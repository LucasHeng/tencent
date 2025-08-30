import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.amp import autocast, GradScaler

from dataset import MyDataset
from model import BaselineModel
from InfoNCE import InfoNCE
import random
import math

class WarmupCosineScheduler:
    """
    Warmup + Cosine Annealing Learning Rate Scheduler
    """
    def __init__(self, optimizer, warmup_steps, total_steps, warmup_lr, max_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_lr = warmup_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_count = 0
        
        # 设置初始学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.warmup_lr
    
    def step(self):
        """更新学习率"""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Warmup阶段：线性增长
            lr = self.warmup_lr + (self.max_lr - self.warmup_lr) * (self.step_count / self.warmup_steps)
        else:
            # 余弦退火阶段
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)  # 确保不超过1.0
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        # 更新所有参数组的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        """获取当前学习率"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]

def set_seed(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)               # 为CPU设置种子
    torch.cuda.manual_seed_all(seed)     # 为所有GPU设置种子
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set as {seed}")

def seed_worker(worker_id):
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)
    torch.manual_seed(42 + worker_id)
    # 若使用CUDA，还需设置cuda种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + worker_id)

def _format_elapsed(second_span: float) -> str:
    hours = int(second_span // 3600)
    minutes = int((second_span % 3600) // 60)
    seconds = int(second_span % 60)
    return f"{hours}小时{minutes}分钟{seconds}秒"

def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=32, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--use_hstu_attn', action='store_true')
    parser.add_argument('--concat_ua', action='store_false')
    parser.add_argument('--use_all_in_batch', action='store_true')
    parser.add_argument('--sample_neg_num',  default=1, type=int)
    parser.add_argument('--skip_mm_emb', action='store_true')
    
    # 性能优化参数
    parser.add_argument('--use_gradient_checkpointing', action='store_true', help='启用梯度检查点以减少内存占用')
    parser.add_argument('--compile_model', action='store_true', help='使用torch.compile优化模型')
    parser.add_argument('--use_amp', action='store_true', default=False, help='使用混合精度训练')
    parser.add_argument('--optimize_backward', action='store_true', default=True, help='启用backward性能优化')
    parser.add_argument('--gradient_clip', default=1.0, type=float, help='梯度裁剪阈值')
    parser.add_argument('--grad_norm_freq', default=20, type=int, help='梯度范数计算频率（每N步计算一次）')
    parser.add_argument('--log_freq', default=10, type=int, help='日志记录频率（每N步记录一次）')
    parser.add_argument('--print_freq', default=10, type=int, help='控制台打印频率（每N步打印一次）')
    
    # 学习率调度器参数
    parser.add_argument('--use_scheduler', action='store_false', default=True, help='使用学习率调度器')
    parser.add_argument('--warmup_steps', default=1000, type=int, help='Warmup步数')
    parser.add_argument('--warmup_lr', default=1e-6, type=float, help='Warmup起始学习率')
    parser.add_argument('--min_lr', default=1e-7, type=float, help='最小学习率')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])
    

    args = parser.parse_args()

    return args


if __name__ == '__main__':   
    set_seed(42)
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')
    save_path = os.environ.get('USER_CACHE_PATH')

    args = get_args()
    dataset = MyDataset(data_path, args, save_path)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, collate_fn=dataset.collate_fn, worker_init_fn=seed_worker, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, collate_fn=dataset.collate_fn, worker_init_fn=seed_worker, pin_memory=True,
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    model = model.cuda()
    
    # 应用模型优化
    if args.compile_model and hasattr(torch, 'compile'):
        print("Using torch.compile to optimize model...")
        model = torch.compile(model, mode='max-autotune')
    
    # 启用backward性能优化
    if args.optimize_backward:
        print("Enabling backward performance optimizations...")
        # 设置torch优化选项
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # 启用内存高效的反向传播
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    scaler = GradScaler('cuda')  # 梯度缩放器，防止 FP16 下溢
    
    # 创建学习率调度器
    scheduler = None
    if args.use_scheduler:
        # 计算总训练步数
        total_steps = len(train_loader) * args.num_epochs * 2
        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=args.warmup_steps,
            total_steps=total_steps,
            warmup_lr=args.warmup_lr,
            max_lr=args.lr,
            min_lr=args.min_lr
        )
        print(f"Created WarmupCosineScheduler: warmup_steps={args.warmup_steps}, total_steps={total_steps}")
        print(f"Learning rate range: {args.warmup_lr} -> {args.lr} -> {args.min_lr}")

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    infonce_criterion = InfoNCE(temperature=0.03, reduction='mean')

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    pre_t = time.time()
    global_step = 0
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_timestamp = batch
            seq = seq.to(args.device, non_blocking=True)
            pos = pos.to(args.device, non_blocking=True)
            neg = neg.to(args.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            t1 = time.time()
            
            # 条件使用混合精度
            if args.use_amp:
                with autocast('cuda'):
                    log_feats, pos_embs, neg_embs = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_timestamp
                    )
            else:
                log_feats, pos_embs, neg_embs = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_timestamp
                )
            
            t2 = time.time()
            indices = np.where(next_token_type == 1)
            if not args.use_all_in_batch:
                loss, acc, pos_sim, neg_sim, future_click_acc = infonce_criterion(log_feats[indices], pos_embs[indices], neg_embs)
            else:
                pos_mask = model.posmask(pos)
                x_index,y_index = indices
                selected_masks = pos_mask[indices]
                loss, acc, pos_sim, neg_sim, future_click_acc = infonce_criterion(log_feats[indices], pos_embs[indices], neg_embs, pos_mask=selected_masks[:,x_index,y_index])
            
            t3 = time.time()
            
            # 日志记录优化 - 降低频率
            if step % args.log_freq == 0:
                elapsed_str = _format_elapsed(time.time() - t0)
                log_json = json.dumps(
                    {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': elapsed_str, 'click_acc': future_click_acc},
                    ensure_ascii=False,
                )
                log_file.write(log_json + '\n')
                log_file.flush()
                
                # TensorBoard日志记录
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Acc/train', acc, global_step)
                writer.add_scalar('Click_acc/train', future_click_acc, global_step)
                writer.add_scalar('Pos_sim/train', pos_sim, global_step)
                writer.add_scalar('Neg_sim/train', neg_sim, global_step)
            
            # 控制台打印优化 - 降低频率
            if step % args.print_freq == 0:
                elapsed_str = _format_elapsed(time.time() - t0)
                log_json = json.dumps(
                    {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': elapsed_str, 'acc': acc.item(), 'click_acc': future_click_acc},
                    ensure_ascii=False,
                )
                print(log_json)

            global_step += 1

            # L2正则化优化 - 使用更高效的向量化计算
            if args.l2_emb > 0:
                l2_loss = 0.0
                for param in model.item_emb.parameters():
                    l2_loss += param.pow(2).sum()
                loss += args.l2_emb * l2_loss
            
            t4 = time.time()
            
            # 条件使用混合精度backward
            if args.use_amp:
                scaler.scale(loss).backward()
                # 优化梯度范数计算 - 使用torch.nn.utils.clip_grad_norm_的底层实现
                scaler.unscale_(optimizer)
                
                # 梯度范数计算优化 - 降低计算频率
                if step % args.grad_norm_freq == 0:
                    total_norm = 0.0
                    # 一个循环同时计算总梯度范数和记录各层梯度范数
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.data.norm(2).item()
                            total_norm += grad_norm ** 2
                            # 记录每一层参数的梯度范数到TensorBoard
                            writer.add_scalar(f'Grad Norm/{name}', grad_norm, step)
                    
                    total_norm = total_norm ** 0.5
                    writer.add_scalar('Grad Norm/Total', total_norm, step)

                # 梯度裁剪优化（总是需要）
                if args.gradient_clip > 0:
                    params_with_grad = [p for p in model.parameters() if p.grad is not None]
                    if len(params_with_grad) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            params_with_grad,
                            args.gradient_clip,
                            error_if_nonfinite=False,
                        )
                
                t5 = time.time()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                
                # 梯度裁剪优化（总是需要）
                if args.gradient_clip > 0:
                    params_with_grad = [p for p in model.parameters() if p.grad is not None]
                    if len(params_with_grad) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            params_with_grad,
                            args.gradient_clip,
                            error_if_nonfinite=False,
                        )
                
                # 梯度范数计算优化 - 降低计算频率
                if step % args.grad_norm_freq == 0:
                    total_norm = 0.0
                    # 一个循环同时计算总梯度范数和记录各层梯度范数
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.data.norm(2).item()
                            total_norm += grad_norm ** 2
                            # 记录每一层参数的梯度范数到TensorBoard
                            writer.add_scalar(f'Grad Norm/{name}', grad_norm, step)
                    
                    total_norm = total_norm ** 0.5
                    writer.add_scalar('Grad Norm', total_norm, step)
                
                t5 = time.time()
                optimizer.step()
            
            # 更新学习率
            if scheduler is not None:
                current_lr = scheduler.step()
                if step % args.log_freq == 0:
                    writer.add_scalar('Learning Rate', current_lr, global_step)
            t6 = time.time()
            total = t6 - pre_t
            
            # 时间统计打印优化 - 降低频率
            if step % args.print_freq == 0:
                print(f'total time: {total:.3f}s load time:{t1-pre_t:.3f}s {(t1-pre_t)/total:.3f} train time: {t2-t1:.2f}s {(t2-t1)/total:.3f}, forward time: {t3-t2:.3f}s {(t3-t2)/total:.3f}, backward time: {t5-t4:.3f}s {(t5-t4)/total:.3f}, optimizer time: {t6-t5:.3f}s {(t6-t5)/total:.3f}')
            
            pre_t = t6
            # print(f"Train Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            # print(f"Train Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        model.eval()
        valid_loss_sum = 0
        valid_acc_sum = 0
        valid_pos_sim_sum = 0
        valid_neg_sim_sum = 0
        valid_future_click_acc_sum = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_timestamp = batch
                seq = seq.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)
                log_feats, pos_embs, neg_embs  = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_timestamp
                )
                indices = np.where(next_token_type == 1)
                if not args.use_all_in_batch:
                    loss, acc, pos_sim, neg_sim, future_click_acc = infonce_criterion(log_feats[indices], pos_embs[indices], neg_embs)
                else:
                    pos_mask = model.posmask(pos)
                    x_index,y_index = indices
                    selected_masks = pos_mask[indices]
                    loss, acc, pos_sim, neg_sim, future_click_acc = infonce_criterion(log_feats[indices], pos_embs[indices], neg_embs, pos_mask=selected_masks[:,x_index,y_index])
                valid_loss_sum += loss.item()
                valid_acc_sum += acc.item()
                valid_pos_sim_sum += pos_sim.item()
                valid_neg_sim_sum += neg_sim.item()
                valid_future_click_acc_sum += future_click_acc
                # print(f"Valid Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                # print(f"Valid Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        valid_acc_sum/=len(valid_loader)
        valid_future_click_acc_sum/=len(valid_loader)
        valid_pos_sim_sum/=len(valid_loader)
        valid_neg_sim_sum/=len(valid_loader)
        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)
        writer.add_scalar('Acc/valid', valid_acc_sum, global_step)
        writer.add_scalar('Click_acc/valid', valid_future_click_acc_sum, global_step)
        writer.add_scalar('Pos_sim/valid', valid_pos_sim_sum, global_step)
        writer.add_scalar('Neg_sim/valid', valid_neg_sim_sum, global_step)

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    dataset.save_statistics(save_path)
    print("Done")
    writer.close()
    log_file.close()
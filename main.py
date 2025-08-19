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

from dataset import MyDataset
from model import BaselineModel
from InfoNCE import InfoNCE
import random

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
    parser.add_argument('--lr', default=0.001, type=float)
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

    args = get_args()
    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, collate_fn=dataset.collate_fn, worker_init_fn=seed_worker
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, collate_fn=dataset.collate_fn, worker_init_fn=seed_worker
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

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

    infonce_criterion = InfoNCE(temperature=0.07, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            log_feats, pos_embs, neg_embs = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )

            optimizer.zero_grad()
            indices = np.where(next_token_type == 1)
            if not args.use_all_in_batch:
                loss, acc, pos_sim, neg_sim, Top10_acc = infonce_criterion(log_feats[indices], pos_embs[indices], neg_embs)
            else:
                pos_mask = model.posmask(pos)
                x_index,y_index = indices
                selected_masks = pos_mask[indices]
                loss, acc, pos_sim, neg_sim, Top10_acc = infonce_criterion(log_feats[indices], pos_embs[indices], neg_embs, pos_mask=selected_masks[:,x_index,y_index])

            elapsed_str = _format_elapsed(time.time() - t0)
            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': elapsed_str},
                ensure_ascii=False,
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Acc/train', acc, global_step)
            writer.add_scalar('Top10_acc/train', Top10_acc, global_step)
            writer.add_scalar('Pos_sim/train', pos_sim, global_step)
            writer.add_scalar('Neg_sim/train', neg_sim, global_step)

            global_step += 1

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)  # L2 范数
                    total_norm += param_norm.item() ** 2
            writer.add_scalar(f'Grad Norm', total_norm, step)
            optimizer.step()
            # print(f"Train Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            # print(f"Train Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        model.eval()
        valid_loss_sum = 0
        valid_acc_sum = 0
        valid_pos_sim_sum = 0
        valid_neg_sim_sum = 0
        valid_Top10_acc_sum = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                seq = seq.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)
                log_feats, pos_embs, neg_embs  = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                )
                indices = np.where(next_token_type == 1)
                if not args.use_all_in_batch:
                    loss, acc, pos_sim, neg_sim, Top10_acc = infonce_criterion(log_feats[indices], pos_embs[indices], neg_embs)
                else:
                    pos_mask = model.posmask(pos)
                    x_index,y_index = indices
                    selected_masks = pos_mask[indices]
                    loss, acc, pos_sim, neg_sim, Top10_acc = infonce_criterion(log_feats[indices], pos_embs[indices], neg_embs, pos_mask=selected_masks[:,x_index,y_index])
                valid_loss_sum += loss.item()
                valid_acc_sum += acc.item()
                valid_pos_sim_sum += pos_sim.item()
                valid_neg_sim_sum += neg_sim.item()
                valid_Top10_acc_sum += Top10_acc
                # print(f"Valid Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                # print(f"Valid Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        valid_acc_sum/=len(valid_loader)
        valid_Top10_acc_sum/=len(valid_loader)
        valid_pos_sim_sum/=len(valid_loader)
        valid_neg_sim_sum/=len(valid_loader)
        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)
        writer.add_scalar('Acc/valid', valid_acc_sum, global_step)
        writer.add_scalar('Top10_acc/valid', valid_Top10_acc_sum, global_step)
        writer.add_scalar('Pos_sim/valid', valid_pos_sim_sum, global_step)
        writer.add_scalar('Neg_sim/valid', valid_neg_sim_sum, global_step)

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
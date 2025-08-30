import json
import pickle
import struct
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
import os
import time
try:
    import fcntl  # Linux/Unix文件锁
    _FCNTL_AVAILABLE = True
except Exception:
    _FCNTL_AVAILABLE = False

import os
def clear_directory(directory):
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或符号链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除子目录
        except Exception as e:
            print(f"删除失败: {file_path}, 错误: {e}")
class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

    def __init__(self, data_dir, args, save_path):
        """
        初始化数据集
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()
        self.maxlen = args.maxlen
        if args.skip_mm_emb:
            self.mm_emb_ids = []
        else:
            self.mm_emb_ids = args.mm_emb_id

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
            # print(f'itemnum {self.itemnum} and usernum {self.usernum}')
        # itemid 与 reconstruct id
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer
        self.sample_neg_num = args.sample_neg_num

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()
        # 统计时间间隔分桶：类别0为padding，1为零间隔，2..为各边界桶

        # 缓存时间分桶边界，避免 __getitem__ 重复创建
        self._tdelta_edges = np.array([
            30, 60, 120, 300, 600, 900, 1800,
            3600, 7200, 14400, 28800,
            43200, 64800, 86400, 129600, 172800,
            259200, 432000, 604800, 1209600
        ], dtype=np.int64)

        self.main_pid = os.getpid()  # 此时在主进程中执行，记录主进程 ID
        # 初始化全局统计信息
        global_stats = {}
        self._save_path = save_path  # 保存save_path以便后续使用
        
        # 设置训练标志，用于调试信息
        self._is_training = True
        
        self._init_global_statistics()

    def _safe_feat_stat(self, k):
        return getattr(self, 'feat_statistics', {}).get(k, 0)

    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        # 仅记录路径，避免在主进程持有打开的文件描述符（便于多worker安全地各自打开）
        self._data_file_path = self.data_dir / "seq.jsonl"
        self.data_file = None
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _ensure_data_file_open(self, use_data_file=True):
        if use_data_file:
            if getattr(self, 'data_file', None) is None:
                self.data_file = open(self._data_file_path, 'rb')
            return self.data_file
        else:
            # 每次调用都创建新的临时文件句柄，避免多进程冲突
            import os
            import threading
            
            # 获取当前进程ID和线程ID
            pid = os.getpid()
            thread_id = threading.get_ident()
            
            # 创建新的文件句柄
            file_handle = open(self._data_file_path, 'rb')
            file_handle_id = id(file_handle)
            
            # 打印调试信息（只在训练时打印，避免推理时过多输出）
            if hasattr(self, '_is_training') and self._is_training:
                print(f"进程 {pid} 线程 {thread_id} 创建新文件句柄: {file_handle_id}")
            
            return file_handle

    def _load_user_data(self, uid, use_data_file=True):
        """
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        with self._ensure_data_file_open(use_data_file=use_data_file) as data_file:
            data_file.seek(self.seq_offsets[uid])
            line = data_file.readline()            
            data = json.loads(line)
            return data


    def save_statistics(self, global_stats):
        """
        仅在主进程持久化一次原始增量raw_stats；若存在同名文件则直接删除后重写。
        """
        if not hasattr(self, 'global_stats') or not global_stats:
            return

        save_dir = Path(self._save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        stats_file = save_dir / 'global_statistics.pkl'

        if not (os.getpid() == self.main_pid):
            return
        
        # 删除同名文件
        if stats_file.exists():
            try:
                stats_file.unlink()
            except Exception:
                pass
        
        # 保存统计信息
        payload = {'raw': global_stats}
        with open(stats_file, 'wb') as f:
            pickle.dump(payload, f)
    
        print(f'dump 完成！！！')

    def _random_neq(self, l, r, s,neg_num=1):
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列

        Returns:
            t: 不在序列s中的随机整数
        """
        neg_samps = []
        for i in range(neg_num):
            t = np.random.randint(l, r)
            while t in s or str(t) not in self.item_feat_dict:
                t = np.random.randint(l, r)
            neg_samps.append(t)
        return neg_samps

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户ID(reid)

        Returns:
            seq: 用户序列ID
            pos: 正样本ID（即下一个真实访问的item）
            neg: 负样本ID
            token_type: 用户序列类型，1表示item，2表示user
            next_token_type: 下一个token类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            pos_feat: 正样本特征，每个元素为字典，key为特征ID，value为特征值
            neg_feat: 负样本特征，每个元素为字典，key为特征ID，value为特征值
        """
        user_sequence = self._load_user_data(uid, use_data_file=False)  # 动态加载用户数据
        
        # 处理空序列（文件读取失败的情况）
        if not user_sequence:
            # 返回一个默认的空序列数据
            seq = np.zeros([self.maxlen + 1], dtype=np.int32)
            pos = np.zeros([self.maxlen + 1], dtype=np.int32)
            neg = np.zeros([self.maxlen + 1, self.sample_neg_num], dtype=np.int32)
            token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
            next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
            next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)
            seq_timestamp = np.zeros([self.maxlen + 1], dtype=np.int32)
            
            seq_feat = np.full([self.maxlen + 1], self.feature_default_value, dtype=object)
            pos_feat = np.full([self.maxlen + 1], self.feature_default_value, dtype=object)
            neg_feat = np.full([self.maxlen + 1, self.sample_neg_num], self.feature_default_value, dtype=object)
            
            return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_timestamp
    
        # 合并遍历：同时构建统计缓存和扩展序列
        user_stats_cache = {}
        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, timestamp = record_tuple
            
            # 构建统计缓存
            if i and item_feat and action_type is not None:
                if i not in user_stats_cache:
                    user_stats_cache[i] = {'clicks': 0, 'impressions': 0}
                
                if action_type == 1:  # 点击
                    user_stats_cache[i]['clicks'] += 1
                else:  # 曝光或其他行为
                    user_stats_cache[i]['impressions'] += 1
            
            # 构建扩展序列
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type, timestamp))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type, timestamp))

        # 检查扩展序列是否为空
        if not ext_user_sequence:
            # 如果扩展序列为空，返回默认的空序列数据
            seq = np.zeros([self.maxlen + 1], dtype=np.int32)
            pos = np.zeros([self.maxlen + 1], dtype=np.int32)
            neg = np.zeros([self.maxlen + 1], self.sample_neg_num, dtype=np.int32)
            token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
            next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
            next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)
            seq_timestamp = np.zeros([self.maxlen + 1], dtype=np.int32)
            
            seq_feat = np.full([self.maxlen + 1], self.feature_default_value, dtype=object)
            pos_feat = np.full([self.maxlen + 1], self.feature_default_value, dtype=object)
            neg_feat = np.full([self.maxlen + 1], self.sample_neg_num, self.feature_default_value, dtype=object)
            
            return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_timestamp

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1, self.sample_neg_num], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_timestamp = np.zeros([self.maxlen + 1], dtype=np.int32)

        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1, self.sample_neg_num], dtype=object)

        nxt = ext_user_sequence[-1]
        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        # left-padding, 从后往前遍历，将用户序列填充到maxlen+1的长度
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type, timestamp = record_tuple
            next_i, next_feat, next_type, next_act_type, next_timestamp = nxt
            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)
            seq[idx] = i
            seq_timestamp[idx] = timestamp
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            # 构造时间特征并写入稀疏特征（整数类别）
            # 时间特征同时用于用户和物品
            hour = int((timestamp // 3600) % 24) if timestamp > 0 else 0
            weekday = int(((timestamp // 86400) + 4) % 7) if timestamp > 0 else 0
            month = int(((timestamp // (86400 * 30)) % 12) + 1) if timestamp > 0 else 0
            # 1304(时间差分桶)稍后统一根据"当前-前一条"的时间差回填
            feat['1301'] = month  # 已为1..12，缺失为0
            feat['1302'] = (weekday + 1) if timestamp > 0 else 0  # 1..7
            feat['1303'] = (hour + 1) if timestamp > 0 else 0      # 1..24
            feat['1304'] = 0  # 先置0，占位，循环结束后回填正确分桶
            
            # 为item添加统计特征
            if type_ == 1 and i != 0:  # item类型且非padding
                item_stats = self._calculate_item_statistics(
                    i, user_sequence, timestamp, idx, 
                    user_stats_cache=user_stats_cache
                )
                feat.update(item_stats)
            
            seq_feat[idx] = feat
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                # pos时间特征使用默认值（不显式赋值）
                pos_feat[idx] = next_feat
                neg_ids = self._random_neq(1, self.itemnum + 1, ts, self.sample_neg_num)
                neg[idx] = neg_ids
                for id,neg_id in enumerate(neg_ids):
                    nf = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
                    # neg时间特征使用默认值（不显式赋值）
                    neg_feat[idx][id] = nf
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        # 回填1304：使用“当前-前一条”的时间差，按固定边界分桶（前密后疏），非零分桶+1；首位为0
        edges = self._tdelta_edges
        # 仅对item位置计算“当前item-上一个item”的时间差；user位置置0
        ts_np = seq_timestamp
        tt_np = token_type
        # 向量化：仅对item位置计算“当前item-上一个item”的时间差；user位置置0
        item_mask = (tt_np == 1) & (ts_np > 0)
        item_pos = np.where(item_mask)[0]
        cats = np.zeros_like(ts_np, dtype=np.int64)
        if item_pos.size > 0:
            item_ts = ts_np[item_mask]
            diffs = np.diff(item_ts, prepend=0)
            cats_item = np.zeros_like(diffs, dtype=np.int64)
            zero_mask = (diffs == 0)
            nonzero_mask = diffs > 0
            cats_item[zero_mask] = 1  # 零间隔→1
            if np.any(nonzero_mask):
                idxs = np.searchsorted(edges, diffs[nonzero_mask], side='right')
                cats_item[nonzero_mask] = idxs + 2  # 非零→2..
            cats_item[0] = 0  # 首个item归0
            cats[item_pos] = cats_item
        max_cat = int(self._safe_feat_stat('1304'))
        cats = np.clip(cats, 0, max_cat)
        for t in range(len(seq_feat)):
            if isinstance(seq_feat[t], dict):
                c = int(cats[t])
                seq_feat[t]['1304'] = c

        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_timestamp

    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)


    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109', '1301', '1302', '1303', '1304']
        feat_types['item_sparse'] = [
            '100',
            '117',
            '111',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            '121',
            '115',
            '122',
            '116',
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = [
            '2001',  # item全局点击次数
            '2002',  # item全局曝光次数  
            '2003',  # item全局点击率
            '2004',  # item被当前用户点击次数
            '2005',  # item被当前用户曝光次数
            '2006'   # item被当前用户点击率
        ]
        # 连续型特征
        feat_types['item_continual'] = []
        # 将时间特征改为稀疏特征（整数类别），使用数字型特征ID
        # 约定：1301-月份，1302-星期，1303-小时，1304-时间差分桶
        # 时间特征同时用于用户和物品

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0.0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        # 为新增稀疏时间特征设置默认值与词表大小（数值型ID）
        feat_default_value['1301'] = 0
        feat_default_value['1302'] = 0
        feat_default_value['1303'] = 0
        feat_default_value['1304'] = 0
        feat_statistics['1301'] = 12
        feat_statistics['1302'] = 7
        feat_statistics['1303'] = 24
        # t_delta 使用固定边界分桶：非零桶 = 1(零间隔) + len(edges) + 1(>max边界)
        # 当前 edges=20 → 非零桶=22（+0 padding）
        feat_statistics['1304'] = 22
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat == None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]

        return filled_feat

    def collate_fn(self, batch):
        """
        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            pos: 正样本ID, torch.Tensor形式
            neg: 负样本ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            next_token_type: 下一个token类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            pos_feat: 正样本特征, list形式
            neg_feat: 负样本特征, list形式
        """
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, seq_timestamp = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        seq_timestamp = torch.from_numpy(np.array(seq_timestamp))

        # helpers
        B = len(seq_feat)
        L = len(seq_feat[0]) if B > 0 else 0

        def build_dense(feat_ids, batch_list, dtype='int'):
            if dtype == 'int':
                out = {k: np.array([[sample[t][k] for t in range(L)] for sample in batch_list], dtype=np.int64) for k in feat_ids}
            elif dtype == 'float':
                out = {k: np.array([[sample[t][k] for t in range(L)] for sample in batch_list], dtype=np.float32) for k in feat_ids}
            else:
                out = {}
            return out

        def build_array(feat_ids, batch_list):
            out = {}
            for k in feat_ids:
                max_a = 1
                for sample in batch_list:
                    for t in range(L):
                        v = sample[t][k]
                        if isinstance(v, list):
                            if len(v) > max_a:
                                max_a = len(v)
                arr = np.zeros((B, L, max_a), dtype=np.int64)
                for b, sample in enumerate(batch_list):
                    for t in range(L):
                        v = sample[t][k]
                        if isinstance(v, list) and len(v) > 0:
                            a = min(len(v), max_a)
                            arr[b, t, :a] = np.asarray(v[:a], dtype=np.int64)
                out[k] = arr
            return out

        def build_emb(feat_ids, batch_list):
            out = {}
            for k in feat_ids:
                dim = self.feature_default_value[k].shape[0]
                arr = np.zeros((B, L, dim), dtype=np.float32)
                for b, sample in enumerate(batch_list):
                    for t in range(L):
                        v = sample[t][k]
                        if isinstance(v, np.ndarray) and v.size > 0:
                            arr[b, t] = v
                out[k] = arr
            return out

        seq_feat_list = list(seq_feat)
        pos_feat_list = list(pos_feat)
        neg_feat_list = list(neg_feat)

        # pack seq & pos features using known feature id groups
        seq_feat_pre = {}
        pos_feat_pre = {}
        # item sparse/continual
        seq_feat_pre.update(build_dense(self.feature_types['item_sparse'], seq_feat_list, 'int'))
        pos_feat_pre.update(build_dense(self.feature_types['item_sparse'], pos_feat_list, 'int'))
        seq_feat_pre.update(build_dense(self.feature_types['item_continual'], seq_feat_list, 'float'))
        pos_feat_pre.update(build_dense(self.feature_types['item_continual'], pos_feat_list, 'float'))
        # user sparse/continual
        seq_feat_pre.update(build_dense(self.feature_types['user_sparse'], seq_feat_list, 'int'))
        seq_feat_pre.update(build_dense(self.feature_types['user_continual'], seq_feat_list, 'float'))
        # arrays
        seq_feat_pre.update(build_array(self.feature_types['item_array'], seq_feat_list))
        pos_feat_pre.update(build_array(self.feature_types['item_array'], pos_feat_list))
        seq_feat_pre.update(build_array(self.feature_types['user_array'], seq_feat_list))
        # multimodal emb
        seq_feat_pre.update(build_emb(self.feature_types['item_emb'], seq_feat_list))
        pos_feat_pre.update(build_emb(self.feature_types['item_emb'], pos_feat_list))

        # pack negatives: shapes -> sparse/continual [B, L, K], array [B, L, K, A], emb [B, L, K, E]
        K = neg.shape[-1]

        def build_dense_neg(feat_ids, batch_list, dtype='int'):
            if dtype == 'int':
                arrs = {}
                for k in feat_ids:
                    arr = np.zeros((B, L, K), dtype=np.int64)
                    for b, sample in enumerate(batch_list):
                        for t in range(L):
                            for n in range(K):
                                arr[b, t, n] = sample[t][n][k]
                    arrs[k] = arr
                return arrs
            else:
                arrs = {}
                for k in feat_ids:
                    arr = np.zeros((B, L, K), dtype=np.float32)
                    for b, sample in enumerate(batch_list):
                        for t in range(L):
                            for n in range(K):
                                arr[b, t, n] = float(sample[t][n][k])
                    arrs[k] = arr
                return arrs

        def build_array_neg(feat_ids, batch_list):
            arrs = {}
            for k in feat_ids:
                max_a = 1
                for sample in batch_list:
                    for t in range(L):
                        for n in range(K):
                            v = sample[t][n][k]
                            if isinstance(v, list) and len(v) > max_a:
                                max_a = len(v)
                arr = np.zeros((B, L, K, max_a), dtype=np.int64)
                for b, sample in enumerate(batch_list):
                    for t in range(L):
                        for n in range(K):
                            v = sample[t][n][k]
                            if isinstance(v, list) and len(v) > 0:
                                a = min(len(v), max_a)
                                arr[b, t, n, :a] = np.asarray(v[:a], dtype=np.int64)
                arrs[k] = arr
            return arrs

        def build_emb_neg(feat_ids, batch_list):
            arrs = {}
            for k in feat_ids:
                dim = self.feature_default_value[k].shape[0]
                arr = np.zeros((B, L, K, dim), dtype=np.float32)
                for b, sample in enumerate(batch_list):
                    for t in range(L):
                        for n in range(K):
                            v = sample[t][n][k]
                            if isinstance(v, np.ndarray) and v.size > 0:
                                arr[b, t, n] = v
                arrs[k] = arr
            return arrs

        neg_feat_pre = {}
        neg_feat_pre.update(build_dense_neg(self.feature_types['item_sparse'], neg_feat_list, 'int'))
        neg_feat_pre.update(build_dense_neg(self.feature_types['item_continual'], neg_feat_list, 'float'))
        neg_feat_pre.update(build_array_neg(self.feature_types['item_array'], neg_feat_list))
        neg_feat_pre.update(build_emb_neg(self.feature_types['item_emb'], neg_feat_list))

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat_pre, pos_feat_pre, neg_feat_pre, seq_timestamp

    def _calculate_item_statistics(self, item_id, user_sequence, current_timestamp, current_idx, global_stats=None, user_stats_cache=None):
        """
        计算item的统计特征
        
        Args:
            item_id: 物品ID
            user_sequence: 用户序列数据
            current_timestamp: 当前时间戳
            current_idx: 当前在序列中的位置
            global_stats: 全局统计信息，如果为None则使用实例的global_stats属性
            user_stats_cache: 用户统计信息缓存
            
        Returns:
            stats: 包含统计特征的字典
        """
        if item_id == 0:  # padding item
            return {
                '2001': 0.0,  # 全局点击次数(log处理)
                '2002': 0.0,  # 全局曝光次数(log处理)
                '2003': 0.0,  # 全局点击率
                '2004': 0.0,  # 当前用户点击次数(log处理)
                '2005': 0.0,  # 当前用户曝光次数(log处理)
                '2006': 0.0   # 当前用户点击率
            }
        
        # 获取全局统计信息（去重后的用户数）
        if global_stats is None:
            global_stats = getattr(self, 'global_stats', {})
        
        if global_stats and item_id in global_stats:
            global_item_stats = self._get_item_statistics_at_timestamp(item_id, current_timestamp, global_stats)
            global_clicks = global_item_stats['clicks']      # 去重后的用户数
            global_impressions = global_item_stats['impressions']  # 去重后的用户数
        else:
            global_clicks = 0
            global_impressions = 0
        
        # 获取用户统计信息（事件数）
        user_clicks = 0
        user_impressions = 0
        
        if user_stats_cache and item_id in user_stats_cache:
            user_item_stats = user_stats_cache[item_id]
            user_clicks = user_item_stats['clicks']      # 事件数
            user_impressions = user_item_stats['impressions']  # 事件数
        
        # 计算点击率（避免除零）
        global_ctr = global_clicks / max(global_impressions + global_clicks, 1)
        user_ctr = user_clicks / max(user_impressions + user_clicks, 1)
        
        # 对点击数和曝光数进行log处理，缓解长尾问题
        # 使用 log(1 + x) 避免 log(0) 的问题
        log_global_clicks = np.log1p(global_clicks)
        log_global_impressions = np.log1p(global_impressions)
        log_user_clicks = np.log1p(user_clicks)
        log_user_impressions = np.log1p(user_impressions)
        
        return {
            '2001': float(log_global_clicks),      # 全局点击次数(log处理)
            '2002': float(log_global_impressions), # 全局曝光次数(log处理)
            '2003': float(global_ctr),             # 全局点击率
            '2004': float(log_user_clicks),        # 当前用户点击次数(log处理)
            '2005': float(log_user_impressions),   # 当前用户曝光次数(log处理)
            '2006': float(user_ctr)                # 当前用户点击率
        }

    def _precompute_global_statistics(self, save_path=None):
        """
        扫描训练集，构建按时间戳的原始增量统计：
        raw_stats: {item_id: {timestamp: {'clicks': c, 'impressions': i}}}
        若提供 save_path，直接持久化 raw_stats。
        同时在内存构建累计缓存self._cumu_cache，供查询使用。
        """
        print("开始预计算全局统计信息(原始时间戳增量结构)...")
        raw_stats = {}  # {item_id: {timestamp: {'clicks': set(user_id), 'impressions': set(user_id)}}}

        # 直接顺序读取文件，比根据uid seek更快
        print("直接顺序读取文件进行统计...")
        with open(self._data_file_path, 'rb') as data_file:
            for line_num, line in enumerate(tqdm(data_file, desc="扫描文件行")):
                try:
                    user_sequence = json.loads(line)
                    for record in user_sequence:
                        u, i, user_feat, item_feat, action_type, timestamp = record
                        if i and item_feat and action_type is not None:
                            # 初始化嵌套字典结构
                            if i not in raw_stats:
                                raw_stats[i] = {}
                            if timestamp not in raw_stats[i]:
                                raw_stats[i][timestamp] = {'clicks': set(), 'impressions': set()}
                            
                            # 将用户ID添加到对应的集合中，自动去重
                            user_id = int(u) if u is not None else 0
                            if action_type == 1:
                                raw_stats[i][timestamp]['clicks'].add(user_id)
                            else:
                                raw_stats[i][timestamp]['impressions'].add(user_id)
                except json.JSONDecodeError:
                    # 静默处理JSON解析错误，继续下一行
                    continue
                except Exception as e:
                    print(f"处理第 {line_num} 行时出错: {e}")
                    continue

        # 持久化 raw_stats - 使用临时文件句柄
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if save_path.exists():
                print(f"清除同名文件: {save_path}")
                save_path.unlink()
            payload = {'raw': raw_stats}
            # 使用临时文件句柄，不作为成员变量
            with open(save_path, 'wb') as f:
                pickle.dump(payload, f)
            print(f"原始增量统计已保存到: {save_path}")

        # 构建累计缓存
        self._build_cumu_cache_from_raw(raw_stats)
        
        # 统计最大值
        self._max_global_clicks = 0
        self._max_global_impressions = 0
        
        return raw_stats
    
    def _build_cumu_cache_from_raw(self, raw_stats):
        """从原始增量raw_stats构建累计缓存，便于按时间戳查询累计点击/曝光。"""
        import numpy as _np
        self._cumu_cache = {}
        # 初始化最大值统计
        self._max_global_clicks = 0
        self._max_global_impressions = 0
        
        for item_id, ts_dict in raw_stats.items():
            ts_sorted = sorted(ts_dict.keys())
            clicks = _np.zeros(len(ts_sorted), dtype=_np.int64)
            imprs = _np.zeros(len(ts_sorted), dtype=_np.int64)
            c_sum = 0
            i_sum = 0
            for idx, ts in enumerate(ts_sorted):
                # 计算集合的长度（去重后的用户数）
                clicks_count = len(ts_dict[ts].get('clicks', set()))
                impressions_count = len(ts_dict[ts].get('impressions', set()))
                
                c_sum += clicks_count
                i_sum += impressions_count
                clicks[idx] = c_sum
                imprs[idx] = i_sum
                # 统计最大值
                if c_sum > self._max_global_clicks:
                    self._max_global_clicks = c_sum
                
                if i_sum > self._max_global_impressions:
                    self._max_global_impressions = i_sum

            self._cumu_cache[item_id] = (ts_sorted, clicks, imprs)
        
        print(f"统计完成 - 最大点击用户数: {self._max_global_clicks}, 最大曝光用户数: {self._max_global_impressions}")

    def _load_global_statistics(self, load_path, build_cache=True):
        """
        加载原始增量统计raw_stats，可选是否立即重建累计缓存。
        返回raw_stats字典。
        """
        load_path = Path(load_path)
        if not load_path.exists():
            print(f"统计信息文件不存在: {load_path}")
            return {}
        print(f"加载全局统计信息(raw): {load_path}")
        # 使用临时文件句柄，不作为成员变量
        with open(load_path, 'rb') as f:
            payload = pickle.load(f)
        raw_stats = payload.get('raw', {})
        if build_cache:
            self._build_cumu_cache_from_raw(raw_stats)
        
        print(f"加载完成，包含 {len(raw_stats)} 个item的统计信息")
        return raw_stats
    
    def _get_item_statistics_at_timestamp(self, item_id, timestamp, raw_stats):
        """
        从累计缓存中获取指定item在指定时间戳的累计统计（clicks, impressions）。
        若无缓存/条目，返回0值。
        """
        cache = self._cumu_cache.get(item_id)
        if cache is None:
            return {'clicks': 0, 'impressions': 0}
        ts_sorted, clicks_cumu, imprs_cumu = cache
        # 二分：找到<=timestamp的最后一个位置
        import bisect as _bisect
        idx = _bisect.bisect_right(ts_sorted, timestamp) - 1
        if idx < 0:
            return {'clicks': 0, 'impressions': 0}
        return {'clicks': int(clicks_cumu[idx]), 'impressions': int(imprs_cumu[idx])}

    def _init_global_statistics(self):
        """
        初始化全局统计信息
        
        Args:
            args: 参数对象，包含统计信息相关配置
            save_path: 统计数据保存路径，如果为None则使用data_dir
        """
        save_path = self._save_path
        print(f"save_path为None，使用默认路径: {save_path}")

        stats_file = Path(save_path) / "global_statistics.pkl"

        is_test_dataset = self.__class__.__name__ == 'MyTestDataset'
        is_primary = (os.getpid() == self.main_pid)

        global_stats = {}
        if is_test_dataset:
            # 测试数据集：主进程加载训练统计->增量更新->再构建累计缓存；子进程只加载不构建，等待主进程完成
            if is_primary:
                print(f"测试数据集模式：从 {save_path} 加载训练集统计信息")
                if stats_file.exists():
                    print(f"加载训练集统计信息: {stats_file}")
                    global_stats = self._load_global_statistics(stats_file, build_cache=False)
                    # 遍历测试数据更新统计信息（仅内存），不重建缓存
                    self._update_statistics_from_test_data(save_path)
                    # 测试数据更新完成后统一构建累计缓存
                    self._build_cumu_cache_from_raw(global_stats)
                else:
                    print(f"警告：训练集统计信息文件不存在 {stats_file}，将使用空统计信息")
            else:
                if stats_file.exists():
                    # 子进程只加载raw，不构建缓存（避免重复构建），等主进程更新后生效
                    global_stats = self._load_global_statistics(stats_file, build_cache=False)
        else:
            # 训练集：每次训练都重新构建统计信息
            if is_primary:
                clear_directory(save_path)
                # 删除现有统计文件
                if stats_file.exists():
                    print(f"删除现有统计文件: {stats_file}")
                    stats_file.unlink()

                print("自动生成全局统计信息...")
                global_stats = self._precompute_global_statistics(stats_file)
                self.save_statistics(global_stats)
            else:
                # 子进程等待主进程生成完成后再加载
                if stats_file.exists():
                    global_stats = self._load_global_statistics(stats_file)


        if global_stats:
            print(f"全局统计信息初始化完成，包含 {len(global_stats)} 个item")
        else:
            print("警告：全局统计信息为空")
        

    def _update_statistics_from_test_data(self, save_path, global_stats):
        """
        从测试数据更新统计信息（仅内存，不持久化）
        
        Args:
            save_path: 保存路径，如果为None则使用data_dir（此处不使用，仅保留参数签名）
        """
        # 如果save_path为None，使用data_dir作为默认值
        if save_path is None:
            save_path = self.data_dir
        
        print("开始从测试数据更新统计信息（仅内存，不持久化）...")
        
        # 记录训练集中已存在的用户-item-时间组合
        print("开始从测试数据更新统计信息...")
        
        # 遍历测试数据更新统计信息（先缓冲增量，不立即重建累计缓存）
        updated_count = 0
        new_item_count = 0
        touched_items = set()
        
        test_data_file = self.data_dir / "predict_seq.jsonl"
        if not test_data_file.exists():
            print(f"测试数据文件不存在: {test_data_file}")
            return
        
        # 使用临时文件句柄，不作为成员变量
        with open(test_data_file, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    user_sequence = json.loads(line.strip())
                    for record in user_sequence:
                        u, i, user_feat, item_feat, action_type, timestamp = record
                        if i and item_feat and action_type is not None:
                            item_id = i
                            evt_key = (int(u) if u is not None else 0, int(action_type))
                            # 直接更新统计信息，使用集合自动去重
                            # 写入raw增量，不排序不重建
                            if item_id not in global_stats:
                                global_stats[item_id] = {}
                                new_item_count += 1
                            if timestamp not in global_stats[item_id]:
                                global_stats[item_id][timestamp] = {'clicks': set(), 'impressions': set()}
                            
                            inc = global_stats[item_id][timestamp]
                            user_id = int(u) if u is not None else 0
                            if action_type == 1:
                                inc['clicks'].add(user_id)
                            else:
                                inc['impressions'].add(user_id)
                            updated_count += 1
                            touched_items.add(item_id)
                except Exception as e:
                    print(f"处理测试数据第 {line_num + 1} 行时出错: {e}")
                    continue
        
        # 为所有item重建累计缓存（因为之前没有构建）
        self._build_cumu_cache_from_raw(global_stats)
        
        print(f"更新完成：更新了 {updated_count} 条事件，影响 {len(touched_items)} 个item（内存） 新增item {new_item_count}")







class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args, save_path):
        super().__init__(data_dir, args, save_path)
        
        # 如果save_path为None，使用data_dir作为默认值
        if save_path is None:
            save_path = self.data_dir
        
        # 测试集不再加载更新后的持久化文件，保持使用训练统计并在内存增量
        # 保留原有global_stats（由父类加载的训练统计）

    def _load_data_and_offsets(self):
        self._data_file_path = self.data_dir / "predict_seq.jsonl"
        self.data_file = None
        # 使用临时文件句柄，不作为成员变量
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
        """
        处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
        """
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if type(feat_value) == list:
                value_list = []
                for v in feat_value:
                    if type(v) == str:
                        value_list.append(0)
                    else:
                        value_list.append(v)
                processed_feat[feat_id] = value_list
            elif type(feat_value) == str:
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户在self.data_file中储存的行号
        Returns:
            seq: 用户序列ID
            token_type: 用户序列类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            user_id: user_id eg. user_xxxxxx ,便于后面对照答案
        """
        user_sequence = self._load_user_data(uid, use_data_file=False)  # 使用临时文件句柄，避免多进程冲突

        # 预计算当前用户的统计信息缓存
        user_stats_cache = {}
        for record in user_sequence:
            u, i, user_feat, item_feat, action_type, timestamp = record
            if i and item_feat and action_type is not None:
                if i not in user_stats_cache:
                    user_stats_cache[i] = {'clicks': 0, 'impressions': 0}
                
                if action_type == 1:  # 点击
                    user_stats_cache[i]['clicks'] += 1
                else:  # 曝光或其他行为
                    user_stats_cache[i]['impressions'] += 1

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, _, timestamp = record_tuple
            if u:
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u]
            if u and user_feat:
                if type(u) == str:
                    u = 0
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)
                ext_user_sequence.insert(0, (u, user_feat, 2, timestamp))

            if i and item_feat:
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, item_feat, 1, timestamp))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        seq_timestamp = np.zeros([self.maxlen + 1], dtype=np.int32)

        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        for record_tuple in reversed(ext_user_sequence[:]):
            i, feat, type_,timestamp = record_tuple
            feat = self.fill_missing_feat(feat, i)
            # 添加时间稀疏特征（与训练集一致，保留0作为padding）
            # 时间特征同时用于用户和物品
            # 本地时区：东八区（UTC+8）
            tz_offset = 8 * 3600
            ts_local = timestamp + tz_offset if timestamp > 0 else 0
            hour = int((ts_local // 3600) % 24) if timestamp > 0 else 0
            weekday = int(((ts_local // 86400) + 4) % 7) if timestamp > 0 else 0
            month = int(((ts_local // (86400 * 30)) % 12) + 1) if timestamp > 0 else 0
            # 1304(时间差分桶)稍后统一根据"当前-前一条"的时间差回填
            feat['1301'] = month  # 已为1..12，缺失为0
            feat['1302'] = (weekday + 1) if timestamp > 0 else 0  # 1..7
            feat['1303'] = (hour + 1) if timestamp > 0 else 0      # 1..24
            feat['1304'] = 0  # 先置0，占位，循环结束后回填正确分桶
            
            # 为item添加统计特征
            if type_ == 1 and i != 0:  # item类型且非padding
                item_stats = self._calculate_item_statistics(
                    i, user_sequence, timestamp, idx,
                    user_stats_cache=user_stats_cache
                )
                feat.update(item_stats)
            
            seq[idx] = i
            seq_timestamp[idx] = timestamp
            token_type[idx] = type_
            seq_feat[idx] = feat
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        
        # 回填1304：使用"当前-前一条"的时间差，按固定边界分桶（前密后疏），非零分桶+1；首位为0
        # 与训练集保持一致的时间差计算逻辑
        edges = self._tdelta_edges
        ts_np = seq_timestamp
        tt_np = token_type
        # 向量化：仅对item位置计算"当前item-上一个item"的时间差；user位置置0
        item_mask = (tt_np == 1) & (ts_np > 0)
        item_pos = np.where(item_mask)[0]
        cats = np.zeros_like(ts_np, dtype=np.int64)
        if item_pos.size > 0:
            item_ts = ts_np[item_mask]
            diffs = np.diff(item_ts, prepend=0)
            cats_item = np.zeros_like(diffs, dtype=np.int64)
            zero_mask = (diffs == 0)
            nonzero_mask = diffs > 0
            cats_item[zero_mask] = 1  # 零间隔→1
            if np.any(nonzero_mask):
                idxs = np.searchsorted(edges, diffs[nonzero_mask], side='right')
                cats_item[nonzero_mask] = idxs + 2  # 非零→2..
            cats_item[0] = 0  # 首个item归0
            cats[item_pos] = cats_item
        max_cat = int(self._safe_feat_stat('1304'))
        cats = np.clip(cats, 0, max_cat)
        for t in range(len(seq_feat)):
            if isinstance(seq_feat[t], dict):
                c = int(cats[t])
                seq_feat[t]['1304'] = c

        return seq, token_type, seq_feat, user_id, seq_timestamp

    def __len__(self):
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)

    def collate_fn(self, batch):
        """
        将多个__getitem__返回的数据拼接成一个batch

        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            user_id: user_id, str
        """
        seq, token_type, seq_feat, user_id, seq_timestamp = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        seq_timestamp = torch.from_numpy(np.array(seq_timestamp))
        token_type = torch.from_numpy(np.array(token_type))
        # helpers
        B = len(seq_feat)
        L = len(seq_feat[0]) if B > 0 else 0

        def build_dense(feat_ids, batch_list, dtype='int'):
            if dtype == 'int':
                out = {k: np.array([[sample[t][k] for t in range(L)] for sample in batch_list], dtype=np.int64) for k in feat_ids}
            elif dtype == 'float':
                out = {k: np.array([[sample[t][k] for t in range(L)] for sample in batch_list], dtype=np.float32) for k in feat_ids}
            else:
                out = {}
            return out

        def build_array(feat_ids, batch_list):
            out = {}
            for k in feat_ids:
                max_a = 1
                for sample in batch_list:
                    for t in range(L):
                        v = sample[t][k]
                        if isinstance(v, list):
                            if len(v) > max_a:
                                max_a = len(v)
                arr = np.zeros((B, L, max_a), dtype=np.int64)
                for b, sample in enumerate(batch_list):
                    for t in range(L):
                        v = sample[t][k]
                        if isinstance(v, list) and len(v) > 0:
                            a = min(len(v), max_a)
                            arr[b, t, :a] = np.asarray(v[:a], dtype=np.int64)
                out[k] = arr
            return out

        def build_emb(feat_ids, batch_list):
            out = {}
            for k in feat_ids:
                dim = self.feature_default_value[k].shape[0]
                arr = np.zeros((B, L, dim), dtype=np.float32)
                for b, sample in enumerate(batch_list):
                    for t in range(L):
                        v = sample[t][k]
                        if isinstance(v, np.ndarray) and v.size > 0:
                            arr[b, t] = v
                out[k] = arr
            return out

        seq_feat_list = list(seq_feat)


        # pack seq & pos features using known feature id groups
        seq_feat_pre = {}
        # item sparse/continual
        seq_feat_pre.update(build_dense(self.feature_types['item_sparse'], seq_feat_list, 'int'))
        seq_feat_pre.update(build_dense(self.feature_types['item_continual'], seq_feat_list, 'float'))
        # user sparse/continual
        seq_feat_pre.update(build_dense(self.feature_types['user_sparse'], seq_feat_list, 'int'))
        seq_feat_pre.update(build_dense(self.feature_types['user_continual'], seq_feat_list, 'float'))
        # arrays
        seq_feat_pre.update(build_array(self.feature_types['item_array'], seq_feat_list))
        seq_feat_pre.update(build_array(self.feature_types['user_array'], seq_feat_list))
        # multimodal emb
        seq_feat_pre.update(build_emb(self.feature_types['item_emb'], seq_feat_list))


        return seq, token_type, seq_feat_pre, user_id, seq_timestamp


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                for json_file in base_path.glob('*.json'):
                    with open(json_file, 'r', encoding='utf-8') as file:
                        for line in file:
                            data_dict_origin = json.loads(line.strip())
                            insert_emb = data_dict_origin['emb']
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)
            except Exception as e:
                print(f"transfer error: {e}")
        if feat_id == '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                # 处理 part-* 格式的文件
                for part_file in base_path.glob('part-*'):
                    if part_file.name.startswith('part-') and not part_file.name.endswith('_SUCCESS'):
                        with open(part_file, 'r', encoding='utf-8') as file:
                            for line in file:
                                try:
                                    data_dict_origin = json.loads(line.strip())
                                    insert_emb = data_dict_origin['emb']
                                    if isinstance(insert_emb, list):
                                        insert_emb = np.array(insert_emb, dtype=np.float32)
                                    data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                                    emb_dict.update(data_dict)
                                except Exception as line_e:
                                    # 跳过无效的行
                                    continue
            except Exception as e:
                print(f"transfer error for feat_id {feat_id}: {e}")
                # 尝试加载 pkl 文件作为备选
                try:
                    with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                        emb_dict = pickle.load(f)
                except Exception as e2:
                    print(f"Failed to load both part files and pkl for feat_id {feat_id}: {e2}")
                    emb_dict = {}
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict

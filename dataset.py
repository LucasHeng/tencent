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
import sys

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
    
    # 移除共享内存相关变量，每个worker独立加载缓存

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
        self.indexer_u_rev = {v: k for k, v in indexer['i'].items()}
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
        self._save_path = save_path  # 保存save_path以便后续使用
        self._cache_name = 'cumu_cache.pkl'
        self._raw_stats_name = 'global_statistics.pkl'
        self._read_stats_name = None

        # 设置训练标志，用于调试信息
        self._is_training = True
        
        # 初始化全局统计信息
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



    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        self._ensure_data_file_open()
        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()            
        data = json.loads(line)
        return data
    

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
        
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据
        
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
                    i, timestamp,
                    user_stats_cache=user_stats_cache
                )
                print(item_stats)
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
        # 回填1304：使用"当前-前一条"的时间差，按固定边界分桶（前密后疏），非零分桶+1；首位为0
        edges = self._tdelta_edges
        # 仅对item位置计算"当前item-上一个item"的时间差；user位置置0
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

    def _calculate_item_statistics(self, item_id, current_timestamp, user_stats_cache=None):
        """
        计算item的统计特征（基于不同时间窗口的统计）
        
        Args:
            item_id: 物品ID
            current_timestamp: 当前时间戳
            user_stats_cache: 用户统计信息缓存
            
        Returns:
            stats: 包含统计特征的字典
            包含全局和用户的所有时间、1天、7天、30天的点击和曝光统计
        """
        if item_id == 0:  # padding item
            return {
                '2001': 0.0,  # 全局所有时间点击次数(log处理)
                '2002': 0.0,  # 全局所有时间曝光次数(log处理)
                '2003': 0.0,  # 全局所有时间点击率
                '2004': 0.0,  # 全局1天前点击次数(log处理)
                '2005': 0.0,  # 全局1天前曝光次数(log处理)
                '2006': 0.0,  # 全局1天前点击率
                '2007': 0.0,  # 全局7天前点击次数(log处理)
                '2008': 0.0,  # 全局7天前曝光次数(log处理)
                '2009': 0.0,  # 全局7天前点击率
                '2010': 0.0,  # 全局30天前点击次数(log处理)
                '2011': 0.0,  # 全局30天前曝光次数(log处理)
                '2012': 0.0,  # 全局30天前点击率
                '2013': 0.0,  # 用户所有时间点击次数(log处理)
                '2014': 0.0,  # 用户所有时间曝光次数(log处理)
                '2015': 0.0,  # 用户所有时间点击率
                '2016': 0.0,  # 用户1天前点击次数(log处理)
                '2017': 0.0,  # 用户1天前曝光次数(log处理)
                '2018': 0.0,  # 用户1天前点击率
                '2019': 0.0,  # 用户7天前点击次数(log处理)
                '2020': 0.0,  # 用户7天前曝光次数(log处理)
                '2021': 0.0,  # 用户7天前点击率
                '2022': 0.0,  # 用户30天前点击次数(log处理)
                '2023': 0.0,  # 用户30天前曝光次数(log处理)
                '2024': 0.0,  # 用户30天前点击率
                '2025': [0] * 10,    # 过去所有时间点击最多的前10个item ID数组
                '2026': [0] * 10,    # 过去所有时间曝光最多的前10个item ID数组
                '2027': [0] * 10,    # 过去1天点击最多的前10个item ID数组
                '2028': [0] * 10,    # 过去1天曝光最多的前10个item ID数组
                '2029': [0] * 10,    # 过去7天点击最多的前10个item ID数组
                '2030': [0] * 10,    # 过去7天曝光最多的前10个item ID数组
                '2031': [0] * 10,    # 过去30天点击最多的前10个item ID数组
                '2032': [0] * 10     # 过去30天曝光最多的前10个item ID数组
            }

        # 获取全局不同时间窗口的统计信息
        global_time_window_stats = self._get_item_statistics_with_time_windows(item_id, current_timestamp)
        
        # 提取全局各个时间窗口的统计
        global_all_time = global_time_window_stats['all_time']
        global_last_1_day = global_time_window_stats['last_1_day']
        global_last_7_days = global_time_window_stats['last_7_days']
        global_last_30_days = global_time_window_stats['last_30_days']
        
        # 计算全局各个时间窗口的点击率（避免除零）
        global_all_time_ctr = global_all_time['clicks'] / max(global_all_time['impressions'] + global_all_time['clicks'], 1)
        global_last_1_day_ctr = global_last_1_day['clicks'] / max(global_last_1_day['impressions'] + global_last_1_day['clicks'], 1)
        global_last_7_days_ctr = global_last_7_days['clicks'] / max(global_last_7_days['impressions'] + global_last_7_days['clicks'], 1)
        global_last_30_days_ctr = global_last_30_days['clicks'] / max(global_last_30_days['impressions'] + global_last_30_days['clicks'], 1)
        
        # 获取用户统计信息（事件数）
        user_all_time_clicks = 0
        user_all_time_impressions = 0
        user_last_1_day_clicks = 0
        user_last_1_day_impressions = 0
        user_last_7_days_clicks = 0
        user_last_7_days_impressions = 0
        user_last_30_days_clicks = 0
        user_last_30_days_impressions = 0
        
        if user_stats_cache and item_id in user_stats_cache:
            user_item_stats = user_stats_cache[item_id]
            user_all_time_clicks = user_item_stats['clicks']      # 事件数
            user_all_time_impressions = user_item_stats['impressions']  # 事件数
            
            # 注意：这里用户统计是基于事件数，不是时间窗口
            # 如果需要时间窗口的用户统计，需要额外的实现
            user_last_1_day_clicks = user_all_time_clicks  # 暂时使用所有时间的数据
            user_last_1_day_impressions = user_all_time_impressions
            user_last_7_days_clicks = user_all_time_clicks
            user_last_7_days_impressions = user_all_time_impressions
            user_last_30_days_clicks = user_all_time_clicks
            user_last_30_days_impressions = user_all_time_impressions
        
        # 计算用户各个时间窗口的点击率（避免除零）
        user_all_time_ctr = user_all_time_clicks / max(user_all_time_impressions + user_all_time_clicks, 1)
        user_last_1_day_ctr = user_last_1_day_clicks / max(user_last_1_day_impressions + user_last_1_day_clicks, 1)
        user_last_7_days_ctr = user_last_7_days_clicks / max(user_last_7_days_impressions + user_last_7_days_clicks, 1)
        user_last_30_days_ctr = user_last_30_days_clicks / max(user_last_30_days_impressions + user_last_30_days_clicks, 1)
        
        # 对点击数和曝光数进行log处理，缓解长尾问题
        # 使用 log(1 + x) 避免 log(0) 的问题
        
        # 全局统计的log处理
        log_global_all_time_clicks = np.log1p(global_all_time['clicks'])
        log_global_all_time_impressions = np.log1p(global_all_time['impressions'])
        log_global_1_day_clicks = np.log1p(global_last_1_day['clicks'])
        log_global_1_day_impressions = np.log1p(global_last_1_day['impressions'])
        log_global_7_days_clicks = np.log1p(global_last_7_days['clicks'])
        log_global_7_days_impressions = np.log1p(global_last_7_days['impressions'])
        log_global_30_days_clicks = np.log1p(global_last_30_days['clicks'])
        log_global_30_days_impressions = np.log1p(global_last_30_days['impressions'])
        
        # 用户统计的log处理
        log_user_all_time_clicks = np.log1p(user_all_time_clicks)
        log_user_all_time_impressions = np.log1p(user_all_time_impressions)
        log_user_1_day_clicks = np.log1p(user_last_1_day_clicks)
        log_user_1_day_impressions = np.log1p(user_last_1_day_impressions)
        log_user_7_days_clicks = np.log1p(user_last_7_days_clicks)
        log_user_7_days_impressions = np.log1p(user_last_7_days_impressions)
        log_user_30_days_clicks = np.log1p(user_last_30_days_clicks)
        log_user_30_days_impressions = np.log1p(user_last_30_days_impressions)
        
        # 获取多个时间窗口的热门item统计
        hot_items = self._get_hot_items_at_timestamp(current_timestamp)
        all_time_clicks_array = hot_items['all_time_clicks']
        all_time_impressions_array = hot_items['all_time_impressions']
        
        # 获取过去时间窗口的热门item统计（从新的缓存结构中获取）
        past_1_day_clicks = hot_items['past_1_day_clicks']
        past_1_day_impressions = hot_items['past_1_day_impressions']
        past_1_week_clicks = hot_items['past_1_week_clicks']
        past_1_week_impressions = hot_items['past_1_week_impressions']
        past_1_month_clicks = hot_items['past_1_month_clicks']
        past_1_month_impressions = hot_items['past_1_month_impressions']
        
        return {
            # 全局统计特征 (2001-2012)
            '2001': float(log_global_all_time_clicks),      # 全局所有时间点击次数(log处理)
            '2002': float(log_global_all_time_impressions), # 全局所有时间曝光次数(log处理)
            '2003': float(global_all_time_ctr),             # 全局所有时间点击率
            '2004': float(log_global_1_day_clicks),         # 全局1天前点击次数(log处理)
            '2005': float(log_global_1_day_impressions),    # 全局1天前曝光次数(log处理)
            '2006': float(global_last_1_day_ctr),           # 全局1天前点击率
            '2007': float(log_global_7_days_clicks),        # 全局7天前点击次数(log处理)
            '2008': float(log_global_7_days_impressions),   # 全局7天前曝光次数(log处理)
            '2009': float(global_last_7_days_ctr),          # 全局7天前点击率
            '2010': float(log_global_30_days_clicks),       # 全局30天前点击次数(log处理)
            '2011': float(log_global_30_days_impressions),  # 全局30天前曝光次数(log处理)
            '2012': float(global_last_30_days_ctr),         # 全局30天前点击率
            
            # 用户统计特征 (2013-2024)
            '2013': float(log_user_all_time_clicks),        # 用户所有时间点击次数(log处理)
            '2014': float(log_user_all_time_impressions),   # 用户所有时间曝光次数(log处理)
            '2015': float(user_all_time_ctr),               # 用户所有时间点击率
            '2016': float(log_user_1_day_clicks),           # 用户1天前点击次数(log处理)
            '2017': float(log_user_1_day_impressions),      # 用户1天前曝光次数(log处理)
            '2018': float(user_last_1_day_ctr),             # 用户1天前点击率
            '2019': float(log_user_7_days_clicks),          # 用户7天前点击次数(log处理)
            '2020': float(log_user_7_days_impressions),     # 用户7天前曝光次数(log处理)
            '2021': float(user_last_7_days_ctr),            # 用户7天前点击率
            '2022': float(log_user_30_days_clicks),         # 用户30天前点击次数(log处理)
            '2023': float(log_user_30_days_impressions),    # 用户30天前曝光次数(log处理)
            '2024': float(user_last_30_days_ctr),           # 用户30天前点击率
            
            # 过去所有时间热门item特征 (2025-2026) - 数组格式，每个包含10个item ID
            '2025': all_time_clicks_array,                  # 过去所有时间点击最多的前10个item ID数组
            '2026': all_time_impressions_array,             # 过去所有时间曝光最多的前10个item ID数组
            
            # 过去时间窗口热门item特征 (2027-2032) - 数组格式，每个包含10个item ID
            '2027': past_1_day_clicks,                      # 过去1天点击最多的前10个item ID数组
            '2028': past_1_day_impressions,                 # 过去1天曝光最多的前10个item ID数组
            '2029': past_1_week_clicks,                     # 过去7天点击最多的前10个item ID数组
            '2030': past_1_week_impressions,                # 过去7天曝光最多的前10个item ID数组
            '2031': past_1_month_clicks,                    # 过去30天点击最多的前10个item ID数组
            '2032': past_1_month_impressions                # 过去30天曝光最多的前10个item ID数组
        }

    def _precompute_global_statistics(self):
        """
        扫描训练集，构建按小时级别的原始增量统计：
        raw_stats: {item_id: {hour_timestamp: {'clicks': c, 'impressions': i}}}
        若提供 save_path，直接持久化 raw_stats。
        同时在内存构建累计缓存self._cumu_cache，供查询使用。
        """
        print(f"主进程 {os.getpid()} 开始预计算全局统计信息(小时级别时间戳增量结构)...")
        raw_stats = {}  # {item_id: {hour_timestamp: {'clicks': set(user_id), 'impressions': set(user_id)}}}
        hourly_stats = {}  # {hour_timestamp: {'clicks': {item_id: count}, 'impressions': {item_id: count}}}

        # 直接顺序读取文件，比根据uid seek更快
        print("直接顺序读取文件进行统计...")
        with open(self._data_file_path, 'rb') as data_file:
            for line_num, line in enumerate(tqdm(data_file, desc="扫描文件行")):
                try:
                    user_sequence = json.loads(line)
                    for record in user_sequence:
                        u, i, user_feat, item_feat, action_type, timestamp = record
                        if i and item_feat and action_type is not None:
                            # 将时间戳转换为小时级别（向下取整到小时）
                            hour_timestamp = (timestamp // 3600) * 3600
                            
                            # 初始化嵌套字典结构
                            if i not in raw_stats:
                                raw_stats[i] = {}
                            if hour_timestamp not in raw_stats[i]:
                                raw_stats[i][hour_timestamp] = {'clicks': set(), 'impressions': set()}
                            
                            # 初始化小时级别统计
                            if hour_timestamp not in hourly_stats:
                                hourly_stats[hour_timestamp] = {'clicks': {}, 'impressions': {}}
                            
                            # 将用户ID添加到对应的集合中，自动去重
                            user_id = int(u) if u is not None else 0
                            if action_type == 1:
                                raw_stats[i][hour_timestamp]['clicks'].add(user_id)
                                # 统计小时级别的点击次数
                                if i not in hourly_stats[hour_timestamp]['clicks']:
                                    hourly_stats[hour_timestamp]['clicks'][i] = 0
                                hourly_stats[hour_timestamp]['clicks'][i] += 1
                            else:
                                raw_stats[i][hour_timestamp]['impressions'].add(user_id)
                                # 统计小时级别的曝光次数
                                if i not in hourly_stats[hour_timestamp]['impressions']:
                                    hourly_stats[hour_timestamp]['impressions'][i] = 0
                                hourly_stats[hour_timestamp]['impressions'][i] += 1
                except json.JSONDecodeError:
                    print(f"第 {line_num} 行JSON解析错误: {line}")
                    # 静默处理JSON解析错误，继续下一行
                    continue
                except Exception as e:
                    print(f"处理第 {line_num} 行时出错: {e}")
                    continue

        # 持久化 raw_stats - 使用临时文件句柄
        if self._raw_stats_name is not None:
            save_path = Path(self._save_path) / self._raw_stats_name
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            # if save_path.exists():
            print(f"清除同名文件: {save_path}")
            clear_directory(self._save_path)
            payload = {'raw': raw_stats, 'hourly': hourly_stats}
            # 使用临时文件句柄，不作为成员变量
            with open(save_path, 'wb') as f:
                pickle.dump(payload, f)
            print(f"原始增量统计已保存到: {save_path}")

        # 保存hourly_stats供后续使用
        self._hourly_stats = hourly_stats
        
        # 构建累计缓存
        self._build_cumu_cache_from_raw(raw_stats, hourly_stats)
        
    
    def _build_cumu_cache_from_raw(self, raw_stats, hourly_stats=None):
        """从原始增量raw_stats构建累计缓存，便于按时间戳查询累计点击/曝光。"""
        cumu_cache = {}
        # 初始化最大值统计
        self._max_global_clicks = 0
        self._max_global_impressions = 0
        
        # 构建热门item统计
        self._hot_items_cache = {}  # {hour_timestamp: {'top_clicks': item_id, 'top_impressions': item_id}}
        
        for item_id, ts_dict in raw_stats.items():
            if not ts_dict:
                continue
                
            ts_sorted = sorted(ts_dict.keys())
            # 使用更紧凑的数据类型：int32 替代 int64，节省一半内存
            clicks = np.zeros(len(ts_sorted), dtype=np.int32)
            imprs = np.zeros(len(ts_sorted), dtype=np.int32)
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

            # 使用numpy数组存储时间戳，提高访问效率
            cumu_cache[item_id] = (np.array(ts_sorted, dtype=np.int64), clicks, imprs)
        
        print(f"统计完成 - 最大点击用户数: {self._max_global_clicks}, 最大曝光用户数: {self._max_global_impressions}")
        
        # 构建热门item统计 - 为每个时间戳构建多个时间窗口的累计热门item
        if hourly_stats:
            # 按时间戳排序
            sorted_timestamps = sorted(hourly_stats.keys())
            
            for hour_timestamp in sorted_timestamps:
                # 计算各个时间窗口的起始时间戳
                past_1_hour = hour_timestamp - 1 * 3600  # 1小时前
                past_1_day = hour_timestamp - 24 * 3600  # 24小时前
                past_1_week = hour_timestamp - 7 * 24 * 3600  # 7天前
                past_1_month = hour_timestamp - 30 * 24 * 3600  # 30天前
                all_time_start = sorted_timestamps[0]  # 所有时间的开始
                
                # 统计各个时间窗口的累计点击和曝光
                def get_hot_items_for_window(start_ts, end_ts):
                    item_clicks = {}
                    item_impressions = {}
                    
                    # 遍历时间窗口内的所有小时
                    for ts in sorted_timestamps:
                        if start_ts <= ts < end_ts:
                            stats = hourly_stats[ts]
                            # 累加点击次数
                            for item_id, clicks in stats.get('clicks', {}).items():
                                item_clicks[item_id] = item_clicks.get(item_id, 0) + clicks
                            # 累加曝光次数
                            for item_id, impressions in stats.get('impressions', {}).items():
                                item_impressions[item_id] = item_impressions.get(item_id, 0) + impressions
                    
                    # 获取前10名
                    top_clicks = sorted(item_clicks.items(), key=lambda x: x[1], reverse=True)[:10]
                    top_clicks_array = [item_id for item_id, count in top_clicks]
                    while len(top_clicks_array) < 10:
                        top_clicks_array.append(0)
                    
                    top_impressions = sorted(item_impressions.items(), key=lambda x: x[1], reverse=True)[:10]
                    top_impressions_array = [item_id for item_id, count in top_impressions]
                    while len(top_impressions_array) < 10:
                        top_impressions_array.append(0)
                    
                    return top_clicks_array, top_impressions_array
                
                # 获取各个时间窗口的热门item
                past_1_hour_clicks, past_1_hour_impressions = get_hot_items_for_window(past_1_hour, hour_timestamp)
                past_1_day_clicks, past_1_day_impressions = get_hot_items_for_window(past_1_day, hour_timestamp)
                past_1_week_clicks, past_1_week_impressions = get_hot_items_for_window(past_1_week, hour_timestamp)
                past_1_month_clicks, past_1_month_impressions = get_hot_items_for_window(past_1_month, hour_timestamp)
                all_time_clicks, all_time_impressions = get_hot_items_for_window(all_time_start, hour_timestamp)
                
                # 存储该时间戳的多个时间窗口热门item
                self._hot_items_cache[hour_timestamp] = {
                    'past_1_hour_clicks': past_1_hour_clicks,
                    'past_1_hour_impressions': past_1_hour_impressions,
                    'past_1_day_clicks': past_1_day_clicks,
                    'past_1_day_impressions': past_1_day_impressions,
                    'past_1_week_clicks': past_1_week_clicks,
                    'past_1_week_impressions': past_1_week_impressions,
                    'past_1_month_clicks': past_1_month_clicks,
                    'past_1_month_impressions': past_1_month_impressions,
                    'all_time_clicks': all_time_clicks,
                    'all_time_impressions': all_time_impressions
                }
        
        # 计算内存占用（传入临时变量计算）
        memory_usage = self.get_cumu_cache_memory_usage_from_dict(cumu_cache)
        print(f"累计缓存内存占用: {memory_usage / 1024 / 1024:.2f} MB")
        
        # 设置缓存（主进程构建后释放，子进程保持）
        self._cumu_cache = cumu_cache
        
        # 持久化累计缓存
        if hasattr(self, '_save_path') and self._save_path:
            self._save_cumu_cache(cumu_cache)
            # 保存完成后释放内存（主进程专用）
            if os.getpid() == self.main_pid:
                print(f"主进程：持久化完成，释放缓存内存")
                self._cumu_cache = {}
    
    def get_cumu_cache_memory_usage_from_dict(self, cumu_cache_dict):
        """计算传入字典的内存占用（用于临时变量）"""
        if not cumu_cache_dict:
            return 0
        
        total_memory = 0
        
        # 基础字典大小
        total_memory += sys.getsizeof(cumu_cache_dict)
        
        for item_id, (timestamps, clicks, impressions) in cumu_cache_dict.items():
            # 每个item的元组大小
            total_memory += sys.getsizeof((timestamps, clicks, impressions))
            
            # 时间戳数组大小
            total_memory += timestamps.nbytes if hasattr(timestamps, 'nbytes') else sys.getsizeof(timestamps)
            
            # 点击数组大小
            total_memory += clicks.nbytes if hasattr(clicks, 'nbytes') else sys.getsizeof(clicks)
            
            # 曝光数组大小
            total_memory += impressions.nbytes if hasattr(impressions, 'nbytes') else sys.getsizeof(impressions)
        
        return total_memory
    


    def _ensure_statistics_loaded(self):
        """
        确保统计信息已加载
        """
        if not hasattr(self, '_cumu_cache') or not self._cumu_cache:
            print(f"进程 {os.getpid()} 加载统计信息")
            self._load_global_statistics()

    def _ensure_data_file_open(self):
        """
        确保数据文件已打开
        """
        if not hasattr(self, 'data_file') or self.data_file is None:
            self.data_file = open(self.data_dir / "seq.jsonl", 'r', encoding='utf-8')

    def _save_cumu_cache(self, cumu_cache):
        """保存累计缓存到文件"""
        if not cumu_cache:
            return
        
        save_dir = Path(self._save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        cache_file = save_dir / self._cache_name
        
        # 删除同名文件
        if cache_file.exists():
            try:
                cache_file.unlink()
            except Exception:
                pass
        
        # 保存累计缓存、热门item缓存和小时统计
        cache_data = {
            'cumu_cache': cumu_cache,
            'hot_items_cache': getattr(self, '_hot_items_cache', {}),
            'hourly_stats': getattr(self, '_hourly_stats', {})
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"主进程 {os.getpid()} 累计缓存和热门item缓存已保存到: {cache_file}")

    def _load_raw_statistics(self, load_path):
        """加载原始统计信息"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and 'raw' in data:
            raw_stats = data['raw']
            hourly_stats = data.get('hourly', {})
            print(f"原始统计信息加载成功，包含 {len(raw_stats)} 个item的统计信息")
            print(f"小时级别统计信息加载成功，包含 {len(hourly_stats)} 个时间点的热门item统计")
            return raw_stats, hourly_stats
        else:
            # 兼容旧格式
            raw_stats = data
            print(f"原始统计信息加载成功，包含 {len(raw_stats)} 个item的统计信息")
            return raw_stats, {}

    def _load_global_statistics(self):
        """
        从文件加载累计缓存
        主进程和子进程都会调用此方法
        """
        save_dir = Path(self._save_path) if hasattr(self, '_save_path') else None
        if not save_dir:
            print("警告：没有设置save_path，无法加载统计信息")
            return {}
        
        # 加载累计缓存文件
        cumu_cache_file = save_dir / self._cache_name
        if cumu_cache_file.exists():
            current_pid = os.getpid()
            is_main = (current_pid == self.main_pid)
            process_type = "主进程" if is_main else f"子进程 {current_pid}"
            
            print(f"{process_type} 加载累计缓存文件: {cumu_cache_file}")
            try:
                with open(cumu_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # 加载新格式缓存数据
                self._cumu_cache = cache_data['cumu_cache']
                self._hot_items_cache = cache_data.get('hot_items_cache', {})
                self._hourly_stats = cache_data.get('hourly_stats', {})
                print(f"{process_type} 成功加载累计缓存、热门item缓存和小时统计")
                
                print(f"{process_type} 累计缓存包含 {len(self._cumu_cache)} 个item")
                print(f"{process_type} 热门item缓存包含 {len(self._hot_items_cache)} 个时间点")
                
                # 计算内存占用
                if hasattr(self, '_cumu_cache') and self._cumu_cache:
                    memory_usage = self.get_cumu_cache_memory_usage_from_dict(self._cumu_cache)
                    print(f"{process_type} 累计缓存内存占用: {memory_usage / 1024 / 1024:.2f} MB")
                
                return True
            except Exception as e:
                print(f"{process_type} 加载累计缓存失败: {e}")
                return False
        else:
            current_pid = os.getpid()
            is_main = (current_pid == self.main_pid)
            process_type = "主进程" if is_main else f"子进程 {current_pid}"
            print(f"{process_type}：没有发现累计缓存文件 {cumu_cache_file}")
            return False

    def _get_item_statistics_at_timestamp(self, item_id, timestamp):
        """
        从累计缓存中获取指定item在指定时间戳之前的累计统计（clicks, impressions）。
        查询逻辑：将时间戳转换为小时级别，查询该小时之前的累计统计，不包括当前小时。
        若无缓存/条目，返回0值。
        """
        # 如果主进程的缓存被释放了，需要重新加载
        if not hasattr(self, '_cumu_cache') or not self._cumu_cache:
            if os.getpid() == self.main_pid:
                print(f"主进程：缓存已释放，重新加载统计信息")
                self._load_global_statistics()
        
        cache = self._cumu_cache.get(item_id)
        if cache is None:
            return {'clicks': 0, 'impressions': 0}
        
        timestamps, clicks_cumu, imprs_cumu = cache
        
        # 添加调试信息
        if len(timestamps) == 0 or len(clicks_cumu) == 0 or len(imprs_cumu) == 0:
            print(f"警告：item {item_id} 的缓存数组为空")
            return {'clicks': 0, 'impressions': 0}
        
        # 将时间戳转换为小时级别（向下取整到小时）
        hour_timestamp = (timestamp // 3600) * 3600
        
        # 查询该小时之前的累计统计（不包括当前小时）
        # 使用numpy的searchsorted进行二分查找，比Python bisect更快
        # 时间复杂度: O(log n)
        idx = np.searchsorted(timestamps, hour_timestamp, side='left')
        
        if idx == 0:
            # 当前小时小于所有记录时间戳，返回0
            return {'clicks': 0, 'impressions': 0}
        elif idx >= len(timestamps):
            # 当前小时大于等于所有记录时间戳，返回最后一个累计值
            return {'clicks': int(clicks_cumu[-1]), 'impressions': int(imprs_cumu[-1])}
        else:
            # 返回前一个位置的累计值（即小于当前小时的最后一个值）
            if idx - 1 < 0 or idx - 1 >= len(clicks_cumu):
                print(f"索引错误：item_id={item_id}, idx={idx}, len(clicks_cumu)={len(clicks_cumu)}, hour_timestamp={hour_timestamp}")
                return {'clicks': 0, 'impressions': 0}
            return {'clicks': int(clicks_cumu[idx-1]), 'impressions': int(imprs_cumu[idx-1])}

    def _get_item_statistics_with_time_windows(self, item_id, timestamp):
        """
        获取指定item在不同时间窗口的统计信息
        
        Args:
            item_id: 物品ID
            timestamp: 当前时间戳
            
        Returns:
            dict: 包含不同时间窗口的统计信息
        """
        # 如果主进程的缓存被释放了，需要重新加载
        if not hasattr(self, '_cumu_cache') or not self._cumu_cache:
            if os.getpid() != self.main_pid:
                print(f"主进程：缓存已释放，重新加载统计信息")
                self._load_global_statistics()
        
        cache = self._cumu_cache.get(item_id)
        if cache is None:
            return {
                'all_time': {'clicks': 0, 'impressions': 0},
                'last_1_day': {'clicks': 0, 'impressions': 0},
                'last_7_days': {'clicks': 0, 'impressions': 0},
                'last_30_days': {'clicks': 0, 'impressions': 0}
            }
        
        timestamps, clicks_cumu, imprs_cumu = cache
        
        if len(timestamps) == 0 or len(clicks_cumu) == 0 or len(imprs_cumu) == 0:
            return {
                'all_time': {'clicks': 0, 'impressions': 0},
                'last_1_day': {'clicks': 0, 'impressions': 0},
                'last_7_days': {'clicks': 0, 'impressions': 0},
                'last_30_days': {'clicks': 0, 'impressions': 0}
            }
        
        # 计算不同时间窗口的截止时间戳
        current_hour = (timestamp // 3600) * 3600
        one_day_ago = current_hour - 24 * 3600  # 24小时前
        seven_days_ago = current_hour - 7 * 24 * 3600  # 7天前
        thirty_days_ago = current_hour - 30 * 24 * 3600  # 30天前
        
        # 获取所有时间的统计（当前小时之前的所有数据）
        all_time_idx = np.searchsorted(timestamps, current_hour, side='left')
        all_time_clicks = int(clicks_cumu[all_time_idx - 1]) if all_time_idx > 0 else 0
        all_time_impressions = int(imprs_cumu[all_time_idx - 1]) if all_time_idx > 0 else 0
        
        # 获取1天前的统计
        one_day_idx = np.searchsorted(timestamps, one_day_ago, side='left')
        one_day_clicks = int(clicks_cumu[one_day_idx - 1]) if one_day_idx > 0 else 0
        one_day_impressions = int(imprs_cumu[one_day_idx - 1]) if one_day_idx > 0 else 0
        
        # 获取7天前的统计
        seven_days_idx = np.searchsorted(timestamps, seven_days_ago, side='left')
        seven_days_clicks = int(clicks_cumu[seven_days_idx - 1]) if seven_days_idx > 0 else 0
        seven_days_impressions = int(imprs_cumu[seven_days_idx - 1]) if seven_days_idx > 0 else 0
        
        # 获取30天前的统计
        thirty_days_idx = np.searchsorted(timestamps, thirty_days_ago, side='left')
        thirty_days_clicks = int(clicks_cumu[thirty_days_idx - 1]) if thirty_days_idx > 0 else 0
        thirty_days_impressions = int(imprs_cumu[thirty_days_idx - 1]) if thirty_days_idx > 0 else 0
        
        # 计算各个时间窗口的统计（当前时间 - 窗口开始时间）
        last_1_day_clicks = all_time_clicks - one_day_clicks
        last_1_day_impressions = all_time_impressions - one_day_impressions
        
        last_7_days_clicks = all_time_clicks - seven_days_clicks
        last_7_days_impressions = all_time_impressions - seven_days_impressions
        
        last_30_days_clicks = all_time_clicks - thirty_days_clicks
        last_30_days_impressions = all_time_impressions - thirty_days_impressions
        
        return {
            'all_time': {'clicks': all_time_clicks, 'impressions': all_time_impressions},
            'last_1_day': {'clicks': last_1_day_clicks, 'impressions': last_1_day_impressions},
            'last_7_days': {'clicks': last_7_days_clicks, 'impressions': last_7_days_impressions},
            'last_30_days': {'clicks': last_30_days_clicks, 'impressions': last_30_days_impressions}
        }

    def _get_hot_items_at_timestamp(self, timestamp):
        """
        获取指定时间戳的多个时间窗口热门item
        
        Args:
            timestamp: 当前时间戳
            
        Returns:
            dict: 包含多个时间窗口热门item的信息
        """
        # 确保热门item缓存已加载
        if not hasattr(self, '_hot_items_cache') or not self._hot_items_cache:
            print(f"进程 {os.getpid()}：热门item缓存未加载，重新加载统计信息")
            self._load_global_statistics()
        
        # 将时间戳转换为小时级别（向下取整到小时）
        hour_timestamp = (timestamp // 3600) * 3600
        
        # 获取该小时的多个时间窗口热门item
        hot_items = self._hot_items_cache.get(hour_timestamp, {
            'past_1_hour_clicks': [0] * 10,
            'past_1_hour_impressions': [0] * 10,
            'past_1_day_clicks': [0] * 10,
            'past_1_day_impressions': [0] * 10,
            'past_1_week_clicks': [0] * 10,
            'past_1_week_impressions': [0] * 10,
            'past_1_month_clicks': [0] * 10,
            'past_1_month_impressions': [0] * 10,
            'all_time_clicks': [0] * 10,
            'all_time_impressions': [0] * 10
        })
        
        return hot_items


        """
        基于hourly_stats获取指定时间戳过去时间窗口的热门item统计
        
        Args:
            timestamp: 当前时间戳
            
        Returns:
            dict: 包含过去时间窗口热门item的信息
        """
        # 确保统计信息已加载
        if not hasattr(self, '_hourly_stats') or not self._hourly_stats:
            print(f"进程 {os.getpid()}：hourly_stats未加载，重新加载统计信息")
            self._load_global_statistics()
        
        # 将时间戳转换为小时级别（向下取整到小时）
        hour_timestamp = (timestamp // 3600) * 3600
        
        # 计算过去时间窗口的截止时间戳
        past_1_day = hour_timestamp - 24 * 3600  # 24小时前
        past_7_days = hour_timestamp - 7 * 24 * 3600  # 7天前
        past_30_days = hour_timestamp - 30 * 24 * 3600  # 30天前
        
        # 统计过去时间窗口内每个item的总点击和曝光次数
        def get_past_hot_items(start_timestamp, end_timestamp):
            item_clicks = {}    # {item_id: total_clicks}
            item_impressions = {}  # {item_id: total_impressions}
            
            # 遍历时间窗口内的每个小时
            for ts in range(start_timestamp, end_timestamp, 3600):
                if ts in self._hourly_stats:
                    # 获取该小时的真实点击和曝光统计
                    stats = self._hourly_stats[ts]
                    
                    # 累加点击次数
                    for item_id, clicks in stats.get('clicks', {}).items():
                        item_clicks[item_id] = item_clicks.get(item_id, 0) + clicks
                    
                    # 累加曝光次数
                    for item_id, impressions in stats.get('impressions', {}).items():
                        item_impressions[item_id] = item_impressions.get(item_id, 0) + impressions
            
            # 获取点击次数最多的前10个item
            top_clicks = sorted(item_clicks.items(), key=lambda x: x[1], reverse=True)[:10]
            top_clicks_result = [item_id for item_id, count in top_clicks]
            while len(top_clicks_result) < 10:
                top_clicks_result.append(0)
            
            # 获取曝光次数最多的前10个item
            top_impressions = sorted(item_impressions.items(), key=lambda x: x[1], reverse=True)[:10]
            top_impressions_result = [item_id for item_id, count in top_impressions]
            while len(top_impressions_result) < 10:
                top_impressions_result.append(0)
            
            return top_clicks_result, top_impressions_result
        
        # 获取过去1天的热门item
        past_1_day_clicks, past_1_day_impressions = get_past_hot_items(past_1_day, hour_timestamp)
        
        # 获取过去7天的热门item
        past_7_days_clicks, past_7_days_impressions = get_past_hot_items(past_7_days, hour_timestamp)
        
        # 获取过去30天的热门item
        past_30_days_clicks, past_30_days_impressions = get_past_hot_items(past_30_days, hour_timestamp)
        

    def _init_global_statistics(self):
        """
        初始化全局统计信息
        
        主进程构建统计信息并保存到文件，子进程从文件加载统计信息
        """
        is_test_dataset = self.__class__.__name__ == 'MyTestDataset'
        is_main_process = (os.getpid() == self.main_pid)

        if is_test_dataset:
            # 测试数据集：加载训练统计->增量更新->再构建累计缓存
            print(f"测试数据集模式：从 {self._save_path} 加载训练集统计信息")
            if self._read_stats_name is not None:
                stats_file = Path(self._save_path) / self._read_stats_name
                if stats_file.exists():
                    print(f"加载训练集统计信息: {stats_file}")
                    global_stats, hourly_stats = self._load_raw_statistics(stats_file)
                    # 遍历测试数据更新统计信息（仅内存），不重建缓存
                    self._update_statistics_from_test_data(self._save_path, global_stats)
                    # 测试数据更新完成后统一构建累计缓存
                    self._build_cumu_cache_from_raw(global_stats, hourly_stats)
                else:
                    print(f"警告：训练集统计信息文件不存在 {stats_file}，将使用空统计信息")
        else:
            # 训练集：主进程构建统计信息，子进程从文件加载
            cache_file = Path(self._save_path) / self._cache_name
            
            if is_main_process:
                # 主进程：检查是否需要构建统计信息
                print("主进程：自动生成全局统计信息...")
                self._precompute_global_statistics()
            else:
                # 子进程：从文件加载统计信息
                print(f"子进程 {os.getpid()}：从文件加载统计信息")
                self._load_global_statistics()

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
                            # 将时间戳转换为小时级别（向下取整到小时）
                            hour_timestamp = (timestamp // 3600) * 3600
                            evt_key = (int(u) if u is not None else 0, int(action_type))
                            # 直接更新统计信息，使用集合自动去重
                            # 写入raw增量，不排序不重建
                            if item_id not in global_stats:
                                global_stats[item_id] = {}
                                new_item_count += 1
                            if hour_timestamp not in global_stats[item_id]:
                                global_stats[item_id][hour_timestamp] = {'clicks': set(), 'impressions': set()}
                            
                            inc = global_stats[item_id][hour_timestamp]
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



    def _ensure_statistics_loaded(self):
        """
        确保统计信息已加载，用于子进程延迟加载
        """
        if not hasattr(self, '_cumu_cache') or not self._cumu_cache:
            # 检查是否是子进程
            if hasattr(self, 'main_pid') and os.getpid() != self.main_pid:
                print(f"子进程 {os.getpid()} 延迟加载统计信息")
                self._load_global_statistics()
            else:
                print(f"主进程 {os.getpid()} 加载统计信息")
                self._load_global_statistics()



class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args, save_path):
        super().__init__(data_dir, args, save_path)
        
        # 如果save_path为None，使用data_dir作为默认值
        if save_path is None:
            save_path = self.data_dir

        self._cache_name = 'test_cumu_cache.pkl'
        self._raw_stats_name = None
        self._read_stats_name = 'global_statistics.pkl'
        
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

        
        user_sequence = self._load_user_data(uid)  # 使用临时文件句柄，避免多进程冲突

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
                    i, timestamp,
                    user_stats_cache=user_stats_cache
                )
                print(item_stats)
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
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict
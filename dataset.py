import json
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
import math
import random
import os
import atexit
import pandas as pd
from datetime import datetime

add_feat_types = {}
# 特征
user_feat_types = {}
item_feat_types= {}

user_feat_types['user_sparse'] = ['103', '104', '105', '109']
item_feat_types['item_sparse'] = [
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
item_feat_types['item_array'] = []
user_feat_types['user_array'] = ['106', '107', '108', '110']
user_feat_types['user_continual'] = []
item_feat_types['item_continual'] = []

add_feat_types['item_manual'] = [
    '99',
    # '98',
    # '97',
    # '96',
]

# # 全局变量（每个worker进程独立拥有一份）
# # 用于缓存文件句柄：{文件路径: 文件句柄}
# _worker_file_cache = {}

# 全局变量：每个worker进程单独存储自己的句柄（进程内可见）
worker_local = {}  # 键：worker_id，值：当前worker的句柄
# Worker初始化函数：为每个worker创建独立句柄
def worker_init_fn(worker_id):
    """
    DataLoader的worker初始化函数
    
    参数:
        worker_id: 当前worker的唯一ID（0, 1, ..., num_workers-1）
    """
    data_path = os.environ.get('TRAIN_DATA_PATH')
    # 为当前worker创建独立句柄
    handle = open(Path(data_path, "seq.jsonl"), 'rb')
    # 存储到全局变量（仅当前worker进程可见）
    worker_local[worker_id] = handle

    dataset = torch.utils.data.get_worker_info().dataset
    with open(Path(data_path, 'indexer.pkl'), 'rb') as ff:
        indexer = pickle.load(ff)
        dataset.itemnum = len(indexer['i'])
        dataset.usernum = len(indexer['u'])
        # print(f'itemnum {self.itemnum} and usernum {self.usernum}')
    # itemid 与 reconstruct id
    dataset.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
    dataset.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
    dataset.indexer = indexer
    # print(worker_id)

    # 注册退出时的清理函数
    import atexit
    atexit.register(handle.close)
    # 设置当前worker的所有随机种子
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)
    torch.manual_seed(42 + worker_id)
    # 若使用CUDA，还需设置cuda种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + worker_id)

def _close_file_handles():
    """worker进程退出时关闭所有缓存的文件句柄"""
    global worker_local
    for handle in worker_local.values():
        if not handle.closed:
            handle.close()
    worker_local.clear()

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

    def __init__(self, data_dir, args):
        """
        初始化数据集
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        print(f'{self.data_dir}')
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
            # print(f'itemnum {self.itemnum} and usernum {self.usernum}')
        # # itemid 与 reconstruct id
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer
        self.sample_neg_num = args.sample_neg_num

        self.feature_default_value, self.feature_types, self.feat_statistics, self.add_feat_types = self._init_feat_info()
        self.file_path = Path(self.data_dir, "seq.jsonl")
        # 确保每个worker进程只注册一次清理函数
        if not hasattr(MyDataset, '_cleanup_registered'):
            atexit.register(_close_file_handles)
            MyDataset._cleanup_registered = True

    def convert_timestamp_to_day_offset(self, timestamp, base_date_str='2024-10-31'):
        """
        将时间戳转换为以基准日期为第0天的天数偏移
        
        参数:
            timestamp: 秒级时间戳（int）
            base_date_str: 基准日期字符串，格式'YYYY-MM-DD'，默认为'2024-10-15'
        
        返回:
            天数偏移（int）：基准日期及之前为0，之后为实际天数差
        """
        
        # 1. 转换时间戳为北京时间的日期
        pdate = pd.to_datetime(timestamp, unit='s') \
                        .tz_localize('UTC') \
                        .tz_convert('Asia/Shanghai') \
                        
        phour = pdate.hour
        
        beijing_date = pdate.date()

        # 2. 解析基准日期
        base_date = datetime.strptime(base_date_str, '%Y-%m-%d').date()
        
        # 3. 计算天数差，基准日期及之前都返回0
        delta_days = (beijing_date - base_date).days
        return min(max(0, delta_days),239), phour

    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        self.data_file = open(self.data_dir / "seq.jsonl", 'rb')
        worker_local[0] = self.data_file
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _get_file_handle(self):
        """每个worker进程只打开一次文件，返回复用的句柄"""
        # 检查当前进程是否已打开文件
        # 若当前worker未打开文件，则打开并缓存句柄

        datafile = open(self.file_path, 'r')
        print(f"Worker {os.getpid()} 打开文件: {self.file_path}")  # 调试用
        return  datafile

    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        # 获取当前worker的ID（从进程名提取，如"DataLoader worker 0"）
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0  # 主进程中worker_id为0
        # print(f'worker_id: {worker_id}')
        data_file = worker_local[worker_id]
        data_file.seek(self.seq_offsets[uid])
        line = data_file.readline()
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

    def cl4srec_aug(self, cur_data):
        def item_crop(seq, length, eta=0.6):
            num_left = math.floor(length * eta)
            crop_begin = random.randint(0, length - num_left)
            croped_item_seq = np.zeros(seq.shape[0])
            if crop_begin + num_left < seq.shape[0]:
                croped_item_seq[:num_left] = seq[crop_begin:crop_begin + num_left]
            else:
                croped_item_seq[:num_left] = seq[crop_begin:]
            return torch.tensor(croped_item_seq, dtype=torch.long), torch.tensor(num_left, dtype=torch.long)
        
        def item_mask(seq, length, gamma=0.3):
            num_mask = math.floor(length * gamma)
            mask_index = random.sample(range(length), k=num_mask)
            masked_item_seq = seq[:]
            masked_item_seq[mask_index] = self.dataset.item_num  # token 0 has been used for semantic masking
            return masked_item_seq, length
        
        def item_reorder(seq, length, beta=0.6):
            num_reorder = math.floor(length * beta)
            reorder_begin = random.randint(0, length - num_reorder)
            reordered_item_seq = seq[:]
            shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
            random.shuffle(shuffle_index)
            reordered_item_seq[reorder_begin:reorder_begin + num_reorder] = reordered_item_seq[shuffle_index]
            return reordered_item_seq, length
        
        seqs = cur_data['item_id_list']
        lengths = cur_data['item_length']

        aug_seq1 = []
        aug_len1 = []
        aug_seq2 = []
        aug_len2 = []
        for seq, length in zip(seqs, lengths):
            if length > 1:
                switch = random.sample(range(3), k=2)
            else:
                switch = [3, 3]
                aug_seq = seq
                aug_len = length
            if switch[0] == 0:
                aug_seq, aug_len = item_crop(seq, length)
            elif switch[0] == 1:
                aug_seq, aug_len = item_mask(seq, length)
            elif switch[0] == 2:
                aug_seq, aug_len = item_reorder(seq, length)
    
            aug_seq1.append(aug_seq)
            aug_len1.append(aug_len)
    
            if switch[1] == 0:
                aug_seq, aug_len = item_crop(seq, length)
            elif switch[1] == 1:
                aug_seq, aug_len = item_mask(seq, length)
            elif switch[1] == 2:
                aug_seq, aug_len = item_reorder(seq, length)
    
            aug_seq2.append(aug_seq)
            aug_len2.append(aug_len)


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

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1, self.sample_neg_num], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        user_feat = np.empty([1], dtype=object)
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
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt

            if type_ == 1:
                if act_type == None:
                    act_type = -1
                feat['99'] = act_type + 1

            feat = self.fill_missing_feat(feat, i, type_)
            next_feat = self.fill_missing_feat(next_feat, next_i, next_type)
            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            if type_ == 1:
                seq_feat[idx] = feat
            else:
                user_feat = feat
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                neg_ids = self._random_neq(1, self.itemnum + 1, ts, self.sample_neg_num)
                neg[idx] = neg_ids
                for id,neg_id in enumerate(neg_ids):
                    negfeat = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id, 1)
                    neg_feat[idx][id] = negfeat
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, user_feat

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

        feat_types = item_feat_types | user_feat_types
        feat_types['item_emb'] = self.mm_emb_ids
        self.item_feat_types = item_feat_types
        self.item_feat_types['item_emb'] = self.mm_emb_ids
        self.user_feat_types = user_feat_types

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
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )
        for feat_id in add_feat_types['item_manual']:
            feat_default_value[feat_id] = 0
        feat_statistics['99'] = 2
        # feat_statistics['98'] = 240
        # feat_statistics['97'] = 24
        # feat_statistics['96'] = 240
        print(f'{feat_statistics}')

        return feat_default_value, feat_types, feat_statistics, add_feat_types

    def feat2tensor(self, seq_feature, k):
        """
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典，形状为 [batch_size, maxlen]
            k: 特征ID

        Returns:
            batch_data: 特征值的tensor，形状为 [batch_size, maxlen, max_array_len(if array)]
        """
        batch_size = len(seq_feature)

        if k in self.item_feat_types or k in self.add_feat_types:
            # 处理某一个特征, 如果是array类型
            # print(f'process {k}')
            if k in self.ITEM_ARRAY_FEAT:
                # 如果特征是Array类型，需要先对array进行padding，然后转换为tensor
                # 序列长度最大值, 这个特征数组的最长值
                max_array_len = 0
                max_seq_len = 0

                # 每一个item是一个用户的序列
                for i in range(batch_size):
                    # 取其中某个feat的特征序列
                    seq_data = [item[k] for item in seq_feature[i]]
                    max_seq_len = max(max_seq_len, len(seq_data))
                    max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))

                batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
                for i in range(batch_size):
                    seq_data = [item[k] for item in seq_feature[i]]
                    for j, item_data in enumerate(seq_data):
                        actual_len = min(len(item_data), max_array_len)
                        batch_data[i, j, :actual_len] = item_data[:actual_len]
                # 后面补充为0

                return torch.from_numpy(batch_data).to(self.dev)
            else:
                # 如果特征是Sparse类型，直接转换为tensor
                max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
                batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)

                for i in range(batch_size):
                    seq_data = [item[k] for item in seq_feature[i]]
                    batch_data[i] = seq_data
            return torch.from_numpy(batch_data).to(self.dev)
        elif k in self.user_feat_types:
            # 处理某一个特征, 如果是array类型
            # print(f"processing {k}")
            if k in self.USER_ARRAY_FEAT:
                # 如果特征是Array类型，需要先对array进行padding，然后转换为tensor
                # 序列长度最大值, 这个特征数组的最长值
                max_array_len = 0
                max_seq_len = 0

                # 每一个item是一个用户的序列
                for i in range(batch_size):
                    # 取其中某个feat的特征序列
                    max_array_len = max(max_array_len, len(seq_feature[i][k]))

                batch_data = np.zeros((batch_size, max_array_len), dtype=np.int64)
                for i in range(batch_size):
                    item_data = seq_feature[i][k]
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, :actual_len] = item_data[:actual_len]
                # 后面补充为0
                return torch.from_numpy(batch_data).to(self.dev)
            else:
                # 如果特征是Sparse类型，直接转换为tensor
                batch_data = np.zeros((batch_size), dtype=np.int64)

                for i in range(batch_size):
                    seq_data =  seq_feature[i][k]
                    # if k == '99':
                    #     print(f'seq_data: {i} {k} {seq_data}')
                    batch_data[i] = seq_data

            return torch.from_numpy(batch_data).to(self.dev)


    def feat2emb(self, seq, feature_array, user_feat=None, mask=None, include_user=False):
        """
        Args:
            seq: 序列ID
            feature_array: 特征list，每个元素为当前时刻的特征字典
            mask: 掩码，1表示item，2表示user
            include_user: 是否处理用户特征，在两种情况下不打开：1) 训练时在转换正负样本的特征时（因为正负样本都是item）;2) 生成候选库item embedding时。

        Returns:
            seqs_emb: 序列特征的Embedding
        """
        seq = seq.to(self.dev)
        # pre-compute embedding
        if include_user:
            user_mask = np.where(mask == 2)
            # user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            # [batch_size, seq_len]
            user_embedding = self.user_emb(seq[user_mask])
            item_embedding = self.item_emb(item_mask * seq)
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            # [batch_size, embed_dim]
            item_embedding = self.item_emb(seq)
            item_feat_list = [item_embedding]

        # batch-process all feature types
        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list),
        ]
        user_feat_types = []
        if include_user:
            user_feat_types.extend(
                [
                    (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                    (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                    (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
                ]
            )
            all_feat_types.extend(
                [(self.ITEM_MANUAL_FEAT, 'item_manual', item_feat_list),]
            )

        add_feat_list = []
        # batch-process each feature type
        for feat_dict, feat_type, feat_list in user_feat_types:
            if not feat_dict:
                continue

            for k in feat_dict:
                tensor_feature = self.feat2tensor(user_feat, k)
                # print(f'item_shape {k} {tensor_feature.shape}')

                # [batch_size, max_len] -> [batch_size, max_len, emb_dim]
                if feat_type.endswith('sparse'):
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                # [batch_size, max_len, ndim] -> [batch_size, max_len, ndim, emb_dim]
                # 对嵌入得到的进行聚合
                elif feat_type.endswith('array'):
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(1))
                # [batch_size, max_len] -> [batch_size, max_len, 1]
                elif feat_type.endswith('continual'):
                    feat_list.append(tensor_feature.unsqueeze(2))

        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue

            for k in feat_dict:
                tensor_feature = self.feat2tensor(feature_array, k)

                # [batch_size, max_len] -> [batch_size, max_len, emb_dim]
                if feat_type.endswith('sparse'):
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                # [batch_size, max_len, ndim] -> [batch_size, max_len, ndim, emb_dim]
                # 对嵌入得到的进行聚合
                elif feat_type.endswith('array'):
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                # [batch_size, max_len] -> [batch_size, max_len, 1]
                elif feat_type.endswith('continual'):
                    feat_list.append(tensor_feature.unsqueeze(2))
                elif feat_type.endswith('manual'):
                    add_feat_list.append(self.sparse_emb[k](tensor_feature))

        for k in self.ITEM_EMB_FEAT:
            # collect all data to numpy, then batch-convert
            batch_size = len(feature_array)
            emb_dim = self.ITEM_EMB_FEAT[k]
            seq_len = len(feature_array[0])

            # 每一个item_emb_feat -> [batch_size, seq_len, mm_emb_dim]-> [batch_size, seq_len, emb_dim]
            # pre-allocate tensor
            batch_emb_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)

            for i, seq in enumerate(feature_array):
                for j, item in enumerate(seq):
                    if k in item:
                        batch_emb_data[i, j] = item[k]

            # batch-convert and transfer to GPU
            tensor_feature = torch.from_numpy(batch_emb_data).to(self.dev)
            item_feat_list.append(self.emb_transform[k](tensor_feature))

        # merge features
        # 总的feature向量 [batch_size, feature_dim]
        all_item_emb = torch.cat(item_feat_list, dim=2)
        # print(f'all_item_emb.shape: {all_item_emb.shape}')
        all_item_emb = F.silu(self.itemdnn(all_item_emb))
        if include_user:
            # print(f'item_shape {[item.shape for item in user_feat_list]}')
            all_user_emb = torch.cat(user_feat_list, dim=1)
            all_user_emb = F.silu(self.userdnn(all_user_emb))
            # 初始化[N,M,D]的全零矩阵
            result = torch.zeros_like(all_item_emb, dtype=all_item_emb.dtype)
            
            # 找到mask中所有值为1的位置（返回两个一维数组：行索引和列索引）
            rows, cols = np.where(mask == 1)
            result[rows, cols] = all_user_emb[rows]
            all_input_emb = torch.cat([all_item_emb, add_feat_list[0]], dim=-1)
            result_input = F.silu(self.inputdnn(all_input_emb))

            seqs_emb = result_input + result
        else:
            seqs_emb = all_item_emb
        # [batch_size, maxlen, hidden_unit]
        return seqs_emb

    def fill_missing_feat(self, feat, item_id, item_type):
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
        if item_type == 1:
            for feat_type in self.item_feat_types.values():
                all_feat_ids.extend(feat_type)
        elif item_type == 2:
            for feat_type in self.user_feat_types.values():
                all_feat_ids.extend(feat_type)


        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]

        if item_type == 1:
            for feat_id in self.feature_types['item_emb']:
                if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                    if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                        filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]
        return filled_feat

    @staticmethod
    def collate_fn(batch):
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
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, user_feat = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        seq_feat = list(seq_feat)
        pos_feat = list(pos_feat)
        neg_feat = list(neg_feat)
        user_feat = list(user_feat)
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat, user_feat

def preprocess_user_feat(user_feats):
    batch_size = len(user_feats)
    user_feat_data = {}
    for k in user_feat_types['user_sparse']:
        batch_data = np.zeros((batch_size), dtype=np.int64)
        for i in range(batch_size):
            batch_data[i] =  user_feats[i][k]
        user_feat_data[k] = batch_data

    for k in user_feat_types['user_array']:
        max_array_len = 0
        for i in range(batch_size):
            max_array_len = max(max_array_len, len(user_feats[i][k]))
        batch_data = np.zeros((batch_size, max_array_len), dtype=np.int64)
        for i in range(batch_size):
            item_data = user_feats[i][k]
            actual_len = min(len(item_data), max_array_len)
            batch_data[i, :actual_len] = item_data[:actual_len]

    for k in user_feat_types['user_continual']:
        batch_data = np.zeros((batch_size), dtype=np.int64)
        for i in range(batch_size):
            batch_data[i] =  user_feats[i][k]
        user_feat_data[k] = batch_data
    return user_feat_data
def preprocess_seq_feat(seq_feats, seq=True):
    batch_size = len(seq_feats)
    seq_feat_data = {}
    for k in item_feat_types['item_sparse']:
        max_seq_len = max(len(seq_feats[i]) for i in range(batch_size))
        batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        for i in range(batch_size):
            seq_data = [item[k] for item in seq_feats[i]]
            batch_data[i] =  seq_data
        seq_feat_data[k] = batch_data

    for k in item_feat_types['item_array']:
        max_array_len = 0
        max_seq_len = 0
        for i in range(batch_size):
            seq_data = [item[k] for item in seq_feats[i]]
            max_seq_len = max(max_seq_len, len(seq_data))
            max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))

        batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
        for i in range(batch_size):
            seq_data = [item[k] for item in seq_feats[i]]
            for j, item_data in enumerate(seq_data):
                actual_len = min(len(item_data), max_array_len)
                batch_data[i, j, :actual_len] = item_data[:actual_len]
        seq_feat_data[k] = batch_data

    for k in item_feat_types['item_continual']:
        max_seq_len = max(len(seq_feats[i]) for i in range(batch_size))
        batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        for i in range(batch_size):
            seq_data = [item[k] for item in seq_feats[i]]
            batch_data[i] =  seq_data
        seq_feat_data[k] = batch_data

    if  seq:
        for k in add_feat_types['item_manual']:
            max_seq_len = max(len(seq_feats[i]) for i in range(batch_size))
            batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feats[i]]
                batch_data[i] =  seq_data
            seq_feat_data[k] = batch_data
    return seq_feat_data


    
class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)

    def _load_data_and_offsets(self):
        self.data_file = open(self.data_dir / "predict_seq.jsonl", 'rb')
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
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action, _ = record_tuple
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
                # user_feat['99'] = 2 # 點擊
                ext_user_sequence.insert(0, (u, user_feat, 2))

            if i and item_feat:
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, item_feat, 1))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)

        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_ = record_tuple
            feat = self.fill_missing_feat(feat, i, type_)
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = feat
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

        return seq, token_type, seq_feat, user_id

    def __len__(self):
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)

    @staticmethod
    def collate_fn(batch):
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
        seq, token_type, seq_feat, user_id = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        seq_feat = list(seq_feat)

        return seq, token_type, seq_feat, user_id


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
        if feat_id == '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                for json_file in base_path.glob('*'):
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
        if feat_id != '81':
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict

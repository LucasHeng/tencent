import json
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


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

    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        # 仅记录路径，避免在主进程持有打开的文件描述符（便于多worker安全地各自打开）
        self._data_file_path = self.data_dir / "seq.jsonl"
        self.data_file = None
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _ensure_data_file_open(self):
        # 在每个worker内懒加载独立的文件句柄，避免文件指针冲突
        if getattr(self, 'data_file', None) is None:
            self.data_file = open(self._data_file_path, 'rb')

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

    def __getstate__(self):
        # 使Dataset可pickle：去掉不可pickle的文件对象，由worker进程内再懒加载
        state = self.__dict__.copy()
        state['data_file'] = None
        return state

    def __del__(self):
        # 防御式关闭（不会在多worker生命周期内共享）
        try:
            if getattr(self, 'data_file', None) is not None:
                self.data_file.close()
        except Exception:
            pass

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

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, timestamp = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type, timestamp))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type, timestamp))

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
            seq_feat[idx] = feat
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                neg_ids = self._random_neq(1, self.itemnum + 1, ts, self.sample_neg_num)
                neg[idx] = neg_ids
                for id,neg_id in enumerate(neg_ids):
                    neg_feat[idx][id] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
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
        feat_types['user_sparse'] = ['103', '104', '105', '109']
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
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []

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


class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)

    def _load_data_and_offsets(self):
        self._data_file_path = self.data_dir / "predict_seq.jsonl"
        self.data_file = None
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
            seq[idx] = i
            seq_timestamp[idx] = timestamp
            token_type[idx] = type_
            seq_feat[idx] = feat
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

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

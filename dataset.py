import json
import pickle
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class PopularityNegativeSampler:
    def __init__(self, data_file_path, seq_offsets, alpha=0.75, popularity_cache_path=None, itemnum=None):
        """
        初始化基于流行度的负采样器，在初始化时从seq.jsonl统计流行度
        :param item_feat_dict: 物品特征字典，用于获取所有有效的物品ID
        :param data_file_path: seq.jsonl文件路径
        :param seq_offsets: 用户序列偏移量字典
        :param popularity_cache_path: 流行度统计缓存文件路径（.npy），存在则直接加载
        :param alpha: 平滑参数，控制流行度权重（默认0.75）
        """
        self.alpha = alpha
        self.popularity_cache_path = popularity_cache_path
        
        # 构建物品ID空间：需要提供itemnum，直接使用1..itemnum
        if itemnum is not None:
            self.num_items = int(itemnum)
            self.items = np.arange(1, self.num_items + 1, dtype=np.int32)
            # counts采用1-based索引，0号位空置，便于直接用item_id作为下标
            self.item_counts = np.zeros(self.num_items + 1, dtype=np.float64)
        else:
            raise ValueError("itemnum is required for PopularityNegativeSampler")
        
        
        # 优先尝试从缓存加载；否则统计后保存
        if isinstance(self.popularity_cache_path, (str, Path)) and Path(self.popularity_cache_path).exists():
            try:
                loaded = np.load(self.popularity_cache_path)
                # 兼容长度不一致时的安全处理
                if loaded.shape[0] == self.item_counts.shape[0]:
                    self.item_counts = loaded.astype(np.float64, copy=False)
                else:
                    # 长度不匹配则重新统计
                    self._compute_popularity_from_file(data_file_path, seq_offsets)
                    np.save(self.popularity_cache_path, self.item_counts)
            except Exception:
                # 读取失败则重新统计并保存
                self._compute_popularity_from_file(data_file_path, seq_offsets)
                try:
                    np.save(self.popularity_cache_path, self.item_counts)
                except Exception:
                    pass
        else:
            # 在初始化时统计流行度并尝试保存
            self._compute_popularity_from_file(data_file_path, seq_offsets)
            if isinstance(self.popularity_cache_path, (str, Path)):
                try:
                    Path(self.popularity_cache_path).parent.mkdir(parents=True, exist_ok=True)
                    np.save(self.popularity_cache_path, self.item_counts)
                except Exception:
                    pass
        
        # 计算采样概率分布
        self._update_sampling_probs()
        print(f"Load popularity cache from {self.popularity_cache_path}")

    def _compute_popularity_from_file(self, data_file_path, seq_offsets):
        """
        从seq.jsonl文件中统计物品流行度，不保留文件句柄
        :param data_file_path: seq.jsonl文件路径
        :param seq_offsets: 用户序列偏移量字典
        """
        total_users = len(seq_offsets)
        print(f"正在从 {total_users} 个用户序列中统计物品流行度...")
        
        # 仅在本方法作用域内打开一次文件句柄，遍历所有偏移读取
        try:
            with open(data_file_path, 'rb') as f:
                for uid in tqdm(range(total_users), desc="统计物品流行度"):
                    try:
                        f.seek(seq_offsets[uid])
                        line = f.readline()
                        user_sequence = json.loads(line)
                        # 统计该用户序列中的物品交互次数
                        for record_tuple in user_sequence:
                            _, item_id, _, _, _, _ = record_tuple
                            if item_id:
                                # 直接按1..itemnum的下标累加（0号位空置）或在items映射中查找
                                if self.item_counts.shape[0] == self.num_items + 1:
                                    if 0 < item_id <= self.num_items:
                                        self.item_counts[item_id] += 1
                                else:
                                    # 退化为查找映射
                                    item_idx = np.where(self.items == item_id)[0]
                                    if len(item_idx) > 0:
                                        self.item_counts[item_idx[0]] += 1
                    except Exception as e:
                        print(f"处理用户 {uid} 时出错: {e}")
                        continue
        except Exception as e:
            print(f"打开数据文件失败: {e}")
        
        print(f"物品流行度统计完成，共处理 {total_users} 个用户序列")

    def _update_sampling_probs(self):
        """
        更新采样概率分布
        """
        # 构造与self.items对齐的计数向量（若1-based则跳过0号位）
        if self.item_counts.shape[0] == self.num_items + 1:
            counts_vec = self.item_counts[1:]
        else:
            counts_vec = self.item_counts
        # 避免除零错误，给所有物品至少1次计数
        counts_safe = counts_vec + 1
    
        # # ✅ 计算逆流行度概率：P(i) ∝ 1 / (freq_i + 1)^α
        # inv_counts_pow = 1.0 / (counts_safe ** self.alpha)
        
        # # 归一化为概率分布
        # self.total = inv_counts_pow.sum()
        # self.sampling_probs = inv_counts_pow / self.total
        # # 预计算累积概率分布（用于加速采样）
        # self.cum_probs = np.cumsum(self.sampling_probs)

        # 计算采样概率分布（流行度^alpha / 总流行度^alpha）
        counts_pow = counts_safe ** self.alpha
        self.total = counts_pow.sum()
        self.sampling_probs = counts_pow / self.total
        
        # 预计算累积概率分布（加速采样）
        self.cum_probs = np.cumsum(self.sampling_probs)

    def sample(self, num_samples=13056, positive_items=None):
        """
        生成指定数量的负样本
        :param num_samples: 负样本数量（默认102*128=13056）
        :param positive_items: 需要排除的正样本集合（如用户已交互的物品，可选）
        :return: 负样本数组（长度为num_samples）
        """
        # 检查cum_probs是否正确初始化
        if self.cum_probs is None or len(self.cum_probs) == 0:
            raise ValueError("cum_probs未正确初始化，请检查_popularity_from_file方法")
        
        # 为避免采样到重复样本或正样本，多采20%作为候选
        candidate_size = int(num_samples * 1.2)
        if candidate_size < num_samples:
            candidate_size = num_samples
        
        # 1. 基于累积概率快速采样（比np.random.choice(p=...)更快）
        rand_vals = np.random.random(candidate_size)  # 生成[0,1)随机数
        # 找到随机数在累积概率中的位置，对应物品索引
        candidate_indices = np.searchsorted(self.cum_probs, rand_vals)
        candidates = self.items[candidate_indices]  # 候选负样本
        
        # 2. 过滤正样本（如果需要）
        if positive_items is not None:
            # 转换为集合加速查找（假设positive_items是列表或数组）
            positive_set = set(positive_items)
            # 过滤掉候选中属于正样本的元素
            candidates = [item for item in candidates if item not in positive_set]
            # 转换回NumPy数组
            candidates = np.array(candidates, dtype=self.items.dtype)
        
        # 3. 去重（保留顺序，确保样本多样性）
        # 用np.unique去重并保留首次出现的顺序
        _, unique_indices = np.unique(candidates, return_index=True)
        unique_candidates = candidates[np.sort(unique_indices)]
        
        # 4. 确保样本数量足够（若候选不足，补充采样）
        if len(unique_candidates) < num_samples:
            # 计算还需要补充的样本数
            need = num_samples - len(unique_candidates)
            # 补充采样（直接用np.random.choice，速度略慢但确保数量）
            additional_samples = np.random.choice(
                self.items,
                size=need,
                p=self.sampling_probs,
                replace=False  # 不重复
            )
            # 再次过滤正样本（如果需要）
            if positive_items is not None:
                additional_samples = [item for item in additional_samples if item not in positive_set]
                additional_samples = np.array(additional_samples, dtype=self.items.dtype)
            # 合并并再次去重
            unique_candidates = np.unique(np.concatenate([unique_candidates, additional_samples]))
        
        # 5. 截取需要的数量并返回
        return unique_candidates[:num_samples]


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
        # 缓存时间分桶边界，避免 __getitem__ 重复创建
        self._tdelta_edges = np.array([
            30, 60, 120, 300, 600, 900, 1800,
            3600, 7200, 14400, 28800,
            43200, 64800, 86400, 129600, 172800,
            259200, 432000, 604800, 1209600
        ], dtype=np.int64)
        
        # 初始化基于流行度的负采样器（仅在需要负采样时创建）
        self.negative_sampler = None
        enable_neg = getattr(args, 'enable_negative_sampling', True)
        if enable_neg and self.sample_neg_num > 0:
            self.save_path = args.save_path
            popularity_cache_path = Path(self.save_path, "popularity_counts.npy")
            self.negative_sampler = PopularityNegativeSampler(
                self._data_file_path,
                self.seq_offsets,
                alpha = args.alpha,
                popularity_cache_path=popularity_cache_path,
                itemnum=self.itemnum
            )

    def _safe_feat_stat(self, k):
        return self._get_feat_stat(self._ensure_feat_stats(), k)

    def _ensure_feat_stats(self):
        return getattr(self, 'feat_statistics', {})

    @staticmethod
    def _get_feat_stat(stat_dict, k):
        return stat_dict[k] if isinstance(stat_dict, dict) and k in stat_dict else 0

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

    def _random_neq(self, s, neg_num=1):
        """
        使用基于流行度的负采样生成不在序列s中的随机整数

        Args:
            l: 随机整数的最小值（保留兼容性，实际不使用）
            r: 随机整数的最大值（保留兼容性，实际不使用）
            s: 序列（正样本集合）
            neg_num: 负样本数量

        Returns:
            neg_samps: 不在序列s中的负样本列表
        """
        # 使用基于流行度的负采样
        neg_samps = self.negative_sampler.sample(num_samples=neg_num, positive_items=s)
        return neg_samps.tolist()

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
            # 构造时间特征并写入稀疏特征（整数类别）
            # 时间特征同时用于用户和物品
            hour = int((timestamp // 3600) % 24) if timestamp > 0 else 0
            weekday = int(((timestamp // 86400) + 4) % 7) if timestamp > 0 else 0
            month = int(((timestamp // (86400 * 30)) % 12) + 1) if timestamp > 0 else 0
            # 1304(时间差分桶)稍后统一根据“当前-前一条”的时间差回填
            feat['1301'] = month  # 已为1..12，缺失为0
            feat['1302'] = (weekday + 1) if timestamp > 0 else 0  # 1..7
            feat['1303'] = (hour + 1) if timestamp > 0 else 0      # 1..24
            feat['1304'] = 0  # 先置0，占位，循环结束后回填正确分桶
            seq_feat[idx] = feat
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                # pos时间特征使用默认值（不显式赋值）
                pos_feat[idx] = next_feat
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        # 训练时的负样本采样移动到 collate_fn 中统一进行；此处不再填充 neg/neg_feat
        

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
            # '111',
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
            feat_default_value[feat_id] = 0
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
        # 重新在collate中统一采样负样本：忽略各样本的neg，重新生成
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

        # 在collate中统一负采样（仅当训练集启用采样器）
        B = seq.shape[0]
        L = seq.shape[1]
        K = self.sample_neg_num
        if getattr(self, 'negative_sampler', None) is not None:
            # 1) 收集整batch的正样本集合（去重）
            pos_np = pos.numpy()
            # next_token_type==1 的位置才有正样本
            ntt_np = next_token_type.numpy()
            mask = (ntt_np == 1)
            positive_set = set(pos_np[mask].tolist())
            if 0 in positive_set:
                positive_set.discard(0)
            # 2) 一次性采样 B*L*K*2 候选并过滤
            total_slots = B * L * K
            candidate_size = total_slots * 2
            pool = self.negative_sampler.sample(num_samples=candidate_size, positive_items=positive_set)
            if len(pool) < total_slots:
                extra_need = total_slots - len(pool)
                if extra_need > 0:
                    extra = self.negative_sampler.sample(num_samples=extra_need, positive_items=positive_set)
                    pool = np.concatenate([pool, extra])
            pool = pool[:total_slots]
            neg = torch.from_numpy(pool.reshape(B, L, K).astype(np.int32))
        else:
            # 测试或未启用采样器：保留占位形状
            neg = torch.zeros((B, L, K), dtype=torch.int32)

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

        # 基于新采样的neg重建neg_feat（字典形式）
        neg_feat_pre = {}
        if getattr(self, 'negative_sampler', None) is not None:
            # 构造与原先接口兼容的list[list[dict]]形式供下游打包
            neg_feat = np.empty([B, L, K], dtype=object)
            for b in range(B):
                for t in range(L):
                    for n in range(K):
                        neg_id = int(neg[b, t, n].item())
                        nf = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
                        neg_feat[b, t, n] = nf
            # 重新使用已有的构建函数打包
            neg_feat_list = list(neg_feat)
            neg_feat_pre.update(build_dense_neg(self.feature_types['item_sparse'], neg_feat_list, 'int'))
            neg_feat_pre.update(build_dense_neg(self.feature_types['item_continual'], neg_feat_list, 'float'))
            neg_feat_pre.update(build_array_neg(self.feature_types['item_array'], neg_feat_list))
            neg_feat_pre.update(build_emb_neg(self.feature_types['item_emb'], neg_feat_list))
        else:
            # 测试集保持空特征
            neg_feat_pre = {k: np.zeros((B, L, K), dtype=np.int64) for k in self.feature_types['item_sparse']}

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat_pre, pos_feat_pre, neg_feat_pre, seq_timestamp


class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        # 强制禁用负采样器
        args.enable_negative_sampling = False
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
            # 添加时间稀疏特征（与训练集一致，保留0作为padding）
            # 时间特征同时用于用户和物品
            hour = int((timestamp // 3600) % 24) if timestamp > 0 else 0
            weekday = int(((timestamp // 86400) + 4) % 7) if timestamp > 0 else 0
            month = int(((timestamp // (86400 * 30)) % 12) + 1) if timestamp > 0 else 0
            # 1304(时间差分桶)稍后统一根据"当前-前一条"的时间差回填
            feat['1301'] = month  # 已为1..12，缺失为0
            feat['1302'] = (weekday + 1) if timestamp > 0 else 0  # 1..7
            feat['1303'] = (hour + 1) if timestamp > 0 else 0      # 1..24
            feat['1304'] = 0  # 先置0，占位，循环结束后回填正确分桶
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

import json
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict, Counter
import psutil


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
        self.save_path = args.save_path
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
        print(f'self.feature_types {self.feature_types}')
        print(f'self.feat_statistics {self.feat_statistics}')
        # 统计时间间隔分桶：类别0为padding，1为零间隔，2..为各边界桶
        try:
            self._tdelta_bucket_counts = np.zeros(int(self._safe_feat_stat('1304')) + 1, dtype=np.int64)
        except Exception:
            self._tdelta_bucket_counts = np.zeros(32, dtype=np.int64)
        self._tdelta_total = 0
        # 缓存时间分桶边界，避免 __getitem__ 重复创建
        self._tdelta_edges = np.array([
            30, 60, 120, 300, 600, 900, 1800,
            3600, 7200, 14400, 28800,
            43200, 64800, 86400, 129600, 172800,
            259200, 432000, 604800, 1209600
        ], dtype=np.int64)


        if self.stats_file_path.exists() and self.offsets_file_path.exists():
            print(f"发现新格式的统计信息文件: {self.stats_file_path}")
            self._load_offsets()
        else:
            print("未发现缓存的统计信息，开始处理数据...")
            self.process_data()
        
        # 检查全局统计文件是否存在，如果不存在则生成
        global_stats_file = Path(self.save_path) / "global_hourly_stats.jsonl"
        if not global_stats_file.exists():
            print("未发现全局统计文件，开始生成...")
            self._generate_global_hourly_stats_from_existing()
        

    def _load_offsets(self):
        """
        加载offset索引
        """
        if self.offsets_file_path and self.offsets_file_path.exists():
            with open(self.offsets_file_path, 'rb') as f:
                self.item_offsets = pickle.load(f)
            print(f"已加载 {len(self.item_offsets)} 个item的offset索引")
        else:
            print("未找到offset索引文件")

    def timestamp_to_hour_stamp(self, timestamp):
        """
        将时间戳转换为小时时间戳
        
        Args:
            timestamp: Unix时间戳
            
        Returns:
            hour_stamp: 小时时间戳（Unix时间戳向下取整到小时）
        """
        # 向下取整到小时
        return (timestamp // 3600) * 3600

    def process_data(self):
        """
        处理数据，构建按小时的统计信息并保存到文件
        """
        print("开始处理item统计信息...")
        
        # 加载数据
        seq_file = self.data_dir / "seq.jsonl"
        
        # 临时存储统计信息（处理完成后写入文件）
        temp_stats = defaultdict(lambda: defaultdict(lambda: {'exposure_users': [], 'click_users': []}))

        # 流式处理数据，避免一次性加载所有记录到内存
        record_count = 0
        with open(seq_file, 'r') as f:
            for line_num, line in enumerate(tqdm(f, desc="处理用户序列")):
                user_sequence = json.loads(line.strip())
                for record in user_sequence:
                    user_id, item_id, user_feat, item_feat, action_type, timestamp = record
                    if item_id is not None and user_id is not None:
                        # 直接处理记录，不存储在内存中
                        hour_stamp = self.timestamp_to_hour_stamp(timestamp)
                        
                        # 更新该小时的统计信息
                        if user_id not in temp_stats[item_id][hour_stamp]['exposure_users']:
                            temp_stats[item_id][hour_stamp]['exposure_users'].append(user_id)
                        
                        if action_type == 1:  # 点击
                            if user_id not in temp_stats[item_id][hour_stamp]['click_users']:
                                temp_stats[item_id][hour_stamp]['click_users'].append(user_id)
                        
                        record_count += 1
                        
        print(f"总记录数: {record_count}")
        
        # 将统计信息写入文件并创建offset索引
        self._save_stats_to_file(temp_stats)
        
        # 生成全局每小时统计
        self._generate_global_hourly_stats(temp_stats)
        
        # 清理临时数据
        del temp_stats
        
        print(f"处理完成！")
        
        # 打印内存使用情况
        memory_usage = self._get_memory_usage()
        print(f"最终内存使用: {memory_usage:.2f}MB")

    def _get_memory_usage(self):
        """获取当前内存使用情况（MB）"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024
        except:
            return 0

    def _save_stats_to_file(self, temp_stats):
        """
        将统计信息保存到文件并创建offset索引
        """
        print("将统计信息写入文件...")
        self.item_offsets = {}
        # 写入统计信息文件
        with open(self.stats_file_path, 'w') as f:
            for item_id, hours in tqdm(temp_stats.items(), desc="写入统计信息"):
                # 记录当前item的offset
                self.item_offsets[item_id] = f.tell()
                
                # 写入item的统计信息
                item_data = {
                    'item_id': item_id,
                    'stats': {}
                }
                
                for hour_stamp, stats in hours.items():
                    item_data['stats'][str(hour_stamp)] = {
                        'exposure_users': stats['exposure_users'],
                        'click_users': stats['click_users']
                    }
                
                f.write(json.dumps(item_data, ensure_ascii=False) + '\n')
        
        # 保存offset索引
        with open(self.offsets_file_path, 'wb') as f:
            pickle.dump(self.item_offsets, f)
        
        print(f"统计信息已保存到: {self.stats_file_path}")
        print(f"Offset索引已保存到: {self.offsets_file_path}")
        print(f"总item数: {len(self.item_offsets)}")

    def _generate_global_hourly_stats(self, temp_stats):
        """
        生成全局每小时统计信息（所有item的总曝光次数和总点击次数）
        
        Args:
            temp_stats: 临时统计信息
        """
        print("生成全局每小时统计...")
        
        # 收集所有小时的时间戳
        all_hours = set()
        for item_id, hours in temp_stats.items():
            all_hours.update(hours.keys())
        
        # 计算每个小时的全局统计
        global_hourly_stats = {}
        for hour_stamp in tqdm(all_hours, desc="计算全局每小时统计"):
            total_exposure = 0
            total_click = 0
            
            for item_id, hours in temp_stats.items():
                if hour_stamp in hours:
                    hour_stats = hours[hour_stamp]
                    total_exposure += len(hour_stats['exposure_users'])
                    total_click += len(hour_stats['click_users'])
            
            global_hourly_stats[hour_stamp] = {
                'exposure_count': total_exposure,
                'click_count': total_click
            }
        
        # 保存全局统计到文件
        global_stats_file = Path(self.save_path) / "global_hourly_stats.jsonl"
        with open(global_stats_file, 'w') as f:
            for hour_stamp in sorted(global_hourly_stats.keys()):
                stats_data = {
                    'hour_stamp': hour_stamp,
                    'exposure_count': global_hourly_stats[hour_stamp]['exposure_count'],
                    'click_count': global_hourly_stats[hour_stamp]['click_count']
                }
                f.write(json.dumps(stats_data, ensure_ascii=False) + '\n')
        
        print(f"全局每小时统计已保存到: {global_stats_file}")
        print(f"总小时数: {len(global_hourly_stats)}")

    def _generate_global_hourly_stats_from_existing(self):
        """
        从现有的item统计文件生成全局每小时统计
        """
        print("从现有统计文件生成全局每小时统计...")
        
        # 收集所有小时的时间戳和统计信息
        all_hours = set()
        hour_stats_map = defaultdict(lambda: {'exposure_count': 0, 'click_count': 0})
        
        # 读取现有的item统计文件
        if not self.stats_file_path.exists():
            print("未找到item统计文件，无法生成全局统计")
            return
        
        with open(self.stats_file_path, 'r') as f:
            for line in tqdm(f, desc="读取item统计文件"):
                try:
                    item_data = json.loads(line.strip())
                    item_id = item_data['item_id']
                    stats = item_data['stats']
                    
                    for hour_stamp_str, hour_stats in stats.items():
                        hour_stamp = int(hour_stamp_str)
                        all_hours.add(hour_stamp)
                        
                        # 累加曝光和点击次数
                        hour_stats_map[hour_stamp]['exposure_count'] += len(hour_stats['exposure_users'])
                        hour_stats_map[hour_stamp]['click_count'] += len(hour_stats['click_users'])
                        
                except Exception as e:
                    print(f"解析item统计行失败: {e}")
                    continue
        
        # 保存全局统计到文件
        global_stats_file = Path(self.save_path) / "global_hourly_stats.jsonl"
        with open(global_stats_file, 'w') as f:
            for hour_stamp in sorted(all_hours):
                stats_data = {
                    'hour_stamp': hour_stamp,
                    'exposure_count': hour_stats_map[hour_stamp]['exposure_count'],
                    'click_count': hour_stats_map[hour_stamp]['click_count']
                }
                f.write(json.dumps(stats_data, ensure_ascii=False) + '\n')
        
        print(f"全局每小时统计已保存到: {global_stats_file}")
        print(f"总小时数: {len(all_hours)}")

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

        self.stats_file_path = Path(self.save_path) / "item_stats.jsonl"
        self.offsets_file_path = Path(self.save_path) / "item_stats_offsets.pkl"

        
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
        state['stats_file'] = None
        state['global_stats_file'] = None
        # 保留全局统计缓存，避免重复加载
        if hasattr(self, '_global_stats_cache'):
            state['_global_stats_cache'] = self._global_stats_cache
        return state

    def __del__(self):
        # 防御式关闭（不会在多worker生命周期内共享）
        try:
            if getattr(self, 'data_file', None) is not None:
                self.data_file.close()
            if getattr(self, 'stats_file', None) is not None:
                self.stats_file.close()
            if getattr(self, 'global_stats_file', None) is not None:
                self.global_stats_file.close()
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

    def _load_item_stats_from_file(self, item_id):
        """
        从文件加载指定item的统计信息
        
        Args:
            item_id: 物品ID
            
        Returns:
            dict: item的统计信息，如果不存在返回None
        """
        if item_id not in self.item_offsets:
            return None
        
        # 在每个worker内懒加载独立的文件句柄，避免文件指针冲突
        if getattr(self, 'stats_file', None) is None:
            self.stats_file = open(self.stats_file_path, 'r')
        
        try:
            # 定位到指定item的位置
            self.stats_file.seek(self.item_offsets[item_id])
            
            # 读取一行
            line = self.stats_file.readline()
            if line:
                item_data = json.loads(line.strip())
                return item_data.get('stats', {})
        except Exception as e:
            print(f"读取item {item_id} 统计信息失败: {e}")
            return None
        
        return None

    def _load_global_hourly_stats(self, hour_stamp):
        """
        加载指定小时的全局统计信息（所有item的总曝光次数和总点击次数）
        
        Args:
            hour_stamp: 小时时间戳
            
        Returns:
            dict: 包含全局统计信息的字典
        """
        # 全局统计文件路径
        global_stats_file = Path(self.save_path) / "global_hourly_stats.jsonl"
        
        if not global_stats_file.exists():
            return {'exposure_count': 0, 'click_count': 0}
        
        try:
            # 懒加载全局统计缓存
            if not hasattr(self, '_global_stats_cache'):
                self._load_global_stats_cache()
            
            # 从缓存中获取统计信息
            return self._global_stats_cache.get(hour_stamp, {'exposure_count': 0, 'click_count': 0})
            
        except Exception as e:
            print(f"读取全局统计信息失败: {e}")
            return {'exposure_count': 0, 'click_count': 0}

    def _load_global_stats_cache(self):
        """
        加载全局统计缓存到内存
        """
        global_stats_file = Path(self.save_path) / "global_hourly_stats.jsonl"
        self._global_stats_cache = {}
        
        try:
            with open(global_stats_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    hour_stamp = data.get('hour_stamp')
                    if hour_stamp is not None:
                        self._global_stats_cache[hour_stamp] = {
                            'exposure_count': data.get('exposure_count', 0),
                            'click_count': data.get('click_count', 0)
                        }
        except Exception as e:
            print(f"加载全局统计缓存失败: {e}")
            self._global_stats_cache = {}

    def get_item_statistics(self, item_id, timestamp, max_users=100):
        """
        获取指定item在指定时间戳的统计信息
        
        Args:
            item_id: 物品ID
            timestamp: 时间戳
            max_users: 每个时间窗口最多返回的用户数量，默认10
            
        Returns:
            dict: 包含统计信息的字典
        """
        target_hour =  (timestamp // 3600) * 3600
        
        # 从文件加载item的统计信息
        item_stats = self._load_item_stats_from_file(item_id)
        if not item_stats:
            return {
                'prev_hour_exposure_users': [],
                'prev_hour_click_users': [],
                'prev_24h_exposure_users': [],
                'prev_24h_click_users': [],
                'all_time_exposure_users': [],
                'all_time_click_users': []
            }
        
        # 获取前1小时、前24小时、之前所有时间的用户
        stats = {
            'prev_hour_exposure_users': [],
            'prev_hour_click_users': [],
            'prev_24h_exposure_users': [],
            'prev_24h_click_users': [],
            'all_time_exposure_users': [],
            'all_time_click_users': [],
            'prev_7d_exposure_users': [],
            'prev_7d_click_users': [],
            'prev_24h_exposure_count': 0,
            'prev_24h_click_count': 0
        }
        
        # 前1小时
        prev_hour = target_hour - 3600
        if str(prev_hour) in item_stats:
            hour_stats = item_stats[str(prev_hour)]
            exposure_users = hour_stats['exposure_users']
            click_users = hour_stats['click_users']
            # Top用户
            exposure_counter = Counter()
            click_counter = Counter()
            exposure_counter.update(exposure_users)
            click_counter.update(click_users)
            top_exposure_users = [user for user, _ in exposure_counter.most_common(max_users)]
            top_click_users = [user for user, _ in click_counter.most_common(max_users)]
            stats['prev_hour_exposure_users'] = top_exposure_users
            stats['prev_hour_click_users'] = top_click_users
            # 计数使用len累加
            stats['prev_hour_exposure_count'] = int(len(exposure_users))
            stats['prev_hour_click_count'] = int(len(click_users))
        
        # 前24小时（最近24小时，不包括当前小时）
        prev_24h_start = target_hour - 24 * 3600
        prev_7d_start = target_hour - 7 * 24 * 3600
        prev_24h_exposure_users = []
        prev_24h_click_users = []
        prev_24h_exposure_count = 0
        prev_24h_click_count = 0
        prev_7d_exposure_users = []
        prev_7d_click_users = []
        prev_7d_exposure_count = 0
        prev_7d_click_count = 0
        
        # 收集前24小时与7天内的所有用户与事件计数，按时间顺序（从早到晚）
        for hour_stamp in range(prev_24h_start, target_hour, 3600):
            if str(hour_stamp) in item_stats:
                hour_stats = item_stats[str(hour_stamp)]
                prev_24h_exposure_users.extend(hour_stats['exposure_users'])
                prev_24h_click_users.extend(hour_stats['click_users'])
                prev_24h_exposure_count += int(len(hour_stats['exposure_users']))
                prev_24h_click_count += int(len(hour_stats['click_users']))

        for hour_stamp in range(prev_7d_start, target_hour, 3600):
            if str(hour_stamp) in item_stats:
                hour_stats = item_stats[str(hour_stamp)]
                prev_7d_exposure_users.extend(hour_stats['exposure_users'])
                prev_7d_click_users.extend(hour_stats['click_users'])
                prev_7d_exposure_count += int(len(hour_stats['exposure_users']))
                prev_7d_click_count += int(len(hour_stats['click_users']))
        
        # 统计用户出现次数，选择出现次数最多的用户
        prev_24h_exposure_counter = Counter()
        prev_24h_click_counter = Counter()
        
        prev_24h_exposure_counter.update(prev_24h_exposure_users)
        prev_24h_click_counter.update(prev_24h_click_users)
        
        # 选择出现次数最多的用户，如果次数相同则按用户ID排序
        top_prev_24h_exposure_users = [user for user, _ in prev_24h_exposure_counter.most_common(max_users)]
        top_prev_24h_click_users = [user for user, _ in prev_24h_click_counter.most_common(max_users)]
        
        stats['prev_24h_exposure_users'] = top_prev_24h_exposure_users
        stats['prev_24h_click_users'] = top_prev_24h_click_users
        stats['prev_24h_exposure_count'] = int(prev_24h_exposure_count)
        stats['prev_24h_click_count'] = int(prev_24h_click_count)

        # 7天窗口Top用户
        prev_7d_exposure_counter = Counter()
        prev_7d_click_counter = Counter()
        prev_7d_exposure_counter.update(prev_7d_exposure_users)
        prev_7d_click_counter.update(prev_7d_click_users)
        top_prev_7d_exposure_users = [user for user, _ in prev_7d_exposure_counter.most_common(max_users)]
        top_prev_7d_click_users = [user for user, _ in prev_7d_click_counter.most_common(max_users)]
        stats['prev_7d_exposure_users'] = top_prev_7d_exposure_users
        stats['prev_7d_click_users'] = top_prev_7d_click_users
        stats['prev_7d_exposure_count'] = int(prev_7d_exposure_count)
        stats['prev_7d_click_count'] = int(prev_7d_click_count)
        
        # 之前所有时间（不包括当前时间戳所在的小时）
        all_exposure_users = []
        all_click_users = []
        all_time_exposure_count = 0
        all_time_click_count = 0
        
        # 收集所有历史用户（不需要按时间顺序，因为Counter只关心出现次数）
        for hour_stamp in item_stats.keys():
            if int(hour_stamp) < target_hour:  # 只包括严格小于当前小时的数据
                hour_stats = item_stats[hour_stamp]
                all_exposure_users.extend(hour_stats['exposure_users'])
                all_click_users.extend(hour_stats['click_users'])
                all_time_exposure_count += int(len(hour_stats['exposure_users']))
                all_time_click_count += int(len(hour_stats['click_users']))
        
        # 统计用户出现次数，选择出现次数最多的用户
        all_exposure_counter = Counter()
        all_click_counter = Counter()
        
        all_exposure_counter.update(all_exposure_users)
        all_click_counter.update(all_click_users)
        
        # 选择出现次数最多的用户，如果次数相同则按用户ID排序
        top_all_exposure_users = [user for user, _ in all_exposure_counter.most_common(max_users)]
        top_all_click_users = [user for user, _ in all_click_counter.most_common(max_users)]
        
        stats['all_time_exposure_users'] = top_all_exposure_users
        stats['all_time_click_users'] = top_all_click_users
        # 全时段与前1小时计数已按事件数统计
        stats['all_time_exposure_count'] = int(all_time_exposure_count)
        stats['all_time_click_count'] = int(all_time_click_count)

        # CTRs（点击率）
        def safe_ctr(clicks, expos):
            return float(clicks) / float(expos) if float(expos) > 0 else 0.0

        stats['ctr_prev_hour'] = safe_ctr(stats.get('prev_hour_click_count', 0), stats.get('prev_hour_exposure_count', 0))
        stats['ctr_prev_24h'] = safe_ctr(stats.get('prev_24h_click_count', 0), stats.get('prev_24h_exposure_count', 0))
        stats['ctr_prev_7d'] = safe_ctr(stats.get('prev_7d_click_count', 0), stats.get('prev_7d_exposure_count', 0))
        stats['ctr_all_time'] = safe_ctr(stats.get('all_time_click_count', 0), stats.get('all_time_exposure_count', 0))
        
        # 计算每小时全局统计（所有item的总曝光次数和总点击次数）
        # 从全局统计文件中读取当前小时的统计
        global_stats = self._load_global_hourly_stats(target_hour)
        stats['hourly_global_exposure_count'] = global_stats.get('exposure_count', 0)
        stats['hourly_global_click_count'] = global_stats.get('click_count', 0)
        
        return stats

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
        max_timestamp = 0
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, timestamp = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type, timestamp))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type, timestamp))
            max_timestamp = max(max_timestamp, timestamp)

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
            feat['1305'] = np.log1p(max_timestamp - timestamp)
            if type_ == 1:
                item_stats = self.get_item_statistics(i, timestamp)
                feat['1401'] = item_stats['prev_hour_exposure_users']
                feat['1402'] = item_stats['prev_hour_click_users']
                feat['1403'] = item_stats['prev_24h_exposure_users']
                feat['1404'] = item_stats['prev_24h_click_users']
                feat['1405'] = item_stats['all_time_exposure_users']
                feat['1406'] = item_stats['all_time_click_users']
                # 新增：7天窗口用户数组
                feat['1407'] = item_stats.get('prev_7d_exposure_users', [])
                feat['1408'] = item_stats.get('prev_7d_click_users', [])
                # 次数特征取log分桶（log1p再向下取整）
                def _log_bucket(v):
                    try:
                        vv = float(v)
                    except Exception:
                        vv = 0.0
                    if vv < 0:
                        vv = 0.0
                    # 次数上限裁剪为4万，再进行log1p与向下取整
                    if vv > 40000.0:
                        vv = 40000.0
                    return int(np.floor(np.log1p(vv)))

                feat['1601'] = _log_bucket(item_stats.get('prev_24h_exposure_count', 0))
                feat['1602'] = _log_bucket(item_stats.get('prev_24h_click_count', 0))
                # 计数（小时、7天、全时）
                feat['1603'] = _log_bucket(item_stats.get('prev_hour_exposure_count', 0))
                feat['1604'] = _log_bucket(item_stats.get('prev_hour_click_count', 0))
                feat['1605'] = _log_bucket(item_stats.get('prev_7d_exposure_count', 0))
                feat['1606'] = _log_bucket(item_stats.get('prev_7d_click_count', 0))
                feat['1607'] = _log_bucket(item_stats.get('all_time_exposure_count', 0))
                feat['1608'] = _log_bucket(item_stats.get('all_time_click_count', 0))
                # CTR（小时、24h、7天、全时）
                feat['1611'] = float(item_stats.get('ctr_prev_hour', 0.0))
                feat['1612'] = float(item_stats.get('ctr_prev_24h', 0.0))
                feat['1613'] = float(item_stats.get('ctr_prev_7d', 0.0))
                feat['1614'] = float(item_stats.get('ctr_all_time', 0.0))
                
                # 每小时全局统计
                feat['1701'] = float(item_stats.get('hourly_global_exposure_count', 0))  # 当前小时全局曝光次数
                feat['1702'] = float(item_stats.get('hourly_global_click_count', 0))     # 当前小时全局点击次数
                if act_type is None:
                    act_type = -1
                feat['1501'] = act_type + 1
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
                # 统计：仅对item记录计数（包含padding=0、零间隔=1、其余=2..）
                if tt_np[t] == 1:
                    idx = c
                    if 0 <= idx < self._tdelta_bucket_counts.shape[0]:
                        self._tdelta_bucket_counts[idx] += 1
                    self._tdelta_total += 1
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

    def report_time_delta_hist(self):
        """
        打印时间间隔分桶的频率统计（占比）。
        0: padding/首个item；1: 真实零间隔；2..: 固定边界桶。
        """
        if getattr(self, '_tdelta_total', 0) == 0:
            print('[time-delta] no samples counted yet')
            return
        total = max(1, int(self._tdelta_total))
        counts = self._tdelta_bucket_counts[:]
        # 输出前若干桶的占比
        print('[time-delta] bucket ratios:')
        for i, c in enumerate(counts):
            ratio = float(c) / total
            print(f'  bucket {i}: count={int(c)} ratio={ratio:.6f}')

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
        feat_types['user_sparse'] = ['103', '104', '105', '109', '1301', '1302', '1303', '1304', '1501',
                                     '1601','1602','1603','1604','1605','1606','1607','1608']
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
        feat_types['user_stats_array'] = ['1401', '1402', '1403', '1404','1405','1406','1407','1408']
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = ['1305', '1611', '1612', '1613', '1614', '1701', '1702']
        # 连续型特征
        feat_types['item_continual'] = []
        # 将时间特征改为稀疏特征（整数类别），使用数字型特征ID
        # 约定：1301-月份，1302-星期，1303-小时，1304-时间差分桶
        # 时间特征同时用于用户和物品

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            try:
                feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
            except Exception:
                feat_statistics[feat_id] = 0
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            try:
                feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
            except Exception:
                feat_statistics[feat_id] = 0
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0.0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0.0
        for feat_id in feat_types['user_stats_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = self.usernum
        # 为新增稀疏时间特征设置默认值与词表大小（数值型ID）
        feat_default_value['1301'] = 0
        feat_default_value['1302'] = 0
        feat_default_value['1303'] = 0
        feat_default_value['1304'] = 0
        feat_default_value['1501'] = 0
        feat_statistics['1301'] = 12
        feat_statistics['1302'] = 7
        feat_statistics['1303'] = 24
        # t_delta 使用固定边界分桶：非零桶 = 1(零间隔) + len(edges) + 1(>max边界)
        # 当前 edges=20 → 非零桶=22（+0 padding）
        feat_statistics['1304'] = 22
        feat_statistics['1501'] = 2
        # 次数分桶：log1p裁剪至4万后floor → 0..10 共11个桶
        for k in ['1601','1602','1603','1604','1605','1606','1607','1608']:
            feat_statistics[k] = 11
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
        seq_feat_pre.update(build_array(self.feature_types['user_stats_array'], seq_feat_list))
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
        seq_feat_pre.update(build_array(self.feature_types['user_stats_array'], seq_feat_list))
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

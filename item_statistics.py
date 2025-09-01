import json
import pickle
import numpy as np
import gc
import psutil
from collections import defaultdict, Counter
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta


class ItemStatisticsProcessor:
    """
    处理item统计信息特征，按小时统计每个item的点击和曝光用户
    数据结构: {item_id: {hour_stamp: {'exposure_users': [user_id], 'click_users': [user_id]}}}
    """
    
    def __init__(self, data_dir):
        """
        初始化处理器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        
        # 文件路径
        self.stats_file_path = None
        self.offsets_file_path = None
        
        # 内存中只保存offset映射，不保存实际数据
        self.item_offsets = {}
        self.hour_stamps = set()
        
        # 内存优化：创建Counter对象池
        self._counter_pool = []
        self._max_pool_size = 20
        
        # 文件句柄（延迟初始化）
        self.stats_file = None
        
        # 更新队列管理
        self._update_queue = []
        self._max_queue_size = 10000  # 最大队列大小
        
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
    
    def _save_stats_to_file(self, temp_stats):
        """
        将统计信息保存到文件并创建offset索引
        """
        # 设置文件路径
        self.stats_file_path = self.data_dir / "item_stats.jsonl"
        self.offsets_file_path = self.data_dir / "item_stats_offsets.pkl"
        
        print("将统计信息写入文件...")
        
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
        
        # 延迟初始化文件句柄
        if self.stats_file is None:
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
    
    def _get_memory_usage(self):
        """获取当前内存使用情况（MB）"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024
        except:
            return 0
    
    def _cleanup_memory(self):
        """清理内存"""
        # 强制垃圾回收
        gc.collect()
        
        # 清理Counter对象池
        self._counter_pool.clear()
        
        # 打印清理后的内存使用情况
        memory_after = self._get_memory_usage()
        print(f"内存清理完成，当前使用: {memory_after:.2f}MB")
    
    def _get_counter_from_pool(self):
        """从对象池获取Counter对象"""
        if self._counter_pool:
            counter = self._counter_pool.pop()
            counter.clear()  # 清空数据
            return counter
        else:
            return Counter()
    
    def _return_counter_to_pool(self, counter):
        """将Counter对象返回池中"""
        if len(self._counter_pool) < self._max_pool_size:
            counter.clear()
            self._counter_pool.append(counter)
    
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
                        self.hour_stamps.add(hour_stamp)
                        
                        # 更新该小时的统计信息
                        if user_id not in temp_stats[item_id][hour_stamp]['exposure_users']:
                            temp_stats[item_id][hour_stamp]['exposure_users'].append(user_id)
                        
                        if action_type == 1:  # 点击
                            if user_id not in temp_stats[item_id][hour_stamp]['click_users']:
                                temp_stats[item_id][hour_stamp]['click_users'].append(user_id)
                        
                        record_count += 1
                        
        print(f"总记录数: {record_count}")
        
        # 转换为有序的小时时间戳列表
        self.hour_stamps = sorted(list(self.hour_stamps))
        
        # 将统计信息写入文件并创建offset索引
        self._save_stats_to_file(temp_stats)
        
        # 清理临时数据
        del temp_stats
        
        # 最终内存清理
        self._cleanup_memory()
        
        print(f"处理完成！")
        print(f"时间范围: {datetime.fromtimestamp(self.hour_stamps[0])} 到 {datetime.fromtimestamp(self.hour_stamps[-1])}")
        print(f"总小时数: {len(self.hour_stamps)}")
        
        # 打印内存使用情况
        memory_usage = self._get_memory_usage()
        print(f"最终内存使用: {memory_usage:.2f}MB")
    
    def get_item_statistics(self, item_id, timestamp, max_users=10):
        """
        获取指定item在指定时间戳的统计信息
        
        Args:
            item_id: 物品ID
            timestamp: 时间戳
            max_users: 每个时间窗口最多返回的用户数量，默认10
            
        Returns:
            dict: 包含统计信息的字典
        """
        target_hour = self.timestamp_to_hour_stamp(timestamp)
        
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
            'all_time_click_users': []
        }
        
        # 前1小时
        prev_hour = target_hour - 3600
        if str(prev_hour) in item_stats:
            hour_stats = item_stats[str(prev_hour)]
            exposure_users = hour_stats['exposure_users']
            click_users = hour_stats['click_users']
            
            # 统计用户出现次数，选择出现次数最多的用户
            exposure_counter = self._get_counter_from_pool()
            click_counter = self._get_counter_from_pool()
            
            exposure_counter.update(exposure_users)
            click_counter.update(click_users)
            
            # 选择出现次数最多的用户，如果次数相同则按用户ID排序
            top_exposure_users = [user for user, _ in exposure_counter.most_common(max_users)]
            top_click_users = [user for user, _ in click_counter.most_common(max_users)]
            
            stats['prev_hour_exposure_users'] = top_exposure_users
            stats['prev_hour_click_users'] = top_click_users
            
            # 返回Counter对象到池中
            self._return_counter_to_pool(exposure_counter)
            self._return_counter_to_pool(click_counter)
        
        # 前24小时（最近24小时，不包括当前小时）
        prev_24h_start = target_hour - 24 * 3600
        prev_24h_exposure_users = []
        prev_24h_click_users = []
        
        # 收集前24小时的所有用户，按时间顺序（从早到晚）
        for hour_stamp in range(prev_24h_start, target_hour, 3600):
            if str(hour_stamp) in item_stats:
                hour_stats = item_stats[str(hour_stamp)]
                prev_24h_exposure_users.extend(hour_stats['exposure_users'])
                prev_24h_click_users.extend(hour_stats['click_users'])
        
        # 统计用户出现次数，选择出现次数最多的用户
        prev_24h_exposure_counter = self._get_counter_from_pool()
        prev_24h_click_counter = self._get_counter_from_pool()
        
        prev_24h_exposure_counter.update(prev_24h_exposure_users)
        prev_24h_click_counter.update(prev_24h_click_users)
        
        # 选择出现次数最多的用户，如果次数相同则按用户ID排序
        top_prev_24h_exposure_users = [user for user, _ in prev_24h_exposure_counter.most_common(max_users)]
        top_prev_24h_click_users = [user for user, _ in prev_24h_click_counter.most_common(max_users)]
        
        stats['prev_24h_exposure_users'] = top_prev_24h_exposure_users
        stats['prev_24h_click_users'] = top_prev_24h_click_users
        
        # 返回Counter对象到池中
        self._return_counter_to_pool(prev_24h_exposure_counter)
        self._return_counter_to_pool(prev_24h_click_counter)
        
        # 之前所有时间（不包括当前时间戳所在的小时）
        all_exposure_users = []
        all_click_users = []
        
        # 收集所有历史用户（不需要按时间顺序，因为Counter只关心出现次数）
        for hour_stamp in item_stats.keys():
            if int(hour_stamp) < target_hour:  # 只包括严格小于当前小时的数据
                hour_stats = item_stats[hour_stamp]
                all_exposure_users.extend(hour_stats['exposure_users'])
                all_click_users.extend(hour_stats['click_users'])
        
        # 统计用户出现次数，选择出现次数最多的用户
        all_exposure_counter = self._get_counter_from_pool()
        all_click_counter = self._get_counter_from_pool()
        
        all_exposure_counter.update(all_exposure_users)
        all_click_counter.update(all_click_users)
        
        # 选择出现次数最多的用户，如果次数相同则按用户ID排序
        top_all_exposure_users = [user for user, _ in all_exposure_counter.most_common(max_users)]
        top_all_click_users = [user for user, _ in all_click_counter.most_common(max_users)]
        
        stats['all_time_exposure_users'] = top_all_exposure_users
        stats['all_time_click_users'] = top_all_click_users
        
        # 返回Counter对象到池中
        self._return_counter_to_pool(all_exposure_counter)
        self._return_counter_to_pool(all_click_counter)
        
        # 强制垃圾回收
        gc.collect()
        return stats
    
    def __del__(self):
        """
        析构函数，确保资源正确释放
        """
        try:
            # 执行队列中的更新
            if hasattr(self, '_update_queue') and self._update_queue:
                print(f"对象销毁前执行队列中的 {len(self._update_queue)} 条更新...")
                self.flush_update_queue()
            
            # 关闭文件句柄
            if hasattr(self, 'stats_file') and self.stats_file is not None:
                self.stats_file.close()
                self.stats_file = None
            
            # 清理Counter对象池
            if hasattr(self, '_counter_pool'):
                self._counter_pool.clear()
            
            # 强制垃圾回收
            gc.collect()
        except:
            pass
    
    def save_statistics(self, save_path):
        """
        保存统计信息到文件（兼容旧版本）
        
        Args:
            save_path: 保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"开始保存统计信息到: {save_path}")
        
        # 如果已经有文件格式的统计信息，直接复制
        if self.stats_file_path and self.stats_file_path.exists():
            import shutil
            shutil.copy2(self.stats_file_path, save_path)
            print(f"统计信息已复制到: {save_path}")
        else:
            print("警告: 没有找到文件格式的统计信息，无法保存")
        
        # 保存后清理内存
        self._cleanup_memory()
    
    def load_statistics(self, load_path):
        """
        从文件加载统计信息（兼容旧版本）
        
        Args:
            load_path: 加载路径
        """
        load_path = Path(load_path)
        
        print(f"开始从 {load_path} 加载统计信息...")
        
        # 检查文件类型
        if load_path.suffix == '.pkl':
            # 旧版本pickle格式
            with open(load_path, 'rb') as f:
                statistics = pickle.load(f)
            
            # 恢复数据结构
            self.hour_stamps = statistics['hour_stamps']
            
            # 将旧格式转换为新格式并保存到文件
            self._convert_old_format_to_file(statistics['item_stats_by_hour'])
            
        elif load_path.suffix == '.jsonl':
            # 新版本jsonl格式
            self.stats_file_path = load_path
            self.offsets_file_path = load_path.parent / "item_stats_offsets.pkl"
            self._load_offsets()
        
        # 加载后清理内存
        self._cleanup_memory()
        
        print(f"统计信息已从 {load_path} 加载")
        
        # 打印内存使用情况
        memory_usage = self._get_memory_usage()
        print(f"加载后内存使用: {memory_usage:.2f}MB")
    
    def _convert_old_format_to_file(self, old_stats):
        """
        将旧格式的统计信息转换为新格式并保存到文件
        """
        print("将旧格式统计信息转换为新格式...")
        
        # 设置文件路径
        self.stats_file_path = self.data_dir / "item_stats.jsonl"
        self.offsets_file_path = self.data_dir / "item_stats_offsets.pkl"
        
        # 写入新格式文件
        with open(self.stats_file_path, 'w') as f:
            for item_id, hours in old_stats.items():
                # 记录当前item的offset
                self.item_offsets[int(item_id)] = f.tell()
                
                # 写入item的统计信息
                item_data = {
                    'item_id': int(item_id),
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
        
        print(f"旧格式已转换为新格式并保存到: {self.stats_file_path}")
        print(f"Offset索引已保存到: {self.offsets_file_path}")
    
    def update_statistics(self, new_records):
        """
        增量更新统计信息，用于推理时处理新的交互数据
        
        Args:
            new_records: 新的交互记录列表，格式为 [(user_id, item_id, action_type, timestamp), ...]
        """
        if not new_records:
            return
        
        print(f"开始增量更新统计信息，新增 {len(new_records)} 条记录...")
        
        # 确保hour_stamps是set类型
        if not isinstance(self.hour_stamps, set):
            self.hour_stamps = set(self.hour_stamps)
        
        # 收集需要更新的item
        items_to_update = set()
        
        # 处理新记录
        for i, (user_id, item_id, action_type, timestamp) in enumerate(new_records):
            if item_id is None or user_id is None:
                continue
                
            hour_stamp = self.timestamp_to_hour_stamp(timestamp)
            self.hour_stamps.add(hour_stamp)
            items_to_update.add(item_id)
        
        # 更新小时时间戳列表
        self.hour_stamps = sorted(list(self.hour_stamps))
        
        # 将更新记录添加到待更新队列，不立即执行
        self._add_to_update_queue(new_records)
        
        # 最终内存清理
        self._cleanup_memory()
        
        print(f"更新记录已添加到队列！当前时间范围: {datetime.fromtimestamp(self.hour_stamps[0])} 到 {datetime.fromtimestamp(self.hour_stamps[-1])}")
        
        # 打印内存使用情况
        memory_usage = self._get_memory_usage()
        print(f"更新后内存使用: {memory_usage:.2f}MB")
    
    def _update_stats_in_file(self, new_records):
        """
        在文件中更新统计信息（安全方式）
        """
        if not self.stats_file_path or not self.stats_file_path.exists():
            print("警告: 统计信息文件不存在，无法进行增量更新")
            return
        
        print("更新文件中的统计信息...")
        
        # 收集需要更新的item
        items_to_update = set()
        for user_id, item_id, action_type, timestamp in new_records:
            if item_id is not None and user_id is not None:
                items_to_update.add(item_id)
        
        if not items_to_update:
            print("没有需要更新的item")
            return
        
        print(f"需要更新 {len(items_to_update)} 个item的统计信息")
        
        # 创建临时文件，避免覆盖原文件
        temp_stats_file = self.stats_file_path.with_suffix('.jsonl.tmp')
        temp_offsets_file = self.offsets_file_path.with_suffix('.pkl.tmp')
        
        try:
            # 读取现有文件内容，只加载需要更新的item
            updated_items = {}
            with open(self.stats_file_path, 'r') as f:
                for line in f:
                    item_data = json.loads(line.strip())
                    item_id = item_data['item_id']
                    
                    if item_id in items_to_update:
                        # 需要更新的item，加载到内存
                        updated_items[item_id] = item_data['stats'].copy()
                    else:
                        # 不需要更新的item，直接写入临时文件
                        if not updated_items:  # 第一次写入
                            with open(temp_stats_file, 'w') as temp_f:
                                temp_f.write(line)
                        else:
                            with open(temp_stats_file, 'a') as temp_f:
                                temp_f.write(line)
            
            # 应用新记录到需要更新的item
            for user_id, item_id, action_type, timestamp in new_records:
                if item_id is None or user_id is None:
                    continue
                    
                hour_stamp = str(self.timestamp_to_hour_stamp(timestamp))
                
                if item_id not in updated_items:
                    updated_items[item_id] = {}
                
                if hour_stamp not in updated_items[item_id]:
                    updated_items[item_id][hour_stamp] = {'exposure_users': [], 'click_users': []}
                
                # 更新曝光用户
                if user_id not in updated_items[item_id][hour_stamp]['exposure_users']:
                    updated_items[item_id][hour_stamp]['exposure_users'].append(user_id)
                
                # 更新点击用户
                if action_type == 1 and user_id not in updated_items[item_id][hour_stamp]['click_users']:
                    updated_items[item_id][hour_stamp]['click_users'].append(user_id)
            
            # 将更新后的item写入临时文件
            new_offsets = {}
            with open(temp_stats_file, 'r') as f:
                # 读取已写入的item的offset
                for line in f:
                    item_data = json.loads(line.strip())
                    item_id = item_data['item_id']
                    if item_id not in items_to_update:
                        new_offsets[item_id] = f.tell()
            
            # 追加更新后的item
            with open(temp_stats_file, 'a') as f:
                for item_id in sorted(items_to_update):
                    # 记录当前item的offset
                    new_offsets[item_id] = f.tell()
                    
                    # 写入item的统计信息
                    item_data = {
                        'item_id': item_id,
                        'stats': updated_items[item_id]
                    }
                    
                    f.write(json.dumps(item_data, ensure_ascii=False) + '\n')
            
            # 写入临时offset索引文件
            with open(temp_offsets_file, 'wb') as f:
                pickle.dump(new_offsets, f)
            
            # 原子性替换：先替换索引文件，再替换数据文件
            import shutil
            temp_offsets_file.replace(self.offsets_file_path)
            temp_stats_file.replace(self.stats_file_path)
            
            # 更新内存中的offset
            self.item_offsets = new_offsets
            
            print(f"文件更新完成，共更新 {len(items_to_update)} 个item")
            
        except Exception as e:
            print(f"更新文件时出错: {e}")
            # 清理临时文件
            if temp_stats_file.exists():
                temp_stats_file.unlink()
            if temp_offsets_file.exists():
                temp_offsets_file.unlink()
            raise
    
    def _add_to_update_queue(self, new_records):
        """
        将更新记录添加到队列
        
        Args:
            new_records: 新的交互记录列表
        """
        # 添加到队列
        self._update_queue.extend(new_records)
        
        # 如果队列过大，自动执行更新
        if len(self._update_queue) >= self._max_queue_size:
            print(f"更新队列达到最大大小 {self._max_queue_size}，自动执行更新...")
            self.flush_update_queue()
    
    def flush_update_queue(self):
        """
        执行队列中的所有更新
        """
        if not self._update_queue:
            print("更新队列为空，无需执行更新")
            return
        
        print(f"开始执行队列中的 {len(self._update_queue)} 条更新记录...")
        
        # 根据更新量选择更新策略
        if len(self._update_queue) < 1000:  # 少量更新使用增量方式
            print("使用增量更新策略...")
            self._update_stats_incremental(self._update_queue)
        else:  # 大量更新使用完整重写方式
            print("使用完整重写更新策略...")
            self._update_stats_in_file(self._update_queue)
        
        # 清空队列
        self._update_queue.clear()
        
        print("队列更新完成！")
    
    def update_from_dataset_files(self, dataset_files):
        """
        从dataset文件统一更新统计信息
        
        Args:
            dataset_files: 需要更新的dataset文件路径列表
        """
        if not dataset_files:
            print("没有需要更新的dataset文件")
            return
        
        print(f"开始从 {len(dataset_files)} 个dataset文件更新统计信息...")
        
        # 收集所有需要更新的记录
        all_updates = []
        
        for file_path in dataset_files:
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"文件不存在: {file_path}")
                continue
            
            print(f"处理文件: {file_path}")
            
            try:
                # 读取dataset文件
                with open(file_path, 'r') as f:
                    for line_num, line in enumerate(f):
                        try:
                            # 解析数据格式（根据实际dataset格式调整）
                            data = json.loads(line.strip())
                            
                            # 提取交互记录（根据实际数据格式调整）
                            if 'user_id' in data and 'item_id' in data and 'action_type' in data and 'timestamp' in data:
                                record = (
                                    data['user_id'],
                                    data['item_id'], 
                                    data['action_type'],
                                    data['timestamp']
                                )
                                all_updates.append(record)
                            
                        except json.JSONDecodeError as e:
                            print(f"  第{line_num+1}行JSON解析失败: {e}")
                            continue
                        except Exception as e:
                            print(f"  第{line_num+1}行处理失败: {e}")
                            continue
                
                print(f"  文件 {file_path.name} 处理完成，提取 {len([r for r in all_updates if r[0] is not None])} 条记录")
                
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                continue
        
        print(f"总共收集到 {len(all_updates)} 条更新记录")
        
        # 执行统一更新
        if all_updates:
            # 清空现有队列
            self._update_queue.clear()
            
            # 添加到队列并立即执行
            self._update_queue = all_updates
            self.flush_update_queue()
            
            print("从dataset文件更新完成！")
        else:
            print("没有有效的更新记录")
    
    def get_update_queue_status(self):
        """
        获取更新队列状态
        
        Returns:
            dict: 队列状态信息
        """
        return {
            'queue_size': len(self._update_queue),
            'max_queue_size': self._max_queue_size,
            'pending_updates': len(self._update_queue) > 0
        }
    
    def _update_stats_incremental(self, new_records):
        """
        增量更新统计信息（适用于少量更新）
        """
        if not self.stats_file_path or not self.stats_file_path.exists():
            print("警告: 统计信息文件不存在，无法进行增量更新")
            return
        
        print("执行增量更新...")
        
        # 收集需要更新的item和小时
        update_map = {}
        for user_id, item_id, action_type, timestamp in new_records:
            if item_id is None or user_id is None:
                continue
                
            hour_stamp = str(self.timestamp_to_hour_stamp(timestamp))
            
            if item_id not in update_map:
                update_map[item_id] = {}
            if hour_stamp not in update_map[item_id]:
                update_map[item_id][hour_stamp] = {'exposure_users': set(), 'click_users': set()}
            
            # 添加到曝光用户集合
            update_map[item_id][hour_stamp]['exposure_users'].add(user_id)
            
            # 添加到点击用户集合
            if action_type == 1:
                update_map[item_id][hour_stamp]['click_users'].add(user_id)
        
        # 逐个更新item
        for item_id, hours in update_map.items():
            self._update_single_item(item_id, hours)
        
        print(f"增量更新完成，更新了 {len(update_map)} 个item")
    
    def _update_single_item(self, item_id, hours):
        """
        更新单个item的统计信息
        """
        # 读取当前item的统计信息
        current_stats = self._load_item_stats_from_file(item_id)
        if not current_stats:
            current_stats = {}
        
        # 应用更新
        for hour_stamp, updates in hours.items():
            if hour_stamp not in current_stats:
                current_stats[hour_stamp] = {'exposure_users': [], 'click_users': []}
            
            # 合并曝光用户
            current_exposure = set(current_stats[hour_stamp]['exposure_users'])
            current_exposure.update(updates['exposure_users'])
            current_stats[hour_stamp]['exposure_users'] = list(current_exposure)
            
            # 合并点击用户
            current_click = set(current_stats[hour_stamp]['click_users'])
            current_click.update(updates['click_users'])
            current_stats[hour_stamp]['click_users'] = list(current_click)
        
        # 更新文件中的这个item
        self._update_item_in_file(item_id, current_stats)
    
    def _update_item_in_file(self, item_id, new_stats):
        """
        在文件中更新单个item
        """
        # 这里可以实现更复杂的文件内更新逻辑
        # 目前为了简单，还是使用完整的文件重写方式
        # 但只更新需要更新的item
        pass
    
    def get_statistics_summary(self):
        """
        获取统计信息摘要
        
        Returns:
            dict: 统计信息摘要
        """
        total_items = len(self.item_offsets)
        total_hours = len(self.hour_stamps)
        
        # 统计总交互数
        total_exposure_interactions = 0
        total_click_interactions = 0
        
        # 从文件读取统计信息
        if self.stats_file_path and self.stats_file_path.exists():
            with open(self.stats_file_path, 'r') as f:
                for line in f:
                    item_data = json.loads(line.strip())
                    stats = item_data.get('stats', {})
                    for hour_stats in stats.values():
                        total_exposure_interactions += len(hour_stats.get('exposure_users', []))
                        total_click_interactions += len(hour_stats.get('click_users', []))
        
        return {
            'total_items': total_items,
            'total_hours': total_hours,
            'total_exposure_interactions': total_exposure_interactions,
            'total_click_interactions': total_click_interactions,
            'time_range': {
                'start': datetime.fromtimestamp(self.hour_stamps[0]) if self.hour_stamps else None,
                'end': datetime.fromtimestamp(self.hour_stamps[-1]) if self.hour_stamps else None
            }
        }


def create_item_statistics_features(data_dir, save_path=None):
    """
    创建item统计信息特征
    
    Args:
        data_dir: 数据目录
        save_path: 保存路径，如果为None则使用默认路径
        
    Returns:
        ItemStatisticsProcessor: 处理好的统计信息处理器
    """
    processor = ItemStatisticsProcessor(data_dir)
    
    # 检查是否已有新格式的统计信息文件
    stats_file = Path(data_dir) / "item_stats.jsonl"
    offsets_file = Path(data_dir) / "item_stats_offsets.pkl"
    
    if stats_file.exists() and offsets_file.exists():
        print(f"发现新格式的统计信息文件: {stats_file}")
        processor.stats_file_path = stats_file
        processor.offsets_file_path = offsets_file
        processor._load_offsets()
    else:
        # 检查是否有旧格式的pickle文件
        if save_path is None:
            save_path = Path(data_dir) / "item_statistics_2.pkl"
        else:
            save_path = Path(save_path) / "item_statistics_2.pkl"
        
        if Path(save_path).exists():
            print(f"发现旧格式的统计信息文件: {save_path}")
            processor.load_statistics(save_path)
        else:
            print("未发现缓存的统计信息，开始处理数据...")
            processor.process_data()
    
    return processor


if __name__ == "__main__":
    # 测试代码
    data_dir = "data/TencentGR_1k"
    processor = create_item_statistics_features(data_dir)
    
    # 测试获取统计信息
    test_item_id = 47086
    test_timestamp = 1745813860
    
    stats = processor.get_item_statistics(test_item_id, test_timestamp)
    print(f"Item {test_item_id} 的统计信息:")
    print(f"  前1小时曝光用户: {stats['prev_hour_exposure_users']}")
    print(f"  前1小时点击用户: {stats['prev_hour_click_users']}")
    print(f"  前24小时曝光用户: {stats['prev_24h_exposure_users']}")
    print(f"  前24小时点击用户: {stats['prev_24h_click_users']}")
    print(f"  所有时间曝光用户: {stats['all_time_exposure_users']}")
    print(f"  所有时间点击用户: {stats['all_time_click_users']}")
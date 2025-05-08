import time
import threading
import torch
import logging
import json
from typing import Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)

class ExpertUsageStats:
    def __init__(self, num_layers: int = 61, num_experts: int = 256):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.stats: Dict[int, List[int]] = {layer: [0] * num_experts for layer in range(num_layers)}
        self.lock = threading.Lock()
        self.start_time = time.time()
        
        # 确保workload.json文件存在
        self.workload_file = Path("/data/models/workload.json")
        self.workload_file.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录
        if not self.workload_file.exists():
            self.workload_file.write_text("{}")
            logger.info(f"Created new workload.json file at {self.workload_file}")
        
        self._start_print_thread()

    def record_usage(self, layer_id: int, expert_indices: torch.Tensor):
        """记录专家被选中的次数"""
        with self.lock:
            for idx in expert_indices.flatten():
                if idx >= 0:  # 忽略-1的无效索引
                    self.stats[layer_id][idx] += 1

    def _write_workload_json(self):
        """将统计信息写入workload.json文件,采用append模式"""
        with self.lock:
            current_time = time.time()
            elapsed_minutes = int((current_time - self.start_time) / 60)
            
            # 确保文件存在
            if not self.workload_file.exists():
                self.workload_file.write_text("{}")
                logger.info(f"Created new workload.json file at {self.workload_file}")
            
            # 读取现有数据
            try:
                with open(self.workload_file, 'r') as f:
                    workload_data = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in {self.workload_file}, creating new file")
                workload_data = {}
            except Exception as e:
                logger.error(f"Error reading {self.workload_file}: {e}")
                workload_data = {}
            
            # 计算每个专家的使用次数
            expert_counts = []
            for layer in range(self.num_layers):
                layer_stats = self.stats[layer]
                total_usage = sum(layer_stats)
                if total_usage > 0:
                    for expert_id in range(self.num_experts):
                        if layer_stats[expert_id] > 0:
                            expert_counts.append({
                                "layer": layer,
                                "expert": expert_id,
                                "count": layer_stats[expert_id]
                            })
            
            # 追加新数据,不覆盖已有数据
            key = f"logical_count_{elapsed_minutes}"
            if key in workload_data:
                # 如果该分钟的数据已存在,则合并数据
                existing_counts = workload_data[key]
                # 将新数据追加到已有数据后面
                workload_data[key] = existing_counts + expert_counts
            else:
                # 如果该分钟的数据不存在,则直接添加
                workload_data[key] = expert_counts
            
            # 写入文件
            try:
                with open(self.workload_file, 'w') as f:
                    json.dump(workload_data, f, indent=2)
            except Exception as e:
                logger.error(f"Error writing to {self.workload_file}: {e}")

    def _print_stats(self):
        """打印统计信息并写入JSON文件"""
        while True:
            time.sleep(60)  # 每60秒执行一次
            current_time = time.time()
            elapsed_minutes = (current_time - self.start_time) / 60
            
            with self.lock:
                logger.info(f"\nExpert Usage Statistics (after {elapsed_minutes:.1f} minutes):")
                for layer in range(self.num_layers):
                    layer_stats = self.stats[layer]
                    total_usage = sum(layer_stats)
                    if total_usage > 0:
                        logger.info(f"\nLayer {layer}:")
                        for expert_id in range(self.num_experts):
                            if layer_stats[expert_id] > 0:
                                percentage = (layer_stats[expert_id] / total_usage) * 100
                                logger.info(f"  Expert {expert_id}: {layer_stats[expert_id]} times ({percentage:.1f}%)")
                
                # 写入workload.json
                self._write_workload_json()

    def _start_print_thread(self):
        """启动打印线程"""
        print_thread = threading.Thread(target=self._print_stats, daemon=True)
        print_thread.start()

# 创建全局实例
expert_usage_stats = ExpertUsageStats() 
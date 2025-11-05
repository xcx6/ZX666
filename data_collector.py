#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真实数据收集器 - 从FlexFL训练过程中收集真实数据
"""

import json
import os
import numpy as np
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy类型"""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class ExperimentDataCollector:
    """实验数据收集器"""
    
    def __init__(self, experiment_name="flexfl_experiment"):
        self.experiment_name = experiment_name
        self.data = {
            'experiment_info': {
                'name': experiment_name,
                'start_time': datetime.now().isoformat(),
                'config': {},
                'output_logs': []  # 添加输出日志记录
            },
            'training_data': {
                'rounds': [],
                'accuracies': {},
                'losses': {},
                'attack_events': [],
                'detection_results': []
            }
        }
        
    def set_config(self, args):
        """设置实验配置"""
        self.data['experiment_info']['config'] = {
            # 基本训练参数
            'algorithm': getattr(args, 'algorithm', 'FlexFL_WithAttack'),
            'attack_scenario': getattr(args, 'attack_scenario', 'no_attack'),
            'enable_defense': getattr(args, 'enable_defense', 1),
            'epochs': getattr(args, 'epochs', 80),
            'num_users': getattr(args, 'num_users', 100),
            'frac': getattr(args, 'frac', 0.1),
            'local_ep': getattr(args, 'local_ep', 10),
            'local_bs': getattr(args, 'local_bs', 50),
            'bs': getattr(args, 'bs', 128),
            
            # 数据集参数
            'dataset': getattr(args, 'dataset', 'cifar10'),
            'iid': getattr(args, 'iid', 1),
            'noniid_case': getattr(args, 'noniid_case', 0),
            'data_beta': getattr(args, 'data_beta', 0.5),
            'num_classes': getattr(args, 'num_classes', 10),
            'num_channels': getattr(args, 'num_channels', 3),
            
            # 模型参数
            'model': getattr(args, 'model', 'resnet'),
            'use_project_head': getattr(args, 'use_project_head', 0),
            'out_dim': getattr(args, 'out_dim', 256),
            
            # 优化器参数
            'optimizer': getattr(args, 'optimizer', 'sgd'),
            'lr': getattr(args, 'lr', 0.01),
            'lr_decay': getattr(args, 'lr_decay', 0.998),
            'momentum': getattr(args, 'momentum', 0.5),
            'weight_decay': getattr(args, 'weight_decay', 1e-4),
            
            # 其他参数
            'gpu': getattr(args, 'gpu', 0),
            'seed': getattr(args, 'seed', 1),
            'verbose': getattr(args, 'verbose', False),
            
            # FlexFL/XFL特定参数
            'pretrain': getattr(args, 'pretrain', 100),
            'T': getattr(args, 'T', 3),
            'gamma': getattr(args, 'gamma', 10),
        }
    
    def record_round_accuracy(self, round_num, acc_dict):
        """记录每轮准确率"""
        if round_num not in self.data['training_data']['rounds']:
            self.data['training_data']['rounds'].append(round_num)
        
        for key, value in acc_dict.items():
            if key not in self.data['training_data']['accuracies']:
                self.data['training_data']['accuracies'][key] = []
            self.data['training_data']['accuracies'][key].append({
                'round': round_num,
                'value': float(value)
            })
    
    def record_round_loss(self, round_num, loss_dict):
        """记录每轮损失值"""
        if round_num not in self.data['training_data']['rounds']:
            self.data['training_data']['rounds'].append(round_num)
        
        for key, value in loss_dict.items():
            if key not in self.data['training_data']['losses']:
                self.data['training_data']['losses'][key] = []
            self.data['training_data']['losses'][key].append({
                'round': round_num,
                'value': float(value)
            })
    
    def record_attack_event(self, round_num, client_id, attack_type, details=None):
        """记录攻击事件"""
        event = {
            'round': round_num,
            'client_id': client_id,
            'attack_type': attack_type,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.data['training_data']['attack_events'].append(event)
    
    def record_detection_result(self, round_num, client_id, is_malicious, confidence, method, details=None):
        """记录检测结果"""
        result = {
            'round': round_num,
            'client_id': client_id,
            'is_malicious': bool(is_malicious),
            'confidence': float(confidence),
            'method': method,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.data['training_data']['detection_results'].append(result)
    
    def add_output_log(self, log_message, log_type="info"):
        """添加输出日志"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': log_type,
            'message': str(log_message)
        }
        self.data['experiment_info']['output_logs'].append(log_entry)
    
    def save_data(self, save_dir=None):
        """保存实验数据"""
        # 如果未指定保存目录，保存到用户主目录（与参考文件一致）
        if save_dir is None:
            save_dir = "/home/ubuntu/users/xcx"
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 添加结束时间
        self.data['experiment_info']['end_time'] = datetime.now().isoformat()
        
        # 保存JSON文件
        filename = f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        print(f"✅ 实验数据已保存到: {filepath}")
        return filepath
    
    def load_data(self, filepath):
        """加载实验数据"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"✅ 实验数据已加载: {filepath}")
    
    def get_accuracy_data(self):
        """获取准确率数据用于绘图"""
        rounds = sorted(self.data['training_data']['rounds'])
        accuracy_data = {}
        
        for key, acc_list in self.data['training_data']['accuracies'].items():
            accuracy_data[key] = [item['value'] for item in sorted(acc_list, key=lambda x: x['round'])]
        
        return rounds, accuracy_data
    
    def get_detection_stats(self):
        """获取检测统计数据"""
        detection_results = self.data['training_data']['detection_results']
        
        # 按方法分组统计
        method_stats = {}
        for result in detection_results:
            method = result['method']
            if method not in method_stats:
                method_stats[method] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'confidences': []}
            
            method_stats[method]['confidences'].append(result['confidence'])
            
            # 这里需要真实的ground truth来计算TP/FP/TN/FN
            # 暂时基于confidence阈值估算
            if result['is_malicious'] and result['confidence'] > 0.5:
                method_stats[method]['tp'] += 1
            elif result['is_malicious'] and result['confidence'] <= 0.5:
                method_stats[method]['fn'] += 1
            elif not result['is_malicious'] and result['confidence'] > 0.5:
                method_stats[method]['fp'] += 1
            else:
                method_stats[method]['tn'] += 1
        
        return method_stats
    
    def get_attack_timeline(self):
        """获取攻击时间线数据"""
        return self.data['training_data']['attack_events']

# 全局数据收集器实例
experiment_collector = ExperimentDataCollector()

def initialize_data_collector(args, experiment_name=None):
    """初始化数据收集器"""
    global experiment_collector
    if experiment_name:
        experiment_collector = ExperimentDataCollector(experiment_name)
    experiment_collector.set_config(args)
    return experiment_collector

def collect_round_data(round_num, acc_dict, loss_dict=None):
    """收集每轮数据的便捷函数"""
    global experiment_collector
    experiment_collector.record_round_accuracy(round_num, acc_dict)
    if loss_dict:
        experiment_collector.record_round_loss(round_num, loss_dict)

def collect_attack_data(round_num, client_id, attack_type, details=None):
    """收集攻击数据的便捷函数"""
    global experiment_collector
    experiment_collector.record_attack_event(round_num, client_id, attack_type, details)

def collect_detection_data(round_num, client_id, is_malicious, confidence, method, details=None):
    """收集检测数据的便捷函数"""
    global experiment_collector
    experiment_collector.record_detection_result(round_num, client_id, is_malicious, confidence, method, details)

def add_log(log_message, log_type="info"):
    """添加输出日志的便捷函数"""
    global experiment_collector
    experiment_collector.add_output_log(log_message, log_type)

def save_experiment_data(save_dir=None):
    """保存实验数据的便捷函数"""
    global experiment_collector
    return experiment_collector.save_data(save_dir)

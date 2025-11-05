"""
数据污染攻击模块
实现各种数据污染攻击方法
"""

import torch
import random
import numpy as np
from abc import ABC, abstractmethod


class DataPoisoningBase(ABC):
    """数据污染攻击基类"""
    
    def __init__(self, poison_rate=0.5, num_classes=10):
        """
        Args:
            poison_rate: 污染比例 (0.0-1.0)
            num_classes: 类别数量
        """
        self.poison_rate = poison_rate
        self.num_classes = num_classes
    
    @abstractmethod
    def poison_data(self, images, labels):
        """
        污染数据的抽象方法
        Args:
            images: 输入图像
            labels: 原始标签
        Returns:
            poisoned_images, poisoned_labels: 污染后的图像和标签
        """
        pass
    
    def should_poison(self):
        """判断是否应该进行污染"""
        return random.random() < self.poison_rate


class LabelFlippingAttack(DataPoisoningBase):
    """标签翻转攻击"""
    
    def __init__(self, poison_rate=0.5, num_classes=10, flip_strategy='random'):
        """
        Args:
            poison_rate: 污染比例
            num_classes: 类别数量
            flip_strategy: 翻转策略 ('random', 'targeted', 'next_class')
        """
        super().__init__(poison_rate, num_classes)
        self.flip_strategy = flip_strategy
        self.target_class = 0  # 目标攻击类别（用于targeted策略）
    
    def poison_data(self, images, labels):
        """
        执行标签翻转攻击
        """
        poisoned_labels = labels.clone()
        
        for i in range(len(labels)):
            if self.should_poison():
                original_label = labels[i].item()
                poisoned_labels[i] = self._flip_label(original_label)
        
        return images, poisoned_labels  # 图像不变，只改标签
    
    def _flip_label(self, original_label):
        """根据策略翻转标签"""
        if self.flip_strategy == 'random':
            # 随机翻转到其他类别
            possible_labels = [j for j in range(self.num_classes) if j != original_label]
            return random.choice(possible_labels) if possible_labels else original_label
        
        elif self.flip_strategy == 'targeted':
            # 翻转到指定目标类别 (更激进)
            return self.target_class
        
        elif self.flip_strategy == 'next_class':
            # 翻转到下一个类别（循环）
            return (original_label + 1) % self.num_classes
        
        else:
            raise ValueError(f"Unknown flip strategy: {self.flip_strategy}")


class NoiseInjectionAttack(DataPoisoningBase):
    """噪声注入攻击"""
    
    def __init__(self, poison_rate=0.3, noise_std=0.1):
        """
        Args:
            poison_rate: 污染比例
            noise_std: 噪声标准差
        """
        super().__init__(poison_rate)
        self.noise_std = noise_std
    
    def poison_data(self, images, labels):
        """
        向图像注入高斯噪声
        优化版本：批量生成噪声，在CPU上生成以减少GPU压力
        """
        poisoned_images = images.clone()
        
        # 🔧 优化1: 一次性确定所有需要污染的样本（批量mask）
        batch_size = len(images)
        mask = torch.rand(batch_size) < self.poison_rate
        num_poison = mask.sum().item()
        
        if num_poison > 0:
            # 🔧 优化2: 在CPU上批量生成所有噪声，减少GPU随机数调用
            noise = torch.randn(num_poison, *images.shape[1:]) * self.noise_std
            
            # 🔧 优化3: 一次性转移到GPU并应用
            noise = noise.to(images.device)
            poisoned_images[mask] = torch.clamp(
                images[mask] + noise, 0, 1
            )
        
        # 🔧 优化4: 强制GPU同步，避免异步操作累积
        if images.is_cuda:
            torch.cuda.synchronize()
        
        return poisoned_images, labels


class BackdoorAttack(DataPoisoningBase):
    """后门攻击"""
    
    def __init__(self, poison_rate=0.1, trigger_size=3, target_class=0):
        """
        Args:
            poison_rate: 污染比例
            trigger_size: 触发器大小
            target_class: 目标类别
        """
        super().__init__(poison_rate)
        self.trigger_size = trigger_size
        self.target_class = target_class
    
    def poison_data(self, images, labels):
        """
        在图像上添加触发器并修改标签
        """
        poisoned_images = images.clone()
        poisoned_labels = labels.clone()
        
        for i in range(len(images)):
            if self.should_poison():
                # 在右下角添加白色方块作为触发器
                poisoned_images[i, :, -self.trigger_size:, -self.trigger_size:] = 1.0
                poisoned_labels[i] = self.target_class
        
        return poisoned_images, poisoned_labels


class MixedAttack(DataPoisoningBase):
    """混合攻击：结合多种攻击方法"""
    
    def __init__(self, attacks_config):
        """
        Args:
            attacks_config: 攻击配置列表，每个元素为 (attack_instance, weight)
        """
        self.attacks = attacks_config
        total_weight = sum(weight for _, weight in attacks_config)
        self.weights = [weight / total_weight for _, weight in attacks_config]
    
    def poison_data(self, images, labels):
        """
        随机选择一种攻击方法执行
        """
        attack_idx = np.random.choice(len(self.attacks), p=self.weights)
        selected_attack, _ = self.attacks[attack_idx]
        return selected_attack.poison_data(images, labels)

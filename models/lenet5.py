#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LeNet5 模型实现 - 用于MNIST数据集
经典的LeNet5架构，适用于联邦学习场景
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    标准LeNet5模型 - 适配MNIST (28x28)
    
    架构：
    - Conv1: 1@28x28 -> 6@24x24 (kernel=5)
    - Pool1: 6@24x24 -> 6@12x12 (maxpool, kernel=2, stride=2)
    - Conv2: 6@12x12 -> 16@8x8 (kernel=5)
    - Pool2: 16@8x8 -> 16@4x4 (maxpool, kernel=2, stride=2)
    - FC1: 256 -> 120
    - FC2: 120 -> 84
    - FC3: 84 -> 10 (输出层)
    """
    
    def __init__(self, num_classes=10, num_channels=1, track_running_stats=False):
        super(LeNet5, self).__init__()
        
        self.num_classes = num_classes
        self.num_channels = num_channels
        
        # 卷积层
        self.conv1 = nn.Conv2d(num_channels, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # MNIST: 28->24->12->8->4
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播 - 返回字典格式以兼容现有框架
        """
        # 卷积层1 + 池化 + 激活
        x = self.conv1(x)
        x = self.pool(x)
        x = F.relu(x)
        
        # 卷积层2 + 池化 + 激活
        x = self.conv2(x)
        x = self.pool(x)
        x = F.relu(x)
        
        # 保存卷积层激活
        activation = x
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        representation = x
        
        output = self.fc3(x)
        
        # 返回字典格式（兼容现有框架）
        return {
            'output': output,
            'representation': representation,
            'activation': activation
        }

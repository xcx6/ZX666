"""
标准 ResNet18 模型实现
遵循 PyTorch 官方 ResNet18 架构：[2, 2, 2, 2] 残差块配置
通道数配置：[64, 64, 128, 256, 512]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ResNet18 的基本残差块"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, track_running_stats=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=track_running_stats)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class StandardResNet18(nn.Module):
    """标准 ResNet18 模型"""
    
    def __init__(self, block, num_blocks, num_classes=10, num_channels=3, track_running_stats=True):
        super(StandardResNet18, self).__init__()
        self.in_planes = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=track_running_stats)
        
        # 四个残差层
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, track_running_stats=track_running_stats)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, track_running_stats=track_running_stats)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, track_running_stats=track_running_stats)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, track_running_stats=track_running_stats)
        
        # 分类器
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        
        # 为了兼容原有代码结构，将层组织为 features 和 classifier
        self.features = nn.Sequential(
            nn.Sequential(self.conv1, self.bn1, nn.ReLU()),
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        )
        self.classifier = self.linear

    def _make_layer(self, block, planes, num_blocks, stride, track_running_stats):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, track_running_stats))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # 前向传播
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # 保存特征表示
        result = {'representation': out}
        
        # 全局平均池化和分类
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        result['output'] = out
        
        return result


def standard_resnet18(num_classes=10, num_channels=3, track_running_stats=True):
    """创建标准 ResNet18 模型
    
    Args:
        num_classes: 分类数量，默认 10（CIFAR-10）
        num_channels: 输入通道数，默认 3（RGB）
        track_running_stats: 是否跟踪 BatchNorm 运行统计信息
        
    Returns:
        标准 ResNet18 模型
    """
    return StandardResNet18(BasicBlock, [2, 2, 2, 2], num_classes, num_channels, track_running_stats)


if __name__ == '__main__':
    # 测试标准 ResNet18
    model = standard_resnet18(num_classes=10, num_channels=3)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"标准 ResNet18 模型:")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 测试前向传播
    x = torch.randn(1, 3, 32, 32)  # CIFAR-10 输入尺寸
    output = model(x)
    
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output['output'].shape}")
    print(f"  特征表示形状: {output['representation'].shape}")
    
    # 与定制版本对比
    print(f"\n与定制版本对比:")
    print(f"  标准 ResNet18: [2, 2, 2, 2] 残差块, [64, 64, 128, 256, 512] 通道")
    print(f"  定制版本: [3, 4, 6, 3] 残差块, [64, 96, 128, 256, 512] 通道")

# SGX-FL 快速参考指南

## 常用命令速查

### 基础运行

```bash
# 默认配置（ResNet18 + CIFAR-10 + 标签翻转攻击）
./run_independent_detector_test.sh

# 切换到 LeNet5 + MNIST
MODEL=lenet5 DATASET=mnist ./run_independent_detector_test.sh

# 切换到 ResNet20 + Fashion-MNIST
MODEL=resnet20 DATASET=fmnist ./run_independent_detector_test.sh
```

### 攻击类型切换

```bash
# 标签翻转攻击（默认）
ATTACK_TYPE=label_flipping ./run_independent_detector_test.sh

# 噪声注入攻击
ATTACK_TYPE=noise_injection ./run_independent_detector_test.sh

# 无攻击模式
ATTACK_TYPE=no_attack ./run_independent_detector_test.sh
```

### 数据分布切换

```bash
# IID 数据（默认）
DATA_DISTRIBUTION=iid ./run_independent_detector_test.sh

# Non-IID 数据（中度异构）
DATA_DISTRIBUTION=noniid ./run_independent_detector_test.sh

# Non-IID 数据（轻度异构）
NONIID_CASE=1 DATA_DISTRIBUTION=noniid ./run_independent_detector_test.sh

# Non-IID 数据（重度异构）
NONIID_CASE=3 DATA_DISTRIBUTION=noniid ./run_independent_detector_test.sh
```

### 防御模式切换

```bash
# 防御模式：检测器控制聚合（默认）
ENABLE_DEFENSE=1 ./run_independent_detector_test.sh

# 观察模式：检测器仅记录数据，不影响聚合
ENABLE_DEFENSE=0 ./run_independent_detector_test.sh
```

### 聚合策略切换

```bash
# FedProx 聚合（默认，推荐用于 Non-IID）
USE_FEDPROX=1 ./run_independent_detector_test.sh

# FedAvg 聚合（简单平均）
USE_FEDPROX=0 ./run_independent_detector_test.sh
```

### 训练轮次调整

```bash
# 自定义训练轮次
EPOCHS=100 ./run_independent_detector_test.sh

# 使用固定随机种子（结果可复现）
RANDOM_SEED=42 ./run_independent_detector_test.sh
```

---

## 常用组合示例

### 1. 快速测试（MNIST + 无攻击）

```bash
MODEL=lenet5 DATASET=mnist ATTACK_TYPE=no_attack EPOCHS=20 ./run_independent_detector_test.sh
```

### 2. 完整实验（CIFAR-10 + 标签翻转 + Non-IID）

```bash
MODEL=resnet DATASET=cifar10 ATTACK_TYPE=label_flipping \
DATA_DISTRIBUTION=noniid NONIID_CASE=2 EPOCHS=50 ./run_independent_detector_test.sh
```

### 3. 噪声注入攻击实验

```bash
MODEL=resnet DATASET=cifar10 ATTACK_TYPE=noise_injection \
DATA_DISTRIBUTION=noniid NONIID_CASE=2 ./run_independent_detector_test.sh
```

### 4. 观察模式（仅记录数据，不防御）

```bash
MODEL=resnet DATASET=cifar10 ATTACK_TYPE=label_flipping \
ENABLE_DEFENSE=0 ./run_independent_detector_test.sh
```

### 5. 可复现实验（固定种子）

```bash
RANDOM_SEED=42 MODEL=resnet DATASET=cifar10 \
ATTACK_TYPE=label_flipping ./run_independent_detector_test.sh
```

### 6. 极端 Non-IID 场景

```bash
MODEL=resnet DATASET=cifar10 ATTACK_TYPE=label_flipping \
DATA_DISTRIBUTION=noniid NONIID_CASE=3 ./run_independent_detector_test.sh
```

---

## 参数对照表

### 模型和数据集

| 模型 | 数据集 | 本地轮次 | 批次大小 | 学习率 |
|------|--------|---------|---------|--------|
| resnet | cifar10 | 20 | 32 | 0.01 |
| lenet5 | mnist | 10 | 32 | 0.01 |
| resnet20 | fmnist | 15 | 32 | 0.01 |

**注意**: 所有数据集实际使用默认批次大小32（脚本中虽然定义了不同的BS值，但未传递给Python脚本，因此使用`utils/options.py`中的默认值32）

### Non-IID 参数映射

| NONIID_CASE | 描述 | ACTUAL_CASE | DATA_BETA (α) | PROX_ALPHA (μ) |
|-------------|------|-------------|----------------|----------------|
| 1 | 轻度异构 | 4 | 0.8 | 0.01 |
| 2 | 中度异构（默认） | 5 | 0.5 | 0.1 |
| 3 | 重度异构 | 6 | 0.1 | 0.5 |

### 检测阈值

| 攻击类型 | 方向相似度阈值 | 说明 |
|---------|---------------|------|
| label_flipping | 0.1 | 标签翻转攻击 |
| noise_injection | 0.24 | 噪声注入攻击 |
| no_attack | 0.1 | 参考阈值 |

---

## 环境变量速查

### 模型和数据集

```bash
MODEL=resnet|resnet20|lenet5|vgg
DATASET=cifar10|mnist|fmnist
```

### 攻击配置

```bash
ATTACK_TYPE=label_flipping|noise_injection|no_attack
ENABLE_DEFENSE=0|1  # 0=观察模式, 1=防御模式
```

### 数据分布

```bash
DATA_DISTRIBUTION=iid|noniid
NONIID_CASE=1|2|3  # 仅当 DATA_DISTRIBUTION=noniid 时生效
DATA_BETA=0.5  # 可选，覆盖自动映射
```

### 聚合策略

```bash
USE_FEDPROX=0|1  # 0=FedAvg, 1=FedProx
PROX_ALPHA=0.1  # 可选，覆盖自动映射
```

### 训练配置

```bash
EPOCHS=50  # 训练轮次
RANDOM_SEED=42  # 随机种子（可选）
```

---

## 输出文件说明

### 文件命名格式

```
independent_test_{MODEL}_{ATTACK_TYPE}_{DISTRIBUTION}_{timestamp}.json
```

### 示例文件名

```
independent_test_resnet_label_flipping_iid_20240101_120000.json
independent_test_lenet5_noise_injection_noniid_case2_beta0.5_20240101_130000.json
```

### 文件内容

- `config`: 实验配置参数
- `rounds`: 每轮训练详情
  - `global_accuracy`: 全局模型准确率
  - `global_loss`: 全局模型损失
  - `detection_results`: 每个客户端的检测结果
- `final_statistics`: 最终统计信息
  - `accuracy`: 检测准确率
  - `precision`: 精确率
  - `recall`: 召回率
  - `f1_score`: F1分数

---

## 常见问题速查

### Q: 如何快速测试系统是否正常工作？

```bash
# 最小配置：MNIST + 无攻击 + 10轮
MODEL=lenet5 DATASET=mnist ATTACK_TYPE=no_attack EPOCHS=10 ./run_independent_detector_test.sh
```

### Q: 如何运行可复现的实验？

```bash
# 使用固定随机种子
RANDOM_SEED=42 ./run_independent_detector_test.sh
```

### Q: 如何对比 FedAvg 和 FedProx？

```bash
# FedAvg
USE_FEDPROX=0 DATA_DISTRIBUTION=noniid ./run_independent_detector_test.sh

# FedProx
USE_FEDPROX=1 DATA_DISTRIBUTION=noniid ./run_independent_detector_test.sh
```

### Q: 如何对比防御和观察模式？

```bash
# 防御模式（检测器控制聚合）
ENABLE_DEFENSE=1 ATTACK_TYPE=label_flipping ./run_independent_detector_test.sh

# 观察模式（检测器仅记录）
ENABLE_DEFENSE=0 ATTACK_TYPE=label_flipping ./run_independent_detector_test.sh
```

### Q: 如何运行多次实验？

```bash
# 运行5次实验（自动生成不同时间戳）
for i in {1..5}; do
  echo "实验 $i/5"
  ./run_independent_detector_test.sh
  sleep 2  # 确保时间戳不同
done
```

---

## 调试技巧

### 1. 查看环境变量

```bash
# 启用调试输出
DEBUG_ENV=1 ./run_independent_detector_test.sh
```

### 2. 检查输出文件

```bash
# 查看最新的结果文件
ls -lt independent_test_*.json | head -1

# 查看JSON文件内容（格式化）
cat independent_test_*.json | python -m json.tool | less
```

### 3. 监控训练进度

训练过程中会输出：
- 每轮的全局模型准确率
- 每个客户端的检测结果
- 聚合决策（接受/拒绝）

### 4. 常见错误排查

```bash
# 检查Python环境
python3 --version

# 检查依赖
pip list | grep torch

# 检查CUDA
nvidia-smi
```

---

## 性能优化建议

### 1. 快速实验（用于调试）

```bash
# 减少训练轮次和客户端数
EPOCHS=10 MODEL=lenet5 DATASET=mnist ./run_independent_detector_test.sh
```

### 2. 完整实验（用于论文）

```bash
# 使用完整配置
EPOCHS=50 MODEL=resnet DATASET=cifar10 \
ATTACK_TYPE=label_flipping DATA_DISTRIBUTION=noniid \
NONIID_CASE=2 ./run_independent_detector_test.sh
```

### 3. 批量实验（用于对比）

```bash
# 创建实验脚本
cat > run_experiments.sh << 'EOF'
#!/bin/bash
# 实验1: IID
DATA_DISTRIBUTION=iid ./run_independent_detector_test.sh
sleep 2

# 实验2: Non-IID轻度
DATA_DISTRIBUTION=noniid NONIID_CASE=1 ./run_independent_detector_test.sh
sleep 2

# 实验3: Non-IID中度
DATA_DISTRIBUTION=noniid NONIID_CASE=2 ./run_independent_detector_test.sh
sleep 2

# 实验4: Non-IID重度
DATA_DISTRIBUTION=noniid NONIID_CASE=3 ./run_independent_detector_test.sh
EOF

chmod +x run_experiments.sh
./run_experiments.sh
```

---

## 相关文档

- [README.md](README.md) - 项目概述
- [WORKFLOW.md](WORKFLOW.md) - 完整流程文档
- [INSTALL.md](INSTALL.md) - 安装指南
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - 项目结构


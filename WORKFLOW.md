# SGX-FL 项目完整流程文档

## 目录
1. [项目概述](#项目概述)
2. [系统架构](#系统架构)
3. [数据流程](#数据流程)
4. [训练流程](#训练流程)
5. [检测流程](#检测流程)
6. [聚合流程](#聚合流程)
7. [输出格式](#输出格式)
8. [关键参数说明](#关键参数说明)

---

## 项目概述

SGX-FL 是一个带恶意客户端检测的联邦学习框架，采用双模型架构：
- **外部模型（External Model）**：使用完整数据训练，可能被污染
- **TEE模型（TEE Model）**：使用采样的干净数据在可信执行环境中训练

通过比较两个模型的更新方向，检测恶意客户端。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     联邦学习服务器                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           全局模型 (Global Model)                    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
   ┌────────┐         ┌────────┐         ┌────────┐
   │客户端1 │         │客户端2 │   ...   │客户端N │
   └────────┘         └────────┘         └────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │        每个客户端本地训练流程          │
        │  ┌──────────────┐  ┌──────────────┐  │
        │  │ 外部模型训练  │  │  TEE模型训练  │  │
        │  │ (完整数据)   │  │ (采样数据)    │  │
        │  └──────────────┘  └──────────────┘  │
        │         │                   │         │
        │         └─────────┬─────────┘         │
        │                   ▼                   │
        │          ┌──────────────────┐         │
        │          │  方向相似度检测   │         │
        │          └──────────────────┘         │
        └───────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  聚合决策      │
                    │ (接受/拒绝)    │
                    └───────────────┘
```

---

## 数据流程

### 1. 数据加载与划分

```
数据集 (CIFAR-10/MNIST/Fashion-MNIST)
    │
    ▼
┌─────────────────────────────────────┐
│  get_dataset()                      │
│  - 加载数据集                       │
│  - 根据分布类型划分客户端数据        │
│    • IID: 随机均匀分配              │
│    • Non-IID: Dirichlet分布分配      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  dict_users[client_id]              │
│  每个客户端的数据索引列表            │
└─────────────────────────────────────┘
```

### 2. 客户端数据采样

每个客户端在训练时：

```
完整数据集 (dict_users[client_id])
    │
    ├─────────────────────────────────┐
    │                                 │
    ▼                                 ▼
┌──────────────┐              ┌──────────────┐
│ 外部数据      │              │  TEE数据      │
│ (100%数据)    │              │ (采样比例)     │
│              │              │              │
│ 用于训练      │              │ 用于训练      │
│ 外部模型      │              │ TEE模型       │
│              │              │              │
│ 可能被污染    │              │ 保证干净      │
│ (攻击场景)    │              │ (TEE保护)     │
└──────────────┘              └──────────────┘
```

**采样比例**：
- 默认：`tee_sample_ratio = 0.1` (10%的数据用于TEE训练)
- TEE训练轮次：`tee_local_ep = int(args.local_ep / tee_sample_ratio * 0.6)`
  - 例如：`local_ep=20` → `tee_local_ep = int(20/0.1*0.6) = 120` epochs

---

## 训练流程

### 完整训练循环

```
Round 0 (Warm-up)
    │
    ├─> 选择 10 个良性客户端
    │
    ├─> 每个客户端：
    │   ├─> 外部模型训练 (20 epochs)
    │   ├─> TEE模型训练 (120 epochs)
    │   └─> 跳过检测 → 直接聚合
    │
    └─> 聚合 → 更新全局模型
    │
Round 1-2 (Warm-up)
    │ (同上)
    │
Round 3+ (正常训练)
    │
    ├─> 选择 20 个客户端 (10良性 + 10恶意)
    │
    ├─> 每个客户端：
    │   ├─> 外部模型训练 (20 epochs)
    │   │   └─> 如果是恶意客户端，数据被污染
    │   │
    │   ├─> TEE模型训练 (120 epochs)
    │   │   └─> 使用干净数据，不受攻击影响
    │   │
    │   └─> 运行检测器：
    │       ├─> 计算更新方向相似度
    │       ├─> 与阈值比较
    │       └─> 决定是否聚合
    │
    └─> 聚合被接受的模型 → 更新全局模型
```

### 客户端本地训练细节

#### 外部模型训练 (`train_external`)

```python
# 1. 初始化优化器
if optimizer == 'sgd':
    optimizer = SGD(params, lr=lr * (lr_decay ** round), momentum=momentum)
elif optimizer == 'adam':
    optimizer = Adam(params, lr=lr)
elif optimizer == 'adaBelief':
    optimizer = AdaBelief(params, lr=lr)

# 2. 训练循环
for epoch in range(local_ep):  # 默认 20 epochs
    for batch in external_data:
        # 2.1 如果是恶意客户端，数据被污染
        if is_malicious:
            images, labels = attack_manager.poison_data(images, labels)
        
        # 2.2 前向传播
        log_probs = external_model(images)['output']
        loss = criterion(log_probs, labels)
        
        # 2.3 FedProx: 添加proximal term
        if use_fedprox:
            loss += (prox_alpha / 2) * ||w - w_global||²
        
        # 2.4 反向传播和优化
        loss.backward()
        optimizer.step()
```

#### TEE模型训练 (`train_tee_secure`)

```python
# 1. 初始化优化器（同上）

# 2. 训练循环
for epoch in range(tee_local_ep):  # 默认 120 epochs
    for batch in clean_data:  # 采样数据，保证干净
        # 2.1 前向传播（数据不会被污染）
        log_probs = tee_model(images)['output']
        loss = criterion(log_probs, labels)
        
        # 2.2 FedProx: 添加proximal term
        if use_fedprox:
            loss += (prox_alpha / 2) * ||w - w_global||²
        
        # 2.3 反向传播和优化
        loss.backward()
        optimizer.step()
```

---

## 检测流程

### 检测器：方向相似度 (Update Direction Similarity)

```python
# 1. 计算更新方向
external_update = external_model - global_model
tee_update = tee_model - global_model

# 2. 展平参数向量
external_vec = flatten(external_update)
tee_vec = flatten(tee_update)

# 3. 计算余弦相似度
similarity = cosine_similarity(external_vec, tee_vec)

# 4. 判断
if similarity < threshold:
    detected_as_malicious = True
else:
    detected_as_malicious = False
```

### 检测阈值

| 攻击类型 | 阈值 | 说明 |
|---------|------|------|
| Label Flipping | 0.1 | 标签翻转导致更新方向差异大 |
| Noise Injection | 0.24 | 噪声注入导致更新方向差异中等 |
| No Attack | 0.1 | 参考阈值（仅用于分析） |

### 检测决策流程

```
客户端训练完成
    │
    ├─> 外部模型更新: Δw_external
    │
    ├─> TEE模型更新: Δw_tee
    │
    └─> 计算方向相似度: sim = cos(Δw_external, Δw_tee)
        │
        ├─> sim >= threshold
        │   └─> 判定为良性 → 聚合
        │
        └─> sim < threshold
            └─> 判定为恶意 → 拒绝聚合
```

### Warm-up 机制

- **前 3 轮（Round 0-2）**：跳过检测，所有客户端都参与聚合
- **原因**：
  - 冷启动期模型不稳定
  - 需要建立基线
  - 避免误检

---

## 聚合流程

### FedAvg 聚合（`USE_FEDPROX=0`）

```python
# 简单平均
w_global_new = (1 / num_accepted) * Σ(w_client_i)
```

### FedProx 聚合（`USE_FEDPROX=1`，默认）

```python
# 加权平均（基于客户端数据量）
weights = [len(client_data_i) for i in accepted_clients]
total_weight = sum(weights)

w_global_new = (1 / total_weight) * Σ(w_client_i * weight_i)
```

### 聚合决策

```python
for client in selected_clients:
    # 1. 训练
    external_model, tee_model = train_models(client)
    
    # 2. 检测（warm-up期跳过）
    if round >= warmup_rounds:
        detection_result = detector.test_update_direction_only(...)
        
        if detection_result['is_malicious']:
            # 拒绝聚合
            continue
        else:
            # 接受聚合
            accepted_clients.append(client)
            accepted_models.append(external_model)
    else:
        # Warm-up期：直接接受
        accepted_clients.append(client)
        accepted_models.append(external_model)

# 3. 聚合
if len(accepted_models) > 0:
    global_model = aggregate(accepted_models)
```

---

## 输出格式

### JSON 输出文件

文件名格式：`independent_test_{MODEL}_{ATTACK_TYPE}_{DISTRIBUTION}_{timestamp}.json`

```json
{
  "config": {
    "model": "resnet",
    "dataset": "cifar10",
    "attack_scenario": "label_flipping",
    "data_distribution": "iid",
    "num_users": 100,
    "epochs": 50,
    "local_ep": 20,
    "lr": 0.01,
    ...
  },
  "rounds": [
    {
      "round": 0,
      "global_accuracy": 10.0,
      "global_loss": 2.3,
      "detection_results": [
        {
          "client_id": 0,
          "is_malicious": false,
          "detected_as_malicious": false,
          "warmup_period": true,
          "detection_skipped": true,
          "detectors": {}
        },
        ...
      ]
    },
    {
      "round": 3,
      "global_accuracy": 25.5,
      "global_loss": 1.8,
      "detection_results": [
        {
          "client_id": 0,
          "is_malicious": false,
          "detected_as_malicious": false,
          "warmup_period": false,
          "detection_skipped": false,
          "detectors": {
            "update_direction": {
              "detection_result": {
                "is_malicious": false,
                "features": {
                  "update_direction_similarity": 0.85
                }
              }
            }
          }
        },
        {
          "client_id": 10,
          "is_malicious": true,
          "detected_as_malicious": true,
          "warmup_period": false,
          "detection_skipped": false,
          "detectors": {
            "update_direction": {
              "detection_result": {
                "is_malicious": true,
                "features": {
                  "update_direction_similarity": 0.05
                }
              }
            }
          }
        },
        ...
      ]
    },
    ...
  ],
  "final_statistics": {
    "total_rounds": 50,
    "warmup_rounds": 3,
    "detection_rounds": 47,
    "total_clients": 1000,
    "true_benign": 850,
    "true_malicious": 150,
    "detected_benign": 840,
    "detected_malicious": 160,
    "accuracy": 0.95,
    "precision": 0.90,
    "recall": 0.96,
    "f1_score": 0.93,
    ...
  }
}
```

---

## 关键参数说明

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `local_ep` | 20 | 外部模型本地训练轮次 |
| `tee_local_ep` | 120 | TEE模型本地训练轮次（自动计算） |
| `lr` | 0.01 | 学习率 |
| `lr_decay` | 0.998 | 学习率衰减率 |
| `local_bs` | 32 | 批次大小（所有数据集统一使用默认值32） |
| `optimizer` | sgd | 优化器（sgd/adam/adaBelief） |

### 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_users` | 100 | 客户端总数 |
| `frac` | 0.2 | 每轮选择的客户端比例 |
| `tee_sample_ratio` | 0.1 | TEE数据采样比例 |
| `data_beta` | 0.5 | Dirichlet分布参数α（Non-IID） |

### 检测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `warmup_rounds` | 3 | Warm-up轮数 |
| `label_direction_threshold` | 0.1 | 标签翻转检测阈值 |
| `noise_direction_threshold` | 0.24 | 噪声注入检测阈值 |

### 聚合参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_fedprox` | 1 | 是否使用FedProx |
| `prox_alpha` | 0.1 | FedProx正则化强度μ |

---

## 完整执行流程示例

### 1. 启动脚本

```bash
# 默认配置（ResNet18 + CIFAR-10 + 标签翻转攻击）
./run_independent_detector_test.sh

# 自定义配置
MODEL=lenet5 DATASET=mnist ATTACK_TYPE=noise_injection \
DATA_DISTRIBUTION=noniid NONIID_CASE=2 \
./run_independent_detector_test.sh
```

### 2. 脚本执行流程

```
run_independent_detector_test.sh
    │
    ├─> 解析环境变量（MODEL, DATASET, ATTACK_TYPE, ...）
    │
    ├─> 根据数据集自动调整超参数
    │
    ├─> Non-IID参数映射（NONIID_CASE → ACTUAL_CASE, DATA_BETA, PROX_ALPHA）
    │
    └─> 调用 test_independent_detectors_training.py
```

### 3. Python 训练流程

```
test_independent_detectors_training.py
    │
    ├─> 解析命令行参数
    │
    ├─> 加载数据集 (get_dataset)
    │
    ├─> 初始化全局模型
    │
    ├─> 初始化攻击管理器 (AttackManager)
    │
    ├─> 初始化检测器 (IndependentDetectorsTester)
    │
    └─> 训练循环 (for round in range(epochs)):
        │
        ├─> 选择客户端（根据攻击类型和轮次）
        │
        ├─> 对每个客户端：
        │   ├─> 创建 LocalUpdate_XFL_SmallData
        │   ├─> 外部模型训练 (train_external)
        │   ├─> TEE模型训练 (train_tee_secure)
        │   └─> 运行检测器 (test_update_direction_only)
        │
        ├─> 聚合被接受的模型 (Aggregation)
        │
        ├─> 测试全局模型准确率
        │
        └─> 保存检测结果
    │
    └─> 计算最终统计 → 保存 JSON 文件
```

### 4. 每个客户端训练流程

```
LocalUpdate_XFL_SmallData
    │
    ├─> 初始化：
    │   ├─> 外部数据加载器 (100% 数据)
    │   └─> TEE数据加载器 (采样数据)
    │
    ├─> train_external():
    │   ├─> 初始化优化器
    │   ├─> for epoch in range(local_ep):
    │   │   ├─> for batch in external_data:
    │   │   │   ├─> 如果是恶意客户端 → 污染数据
    │   │   │   ├─> 前向传播
    │   │   │   ├─> 计算损失（+ FedProx项）
    │   │   │   ├─> 反向传播
    │   │   │   └─> 优化器更新
    │   │   └─> 清理GPU缓存
    │   └─> 返回模型状态和损失
    │
    └─> train_tee_secure():
        ├─> 初始化优化器
        ├─> for epoch in range(tee_local_ep):
        │   ├─> for batch in clean_data:
        │   │   ├─> 前向传播（数据不会被污染）
        │   │   ├─> 计算损失（+ FedProx项）
        │   │   ├─> 反向传播
        │   │   └─> 优化器更新
        │   └─> 清理GPU缓存
        └─> 返回模型状态和损失
```

---

## 关键设计决策

### 1. 双模型架构

- **外部模型**：使用完整数据，可能被攻击，用于实际推理
- **TEE模型**：使用采样数据，保证干净，用于检测恶意行为

### 2. TEE训练轮次调整

- TEE数据量少（10%），但训练轮次多（120 epochs）
- 保证TEE模型在少量数据上也能充分学习
- 公式：`tee_local_ep = int(local_ep / sample_ratio * 0.6)`

### 3. Warm-up 机制

- 前3轮跳过检测，建立基线
- 避免冷启动期误检
- 确保模型稳定后再启用检测

### 4. FedProx 聚合

- 默认使用FedProx，适合Non-IID场景
- 通过proximal term约束客户端更新，防止过度偏离
- 正则化强度根据数据异构程度自动调整

### 5. 方向相似度检测

- 简单有效，无需额外训练
- 基于余弦相似度，对参数规模不敏感
- 恶意客户端更新方向与TEE模型差异大

---

## 常见问题

### Q1: 为什么第一轮准确率只有10%？

**A**: 这是正常的，因为：
- 模型初始化为随机权重
- CIFAR-10有10个类别，随机猜测准确率就是10%
- 需要多轮训练才能提升准确率

### Q2: TEE训练轮次为什么这么多？

**A**: TEE数据量只有10%，为了在少量数据上充分学习，需要更多轮次：
- 外部模型：20 epochs × 100% 数据
- TEE模型：120 epochs × 10% 数据
- 总训练量相近，保证两个模型都能充分学习

### Q3: 检测阈值如何选择？

**A**: 阈值根据攻击类型和实验优化：
- 标签翻转：阈值0.1（攻击性强，差异明显）
- 噪声注入：阈值0.24（攻击性较弱，差异较小）
- 可通过实验调整，平衡准确率和召回率

### Q4: 为什么使用FedProx而不是FedAvg？

**A**: FedProx更适合Non-IID场景：
- 添加proximal term约束，防止客户端过度偏离全局模型
- 提高收敛稳定性
- 可通过`USE_FEDPROX=0`切换到FedAvg

---

## 扩展阅读

- [README.md](README.md) - 项目概述和快速开始
- [INSTALL.md](INSTALL.md) - 环境安装指南
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - 项目结构说明


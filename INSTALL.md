# SGX-FL 环境安装指南

## 问题说明

如果遇到以下错误：
```
RuntimeError: operator torchvision::nms does not exist
```

这通常是因为 `torch` 和 `torchvision` 版本不匹配或安装方式不一致导致的。

## 推荐解决方案

### 方案 1: 使用 conda 环境（推荐）

项目已经在 `flexfl` conda 环境中测试通过。如果您的系统中已有该环境，直接激活即可：

```bash
conda activate flexfl
```

### 方案 2: 创建新的 conda 环境

```bash
# 创建 Python 3.8 环境
conda create -n sgx-fl python=3.8 -y
conda activate sgx-fl

# 安装依赖
pip install -r requirements.txt
```

### 方案 3: 使用 pip 安装（确保版本匹配）

**重要**：PyTorch 和 torchvision 必须从同一个源安装，且版本必须匹配。

#### 对于 CUDA 12.1（推荐）

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

#### 对于 CUDA 11.8

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

#### 对于 CPU 版本

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 验证安装

安装完成后，验证环境是否正确：

```bash
python -c "import torch; import torchvision; print(f'torch: {torch.__version__}'); print(f'torchvision: {torchvision.__version__}'); print('✅ 导入成功')"
```

应该输出类似：
```
torch: 2.4.1
torchvision: 0.19.1
✅ 导入成功
```

## 依赖说明

### 核心依赖

- **PyTorch 2.4.1**: 深度学习框架
- **torchvision 0.19.1**: 计算机视觉工具和数据集
- **numpy 1.24.4**: 数值计算
- **scikit-learn 1.3.2**: 机器学习工具

### 可选依赖

- **wandb 0.21.1**: 实验跟踪（如果不需要可以注释掉）
- **transformers 4.46.3**: Hugging Face 模型（仅用于文本数据集）
- **datasets 3.1.0**: Hugging Face 数据集（仅用于文本数据集）

## 故障排除

### 问题 1: torchvision 导入失败

**症状**：`RuntimeError: operator torchvision::nms does not exist`

**解决方案**：
1. 卸载 torch 和 torchvision：`pip uninstall torch torchvision`
2. 从 PyTorch 官方源重新安装（见方案 3）

### 问题 2: CUDA 版本不匹配

如果您的系统 CUDA 版本与安装的不匹配，请：
1. 检查 CUDA 版本：`nvidia-smi` 或 `nvcc --version`
2. 根据 CUDA 版本选择对应的安装命令（见方案 3）

### 问题 3: Python 版本不兼容

项目要求 Python 3.8。如果使用其他版本，可能需要调整依赖版本。

## 快速开始

```bash
# 1. 激活环境（如果使用 flexfl）
conda activate flexfl

# 2. 运行测试
./run_independent_detector_test.sh
```


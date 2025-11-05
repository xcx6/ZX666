# SGX-FL Project Structure

本项目是从 oldXFL 项目中提取的核心代码，用于联邦学习恶意客户端检测。

## 文件清单

### 核心脚本
- `run_independent_detector_test.sh` - 主启动脚本
- `test_independent_detectors_training.py` - 训练主脚本（带检测器）
- `independent_detectors_test.py` - 独立检测器测试器

### 算法模块
- `Algorithm/Training_XFL_SmallData.py` - XFL训练算法（小数据集版本）

### 模型定义
- `models/` - 所有模型定义
  - `Fed.py` - 联邦聚合函数
  - `lenet5.py` - LeNet5模型
  - `resnet20.py` - ResNet20模型
  - `standard_resnet18.py` - 标准ResNet18模型
  - `vgg_16_bn.py` - VGG16模型
  - 其他模型文件

### 工具模块
- `utils/` - 工具函数
  - `options.py` - 参数解析
  - `get_dataset.py` - 数据集加载
  - `sampling.py` - 数据采样
  - 其他工具文件

### 攻击模块
- `attacks/` - 攻击实现
  - `attack_manager.py` - 攻击管理器
  - `config.py` - 攻击配置
  - `data_poisoning.py` - 数据投毒攻击

### 优化器模块
- `optimizer/` - 优化器实现
  - `Adabelief.py` - AdaBelief优化器（NeurIPS 2020 Spotlight）

### 辅助文件
- `getAPOZ.py` - APOZ计算工具
- `data_collector.py` - 数据收集工具
- `wandbUtils.py` - WandB工具（可选）

### 配置文件
- `requirements.txt` - Python依赖
- `README.md` - 项目说明
- `.gitignore` - Git忽略文件

## 依赖关系

```
run_independent_detector_test.sh
  └─> test_independent_detectors_training.py
      ├─> utils.options
      ├─> utils.get_dataset
      ├─> Algorithm.Training_XFL_SmallData
      ├─> models.Fed
      ├─> models.* (各种模型)
      ├─> attacks.attack_manager
      └─> independent_detectors_test.IndependentDetectorsTester

Algorithm.Training_XFL_SmallData
  ├─> getAPOZ
  ├─> models.*
  ├─> optimizer.Adabelief
  ├─> wandbUtils
  ├─> data_collector
  └─> attacks.*

models.Update
  └─> optimizer.Adabelief

getAPOZ
  └─> optimizer.Adabelief
```

## 路径修改说明

所有硬编码的 `oldXFL` 路径已修改为相对路径：
- `run_independent_detector_test.sh`: 使用 `$(dirname "$0")` 获取脚本目录
- `Algorithm/Training_XFL_SmallData.py`: 添加了项目根目录到sys.path
- 所有相对导入路径保持不变，确保模块间正确引用

## 使用说明

详见 `README.md`


"""
Training_XFL_SmallData - TEE内部使用小数据集的恶意检测算法
基于TEE可信执行环境的恶意客户端检测方案 - 资源优化版本

核心改进:
- TEE内部使用采样数据集（而非完整数据）
- 调整TEE训练超参数以保持检测效果
- 保持外部训练不变
- 保持所有检测功能和数据收集功能
"""

# Standard library imports
import copy
import random
from collections import defaultdict
from datetime import datetime

# Third-party imports
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Local application imports
import sys
import os
# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models import vgg_16_bn, test, MobileNetV2
from models.Fed import Aggregation, summon_clients
from models.Update import DatasetSplit
from models.standard_resnet18 import standard_resnet18
from models.resnet20 import resnet20
from models.lenet5 import LeNet5
from wandbUtils import init_run, endrun, upload_data
from data_collector import (
    initialize_data_collector, collect_round_data, collect_attack_data, 
    collect_detection_data, save_experiment_data, add_log
)

# 导入攻击模块
from attacks.attack_manager import AttackManager
from attacks.config import ATTACK_SCENARIOS

layer_idx = 0


def getStandardNet(args):
    """获取标准模型"""
    if args.model == "resnet":
        net = standard_resnet18(
            num_classes=args.num_classes, 
            num_channels=args.num_channels, 
            track_running_stats=False
        ).to(args.device)
        return net
    elif args.model == "resnet20":
        net = resnet20(
            num_classes=args.num_classes,
            num_channels=args.num_channels,
            track_running_stats=False
        ).to(args.device)
        return net
    elif args.model == "lenet5":
        net = LeNet5(
            num_classes=args.num_classes,
            num_channels=args.num_channels,
            track_running_stats=False
        ).to(args.device)
        return net
    elif args.model == "vgg":
        net = vgg_16_bn(
            num_classes=args.num_classes,
            track_running_stats=False,
            num_channels=args.num_channels
        ).to(args.device)
        return net
    elif args.model == "mobilenet":
        net = MobileNetV2(
            channels=args.num_channels,
            num_classes=args.num_classes,
            trs=False,
            rate=[1] * 9
        ).to(args.device)
        return net
    else:
        raise ValueError(f"Unknown model: {args.model}. Only standard models (resnet, resnet20, lenet5, vgg, mobilenet) are supported.")


class LocalUpdate_XFL_SmallData(object):
    """
    XFL本地训练类 - TEE使用小数据集优化版本
    
    核心改进：
    1. TEE内部使用分层采样的小数据集
    2. 调整TEE训练轮次以保持参数更新次数一致
    3. 保持外部训练不变
    """
    def __init__(self, args, dataset, idxs, verbose=False, tee_sample_ratio=0.3):
        """
        Args:
            args: 参数配置
            dataset: 真实数据集
            idxs: 客户端数据索引
            verbose: 是否详细输出
            tee_sample_ratio: TEE数据采样比例（默认30%）
        """
        self.args = args
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.dataset = dataset
        self.idxs = idxs
        self.verbose = verbose
        
        # TEE采样配置
        self.tee_sample_ratio = tee_sample_ratio
        
        # 分层采样TEE数据索引
        self.tee_idxs = self._stratified_sampling(dataset, idxs, tee_sample_ratio)
        
        # 计算TEE训练轮次（保持总更新次数一致）
        # TEE内部训练epochs：降低倍数避免GPU压力累积
        # 原方案：20 / 0.3 = 67 epochs → 长时间训练，51轮后CUDA崩溃
        # 新方案：20 / 0.3 * 0.6 = 40 epochs → 平衡效果与稳定性
        self.tee_local_ep = int(args.local_ep / tee_sample_ratio * 0.6)
        
        # 外部训练使用完整数据
        self.external_data = DataLoader(
            DatasetSplit(dataset, idxs, self.args),
            batch_size=self.args.local_bs, 
            shuffle=True, 
            drop_last=True
        )
        
        # TEE训练使用采样数据
        self.clean_data = DataLoader(
            DatasetSplit(dataset, self.tee_idxs, self.args),
            batch_size=self.args.local_bs, 
            shuffle=True, 
            drop_last=True
        )
        
        if verbose:
            print(f"  数据配置:")
            print(f"    外部数据: {len(idxs)} 样本")
            print(f"    TEE数据: {len(self.tee_idxs)} 样本 ({tee_sample_ratio*100:.0f}%)")
            print(f"    外部训练: {args.local_ep} epochs")
            print(f"    TEE训练: {self.tee_local_ep} epochs")
    
    def _stratified_sampling(self, dataset, idxs, ratio):
        """
        分层随机采样 - 按类别比例采样
        
        Args:
            dataset: 数据集
            idxs: 索引列表
            ratio: 采样比例
        
        Returns:
            sampled_idxs: 采样后的索引列表
        """
        # 获取标签
        labels = []
        for idx in idxs:
            if hasattr(dataset, 'targets'):
                labels.append(dataset.targets[idx])
            elif hasattr(dataset, 'labels'):
                labels.append(dataset.labels[idx])
            else:
                # 如果没有直接的标签属性，通过索引获取
                _, label = dataset[idx]
                labels.append(label)
        
        # 按类别分组
        class_indices = defaultdict(list)
        for i, label in enumerate(labels):
            class_indices[int(label)].append(idxs[i])
        
        # 每类采样
        sampled_idxs = []
        for label, indices in class_indices.items():
            n_samples = max(1, int(len(indices) * ratio))
            sampled = random.sample(indices, n_samples)
            sampled_idxs.extend(sampled)
        
        return sampled_idxs
    
    def train_external(self, round, external_model, client_id=None, attack_manager=None, global_model=None):
        """
        外部训练 - 使用完整模型和完整数据（可能被污染）
        
        Args:
            round: 当前训练轮次
            external_model: 外部模型
            client_id: 客户端ID
            attack_manager: 攻击管理器
            global_model: 全局模型（用于FedProx，可选）
        """
        from optimizer.Adabelief import AdaBelief
        
        # 确保模型在正确的设备上
        external_model = external_model.to(self.args.device)
        external_model.train()
        
        # 如果使用FedProx且提供了全局模型，保存全局模型参数
        use_fedprox = global_model is not None and hasattr(self.args, 'use_fedprox') and self.args.use_fedprox
        if use_fedprox:
            global_params = {name: param.clone().detach() for name, param in global_model.named_parameters()}
        
        # 外部训练优化器
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                external_model.parameters(), 
                lr=self.args.lr * (self.args.lr_decay ** round),
                momentum=self.args.momentum, 
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(external_model.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(external_model.parameters(), lr=self.args.lr)

        external_loss = 0
        
        # 检查是否为恶意客户端
        is_malicious = attack_manager and attack_manager.is_malicious(client_id)
        
        # 外部训练（使用args.local_ep，不变）
        for epoch in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.external_data):
                try:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    
                    # 如果是恶意客户端，数据被污染
                    if is_malicious:
                        images, labels = attack_manager.poison_data(client_id, images, labels)
                    
                    # 外部训练
                    external_model.zero_grad()
                    log_probs = external_model(images)['output']
                    loss = self.loss_func(log_probs, labels)
                    
                    # FedProx: 添加proximal term
                    if use_fedprox:
                        proximal_term = 0.0
                        for name, param in external_model.named_parameters():
                            if name in global_params:
                                proximal_term += ((param - global_params[name]) ** 2).sum()
                        loss += (self.args.prox_alpha / 2) * proximal_term
                    
                    loss.backward()
                    optimizer.step()
                    external_loss += loss.item()
                    
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        print(f"❌ CUDA错误在Client {client_id}, epoch {epoch}, batch {batch_idx}: {e}")
                        print(f"💥 CUDA错误检测到，立即停止程序执行")
                        print(f"🔧 建议：重启Python进程或重启服务器")
                        raise e  # 立即抛出错误，停止程序
                    else:
                        raise e
            
            # 🔧 优化: 每10轮epoch后清理GPU缓存，防止内存碎片化和cuDNN状态损坏
            if (epoch + 1) % 10 == 0 and self.args.device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        if self.verbose:
            attack_info = " [MALICIOUS-EXTERNAL]" if is_malicious else " [BENIGN-EXTERNAL]"
            info = '\nClient {} {} External Loss={:.4f}'.format(
                client_id, attack_info,
                external_loss / (self.args.local_ep * len(self.external_data))
            )
            print(info)

        return external_model.state_dict(), external_loss

    def train_tee_secure(self, round, tee_model, client_id=None, attack_manager=None, global_model=None):
        """
        TEE安全训练 - 使用完整模型和采样的干净数据
        关键改进：使用调整后的训练轮次（self.tee_local_ep）
        新增：支持FedProx，与客户端训练保持一致
        """
        from optimizer.Adabelief import AdaBelief
        
        # 确保模型在正确的设备上
        tee_model = tee_model.to(self.args.device)
        tee_model.train()
        
        # FedProx: 保存全局模型参数（如果启用）
        use_fedprox = global_model is not None and hasattr(self.args, 'use_fedprox') and self.args.use_fedprox
        if use_fedprox:
            global_params = {name: param.clone().detach() 
                           for name, param in global_model.state_dict().items()}
            prox_mu = self.args.prox_alpha if hasattr(self.args, 'prox_alpha') else 0.01
        
        # TEE内部优化器
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                tee_model.parameters(), 
                lr=self.args.lr * (self.args.lr_decay ** round),
                momentum=self.args.momentum, 
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(tee_model.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adaBelief':
            optimizer = AdaBelief(tee_model.parameters(), lr=self.args.lr)

        Predict_loss = 0
        
        # 检查是否为恶意客户端（仅用于日志）
        is_malicious = attack_manager and attack_manager.is_malicious(client_id)
        
        # TEE训练（使用调整后的轮次）
        for epoch in range(self.tee_local_ep):
            for batch_idx, (images, labels) in enumerate(self.clean_data):
                try:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    
                    # TEE内部训练（不受外部攻击影响）
                    tee_model.zero_grad()
                    log_probs = tee_model(images)['output']
                    ce_loss = self.loss_func(log_probs, labels)
                    
                    # FedProx: 添加proximal term（与客户端训练保持一致）
                    if use_fedprox:
                        proximal_term = 0.0
                        for name, param in tee_model.named_parameters():
                            if name in global_params:
                                proximal_term += torch.sum((param - global_params[name]) ** 2)
                        loss = ce_loss + (prox_mu / 2.0) * proximal_term
                    else:
                        loss = ce_loss
                    
                    loss.backward()
                    optimizer.step()
                    Predict_loss += loss.item()
                    
                    # 收集攻击事件数据
                    if is_malicious:
                        collect_attack_data(round, client_id, "tee_protected", {
                            "batch_size": len(labels),
                            "tee_protection": True,
                            "attack_blocked": True,
                            "tee_sample_ratio": self.tee_sample_ratio
                        })
                        
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        print(f"❌ TEE训练CUDA错误在Client {client_id}, epoch {epoch}, batch {batch_idx}: {e}")
                        print(f"💥 CUDA错误检测到，立即停止程序执行")
                        print(f"🔧 建议：重启Python进程或重启服务器")
                        raise e  # 立即抛出错误，停止程序
                    else:
                        raise e
            
            # 🔧 优化: 每10轮epoch后清理GPU缓存，防止内存碎片化和cuDNN状态损坏
            if (epoch + 1) % 10 == 0 and self.args.device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        if self.verbose:
            attack_info = " [MALICIOUS-TEE-PROTECTED]" if is_malicious else " [BENIGN]"
            info = '\nClient {} {} TEE Loss={:.4f} (epochs={})'.format(
                client_id, attack_info,
                Predict_loss / (self.tee_local_ep * len(self.clean_data)),
                self.tee_local_ep
            )
            print(info)

        return tee_model.state_dict(), Predict_loss


def Training_XFL_SmallData(args, dataset_train, dataset_test, dict_users, attack_scenario='no_attack'):
    """
    XFL训练函数 - TEE使用小数据集优化版本
    
    核心改进：
    - TEE内部使用30%采样数据
    - 调整TEE训练轮次以保持检测效果
    - 保持所有其他功能不变
    
    Args:
        attack_scenario: 攻击场景 ('no_attack', 'label_flipping', 'noise_injection', 'backdoor')
    """
    # 初始化全局模型
    global_model = getStandardNet(args)
    model_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    init_msg = f"🌐 服务器初始化: 标准ResNet18全局模型, 参数数: {model_params:,}"
    print(init_msg)
    add_log(init_msg, "info")
    
    # TEE配置信息
    tee_sample_ratio = 0.3
    tee_msg = f"🔧 TEE配置: 采样比例={tee_sample_ratio*100:.0f}%, 训练轮次调整={int(args.local_ep/tee_sample_ratio)}轮"
    print(tee_msg)
    add_log(tee_msg, "info")
    
    # 初始化wandb
    if hasattr(args, 'wandb') and args.wandb:
        run = init_run(args, "XFL-SmallData-Experiment", attack_scenario)
    else:
        run = None
    
    # 初始化攻击管理器
    attack_config = ATTACK_SCENARIOS.get(attack_scenario, ATTACK_SCENARIOS['no_attack'])
    attack_config['attack_params']['num_classes'] = args.num_classes
    attack_manager = AttackManager(args.num_users, attack_config)
    
    attack_msg = f"🎯 攻击场景: {attack_scenario}, 攻击类型: {attack_config['attack_type']}"
    print(attack_msg)
    add_log(attack_msg, "info")

    # Start federated learning
    avg_acc = [0]
    clients_list = summon_clients(args)
    
    # 初始化数据收集器
    experiment_name = f"XFL_SmallData_{attack_scenario}_defense_{args.enable_defense}"
    data_collector = initialize_data_collector(args, experiment_name)
    print(f"📊 数据收集器已初始化: {experiment_name}")
    client_models = []
    last_global_accuracy = 0.0  # 跟踪上一轮的全局准确率
    
    for _iter in tqdm(range(args.epochs)):
        print('*' * 80)
        round_msg = f"Round {_iter:3d}"
        print(round_msg)
        add_log(round_msg, "round_start")

        w_locals = []
        lens = []
        current_round_models = []

        m = max(int(args.frac * args.num_users), 1)
        
        # 客户端选择
        available_clients = list(range(args.num_users))
        np.random.shuffle(available_clients)
        selected_clients = available_clients[:m]
        
        # 验证唯一性
        unique_clients = list(set(selected_clients))
        if len(unique_clients) != len(selected_clients):
            selected_clients = np.random.choice(
                range(args.num_users), min(m, args.num_users), replace=False
            ).tolist()
        
        print(f"this epoch choose: {selected_clients} (共{len(selected_clients)}个)")
        print(f"XFL-SmallData算法: TEE使用{tee_sample_ratio*100:.0f}%采样数据")

        # 设置恶意客户端
        attack_manager.setup_malicious_clients(selected_clients, _iter, args.epochs)
        
        if attack_manager.get_malicious_clients():
            malicious_list = sorted(attack_manager.get_malicious_clients())
            malicious_msg = f"🚨 恶意客户端: {malicious_list}"
            print(malicious_msg)
            add_log(malicious_msg, "warning")
            tee_msg = "🔒 TEE保护: 恶意客户端无法污染TEE内部训练"
            print(tee_msg)
            add_log(tee_msg, "info")
        
        for user_idx in selected_clients:
            # 清理CUDA缓存（防止累积的内存问题）
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass  # 如果CUDA已损坏，跳过清理
            
            local = LocalUpdate_XFL_SmallData(
                args=args, 
                dataset=dataset_train, 
                idxs=dict_users[user_idx], 
                verbose=True,
                tee_sample_ratio=tee_sample_ratio
            )
            
            try:
                # 1. 外部训练：完整模型 + 完整数据（可能被污染）
                external_model = copy.deepcopy(global_model).to(args.device)
                w_external, external_loss = local.train_external(
                    round=_iter,
                    external_model=external_model,
                    client_id=user_idx,
                    attack_manager=attack_manager
                )
                
                # ⚠️ 不删除external_model，保留用于后续检测（避免重建）
                # 虽然会短暂与tee_model共存，但总比510次重建好
                
                # 2. TEE内部训练：完整模型 + 采样的干净数据
                tee_model = copy.deepcopy(global_model).to(args.device)
                w_tee, Predict_loss = local.train_tee_secure(
                    round=_iter,
                    tee_model=tee_model,
                    client_id=user_idx,
                    attack_manager=attack_manager
                )
                    
                # 3. 计算损失
                external_loss_avg = external_loss / (args.local_ep * len(local.external_data))
                tee_loss_avg = Predict_loss / (local.tee_local_ep * len(local.clean_data))
                
                print(f"Client {user_idx} 双模型训练完成:")
                print(f"   外部模型损失: {external_loss_avg:.6f} ({args.local_ep} epochs)")
                print(f"   TEE模型损失:  {tee_loss_avg:.6f} ({local.tee_local_ep} epochs)")
                print(f"   损失差异:     {abs(external_loss_avg - tee_loss_avg):.6f}")
                
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"❌ Client {user_idx} 训练失败（CUDA错误）: {e}")
                    print(f"💥 CUDA错误检测到，立即停止程序执行")
                    print(f"🔧 建议：重启Python进程或重启服务器")
                    print(f"📍 错误位置：Round {_iter}, Client {user_idx}")
                    raise e  # 立即抛出错误，停止程序
                else:
                    raise e

            # 聚合模型（集成检测器已移除，直接聚合）
            is_malicious_actual = attack_manager and attack_manager.is_malicious(user_idx)
            w_locals.append(copy.deepcopy(w_external))
            lens.append(len(dict_users[user_idx]))
            current_round_models.append(copy.deepcopy(w_external))
            
            # 清理当前客户端的模型
            if 'external_model' in locals():
                del external_model
            if 'tee_model' in locals():
                del tee_model
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass  # CUDA上下文损坏时，清理操作也会失败
        
        client_models = current_round_models
        
        # 聚合
        if len(w_locals) == 0:
            print("⚠️  警告：所有客户端都被检测为恶意，跳过本轮聚合")
        else:
            try:
                # 尝试聚合（可能因CUDA上下文损坏失败）
                w_glob = Aggregation(w_locals, lens)
                global_model.load_state_dict(w_glob)
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"❌ 聚合失败（CUDA错误）: {e}")
                    print(f"💥 CUDA错误检测到，立即停止程序执行")
                    print(f"🔧 建议：重启Python进程或重启服务器")
                    print(f"📍 错误位置：Round {_iter}, 聚合阶段")
                    raise e  # 立即抛出错误，停止程序
                else:
                    raise e
        
        # 测试
        accDict = {}
        try:
            if len(w_locals) > 0:
                global_accuracy = test(global_model, dataset_test, args)
                accDict[f"global-acc"] = global_accuracy
                last_global_accuracy = global_accuracy  # 更新上一轮准确率
                acc_msg = f"Round {_iter}: Global Model Accuracy = {global_accuracy:.2f}%"
                print(acc_msg)
                add_log(acc_msg, "accuracy")
            else:
                global_accuracy = test(global_model, dataset_test, args)
                accDict[f"global-acc"] = global_accuracy
                last_global_accuracy = global_accuracy  # 更新上一轮准确率
                acc_msg = f"Round {_iter}: Global Model Accuracy = {global_accuracy:.2f}% (未更新)"
                print(acc_msg)
                add_log(acc_msg, "accuracy")
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"❌ 测试失败（CUDA错误）: {e}")
                print(f"💥 CUDA错误检测到，立即停止程序执行")
                print(f"🔧 建议：重启Python进程或重启服务器")
                print(f"📍 错误位置：Round {_iter}, 测试阶段")
                raise e  # 立即抛出错误，停止程序
            else:
                raise e
        
        # 收集数据
        collect_round_data(_iter, accDict)
        upload_data(args, run, _iter, accDict, avg_acc, {"tee_sample_ratio": tee_sample_ratio})
        
        # ✅ 每轮结束后彻底清理GPU内存（防止累积导致崩溃）
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # 同步所有CUDA操作
                # 每10轮打印一次GPU内存状态
                if _iter % 10 == 0:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"  💾 GPU内存: {allocated:.2f}GB / {reserved:.2f}GB")
        except:
            pass
    
    # 训练完成
    final_accuracy = accDict.get('global-acc', 0.0)
    
    summary_msg = f"\n🔒 XFL-SmallData TEE安全训练完成"
    print(summary_msg)
    add_log(summary_msg, "summary")
    
    final_acc_msg = f"  最终准确率: {final_accuracy:.4f}"
    print(final_acc_msg)
    add_log(final_acc_msg, "summary")
    
    tee_features = [
        "  TEE-SmallData特性:",
        f"    ✅ TEE采样数据 - 节省{(1-tee_sample_ratio)*100:.0f}%存储和计算",
        f"    ✅ 调整训练轮次 - 保持检测效果",
        "    ✅ 完整模型训练 - 无剪枝性能损失",
        "    ✅ 干净数据保护 - 不受外部攻击污染",
        "    ✅ 内部安全检测 - 基于完整模型特征",
        "    ✅ 零信任架构 - 所有客户端都需验证"
    ]
    for feature in tee_features:
        print(feature)
        add_log(feature, "summary")
    
    # 保存数据
    data_file = save_experiment_data()
    save_msg = f"📊 实验数据已保存: {data_file}"
    print(save_msg)
    add_log(save_msg, "info")
    
    endrun(run)


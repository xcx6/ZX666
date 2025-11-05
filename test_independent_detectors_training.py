#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
联邦学习 + 检测器训练脚本（简化版）
完整流程：正常FL训练 → update_direction检测 → 基于检测结果决定是否上传

检测器配置：
  - 仅使用 update_direction 检测器（通用性最强，所有场景有效）
  - 检测为良性 → 上传模型参与聚合
  - 检测为恶意 → 拒绝上传，不参与聚合

输出信息：
  - 每个客户端的检测结果（方向相似度、判断结果、匹配情况）
  - 每轮全局模型准确率和损失
  - 训练结束后的总体统计（准确率、精确率、召回率等）
"""

import sys
import os
import copy
import torch
import numpy as np
import json
from tqdm import tqdm

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.options import args_parser
from utils.get_dataset import get_dataset
from Algorithm.Training_XFL_SmallData import LocalUpdate_XFL_SmallData
from models.Fed import Aggregation
from models import vgg_16_bn
from models.resnet20 import resnet20
from models.lenet5 import LeNet5
from models.standard_resnet18 import standard_resnet18
from attacks.attack_manager import AttackManager
from independent_detectors_test import IndependentDetectorsTester

# ==================== cuDNN错误修复方案 ====================
# 方案1: 完全禁用cuDNN（最激进方案）
torch.backends.cudnn.enabled = False

# 方案2: 禁用cuDNN的benchmark模式（避免kernel自动选择bug）
torch.backends.cudnn.benchmark = False

# 方案3: 启用确定性模式（避免随机kernel选择）
torch.backends.cudnn.deterministic = True

# 方案4: 设置环境变量（精确定位错误）
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 方案5: 设置更严格的内存管理
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# ==========================================================


def load_cluster_mapping(args):
    """加载客户端簇归属映射信息"""
    if not args.use_clustered_data:
        return None, None
    
    # 构建簇映射文件路径
    mapping_file = f"cluster_mappings/{args.dataset}_{args.num_users}_clusters_{args.num_clusters}_mapping.json"
    
    if not os.path.exists(mapping_file):
        print(f"⚠️  簇映射文件未找到: {mapping_file}")
        return None, None
    
    try:
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)
        
        client_cluster_map = mapping_data['client_cluster_map']
        cluster_summary = mapping_data.get('cluster_summary', {})
        
        print(f"✅ 已加载簇映射信息: {mapping_file}")
        print(f"📊 总客户端数: {len(client_cluster_map)}")
        print(f"📊 总簇数: {len(set(client_cluster_map.values()))}")
        
        return client_cluster_map, cluster_summary
    except Exception as e:
        print(f"❌ 加载簇映射文件失败: {e}")
        return None, None

def get_client_cluster_info(client_id, client_cluster_map, cluster_summary):
    """获取客户端簇信息"""
    if not client_cluster_map or client_id not in client_cluster_map:
        return f"客户端{client_id} (无簇信息)"
    
    cluster_id = client_cluster_map[client_id]
    cluster_info = cluster_summary.get(str(cluster_id), {})
    client_count = cluster_info.get('client_count', 0)
    
    return f"客户端{client_id} (簇{cluster_id}, 簇内{client_count}个客户端)"

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
    elif args.model == "vgg":
        net = vgg_16_bn(args).to(args.device)
        return net
    elif args.model == "lenet5":
        net = LeNet5(
            num_classes=args.num_classes,
            num_channels=args.num_channels,
            track_running_stats=False
        ).to(args.device)
        return net
    else:
        raise ValueError(f"Unknown model: {args.model}")


def prepare_tee_validation_set(dataset_train, num_samples=500):
    """
    为TEE准备全局IID验证集（分层采样）
    从训练集中均衡采样，所有客户端共用
    
    重要说明：
    - 使用分层采样（stratified sampling）确保每个类别的样本数量相同
    - 无论客户端数据是IID还是Non-IID，TEE验证集都保持平衡分布
    - 这确保TEE模型在干净的、平衡的数据上训练
    
    Args:
        dataset_train: 训练数据集
        num_samples: 总样本数（默认500，每类50个）
    
    Returns:
        validation_loader: 验证集数据加载器
    """
    import torch
    from torch.utils.data import DataLoader, Subset
    import random
    
    # CIFAR-10有10个类，每类采样50个
    samples_per_class = num_samples // 10
    
    # 收集每个类的索引
    class_indices = {i: [] for i in range(10)}
    for idx in range(len(dataset_train)):
        _, label = dataset_train[idx]
        class_indices[label].append(idx)
    
    # 从每个类中均衡采样
    validation_indices = []
    # random.seed(1)  # 已禁用固定种子，允许每次运行产生不同的验证集
    for class_id in range(10):
        if len(class_indices[class_id]) >= samples_per_class:
            sampled = random.sample(class_indices[class_id], samples_per_class)
            validation_indices.extend(sampled)
    
    validation_dataset = Subset(dataset_train, validation_indices)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)  # validation用与训练相同的batch size
    
    print(f"✅ TEE验证集已准备: {len(validation_indices)}个样本（每类{samples_per_class}个）")
    
    return validation_loader


def test_independent_detectors(args, dataset_train, dataset_test, dict_users, attack_scenario='label_flipping', client_cluster_map=None, cluster_summary=None):
    """
    测试独立检测器
    
    Args:
        args: 参数配置
        dataset_train: 训练数据集
        dataset_test: 测试数据集
        dict_users: 用户数据索引
        attack_scenario: 攻击场景 ('label_flipping', 'noise_injection', 'no_attack')
    """
    # ==================== 防御开关配置 ====================
    # 通过环境变量控制是否启用防御机制
    # ENABLE_DEFENSE=1: 防御模式（检测器控制聚合，拒绝恶意客户端）
    # ENABLE_DEFENSE=0: 观察模式（检测器仅记录数据，不影响聚合）
    enable_defense = os.environ.get('ENABLE_DEFENSE', '1') == '1'
    defense_status = "🛡️ 防御模式" if enable_defense else "📊 观察模式"
    # =====================================================
    
    print("\n" + "="*80)
    print("联邦学习训练（带检测器过滤）")
    print("="*80)
    print(f"模型: {args.model}")
    print(f"数据集: {args.dataset}")
    print(f"客户端数: {args.num_users}")
    print(f"训练轮数: {args.epochs}")
    print(f"攻击场景: {attack_scenario}")
    
    # 防御状态
    print(f"防御状态: {defense_status}")
    if enable_defense:
        print(f"  └─ 检测器主动防御，拒绝恶意客户端")
    else:
        print(f"  └─ 检测器仅记录数据，不影响聚合决策（用于数据收集和对比实验）")
    
    # 攻击方式说明
    if attack_scenario == 'label_flipping':
        print(f"\n🔴 攻击方式: 标签翻转攻击")
        print(f"   - 100%翻转率（所有恶意客户端的训练数据标签全部随机翻转）")
    elif attack_scenario == 'noise_injection':
        print(f"\n🔴 攻击方式: 噪声注入攻击")
        print(f"   - 100%的训练数据被添加高斯噪声（std=0.25）")
    elif attack_scenario == 'no_attack':
        print(f"\n✅ 无攻击模式")
    
    # 根据攻击场景动态显示恶意客户端信息
    print(f"\n客户端配置:")
    if attack_scenario == 'no_attack':
        print(f"  恶意客户端: 0个（无攻击模式）")
        print(f"  每轮客户端数: 暖机期10个良性，正常期20个良性")
        print(f"  聚合策略: 全部聚合")
    elif attack_scenario == 'noise_injection':
        print(f"  每轮客户端数: 暖机期10个良性，正常期20个（10良性+10恶意）")
        print(f"  暖机轮数: {args.warmup_rounds}轮（Round 0-{args.warmup_rounds-1}全部良性客户端）")
        print(f"  恶意客户端: 10个（第{args.warmup_rounds}轮起，良性：恶意=10:10）")
        if enable_defense:
            threshold = 0.24
            print(f"  聚合策略: 🛡️ 检测器控制（direction_similarity < {threshold}拒绝）")
        else:
            print(f"  聚合策略: 📊 全部聚合（检测器仅记录数据）")
    elif attack_scenario == 'label_flipping':
        print(f"  每轮客户端数: 暖机期10个良性，正常期20个（10良性+10恶意）")
        print(f"  暖机轮数: {args.warmup_rounds}轮（Round 0-{args.warmup_rounds-1}全部良性客户端）")
        print(f"  恶意客户端: 10个（第{args.warmup_rounds}轮起，良性：恶意=10:10）")
        if enable_defense:
            threshold = 0.1
            print(f"  聚合策略: 🛡️ 检测器控制（direction_similarity < {threshold}拒绝）")
            print(f"             └─ 统一阈值，适用于所有数据分布")
        else:
            print(f"  聚合策略: 📊 全部聚合（检测器仅记录数据）")
    else:
        print(f"  每轮客户端数: 动态配置")
        if enable_defense:
            print(f"  聚合策略: 🛡️ 检测器控制聚合决策")
        else:
            print(f"  聚合策略: 📊 全部聚合（检测器仅记录数据）")
    
    print(f"\n检测器: update_direction（方向相似度检测）")
    print(f"  - 前{args.warmup_rounds}轮（0-{args.warmup_rounds-1}）：跳过检测（冷启动期）")
    if enable_defense:
        if attack_scenario == 'noise_injection':
            print(f"  - 第{args.warmup_rounds}轮起：启用检测（direction_similarity阈值={threshold}）")
        elif attack_scenario == 'label_flipping':
            print(f"  - 第{args.warmup_rounds}轮起：启用检测（direction_similarity阈值={threshold}）")
        else:
            print(f"  - 第{args.warmup_rounds}轮起：启用检测")
    print("="*80 + "\n")
    
    # 初始化全局模型
    global_model = getStandardNet(args)
    global_model.train()
    
    # 初始化攻击管理器
    if attack_scenario != 'no_attack':
        # 根据不同的攻击类型配置参数
        if attack_scenario == 'label_flipping':
            attack_params = {
                'poison_rate': 1.0,  # 100%翻转率（所有数据都翻转标签）
                'num_classes': args.num_classes,
                'flip_strategy': 'random'  # 随机翻转策略
            }
            attack_desc = f"标签翻转，翻转率=100%"
        elif attack_scenario == 'noise_injection':
            attack_params = {
                'poison_rate': 1.0,  # 100%加噪率（所有数据都加噪声）
                'noise_std': 0.25  # 噪声标准差
            }
            attack_desc = f"噪声注入，加噪率=100%，噪声标准差={attack_params['noise_std']}"
        else:
            raise ValueError(f"Unknown attack scenario: {attack_scenario}")
        
        attack_config = {
            'attack_type': attack_scenario,
            'malicious_ratio': args.num_corrupt / args.num_users,
            'attack_timing': 'all_rounds',  # 每轮都攻击
            'attack_start_round': 0,
            'attack_params': attack_params
        }
        attack_manager = AttackManager(
            num_clients=args.num_users,
            attack_config=attack_config
        )
        print(f"攻击配置: {attack_desc}")
    else:
        # 无攻击模式：创建测试用attack_manager（不污染模型）
        # 用于测试检测器对恶意客户端的识别能力
        attack_params = {
            'poison_rate': 1.0,
            'num_classes': args.num_classes,
            'flip_strategy': 'random'
        }
        attack_config = {
            'attack_type': 'label_flipping',
            'malicious_ratio': 0,  # 实际不污染
            'attack_timing': 'test_only',  # 仅测试用
            'attack_start_round': 0,
            'attack_params': attack_params
        }
        attack_manager = AttackManager(
            num_clients=args.num_users,
            attack_config=attack_config
        )
        attack_manager.test_mode = True  # 标记为测试模式
        print(f"✅ 无攻击模式：暖机期10个良性，正常期20个良性，全部聚合")
    
    # 注：update_direction检测器不需要TEE验证集，因此不再准备
    # （layer_wise_direction检测器需要，但已不使用）
    print("✅ 检测器配置: 仅使用update_direction（无需验证集）\n")
    
    # 初始化独立检测器测试器
    detector_tester = IndependentDetectorsTester(args)
    
    # 使用参数解析器中的warmup_rounds
    print(f"⏰ Warm-up轮数: {args.warmup_rounds}轮\n")
    
    # 记录所有轮次的检测结果
    all_detection_results = []
    
    # 记录每轮的训练详情
    round_details = []
    
    # 训练循环
    for round_idx in tqdm(range(args.epochs), desc="训练进度"):
        print(f"\n{'='*80}")
        print(f"Round {round_idx + 1}/{args.epochs}")
        print(f"{'='*80}")
        
        # 保存当前全局模型（用于计算更新）
        global_model_copy = copy.deepcopy(global_model)
        
        # 客户端选择（根据攻击类型确定客户端数）
        if attack_scenario == 'noise_injection':
            m = 15  # 噪声注入：暖机后15个客户端（10良性+5恶意）
        elif attack_scenario == 'label_flipping':
            m = 15  # 标签翻转：暖机后15个客户端（10良性+5恶意）
        else:
            m = max(int(args.frac * args.num_users), 1)  # 其他攻击：默认配置
        
        available_clients = list(range(args.num_users))
        np.random.shuffle(available_clients)
        
        # 区分无攻击模式和其他攻击模式
        if hasattr(attack_manager, 'test_mode') and attack_manager.test_mode:
            # 无攻击模式：暖机期10个良性，正常期20个良性
            if round_idx < args.warmup_rounds:
                # 暖机期：选择前10个客户端，全部良性
                selected_clients = available_clients[:10]
                print(f"选中客户端: {selected_clients}")
                
                # 显示客户端簇信息
                if client_cluster_map:
                    print(f"\n📊 选中客户端簇信息:")
                    for client_id in selected_clients:
                        cluster_info = get_client_cluster_info(client_id, client_cluster_map, cluster_summary)
                        print(f"  {cluster_info}")
                    print(f"  → 良性客户端（暖机期）: {selected_clients}")
                    print(f"  → 前{args.warmup_rounds}轮warm-up，全部10个良性客户端聚合")
                
                attack_manager.malicious_clients = set()
            else:
                # 正常期：选择前20个客户端，全部良性
                selected_clients = available_clients[:20]
                print(f"选中客户端: {selected_clients}")
                
                # 显示客户端簇信息
                if client_cluster_map:
                    print(f"\n📊 选中客户端簇信息:")
                    for client_id in selected_clients:
                        cluster_info = get_client_cluster_info(client_id, client_cluster_map, cluster_summary)
                        print(f"  {cluster_info}")
                    print(f"  → 良性客户端（正常期）: {selected_clients}")
                    print(f"  → 无攻击模式，全部20个良性客户端聚合")
                
                attack_manager.malicious_clients = set()
        elif attack_scenario == 'noise_injection':
            # 噪声注入攻击：暖机期10个良性，正常期20个（10良性+10恶意）
            if round_idx < args.warmup_rounds:
                # 暖机期：选择前10个客户端，全部良性
                benign_clients = available_clients[:10]
                malicious_clients = []
                selected_clients = benign_clients
                
                print(f"选中客户端: {selected_clients}")
                
                # 显示客户端簇信息
                if client_cluster_map:
                    print(f"\n📊 选中客户端簇信息:")
                    for client_id in selected_clients:
                        cluster_info = get_client_cluster_info(client_id, client_cluster_map, cluster_summary)
                        print(f"  {cluster_info}")
                    print(f"  → 良性客户端（暖机期）: {benign_clients}")
                    print(f"  → 前{args.warmup_rounds}轮warm-up，全部10个良性客户端聚合")
                
                attack_manager.malicious_clients = set()
            else:
                # 正常期：固定20个客户端，10个良性 + 10个恶意
                selected_clients = available_clients[:20]
                benign_clients = selected_clients[:10]  # 前10个是良性
                malicious_clients = selected_clients[10:]  # 后10个是恶意
                
                print(f"选中客户端: {selected_clients}")
                
                # 显示客户端簇信息
                if client_cluster_map:
                    print(f"\n📊 选中客户端簇信息:")
                    for client_id in selected_clients:
                        cluster_info = get_client_cluster_info(client_id, client_cluster_map, cluster_summary)
                        print(f"  {cluster_info}")
                    print(f"  → 良性客户端: {benign_clients}")
                    print(f"  → 恶意客户端: {malicious_clients}")
                    print(f"  → 检测器工作中（direction_similarity检测，控制聚合决策）")
                
                attack_manager.malicious_clients = set(malicious_clients)
        elif attack_scenario == 'label_flipping':
            # 标签翻转攻击：暖机期10个良性，正常期20个（10良性+10恶意）
            if round_idx < args.warmup_rounds:
                # 暖机期：选择前10个客户端，全部良性
                benign_clients = available_clients[:10]
                malicious_clients = []
                selected_clients = benign_clients
                
                print(f"选中客户端: {selected_clients}")
                
                # 显示客户端簇信息
                if client_cluster_map:
                    print(f"\n📊 选中客户端簇信息:")
                    for client_id in selected_clients:
                        cluster_info = get_client_cluster_info(client_id, client_cluster_map, cluster_summary)
                        print(f"  {cluster_info}")
                    print(f"  → 良性客户端（暖机期）: {benign_clients}")
                    print(f"  → 前{args.warmup_rounds}轮warm-up，全部10个良性客户端聚合")
                
                attack_manager.malicious_clients = set()
            else:
                # 正常期：固定20个客户端，10个良性 + 10个恶意
                selected_clients = available_clients[:20]
                benign_clients = selected_clients[:10]  # 前10个是良性
                malicious_clients = selected_clients[10:]  # 后10个是恶意
                
                print(f"选中客户端: {selected_clients}")
                
                # 显示客户端簇信息
                if client_cluster_map:
                    print(f"\n📊 选中客户端簇信息:")
                    for client_id in selected_clients:
                        cluster_info = get_client_cluster_info(client_id, client_cluster_map, cluster_summary)
                        print(f"  {cluster_info}")
                    print(f"  → 良性客户端: {benign_clients}")
                    print(f"  → 恶意客户端: {malicious_clients}")
                    print(f"  → 检测器工作中（direction_similarity检测）")
                
                attack_manager.malicious_clients = set(malicious_clients)
        else:
            # 其他攻击模式：正常选择m个客户端
            selected_clients = available_clients[:m]
            print(f"选中客户端: {selected_clients}")
            
            # 显示客户端簇信息
            if client_cluster_map:
                print(f"\n📊 选中客户端簇信息:")
                for client_id in selected_clients:
                    cluster_info = get_client_cluster_info(client_id, client_cluster_map, cluster_summary)
                    print(f"  {cluster_info}")
            
            # warm-up期不设置恶意客户端
            if attack_manager:
                if round_idx >= args.warmup_rounds:
                    # 这里是其他未明确指定的攻击类型的处理
                    attack_manager.malicious_clients = set()
                    print(f"  → 前{args.warmup_rounds}轮warm-up后，需要明确定义攻击模式")
                else:
                    attack_manager.malicious_clients = set()
                    print(f"  → 前{args.warmup_rounds}轮warm-up，无攻击")
        
        w_locals = []
        aggregated_clients = []  # 记录被聚合的客户端ID
        
        for user_idx in selected_clients:
            # 判断是否是恶意客户端
            is_malicious = False
            if attack_manager and attack_manager.is_malicious(user_idx):
                is_malicious = True
            
            print(f"\n--- 客户端 {user_idx} ({'实际恶意' if is_malicious else '实际良性'}) ---")
            
            # 1. 外部训练
            print("  [1/3] 外部训练中...")
            local = LocalUpdate_XFL_SmallData(
                args=args,
                dataset=dataset_train,
                idxs=dict_users[user_idx]
            )
            
            external_model = copy.deepcopy(global_model).to(args.device)
            w_external, external_loss = local.train_external(
                round=round_idx,
                external_model=external_model,
                client_id=user_idx,
                attack_manager=attack_manager,
                global_model=global_model if args.use_fedprox else None
            )
            external_model.load_state_dict(w_external)
            
            # 2. TEE训练
            print("  [2/3] TEE训练中...")
            tee_model = copy.deepcopy(global_model).to(args.device)
            w_tee, tee_loss = local.train_tee_secure(
                round=round_idx,
                tee_model=tee_model,
                client_id=user_idx,
                attack_manager=None,  # TEE内部不受攻击
                global_model=global_model  # FedProx: 传递全局模型以添加proximal term
            )
            tee_model.load_state_dict(w_tee)
            
            # 3. 决定是否聚合
            # warm-up期跳过检测：冷启动期和过渡期数据不稳定
            # warm-up结束后使用检测：direction_similarity检测 + (noise时)BN欧氏距离检测
            
            if round_idx < args.warmup_rounds:
                # warm-up期：跳过检测
                detected_as_malicious = False
                
                # 测试模式：warm-up期也不聚合恶意测试客户端
                if hasattr(attack_manager, 'test_mode') and attack_manager.test_mode:
                    should_aggregate = not is_malicious
                else:
                    should_aggregate = True
                
                print(f"\n  📊 客户端 {user_idx} 检测详情：")
                print(f"     ⏭️  第{round_idx}轮跳过检测（冷启动/过渡期，共跳过前{args.warmup_rounds}轮）")
                if hasattr(attack_manager, 'test_mode') and attack_manager.test_mode and is_malicious:
                    print(f"     ➜ 🧪 测试客户端（不聚合）")
                else:
                    print(f"     ➜ ✅ 聚合模型（无检测）")
                
                # 暖机期创建空检测结果（保持数据结构一致性）
                detection_result = {
                    'client_id': user_idx,
                    'is_malicious': is_malicious,
                    'round': round_idx,
                    'warmup_period': True,
                    'detection_skipped': True,
                    'detectors': {}  # 空检测器字典，避免后续统计时KeyError
                }
                all_detection_results.append(detection_result)
            else:
                # 第5轮起：运行检测器
                print("  [3/3] 运行检测器...")
                detection_result = detector_tester.test_update_direction_only(
                    global_model=global_model_copy,
                    external_model=external_model,
                    tee_model=tee_model,
                    client_id=user_idx,
                    is_malicious=is_malicious,
                    attack_scenario=attack_scenario  # 传递攻击场景
                )
                detection_result['round'] = round_idx
                all_detection_results.append(detection_result)
                
                # 提取检测结果
                detected_as_malicious = False
                direction_sim = None
                bn_distance = None
                
                # 1. 提取direction_similarity
                if 'detectors' in detection_result and 'update_direction' in detection_result['detectors']:
                    update_direction_result = detection_result['detectors']['update_direction']
                    if 'detection_result' in update_direction_result:
                        features = update_direction_result['detection_result'].get('features', {})
                        direction_sim = features.get('update_direction_similarity')
                
                # 2. BN欧氏距离检测已屏蔽，不再提取相关数据
                # if attack_scenario == 'noise_injection' and 'batchnorm_euclidean' in detection_result.get('detectors', {}):
                #     bn_result = detection_result['detectors']['batchnorm_euclidean']
                #     if 'detection_result' in bn_result:
                #         bn_features = bn_result['detection_result'].get('features', {})
                #         bn_distance = bn_features.get('mean_distance')
                
                # 检测判断逻辑：根据攻击类型和防御开关
                if attack_scenario == 'noise_injection':
                    # 噪声注入：任一检测器判断为异常即判定为恶意（OR逻辑）
                    # 理由：两个都正常才聚合，有一个异常就拒绝
                    
                    # 🎚️ 阈值控制层：根据防御开关动态调整阈值
                    if enable_defense:
                        # 防御模式：使用实际检测阈值
                        direction_threshold = 0.24  # 基于std=0.25优化（确保绝大多数恶意被检测）
                        bn_threshold = 0.008  # 基于实际运行数据（深层BN：L2-L3-L4，能捕获伪装型攻击者）
                    else:
                        # 观察模式：使用极端阈值屏蔽检测（确保所有客户端都被判定为"正常"）
                        direction_threshold = -1.0  # 负值阈值 → 所有相似度(≥-1)都正常
                        bn_threshold = 999.0  # 极大阈值 → 所有距离(<999)都正常
                    
                    direction_anomaly = (direction_sim is not None and direction_sim < direction_threshold)
                    
                    # BN欧氏距离检测已屏蔽，不再提取相关数据
                    # sensitive_bn_distance = None
                    # if attack_scenario == 'noise_injection' and 'batchnorm_euclidean' in detection_result.get('detectors', {}):
                    #     bn_result = detection_result['detectors']['batchnorm_euclidean']
                    #     if 'detection_result' in bn_result:
                    #         bn_features = bn_result['detection_result'].get('features', {})
                    #         sensitive_bn_distance = bn_features.get('sensitive_mean')  # 使用深层BN (L2-L3-L4)
                    # 
                    # bn_anomaly = (sensitive_bn_distance is not None and sensitive_bn_distance > bn_threshold)
                    
                    # 单一检测器：仅使用direction_similarity
                    detected_as_malicious = direction_anomaly
                else:
                    # 标签翻转：仅使用direction_similarity
                    # 🎚️ 阈值控制层：根据防御开关调整阈值
                    if enable_defense:
                        # 防御模式：统一阈值为0.1
                        direction_threshold_label = 0.1
                    else:
                        # 观察模式：使用极端阈值屏蔽检测
                        direction_threshold_label = -1.0  # 负值阈值 → 所有客户端都正常
                    
                    if direction_sim is not None and direction_sim < direction_threshold_label:
                        detected_as_malicious = True
                
                # ⚖️ 聚合决策层：根据模式和防御开关决定聚合策略
                if hasattr(attack_manager, 'test_mode') and attack_manager.test_mode:
                    # 测试模式：恶意测试客户端不聚合，良性客户端全部聚合
                    should_aggregate = not is_malicious
                elif attack_manager is None:
                    # 无攻击模式：检测器仅作分数评估，不决定聚合，固定全部聚合
                    should_aggregate = True
                else:
                    # 攻击模式（包括噪声注入和标签翻转）
                    if enable_defense:
                        # 🛡️ 防御模式：检测器控制聚合决策
                        should_aggregate = not detected_as_malicious
                    else:
                        # 📊 观察模式：检测器仅记录数据，不影响聚合（全部聚合）
                        should_aggregate = True
                
                # 输出详细检测信息
                print(f"\n  📊 客户端 {user_idx} 检测详情：")
                if attack_scenario == 'noise_injection':
                    # 噪声攻击：仅显示direction_similarity检测器
                    if direction_sim is not None:
                        dir_status = "✓正常" if direction_sim >= direction_threshold else "✗异常"
                        print(f"     • 更新方向相似度: {direction_sim:.4f}  (阈值: {direction_threshold:.2f})  {dir_status}")
                    # BN欧氏距离检测已屏蔽，不再显示相关信息
                    # if sensitive_bn_distance is not None:
                    #     bn_status = "✓正常" if sensitive_bn_distance <= bn_threshold else "✗异常"
                    #     print(f"     • 深层BN欧氏距离(L2-L3-L4): {sensitive_bn_distance:.4f}  (阈值: {bn_threshold:.3f})  {bn_status}")
                    #     # 额外显示浅层对比（如果有）
                    #     if 'batchnorm_euclidean' in detection_result.get('detectors', {}):
                    #         bn_result = detection_result['detectors']['batchnorm_euclidean']
                    #         if 'detection_result' in bn_result:
                    #             bn_features = bn_result['detection_result'].get('features', {})
                    #             shallow_mean = bn_features.get('shallow_mean')
                    #             if shallow_mean is not None:
                    #                 print(f"       (浅层L1: {shallow_mean:.4f})")
                    print(f"     • 检测策略: 单一检测器（direction_similarity）")
                else:
                    # 标签翻转：只显示direction_similarity
                    if direction_sim is not None:
                        dir_status = "✓通过" if direction_sim >= direction_threshold_label else "✗异常"
                        print(f"     • 更新方向相似度: {direction_sim:.4f}  (阈值: {direction_threshold_label:.2f})  {dir_status}")
                
                # 输出判断结果和真实情况（仅第2轮起）
                detection_status = "检测为恶意" if detected_as_malicious else "检测为良性"
                actual_status = "实际恶意" if is_malicious else "实际良性"
                match_status = "✓" if (detected_as_malicious == is_malicious) else "✗"
                
                # 根据模式输出不同信息
                if hasattr(attack_manager, 'test_mode') and attack_manager.test_mode:
                    # 测试模式：显示检测分数和聚合决策
                    if is_malicious:
                        print(f"     ➜ 🧪 测试客户端（不聚合） - {detection_status}, {actual_status} {match_status}")
                    else:
                        print(f"     ➜ ✅ 良性客户端（聚合） - {detection_status}, {actual_status} {match_status}")
                elif attack_manager is None:
                    # 无攻击模式：始终聚合，显示检测分数
                    print(f"     ➜ ✅ 固定聚合（无攻击模式） - 检测分数: {detection_status}")
                else:
                    # 攻击模式（包括噪声注入和标签翻转）
                    if enable_defense:
                        # 防御模式：检测器控制聚合
                        if should_aggregate:
                            print(f"     ➜ ✅ 聚合模型 - {detection_status}, {actual_status} {match_status} [🛡️ 防御模式]")
                        else:
                            print(f"     ➜ ❌ 拒绝聚合 - {detection_status}, {actual_status} {match_status} [🛡️ 防御模式]")
                    else:
                        # 观察模式：全部聚合，但显示检测结果
                        print(f"     ➜ ✅ 聚合模型 - {detection_status}, {actual_status} {match_status} [📊 观察模式 - 检测器不影响聚合]")
            
            # 聚合模型（所有轮次都执行）
            if should_aggregate:
                w_locals.append(copy.deepcopy(w_external))
                aggregated_clients.append(user_idx)
            
            # 清理内存
            del external_model, tee_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # 等待GPU操作完成，避免内存碎片
        
        # 聚合
        if w_locals:
            print(f"\n聚合 {len(w_locals)} 个客户端模型...")
            # Aggregation需要lens参数（每个客户端的数据量）
            # 使用被聚合的客户端列表（基于检测器判断）
            lens = [len(dict_users[idx]) for idx in aggregated_clients]
            w_glob = Aggregation(w_locals, lens)
            global_model.load_state_dict(w_glob)
            print(f"✅ 聚合完成，共 {len(aggregated_clients)} 个客户端: {aggregated_clients}")
        
        # 测试全局模型（每轮都测试）
        global_model.eval()
        
        # 清理GPU缓存，避免cuDNN错误
        if args.gpu != -1:
            torch.cuda.empty_cache()
        
        try:
            test_acc, test_loss = test_model(global_model, dataset_test, args)
        except RuntimeError as e:
            if "cuDNN" in str(e):
                print(f"⚠️  cuDNN错误，尝试重新初始化模型: {e}")
                # 重新初始化模型
                global_model = init_model(args)
                global_model.load_state_dict(w_glob)
                global_model.to(args.device)
                global_model.eval()
                torch.cuda.empty_cache()
                test_acc, test_loss = test_model(global_model, dataset_test, args)
            else:
                raise e
        
        global_model.train()
        print(f"\n📊 轮次 {round_idx + 1}/{args.epochs} 全局模型准确率: {test_acc:.2%} (损失: {test_loss:.4f})")
        
        # 记录本轮详情
        round_detail = {
            'round': round_idx,
            'selected_clients': selected_clients,
            'malicious_clients': sorted(list(attack_manager.malicious_clients)) if attack_manager else [],
            'aggregated_clients': aggregated_clients,
            'num_aggregated': len(aggregated_clients),
            'global_accuracy': float(test_acc),
            'global_loss': float(test_loss),
            'client_details': []
        }
        
        # 添加每个客户端的检测详情（包括所有检测器数据）
        for result in all_detection_results:
            if result['round'] == round_idx:
                client_detail = {
                    'client_id': result['client_id'],
                    'is_malicious': result['is_malicious'],
                    'aggregated': result['client_id'] in aggregated_clients,
                }
                
                # 记录所有检测器的结果
                detectors_data = {}
                if 'detectors' in result:
                    # direction_similarity检测器
                    if 'update_direction' in result['detectors']:
                        update_dir = result['detectors']['update_direction'].get('detection_result', {})
                        detectors_data['update_direction'] = {
                            'is_anomaly': update_dir.get('is_anomaly'),
                            'features': update_dir.get('features', {}),
                            'evidence': update_dir.get('evidence', [])
                        }
                    
                    # BN欧氏距离检测器已屏蔽，不再记录相关数据
                    # if 'batchnorm_euclidean' in result['detectors']:
                    #     bn_euclidean = result['detectors']['batchnorm_euclidean'].get('detection_result', {})
                    #     detectors_data['batchnorm_euclidean'] = {
                    #         'is_anomaly': bn_euclidean.get('is_anomaly'),
                    #         'features': bn_euclidean.get('features', {}),
                    #         'evidence': bn_euclidean.get('evidence', [])
                    #     }
                
                client_detail['detectors'] = detectors_data
                round_detail['client_details'].append(client_detail)
        
        round_details.append(round_detail)
    
    # 训练结束，计算统计
    print(f"\n{'='*80}")
    print("训练完成，计算检测器统计...")
    print(f"{'='*80}\n")
    
    detector_stats = detector_tester.calculate_simple_statistics(all_detection_results)
    
    # 构建包含数据分布信息和时间戳的文件名
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.iid:
        distribution_suffix = "iid"
    else:
        # Non-IID: 包含case和beta信息
        distribution_suffix = f"noniid_case{args.noniid_case}_beta{args.data_beta}"
    
    result_filename = f"independent_test_{args.model}_{attack_scenario}_{distribution_suffix}_{timestamp}.json"
    
    # 保存结果（包括攻击配置信息和每轮训练详情）
    filename = detector_tester.save_results(
        all_detection_results, 
        detector_stats,
        filename=result_filename,
        attack_config=attack_config if attack_scenario != 'no_attack' else None,
        round_details=round_details
    )
    
    return detector_stats, all_detection_results


def test_model(model, dataset_test, args):
    """测试模型准确率"""
    model.eval()
    test_loss = 0
    correct = 0
    
    data_loader = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=args.local_bs,
        shuffle=False
    )
    
    loss_func = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            
            # 检查输入数据是否有效
            if torch.isnan(data).any() or torch.isinf(data).any():
                print(f"⚠️  检测到无效输入数据 (NaN/Inf) 在批次 {idx}")
                continue
                
            try:
                output = model(data)['output']
                
                # 检查输出是否有效
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"⚠️  检测到无效模型输出 (NaN/Inf) 在批次 {idx}")
                    continue
                    
                test_loss += loss_func(output, target).item()
                
                y_pred = output.argmax(dim=1, keepdim=True)
                correct += y_pred.eq(target.view_as(y_pred)).sum().item()
                
            except RuntimeError as e:
                if "cuDNN" in str(e):
                    print(f"⚠️  cuDNN错误在批次 {idx}: {e}")
                    # 清理缓存并跳过这个批次
                    if args.gpu != -1:
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
    
    return accuracy, test_loss


def main():
    """主函数"""
    # 解析参数
    args = args_parser()
    
    # 设置必要的默认参数（不覆盖命令行参数）
    # dataset, model, epochs, frac, local_ep, lr 等从命令行参数或 options.py 默认值获取
    
    # 根据数据集设置类别数和通道数（如果未设置）
    if args.dataset == 'mnist':
        if not hasattr(args, 'num_classes') or args.num_classes == 10:
            args.num_classes = 10
        if not hasattr(args, 'num_channels') or args.num_channels == 3:
            args.num_channels = 1
    elif args.dataset == 'fmnist':
        if not hasattr(args, 'num_classes') or args.num_classes == 10:
            args.num_classes = 10
        if not hasattr(args, 'num_channels') or args.num_channels == 3:
            args.num_channels = 1
    elif args.dataset == 'cifar10':
        if not hasattr(args, 'num_classes') or args.num_classes != 10:
            args.num_classes = 10
        if not hasattr(args, 'num_channels') or args.num_channels != 3:
            args.num_channels = 3
    
    # num_corrupt 不从命令行传入，需要设置默认值
    args.num_corrupt = 10
    
    # 确保 momentum 被设置（SGD 需要）
    if not hasattr(args, 'momentum'):
        args.momentum = 0.9
    else:
        args.momentum = 0.9
    
    # 数据分布设置（优先级：命令行参数 > 环境变量 > 默认值）
    # 如果环境变量存在且非空，则使用环境变量覆盖默认值
    # 注意：argparse总是会设置默认值，所以这里直接检查环境变量
    data_distribution_env = os.environ.get('DATA_DISTRIBUTION', '').strip()
    if data_distribution_env:
        data_distribution = data_distribution_env.lower()
        args.iid = (data_distribution == 'iid')
    # 如果环境变量不存在或为空，使用args中的默认值（已在args_parser中设置，默认iid=1）
    
    # 聚合策略设置（Non-IID环境下默认使用FedProx）
    use_fedprox = int(os.environ.get('USE_FEDPROX', '1')) == 1
    
    # Non-IID设置
    if not args.iid:
        # 保留原始NONIID_CASE用于逻辑判断（检测阈值等）
        logical_noniid_case = int(os.environ.get('NONIID_CASE', '2'))
        
        # 使用ACTUAL_NONIID_CASE用于数据分割（shell脚本已映射到 case >= 4 以使用Dirichlet分布）
        # 重新组织映射关系：从低到高排序
        # NONIID_CASE=1 -> ACTUAL_NONIID_CASE=4 (α=0.8, 轻度异构)
        # NONIID_CASE=2 -> ACTUAL_NONIID_CASE=5 (α=0.5, 中度异构)
        # NONIID_CASE=3 -> ACTUAL_NONIID_CASE=6 (α=0.1, 重度异构)
        args.noniid_case = int(os.environ.get('ACTUAL_NONIID_CASE', os.environ.get('NONIID_CASE', '5')))
        
        # 存储逻辑case用于检测阈值判断
        args.logical_noniid_case = logical_noniid_case
        
        # 读取 DATA_BETA (α值) 从环境变量
        args.data_beta = float(os.environ.get('DATA_BETA', '0.5'))
        
        # Non-IID时的聚合策略（由USE_FEDPROX环境变量控制）
        args.use_fedprox = use_fedprox
        
        # FedProx参数自动映射（仅当使用FedProx时有效）：数据越不平衡，正则化强度越大
        # 可通过环境变量PROX_ALPHA手动覆盖
        # 使用logical_noniid_case (1, 2, 3) 作为映射键
        prox_alpha_map = {
            1: 0.01,  # mild (轻度异构, α=0.8): 极弱正则化
            2: 0.1,   # moderate (中度异构, α=0.5): 中等正则化
            3: 0.5    # extreme (重度异构, α=0.1): 强正则化（极度异构需要更强约束）
        }
        default_prox_alpha = prox_alpha_map.get(args.logical_noniid_case, 0.01)
        
        # 处理 PROX_ALPHA 环境变量（可能为空字符串）
        prox_alpha_str = os.environ.get('PROX_ALPHA', str(default_prox_alpha))
        if prox_alpha_str and prox_alpha_str.strip():
            args.prox_alpha = float(prox_alpha_str)
        else:
            args.prox_alpha = default_prox_alpha
        
        # 输出配置信息
        noniid_case_names = {1: "mild (轻度异构)", 2: "moderate (中度异构)", 3: "extreme (重度异构)"}
        case_name = noniid_case_names.get(args.logical_noniid_case, "unknown")
        case_desc_map = {1: "Dirichlet α=0.8", 2: "Dirichlet α=0.5", 3: "Dirichlet α=0.1"}
        case_desc = case_desc_map.get(args.logical_noniid_case, "")
        
        print(f"\n📊 数据分布: Non-IID")
        print(f"   noniid_case: {args.logical_noniid_case} -> 实际case={args.noniid_case} ({case_name})")
        if case_desc:
            print(f"   └─ 数据分割方法: {case_desc}")
        
        # 检查data_beta是否由环境变量设置
        data_beta_from_env = os.environ.get('DATA_BETA', '')
        if data_beta_from_env:
            print(f"   data_beta (α): {args.data_beta} (来源: 环境变量)")
        else:
            print(f"   data_beta (α): {args.data_beta} (来源: 自动映射)")
        
        if args.use_fedprox:
            print(f"   聚合方法: FedProx (本地正则化μ={args.prox_alpha})")
            print(f"   └─ 正则化项: loss += (μ/2)||w - w_global||²")
            # 显示参数来源
            prox_alpha_str = os.environ.get('PROX_ALPHA', '').strip()
            if prox_alpha_str:
                # 环境变量中有值（可能是自动映射设置的，也可能是用户手动设置的）
                # 检查是否与默认值一致来判断来源
                if abs(args.prox_alpha - default_prox_alpha) < 1e-6:
                    print(f"   └─ prox_alpha 来源: 自动映射 (logical_case {args.logical_noniid_case} -> {args.prox_alpha})")
                else:
                    print(f"   └─ prox_alpha 来源: 手动设置")
            else:
                print(f"   └─ prox_alpha 来源: 自动映射 (logical_case {args.logical_noniid_case} -> {args.prox_alpha})")
            print(f"   └─ 说明: μ值根据数据异构程度自动调整（α越小，μ越大）")
        else:
            print(f"   聚合方法: FedAvg (无正则化约束)")
            print(f"   └─ 说明: 可能在学习初期收敛较慢")
    else:
        args.use_fedprox = False
        print(f"\n📊 数据分布: IID")
        print(f"   聚合方法: FedAvg")
    
    # GPU设置
    args.gpu = 0 if torch.cuda.is_available() else -1
    args.device = torch.device(f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu')
    
    # 随机种子设置（优先级：环境变量 > 默认值）
    if 'RANDOM_SEED' in os.environ and os.environ['RANDOM_SEED']:
        args.seed = int(os.environ['RANDOM_SEED'])
        print(f"\n🎲 随机种子: {args.seed} (来源: 环境变量)")
    elif args.seed is not None:
        print(f"\n🎲 随机种子: {args.seed} (来源: 默认值)")
    else:
        print(f"\n🎲 使用随机种子（每次运行不同）")
    
    # 攻击类型设置（优先级：环境变量 > 命令行参数 > 默认值）
    # 1. 先尝试从环境变量获取
    attack_type = os.environ.get('ATTACK_TYPE', None)
    
    # 2. 如果环境变量未设置，使用命令行参数（如果有且不是默认值）
    if attack_type is None:
        if hasattr(args, 'attack_scenario') and args.attack_scenario:
            attack_type = args.attack_scenario
        else:
            # 3. 最后使用默认值
            attack_type = 'label_flipping'
    
    # 输出实际使用的攻击类型
    print(f"\n🎯 攻击类型确认: {attack_type}")
    if 'ATTACK_TYPE' in os.environ:
        print(f"   来源: 环境变量 ATTACK_TYPE={os.environ['ATTACK_TYPE']}")
    elif hasattr(args, 'attack_scenario'):
        print(f"   来源: 命令行参数 --attack_scenario={args.attack_scenario}")
    else:
        print(f"   来源: 默认值")
    
    # 加载数据集
    print("\n加载数据集...")
    dataset_train, dataset_test, dict_users = get_dataset(args)
    print(f"训练集大小: {len(dataset_train)}")
    print(f"测试集大小: {len(dataset_test)}")
    
    # 加载簇映射信息
    print("\n加载簇映射信息...")
    client_cluster_map, cluster_summary = load_cluster_mapping(args)
    
    # 运行测试（使用指定的攻击类型）
    attack_scenarios = [attack_type]  # 使用指定的攻击类型
    
    for attack_scenario in attack_scenarios:
        print(f"\n{'#'*80}")
        print(f"测试攻击场景: {attack_scenario}")
        print(f"{'#'*80}")
        
        detector_stats, all_results = test_independent_detectors(
            args, 
            dataset_train, 
            dataset_test, 
            dict_users, 
            attack_scenario
        )
    
    print("\n✅ 所有测试完成！")


if __name__ == "__main__":
    main()


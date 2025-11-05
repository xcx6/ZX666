"""
攻击配置文件
定义各种攻击场景的配置
"""

# 标签翻转攻击配置
LABEL_FLIPPING_CONFIG = {
    'malicious_ratio': 0.5,  # 70%的客户端为恶意 (增强)
    'attack_type': 'label_flipping',
    'attack_params': {
        'poison_rate': 1.0,  # 污染100%的数据 (增强)
        'num_classes': 10,
        'flip_strategy': 'random'  # 随机翻转策略
    },
    'attack_timing': 'all_rounds'  # 每轮都可能攻击
}

# 噪声注入攻击配置
NOISE_INJECTION_CONFIG = {
    'malicious_ratio': 0.15,
    'attack_type': 'noise_injection',
    'attack_params': {
        'poison_rate': 0.3,
        'noise_std': 0.1
    },
    'attack_timing': 'all_rounds'
}

# 后门攻击配置
BACKDOOR_CONFIG = {
    'malicious_ratio': 0.1,
    'attack_type': 'backdoor',
    'attack_params': {
        'poison_rate': 0.1,
        'trigger_size': 3,
        'target_class': 0
    },
    'attack_timing': 'random_rounds'
}

# 无攻击配置（用于对比实验）
NO_ATTACK_CONFIG = {
    'malicious_ratio': 0.0,
    'attack_type': None,
    'attack_params': {},
    'attack_timing': 'never'
}

# 预定义攻击场景
# 极端攻击配置 (用于测试攻击效果)
EXTREME_ATTACK_CONFIG = {
    'malicious_ratio': 0.8,  # 80%的客户端为恶意
    'attack_type': 'label_flipping',
    'attack_params': {
        'poison_rate': 1.0,  # 污染100%的数据
        'num_classes': 10,
        'flip_strategy': 'targeted',  # 目标攻击更有效
        'target_class': 0  # 全部翻转到类别0
    },
    'attack_timing': 'all_rounds'
}

ATTACK_SCENARIOS = {
    'no_attack': NO_ATTACK_CONFIG,
    'label_flipping': LABEL_FLIPPING_CONFIG,
    'noise_injection': NOISE_INJECTION_CONFIG,
    'backdoor': BACKDOOR_CONFIG,
    'extreme_attack': EXTREME_ATTACK_CONFIG  # 新增极端攻击
}

import random
from typing import Dict, Any, List, Set, Tuple

from .data_poisoning import LabelFlippingAttack, NoiseInjectionAttack, BackdoorAttack


class AttackManager:
    def __init__(self, num_clients: int, attack_config: Dict[str, Any]):
        self.num_clients: int = num_clients
        self.attack_config: Dict[str, Any] = attack_config
        self.attack_type: str | None = attack_config.get('attack_type')
        self.attack_params: Dict[str, Any] = attack_config.get('attack_params', {})
        self.malicious_ratio: float = attack_config.get('malicious_ratio', 0.0)
        self.attack_timing: str = attack_config.get('attack_timing', 'never')

        self.malicious_clients: Set[int] = set()
        self.attack_summary: Dict[int, Dict[str, Any]] = {}

        # 初始化对应攻击实例
        self.attack_instance = None
        if self.attack_type == 'label_flipping':
            self.attack_instance = LabelFlippingAttack(
                poison_rate=self.attack_params.get('poison_rate', 0.5),
                num_classes=self.attack_params.get('num_classes', 10),
                flip_strategy=self.attack_params.get('flip_strategy', 'random'),
            )
        elif self.attack_type == 'noise_injection':
            self.attack_instance = NoiseInjectionAttack(
                poison_rate=self.attack_params.get('poison_rate', 0.3),
                noise_std=self.attack_params.get('noise_std', 0.1),
            )
        elif self.attack_type == 'backdoor':
            self.attack_instance = BackdoorAttack(
                poison_rate=self.attack_params.get('poison_rate', 0.1),
                trigger_size=self.attack_params.get('trigger_size', 3),
                target_class=self.attack_params.get('target_class', 0),
            )

    def setup_malicious_clients(self, participating_clients: List[int], current_round: int, total_rounds: int) -> None:
        self.malicious_clients.clear()
        if self.malicious_ratio <= 0 or not self._should_attack_in_round(current_round, total_rounds):
            return
        num_malicious = max(1, int(round(len(participating_clients) * self.malicious_ratio)))
        num_malicious = min(num_malicious, len(participating_clients))
        self.malicious_clients = set(random.sample(participating_clients, num_malicious))
        self.attack_summary[current_round] = {
            'malicious_clients': sorted(self.malicious_clients),
            'attack_type': self.attack_type,
        }

    def is_malicious(self, client_id: int) -> bool:
        return client_id in self.malicious_clients

    def poison_data(self, client_id, images, labels):
        if not self.attack_instance or not self.is_malicious(client_id):
            return images, labels
        # 兼容 data_poisoning.py 的方法名 poison_data
        return self.attack_instance.poison_data(images, labels)

    def get_malicious_clients(self) -> List[int]:
        return sorted(self.malicious_clients)

    def get_attack_summary(self) -> Dict[int, Dict[str, Any]]:
        return self.attack_summary

    def _should_attack_in_round(self, current_round: int, total_rounds: int) -> bool:
        if self.attack_timing == 'all_rounds':
            return True
        if self.attack_timing == 'last_round':
            return current_round == total_rounds - 1
        if self.attack_timing == 'random_rounds':
            return random.random() < 0.5
        if self.attack_timing == 'never' or self.attack_type is None:
            return False
        return False

import random
import numpy as np
import torch


def set_random_seed(seed):
    """
        set random seed
        If seed is None, use random seed (no fixed seed)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        # Use random seed based on current time
        import time
        random_seed = int(time.time() * 1000) % (2**32)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        print(f"🎲 使用随机种子: {random_seed}")

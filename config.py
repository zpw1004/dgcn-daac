import torch
import numpy as np
import random
import os

def set_seed(seed: int = 42):
    random.seed(seed)                            # Python 内置随机库
    np.random.seed(seed)                         # NumPy 随机
    torch.manual_seed(seed)                      # PyTorch CPU
    torch.cuda.manual_seed(seed)                 # PyTorch 当前GPU
    torch.cuda.manual_seed_all(seed)             # PyTorch 所有GPU

    torch.backends.cudnn.deterministic = True    # 禁用cudnn中非确定性算法
    torch.backends.cudnn.benchmark = False       # 禁用cudnn autotune以固定算法路径

    os.environ['PYTHONHASHSEED'] = str(seed)     # 固定Python哈希种子（影响某些操作顺序）

# Set seed at the beginning
set_seed(42)
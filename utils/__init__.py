import random
from datetime import datetime

import numpy as np
import torch


def setup_seed(seed: int):
    """
    保证每次实验的结果是一样的
    :param seed: 随机数种子
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def datetime_now_str() -> str:
    # 获取处理后的时间
    return datetime.now().strftime('%Y%m%d%H%M%S')

import torch
import numpy as np
import os


def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_seed(seed):
    if seed == -1:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)




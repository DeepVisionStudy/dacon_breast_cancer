import os
import os.path as osp
import yaml
import torch
import random
import argparse
import numpy as np
from types import SimpleNamespace


def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_exp_dir(work_dir):
    work_dir = work_dir.split('./')[-1]
    if not osp.exists(osp.join(os.getcwd(), work_dir)):
        exp_dir = osp.join(os.getcwd(), work_dir, 'exp0')
    else:
        idx = 1
        exp_dir = osp.join(os.getcwd(), work_dir, f'exp{idx}')
        while osp.exists(exp_dir):
            idx += 1
            exp_dir = osp.join(os.getcwd(), work_dir, f'exp{idx}')
    
    os.makedirs(exp_dir)
    return exp_dir


def save_config(args, save_dir):
    with open(save_dir, 'w') as f:
        yaml.safe_dump(args.__dict__, f)


def load_config(config_dir):
    with open(config_dir, 'r') as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(**config)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
# Finished by zfang 2020/02/14 13:50pm
import yaml
from torch.utils.data import Dataset, DataLoader

class CustomException(Exception):
    pass

class COLORS:
    """Color scheme for logging to console"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_cfg(cfg_path):
    cfg = yaml.load(open(cfg_path, 'r'))
    return cfg

def get_dataloader(cfg, dataset):
    return DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])

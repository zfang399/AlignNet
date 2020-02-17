# Finished by zfang 2020/02/16 12:11pm
import torch
from torchvision.utils import save_image
import torch.nn as nn

import argparse
from tqdm import tqdm
import numpy as np
import os
import os.path as osp

from dataset.lip_dataset import VoxCelebDataset
from model.Networks import AlignNet
from utils.utils import get_cfg, get_dataloader, CustomException
from code.trainer_AlignNet import AlignNetTrainer

def train(args, cfg):
    trainer = AlignNetTrainer(args, cfg)
    if args.mode == "train":
        trainer.train()
    elif args.mode == "test":
        trainer.val()
    else:
        raise CustomException("Unrecognized mode!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser for extracting the identities in LRS3 dataset")
    parser.add_argument("--cfg_path", 
                        type=str, 
                        default="/home/jianrenw/research/AlignNet/cfg/cfg.yaml", 
                        help="path of the cfg file")
    parser.add_argument("--mode", 
                        type=str, 
                        default="train",
                        help="mode: train / test")
    parser.add_argument('--local_distortion',
                        action='store_true',
                        help="whether to enable local distortion")
    parser.add_argument('--global_stretch',
                        action='store_true',
                        help="whether to enable global_stretch")
    parser.add_argument('--global_shift',
                        action='store_true',
                        help="whether to enable global_shift")
    parser.add_argument("--data_parallel", 
                        action="store_true", 
                        help="whether to use data parallelism")
    parser.add_argument("--log", 
                        action="store_true", 
                        help="whether to log")
    parser.add_argument("--log_dir", 
                        type=str, 
                        default="/home/jianrenw/research/AlignNet/experiments/%s/checkpoints",
                        help="The directory to save models/tb logs/examples", 
                        required=True)
    parser.add_argument("--experiment_name", 
                        type=str, 
                        help="experiment name of the current experiment",
                        required=True)
    parser.add_argument("--resume", 
                        action="store_true", 
                        help="whether to resume from previous checkpoint")
    parser.add_argument("--resume_exp", 
                        type=str, 
                        help="experiment name to resume")
    parser.add_argument("--resume_epoch",
                        type=int, 
                        help="epoch to resume")
    args = parser.parse_args()
    cfg = get_cfg(args.cfg_path)
    train(args, cfg)



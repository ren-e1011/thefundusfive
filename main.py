
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# Revised by RNZ
# --------------------------------------------------------

import argparse
import os
from pathlib import Path

import torch

import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

# import timm
# assert timm.__version__ == "0.3.2" # version check version 1.0.7 conda install if problematic


from omegaconf import DictConfig, OmegaConf

from train import train_retfund_fives
from test import test_retfund_fives



# import util.lr_decay as lrd
# import util.misc as misc
# from util.datasets import build_dataset
# from util.pos_embed import interpolate_pos_embed
# from util.misc import NativeScalerWithGradNormCount as NativeScaler

# import models_vit

# from engine_finetune import train_one_epoch, evaluate

from dataset import DataSet


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--eval',type=bool,default=True)

    parser.add_argument('--data_path',type=str, default='/Users/renee/Documents/Projects/FundUs/FIVES A Fundus Image Dataset for AI-based Vessel Segmentation/')
    
    # parser.add_argument('--datatype', type=str, default='Original', choices=['Original', 'Ground truth'])

    # parser.add_argument('--traintype', type=str, default='finetune', choices=['finetune', 'freeze'] )

    return parser


def main(args,config: DictConfig):
    
    Training = not args.eval 

    subdir = 'train' if Training else 'test'
    data_path = os.path.join(args.data_path,subdir,config.data.type)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = DataSet(data_path=data_path, train=Training, data_params = config.data,device=device)
    batch_size = config.training.batch_size 

    if Training: 

        train_retfund_fives(dataset, data_params= config.data, training_params=config.training, device=device, k=config.training.folds)

       
    else: 
        test_retfund_fives(dataset, data_params= config.data, training_params=config.training, device=device, folds = config.training.folds)
    

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    conf = OmegaConf.load('./config.yaml')
    main(args,conf)



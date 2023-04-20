import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
from argparse import Namespace
from logging import Logger
import os
from typing import Tuple

import numpy as np

from chemprop.train.run_training import pre_training
from chemprop.parsing import parse_train_args, modify_train_args
from chemprop.torchlight import initialize_exp

def pretrain(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    pre_training(args, logger)



if __name__ == '__main__':
    args = parse_train_args()
    args.data_path = './data/zinc15_250K.csv'
    args.gpu = 0
    args.epochs = 50
    args.batch_size = 1024
    
    modify_train_args(args)
    logger, args.save_dir = initialize_exp(Namespace(**args.__dict__))
    pretrain(args, logger)

import os, sys, re, datetime, random, gzip, json, copy
from tqdm import tqdm
import pandas as pd
import numpy as np
from time import time
from math import ceil
from pathlib import Path
import itertools
import argparse
import networkx as nx
from stellargraph import StellarGraph

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

PROJ_PATH = Path(os.path.join(re.sub("/DyHNet.*$", '', os.getcwd()), 'DyHNet'))
sys.path.insert(1, str(PROJ_PATH / 'DyHNet' / 'src'))
import utils
from datasets import DyHNetDataModule
from model import DyHNet
from trainer import build_trainer
from inference import InferenceAgent
from prediction import eval_lp

def main():
    parser = argparse.ArgumentParser(description='Training.')
    parser.add_argument('-config_file', help='config file path', default=str(PROJ_PATH / 'src/config/fold_1.json'), type=str)
    args = parser.parse_args()
    print('Reading config file:', args.config_file)
    args.config = utils.read_json(args.config_file)
    seed_everything(args.config['seed'], workers=True)
    
    data_module = DyHNetDataModule(args.config)
    model_module = DyHNet(args.config)
    trainer, _ = build_trainer(args.config)

    # Train
    print('### Train')
    trainer.fit(model_module, data_module)
    
    # Test
    print('### Test')
    checkpoint_dir = Path(args.config['checkpoint_dir'])
    print(f'Load checkpoint from: {str(checkpoint_dir)}')
    paths = sorted(checkpoint_dir.glob('*.ckpt'))
    name = args.config['name']
    filtered_paths = [p for p in paths if f'model={name}-' in str(p)]
    results = []
    lp_results = []
    for i, p in enumerate(filtered_paths):
        print(f'Load model {i}: {p}')
        # test
        model_test = model_module.load_from_checkpoint(checkpoint_path=p) 
        result = trainer.test(model_test, datamodule=data_module)
        # link prediction
        args_inference = {
            'task': name,
            'restore_model_dir': str(PROJ_PATH / 'model'),
            'restore_model_name': p.name,
            'device': 'cuda:0',
        }
        agent = InferenceAgent(**args_inference)
        agent.inference()
        node_embedding = agent.output 
        lp_result, _ = eval_lp(name, node_embedding)
        print(lp_result.head())

        results.append(result)
        lp_results.append(lp_result)
        del model_test

    for p, r, lp in zip(filtered_paths, results, lp_results):
        print(p)
        print(lp)
        print('\n')

    del data_module
    del model_module
    del trainer
    
if __name__ == "__main__":
    main()
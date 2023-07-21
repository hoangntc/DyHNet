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
from sklearn.model_selection import ParameterGrid

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
from data_preparation import prepare_data, SnapshotGraph
from DyHNet import DyHNetPipeline

parser = argparse.ArgumentParser(prog='DyHNet', description='Dynamic Heterogeneous Network Representation Learning')
parser.add_argument('--name', type=str, default='yelp', help='dataset name', required=True)
args = parser.parse_args()
name = args.name
print(f'######### Experiment: {name} #########')

config_path = str(PROJ_PATH /  'DyHNet' / f'config/{name}.json')
config = utils.read_json(config_path)
config['data_fname'] = 'data.pkl'
config['checkpoint_dir'] = '../model_analysis/'

params = {
    'random_walk_len': [5, 10, 15, 20, 25],
    'n_anchor_patches_structure': [35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
    'subg_n_layers': [5], # [1, 2, 3, 4, 5]
    'max_size': [5, 7, 9, 11, 13, 15], # [5, 7, 9, 11, 13, 15]
    'subg_hidden_dim': [32, 64, 128, 256],
    'emb_dim': [16, 32, 64, 128, 256],
    'n_heads': [2, 4, 8, 16],
    }

for k, values in params.items():
    print(f'Run analysis for {k}:')
    for i, value in enumerate(values):
        print(f'- Run experiment {i}: {k} = {value}')
        config_dict = copy.deepcopy(config)
        config_dict[k] = value
        config_dict['name_suffix'] = f'sen_{k}={value}'
        print(config_dict)
        dyhnet = DyHNetPipeline(config_dict=config_dict)
        data_module, model_module, trainer = dyhnet.initialize()
        dyhnet.train(data_module, model_module, trainer)
            
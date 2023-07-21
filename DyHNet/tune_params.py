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

# Grid search
config = utils.read_json(config_path)

params = {
    'sample_walk_len': [10, 15, 20],
    'random_walk_len': [10, 15, 20],
    'n_triangular_walks': [10, 15],
    'n_anchor_patches_structure': [45, 75],
    'learning_rate': [0.001], # [0.0001, 0.005, 0.001],
    'subg_hidden_dim': [128],
    'subg_n_layers': [2, 3],
    }
param_grid = ParameterGrid(params)
for dict_ in param_grid:
    config_dict = copy.deepcopy(config)
    config_dict.update(dict_)
    print(config_dict)
    dyhnet = DyHNetPipeline(config_dict=config_dict)
    data_module, model_module, trainer = dyhnet.initialize()
    dyhnet.train(data_module, model_module, trainer)
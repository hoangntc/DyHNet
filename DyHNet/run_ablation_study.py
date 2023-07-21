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
if name == 'yelp' or name == 'dblp_four_area':
    config['data_fname'] = 'data_train=80.pkl'
    config['checkpoint_dir'] = '../model_80/'
else:
    config['data_fname'] = 'data.pkl'
    config['checkpoint_dir'] = '../model_lp/'

print('Run DyHNet:')
config_dict = copy.deepcopy(config)
print(config_dict)
dyhnet = DyHNetPipeline(config_dict=config_dict)
data_module, model_module, trainer = dyhnet.initialize()
dyhnet.train(data_module, model_module, trainer)

print('Run ablation study 1: Drop local information')
config_dict = copy.deepcopy(config)
config_dict['drop_local_info'] = True
config_dict['name_suffix'] = 'ablation=drop_local_info'
print(config_dict)
dyhnet = DyHNetPipeline(config_dict=config_dict)
data_module, model_module, trainer = dyhnet.initialize()
dyhnet.train(data_module, model_module, trainer)

print('Run ablation study 2: Drop global information')
config_dict = copy.deepcopy(config)
config_dict['drop_global_info'] = True
config_dict['name_suffix'] = 'ablation=drop_global_info'
print(config_dict)
dyhnet = DyHNetPipeline(config_dict=config_dict)
data_module, model_module, trainer = dyhnet.initialize()
dyhnet.train(data_module, model_module, trainer)

print('Run ablation study 3: Drop temporal information')
config_dict = copy.deepcopy(config)
config_dict['drop_temporal_info'] = True
config_dict['name_suffix'] = 'ablation=drop_temporal_info'
print(config_dict)
dyhnet = DyHNetPipeline(config_dict=config_dict)
data_module, model_module, trainer = dyhnet.initialize()
dyhnet.train(data_module, model_module, trainer)
    
print('Run ablation study 4: Drop local and global information')
config_dict = copy.deepcopy(config)
config_dict['drop_local_info'] = True
config_dict['drop_global_info'] = True
config_dict['name_suffix'] = 'ablation=drop_LG_info'
print(config_dict)
dyhnet = DyHNetPipeline(config_dict=config_dict)
data_module, model_module, trainer = dyhnet.initialize()
dyhnet.train(data_module, model_module, trainer)
    
print('Run ablation study 5: Drop global and temporal information')
config_dict = copy.deepcopy(config)
config_dict['drop_global_info'] = True
config_dict['drop_temporal_info'] = True
config_dict['name_suffix'] = 'ablation=drop_GT_info'
print(config_dict)
dyhnet = DyHNetPipeline(config_dict=config_dict)
data_module, model_module, trainer = dyhnet.initialize()
dyhnet.train(data_module, model_module, trainer)
    
print('Run ablation study 6: Drop local and temporal information')
config_dict = copy.deepcopy(config)
config_dict['drop_local_info'] = True
config_dict['drop_temporal_info'] = True
config_dict['name_suffix'] = 'ablation=drop_LT_info'
print(config_dict)
dyhnet = DyHNetPipeline(config_dict=config_dict)
data_module, model_module, trainer = dyhnet.initialize()
dyhnet.train(data_module, model_module, trainer)
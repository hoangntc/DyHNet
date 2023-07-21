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
sys.path.insert(1, str(PROJ_PATH / 'DyHNet'))
sys.path.insert(1, str(PROJ_PATH / 'DyHNet' / 'src'))
import utils
from datasets import DyHNetDataModule
from model import DyHNet
from trainer import build_trainer
from inference import InferenceAgent
from data_preparation import prepare_data, SnapshotGraph
from DyHNet import DyHNetPipeline
from prediction import eval_nc, eval_lp

parser = argparse.ArgumentParser(prog='DyHNet', description='Dynamic Heterogeneous Network Representation Learning')
parser.add_argument('--name', type=str, default='yelp', help='dataset name', required=True)
parser.add_argument('--data_fname', type=str, default='data.pkl', help='dataset file name', required=True)
parser.add_argument('--checkpoint_dir', type=str, default='model_analysis', help='checkpoint folder', required=True)
parser.add_argument('--checkpoint_fname', type=str, default='', help='checkpoint file name', required=True)
args = parser.parse_args()
name = args.name
print(f'######### Experiment: {name} #########')

# This is for evaluation including:
# 1. Initialize the main modules: data_module, model_module and trainer
# 2. Generate the graph embedding (for each node, generate t embeddings, t equals to number of timestamps)
# 3. Evaluate the results for two tasks: node classification or link prediction

device = 'cuda:0' 
save = False

config_path = str(PROJ_PATH /  'DyHNet' / f'config/{name}.json')
config = utils.read_json(config_path)
config_dict = copy.deepcopy(config)

restore_model_dir = str(PROJ_PATH / args.checkpoint_dir)
restore_model_name = args.checkpoint_fname
print(restore_model_name)

# 1. Initialize main modules
config_dict['data_fname'] = args.data_fname
# config_dict['name_suffix'] = f'sen_{k}={value}'
dyhnet = DyHNetPipeline(config_dict=config_dict)
data_module, model_module, trainer = dyhnet.initialize()

# 2. Generate the embedding
    agent = dyhnet.generate_embedding(
        data_module, model_module, restore_model_dir, restore_model_name, output_dir='', device=device, save=save)
    node_embeddings = agent.output

# 3. Evaluate tasks
## 3.1 Link prediction
if False:
    operators = ['HAD', 'AVG', 'L1', 'L2']
    pd_res_lp, models = eval_lp(name, node_embeddings, embed_path=None, operators=operators, threshold=0.1)
    print(pd_res_lp)
    pd_res_ranking = eval_lp_ranking(name, node_embeddings, embed_path=None, models=models, k=5)
    print(pd_res_ranking)

## 3.2 Node classification
if False:
    pd_res_nc, model = eval_nc(name, node_embeddings)
    print(pd_res_nc)


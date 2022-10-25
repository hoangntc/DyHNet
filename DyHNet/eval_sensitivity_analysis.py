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
from prediction import eval_nc, eval_lp, eval_lp_ranking

parser = argparse.ArgumentParser(prog='DyHNet', description='Dynamic Heterogeneous Network Representation Learning')
parser.add_argument('--name', type=str, default='yelp', help='dataset name', required=True)
parser.add_argument('--data_fname', type=str, default='data.pkl', help='dataset file name', required=False)
parser.add_argument('--checkpoint_dir', type=str, default='model_analysis', help='checkpoint folder', required=False)
parser.add_argument('--checkpoint_fname', type=str, default='', help='checkpoint file name', required=False)
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
paths = sorted(Path(restore_model_dir).glob('*.ckpt'))

operators = ['HAD']
list_k = [1, 2]

params = {
    'random_walk_len': [5, 10, 15, 20, 25],
    'n_anchor_patches_structure': [35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
    'subg_n_layers': [2, 3],
    'max_size': [5, 7, 9],
    # 'subg_hidden_dim': [16, 32, 64, 128, 256],
    # 'emb_dim': [16, 32, 64, 128, 256],
    # 'n_heads': [2, 4, 8, 16],
    }
list_results = []
for param, values in params.items():
    print(f'Evaluate sensitivity analysis for {param}:')
    for i, value in enumerate(values):
        print(f'- Result {i}: {param} = {value}')
        config_dict = copy.deepcopy(config)
        config_dict[param] = value
        config_dict['name_suffix'] = f'sen_{param}={value}'
        print(config_dict)
        checkpoint_paths = [p for p in paths if f'model={name}_sen_{param}={value}-' in str(p)]

        for p in checkpoint_paths:
            # 1. Initialize main modules
            config_dict['data_fname'] = args.data_fname
            # config_dict['name_suffix'] = f'sen_{k}={value}'
            dyhnet = DyHNetPipeline(config_dict=config_dict)
            data_module, model_module, trainer = dyhnet.initialize()

            # 2. Generate the embedding
            agent = dyhnet.generate_embedding(
                data_module, model_module, restore_model_dir, restore_model_name=p, output_dir='', device=device, save=save)
            node_embeddings = agent.output

            # 3. Evaluate task
            ## 3.1 Link prediction
            pd_res_lp, models = eval_lp(name, node_embeddings, embed_path=None, operators=operators, threshold=0.1)
            pd_res_ranking = eval_lp_ranking(name, node_embeddings, embed_path=None, models=models, k=5)

            # 4. Generate result table and append to list (to save to file)
            auc = pd_res_lp['HAD_auc']['test']
            pd_result = pd_res_ranking.copy()
            pd_result['method'] = 'DyHNet'
            pd_result = pd_result[pd_result['k'].isin(list_k)][['method', 'k', 'micro_f1', 'micro_recall', 'micro_precision']].pivot_table(
                index='method', columns='k', values=['micro_f1', 'micro_recall', 'micro_precision'])
            pd_result['AUC'] = auc    
            pd_result['param'] = param
            pd_result['value'] = value
            pd_result['path'] = p
            cnames =[
                ('AUC', ''), ('micro_f1', 1), ('micro_recall', 1), ('micro_precision', 1), 
                ('micro_f1', 2), ('micro_recall', 2), ('micro_precision', 2), ('param', ''), ('value', ''), ('path', '')]
            list_results.append(pd_result[cnames])

pd_report = pd.concat(list_results)
pd_report.to_csv(f'./report_lp_{name}.csv', index=False)
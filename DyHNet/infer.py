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

# This is for generate the graph embedding once the model is trained and saved:
# 1. Initialize the main modules: data_module, model_module and trainer
# 2. Generate the graph embedding with the checkpoint

dyhnet = DyHNetPipeline(config_path=config_path)

# 1. Initialize data, model, trainer
data_module, model_module, trainer = dyhnet.initialize()

# 2. Infer
restore_model_dir = str(dyhnet.config['checkpoint_dir'])
restore_model_name = str(checkpoint_paths[-1].name)
output_dir = str(PROJ_PATH / 'output')
dyhnet.generate_embedding(data_module, model_module, restore_model_dir, restore_model_name, output_dir)


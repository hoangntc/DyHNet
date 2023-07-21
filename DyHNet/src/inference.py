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

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything

PROJ_PATH = Path(os.path.join(re.sub("/DyHNet.*$", '', os.getcwd()), 'DyHNet'))
sys.path.insert(1, str(PROJ_PATH / 'DyHNet' / 'src'))
import utils
from datasets import DyHNetDataModule
from model import DyHNet
from trainer import build_trainer

class InferenceAgent:
    def __init__(self,
                 config,
                 data_module,
                 model_module,
                 restore_model_dir=str(PROJ_PATH / 'model'),
                 restore_model_name='.ckpt',
                 output_dir=str(PROJ_PATH / 'output'),
                 output_fname='',
                 device='cuda:0',
                ):
        
       
        self.restore_model_dir = restore_model_dir
        self.restore_model_name = restore_model_name
        self.output_dir = output_dir
        self.output_fname = output_fname

        # initial data/model
        seed_everything(config['seed'], workers=True)
        self.data_module = data_module
        self.data_module.setup()
        self.model_module = model_module
        self.device = device
        
        # load checkpoint
        map_location = lambda storage, loc: storage.cuda()
        checkpoint_path = Path(self.restore_model_dir)/self.restore_model_name
        print(f'Load checkpoint from: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model_dict = self.model_module.state_dict()
        pretrain_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
        self.model_module.load_state_dict(pretrain_dict)
        self.model_module.eval()
        self.model_module.to(self.device)
        
    def single_dataset_inference(self, loader):
        time_ids = []
        embeddings = []
        node_ids = []
        
        for batch in loader:
            # to device
            temporal_initial_embed = batch['temporal_initial_embed'].to(self.device)
            temporal_subgraph_idx = batch['temporal_subgraph_idx'].to(self.device)
            temporal_subgraph_mask = batch['temporal_subgraph_mask'].to(self.device)
            temporal_cc_ids = [[d.to(self.device) for d in b] for b in batch['temporal_cc_ids']]
            temporal_I_S_sim = [[d.to(self.device) for d in b] for b in batch['temporal_I_S_sim']]
            temporal_B_S_sim = [[d.to(self.device) for d in b] for b in batch['temporal_B_S_sim']]
            temporal_S_I_cc_embed = [[d.to(self.device) for d in b] for b in batch['temporal_S_I_cc_embed']]
            temporal_S_B_cc_embed = [[d.to(self.device) for d in b] for b in batch['temporal_S_B_cc_embed']]
            
            # forward
            node_embedding = self.model_module.forward_embedding(
                node_id=batch['node_id'],
                temporal_initial_embed=temporal_initial_embed,
                temporal_subgraph_idx=temporal_subgraph_idx,
                temporal_subgraph_mask=temporal_subgraph_mask,
                temporal_cc_ids=temporal_cc_ids,
                temporal_I_S_sim=temporal_I_S_sim,
                temporal_B_S_sim=temporal_B_S_sim,
                temporal_S_I_cc_embed=temporal_S_I_cc_embed,
                temporal_S_B_cc_embed=temporal_S_B_cc_embed,
                temporal_mask=batch['temporal_mask'].to(self.device),
            )    
            embed = node_embedding.cpu().detach().numpy().tolist()
            node_id = batch['node_id'].cpu().detach().numpy().tolist()
            time_id = batch['time_id'].cpu().detach().numpy().tolist()
            time_ids += time_id
            embeddings += embed
            node_ids += node_id 

        print(f'Number of samples: {len(embeddings)}')
        return time_ids, node_ids, embeddings
    
    def inference(self):
        tids = []
        nids = []
        embs = []
        loaders = [
                self.data_module.mytrain_dataloader(), 
                self.data_module.val_dataloader(), 
                self.data_module.test_dataloader()]
        if len(self.data_module.data_full.inference_idx) > 0:
            loaders.append(self.data_module.inference_dataloader())
            
        with torch.no_grad():
            for loader in loaders:
                time_ids, node_ids, embeddings = self.single_dataset_inference(loader)
                tids += time_ids
                nids += node_ids
                embs += embeddings
        print(f'Total number of samples: {len(nids)}')
        self.output = {}
        for t, n, e in zip(tids, nids, embs):
            if t in self.output:
                self.output[t][n] = e
            else:
                self.output[t] = {n: e}
        
    def save_output(self):
        if self.output_dir != '':
            if not os.path.exists(self.output_dir): os.mkdir(self.output_dir)
            if self.output_fname == '':
                self.output_fname = os.path.basename(str(self.restore_model_name)).replace('.ckpt', '.pkl')
            save_path = str(Path(self.output_dir) / self.output_fname)
            print(f'Save embeddings to: {save_path}')
            pd.to_pickle(self.output, save_path)
import os, sys, re, datetime, random, gzip, json, copy
import tqdm
import pandas as pd
import numpy as np
from time import time
from math import ceil
from pathlib import Path
import itertools
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

PROJ_PATH = Path(os.path.join(re.sub("/DyHNet.*$", '', os.getcwd()), 'DyHNet'))
sys.path.insert(1, str(PROJ_PATH / 'DyHNet' / 'src'))
import utils
from layer import SubgEncoder, StructuralEncoder, TemporalEncoder, CrossEntropyLossList
from datasets import DyHNetDataModule, GraphSnapshot

class DyHNet(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        self.subg_encoder = SubgEncoder(
            self.hparams.node_embed_size, 
            self.hparams.n_anchor_patches_structure, 
            self.hparams.subg_n_layers, 
            self.hparams.subg_hidden_dim, 
            self.hparams.dropout_prob, 
            self.hparams.lstm_n_layers, 
            self.hparams.lstm_aggregator, 
            hparams=self.hparams)
        self.subg_lin = nn.Linear(self.hparams.subg_hidden_dim, self.hparams.hidden_dim)
        self.structural_encoder = StructuralEncoder(input_dim=self.hparams.hidden_dim)
        self.temporal_encoder = TemporalEncoder(
            input_dim=self.hparams.hidden_dim+self.hparams.node_embed_size, 
            n_heads=self.hparams.n_heads, 
            num_time_steps=self.hparams.num_time_steps, 
            dropout_prob=self.hparams.dropout_prob, 
            residual=True)
        
        
        self.lin = nn.Linear(self.hparams.hidden_dim+self.hparams.node_embed_size, self.hparams.emb_dim)
        self.bn = nn.BatchNorm1d(num_features=self.hparams.emb_dim)
        self.dropout = nn.Dropout(self.hparams.dropout_prob)
        self.lin1 = nn.Linear(self.hparams.emb_dim, self.hparams.emb_dim)
        self.dropout1 = nn.Dropout(self.hparams.dropout_prob)
        self.classifier = nn.Linear(self.hparams.emb_dim, self.hparams.num_labels)
        
        if self.hparams.multilabel:
            if self.hparams.loss == 'KL':
                self.loss = nn.KLDivLoss(reduction='batchmean')
            elif self.hparams.loss == 'CE':
                # weight = torch.tensor([4, 2, 1, 7])
                self.loss = CrossEntropyLossList()
            elif self.hparams.loss == 'BCE':
                self.loss = nn.BCEWithLogitsLoss()
            elif self.hparams.loss == 'MULTI':
                self.loss_KL = nn.KLDivLoss(reduction='batchmean')
                self.loss_BCE = nn.BCEWithLogitsLoss()
                self.alpha = self.hparams.alpha
                self.beta = self.hparams.beta
        else:
            self.loss = nn.CrossEntropyLoss()
        self.read_snapshots()
        
    def read_snapshots(self):
        self.node_embedding = []
        self.anchors_structure = []
        for i in range(self.hparams.num_time_steps):
            snapshot_path = PROJ_PATH / 'dataset' / self.hparams.name / 't_{:02d}'.format(i)
            print(f'### Processing {snapshot_path}')
            ss = GraphSnapshot(
                graph_path=snapshot_path / 'edge_list.txt', 
                node_path=snapshot_path / 'node_types.csv',
                subgraph_path=snapshot_path / 'subgraphs.pth',
                embedding_path=snapshot_path / 'gin_gcn_embeddings.pth',
                similarities_path=snapshot_path / 'similarities',
                degree_dict_path=snapshot_path / 'degree_sequence.txt',
                params=self.hparams,
            )
            self.node_embedding.append(ss.node_embeddings)
            self.anchors_structure.append(ss.anchors_structure)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer  

    def run_subg(
        self, 
        node_embedding, 
        anchors_structure,
        temporal_S_I_cc_embed,
        temporal_S_B_cc_embed,
        temporal_cc_ids,
        temporal_subgraph_idx,
        temporal_subgraph_mask,
        temporal_I_S_sim,
        temporal_B_S_sim,
        batch_size,
    ):
        snapshots = []
        for time_id in range(self.hparams.num_time_steps):
            S_I_cc_embed = torch.cat([cc_embed[time_id] for cc_embed in temporal_S_I_cc_embed])
            S_B_cc_embed = torch.cat([cc_embed[time_id] for cc_embed in temporal_S_B_cc_embed])
            cc_ids = torch.cat([cc_id[time_id] for cc_id in temporal_cc_ids])
            I_S_sim = torch.cat([sim[time_id] for sim in temporal_I_S_sim])
            B_S_sim = torch.cat([sim[time_id] for sim in temporal_B_S_sim])

            # Subgraph Embedding
            subgs_embed = self.subg_encoder.forward(
                node_embedding[time_id].to(self.device),
                anchors_structure[time_id],
                S_I_cc_embed=S_I_cc_embed,
                S_B_cc_embed=S_B_cc_embed,
                cc_ids=cc_ids, 
                subgraph_idx='', 
                I_S_sim=I_S_sim, 
                B_S_sim=B_S_sim)
            
            # (batch_size, max_size, subg_hidden_dim)
            subgs_embed = subgs_embed.reshape(batch_size, self.hparams.max_size, subgs_embed.shape[1]) 
            subgs_embed = self.subg_lin(subgs_embed)
            subgs_mask = torch.repeat_interleave(temporal_subgraph_mask[:, time_id, :], subgs_embed.shape[2]).reshape(subgs_embed.shape)
            subgs_embed = subgs_embed.masked_fill((subgs_mask==1), 0.0)
            # Structural Encoder
            # (batch_size, hidden_dim)
            snapshot = self.structural_encoder(subgs_embed)
            snapshots.append(snapshot)

        # (batch_size, num_time_steps, hidden_dim)
        structural_embed = torch.cat(snapshots, dim=1).reshape(batch_size, self.hparams.num_time_steps, snapshot.shape[1])
        return structural_embed
    
    def forward(
        self, 
        node_id,
        temporal_initial_embed,
        temporal_subgraph_idx,
        temporal_subgraph_mask,
        temporal_cc_ids,
        temporal_I_S_sim,
        temporal_B_S_sim,
        temporal_S_I_cc_embed,
        temporal_S_B_cc_embed,
        temporal_mask,
    ):
        batch_size = temporal_initial_embed.shape[0]

        # Initial Node Embedding: (batch_size, num_time_steps, hidden)
        if 'drop_local_info' in self.hparams and self.hparams.drop_local_info:
            initial_embed = torch.zeros(temporal_initial_embed.size()).to(self.device)
        else:
            initial_embed = temporal_initial_embed

        # Structural Encoding: (batch_size, num_time_steps, hidden_dim)
        if 'drop_global_info' in self.hparams and self.hparams.drop_global_info:
            structural_embed = torch.zeros((batch_size, self.hparams.num_time_steps, self.hparams.hidden_dim)).to(self.device)
        else:
            structural_embed = self.run_subg(
            node_embedding=self.node_embedding, 
            anchors_structure=self.anchors_structure,
            temporal_S_I_cc_embed=temporal_S_I_cc_embed,
            temporal_S_B_cc_embed=temporal_S_B_cc_embed,
            temporal_cc_ids=temporal_cc_ids,
            temporal_subgraph_idx=temporal_subgraph_idx,
            temporal_subgraph_mask=temporal_subgraph_mask,
            temporal_I_S_sim=temporal_I_S_sim,
            temporal_B_S_sim=temporal_B_S_sim,
            batch_size=batch_size,
        )
        # Information Fusion
        snapshots_embed = torch.cat([structural_embed, initial_embed], dim=2)

        # Temporal Encoding
        if 'drop_temporal_info' in self.hparams and self.hparams.drop_temporal_info:
            outputs = snapshots_embed    
        else:
            outputs = self.temporal_encoder(snapshots_embed)
 
        time_mask = torch.repeat_interleave(temporal_mask, self.hparams.hidden_dim+self.hparams.node_embed_size, dim=-1).reshape((batch_size, self.hparams.num_time_steps, self.hparams.hidden_dim+self.hparams.node_embed_size)).to(self.device)
        h = utils.masked_sum(vector=outputs, mask=(time_mask==1), dim=1)
        del time_mask
        
        # FF
        # h = self.lin(h)
        # h = self.bn(h)
        h = torch.tanh(self.lin(h))
        h = self.dropout(h)
        h = torch.tanh(self.lin1(h))
        node_embed = self.dropout1(h)
        logits = self.classifier(node_embed)
        return logits
        
    def training_step(self, batch, batch_idx):
        logits = self.forward(
            node_id=batch['node_id'],
            temporal_initial_embed=batch['temporal_initial_embed'],
            temporal_subgraph_idx=batch['temporal_subgraph_idx'],
            temporal_subgraph_mask=batch['temporal_subgraph_mask'],
            temporal_cc_ids=batch['temporal_cc_ids'],
            temporal_I_S_sim=batch['temporal_I_S_sim'],
            temporal_B_S_sim=batch['temporal_B_S_sim'],
            temporal_S_I_cc_embed=batch['temporal_S_I_cc_embed'],
            temporal_S_B_cc_embed=batch['temporal_S_B_cc_embed'],
            temporal_mask=batch['temporal_mask'],
        )
        labels = batch['labels']
        if self.hparams.multilabel:
            if self.hparams.loss == 'KL':
                # logits = torch.clip(logits, 1e-7)
                loss = self.loss(F.log_softmax(logits.squeeze(1), -1), labels) # KL
            elif self.hparams.loss == 'CE':
                loss = self.loss(logits.squeeze(1), labels)
            elif self.hparams.loss == 'BCE':
                loss = self.loss(logits.squeeze(1), labels)
            elif self.hparams.loss == 'MULTI':
                loss_KL = self.loss_KL(F.log_softmax(logits.squeeze(1), -1), labels) # KL
                loss_BCE = self.loss_BCE(logits.squeeze(1), labels)
                loss = self.alpha * loss_KL + self.beta * loss_BCE
        else:
            loss = self.loss(logits.squeeze(1), labels)
        self.log('train_loss', loss, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self.forward(
            node_id=batch['node_id'],
            temporal_initial_embed=batch['temporal_initial_embed'],
            temporal_subgraph_idx=batch['temporal_subgraph_idx'],
            temporal_subgraph_mask=batch['temporal_subgraph_mask'],
            temporal_cc_ids=batch['temporal_cc_ids'],
            temporal_I_S_sim=batch['temporal_I_S_sim'],
            temporal_B_S_sim=batch['temporal_B_S_sim'],
            temporal_S_I_cc_embed=batch['temporal_S_I_cc_embed'],
            temporal_S_B_cc_embed=batch['temporal_S_B_cc_embed'],
            temporal_mask=batch['temporal_mask'],
        )
        labels = batch['labels']
        if self.hparams.multilabel:
            if self.hparams.loss == 'KL':
                # logits = torch.clip(logits, 1e-7)
                loss = self.loss(F.log_softmax(logits.squeeze(1), -1), labels) # KL
            elif self.hparams.loss == 'CE':
                loss = self.loss(logits.squeeze(1), labels)
            elif self.hparams.loss == 'BCE':
                loss = self.loss(logits.squeeze(1), labels)
            elif self.hparams.loss == 'MULTI':
                loss_KL = self.loss_KL(F.log_softmax(logits.squeeze(1), -1), labels) # KL
                loss_BCE = self.loss_BCE(logits.squeeze(1), labels)
                loss = self.alpha * loss_KL + self.beta * loss_BCE
        else:
            loss = self.loss(logits.squeeze(1), labels)
        acc = utils.calc_accuracy(logits, labels, multilabel=self.hparams.multilabel).squeeze()
        hamming_loss = utils.calc_hamming_loss(logits, labels, multilabel=self.hparams.multilabel).squeeze()
        macro_f1 = utils.calc_f1(logits, labels, avg_type='macro', multilabel=self.hparams.multilabel).squeeze()
        micro_f1 = utils.calc_f1(logits, labels, avg_type='micro', multilabel=self.hparams.multilabel).squeeze()

        logs = {
            'loss': loss, 
            'acc': acc,
            'hamming_loss': hamming_loss,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
        }
        # self.log_dict(logs, prog_bar=True)
        return logs
    
    def validation_epoch_end(self, val_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in val_step_outputs]).mean().cpu()
        avg_acc = torch.stack([x['acc'] for x in val_step_outputs]).mean().cpu()
        avg_hamming_loss = torch.stack([x['hamming_loss'] for x in val_step_outputs]).mean().cpu()
        avg_macro_f1 = torch.stack([x['macro_f1'] for x in val_step_outputs]).mean().cpu()
        avg_micro_f1 = torch.stack([x['micro_f1'] for x in val_step_outputs]).mean().cpu()
        logs = {
            'val_loss': avg_loss, 
            'val_acc': avg_acc,
            'val_hamming_loss': avg_hamming_loss,
            'val_macro_f1': avg_macro_f1,
            'val_micro_f1': avg_micro_f1,
        }
        self.log_dict(logs, prog_bar=True)
     
    def test_step(self, batch, batch_idx):
        logits = self.forward(
            node_id=batch['node_id'],
            temporal_initial_embed=batch['temporal_initial_embed'],
            temporal_subgraph_idx=batch['temporal_subgraph_idx'],
            temporal_subgraph_mask=batch['temporal_subgraph_mask'],
            temporal_cc_ids=batch['temporal_cc_ids'],
            temporal_I_S_sim=batch['temporal_I_S_sim'],
            temporal_B_S_sim=batch['temporal_B_S_sim'],
            temporal_S_I_cc_embed=batch['temporal_S_I_cc_embed'],
            temporal_S_B_cc_embed=batch['temporal_S_B_cc_embed'],
            temporal_mask=batch['temporal_mask'],
        )
        labels = batch['labels']
        if self.hparams.multilabel:
            if self.hparams.loss == 'KL':
                # logits = torch.clip(logits, 1e-7)
                loss = self.loss(F.log_softmax(logits.squeeze(1), -1), labels) # KL
            elif self.hparams.loss == 'CE':
                loss = self.loss(logits.squeeze(1), labels)
            elif self.hparams.loss == 'BCE':
                loss = self.loss(logits.squeeze(1), labels)
            elif self.hparams.loss == 'MULTI':
                loss_KL = self.loss_KL(F.log_softmax(logits.squeeze(1), -1), labels) # KL
                loss_BCE = self.loss_BCE(logits.squeeze(1), labels)
                loss = self.alpha * loss_KL + self.beta * loss_BCE

        else: 
            loss = self.loss(logits.squeeze(1), labels)
        acc = utils.calc_accuracy(logits, labels, multilabel=self.hparams.multilabel).squeeze()
        hamming_loss = utils.calc_hamming_loss(logits, labels, multilabel=self.hparams.multilabel).squeeze()
        macro_f1 = utils.calc_f1(logits, labels, avg_type='macro', multilabel=self.hparams.multilabel).squeeze()
        micro_f1 = utils.calc_f1(logits, labels, avg_type='micro', multilabel=self.hparams.multilabel).squeeze()

        logs = {
            'loss': loss, 
            'acc': acc,
            'hamming_loss': hamming_loss,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1
        }
        return logs
    
    def test_epoch_end(self, test_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in test_step_outputs]).mean().cpu()
        avg_acc = torch.stack([x['acc'] for x in test_step_outputs]).mean().cpu()
        avg_hamming_loss = torch.stack([x['hamming_loss'] for x in test_step_outputs]).mean().cpu()
        avg_macro_f1 = torch.stack([x['macro_f1'] for x in test_step_outputs]).mean().cpu()
        avg_micro_f1 = torch.stack([x['micro_f1'] for x in test_step_outputs]).mean().cpu()
        logs = {
            'test_loss': avg_loss, 
            'test_acc': avg_acc,
            'test_hamming_loss': avg_hamming_loss,
            'test_macro_f1': avg_macro_f1,
            'test_micro_f1': avg_micro_f1,
        }
        self.log_dict(logs, prog_bar=True)
        return logs

    def forward_embedding(
        self, 
        node_id,
        temporal_initial_embed,
        temporal_subgraph_idx,
        temporal_subgraph_mask,
        temporal_cc_ids,
        temporal_I_S_sim,
        temporal_B_S_sim,
        temporal_S_I_cc_embed,
        temporal_S_B_cc_embed,
        temporal_mask,
    ):
        batch_size = temporal_initial_embed.shape[0]

        # Initial Node Embedding: (batch_size, num_time_steps, hidden)
        if 'drop_local_info' in self.hparams and self.hparams.drop_local_info:
            initial_embed = torch.zeros(temporal_initial_embed.size()).to(self.device)
        else:
            initial_embed = temporal_initial_embed

        # Structural Encoding: (batch_size, num_time_steps, hidden_dim)
        if 'drop_global_info' in self.hparams and self.hparams.drop_global_info:
            structural_embed = torch.zeros((batch_size, self.hparams.num_time_steps, self.hparams.hidden_dim)).to(self.device)
        else:
            structural_embed = self.run_subg(
                node_embedding=self.node_embedding, 
                anchors_structure=self.anchors_structure,
                temporal_S_I_cc_embed=temporal_S_I_cc_embed,
                temporal_S_B_cc_embed=temporal_S_B_cc_embed,
                temporal_cc_ids=temporal_cc_ids,
                temporal_subgraph_idx=temporal_subgraph_idx,
                temporal_subgraph_mask=temporal_subgraph_mask,
                temporal_I_S_sim=temporal_I_S_sim,
                temporal_B_S_sim=temporal_B_S_sim,
                batch_size=batch_size,
            )
        
        # Information Fusion
        snapshots_embed = torch.cat([structural_embed, initial_embed], dim=2)

        # Temporal Encoding
        if 'drop_temporal_info' in self.hparams and self.hparams.drop_temporal_info:
            outputs = snapshots_embed    
        else:
            outputs = self.temporal_encoder(snapshots_embed)
 
        time_mask = torch.repeat_interleave(temporal_mask, self.hparams.hidden_dim+self.hparams.node_embed_size, dim=-1).reshape((batch_size, self.hparams.num_time_steps, self.hparams.hidden_dim+self.hparams.node_embed_size)).to(self.device)
        h = utils.masked_sum(vector=outputs, mask=(time_mask==1), dim=1)
        del time_mask

        # FF
        h = torch.tanh(self.lin(h))
        h = self.dropout(h)
        h = torch.tanh(self.lin1(h))
        node_embed = self.dropout1(h)
        return node_embed
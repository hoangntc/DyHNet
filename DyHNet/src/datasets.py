import os, sys, re, datetime, random, gzip, json, copy
from tqdm.autonotebook import tqdm
import pandas as pd
import numpy as np
from time import time
from math import ceil
from pathlib import Path
import multiprocessing
from multiprocessing import Pool
import itertools
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, Subset

import pytorch_lightning as pl
import networkx as nx
from stellargraph import StellarGraph

PROJ_PATH = Path(os.path.join(re.sub("/DyHNet.*$", '', os.getcwd()), 'DyHNet'))
sys.path.insert(1, str(PROJ_PATH / 'DyHNet' / 'src'))

from anchor_patch_samplers import *
import gamma
from sklearn.preprocessing import MultiLabelBinarizer

PAD_VALUE = 0

class GraphSnapshot:
    def __init__(
        self, 
        graph_path, 
        node_path,
        subgraph_path,
        embedding_path,
        similarities_path,
        degree_dict_path,
        params={},
       ):
        default_params = {
            'sample_walk_len': 20, # sampling anchor nodes
            'random_walk_len': 20, # sampling neighbors of anchor nodes
            'structure_patch_type': 'triangular_random_walk',
            'max_sim_epochs': 5,
            'n_anchor_patches_structure': 45,
            'subg_n_layers': 2,
            'n_triangular_walks': 10,
            'n_processes': 4,
            'meta_paths': '0-1-0-1-0-1-0-1-0-1-0-1-0-1-0 \
            1-0-1-0-1-0-1-0-1-0-1-0-1-0-1 \
            1-2-1-2-1-2-1-2-1-2-1-2-1-2-1 \
            1-3-1-3-1-3-1-3-1-3-1-3-1-3-1 \
            2-1-2-1-2-1-2-1-2-1-2-1-2-1-2 \
            0-1-2-1-0-1-2-1-0-1-2-1-0-1-2-1-0',
        }
        default_params.update(params)
        self.hparams = copy.deepcopy(default_params)
        self.graph_path = graph_path
        self.node_path = node_path
        self.subgraph_path = subgraph_path
        self.embedding_path = embedding_path
        self.similarities_path = similarities_path
        self.degree_dict_path = degree_dict_path
        self.meta_paths = [p.strip().split('-') for p in self.hparams['meta_paths'].split(' ')]
        self.prepare_data()
        print('\n')
        
    def read_subgraphs(self, sub_f, split=True):
        '''
        Read all subgraphs from file
        '''
        sub_G = []
        with open(sub_f) as fin:
            for line in fin:
                nodes = [int(n) for n in line.split("\t")[0].split("-") if n != ""]
                if len(nodes) > 0:
                    sub_G.append(nodes)
        print(f'Number of subgraphs: {len(sub_G)}')
        return sub_G
    
    def reindex_data(self, data):
        '''
        Relabel node indices in the train/val/test sets to be 1-indexed instead of 0 indexed
        so that we can use 0 for padding
        '''
        new_subg = []
        for subg in data:
            new_subg.append([c + 1 for c in subg])
        return new_subg
    
    def read_data(self):
        print('--- Reading subgraphs ---')
        self.networkx_graph = nx.read_edgelist(str(self.graph_path))
        self.node_type = pd.read_csv(str(self.node_path))
        all_nodes = self.node_type['node_id'].astype(type(list(self.networkx_graph.nodes)[0])).values
        self.networkx_graph.add_nodes_from(all_nodes)
        self.sub_G = self.read_subgraphs(str(self.subgraph_path))

        # renumber nodes to start with index 1 instead of 0
        mapping = {n:int(n)+1 for n in self.networkx_graph.nodes()}
        self.networkx_graph = nx.relabel_nodes(self.networkx_graph, mapping)
        self.sub_G = self.reindex_data(self.sub_G)
        self.node_type['node_id'] = self.node_type['node_id'] + 1

        # get node type
        self.node_type['node_type'] = self.node_type['node_type'].astype(str)
        self.nodeid_to_nodetype = self.node_type.set_index('node_id').to_dict()['node_type']
        nx.set_node_attributes(self.networkx_graph, self.nodeid_to_nodetype, name='label')
        self.stellar_graph = StellarGraph.from_networkx(self.networkx_graph)

        # Initialize pretrained node embeddings
        self.pretrained_node_embeds = torch.load(str(self.embedding_path)) # feature matrix should be initialized to the node embeddings
        node_embed_size = self.pretrained_node_embeds.shape[1]
        zeros = torch.zeros(1, self.pretrained_node_embeds.shape[1])
        embeds = torch.cat((zeros, self.pretrained_node_embeds), 0) #there's a zeros in the first index for padding

        # optionally freeze the node embeddings
        self.node_embeddings = nn.Embedding.from_pretrained(embeds, freeze=True, padding_idx=PAD_VALUE)
        
    def initialize_cc_ids(self, subgraph_ids):
        '''
        Initialize the 3D matrix of (n_subgraphs X max number of cc X max length of cc)
        Input:
            - subgraph_ids: list of subgraphs where each subgraph is a list of node ids 
        Output:
            - reshaped_cc_ids_pad: padded tensor of shape (n_subgraphs, max_n_cc, max_len_cc)
        '''
        n_subgraphs = len(subgraph_ids) # number of subgraphs

        # Create connected component ID list from subgraphs
        cc_id_list = []
        for curr_subgraph_ids in subgraph_ids:
            subgraph = nx.subgraph(self.networkx_graph, curr_subgraph_ids) #networkx version of subgraph
            con_components = list(nx.connected_components(subgraph)) # get connected components in subgraph
            cc_id_list.append([torch.LongTensor(list(cc_ids)) for cc_ids in con_components])

        # pad number of connected components
        max_n_cc = max([len(cc) for cc in cc_id_list]) #max number of cc across all subgraphs
        for cc_list in cc_id_list:
            while True:
                if len(cc_list) == max_n_cc: break
                cc_list.append(torch.LongTensor([PAD_VALUE]))

        # pad number of nodes in connected components
        all_pad_cc_ids = [cc for cc_list in cc_id_list for cc in cc_list]
        assert len(all_pad_cc_ids) % max_n_cc == 0
        con_component_ids_pad = pad_sequence(all_pad_cc_ids, batch_first=True, padding_value=PAD_VALUE) # (batch_sz * max_n_cc, max_cc_len)
        reshaped_cc_ids_pad = con_component_ids_pad.view(n_subgraphs, max_n_cc, -1) # (batch_sz, max_n_cc, max_cc_len)

        return reshaped_cc_ids_pad # (n_subgraphs, max_n_cc, max_len_cc)
    
    def initialize_cc_embeddings(self, cc_id_list, aggregator='sum'):
        '''
        Initialize connected component embeddings as either the sum or max of node embeddings in the connected component
        Input:
            - cc_id_list: 3D tensor of shape (n subgraphs, max n CC, max length CC)
        Output:
            - 3D tensor of shape (n_subgraphs, max n_cc, node embedding dim)
        '''
        if aggregator == 'sum':
            return torch.sum(self.node_embeddings(cc_id_list), dim=2)
        elif aggregator == 'max':
            return torch.max(self.node_embeddings(cc_id_list), dim=2)[0]
        
    def initialize_channel_embeddings(self, cc_embeddings, trainable=False):
        '''
        Initialize CC embeddings for each channel (N, S, P X internal, border)
        '''

        if trainable: # if the embeddings are trainable, make them a parameter
            S_I_cc_embeds = Parameter(cc_embeddings.detach().clone())
            S_B_cc_embeds = Parameter(cc_embeddings.detach().clone())
        else:
            S_I_cc_embeds = cc_embeddings
            S_B_cc_embeds = cc_embeddings
            
        return S_I_cc_embeds, S_B_cc_embeds
    
    def init_all_embeddings(self, trainable=False):
        '''
        Initialize the CC and channel-specific CC embeddings for the subgraphs in the specified split
        '''
        # initialize CC embeddings
        cc_embeddings = self.initialize_cc_embeddings(self.cc_ids)
        # initialize  CC embeddings for each channel
        self.S_I_cc_embed, self.S_B_cc_embed = self.initialize_channel_embeddings(cc_embeddings, trainable)

    def compute_structure_patch_similarities(self, degree_dict, fname, internal, cc_ids, sim_path):
        '''
        Calculate the similarity between the sampled anchor patches and the connected components
        The default structure similarity function is DTW over the patch and component degree sequences.
        Returns tensor of similarities of shape (n_subgraphs, max_n_cc, n anchor patches)
        '''
        n_anchors = self.structure_anchors.shape[0]
        n_subgraphs, max_n_cc, _ = cc_ids.shape
        cc_id_mask = (cc_ids[:,:,0] != PAD_VALUE)
        
        # store the degree sequence for each anchor patch into a dict
        anchor_degree_seq_dict = {}
        for a, anchor_patch in enumerate(self.structure_anchors):
            anchor_degree_seq_dict[a] = gamma.get_degree_sequence(self.networkx_graph, anchor_patch, degree_dict, internal=internal)

        # store the degree sequence for each connected component into a dict
        component_degree_seq_dict = {}
        cc_ids_reshaped = cc_ids.view(n_subgraphs*max_n_cc, -1)
        for c, component in enumerate(cc_ids_reshaped):
            component_degree_seq_dict[c] = gamma.get_degree_sequence(self.networkx_graph, component, degree_dict, internal=internal)

        # to use multiprocessing to calculate the similarity, we first create a list of all of the inputs
        inputs = []
        for c in range(len(cc_ids_reshaped)):
            for a in range(len(self.structure_anchors)):
                inputs.append((component_degree_seq_dict[c], anchor_degree_seq_dict[a]))

        # use starmap to calculate DTW between the anchor patches & connected components' degree sequences
        with multiprocessing.Pool(processes=self.hparams['n_processes']) as pool: 
            sims = pool.starmap(gamma.calc_dtw, inputs)

        # reshape similarities to a matrix of shape (n_subgraphs, max_n_cc, n anchor patches)
        similarities = torch.tensor(sims, dtype=torch.float).view(n_subgraphs, max_n_cc, -1)     

        # add padding & save to file
        if not fname.parent.exists(): fname.parent.mkdir(parents=True)
        similarities[~cc_id_mask] = PAD_VALUE
        np.save(fname, similarities.cpu().numpy())
        return similarities
    
    def get_similarities(self):
        # path where similarities are stored
        sim_path = self.similarities_path 
        degree_path = self.degree_dict_path
        if degree_path.exists():
            with open(str(degree_path), 'r') as f:
                degree_dict = json.load(f)
            degree_dict = {int(key): value for key, value in degree_dict.items()}
        else: degree_dict = None

        if not sim_path.exists(): sim_path.mkdir()
        # (1) sample structure anchor patches
        # struc_anchor_patches_path = sim_path / str('struc_patches_' + str(self.hparams['sample_walk_len']) + '_' + self.hparams['structure_patch_type'] + '_' + str(self.hparams['max_sim_epochs']) + '.npy') 
        struc_anchor_patches_path = sim_path / 'struc_patches_max_sim_epochs={}_sample_walk_len={}_random_walk_len={}_n_anchor_patches_structure={}_n_triangular_walks={}_structure_patch_type={}.npy'.format(
            str(self.hparams['max_sim_epochs']),
            str(self.hparams['sample_walk_len']),
            str(self.hparams['random_walk_len']),
            str(self.hparams['n_triangular_walks']),
            str(self.hparams['n_anchor_patches_structure']),
            self.hparams['structure_patch_type'],
        )
        if struc_anchor_patches_path.exists():
            print('--- Loading structure anchor patches from File ---')
            self.structure_anchors = torch.tensor(np.load(struc_anchor_patches_path, allow_pickle=True))
        else:
            print('--- Computing structure anchor patches and save to File ---')
            self.structure_anchors = sample_structure_anchor_patches(self.hparams, self.networkx_graph, self.meta_paths, self.nodeid_to_nodetype, self.hparams['max_sim_epochs'])
            np.save(struc_anchor_patches_path, self.structure_anchors.cpu().numpy())
        print(struc_anchor_patches_path)
        
        # (2) perform internal and border random walks over sampled anchor patches
        # border
        # bor_struc_patch_random_walks_path = sim_path / str('bor_struc_patch_random_walks_' + str(self.hparams['n_triangular_walks']) +  '_' + str(self.hparams['random_walk_len']) +  '_' + str(self.hparams['sample_walk_len']) +  '_' + self.hparams['structure_patch_type'] + '_' + str(self.hparams['max_sim_epochs']) + '.npy')
        # int_struc_patch_random_walks_path = sim_path / str('int_struc_patch_random_walks_' + str(self.hparams['n_triangular_walks']) +  '_' + str(self.hparams['random_walk_len']) +  '_' + str(self.hparams['sample_walk_len']) +  '_' + self.hparams['structure_patch_type'] + '_' + str(self.hparams['max_sim_epochs']) + '.npy') 
        bor_struc_patch_random_walks_path = sim_path / 'bor_struc_patch_random_walks_max_sim_epochs={}_sample_walk_len={}_random_walk_len={}_n_anchor_patches_structure={}_n_triangular_walks={}_structure_patch_type={}.npy'.format(
            str(self.hparams['max_sim_epochs']),
            str(self.hparams['sample_walk_len']),
            str(self.hparams['random_walk_len']),
            str(self.hparams['n_anchor_patches_structure']),
            str(self.hparams['n_triangular_walks']),
            self.hparams['structure_patch_type'],
        )
        int_struc_patch_random_walks_path = sim_path / 'int_struc_patch_random_walks_max_sim_epochs={}_sample_walk_len={}_random_walk_len={}_n_anchor_patches_structure={}_n_triangular_walks={}_structure_patch_type={}.npy'.format(
            str(self.hparams['max_sim_epochs']),
            str(self.hparams['sample_walk_len']),
            str(self.hparams['random_walk_len']),
            str(self.hparams['n_anchor_patches_structure']),
            str(self.hparams['n_triangular_walks']),
            self.hparams['structure_patch_type'],
        )
        if bor_struc_patch_random_walks_path.exists(): 
            print('--- Loading border structure anchor random walks from File ---')
            self.bor_structure_anchor_random_walks = torch.tensor(np.load(bor_struc_patch_random_walks_path, allow_pickle=True))
        else:
            print('--- Computing border structure anchor random walks and save to File ---')
            self.bor_structure_anchor_random_walks = perform_random_walks(self.hparams, self.networkx_graph, self.meta_paths, self.nodeid_to_nodetype, self.structure_anchors, inside=False)
            np.save(bor_struc_patch_random_walks_path, self.bor_structure_anchor_random_walks.cpu().numpy())
        print(bor_struc_patch_random_walks_path)
        # internal
        if int_struc_patch_random_walks_path.exists():
            print('--- Loading internal structure anchor random walks from File ---')
            self.int_structure_anchor_random_walks = torch.tensor(np.load(int_struc_patch_random_walks_path, allow_pickle=True))
        else:
            print('--- Computing internal structure anchor random walks and save to File ---')
            self.int_structure_anchor_random_walks = perform_random_walks(self.hparams, self.networkx_graph, self.meta_paths, self.nodeid_to_nodetype, self.structure_anchors, inside=True)
            np.save(int_struc_patch_random_walks_path, self.int_structure_anchor_random_walks.cpu().numpy())
        print(int_struc_patch_random_walks_path)

        # (3) calculate similarities between anchor patches and connected components
        # int_struc_path = sim_path /  str('int_struc_' + str(self.hparams['sample_walk_len']) + '_' + self.hparams['structure_patch_type'] + '_' + str(self.hparams['max_sim_epochs']) + '_similarities.npy') 
        # bor_struc_path = sim_path /  str('bor_struc_' + str(self.hparams['sample_walk_len']) + '_'  + self.hparams['structure_patch_type'] + '_' + str(self.hparams['max_sim_epochs']) + '_similarities.npy')
        bor_struc_path = sim_path /  'bor_struc_max_sim_epochs={}_sample_walk_len={}_random_walk_len={}_n_anchor_patches_structure={}_n_triangular_walks={}_structure_patch_type={}_similarities.npy'.format(
            str(self.hparams['max_sim_epochs']),
            str(self.hparams['sample_walk_len']),
            str(self.hparams['random_walk_len']),
            str(self.hparams['n_anchor_patches_structure']),
            str(self.hparams['n_triangular_walks']),
            self.hparams['structure_patch_type'],
        )
        int_struc_path = sim_path /  'int_struc_max_sim_epochs={}_sample_walk_len={}_random_walk_len={}_n_anchor_patches_structure={}_n_triangular_walks={}_structure_patch_type={}_similarities.npy'.format(
            str(self.hparams['max_sim_epochs']),
            str(self.hparams['sample_walk_len']),
            str(self.hparams['random_walk_len']),
            str(self.hparams['n_anchor_patches_structure']),
            str(self.hparams['n_triangular_walks']),
            self.hparams['structure_patch_type'],
        )
        # border
        if bor_struc_path.exists():
            print('--- Loading border structure similarities from File ---')
            self.bor_struc_similarities = torch.tensor(np.load(bor_struc_path, allow_pickle=True))
        else:
            print('--- Computing border structure similarities ---')
            self.bor_struc_similarities = self.compute_structure_patch_similarities(degree_dict, bor_struc_path, internal=False, cc_ids=self.cc_ids, sim_path=sim_path)
        print(bor_struc_path)
        # internal    
        if int_struc_path.exists():
            print('--- Loading internal structure similarities from File ---', flush=True)
            self.int_struc_similarities = torch.tensor(np.load(int_struc_path, allow_pickle=True))
        else:
            print('--- Computing internal structure similarities ---')
            self.int_struc_similarities = self.compute_structure_patch_similarities(degree_dict, int_struc_path, internal=True, cc_ids=self.cc_ids, sim_path=sim_path)
        print(int_struc_path)
        
    def prepare_data(self):
        self.read_data()
        self.cc_ids = self.initialize_cc_ids(self.sub_G)
        self.init_all_embeddings()
        self.get_similarities()
        self.anchors_structure = init_anchors_structure(
            self.hparams,  
            self.structure_anchors, 
            self.int_structure_anchor_random_walks, 
            self.bor_structure_anchor_random_walks) 
        
class DyHNetDataset(Dataset):
    def __init__(self, data_dir, data_fname='data.pkl', num_time_steps=5, max_size=5, seed=0, params={}):
        random.seed(seed)
        np.random.seed(seed)
        self.data_dir = Path(data_dir)
        self.num_time_steps = num_time_steps
        self.max_size = max_size
        self.params = params
        self.data = pd.read_pickle(str(self.data_dir / data_fname))
        self.sample_subgraphs()
        self.encode_labels()
        self.read_snapshots()
        self.split_data()
    
    def sample_subgraphs(self):
        '''
        For each node, sample max_size number of subgraphs
        '''
        self.sampled_data = copy.deepcopy(self.data)
        for idx, d in self.data.items():
            sampled_subgraph_idx = {}
            for time_id in range(self.num_time_steps):
                subgraph_idx = d['subgraph_idx'][time_id]
                if isinstance(subgraph_idx, list) and len(subgraph_idx)>self.max_size:
                    random.shuffle(subgraph_idx)
                    sampled_subgraph_idx[time_id] = subgraph_idx[:self.max_size]
                else:
                    sampled_subgraph_idx[time_id] = subgraph_idx
                self.sampled_data[idx]['subgraph_idx'] = sampled_subgraph_idx
            
    def encode_labels(self):
        labels_str = [d['label'] for d in list(self.sampled_data.values())]
        if isinstance(labels_str[0], list) and self.params.multilabel:
            distinct_labels_str = sorted(set(itertools.chain(*labels_str)))
            self.label_encoder = {j:i for i,j in enumerate(distinct_labels_str)}
            self.labels = []
            for lab in labels_str:
                self.labels.append([self.label_encoder[l] for l in lab])
            self.multilabel_binarizer = MultiLabelBinarizer().fit(self.labels)
            
        else:
            distinct_labels_str = sorted(set(labels_str))
            print(distinct_labels_str)
            self.label_encoder = {j:i for i,j in enumerate(distinct_labels_str)}
            self.multilabel_binarizer = None
        
        for idx, d in self.sampled_data.items():
            if self.multilabel_binarizer is not None:
                lab = [self.label_encoder[l] for l in d['label']]
                self.sampled_data[idx]['label'] = self.multilabel_binarizer.transform([lab])[0]
            else:
                self.sampled_data[idx]['label'] = self.label_encoder[d['label']]

    def read_snapshots(self):
        self.snapshots = []
        for i in range(self.num_time_steps):
            snapshot_path = self.data_dir / 't_{:02d}'.format(i)
            print(f'### Processing {snapshot_path}')
            ss = GraphSnapshot(
                graph_path=snapshot_path / 'edge_list.txt', 
                node_path=snapshot_path / 'node_types.csv',
                subgraph_path=snapshot_path / 'subgraphs.pth',
                embedding_path=snapshot_path / 'gin_gcn_embeddings.pth',
                similarities_path=snapshot_path / 'similarities',
                degree_dict_path=snapshot_path / 'degree_sequence.txt',
                params=self.params,
            )
            self.snapshots.append(ss)

    def split_data(self):
        if 'dataset' in list(self.sampled_data.values())[0]:
            self.train_idx = []
            self.val_idx = []
            self.test_idx = []
            self.inference_idx = []

            for idx, d in self.sampled_data.items():
                if d['dataset'] == 'train':
                    self.train_idx.append(idx)
                elif d['dataset'] == 'val':
                    self.val_idx.append(idx)
                elif d['dataset'] == 'test':
                    self.test_idx.append(idx)
                else:
                    self.inference_idx.append(idx)

        else:
            total_samples = self.__len__()
            train_samples = int(total_samples * 0.6)
            val_samples = int(total_samples * 0.2)
            dataset_indices = list(self.sampled_data.keys())
            np.random.shuffle(dataset_indices)
            self.train_idx = dataset_indices[:train_samples]
            self.val_idx = dataset_indices[train_samples:(train_samples+val_samples)]
            self.test_idx = dataset_indices[(train_samples+val_samples):]

    def __getitem__(self, idx):
        d = self.sampled_data[idx]
        global_node_id = d['node_id']
        temporal_initial_embed = []
        temporal_subgraph_idx = []
        temporal_subgraph_mask = []
        temporal_cc_ids = []
        temporal_I_S_sim = []
        temporal_B_S_sim = []
        temporal_S_I_cc_embed = []
        temporal_S_B_cc_embed = []
        temporal_label = []
        
        global_time_id = d['time_id']
        temporal_mask = [1] * self.num_time_steps
        temporal_mask[d['time_id']] = 0

        for time_id in range(self.num_time_steps):
            snapshot = self.snapshots[time_id]
            subgraph_idx = d['subgraph_idx'][time_id]
            initial_embed = snapshot.node_embeddings(torch.LongTensor([global_node_id+1])).squeeze()
            if len(subgraph_idx) > 0:
                subgraph_mask = [1] * (self.max_size - len(subgraph_idx)) + [0] * len(subgraph_idx)
                subgraph_idx = tuple([0] * (self.max_size - len(subgraph_idx)) + subgraph_idx)
                cc_ids = snapshot.cc_ids[subgraph_idx, :, :]
                I_S_sim = snapshot.int_struc_similarities[subgraph_idx, :, :]
                B_S_sim = snapshot.bor_struc_similarities[subgraph_idx, :, :]
                S_I_cc_embed = snapshot.S_I_cc_embed[subgraph_idx, :, :].detach()
                S_B_cc_embed = snapshot.S_B_cc_embed[subgraph_idx, :, :].detach()
                
            else:
                subgraph_idx = tuple([0] * self.max_size)
                subgraph_mask = [1] * self.max_size
                cc_ids = snapshot.cc_ids[subgraph_idx, :, :]
                I_S_sim = snapshot.int_struc_similarities[subgraph_idx, :, :]
                B_S_sim = snapshot.bor_struc_similarities[subgraph_idx, :, :]
                S_I_cc_embed = snapshot.S_I_cc_embed[subgraph_idx, :, :].detach()
                S_B_cc_embed = snapshot.S_B_cc_embed[subgraph_idx, :, :].detach()
                
            temporal_initial_embed.append(initial_embed)
            temporal_subgraph_idx.append(subgraph_idx)
            temporal_subgraph_mask.append(subgraph_mask)
            temporal_cc_ids.append(cc_ids)
            temporal_I_S_sim.append(I_S_sim)
            temporal_B_S_sim.append(B_S_sim)
            temporal_S_I_cc_embed.append(S_I_cc_embed)
            temporal_S_B_cc_embed.append(S_B_cc_embed)
        
        temporal_initial_embed = torch.stack(temporal_initial_embed)
        temporal_subgraph_idx = torch.tensor(temporal_subgraph_idx, dtype=torch.long)
        temporal_subgraph_mask = torch.tensor(temporal_subgraph_mask, dtype=torch.long) 
        temporal_mask = torch.tensor(temporal_mask, dtype=torch.long)
        
        if self.params.multilabel:
            labels = torch.tensor(self.sampled_data[idx]['label'], dtype=torch.float)
        else:
            labels = torch.tensor(self.sampled_data[idx]['label'], dtype=torch.long)

        single_input = {
            'node_id': global_node_id,
            'time_id': global_time_id,
            'temporal_initial_embed': temporal_initial_embed,
            'temporal_subgraph_idx': temporal_subgraph_idx,
            'temporal_subgraph_mask': temporal_subgraph_mask,
            'temporal_cc_ids': temporal_cc_ids,
            'temporal_I_S_sim': temporal_I_S_sim,
            'temporal_B_S_sim': temporal_B_S_sim,
            'temporal_S_I_cc_embed': temporal_S_I_cc_embed,
            'temporal_S_B_cc_embed': temporal_S_B_cc_embed,
            'temporal_mask': temporal_mask,
            'labels': labels,
        }
        return single_input

    def __len__(self):
        return len(self.data)
    
class DyHNetDataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        
    def setup(self, stage=None):
        if 'data_fname' not in self.hparams:
            self.hparams.data_fname = 'data.pkl'
        self.data_full = DyHNetDataset(
            data_dir=PROJ_PATH / 'dataset' / self.hparams.name, 
            data_fname=self.hparams.data_fname,
            num_time_steps=self.hparams.num_time_steps, 
            max_size=self.hparams.max_size,
            seed=self.hparams.seed,
            params=self.hparams,
        )

        self.data_train = Subset(self.data_full, self.data_full.train_idx)
        self.data_val = Subset(self.data_full, self.data_full.val_idx)
        self.data_test = Subset(self.data_full, self.data_full.test_idx)
        self.data_inference = Subset(self.data_full, self.data_full.inference_idx)

        self.multilabel_binarizer = self.data_full.multilabel_binarizer
        self.label_encoder = self.data_full.label_encoder
        print(f'Multiple label: {self.hparams.multilabel}')
        print(f'Number of labels: {len(self.label_encoder)}')
        
    def collate_fn(self, batch): 
        node_id = torch.tensor([b['node_id'] for b in batch], dtype=torch.long)
        time_id = torch.tensor([b['time_id'] for b in batch], dtype=torch.long)
        temporal_initial_embed = torch.stack([b['temporal_initial_embed'] for b in batch])
        temporal_subgraph_idx = torch.stack([b['temporal_subgraph_idx'] for b in batch])
        temporal_subgraph_mask = torch.stack([b['temporal_subgraph_mask'] for b in batch]) 
        
        temporal_cc_ids = [b['temporal_cc_ids'] for b in batch]
        temporal_I_S_sim = [b['temporal_I_S_sim'] for b in batch]
        temporal_B_S_sim = [b['temporal_B_S_sim'] for b in batch]
        temporal_S_I_cc_embed = [b['temporal_S_I_cc_embed'] for b in batch]
        temporal_S_B_cc_embed = [b['temporal_S_B_cc_embed'] for b in batch]

        temporal_mask = torch.stack([b['temporal_mask'] for b in batch])
        labels = torch.stack([b['labels'] for b in batch])

        return {
            'node_id': node_id,
            'time_id': time_id,
            'temporal_initial_embed': temporal_initial_embed,
            'temporal_subgraph_idx': temporal_subgraph_idx,
            'temporal_subgraph_mask': temporal_subgraph_mask,
            'temporal_cc_ids': temporal_cc_ids,
            'temporal_I_S_sim': temporal_I_S_sim,
            'temporal_B_S_sim': temporal_B_S_sim,
            'temporal_S_I_cc_embed': temporal_S_I_cc_embed,
            'temporal_S_B_cc_embed': temporal_S_B_cc_embed,
            'temporal_mask': temporal_mask,
            'labels': labels,
        }
    
    def train_dataloader(self):
        return DataLoader(
            self.data_train, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            shuffle=True,
            drop_last=True, 
            collate_fn=self.collate_fn,
        )
    
    def mytrain_dataloader(self):
        return DataLoader(
            self.data_train, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            shuffle=False,
            drop_last=False, 
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.data_val, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def inference_dataloader(self):
        return DataLoader(
            self.data_inference, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            shuffle=False,
            collate_fn=self.collate_fn,
        )
            
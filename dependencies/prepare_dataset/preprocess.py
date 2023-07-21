# General
import numpy as np
import pandas as pd
import random
import pickle
from collections import Counter

# Pytorch
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, negative_sampling
from torch_geometric.utils.convert import to_networkx

# NetworkX
import networkx as nx
from networkx.relabel import convert_node_labels_to_integers, relabel_nodes
from networkx.generators.random_graphs import barabasi_albert_graph

# Sklearn
from sklearn.feature_extraction.text import CountVectorizer

import sys
import utils


def read_graphs(edge_f, node_f):
    """
    Read in base graph and create a Data object for Pytorch geometric

    Args
        - edge_f (str): directory of edge list

    Return
        - all_data (Data object): Data object of base graph
    """
    nx_G = nx.read_edgelist(edge_f, nodetype = int)
    all_nodes = pd.read_csv(node_f)['node_id'].astype(type(list(nx_G.nodes)[0])).values
    nx_G.add_nodes_from(all_nodes)
    feat_mat = np.eye(len(nx_G.nodes), dtype=np.uint8)
    print("Graph density", nx.density(nx_G))
    all_data = create_dataset(nx_G, feat_mat)
    print(all_data)
    # assert nx.is_connected(nx_G)
    assert len(nx_G) == all_data.x.shape[0]
    return all_data


def create_dataset(G, feat_mat, split=False):
    """
    Create Data object of the base graph for Pytorch geometric

    Args
        - G (object): NetworkX graph
        - feat_mat (tensor): feature matrix for each node

    Return
        - new_G (Data object): new Data object of base graph for Pytorch geometric 
    """

    edge_index = torch.tensor(list(G.edges)).t().contiguous() 
    x = torch.tensor(feat_mat, dtype=torch.float) # Feature matrix    
    y = torch.ones(edge_index.shape[1]) 
    num_classes = len(torch.unique(y)) 

    split_idx = np.arange(len(y))
    np.random.shuffle(split_idx)
    train_samples = int(0.7 * len(split_idx))
    val_samples = int(0.15 * len(split_idx))
    train_idx = split_idx[:train_samples]
    val_idx = split_idx[train_samples:(train_samples + val_samples)]
    test_idx = split_idx[(train_samples + val_samples):]

    # Train set
    train_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[train_idx] = 1

    # Val set
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask[val_idx] = 1

    # Test set
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask[test_idx] = 1

    new_G = Data(x = x, y = y, num_classes = num_classes, edge_index = edge_index, train_mask = train_mask, val_mask = val_mask, test_mask = test_mask) 
    return new_G


def set_data(data, all_data, minibatch):
    """
    Create per-minibatch Data object

    Args
        - data (Data object): batched dataset
        - all_data (Data object): full dataset
        - minibatch (str): NeighborSampler

    Return
        - data (Data object): base graph as Pytorch Geometric Data object
    """

    batch_size, n_id, adjs = data
    data = Data(edge_index = adjs[0], n_id = n_id, e_id = adjs[1]) 
    data.x = all_data.x[data.n_id]
    data.train_mask = all_data.train_mask[data.e_id]
    data.val_mask = all_data.val_mask[data.e_id]
    data.y = torch.ones(len(data.e_id)) 
    return data


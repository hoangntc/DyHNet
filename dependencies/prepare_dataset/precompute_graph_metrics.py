# General
import networkx as nx
import sys
import argparse
import snap
from pathlib import Path
import numpy as np
import json
import os
import multiprocessing
from functools import partial

'''
Use this script to precompute information about the underlying base graph.
'''

def get_shortest_path(node_id):
    NIdToDistH = snap.TIntH()
    path_len = snap.GetShortPath(snap_graph, int(node_id), NIdToDistH)
    paths = np.zeros((max(node_ids) + 1)) #previously was n_nodes
    for dest_node in NIdToDistH: 
        paths[dest_node] = NIdToDistH[dest_node]
    return paths

def calculate_stats(snap_graph, node_ids, config):

    # create similarities folder
    if not os.path.exists(config.DATASET_DIR / 'similarities'):
        os.makedirs(config.DATASET_DIR / 'similarities')

    # if config.CALCULATE_EGO_GRAPHS:
    #     print(f'Calculating ego graphs for {config.DATASET_DIR }...')
    #     if not (config.DATASET_DIR / 'ego_graphs.txt').exists() or config.OVERRIDE:
    #         ego_graph_dict = {}
    #         for node in snap_graph.Nodes():
    #             node_id = int(node.GetId())
    #             nodes_vec = snap.TIntV()
    #             snap.GetNodesAtHop(snap_graph, node_id, 1, nodes_vec, False)
    #             ego_graph_dict[node_id] = list(nodes_vec)
            
    #         with open(str(config.DATASET_DIR / 'ego_graphs.txt'), 'w') as f:
    #             json.dump(ego_graph_dict, f)

    if config.CALCULATE_DEGREE_SEQUENCE:
        print(f'Calculating degree sequences for {config.DATASET_DIR}...')
        if not (config.DATASET_DIR / 'degree_sequence.txt').exists() or config.OVERRIDE:
            n_nodes = len(list(snap_graph.Nodes()))
            degrees = {}
            InDegV = snap.TIntPrV()
            snap.GetNodeInDegV(snap_graph, InDegV)
            OutDegV = snap.TIntPrV()
            snap.GetNodeOutDegV(snap_graph, OutDegV)
            
            for item1, item2 in zip(InDegV,OutDegV) :
                degrees[item1.GetVal1()] = item1.GetVal2()

            node_ids = [int(n) for n in node_ids]
            isolated_nodes = list(set(node_ids).difference(list(degrees.keys())))
            print('Number of node_ids:', len(node_ids))
            for n in isolated_nodes:
                degrees[n] = 0
            with open(str(config.DATASET_DIR / 'degree_sequence.txt'), 'w') as f:
                json.dump(degrees, f)

    # if config.CALCULATE_SHORTEST_PATHS:
    #     print(f'Calculating shortest paths for {config.DATASET_DIR}...')
    #     if not (config.DATASET_DIR /'shortest_path_matrix.npy').exists() or config.OVERRIDE:


    #         with multiprocessing.Pool(processes=config.N_PROCESSSES) as pool:
    #             # func = partial(get_shortest_path, snap_graph=snap_graph)
    #             # pool.map(func, iterable)
    #             shortest_paths = pool.map(get_shortest_path, node_ids)

    #         all_shortest_paths = np.stack(shortest_paths)
    #         np.save(str(config.DATASET_DIR / 'shortest_path_matrix.npy'), all_shortest_paths)

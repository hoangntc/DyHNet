# General
import os, sys,re
import numpy as np
import pandas as pd
from pathlib import Path
import snap
PROJ_PATH = Path(os.path.join(re.sub("/DyHNet.*$", '', os.getcwd()), 'DyHNet'))
sys.path.insert(1, str(PROJ_PATH / 'dependencies'/ 'prepare_dataset'))
import config_prepare_dataset
import train_node_emb
import precompute_graph_metrics

class SnapshotGraph:
    def __init__(
        self, 
        data_dir='',
        num_time_steps=2,
    ):
        self.data_dir = data_dir
        self.temporal_subgraph_path = os.path.join(self.data_dir, 'temporal_subgraphs.pth')
        self.pd_temporal_edges = pd.read_csv(
            os.path.join(self.data_dir, 'temporal_edge_list.txt'), 
            sep=' ', names=['source', 'target', 'time_id'])
        self.pd_node_types = pd.read_csv(os.path.join(self.data_dir, 'node_types.csv'))
        self.num_time_steps = self.pd_temporal_edges['time_id'].max() + 1

        assert (self.num_time_steps == num_time_steps) and (num_time_steps > 1), 'Incorrect number of timesteps!'
        self.create_temporal_data()

    def create_temporal_data(self):
        self.create_folder()
        self.get_subgraphs()
        self.get_edges()

    def create_folder(self):
        print('Create snapshot folders')
        for tid in range(self.num_time_steps):
            f_name = os.path.join(self.data_dir, 't_{:02d}'.format(int(tid)))
            if not os.path.exists(f_name): os.mkdir(f_name)
            
    def get_subgraphs(self):
        print('Get temporal subgraphs for each snapshot')
        with open(self.temporal_subgraph_path) as fin:
            for line in fin:
                lst_line = line.split('\t')
                tid = lst_line[1]
                pout = os.path.join(self.data_dir, 't_{:02d}'.format(int(tid)), 'subgraphs.pth')
                with open(pout, 'a') as fout:
                    fout.write(line)
                    fout.close()

    def get_edges(self):
        print('Get temporal edges for each snapshot')
        for tid in range(self.num_time_steps):
            df_edges = self.pd_temporal_edges[self.pd_temporal_edges['time_id']==tid][['source', 'target']]
            nodes = list(set(df_edges['source'].values.tolist() + df_edges['target'].values.tolist()))
            df_nodes = self.pd_node_types.copy()
            
            save_path = os.path.join(self.data_dir, 't_{:02d}'.format(int(tid)), 'edge_list.txt')
            print(save_path)
            df_edges.to_csv(save_path, header=None, index=None, sep=' ')
            
            save_path = os.path.join(self.data_dir, 't_{:02d}'.format(int(tid)), 'node_types.csv')
            print(save_path)
            df_nodes.to_csv(save_path, index=None)

def prepare_data(config):
    num_time_steps = config['num_time_steps']
    dataset_name = config['name']
    data_dir = str(PROJ_PATH / 'dataset' / dataset_name)
    
    # Split input data into folders
    SnapshotGraph(data_dir=data_dir, num_time_steps=num_time_steps)

    # Prepare input data for the model training
    for t in range(0, num_time_steps):
        config_prepare_dataset.DATASET_DIR = PROJ_PATH / 'dataset' / dataset_name / 't_{:02d}'.format(t)
        # Compute metrics
        snap_graph = snap.LoadEdgeList(snap.PUNGraph, str(config_prepare_dataset.DATASET_DIR / 'edge_list.txt'), 0, 1)
        node_ids = np.sort(pd.read_csv(str(config_prepare_dataset.DATASET_DIR / 'node_types.csv'))['node_id'].values)
        precompute_graph_metrics.calculate_stats(snap_graph, node_ids, config_prepare_dataset)
        # Extract local features for nodes
        train_node_emb.generate_emb(config_prepare_dataset)
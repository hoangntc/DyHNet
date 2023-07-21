import os, sys, re, datetime, random, gzip, json, copy
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from time import time
from math import ceil
from pathlib import Path
import itertools
import argparse
import networkx as nx
from stellargraph import StellarGraph

import matplotlib.pyplot as plt
PROJ_PATH = Path(os.path.join(re.sub("/DyHNet.*$", '', os.getcwd()), 'DyHNet'))
sys.path.insert(1, str(PROJ_PATH / 'DyHNet' / 'src'))
sys.path.insert(1, str(PROJ_PATH / 'dependencies'))

from littleballoffur.exploration_sampling import (
    DiffusionSampler, RandomWalkSampler, CommonNeighborAwareRandomWalkSampler, 
    RandomWalkWithJumpSampler, RandomWalkWithRestartSampler, CirculatedNeighborsRandomWalkSampler)
            
class TemporalSubgraphSampler():
    def __init__(self, node_path, edge_path, sampled_node_ids, max_size=5, number_of_nodes=20, seed=0, output_dir='./'):
        self.pd_edges = pd.read_csv(edge_path, sep=' ', names=['source', 'target', 'time_id'])
        pd_node_types = pd.read_csv(node_path)
        # self.nodes = pd_node_types[pd_node_types['node_type_name']==sampled_node_type]['node_id'].values.tolist()
        self.all_nodes = pd_node_types['node_id'].values
        self.sampled_node_ids = sampled_node_ids
        self.num_time_steps = self.pd_edges['time_id'].max() + 1
        self.max_size = max_size
        self.number_of_nodes = number_of_nodes
        self.seed = seed
        self.output_dir = output_dir
        # self.sampling_temporal_subgraph()
        # self.write_temporal_subgraphs()
        # self.write_data()
        
    def sampling_subgraph(self, G_reindex, start_node, inversed_mapping=None):
        '''
        Sampling subgraphs based on different criteria: 
        https://little-ball-of-fur.readthedocs.io/en/latest/index.html
        '''
        subgs = [
            DiffusionSampler(number_of_nodes=self.number_of_nodes, seed=self.seed).sample(
                G_reindex, start_node=start_node),
            RandomWalkSampler(number_of_nodes=self.number_of_nodes, seed=self.seed).sample(
                G_reindex, start_node=start_node),
            CommonNeighborAwareRandomWalkSampler(number_of_nodes=self.number_of_nodes, seed=self.seed).sample(
                G_reindex, start_node=start_node),
            RandomWalkWithRestartSampler(number_of_nodes=self.number_of_nodes, seed=self.seed).sample(
                G_reindex, start_node=start_node),
            CirculatedNeighborsRandomWalkSampler(number_of_nodes=self.number_of_nodes, seed=self.seed).sample(
                G_reindex, start_node=start_node),
        ][:self.max_size]
        subgs = [g for g in subgs if len(list(g.nodes)) > 1] # remove subgraphs of isolated nodes
        tmp = [sorted(list(g.nodes)) for g in subgs]
        subgraphs = list(k for k,_ in itertools.groupby(tmp))
        return subgraphs
    
    def build_graph(self, all_nodes, edge_tuple):
        '''
        Prepare graph for sampling
        '''
        G = nx.Graph()
        G.add_edges_from(edge_tuple)
        G.add_nodes_from(all_nodes)
        return G
    
    def sampling_temporal_subgraph(self):
        '''
        # temporal_subgraph = {time_id: [subgraphs]}
        # data = {
        #     node_id: {time_id: {'subgraph_idx': [], 'label': []},
        # }
        # mapping = {time_id: {snapshot node_id: global node_id}}
        '''
        self.temporal_subgraph = {}
        self.data = {}
        self.subgraphs_node_mapping = {}
        
        for tid in range(self.num_time_steps):
            self.temporal_subgraph[tid] = []
                
        for nid in self.sampled_node_ids:
            self.data[nid] = {}
            for tid in range(self.num_time_steps):
                self.data[nid][tid] = {'subgraph_idx': []}
        
        for tid in range(self.num_time_steps):
            print(f'Sampling subgraph at time id: {tid}')
            i_pd_edges = self.pd_edges[self.pd_edges['time_id']==tid]
            i_nodes = sorted(set(i_pd_edges['source'].values.tolist() + i_pd_edges['target'].values.tolist()))
            i_edge_tuple = list(i_pd_edges[['source', 'target']].to_records(index=False))
            G = self.build_graph(self.all_nodes, i_edge_tuple)
            
            for nid in tqdm(self.sampled_node_ids, total=len(self.sampled_node_ids)):
                i_subgs = self.sampling_subgraph(G, start_node=nid)
                no_subgs = len(self.temporal_subgraph[tid])
                self.data[nid][tid]['subgraph_idx'] = list(range(no_subgs, no_subgs + len(i_subgs)))
                self.temporal_subgraph[tid] += i_subgs
            
    def write_temporal_subgraphs(self):
        with open(os.path.join(self.output_dir, 'temporal_subgraphs.pth'), 'w') as fout:
            for tid in range(self.num_time_steps):
                for subgraph in self.temporal_subgraph[tid]:
                    subgraph_str = '-'.join([str(n) for n in subgraph])
                    fout.write('\t'.join([subgraph_str, str(tid), '\n']))
    
    def write_data(self):
        pd.to_pickle(self.data, './temporal_subgraph_data.pkl')
        
        
class SubGraphSampler():
    def __init__(self, node_list, edge_list, subgraph_type, **kwargs):
        self.subgraph_type = subgraph_type
        self.node_list = node_list
        self.edge_list = edge_list
        self.graph = self.create_base_graph()
        self.subgraphs = self.generate_and_add_subgraphs(**kwargs)
        
    def create_base_graph(self):
        '''
        Create a base graph
        '''
        graph = nx.Graph()
        graph.add_nodes_from(self.node_list)
        graph.add_edges_from(self.edge_list)
        print('Number of nodes:', len(graph.nodes))
        print('Number of edges:', len(graph.edges))
        return graph
    
    def generate_and_add_subgraphs(self, **kwargs):
        """
        Generate and add subgraphs to the base graph.
        Return
            - subgraphs (list of lists): list of subgraphs, where each subgraph is a list of nodes
        """

        n_subgraphs = kwargs.pop('n_subgraphs', 3)
        n_nodes_in_subgraph = kwargs.pop('n_subgraph_nodes', 5)
        n_connected_components = kwargs.pop('n_connected_components', 1)

        if self.subgraph_type == 'bfs':
            subgraphs =  self._get_subgraphs_by_bfs(n_subgraphs, n_nodes_in_subgraph, n_connected_components)
        elif self.subgraph_type == 'coreness':
            subgraphs = self._get_subgraphs_by_coreness(n_subgraphs, n_nodes_in_subgraph, n_connected_components)
        else:
            raise Exception('The subgraph generation you specified is not implemented')

        return subgraphs

    def _get_subgraphs_by_coreness(self, n_subgraphs, n_nodes_in_subgraph, n_connected_components, remove_edges=False, **kwargs):
        """
        Sample nodes from the base graph that have at least n nodes with k core. Merge the edges from the generated
        subgraph with the edges from the base graph. Optionally, remove all other edges in the subgraphs
        Args
            - n_subgraphs (int): number of subgraphs
            - n_nodes_in_subgraph (int): number of nodes in each subgraph
            - n_connected_components (int): number of connected components in each subgraph
            - remove_edges (bool): true if should remove unmerged edges in subgraphs, false otherwise
        
        Return
            - subgraphs (list of lists): list of subgraphs, where each subgraph is a list of nodes
        """

        subgraphs = []

        k_core_dict = nx.core_number(self.graph)        
        nodes_per_k_core = Counter(list(k_core_dict.values()))
        print(nodes_per_k_core)
        
        nodes_with_core_number = defaultdict()
        for n, k in k_core_dict.items():
            if k in nodes_with_core_number: nodes_with_core_number[k].append(n)
            else: nodes_with_core_number[k] = [n]

        for k in nodes_with_core_number:

            # Get nodes with core number k that have not been sampled already
            nodes_with_k_cores = nodes_with_core_number[k]
            
            # Sample n_subgraphs subgraphs per core number
            for s in range(n_subgraphs):

                curr_subgraph = []
                for c in range(n_connected_components):
                    if len(nodes_with_k_cores) < n_nodes_in_subgraph: break

                    con_component = self.generate_subgraph(n_nodes_in_subgraph, **kwargs)
                    cc_node_ids = random.sample(nodes_with_k_cores, n_nodes_in_subgraph)

                    # Relabel subgraph to have the same ids as the randomly sampled nodes
                    cc_id_mapping = {curr_id:new_id for curr_id, new_id in zip(con_component.nodes, cc_node_ids)}
                    nx.relabel_nodes(con_component, cc_id_mapping, copy=False)
            
                    if remove_edges:
                        # Remove the existing edges between nodes in the planted subgraph (except the ones to be added)
                        self.graph.remove_edges_from(self.graph.subgraph(cc_node_ids).edges)

                    # Combine the base graph & subgraph. Nodes with the same ID are merged
                    joined_graph = nx.compose(self.graph, con_component) #NOTE: attributes from subgraph take precedent over attributes from self.graph
                    self.graph = joined_graph.copy()
                    
                    curr_subgraph.extend(cc_node_ids) # add nodes to subgraph
                    nodes_with_k_cores = list(set(nodes_with_k_cores).difference(set(cc_node_ids)))
                    nodes_with_core_number[k] = nodes_with_k_cores
                
                if len(curr_subgraph) > 0: subgraphs.append(curr_subgraph)

        return subgraphs

    def _get_subgraphs_by_bfs(self, n_subgraphs, n_nodes_in_subgraph, n_connected_components,  **kwargs):
        """
        Sample n_connected_components number of start nodes from the base graph. Perform BFS to create subgraphs
        of size n_nodes_in_subgraph.
        Args
            - n_subgraphs (int): number of subgraphs
            - n_nodes_in_subgraph (int): number of nodes in each subgraph
            - n_connected_components (int): number of connected components in each subgraph
        
        Return
            - subgraphs (list of lists): list of subgraphs, where each subgraph is a list of nodes
        """

        max_depth = kwargs.pop('max_depth', 3)

        subgraphs = []
        for s in range(n_subgraphs):

            #randomly select start nodes. # of start nodes == n connected components
            curr_subgraph = []
            start_nodes = random.sample(self.graph.nodes, n_connected_components)            
            for start_node in start_nodes:
                edges = nx.bfs_edges(self.graph, start_node, depth_limit=max_depth)
                nodes = [start_node] + [v for u, v in edges]
                nodes = nodes[:n_nodes_in_subgraph] #limit nodes to n_nodes_in_subgraph

                if max(nodes) > max(self.graph.nodes): print(max(nodes), max(self.graph.nodes))
                assert max(nodes) <= max(self.graph.nodes)

                assert nx.is_connected(self.graph.subgraph(nodes)) #check to see if selected nodes represent a conencted component
                curr_subgraph.extend(nodes)
            subgraphs.append(sorted(curr_subgraph))
        
        seen = []
        for g in subgraphs:
            seen += g
        assert max(seen) <= max(self.graph.nodes)
        return subgraphs
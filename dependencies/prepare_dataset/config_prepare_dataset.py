from pathlib import Path
import sys, os, re

PROJ_PATH = Path(os.path.join(re.sub("/DyHNet.*$", '', os.getcwd()), 'DyHNet'))

DATASET_DIR = Path(PROJ_PATH) / 'dataset'
GENERATE_NODE_EMB = True # whether to generate node embeddings
RANDOM_SEED = 42
CONV = "gin_gcn"
MINIBATCH = "NeighborSampler"
POSSIBLE_BATCH_SIZES = [512]
POSSIBLE_HIDDEN = [256]
POSSIBLE_OUTPUT = [128]
POSSIBLE_LR = [0.0001, 0.001, 0.005]
POSSIBLE_WD = [5e-5]
POSSIBLE_DROPOUT = [0.01, 0.001]
POSSIBLE_NB_SIZE = [-1]
POSSIBLE_NUM_HOPS = [3]
POSSIBLE_WALK_LENGTH = [32]
POSSIBLE_NUM_STEPS = [32]
EPOCHS = 50

# Flags for precomputing similarity metrics
CALCULATE_SHORTEST_PATHS = True # Calculate pairwise shortest paths between all nodes in the graph
CALCULATE_DEGREE_SEQUENCE = True # Create a dictionary containing degrees of the nodes in the graph
CALCULATE_EGO_GRAPHS = True # Calculate the 1-hop ego graph associated with each node in the graph
TEMPORAL_INFO = True
OVERRIDE = True # Overwrite a similarity file even if it exists
N_PROCESSSES = 4 # Number of cores to use for multi-processsing when precomputing similarity metrics

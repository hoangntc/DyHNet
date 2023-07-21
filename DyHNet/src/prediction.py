import os, sys, re, datetime, random, gzip, json, copy
from tqdm import tqdm
import pandas as pd
import numpy as np
from time import time
from math import ceil
from pathlib import Path
from collections import OrderedDict
import itertools
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.metrics import roc_auc_score, precision_score, recall_score

PROJ_PATH = Path(os.path.join(re.sub("/DyHNet.*$", '', os.getcwd()), 'DyHNet'))
import utils
from evaluation import eval_link_prediction, eval_node_classification
from ranking import make_prediction

def get_training_data_link_prediction(data, node_embeddings, all_labels, label_mapping):
    source_features = []
    target_features = []
    labels = []
    train_val_test_index = []

    for i,d in data.items():
        node_id = d['node_id']
        time_id = d['time_id']
        pos_labels = d['label']
        neg_labels = [i for i in all_labels if i not in pos_labels]
        if len(pos_labels) > 0:
            for l in pos_labels:
                labels.append(1)
                label_nid = label_mapping[l]
                source_features.append(node_embeddings[time_id][node_id])
                target_features.append(node_embeddings[time_id][label_nid])
                train_val_test_index.append(d['dataset'])
        if len(neg_labels) > 0:   
            for l in neg_labels:
                labels.append(0)
                label_nid = label_mapping[l]
                source_features.append(node_embeddings[time_id][node_id])
                target_features.append(node_embeddings[time_id][label_nid])
                train_val_test_index.append(d['dataset'])
    return source_features, target_features, labels, train_val_test_index

def eval_lp(name, node_embeddings, embed_path=None, operators=[], threshold=0.5):
    """
    Return AUC, f1 score for each operator
    """
    assert name in ['dblp', 'imdb', 'dblp_lp'], 'Unknown dataset'

    if embed_path is not None:
        node_embeddings = pd.read_pickle(str(PROJ_PATH / 'output'/ embed_path))
    
    if len(operators) == 0:
        operators = ['HAD', 'AVG', 'L1', 'L2']
    
    if name == 'imdb':
        data = pd.read_pickle(str(PROJ_PATH / 'dataset' / name / 'data.pkl'))
        all_labels = list(set(itertools.chain(*[d['label'] for i, d in data.items()])))
        label_mapping = pd.read_pickle(str(PROJ_PATH / 'dataset' / name / 'entity_mapping.pkl'))['genre']
    elif name == 'dblp' or 'dblp_lp':
        data = pd.read_pickle(str(PROJ_PATH / 'dataset' / name / 'data.pkl'))
        cid2cname = pd.read_pickle(str(PROJ_PATH / 'dataset' / name / 'cid2cname.pkl'))
        all_labels = list(set(itertools.chain(*[d['label'] for i, d in data.items()])))
        label_mapping = {j:i for i,j in cid2cname.items() if j in all_labels}

    source_features, target_features, labels, train_val_test_index = get_training_data_link_prediction(data, node_embeddings, all_labels, label_mapping)
    results, models = eval_link_prediction(source_features, target_features, labels, train_val_test_index, operators, threshold)
    # create report table
    pd_results = pd.DataFrame(results)
    cnames_auc =  [c for c in pd_results.columns if '_auc' in c]
    cnames_f1 = [c for c in pd_results.columns if '_f1' in c] 
    pd_results['best_auc'] = pd_results[cnames_auc].max(axis=1)
    pd_results['best_f1'] = pd_results[cnames_f1].max(axis=1)
    pd_results = pd_results[cnames_auc + ['best_auc'] + cnames_f1 + ['best_f1']]
    return pd_results, models

def get_ground_truth_lp_ranking(data, all_labels, label_mapping):
    ground_truth = {}
    pred_idx = {}
    for i,d in data.items():
        node_id = d['node_id']
        time_id = d['time_id']
        pos_labels = d['label']
        neg_labels = [i for i in all_labels if i not in pos_labels]
        if len(pos_labels) > 0:
            ground_truth[node_id] = [label_mapping[l] for l in pos_labels]
        pred_idx[node_id] = [label_mapping[l] for l in all_labels]
    return ground_truth, pred_idx

def evaluate_multilabel_classification(gt, prd):
    micro_f1 = f1_score(gt, prd, average='micro', zero_division=0)
    macro_f1 = f1_score(gt, prd, average='macro', zero_division=0)
    f1 = f1_score(gt, prd, average='weighted', zero_division=0)
    micro_recall = recall_score(gt, prd, average='micro', zero_division=0)
    macro_recall = recall_score(gt, prd, average='macro', zero_division=0)
    recall = recall_score(gt, prd, average='weighted', zero_division=0)
    micro_precision = precision_score(gt, prd, average='micro', zero_division=0)
    macro_precision = precision_score(gt, prd, average='macro', zero_division=0)
    precision = precision_score(gt, prd, average='weighted', zero_division=0)
    return f1, micro_f1, macro_f1, recall, micro_recall, macro_recall, precision, micro_precision, macro_precision

def eval_lp_ranking(name, node_embeddings, embed_path=None, models={}, k=5):
    '''
    Args:
        models: trained link detectors in eval_lp --> Dont need to train these models again in this step
    Return:
        f1 score, recall, precision @k (top k prediction)
    '''
    assert name in ['dblp', 'imdb', 'dblp_lp'], 'Unknown dataset'

    if embed_path is not None:
        node_embeddings = pd.read_pickle(str(PROJ_PATH / 'output'/ embed_path))
    operators = list(models.keys())
    
    if name == 'imdb':
        time_id = 4 - 1
        data = pd.read_pickle(str(PROJ_PATH / 'dataset' / name / 'data.pkl'))
        test_data = {i: d for i,d in data.items() if d['dataset']=='test'}
        label_mapping = pd.read_pickle(str(PROJ_PATH / 'dataset' / name / 'entity_mapping.pkl'))['genre']
        all_labels = list(set(itertools.chain(*[d['label'] for i, d in data.items()])))
    elif name == 'dblp':
        time_id = 7 - 1
        data = pd.read_pickle(str(PROJ_PATH / 'dataset' / name / 'data.pkl'))
        test_data = {i: d for i,d in data.items() if d['dataset']=='test'}
        cid2cname = pd.read_pickle(str(PROJ_PATH / 'dataset' / name / 'cid2cname.pkl'))
        all_labels = list(set(itertools.chain(*[d['label'] for i, d in data.items()])))
        label_mapping = {j:i for i,j in cid2cname.items() if j in all_labels}
    elif name == 'dblp_lp':
        time_id = 6 - 1
        data = pd.read_pickle(str(PROJ_PATH / 'dataset' / name / 'data.pkl'))
        test_data = {i: d for i,d in data.items() if d['dataset']=='test'}
        cid2cname = pd.read_pickle(str(PROJ_PATH / 'dataset' / name / 'cid2cname.pkl'))
        all_labels = list(set(itertools.chain(*[d['label'] for i, d in data.items()])))
        label_mapping = {j:i for i,j in cid2cname.items() if j in all_labels}

    ground_truth, pred_idx = get_ground_truth_lp_ranking(test_data, all_labels, label_mapping)
    multilabel_binarizer = MultiLabelBinarizer().fit(list(ground_truth.values()))
    gt = multilabel_binarizer.transform(list(ground_truth.values()))
    
    node_embedding = node_embeddings[time_id]
    ranking = make_prediction(pred_idx, node_embedding, models)
    rk = ranking[operators[0]]
    pd_pred = pd.DataFrame(rk, columns=['source', 'target', 'sims']).sort_values(['sims'], ascending=False)
    pred = pd_pred.groupby('source').agg({'target': list}).to_dict()['target']

    result = {}
    for k in range(1, k+1):
        prd = multilabel_binarizer.transform(list({i: j[:k] for i, j in pred.items()}.values()))
        result[k] = evaluate_multilabel_classification(gt, prd)
    df = pd.DataFrame(result).T
    df.columns = [
        'f1', 'micro_f1', 'macro_f1', 
        'recall', 'micro_recall', 'macro_recall', 
        'precision', 'micro_precision', 'macro_precision']
    df['k'] = range(1, k+1)
    return df

def get_training_data_node_classification(data, node_embeddings):
    # Get labels, tvts
    nids = []
    features = []
    tvts = []
    labels_str = []
    for i, d in data.items():
        time_id = d['time_id']
        nid = d['node_id']
        nids.append(nid)
        tvts.append(d['dataset'])
        labels_str.append(d['label'])
        features.append(node_embeddings[time_id][nid])

    label_mapping = {j:i for i,j in enumerate(sorted(set(labels_str)))}
    print('Label mapping:', label_mapping)
    labels = [label_mapping[l] for l in labels_str]

    # # Get features
    # features = []
    # for nid in nids:
    #     features.append(node_embedding[nid])
    return nids, features, labels, tvts

def eval_nc(name, node_embeddings, embed_path=None):
    if embed_path is not None:
        node_embeddings = pd.read_pickle(str(PROJ_PATH / 'output'/ embed_path))
        # time_id = max(list(node_embeddings.keys()))
        # node_embedding = node_embeddings[time_id]
    if name.startswith('yelp') or name.startswith('dblp_four_area'):
        data = pd.read_pickle(str(PROJ_PATH / 'dataset' / name / 'data.pkl'))
    else:
        print('Unknown dataset')
    nids, features, labels, tvts = get_training_data_node_classification(data, node_embeddings)
    results, model = eval_node_classification(features, labels, train_val_test_index=tvts)
    pd_results = pd.DataFrame(results)
    return pd_results, model
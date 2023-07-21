# -*- coding: utf-8 -*-
#
# Author: Taylor G Smith
#
# Recommender system ranking metrics derived from Spark source for use with
# Python-based recommender libraries (i.e., implicit, 
# http://github.com/benfred/implicit/). These metrics are derived from the
# original Spark Scala source code for recommender metrics.
# https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/evaluation/RankingMetrics.scala

from __future__ import absolute_import, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from evaluation import sigmoid, get_link_score

__all__ = [
    'mean_average_precision',
    'ndcg_at',
    'precision_at',
]

try:
    xrange
except NameError:  # python 3 does not have an 'xrange'
    xrange = range


def _require_positive_k(k):
    """Helper function to avoid copy/pasted code for validating K"""
    if k <= 0:
        raise ValueError("ranking position k should be positive")


def _mean_ranking_metric(predictions, labels, metric):
    """Helper function for precision_at_k and mean_average_precision"""
    # do not zip, as this will require an extra pass of O(N). Just assert
    # equal length and index (compute in ONE pass of O(N)).
    # if len(predictions) != len(labels):
    #     raise ValueError("dim mismatch in predictions and labels!")
    # return np.mean([
    #     metric(np.asarray(predictions[i]), np.asarray(labels[i]))
    #     for i in xrange(len(predictions))
    # ])
    
    # Actually probably want lazy evaluation in case preds is a 
    # generator, since preds can be very dense and could blow up 
    # memory... but how to assert lengths equal? FIXME
    return np.mean([
        metric(np.asarray(prd), np.asarray(labels[i]))
        for i, prd in enumerate(predictions)  # lazy eval if generator
    ])


def _warn_for_empty_labels():
    """Helper for missing ground truth sets"""
    warnings.warn("Empty ground truth set! Check input data")
    return 0.


def precision_at(predictions, labels, k=10, assume_unique=True):
    """Compute the precision at K.
    Compute the average precision of all the queries, truncated at
    ranking position k. If for a query, the ranking algorithm returns
    n (n is less than k) results, the precision value will be computed
    as #(relevant items retrieved) / k. This formula also applies when
    the size of the ground truth set is less than k.
    If a query has an empty ground truth set, zero will be used as
    precision together with a warning.
    Parameters
    ----------
    predictions : array-like, shape=(n_predictions,)
        The prediction array. The items that were predicted, in descending
        order of relevance.
    labels : array-like, shape=(n_ratings,)
        The labels (positively-rated items).
    k : int, optional (default=10)
        The rank at which to measure the precision.
    assume_unique : bool, optional (default=True)
        Whether to assume the items in the labels and predictions are each
        unique. That is, the same item is not predicted multiple times or
        rated multiple times.
    Examples
    --------
    >>> # predictions for 3 users
    >>> preds = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5],
    ...          [4, 1, 5, 6, 2, 7, 3, 8, 9, 10],
    ...          [1, 2, 3, 4, 5]]
    >>> # labels for the 3 users
    >>> labels = [[1, 2, 3, 4, 5], [1, 2, 3], []]
    >>> precision_at(preds, labels, 1)
    0.33333333333333331
    >>> precision_at(preds, labels, 5)
    0.26666666666666666
    >>> precision_at(preds, labels, 15)
    0.17777777777777778
    """
    # validate K
    _require_positive_k(k)

    def _inner_pk(pred, lab):
        # need to compute the count of the number of values in the predictions
        # that are present in the labels. We'll use numpy in1d for this (set
        # intersection in O(1))
        if lab.shape[0] > 0:
            n = min(pred.shape[0], k)
            cnt = np.in1d(pred[:n], lab, assume_unique=assume_unique).sum()
            return float(cnt) / k
        else:
            return _warn_for_empty_labels()

    return _mean_ranking_metric(predictions, labels, _inner_pk)


def mean_average_precision(predictions, labels, assume_unique=True):
    """Compute the mean average precision on predictions and labels.
    Returns the mean average precision (MAP) of all the queries. If a query
    has an empty ground truth set, the average precision will be zero and a
    warning is generated.
    Parameters
    ----------
    predictions : array-like, shape=(n_predictions,)
        The prediction array. The items that were predicted, in descending
        order of relevance.
    labels : array-like, shape=(n_ratings,)
        The labels (positively-rated items).
    assume_unique : bool, optional (default=True)
        Whether to assume the items in the labels and predictions are each
        unique. That is, the same item is not predicted multiple times or
        rated multiple times.
    Examples
    --------
    >>> # predictions for 3 users
    >>> preds = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5],
    ...          [4, 1, 5, 6, 2, 7, 3, 8, 9, 10],
    ...          [1, 2, 3, 4, 5]]
    >>> # labels for the 3 users
    >>> labels = [[1, 2, 3, 4, 5], [1, 2, 3], []]
    >>> mean_average_precision(preds, labels)
    0.35502645502645497
    """
    def _inner_map(pred, lab):
        if lab.shape[0]:
            # compute the number of elements within the predictions that are
            # present in the actual labels, and get the cumulative sum weighted
            # by the index of the ranking
            n = pred.shape[0]

            # Scala code from Spark source:
            # var i = 0
            # var cnt = 0
            # var precSum = 0.0
            # val n = pred.length
            # while (i < n) {
            #     if (labSet.contains(pred(i))) {
            #         cnt += 1
            #         precSum += cnt.toDouble / (i + 1)
            #     }
            #     i += 1
            # }
            # precSum / labSet.size

            arange = np.arange(n, dtype=np.float32) + 1.  # this is the denom
            present = np.in1d(pred[:n], lab, assume_unique=assume_unique)
            prec_sum = np.ones(present.sum()).cumsum()
            denom = arange[present]
            return (prec_sum / denom).sum() / lab.shape[0]

        else:
            return _warn_for_empty_labels()

    return _mean_ranking_metric(predictions, labels, _inner_map)


def ndcg_at(predictions, labels, k=10, assume_unique=True):
    """Compute the normalized discounted cumulative gain at K.
    Compute the average NDCG value of all the queries, truncated at ranking
    position k. The discounted cumulative gain at position k is computed as:
        sum,,i=1,,^k^ (2^{relevance of ''i''th item}^ - 1) / log(i + 1)
    and the NDCG is obtained by dividing the DCG value on the ground truth set.
    In the current implementation, the relevance value is binary.
    If a query has an empty ground truth set, zero will be used as
    NDCG together with a warning.
    Parameters
    ----------
    predictions : array-like, shape=(n_predictions,)
        The prediction array. The items that were predicted, in descending
        order of relevance.
    labels : array-like, shape=(n_ratings,)
        The labels (positively-rated items).
    k : int, optional (default=10)
        The rank at which to measure the NDCG.
    assume_unique : bool, optional (default=True)
        Whether to assume the items in the labels and predictions are each
        unique. That is, the same item is not predicted multiple times or
        rated multiple times.
    Examples
    --------
    >>> # predictions for 3 users
    >>> preds = [[1, 6, 2, 7, 8, 3, 9, 10, 4, 5],
    ...          [4, 1, 5, 6, 2, 7, 3, 8, 9, 10],
    ...          [1, 2, 3, 4, 5]]
    >>> # labels for the 3 users
    >>> labels = [[1, 2, 3, 4, 5], [1, 2, 3], []]
    >>> ndcg_at(preds, labels, 3)
    0.3333333432674408
    >>> ndcg_at(preds, labels, 10)
    0.48791273434956867
    References
    ----------
    .. [1] K. Jarvelin and J. Kekalainen, "IR evaluation methods for
           retrieving highly relevant documents."
    """
    # validate K
    _require_positive_k(k)

    def _inner_ndcg(pred, lab):
        if lab.shape[0]:
            # if we do NOT assume uniqueness, the set is a bit different here
            if not assume_unique:
                lab = np.unique(lab)

            n_lab = lab.shape[0]
            n_pred = pred.shape[0]
            n = min(max(n_pred, n_lab), k)  # min(min(p, l), k)?

            # similar to mean_avg_prcsn, we need an arange, but this time +2
            # since python is zero-indexed, and the denom typically needs +1.
            # Also need the log base2...
            arange = np.arange(n, dtype=np.float32)  # length n

            # since we are only interested in the arange up to n_pred, truncate
            # if necessary
            arange = arange[:n_pred]
            denom = np.log2(arange + 2.)  # length n
            gains = 1. / denom  # length n

            # compute the gains where the prediction is present in the labels
            dcg_mask = np.in1d(pred[:n], lab, assume_unique=assume_unique)
            dcg = gains[dcg_mask].sum()

            # the max DCG is sum of gains where the index < the label set size
            max_dcg = gains[arange < n_lab].sum()
            return dcg / max_dcg

        else:
            return _warn_for_empty_labels()

    return _mean_ranking_metric(predictions, labels, _inner_ndcg)

def make_prediction(pred_idx, node_embedding, models):
    ranking = {}
    distances = []
    for s, ts in pred_idx.items():
        for t in ts:
            source = node_embedding[s]
            target = node_embedding[t]
            probs = sigmoid(np.dot(np.array(source), np.array(target).T))
            distances.append((s, t, probs))
        distances.sort(key=lambda tup: tup[2])
    ranking['sigmoid'] = distances  
    
    for operator, model in models.items():
        distances = []
        for s, ts in pred_idx.items():
            for t in ts:
                source = node_embedding[s]
                target = node_embedding[t]
                feats = np.array([get_link_score(source, target, operator)])
                probs = model.predict_proba(feats)[:, 1][0]
                distances.append((s, t, probs))
            distances.sort(key=lambda tup: tup[2])
        ranking[operator] = distances
    return ranking

def eval_ranking(pred_dict, true_dict, k):
    predicted_indices = [] # collections.OrderedDict()
    true_indices = [] # collections.OrderedDict()
    author_indices = sorted(list(true_dict.keys()))
    for aid in author_indices:
        predicted_indices.append(pred_dict[aid])
        true_indices.append(true_dict[aid])
    eval_agent = Evaluation(predicted_indices, true_indices, k)
    return eval_agent.result

class Evaluation:
    '''
    https://github.com/radinhamidi/Team_Formation_Library
    '''
    def __init__(self, predicted_indices, true_indices, max_k=20):
        self.predicted_indices = predicted_indices
        self.true_indices = true_indices
        self.max_k = max_k
        self.evaluate()
        
    def recall_at_k(self, k):
        all_recall = []
        for pred_indices, t_indices in zip(self.predicted_indices, self.true_indices):
            recall = 0
            for t_index in t_indices:
                if t_index in pred_indices[:k]:
                    recall += 1
            all_recall.append(recall / len(t_indices))
        return np.mean(all_recall), all_recall
    
    def ndcg_at(self, k):
        if k == 0:
            return 0
        else:
            return ndcg_at(self.predicted_indices, self.true_indices, k)
    
    def cal_relevance_score(self, k):
        rs = []
        for p, t in zip(self.predicted_indices, self.true_indices):
            r = []
            for p_record in p[:k]:
                if p_record in t:
                    r.append(1)
                else:
                    r.append(0)
            rs.append(r)
        return rs

    def mean_reciprocal_rank(self, rs):
        """Compute Mean Reciprocal Rank
        """
        rs = (np.asarray(r).nonzero()[0] for r in rs)
        return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])
    
    def evaluate(self):
        """Print all the metrics for evaluation
        """
        self.all_recall = []
        self.all_mrr = []
        self.all_map = []
        self.all_ndcg = []
        for k in range(0, self.max_k+1):
            # Recall:
            self.all_recall.append(self.recall_at_k(k)[0])
            # MRR:
            rs = self.cal_relevance_score(k)
            self.all_mrr.append(self.mean_reciprocal_rank(rs))
            # MAP:
            top_k_pred = [item[:k] for item in self.predicted_indices]
            self.all_map.append(mean_average_precision(top_k_pred, self.true_indices))
            # NDCG:
            self.all_ndcg.append(self.ndcg_at(k))

        self.result = pd.DataFrame({
            'k': list(range(self.max_k+1)),
            'recall': self.all_recall,
            'mrr': self.all_mrr,
            'map': self.all_map,
            'ndcg': self.all_ndcg,
        })

    def print_metrics(self, at=None):
        if at is not None:
            display(self.result[self.result['k']==at])
        else:
            display(self.result)
            
    def metric_visualization(self, max_k=20, save_path=None):
        x = list(range(max_k))
        recall = self.all_recall[:max_k]
        mrr = self.all_mrr[:max_k]
        map = self.all_map[:max_k]
        ndcg = self.all_ndcg[:max_k]
        
        fig, axs = plt.subplots(2, 2)
        # Recall plot:
        axs[0,0].set_ylim([min(recall) * 0.85, max(recall) * 1.05])
        axs[0,0].scatter(x, recall, c='red')
        axs[0,0].plot(x, recall)
        axs[0,0].grid()
        axs[0,0].set(xlabel='top-k', ylabel='recall')

        # MRR plot:
        axs[0,1].set_ylim([min(mrr) * 0.85, max(mrr) * 1.05])
        axs[0,1].scatter(x, mrr, c='red')
        axs[0,1].plot(x, mrr)
        axs[0,1].grid()
        axs[0,1].set(xlabel='top-k', ylabel='mrr')

        # MAP plot:
        axs[1,0].set_ylim([min(map) * 0.85, max(map) * 1.05])
        axs[1,0].scatter(x, map, c='red')
        axs[1,0].plot(x, map)
        axs[1,0].grid()
        axs[1,0].set(xlabel='top-k', ylabel='map')

        # NDCG plot:
        axs[1,1].set_ylim([min(ndcg) * 0.85, max(ndcg) * 1.05])
        axs[1,1].scatter(x, ndcg, c='red')
        axs[1,1].plot(x, ndcg)
        axs[1,1].grid()
        axs[1,1].set(xlabel='top-k', ylabel='ndcg')

        # Show plot
        plt.tight_layout()
        metrics_fig = plt.gcf()
        plt.show()
        plt.draw()
        
        if save_path is not None:
            metrics_fig.savefig(save_path)
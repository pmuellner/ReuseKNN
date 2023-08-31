import numpy as np
from collections import defaultdict

def dict3d_avg(listlistdict, K, n_folds):
    avg = []
    for k in range(len(K)):
        avg_at_k = dict()
        for f in range(n_folds):
            for key, value in listlistdict[f][k].items():
                avg_at_k[key] = avg_at_k.get(key, 0) + value
        for key, value in avg_at_k.items():
            avg_at_k[key] /= n_folds
        avg.append(avg_at_k)
    return avg

def avg_over_q(data, n_folds, n_ks):
    average = []
    for k in range(n_ks):
        min_queries = min([len(data[f][k]) for f in range(n_folds)])
        average.append(np.mean([data[f][k][:min_queries] for f in range(n_folds)], axis=0))

    return average

def get_groundtruth(predictions, threshold=3.5):
    relevant_items = defaultdict(list)
    for uid, iid, true_r, _, _ in predictions:
        if true_r >= threshold:
            relevant_items[uid].append(iid)

    return relevant_items
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

def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = [item_id for item_id, _ in user_ratings[:n]]

    return top_n


def get_groundtruth(predictions, threshold=3.5):
    relevant_items = defaultdict(list)
    for uid, iid, true_r, _, _ in predictions:
        if true_r >= threshold:
            relevant_items[uid].append(iid)

    return relevant_items
import numpy as np
from collections import defaultdict
from algorithms.utils import get_top_n, get_groundtruth


def avg_neighborhood_size_q(model, n_queries):
    """
    Calculates the size of each user's neighborhood after processing q recommendation queries (for all q < n_queries).
    """
    q_max = np.max([len(n_neighbors_at_q) for n_neighbors_at_q in model.n_neighbors_at_q.values()])
    n_neighbors_over_q = [0]
    n_neighbors_over_q_sample = [[0] * len(model.n_neighbors_at_q)]
    for q in range(1, min(n_queries, q_max) + 1):
        n_neighbors_at_q = []
        for iuid, n_neighbors in model.n_neighbors_at_q.items():
            if len(n_neighbors) >= q:
                n_neighbors_at_q.append(n_neighbors[q-1])
        n_neighbors_over_q.append(np.mean(n_neighbors_at_q))
        n_neighbors_over_q_sample.append(n_neighbors_at_q)

    return n_neighbors_over_q, n_neighbors_over_q_sample


def avg_rating_overlap_q(model, n_queries=100):
    """
    Calculates the rating overlap of each user's neighborhood after processing q recommendation queries (for all q < n_queries).
    """
    q_max = np.max([len(rating_overlap_at_q) for rating_overlap_at_q in model.rating_overlap_at_q.values()])
    rating_overlap_over_q = [0]
    rating_overlap_over_q_sample = [[0] * len(model.rating_overlap_at_q)]
    for q in range(1, min(n_queries, q_max) + 1):
        rating_overlap_at_q = []
        for iuid, overlap in model.rating_overlap_at_q.items():
            if len(overlap) >= q:
                rating_overlap_at_q.append(overlap[q - 1])
        rating_overlap_over_q.append(np.mean(rating_overlap_at_q))
        rating_overlap_over_q_sample.append(rating_overlap_at_q)

    return rating_overlap_over_q, rating_overlap_over_q_sample


def mean_absolute_error(model):
    """
    the mean absolute error for the predictions of all users
    """
    absolute_errors = []
    for uid, aes in model.absolute_errors.items():
        absolute_errors.extend(aes)

    return np.mean(absolute_errors), absolute_errors


def ndcg(model, n=10):
    """
    the normalized discounted cumulative gain of the top n items with the highest estimated rating score
    """
    groundtruth = get_groundtruth(model.predictions, threshold=model.trainset.global_mean)
    recommendation_lists = get_top_n(model.predictions, n=n)
    ndcgs = []
    for uid, rec_list in recommendation_lists.items():
        n = len(rec_list)
        dcg = 0.0
        for i in range(1, n + 1):
            item_at_i = rec_list[i - 1]
            rel_i = item_at_i in groundtruth[uid]
            dcg += rel_i / np.log2(i + 1)

        idcg = np.sum([1 / np.log2(i + 1) for i in range(1, n + 1)])
        ndcgs.append(dcg / idcg)

    return np.mean(ndcgs), ndcgs


def avg_neighborhood_size(model):
    sizes = [len(neighborhood) for neighborhood in model.neighbors.values()]
    return np.mean(sizes), sizes


def _top_recommendations(model, n):
    # generates a top n recommendation list
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in model.predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = [iid for iid, _ in user_ratings[:n]]

    return top_n


def recommendation_frequency(model, n):
    """
    calculates how often an item occurs in all users top n recommendations
    """
    positive_recommendations = _top_recommendations(model, n=n)
    frequencies = dict()
    for ruid, riids in positive_recommendations.items():
        for riid in riids:
            frequencies[riid] = frequencies.get(riid, 0) + 1

    return frequencies


def fraction_vulnerables(model):
    return np.mean(model.data_usage >= model.threshold)


def avg_data_usage(model):
    return np.mean(model.data_usage), model.data_usage

def avg_privacy_risk(model):
    return np.mean(model.privacy_risk), model.privacy_risk


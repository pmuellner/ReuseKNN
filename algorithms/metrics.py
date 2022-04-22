import numpy as np
from collections import defaultdict


def avg_neighborhood_size_q(model, n_queries):
    q_max = np.max([len(n_mentors_at_q) for n_mentors_at_q in model.n_mentors_at_q.values()])
    n_mentors_over_q = [0]
    n_mentors_over_q_sample = [[0] * len(model.n_mentors_at_q)]
    for q in range(1, min(n_queries, q_max) + 1):
        n_mentors_at_q = []
        for iuid, n_mentors in model.n_mentors_at_q.items():
            if len(n_mentors) >= q:
                n_mentors_at_q.append(n_mentors[q-1])
        n_mentors_over_q.append(np.mean(n_mentors_at_q))
        n_mentors_over_q_sample.append(n_mentors_at_q)

    return n_mentors_over_q, n_mentors_over_q_sample


def avg_item_coverage_q(model, n_queries):
    q_max = np.max([len(item_coverage_at_q) for item_coverage_at_q in model.item_coverage_at_q.values()])
    item_coverage_over_q = [0]
    item_coverage_over_q_sample = [[0] * len(model.item_coverage_at_q)]
    for q in range(1, min(n_queries, q_max) + 1):
        item_coverage_at_q = []
        for iuid, covered_items in model.item_coverage_at_q.items():
            if len(covered_items) >= q:
                item_coverage_at_q.append(covered_items[q-1])
        item_coverage_over_q.append(np.mean(item_coverage_at_q))
        item_coverage_over_q_sample.append(item_coverage_at_q)

    return item_coverage_over_q, item_coverage_over_q_sample


def avg_rating_overlap_q(model, n_queries):
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


def mean_absolute_error_q(model, n_queries):
    q_max = np.max([len(absolute_errors) for absolute_errors in model.absolute_errors.values()])
    mae_over_q = [0]
    mae_over_q_sample = [[0] * len(model.absolute_errors)]
    for q in range(1, min(n_queries, q_max) + 1):
        absolute_errors_at_q = []
        for iuid, absolute_errors in model.absolute_errors.items():
            if len(absolute_errors) >= q:
                if len(absolute_errors) == 1:
                    absolute_errors_at_q.append(absolute_errors[q - 1])
                else:
                    absolute_errors_at_q.extend(absolute_errors[:q - 1])
        mae_over_q.append(np.mean(absolute_errors_at_q))
        mae_over_q_sample.append(absolute_errors_at_q)

    return mae_over_q, mae_over_q_sample


def mean_absolute_error(model, users=None):
    absolute_errors = []
    if users:
        for iuid in users:
            absolute_errors.extend(model.absolute_errors[iuid])
    else:
        for uid, aes in model.absolute_errors.items():
            absolute_errors.extend(aes)

    return np.mean(absolute_errors), absolute_errors


def avg_neighborhood_size(model, users=None):
    sizes = []
    if users:
        for iuid in users:
            sizes.append(len(model.mentors[iuid]))
    else:
        sizes = [len(neighborhood) for neighborhood in model.mentors.values()]

    return np.mean(sizes), sizes


def _relevant_recommendations(model, threshold):
    relevant_riids = defaultdict(list)
    for ruid, riid, _, est, _ in model.predictions:
        if est >= threshold:
            relevant_riids[ruid].append(riid)

    return relevant_riids


def recommendation_frequency(model, threshold, users=None):
    positive_recommendations = _relevant_recommendations(model, threshold=threshold)
    if users:
        positive_recommendations = {ruid: positive_recommendations[ruid] for ruid in users}

    frequencies = dict()
    for ruid, riids in positive_recommendations.items():
        for riid in riids:
            frequencies[riid] = frequencies.get(riid, 0) + 1

    return frequencies


def avg_recommendation_popularity(model, threshold):
    item_popularities = np.zeros(model.trainset.n_items)
    for i, ratings in model.trainset.ir.items():
        item_popularities[i] = float(len(ratings)) / model.trainset.n_users

    positive_recommendations = _relevant_recommendations(model, threshold=threshold)
    avg_popularities = []
    for _, riids in positive_recommendations.items():
        avg_popularity_u = np.mean([item_popularities[model.trainset.to_inner_iid(riid)] for riid in riids])
        avg_popularities.append(avg_popularity_u)

    return np.mean(avg_popularities)


def gini_index(model, threshold):
    """frequencies = recommendation_frequency(model, threshold)
    n_lists = len(set([ruid for ruid, _, _, _, _ in model.predictions]))
    n_items = model.trainset.n_items

    print(n_lists, n_items, len(frequencies))

    d = {k+1: float(freq) / n_lists for k, freq in enumerate(sorted(frequencies.values()))}

    summation = 0.0
    #for k in range(1, n_items+1):
    for k, ratio in d.items():
        summation += (2 * k - n_items - 1) * ratio
    gini = 1 - (1. / (n_items - 1)) * summation

    return gini"""

    frequencies = recommendation_frequency(model, threshold)
    #array = np.zeros((len(frequencies)))
    array = np.zeros(model.trainset.n_items)
    for idx, freq in enumerate(sorted(frequencies.values())):
        array[idx] = freq

    #print(len(frequencies))

    array += 0.0000001  # values cannot be 0
    array = np.sort(array)  # values must be sorted
    index = np.arange(1, array.shape[0] + 1)  # index per array element
    n = array.shape[0]  # number of array elements
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))  # Gini coefficient

def gap(model, threshold):
    positive_recommendations = _relevant_recommendations(model, threshold=threshold)

    item_popularities = np.zeros(model.trainset.n_items)
    for i, ratings in model.trainset.ir.items():
        item_popularities[i] = float(len(ratings)) / model.trainset.n_users

    gap_r = np.zeros(model.trainset.n_users)
    for ruid, riids in positive_recommendations.items():
        avg_item_popularity = []
        for riid in riids:
            iiid = model.trainset.to_inner_iid(riid)
            avg_item_popularity.append(item_popularities[iiid])
        avg_item_popularity = np.mean(avg_item_popularity)
        iuid = model.trainset.to_inner_uid(ruid)
        gap_r[iuid] = avg_item_popularity

    gap_p = np.zeros(model.trainset.n_users)
    for iuid, ratings in model.trainset.ur.items():
        avg_user_popularity = np.mean([item_popularities[iiid] for iiid, _ in ratings])
        gap_p[iuid] = avg_user_popularity

    return np.mean((gap_r - gap_p) / gap_p)

def fraction_vulnerables(model, users=None):
    if users:
        return np.mean(model.privacy_risk[users] >= model.threshold[users])
    else:
        return np.mean(model.privacy_risk >= model.threshold)


def avg_privacy_risk(model, users=None):
    if users:
        privacy_risk = []
        for ruid in users:
            iuid = model.trainset.to_inner_uid(ruid)
            privacy_risk.append(model.privacy_risk[iuid])
        return np.mean(privacy_risk)
    else:
        return np.mean(model.privacy_risk), model.privacy_risk


def avg_privacy_risk_dp(model, users=None):
    if users:
        privacy_risk_dp = []
        for ruid in users:
            iuid = model.trainset.to_inner_uid(ruid)
            privacy_risk_dp.append(model.privacy_risk_dp[iuid])
        return np.mean(privacy_risk_dp)
    else:
        return np.mean(model.privacy_risk_dp), model.privacy_risk_dp


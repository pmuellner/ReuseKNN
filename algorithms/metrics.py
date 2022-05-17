import numpy as np
from collections import defaultdict
from scipy.stats import rankdata, spearmanr


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


def _top_recommendations(model, n):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in model.predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = [iid for iid, _ in user_ratings[:n]]

    return top_n


def recommendation_frequency(model, n, users=None):
    positive_recommendations = _top_recommendations(model, n=n)
    if users:
        positive_recommendations = {ruid: positive_recommendations[ruid] for ruid in users}

    frequencies = dict()
    for ruid, riids in positive_recommendations.items():
        for riid in riids:
            frequencies[riid] = frequencies.get(riid, 0) + 1

    return frequencies


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
        return np.mean(privacy_risk_dp), privacy_risk_dp
    else:
        return np.mean(model.privacy_risk_dp), model.privacy_risk_dp


import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True)
from algorithms.knn_neighborhood import UserKNN
import pandas as pd
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import KFold
import matplotlib.pyplot as plt
from datetime import datetime as dt
from collections import defaultdict
import os
import psutil
import pickle as pl
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import mannwhitneyu, norm, shapiro, ttest_ind
import sys
from algorithms.metrics import *

def mann_whitney_u_test(x, y, alternative="less"):
    u, p = mannwhitneyu(x, y, alternative=alternative)
    nx = len(x)
    ny = len(y)
    mu_u = (nx * ny) / 2
    sigma_u = np.sqrt(nx * ny * (nx + ny + 1) / 12)
    z = (u - mu_u) / sigma_u
    n = len(x) + len(y)
    r = z / np.sqrt(n)
    return u, p, r

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

def get_relevant(predictions, threshold=4):
    relevant = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        if est >= threshold:
            relevant[uid].append(iid)

    return relevant

def get_top_frac(predictions, frac=0.25):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        n = np.ceil(len(user_ratings) * frac).astype(int)
        top_n[uid] = [iid for iid, _ in user_ratings[:n]]

    return top_n

def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = [iid for iid, _ in user_ratings[:n]]

    return top_n

def run(trainset, testset, K, configuration={}):
    reuse = configuration.get("reuse", False)
    sim = configuration.get("precomputed_sim", None)
    act = configuration.get("preocmputed_act", None)
    pop = configuration.get("precomputed_pop", None)
    rr = configuration.get("precomputed_rr", None)
    gain = configuration.get("precomputed_gain", None)
    tau_1 = configuration.get("tau_1", 0) #activity
    tau_2 = configuration.get("tau_2", 0) #expect
    tau_3 = configuration.get("tau_3", 0) #rr expect
    tau_4 = configuration.get("tau_4", 0) #gain
    tau_5 = configuration.get("tau_5", 0) #rr pop
    tau_6 = configuration.get("tau_6", 0) #pop

    thresholds = configuration.get("thresholds", None)
    protected = configuration.get("protected", False)

    config_str = str({"reuse": reuse, "tau_1": tau_1, "tau_2": tau_2, "tau_3": tau_3, "tau_4": tau_4, "tau_5": tau_5, "tau_6": tau_6,
                      "precomputed_sim": sim is not None, "precomputed_act": act is not None,
                      "precomputed_pop": pop is not None, "precomputed_rr": rr is not None,
                      "precomputed_gain": gain is not None, "protected": protected})

    t0 = dt.now()
    print("Started training model with K: " + str(K) + " and " + config_str)
    results = defaultdict(list)
    for idx, k in enumerate(K):
        if thresholds is not None:
            th = thresholds[idx]
        else:
            th = 0
        model = UserKNN(k=k, reuse=reuse, precomputed_sim=sim, precomputed_act=act, precomputed_pop=pop,
                        precomputed_rr=rr, precomputed_gain=gain, tau_1=tau_1, tau_2=tau_2, tau_3=tau_3, tau_4=tau_4, tau_5=tau_5, tau_6=tau_6,
                        threshold=th, protected=protected)
        model.fit(trainset)
        predictions = model.test(testset)
        results["models"].append(model)
        results["predictions"].append(predictions)

        del model.ranking

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print("Mb: " + str(mem_info.rss / (1024 * 1024)))

    print("Training finished after " + str(dt.now() - t0))

    return results["models"], results["predictions"]

def eval_ratings(models, measurements=[]):
    results = defaultdict(list)
    for m_at_k in models:
        protected_neighbors = m_at_k.protected_neighbors()
        if "mae" in measurements:
            mae_below, mae_above, mae_all = [], [], []
            for uid, aes in m_at_k.absolute_errors.items():
                if uid in protected_neighbors:
                    mae_above.extend(aes)
                else:
                    mae_below.extend(aes)
                mae_all.extend(aes)

            if len(mae_below) > 0:
                results["mae_below"].append(np.mean(mae_below))
            else:
                results["mae_below"].append(0)

            if len(mae_above) > 0:
                results["mae_above"].append(np.mean(mae_above))
            else:
                results["mae_above"].append(0)

            results["mae_all"].append(np.mean(mae_all))
            results["mae_pointwise"].append(mae_all)
            #results["mae_pointwise"].append(mae_per_user)

        if "item_frequency" in measurements:
            item_frequencies = dict()
            #top_n = get_top_n(m_at_k.predictions, n=10)
            #top_n = get_top_frac(m_at_k.predictions, frac=0.25)
            top_n = get_relevant(m_at_k.predictions, threshold=4)
            for uid, iids in top_n.items():
                for iid in iids:
                    item_frequencies[iid] = item_frequencies.get(iid, 0) + 1

            results["item_frequency"].append(item_frequencies)

    return results

def eval_network(models, measurements=[]):
    results = defaultdict(list)
    for m_at_k in models:
        protected_neighbors = m_at_k.protected_neighbors()
        if "privacy_risk" in measurements:
            pr_below, pr_above, pr_all = [], [], []
            for uid in m_at_k.trainset.all_users():
                if uid in protected_neighbors:
                    pr_above.append(m_at_k.privacy_risk_dp[uid])
                else:
                    pr_below.append(m_at_k.privacy_risk_dp[uid])
                pr_all.append(m_at_k.privacy_risk_dp[uid])

            if len(pr_below) > 0:
                results["pr_below"].append(np.mean(pr_below))
            else:
                results["pr_below"].append(0)

            if len(pr_above) > 0:
                results["pr_above"].append(np.mean(pr_above))
            else:
                results["pr_above"].append(0)

            results["pr_all"].append(np.mean(pr_all))
            results["pr_pointwise"].append(pr_all)

        if "coverage":
            coverage = []
            for target_user, neighbors in m_at_k.mentors.items():
                coverage_target_user = set()
                for neighbor in neighbors:
                    coverage_target_user = coverage_target_user.union([iid for iid, r in m_at_k.trainset.ur[neighbor]])
                coverage.append(len(coverage_target_user))
            results["coverage"].append(np.mean(coverage))

        if "n_queries" in measurements:
            nq_below, nq_above, nq_all = [], [], []
            for uid in m_at_k.trainset.all_users():
                if uid in protected_neighbors:
                    nq_above.append(m_at_k.privacy_risk[uid])
                else:
                    nq_below.append(m_at_k.privacy_risk[uid])
                nq_all.append(m_at_k.privacy_risk[uid])

            if len(nq_below) > 0:
                results["nq_below"].append(np.mean(nq_below))
            else:
                results["nq_below"].append(0)

            if len(nq_above) > 0:
                results["nq_above"].append(np.mean(nq_above))
            else:
                results["nq_above"].append(0)

            results["nq_all"].append(np.mean(nq_all))
            results["nq_pointwise_v"].append(nq_above)
            results["nq_pointwise_s"].append(nq_below)

        if "n_neighbors" in measurements:
            q_max = np.max([len(nmentors) for nmentors in m_at_k.n_mentors_at_q.values()])
            avg_n_mentors_at_q = [0]
            n_mentors_at_q = [[0 for _ in range(m_at_k.trainset.n_users)]]
            for q in range(1, q_max + 1):
                n_at_q = []
                n = 0
                for iuid, mentors in m_at_k.n_mentors_at_q.items():
                    if len(mentors) >= q:
                        n_at_q.append(mentors[q - 1])
                        n += 1
                avg_n_mentors_at_q.append(np.mean(n_at_q))
                n_mentors_at_q.append(n_at_q)

            results["nn_pointwise"].append(n_mentors_at_q)
            results["nr_neighbors"].append(avg_n_mentors_at_q)


        if "pr_growth" in measurements:
            q_max = np.max([len(nmentors) for nmentors in m_at_k.n_mentors_at_q.values()])
            avg_pr_of_mentors_at_q = [0]
            for q in range(1, q_max + 1):
                avg_at_q = [0]
                """for iuid, mentors in m_at_k.n_mentors_at_q.items():
                    if len(mentors) >= q:
                        avg_at_q.append(m_at_k.pr_mentors_at_q[iuid][q-1])"""
                for iuid, pr in m_at_k.pr_mentors_at_q.items():
                    if len(pr) >= q:
                        avg_at_q.append(pr[q-1])
                #print(avg_at_q)
                avg_pr_of_mentors_at_q.append(np.mean(avg_at_q))
            results["pr_growth"].append(avg_pr_of_mentors_at_q)

        if "pathlength" in measurements:
            pathlength = m_at_k.get_path_length()
            results["pathlength"].append(pathlength)
        if "privacy_score" in measurements:
            ps = [m_at_k.privacy_score[uid] for uid in m_at_k.trainset.all_users() if uid not in m_at_k.protected_neighbors]
            #ps = [m_at_k.privacy_score[uid] for uid in m_at_k.trainset.all_users()]
            results["privacy_score"].append(np.mean(ps))

    return results


def size_of_groups(models):
    n_vulnerables, n_secure = [], []
    for m in models:
        V = m.protected_neighbors()
        n_vulnerables.append(len(V))
        n_secure.append(m.trainset.n_users - len(V))

    return n_secure, n_vulnerables

def identify_user_groups(trainset, size_in_frac=0.05):
    item_popularities = np.zeros(trainset.n_items)
    for iid, ratings in trainset.ir.items():
        item_popularities[iid] = float(len(ratings)) / trainset.n_users

    user_popularities = np.zeros(trainset.n_items)
    for uid, ratings in trainset.ur.items():
        user_popularities[uid] = np.mean([item_popularities[iid] for iid, _ in ratings])

    n = np.round(trainset.n_users * size_in_frac).astype(int)
    sorted_users = np.argsort(user_popularities)
    low = sorted_users[:n]
    high = sorted_users[-n:]
    med = np.argsort(user_popularities - np.median(user_popularities))[:n]

    return low, med, high



def mae_per_group(models):
    results = defaultdict(list)
    for m_at_k in models:
        avg_popularities = np.argsort(UserKNN.compute_pop(m_at_k.trainset))
        n_5p_users = np.round(m_at_k.trainset.n_users * 0.05).astype(int)
        low_users = avg_popularities[:n_5p_users]
        high_users = avg_popularities[-n_5p_users:]
        med_users = np.argsort(np.abs(avg_popularities - np.median(avg_popularities)))[:n_5p_users]

        mae_low, mae_med, mae_high = [], [], []
        for uid, aes in m_at_k.absolute_errors.items():
            if uid in low_users:
                mae_low.extend(aes)
            elif uid in med_users:
                mae_med.extend(aes)
            else:
                mae_high.extend(aes)

        results["low"].append(np.mean(mae_low))
        results["med"].append(np.mean(mae_med))
        results["high"].append(np.mean(mae_high))

        print(m_at_k.trainset.n_users, len(low_users), len(med_users), len(high_users))

    return results

def evaluate(models):
    results = dict()
    #results["mean_absolute_error"] = [mean_absolute_error(m) for m in models]
    #results["avg_neighborhood_size"] = [avg_neighborhood_size(m) for m in models]
    #results["recommendation_frequency"] = [recommendation_frequency(m, threshold=4) for m in models]
    #results["fraction_vulnerables"] = [fraction_vulnerables(m) for m in models]
    #results["avg_privacy_risk"] = [avg_privacy_risk(m) for m in models]
    #results["avg_privacy_risk_dp"] = [avg_privacy_risk_dp(m) for m in models]
    #results["avg_neighborhood_size_q"] = [avg_neighborhood_size_q(m, n_queries=100) for m in models]
    #results["avg_item_coverage_q"] = [avg_item_coverage_q(m, n_queries=100) for m in models]
    results["avg_rating_overlap_q"] = [avg_rating_overlap_q(m, n_queries=100) for m in models]
    #results["mean_absolute_error_q"] = [mean_absolute_error_q(m, n_queries=100) for m in models]

    return results


if len(sys.argv) == 3:
    NAME = sys.argv[1]
    if sys.argv[2] == "True":
        PROTECTED = True
    else:
        PROTECTED = False
else:
    NAME = "ml-100k"
    PROTECTED = True

NAME = "ml-100k"
PROTECTED = True


if NAME == "ml-100k":
    data_df = pd.read_csv("data/ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"], usecols=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(1, 5))
elif NAME == "ml-1m":
    data_df = pd.read_csv("data/ml-1m/ratings.dat", sep="::", names=["user_id", "item_id", "rating", "timestamp"], usecols=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(1, 5))
elif NAME == "goodreads":
    data_df = pd.read_csv("data/goodreads/sample.csv", sep=";", names=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(1, 5))
elif NAME == "lfm":
    data_df = pd.read_csv("data/lfm/artist_ratings.csv", sep=";", names=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(1, 1000))
elif NAME == "ciao":
    data_df = pd.read_csv("data/ciao/ciao.csv", sep=";", names=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(1, 5))
elif NAME == "douban":
    data_df = pd.read_csv("data/douban/douban.csv", sep=";", names=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(1, 5))
else:
    print("error")
    data_df = pd.DataFrame()
    reader = Reader()

if PROTECTED:
    PATH = "protected/" + NAME
else:
    PATH = "unprotected/" + NAME

print(PATH)

dataset = Dataset.load_from_df(data_df, reader=reader)
n_folds = 0
folds = KFold(n_splits=5)

K = [5, 10, 15, 20, 25, 30]
#K = [10]

mae_all_0, mae_below_0, mae_above_0, pr_below_0, pr_above_0, pr_all_0, nq_below_0, nq_above_0, nq_all_0, vulnerables_0, secure_0, nr_noisy_ratings_0 = [], [], [], [], [], [], [], [], [], [], [], []
mae_all_1, mae_below_1, mae_above_1, pr_below_1, pr_above_1, pr_all_1, nq_below_1, nq_above_1, nq_all_1, vulnerables_1, secure_1, nr_noisy_ratings_1 = [], [], [], [], [], [], [], [], [], [], [], []
mae_all_2, mae_below_2, mae_above_2, pr_below_2, pr_above_2, pr_all_2, nq_below_2, nq_above_2, nq_all_2, vulnerables_2, secure_2, nr_noisy_ratings_2 = [], [], [], [], [], [], [], [], [], [], [], []
mae_all_3, mae_below_3, mae_above_3, pr_below_3, pr_above_3, pr_all_3, nq_below_3, nq_above_3, nq_all_3, vulnerables_3, secure_3, nr_noisy_ratings_3 = [], [], [], [], [], [], [], [], [], [], [], []
mae_all_4, mae_below_4, mae_above_4, pr_below_4, pr_above_4, pr_all_4, nq_below_4, nq_above_4, nq_all_4, vulnerables_4, secure_4, nr_noisy_ratings_4 = [], [], [], [], [], [], [], [], [], [], [], []
mae_all_5, mae_below_5, mae_above_5, pr_below_5, pr_above_5, pr_all_5, nq_below_5, nq_above_5, nq_all_5, vulnerables_5, secure_5, nr_noisy_ratings_5 = [], [], [], [], [], [], [], [], [], [], [], []
mae_all_6, mae_below_6, mae_above_6, pr_below_6, pr_above_6, pr_all_6, nq_below_6, nq_above_6, nq_all_6, vulnerables_6, secure_6, nr_noisy_ratings_6 = [], [], [], [], [], [], [], [], [], [], [], []
mae_all_7, mae_below_7, mae_above_7, pr_below_7, pr_above_7, pr_all_7, nq_below_7, nq_above_7, nq_all_7, vulnerables_7, secure_7, nr_noisy_ratings_7 = [], [], [], [], [], [], [], [], [], [], [], []
n_neighbors_0, n_neighbors_1, n_neighbors_2, n_neighbors_3, n_neighbors_4, n_neighbors_5, n_neighbors_6, n_neighbors_7 = [], [], [], [], [], [], [], []
pr_growth_0, pr_growth_1, pr_growth_2, pr_growth_3, pr_growth_4, pr_growth_5, pr_growth_6, pr_growth_7 = [], [], [], [], [], [], [], []
itemfreq_0, itemfreq_1, itemfreq_2, itemfreq_3, itemfreq_4, itemfreq_5, itemfreq_6, itemfreq_7 = [], [], [], [], [], [], [], []
mae_pointwise_0, mae_pointwise_1, mae_pointwise_2, mae_pointwise_3, mae_pointwise_4, mae_pointwise_5, mae_pointwise_6, mae_pointwise_7 = [], [], [], [], [], [], [], []
pr_pointwise_0, pr_pointwise_1, pr_pointwise_2, pr_pointwise_3, pr_pointwise_4, pr_pointwise_5, pr_pointwise_6, pr_pointwise_7 = [], [], [], [], [], [], [], []
nn_pointwise_0, nn_pointwise_1, nn_pointwise_2, nn_pointwise_3, nn_pointwise_4, nn_pointwise_5, nn_pointwise_6, nn_pointwise_7 = [], [], [], [], [], [], [], []
nq_pw_v_0, nq_pw_v_1, nq_pw_v_2, nq_pw_v_3, nq_pw_v_4, nq_pw_v_5, nq_pw_v_6, nq_pw_v_7 = [], [], [], [], [], [], [], []
nq_pw_s_0, nq_pw_s_1, nq_pw_s_2, nq_pw_s_3, nq_pw_s_4, nq_pw_s_5, nq_pw_s_6, nq_pw_s_7 = [], [], [], [], [], [], [], []
coverage_0, coverage_1, coverage_2, coverage_3, coverage_4, coverage_5, coverage_6, coverage_7 = [], [], [], [], [], [], [], []

mae_all_8, mae_all_9, mae_all_10, mae_all_11 = [], [], [], []
mae_low_1, mae_low_3, mae_low_5, mae_low_8, mae_low_9, mae_low_10, mae_low_11 = [], [], [], [], [], [], []
mae_med_1, mae_med_3, mae_med_5, mae_med_8, mae_med_9, mae_med_10, mae_med_11 = [], [], [], [], [], [], []
mae_high_1, mae_high_3, mae_high_5, mae_high_8, mae_high_9, mae_high_10, mae_high_11 = [], [], [], [], [], [], []

thresholds = []
for trainset, testset in folds.split(dataset):
    sim = UserKNN.compute_similarities(trainset, min_support=1)
    pop = UserKNN.compute_popularities(trainset)
    gain = UserKNN.compute_gain(trainset)

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print("Mb: " + str(mem_info.rss / (1024 * 1024)))


    # TODO delete this
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "protected": False})
    print(evaluate(models)["avg_rating_overlap_q"][1])
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "protected": False, "tau_2": 0.5})
    print(evaluate(models)["avg_rating_overlap_q"][1])
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "protected": False, "tau_4": 0.5})
    print(evaluate(models)["avg_rating_overlap_q"][1])
    exit()
    # TODO end


    # Threshold
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "protected": False})
    threshs = [m.get_privacy_threshold() for m in models]
    thresholds.append(threshs)

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print("Mb: " + str(mem_info.rss / (1024 * 1024)))

    del models

    # KNN
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "thresholds": threshs, "protected": PROTECTED})

    resratings = eval_ratings(models, measurements=["mae", "item_frequency"])
    resnetwork = eval_network(models, measurements=["privacy_risk", "n_queries", "n_neighbors", "coverage"])
    mae_all_1.append(resratings["mae_all"])
    mae_below_1.append(resratings["mae_below"])
    mae_above_1.append(resratings["mae_above"])
    pr_all_1.append(resnetwork["pr_all"])
    pr_below_1.append(resnetwork["pr_below"])
    pr_above_1.append(resnetwork["pr_above"])
    nq_all_1.append(resnetwork["nq_all"])
    nq_below_1.append(resnetwork["nq_below"])
    nq_above_1.append(resnetwork["nq_above"])
    n_neighbors_1.append(resnetwork["nr_neighbors"])
    itemfreq_1.append(resratings["item_frequency"])
    mae_pointwise_1.append(resratings["mae_pointwise"])
    pr_pointwise_1.append(resnetwork["pr_pointwise"])
    nq_pw_v_1.append(resnetwork["nq_pointwise_v"])
    nq_pw_s_1.append(resnetwork["nq_pointwise_s"])
    nn_pointwise_1.append(resnetwork["nn_pointwise"])
    n_secure, n_vulnerables = size_of_groups(models)
    secure_1.append(n_secure)
    vulnerables_1.append(n_vulnerables)
    nr_noisy_ratings_1.append([m.nr_noisy_ratings for m in models])
    r = mae_per_group(models)
    mae_low_1.append(r["low"])
    mae_med_1.append(r["med"])
    mae_high_1.append(r["high"])
    coverage_1.append(resnetwork["coverage"])

    print(mae_all_1)
    print(mae_high_1)

    del models, resratings, resnetwork



    # KNN + no protection
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "protected": False})
    resratings = eval_ratings(models, measurements=["mae", "item_frequency"])
    resnetwork = eval_network(models, measurements=["privacy_risk", "n_queries", "n_neighbors", "coverage"])
    mae_all_7.append(resratings["mae_all"])
    mae_below_7.append(resratings["mae_below"])
    mae_above_7.append(resratings["mae_above"])
    pr_all_7.append(resnetwork["pr_all"])
    pr_below_7.append(resnetwork["pr_below"])
    pr_above_7.append(resnetwork["pr_above"])
    nq_all_7.append(resnetwork["nq_all"])
    nq_below_7.append(resnetwork["nq_below"])
    nq_above_7.append(resnetwork["nq_above"])
    n_neighbors_7.append(resnetwork["nr_neighbors"])
    itemfreq_7.append(resratings["item_frequency"])
    n_secure, n_vulnerables = size_of_groups(models)
    secure_7.append(n_secure)
    vulnerables_7.append(n_vulnerables)
    nr_noisy_ratings_7.append([m.nr_noisy_ratings for m in models])
    pr_growth_7.append(resnetwork["pr_growth"])
    mae_pointwise_7.append(resratings["mae_pointwise"])
    pr_pointwise_7.append(resnetwork["pr_pointwise"])
    nn_pointwise_7.append(resnetwork["nn_pointwise"])
    nq_pw_v_7.append(resnetwork["nq_pointwise_v"])
    nq_pw_s_7.append(resnetwork["nq_pointwise_s"])
    coverage_7.append(resnetwork["coverage"])

    del models, resratings, resnetwork

    # KNN + full protection
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "thresholds": [0 for _ in range(len(K))], "protected": True})
    resratings = eval_ratings(models, measurements=["mae", "item_frequency"])
    resnetwork = eval_network(models, measurements=["privacy_risk", "n_queries", "n_neighbors", "coverage"])
    mae_all_0.append(resratings["mae_all"])
    mae_below_0.append(resratings["mae_below"])
    mae_above_0.append(resratings["mae_above"])
    pr_all_0.append(resnetwork["pr_all"])
    pr_below_0.append(resnetwork["pr_below"])
    pr_above_0.append(resnetwork["pr_above"])
    nq_all_0.append(resnetwork["nq_all"])
    nq_below_0.append(resnetwork["nq_below"])
    nq_above_0.append(resnetwork["nq_above"])
    n_neighbors_0.append(resnetwork["nr_neighbors"])
    itemfreq_0.append(resratings["item_frequency"])
    n_secure, n_vulnerables = size_of_groups(models)
    secure_0.append(n_secure)
    vulnerables_0.append(n_vulnerables)
    nr_noisy_ratings_0.append([m.nr_noisy_ratings for m in models])
    mae_pointwise_0.append(resratings["mae_pointwise"])
    pr_pointwise_0.append(resnetwork["pr_pointwise"])
    nn_pointwise_0.append(resnetwork["nn_pointwise"])
    nq_pw_v_0.append(resnetwork["nq_pointwise_v"])
    nq_pw_s_0.append(resnetwork["nq_pointwise_s"])
    coverage_0.append(resnetwork["coverage"])

    del models, resratings, resnetwork


    # KNN + reuse
    models, _ = run(trainset, testset, K=K, configuration={"reuse": True, "precomputed_sim": sim, "thresholds": threshs, "protected": PROTECTED})
    resratings = eval_ratings(models, measurements=["mae", "item_frequency"])
    resnetwork = eval_network(models, measurements=["privacy_risk", "n_queries", "n_neighbors", "coverage"])
    mae_all_2.append(resratings["mae_all"])
    mae_below_2.append(resratings["mae_below"])
    mae_above_2.append(resratings["mae_above"])
    pr_all_2.append(resnetwork["pr_all"])
    pr_below_2.append(resnetwork["pr_below"])
    pr_above_2.append(resnetwork["pr_above"])
    nq_all_2.append(resnetwork["nq_all"])
    nq_below_2.append(resnetwork["nq_below"])
    nq_above_2.append(resnetwork["nq_above"])
    n_neighbors_2.append(resnetwork["nr_neighbors"])
    itemfreq_2.append(resratings["item_frequency"])
    n_secure, n_vulnerables = size_of_groups(models)

    secure_2.append(n_secure)
    vulnerables_2.append(n_vulnerables)
    nr_noisy_ratings_2.append([m.nr_noisy_ratings for m in models])
    mae_pointwise_2.append(resratings["mae_pointwise"])
    pr_pointwise_2.append(resnetwork["pr_pointwise"])
    nn_pointwise_2.append(resnetwork["nn_pointwise"])
    nq_pw_v_2.append(resnetwork["nq_pointwise_v"])
    nq_pw_s_2.append(resnetwork["nq_pointwise_s"])
    coverage_2.append(resnetwork["coverage"])


    del models, resratings, resnetwork

    # Popularity
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "precomputed_pop": pop, "tau_2": 0.5, "thresholds": threshs, "protected": PROTECTED})
    resratings = eval_ratings(models, measurements=["mae", "item_frequency"])
    resnetwork = eval_network(models, measurements=["privacy_risk", "n_queries", "n_neighbors", "coverage"])
    mae_all_3.append(resratings["mae_all"])
    mae_below_3.append(resratings["mae_below"])
    mae_above_3.append(resratings["mae_above"])
    pr_all_3.append(resnetwork["pr_all"])
    pr_below_3.append(resnetwork["pr_below"])
    pr_above_3.append(resnetwork["pr_above"])
    nq_all_3.append(resnetwork["nq_all"])
    nq_below_3.append(resnetwork["nq_below"])
    nq_above_3.append(resnetwork["nq_above"])
    n_neighbors_3.append(resnetwork["nr_neighbors"])
    itemfreq_3.append(resratings["item_frequency"])
    n_secure, n_vulnerables = size_of_groups(models)
    secure_3.append(n_secure)
    vulnerables_3.append(n_vulnerables)
    nr_noisy_ratings_3.append([m.nr_noisy_ratings for m in models])
    mae_pointwise_3.append(resratings["mae_pointwise"])
    pr_pointwise_3.append(resnetwork["pr_pointwise"])
    nn_pointwise_3.append(resnetwork["nn_pointwise"])
    nq_pw_v_3.append(resnetwork["nq_pointwise_v"])
    nq_pw_s_3.append(resnetwork["nq_pointwise_s"])
    coverage_3.append(resnetwork["coverage"])

    r = mae_per_group(models)
    mae_low_3.append(r["low"])
    mae_med_3.append(r["med"])
    mae_high_3.append(r["high"])


    del models, resratings, resnetwork


    # Popularity + reuse
    models, _ = run(trainset, testset, K=K, configuration={"reuse": True, "precomputed_sim": sim, "precomputed_pop": pop, "tau_2": 0.5, "thresholds": threshs, "protected": PROTECTED})
    resratings = eval_ratings(models, measurements=["mae", "item_frequency"])
    resnetwork = eval_network(models, measurements=["privacy_risk", "n_queries", "n_neighbors", "coverage"])
    mae_all_4.append(resratings["mae_all"])
    mae_below_4.append(resratings["mae_below"])
    mae_above_4.append(resratings["mae_above"])
    pr_all_4.append(resnetwork["pr_all"])
    pr_below_4.append(resnetwork["pr_below"])
    pr_above_4.append(resnetwork["pr_above"])
    nq_all_4.append(resnetwork["nq_all"])
    nq_below_4.append(resnetwork["nq_below"])
    nq_above_4.append(resnetwork["nq_above"])
    n_neighbors_4.append(resnetwork["nr_neighbors"])
    itemfreq_4.append(resratings["item_frequency"])
    n_secure, n_vulnerables = size_of_groups(models)
    secure_4.append(n_secure)
    vulnerables_4.append(n_vulnerables)
    nr_noisy_ratings_4.append([m.nr_noisy_ratings for m in models])
    mae_pointwise_4.append(resratings["mae_pointwise"])
    pr_pointwise_4.append(resnetwork["pr_pointwise"])
    nn_pointwise_4.append(resnetwork["nn_pointwise"])
    nq_pw_v_4.append(resnetwork["nq_pointwise_v"])
    nq_pw_s_4.append(resnetwork["nq_pointwise_s"])
    coverage_4.append(resnetwork["coverage"])


    del models, resratings, resnetwork

    # Gain
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "precomputed_gain": gain, "tau_4": 0.5, "thresholds": threshs, "protected": PROTECTED})
    resratings = eval_ratings(models, measurements=["mae", "item_frequency"])
    resnetwork = eval_network(models, measurements=["privacy_risk", "n_queries", "n_neighbors", "coverage"])
    mae_all_5.append(resratings["mae_all"])
    mae_below_5.append(resratings["mae_below"])
    mae_above_5.append(resratings["mae_above"])
    pr_all_5.append(resnetwork["pr_all"])
    pr_below_5.append(resnetwork["pr_below"])
    pr_above_5.append(resnetwork["pr_above"])
    nq_all_5.append(resnetwork["nq_all"])
    nq_below_5.append(resnetwork["nq_below"])
    nq_above_5.append(resnetwork["nq_above"])
    n_neighbors_5.append(resnetwork["nr_neighbors"])
    itemfreq_5.append(resratings["item_frequency"])
    n_secure, n_vulnerables = size_of_groups(models)
    secure_5.append(n_secure)
    vulnerables_5.append(n_vulnerables)
    nr_noisy_ratings_5.append([m.nr_noisy_ratings for m in models])
    mae_pointwise_5.append(resratings["mae_pointwise"])
    pr_pointwise_5.append(resnetwork["pr_pointwise"])
    nn_pointwise_5.append(resnetwork["nn_pointwise"])
    nq_pw_v_5.append(resnetwork["nq_pointwise_v"])
    nq_pw_s_5.append(resnetwork["nq_pointwise_s"])
    coverage_5.append(resnetwork["coverage"])

    r = mae_per_group(models)
    mae_low_5.append(r["low"])
    mae_med_5.append(r["med"])
    mae_high_5.append(r["high"])


    del models, resratings, resnetwork


    # Gain + reuse
    models, _ = run(trainset, testset, K=K, configuration={"reuse": True, "precomputed_sim": sim, "precomputed_gain": gain, "tau_4": 0.5, "thresholds": threshs, "protected": PROTECTED})
    resratings = eval_ratings(models, measurements=["mae", "item_frequency"])
    resnetwork = eval_network(models, measurements=["privacy_risk", "n_queries", "n_neighbors", "coverage"])
    mae_all_6.append(resratings["mae_all"])
    mae_below_6.append(resratings["mae_below"])
    mae_above_6.append(resratings["mae_above"])
    pr_all_6.append(resnetwork["pr_all"])
    pr_below_6.append(resnetwork["pr_below"])
    pr_above_6.append(resnetwork["pr_above"])
    nq_all_6.append(resnetwork["nq_all"])
    nq_below_6.append(resnetwork["nq_below"])
    nq_above_6.append(resnetwork["nq_above"])
    n_neighbors_6.append(resnetwork["nr_neighbors"])
    itemfreq_6.append(resratings["item_frequency"])
    n_secure, n_vulnerables = size_of_groups(models)
    secure_6.append(n_secure)
    vulnerables_6.append(n_vulnerables)
    nr_noisy_ratings_6.append([m.nr_noisy_ratings for m in models])
    mae_pointwise_6.append(resratings["mae_pointwise"])
    pr_pointwise_6.append(resnetwork["pr_pointwise"])
    nn_pointwise_6.append(resnetwork["nn_pointwise"])
    nq_pw_v_6.append(resnetwork["nq_pointwise_v"])
    nq_pw_s_6.append(resnetwork["nq_pointwise_s"])
    coverage_6.append(resnetwork["coverage"])

    del models, resratings, resnetwork
    del sim, gain, pop

    """
    ################################################################################################################
    # Activity
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "tau_1": 0.5, "thresholds": threshs, "protected": PROTECTED})
    resratings = eval_ratings(models, measurements=["mae"])
    mae_all_8.append(resratings["mae_all"])
    r = mae_per_group(models)
    mae_low_8.append(r["low"])
    mae_med_8.append(r["med"])
    mae_high_8.append(r["high"])

    # RR Expect
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "tau_3": 0.5, "thresholds": threshs, "protected": PROTECTED})
    resratings = eval_ratings(models, measurements=["mae"])
    mae_all_9.append(resratings["mae_all"])
    r = mae_per_group(models)
    mae_low_9.append(r["low"])
    mae_med_9.append(r["med"])
    mae_high_9.append(r["high"])

    # RR Popularity
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "tau_5": 0.5, "thresholds": threshs, "protected": PROTECTED})
    resratings = eval_ratings(models, measurements=["mae"])
    mae_all_10.append(resratings["mae_all"])
    r = mae_per_group(models)
    mae_low_10.append(r["low"])
    mae_med_10.append(r["med"])
    mae_high_10.append(r["high"])

    # Popularity
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "tau_6": 0.5, "thresholds": threshs, "protected": PROTECTED})
    resratings = eval_ratings(models, measurements=["mae"])
    mae_all_11.append(resratings["mae_all"])
    r = mae_per_group(models)
    mae_low_11.append(r["low"])
    mae_med_11.append(r["med"])
    mae_high_11.append(r["high"])
    """


    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print("Mb: " + str(mem_info.rss / (1024 * 1024)))

    """original_stdout = sys.stdout
    #f = open("results/" + PATH + "/significance_tests.txt", 'a')
    f = open("results/" + PATH + "/significance_privacy_risk.txt", 'a')
    sys.stdout = f  # Change the standard output to the file we created.

    print("=== [Accuracy] H0: DP vs. ReuseKNN ===")
    for k_idx, _ in enumerate(K):
        if not np.array_equal(mae_pointwise_1[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(mae_pointwise_1[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx], alternative="less")
            print("[one-tailed MWU] UserKNN k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] UserKNN k=%d: No Difference" % K[k_idx])
        if not np.array_equal(mae_pointwise_2[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(mae_pointwise_2[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx], alternative="less")
            print("[one-tailed MWU] UserKNN+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] UserKNN+Reuse k=%d: No Difference" % K[k_idx])
        if not np.array_equal(mae_pointwise_3[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(mae_pointwise_3[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx], alternative="less")
            print("[one-tailed MWU] Popularity k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Popularity k=%d: No Difference" % K[k_idx])
        if not np.array_equal(mae_pointwise_4[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(mae_pointwise_4[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx], alternative="less")
            print("[one-tailed MWU] Popularity+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Popularity+Reuse k=%d: No Difference" % K[k_idx])
        if not np.array_equal(mae_pointwise_5[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(mae_pointwise_5[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx], alternative="less")
            print("[one-tailed MWU] Gain k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Gain k=%d: No Difference" % K[k_idx])
        if not np.array_equal(mae_pointwise_6[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(mae_pointwise_6[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx], alternative="less")
            print("[one-tailed MWU] Gain+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Gain+Reuse k=%d: No Difference" % K[k_idx])
        print()
    for k_idx, _ in enumerate(K):
        if not np.array_equal(mae_pointwise_1[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(mae_pointwise_1[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx], alternative="two-sided")
            print("[two-tailed MWU] UserKNN k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[two-tailed MWU] UserKNN k=%d: No Difference" % K[k_idx])
        if not np.array_equal(mae_pointwise_2[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(mae_pointwise_2[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx], alternative="two-sided")
            print("[two-tailed MWU] UserKNN+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[two-tailed MWU] UserKNN+Reuse k=%d: No Difference" % K[k_idx])
        if not np.array_equal(mae_pointwise_3[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(mae_pointwise_3[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx], alternative="two-sided")
            print("[two-tailed MWU] Popularity k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[two-tailed MWU] Popularity k=%d: No Difference" % K[k_idx])
        if not np.array_equal(mae_pointwise_4[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(mae_pointwise_4[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx], alternative="two-sided")
            print("[two-tailed MWU] Popularity+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[two-tailed MWU] Popularity+Reuse k=%d: No Difference" % K[k_idx])
        if not np.array_equal(mae_pointwise_5[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(mae_pointwise_5[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx], alternative="two-sided")
            print("[two-tailed MWU] Gain k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[two-tailed MWU] Gain k=%d: No Difference" % K[k_idx])
        if not np.array_equal(mae_pointwise_6[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(mae_pointwise_6[n_folds][k_idx], mae_pointwise_0[n_folds][k_idx], alternative="two-sided")
            print("[two-tailed MWU] Gain+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[two-tailed MWU] Gain+Reuse k=%d: No Difference" % K[k_idx])
        print()

    print("=== [Accuracy] H0: UserKNN vs. ReuseKNN ===")
    for k_idx, _ in enumerate(K):
        if not np.array_equal(mae_pointwise_2[n_folds][k_idx], mae_pointwise_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(mae_pointwise_2[n_folds][k_idx], mae_pointwise_1[n_folds][k_idx], alternative="less")
            print("[one-tailed MWU] UserKNN+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] UserKNN+Reuse k=%d: No Difference" % K[k_idx])
        if not np.array_equal(mae_pointwise_3[n_folds][k_idx], mae_pointwise_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(mae_pointwise_3[n_folds][k_idx], mae_pointwise_1[n_folds][k_idx], alternative="less")
            print("[one-tailed MWU] Popularity k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Popularity k=%d: No Difference" % K[k_idx])
        if not np.array_equal(mae_pointwise_4[n_folds][k_idx], mae_pointwise_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(mae_pointwise_4[n_folds][k_idx], mae_pointwise_1[n_folds][k_idx], alternative="less")
            print("[one-tailed MWU] Popularity+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Popularity+Reuse k=%d: No Difference" % K[k_idx])
        if not np.array_equal(mae_pointwise_5[n_folds][k_idx], mae_pointwise_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(mae_pointwise_5[n_folds][k_idx], mae_pointwise_1[n_folds][k_idx], alternative="less")
            print("[one-tailed MWU] Gain k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Gain k=%d: No Difference" % K[k_idx])
        if not np.array_equal(mae_pointwise_6[n_folds][k_idx], mae_pointwise_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(mae_pointwise_6[n_folds][k_idx], mae_pointwise_1[n_folds][k_idx], alternative="less")
            print("[one-tailed MWU] Gain+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Gain+Reuse k=%d: No Difference" % K[k_idx])
        print()
    for k_idx, _ in enumerate(K):
        if not np.array_equal(mae_pointwise_2[n_folds][k_idx], mae_pointwise_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(mae_pointwise_2[n_folds][k_idx], mae_pointwise_1[n_folds][k_idx], alternative="two-sided")
            print("[two-tailed MWU] UserKNN+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] UserKNN+Reuse k=%d: No Difference" % K[k_idx])
        if not np.array_equal(mae_pointwise_2[n_folds][k_idx], mae_pointwise_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(mae_pointwise_3[n_folds][k_idx], mae_pointwise_1[n_folds][k_idx], alternative="two-sided")
            print("[two-tailed MWU] Popularity k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Popularity k=%d: No Difference" % K[k_idx])
        if not np.array_equal(mae_pointwise_2[n_folds][k_idx], mae_pointwise_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(mae_pointwise_4[n_folds][k_idx], mae_pointwise_1[n_folds][k_idx], alternative="two-sided")
            print("[two-tailed MWU] Popularity+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Popularity+Reuse k=%d: No Difference" % K[k_idx])
        if not np.array_equal(mae_pointwise_2[n_folds][k_idx], mae_pointwise_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(mae_pointwise_5[n_folds][k_idx], mae_pointwise_1[n_folds][k_idx], alternative="two-sided")
            print("[two-tailed MWU] Gain k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Gain k=%d: No Difference" % K[k_idx])
        if not np.array_equal(mae_pointwise_2[n_folds][k_idx], mae_pointwise_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(mae_pointwise_6[n_folds][k_idx], mae_pointwise_1[n_folds][k_idx], alternative="two-sided")
            print("[two-tailed MWU] Gain+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Gain+Reuse k=%d: No Difference" % K[k_idx])
        print()

    print("=== [Privacy] H0: UserKNN vs. ReuseKNN ===")
    for k_idx, _ in enumerate(K):
        if not np.array_equal(pr_pointwise_2[n_folds][k_idx], pr_pointwise_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(pr_pointwise_2[n_folds][k_idx], pr_pointwise_1[n_folds][k_idx], alternative="less")
            print("[one-tailed MWU] UserKNN+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] UserKNN+Reuse k=%d: No Difference" % K[k_idx])
        if not np.array_equal(pr_pointwise_3[n_folds][k_idx], pr_pointwise_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(pr_pointwise_3[n_folds][k_idx], pr_pointwise_1[n_folds][k_idx], alternative="less")
            print("[one-tailed MWU] Popularity k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Popularity k=%d: No Difference" % K[k_idx])
        if not np.array_equal(pr_pointwise_4[n_folds][k_idx], pr_pointwise_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(pr_pointwise_4[n_folds][k_idx], pr_pointwise_1[n_folds][k_idx], alternative="less")
            print("[one-tailed MWU] Popularity+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Popularity+Reuse k=%d: No Difference" % K[k_idx])
        if not np.array_equal(pr_pointwise_5[n_folds][k_idx], pr_pointwise_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(pr_pointwise_5[n_folds][k_idx], pr_pointwise_1[n_folds][k_idx], alternative="less")
            print("[one-tailed MWU] Gain k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Gain k=%d: No Difference" % K[k_idx])
        if not np.array_equal(pr_pointwise_6[n_folds][k_idx], pr_pointwise_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(pr_pointwise_6[n_folds][k_idx], pr_pointwise_1[n_folds][k_idx], alternative="less")
            print("[one-tailed MWU] Gain+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Gain+Reuse k=%d: No Difference" % K[k_idx])
        print()
    for k_idx, _ in enumerate(K):
        if not np.array_equal(pr_pointwise_2[n_folds][k_idx], pr_pointwise_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(pr_pointwise_2[n_folds][k_idx], pr_pointwise_1[n_folds][k_idx], alternative="two-sided")
            print("[two-tailed MWU] UserKNN+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] UserKNN+Reuse k=%d: No Difference" % K[k_idx])
        if not np.array_equal(pr_pointwise_3[n_folds][k_idx], pr_pointwise_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(pr_pointwise_3[n_folds][k_idx], pr_pointwise_1[n_folds][k_idx], alternative="two-sided")
            print("[two-tailed MWU] Popularity k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Popularity k=%d: No Difference" % K[k_idx])
        if not np.array_equal(pr_pointwise_4[n_folds][k_idx], pr_pointwise_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(pr_pointwise_4[n_folds][k_idx], pr_pointwise_1[n_folds][k_idx], alternative="two-sided")
            print("[two-tailed MWU] Popularity+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Popularity+Reuse k=%d: No Difference" % K[k_idx])
        if not np.array_equal(pr_pointwise_5[n_folds][k_idx], pr_pointwise_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(pr_pointwise_5[n_folds][k_idx], pr_pointwise_1[n_folds][k_idx], alternative="two-sided")
            print("[two-tailed MWU] Gain k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Gain k=%d: No Difference" % K[k_idx])
        if not np.array_equal(pr_pointwise_6[n_folds][k_idx], pr_pointwise_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(pr_pointwise_6[n_folds][k_idx], pr_pointwise_1[n_folds][k_idx], alternative="two-sided")
            print("[two-tailed MWU] Gain+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Gain+Reuse k=%d: No Difference" % K[k_idx])
        print()

    print("=== [Privacy Risk - Secures] H0: Popularity > UserKNN")
    for k_idx, _ in enumerate(K):
        if not np.array_equal(nq_pw_s_2[n_folds][k_idx], nq_pw_s_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(nq_pw_s_2[n_folds][k_idx], nq_pw_s_1[n_folds][k_idx], alternative="less")
            print("[one-tailed MWU] UserKNN+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] UserKNN+Reuse k=%d: No Difference" % K[k_idx])
        if not np.array_equal(nq_pw_s_3[n_folds][k_idx], nq_pw_s_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(nq_pw_s_3[n_folds][k_idx], nq_pw_s_1[n_folds][k_idx], alternative="less")
            print("[one-tailed MWU] Popularity k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Popularity k=%d: No Difference" % K[k_idx])
        if not np.array_equal(nq_pw_s_4[n_folds][k_idx], nq_pw_s_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(nq_pw_s_4[n_folds][k_idx], nq_pw_s_1[n_folds][k_idx], alternative="less")
            print("[one-tailed MWU] Popularity+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Popularity+Reuse k=%d: No Difference" % K[k_idx])
        if not np.array_equal(nq_pw_s_5[n_folds][k_idx], nq_pw_s_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(nq_pw_s_5[n_folds][k_idx], nq_pw_s_1[n_folds][k_idx], alternative="less")
            print("[one-tailed MWU] Gain k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Gain k=%d: No Difference" % K[k_idx])
        if not np.array_equal(nq_pw_s_6[n_folds][k_idx], nq_pw_s_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(nq_pw_s_6[n_folds][k_idx], nq_pw_s_1[n_folds][k_idx], alternative="less")
            print("[one-tailed MWU] Gain+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Gain+Reuse k=%d: No Difference" % K[k_idx])
        print()

    print("=== [Privacy Risk - Vulnerables] H0: Popularity < UserKNN")
    for k_idx, _ in enumerate(K):
        if not np.array_equal(nq_pw_v_2[n_folds][k_idx], nq_pw_v_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(nq_pw_v_2[n_folds][k_idx], nq_pw_v_1[n_folds][k_idx], alternative="greater")
            print("[one-tailed MWU] UserKNN+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] UserKNN+Reuse k=%d: No Difference" % K[k_idx])
        if not np.array_equal(nq_pw_v_3[n_folds][k_idx], nq_pw_v_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(nq_pw_v_3[n_folds][k_idx], nq_pw_v_1[n_folds][k_idx], alternative="greater")
            print("[one-tailed MWU] Popularity k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Popularity k=%d: No Difference" % K[k_idx])
        if not np.array_equal(nq_pw_v_4[n_folds][k_idx], nq_pw_v_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(nq_pw_v_4[n_folds][k_idx], nq_pw_v_1[n_folds][k_idx], alternative="greater")
            print("[one-tailed MWU] Popularity+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Popularity+Reuse k=%d: No Difference" % K[k_idx])
        if not np.array_equal(nq_pw_v_5[n_folds][k_idx], nq_pw_v_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(nq_pw_v_5[n_folds][k_idx], nq_pw_v_1[n_folds][k_idx], alternative="greater")
            print("[one-tailed MWU] Gain k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Gain k=%d: No Difference" % K[k_idx])
        if not np.array_equal(nq_pw_v_6[n_folds][k_idx], nq_pw_v_1[n_folds][k_idx]):
            u, p, r = mann_whitney_u_test(nq_pw_v_6[n_folds][k_idx], nq_pw_v_1[n_folds][k_idx], alternative="greater")
            print("[one-tailed MWU] Gain+Reuse k=%d: %f (U), %f (p), %f (r)" % (K[k_idx], u, p, r))
        else:
            print("[one-tailed MWU] Gain+Reuse k=%d: No Difference" % K[k_idx])


    print("=== [Neighborhood Growth] H0: UserKNN vs. ReuseKNN ===")
    for k_idx, _ in enumerate(K):
        n_queries = len(nn_pointwise_1[n_folds][k_idx])
        for q in range(1, n_queries):
            print("=== k: %d, q: %d, sample size: %d ===" % (K[k_idx], q, len(nn_pointwise_1[n_folds][k_idx][q])))
            if not np.array_equal(nn_pointwise_2[n_folds][k_idx][q], nn_pointwise_1[n_folds][k_idx][q]):
                u, p, r = mann_whitney_u_test(nn_pointwise_2[n_folds][k_idx][q], nn_pointwise_1[n_folds][k_idx][q], alternative="less")
                print("[one-tailed MWU] UserKNN+Reuse: %f (U), %f (p), %f (r)" % (u, p, r))
            else:
                print("[one-tailed MWU] UserKNN+Reuse k=%d: No Difference" % K[k_idx])
            if not np.array_equal(nn_pointwise_3[n_folds][k_idx][q], nn_pointwise_1[n_folds][k_idx][q]):
                u, p, r = mann_whitney_u_test(nn_pointwise_3[n_folds][k_idx][q], nn_pointwise_1[n_folds][k_idx][q], alternative="less")
                print("[one-tailed MWU] Popularity: %f (U), %f (p), %f (r)" % (u, p, r))
            else:
                print("[one-tailed MWU] Popularity k=%d: No Difference" % K[k_idx])
            if not np.array_equal(nn_pointwise_4[n_folds][k_idx][q], nn_pointwise_1[n_folds][k_idx][q]):
                u, p, r = mann_whitney_u_test(nn_pointwise_4[n_folds][k_idx][q], nn_pointwise_1[n_folds][k_idx][q], alternative="less")
                print("[one-tailed MWU] Popularity+Reuse: %f (U), %f (p), %f (r)" % (u, p, r))
            else:
                print("[one-tailed MWU] Popularity+Reuse k=%d: No Difference" % K[k_idx])
            if not np.array_equal(nn_pointwise_5[n_folds][k_idx][q], nn_pointwise_1[n_folds][k_idx][q]):
                u, p, r = mann_whitney_u_test(nn_pointwise_5[n_folds][k_idx][q], nn_pointwise_1[n_folds][k_idx][q], alternative="less")
                print("[one-tailed MWU] Gain: %f (U), %f (p), %f (r)" % (u, p, r))
            else:
                print("[one-tailed MWU] Gain k=%d: No Difference" % K[k_idx])
            if not np.array_equal(nn_pointwise_6[n_folds][k_idx][q], nn_pointwise_1[n_folds][k_idx][q]):
                u, p, r = mann_whitney_u_test(nn_pointwise_6[n_folds][k_idx][q], nn_pointwise_1[n_folds][k_idx][q], alternative="less")
                print("[one-tailed MWU] Gain+Reuse: %f (U), %f (p), %f (r)" % (u, p, r))
            else:
                print("[one-tailed MWU] Gain+Reuse k=%d: No Difference" % K[k_idx])
        print()

    for k_idx, _ in enumerate(K):
        n_queries = len(nn_pointwise_1[n_folds][k_idx])
        for q in range(1, n_queries):
            print("=== k: %d, q: %d, sample size: %d ===" % (K[k_idx], q, len(nn_pointwise_1[n_folds][k_idx][q])))
            if not np.array_equal(nn_pointwise_2[n_folds][k_idx][q], nn_pointwise_1[n_folds][k_idx][q]):
                u, p, r = mann_whitney_u_test(nn_pointwise_2[n_folds][k_idx][q], nn_pointwise_1[n_folds][k_idx][q], alternative="two-sided")
                print("[two-tailed MWU] UserKNN+Reuse: %f (U), %f (p), %f (r)" % (u, p, r))
            else:
                print("[two-tailed MWU] UserKNN+Reuse k=%d: No Difference" % K[k_idx])
            if not np.array_equal(nn_pointwise_3[n_folds][k_idx][q], nn_pointwise_1[n_folds][k_idx][q]):
                u, p, r = mann_whitney_u_test(nn_pointwise_3[n_folds][k_idx][q], nn_pointwise_1[n_folds][k_idx][q], alternative="two-sided")
                print("[two-tailed MWU] Popularity: %f (U), %f (p), %f (r)" % (u, p, r))
            else:
                print("[two-tailed MWU] Popularity k=%d: No Difference" % K[k_idx])
            if not np.array_equal(nn_pointwise_4[n_folds][k_idx][q], nn_pointwise_1[n_folds][k_idx][q]):
                u, p, r = mann_whitney_u_test(nn_pointwise_4[n_folds][k_idx][q], nn_pointwise_1[n_folds][k_idx][q], alternative="two-sided")
                print("[two-tailed MWU] Popularity+Reuse: %f (U), %f (p), %f (r)" % (u, p, r))
            else:
                print("[two-tailed MWU] Popularity+Reuse k=%d: No Difference" % K[k_idx])
            if not np.array_equal(nn_pointwise_5[n_folds][k_idx][q], nn_pointwise_1[n_folds][k_idx][q]):
                u, p, r = mann_whitney_u_test(nn_pointwise_5[n_folds][k_idx][q], nn_pointwise_1[n_folds][k_idx][q], alternative="two-sided")
                print("[two-tailed MWU] Gain: %f (U), %f (p), %f (r)" % (u, p, r))
            else:
                print("[two-tailed MWU] Gain k=%d: No Difference" % K[k_idx])
            if not np.array_equal(nn_pointwise_6[n_folds][k_idx][q], nn_pointwise_1[n_folds][k_idx][q]):
                u, p, r = mann_whitney_u_test(nn_pointwise_6[n_folds][k_idx][q], nn_pointwise_1[n_folds][k_idx][q], alternative="two-sided")
                print("[two-tailed MWU] Gain+Reuse: %f (U), %f (p), %f (r)" % (u, p, r))
            else:
                print("[two-tailed MWU] Gain+Reuse k=%d: No Difference" % K[k_idx])
        print()

    sys.stdout = original_stdout  # Reset the standard output to its original value
    f.close()
    """
    n_folds += 1
    break


"""avg_pr0, avg_pr1, avg_pr2, avg_pr3, avg_pr4, avg_pr5, avg_pr6, avg_pr7 = [], [], [], [], [], [], [], []
for k in range(len(K)):
    min_queries = min([len(pr_growth_0[f][k]) for f in range(n_folds)])
    avg_pr0.append(np.mean([pr_growth_0[f][k][:min_queries] for f in range(n_folds)], axis=0))

    min_queries = min([len(pr_growth_1[f][k]) for f in range(n_folds)])
    avg_pr1.append(np.mean([pr_growth_1[f][k][:min_queries] for f in range(n_folds)], axis=0))

    min_queries = min([len(pr_growth_2[f][k]) for f in range(n_folds)])
    avg_pr2.append(np.mean([pr_growth_2[f][k][:min_queries] for f in range(n_folds)], axis=0))

    min_queries = min([len(pr_growth_3[f][k]) for f in range(n_folds)])
    avg_pr3.append(np.mean([pr_growth_3[f][k][:min_queries] for f in range(n_folds)], axis=0))

    min_queries = min([len(pr_growth_4[f][k]) for f in range(n_folds)])
    avg_pr4.append(np.mean([pr_growth_4[f][k][:min_queries] for f in range(n_folds)], axis=0))

    min_queries = min([len(pr_growth_5[f][k]) for f in range(n_folds)])
    avg_pr5.append(np.mean([pr_growth_5[f][k][:min_queries] for f in range(n_folds)], axis=0))

    min_queries = min([len(pr_growth_6[f][k]) for f in range(n_folds)])
    avg_pr6.append(np.mean([pr_growth_6[f][k][:min_queries] for f in range(n_folds)], axis=0))

    min_queries = min([len(pr_growth_7[f][k]) for f in range(n_folds)])
    avg_pr7.append(np.mean([pr_growth_7[f][k][:min_queries] for f in range(n_folds)], axis=0))"""

avg_n_neighbors0, avg_n_neighbors1, avg_n_neighbors2, avg_n_neighbors3, avg_n_neighbors4, avg_n_neighbors5, avg_n_neighbors6, avg_n_neighbors7 = [], [], [], [], [], [], [], []
for k in range(len(K)):
    min_queries = min([len(n_neighbors_0[f][k]) for f in range(n_folds)])
    avg_n_neighbors0.append(np.mean([n_neighbors_0[f][k][:min_queries] for f in range(n_folds)], axis=0))

    min_queries = min([len(n_neighbors_1[f][k]) for f in range(n_folds)])
    avg_n_neighbors1.append(np.mean([n_neighbors_1[f][k][:min_queries] for f in range(n_folds)], axis=0))

    min_queries = min([len(n_neighbors_2[f][k]) for f in range(n_folds)])
    avg_n_neighbors2.append(np.mean([n_neighbors_2[f][k][:min_queries] for f in range(n_folds)], axis=0))

    min_queries = min([len(n_neighbors_3[f][k]) for f in range(n_folds)])
    avg_n_neighbors3.append(np.mean([n_neighbors_3[f][k][:min_queries] for f in range(n_folds)], axis=0))

    min_queries = min([len(n_neighbors_4[f][k]) for f in range(n_folds)])
    avg_n_neighbors4.append(np.mean([n_neighbors_4[f][k][:min_queries] for f in range(n_folds)], axis=0))

    min_queries = min([len(n_neighbors_5[f][k]) for f in range(n_folds)])
    avg_n_neighbors5.append(np.mean([n_neighbors_5[f][k][:min_queries] for f in range(n_folds)], axis=0))

    min_queries = min([len(n_neighbors_6[f][k]) for f in range(n_folds)])
    avg_n_neighbors6.append(np.mean([n_neighbors_6[f][k][:min_queries] for f in range(n_folds)], axis=0))

    min_queries = min([len(n_neighbors_7[f][k]) for f in range(n_folds)])
    avg_n_neighbors7.append(np.mean([n_neighbors_7[f][k][:min_queries] for f in range(n_folds)], axis=0))


f = open("results/" + PATH + "/item_frequency_userknn_full.pkl", "wb")
pl.dump(dict3d_avg(itemfreq_0, n_folds=n_folds, K=K), f)
f = open("results/" + PATH + "/item_frequency_userknn.pkl", "wb")
pl.dump(dict3d_avg(itemfreq_1, n_folds=n_folds, K=K), f)
f = open("results/" + PATH + "/item_frequency_pop.pkl", "wb")
pl.dump(dict3d_avg(itemfreq_3, n_folds=n_folds, K=K), f)
f = open("results/" + PATH + "/item_frequency_gain.pkl", "wb")
pl.dump(dict3d_avg(itemfreq_5, n_folds=n_folds, K=K), f)
f = open("results/" + PATH + "/item_frequency_userknn_reuse.pkl", "wb")
pl.dump(dict3d_avg(itemfreq_2, n_folds=n_folds, K=K), f)
f = open("results/" + PATH + "/item_frequency_pop_reuse.pkl", "wb")
pl.dump(dict3d_avg(itemfreq_4, n_folds=n_folds, K=K), f)
f = open("results/" + PATH + "/item_frequency_gain_reuse.pkl", "wb")
pl.dump(dict3d_avg(itemfreq_6, n_folds=n_folds, K=K), f)
f = open("results/" + PATH + "/item_frequency_userknn_no.pkl", "wb")
pl.dump(dict3d_avg(itemfreq_7, n_folds=n_folds, K=K), f)
f.close()

np.save("results/" + PATH + "/nr_neighbors_userknn_full.npy", avg_n_neighbors0)
np.save("results/" + PATH + "/nr_neighbors_userknn.npy", avg_n_neighbors1)
np.save("results/" + PATH + "/nr_neighbors_pop.npy", avg_n_neighbors3)
np.save("results/" + PATH + "/nr_neighbors_gain.npy", avg_n_neighbors5)
np.save("results/" + PATH + "/nr_neighbors_userknn_reuse.npy", avg_n_neighbors2)
np.save("results/" + PATH + "/nr_neighbors_pop_reuse.npy", avg_n_neighbors4)
np.save("results/" + PATH + "/nr_neighbors_gain_reuse.npy", avg_n_neighbors6)
np.save("results/" + PATH + "/nr_neighbors_userknn_no.npy", avg_n_neighbors7)

np.save("results/" + PATH + "/K.npy", K)
np.save("results/" + PATH + "/thresholds.npy", np.mean(thresholds, axis=0))

np.save("results/" + PATH + "/coverage_userknn_full.npy", np.mean(coverage_0, axis=0))
np.save("results/" + PATH + "/coverage_userknn.npy", np.mean(coverage_1, axis=0))
np.save("results/" + PATH + "/coverage_pop.npy", np.mean(coverage_3, axis=0))
np.save("results/" + PATH + "/coverage_gain.npy", np.mean(coverage_5, axis=0))
np.save("results/" + PATH + "/coverage_userknn_reuse.npy", np.mean(coverage_2, axis=0))
np.save("results/" + PATH + "/coverage_pop_reuse.npy", np.mean(coverage_4, axis=0))
np.save("results/" + PATH + "/coverage_gain_reuse.npy", np.mean(coverage_6, axis=0))
np.save("results/" + PATH + "/coverage_userknn_no.npy", np.mean(coverage_7, axis=0))

np.save("results/" + PATH + "/mae_all_userknn_full.npy", np.mean(mae_all_0, axis=0))
np.save("results/" + PATH + "/mae_all_userknn.npy", np.mean(mae_all_1, axis=0))
np.save("results/" + PATH + "/mae_all_pop.npy", np.mean(mae_all_3, axis=0))
np.save("results/" + PATH + "/mae_all_gain.npy", np.mean(mae_all_5, axis=0))
np.save("results/" + PATH + "/mae_all_userknn_reuse.npy", np.mean(mae_all_2, axis=0))
np.save("results/" + PATH + "/mae_all_pop_reuse.npy", np.mean(mae_all_4, axis=0))
np.save("results/" + PATH + "/mae_all_gain_reuse.npy", np.mean(mae_all_6, axis=0))
np.save("results/" + PATH + "/mae_all_userknn_no.npy", np.mean(mae_all_7, axis=0))

np.save("results/" + PATH + "/mae_below_userknn_full.npy", np.mean(mae_below_0, axis=0))
np.save("results/" + PATH + "/mae_below_userknn.npy", np.mean(mae_below_1, axis=0))
np.save("results/" + PATH + "/mae_below_pop.npy", np.mean(mae_below_3, axis=0))
np.save("results/" + PATH + "/mae_below_gain.npy", np.mean(mae_below_5, axis=0))
np.save("results/" + PATH + "/mae_below_userknn_reuse.npy", np.mean(mae_below_2, axis=0))
np.save("results/" + PATH + "/mae_below_pop_reuse.npy", np.mean(mae_below_4, axis=0))
np.save("results/" + PATH + "/mae_below_gain_reuse.npy", np.mean(mae_below_6, axis=0))
np.save("results/" + PATH + "/mae_below_userknn_no.npy", np.mean(mae_below_7, axis=0))

np.save("results/" + PATH + "/mae_above_userknn_full.npy", np.mean(mae_above_0, axis=0))
np.save("results/" + PATH + "/mae_above_userknn.npy", np.mean(mae_above_1, axis=0))
np.save("results/" + PATH + "/mae_above_pop.npy", np.mean(mae_above_3, axis=0))
np.save("results/" + PATH + "/mae_above_gain.npy", np.mean(mae_above_5, axis=0))
np.save("results/" + PATH + "/mae_above_userknn_reuse.npy", np.mean(mae_above_5, axis=0))
np.save("results/" + PATH + "/mae_above_pop_reuse.npy", np.mean(mae_above_4, axis=0))
np.save("results/" + PATH + "/mae_above_gain_reuse.npy", np.mean(mae_above_6, axis=0))
np.save("results/" + PATH + "/mae_above_userknn_no.npy", np.mean(mae_above_7, axis=0))

np.save("results/" + PATH + "/pr_all_userknn_full.npy", np.mean(pr_all_0, axis=0))
np.save("results/" + PATH + "/pr_all_userknn.npy", np.mean(pr_all_1, axis=0))
np.save("results/" + PATH + "/pr_all_pop.npy", np.mean(pr_all_3, axis=0))
np.save("results/" + PATH + "/pr_all_gain.npy", np.mean(pr_all_5, axis=0))
np.save("results/" + PATH + "/pr_all_userknn_reuse.npy", np.mean(pr_all_2, axis=0))
np.save("results/" + PATH + "/pr_all_pop_reuse.npy", np.mean(pr_all_4, axis=0))
np.save("results/" + PATH + "/pr_all_gain_reuse.npy", np.mean(pr_all_6, axis=0))
np.save("results/" + PATH + "/pr_all_userknn_no.npy", np.mean(pr_all_7, axis=0))

np.save("results/" + PATH + "/pr_below_userknn_full.npy", np.mean(pr_below_0, axis=0))
np.save("results/" + PATH + "/pr_below_userknn.npy", np.mean(pr_below_1, axis=0))
np.save("results/" + PATH + "/pr_below_pop.npy", np.mean(pr_below_3, axis=0))
np.save("results/" + PATH + "/pr_below_gain.npy", np.mean(pr_below_5, axis=0))
np.save("results/" + PATH + "/pr_below_userknn_reuse.npy", np.mean(pr_below_2, axis=0))
np.save("results/" + PATH + "/pr_below_pop_reuse.npy", np.mean(pr_below_4, axis=0))
np.save("results/" + PATH + "/pr_below_gain_reuse.npy", np.mean(pr_below_6, axis=0))
np.save("results/" + PATH + "/pr_below_userknn_no.npy", np.mean(pr_below_7, axis=0))

np.save("results/" + PATH + "/pr_above_userknn_full.npy", np.mean(pr_above_0, axis=0))
np.save("results/" + PATH + "/pr_above_userknn.npy", np.mean(pr_above_1, axis=0))
np.save("results/" + PATH + "/pr_above_pop.npy", np.mean(pr_above_3, axis=0))
np.save("results/" + PATH + "/pr_above_gain.npy", np.mean(pr_above_5, axis=0))
np.save("results/" + PATH + "/pr_above_userknn_reuse.npy", np.mean(pr_above_2, axis=0))
np.save("results/" + PATH + "/pr_above_pop_reuse.npy", np.mean(pr_above_4, axis=0))
np.save("results/" + PATH + "/pr_above_gain_reuse.npy", np.mean(pr_above_6, axis=0))
np.save("results/" + PATH + "/pr_above_userknn_no.npy", np.mean(pr_above_7, axis=0))

np.save("results/" + PATH + "/nq_all_userknn_full.npy", np.mean(nq_all_0, axis=0))
np.save("results/" + PATH + "/nq_all_userknn.npy", np.mean(nq_all_1, axis=0))
np.save("results/" + PATH + "/nq_all_pop.npy", np.mean(nq_all_3, axis=0))
np.save("results/" + PATH + "/nq_all_gain.npy", np.mean(nq_all_5, axis=0))
np.save("results/" + PATH + "/nq_all_userknn_reuse.npy", np.mean(nq_all_2, axis=0))
np.save("results/" + PATH + "/nq_all_pop_reuse.npy", np.mean(nq_all_4, axis=0))
np.save("results/" + PATH + "/nq_all_gain_reuse.npy", np.mean(nq_all_6, axis=0))
np.save("results/" + PATH + "/nq_all_userknn_no.npy", np.mean(nq_all_7, axis=0))

np.save("results/" + PATH + "/nq_below_userknn_full.npy", np.mean(nq_below_0, axis=0))
np.save("results/" + PATH + "/nq_below_userknn.npy", np.mean(nq_below_1, axis=0))
np.save("results/" + PATH + "/nq_below_pop.npy", np.mean(nq_below_3, axis=0))
np.save("results/" + PATH + "/nq_below_gain.npy", np.mean(nq_below_5, axis=0))
np.save("results/" + PATH + "/nq_below_userknn_reuse.npy", np.mean(nq_below_2, axis=0))
np.save("results/" + PATH + "/nq_below_pop_reuse.npy", np.mean(nq_below_4, axis=0))
np.save("results/" + PATH + "/nq_below_gain_reuse.npy", np.mean(nq_below_6, axis=0))
np.save("results/" + PATH + "/nq_below_userknn_no.npy", np.mean(nq_below_7, axis=0))

np.save("results/" + PATH + "/nq_above_userknn_full.npy", np.mean(nq_above_0, axis=0))
np.save("results/" + PATH + "/nq_above_userknn.npy", np.mean(nq_above_1, axis=0))
np.save("results/" + PATH + "/nq_above_pop.npy", np.mean(nq_above_3, axis=0))
np.save("results/" + PATH + "/nq_above_gain.npy", np.mean(nq_above_5, axis=0))
np.save("results/" + PATH + "/nq_above_userknn_reuse.npy", np.mean(nq_above_2, axis=0))
np.save("results/" + PATH + "/nq_above_pop_reuse.npy", np.mean(nq_above_4, axis=0))
np.save("results/" + PATH + "/nq_above_gain_reuse.npy", np.mean(nq_above_6, axis=0))
np.save("results/" + PATH + "/nq_above_userknn_no.npy", np.mean(nq_above_7, axis=0))

np.save("results/" + PATH + "/secures_userknn_full.npy", np.mean(secure_0, axis=0))
np.save("results/" + PATH + "/secures_userknn.npy", np.mean(secure_1, axis=0))
np.save("results/" + PATH + "/secures_pop.npy", np.mean(secure_3, axis=0))
np.save("results/" + PATH + "/secures_gain.npy", np.mean(secure_5, axis=0))
np.save("results/" + PATH + "/secures_userknn_reuse.npy", np.mean(secure_2, axis=0))
np.save("results/" + PATH + "/secures_pop_reuse.npy", np.mean(secure_4, axis=0))
np.save("results/" + PATH + "/secures_gain_reuse.npy", np.mean(secure_6, axis=0))
np.save("results/" + PATH + "/secures_userknn_no.npy", np.mean(secure_7, axis=0))

np.save("results/" + PATH + "/vulnerables_userknn_full.npy", np.mean(vulnerables_0, axis=0))
np.save("results/" + PATH + "/vulnerables_userknn.npy", np.mean(vulnerables_1, axis=0))
np.save("results/" + PATH + "/vulnerables_pop.npy", np.mean(vulnerables_3, axis=0))
np.save("results/" + PATH + "/vulnerables_gain.npy", np.mean(vulnerables_5, axis=0))
np.save("results/" + PATH + "/vulnerables_userknn_reuse.npy", np.mean(vulnerables_2, axis=0))
np.save("results/" + PATH + "/vulnerables_pop_reuse.npy", np.mean(vulnerables_4, axis=0))
np.save("results/" + PATH + "/vulnerables_gain_reuse.npy", np.mean(vulnerables_6, axis=0))
np.save("results/" + PATH + "/vulnerables_userknn_no.npy", np.mean(vulnerables_7, axis=0))

np.save("results/" + PATH + "/nr_noisy_ratings_userknn_full.npy", np.mean(nr_noisy_ratings_0, axis=0))
np.save("results/" + PATH + "/nr_noisy_ratings_userknn.npy", np.mean(nr_noisy_ratings_1, axis=0))
np.save("results/" + PATH + "/nr_noisy_ratings_pop.npy", np.mean(nr_noisy_ratings_3, axis=0))
np.save("results/" + PATH + "/nr_noisy_ratings_gain.npy", np.mean(nr_noisy_ratings_5, axis=0))
np.save("results/" + PATH + "/nr_noisy_ratings_userknn_reuse.npy", np.mean(nr_noisy_ratings_2, axis=0))
np.save("results/" + PATH + "/nr_noisy_ratings_pop_reuse.npy", np.mean(nr_noisy_ratings_4, axis=0))
np.save("results/" + PATH + "/nr_noisy_ratings_gain_reuse.npy", np.mean(nr_noisy_ratings_6, axis=0))
np.save("results/" + PATH + "/nr_noisy_ratings_userknn_no.npy", np.mean(nr_noisy_ratings_7, axis=0))

"""
1. KNN, 2. KNN + Reuse, 3. Popularity, 4. Popularity + Reuse, 5. Gain, 6. Gain + Reuse
"""

"""plt.figure()
plt.plot(np.mean(nr_noisy_ratings_1, axis=0), np.mean(mae_all_1, axis=0), color="C0", linestyle="dashed", label="UserKNN", alpha=0.5)
plt.plot(np.mean(nr_noisy_ratings_3, axis=0), np.mean(mae_all_3, axis=0), color="C1", linestyle="dashed", label="Popularity", alpha=0.5)
plt.plot(np.mean(nr_noisy_ratings_5, axis=0), np.mean(mae_all_5, axis=0), color="C2", linestyle="dashed", label="Gain", alpha=0.5)
plt.plot(np.mean(nr_noisy_ratings_2, axis=0), np.mean(mae_all_2, axis=0), color="C0", linestyle="solid", label="UserKNN + Reuse")
plt.plot(np.mean(nr_noisy_ratings_4, axis=0), np.mean(mae_all_4, axis=0), color="C1", linestyle="solid", label="Popularity + Reuse")
plt.plot(np.mean(nr_noisy_ratings_6, axis=0), np.mean(mae_all_6, axis=0), color="C2", linestyle="solid", label="Gain + Reuse")
plt.ylabel("Mean absolute error")
plt.xlabel("Nr. noisy ratings")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()
"""

"""plt.figure()
plt.plot(K, np.mean(mae_all_1, axis=0), color="C0", linestyle="dashed", label="UserKNN", alpha=1)
plt.plot(K, np.mean(mae_all_3, axis=0), color="C1", linestyle="dashed", label="Expect", alpha=1)
plt.plot(K, np.mean(mae_all_5, axis=0), color="C2", linestyle="dashed", label="Gain", alpha=1)
plt.plot(K, np.mean(mae_all_8, axis=0), color="C3", linestyle="dashed", label="Activity", alpha=1)
plt.plot(K, np.mean(mae_all_9, axis=0), color="C4", linestyle="dashed", label="RR Expect", alpha=1)
plt.plot(K, np.mean(mae_all_10, axis=0), color="C5", linestyle="dashed", label="RR Popularity", alpha=1)
plt.plot(K, np.mean(mae_all_11, axis=0), color="C6", linestyle="dashed", label="Popularity", alpha=1)
plt.ylabel("Mean absolute error")
plt.xlabel("Nr. of neighbors")
plt.legend(ncol=3)
plt.tight_layout()
plt.title("All Users")
plt.show()

plt.figure()
plt.plot(K, np.mean(mae_high_1, axis=0), color="C0", linestyle="dashed", label="UserKNN", alpha=1)
plt.plot(K, np.mean(mae_high_3, axis=0), color="C1", linestyle="dashed", label="Expect", alpha=1)
plt.plot(K, np.mean(mae_high_5, axis=0), color="C2", linestyle="dashed", label="Gain", alpha=1)
plt.plot(K, np.mean(mae_high_8, axis=0), color="C3", linestyle="dashed", label="Activity", alpha=1)
plt.plot(K, np.mean(mae_high_9, axis=0), color="C4", linestyle="dashed", label="RR Expect", alpha=1)
plt.plot(K, np.mean(mae_high_10, axis=0), color="C5", linestyle="dashed", label="RR Popularity", alpha=1)
plt.plot(K, np.mean(mae_high_11, axis=0), color="C6", linestyle="dashed", label="Popularity", alpha=1)
plt.ylabel("Mean absolute error")
plt.xlabel("Nr. of neighbors")
plt.legend(ncol=3)
plt.tight_layout()
plt.title("High Users")
plt.show()

plt.figure()
plt.plot(K, np.mean(mae_med_1, axis=0), color="C0", linestyle="dashed", label="UserKNN", alpha=1)
plt.plot(K, np.mean(mae_med_3, axis=0), color="C1", linestyle="dashed", label="Expect", alpha=1)
plt.plot(K, np.mean(mae_med_5, axis=0), color="C2", linestyle="dashed", label="Gain", alpha=1)
plt.plot(K, np.mean(mae_med_8, axis=0), color="C3", linestyle="dashed", label="Activity", alpha=1)
plt.plot(K, np.mean(mae_med_9, axis=0), color="C4", linestyle="dashed", label="RR Expect", alpha=1)
plt.plot(K, np.mean(mae_med_10, axis=0), color="C5", linestyle="dashed", label="RR Popularity", alpha=1)
plt.plot(K, np.mean(mae_med_11, axis=0), color="C6", linestyle="dashed", label="Popularity", alpha=1)
plt.ylabel("Mean absolute error")
plt.xlabel("Nr. of neighbors")
plt.legend(ncol=3)
plt.tight_layout()
plt.title("Med Users")
plt.show()

plt.figure()
plt.plot(K, np.mean(mae_low_1, axis=0), color="C0", linestyle="dashed", label="UserKNN", alpha=1)
plt.plot(K, np.mean(mae_low_3, axis=0), color="C1", linestyle="dashed", label="Expect", alpha=1)
plt.plot(K, np.mean(mae_low_5, axis=0), color="C2", linestyle="dashed", label="Gain", alpha=1)
plt.plot(K, np.mean(mae_low_8, axis=0), color="C3", linestyle="dashed", label="Activity", alpha=1)
plt.plot(K, np.mean(mae_low_9, axis=0), color="C4", linestyle="dashed", label="RR Expect", alpha=1)
plt.plot(K, np.mean(mae_low_10, axis=0), color="C5", linestyle="dashed", label="RR Popularity", alpha=1)
plt.plot(K, np.mean(mae_low_11, axis=0), color="C6", linestyle="dashed", label="Popularity", alpha=1)
plt.ylabel("Mean absolute error")
plt.xlabel("Nr. of neighbors")
plt.legend(ncol=3)
plt.tight_layout()
plt.title("Low Users")
plt.show()"""

"""plt.figure()
plt.plot(np.mean(pr_all_1, axis=0), np.mean(mae_all_1, axis=0), color="C0", linestyle="dashed", label="UserKNN", alpha=0.5)
plt.plot(np.mean(pr_all_3, axis=0), np.mean(mae_all_3, axis=0), color="C1", linestyle="dashed", label="Popularity", alpha=0.5)
plt.plot(np.mean(pr_all_5, axis=0), np.mean(mae_all_5, axis=0), color="C2", linestyle="dashed", label="Gain", alpha=0.5)
plt.plot(np.mean(pr_all_2, axis=0), np.mean(mae_all_2, axis=0), color="C0", linestyle="solid", label="UserKNN + Reuse")
plt.plot(np.mean(pr_all_4, axis=0), np.mean(mae_all_4, axis=0), color="C1", linestyle="solid", label="Popularity + Reuse")
plt.plot(np.mean(pr_all_6, axis=0), np.mean(mae_all_6, axis=0), color="C2", linestyle="solid", label="Gain + Reuse")
plt.ylabel("Mean absolute error")
plt.xlabel("Avg. privacy risk")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()"""

"""plt.figure()
plt.plot(np.mean(exposure_all_1, axis=0), np.mean(mae_1, axis=0), color="C0", linestyle="dashed", label="UserKNN", alpha=0.5)
plt.plot(np.mean(exposure_all_3, axis=0), np.mean(mae_3, axis=0), color="C1", linestyle="dashed", label="Popularity", alpha=0.5)
plt.plot(np.mean(exposure_all_5, axis=0), np.mean(mae_5, axis=0), color="C2", linestyle="dashed", label="Gain", alpha=0.5)
plt.plot(np.mean(exposure_all_2, axis=0), np.mean(mae_2, axis=0), color="C0", linestyle="solid", label="UserKNN + Reuse")
plt.plot(np.mean(exposure_all_4, axis=0), np.mean(mae_4, axis=0), color="C1", linestyle="solid", label="Popularity + Reuse")
plt.plot(np.mean(exposure_all_6, axis=0), np.mean(mae_6, axis=0), color="C2", linestyle="solid", label="Gain + Reuse")
plt.ylabel("Mean absolute error")
plt.xlabel("Avg. Exposure")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()


plt.figure()
plt.plot(np.mean(vulnerables_1, axis=0), np.mean(mae_1, axis=0), color="C0", linestyle="dashed", label="UserKNN", alpha=0.5)
plt.plot(np.mean(vulnerables_3, axis=0), np.mean(mae_3, axis=0), color="C1", linestyle="dashed", label="Popularity", alpha=0.5)
plt.plot(np.mean(vulnerables_5, axis=0), np.mean(mae_5, axis=0), color="C2", linestyle="dashed", label="Gain", alpha=0.5)
plt.plot(np.mean(vulnerables_2, axis=0), np.mean(mae_2, axis=0), color="C0", linestyle="solid", label="UserKNN + Reuse")
plt.plot(np.mean(vulnerables_4, axis=0), np.mean(mae_4, axis=0), color="C1", linestyle="solid", label="Popularity + Reuse")
plt.plot(np.mean(vulnerables_6, axis=0), np.mean(mae_6, axis=0), color="C2", linestyle="solid", label="Gain + Reuse")
plt.ylabel("Mean absolute error")
plt.xlabel("Vulnerables")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()"""
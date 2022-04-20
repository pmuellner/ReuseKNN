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
from algorithms import metrics

def identify_user_groups(trainset, size_in_frac=0.2):
    """item_popularities = np.zeros(trainset.n_items)
    for iid, ratings in trainset.ir.items():
        item_popularities[iid] = float(len(ratings)) / trainset.n_users

    user_popularities = np.zeros(trainset.n_items)
    for uid, ratings in trainset.ur.items():
        user_popularities[uid] = np.mean([item_popularities[iid] for iid, _ in ratings])

    n = np.round(trainset.n_users * size_in_frac).astype(int)
    sorted_users = np.argsort(user_popularities)
    low = sorted_users[:n]
    high = sorted_users[-n:]
    med = np.argsort(user_popularities - np.median(user_popularities))[:n]"""

    # user groups as in "The Unfairness of Popularity Bias in Recommendation" (Abdollahpouri, Mansoury, Burke, Mobasher)
    item_popularities = np.zeros(trainset.n_items)
    for iid, ratings in trainset.ir.items():
        item_popularities[iid] = float(len(ratings)) / trainset.n_users

    n = np.round(trainset.n_items * 0.2).astype(int)
    popular_items = np.argsort(item_popularities)[-n:]
    user_popularities = np.zeros(trainset.n_users)
    for uid, ratings in trainset.ur.items():
        n_popular_items = 0
        for iid, _ in ratings:
            if iid in popular_items:
                n_popular_items += 1
        user_popularities[uid] = n_popular_items / len(ratings)

    n = np.round(trainset.n_users * size_in_frac).astype(int)
    sorted_users = np.argsort(user_popularities)
    low = sorted_users[:n]
    high = sorted_users[-n:]
    med = set(trainset.all_users()).difference(low).difference(high)

    return low, med, high

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


def run(trainset, testset, K, configuration={}):
    reuse = configuration.get("reuse", False)
    sim = configuration.get("precomputed_sim", None)
    act = configuration.get("preocmputed_act", None)
    pop = configuration.get("precomputed_pop", None)
    rr = configuration.get("precomputed_rr", None)
    gain = configuration.get("precomputed_gain", None)
    overlap = configuration.get("precomputed_overlap", None)
    rated_items = configuration.get("rated_items", None)
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
                      "precomputed_gain": gain is not None, "protected": protected,
                      "precomputed_overlap": overlap is not None, "rated_items": rated_items is not None})

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
                        threshold=th, protected=protected, precomputed_overlap=overlap, rated_items=rated_items)
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

def evaluate_user_groups(models, **kwargs):
    results = dict()
    for group, users in kwargs.items():
        results[group] = evaluate(models, users=users)

    return results


def evaluate(models, users=None):
    results = dict()
    results["mean_absolute_error"] = [metrics.mean_absolute_error(m, users=users) for m in models]
    results["recommendation_frequency"] = [metrics.recommendation_frequency(m, threshold=4, users=users) for m in models]
    results["fraction_vulnerables"] = [metrics.fraction_vulnerables(m, users=users) for m in models]
    results["avg_privacy_risk_dp"] = [metrics.avg_privacy_risk_dp(m, users=users) for m in models]
    results["avg_neighborhood_size_q"] = [metrics.avg_neighborhood_size_q(m, n_queries=100) for m in models]
    results["avg_item_coverage_q"] = [metrics.avg_item_coverage_q(m, n_queries=100) for m in models]
    results["avg_rating_overlap_q"] = [metrics.avg_rating_overlap_q(m, n_queries=100) for m in models]
    results["mean_absolute_error_q"] = [metrics.mean_absolute_error_q(m, n_queries=100) for m in models]

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
PROTECTED = False


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
folds = KFold(n_splits=5, random_state=42)

K = [5, 10, 15, 20, 25, 30]
mean_absolute_error = defaultdict(list)
recommendation_frequency = defaultdict(list)
fraction_vulnerables = defaultdict(list)
privacy_risk_dp = defaultdict(list)
neighborhood_size_q = defaultdict(list)
item_coverage_q = defaultdict(list)
rating_overlap_q = defaultdict(list)
mean_absolute_error_q = defaultdict(list)
thresholds = []
for trainset, testset in folds.split(dataset):
    #sim = UserKNN.compute_similarities(trainset, min_support=1)
    #sim = UserKNN.compute_similarities(trainset, min_support=1, kind="adjusted_cosine")
    sim = UserKNN.compute_similarities(trainset, min_support=1, kind="pearson")
    print(sim)
    pop = UserKNN.compute_popularities(trainset)
    gain = UserKNN.compute_gain(trainset)
    overlap = UserKNN.compute_overlap(trainset)
    rated_items = UserKNN.compute_rated_items(trainset)

    low, med, high = identify_user_groups(trainset)

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print("Mb: " + str(mem_info.rss / (1024 * 1024)))

    # Threshold
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "precomputed_overlap": overlap, "rated_items": rated_items, "protected": False})
    threshs = [m.get_privacy_threshold() for m in models]
    thresholds.append(threshs)

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print("Mb: " + str(mem_info.rss / (1024 * 1024)))
    del models

    # KNN
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "precomputed_overlap": overlap, "rated_items": rated_items, "thresholds": threshs, "protected": PROTECTED})
    results = evaluate(models)
    mean_absolute_error["userknn"].append(results["mean_absolute_error"])
    recommendation_frequency["userknn"].append(results["recommendation_frequency"])
    fraction_vulnerables["userknn"].append(results["fraction_vulnerables"])
    privacy_risk_dp["userknn"].append(results["avg_privacy_risk_dp"])
    neighborhood_size_q["userknn"].append(results["avg_neighborhood_size_q"])
    item_coverage_q["userknn"].append(results["avg_item_coverage_q"])
    rating_overlap_q["userknn"].append(results["avg_rating_overlap_q"])
    mean_absolute_error_q["userknn"].append(results["mean_absolute_error_q"])
    del models, results

    # KNN + no protection
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "precomputed_overlap": overlap, "rated_items": rated_items, "protected": False})
    results = evaluate(models)
    mean_absolute_error["userknn_no"].append(results["mean_absolute_error"])
    recommendation_frequency["userknn_no"].append(results["recommendation_frequency"])
    fraction_vulnerables["userknn_no"].append(results["fraction_vulnerables"])
    privacy_risk_dp["userknn_no"].append(results["avg_privacy_risk_dp"])
    neighborhood_size_q["userknn_no"].append(results["avg_neighborhood_size_q"])
    item_coverage_q["userknn_no"].append(results["avg_item_coverage_q"])
    rating_overlap_q["userknn_no"].append(results["avg_rating_overlap_q"])
    mean_absolute_error_q["userknn_no"].append(results["mean_absolute_error_q"])
    del models, results


    # KNN + full protection
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "precomputed_overlap": overlap, "rated_items": rated_items, "thresholds": [0 for _ in range(len(K))], "protected": True})
    results = evaluate(models)
    mean_absolute_error["userknn_full"].append(results["mean_absolute_error"])
    recommendation_frequency["userknn_full"].append(results["recommendation_frequency"])
    fraction_vulnerables["userknn_full"].append(results["fraction_vulnerables"])
    privacy_risk_dp["userknn_full"].append(results["avg_privacy_risk_dp"])
    neighborhood_size_q["userknn_full"].append(results["avg_neighborhood_size_q"])
    item_coverage_q["userknn_full"].append(results["avg_item_coverage_q"])
    rating_overlap_q["userknn_full"].append(results["avg_rating_overlap_q"])
    mean_absolute_error_q["userknn_full"].append(results["mean_absolute_error_q"])
    del models, results


    # KNN + reuse
    models, _ = run(trainset, testset, K=K, configuration={"reuse": True, "precomputed_sim": sim, "precomputed_overlap": overlap, "rated_items": rated_items, "thresholds": threshs, "protected": PROTECTED})
    results = evaluate(models)
    mean_absolute_error["userknn_reuse"].append(results["mean_absolute_error"])
    recommendation_frequency["userknn_reuse"].append(results["recommendation_frequency"])
    fraction_vulnerables["userknn_reuse"].append(results["fraction_vulnerables"])
    privacy_risk_dp["userknn_reuse"].append(results["avg_privacy_risk_dp"])
    neighborhood_size_q["userknn_reuse"].append(results["avg_neighborhood_size_q"])
    item_coverage_q["userknn_reuse"].append(results["avg_item_coverage_q"])
    rating_overlap_q["userknn_reuse"].append(results["avg_rating_overlap_q"])
    mean_absolute_error_q["userknn_reuse"].append(results["mean_absolute_error_q"])
    del models, results

    # Popularity
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "precomputed_pop": pop, "precomputed_overlap": overlap, "rated_items": rated_items, "tau_2": 0.5, "thresholds": threshs, "protected": PROTECTED})
    results = evaluate(models)
    mean_absolute_error["expect"].append(results["mean_absolute_error"])
    recommendation_frequency["expect"].append(results["recommendation_frequency"])
    fraction_vulnerables["expect"].append(results["fraction_vulnerables"])
    privacy_risk_dp["expect"].append(results["avg_privacy_risk_dp"])
    neighborhood_size_q["expect"].append(results["avg_neighborhood_size_q"])
    item_coverage_q["expect"].append(results["avg_item_coverage_q"])
    rating_overlap_q["expect"].append(results["avg_rating_overlap_q"])
    mean_absolute_error_q["expect"].append(results["mean_absolute_error_q"])
    del models, results


    # Popularity + reuse
    models, _ = run(trainset, testset, K=K, configuration={"reuse": True, "precomputed_sim": sim, "precomputed_pop": pop, "precomputed_overlap": overlap, "rated_items": rated_items, "tau_2": 0.5, "thresholds": threshs, "protected": PROTECTED})
    results = evaluate(models)
    mean_absolute_error["expect_reuse"].append(results["mean_absolute_error"])
    recommendation_frequency["expect_reuse"].append(results["recommendation_frequency"])
    fraction_vulnerables["expect_reuse"].append(results["fraction_vulnerables"])
    privacy_risk_dp["expect_reuse"].append(results["avg_privacy_risk_dp"])
    neighborhood_size_q["expect_reuse"].append(results["avg_neighborhood_size_q"])
    item_coverage_q["expect_reuse"].append(results["avg_item_coverage_q"])
    rating_overlap_q["expect_reuse"].append(results["avg_rating_overlap_q"])
    mean_absolute_error_q["expect_reuse"].append(results["mean_absolute_error_q"])
    del models, results

    # Gain
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "precomputed_gain": gain, "precomputed_overlap": overlap, "rated_items": rated_items, "tau_4": 0.5, "thresholds": threshs, "protected": PROTECTED})
    results = evaluate(models)
    mean_absolute_error["gain"].append(results["mean_absolute_error"])
    recommendation_frequency["gain"].append(results["recommendation_frequency"])
    fraction_vulnerables["gain"].append(results["fraction_vulnerables"])
    privacy_risk_dp["gain"].append(results["avg_privacy_risk_dp"])
    neighborhood_size_q["gain"].append(results["avg_neighborhood_size_q"])
    item_coverage_q["gain"].append(results["avg_item_coverage_q"])
    rating_overlap_q["gain"].append(results["avg_rating_overlap_q"])
    mean_absolute_error_q["gain"].append(results["mean_absolute_error_q"])
    del models, results


    # Gain + reuse
    models, _ = run(trainset, testset, K=K, configuration={"reuse": True, "precomputed_sim": sim, "precomputed_gain": gain, "precomputed_overlap": overlap, "rated_items": rated_items, "tau_4": 0.5, "thresholds": threshs, "protected": PROTECTED})
    results = evaluate(models)
    mean_absolute_error["gain_reuse"].append(results["mean_absolute_error"])
    recommendation_frequency["gain_reuse"].append(results["recommendation_frequency"])
    fraction_vulnerables["gain_reuse"].append(results["fraction_vulnerables"])
    privacy_risk_dp["gain_reuse"].append(results["avg_privacy_risk_dp"])
    neighborhood_size_q["gain_reuse"].append(results["avg_neighborhood_size_q"])
    item_coverage_q["gain_reuse"].append(results["avg_item_coverage_q"])
    rating_overlap_q["gain_reuse"].append(results["avg_rating_overlap_q"])
    mean_absolute_error_q["gain_reuse"].append(results["mean_absolute_error_q"])
    del models, results

    del sim, gain, pop, overlap, rated_items
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print("Mb: " + str(mem_info.rss / (1024 * 1024)))

    n_folds += 1
    break

avg_neighborhood_size_q_userknn = avg_over_q(neighborhood_size_q["userknn"], n_folds=n_folds, n_ks=len(K))
avg_neighborhood_size_q_userknn_reuse = avg_over_q(neighborhood_size_q["userknn_reuse"], n_folds=n_folds, n_ks=len(K))
avg_neighborhood_size_q_expect = avg_over_q(neighborhood_size_q["expect"], n_folds=n_folds, n_ks=len(K))
avg_neighborhood_size_q_expect_reuse = avg_over_q(neighborhood_size_q["expect_reuse"], n_folds=n_folds, n_ks=len(K))
avg_neighborhood_size_q_gain = avg_over_q(neighborhood_size_q["gain"], n_folds=n_folds, n_ks=len(K))
avg_neighborhood_size_q_gain_reuse = avg_over_q(neighborhood_size_q["gain_reuse"], n_folds=n_folds, n_ks=len(K))

avg_item_coverage_q_userknn = avg_over_q(item_coverage_q["userknn"], n_folds=n_folds, n_ks=len(K))
avg_item_coverage_q_userknn_reuse = avg_over_q(item_coverage_q["userknn_reuse"], n_folds=n_folds, n_ks=len(K))
avg_item_coverage_q_expect = avg_over_q(item_coverage_q["expect"], n_folds=n_folds, n_ks=len(K))
avg_item_coverage_q_expect_reuse = avg_over_q(item_coverage_q["expect_reuse"], n_folds=n_folds, n_ks=len(K))
avg_item_coverage_q_gain = avg_over_q(item_coverage_q["gain"], n_folds=n_folds, n_ks=len(K))
avg_item_coverage_q_gain_reuse = avg_over_q(item_coverage_q["gain_reuse"], n_folds=n_folds, n_ks=len(K))

avg_rating_overlap_q_userknn = avg_over_q(rating_overlap_q["userknn"], n_folds=n_folds, n_ks=len(K))
avg_rating_overlap_q_userknn_reuse = avg_over_q(rating_overlap_q["userknn_reuse"], n_folds=n_folds, n_ks=len(K))
avg_rating_overlap_q_expect = avg_over_q(rating_overlap_q["expect"], n_folds=n_folds, n_ks=len(K))
avg_rating_overlap_q_expect_reuse = avg_over_q(rating_overlap_q["expect_reuse"], n_folds=n_folds, n_ks=len(K))
avg_rating_overlap_q_gain = avg_over_q(rating_overlap_q["gain"], n_folds=n_folds, n_ks=len(K))
avg_rating_overlap_q_gain_reuse = avg_over_q(rating_overlap_q["gain_reuse"], n_folds=n_folds, n_ks=len(K))

avg_mae_q_userknn = avg_over_q(mean_absolute_error_q["userknn"], n_folds=n_folds, n_ks=len(K))
avg_mae_q_userknn_reuse = avg_over_q(mean_absolute_error_q["userknn_reuse"], n_folds=n_folds, n_ks=len(K))
avg_mae_q_expect = avg_over_q(mean_absolute_error_q["expect"], n_folds=n_folds, n_ks=len(K))
avg_mae_q_expect_reuse = avg_over_q(mean_absolute_error_q["expect_reuse"], n_folds=n_folds, n_ks=len(K))
avg_mae_q_gain = avg_over_q(mean_absolute_error_q["gain"], n_folds=n_folds, n_ks=len(K))
avg_mae_q_gain_reuse = avg_over_q(mean_absolute_error_q["gain_reuse"], n_folds=n_folds, n_ks=len(K))

np.save("results/" + PATH + "/K.npy", K)
np.save("results/" + PATH + "/thresholds.npy", np.mean(thresholds, axis=0))

np.save("results/" + PATH + "/neighborhood_size_q_userknn.npy", avg_neighborhood_size_q_userknn)
np.save("results/" + PATH + "/neighborhood_size_q_userknn_reuse.npy", avg_neighborhood_size_q_userknn_reuse)
np.save("results/" + PATH + "/neighborhood_size_q_expect.npy", avg_neighborhood_size_q_expect)
np.save("results/" + PATH + "/neighborhood_size_q_expect_reuse.npy", avg_neighborhood_size_q_expect_reuse)
np.save("results/" + PATH + "/neighborhood_size_q_gain.npy", avg_neighborhood_size_q_gain)
np.save("results/" + PATH + "/neighborhood_size_q_gain_reuse.npy", avg_neighborhood_size_q_gain_reuse)

np.save("results/" + PATH + "/item_coverage_q_userknn.npy", avg_item_coverage_q_userknn)
np.save("results/" + PATH + "/item_coverage_q_userknn_reuse.npy", avg_item_coverage_q_userknn_reuse)
np.save("results/" + PATH + "/item_coverage_q_expect.npy", avg_item_coverage_q_expect)
np.save("results/" + PATH + "/item_coverage_q_expect_reuse.npy", avg_item_coverage_q_expect_reuse)
np.save("results/" + PATH + "/item_coverage_q_gain.npy", avg_item_coverage_q_gain)
np.save("results/" + PATH + "/item_coverage_q_gain_reuse.npy", avg_item_coverage_q_gain_reuse)

np.save("results/" + PATH + "/rating_overlap_q_userknn.npy", avg_rating_overlap_q_userknn)
np.save("results/" + PATH + "/rating_overlap_q_userknn_reuse.npy", avg_rating_overlap_q_userknn_reuse)
np.save("results/" + PATH + "/rating_overlap_q_expect.npy", avg_rating_overlap_q_expect)
np.save("results/" + PATH + "/rating_overlap_q_expect_reuse.npy", avg_rating_overlap_q_expect_reuse)
np.save("results/" + PATH + "/rating_overlap_q_gain.npy", avg_rating_overlap_q_gain)
np.save("results/" + PATH + "/rating_overlap_q_gain_reuse.npy", avg_rating_overlap_q_gain_reuse)

np.save("results/" + PATH + "/mae_q_userknn.npy", avg_mae_q_userknn)
np.save("results/" + PATH + "/mae_q_userknn_reuse.npy", avg_mae_q_userknn_reuse)
np.save("results/" + PATH + "/mae_q_expect.npy", avg_mae_q_expect)
np.save("results/" + PATH + "/mae_q_expect_reuse.npy", avg_mae_q_expect_reuse)
np.save("results/" + PATH + "/mae_q_gain.npy", avg_mae_q_gain)
np.save("results/" + PATH + "/mae_q_gain_reuse.npy", avg_mae_q_gain_reuse)

np.save("results/" + PATH + "/mae_userknn_no.npy", np.mean(mean_absolute_error["userknn_no"], axis=0))
np.save("results/" + PATH + "/mae_userknn_full.npy", np.mean(mean_absolute_error["userknn_full"], axis=0))
np.save("results/" + PATH + "/mae_userknn.npy", np.mean(mean_absolute_error["userknn"], axis=0))
np.save("results/" + PATH + "/mae_userknn_reuse.npy", np.mean(mean_absolute_error["userknn_reuse"], axis=0))
np.save("results/" + PATH + "/mae_expect.npy", np.mean(mean_absolute_error["expect"], axis=0))
np.save("results/" + PATH + "/mae_expect_reuse.npy", np.mean(mean_absolute_error["expect_reuse"], axis=0))
np.save("results/" + PATH + "/mae_gain.npy", np.mean(mean_absolute_error["gain"], axis=0))
np.save("results/" + PATH + "/mae_gain_reuse.npy", np.mean(mean_absolute_error["gain_reuse"], axis=0))

np.save("results/" + PATH + "/privacy_risk_dp_userknn_no.npy", np.mean(privacy_risk_dp["userknn_no"], axis=0))
np.save("results/" + PATH + "/privacy_risk_dp_userknn_full.npy", np.mean(privacy_risk_dp["userknn_full"], axis=0))
np.save("results/" + PATH + "/privacy_risk_dp_userknn.npy", np.mean(privacy_risk_dp["userknn"], axis=0))
np.save("results/" + PATH + "/privacy_risk_dp_userknn_reuse.npy", np.mean(privacy_risk_dp["userknn_reuse"], axis=0))
np.save("results/" + PATH + "/privacy_risk_dp_expect.npy", np.mean(privacy_risk_dp["expect"], axis=0))
np.save("results/" + PATH + "/privacy_risk_dp_expect_reuse.npy", np.mean(privacy_risk_dp["expect_reuse"], axis=0))
np.save("results/" + PATH + "/privacy_risk_dp_gain.npy", np.mean(privacy_risk_dp["gain"], axis=0))
np.save("results/" + PATH + "/privacy_risk_dp_gain_reuse.npy", np.mean(privacy_risk_dp["gain_reuse"], axis=0))

np.save("results/" + PATH + "/fraction_vulnerables_userknn_no.npy", np.mean(fraction_vulnerables["userknn_no"], axis=0))
np.save("results/" + PATH + "/fraction_vulnerables_userknn_full.npy", np.mean(fraction_vulnerables["userknn_full"], axis=0))
np.save("results/" + PATH + "/fraction_vulnerables_userknn.npy", np.mean(fraction_vulnerables["userknn"], axis=0))
np.save("results/" + PATH + "/fraction_vulnerables_userknn_reuse.npy", np.mean(fraction_vulnerables["userknn_reuse"], axis=0))
np.save("results/" + PATH + "/fraction_vulnerables_expect.npy", np.mean(fraction_vulnerables["expect"], axis=0))
np.save("results/" + PATH + "/fraction_vulnerables_expect_reuse.npy", np.mean(fraction_vulnerables["expect_reuse"], axis=0))
np.save("results/" + PATH + "/fraction_vulnerables_gain.npy", np.mean(fraction_vulnerables["gain"], axis=0))
np.save("results/" + PATH + "/fraction_vulnerables_gain_reuse.npy", np.mean(fraction_vulnerables["gain_reuse"], axis=0))

f = open("results/" + PATH + "/recommendation_frequency_userknn_no.pkl", "wb")
pl.dump(dict3d_avg(recommendation_frequency["userknn_no"], n_folds=n_folds, K=K), f)
f = open("results/" + PATH + "/recommendation_frequency_userknn_full.pkl", "wb")
pl.dump(dict3d_avg(recommendation_frequency["userknn_full"], n_folds=n_folds, K=K), f)
f = open("results/" + PATH + "/recommendation_frequency_userknn.pkl", "wb")
pl.dump(dict3d_avg(recommendation_frequency["userknn"], n_folds=n_folds, K=K), f)
f = open("results/" + PATH + "/recommendation_frequency_userknn_reuse.pkl", "wb")
pl.dump(dict3d_avg(recommendation_frequency["userknn_reuse"], n_folds=n_folds, K=K), f)
f = open("results/" + PATH + "/recommendation_frequency_expect.pkl", "wb")
pl.dump(dict3d_avg(recommendation_frequency["expect"], n_folds=n_folds, K=K), f)
f = open("results/" + PATH + "/recommendation_frequency_expect_reuse.pkl", "wb")
pl.dump(dict3d_avg(recommendation_frequency["expect_reuse"], n_folds=n_folds, K=K), f)
f = open("results/" + PATH + "/recommendation_frequency_gain.pkl", "wb")
pl.dump(dict3d_avg(recommendation_frequency["gain"], n_folds=n_folds, K=K), f)
f = open("results/" + PATH + "/recommendation_frequency_gain_reuse.pkl", "wb")
pl.dump(dict3d_avg(recommendation_frequency["gain_reuse"], n_folds=n_folds, K=K), f)
f.close()



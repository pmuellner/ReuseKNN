import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True)
from algorithms.knn_neighborhood import UserKNN
import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import KFold
from datetime import datetime as dt
from collections import defaultdict
import os
import psutil
import pickle as pl
import sys
from algorithms import evaluation, utils

def run(trainset, testset, K, configuration={}):
    reuse = configuration.get("reuse", False)
    sim = configuration.get("precomputed_sim", None)
    pop = configuration.get("precomputed_pop", None)
    gain = configuration.get("precomputed_gain", None)
    overlap = configuration.get("precomputed_overlap", None)
    rated_items = configuration.get("rated_items", None)
    tau_2 = configuration.get("tau_2", 0) #expect
    tau_4 = configuration.get("tau_4", 0) #gain

    thresholds = configuration.get("thresholds", None)
    protected = configuration.get("protected", False)

    config_str = str({"reuse": reuse, "tau_2": tau_2, "tau_4": tau_4, "precomputed_sim": sim is not None,
                      "precomputed_pop": pop is not None, "precomputed_gain": gain is not None, "protected": protected,
                      "precomputed_overlap": overlap is not None, "rated_items": rated_items is not None})

    t0 = dt.now()
    print("Started training model with K: " + str(K) + " and " + config_str)
    results = defaultdict(list)
    for idx, k in enumerate(K):
        if thresholds is not None:
            th = thresholds[idx]
        else:
            th = 0
        model = UserKNN(k=k, reuse=reuse, precomputed_sim=sim, precomputed_pop=pop,
                        precomputed_gain=gain, tau_2=tau_2, tau_4=tau_4,
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



if len(sys.argv) == 3:
    NAME = sys.argv[1]
    if sys.argv[2] == "True":
        PROTECTED = True
    else:
        PROTECTED = False
else:
    NAME = "ml-100k"
    PROTECTED = True


if NAME == "ml-100k":
    data_df = pd.read_csv("../datasets/ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"], usecols=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(1, 5))
elif NAME == "ml-1m":
    data_df = pd.read_csv("../datasets/ml-1m/ratings.dat", sep="::", names=["user_id", "item_id", "rating", "timestamp"], usecols=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(1, 5))
elif NAME == "goodreads":
    data_df = pd.read_csv("../datasets/goodreads/sample.csv", sep=";", names=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(1, 5))
elif NAME == "lfm":
    data_df = pd.read_csv("../datasets/lfm/artist_ratings.csv", sep=";", names=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(1, 1000))
elif NAME == "ciao":
    data_df = pd.read_csv("../datasets/ciao/ciao.csv", sep=";", names=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(1, 5))
elif NAME == "douban":
    data_df = pd.read_csv("../datasets/douban/douban.csv", sep=";", names=["user_id", "item_id", "rating"])
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
K_q_idx = 1
privacy_risk = defaultdict(list)
mean_absolute_error = defaultdict(list)
ndcg = defaultdict(list)
recommendation_frequency = defaultdict(list)
fraction_vulnerables = defaultdict(list)
privacy_risk_dp = defaultdict(list)
neighborhood_size_q = defaultdict(list)
rating_overlap_q = defaultdict(list)
privacy_risk_dp_secures = defaultdict(list)
significance_test_results = defaultdict(list)
significance_test_results_full = defaultdict(list)
thresholds = []
for trainset, testset in folds.split(dataset):
    sim = UserKNN.compute_similarities(trainset, min_support=1, kind="cosine")
    pop = UserKNN.compute_popularities(trainset)
    gain = UserKNN.compute_gain(trainset)
    overlap = UserKNN.compute_overlap(trainset)
    rated_items = UserKNN.compute_rated_items(trainset)

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
    results, userknn_results_samples = evaluation.evaluate(models, [models[K_q_idx]])
    mean_absolute_error["userknn"].append(results["mean_absolute_error"])
    ndcg["userknn"].append(results["avg_ndcg"])
    recommendation_frequency["userknn"].append(results["recommendation_frequency"])
    fraction_vulnerables["userknn"].append(results["fraction_vulnerables"])
    privacy_risk_dp["userknn"].append(results["avg_privacy_risk_dp"])
    neighborhood_size_q["userknn"].append(results["avg_neighborhood_size_q"])
    rating_overlap_q["userknn"].append(results["avg_rating_overlap_q"])
    privacy_risk_dp_secures["userknn"].append(results["avg_privacy_risk_dp_secures"])
    privacy_risk["userknn"].append(userknn_results_samples["avg_privacy_risk"])
    del models, results

    # KNN + no protection
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "precomputed_overlap": overlap, "rated_items": rated_items, "thresholds": [np.inf for _ in range(len(K))], "protected": False})
    results, results_samples = evaluation.evaluate(models, [models[K_q_idx]])
    mean_absolute_error["userknn_no"].append(results["mean_absolute_error"])
    ndcg["userknn_no"].append(results["avg_ndcg"])
    recommendation_frequency["userknn_no"].append(results["recommendation_frequency"])
    fraction_vulnerables["userknn_no"].append(results["fraction_vulnerables"])
    privacy_risk_dp["userknn_no"].append(results["avg_privacy_risk_dp"])
    neighborhood_size_q["userknn_no"].append(results["avg_neighborhood_size_q"])
    rating_overlap_q["userknn_no"].append(results["avg_rating_overlap_q"])
    privacy_risk_dp_secures["userknn_no"].append(results["avg_privacy_risk_dp_secures"])
    privacy_risk["userknn_no"].append(results_samples["avg_privacy_risk"])
    significance_test_results["userknn_no"].append(evaluation.significance_tests(userknn_results_samples, results_samples))
    del models, results, results_samples

    # KNN + full protection
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "precomputed_overlap": overlap, "rated_items": rated_items, "thresholds": [0 for _ in range(len(K))], "protected": True})
    results, userknn_full_results_samples = evaluation.evaluate(models, [models[K_q_idx]])
    mean_absolute_error["userknn_full"].append(results["mean_absolute_error"])
    ndcg["userknn_full"].append(results["avg_ndcg"])
    recommendation_frequency["userknn_full"].append(results["recommendation_frequency"])
    fraction_vulnerables["userknn_full"].append(results["fraction_vulnerables"])
    privacy_risk_dp["userknn_full"].append(results["avg_privacy_risk_dp"])
    neighborhood_size_q["userknn_full"].append(results["avg_neighborhood_size_q"])
    rating_overlap_q["userknn_full"].append(results["avg_rating_overlap_q"])
    privacy_risk_dp_secures["userknn_full"].append(results["avg_privacy_risk_dp_secures"])
    privacy_risk["userknn_full"].append(userknn_full_results_samples["avg_privacy_risk"])
    significance_test_results["userknn_full"].append(evaluation.significance_tests(userknn_results_samples, userknn_full_results_samples))
    significance_test_results_full["userknn_full"].append(evaluation.significance_tests(userknn_full_results_samples, userknn_results_samples))
    del models, results

    # KNN + reuse
    models, _ = run(trainset, testset, K=K, configuration={"reuse": True, "precomputed_sim": sim, "precomputed_overlap": overlap, "rated_items": rated_items, "thresholds": threshs, "protected": PROTECTED})
    results, results_samples = evaluation.evaluate(models, [models[K_q_idx]])
    mean_absolute_error["userknn_reuse"].append(results["mean_absolute_error"])
    ndcg["userknn_reuse"].append(results["avg_ndcg"])
    recommendation_frequency["userknn_reuse"].append(results["recommendation_frequency"])
    fraction_vulnerables["userknn_reuse"].append(results["fraction_vulnerables"])
    privacy_risk_dp["userknn_reuse"].append(results["avg_privacy_risk_dp"])
    neighborhood_size_q["userknn_reuse"].append(results["avg_neighborhood_size_q"])
    rating_overlap_q["userknn_reuse"].append(results["avg_rating_overlap_q"])
    privacy_risk_dp_secures["userknn_reuse"].append(results["avg_privacy_risk_dp_secures"])
    privacy_risk["userknn_reuse"].append(results_samples["avg_privacy_risk"])
    significance_test_results["userknn_reuse"].append(evaluation.significance_tests(userknn_results_samples, results_samples))
    significance_test_results_full["userknn_reuse"].append(evaluation.significance_tests(userknn_full_results_samples, results_samples))
    del models, results, results_samples

    # Popularity
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "precomputed_pop": pop, "precomputed_overlap": overlap, "rated_items": rated_items, "tau_2": 0.5, "thresholds": threshs, "protected": PROTECTED})
    results, results_samples = evaluation.evaluate(models, [models[K_q_idx]])
    mean_absolute_error["expect"].append(results["mean_absolute_error"])
    ndcg["expect"].append(results["avg_ndcg"])
    recommendation_frequency["expect"].append(results["recommendation_frequency"])
    fraction_vulnerables["expect"].append(results["fraction_vulnerables"])
    privacy_risk_dp["expect"].append(results["avg_privacy_risk_dp"])
    neighborhood_size_q["expect"].append(results["avg_neighborhood_size_q"])
    rating_overlap_q["expect"].append(results["avg_rating_overlap_q"])
    privacy_risk_dp_secures["expect"].append(results["avg_privacy_risk_dp_secures"])
    privacy_risk["expect"].append(results_samples["avg_privacy_risk"])
    significance_test_results["expect"].append(evaluation.significance_tests(userknn_results_samples, results_samples))
    significance_test_results_full["expect"].append(evaluation.significance_tests(userknn_full_results_samples, results_samples))
    del models, results, results_samples

    # Popularity + reuse
    models, _ = run(trainset, testset, K=K, configuration={"reuse": True, "precomputed_sim": sim, "precomputed_pop": pop, "precomputed_overlap": overlap, "rated_items": rated_items, "tau_2": 0.5, "thresholds": threshs, "protected": PROTECTED})
    results, results_samples = evaluation.evaluate(models, [models[K_q_idx]])
    mean_absolute_error["expect_reuse"].append(results["mean_absolute_error"])
    ndcg["expect_reuse"].append(results["avg_ndcg"])
    recommendation_frequency["expect_reuse"].append(results["recommendation_frequency"])
    fraction_vulnerables["expect_reuse"].append(results["fraction_vulnerables"])
    privacy_risk_dp["expect_reuse"].append(results["avg_privacy_risk_dp"])
    neighborhood_size_q["expect_reuse"].append(results["avg_neighborhood_size_q"])
    rating_overlap_q["expect_reuse"].append(results["avg_rating_overlap_q"])
    privacy_risk_dp_secures["expect_reuse"].append(results["avg_privacy_risk_dp_secures"])
    privacy_risk["expect_reuse"].append(results_samples["avg_privacy_risk"])
    significance_test_results["expect_reuse"].append(evaluation.significance_tests(userknn_results_samples, results_samples))
    significance_test_results_full["expect_reuse"].append(evaluation.significance_tests(userknn_full_results_samples, results_samples))
    del models, results, results_samples

    # Gain
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "precomputed_gain": gain, "precomputed_overlap": overlap, "rated_items": rated_items, "tau_4": 0.5, "thresholds": threshs, "protected": PROTECTED})
    results, results_samples = evaluation.evaluate(models, [models[K_q_idx]])
    mean_absolute_error["gain"].append(results["mean_absolute_error"])
    ndcg["gain"].append(results["avg_ndcg"])
    recommendation_frequency["gain"].append(results["recommendation_frequency"])
    fraction_vulnerables["gain"].append(results["fraction_vulnerables"])
    privacy_risk_dp["gain"].append(results["avg_privacy_risk_dp"])
    neighborhood_size_q["gain"].append(results["avg_neighborhood_size_q"])
    rating_overlap_q["gain"].append(results["avg_rating_overlap_q"])
    privacy_risk_dp_secures["gain"].append(results["avg_privacy_risk_dp_secures"])
    privacy_risk["gain"].append(results_samples["avg_privacy_risk"])
    significance_test_results["gain"].append(evaluation.significance_tests(userknn_results_samples, results_samples))
    significance_test_results_full["gain"].append(evaluation.significance_tests(userknn_full_results_samples, results_samples))
    del models, results, results_samples

    # Gain + reuse
    models, _ = run(trainset, testset, K=K, configuration={"reuse": True, "precomputed_sim": sim, "precomputed_gain": gain, "precomputed_overlap": overlap, "rated_items": rated_items, "tau_4": 0.5, "thresholds": threshs, "protected": PROTECTED})
    results, results_samples = evaluation.evaluate(models, [models[K_q_idx]])
    mean_absolute_error["gain_reuse"].append(results["mean_absolute_error"])
    ndcg["gain_reuse"].append(results["avg_ndcg"])
    recommendation_frequency["gain_reuse"].append(results["recommendation_frequency"])
    fraction_vulnerables["gain_reuse"].append(results["fraction_vulnerables"])
    privacy_risk_dp["gain_reuse"].append(results["avg_privacy_risk_dp"])
    neighborhood_size_q["gain_reuse"].append(results["avg_neighborhood_size_q"])
    rating_overlap_q["gain_reuse"].append(results["avg_rating_overlap_q"])
    privacy_risk_dp_secures["gain_reuse"].append(results["avg_privacy_risk_dp_secures"])
    privacy_risk["gain_reuse"].append(results_samples["avg_privacy_risk"])
    significance_test_results["gain_reuse"].append(evaluation.significance_tests(userknn_results_samples, results_samples))
    significance_test_results_full["gain_reuse"].append(evaluation.significance_tests(userknn_full_results_samples, results_samples))
    del models, results, results_samples

    del sim, gain, pop, overlap, rated_items
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print("Mb: " + str(mem_info.rss / (1024 * 1024)))

    n_folds += 1
    #break

#exit()

f = open("results/" + PATH + "/privacy_risk_distribution.pkl", "wb")
pl.dump(privacy_risk, f)
f.close()


f = open("results/" + PATH + "/significance_test_results.pkl", "wb")
pl.dump(significance_test_results, f)
f.close()

f = open("results/" + PATH + "/significance_test_results_full.pkl", "wb")
pl.dump(significance_test_results_full, f)
f.close()

avg_neighborhood_size_q_userknn = utils.avg_over_q(neighborhood_size_q["userknn"], n_folds=n_folds, n_ks=1)
avg_neighborhood_size_q_userknn_reuse = utils.avg_over_q(neighborhood_size_q["userknn_reuse"], n_folds=n_folds, n_ks=1)
avg_neighborhood_size_q_expect = utils.avg_over_q(neighborhood_size_q["expect"], n_folds=n_folds, n_ks=1)
avg_neighborhood_size_q_expect_reuse = utils.avg_over_q(neighborhood_size_q["expect_reuse"], n_folds=n_folds, n_ks=1)
avg_neighborhood_size_q_gain = utils.avg_over_q(neighborhood_size_q["gain"], n_folds=n_folds, n_ks=1)
avg_neighborhood_size_q_gain_reuse = utils.avg_over_q(neighborhood_size_q["gain_reuse"], n_folds=n_folds, n_ks=1)

avg_rating_overlap_q_userknn = utils.avg_over_q(rating_overlap_q["userknn"], n_folds=n_folds, n_ks=1)
avg_rating_overlap_q_userknn_reuse = utils.avg_over_q(rating_overlap_q["userknn_reuse"], n_folds=n_folds, n_ks=1)
avg_rating_overlap_q_expect = utils.avg_over_q(rating_overlap_q["expect"], n_folds=n_folds, n_ks=1)
avg_rating_overlap_q_expect_reuse = utils.avg_over_q(rating_overlap_q["expect_reuse"], n_folds=n_folds, n_ks=1)
avg_rating_overlap_q_gain = utils.avg_over_q(rating_overlap_q["gain"], n_folds=n_folds, n_ks=1)
avg_rating_overlap_q_gain_reuse = utils.avg_over_q(rating_overlap_q["gain_reuse"], n_folds=n_folds, n_ks=1)

np.save("results/" + PATH + "/K.npy", K)
np.save("results/" + PATH + "/thresholds.npy", np.mean(thresholds, axis=0))

np.save("results/" + PATH + "/neighborhood_size_q_userknn.npy", avg_neighborhood_size_q_userknn)
np.save("results/" + PATH + "/neighborhood_size_q_userknn_reuse.npy", avg_neighborhood_size_q_userknn_reuse)
np.save("results/" + PATH + "/neighborhood_size_q_expect.npy", avg_neighborhood_size_q_expect)
np.save("results/" + PATH + "/neighborhood_size_q_expect_reuse.npy", avg_neighborhood_size_q_expect_reuse)
np.save("results/" + PATH + "/neighborhood_size_q_gain.npy", avg_neighborhood_size_q_gain)
np.save("results/" + PATH + "/neighborhood_size_q_gain_reuse.npy", avg_neighborhood_size_q_gain_reuse)

np.save("results/" + PATH + "/rating_overlap_q_userknn.npy", avg_rating_overlap_q_userknn)
np.save("results/" + PATH + "/rating_overlap_q_userknn_reuse.npy", avg_rating_overlap_q_userknn_reuse)
np.save("results/" + PATH + "/rating_overlap_q_expect.npy", avg_rating_overlap_q_expect)
np.save("results/" + PATH + "/rating_overlap_q_expect_reuse.npy", avg_rating_overlap_q_expect_reuse)
np.save("results/" + PATH + "/rating_overlap_q_gain.npy", avg_rating_overlap_q_gain)
np.save("results/" + PATH + "/rating_overlap_q_gain_reuse.npy", avg_rating_overlap_q_gain_reuse)

np.save("results/" + PATH + "/mae_userknn_no.npy", np.mean(mean_absolute_error["userknn_no"], axis=0))
np.save("results/" + PATH + "/mae_userknn_full.npy", np.mean(mean_absolute_error["userknn_full"], axis=0))
np.save("results/" + PATH + "/mae_userknn.npy", np.mean(mean_absolute_error["userknn"], axis=0))
np.save("results/" + PATH + "/mae_userknn_reuse.npy", np.mean(mean_absolute_error["userknn_reuse"], axis=0))
np.save("results/" + PATH + "/mae_expect.npy", np.mean(mean_absolute_error["expect"], axis=0))
np.save("results/" + PATH + "/mae_expect_reuse.npy", np.mean(mean_absolute_error["expect_reuse"], axis=0))
np.save("results/" + PATH + "/mae_gain.npy", np.mean(mean_absolute_error["gain"], axis=0))
np.save("results/" + PATH + "/mae_gain_reuse.npy", np.mean(mean_absolute_error["gain_reuse"], axis=0))

np.save("results/" + PATH + "/ndcg_userknn_no.npy", np.mean(ndcg["userknn_no"], axis=0))
np.save("results/" + PATH + "/ndcg_userknn_full.npy", np.mean(ndcg["userknn_full"], axis=0))
np.save("results/" + PATH + "/ndcg_userknn.npy", np.mean(ndcg["userknn"], axis=0))
np.save("results/" + PATH + "/ndcg_userknn_reuse.npy", np.mean(ndcg["userknn_reuse"], axis=0))
np.save("results/" + PATH + "/ndcg_expect.npy", np.mean(ndcg["expect"], axis=0))
np.save("results/" + PATH + "/ndcg_expect_reuse.npy", np.mean(ndcg["expect_reuse"], axis=0))
np.save("results/" + PATH + "/ndcg_gain.npy", np.mean(ndcg["gain"], axis=0))
np.save("results/" + PATH + "/ndcg_gain_reuse.npy", np.mean(ndcg["gain_reuse"], axis=0))

np.save("results/" + PATH + "/privacy_risk_dp_userknn_no.npy", np.mean(privacy_risk_dp["userknn_no"], axis=0))
np.save("results/" + PATH + "/privacy_risk_dp_userknn_full.npy", np.mean(privacy_risk_dp["userknn_full"], axis=0))
np.save("results/" + PATH + "/privacy_risk_dp_userknn.npy", np.mean(privacy_risk_dp["userknn"], axis=0))
np.save("results/" + PATH + "/privacy_risk_dp_userknn_reuse.npy", np.mean(privacy_risk_dp["userknn_reuse"], axis=0))
np.save("results/" + PATH + "/privacy_risk_dp_expect.npy", np.mean(privacy_risk_dp["expect"], axis=0))
np.save("results/" + PATH + "/privacy_risk_dp_expect_reuse.npy", np.mean(privacy_risk_dp["expect_reuse"], axis=0))
np.save("results/" + PATH + "/privacy_risk_dp_gain.npy", np.mean(privacy_risk_dp["gain"], axis=0))
np.save("results/" + PATH + "/privacy_risk_dp_gain_reuse.npy", np.mean(privacy_risk_dp["gain_reuse"], axis=0))

np.save("results/" + PATH + "/privacy_risk_dp_secures_userknn_no.npy", np.mean(privacy_risk_dp_secures["userknn_no"], axis=0))
np.save("results/" + PATH + "/privacy_risk_dp_secures_userknn_full.npy", np.mean(privacy_risk_dp_secures["userknn_full"], axis=0))
np.save("results/" + PATH + "/privacy_risk_dp_secures_userknn.npy", np.mean(privacy_risk_dp_secures["userknn"], axis=0))
np.save("results/" + PATH + "/privacy_risk_dp_secures_userknn_reuse.npy", np.mean(privacy_risk_dp_secures["userknn_reuse"], axis=0))
np.save("results/" + PATH + "/privacy_risk_dp_secures_expect.npy", np.mean(privacy_risk_dp_secures["expect"], axis=0))
np.save("results/" + PATH + "/privacy_risk_dp_secures_expect_reuse.npy", np.mean(privacy_risk_dp_secures["expect_reuse"], axis=0))
np.save("results/" + PATH + "/privacy_risk_dp_secures_gain.npy", np.mean(privacy_risk_dp_secures["gain"], axis=0))
np.save("results/" + PATH + "/privacy_risk_dp_secures_gain_reuse.npy", np.mean(privacy_risk_dp_secures["gain_reuse"], axis=0))

np.save("results/" + PATH + "/fraction_vulnerables_userknn_no.npy", np.mean(fraction_vulnerables["userknn_no"], axis=0))
np.save("results/" + PATH + "/fraction_vulnerables_userknn_full.npy", np.mean(fraction_vulnerables["userknn_full"], axis=0))
np.save("results/" + PATH + "/fraction_vulnerables_userknn.npy", np.mean(fraction_vulnerables["userknn"], axis=0))
np.save("results/" + PATH + "/fraction_vulnerables_userknn_reuse.npy", np.mean(fraction_vulnerables["userknn_reuse"], axis=0))
np.save("results/" + PATH + "/fraction_vulnerables_expect.npy", np.mean(fraction_vulnerables["expect"], axis=0))
np.save("results/" + PATH + "/fraction_vulnerables_expect_reuse.npy", np.mean(fraction_vulnerables["expect_reuse"], axis=0))
np.save("results/" + PATH + "/fraction_vulnerables_gain.npy", np.mean(fraction_vulnerables["gain"], axis=0))
np.save("results/" + PATH + "/fraction_vulnerables_gain_reuse.npy", np.mean(fraction_vulnerables["gain_reuse"], axis=0))

f = open("results/" + PATH + "/recommendation_frequency_userknn_no.pkl", "wb")
pl.dump(utils.dict3d_avg(recommendation_frequency["userknn_no"], n_folds=n_folds, K=K), f)
f = open("results/" + PATH + "/recommendation_frequency_userknn_full.pkl", "wb")
pl.dump(utils.dict3d_avg(recommendation_frequency["userknn_full"], n_folds=n_folds, K=K), f)
f = open("results/" + PATH + "/recommendation_frequency_userknn.pkl", "wb")
pl.dump(utils.dict3d_avg(recommendation_frequency["userknn"], n_folds=n_folds, K=K), f)
f = open("results/" + PATH + "/recommendation_frequency_userknn_reuse.pkl", "wb")
pl.dump(utils.dict3d_avg(recommendation_frequency["userknn_reuse"], n_folds=n_folds, K=K), f)
f = open("results/" + PATH + "/recommendation_frequency_expect.pkl", "wb")
pl.dump(utils.dict3d_avg(recommendation_frequency["expect"], n_folds=n_folds, K=K), f)
f = open("results/" + PATH + "/recommendation_frequency_expect_reuse.pkl", "wb")
pl.dump(utils.dict3d_avg(recommendation_frequency["expect_reuse"], n_folds=n_folds, K=K), f)
f = open("results/" + PATH + "/recommendation_frequency_gain.pkl", "wb")
pl.dump(utils.dict3d_avg(recommendation_frequency["gain"], n_folds=n_folds, K=K), f)
f = open("results/" + PATH + "/recommendation_frequency_gain_reuse.pkl", "wb")
pl.dump(utils.dict3d_avg(recommendation_frequency["gain_reuse"], n_folds=n_folds, K=K), f)
f.close()



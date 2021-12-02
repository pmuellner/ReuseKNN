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
import gc
import sys
import pickle as pl
from sklearn.preprocessing import MinMaxScaler

def run(trainset, testset, K, configuration={}):
    reuse = configuration.get("reuse", False)
    sim = configuration.get("precomputed_sim", None)
    act = configuration.get("preocmputed_act", None)
    pop = configuration.get("precomputed_pop", None)
    rr = configuration.get("precomputed_rr", None)
    gain = configuration.get("precomputed_gain", None)
    tau_1 = configuration.get("tau_1", 0)
    tau_2 = configuration.get("tau_2", 0)
    tau_3 = configuration.get("tau_3", 0)
    tau_4 = configuration.get("tau_4", 0)
    thresholds = configuration.get("thresholds", None)
    protected = configuration.get("protected", False)

    config_str = str({"reuse": reuse, "tau_1": tau_1, "tau_2": tau_2, "tau_3": tau_3, "tau_4": tau_4,
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
                        precomputed_rr=rr, precomputed_gain=gain, tau_1=tau_1, tau_2=tau_2, tau_3=tau_3, tau_4=tau_4,
                        threshold=th, protected=protected)
        model.fit(trainset)
        predictions = model.test(testset)
        results["models"].append(model)
        results["predictions"].append(predictions)

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

    return results

def eval_network(models, measurements=[]):
    results = defaultdict(list)
    for m_at_k in models:
        protected_neighbors = m_at_k.protected_neighbors()
        if "privacy_risk" in measurements:
            pr_below, pr_above, pr_all = [], [], []
            for uid in m_at_k.trainset.all_users():
                if uid in protected_neighbors:
                    pr_above.append(m_at_k.privacy_risk[uid])
                else:
                    pr_below.append(m_at_k.privacy_risk[uid])
                pr_all.append(m_at_k.privacy_risk[uid])

            if len(pr_below) > 0:
                results["pr_below"].append(np.mean(pr_below))
            else:
                results["pr_below"].append(0)

            if len(pr_above) > 0:
                results["pr_above"].append(np.mean(pr_above))
            else:
                results["pr_above"].append(0)

            results["pr_all"].append(np.mean(pr_all))

        if "n_queries" in measurements:
            nq_below, nq_above, nq_all = [], [], []
            for uid in m_at_k.trainset.all_users():
                if uid in protected_neighbors:
                    nq_above.append(m_at_k.n_queries[uid])
                else:
                    nq_below.append(m_at_k.n_queries[uid])
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

        if "n_neighbors" in measurements:
            q_max = np.max([len(nmentors) for nmentors in m_at_k.n_mentors_at_q.values()])
            avg_n_mentors_at_q = [0]
            for q in range(1, q_max + 1):
                avg_at_q = []
                n = 0
                for iuid, mentors in m_at_k.n_mentors_at_q.items():
                    if len(mentors) >= q:
                        avg_at_q.append(mentors[q - 1])
                        n += 1
                avg_n_mentors_at_q.append(np.mean(avg_at_q))
            results["nr_neighbors"].append(avg_n_mentors_at_q)

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
    data_df = pd.read_csv("data/ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"], usecols=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(1, 5))
elif NAME == "ml-1m":
    data_df = pd.read_csv("data/ml-1m/ratings.dat", sep="::", names=["user_id", "item_id", "rating", "timestamp"], usecols=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(1, 5))
elif NAME == "goodreads":
    data_df = pd.read_csv("data/goodreads/sample.csv", sep=";", names=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(1, 5))
elif NAME == "jester":
    data_df = pd.read_csv("data/jester/sample.csv", sep=";", names=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(-10, 10))
elif NAME == "foursquare":
    data_df = pd.read_csv("data/foursquare/sample.csv", sep=";", names=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(2, 5))
else:
    print("error")
    data_df = pd.DataFrame()
    reader = Reader()

"""data_df = pd.read_csv("data/lfm/artist_ratings.csv", sep=";", names=["user_id", "item_id", "rating"])
relevant_users = np.random.choice(data_df["user_id"].unique(), replace=False, size=1000)
data_df = data_df[data_df["user_id"].isin(relevant_users)]
reader = Reader(rating_scale=(1, 100))
NAME = "lfm"
PROTECTED = True"""

data_df = pd.read_csv("data/ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"], usecols=["user_id", "item_id", "rating"])
profile_size = data_df.groupby("user_id").size()
relevant_users = profile_size[profile_size >= 50].index.tolist()
data_df = data_df[data_df["user_id"].isin(relevant_users)]
reader = Reader(rating_scale=(1, 5))
NAME = "ml-100k"
PROTECTED = True


if PROTECTED:
    PATH = "protected/" + NAME
else:
    PATH = "unprotected/" + NAME

print(PATH)

dataset = Dataset.load_from_df(data_df, reader=reader)
n_folds = 0
folds = KFold(n_splits=5)

K = [5, 10, 15, 20, 25, 30]

mae_all_0, mae_below_0, mae_above_0, pr_below_0, pr_above_0, pr_all_0, nq_below_0, nq_above_0, nq_all_0, vulnerables_0, secure_0, nr_noisy_ratings_0 = [], [], [], [], [], [], [], [], [], [], [], []
mae_all_1, mae_below_1, mae_above_1, pr_below_1, pr_above_1, pr_all_1, nq_below_1, nq_above_1, nq_all_1, vulnerables_1, secure_1, nr_noisy_ratings_1 = [], [], [], [], [], [], [], [], [], [], [], []
mae_all_2, mae_below_2, mae_above_2, pr_below_2, pr_above_2, pr_all_2, nq_below_2, nq_above_2, nq_all_2, vulnerables_2, secure_2, nr_noisy_ratings_2 = [], [], [], [], [], [], [], [], [], [], [], []
mae_all_3, mae_below_3, mae_above_3, pr_below_3, pr_above_3, pr_all_3, nq_below_3, nq_above_3, nq_all_3, vulnerables_3, secure_3, nr_noisy_ratings_3 = [], [], [], [], [], [], [], [], [], [], [], []
mae_all_4, mae_below_4, mae_above_4, pr_below_4, pr_above_4, pr_all_4, nq_below_4, nq_above_4, nq_all_4, vulnerables_4, secure_4, nr_noisy_ratings_4 = [], [], [], [], [], [], [], [], [], [], [], []
mae_all_5, mae_below_5, mae_above_5, pr_below_5, pr_above_5, pr_all_5, nq_below_5, nq_above_5, nq_all_5, vulnerables_5, secure_5, nr_noisy_ratings_5 = [], [], [], [], [], [], [], [], [], [], [], []
mae_all_6, mae_below_6, mae_above_6, pr_below_6, pr_above_6, pr_all_6, nq_below_6, nq_above_6, nq_all_6, vulnerables_6, secure_6, nr_noisy_ratings_6 = [], [], [], [], [], [], [], [], [], [], [], []
mae_all_7, mae_below_7, mae_above_7, pr_below_7, pr_above_7, pr_all_7, nq_below_7, nq_above_7, nq_all_7, vulnerables_7, secure_7, nr_noisy_ratings_7 = [], [], [], [], [], [], [], [], [], [], [], []
n_neighbors_0, n_neighbors_1, n_neighbors_2, n_neighbors_3, n_neighbors_4, n_neighbors_5, n_neighbors_6, n_neighbors_7 = [], [], [], [], [], [], [], []

thresholds = []
for trainset, testset in folds.split(dataset):
    sim = UserKNN.compute_similarities(trainset, min_support=1)
    pop = UserKNN.compute_popularities(trainset)
    gain = UserKNN.compute_gain(trainset)

    # Threshold
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "protected": False})
    threshs = [m.get_privacy_threshold() for m in models]
    thresholds.append(threshs)

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print("Mb: " + str(mem_info.rss / (1024 * 1024)))

    # KNN
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "thresholds": threshs, "protected": PROTECTED})
    resratings = eval_ratings(models, measurements=["mae"])
    resnetwork = eval_network(models, measurements=["privacy_risk", "n_queries", "n_neighbors"])
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

    n_secure, n_vulnerables = size_of_groups(models)
    secure_1.append(n_secure)
    vulnerables_1.append(n_vulnerables)
    nr_noisy_ratings_1.append([m.nr_noisy_ratings for m in models])


    # KNN + no protection
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "protected": False})
    resratings = eval_ratings(models, measurements=["mae"])
    resnetwork = eval_network(models, measurements=["privacy_risk", "n_queries", "n_neighbors"])
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
    n_secure, n_vulnerables = size_of_groups(models)
    secure_7.append(n_secure)
    vulnerables_7.append(n_vulnerables)
    nr_noisy_ratings_7.append([m.nr_noisy_ratings for m in models])

    # KNN + full protection
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "thresholds": [0 for _ in range(len(K))], "protected": True})
    resratings = eval_ratings(models, measurements=["mae"])
    resnetwork = eval_network(models, measurements=["privacy_risk", "n_queries", "n_neighbors"])
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
    n_secure, n_vulnerables = size_of_groups(models)
    secure_0.append(n_secure)
    vulnerables_0.append(n_vulnerables)
    nr_noisy_ratings_0.append([m.nr_noisy_ratings for m in models])

    # KNN + reuse
    models, _ = run(trainset, testset, K=K, configuration={"reuse": True, "precomputed_sim": sim, "thresholds": threshs, "protected": PROTECTED})
    resratings = eval_ratings(models, measurements=["mae"])
    resnetwork = eval_network(models, measurements=["privacy_risk", "n_queries", "n_neighbors"])
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
    n_secure, n_vulnerables = size_of_groups(models)
    secure_2.append(n_secure)
    vulnerables_2.append(n_vulnerables)
    nr_noisy_ratings_2.append([m.nr_noisy_ratings for m in models])


    # Popularity
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "precomputed_pop": pop, "tau_2": 0.5, "thresholds": threshs, "protected": PROTECTED})
    resratings = eval_ratings(models, measurements=["mae"])
    resnetwork = eval_network(models, measurements=["privacy_risk", "n_queries", "n_neighbors"])
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
    n_secure, n_vulnerables = size_of_groups(models)
    secure_3.append(n_secure)
    vulnerables_3.append(n_vulnerables)
    nr_noisy_ratings_3.append([m.nr_noisy_ratings for m in models])


    # Popularity + reuse
    models, _ = run(trainset, testset, K=K, configuration={"reuse": True, "precomputed_sim": sim, "precomputed_pop": pop, "tau_2": 0.5, "thresholds": threshs, "protected": PROTECTED})
    resratings = eval_ratings(models, measurements=["mae"])
    resnetwork = eval_network(models, measurements=["privacy_risk", "n_queries", "n_neighbors"])
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
    n_secure, n_vulnerables = size_of_groups(models)
    secure_4.append(n_secure)
    vulnerables_4.append(n_vulnerables)
    nr_noisy_ratings_4.append([m.nr_noisy_ratings for m in models])

    # Gain
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "precomputed_gain": gain, "tau_4": 0.5, "thresholds": threshs, "protected": PROTECTED})
    resratings = eval_ratings(models, measurements=["mae"])
    resnetwork = eval_network(models, measurements=["privacy_risk", "n_queries", "n_neighbors"])
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
    n_secure, n_vulnerables = size_of_groups(models)
    secure_5.append(n_secure)
    vulnerables_5.append(n_vulnerables)
    nr_noisy_ratings_5.append([m.nr_noisy_ratings for m in models])


    # Gain + reuse
    models, _ = run(trainset, testset, K=K, configuration={"reuse": True, "precomputed_sim": sim, "precomputed_gain": gain, "tau_4": 0.5, "thresholds": threshs, "protected": PROTECTED})
    resratings = eval_ratings(models, measurements=["mae"])
    resnetwork = eval_network(models, measurements=["privacy_risk", "n_queries", "n_neighbors"])
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
    n_secure, n_vulnerables = size_of_groups(models)
    secure_6.append(n_secure)
    vulnerables_6.append(n_vulnerables)
    nr_noisy_ratings_6.append([m.nr_noisy_ratings for m in models])


    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print("Mb: " + str(mem_info.rss / (1024 * 1024)))

    n_folds += 1
    break

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


plt.figure()
plt.plot(np.mean(pr_all_1, axis=0), np.mean(mae_all_1, axis=0), color="C0", linestyle="dashed", label="UserKNN", alpha=0.5)
plt.plot(np.mean(pr_all_3, axis=0), np.mean(mae_all_3, axis=0), color="C1", linestyle="dashed", label="Popularity", alpha=0.5)
plt.plot(np.mean(pr_all_5, axis=0), np.mean(mae_all_5, axis=0), color="C2", linestyle="dashed", label="Gain", alpha=0.5)
plt.plot(np.mean(pr_all_2, axis=0), np.mean(mae_all_2, axis=0), color="C0", linestyle="solid", label="UserKNN + Reuse")
plt.plot(np.mean(pr_all_4, axis=0), np.mean(mae_all_4, axis=0), color="C1", linestyle="solid", label="Popularity + Reuse")
plt.plot(np.mean(pr_all_6, axis=0), np.mean(mae_all_6, axis=0), color="C2", linestyle="solid", label="Gain + Reuse")
plt.ylabel("Mean absolute error")
plt.xlabel("Avg. Exposure")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()

plt.figure()
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
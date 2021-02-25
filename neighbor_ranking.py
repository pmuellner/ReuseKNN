import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True)
from algorithms.knn_neighborhood import UserKNN
import pandas as pd
from surprise import Dataset, Reader, accuracy, NMF
from surprise.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
from datetime import datetime as dt
from collections import defaultdict
from scipy.stats import skew
from networkx.algorithms.approximation.clustering_coefficient import average_clustering
from networkx import Graph
import threading, queue
import os
import psutil
import gc

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

    config_str = str({"reuse": reuse, "tau_1": tau_1, "tau_2": tau_2, "tau_3": tau_3, "tau_4": tau_4,
                      "precomputed_sim": sim is not None, "precomputed_act": act is not None,
                      "precomputed_pop": pop is not None, "precomputed_rr": rr is not None,
                      "precomputed_gain": gain is not None})

    t0 = dt.now()
    print("Started training model with K: " + str(K) + " and " + config_str)
    results = defaultdict(list)
    for k in K:
        model = UserKNN(k=k, reuse=reuse, precomputed_sim=sim, precomputed_act=act, precomputed_pop=pop,
                        precomputed_rr=rr, precomputed_gain=gain, tau_1=tau_1, tau_2=tau_2, tau_3=tau_3, tau_4=tau_4)
        model.fit(trainset)
        predictions = model.test(testset)
        results["models"].append(model)
        results["predictions"].append(predictions)

    print("Training finished after " + str(dt.now() - t0))

    return results["models"], results["predictions"]

def eval_ratings(predictions, measurements=[]):
    results = defaultdict(list)
    for p_at_k in predictions:
        if "mae" in measurements:
            mae = accuracy.mae(p_at_k, verbose=False)
            results["mae"].append(mae)

    return results

def eval_network(models, measurements=[]):
    results = defaultdict(list)
    for m_at_k in models:
        if "outdegree" in measurements:
            outdegree = m_at_k.get_degree()
            results["outdegree"].append(outdegree)
        if "pathlength" in measurements:
            pathlength = m_at_k.get_path_length()
            results["pathlength"].append(pathlength)
        if "skew" in measurements:
            ratios = []
            for uid in sorted(m_at_k.mae_u.keys()):
                s = len(m_at_k.students[uid])
                r = m_at_k.mae_u[uid] / s if s > 0 else 0
                ratios.append(r)
            results["skew"].append(skew(ratios))

    return results

data_df = pd.read_csv("data/ml-100k/u.data", sep="\t")
#data_df = pd.read_csv("data/ml-1m/ratings.dat", sep="::", header=None)
data_df.columns = ["user_id", "item_id", "rating", "timestamp"]
data_df.drop(columns=["timestamp"], axis=1, inplace=True)
"""data_df = pd.read_csv("data/anime_small.csv", sep=";")
data_df.columns = ["user_id", "item_id", "rating"]

n_users = data_df["user_id"].nunique()
print(n_users)
sample = np.random.choice(data_df["user_id"].unique(), size=int(n_users * 0.01), replace=False)
data_df = data_df[data_df["user_id"].isin(sample)]
print(data_df["user_id"].nunique())"""

data_df["user_id"] = data_df["user_id"].map({b: a for a, b in enumerate(data_df["user_id"].unique())})
data_df["item_id"] = data_df["item_id"].map({b: a for a, b in enumerate(data_df["item_id"].unique())})
n_items = data_df["item_id"].nunique()

reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data_df, reader=reader)
folds = KFold(n_splits=5)

K = np.arange(1, 10, 2)
#K = np.arange(1, 30, 2)

mae_1, outdegrees_1, pathlength_1, skew_1 = [], [], [], []
mae_2, outdegrees_2, pathlength_2, skew_2 = [], [], [], []
mae_3, outdegrees_3, pathlength_3, skew_3 = [], [], [], []
mae_4, outdegrees_4, pathlength_4, skew_4 = [], [], [], []
mae_5, outdegrees_5, pathlength_5, skew_5 = [], [], [], []
mae_6, outdegrees_6, pathlength_6, skew_6 = [], [], [], []
for trainset, testset in folds.split(dataset):
    sim = UserKNN.compute_similarities(trainset, min_support=1)
    pop = UserKNN.compute_popularities(trainset)
    gain = UserKNN.compute_gain(trainset)

    # KNN
    models, predictions = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim})
    resratings = eval_ratings(predictions, measurements=["mae"])
    resnetwork = eval_network(models, measurements=["outdegree", "skew"])
    mae_1.append(resratings["mae"])
    outdegrees_1.append(resnetwork["outdegree"])
    skew_1.append(resnetwork["skew"])

    # KNN + reuse
    models, predictions = run(trainset, testset, K=K, configuration={"reuse": True, "precomputed_sim": sim})
    resratings = eval_ratings(predictions, measurements=["mae"])
    resnetwork = eval_network(models, measurements=["outdegree", "skew"])
    mae_2.append(resratings["mae"])
    outdegrees_2.append(resnetwork["outdegree"])
    skew_2.append(resnetwork["skew"])

    # Popularity
    models, predictions = run(trainset, testset, K=K, configuration={"reuse": False, "tau_2": 0.5,
                                                                     "precomputed_sim": sim,
                                                                     "precomputed_pop": pop})
    resratings = eval_ratings(predictions, measurements=["mae"])
    resnetwork = eval_network(models, measurements=["outdegree", "skew"])
    mae_3.append(resratings["mae"])
    outdegrees_3.append(resnetwork["outdegree"])
    skew_3.append(resnetwork["skew"])

    # Popularity + reuse
    models, predictions = run(trainset, testset, K=K, configuration={"reuse": True, "tau_2": 0.5,
                                                                     "precomputed_sim": sim,
                                                                     "precomputed_pop": pop})
    resratings = eval_ratings(predictions, measurements=["mae"])
    resnetwork = eval_network(models, measurements=["outdegree", "skew"])
    mae_4.append(resratings["mae"])
    outdegrees_4.append(resnetwork["outdegree"])
    skew_4.append(resnetwork["skew"])

    # Gain
    models, predictions = run(trainset, testset, K=K, configuration={"reuse": False, "tau_4": 0.5,
                                                                     "precomputed_sim": sim,
                                                                     "precomputed_gain": gain})
    resratings = eval_ratings(predictions, measurements=["mae"])
    resnetwork = eval_network(models, measurements=["outdegree", "skew"])
    mae_5.append(resratings["mae"])
    outdegrees_5.append(resnetwork["outdegree"])
    skew_5.append(resnetwork["skew"])

    # Gain + reuse
    models, predictions = run(trainset, testset, K=K, configuration={"reuse": True, "tau_4": 0.5,
                                                                     "precomputed_sim": sim,
                                                                     "precomputed_gain": gain})
    resratings = eval_ratings(predictions, measurements=["mae"])
    resnetwork = eval_network(models, measurements=["outdegree", "skew"])
    mae_6.append(resratings["mae"])
    outdegrees_6.append(resnetwork["outdegree"])
    skew_6.append(resnetwork["skew"])

    del models
    del predictions
    gc.collect()
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print("Mb: " + str(mem_info.rss / (1024 * 1024)))

"""
1. KNN, 2. KNN + Reuse, 3. Popularity, 4. Popularity + Reuse, 5. Gain, 6. Gain + Reuse
"""
plt.figure()
plt.plot(K, np.mean(mae_1, axis=0), color="C0", linestyle="dashed", label="UserKNN", alpha=0.5)
plt.plot(K, np.mean(mae_3, axis=0), color="C1", linestyle="dashed", label="Popularity", alpha=0.5)
plt.plot(K, np.mean(mae_5, axis=0), color="C2", linestyle="dashed", label="Gain", alpha=0.5)
plt.plot(K, np.mean(mae_2, axis=0), color="C0", linestyle="solid", label="UserKNN + Reuse")
plt.plot(K, np.mean(mae_4, axis=0), color="C1", linestyle="solid", label="Popularity + Reuse")
plt.plot(K, np.mean(mae_6, axis=0), color="C2", linestyle="solid", label="Gain + Reuse")
plt.xlabel("Nr. of neighbors")
plt.ylabel("Mean absolute error")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()
#plt.savefig("plots/ml-1m/k_vs_mae.png", dpi=300)

plt.figure()
plt.plot(np.mean(mae_1, axis=0), np.mean(outdegrees_1, axis=0), color="C0", linestyle="dashed", label="UserKNN", alpha=0.5)
plt.plot(np.mean(mae_3, axis=0), np.mean(outdegrees_3, axis=0), color="C1", linestyle="dashed", label="Popularity", alpha=0.5)
plt.plot(np.mean(mae_5, axis=0), np.mean(outdegrees_5, axis=0), color="C2", linestyle="dashed", label="Gain", alpha=0.5)
plt.plot(np.mean(mae_2, axis=0), np.mean(outdegrees_2, axis=0), color="C0", linestyle="solid", label="UserKNN + Reuse")
plt.plot(np.mean(mae_4, axis=0), np.mean(outdegrees_4, axis=0), color="C1", linestyle="solid", label="Popularity + Reuse")
plt.plot(np.mean(mae_6, axis=0), np.mean(outdegrees_6, axis=0), color="C2", linestyle="solid", label="Gain + Reuse")
plt.ylabel("Outdegree")
plt.xlabel("Mean absolute error")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()
#plt.savefig("plots/ml-1m/outdegree_vs_mae.png", dpi=300)

"""plt.plot(np.mean(mae_1, axis=0), np.mean(pathlength_1, axis=0), color="C0", linestyle="dashed", label="UserKNN", alpha=0.5)
plt.plot(np.mean(mae_3, axis=0), np.mean(pathlength_3, axis=0), color="C1", linestyle="dashed", label="Popularity", alpha=0.5)
plt.plot(np.mean(mae_5, axis=0), np.mean(pathlength_5, axis=0), color="C2", linestyle="dashed", label="Gain", alpha=0.5)
plt.plot(np.mean(mae_2, axis=0), np.mean(pathlength_2, axis=0), color="C0", linestyle="solid", label="UserKNN + Reuse")
plt.plot(np.mean(mae_4, axis=0), np.mean(pathlength_4, axis=0), color="C1", linestyle="solid", label="Popularity + Reuse")
plt.plot(np.mean(mae_6, axis=0), np.mean(pathlength_6, axis=0), color="C2", linestyle="solid", label="Gain + Reuse")
plt.ylabel("Path Length")
plt.xlabel("Mean absolute error")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()"""

plt.figure()
plt.plot(K, np.mean(skew_1, axis=0), color="C0", linestyle="dashed", label="UserKNN", alpha=0.5)
plt.plot(K, np.mean(skew_3, axis=0), color="C1", linestyle="dashed", label="Popularity", alpha=0.5)
plt.plot(K, np.mean(skew_5, axis=0), color="C2", linestyle="dashed", label="Gain", alpha=0.5)
plt.plot(K, np.mean(skew_2, axis=0), color="C0", linestyle="solid", label="UserKNN + Reuse")
plt.plot(K, np.mean(skew_4, axis=0), color="C1", linestyle="solid", label="Popularity + Reuse")
plt.plot(K, np.mean(skew_6, axis=0), color="C2", linestyle="solid", label="Gain + Reuse")
plt.ylabel("Skewness of the ratio distribution")
plt.xlabel("Nr. of neighbors")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()
#plt.savefig("plots/ml-1m/skew_vs_k.png", dpi=300)




"""ranking_functions = [cumsum, softneg, reciprocal, reciprocal_plus_1, log_reciprocal]
for fold_it, data in enumerate(folds.split(dataset)):
    trainset, testset = data
    mean_absolute_errors = defaultdict(list)
    outdegrees = defaultdict(list)
    for function_it, f in enumerate(ranking_functions):
        sim = UserKNN.compute_similarities(trainset, min_support=1)
        rr = UserKNN.compute_rr(trainset, f)
        models, predictions = run(trainset, testset, K=np.arange(1, 30, 2), configuration={"precomputed_sim": sim,
                                                                                           "precomputed_rr": rr,
                                                                                           "reuse": True,
                                                                                           "tau_2": 1})
        resratings = eval_ratings(predictions, measurements=["mae"])
        resnetwork = eval_network(models, measurements=["outdegree"])

        mean_absolute_errors[f.__name__].append(resratings["mae"])
        outdegrees[f.__name__].append(resnetwork["outdegree"])

        print("Function %d fold %d finished" % (function_it, fold_it))
        print()

    models, predictions = run(trainset, testset, K=np.arange(1, 30, 2), configuration={"reuse": True, "tau_1": 1})
    resratings = eval_ratings(predictions, measurements=["mae"])
    resnetwork = eval_network(models, measurements=["outdegree"])
    mean_absolute_errors["reuse+pop"].append(resratings["mae"])
    outdegrees["reuse+pop"].append(resnetwork["outdegree"])

    break


for f in ranking_functions:
    mean_mae_f = np.mean(mean_absolute_errors[f.__name__], axis=0)
    mean_outdegree_f = np.mean(outdegrees[f.__name__], axis=0)
    plt.plot(mean_mae_f, mean_outdegree_f, label=f.__name__, linewidth=1)

mean_mae_baseline = np.mean(mean_absolute_errors["reuse+pop"], axis=0)
mean_outdegree_baseline = np.mean(outdegrees["reuse+pop"], axis=0)

plt.plot(mean_mae_baseline, mean_outdegree_baseline, label="reuse+pop", linewidth=1)
plt.xlabel("Mean absolute error")
plt.ylabel("Outdegree")
plt.legend()
plt.show()"""


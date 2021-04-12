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
import os
import psutil
import gc
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
            #outdegree = m_at_k.get_degree()
            exposure = np.mean(list(m_at_k.exposure_u.values()))
            results["outdegree"].append(exposure)
        if "pathlength" in measurements:
            pathlength = m_at_k.get_path_length()
            results["pathlength"].append(pathlength)

    return results

def rank_utility(model):
    sorted_uids = [uid for uid, _ in sorted(model.mae_u.items(), key=lambda t: t[1])[::-1]]
    return {uid: r for r, uid in enumerate(sorted_uids)}

def rank_privacy(model):
    sorted_uids = [uid for uid, _ in sorted(model.exposure_u.items(), key=lambda t: t[1])[::-1]]
    return {uid: r for r, uid in enumerate(sorted_uids)}

def tradeoff(model, lam):
    utility_ranks = rank_utility(model)
    privacy_ranks = rank_privacy(model)
    uids = set(utility_ranks.keys()).intersection(privacy_ranks.keys())
    avg_ranks = {uid: lam * utility_ranks[uid] + (1 - lam) * privacy_ranks[uid] for uid in uids}
    return avg_ranks

def delta_tradeoff(model1, model2, lam):
    ranks1 = tradeoff(model1, lam)
    ranks2 = tradeoff(model2, lam)
    uids = set(ranks1.keys()).intersection(ranks2.keys())
    return [(ranks2[uid] - ranks1[uid]) for uid in uids]

def calculate_redistribution(base_models, models, tau=0.5, how="gini"):
    def gini(x):
        x = MinMaxScaler().fit_transform(np.array(x).reshape(-1, 1))
        mad = np.abs(np.subtract.outer(x, x)).mean()
        rmad = mad / np.mean(x)
        g = 0.5 * rmad
        return g

    if how == "gini":
        ginis = []
        for b, m in zip(base_models, models):
            deltas = delta_tradeoff(b, m, lam=tau)
            ginis.append(gini(deltas))
        return ginis
    elif how == "skew":
        skews = []
        for b, m in zip(base_models, models):
            deltas = delta_tradeoff(b, m, lam=tau)
            skews.append(skew(deltas))
        return skews
    else:
        return None

data_df = pd.read_csv("data/ml-100k/u.data", sep="\t")
data_df.columns = ["user_id", "item_id", "rating", "timestamp"]
data_df.drop(columns=["timestamp"], axis=1, inplace=True)

#data_df = pd.read_csv("data/ml-1m/ratings.dat", sep="::", names=["user_id", "item_id", "rating", "timestamp"], usecols=["user_id", "item_id", "rating"])
#data_df = pd.read_csv("data/goodreads/sample.csv", sep=";", names=["user_id", "item_id", "rating"])

#data_df["user_id"] = data_df["user_id"].map({b: a for a, b in enumerate(data_df["user_id"].unique())})
#data_df["item_id"] = data_df["item_id"].map({b: a for a, b in enumerate(data_df["item_id"].unique())})
n_items = data_df["item_id"].nunique()

reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data_df, reader=reader)
folds = KFold(n_splits=5)

K = np.arange(1, 30, 5)

mae_1, outdegrees_1, pathlength_1, skew_1, gini_1 = [], [], [], [], []
mae_2, outdegrees_2, pathlength_2, skew_2, gini_2 = [], [], [], [], []
mae_3, outdegrees_3, pathlength_3, skew_3, gini_3 = [], [], [], [], []
mae_4, outdegrees_4, pathlength_4, skew_4, gini_4 = [], [], [], [], []
mae_5, outdegrees_5, pathlength_5, skew_5, gini_5 = [], [], [], [], []
mae_6, outdegrees_6, pathlength_6, skew_6, gini_6 = [], [], [], [], []
for trainset, testset in folds.split(dataset):
    sim = UserKNN.compute_similarities(trainset, min_support=1)
    pop = UserKNN.compute_popularities(trainset)
    gain = UserKNN.compute_gain(trainset)

    # KNN
    userknn_models, predictions = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim})
    resratings = eval_ratings(predictions, measurements=["mae"])
    resnetwork = eval_network(userknn_models, measurements=["outdegree", "skew", "similarity"])
    mae_1.append(resratings["mae"])
    outdegrees_1.append(resnetwork["outdegree"])

    del predictions
    gc.collect()

    # KNN + reuse
    userknn_reuse_models, predictions = run(trainset, testset, K=K, configuration={"reuse": True, "precomputed_sim": sim})
    resratings = eval_ratings(predictions, measurements=["mae"])
    resnetwork = eval_network(userknn_reuse_models, measurements=["outdegree", "skew", "similarity"])
    mae_2.append(resratings["mae"])
    outdegrees_2.append(resnetwork["outdegree"])
    gini_2.append(calculate_redistribution(userknn_models, userknn_reuse_models, how="gini"))
    skew_2.append(calculate_redistribution(userknn_models, userknn_reuse_models, how="skew"))

    del userknn_reuse_models
    del predictions
    gc.collect()

    # Popularity
    popularity_models, predictions = run(trainset, testset, K=K, configuration={"reuse": False, "tau_2": 0.5,
                                                                     "precomputed_sim": sim,
                                                                     "precomputed_pop": pop})
    resratings = eval_ratings(predictions, measurements=["mae"])
    resnetwork = eval_network(popularity_models, measurements=["outdegree", "skew", "similarity"])
    mae_3.append(resratings["mae"])
    outdegrees_3.append(resnetwork["outdegree"])
    gini_3.append(calculate_redistribution(userknn_models, popularity_models, how="gini"))
    skew_3.append(calculate_redistribution(userknn_models, popularity_models, how="skew"))

    del popularity_models
    del predictions
    gc.collect()

    # Popularity + reuse
    popularity_reuse_models, predictions = run(trainset, testset, K=K, configuration={"reuse": True, "tau_2": 0.5,
                                                                     "precomputed_sim": sim,
                                                                     "precomputed_pop": pop})
    resratings = eval_ratings(predictions, measurements=["mae"])
    resnetwork = eval_network(popularity_reuse_models, measurements=["outdegree", "skew", "similarity"])
    mae_4.append(resratings["mae"])
    outdegrees_4.append(resnetwork["outdegree"])
    gini_4.append(calculate_redistribution(userknn_models, popularity_reuse_models, how="gini"))
    skew_4.append(calculate_redistribution(userknn_models, popularity_reuse_models, how="skew"))

    del popularity_reuse_models
    del predictions
    gc.collect()

    # Gain
    gain_models, predictions = run(trainset, testset, K=K, configuration={"reuse": False, "tau_4": 0.5,
                                                                     "precomputed_sim": sim,
                                                                     "precomputed_gain": gain})
    resratings = eval_ratings(predictions, measurements=["mae"])
    resnetwork = eval_network(gain_models, measurements=["outdegree", "skew", "similarity"])
    mae_5.append(resratings["mae"])
    outdegrees_5.append(resnetwork["outdegree"])
    gini_5.append(calculate_redistribution(userknn_models, gain_models, how="gini"))
    skew_5.append(calculate_redistribution(userknn_models, gain_models, how="skew"))

    del gain_models
    del predictions
    gc.collect()

    # Gain + reuse
    gain_reuse_models, predictions = run(trainset, testset, K=K, configuration={"reuse": True, "tau_4": 0.5,
                                                                     "precomputed_sim": sim,
                                                                     "precomputed_gain": gain})
    resratings = eval_ratings(predictions, measurements=["mae"])
    resnetwork = eval_network(gain_reuse_models, measurements=["outdegree", "skew", "similarity"])
    mae_6.append(resratings["mae"])
    outdegrees_6.append(resnetwork["outdegree"])
    gini_6.append(calculate_redistribution(userknn_models, gain_reuse_models, how="gini"))
    skew_6.append(calculate_redistribution(userknn_models, gain_reuse_models, how="skew"))

    del gain_reuse_models
    del predictions
    gc.collect()

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print("Mb: " + str(mem_info.rss / (1024 * 1024)))

NAME = "ml-100k"
np.save("results/" + NAME + "/K.npy", K)

np.save("results/" + NAME + "/mae_userknn.npy", np.mean(mae_1, axis=0))
np.save("results/" + NAME + "/mae_pop.npy", np.mean(mae_3, axis=0))
np.save("results/" + NAME + "/mae_gain.npy", np.mean(mae_5, axis=0))
np.save("results/" + NAME + "/mae_userknn_reuse.npy", np.mean(mae_2, axis=0))
np.save("results/" + NAME + "/mae_pop_reuse.npy", np.mean(mae_4, axis=0))
np.save("results/" + NAME + "/mae_gain_reuse.npy", np.mean(mae_6, axis=0))

np.save("results/" + NAME + "/deg_userknn.npy", np.mean(outdegrees_1, axis=0))
np.save("results/" + NAME + "/deg_pop.npy", np.mean(outdegrees_3, axis=0))
np.save("results/" + NAME + "/deg_gain.npy", np.mean(outdegrees_5, axis=0))
np.save("results/" + NAME + "/deg_userknn_reuse.npy", np.mean(outdegrees_2, axis=0))
np.save("results/" + NAME + "/deg_pop_reuse.npy", np.mean(outdegrees_4, axis=0))
np.save("results/" + NAME + "/deg_gain_reuse.npy", np.mean(outdegrees_6, axis=0))

np.save("results/" + NAME + "/skew_userknn.npy", np.mean(skew_1, axis=0))
np.save("results/" + NAME + "/skew_pop.npy", np.mean(skew_3, axis=0))
np.save("results/" + NAME + "/skew_gain.npy", np.mean(skew_5, axis=0))
np.save("results/" + NAME + "/skew_userknn_reuse.npy", np.mean(skew_2, axis=0))
np.save("results/" + NAME + "/skew_pop_reuse.npy", np.mean(skew_4, axis=0))
np.save("results/" + NAME + "/skew_gain_reuse.npy", np.mean(skew_6, axis=0))

np.save("results/" + NAME + "/gini_userknn.npy", np.mean(gini_1, axis=0))
np.save("results/" + NAME + "/gini_pop.npy", np.mean(gini_3, axis=0))
np.save("results/" + NAME + "/gini_gain.npy", np.mean(gini_5, axis=0))
np.save("results/" + NAME + "/gini_userknn_reuse.npy", np.mean(gini_2, axis=0))
np.save("results/" + NAME + "/gini_pop_reuse.npy", np.mean(gini_4, axis=0))
np.save("results/" + NAME + "/gini_gain_reuse.npy", np.mean(gini_6, axis=0))

"""
1. KNN, 2. KNN + Reuse, 3. Popularity, 4. Popularity + Reuse, 5. Gain, 6. Gain + Reuse
"""


"""plt.figure()
plt.plot(K, sim_1[0], color="C0", linestyle="dashed", label="UserKNN", alpha=0.5)
plt.plot(K, sim_3[0], color="C1", linestyle="dashed", label="Popularity", alpha=0.5)
plt.plot(K, sim_5[0], color="C2", linestyle="dashed", label="Gain", alpha=0.5)
plt.plot(K, sim_2[0], color="C0", linestyle="solid", label="UserKNN + Reuse")
plt.plot(K, sim_4[0], color="C1", linestyle="solid", label="Popularity + Reuse")
plt.plot(K, sim_6[0], color="C2", linestyle="solid", label="Gain + Reuse")
plt.xlabel("Nr. of neighbors")
plt.ylabel("Avg. mentor similarity")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()

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

plt.figure()
plt.plot(np.mean(outdegrees_1, axis=0), np.mean(mae_1, axis=0), color="C0", linestyle="dashed", label="UserKNN", alpha=0.5)
plt.plot(np.mean(outdegrees_3, axis=0), np.mean(mae_3, axis=0), color="C1", linestyle="dashed", label="Popularity", alpha=0.5)
plt.plot(np.mean(outdegrees_5, axis=0), np.mean(mae_5, axis=0), color="C2", linestyle="dashed", label="Gain", alpha=0.5)
plt.plot(np.mean(outdegrees_2, axis=0), np.mean(mae_2, axis=0), color="C0", linestyle="solid", label="UserKNN + Reuse")
plt.plot(np.mean(outdegrees_4, axis=0), np.mean(mae_4, axis=0), color="C1", linestyle="solid", label="Popularity + Reuse")
plt.plot(np.mean(outdegrees_6, axis=0), np.mean(mae_6, axis=0), color="C2", linestyle="solid", label="Gain + Reuse")
plt.ylabel("Mean absolute error")
plt.xlabel("Outdegree")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()

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
plt.show()"""
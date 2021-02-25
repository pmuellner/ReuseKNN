import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=False)
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
import multiprocessing as mp
from multiprocessing import Pool, Process

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

def knn(trainset, testset, K, reuse, sim):
    models, predictions = run(trainset, testset, K=K, configuration={"reuse": reuse, "precomputed_sim": sim})
    resratings = eval_ratings(predictions, measurements=["mae"])
    resnetwork = eval_network(models, measurements=["outdegree", "skew"])

    return resratings["mae"], resnetwork["outdegree"], resnetwork["skew"]

def popularity(trainset, testset, K, reuse, tau_2, sim, pop):
    models, predictions = run(trainset, testset, K=K, configuration={"reuse": reuse, "tau_2": tau_2,
                                                                     "precomputed_sim": sim, "precomputed_pop": pop})
    resratings = eval_ratings(predictions, measurements=["mae"])
    resnetwork = eval_network(models, measurements=["outdegree", "skew"])

    return resratings["mae"], resnetwork["outdegree"], resnetwork["skew"]

def gain(trainset, testset, K, reuse, tau_4, sim, gain):
    models, predictions = run(trainset, testset, K=K, configuration={"reuse": reuse, "tau_4": tau_4,
                                                                     "precomputed_sim": sim, "precomputed_gain": gain})
    resratings = eval_ratings(predictions, measurements=["mae"])
    resnetwork = eval_network(models, measurements=["outdegree", "skew"])

    return resratings["mae"], resnetwork["outdegree"], resnetwork["skew"]

mae_1, outdegrees_1, skew_1 = [], [], []
mae_2, outdegrees_2, skew_2 = [], [], []
mae_3, outdegrees_3, skew_3 = [], [], []
mae_4, outdegrees_4, skew_4 = [], [], []
mae_5, outdegrees_5, skew_5 = [], [], []
mae_6, outdegrees_6, skew_6 = [], [], []
def save_results(identifier, mae, deg, skew):
    if identifier == 1:
        mae_1.append(mae)
        outdegrees_1.append(deg)
        skew_1.append(skew)
    elif identifier == 2:
        mae_2.append(mae)
        outdegrees_2.append(deg)
        skew_2.append(skew)
    elif identifier == 3:
        mae_3.append(mae)
        outdegrees_3.append(deg)
        skew_3.append(skew)
    elif identifier == 4:
        mae_4.append(mae)
        outdegrees_4.append(deg)
        skew_4.append(skew)
    elif identifier == 5:
        mae_5.append(mae)
        outdegrees_5.append(deg)
        skew_5.append(skew)
    elif identifier == 6:
        mae_6.append(mae)
        outdegrees_6.append(deg)
        skew_6.append(skew)

if __name__ == "__main__":
    #data_df = pd.read_csv("data/ml-1m/ratings.dat", sep="::", header=None)
    #data_df.columns = ["user_id", "item_id", "rating", "timestamp"]
    #data_df.drop(columns=["timestamp"], axis=1, inplace=True)

    data_df = pd.read_csv("data/ml-100k/u.data", sep="\t")
    data_df.columns = ["user_id", "item_id", "rating", "timestamp"]
    data_df.drop(columns=["timestamp"], axis=1, inplace=True)

    #data_df = pd.read_csv("data/bx_sample.csv", sep=";")
    #data_df.columns = ["user_id", "item_id", "rating"]

    data_df["user_id"] = data_df["user_id"].map({b: a for a, b in enumerate(data_df["user_id"].unique())})
    data_df["item_id"] = data_df["item_id"].map({b: a for a, b in enumerate(data_df["item_id"].unique())})
    n_items = data_df["item_id"].nunique()

    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(data_df, reader=reader)
    folds = KFold(n_splits=5)

    K = np.arange(1, 30, 2)
    for trainset, testset in folds.split(dataset):
        S = UserKNN.compute_similarities(trainset, min_support=1)
        P = UserKNN.compute_popularities(trainset)
        G = UserKNN.compute_gain(trainset)

        used_cores = int(mp.cpu_count() * 0.5)
        print("Usage of %d/%d cores" % (used_cores, mp.cpu_count()))
        with Pool(processes=used_cores) as pool:
            results_knn = pool.apply_async(knn, (trainset, testset, K, False, S),
                                           callback=lambda res: save_results(1, *res))
            results_knn_reuse = pool.apply_async(knn, (trainset, testset, K, True, S),
                                           callback=lambda res: save_results(2, *res))

            results_pop = pool.apply_async(popularity, (trainset, testset, K, False, 0.5, S, P),
                                           callback=lambda res: save_results(3, *res))
            results_pop_reuse = pool.apply_async(popularity, (trainset, testset, K, True, 0.5, S, P),
                                           callback=lambda res: save_results(4, *res))

            results_gain = pool.apply_async(gain, (trainset, testset, K, False, 0.5, S, G),
                                           callback=lambda res: save_results(5, *res))
            results_gain_reuse = pool.apply_async(gain, (trainset, testset, K, True, 0.5, S, G),
                                           callback=lambda res: save_results(6, *res))

            pool.close()
            pool.join()

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
    plt.savefig("plots/bx/k_vs_mae.png", dpi=300)

    plt.figure()
    plt.plot(np.mean(mae_1, axis=0), np.mean(outdegrees_1, axis=0), color="C0", linestyle="dashed", label="UserKNN",
             alpha=0.5)
    plt.plot(np.mean(mae_3, axis=0), np.mean(outdegrees_3, axis=0), color="C1", linestyle="dashed", label="Popularity",
             alpha=0.5)
    plt.plot(np.mean(mae_5, axis=0), np.mean(outdegrees_5, axis=0), color="C2", linestyle="dashed", label="Gain",
             alpha=0.5)
    plt.plot(np.mean(mae_2, axis=0), np.mean(outdegrees_2, axis=0), color="C0", linestyle="solid",
             label="UserKNN + Reuse")
    plt.plot(np.mean(mae_4, axis=0), np.mean(outdegrees_4, axis=0), color="C1", linestyle="solid",
             label="Popularity + Reuse")
    plt.plot(np.mean(mae_6, axis=0), np.mean(outdegrees_6, axis=0), color="C2", linestyle="solid", label="Gain + Reuse")
    plt.ylabel("Outdegree")
    plt.xlabel("Mean absolute error")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig("plots/bx/outdegree_vs_mae.png", dpi=300)

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
    plt.savefig("plots/bx/skew_vs_k.png", dpi=300)
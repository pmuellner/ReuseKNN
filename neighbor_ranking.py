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
        if "mae" in measurements:
            #mae = accuracy.mae(m_at_k.predictions, verbose=False)
            #results["mae"].append(mae)

            mae_below, mae_above, mae_all = [], [], []
            for uid, aes in m_at_k.absolute_errors.items():
                if uid not in m_at_k.protected_neighbors:
                    mae_below.extend(aes)
                else:
                    mae_above.extend(aes)
                mae_all.extend(aes)
            results["mae_below"].append(np.mean(mae_below))
            results["mae_above"].append(np.mean(mae_above))
            results["mae_all"].append(np.mean(mae_all))

    return results

def eval_network(models, measurements=[]):
    results = defaultdict(list)
    for m_at_k in models:
        if "privacy_risk" in measurements:
            pr_below, pr_above, pr_all = [], [], []
            for uid in m_at_k.trainset.all_users():
                if uid not in m_at_k.protected_neighbors:
                    pr_below.append(m_at_k.n_queries[uid])
                else:
                    pr_above.append(m_at_k.n_queries[uid])
                pr_all.append(m_at_k.n_queries[uid])

            results["pr_below"].append(np.mean(pr_below))
            results["pr_above"].append(np.mean(pr_above))
            results["pr_all"].append(np.mean(pr_all))

        if "pathlength" in measurements:
            pathlength = m_at_k.get_path_length()
            results["pathlength"].append(pathlength)
        if "privacy_score" in measurements:
            ps = [m_at_k.privacy_score[uid] for uid in m_at_k.trainset.all_users() if uid not in m_at_k.protected_neighbors]
            #ps = [m_at_k.privacy_score[uid] for uid in m_at_k.trainset.all_users()]
            results["privacy_score"].append(np.mean(ps))

    return results


def nr_protected(models):
    n_protected = np.array([len(m.protected_neighbors) for m in models])
    return n_protected

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
    reader = Reader(rating_scale=(0, 5))
elif NAME == "jester":
    data_df = pd.read_csv("data/jester/sample.csv", sep=";", names=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(-10, 10))
else:
    print("error")
    data_df = pd.DataFrame()
    reader = Reader()
if PROTECTED:
    PATH = "protected/" + NAME
else:
    PATH = "unprotected/" + NAME


dataset = Dataset.load_from_df(data_df, reader=reader)
folds = KFold(n_splits=5)

K = [5, 10, 15, 20]
#K = [1]
#K = np.arange(1, 30, 2)

mae_all_1, mae_below_1, mae_above_1, pr_below_1, pr_above_1, pr_all_1, vulnerables_1, nr_noisy_ratings_1 = [], [], [], [], [], [], [], []
mae_all_2, mae_below_2, mae_above_2, pr_below_2, pr_above_2, pr_all_2, vulnerables_2, nr_noisy_ratings_2 = [], [], [], [], [], [], [], []
mae_all_3, mae_below_3, mae_above_3, pr_below_3, pr_above_3, pr_all_3, vulnerables_3, nr_noisy_ratings_3 = [], [], [], [], [], [], [], []
mae_all_4, mae_below_4, mae_above_4, pr_below_4, pr_above_4, pr_all_4, vulnerables_4, nr_noisy_ratings_4 = [], [], [], [], [], [], [], []
mae_all_5, mae_below_5, mae_above_5, pr_below_5, pr_above_5, pr_all_5, vulnerables_5, nr_noisy_ratings_5 = [], [], [], [], [], [], [], []
mae_all_6, mae_below_6, mae_above_6, pr_below_6, pr_above_6, pr_all_6, vulnerables_6, nr_noisy_ratings_6 = [], [], [], [], [], [], [], []
i = 0
for trainset, testset in folds.split(dataset):
    sim = UserKNN.compute_similarities(trainset, min_support=1)
    pop = UserKNN.compute_popularities(trainset)
    gain = UserKNN.compute_gain(trainset)

    # KNN
    baseline_models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "protected": PROTECTED})
    threshs = [m.get_privacy_threshold() for m in baseline_models]

    userknn_models, predictions = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "thresholds": threshs, "protected": PROTECTED})
    resratings = eval_ratings(userknn_models, measurements=["mae"])
    resnetwork = eval_network(userknn_models, measurements=["privacy_risk"])
    mae_all_1.append(resratings["mae_all"])
    mae_below_1.append(resratings["mae_below"])
    mae_above_1.append(resratings["mae_above"])
    pr_all_1.append(resnetwork["pr_all"])
    pr_below_1.append(resnetwork["pr_below"])
    pr_above_1.append(resnetwork["pr_above"])
    vulnerables_1.append(nr_protected(userknn_models))
    nr_noisy_ratings_1.append([m.nr_noisy_ratings for m in userknn_models])

    del predictions
    gc.collect()

    # KNN + reuse
    userknn_reuse_models, predictions = run(trainset, testset, K=K, configuration={"reuse": True, "precomputed_sim": sim, "thresholds": threshs, "protected": PROTECTED})
    resratings = eval_ratings(userknn_reuse_models, measurements=["mae"])
    resnetwork = eval_network(userknn_reuse_models, measurements=["privacy_risk"])
    mae_all_2.append(resratings["mae_all"])
    mae_below_2.append(resratings["mae_below"])
    mae_above_2.append(resratings["mae_above"])
    pr_all_2.append(resnetwork["pr_all"])
    pr_below_2.append(resnetwork["pr_below"])
    pr_above_2.append(resnetwork["pr_above"])
    vulnerables_2.append(nr_protected(userknn_models))
    nr_noisy_ratings_2.append([m.nr_noisy_ratings for m in userknn_reuse_models])

    del userknn_reuse_models
    del predictions
    gc.collect()

    # Popularity
    popularity_models, predictions = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "precomputed_pop": pop, "tau_2": 0.5, "thresholds": threshs, "protected": PROTECTED})
    resratings = eval_ratings(popularity_models, measurements=["mae"])
    resnetwork = eval_network(popularity_models, measurements=["privacy_risk"])
    mae_all_3.append(resratings["mae_all"])
    mae_below_3.append(resratings["mae_below"])
    mae_above_3.append(resratings["mae_above"])
    pr_all_3.append(resnetwork["pr_all"])
    pr_below_3.append(resnetwork["pr_below"])
    pr_above_3.append(resnetwork["pr_above"])
    vulnerables_3.append(nr_protected(userknn_models))
    nr_noisy_ratings_3.append([m.nr_noisy_ratings for m in popularity_models])

    del popularity_models
    del predictions
    gc.collect()

    # Popularity + reuse
    popularity_reuse_models, predictions = run(trainset, testset, K=K, configuration={"reuse": True, "precomputed_sim": sim, "precomputed_pop": pop, "tau_2": 0.5, "thresholds": threshs, "protected": PROTECTED})
    resratings = eval_ratings(popularity_reuse_models, measurements=["mae"])
    resnetwork = eval_network(popularity_reuse_models, measurements=["privacy_risk"])
    mae_all_4.append(resratings["mae_all"])
    mae_below_4.append(resratings["mae_below"])
    mae_above_4.append(resratings["mae_above"])
    pr_all_4.append(resnetwork["pr_all"])
    pr_below_4.append(resnetwork["pr_below"])
    pr_above_4.append(resnetwork["pr_above"])
    vulnerables_4.append(nr_protected(userknn_models))
    nr_noisy_ratings_4.append([m.nr_noisy_ratings for m in popularity_reuse_models])

    del popularity_reuse_models
    del predictions
    gc.collect()

    # Gain
    gain_models, predictions = run(trainset, testset, K=K, configuration={"reuse": False, "precomputed_sim": sim, "precomputed_gain": gain, "tau_4": 0.5, "thresholds": threshs, "protected": PROTECTED})
    resratings = eval_ratings(gain_models, measurements=["mae"])
    resnetwork = eval_network(gain_models, measurements=["privacy_risk"])
    mae_all_5.append(resratings["mae_all"])
    mae_below_5.append(resratings["mae_below"])
    mae_above_5.append(resratings["mae_above"])
    pr_all_5.append(resnetwork["pr_all"])
    pr_below_5.append(resnetwork["pr_below"])
    pr_above_5.append(resnetwork["pr_above"])
    vulnerables_5.append(nr_protected(userknn_models))
    nr_noisy_ratings_5.append([m.nr_noisy_ratings for m in gain_models])

    del gain_models
    del predictions
    gc.collect()

    # Gain + reuse
    gain_reuse_models, predictions = run(trainset, testset, K=K, configuration={"reuse": True, "precomputed_sim": sim, "precomputed_gain": gain, "tau_4": 0.5, "thresholds": threshs, "protected": PROTECTED})
    resratings = eval_ratings(gain_reuse_models, measurements=["mae"])
    resnetwork = eval_network(gain_reuse_models, measurements=["privacy_risk"])
    mae_all_6.append(resratings["mae_all"])
    mae_below_6.append(resratings["mae_below"])
    mae_above_6.append(resratings["mae_above"])
    pr_all_6.append(resnetwork["pr_all"])
    pr_below_6.append(resnetwork["pr_below"])
    pr_above_6.append(resnetwork["pr_above"])
    vulnerables_6.append(nr_protected(userknn_models))
    nr_noisy_ratings_6.append([m.nr_noisy_ratings for m in gain_reuse_models])

    del gain_reuse_models
    del predictions
    gc.collect()

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print("Mb: " + str(mem_info.rss / (1024 * 1024)))

    break

np.save("results/" + PATH + "/K.npy", K)

np.save("results/" + PATH + "/mae_all_userknn.npy", np.mean(mae_all_1, axis=0))
np.save("results/" + PATH + "/mae_all_pop.npy", np.mean(mae_all_3, axis=0))
np.save("results/" + PATH + "/mae_all_gain.npy", np.mean(mae_all_5, axis=0))
np.save("results/" + PATH + "/mae_all_userknn_reuse.npy", np.mean(mae_all_2, axis=0))
np.save("results/" + PATH + "/mae_all_pop_reuse.npy", np.mean(mae_all_4, axis=0))
np.save("results/" + PATH + "/mae_all_gain_reuse.npy", np.mean(mae_all_6, axis=0))

np.save("results/" + PATH + "/mae_below_userknn.npy", np.mean(mae_below_1, axis=0))
np.save("results/" + PATH + "/mae_below_pop.npy", np.mean(mae_below_3, axis=0))
np.save("results/" + PATH + "/mae_below_gain.npy", np.mean(mae_below_5, axis=0))
np.save("results/" + PATH + "/mae_below_userknn_reuse.npy", np.mean(mae_below_2, axis=0))
np.save("results/" + PATH + "/mae_below_pop_reuse.npy", np.mean(mae_below_4, axis=0))
np.save("results/" + PATH + "/mae_below_gain_reuse.npy", np.mean(mae_below_6, axis=0))

np.save("results/" + PATH + "/mae_above_userknn.npy", np.mean(mae_above_1, axis=0))
np.save("results/" + PATH + "/mae_above_pop.npy", np.mean(mae_above_3, axis=0))
np.save("results/" + PATH + "/mae_above_gain.npy", np.mean(mae_above_5, axis=0))
np.save("results/" + PATH + "/mae_above_userknn_reuse.npy", np.mean(mae_above_5, axis=0))
np.save("results/" + PATH + "/mae_above_pop_reuse.npy", np.mean(mae_above_4, axis=0))
np.save("results/" + PATH + "/mae_above_gain_reuse.npy", np.mean(mae_above_6, axis=0))

np.save("results/" + PATH + "/pr_all_userknn.npy", np.mean(pr_all_1, axis=0))
np.save("results/" + PATH + "/pr_all_pop.npy", np.mean(pr_all_3, axis=0))
np.save("results/" + PATH + "/pr_all_gain.npy", np.mean(pr_all_5, axis=0))
np.save("results/" + PATH + "/pr_all_userknn_reuse.npy", np.mean(pr_all_2, axis=0))
np.save("results/" + PATH + "/pr_all_pop_reuse.npy", np.mean(pr_all_4, axis=0))
np.save("results/" + PATH + "/pr_all_gain_reuse.npy", np.mean(pr_all_6, axis=0))

np.save("results/" + PATH + "/pr_below_userknn.npy", np.mean(pr_below_1, axis=0))
np.save("results/" + PATH + "/pr_below_pop.npy", np.mean(pr_below_3, axis=0))
np.save("results/" + PATH + "/pr_below_gain.npy", np.mean(pr_below_5, axis=0))
np.save("results/" + PATH + "/pr_below_userknn_reuse.npy", np.mean(pr_below_2, axis=0))
np.save("results/" + PATH + "/pr_below_pop_reuse.npy", np.mean(pr_below_4, axis=0))
np.save("results/" + PATH + "/pr_below_gain_reuse.npy", np.mean(pr_below_6, axis=0))

np.save("results/" + PATH + "/pr_above_userknn.npy", np.mean(pr_above_1, axis=0))
np.save("results/" + PATH + "/pr_above_pop.npy", np.mean(pr_above_3, axis=0))
np.save("results/" + PATH + "/pr_above_gain.npy", np.mean(pr_above_5, axis=0))
np.save("results/" + PATH + "/pr_above_userknn_reuse.npy", np.mean(pr_above_2, axis=0))
np.save("results/" + PATH + "/pr_above_pop_reuse.npy", np.mean(pr_above_4, axis=0))
np.save("results/" + PATH + "/pr_above_gain_reuse.npy", np.mean(pr_above_6, axis=0))

np.save("results/" + PATH + "/vulnerables_userknn.npy", np.mean(vulnerables_1, axis=0))
np.save("results/" + PATH + "/vulnerables_pop.npy", np.mean(vulnerables_3, axis=0))
np.save("results/" + PATH + "/vulnerables_gain.npy", np.mean(vulnerables_5, axis=0))
np.save("results/" + PATH + "/vulnerables_userknn_reuse.npy", np.mean(vulnerables_2, axis=0))
np.save("results/" + PATH + "/vulnerables_pop_reuse.npy", np.mean(vulnerables_4, axis=0))
np.save("results/" + PATH + "/vulnerables_gain_reuse.npy", np.mean(vulnerables_6, axis=0))

np.save("results/" + PATH + "/nr_noisy_ratings_userknn.npy", np.mean(nr_noisy_ratings_1, axis=0))
np.save("results/" + PATH + "/nr_noisy_ratings_pop.npy", np.mean(nr_noisy_ratings_3, axis=0))
np.save("results/" + PATH + "/nr_noisy_ratings_gain.npy", np.mean(nr_noisy_ratings_5, axis=0))
np.save("results/" + PATH + "/nr_noisy_ratings_userknn_reuse.npy", np.mean(nr_noisy_ratings_2, axis=0))
np.save("results/" + PATH + "/nr_noisy_ratings_pop_reuse.npy", np.mean(nr_noisy_ratings_4, axis=0))
np.save("results/" + PATH + "/nr_noisy_ratings_gain_reuse.npy", np.mean(nr_noisy_ratings_6, axis=0))

"""
1. KNN, 2. KNN + Reuse, 3. Popularity, 4. Popularity + Reuse, 5. Gain, 6. Gain + Reuse
"""

plt.figure()
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
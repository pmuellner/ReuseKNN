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
import sys
from algorithms import evaluation
from embeddings.embeddings import Embeddings

def run(trainset, testset, K, configuration={}):
    reuse = configuration.get("reuse", False)
    sim = configuration.get("precomputed_sim", None)
    pop = configuration.get("precomputed_pop", None)
    gain = configuration.get("precomputed_gain", None)
    overlap = configuration.get("precomputed_overlap", None)
    rated_items = configuration.get("rated_items", None)
    tau_2 = configuration.get("tau_2", 0) #expect
    tau_4 = configuration.get("tau_4", 0) #gain
    tau_6 = configuration.get("tau_6", 0) #gainplus
    user_embedding = configuration.get("user_embedding", None)
    item_embedding = configuration.get("item_embedding", None)


    thresholds = configuration.get("thresholds", None)
    protected = configuration.get("protected", False)

    config_str = str({"reuse": reuse, "tau_2": tau_2, "tau_4": tau_4, "tau_6": tau_6, "precomputed_sim": sim is not None,
                      "precomputed_pop": pop is not None, "precomputed_gain": gain is not None, "protected": protected,
                      "precomputed_overlap": overlap is not None, "rated_items": rated_items is not None,
                      "user_embedding": user_embedding is not None, "item_embedding": item_embedding is not None})

    t0 = dt.now()
    print("Started training model with K: " + str(K) + " and " + config_str)
    results = defaultdict(list)
    for idx, k in enumerate(K):
        if thresholds is not None:
            th = thresholds[idx]
        else:
            th = 0
        model = UserKNN(k=k, reuse=reuse, user_embedding=user_embedding, item_embedding=item_embedding, tau_2=tau_2, tau_4=tau_4, tau_6=tau_6,
                        threshold=th, protected=protected, rated_items=rated_items)
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
#K = [10]
#K_q_idx = 0
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
    user_embedding = Embeddings("results/embeddings/" + NAME + "/UB-10-cbow-ns10-w10-c1-i25-id-fold1.embeddings")
    item_embedding = Embeddings("results/embeddings/" + NAME + "/IB-10-cbow-ns10-w10-c1-i25-id-fold1.embeddings")

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print("Mb: " + str(mem_info.rss / (1024 * 1024)))

    # Threshold
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "user_embedding": user_embedding, "item_embedding": item_embedding, "protected": False})
    threshs = [m.get_privacy_threshold() for m in models]
    thresholds.append(threshs)

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print("Mb: " + str(mem_info.rss / (1024 * 1024)))
    del models

    # UserKNN_DP
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "user_embedding": user_embedding, "item_embedding": item_embedding, "protected": PROTECTED})
    results, userknn_results_samples = evaluation.evaluate(models, [models[K_q_idx]])
    mean_absolute_error["userknn"].append(results["mean_absolute_error"])
    del models, results

    # UserKNN
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "user_embedding": user_embedding, "item_embedding": item_embedding,
                                                           "thresholds": [np.inf for _ in range(len(K))], "protected": False})
    results, results_samples = evaluation.evaluate(models, [models[K_q_idx]])
    mean_absolute_error["userknn_no"].append(results["mean_absolute_error"])
    del models, results, results_samples

    # Full_DP
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "user_embedding": user_embedding, "item_embedding": item_embedding,
                                                           "thresholds": [0 for _ in range(len(K))], "protected": True})
    results, userknn_full_results_samples = evaluation.evaluate(models, [models[K_q_idx]])
    mean_absolute_error["userknn_full"].append(results["mean_absolute_error"])
    del models, results

    # Expect_DP
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "tau_2": 0.5, "user_embedding": user_embedding, "item_embedding": item_embedding,
                                                           "thresholds": threshs, "protected": PROTECTED})
    results, results_samples = evaluation.evaluate(models, [models[K_q_idx]])
    mean_absolute_error["expect"].append(results["mean_absolute_error"])
    del models, results, results_samples

    # Gain_DP
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "tau_4": 0.5, "user_embedding": user_embedding, "item_embedding": item_embedding,
                                                           "thresholds": threshs, "protected": PROTECTED})
    results, results_samples = evaluation.evaluate(models, [models[K_q_idx]])
    mean_absolute_error["gain"].append(results["mean_absolute_error"])
    del models, results, results_samples

    # Gain+_DP
    models, _ = run(trainset, testset, K=K, configuration={"reuse": False, "tau_6": 0.5, "user_embedding": user_embedding, "item_embedding": item_embedding,
                                                           "thresholds": threshs, "protected": PROTECTED})
    results, results_samples = evaluation.evaluate(models, [models[K_q_idx]])
    mean_absolute_error["gain+"].append(results["mean_absolute_error"])
    del models, results, results_samples

    print(mean_absolute_error)

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print("Mb: " + str(mem_info.rss / (1024 * 1024)))

    n_folds += 1


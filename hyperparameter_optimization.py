import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True)
from algorithms.knn_neighborhood import UserKNN
from algorithms.metrics import mean_absolute_error, avg_privacy_risk_dp, fraction_vulnerables, avg_neighborhood_size, avg_rating_overlap_q
import pandas as pd
import pickle as pl
from surprise import Dataset, Reader
from sklearn.model_selection import train_test_split
from datetime import datetime as dt
from collections import defaultdict
import os
import psutil
import sys
from algorithms import evaluation
from embeddings.embeddings import Embeddings
from itertools import product as cartesian_product

np.random.seed(42)

def print_mem_consumption():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print("Mb: " + str(mem_info.rss / (1024 * 1024)))

def map_user_embedding(trainset, embedding):
    def normalize(word_vec):
        norm = np.linalg.norm(word_vec)
        if norm == 0:
            return word_vec
        return word_vec / norm

    embedding_mapped = np.zeros_like(embedding.embeddings)
    for inner_uid in trainset.all_users():
        raw_uid = trainset.to_raw_uid(inner_uid)
        index = embedding.item2index[raw_uid]
        #embedding_mapped[inner_uid] = normalize(embedding.embeddings[index])
        embedding_mapped[inner_uid] = embedding.embeddings[index]
    return embedding_mapped

def map_item_embedding(trainset, embedding):
    def normalize(word_vec):
        norm = np.linalg.norm(word_vec)
        if norm == 0:
            return word_vec
        return word_vec / norm

    embedding_mapped = np.zeros_like(embedding.embeddings)
    for inner_iid in trainset.all_items():
        raw_iid = trainset.to_raw_iid(inner_iid)
        index = embedding.item2index[raw_iid]
        #embedding_mapped[inner_uid] = normalize(embedding.embeddings[index])
        embedding_mapped[inner_iid] = embedding.embeddings[index]
    return embedding_mapped

if len(sys.argv) == 3:
    NAME = sys.argv[1]
else:
    NAME = "ml-100k"


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

print(NAME)
dataset = Dataset.load_from_df(data_df, reader=reader)
trainset_df, valtestset_df = train_test_split(data_df, test_size=0.2, random_state=42)
valset_df, testset_df = train_test_split(valtestset_df, test_size=0.5)

# todo change that
#valset_df = valtestset_df
#user_sample = np.random.choice(valtestset_df["user_id"].unique(), size=int(valtestset_df["user_id"].nunique()*0.5), replace=False)
#valset_df = valtestset_df[valtestset_df["user_id"].isin(user_sample)]
#testset_df = valtestset_df[~valtestset_df["user_id"].isin(user_sample)]
#trainset_df, valset_df = train_test_split(data_df, test_size=0.2, random_state=42)
#trainset_df, valtestset_df = train_test_split(data_df, test_size=0.4, random_state=42)
#valset_df, testset_df = train_test_split(valtestset_df, test_size=0.5, random_state=42)

trainset = dataset.construct_trainset([(*record, None) for record in trainset_df.to_records(index=False)])
valset = dataset.construct_testset([(*record, None) for record in valset_df.to_records(index=False)])

testset_df = pd.DataFrame(testset_df)
testset_df.to_csv("results/parameter_search/" + NAME + "/testset.csv", sep=";", index=False, header=False)
print("Testset saved")

#Ks = [5, 10, 15, 20, 25, 30]
#Ks = [10]
Ks = [10, 20, 30, 40, 50, 60, 70, 80]
#D = [10, 25, 50]
D = [100, 200, 500]
N = [10]
I = [25, 50, 100]
W = [10, 25, 50]
parameter_configurations = list(cartesian_product(D, N, I, W))
mean_absolute_errors = dict()
privacy_risks = dict()
fraction_protected_users = dict()
progress = 1
for n_dimensions, n_negative_samples, n_iterations, window_size in parameter_configurations:
    print("Embedding %d/%d ..." % (progress, len(parameter_configurations)))
    ub_embedding_str = "UB-{d}-cbow-ns{n}-w{w}-c1-i{i}-id-fold1.embeddings".format(d=n_dimensions, n=n_negative_samples, w=window_size, i=n_iterations)
    ib_embedding_str = "IB-{d}-cbow-ns{n}-w{w}-c1-i{i}-id-fold1.embeddings".format(d=n_dimensions, n=n_negative_samples, w=window_size, i=n_iterations)
    #ub_embedding_str = "UB-{d}-sg-ns{n}-w{w}-c1-i{i}-id-fold1.embeddings".format(d=n_dimensions, n=n_negative_samples, w=window_size, i=n_iterations)
    #ib_embedding_str = "IB-{d}-sg-ns{n}-w{w}-c1-i{i}-id-fold1.embeddings".format(d=n_dimensions, n=n_negative_samples, w=window_size, i=n_iterations)

    user_embedding = Embeddings("results/embeddings/" + NAME + "/" + ub_embedding_str)
    item_embedding = Embeddings("results/embeddings/" + NAME + "/" + ib_embedding_str)

    print(user_embedding.embeddings.shape, item_embedding.embeddings.shape)
    print(ub_embedding_str)

    rated_items = UserKNN.compute_rated_items(trainset)
    overlap = UserKNN.compute_overlap(trainset)

    user_embedding = map_user_embedding(trainset, user_embedding)
    user_sim = UserKNN.cosine_similarity(user_embedding)
    #user_sim = UserKNN.compute_similarities(trainset, min_support=1)

    item_embedding = map_item_embedding(trainset, item_embedding)
    item_neighbors = UserKNN.topk_item_neighbors(item_embedding, k=1)
    gainplus_scores = UserKNN.compute_gainplus(trainset, item_neighbors)

    expect_scores = UserKNN.compute_popularities(trainset)
    gain_scores = UserKNN.compute_gain(trainset)

    #user_embedding = None
    #item_embedding = None
    #gainplus_scores = None

    tau = 0.5
    use_dp = True
    print_mem_consumption()
    maes_per_emb = defaultdict(list)
    prs_per_emb = defaultdict(list)
    fpu_per_emb = defaultdict(list)
    for k_idx, k in enumerate(Ks):
        # todo train each method separately + monitor accuracy/privacyrisk_dp
        # todo do we need an additional optimization for the item embeddings in case of gain+?

        model = UserKNN(k=k, user_embedding=user_embedding, item_embedding=item_embedding, threshold=np.inf, protected=False, rated_items=rated_items, precomputed_overlap=overlap,
                        precomputed_sim=user_sim, precomputed_gainplus=gainplus_scores, precomputed_pop=expect_scores, precomputed_gain=gain_scores)
        model.fit(trainset).test(valset)
        threshold = model.get_privacy_threshold()

        # todo change
        print(threshold)

        # UserKNN
        print("UserKNN (k=%d) ..." % k)
        model = UserKNN(k=k, user_embedding=user_embedding, item_embedding=item_embedding, threshold=np.inf, protected=False, rated_items=rated_items, precomputed_overlap=overlap,
                        precomputed_sim=user_sim, precomputed_gainplus=gainplus_scores, precomputed_pop=expect_scores, precomputed_gain=gain_scores)
        model.fit(trainset).test(valset)
        #print(mean_absolute_error(model)[0])
        maes_per_emb["UserKNN"].append(mean_absolute_error(model)[0])
        prs_per_emb["UserKNN"].append(avg_privacy_risk_dp(model)[0])
        fpu_per_emb["UserKNN"].append(fraction_vulnerables(model))
        print(avg_rating_overlap_q(model)[0][-1])
        del model

        # UserKNN_DP
        print("UserKNN_DP (k=%d) ..." % k)
        model = UserKNN(k=k, user_embedding=user_embedding, item_embedding=item_embedding, threshold=threshold, protected=use_dp, rated_items=rated_items,
                        precomputed_overlap=overlap, precomputed_sim=user_sim, precomputed_gainplus=gainplus_scores, precomputed_pop=expect_scores, precomputed_gain=gain_scores)
        model.fit(trainset).test(valset)
        #print(mean_absolute_error(model)[0])
        maes_per_emb["UserKNN_DP"].append(mean_absolute_error(model)[0])
        prs_per_emb["UserKNN_DP"].append(avg_privacy_risk_dp(model)[0])
        fpu_per_emb["UserKNN_DP"].append(fraction_vulnerables(model))
        print(avg_rating_overlap_q(model)[0][-1])
        del model

        # Full_DP
        print("Full_DP (k=%d) ..." % k)
        model = UserKNN(k=k, user_embedding=user_embedding, item_embedding=item_embedding, threshold=0, protected=use_dp, rated_items=rated_items, precomputed_overlap=overlap,
                        precomputed_sim=user_sim, precomputed_gainplus=gainplus_scores, precomputed_pop=expect_scores, precomputed_gain=gain_scores)
        model.fit(trainset).test(valset)
        #print(mean_absolute_error(model)[0])
        maes_per_emb["Full_DP"].append(mean_absolute_error(model)[0])
        prs_per_emb["Full_DP"].append(avg_privacy_risk_dp(model)[0])
        fpu_per_emb["Full_DP"].append(fraction_vulnerables(model))
        print(avg_rating_overlap_q(model)[0][-1])
        del model

        # Expect_DP
        print("Expect_DP (k=%d) ..." % k)
        model = UserKNN(k=k, user_embedding=user_embedding, item_embedding=item_embedding, threshold=threshold, protected=use_dp, rated_items=rated_items, tau_2=tau,
                        precomputed_overlap=overlap, precomputed_sim=user_sim, precomputed_gainplus=gainplus_scores, precomputed_pop=expect_scores, precomputed_gain=gain_scores)
        model.fit(trainset).test(valset)
        #print(mean_absolute_error(model)[0])
        maes_per_emb["Expect_DP"].append(mean_absolute_error(model)[0])
        prs_per_emb["Expect_DP"].append(avg_privacy_risk_dp(model)[0])
        fpu_per_emb["Expect_DP"].append(fraction_vulnerables(model))
        print(avg_rating_overlap_q(model)[0][-1])
        del model

        # Gain_DP
        print("Gain_DP (k=%d) ..." % k)
        model = UserKNN(k=k, user_embedding=user_embedding, item_embedding=item_embedding, threshold=threshold, protected=use_dp, rated_items=rated_items, tau_4=tau,
                        precomputed_overlap=overlap, precomputed_sim=user_sim, precomputed_gainplus=gainplus_scores, precomputed_pop=expect_scores, precomputed_gain=gain_scores)
        model.fit(trainset).test(valset)
        #print(mean_absolute_error(model)[0])
        maes_per_emb["Gain_DP"].append(mean_absolute_error(model)[0])
        prs_per_emb["Gain_DP"].append(avg_privacy_risk_dp(model)[0])
        fpu_per_emb["Gain_DP"].append(fraction_vulnerables(model))
        print(avg_rating_overlap_q(model)[0][-1])
        del model

        # Gain+_DP
        print("Gain+_DP (k=%d) ..." % k)
        model = UserKNN(k=k, user_embedding=user_embedding, item_embedding=item_embedding, threshold=threshold, protected=use_dp, rated_items=rated_items, tau_6=tau,
                        precomputed_overlap=overlap, precomputed_sim=user_sim, precomputed_gainplus=gainplus_scores, precomputed_pop=expect_scores, precomputed_gain=gain_scores)
        model.fit(trainset).test(valset)
        #print(mean_absolute_error(model)[0])
        maes_per_emb["Gain+_DP"].append(mean_absolute_error(model)[0])
        prs_per_emb["Gain+_DP"].append(avg_privacy_risk_dp(model)[0])
        fpu_per_emb["Gain+_DP"].append(fraction_vulnerables(model))
        print(avg_rating_overlap_q(model)[0][-1])
        del model

        print_mem_consumption()

    embedding_str = "{d}d-{n}ns-{w}w-{i}i".format(d=n_dimensions, n=n_negative_samples, w=window_size, i=n_iterations)
    mean_absolute_errors[embedding_str] = maes_per_emb
    privacy_risks[embedding_str] = prs_per_emb
    fraction_protected_users[embedding_str] = fpu_per_emb

    for method, results in maes_per_emb.items():
        print("Mean Absolute Error " + method + ": " + str(results))
    for method, results in prs_per_emb.items():
        print("Avg. Privacy Risk DP " + method + ": " + str(results))
    for method, results in fpu_per_emb.items():
        print("Fraction of Vulnerables " + method + ": " + str(results))
    print()
    progress += 1


with open("results/parameter_search/" + NAME + "/mean_absolute_errors.pkl", "wb") as f:
    pl.dump(mean_absolute_errors, f)
with open("results/parameter_search/" + NAME + "/privacy_risks.pkl", "wb") as f:
    pl.dump(privacy_risks, f)
with open("results/parameter_search/" + NAME + "/frac_vulnerables.pkl", "wb") as f:
    pl.dump(fraction_protected_users, f)



    # todo for every method, select parameters with max accuracy and min privacyrisk_dp

# todo overall result: for every k and every method, we have two parameter configurations (max accuracy and max privacy)





import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True)
from algorithms.knn_neighborhood_embedding import UserKNN
#from algorithms.knn_neighborhood import UserKNN

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import sys
from surprise import Dataset, Reader
from collections import defaultdict
from algorithms.metrics import mean_absolute_error, avg_privacy_risk, fraction_vulnerables
import pickle as pl
import argparse
import os
from sklearn.metrics.pairwise import cosine_similarity


def parse_input_str(input):
    if type(input) is str:
        if input.lower() == "none":
            return None
        if input.lower() == "false":
            return False
        if input.lower() == "true":
            return True
    else:
        return input


def map_embedding(embedding_dict, trainset, user_based=True):
    n = len(list(embedding_dict.values())[0])
    if user_based:
        embedding_array = np.zeros((trainset.n_users, n))
        for iuid in trainset.all_users():
            ruid = trainset.to_raw_uid(iuid)
            embedding_array[iuid] = embedding_dict[ruid]
    else:
        embedding_array = np.zeros((trainset.n_items, n))
        for iiid in trainset.all_items():
            riid = trainset.to_raw_iid(iiid)
            embedding_array[iiid] = embedding_dict[riid]
    return embedding_array


def EmbeddingModel(embedding_size, n_users, n_items, n_samples):
    init_normal = keras.initializers.RandomNormal(mean=0, stddev=0.01, seed=1234)

    user_input = keras.Input(shape=(1,), name='user_id')
    user_emb = layers.Embedding(output_dim=embedding_size, input_dim=n_users, name='user_emb', input_length=n_samples,
                                embeddings_initializer=init_normal)(user_input)
    user_vec = layers.Flatten(name='FlattenUser')(user_emb)
    user_model = keras.Model(inputs=user_input, outputs=user_vec)

    item_input = keras.Input(shape=(1,), name='item_id')
    item_emb = layers.Embedding(output_dim=embedding_size, input_dim=n_items, name='item_emb', input_length=n_samples,
                                embeddings_initializer=init_normal)(item_input)
    item_vec = layers.Flatten(name='FlattenItem')(item_emb)
    item_model = keras.Model(inputs=item_input, outputs=item_vec)

    # link user and item embedding
    merged = layers.Dot(name='dot_product', normalize=True, axes=2)([user_emb, item_emb])
    result = layers.Dense(1, name='result', activation="relu", kernel_initializer=init_normal)(merged)

    # optimization
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model = keras.Model([user_input, item_input], result)
    model.compile(optimizer=optimizer, loss='mean_absolute_error')

    return model, user_model, item_model


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default="ml-100k")
parser.add_argument('--generate_embeddings', default=True)
parser.add_argument('--generate_recommendations', default=True)
parser.add_argument('--only_first_fold', default=False)
parser.add_argument('--gpus', default="7")
args = parser.parse_args()

args.generate_embeddings = parse_input_str(str(args.generate_embeddings))
args.generate_recommendations = parse_input_str(str(args.generate_recommendations))
args.only_first_fold = parse_input_str(str(args.only_first_fold))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus  # specify which GPU(s) to be used

DATASET_PATH = "../../datasets/"
RESULT_PATH = "results/"

if args.dataset_name == "ml-100k":
    data_df = pd.read_csv(DATASET_PATH + "ml-100k/u.data", sep="\t",
                          names=["user_id", "item_id", "rating", "timestamp"],
                          usecols=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(1, 5))
elif args.dataset_name == "ml-1m":
    data_df = pd.read_csv(DATASET_PATH + "ml-1m/ratings.dat", sep="::",
                          names=["user_id", "item_id", "rating", "timestamp"],
                          usecols=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(1, 5))
elif args.dataset_name == "goodreads":
    data_df = pd.read_csv(DATASET_PATH + "goodreads/sample.csv", sep=";", names=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(1, 5))
elif args.dataset_name == "lfm":
    data_df = pd.read_csv(DATASET_PATH + "lfm/artist_ratings.csv", sep=";", names=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(1, 1000))
elif args.dataset_name == "ciao":
    data_df = pd.read_csv(DATASET_PATH + "ciao/ciao.csv", sep=";", names=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(1, 5))
elif args.dataset_name == "douban":
    data_df = pd.read_csv(DATASET_PATH + "douban/douban.csv", sep=";", names=["user_id", "item_id", "rating"])
    reader = Reader(rating_scale=(1, 5))
else:
    print("error")
    data_df = pd.DataFrame()
    reader = Reader()

print("=====================")
print("Dataset: %s, Embedding Size: %d, Generate Embeddings: %s, Generate Recommendations: %s"
      % (args.dataset_name, 16, args.generate_embeddings, args.generate_recommendations))

mean_absolute_errors = defaultdict(list)
privacy_risks = defaultdict(list)
fraction_protected_users = defaultdict(list)

raw2inner_uid = {b: a for a, b in enumerate(data_df["user_id"].unique())}
raw2inner_iid = {b: a for a, b in enumerate(data_df["item_id"].unique())}
inner2raw_uid = {b: a for a, b in raw2inner_uid.items()}
inner2raw_iid = {b: a for a, b in raw2inner_iid.items()}
dataset = Dataset.load_from_df(data_df, reader=reader)

kf = KFold(n_splits=5, shuffle=True, random_state=1234)
all_methods = ["UserKNN", "UserKNN_DP", "Full_DP", "UserKNN_Static_DP", "UserKNN_Dynamic_DP"]
for i, (train_index, valtest_index) in enumerate(kf.split(data_df)):
    if args.generate_embeddings:
        # generate user and item embeddings
        print("=====================")
        print("[Fold %d] Generate Embeddings" % (i+1))
        # trainset used for training recsys and embedding, val used for evaluating embedding, test used for evaluating recsys
        train_df = data_df.iloc[train_index]
        valtest_df = data_df.iloc[valtest_index]
        val_df, test_df = train_test_split(valtest_df, test_size=0.5, shuffle=True, random_state=1234)

        # train embeddings
        n_users = data_df["user_id"].nunique()
        n_items = data_df["item_id"].nunique()
        X = train_df.copy()
        X["user_id"] = X["user_id"].map(raw2inner_uid)
        X["item_id"] = X["item_id"].map(raw2inner_iid)
        Y = val_df.copy()
        Y["user_id"] = Y["user_id"].map(raw2inner_uid)
        Y["item_id"] = Y["item_id"].map(raw2inner_iid)

        model, user_model, item_model = EmbeddingModel(16, n_users, n_items, n_samples=len(train_df))
        callbacks = [keras.callbacks.EarlyStopping('val_loss', patience=10, restore_best_weights=True)]
        history = model.fit([X["user_id"], X["item_id"]], X["rating"], batch_size=128, epochs=50,
                            validation_data=([Y["user_id"], Y["item_id"]], Y["rating"]), verbose=1,
                            callbacks=callbacks)

        # save user and item embeddings
        user_embeddings_dict = dict()
        inner_uids = list(raw2inner_uid.values())
        embeddings = user_model.predict(inner_uids)
        for idx, emb_u in enumerate(embeddings):
            iuid = inner_uids[idx]
            user_embeddings_dict[inner2raw_uid[iuid]] = emb_u

        item_embeddings_dict = dict()
        inner_iids = list(raw2inner_iid.values())
        embeddings = item_model.predict(inner_iids)
        for idx, emb_i in enumerate(embeddings):
            iiid = inner_iids[idx]
            item_embeddings_dict[inner2raw_iid[iiid]] = emb_i

        # save embeddings to disk
        with open(RESULT_PATH + "NeuReuse/" + args.dataset_name + "/user_embeddings_f" + str(i) + ".pkl", "wb") as f:
            pl.dump(user_embeddings_dict, f)
        with open(RESULT_PATH + "NeuReuse/" + args.dataset_name + "/item_embeddings_f" + str(i) + ".pkl", "wb") as f:
            pl.dump(item_embeddings_dict, f)

        train_df.to_csv(RESULT_PATH + "NeuReuse/" + args.dataset_name + "/train_f" + str(i) + ".csv", sep=";", index=False, header=None)
        test_df.to_csv(RESULT_PATH + "NeuReuse/" + args.dataset_name + "/test_f" + str(i) + ".csv", sep=";", index=False, header=None)
    else:
        # read embeddings
        user_embeddings_dict = pl.load(open(RESULT_PATH + "NeuReuse/" + args.dataset_name + "/user_embeddings_f" + str(i) + ".pkl", "rb"))
        item_embeddings_dict = pl.load(open(RESULT_PATH + "NeuReuse/" + args.dataset_name + "/item_embeddings_f" + str(i) + ".pkl", "rb"))

        train_df = pd.read_csv(RESULT_PATH + "NeuReuse/" + args.dataset_name + "/train_f" + str(i) + ".csv", sep=";", header=None)
        test_df = pd.read_csv(RESULT_PATH + "NeuReuse/" + args.dataset_name + "/test_f" + str(i) + ".csv", sep=";", header=None)

    if args.generate_recommendations:
        # generate recommendations
        print("=====================")
        print("[Fold %d] Generate Recommendations" % (i+1))
        trainset = dataset.construct_trainset([(*record, None) for record in train_df.to_records(index=False)])
        testset = dataset.construct_testset([(*record, None) for record in test_df.to_records(index=False)])

        user_embeddings = map_embedding(user_embeddings_dict, trainset=trainset, user_based=True)
        item_embeddings = map_embedding(item_embeddings_dict, trainset=trainset, user_based=False)
        user_sim = cosine_similarity(user_embeddings, user_embeddings)

        Ks = [5, 10, 15, 20, 25, 30]
        mean_absolute_errors_f = defaultdict(list)
        privacy_risks_f = defaultdict(list)
        fraction_protected_users_f = defaultdict(list)
        for k_idx, k in enumerate(Ks):
            # UserKNN and compute privacy risk threshold tau
            model = UserKNN(k=k, threshold=np.inf, use_dp=False, sim=user_sim)
            model.fit(trainset).test(testset)
            mean_absolute_errors_f["UserKNN"].append(mean_absolute_error(model)[0])
            privacy_risks_f["UserKNN"].append(avg_privacy_risk(model)[0])
            fraction_protected_users_f["UserKNN"].append(fraction_vulnerables(model))
            tau = model.get_privacy_threshold()
            print("[" + str(k) + " Neighbors] UserKNN", end=", ")

            # UserKNN_DP
            model = UserKNN(k=k, threshold=tau, use_dp=True, sim=user_sim)
            model.fit(trainset).test(testset)
            mean_absolute_errors_f["UserKNN_DP"].append(mean_absolute_error(model)[0])
            privacy_risks_f["UserKNN_DP"].append(avg_privacy_risk(model)[0])
            fraction_protected_users_f["UserKNN_DP"].append(fraction_vulnerables(model))
            print("UserKNN_DP", end=", ")

            # Full_DP
            model = UserKNN(k=k, threshold=0, use_dp=True, sim=user_sim)
            model.fit(trainset).test(testset)
            mean_absolute_errors_f["Full_DP"].append(mean_absolute_error(model)[0])
            privacy_risks_f["Full_DP"].append(avg_privacy_risk(model)[0])
            fraction_protected_users_f["Full_DP"].append(fraction_vulnerables(model))
            print("Full_DP", end=", ")

            # UserKNN+Reuse_DP
            model = UserKNN(k=k, threshold=tau, use_dp=True, sim=user_sim, explicit_reuse_option="static")
            model.fit(trainset).test(testset)
            mean_absolute_errors_f["UserKNN_Static_DP"].append(mean_absolute_error(model)[0])
            privacy_risks_f["UserKNN_Static_DP"].append(avg_privacy_risk(model)[0])
            fraction_protected_users_f["UserKNN_Static_DP"].append(fraction_vulnerables(model))
            print("UserKNN+Reuse_DP", end=", ")

            # NeuReuse_DP
            model = UserKNN(k=k, threshold=tau, use_dp=True, sim=user_sim, explicit_reuse_option="dynamic",
                            item_embeddings=item_embeddings)
            model.fit(trainset).test(testset)
            mean_absolute_errors_f["UserKNN_Dynamic_DP"].append(mean_absolute_error(model)[0])
            privacy_risks_f["UserKNN_Dynamic_DP"].append(avg_privacy_risk(model)[0])
            fraction_protected_users_f["UserKNN_Dynamic_DP"].append(fraction_vulnerables(model))
            print("NeuReuse_DP")

        for method in all_methods:
            mean_absolute_errors[method].append(mean_absolute_errors_f[method])
            privacy_risks[method].append(privacy_risks_f[method])
            fraction_protected_users[method].append(fraction_protected_users_f[method])

    if args.only_first_fold:
        break

if args.generate_recommendations:
    for method in all_methods:
        mean_absolute_errors[method] = np.mean(mean_absolute_errors[method], axis=0)
        privacy_risks[method] = np.mean(privacy_risks[method], axis=0)
        fraction_protected_users[method] = np.mean(fraction_protected_users[method], axis=0)

    with open("results/NeuReuse/" + args.dataset_name + "/mean_absolute_error" + ".pkl", "wb") as f:
        pl.dump(mean_absolute_errors, f)
    with open("results/NeuReuse/" + args.dataset_name + "/privacy_risk" + ".pkl", "wb") as f:
        pl.dump(privacy_risks, f)
    with open("results/NeuReuse/" + args.dataset_name + "/fraction_vulnerables" + ".pkl", "wb") as f:
        pl.dump(fraction_protected_users, f)



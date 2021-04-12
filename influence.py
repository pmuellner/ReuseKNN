import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True)
from algorithms.knn_neighborhood import UserKNN
from surprise import Dataset, Reader, accuracy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from datetime import datetime as dt
import pickle


def get_top_n(predictions, n=10):
    top_n_recommendations = defaultdict(list)
    for ruid, riid, true_r, est_r, _ in predictions:
        top_n_recommendations[ruid].append((riid, est_r))

    for ruid, user_ratings in top_n_recommendations.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n_recommendations[ruid] = [riid for riid, _ in user_ratings[:n]]

    return top_n_recommendations

def get_mae(predictions):
    errors = defaultdict(list)
    for ruid, _, true_r, est_r, _ in predictions:
        errors[ruid].append(np.abs(true_r - est_r))
    return {ruid: np.mean(errors[ruid]) for ruid in errors.keys()}

def measure_influence_top_n(base_top_n, top_n):  
    jdists = []
    for ruid in top_n.keys():
        jsim = len(set(top_n[ruid]).intersection(base_top_n[ruid])) / len(set(top_n[ruid]).union(base_top_n[ruid]))
        jdists.append(1 - jsim)
    return np.mean(jdists)

def measure_influence_mae(base_mae, mae):
    dists = [mae[ruid] / base_mae[ruid] for ruid in mae.keys()]
    return np.mean(dists)

def get_top_n_mentors(model, n=10):
    nr_of_students = [(model.trainset.to_raw_uid(iuid), len(students)) for iuid, students in model.students.items()]
    top_n_mentors = sorted(nr_of_students, key=lambda t: t[1])[::-1][:n]
    top_n_mentors = [ruid for ruid, _ in top_n_mentors]
    
    return top_n_mentors


# # Dataset
# ## Read dataset
name = "ml-100k"
if name == "ml-100k":
    data_df = pd.read_csv("data/ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"], usecols=["user_id", "item_id", "rating"])
elif name == "ml-1m":
    data_df = pd.read_csv("data/ml-1m/ratings.dat", sep="::", names=["user_id", "item_id", "rating", "timestamp"], usecols=["user_id", "item_id", "rating"])
elif name == "jester":
    data_df = pd.read_csv("data/jester/sample.csv", sep=";", names=["user_id", "item_id", "rating"])
elif name == "goodreads":
    data_df = pd.read_csv("data/goodreads/sample.csv", sep=";", names=["user_id", "item_id", "rating"])
else:
    raise Exception

data_df["user_id"] = data_df["user_id"].map({b: a for a, b in enumerate(data_df["user_id"].unique())})
data_df["item_id"] = data_df["item_id"].map({b: a for a, b in enumerate(data_df["item_id"].unique())})

# ## Train- and testset
reader = Reader(rating_scale=(1, 5))
train_df, test_df = train_test_split(data_df, test_size=0.2)
dataset = Dataset.load_from_df(data_df, reader=reader)
raw_trainset = [(ruid, riid, r, None) for ruid, riid, r in train_df.to_records(index=False)]
raw_testset = [(ruid, riid, r, None) for ruid, riid, r in test_df.to_records(index=False)]
trainset = Dataset.construct_trainset(dataset, raw_trainset)
testset = Dataset.construct_testset(dataset, raw_testset)


# # Model
Ks = [5, 10, 15, 30]
sim = UserKNN().compute_similarities(trainset, min_support=1)
pop = UserKNN().compute_popularities(trainset)
gain = UserKNN().compute_gain(trainset)

# ## Baseline models
base_top_n = defaultdict(list)
base_mae = defaultdict(list)
top_n_mentors = defaultdict(list)
for k in Ks:
    # UserKNN
    model = UserKNN(k=k, precomputed_sim=sim)
    model.fit(trainset)
    predictions = model.test(testset)
    base_top_n["UserKNN"].append(get_top_n(predictions))
    base_mae["UserKNN"].append(get_mae(predictions))
    top_n_mentors["UserKNN"].append(get_top_n_mentors(model))
    
    # UserKNN + reuse
    model = UserKNN(k=k, precomputed_sim=sim, reuse=True)
    model.fit(trainset)
    predictions = model.test(testset)
    base_top_n["UserKNN + Reuse"].append(get_top_n(predictions))
    base_mae["UserKNN + Reuse"].append(get_mae(predictions))
    top_n_mentors["UserKNN + Reuse"].append(get_top_n_mentors(model))
    
    # Popularity
    model = UserKNN(k=k, precomputed_sim=sim, precomputed_pop=pop, tau_2=0.5)
    model.fit(trainset)
    predictions = model.test(testset)
    base_top_n["Popularity"].append(get_top_n(predictions))
    base_mae["Popularity"].append(get_mae(predictions))
    top_n_mentors["Popularity"].append(get_top_n_mentors(model))
    
    # Popularity + Reuse
    model = UserKNN(k=k, precomputed_sim=sim, precomputed_pop=pop, tau_2=0.5, reuse=True)
    model.fit(trainset)
    predictions = model.test(testset)
    base_top_n["Popularity + Reuse"].append(get_top_n(predictions))
    base_mae["Popularity + Reuse"].append(get_mae(predictions))
    top_n_mentors["Popularity + Reuse"].append(get_top_n_mentors(model))
    
    # Gain
    model = UserKNN(k=k, precomputed_sim=sim, precomputed_pop=pop, precomputed_gain=gain, tau_4=0.5)
    model.fit(trainset)
    predictions = model.test(testset)
    base_top_n["Gain"].append(get_top_n(predictions))
    base_mae["Gain"].append(get_mae(predictions))
    top_n_mentors["Gain"].append(get_top_n_mentors(model))
    
    # Gain + reuse
    model = UserKNN(k=k, precomputed_sim=sim, precomputed_pop=pop, precomputed_gain=gain, tau_4=0.5, reuse=True)
    model.fit(trainset)
    predictions = model.test(testset)
    base_top_n["Gain + Reuse"].append(get_top_n(predictions))
    base_mae["Gain + Reuse"].append(get_mae(predictions))
    top_n_mentors["Gain + Reuse"].append(get_top_n_mentors(model))

print("Base models finished")

all_mentors = set()
for mentors in top_n_mentors.values():
    all_mentors = all_mentors.union(set(np.ravel(mentors)))

influence_top_n = defaultdict(list)
influence_mae = defaultdict(list)

all_users = set([trainset.to_raw_uid(iuid) for iuid in trainset.all_users()])
no_mentors = set(np.random.choice(list(all_users.difference(all_mentors)), replace=False, size=100))

starttime = dt.now()
for ruid in no_mentors.union(all_mentors):
    train_wo_df = train_df[train_df["user_id"] != ruid]
    raw_trainset_wo = [(ruid, riid, r, None) for ruid, riid, r in train_wo_df.to_records(index=False)]
    trainset_wo = Dataset.construct_trainset(dataset, raw_trainset_wo)
    
    sim = UserKNN().compute_similarities(trainset_wo, min_support=1)
    pop = UserKNN().compute_popularities(trainset_wo)
    gain = UserKNN().compute_gain(trainset_wo)
    
    for k in Ks:
        # UserKNN
        if ruid in top_n_mentors["UserKNN"][Ks.index(k)] or ruid in no_mentors:
            model = UserKNN(k=k, precomputed_sim=sim)
            model.fit(trainset_wo)
            predictions = model.test(testset)
        
            ruid_influence_top_n = measure_influence_top_n(base_top_n["UserKNN"][Ks.index(k)], get_top_n(predictions))
            if len(influence_top_n["UserKNN"]) < len(Ks):
                influence_top_n["UserKNN"] = [[] for _ in Ks]
                influence_top_n["UserKNN"][Ks.index(k)] = [(ruid, ruid_influence_top_n)]
            else:
                influence_top_n["UserKNN"][Ks.index(k)].append((ruid, ruid_influence_top_n))
                
            ruid_influence_mae = measure_influence_mae(base_mae["UserKNN"][Ks.index(k)], get_mae(predictions))
            if len(influence_mae["UserKNN"]) < len(Ks):
                influence_mae["UserKNN"] = [[] for _ in Ks]
                influence_mae["UserKNN"][Ks.index(k)] = [(ruid, ruid_influence_mae)]
            else:
                influence_mae["UserKNN"][Ks.index(k)].append((ruid, ruid_influence_mae))
        
        # UserKNN + Reuse
        if ruid in top_n_mentors["UserKNN + Reuse"][Ks.index(k)] or ruid in no_mentors:
            model = UserKNN(k=k, precomputed_sim=sim, reuse=True)
            model.fit(trainset_wo)
            predictions = model.test(testset)
        
            ruid_influence_top_n = measure_influence_top_n(base_top_n["UserKNN + Reuse"][Ks.index(k)], get_top_n(predictions))
            if len(influence_top_n["UserKNN + Reuse"]) < len(Ks):
                influence_top_n["UserKNN + Reuse"] = [[] for _ in Ks]
                influence_top_n["UserKNN + Reuse"][Ks.index(k)] = [(ruid, ruid_influence_top_n)]
            else:
                influence_top_n["UserKNN + Reuse"][Ks.index(k)].append((ruid, ruid_influence_top_n))

            ruid_influence_mae = measure_influence_mae(base_mae["UserKNN + Reuse"][Ks.index(k)], get_mae(predictions))
            if len(influence_mae["UserKNN + Reuse"]) < len(Ks):
                influence_mae["UserKNN + Reuse"] = [[] for _ in Ks]
                influence_mae["UserKNN + Reuse"][Ks.index(k)] = [(ruid, ruid_influence_mae)]
            else:
                influence_mae["UserKNN + Reuse"][Ks.index(k)].append((ruid, ruid_influence_mae))
        
        # Popularity
        if ruid in top_n_mentors["Popularity"][Ks.index(k)] or ruid in no_mentors:
            model = UserKNN(k=k, precomputed_sim=sim, precomputed_pop=pop, tau_2=0.5)
            model.fit(trainset_wo)
            predictions = model.test(testset)
        
            ruid_influence_top_n = measure_influence_top_n(base_top_n["Popularity"][Ks.index(k)], get_top_n(predictions))
            if len(influence_top_n["Popularity"]) < len(Ks):
                influence_top_n["Popularity"] = [[] for _ in Ks]
                influence_top_n["Popularity"][Ks.index(k)] = [(ruid, ruid_influence_top_n)]
            else:
                influence_top_n["Popularity"][Ks.index(k)].append((ruid, ruid_influence_top_n))

            ruid_influence_mae = measure_influence_mae(base_mae["Popularity"][Ks.index(k)], get_mae(predictions))
            if len(influence_mae["Popularity"]) < len(Ks):
                influence_mae["Popularity"] = [[] for _ in Ks]
                influence_mae["Popularity"][Ks.index(k)] = [(ruid, ruid_influence_mae)]
            else:
                influence_mae["Popularity"][Ks.index(k)].append((ruid, ruid_influence_mae))
        
        # Popularity + Reuse
        if ruid in top_n_mentors["Popularity + Reuse"][Ks.index(k)] or ruid in no_mentors:
            model = UserKNN(k=k, precomputed_sim=sim, precomputed_pop=pop, tau_2=0.5, reuse=True)
            model.fit(trainset_wo)
            predictions = model.test(testset)
        
            ruid_influence_top_n = measure_influence_top_n(base_top_n["Popularity + Reuse"][Ks.index(k)], get_top_n(predictions))
            if len(influence_top_n["Popularity + Reuse"]) < len(Ks):
                influence_top_n["Popularity + Reuse"] = [[] for _ in Ks]
                influence_top_n["Popularity + Reuse"][Ks.index(k)] = [(ruid, ruid_influence_top_n)]
            else:
                influence_top_n["Popularity + Reuse"][Ks.index(k)].append((ruid, ruid_influence_top_n))

            ruid_influence_mae = measure_influence_mae(base_mae["Popularity + Reuse"][Ks.index(k)], get_mae(predictions))
            if len(influence_mae["Popularity + Reuse"]) < len(Ks):
                influence_mae["Popularity + Reuse"] = [[] for _ in Ks]
                influence_mae["Popularity + Reuse"][Ks.index(k)] = [(ruid, ruid_influence_mae)]
            else:
                influence_mae["Popularity + Reuse"][Ks.index(k)].append((ruid, ruid_influence_mae))
        
        # Gain
        if ruid in top_n_mentors["Gain"][Ks.index(k)] or ruid in no_mentors:
            model = UserKNN(k=k, precomputed_sim=sim, precomputed_gain=gain, tau_4=0.5)
            model.fit(trainset_wo)
            predictions = model.test(testset)
        
            ruid_influence_top_n = measure_influence_top_n(base_top_n["Gain"][Ks.index(k)], get_top_n(predictions))
            if len(influence_top_n["Gain"]) < len(Ks):
                influence_top_n["Gain"] = [[] for _ in Ks]
                influence_top_n["Gain"][Ks.index(k)] = [(ruid, ruid_influence_top_n)]
            else:
                influence_top_n["Gain"][Ks.index(k)].append((ruid, ruid_influence_top_n))

            ruid_influence_mae = measure_influence_mae(base_mae["Gain"][Ks.index(k)], get_mae(predictions))
            if len(influence_mae["Gain"]) < len(Ks):
                influence_mae["Gain"] = [[] for _ in Ks]
                influence_mae["Gain"][Ks.index(k)] = [(ruid, ruid_influence_mae)]
            else:
                influence_mae["Gain"][Ks.index(k)].append((ruid, ruid_influence_mae))
        
        # Gain + Reuse
        if ruid in top_n_mentors["Gain + Reuse"][Ks.index(k)] or ruid in no_mentors:
            model = UserKNN(k=k, precomputed_sim=sim, precomputed_gain=gain, tau_4=0.5, reuse=True)
            model.fit(trainset_wo)
            predictions = model.test(testset)
        
            ruid_influence_top_n = measure_influence_top_n(base_top_n["Gain + Reuse"][Ks.index(k)], get_top_n(predictions))
            if len(influence_top_n["Gain + Reuse"]) < len(Ks):
                influence_top_n["Gain + Reuse"] = [[] for _ in Ks]
                influence_top_n["Gain + Reuse"][Ks.index(k)] = [(ruid, ruid_influence_top_n)]
            else:
                influence_top_n["Gain + Reuse"][Ks.index(k)].append((ruid, ruid_influence_top_n))

            ruid_influence_mae = measure_influence_mae(base_mae["Gain + Reuse"][Ks.index(k)], get_mae(predictions))
            if len(influence_mae["Gain + Reuse"]) < len(Ks):
                influence_mae["Gain + Reuse"] = [[] for _ in Ks]
                influence_mae["Gain + Reuse"][Ks.index(k)] = [(ruid, ruid_influence_mae)]
            else:
                influence_mae["Gain + Reuse"][Ks.index(k)].append((ruid, ruid_influence_mae))
        
    print("User id: %d, Time elapsed: %s" % (ruid, dt.now() - starttime))


f = open("results/" + name + "/influence/influence_top_n.pkl", "wb")
pickle.dump(influence_top_n, f)
f.close()

f = open("results/" + name + "/influence/influence_mae.pkl", "wb")
pickle.dump(influence_mae, f)
f.close()

f = open("results/" + name + "/influence/k.pkl", "wb")
pickle.dump(Ks, f)
f.close()

f = open("results/" + name + "/influence/top_10_mentors.pkl", "wb")
pickle.dump(top_n_mentors, f)
f.close()

f = open("results/" + name + "/influence/base_top_10.pkl", "wb")
pickle.dump(base_top_n, f)
f.close()

f = open("results/" + name + "/influence/base_mae.pkl", "wb")
pickle.dump(base_mae, f)
f.close()
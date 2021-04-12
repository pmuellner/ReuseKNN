#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True)
from algorithms.knn_neighborhood import UserKNN
import pandas as pd
from surprise import Dataset, Reader, accuracy, NMF
from surprise.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle


# In[46]:


def mentors_at_q(model):
    q_max = np.max([len(nmentors) for nmentors in model.n_mentors_at_q.values()])
    avg_n_mentors_at_q = [0]
    for q in range(1, q_max+1):
        avg_at_q = []
        n = 0
        for iuid, mentors in model.n_mentors_at_q.items():
            if len(mentors) >= q:
                avg_at_q.append(mentors[q-1])
                n += 1
        avg_n_mentors_at_q.append(np.mean(avg_at_q))
    
    return avg_n_mentors_at_q

def students_at_q(model):
    q_max = np.max([len(nstudents) for nstudents in model.n_students_at_q.values()])
    avg_n_students_at_q = [0]
    for q in range(1, q_max+1):
        avg_at_q = []
        n = 0
        for iuid, students in model.n_students_at_q.items():
            if len(students) >= q:
                avg_at_q.append(students[q-1])
                n += 1
        avg_n_students_at_q.append(np.mean(avg_at_q))

    return avg_n_students_at_q


# # Read data 

# In[47]:


data_df = pd.read_csv("data/ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"], usecols=["user_id", "item_id", "rating"])

reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data_df, reader=reader)
folds = KFold(n_splits=5)


# In[48]:


avg_n_mentors_at_q_per_fold = defaultdict(list)
avg_n_students_at_q_per_fold = defaultdict(list)
for trainset, testset in folds.split(dataset):
    sim = UserKNN.compute_similarities(trainset, min_support=1)
    pop = UserKNN.compute_popularities(trainset)
    gain = UserKNN.compute_gain(trainset)
    
    # KNN
    model = UserKNN(k=10, precomputed_sim=sim)
    model.fit(trainset)
    _ = model.test(testset)
    avg_n_mentors_at_q = mentors_at_q(model)
    avg_n_students_at_q = students_at_q(model)
    avg_n_mentors_at_q_per_fold["UserKNN"].append(avg_n_mentors_at_q)
    avg_n_students_at_q_per_fold["UserKNN"].append(n_students_at_q)
    
    # Popularity
    model = UserKNN(k=10, precomputed_sim=sim, precomputed_pop=pop, tau_2=0.5)
    model.fit(trainset)
    _ = model.test(testset)
    avg_n_mentors_at_q = mentors_at_q(model)
    n_students_at_q = students_at_q(model)
    avg_n_mentors_at_q_per_fold["Popularity"].append(avg_n_mentors_at_q)
    avg_n_students_at_q_per_fold["Popularity"].append(n_students_at_q)
    
    # Gain
    model = UserKNN(k=10, precomputed_sim=sim, precomputed_gain=gain, tau_4=0.5)
    model.fit(trainset)
    _ = model.test(testset)
    avg_n_mentors_at_q = mentors_at_q(model)
    n_students_at_q = students_at_q(model)
    avg_n_mentors_at_q_per_fold["Gain"].append(avg_n_mentors_at_q)
    avg_n_students_at_q_per_fold["Gain"].append(n_students_at_q)

    # KNN + reuse
    model = UserKNN(k=10, reuse=True, precomputed_sim=sim)
    model.fit(trainset)
    _ = model.test(testset)
    avg_n_mentors_at_q = mentors_at_q(model)
    n_students_at_q = students_at_q(model)
    avg_n_mentors_at_q_per_fold["UserKNN + Reuse"].append(avg_n_mentors_at_q)
    avg_n_students_at_q_per_fold["UserKNN + Reuse"].append(n_students_at_q)

    # Popularity + reuse
    model = UserKNN(k=10, reuse=True, precomputed_sim=sim, precomputed_pop=pop, tau_2=0.5)
    model.fit(trainset)
    _ = model.test(testset)
    avg_n_mentors_at_q = mentors_at_q(model)
    n_students_at_q = students_at_q(model)
    avg_n_mentors_at_q_per_fold["Popularity + Reuse"].append(avg_n_mentors_at_q)
    avg_n_students_at_q_per_fold["Popularity + Reuse"].append(n_students_at_q)

    # Gain + reuse
    model = UserKNN(k=10, reuse=True, precomputed_sim=sim, precomputed_gain=gain, tau_4=0.5)
    model.fit(trainset)
    _ = model.test(testset)
    avg_n_mentors_at_q = mentors_at_q(model)
    n_students_at_q = students_at_q(model)
    avg_n_mentors_at_q_per_fold["Gain + Reuse"].append(avg_n_mentors_at_q)
    avg_n_students_at_q_per_fold["Gain + Reuse"].append(n_students_at_q)


# In[49]:


def compute_avg(avg_nr):
    min_n_queries = min([len(l) for l in avg_nr])
    avg = np.mean([l[:min_n_queries] for l in avg_nr], axis=0)
    return avg

avg_userknn = compute_avg(avg_n_mentors_at_q_per_fold["UserKNN"])
avg_userknn_reuse = compute_avg(avg_n_mentors_at_q_per_fold["UserKNN + Reuse"])
avg_pop = compute_avg(avg_n_mentors_at_q_per_fold["Popularity"])
avg_pop_reuse = compute_avg(avg_n_mentors_at_q_per_fold["Popularity + Reuse"])
avg_gain = compute_avg(avg_n_mentors_at_q_per_fold["Gain"])
avg_gain_reuse = compute_avg(avg_n_mentors_at_q_per_fold["Gain + Reuse"])

n_mentors = {"UserKNN": avg_userknn, "UserKNN + Reuse": avg_userknn_reuse, "Popularity": avg_pop, 
             "Popularity + Reuse": avg_pop_reuse, "Gain": avg_gain, "Gain + Reuse": avg_gain_reuse}

f = open("results/ml-100k/nr_of_mentors.pkl", "wb")
pickle.dump(n_mentors, f)
f.close()

avg_userknn = compute_avg(avg_n_students_at_q_per_fold["UserKNN"])
avg_userknn_reuse = compute_avg(avg_n_students_at_q_per_fold["UserKNN + Reuse"])
avg_pop = compute_avg(avg_n_students_at_q_per_fold["Popularity"])
avg_pop_reuse = compute_avg(avg_n_students_at_q_per_fold["Popularity + Reuse"])
avg_gain = compute_avg(avg_n_students_at_q_per_fold["Gain"])
avg_gain_reuse = compute_avg(avg_n_students_at_q_per_fold["Gain + Reuse"])

n_mentors = {"UserKNN": avg_userknn, "UserKNN + Reuse": avg_userknn_reuse, "Popularity": avg_pop, 
             "Popularity + Reuse": avg_pop_reuse, "Gain": avg_gain, "Gain + Reuse": avg_gain_reuse}

f = open("results/ml-100k/nr_of_students.pkl", "wb")
pickle.dump(n_mentors, f)
f.close()


# In[ ]:





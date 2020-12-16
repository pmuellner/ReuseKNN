import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs": np.get_include()},
                  reload_support=True)
from algorithms.knn_neighborhood import UserKNN
import pandas as pd
from surprise import Dataset, Reader, accuracy, KNNBasic
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import skew, pearsonr, spearmanr
from networkx.algorithms.centrality import degree_centrality
from networkx.algorithms.approximation.clustering_coefficient import average_clustering
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community.quality import modularity
from datetime import datetime as dt
from sklearn.metrics.pairwise import cosine_similarity

def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def get_ratio(model):
    ratios = []
    for uid in model.mae_u.keys():
        mae = model.mae_u[uid]
        s = len(model.students[uid])
        if s > 0:
            ratios.append(mae / s)
            #ratios.append(mae)
        else:
            ratios.append(0)

    return ratios

def get_degree_centrality(model):
    centralities = degree_centrality(model.trust_graph)
    degree_centralities = []
    for students in model.students.values():
        leakage = np.sum([centralities[s] for s in students])
        degree_centralities.append(leakage)
    return np.mean(degree_centralities)


def get_modularity(model):
    communities = [c for c in girvan_newman(model.trust_graph)]
    print(len(communities))
    for c in communities:
        print(len(c))
    return modularity(model.trust_graph, communities)

def popularity_degree_correlation(model, popularity_distribution):
    N = len(popularity_distribution)
    degree = [len(model.students[u]) / N for u in range(N)]
    popularity = [popularity_distribution[u] / np.sum(popularity_distribution) for u in range(N)]

    r, p = pearsonr(degree, popularity)
    return r, p

def run(trainset, testset, K, configuration={}):
    random = configuration.get("random", False)
    reuse = configuration.get("reuse", False)
    tau_1 = configuration.get("tau_1", 0.0)
    tau_2 = configuration.get("tau_2", 0.0)
    sim = configuration.get("precomputed_sim", None)
    pop = configuration.get("precomputed_pop", None)
    act = configuration.get("precomputed_act", None)

    config_str = str({"random": random, "reuse": reuse, "tau_1": tau_1, "tau_2": tau_2, "precomputed_sim": sim is None, "precomputed_pop": pop is None, "precomputed_act": act is None})

    t0 = dt.now()
    print("Started training model with K: " + str(K) + " and " + config_str)
    results = defaultdict(list)
    for k in K:
        model = UserKNN(k=k, random=random, reuse=reuse, tau_1=tau_1, tau_2=tau_2, precomputed_sim=sim, precomputed_pop=pop, precomputed_act=act)
        model.fit(trainset)
        predictions = model.test(testset)
        results["models"].append(model)
        results["predictions"].append(predictions)
    print("Training finished after " + str(dt.now() - t0))

    return results["models"], results["predictions"]

def eval_network(models, measurements=[]):
    results = defaultdict(list)
    for m_at_k in models:
        if "outdegree" in measurements:
            outdegree = np.mean([len(students) for students in m_at_k.students.values()])
            results["outdegree"].append(outdegree)
        if "pathlength" in measurements:
            pathlength = m_at_k.get_path_length()
            results["pathlength"].append(pathlength)
        if "clustering_coefficient" in measurements:
            cc = average_clustering(m_at_k.trust_graph)
            results["clustering_coefficient"].append(cc)
        if "ratio" in measurements:
            ratio = get_ratio(m_at_k)
            results["ratio"].append(ratio)

    return results

def eval_ratings(predictions, measurements=[]):
    results = defaultdict(list)
    for p_at_k in predictions:
        if "mae" in measurements:
            mae = accuracy.mae(p_at_k, verbose=False)
            results["mae"].append(mae)

    return results

def recommendation_frequency(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    item_distribution = dict()
    for uid, ratings_u in top_n.items():
        for iid, _ in ratings_u:
            item_distribution[iid] = item_distribution.get(iid, 0) + 1

    return item_distribution

def eval_topn(predictions, trainset, n, measurements=[]):
    results = defaultdict(list)
    for p_at_k in predictions:
        top_n_data = defaultdict(list)
        for uid, iid, true_r, est, _ in p_at_k:
            top_n_data[uid].append((iid, est))

        est_top_items = defaultdict(set)
        for uid, user_ratings in top_n_data.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            est_top_items[uid] = set(iid for iid, _ in user_ratings[:n])

        if "diversity" in measurements:
            for uid in trainset.all_users():
               items = est_top_items[uid]



        if "serendipity" in measurements:
            pass
        if "novelty" in measurements:
            pass

    return results

def diversity(predictions, data_df, n=10):
    results = []
    df = data_df.pivot(index="item_id", columns="user_id", values="rating").fillna(0)
    item_similarity = cosine_similarity(df)
    for p_at_k in predictions:
        top_n_data = defaultdict(list)
        for uid, iid, true_r, est, _ in p_at_k:
            top_n_data[uid].append((iid, est))

        diversities = []
        for uid, user_ratings in top_n_data.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            items = set(iid for iid, _ in user_ratings[:n])
            sum = 0
            for i1 in items:
                for i2 in items:
                    if i1 != i2:
                        sum += 1 - item_similarity[i1, i2]
            diversity_u = sum / (n**2 - n)
            diversities.append(diversity_u)
        results.append(np.mean(diversities))
    return results

def novelty(predictions, data_df, n=10):
    results = []
    p_seen = data_df.groupby("item_id").size() / len(data_df)
    for p_at_k in predictions:
        top_n_data = defaultdict(list)
        for uid, iid, true_r, est, _ in p_at_k:
            top_n_data[uid].append((iid, est))

        novelties = []
        for uid, user_ratings in top_n_data.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            items = set(iid for iid, _ in user_ratings[:n])
            #novelty_u = np.mean([1 - p_seen[i] for i in items])
            novelty_u = np.mean([1 - np.log(p_seen[i]) for i in items])
            novelties.append(novelty_u)
        results.append(np.mean(novelties))
    return results

def serendipity(predictions, trainset, data_df, n=10):
    results = []
    df = data_df.pivot(index="item_id", columns="user_id", values="rating").fillna(0)
    item_similarity = cosine_similarity(df)
    np.fill_diagonal(item_similarity, 0)
    for p_at_k in predictions:
        top_n_data = defaultdict(list)
        for uid, iid, true_r, est, _ in p_at_k:
            top_n_data[uid].append((iid, est))

        serendipities = []
        for uid, user_ratings in top_n_data.items():
            train_items = np.array([iid for iid, _ in trainset.ur[uid]])
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            serendipities_u = []
            for iid, score in user_ratings[:n]:
                unexpectedness = []
                for iid_train in train_items:
                    s = item_similarity[iid, iid_train]
                    if s != 0:
                        unexpectedness.append(1. / s)
                    else:
                        unexpectedness.append(0)
                s = score * np.mean(unexpectedness)
                serendipities_u.append(s)

            serendipities.append(np.mean(serendipities_u))
        results.append(np.mean(serendipities))
    return results

data_df = pd.read_csv("data/ml-100k/u.data", sep="\t")
#data_df = pd.read_csv("data/ml-latest-small/ratings.csv", sep=",")
data_df.columns = ["user_id", "item_id", "rating", "timestamp"]
data_df.drop(columns=["timestamp"], axis=1, inplace=True)
data_df["user_id"] = data_df["user_id"].map({b: a for a, b in enumerate(data_df["user_id"].unique())})
data_df["item_id"] = data_df["item_id"].map({b: a for a, b in enumerate(data_df["item_id"].unique())})

# train test split for top n recommendations
"""data_df.sort_values("timestamp", ascending=True, inplace=True)
trainset, testset = [], []
for u, df in data_df.groupby("user_id"):
    n_test_ratings = np.ceil(len(df) * 0.20).astype(int)
    rating_tuples = list(df.to_records(index=False))
    trainset_u = rating_tuples[:-n_test_ratings]
    testset_u = rating_tuples[-n_test_ratings:]
    trainset.extend(trainset_u)
    testset.extend(testset_u)

reader = Reader(rating_scale=(1, 5))
dataset = Dataset(reader)
trainset = dataset.construct_trainset(trainset)
testset = dataset.construct_testset(testset)"""

reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data_df, reader=reader)
trainset, testset = train_test_split(dataset, test_size=0.2)

train_df = pd.DataFrame(trainset.build_testset(), columns=["user_id", "item_id", "rating"])
test_df = pd.DataFrame(testset, columns=["user_id", "item_id", "rating"])

user_popularity = (train_df.groupby("user_id").size() / train_df["item_id"].nunique()).to_dict()
item_popularity = (train_df.groupby("item_id").size() / train_df["user_id"].nunique()).to_dict()
item_popularity_ranking = dict(sorted(item_popularity.items(), key=lambda t: t[1], reverse=True))

"""covered_items = set()
coverage = []
average_popularity = []
for uid, size in train_df.groupby("user_id").size().sort_values(ascending=False).iteritems():
    new_items = set(train_df[train_df["user_id"] == uid]["item_id"].unique())
    new_covered_items = new_items.intersection(test_df["item_id"].unique())
    average_popularity.append(np.mean(item_popularity[new_items].values))

    covered_items = covered_items.union(new_covered_items)
    fraction = len(covered_items) / test_df["item_id"].nunique()
    coverage.append(fraction)

plt.plot(average_popularity)
plt.xlabel(r"Most Popular Users")
plt.ylabel("Average Item Popularity")
plt.show()

plt.plot(coverage)
plt.xlabel(r"Top $n$ users")
plt.ylabel("Coverage")
plt.show()"""

sim = UserKNN.compute_similarities(trainset, min_support=1)
pop = UserKNN.compute_popularities(trainset)
act = UserKNN.compute_activities(trainset)
rr = UserKNN.compute_rr(trainset)

K = np.arange(1, 30, 5)

# Similarity neighborhood
model_at_k, predictions_at_k = run(trainset, testset, K=K, configuration={"precomputed_sim": sim})
diversity_sim = diversity(predictions_at_k, data_df, n=10)
novelty_sim = novelty(predictions_at_k, data_df, n=10)
serendipity_sim = serendipity(predictions_at_k, trainset, data_df, n=10)
resratings_sim = eval_ratings(predictions_at_k, measurements=["mae"])
resnetwork_sim = eval_network(model_at_k, measurements=["outdegree"])

# Similarity neighborhood and reusage
model_at_k, predictions_at_k = run(trainset, testset, K=K, configuration={"reuse": True, "precomputed_sim": sim})
diversity_simreuse = diversity(predictions_at_k, data_df, n=10)
novelty_simreuse = novelty(predictions_at_k, data_df, n=10)
serendipity_simreuse = serendipity(predictions_at_k, trainset, data_df, n=10)
resratings_simreuse = eval_ratings(predictions_at_k, measurements=["mae"])
resnetwork_simreuse = eval_network(model_at_k, measurements=["outdegree"])

# Popularity neighborhood
model_at_k, predictions_at_k = run(trainset, testset, K=K, configuration={"tau_1": 0.5, "tau_2": 0.0, "reuse": True, "precomputed_sim": sim, "precomputed_pop": pop, "precomputed_act": rr})
diversity_simpop = diversity(predictions_at_k, data_df, n=10)
novelty_simpop = novelty(predictions_at_k, data_df, n=10)
serendipity_simpop = serendipity(predictions_at_k, trainset, data_df, n=10)
resratings_simpop = eval_ratings(predictions_at_k, measurements=["mae"])
resnetwork_simpop = eval_network(model_at_k, measurements=["outdegree"])

# Popularity neighborhood and reusage
model_at_k, predictions_at_k = run(trainset, testset, K=K, configuration={"tau_1": 0.5, "tau_2": 0.0, "reuse": True, "precomputed_sim": sim, "precomputed_pop": pop, "precomputed_act": act})
diversity_simpopreuse = diversity(predictions_at_k, data_df, n=10)
novelty_simpopreuse = novelty(predictions_at_k, data_df, n=10)
serendipity_simpopreuse = serendipity(predictions_at_k, trainset, data_df, n=10)
resratings_simpopreuse = eval_ratings(predictions_at_k, measurements=["mae"])
resnetwork_simpopreuse = eval_network(model_at_k, measurements=["outdegree"])

plt.plot(diversity_sim, resnetwork_sim["outdegree"], label="Sim", linewidth=1, linestyle="solid")
plt.plot(diversity_simreuse, resnetwork_simreuse["outdegree"], label="Sim+Reuse", linewidth=1, linestyle="dotted")
plt.plot(diversity_simpop, resnetwork_simpop["outdegree"], label="Sim+Pop", linewidth=1, linestyle="dashed")
plt.plot(diversity_simpopreuse, resnetwork_simpopreuse["outdegree"], label="Sim+Pop+Reuse", linewidth=1, linestyle="dashdot")
plt.ylabel("Outdegree")
plt.xlabel("Diversity")
plt.legend(title="Neighborhood Strategy")
plt.show()

plt.plot(novelty_sim, resnetwork_sim["outdegree"], label="Sim", linewidth=1, linestyle="solid")
plt.plot(novelty_simreuse, resnetwork_simreuse["outdegree"], label="Sim+Reuse", linewidth=1, linestyle="dotted")
plt.plot(novelty_simpop, resnetwork_simpop["outdegree"], label="Sim+Pop", linewidth=1, linestyle="dashed")
plt.plot(novelty_simpopreuse, resnetwork_simpopreuse["outdegree"], label="Sim+Pop+Reuse", linewidth=1, linestyle="dashdot")
plt.ylabel("Outdegree")
plt.xlabel("Novelty")
plt.legend(title="Neighborhood Strategy")
plt.show()

plt.plot(serendipity_sim, resnetwork_sim["outdegree"], label="Sim", linewidth=1, linestyle="solid")
plt.plot(serendipity_simreuse, resnetwork_simreuse["outdegree"], label="Sim+Reuse", linewidth=1, linestyle="dotted")
plt.plot(serendipity_simpop, resnetwork_simpop["outdegree"], label="Sim+Pop", linewidth=1, linestyle="dashed")
plt.plot(serendipity_simpopreuse, resnetwork_simpopreuse["outdegree"], label="Sim+Pop+Reuse", linewidth=1, linestyle="dashdot")
plt.ylabel("Outdegree")
plt.xlabel("Serendipity")
plt.legend(title="Neighborhood Strategy")
plt.show()


plt.plot(resratings_sim["mae"], resnetwork_sim["outdegree"], label="Sim", linewidth=1, linestyle="solid")
plt.plot(resratings_simreuse["mae"], resnetwork_simreuse["outdegree"], label="Sim+Reuse", linewidth=1, linestyle="dotted")
plt.plot(resratings_simpop["mae"], resnetwork_simpop["outdegree"], label="Sim+Pop", linewidth=1, linestyle="dashed")
plt.plot(resratings_simpopreuse["mae"], resnetwork_simpopreuse["outdegree"], label="Sim+Pop+Reuse", linewidth=1, linestyle="dashdot")
plt.xlabel("Mean Absolute Error")
plt.ylabel("Outdegree")
plt.legend(title="Neighborhood Strategy")
plt.show()

"""
skew_baseline = [skew(resnetwork_sim["ratio"][i]) for i in range(len(K))]
skew_reuse = [skew(resnetwork_simreuse["ratio"][i]) for i in range(len(K))]
skew_tau = [skew(resnetwork_simpop["ratio"][i]) for i in range(len(K))]
skew_reuse_tau = [skew(resnetwork_simpopreuse["ratio"][i]) for i in range(len(K))]
plt.plot(K, skew_baseline, label="Baseline", linestyle="solid")
plt.plot(K, skew_reuse, label="Reuse", linestyle="dotted")
plt.plot(K, skew_tau, label=r"$\tau=0.5$", linestyle="dashed")
plt.plot(K, skew_reuse_tau, label=r"Reuse+$\tau=0.5$", linestyle="dashdot")
plt.xlabel(r"Nr. of neighbors $k$")
plt.ylabel("Skew")
plt.legend()
plt.show()"""
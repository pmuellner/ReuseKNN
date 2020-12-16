cimport numpy as np
import numpy as np
import heapq
from collections import defaultdict, Counter
import networkx as nx
import pandas as pd

class PredictionImpossible(Exception):
    pass

class UserKNN:
    def __init__(self, k=40, min_k=1, random=False, reuse=False, tau_1=0, tau_2=0, tau_3=0, tau_4=0, precomputed_sim=None,
                 precomputed_pop=None, precomputed_act=None, precomputed_rr=None, precomputed_gain=None):
        self.k = k
        self.min_k = min_k
        self.mentors = defaultdict(set)
        self.students = defaultdict(set)
        self.reuse_neighbors = reuse
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.tau_3 = tau_3
        self.tau_4 = tau_4
        self.precomputed_sim = precomputed_sim
        self.precomputed_pop = precomputed_pop
        self.precomputed_act = precomputed_act
        self.precomputed_rr = precomputed_rr
        self.precomputed_gain = precomputed_gain
        self.random_neighbors = random

    def fit(self, trainset):
        self.trainset = trainset

        if self.precomputed_sim is None:
            self.sim = self.compute_similarities(self.trainset, self.min_k)
        else:
            self.sim = self.precomputed_sim.copy()

        if self.precomputed_pop is None:
            self.pop = self.compute_popularities(self.trainset)
        else:
            self.pop = self.precomputed_pop.copy()

        if self.precomputed_act is None:
            self.act = self.compute_activities(self.trainset)
        else:
            self.act = self.precomputed_act.copy()

        if self.precomputed_rr is None:
            self.rr = self.compute_rr(self.trainset)
        else:
            self.rr = self.precomputed_rr.copy()

        if self.precomputed_gain is None:
            self.gain = self.compute_gain(self.trainset)
        else:
            self.gain = self.precomputed_gain.copy()

        self.ranking = dict()

        # Tradeoff
        for u in self.trainset.all_users():
            simrank = {v: k for k, v in dict(enumerate(np.argsort(self.sim[u, :]))).items()}
            #poprank = {v: k for k, v in dict(enumerate(np.argsort(self.pop))).items()}
            poprank = {v: k for k, v in dict(enumerate(np.argsort(self.pop[u, :]))).items()}
            actrank = {v: k for k, v in dict(enumerate(np.argsort(self.act))).items()}
            rrrank = {v: k for k, v in dict(enumerate(np.argsort(self.rr))).items()}
            gainrank = {v: k for k, v in dict(enumerate(np.argsort(self.gain[u, :]))).items()}

            ranking_u = dict()
            for u_ in self.trainset.all_users():
                if u_ != u:
                    ranking_u[u_] = self.tau_1 * actrank[u_] + \
                                    self.tau_2 * poprank[u_] + \
                                    self.tau_3 * rrrank[u_] + \
                                    self.tau_4 * gainrank[u_] + \
                                    (1.0 - self.tau_1 - self.tau_2 - self.tau_3 - self.tau_4) * simrank[u_]
            self.ranking[u] = ranking_u
        return self

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        possible_mentors = set(u_ for u_, _ in self.trainset.ir[i])

        ranks = self.ranking[u]
        possible_mentors_data = [(u_, self.sim[u, u_], ranks[u_], r) for u_, r in self.trainset.ir[i]]
        possible_mentors_data = sorted(possible_mentors_data, key=lambda t: t[2])[::-1]

        possible_mentors_act = sorted([(u_, self.act[u_]) for u_ in possible_mentors], key=lambda t: t[1])[::-1]
        possible_mentors_actrank = {v: k for k, v in dict(enumerate([u_ for u_, _ in possible_mentors_act])).items()}

        if self.random_neighbors:
            mentors = np.random.choice(list(possible_mentors), replace=False, size=min(self.k, len(possible_mentors)))
            self.mentors[u] = self.mentors[u].union(set(mentors))
            for m in mentors:
                self.students[m] = self.students[m].union({u})
            neighbors = [(s, rank, r) for u_, s, rank, r in possible_mentors_data if u_ in mentors]
            k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        if self.reuse_neighbors:
            already_mentors = self.mentors[u].intersection(possible_mentors)

            n_new_mentors = self.k - len(already_mentors) if self.k > len(already_mentors) else 0
            new_mentors = []
            for u_, _, _, _ in possible_mentors_data:
                if len(new_mentors) >= n_new_mentors:
                    break
                elif u_ not in already_mentors:
                    new_mentors.append(u_)
                    self.mentors[u] = self.mentors[u].union({u_})
                    self.students[u_] = self.students[u_].union({u})
            new_mentors = set(new_mentors)

            mentors = new_mentors.union(already_mentors)
            neighbors = [(s, rank, r) for u_, s, rank, r in possible_mentors_data if u_ in mentors]
            k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        else:
            k_neighbors = heapq.nlargest(self.k, possible_mentors_data, key=lambda t: t[2])
            self.mentors[u] = self.mentors[u].union(set(u_ for u_, _, _, _  in k_neighbors))
            for u_, _, _, _ in k_neighbors:
                self.students[u_] = self.students[u_].union({u})
            k_neighbors = [(s, rank, r) for _, s, rank, r in k_neighbors]

        # UserKNN
        sum_sim = sum_ratings = actual_k = 0.0
        for (sim, _, r) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * r
                actual_k += 1

        if actual_k < self.min_k:
            raise PredictionImpossible('Not enough neighbors.')

        est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details

    def predict(self, uid, iid, r, clip=True):
        try:
            iuid = self.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = 'UKN__' + str(uid)
        try:
            iiid = self.trainset.to_inner_iid(iid)
        except ValueError:
            iiid = 'UKN__' + str(iid)

        details = {}
        try:
            est = self.estimate(iuid, iiid)

            # If the details dict was also returned
            if isinstance(est, tuple):
                est, details = est

            details['was_impossible'] = False

        except PredictionImpossible as e:
            est = self.default_prediction()
            details['was_impossible'] = True
            details['reason'] = str(e)

        # clip estimate into [lower_bound, higher_bound]
        if clip:
            lower_bound, higher_bound = self.trainset.rating_scale
            est = min(higher_bound, est)
            est = max(lower_bound, est)

        return uid, iid, r, est, details

    def test(self, testset):
        predictions = []
        absolute_errors = defaultdict(list)
        for user_id, item_id, rating in testset:
            uid, iid, r, r_, details = self.predict(user_id, item_id,  rating)
            predictions.append((uid, iid, r, r_, details))
            absolute_errors[uid].append(np.abs(r - r_))
        self.mae_u = {uid: np.mean(aes) for uid, aes in absolute_errors.items()}

        return predictions

    def default_prediction(self):
        return self.trainset.global_mean

    @staticmethod
    def compute_similarities(trainset, min_support):
        n_users = trainset.n_users
        ir = trainset.ir

        # sum (r_xy * r_x'y) for common ys
        cdef np.ndarray[np.double_t, ndim=2] prods
        # number of common ys
        cdef np.ndarray[np.int_t, ndim=2] freq
        # sum (r_xy ^ 2) for common ys
        cdef np.ndarray[np.double_t, ndim=2] sqi
        # sum (r_x'y ^ 2) for common ys
        cdef np.ndarray[np.double_t, ndim=2] sqj
        # the similarity matrix
        cdef np.ndarray[np.double_t, ndim=2] sim

        cdef int xi, xj
        cdef double ri, rj
        cdef int min_sprt = min_support

        prods = np.zeros((n_users, n_users), np.double)
        freq = np.zeros((n_users, n_users), np.int)
        sqi = np.zeros((n_users, n_users), np.double)
        sqj = np.zeros((n_users, n_users), np.double)
        sim = np.zeros((n_users, n_users), np.double)

        for y, y_ratings in ir.items():
            for xi, ri in y_ratings:
                for xj, rj in y_ratings:
                    freq[xi, xj] += 1
                    prods[xi, xj] += ri * rj
                    sqi[xi, xj] += ri**2
                    sqj[xi, xj] += rj**2

        for xi in range(n_users):
            sim[xi, xi] = 1
            for xj in range(xi + 1, n_users):
                if freq[xi, xj] < min_sprt:
                    sim[xi, xj] = 0
                else:
                    denum = np.sqrt(sqi[xi, xj] * sqj[xi, xj])
                    sim[xi, xj] = prods[xi, xj] / denum

                sim[xj, xi] = sim[xi, xj]

        return sim

    @staticmethod
    def compute_activities(trainset):
        n_ratings = np.zeros(trainset.n_users)
        for u, ratings in trainset.ur.items():
            n_ratings[u] = len(ratings)

        return n_ratings

    @staticmethod
    def compute_popularities(trainset):
        """item_popularities = np.zeros(trainset.n_items)
        for i, ratings in trainset.ir.items():
            item_popularities[i] = float(len(ratings)) / trainset.n_users

        reuse_potential = np.zeros(trainset.n_users)
        for u, ratings in trainset.ur.items():
            acc_rp = 0.0
            for i, _ in ratings:
                acc_rp += item_popularities[i]
            reuse_potential[u] = acc_rp
        return reuse_potential"""
        knowledge = defaultdict(list)
        for uid, ratings in trainset.ur.items():
            knowledge[uid].extend([iid for iid, _ in ratings])

        gain = np.zeros((trainset.n_users, trainset.n_users))
        for mentor in trainset.all_users():
            for student in trainset.all_users():
                n_queries = len(knowledge[student])
                if n_queries > 0:
                    g = float(len(set(knowledge[mentor]).intersection(knowledge[student]))) / len(knowledge[student])
                else:
                    g = 0.0
                gain[student, mentor] = g
        return gain

    @staticmethod
    def compute_rr(trainset, function=None):
        if function:
            f = function
        else:
            f = lambda x: 1.0 / x

        item_popularities = np.zeros(trainset.n_items)
        for i, ratings in trainset.ir.items():
            item_popularities[i] = float(len(ratings)) / trainset.n_users

        item_ranks = {v: k+1 for k, v in dict(enumerate(np.argsort(item_popularities)[::-1])).items()}

        rr = np.zeros(trainset.n_users)
        for u, ratings in trainset.ur.items():
            rr_u = 0.0
            for i, _ in ratings:
                rr_u += f(item_ranks[i])
            rr[u] = rr_u
        return rr

    @staticmethod
    def compute_gain(trainset):
        item_popularities = np.zeros(trainset.n_items)
        for i, ratings in trainset.ir.items():
            item_popularities[i] = float(len(ratings)) / trainset.n_users

        knowledge = defaultdict(list)
        for uid, ratings in trainset.ur.items():
            knowledge[uid].extend([iid for iid, _ in ratings])

        gain = np.zeros((trainset.n_users, trainset.n_users))
        for mentor in trainset.all_users():
            for student in trainset.all_users():
                n_queries = len(knowledge[student])
                if n_queries > 0:
                    g = float(len(set(knowledge[mentor]).intersection(knowledge[student]))) / len(knowledge[student])
                else:
                    g = 0.0

                unused_items = set(knowledge[mentor]).difference(knowledge[student])
                gain[student, mentor] = g
        return gain

    @property
    def trust_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.trainset.all_users())
        for u, students in self.students.items():
            for s in students:
                G.add_edge(u, s)

        return G

    def get_degree(self):
        G = self.trust_graph
        degree_counter = Counter(list(dict(nx.degree(G)).values()))
        P_degrees = {d: float(degree_counter[d]) / G.number_of_nodes() for d in degree_counter.keys()}
        return np.mean([d * p for d, p in P_degrees.items()])

    def get_path_length(self):
        G = self.trust_graph
        z1 = np.mean(list(dict(nx.degree(G)).values()))
        n_second_neighbors = []
        for v in G.nodes():
            N_v = G.neighbors(v)
            N_of_N = set()
            for n in N_v:
                N_of_N = N_of_N.union(set(G.neighbors(n)))

            n_second_neighbors.append(len(N_of_N))
        z2 = np.mean(n_second_neighbors)

        N = G.number_of_nodes()
        l = (np.log((N - 1) * (z2 - z1) + np.power(z1, 2)) - np.log(np.power(z1, 2))) / np.log(z2 / z1)

        return l

    def get_betweenness_centrality(self):
        G = self.trust_graph
        if nx.is_connected(G):
            return np.mean([c for _, c in dict(nx.algorithms.centrality.betweenness_centrality(G)).items()])
        else:
            return 0

    def get_current_flow_betweenness_centrality(self):
        G = self.trust_graph
        if nx.is_connected(G):
            return np.mean([c for _, c in dict(nx.algorithms.centrality.current_flow_betweenness_centrality(G)).items()])
        else:
            return 0
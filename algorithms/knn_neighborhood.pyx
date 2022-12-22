cimport numpy as np
from numpy cimport ndarray
import numpy as np
import heapq
from collections import defaultdict
from six import iteritems
from sklearn.neighbors import KernelDensity
import sys
from scipy.special import softmax
from scipy.stats import rankdata
from embeddings.embeddings import Embeddings
from scipy.spatial.distance import cdist

class PredictionImpossible(Exception):
    pass

class UserKNN:
    def __init__(self, k=40, min_k=1, random=False, reuse=False, tau_2=0, tau_4=0, tau_6=0, precomputed_sim=None,
                 precomputed_pop=None, precomputed_gain=None, precomputed_overlap=None, precomputed_gainplus=None, threshold=0, rated_items=None, protected=False,
                 user_embedding=None, item_embedding=None):
        self.k = k
        self.min_k = min_k
        self.mentors = defaultdict(set)
        self.n_mentors_at_q = defaultdict(list)
        self.item_coverage_at_q = defaultdict(list)
        self.accuracy_at_q = defaultdict(list)
        self.rating_overlap_at_q = defaultdict(list)
        self.reuse_neighbors = reuse
        self.tau_2 = tau_2
        self.tau_4 = tau_4
        self.tau_6 = tau_6
        self.user_sim = precomputed_sim if precomputed_sim is not None else None
        self.pop = precomputed_pop if precomputed_pop is not None else None
        self.gain = precomputed_gain if precomputed_gain is not None else None
        self.overlap = precomputed_overlap if precomputed_overlap is not None else None
        self.gainplus = precomputed_gainplus if precomputed_gainplus is not None else None
        self.random_neighbors = random
        self.privacy_score = None
        self.threshold = threshold
        self.protected = protected
        self.nr_noisy_ratings = []
        self.avg_sims = []
        self.rated_items = None
        self.user_embedding = user_embedding
        self.item_embedding = item_embedding

    def fit(self, trainset):
        self.trainset = trainset
        if self.rated_items is None:
            self.rated_items = self.compute_rated_items(self.trainset)

        self.n_ratings = np.zeros(self.trainset.n_users)
        for uid, ratings in self.trainset.ur.items():
            self.n_ratings[uid] = len(ratings)

        if self.overlap is None:
            self.overlap = self.compute_overlap(self.trainset)

        if self.user_embedding is not None:
            """user_embedding_mapped = np.zeros_like(self.user_embedding.embeddings)
            for inner_uid in self.trainset.all_users():
                raw_uid = self.trainset.to_raw_uid(inner_uid)
                index = self.user_embedding.item2index[raw_uid]
                user_embedding_mapped[inner_uid] = self.user_embedding.embeddings[index]
            self.user_embedding = user_embedding_mapped
            #self.user_sim = self.cosine(self.user_embedding)"""
            self.user_sim = 1 - UserKNN.cosine_distance(self.user_embedding)
        elif self.user_sim is None:
            self.user_sim = self.compute_similarities(self.trainset, self.min_k)

        if self.pop is None and self.tau_2 > 0:
            self.pop = self.compute_popularities(self.trainset)


        if self.gain is None and self.tau_4 > 0:
            self.gain = self.compute_gain(self.trainset)

        if self.gainplus is None and self.tau_6 > 0:
            if self.item_embedding:
                """item_embedding_mapped = np.zeros_like(self.item_embedding.embeddings)
                for inner_iid in self.trainset.all_items():
                    raw_iid = self.trainset.to_raw_iid(inner_iid)
                    index = self.item_embedding.item2index[raw_iid]
                    item_embedding_mapped[inner_iid] = self.item_embedding.embeddings[index]
                self.item_embedding = item_embedding_mapped"""
                self.top_item_neighbors = UserKNN.topk_item_neighbors(self.item_embedding, k=10)
            else:
                # generate item vectors based on ratings
                rating_matrix = np.zeros((self.trainset.n_users, self.trainset.n_items))
                for uid, iid, r in self.trainset.all_ratings():
                    rating_matrix[uid, iid] = r
                self.top_item_neighbors = UserKNN.topk_item_neighbors(rating_matrix.T, k=10)

            self.gainplus = UserKNN.compute_gainplus(self.trainset, self.top_item_neighbors)

        self.privacy_risk = np.zeros((self.trainset.n_users))
        self.privacy_risk_dp = np.zeros((self.trainset.n_users))

        # Tradeoff
        self.ranking = np.zeros((self.trainset.n_users, self.trainset.n_users))
        for u in self.trainset.all_users():
            """if self.user_sim is not None:
                simrank = rankdata(self.user_sim[u, :], method="max")
            if self.pop is not None:
                poprank = rankdata(self.pop, method="max")
            if self.gain is not None:
                gainrank = rankdata(self.gain[u, :], method="max")
            if self.gainplus is not None:
                gainplusrank = rankdata(self.gainplus[u, :], method="max")

            if self.pop is not None:
                 self.ranking[u] += self.tau_2 * poprank
            if self.gain is not None:
                 self.ranking[u] += self.tau_4 * gainrank
            if self.gainplus is not None:
                self.ranking[u] += self.tau_6 * gainplusrank
            if self.user_sim is not None:
                 self.ranking[u] += (1.0 - self.tau_2 - self.tau_4 - self.tau_6) * simrank"""

            simrank = rankdata(self.user_sim[u, :], method="max")
            self.ranking[u] = (1 - self.tau_2 - self.tau_4 - self.tau_6) * simrank
            if self.tau_2 > 0:
                poprank = rankdata(self.pop, method="max")
                self.ranking[u] += self.tau_2 * poprank
            if self.tau_4 > 0:
                gainrank = rankdata(self.gain[u, :], method="max")
                self.ranking[u] += self.tau_4 * gainrank
            if self.tau_6 > 0:
                gainplusrank = rankdata(self.gainplus[u, :], method="max")
                self.ranking[u] += self.tau_6 * gainplusrank

        return self

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        def deniable_answer(model, u, i):
            coin_1 = np.random.uniform() >= 0.5
            coin_2 = np.random.uniform() >= 0.5
            for iid, r in model.trainset.ur[u]:
                if iid == i:
                    r_true = r

            if coin_1:
                # answer truthfully
                return r_true
            else:
                if coin_2:
                    # answer with real rating
                    return r_true
                else:
                    # answer with random rating
                    min_rating, max_rating = model.trainset.rating_scale
                    r_random = np.random.uniform(min_rating, max_rating)
                    return r_random

        modified_ir = self.trainset.ir[i]
        possible_mentors = set(u_ for u_, _ in modified_ir)

        ranks = self.ranking[u]
        possible_mentors_data = [(u_, self.user_sim[u, u_], ranks[u_], r) for u_, r in modified_ir if u_ != u]
        np.random.shuffle(possible_mentors_data)
        possible_mentors_data = sorted(possible_mentors_data, key=lambda t: t[2])[::-1]


        if self.random_neighbors:
            mentors = np.random.choice(list(possible_mentors), replace=False, size=min(self.k, len(possible_mentors)))
            self.mentors[u] = self.mentors[u].union(set(mentors))
            neighbors = [(s, rank, r, u_) for u_, s, rank, r in possible_mentors_data if u_ in mentors]
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
            new_mentors = set(new_mentors)

            mentors = new_mentors.union(already_mentors)
            neighbors = [(s, rank, r, u_) for u_, s, rank, r in possible_mentors_data if u_ in mentors]
            k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        else:
            k_neighbors = heapq.nlargest(self.k, possible_mentors_data, key=lambda t: t[2])
            self.mentors[u] = self.mentors[u].union(set(u_ for u_, _, _, _  in k_neighbors))
            k_neighbors = [(s, rank, r, u_) for u_, s, rank, r in k_neighbors]

        n_mentors = len(self.mentors[u])
        self.n_mentors_at_q[u].append(n_mentors)

        neighborhood = list(self.mentors[u])
        avg_overlap = np.mean(self.overlap[u, neighborhood])
        self.rating_overlap_at_q[u].append(avg_overlap)

        # UserKNN
        sum_sim = sum_ratings = actual_k = 0.0
        sum_rank = 0.0
        est = 0
        pr_unprotected = []
        noisy = 0
        avg_sim = np.mean([sim for sim, _, _, _ in k_neighbors])
        for (sim, rank, r, u_) in k_neighbors:
            #if sim <= 0:
            #    continue

            self.privacy_risk[u_] += 1

            response = r
            if self.protected and self.privacy_risk[u_] > self.threshold:
                response = deniable_answer(self, u_, i)
            else:
                pr_unprotected.append(self.privacy_risk_dp[u_])
                self.privacy_risk_dp[u_] += 1

            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * response
                actual_k += 1
                sum_rank += rank
        if actual_k < self.min_k:
            raise PredictionImpossible('Not enough neighbors.')

        est += sum_ratings / sum_sim

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
        self.predictions = []
        self.absolute_errors = defaultdict(list)
        for user_id, item_id, rating in testset:
            uid, iid, r, r_, details = self.predict(user_id, item_id,  rating, clip=False)
            self.predictions.append((uid, iid, r, r_, details))
            try:
                iuid = self.trainset.to_inner_uid(uid)
            except ValueError:
                iuid = 'UKN__' + str(uid)
            self.absolute_errors[iuid].append(np.abs(r - r_))

        return self.predictions

    def default_prediction(self):
        return self.trainset.global_mean

    @staticmethod
    def compute_rated_items(trainset):
        rated_items = defaultdict(set)
        for uid, ratings in trainset.ur.items():
            rated_items[uid].update([iid for iid, _ in ratings])
        return rated_items

    @staticmethod
    def compute_overlap(trainset):
        overlap = np.zeros((trainset.n_users, trainset.n_users))
        for _, ratings in trainset.ir.items():
            for u1, _ in ratings:
                for u2, _ in ratings:
                    overlap[u1, u2] += 1

        return overlap


    @staticmethod
    def _cosine(trainset, min_support):
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
    def _pearson(trainset, min_support):
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

        avg_user_ratings = np.zeros(trainset.n_users)
        for uid, ratings in trainset.ur.items():
            avg_user_ratings[uid] = np.mean([r for _, r in ratings])

        prods = np.zeros((n_users, n_users), np.double)
        freq = np.zeros((n_users, n_users), np.int)
        sqi = np.zeros((n_users, n_users), np.double)
        sqj = np.zeros((n_users, n_users), np.double)
        sim = np.zeros((n_users, n_users), np.double)

        for y, y_ratings in ir.items():
            for xi, ri in y_ratings:
                for xj, rj in y_ratings:
                    freq[xi, xj] += 1
                    prods[xi, xj] += (ri - avg_user_ratings[xi]) * (rj - avg_user_ratings[xj])
                    sqi[xi, xj] += (ri - avg_user_ratings[xi])**2
                    sqj[xi, xj] += (rj - avg_user_ratings[xj])**2

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
    def _adjusted_cosine(trainset, min_support):
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

        avg_item_ratings = np.zeros(trainset.n_items)
        for iid, ratings in trainset.ir.items():
            avg_item_ratings[iid] = np.mean([r for _, r in ratings])

        prods = np.zeros((n_users, n_users), np.double)
        freq = np.zeros((n_users, n_users), np.int)
        sqi = np.zeros((n_users, n_users), np.double)
        sqj = np.zeros((n_users, n_users), np.double)
        sim = np.zeros((n_users, n_users), np.double)

        for y, y_ratings in ir.items():
            for xi, ri in y_ratings:
                for xj, rj in y_ratings:
                    freq[xi, xj] += 1
                    prods[xi, xj] += (ri - avg_item_ratings[xi]) * (rj - avg_item_ratings[xj])
                    sqi[xi, xj] += (ri - avg_item_ratings[xi])**2
                    sqj[xi, xj] += (rj - avg_item_ratings[xj])**2

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
    def _jaccard(trainset, min_support):
        n_users = trainset.n_users
        ir = trainset.ir

        # |r_x and r_y|
        cdef np.ndarray[np.double_t, ndim=2] overlap_size
        # |r_x or r_y|
        cdef np.ndarray[np.double_t, ndim=1] profile_size
        # the similarity matrix
        cdef np.ndarray[np.double_t, ndim=2] sim

        cdef int xi, xj
        cdef int min_sprt = min_support

        overlap_size = np.zeros((n_users, n_users), np.double)
        profile_size = np.zeros(n_users, np.double)
        sim = np.zeros((n_users, n_users), np.double)

        for _, y_ratings in ir.items():
            for xi, _ in y_ratings:
                profile_size[xi] += 1
                for xj, _ in y_ratings:
                    overlap_size[xi, xj] += 1

        for xi in range(n_users):
            sim[xi, xi] = 1
            for xj in range(xi + 1, n_users):
                if overlap_size[xi, xj] < min_sprt:
                    sim[xi, xj] = 0
                else:
                    sim[xi, xj] = overlap_size[xi, xj] / (profile_size[xi] + profile_size[xj])

                sim[xj, xi] = sim[xi, xj]

        return sim

    @staticmethod
    def compute_similarities(trainset, min_support, kind="cosine"):
        if kind == "cosine":
            sim = UserKNN._cosine(trainset, min_support)
        elif kind == "adjusted_cosine":
            sim = UserKNN._adjusted_cosine(trainset, min_support)
        elif kind == "pearson":
            sim = UserKNN._pearson(trainset, min_support)
        elif kind == "jaccard":
            sim = UserKNN._jaccard(trainset, min_support)
        else:
            sim = None

        return sim

    @staticmethod
    def compute_popularities(trainset):
        item_popularities = np.zeros(trainset.n_items)
        for i, ratings in trainset.ir.items():
            item_popularities[i] = float(len(ratings)) / trainset.n_users

        reuse_potential = np.zeros(trainset.n_users)
        for u, ratings in trainset.ur.items():
            acc_rp = 0.0
            for i, _ in ratings:
                acc_rp += item_popularities[i]
            #reuse_potential[u] = acc_rp / (trainset.n_ratings / trainset.n_users)
            reuse_potential[u] = acc_rp
        return reuse_potential


    @staticmethod
    def compute_gain(trainset):
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
    def compute_gainplus(trainset, top_neighbors):
        knowledge = defaultdict(set)
        for uid, ratings in trainset.ur.items():
            knowledge[uid] = knowledge[uid].union([iid for iid, _ in ratings])

        gainplus = np.zeros((trainset.n_users, trainset.n_users))
        for mentor in trainset.all_users():
            for student in trainset.all_users():
                g = 0.0
                n_queries = len(knowledge[student])
                if n_queries > 0:
                    #g = float(len(set(knowledge[mentor]).intersection(knowledge[student]))) / len(knowledge[student])
                    iids_no_match = list(set(knowledge[student]).difference(knowledge[mentor]))
                    g += n_queries - len(iids_no_match)

                    for iid in iids_no_match:
                        for neighbor, sim in top_neighbors[iid]:
                            if neighbor in knowledge[mentor]:
                                g += sim
                                break

                g /= len(knowledge[student])
                gainplus[student, mentor] = g

        return gainplus

    def get_privacy_threshold(self):
        scores = self.privacy_risk
        counts, edges = np.histogram(scores, bins=25)
        kde = KernelDensity(bandwidth=50.0, kernel='gaussian')
        kde.fit(counts.reshape(-1, 1))
        logprob = kde.score_samples(np.linspace(0, np.max(edges), 1000).reshape(-1, 1))
        threshold = np.linspace(0, np.max(edges), 1000)[np.argmax(np.gradient(np.gradient(np.exp(logprob))))]

        return threshold

    def protected_neighbors(self):
        protected_neighbors = set()
        for uid, q in enumerate(self.privacy_risk_dp):
            if self.protected and q >= self.threshold:
                protected_neighbors.add(uid)
        return protected_neighbors

    @staticmethod
    def cosine_distance(A, B=None):
        if B is None:
            return cdist(A, A, "cosine")
        else:
            return cdist(A, B, "cosine")

    @staticmethod
    def topk_item_neighbors(item_vectors, k=10):
        sim = defaultdict(list)
        for item, vector_i in enumerate(item_vectors):
            sim_i = cdist([vector_i], item_vectors)
            sim_i[0, item] = 0
            random_vector = np.random.randn(len(sim_i[0]))
            topn_items = np.lexsort((random_vector, sim_i[0]))[-k:]
            for neighbor in topn_items:
                sim[item].append((neighbor, sim_i[0, neighbor]))
            sim[item] = sorted(sim[item], key=lambda t: t[1], reverse=True)
        return sim
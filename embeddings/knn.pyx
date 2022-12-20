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

class PredictionImpossible(Exception):
    pass

class UserKNNEmbedding:
    def __init__(self, user_embedding, item_embedding, k=40, min_k=1, reuse=False, tau_2=0, tau_4=0,
                 expect_scores=None, gain_scores=None, overlap=None, threshold=0, rated_items=None, protected=False):
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
        self.pop = expect_scores if expect_scores is not None else None
        self.gain = gain_scores if gain_scores is not None else None
        self.overlap = overlap if overlap is not None else None
        self.privacy_score = None
        self.threshold = threshold
        self.protected = protected
        self.nr_noisy_ratings = []
        self.avg_sims = []
        self.rated_items = rated_items

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

        item_embedding_mapped = np.zeros_like(self.item_embedding.embeddings)
        for inner_iid in self.trainset.all_items():
            raw_iid = self.trainset.to_raw_iid(inner_iid)
            index = self.item_embedding.item2index[raw_iid]
            item_embedding_mapped[inner_iid] = self.item_embedding.embeddings[index]
        self.item_embedding = item_embedding_mapped
        self.item_sim = self.cosine(self.item_embedding)

        user_embedding_mapped = np.zeros_like(self.user_embedding.embeddings)
        for inner_uid in self.trainset.all_users():
            raw_uid = self.trainset.to_raw_uid(inner_uid)
            index = self.user_embedding.item2index[raw_uid]
            user_embedding_mapped[inner_uid] = self.user_embedding.embeddings[index]
        self.user_embedding = user_embedding_mapped
        self.user_sim = self.cosine(self.user_embedding)

        if self.pop is None and self.tau_2 > 0:
            pass

        if self.gain is None and self.tau_4 > 0:
            pass

        self.privacy_risk = np.zeros((self.trainset.n_users))
        self.privacy_risk_dp = np.zeros((self.trainset.n_users))

        # Tradeoff
        self.ranking = np.zeros((self.trainset.n_users, self.trainset.n_users))
        for u in self.trainset.all_users():
            if self.user_sim is not None:
                simrank = rankdata(self.user_sim[u, :], method="max")
                self.ranking[u] += (1.0 - self.tau_2 - self.tau_4) * simrank
            if self.pop is not None:
                poprank = rankdata(self.pop, method="max")
                self.ranking[u] += self.tau_2 * poprank
            if self.gain is not None:
                gainrank = rankdata(self.gain[u, :], method="max")
                self.ranking[u] += self.tau_4 * gainrank

            """if self.pop is not None:
                 self.ranking[u] += self.tau_2 * poprank
            if self.gain is not None:
                 self.ranking[u] += self.tau_4 * gainrank
            if self.user_sim is not None:
                 self.ranking[u] += (1.0 - self.tau_2 - self.tau_4) * simrank"""

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
        for (sim, rank, r, u_) in k_neighbors:
            if sim <= 0:
                continue

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

                gain[student, mentor] = g
        return gain

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
    def cosine(X):
        norms = np.linalg.norm(X, axis=1)
        sim = np.dot(X, X.transpose())
        sim /= norms
        sim /= norms[:, np.newaxis]
        sim[np.isnan(sim)] = 0
        return sim
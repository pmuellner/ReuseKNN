cimport numpy as np
import numpy as np
import heapq
from collections import defaultdict
from sklearn.neighbors import KernelDensity
from scipy.stats import rankdata

class PredictionImpossible(Exception):
    pass

class UserKNN:
    """
    Class for user-based KNN-based recommendations, i.e., UserKNN, ReuseKNN. In detail, UserKNN, UserKNN+Reuse,
    Expect, Gain. Also, DP is implemented here.
    (large parts from https://github.com/NicolasHug/Surprise/blob/master/surprise/prediction_algorithms/knns.py)
    """
    def __init__(self, k=40, min_k=1, reuse=False, sim=None, expect_scores=None, gain_scores=None, overlap=None,
                 threshold=0, rated_items=None, use_dp=False):
        self.k = k
        self.min_k = min_k
        self.neighbors = defaultdict(set)
        self.n_neighbors_at_q = defaultdict(list)
        self.rating_overlap_at_q = defaultdict(list)
        self.reuse = reuse
        self.user_sim = sim if sim is not None else None
        self.expect = expect_scores if expect_scores is not None else None
        self.gain = gain_scores if gain_scores is not None else None
        self.overlap = overlap if overlap is not None else None
        self.threshold = threshold
        self.use_dp = use_dp
        self.rated_items = None
        self.trainset = None
        self.n_ratings = None
        self.data_usage = None
        self.privacy_risk = None
        self.ranking = None
        self.predictions = []
        self.absolute_errors = []

    def fit(self, trainset):
        """
        Prepares everything for the recommendation generation stage. E.g., calculates the trade-off between similarity
        and reusability for all users
        """

        self.trainset = trainset
        if self.rated_items is None:
            self.rated_items = self.compute_rated_items(self.trainset)

        self.n_ratings = np.zeros(self.trainset.n_users)
        for uid, ratings in self.trainset.ur.items():
            self.n_ratings[uid] = len(ratings)

        if self.overlap is None:
            self.overlap = self.compute_overlap(self.trainset)

        self.data_usage = np.zeros(self.trainset.n_users)
        self.privacy_risk = np.zeros(self.trainset.n_users)

        # Tradeoff
        self.ranking = np.zeros((self.trainset.n_users, self.trainset.n_users))
        for u in self.trainset.all_users():
            simrank = rankdata(self.user_sim[u, :], method="max")
            self.ranking[u] = 0.5 * simrank

            if self.expect is not None:
                expectrank = rankdata(self.expect, method="max")
                self.ranking[u] += 0.5 * expectrank
            elif self.gain is not None:
                gainrank = rankdata(self.gain[u, :], method="max")
                self.ranking[u] += 0.5 * gainrank
            else:
                self.ranking[u] += 0.5 * simrank

        return self

    def estimate(self, u, i):
        """
        Selects neighbors based on the trade-off between similarity and reusability to generate estimated rating scores.
        Applies DP via the plausible deniability mechanism. Monitors data usage and privacy risk.
        """
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        def deniable_answer(model, u, i):
            """
            The DP mechanism.
            """
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

        ir = self.trainset.ir[i]
        candidate_neighbors = set(u_ for u_, _ in ir)

        ranks = self.ranking[u]
        candidate_neighbors_data = [(u_, self.user_sim[u, u_], ranks[u_], r) for u_, r in ir if u_ != u]
        np.random.shuffle(candidate_neighbors_data)
        candidate_neighbors_data = sorted(candidate_neighbors_data, key=lambda t: t[2])[::-1]

        # only used for UserKNN+Reuse
        if self.reuse:
            already_neighbors = self.neighbors[u].intersection(candidate_neighbors)
            n_new_neighbors = self.k - len(already_neighbors) if self.k > len(already_neighbors) else 0
            new_neighbors = []
            for u_, _, _, _ in candidate_neighbors_data:
                if len(new_neighbors) >= n_new_neighbors:
                    break
                elif u_ not in already_neighbors:
                    new_neighbors.append(u_)
                    self.neighbors[u] = self.neighbors[u].union({u_})
            new_neighbors = set(new_neighbors)

            neighbors = new_neighbors.union(already_neighbors)
            neighbors = [(s, rank, r, u_) for u_, s, rank, r in candidate_neighbors_data if u_ in neighbors]
            k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        else:
            k_neighbors = heapq.nlargest(self.k, candidate_neighbors_data, key=lambda t: t[2])
            self.neighbors[u] = self.neighbors[u].union(set(u_ for u_, _, _, _  in k_neighbors))
            k_neighbors = [(s, rank, r, u_) for u_, s, rank, r in k_neighbors]

        n_mentors = len(self.neighbors[u])
        self.n_neighbors_at_q[u].append(n_mentors)

        neighborhood = list(self.neighbors[u])
        avg_overlap = np.mean(self.overlap[u, neighborhood])
        self.rating_overlap_at_q[u].append(avg_overlap)

        # estimation of the rating score
        sum_sim = sum_ratings = actual_k = 0.0
        est = 0
        for (sim, rank, r, u_) in k_neighbors:
            if sim <= 0:
                continue

            self.data_usage[u_] += 1

            response = r
            if self.use_dp and self.data_usage[u_] > self.threshold:
                response = deniable_answer(self, u_, i)
            else:
                self.privacy_risk[u_] += 1

            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * response
                actual_k += 1
        if actual_k < self.min_k:
            raise PredictionImpossible('Not enough neighbors.')

        est += sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details

    def predict(self, uid, iid, r, clip=True):
        """
        Wrapper for estimate. Predicts the rating between user uid and item iid
        """
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
        """
        predicts rating scores between all user and item pairs in testset
        """
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
        """
        Copied from https://github.com/NicolasHug/Surprise/blob/master/surprise/similarities.pyx
        """
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
    def compute_similarities(trainset, min_support, kind="cosine"):
        if kind == "cosine":
            sim = UserKNN._cosine(trainset, min_support)
        else:
            sim = None
        return sim

    @staticmethod
    def compute_expect_scores(trainset):
        """
        Scores neighbors according to the Expect reusability strategy
        """
        item_popularities = np.zeros(trainset.n_items)
        for i, ratings in trainset.ir.items():
            item_popularities[i] = float(len(ratings)) / trainset.n_users

        expect_scores = np.zeros(trainset.n_users)
        for u, ratings in trainset.ur.items():
            expect_scores[u] =  np.sum([item_popularities[i] for i, _ in ratings])
        return expect_scores


    @staticmethod
    def compute_gain_scores(trainset):
        """
        Scores neighbors according to the Gain reusability strategy
        """
        items = defaultdict(list)
        for uid, ratings in trainset.ur.items():
            items[uid].extend([iid for iid, _ in ratings])

        gain = np.zeros((trainset.n_users, trainset.n_users))
        for candidate_neighbor in trainset.all_users():
            for target_user in trainset.all_users():
                n_queries = len(items[target_user])
                if n_queries > 0:
                    overlap = set(items[candidate_neighbor]).intersection(items[target_user])
                    g = float(len(overlap) / len(items[target_user]))
                else:
                    g = 0.0

                gain[target_user, candidate_neighbor] = g
        return gain

    def get_privacy_threshold(self):
        """
        Calculate the threshold tau that defines what users are regarded as vulnerable and need to be protected with DP
        """
        scores = self.data_usage
        counts, edges = np.histogram(scores, bins=25)
        kde = KernelDensity(bandwidth=50.0, kernel='gaussian')
        kde.fit(counts.reshape(-1, 1))
        logprob = kde.score_samples(np.linspace(0, np.max(edges), 1000).reshape(-1, 1))
        threshold = np.linspace(0, np.max(edges), 1000)[np.argmax(np.gradient(np.gradient(np.exp(logprob))))]

        return threshold

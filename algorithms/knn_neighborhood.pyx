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

class PredictionImpossible(Exception):
    pass

class UserKNN:
    def __init__(self, k=40, min_k=1, random=False, reuse=False, tau_1=0, tau_2=0, tau_3=0, tau_4=0, tau_5=0, tau_6=0, precomputed_sim=None,
                 precomputed_pop=None, precomputed_act=None, precomputed_rr=None, precomputed_gain=None, precomputed_overlap=None, threshold=0, rated_items=None, protected=False):
        self.k = k
        self.min_k = min_k
        self.mentors = defaultdict(set)
        self.n_mentors_at_q = defaultdict(list)
        self.item_coverage_at_q = defaultdict(list)
        self.accuracy_at_q = defaultdict(list)
        self.rating_overlap_at_q = defaultdict(list)
        self.reuse_neighbors = reuse
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.tau_3 = tau_3
        self.tau_4 = tau_4
        self.tau_5 = tau_5
        self.tau_6 = tau_6
        self.sim = precomputed_sim if precomputed_sim is not None else None
        self.pop = precomputed_pop if precomputed_pop is not None else None
        self.act = precomputed_act if precomputed_act is not None else None
        self.rr = precomputed_rr if precomputed_rr is not None else None
        self.gain = precomputed_gain if precomputed_gain is not None else None
        self.overlap = precomputed_overlap if precomputed_overlap is not None else None
        self.random_neighbors = random
        self.privacy_score = None
        self.threshold = int(threshold)
        self.protected = protected
        self.nr_noisy_ratings = 0
        self.rated_items = None

    def fit(self, trainset):
        self.trainset = trainset
        if self.rated_items is None:
            self.rated_items = self.compute_rated_items(self.trainset)

        self.n_ratings = np.zeros(self.trainset.n_users)
        for uid, ratings in self.trainset.ur.items():
            self.n_ratings[uid] = len(ratings)

        if self.overlap is None:
            self.overlap = self.compute_overlap(self.trainset)

        if self.sim is None:
            self.sim = self.compute_similarities(self.trainset, self.min_k)

        if self.act is None and self.tau_1 > 0:
            self.act = self.compute_activities(self.trainset)

        if self.pop is None and self.tau_2 > 0:
            self.pop = self.compute_popularities(self.trainset)

        if self.rr is None and self.tau_3 > 0:
            self.rr = self.compute_rr(self.trainset)

        if self.gain is None and self.tau_4 > 0:
            self.gain = self.compute_gain(self.trainset)

        self.rr_expect = self.compute_rr_expect(self.trainset)
        self.rr_pop = self.compute_rr_pop(self.trainset)
        self.pop_new = self.compute_pop(self.trainset)

        #self.ranking = dict()
        self.privacy_risk = np.zeros((self.trainset.n_users))
        self.privacy_risk_dp = np.zeros((self.trainset.n_users))

        # Tradeoff
        self.ranking = np.zeros((self.trainset.n_users, self.trainset.n_users))
        for u in self.trainset.all_users():
            if self.sim is not None:
                simrank = rankdata(self.sim[u, :], method="max")
            if self.pop is not None:
                poprank = rankdata(self.pop, method="max")
            if self.act is not None:
                actrank = rankdata(self.act, method="max")
            if self.rr is not None:
                rrrank = rankdata(self.rr, method="max")
            if self.gain is not None:
                gainrank = rankdata(self.gain[u, :], method="max")
            if self.rr_expect is not None:
                rr_expect_rank = rankdata(self.rr_expect, method="max")
            if self.rr_pop is not None:
                rr_pop_rank = rankdata(self.rr_pop, method="max")
            if self.pop_new is not None:
                pop_new_rank = rankdata(self.pop_new, method="max")


            if self.act is not None:
                self.ranking[u] += self.tau_1 * actrank
            if self.pop is not None:
                 self.ranking[u] += self.tau_2 * poprank
            if self.rr is not None:
                 self.ranking[u] += self.tau_3 * rrrank
            if self.gain is not None:
                 self.ranking[u] += self.tau_4 * gainrank
            if self.sim is not None:
                 self.ranking[u] += (1.0 - self.tau_1 - self.tau_2 - self.tau_3 - self.tau_4) * simrank

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

        #possible_mentors_old = set(u_ for u_, _ in self.trainset.ir[i])
        modified_ir = self.trainset.ir[i]
        possible_mentors = set(u_ for u_, _ in modified_ir)

        ranks = self.ranking[u]
        possible_mentors_data = [(u_, self.sim[u, u_], ranks[u_], r) for u_, r in modified_ir if u_ != u]
        np.random.shuffle(possible_mentors_data)
        possible_mentors_data = sorted(possible_mentors_data, key=lambda t: t[2])[::-1]


        if self.random_neighbors:
            mentors = np.random.choice(list(possible_mentors), replace=False, size=min(self.k, len(possible_mentors)))
            self.mentors[u] = self.mentors[u].union(set(mentors))
            #for m in mentors:
            #    self.students[m] = self.students[m].union({u})
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
                    #self.students[u_] = self.students[u_].union({u})
            new_mentors = set(new_mentors)

            mentors = new_mentors.union(already_mentors)
            neighbors = [(s, rank, r, u_) for u_, s, rank, r in possible_mentors_data if u_ in mentors]
            k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        else:
            k_neighbors = heapq.nlargest(self.k, possible_mentors_data, key=lambda t: t[2])
            self.mentors[u] = self.mentors[u].union(set(u_ for u_, _, _, _  in k_neighbors))
            #for u_, _, _, _ in k_neighbors:
            #    self.students[u_] = self.students[u_].union({u})
            k_neighbors = [(s, rank, r, u_) for u_, s, rank, r in k_neighbors]

        n_mentors = len(self.mentors[u])
        self.n_mentors_at_q[u].append(n_mentors)

        neighborhood = list(self.mentors[u])
        items_in_neighborhood = set()
        for neighbor in sorted(zip(neighborhood, self.n_ratings[neighborhood]), key=lambda t: t[1], reverse=True):
            items_in_neighborhood.update(self.rated_items[neighbor])
            if len(items_in_neighborhood) >= self.trainset.n_items:
                break
        self.item_coverage_at_q[u].append(len(items_in_neighborhood))

        avg_overlap = np.mean(self.overlap[u, list(self.mentors[u])])
        self.rating_overlap_at_q[u].append(avg_overlap)

        # UserKNN
        sum_sim = sum_ratings = actual_k = 0.0
        sum_rank = 0.0
        est = 0
        pr_unprotected = []
        for (sim, rank, r, u_) in k_neighbors:
            self.privacy_risk[u_] += 1

            response = r
            if self.protected and self.privacy_risk[u_] > self.threshold:
                self.nr_noisy_ratings += 1
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

        #for _, _, r, u_ in k_neighbors:
        #    self.known_ratings[u].append((u_, i, r, est))

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
            #iuid = self.trainset.to_inner_uid(uid)
            #self.absolute_errors[iuid].append(np.abs(r - r_))

            try:
                iuid = self.trainset.to_inner_uid(uid)
            except ValueError:
                iuid = 'UKN__' + str(uid)
            self.absolute_errors[iuid].append(np.abs(r - r_))

        #self.absolute_errors = absolute_errors
        #self.mae_u = {uid: np.mean(aes) for uid, aes in absolute_errors.items()}
        #self.predictions = predictions

        #self.exposure_u = {iuid: 0 for iuid in self.trainset.all_users()}
        #for iuid, students in self.students.items():
        #    self.exposure_u[iuid] = len(students)

        #self.privacy_score = self._get_privacy_scores()
        self.nr_noisy_ratings /= len(testset)

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
        elif kind=="jaccard":
            sim = UserKNN._jaccard(trainset, min_support)
        else:
            sim = None

        return sim

    @staticmethod
    def compute_activities(trainset):
        n_ratings = np.zeros(trainset.n_users)
        for u, ratings in trainset.ur.items():
            n_ratings[u] = len(ratings)

        return n_ratings

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
    def compute_rr_expect(trainset):
        item_popularities = np.zeros(trainset.n_items)
        for i, ratings in trainset.ir.items():
            item_popularities[i] = float(len(ratings)) / trainset.n_users

        rr = np.zeros(trainset.n_users)
        for u, ratings in trainset.ur.items():
            acc_rp = 0.0
            for i, _ in ratings:
                acc_rp += item_popularities[i]
            rr[u] = 1 / acc_rp
        return rr

    @staticmethod
    def compute_rr_pop(trainset):
        item_popularities = np.zeros(trainset.n_items)
        for i, ratings in trainset.ir.items():
            item_popularities[i] = float(len(ratings)) / trainset.n_users

        rr = np.zeros(trainset.n_users)
        for uid, ratings in trainset.ur.items():
            rr[uid] = 1 / np.mean([item_popularities[iid] for iid, r in ratings])
        return rr

    @staticmethod
    def compute_pop(trainset):
        item_popularities = np.zeros(trainset.n_items)
        for i, ratings in trainset.ir.items():
            item_popularities[i] = float(len(ratings)) / trainset.n_users

        rr = np.zeros(trainset.n_users)
        for uid, ratings in trainset.ur.items():
            rr[uid] = np.mean([item_popularities[iid] for iid, r in ratings])
        return rr

    @staticmethod
    def compute_rr(trainset, function=None):
        item_popularities = np.zeros(trainset.n_items)
        for i, ratings in trainset.ir.items():
            item_popularities[i] = float(len(ratings)) / trainset.n_users

        """item_ranks = {v: k+1 for k, v in dict(enumerate(np.argsort(item_popularities)[::-1])).items()}

        rr = np.zeros(trainset.n_users)
        for u, ratings in trainset.ur.items():
            rr_u = 0.0
            for i, _ in ratings:
                rr_u += f(item_ranks[i])
            rr[u] = rr_u"""

        rr = np.zeros(trainset.n_users)
        if function == "expectation":
            for u, ratings in trainset.ur.items():
                acc_rp = 0.0
                for i, _ in ratings:
                    acc_rp += item_popularities[i]
                rr[u] = 1 / acc_rp
        if function == "popularity":
            rr = np.zeros(trainset.n_users)
            for uid, ratings in trainset.ur.items():
                rr[uid] = 1 / np.mean([item_popularities[iid] for iid, r in ratings])

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

    def  _get_privacy_scores(self):
        item_popularity = {iid: len(ratings) / self.trainset.n_users for iid, ratings in self.trainset.ir.items()}
        privacy_score = {iuid: 0 for iuid in self.trainset.all_users()}
        privacy_score_pairwise = np.zeros((self.trainset.n_users, self.trainset.n_users))
        for alice, secrets in self.known_secrets.items():
            sensitivity = dict()
            for bob, iid in secrets:
                sensitivity_i = np.log(1 / item_popularity[iid])
                sensitivity[bob] = sensitivity.get(bob, 0) + sensitivity_i

            for bob, s in sensitivity.items():
                privacy_score[bob] = privacy_score.get(bob, 0) + s
                privacy_score_pairwise[alice, bob] = s

        return privacy_score
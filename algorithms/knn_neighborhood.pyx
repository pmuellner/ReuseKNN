cimport numpy as np
from numpy cimport ndarray
import numpy as np
import heapq
from collections import defaultdict
from sklearn.neighbors import KernelDensity

class PredictionImpossible(Exception):
    pass

class UserKNN:
    def __init__(self, k=40, min_k=1, random=False, reuse=False, tau_1=0, tau_2=0, tau_3=0, tau_4=0, precomputed_sim=None,
                 precomputed_pop=None, precomputed_act=None, precomputed_rr=None, precomputed_gain=None, protected_neighbors=None):
        self.k = k
        self.min_k = min_k
        self.mentors = defaultdict(set)
        self.students = defaultdict(set)
        self.n_mentors_at_q = defaultdict(list)
        self.n_students_at_q = defaultdict(list)
        self.reuse_neighbors = reuse
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.tau_3 = tau_3
        self.tau_4 = tau_4
        self.sim = precomputed_sim.copy() if precomputed_sim is not None else None
        self.pop = precomputed_pop.copy() if precomputed_pop is not None else None
        self.act = precomputed_act.copy() if precomputed_act is not None else None
        self.rr = precomputed_rr.copy() if precomputed_rr is not None else None
        self.gain = precomputed_gain.copy() if precomputed_gain is not None else None
        self.random_neighbors = random
        self.privacy_score = None

        self.known_secrets = defaultdict(list)
        self.known_ratings = defaultdict(list)
        self.protected_neighbors = protected_neighbors

    def fit(self, trainset):
        self.trainset = trainset
        self.rated_items = defaultdict(list)

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

        if self.protected_neighbors is None:
            self.protected_neighbors = {u: set() for u in self.trainset.all_users()}

        self.ranking = dict()

        # Tradeoff
        for u in self.trainset.all_users():
            if self.sim is not None:
                simrank = {v: k for k, v in dict(enumerate(np.argsort(self.sim[u, :]))).items()}
            if self.pop is not None:
                poprank = {v: k for k, v in dict(enumerate(np.argsort(self.pop))).items()}
            if self.act is not None:
                actrank = {v: k for k, v in dict(enumerate(np.argsort(self.act))).items()}
            if self.rr is not None:
                rrrank = {v: k for k, v in dict(enumerate(np.argsort(self.rr))).items()}
            if self.gain is not None:
                gainrank = {v: k for k, v in dict(enumerate(np.argsort(self.gain[u, :]))).items()}

            ranking_u = dict()
            for u_ in self.trainset.all_users():
                if u_ != u:
                    ranking_u[u_] = 0
                    if self.act is not None:
                        ranking_u[u_] += self.tau_1 * actrank[u_]
                    if self.pop is not None:
                        ranking_u[u_] += self.tau_2 * poprank[u_]
                    if self.rr is not None:
                        ranking_u[u_] += self.tau_3 * rrrank[u_]
                    if self.gain is not None:
                        ranking_u[u_] += self.tau_4 * gainrank[u_]
                    if self.sim is not None:
                        ranking_u[u_] += (1.0 - self.tau_1 - self.tau_2 - self.tau_3 - self.tau_4) * simrank[u_]

            self.ranking[u] = ranking_u


        return self

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        def deniable_answer(model, u, i):
            if np.random.uniform() >= 0.5:
                # answer truthfully
                for iid, r in model.trainset.ur[u]:
                    if iid == i:
                        return r
                else:
                    return None
            else:
                # answer randomly
                if np.random.uniform() >= 0.5:
                    # generate random rating
                    min_rating, max_rating = model.trainset.rating_scale
                    return np.random.uniform(min_rating, max_rating)
                else:
                    # return no rating
                    return None

        possible_mentors_old = set(u_ for u_, _ in self.trainset.ir[i])
        modified_ir = []
        for u_, r in self.trainset.ir[i]:
            if u_ in possible_mentors_old.intersection(self.protected_neighbors):
                if np.random.uniform() <= 0.5:
                    # real rating
                    modified_ir.append((u_, r))
                else:
                    # fake rating
                    min_rating, max_rating = self.trainset.rating_scale
                    r_ =  np.random.uniform(min_rating, max_rating)
                    modified_ir.append((u_, r_))
            else:
                modified_ir.append((u_, r))


        #modified_ir = self.trainset.ir[i]
        possible_mentors = set(u_ for u_, _ in modified_ir)

        ranks = self.ranking[u]
        possible_mentors_data = [(u_, self.sim[u, u_], ranks[u_], r) for u_, r in modified_ir]
        possible_mentors_data = sorted(possible_mentors_data, key=lambda t: t[2])[::-1]


        if self.random_neighbors:
            mentors = np.random.choice(list(possible_mentors), replace=False, size=min(self.k, len(possible_mentors)))
            self.mentors[u] = self.mentors[u].union(set(mentors))
            for m in mentors:
                self.students[m] = self.students[m].union({u})
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
                    self.students[u_] = self.students[u_].union({u})
            new_mentors = set(new_mentors)

            mentors = new_mentors.union(already_mentors)
            neighbors = [(s, rank, r, u_) for u_, s, rank, r in possible_mentors_data if u_ in mentors]
            k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        else:
            k_neighbors = heapq.nlargest(self.k, possible_mentors_data, key=lambda t: t[2])
            self.mentors[u] = self.mentors[u].union(set(u_ for u_, _, _, _  in k_neighbors))
            for u_, _, _, _ in k_neighbors:
                self.students[u_] = self.students[u_].union({u})
            k_neighbors = [(s, rank, r, u_) for u_, s, rank, r in k_neighbors]

        n_mentors = len(self.mentors[u])
        self.n_mentors_at_q[u].append(n_mentors)

        for _, _, _, u_ in k_neighbors:
            n_students = len(self.students[u_])
            self.n_students_at_q[u_].append(n_students)
            self.known_secrets[u].append((u_, i))

        # UserKNN
        sum_sim = sum_ratings = actual_k = 0.0
        sum_rank = 0.0
        for (sim, rank, r, u_) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * r
                actual_k += 1
                sum_rank += rank

        if actual_k < self.min_k:
            raise PredictionImpossible('Not enough neighbors.')

        est = sum_ratings / sum_sim

        for _, _, r, u_ in k_neighbors:
            self.known_ratings[u].append((u_, i, r, est))

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
            iuid = self.trainset.to_inner_uid(uid)
            absolute_errors[iuid].append(np.abs(r - r))

        self.mae_u = {iuid: np.mean(aes) for iuid, aes in absolute_errors.items()}

        self.exposure_u = {iuid: 0 for iuid in self.trainset.all_users()}
        pop = self.compute_popularities(self.trainset)
        for iuid, students in self.students.items():
            self.exposure_u[iuid] = len(students)

        self.privacy_score = self._get_privacy_scores()

        return predictions

    def default_prediction(self):
        return self.trainset.global_mean

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
            reuse_potential[u] = acc_rp / (trainset.n_ratings / trainset.n_users)
        return reuse_potential

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

    def get_privacy_threshold(self):
        """if self.privacy_score is None:
            self.privacy_score = self._get_privacy_scores()

        scores = np.array(list(self.privacy_score.values()))

        counts, edges = np.histogram(scores, bins=25)
        kde = KernelDensity(bandwidth=100.0, kernel='gaussian')
        kde.fit(counts.reshape(-1, 1))
        logprob = kde.score_samples(np.linspace(0, np.max(edges), 1000).reshape(-1, 1))
        threshold = np.linspace(0, np.max(edges), 1000)[np.argmax(np.gradient(np.gradient(np.exp(logprob))))]"""

        scores = np.array(list(self.exposure_u.values()))
        counts, edges = np.histogram(scores, bins=25)
        kde = KernelDensity(bandwidth=30.0, kernel='gaussian')
        kde.fit(counts.reshape(-1, 1))
        logprob = kde.score_samples(np.linspace(0, np.max(edges), 1000).reshape(-1, 1))
        threshold = np.linspace(0, np.max(edges), 1000)[np.argmax(np.gradient(np.gradient(np.exp(logprob))))]

        return threshold

    def _get_privacy_scores(self):
        item_popularity = {iid: len(ratings) / self.trainset.n_users for iid, ratings in self.trainset.ir.items()}
        privacy_score = dict()
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

    @staticmethod
    def identify_vulnerable_users(model, threshold):
        vulnerable_users = [uid for uid, score in model.privacy_score.items() if score > threshold]
        return vulnerable_users
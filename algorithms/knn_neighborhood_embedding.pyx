cimport numpy as np
from numpy cimport ndarray
import numpy as np
import heapq
from collections import defaultdict
from sklearn.neighbors import KernelDensity
from scipy.stats import rankdata
from sklearn.metrics.pairwise import cosine_similarity

class PredictionImpossible(Exception):
    pass

class UserKNN:
    def __init__(self, k=40, min_k=1, explicit_reuse_option=None, precomputed_sim=None, threshold=0, protected=False, user_embeddings=None, item_embeddings=None):
        self.k = k
        self.min_k = min_k
        self.mentors = defaultdict(set)
        self.explicit_reuse_option = explicit_reuse_option
        self.user_sim = precomputed_sim
        self.threshold = threshold
        self.protected = protected
        self.previous_query_neighborhoods = defaultdict(dict)
        self.trainset = None
        self.privacy_risk = 0
        self.privacy_risk_dp = 0
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings

    def fit(self, trainset):
        self.trainset = trainset

        if self.user_sim is None:
            self.user_sim = UserKNN.compute_similarity(self.user_embeddings)

        self.privacy_risk = np.zeros(self.trainset.n_users)
        self.privacy_risk_dp = np.zeros(self.trainset.n_users)

        self.ranking = np.zeros((self.trainset.n_users, self.trainset.n_users))
        for u in self.trainset.all_users():
            self.ranking[u] = rankdata(self.user_sim[u, :], method="max")

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

        if self.explicit_reuse_option is not None:
            if str(self.explicit_reuse_option).lower() == "static":
                np.random.shuffle(possible_mentors_data)
                possible_mentors_data = sorted(possible_mentors_data, key=lambda t: t[2])[::-1]
                already_mentors = self.mentors[u].intersection(possible_mentors)
                n_new_mentors = self.k - len(already_mentors) if self.k > len(already_mentors) else 0
                new_mentors = []
                for u_, _, _, _ in possible_mentors_data:
                    if len(new_mentors) >= n_new_mentors:
                        break
                    elif u_ not in already_mentors:
                        new_mentors.append(u_)
                        self.mentors[u] = self.mentors[u].union({u_})

                mentors = set(new_mentors).union(already_mentors)
                new_possible_mentors_data = [record for record in possible_mentors_data if u_ in mentors]
                possible_mentors_data = new_possible_mentors_data
            elif str(self.explicit_reuse_option).lower() == "dynamic":
                previous_items = self.previous_query_neighborhoods[u].keys()
                neighbor_counts = np.zeros(self.trainset.n_users)
                for item in previous_items:
                    for n in self.previous_query_neighborhoods[u][item]:
                        e1 = self.item_embeddings[i]
                        e2 = self.item_embeddings[item]
                        sim = e1.dot(e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
                        neighbor_counts[n] += sim if sim > 0 else 0

                reuse_ranks = rankdata(neighbor_counts, method="max")
                reranked = []
                for u_, sim, sim_rank, r in possible_mentors_data:
                    reuse_rank = reuse_ranks[u_]
                    reranked.append((u_, sim, sim_rank + reuse_rank, r))

                possible_mentors_data = reranked
            else:
                print("Invalid value for explicit_reuse_option: %s" % self.explicit_reuse_option)

        k_neighbors = heapq.nlargest(self.k, possible_mentors_data, key=lambda t: t[2])
        self.previous_query_neighborhoods[u][i] = set(u_ for u_, _, _, _  in k_neighbors)
        self.mentors[u] = self.mentors[u].union(set(u_ for u_, _, _, _  in k_neighbors))

        # UserKNN
        sum_sim = sum_ratings = actual_k = 0.0
        est = 0
        for (u_, sim, _, r) in k_neighbors:
            if sim <= 0:
                continue

            self.privacy_risk[u_] += 1

            response = r
            if self.protected and self.privacy_risk[u_] > self.threshold:
                response = deniable_answer(self, u_, i)
            else:
                self.privacy_risk_dp[u_] += 1

            sum_sim += sim
            sum_ratings += sim * response
            actual_k += 1

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

    def get_privacy_threshold(self):
        scores = self.privacy_risk
        counts, edges = np.histogram(scores, bins=25)
        kde = KernelDensity(bandwidth=50.0, kernel='gaussian')
        kde.fit(counts.reshape(-1, 1))
        logprob = kde.score_samples(np.linspace(0, np.max(edges), 1000).reshape(-1, 1))
        threshold = np.linspace(0, np.max(edges), 1000)[np.argmax(np.gradient(np.gradient(np.exp(logprob))))]

        return threshold

    @staticmethod
    def compute_similarity(array):
        sim = cosine_similarity(array, array)
        return sim
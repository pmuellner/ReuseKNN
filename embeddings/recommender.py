# -*- coding: utf-8 -*-

# Copyright 2019 Information Retrieval Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import heapq
from multiprocessing import Pool

import numpy as np


class WSRRec(object):
    """
    Weighted Sum Recommender.
    """

    def __init__(self, item_based, items, embeddings, ratings, users=None):
        """
        - item_based: if the recommender is item or used based. Should be
                      of the same type as the embeddings.
        - items: set of items to consider when looking for unrated
                 items
        - embeddings: embeddings for a set of items. Of type
                      embeddings.Embeddings
        - ratings_matrix: ratings matrix. Of type ratings.RatingsMatrix
        - users: user set to calculate the rakings. If None is given
                 it will calculate the rankings for all the users in
                 the ratings matrix.
        """
        self.item_based = item_based
        self.embeddings = embeddings
        # Only rank items of which we have embeddings.
        self.items = items & embeddings.item_set()
        self.ratings_matrix = ratings
        if users is None:
            self.users = ratings.users_set()
        else:
            self.users = users

    def calculate_user_ranking(self, user, k, max_items=None):
        """
        Calculates the scores of unrated items for a given user,
        returning them in a list of max_items ordered in descending
        order of score of pairs (item, score). If max_items is None
        it return a list with a ranking of all unrated items.
        """
        scores = []

        if self.item_based:
            ratings = self.ratings_matrix.user_ratings(user)
        else:
            neighbors, cosines = self.embeddings.nearest_neighbors(k, user)

        rated_items = self.ratings_matrix.items_rated_by_user(user)
        unrated_items = self.items - rated_items
        for item in unrated_items:
            if self.item_based:
                neighbors, cosines = self.embeddings.nearest_neighbors(k, item)
            else:
                ratings = self.ratings_matrix.item_ratings(item)
            neighbors_ratings = np.zeros(len(neighbors))
            for i, neighbor in enumerate(neighbors):
                if neighbor in ratings:
                    neighbors_ratings[i] = ratings[neighbor]
            score = np.dot(neighbors_ratings, cosines)
            scores.append((item, score))

        if max_items is None:
            scores.sort(key=lambda x: x[1], reverse=True)
            return user, scores
        else:
            return user, heapq.nlargest(max_items, scores, key=lambda x: x[1])

    def calculate_rankings_with_k_neighbors(self, k, max_items=None):
        """
        Returns a dict mapping users to a list of (item, score) of
        max_items unrated items in the item set, sorted in descending
        order by score. If max_items is None it returns a ranking of
        all unrated_items.
        """
        user_rankings = {}
        for user in self.users:
            user_rankings[user] = self.calculate_user_ranking(user, k,
                                                              max_items)

        return user_rankings

    def calculate_rankings_to_run_file(self, k, run_file, max_items=None,
                                       workers=1):
        """
        Calculate a ranking for each user of max_item items of the
        items unrated by the user. If max_items is None the ranking
        includes the score for all unrated items for each user.

        It writes the result to a file in a format recognized by the
        trec_eval tool.
        """
        if workers is not None and workers == 1:
            with open(run_file, 'w') as f_out:
                for user in self.users:
                    _, ranking = self.calculate_user_ranking(user, k, max_items)
                    for i, entry in enumerate(ranking):
                        line = '\t'.join([str(user), 'Q0', str(entry[0]),
                                          str(i), str(entry[1]), '-'])
                        line += '\n'
                        f_out.write(line)
        else:
            self.parallel_calculate_rankings_to_run_file(k, run_file,
                                                         max_items, workers)

    def parallel_calculate_rankings_to_run_file(self, k, run_file,
                                                max_items=None, workers=None):
        with Pool(workers) as pool:
            args = ((user, k, max_items) for user in self.users)

            results = pool.starmap(self.calculate_user_ranking, args)

            pool.close()
            pool.join()

        with open(run_file, 'w') as f_out:
            for user, ranking in results:
                for i, (item, score) in enumerate(ranking):
                    line = '\t'.join([str(user), 'Q0', str(item), str(i),
                                      str(score), '-'])
                    line += '\n'
                    f_out.write(line)

        pool.join()

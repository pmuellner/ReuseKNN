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

import numpy as np


class CosineSimilaritiesProxy(object):
    def __init__(self, arr):
        norms = np.linalg.norm(arr, axis=1)
        normalized_embeddings = arr.copy()
        normalized_embeddings /= norms
        self._normalized_embeddings = normalized_embeddings

    def __getitem__(self, i):
        return np.dot(self._normalized_embeddings,
                      self._normalized_embeddings[i].transpose())


def calc_cosines_all(arr):
    norms = np.linalg.norm(arr, axis=1)
    products = np.dot(arr, arr.transpose())
    products /= norms
    products /= norms[:, np.newaxis]
    return products


def calc_cosines(arr):
    if arr.shape[0] > 5000:
        return CosineSimilaritiesProxy(arr)
    else:
        return calc_cosines_all(arr)


class Embeddings(object):

    def __init__(self, load_from, similarity_function=calc_cosines):
        """
          - load_from: load items embeddings from this file.
          - similarity_function: used to calculate the similarities
                                 between embeddings.
        """
        with open(load_from) as f_in:
            line = f_in.readline().split()
            self.total = int(line[0])
            self.dim = int(line[1])
            self.item2index = {}
            self.index2item = []
            self.embeddings = np.zeros((self.total, self.dim))
            for i, line in enumerate(f_in):
                line = line.split()
                item = int(line[0])
                self.item2index[item] = i
                self.index2item.append(item)
                self.embeddings[i] = [float(s) for s in line[1:]]

        #self.similarities = similarity_function(self.embeddings)
        #self.optimized_for_k = None

    def optimize_knn_search(self, k):
        self.sorted_neighbors_items = [None] * self.total
        self.sorted_neighbors_similarities = [None] * self.total
        self.sorted_both = [None] * self.total
        for i in range(self.total):
            item_similarities = self.similarities[i]
            if k >= len(item_similarities):
                this_k = len(item_similarities)
                neighbor_items = np.argpartition(-item_similarities,
                                                 this_k - 1)[:this_k]
            else:
                neighbor_items = np.argpartition(-item_similarities, k)[:k+1]
            neighbor_items = np.delete(neighbor_items,
                                       np.where(neighbor_items == i))
            neighbor_similarities = item_similarities[neighbor_items]
            self.sorted_neighbors_items[i] = [self.index2item[x]
                                              for x in neighbor_items]
            self.sorted_neighbors_similarities[i] = neighbor_similarities

        self.optimized_for_k = k

    def nearest_neighbors(self, k, item):
        i = self.item2index[item]

        if self.optimized_for_k == k:
            return (self.sorted_neighbors_items[i],
                    self.sorted_neighbors_similarities[i])
        else:
            pairs = heapq.nlargest(
                k + 1,
                [(self.index2item[x], y)
                 for x, y in enumerate(self.similarities[i].tolist())],
                key=lambda x: x[1])
            pairs.pop(0)
            ids, sims = [np.array(t) for t in zip(*pairs)]
            return ids, sims

    def item_set(self):
        return set(self.item2index.keys())

import numpy as np
import networkx as nx
from collections import Counter

class Graph:
    def __init__(self, graph, degree_distribution=None):
        self.graph = graph
        if degree_distribution:
            self.p = degree_distribution
        else:
            self.p = self._degree_distribution(self.graph)

        self.n_nodes = self.graph.number_of_nodes()

    def prime_0(self, x):
        n = len(self.p)
        expectation = 0.0
        for k in range(1, n):
            expectation += k * self.p[k] * np.power(x, k - 1)
        return expectation

    def primeprime_0(self, x):
        n = len(self.p)
        expectation = 0.0
        for k in range(2, n):
            expectation += k * (k - 1) * self.p[k] * np.power(x, k - 1)
        return expectation


    @staticmethod
    def _degree_distribution(graph):
        n = graph.number_of_nodes()
        degrees = np.zeros(n)
        for degree, count in Counter([d for _, d in graph.out_degree()]).items():
            degrees[degree] = count
        p = degrees / n
        return p

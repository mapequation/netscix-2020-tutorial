import random
from heapq import heappush, heapreplace
from math import fabs

import numpy as np
#from numba import jit


def plogp(x):
    p = np.asarray(x, dtype=np.float)
    logp = np.zeros(p.shape, dtype=np.float)
    np.log2(p, out=logp, where=p > 0)
    return p * logp


#@jit(nopython=True)
def entropy(x):
    # return -np.sum(plogp(x))

    h = 0

    for p in x:
        if p > 0:
            h -= p * np.log2(p)

    return h


#@jit(nopython=True)
def js_distance(x1, x2):
    if x1 is x2:
        return 0.0

    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    x1_w = 0
    x2_w = 0
    mix = np.empty(x1.shape, dtype=x1.dtype)

    for i in range(x1.shape[0]):
        mix[i] = x1[i] + x2[i]  # mix = x1 + x2
        x1_w += x1[i]  # x1_w = np.sum(x1)
        x2_w += x2[i]  # x2_w = np.sum(x2)

    mix_w = x1_w + x2_w

    return mix_w * entropy(mix / mix_w) - x1_w * entropy(x1 / x1_w) - x2_w * entropy(x2 / x2_w)


def js_label_indices_distance(X, label_indices):
    num_features = X.shape[1]
    mix = np.zeros(num_features)
    js_div = 0.0

    for i in label_indices:
        x = np.asarray(X[i])
        x_w = np.sum(x)
        mix += x
        js_div -= x_w * entropy(x / x_w)

    mix_w = np.sum(mix)
    js_div += mix_w * entropy(mix / mix_w)

    return js_div


class JSdivisiveClustering:
    min_distance = 1e-10

    __slots__ = ("X", "labels", "tot_weight", "js_div_initial", "label_indices_heap")

    def __init__(self, X, tot_weight):
        self.X = X
        num_rows, num_features = X.shape
        self.labels = np.zeros(num_rows, dtype=np.int)
        self.tot_weight = tot_weight

        label_indices = list(range(X.shape[0]))

        # Measure Jensen-Shannon divergence if all state nodes are lumped into one cluster
        self.js_div_initial = js_label_indices_distance(self.X, label_indices) / tot_weight

        # Initiate a heap with clusters sorted in descending divergence order
        # (all are negated because Python offers a min heap)
        self.label_indices_heap = []
        heappush(self.label_indices_heap, (-self.js_div_initial, label_indices))

        # print("Initiated states in one cluster with JS divergence {0:.4f} bits".format(self.js_div_initial))

    def divide_labels(self, js_div_threshold=None, n_clusters=None):
        if js_div_threshold is not None:
            while self.label_indices_heap[0][0] < -js_div_threshold:
                self.divide_label_indices()
        elif n_clusters is not None:
            while len(self.label_indices_heap) < n_clusters and self.label_indices_heap[0][0] < 0.0:
                self.divide_label_indices()
        else:
            print("Please provide objective value.")

        return self.labels

    def proportional_label(self, label_distances, label_indices=None):
        if not label_indices:
            label_indices = self.label_indices_heap[0][1]
        return label_indices[np.argmax(label_distances)]

    def label_distances(self, reference_label_index, label_indices=None):
        if not label_indices:
            label_indices = self.label_indices_heap[0][1]
        return [js_distance(self.X[reference_label_index], self.X[i]) for i in label_indices]

    def divide_label_indices(self):
        # Get split cluster with highest JS divergence
        split_cluster_js_div, split_cluster_label_indices = self.label_indices_heap[0]
        split_cluster_js_div *= -1

        best_old_label_indices = []
        best_new_label_indices = []
        best_old_label_cluster_js_div = 0.0
        best_new_label_cluster_js_div = 0.0

        for attempt in range(5):
            # Pick initial random label index and find distances to other label indices
            random_label_index = random.choice(split_cluster_label_indices)
            random_label_distances = self.label_distances(random_label_index)

            # Pick first center proportional to distance from initial index and find distances to label indices
            first_label_index = self.proportional_label(random_label_distances)
            first_label_distances = self.label_distances(first_label_index)

            # Pick second center proportional to distance from first center and find distances to label indices
            second_label_index = self.proportional_label(first_label_distances)
            second_label_distances = self.label_distances(second_label_index)

            # To split the cluster, assign each label index to the center that it is closest to.
            # Label indices closer to the second center receive a new label
            old_label_indices, new_label_indices = [], []

            for i, label_index in enumerate(split_cluster_label_indices):
                if fabs(first_label_distances[i] - second_label_distances[i]) < self.min_distance:
                    random_cluster = random.choice([new_label_indices, old_label_indices])
                    random_cluster.append(label_index)
                elif second_label_distances[i] < first_label_distances[i]:
                    new_label_indices.append(label_index)
                else:
                    old_label_indices.append(label_index)

            old_label_cluster_js_div = js_label_indices_distance(self.X, old_label_indices) / self.tot_weight
            new_label_cluster_js_div = js_label_indices_distance(self.X, new_label_indices) / self.tot_weight
            total_best_js_div = best_old_label_cluster_js_div + best_new_label_cluster_js_div

            if attempt == 0 or old_label_cluster_js_div + new_label_cluster_js_div < total_best_js_div:
                best_old_label_indices = old_label_indices
                best_new_label_indices = new_label_indices
                best_old_label_cluster_js_div = old_label_cluster_js_div
                best_new_label_cluster_js_div = new_label_cluster_js_div

        new_label = len(self.label_indices_heap)
        for label_index in best_new_label_indices:
            self.labels[label_index] = new_label

        heapreplace(self.label_indices_heap, (-best_old_label_cluster_js_div, best_old_label_indices))
        heappush(self.label_indices_heap, (-best_new_label_cluster_js_div, best_new_label_indices))
        # print("Final: {0:.4f} bits".format(
        #    best_old_label_cluster_js_div + best_new_label_cluster_js_div + split_cluster_js_div))

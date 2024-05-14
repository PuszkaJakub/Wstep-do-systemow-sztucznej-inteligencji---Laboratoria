import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

        def is_leaf_node(self):
            return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None


        def fit(self, X, y):

            pass

        def _grow_tree(self, X, y, depth=0):
            pass

        def _best_split(self, X, y, feat_idxs):
            pass

        def _information_gain(self, X, y, threshold):
            pass

        def _split(self, X, y, split_thresh):
            pass

        def _entropy(self, y):
            hist = np.bincount(y)
            ps = hist / len(y)
            return -np.sum([p * np.log(p) for p in ps if p>0])

        def _traverse_tree(self, x, node):
            pass

        def predict(self, X):
            return np.array([self._traverse_tree(x, self.root) for x in X])
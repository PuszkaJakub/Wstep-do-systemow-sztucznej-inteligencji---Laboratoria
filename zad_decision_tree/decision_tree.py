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
            self.root = self._grow_tree(X, y, 0)

        def _grow_tree(self, X, y, depth=0):
            # ilosc probek, ilosc cech
            num_of_samples, num_of_features = np.shape(X)

            # Jesli warunek stopu jest spelniony, wezel jest lisciem
            if num_of_samples <= self.min_samples_split or depth > self.max_depth:
                # znajduje i przypisuje jako wartosc liscia najczesciej wystepujacy typ
                counter = Counter(y)
                most_common = counter.most_common(1)[0][0]
                return Node(value=most_common)

            # Jesli nie jest spelniony, to jest to wezel decyzyjny

            # losowanie probki do podzialu
            feat_idxs = np.random.choice(num_of_features, self.n_features, replace=False)
            # szukanie najlepszego podzialu, zwraca dane nt. najlepszej cechy i jej wartosci progowej
            best_feature, best_threshold = self._best_split(X, y, feat_idxs)
            # podzial
            left_features, right_features = self._split(X[:, best_feature], y, best_threshold)
            # stworzenie dzieci wezla
            left_subtree = self._grow_tree(X[left_features, :], y[left_features], depth+1)
            right_subtree = self._grow_tree(X[right_features, :], y[right_features], depth+1)
            # zwrocenie kompletnego wezla
            return Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

        def _best_split(self, X, y, feat_idxs):
            best_feature, best_threshold = None, None
            best_info_gain = float('-inf')
            # iteruje przez wszystkie cechy
            for feat_idx in feat_idxs:
                # iteruje przez wszystkie wartosci cech
                thresholds = np.unique(X[:, feat_idx])
                for threshold in thresholds:
                    # jesli znajdzie lepsze info_gain to zamienia
                    gain = self._information_gain(X[:, feat_idx], y, threshold)
                    if gain > best_info_gain:
                        best_info_gain = gain
                        best_feature = feat_idx
                        best_threshold = threshold
            return best_feature, best_threshold


        def _information_gain(self, X, y, threshold):
            pass

        def _split(self, X, y, split_thresh):
            left_features = np.where(X <= split_thresh)[0]
            right_features = np.where(X > split_thresh)[0]
            return left_features, right_features

        def _entropy(self, y):
            hist = np.bincount(y)
            ps = hist / len(y)
            return -np.sum([p * np.log(p) for p in ps if p>0])

        def _traverse_tree(self, x, node):
            pass

        def predict(self, X):
            return np.array([self._traverse_tree(x, self.root) for x in X])

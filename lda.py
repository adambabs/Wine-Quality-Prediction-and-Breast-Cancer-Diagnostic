#%%
#!/usr/bin/python
# essential imports
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import timeit

# LDA test
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class lda(object):
    def __init__(self, rate=0.1, maxGD_iter=1000, tol=0.1):
        # learning rate
        self.rate = rate
        # gradient descent iterations
        self.maxGD_iter = maxGD_iter
        # stopping tolerance
        self.tol = tol

        # initialize weight values
        self.w0 = None
        self.w = None

    def fit(self, X, y):
        # number of data points
        n = X.shape[0]
        # lda mean feature vector u1, u0
        x1 = X[np.where(y == 1)]
        x0 = X[np.where(y == 0)]
        u1 = x1.mean(axis=0)
        u0 = x0.mean(axis=0)
        # log(y1/y0)
        odds = np.count_nonzero(y == 1) / np.count_nonzero(y == 0)
        log_odds = np.log(odds)
        # sigma
        m1 = x1 - u1
        m0 = x0 - u0

        # print((m1 ** 2)[0].sum())
        # m1 = x1 - u1

        m1m = np.zeros((m1.shape[1], m1.shape[1]))
        for i in range(m1.shape[0]):
            # m1t = m1[i]
            b1 = m1[i].reshape((m1.shape[1], 1))
            m1m += b1 @ b1.T

        m0m = np.zeros((m0.shape[1], m0.shape[1]))
        for i in range(m0.shape[0]):
            # m0t = m0[i]
            b0 = m0[i].reshape((m0.shape[1], 1))
            m0m += b0 @ b0.T

        # s1 = (m1 ** 2).sum(axis=0)
        # s0 = (m0 ** 2).sum(axis=0)
        s = (m1m + m0m) / (n - 2)
        s_inv = np.linalg.inv(s)
        # ii = s @ s_inv
        self.w0 = (
            log_odds - (((0.5) * u1.T) @ (s_inv @ u1)) + ((0.5 * u0.T) @ (s_inv @ u0))
        )
        self.w = s_inv @ (u1 - u0)

        # (self.w ** u1)
        # (self.w ** u0)
        pass

    def predict(self, X):
        # check for fitted values
        if self.w0 == None or np.all(self.w) == None:
            return print("please fit the model first")

        y_pred = []
        for xs in X:
            boundary = self.w0 + (xs.T @ self.w)
            if boundary > 0:
                y_pred.append(1)
            else:
                y_pred.append(0)

        return np.asarray(y_pred)


def standardize(X):
    mean = X.mean(axis=0)
    sd = X.std(axis=0)
    return (X - mean) / sd


def normalize(X):
    max = X.max(axis=0)
    min = X.min(axis=0)
    return (X - min) / (max - min)


# computing accuracy
def evaluate_acc(y_true, y_pred):
    # check for same length
    if y_true.shape != y_pred.shape:
        raise ValueError("input lengths are not equal")
    # convert to integer for safer comparison
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    score = y_true == y_pred
    return np.average(score)


# kfold
def kfold_index(k, X):
    n_samples = len(X)
    indices = np.arange(n_samples)
    random.shuffle(indices)
    fold_sizes = np.full(k, n_samples // k, dtype=np.int)
    fold_sizes[: n_samples % k] += 1
    current = 0

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_index = indices[start:stop]
        # print(test_index)
        # print(indices[:start])
        # print(indices[stop:])
        train_index = np.concatenate((indices[:start], indices[stop:]))
        # print(train_index)
        yield train_index, test_index
        current = stop


#%%
# chisquared features test
from scipy.stats import chisquare
def chi2(X, y):
    Y = np.vstack([1 - y, y])
    observed = np.dot(Y, X)
    feature_count = X.sum(axis=0)
    class_prob = Y.mean(axis=1)
    expected = np.dot(feature_count.reshape(-1, 1), class_prob.reshape(1, -1)).T
    score, pval = chisquare(observed, expected)

    return score, pval

#%%
#!/usr/bin/python
# essential imports
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import timeit

import pandas as pd

# user class functions
from lda import evaluate_acc
from lda import lda
from lda import kfold_index
from lda import standardize
from lda import normalize
from logreg import LogReg

# sklearn tests for comparison
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#%%
# Load wine datasets
wine_path = "./Data/winequality-red.csv"
data = np.genfromtxt(wine_path, delimiter=";", skip_header=1)

X_wine = data[:, 0:11]  # select columns 1 through 11
y_wine = data[:, 11]

# convert y values to 0, 1
y_wine = np.where(y_wine < 5.5, 0, 1)

# standardize
# X_wine = standardize(X_wine)

# normalize
X_wine = normalize(X_wine)

# y_wine = y_wine.reshape(-1,1)

#%%
# Load cancer datasets
cancer_path = "./Data/breast-cancer-wisconsin.data"
data = np.genfromtxt(cancer_path, delimiter=",")
X_cancer = data[:, 1:10]  # select columns 2 through 9
y_cancer = data[:, 10]

# remove "?" nan values
index_of_nan = np.where(X_cancer != X_cancer)
X_cancer = np.delete(X_cancer, index_of_nan[0], axis=0)
y_cancer = np.delete(y_cancer, index_of_nan[0], axis=0)
# convert y values to 0,1
y_cancer = np.where(y_cancer < 3, 0, 1)

# standardize
# X_cancer = standardize(X_cancer)

# normalize
X_cancer = normalize(X_cancer)

# y_cancer = y_cancer.reshape(-1,1)

#%%
# repeated kfold test
# wine dataset
import warnings

warnings.filterwarnings("ignore")
# classifier options
ld = lda()
clf = LinearDiscriminantAnalysis()
# ld = LogReg(learning_rate=0.001)
# clf = LogisticRegression(solver='liblinear',C=1000)

score = []
scorec = []
for i in range(100):
    for train_index, test_index in kfold_index(5, X_wine):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_wine[train_index], X_wine[test_index]
        y_train, y_test = y_wine[train_index], y_wine[test_index]
        # project
        ld.fit(X_train, y_train)
        y_pred = ld.predict(X_test)
        score.append(evaluate_acc(y_test, y_pred))

        # scikit learn
        clf.fit(X_train, y_train)
        y_predT = clf.predict(X_test)
        # y_predT = y_predT.reshape(-1,1)
        scorec.append(evaluate_acc(y_test, y_predT))

print(np.mean(score))
print(np.std(score))
print(np.mean(scorec))
print(np.std(scorec))
#%%
# cancer dataset

# classifier options
ld = lda()
clf = LinearDiscriminantAnalysis()
# ld = LogReg(learning_rate=0.001)
# clf = LogisticRegression(solver='liblinear',C=1000)

score = []
scorec = []
for i in range(100):
    for train_index, test_index in kfold_index(5, X_cancer):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_cancer[train_index], X_cancer[test_index]
        y_train, y_test = y_cancer[train_index], y_cancer[test_index]
        # project
        ld.fit(X_train, y_train)
        y_pred = ld.predict(X_test)
        score.append(evaluate_acc(y_test, y_pred))

        # scikit learn
        clf.fit(X_train, y_train)
        y_predT = clf.predict(X_test)
        # y_predT = y_predT.reshape(-1,1)
        scorec.append(evaluate_acc(y_test, y_predT))

print(np.mean(score))
print(np.std(score))
print(np.mean(scorec))
print(np.std(scorec))
#%%
# execution time evaluation
execution_time = []

ld = lda()
ld = LinearDiscriminantAnalysis()
# ld = LogReg(learning_rate=0.001)
# ld = LogisticRegression(C=1000)
for i in range(100):
    import timeit

    start = timeit.default_timer()
    # ALL THE PROGRAM STATEMETNS
    ld.fit(X_wine, y_wine)
    # ld.fit(X_cancer, y_cancer)

    stop = timeit.default_timer()
    etime = stop - start
    execution_time.append(etime)
    # print("Program Executed in ", etime)  # It returns time in sec

mean = np.mean(execution_time)
sd = np.std(execution_time)
print("Mean time: %.6f" % np.mean(execution_time))  # It returns time in sec
print("SD time: %.6f" % np.std(execution_time))
print("95%% confidence interval: (%.6f, %.6f)" % (mean - 2 * sd, mean + 2 * sd))

#%%

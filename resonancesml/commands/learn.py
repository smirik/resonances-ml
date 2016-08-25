import numpy as np
from resonancesml.shortcuts import ProgressBar
from texttable import Texttable
from typing import Tuple
from typing import List
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from resonancesml.shortcuts import get_target_vector
from resonancesml.shortcuts import get_feuture_matrix
import pandas
from resonancesml.settings import CATALOG_PATH
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


def _validate(data: DataFrame):
    flag = False
    for i in data.keys():
        if data[i].hasnans():
            flag = True
            print(i)

    if flag:
        raise Exception('syntetic elements has nan values')


def _classify(clf: ClassifierMixin, kf: cross_validation.KFold, X: np.ndarray, Y: np.ndarray)\
        -> Tuple[float, int, int]:
    precisions = []
    recalls = []
    scores = []

    TPs = []
    FPs = []
    TNs = []
    FNs = []
    for train_index, test_index in kf:
        X_train = X[train_index]
        Y_train = Y[train_index]
        X_test = X[test_index]
        Y_test = Y[test_index]

        clf.fit(X_train, Y_train)
        res = clf.predict(X_test)  # type: np.ndarray
        precisions.append(precision_score(Y_test, res))
        recalls.append(recall_score(Y_test, res))
        scores.append(accuracy_score(Y_test, res))
        TP, FP, TN, FN = _perf_measure(Y_test, res)

        TPs.append(TP)
        FPs.append(FP)
        TNs.append(TN)
        FNs.append(FN)

    return (np.mean(precisions), np.mean(recalls), np.mean(scores),
            np.mean(TPs), np.mean(FPs), np.mean(TNs), np.mean(FNs))


def _get_headers(indices: List[int]) -> List[str]:
    headers = []
    with open(CATALOG_PATH) as f:
        for i, line in enumerate(f):
            if i == 1:
                headers = [x.strip() for x in line.split('  ') if x]
            elif i > 1:
                break
    res = []
    for index in indices:
        res.append(headers[index])
    return res

def _perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
    for i in range(len(y_hat)):
        if y_actual[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==0:
           TN += 1
    for i in range(len(y_hat)):
        if y_actual[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)


def learn(librate_list: str):
    indices_cases = [
        [2,3,4],
        [2,3,4,5]
    ]
    librated_asteroids = np.loadtxt(librate_list, dtype=int)
    slice_len = int(librated_asteroids[-1])
    dtype = {0:str}
    dtype.update({x: float for x in range(1,10)})
    syntetic_elems = pandas.read_csv(CATALOG_PATH, delim_whitespace=True,  # type: DataFrame
                                     skiprows=2, header=None, dtype=dtype)

    learn_feature_set = syntetic_elems.values[:slice_len]  # type: np.ndarray
    table = Texttable(max_width=120)
    table.header(['Classifier', 'Input data (fields)', 'precision', 'recall', 'accuracy', 'TP', 'FP', 'TN', 'FN'])
    table.set_cols_width([30, 30, 15, 15, 15, 5, 5, 5, 5])
    table.set_precision(5)

    bar = ProgressBar(len(indices_cases) * 5, 'Learning', 1)

    for indices in indices_cases:
        classifiers = {
            'Decision tree': DecisionTreeClassifier(random_state=241),
            'Gradient boosting (10 trees)': GradientBoostingClassifier(n_estimators=10),
            'Gradient boosting (50 trees)': GradientBoostingClassifier(n_estimators=50),
            'K neighbors': KNeighborsClassifier(weights='distance', p=100, n_jobs=4),
            'Logistic regression': LogisticRegression(C=10)
        }
        for name, clf in classifiers.items():
            headers = _get_headers(indices)
            Y = get_target_vector(librated_asteroids, learn_feature_set.astype(int))
            X = get_feuture_matrix(learn_feature_set, False, indices)

            kf = cross_validation.KFold(X.shape[0], 5, shuffle=True, random_state=42)
            precision, recall, accuracy, TP, FP, TN, FN = _classify(clf, kf, X, Y)
            table.add_row([name, ', '.join(headers), precision, recall, accuracy,
                           int(TP), int(FP), int(TN), int(FN)])
            bar.update()

    print('\n')
    print(table.draw())
    print('Dataset for used for learning and testing by k-fold cross-validation: %d' % Y.shape)

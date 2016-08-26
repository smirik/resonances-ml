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
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

from enum import Enum
from enum import unique
from settings import SYN_CATALOG_PATH
from settings import CAT_CATALOG_PATH

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
        TP, FP, TN, FN = _perf_measure(res, Y_test)

        precisions.append(TP/(TP+FP) if TP+FP != 0 else 0)
        recalls.append(TP/(TP+FN) if TP+FN != 0 else 0)
        scores.append((TN+TP)/(TN+TP+FN+FP))

        TPs.append(TP)
        FPs.append(FP)
        TNs.append(TN)
        FNs.append(FN)

    return (np.mean(precisions), np.mean(recalls), np.mean(scores),
            np.sum(TPs), np.sum(FPs), np.sum(TNs), np.sum(FNs))


def _get_headers(from_catalog_path:str, by_indices: List[int]) -> List[str]:
    headers = []
    with open(from_catalog_path) as f:
        for i, line in enumerate(f):
            if i == 1:
                headers = [x.strip() for x in line.split('  ') if x]
            elif i > 1:
                break
    res = []
    for index in by_indices:
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


@unique
class Catalog(Enum):
    syn = 'syn'
    cat = 'cat'


class TesterParameters:
    def __init__(self, indices_cases: List[List[int]], catalog_path: str,
                 catalog_width: int):
        self.indices_cases = indices_cases
        self.catalog_path = catalog_path
        self.catalog_width = catalog_width


def get_tester_parameters(catalog: Catalog) -> TesterParameters:
    return {
        Catalog.syn: TesterParameters([[2,3,4],[2,3,4,5]], SYN_CATALOG_PATH, 10),
        Catalog.cat: TesterParameters([[2,3,4],[2,3,4,5]], CAT_CATALOG_PATH, 11),
    }[catalog]


class MethodComparer:
    def __init__(self, librate_list: str, parameters: TesterParameters):
        dtype = {0:str}
        dtype.update({x: float for x in range(1,parameters.catalog_width)})
        self._catalog_feautures = pandas.read_csv(  # type: DataFrame
            parameters.catalog_path, delim_whitespace=True, skiprows=2, header=None, dtype=dtype)
        self._librated_asteroids = np.loadtxt(librate_list, dtype=int)
        self._indices_cases = parameters.indices_cases
        self._catalog_path = parameters.catalog_path

    def learn(self):
        slice_len = int(self._librated_asteroids[-1])
        dtype = {0:str}
        dtype.update({x: float for x in range(1,10)})
        syntetic_elems = pandas.read_csv(self._catalog_path, delim_whitespace=True,  # type: DataFrame
                                         skiprows=2, header=None, dtype=dtype)

        learn_feature_set = syntetic_elems.values[:slice_len]  # type: np.ndarray
        table = Texttable(max_width=120)
        table.header(['Classifier', 'Input data (fields)', 'precision', 'recall',
                      'accuracy', 'TP', 'FP', 'TN', 'FN'])
        table.set_cols_width([30, 30, 15, 15, 15, 5, 5, 5, 5])
        table.set_precision(5)

        bar = ProgressBar(len(self._indices_cases) * 5, 'Learning', 1)

        for indices in self._indices_cases:
            classifiers = {
                'Decision tree': DecisionTreeClassifier(random_state=241),
                'Gradient boosting (10 trees)': GradientBoostingClassifier(n_estimators=10),
                'Gradient boosting (50 trees)': GradientBoostingClassifier(n_estimators=50),
                'K neighbors': KNeighborsClassifier(weights='distance', p=1, n_jobs=4),
                'Logistic regression': LogisticRegression(C=10)
            }
            for name, clf in classifiers.items():
                headers = _get_headers(self._catalog_path, indices)
                Y = get_target_vector(self._librated_asteroids, learn_feature_set.astype(int))
                X = get_feuture_matrix(learn_feature_set, False, indices)

                kf = cross_validation.KFold(X.shape[0], 5, shuffle=True, random_state=42)
                precision, recall, accuracy, TP, FP, TN, FN = _classify(clf, kf, X, Y)
                table.add_row([name, ', '.join(headers), precision, recall, accuracy,
                               int(TP), int(FP), int(TN), int(FN)])
                bar.update()

        print('\n')
        print(table.draw())
        print('Dataset for used for learning and testing by k-fold cross-validation: %d' % Y.shape)

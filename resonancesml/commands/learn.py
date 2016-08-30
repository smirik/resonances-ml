import numpy as np
from resonancesml.shortcuts import ProgressBar
from texttable import Texttable
from typing import List
from typing import Dict
from resonancesml.shortcuts import get_target_vector
from resonancesml.shortcuts import get_feuture_matrix
from .shortcuts import perf_measure
import pandas
from pandas import DataFrame
from sklearn import cross_validation

from enum import Enum
from enum import unique
from resonancesml.settings import SYN_CATALOG_PATH
from resonancesml.settings import CAT_CATALOG_PATH
from abc import abstractclassmethod

from typing import Tuple
from sklearn.base import ClassifierMixin


def _validate(data: DataFrame):
    flag = False
    for i in data.keys():
        if data[i].hasnans():
            flag = True
            print(i)

    if flag:
        raise Exception('syntetic elements has nan values')


@unique
class Catalog(Enum):
    syn = 'syn'
    cat = 'cat'


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
        TP, FP, TN, FN = perf_measure(res, Y_test)

        precisions.append(TP/(TP+FP) if TP+FP != 0 else 0)
        recalls.append(TP/(TP+FN) if TP+FN != 0 else 0)
        scores.append((TN+TP)/(TN+TP+FN+FP))

        TPs.append(TP)
        FPs.append(FP)
        TNs.append(TN)
        FNs.append(FN)

    return (np.mean(precisions), np.mean(recalls), np.mean(scores),
            np.sum(TPs), np.sum(FPs), np.sum(TNs), np.sum(FNs))


class ADatasetInjector:
    def __init__(self, headers: List[str]):
        self.headers = headers

    @abstractclassmethod
    def update_data(self, X: np.ndarray) -> np.ndarray:
        pass


class TesterParameters:
    def __init__(self, indices_cases: List[List[int]], catalog_path: str,
                 catalog_width: int, delimiter: str, skiprows: int, injector: ADatasetInjector = None):
        self.indices_cases = indices_cases
        self.catalog_path = catalog_path
        self.catalog_width = catalog_width
        self.delimiter = delimiter
        self.skiprows = skiprows
        self.injector = injector


class KeplerInjector(ADatasetInjector):
    def update_data(self, X: np.ndarray) -> np.ndarray:
        mean_motions = np.sqrt([0.0002959122082855911025 / X[:,0] ** 3.])
        X = np.hstack((X, mean_motions.T))
        return X


def get_tester_parameters(catalog: Catalog) -> TesterParameters:
    return {
        Catalog.syn: TesterParameters([[2,3,4],[2,3,4,5]], SYN_CATALOG_PATH, 10, '  ', 2),
        Catalog.cat: TesterParameters([[2,3,4]], CAT_CATALOG_PATH, 11, ', ', 6, KeplerInjector(['n'])),
    }[catalog]


class MethodComparer:
    def __init__(self, librate_list: str, parameters: TesterParameters):
        dtype = {0:str}
        dtype.update({x: float for x in range(1,parameters.catalog_width)})
        self._catalog_feautures = pandas.read_csv(  # type: DataFrame
            parameters.catalog_path, delim_whitespace=True,
            skiprows=parameters.skiprows, header=None, dtype=dtype)
        self._librated_asteroids = np.loadtxt(librate_list, dtype=int)
        self._parameters = parameters
        self._classifiers = None  # type: Dict[str, ClassifierMixin]

    def _get_headers(self, by_indices: List[int]) -> List[str]:
        headers = []
        header_line_number = self._parameters.skiprows - 1
        with open(self._parameters.catalog_path) as f:
            for i, line in enumerate(f):
                if i == header_line_number:
                    headers = [x.strip() for x in line.split(self._parameters.delimiter) if x]
                elif i > header_line_number:
                    break
        res = []
        for index in by_indices:
            res.append(headers[index])
        return res

    def set_methods(self, values: Dict[str, ClassifierMixin]):
        self._classifiers = values

    def learn(self):
        slice_len = int(self._librated_asteroids[-1])

        learn_feature_set = self._catalog_feautures.values[:slice_len]  # type: np.ndarray
        table = Texttable(max_width=120)
        table.header(['Classifier', 'Input data (fields)', 'precision', 'recall',
                      'accuracy', 'TP', 'FP', 'TN', 'FN'])
        table.set_cols_width([30, 30, 15, 15, 15, 5, 5, 5, 5])
        table.set_precision(5)

        bar = ProgressBar(len(self._parameters.indices_cases) * len(self._classifiers),
                          'Learning', 1)
        Y = get_target_vector(self._librated_asteroids, learn_feature_set.astype(int))

        for indices in self._parameters.indices_cases:
            headers = self._get_headers(indices)
            X = get_feuture_matrix(learn_feature_set, False, indices)

            if self._parameters.injector:
                headers += self._parameters.injector.headers
                X = self._parameters.injector.update_data(X)

            for name, clf in self._classifiers.items():
                kf = cross_validation.KFold(X.shape[0], 5, shuffle=True, random_state=42)
                precision, recall, accuracy, TP, FP, TN, FN = _classify(clf, kf, X, Y)
                table.add_row([name, ', '.join(headers), precision, recall, accuracy,
                               int(TP), int(FP), int(TN), int(FN)])
                bar.update()

        print('\n')
        print(table.draw())
        print('Dataset for used for learning and testing by k-fold cross-validation: %d' % Y.shape)

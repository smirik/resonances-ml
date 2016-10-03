from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas
from pandas import DataFrame
from typing import Tuple
import numpy as np
from .shortcuts import perf_measure
from resonancesml.shortcuts import get_target_vector
from resonancesml.shortcuts import get_feuture_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from texttable import Texttable
from sklearn.base import ClassifierMixin

from .parameters import TesterParameters


def _classify(clf: ClassifierMixin, X: np.ndarray, Y: np.ndarray,
              X_test: np.ndarray, Y_test: np.ndarray) -> Tuple[float, int, int]:
    clf.fit(X, Y)
    res = clf.predict(X_test)
    TP, FP, TN, FN = perf_measure(res, Y_test)

    precision = precision_score(Y_test, res)
    recall = recall_score(Y_test, res)
    accuracy = accuracy_score(Y_test, res)

    return (precision, recall, accuracy, TP, FP, TN, FN)


class _DataSets:
    def __init__(self, librated_asteroids, learn_feature_set, all_librated_asteroids,
                 test_feature_set):
        self.librated_asteroids = librated_asteroids
        self.learn_feature_set = learn_feature_set
        self.all_librated_asteroids = all_librated_asteroids
        self.test_feature_set = test_feature_set


NUMBERED_ASTEROID_COUNT = 406253


def _get_datasets(librate_list: str, all_librated: str, parameters: TesterParameters,
                  slice_len: int = None) -> _DataSets:
    librated_asteroids = np.loadtxt(librate_list, dtype=int)
    all_librated_asteroids = np.loadtxt(all_librated, dtype=int)
    dtype = {0:str}
    dtype.update({x: float for x in range(1, parameters.catalog_width)})
    skiprows = parameters.skiprows
    catalog_features = pandas.read_csv(  # type: DataFrame
        parameters.catalog_path, delim_whitespace=True,
        skiprows=skiprows, header=None, dtype=dtype).values
    tail = NUMBERED_ASTEROID_COUNT - skiprows

    if parameters.injection:
        catalog_features = parameters.injection.update_data(
            catalog_features[:tail])

    if slice_len is None:
        slice_len = int(librated_asteroids[-1])
    learn_feature_set = catalog_features[:slice_len]  # type: np.ndarray
    test_feature_set = catalog_features[slice_len:tail]  # type: np.ndarray
    return _DataSets(librated_asteroids, learn_feature_set,
                     all_librated_asteroids, test_feature_set)


def _build_table() -> Texttable:
    table = Texttable(max_width=120)
    table.header(['Classifier', 'precision', 'recall', 'accuracy', 'TP', 'FP', 'TN', 'FN'])
    table.set_cols_width([30, 15, 15, 15, 5, 5, 5, 5])
    table.set_precision(5)
    return table


def _classify_all(datasets: _DataSets, parameters: TesterParameters, clf_name: str = None):
    table = _build_table()
    classifiers = {
        'Decision tree': DecisionTreeClassifier(random_state=241),
        'K neighbors': KNeighborsClassifier(weights='distance', p=1, n_jobs=4),
        'GB': GradientBoostingClassifier(n_estimators=7, learning_rate=0.6, min_samples_split=150),
    }

    data = []
    for indices in parameters.indices_cases:
        X = get_feuture_matrix(datasets.learn_feature_set, False, indices)
        Y = get_target_vector(datasets.librated_asteroids, datasets.learn_feature_set.astype(int))

        X_test = get_feuture_matrix(datasets.test_feature_set, False, indices)
        Y_test = get_target_vector(datasets.all_librated_asteroids,
                                   datasets.test_feature_set.astype(int))


        for name, clf in classifiers.items():
            precision, recall, accuracy, TP, FP, TN, FN = _classify(clf, X, Y, X_test, Y_test)
            data.append('%s;%s;%s' % (name, TP, FP))
            data.append('%s;%s;%s' % (name, FN, TN))
            table.add_row([name, precision, recall, accuracy, TP, FP, TN, FN])


    with open('data.csv', 'w') as f:
        for item in data:
            f.write('%s\n' % item)

    print('\n')
    print(table.draw())
    print('resonant %i' % Y[Y==1].shape[0])
    print('learn %i' % datasets.learn_feature_set.shape[0])
    print('total %i' % (datasets.learn_feature_set.shape[0] + datasets.test_feature_set.shape[0]))


def clear_classify_all(all_librated: str, parameters: TesterParameters, length):
    datasets = _get_datasets(all_librated, all_librated, parameters, length)
    _classify_all(datasets, parameters)


def classify_all(librate_list: str, all_librated: str, parameters: TesterParameters, clf_name: str = None):
    datasets = _get_datasets(librate_list, all_librated, parameters)
    _classify_all(datasets, parameters, clf_name)

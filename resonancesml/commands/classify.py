from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas
from pandas import DataFrame
import numpy as np
from .shortcuts import perf_measure
from resonancesml.shortcuts import get_target_vector
from resonancesml.shortcuts import get_feuture_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from texttable import Texttable

from typing import Tuple
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


def classify_all(librate_list: str, all_librated: str, parameters: TesterParameters):
    table = Texttable(max_width=120)
    table.header(['Classifier', 'precision', 'recall', 'accuracy', 'TP', 'FP', 'TN', 'FN'])
    table.set_cols_width([30, 15, 15, 15, 5, 5, 5, 5])
    table.set_precision(5)
    librated_asteroids = np.loadtxt(librate_list, dtype=int)
    all_librated_asteroids = np.loadtxt(all_librated, dtype=int)
    classifiers = {
        'Decision tree': DecisionTreeClassifier(random_state=241, max_depth=5),
        'K neighbors': KNeighborsClassifier(weights='distance', p=1, n_jobs=4),
    }
    dtype = {0:str}
    dtype.update({x: float for x in range(1, parameters.catalog_width)})
    catalog_feautures = pandas.read_csv(  # type: DataFrame
        parameters.catalog_path, delim_whitespace=True,
        skiprows=parameters.skiprows, header=None, dtype=dtype)

    slice_len = int(librated_asteroids[-1])
    learn_feature_set = catalog_feautures.values[:slice_len]  # type: np.ndarray
    test_feature_set = catalog_feautures.values[:400000]  # type: np.ndarray


    data = []  # type: List[str]
    for indices in parameters.indices_cases:
        Y = get_target_vector(librated_asteroids, learn_feature_set.astype(int))
        X = get_feuture_matrix(learn_feature_set, False, indices)

        Y_test = get_target_vector(all_librated_asteroids, test_feature_set.astype(int))
        X_test = get_feuture_matrix(test_feature_set, False, indices)

        for name, clf in classifiers.items():
            precision, recall, accuracy, TP, FP, TN, FN = _classify(clf, X, Y, X_test, Y_test)

            data.append('%s;%s;%s' % (name, TP, FP))
            data.append('%s;%s;%s' % (name, FN, TN))
            table.add_row([
                name, precision, recall, accuracy, TP, FP, TN, FN
            ])

    with open('data.csv', 'w') as f:
        for item in data:
            f.write('%s\n' % item)

    print('\n')
    print(table.draw())

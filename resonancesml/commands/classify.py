from sklearn.neighbors import KNeighborsClassifier
from resonancesml.loader import get_asteroids
from resonancesml.loader import get_catalog_dataset
from resonancesml.loader import get_learn_set
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from typing import Dict
from typing import List
import numpy as np
from .shortcuts import perf_measure
from resonancesml.shortcuts import get_target_vector
from resonancesml.shortcuts import get_feuture_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from texttable import Texttable
from sklearn.base import ClassifierMixin
from .parameters import DatasetParameters
from resonancesml.knezevic import knezevic_metric


class ClassifyResult:
    def __init__(self, Y_pred, Y_test):
        self.predictions = Y_pred
        self.TP, self.FP, self.TN, self.FN = perf_measure(Y_pred, Y_test)
        self.precision = precision_score(Y_test, Y_pred)
        self.recall = recall_score(Y_test, Y_pred)
        self.accuracy = accuracy_score(Y_test, Y_pred)


def _classify(clf: ClassifierMixin, X: np.ndarray, Y: np.ndarray,
              X_test: np.ndarray, Y_test: np.ndarray) -> ClassifyResult:
    clf.fit(X, Y)
    res = clf.predict(X_test)
    return ClassifyResult(res, Y_test)


class _DataSets:
    def __init__(self, librated_asteroids, learn_feature_set, all_librated_asteroids,
                 test_feature_set):
        self.librated_asteroids = librated_asteroids
        self.learn_feature_set = learn_feature_set
        self.all_librated_asteroids = all_librated_asteroids
        self.test_feature_set = test_feature_set


def _get_datasets(librate_list: str, all_librated: str, parameters: DatasetParameters,
                  slice_len: int = None) -> _DataSets:
    all_librated_asteroids = np.loadtxt(all_librated, dtype=int)
    catalog_features = get_catalog_dataset(parameters).values
    librated_asteroids = get_asteroids(librate_list,  catalog_features[:, 0].astype(int))
    if slice_len is None:
        slice_len = librated_asteroids[-1]

    learn_feature_set = get_learn_set(catalog_features, str(slice_len))  # type: np.ndarray
    test_feature_set = catalog_features[learn_feature_set.shape[0]:]  # type: np.ndarray

    max_number_catalog = test_feature_set[:, 0][-1]
    all_librated_asteroids = all_librated_asteroids[np.where(
        all_librated_asteroids <= int(max_number_catalog)
    )]
    mask = np.in1d(all_librated_asteroids, catalog_features[:, 0].astype(int))
    all_librated_asteroids = all_librated_asteroids[mask]
    return _DataSets(librated_asteroids, learn_feature_set,
                     all_librated_asteroids, test_feature_set)


def _build_table() -> Texttable:
    table = Texttable(max_width=120)
    table.header(['Classifier', 'indices', 'precision', 'recall', 'accuracy', 'TP', 'FP', 'TN', 'FN'])
    table.set_cols_width([20, 8, 11, 11, 11, 7, 7, 7, 7])
    table.set_precision(5)
    return table


def _classify_all(datasets: _DataSets, parameters: DatasetParameters,
                  clf_name: str = None) -> Dict[str, ClassifyResult]:
    table = _build_table()
    classifiers = {
        'DT': DecisionTreeClassifier(random_state=241),
        'KNN': KNeighborsClassifier(weights='distance', p=1, n_jobs=4),
        'GB': GradientBoostingClassifier(n_estimators=7, learning_rate=0.6, min_samples_split=150),
    }
    result = {}

    data = []
    for indices in parameters.indices_cases:
        X = get_feuture_matrix(datasets.learn_feature_set, False, indices)
        Y = get_target_vector(datasets.librated_asteroids, datasets.learn_feature_set.astype(int))

        X_test = get_feuture_matrix(datasets.test_feature_set, False, indices)
        Y_test = get_target_vector(datasets.all_librated_asteroids,
                                   datasets.test_feature_set.astype(int))

        for name, clf in classifiers.items():
            if clf_name and clf_name != name:
                continue
            res = _classify(clf, X, Y, X_test, Y_test)
            data.append('%s;%s;%s;%s' % (name, ' '.join([str(x) for x in indices]), res.TP, res.FP))
            data.append('%s;%s;%s;%s' % (name, ' '.join([str(x) for x in indices]), res.FN, res.TN))
            table.add_row([name, ' '.join([str(x) for x in indices]), res.precision, res.recall,
                           res.accuracy, res.TP, res.FP, res.TN, res.FN])
            result[name + '-' + '-'.join([str(x) for x in indices])] = res

    with open('data.csv', 'w') as fd:
        for item in data:
            fd.write('%s\n' % item)

    print('\n')
    print(table.draw())
    print('Amount of resonant asteroids in learning dataset %i' % Y[Y==1].shape[0])
    print('Learning dataset shape %i' % datasets.learn_feature_set.shape[0])
    print('Total amount of asteroids %i' % (datasets.learn_feature_set.shape[0] + datasets.test_feature_set.shape[0]))

    return result


def _build_clf_kwargs(metric: str) -> dict:
    if metric == 'euclidean':
        return {'p': 2}
    elif metric == 'knezevic':
        return {'metric': knezevic_metric}
    else:
        raise Exception('wrong metric')


def test_classifier(X_train: np.ndarray, X_test: np.ndarray, Y_train: np.ndarray,
                    Y_test: np.ndarray, indices: List[int], metric: str):
    """
    :param X_train: feature 2d matrix for learning.
    :param X_test: feature 2d matrix with testing.
    :param Y_train: response vector for learning.
    :param Y_test: response vector for testing.
    :param indices: indices of columns selected from catalog for building data.
    They are need for report.
    :param metric: metric used by KNN classifier.

    Trains classifier and test it. After testing show results divided on basis
    of general metrics (precision, recall, accuracy). Also shows numbers of
    classified objects separated on true positive (TP), false positive (FP),
    true negative (TN), false negative (FN).
    """
    clf = KNeighborsClassifier(weights='distance', n_jobs=1, algorithm='ball_tree',
                               **_build_clf_kwargs(metric))
    res = _classify(clf, X_train, Y_train, X_test, Y_test)
    result = [res.precision, res.recall, res.accuracy, res.TP, res.FP, res.TN, res.FN]
    table = _build_table()
    table.add_row([str(clf.__class__), ' '.join([str(x) for x in indices])] + result)
    print(table.draw())


def get_librated_asteroids(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray,
                           metric: str) -> np.ndarray:
    """
    Returns vector of asteroid's numbers that librates. This vector is get by classifier.
    """
    clf = KNeighborsClassifier(weights='distance', n_jobs=4, algorithm='ball_tree',
                               **_build_clf_kwargs(metric))
    clf.fit(X_train, Y_train)
    return clf.predict(X_test)


def clear_classify_all(all_librated: str, parameters: DatasetParameters, length):
    datasets = _get_datasets(all_librated, all_librated, parameters, length)
    _classify_all(datasets, parameters)


def classify_all(librate_list: str, all_librated: str, parameters: DatasetParameters,
                 clf_name: str = None):
    datasets = _get_datasets(librate_list, all_librated, parameters)
    res = _classify_all(datasets, parameters, clf_name)
    for name, result in res.items():
        numbers_int = np.array([datasets.test_feature_set[:, 0].astype(int)]).T
        all_objects = np.hstack((
            numbers_int, datasets.test_feature_set, np.array([result.predictions]).T
        ))

        predicted_objects = all_objects[np.where(all_objects[:, -1] == 1)]
        predicted_objects_2 = predicted_objects[np.where(predicted_objects[:, 0] > 249567)][:, 1]

        mask = np.in1d(predicted_objects[:, 0], datasets.all_librated_asteroids[50:])
        predicted_objects_FP = predicted_objects[np.invert(mask)][:, 1]
        mask = np.in1d(datasets.all_librated_asteroids[50:], predicted_objects[:, 0])
        predicted_objects_FN = datasets.all_librated_asteroids[50:][np.invert(mask)].astype(str)

        with open('report-%s.txt' % name, 'w') as fd:
            fd.write('Predicted asteroids:\n%s\n' % ','.join(predicted_objects[:, 1]))
            fd.write('Predicted asteroids after 249567:\n%s\n' % ','.join(predicted_objects_2))
            fd.write('FP:\n%s\n' % ','.join(predicted_objects_FP))
            fd.write('FN:\n%s\n' % ','.join(predicted_objects_FN))
            fd.write('Asteroids was found by integration: %s\n' % datasets.all_librated_asteroids.shape[0])
            fd.write('Asteroids was found by ML: %s' % predicted_objects.shape[0])

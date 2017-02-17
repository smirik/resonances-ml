from resonancesml.loader import get_asteroids
from resonancesml.loader import get_learn_set
from typing import Dict
from typing import List
from typing import Tuple
from typing import Callable
from typing import Any
import numpy as np
from .shortcuts import perf_measure
from resonancesml.shortcuts import get_target_vector
from resonancesml.shortcuts import get_feature_matrix
from resonancesml.shortcuts import ClfPreset
from resonancesml.shortcuts import get_classifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from texttable import Texttable
from sklearn.base import ClassifierMixin
from resonancesml.reader import CatalogReader
from time import time


class ClassifyResult:
    def __init__(self, Y_pred: np.ndarray, Y_test: np.ndarray,
                 fit_time: float, predict_time: float):
        self.predictions = Y_pred
        self.TP, self.FP, self.TN, self.FN = perf_measure(Y_pred, Y_test)
        self.precision = precision_score(Y_test, Y_pred)
        self.recall = recall_score(Y_test, Y_pred)
        self.accuracy = accuracy_score(Y_test, Y_pred)

        self._fit_time = fit_time
        self._predict_time = predict_time

    @property
    def fit_time(self) -> float:
        return self._fit_time

    @property
    def predict_time(self) -> float:
        return self._predict_time


def _time_exec(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    start_time = time()
    res = func(*args, **kwargs)
    stop_time = time()
    execution_time = stop_time - start_time
    return res, execution_time


def _classify(clf: ClassifierMixin, X: np.ndarray, Y: np.ndarray,
              X_test: np.ndarray, Y_test: np.ndarray, verbose=0) -> ClassifyResult:
    fit_time, predict_time = None, None
    if verbose > 0:
        clf, fit_time = _time_exec(clf.fit, X, Y)
    else:
        clf = clf.fit(X, Y)

    if verbose > 0:
        res, predict_time = _time_exec(clf.predict, X_test)
    else:
        res = clf.predict(X_test)

    return ClassifyResult(res, Y_test, fit_time, predict_time)


class _DataSets:
    def __init__(self, librated_asteroids, learn_feature_set, all_librated_asteroids,
                 test_feature_set):
        self.librated_asteroids = librated_asteroids
        self.learn_feature_set = learn_feature_set
        self.all_librated_asteroids = all_librated_asteroids
        self.test_feature_set = test_feature_set


def _get_datasets(librate_list: str, all_librated: str, catalog_reader: CatalogReader,
                  slice_len: int = None) -> _DataSets:
    """
    Gets feature dataset from catalog by pointed catalog reader argument, loads
    vector of librated asteroids and separate it on train and test datasets.

    :param librate_list: path to file contains vector of asteroid's numbers that librates.
    It will be used for learning dataset.
    :param all_librated: path to file contains vector of asteroid's numbers
    that librates. By words from ML this numbers of objects that from true
    class. It will be used for test set.
    :param catalog_reader: catalog reader.
    :param slice_len: points length of learning dataset. If not pointed the
    length will be equal to last number from vector  from file pointed by path librate_list.
    """
    all_librated_asteroids = np.loadtxt(all_librated, dtype=int)
    catalog_features = catalog_reader.read().values
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


def _get_classifiers(clf_presets: Tuple[ClfPreset, ...])\
        -> Tuple[Dict[str, ClassifierMixin], List[str]]:
    classifiers = {}
    preset_names = []
    for clf_preset in clf_presets:
        preset_name = '%s %s' % clf_preset
        clf = get_classifier(clf_preset)
        classifiers[preset_name] = clf
        preset_names.append(preset_name)
    return classifiers, preset_names


def _classify_all(datasets: _DataSets, parameters: CatalogReader,
                  clf_presets: Tuple[ClfPreset, ...], verbose=0)\
        -> Dict[str, ClassifyResult]:
    table = _build_table()
    result = {}
    classifiers, preset_names = _get_classifiers(clf_presets)

    data = []
    for indices in parameters.indices_cases:
        X = get_feature_matrix(datasets.learn_feature_set, False, indices)
        Y = get_target_vector(datasets.librated_asteroids, datasets.learn_feature_set.astype(int))

        X_test = get_feature_matrix(datasets.test_feature_set, False, indices)
        Y_test = get_target_vector(datasets.all_librated_asteroids,
                                   datasets.test_feature_set.astype(int))

        for name in preset_names:
            clf = classifiers[name]
            res = _classify(clf, X, Y, X_test, Y_test, verbose)
            data.append('%s;%s;%s;%s' % (name, ' '.join([str(x) for x in indices]), res.TP, res.FP))
            data.append('%s;%s;%s;%s' % (name, ' '.join([str(x) for x in indices]), res.FN, res.TN))
            table.add_row([name, ' '.join([str(x) for x in indices]), res.precision, res.recall,
                           res.accuracy, res.TP, res.FP, res.TN, res.FN])
            result[name + '-' + '-'.join([str(x) for x in indices])] = res

            if verbose > 0:
                fit_time = res.fit_time
                predict_time = res.predict_time
                print('%s. Fit time: %f. Predict time: %f.' % (name, fit_time, predict_time))

    with open('data.csv', 'w') as fd:
        for item in data:
            fd.write('%s\n' % item)

    print('\n')
    print(table.draw())
    print('Amount of resonant asteroids in learning dataset %i' % Y[Y==1].shape[0])
    print('Learning dataset shape %i' % datasets.learn_feature_set.shape[0])
    print('Total amount of asteroids %i' % (datasets.learn_feature_set.shape[0] +
                                            datasets.test_feature_set.shape[0]))

    return result


def test_classifier(X_train: np.ndarray, X_test: np.ndarray, Y_train: np.ndarray,
                    Y_test: np.ndarray, indices: List[int], clf_preset: ClfPreset):
    """
    :param X_train: feature 2d matrix for learning.
    :param X_test: feature 2d matrix with testing.
    :param Y_train: response vector for learning.
    :param Y_test: response vector for testing.
    :param indices: indices of columns selected from catalog for building data.
    They are need for report.
    :param clf_preset: data pointing on preset in configuration.

    Trains classifier and test it. After testing show results divided on basis
    of general metrics (precision, recall, accuracy). Also shows numbers of
    classified objects separated on true positive (TP), false positive (FP),
    true negative (TN), false negative (FN).
    """
    clf = get_classifier(clf_preset)
    res = _classify(clf, X_train, Y_train, X_test, Y_test)
    result = [res.precision, res.recall, res.accuracy, res.TP, res.FP, res.TN, res.FN]
    table = _build_table()
    table.add_row([str(clf.__class__), ' '.join([str(x) for x in indices])] + result)
    print(table.draw())


def get_librated_asteroids(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray,
                           clf_preset: ClfPreset) -> np.ndarray:
    """
    Returns classes of X_test matrix.
    """
    clf = get_classifier(clf_preset)
    clf.fit(X_train, Y_train)
    return clf.predict(X_test)


def clear_classify_all(all_librated: str, parameters: CatalogReader, length,
                       clf_presets: Tuple[ClfPreset, ...]):
    datasets = _get_datasets(all_librated, all_librated, parameters, length)
    _classify_all(datasets, parameters, clf_presets)


SMIRIK_DISCOVERED_AST_COUNT = 249567


def classify_all(librate_list: str, all_librated: str, parameters: CatalogReader,
                 clf_presets: Tuple[ClfPreset, ...], verbose=0):

    datasets = _get_datasets(librate_list, all_librated, parameters)
    res = _classify_all(datasets, parameters, clf_presets, verbose)
    for name, result in res.items():
        numbers_int = np.array([datasets.test_feature_set[:, 0].astype(int)]).T
        all_objects = np.hstack((
            numbers_int, datasets.test_feature_set, np.array([result.predictions]).T
        ))

        predicted_objects = all_objects[np.where(all_objects[:, -1] == 1)]
        mask = np.where(predicted_objects[:, 0] > SMIRIK_DISCOVERED_AST_COUNT)
        predicted_objects_2 = predicted_objects[mask][:, 1]

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

from typing import Tuple
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.cross_validation import KFold
from typing import Callable
from threading import Lock


def perf_measure(y_actual, y_hat):
    """
    y_actual is predicated
    y_hat is real
    """
    TP = y_actual[np.where((y_actual == y_hat) & (y_hat == 1))].shape[0]
    FP = y_actual[np.where((y_actual == 1) & (y_hat != y_actual))].shape[0]
    TN = y_actual[np.where((y_actual == y_hat) & (y_hat == 0))].shape[0]
    FN = y_actual[np.where((y_actual == 0) & (y_hat != y_actual))].shape[0]

    return (TP, FP, TN, FN)


Modifier = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def classify(clf: ClassifierMixin, kf: KFold, X: np.ndarray, Y: np.ndarray,
             trainset_modifier: Modifier = None)\
        -> Tuple[float, float, float, int, int, int, int]:
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
        if trainset_modifier:
            X_train, Y_train = trainset_modifier(X_train, Y_train)
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


def classify_as_dict(clf: ClassifierMixin, kf: KFold, features: np.ndarray, targets: np.ndarray,
                     trainset_modifier: Modifier = None) -> dict:
    precision, recall, accuracy, TP, FP, TN, FN = classify(
        clf, kf, features, targets, trainset_modifier)
    scores = {
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
    }
    return scores


class DataLockAdapter(object):
    """
    Adapter for list aims to locking it for appending.
    """
    def __init__(self):
        self._lock = Lock()
        self._data = []

    def append(self, value):
        self._lock.acquire()
        self._data.append(value)
        self._lock.release()

    @property
    def data(self) -> list:
        return self._data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]

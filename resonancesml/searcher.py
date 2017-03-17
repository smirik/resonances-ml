from abc import abstractmethod
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold


class ASearcher(object):
    """
    Custom Grid Search returns:
        precision
        recall
        accuracy
        true positive
        false positive
        true negative
        false negative
    """
    _SCORE_NAMES = ['precision', 'recall', 'accuracy', 'TP', 'FP', 'TN', 'FN']

    def __init__(self, verbose=False):
        self._verbose = verbose

    @abstractmethod
    def fit(self, features: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
        pass

    def _build_cv(self, size) -> KFold:
        cv = KFold(size, n_folds=5, random_state=42, shuffle=True)
        return cv

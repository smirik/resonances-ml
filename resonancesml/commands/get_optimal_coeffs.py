from resonancesml.reader import build_reader
from resonancesml.reader import Catalog
from resonancesml.builders.learningset import LearningSetBuilder
from typing import List
import numpy as np
import pandas as pd
from resonancesml.searcher import ASearcher
from resonancesml.shortcuts import get_classifier_with_kwargs
from resonancesml.shortcuts import ClfPreset
from resonancesml.report import view_for_total_comparing
from resonancesml.settings import params
from itertools import product
from .shortcuts import classify_as_dict
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from threading import Lock
from asyncio.futures import Future
from sklearn.base import clone as sklearn_clone


class _DataLockAdapter(object):
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



class _CoeffitientsSearcher(ASearcher):
    """
    Aims to search optimal coeffitients of features for pointed classifier
    preset.  It makes vector of coeffitients contains real numbers from 0 to 1
    then it makes set of coeff combinations from this vector with length equals
    number of features.
    """
    def __init__(self, clf_preset: ClfPreset, verbose: bool):
        super(_CoeffitientsSearcher, self).__init__(verbose)
        self._clf, self._classifier_params = get_classifier_with_kwargs(clf_preset)
        p = self._classifier_params['p']
        self._coeff_vals = [(x / 10) ** (1 / p) for x in range(11)]
        self._data = None
        self._worker_number = params()['system']['threads']

    def _accumulate_scores(self, features: np.ndarray, targets: np.ndarray, coeffs: dict) -> dict:
        """
        Clones classifier prototype, measures features by coeffitients and
        returns scores of metrics.
        """
        mutable_features = features.copy()
        mutable_features[:,0] *= coeffs['ka']
        mutable_features[:,1] *= coeffs['ke']
        mutable_features[:,2] *= coeffs['ki']
        mutable_features[:,3] *= coeffs['kn']
        cv = self._build_cv(targets.shape[0])
        clf_copy = sklearn_clone(self._clf)
        scores = classify_as_dict(clf_copy, cv, mutable_features, targets)
        if self._verbose:
            sorded_coeffs = sorted(coeffs.items())
            print('{%s}' % ', '.join('%s: %s' % (x, y) for x, y in sorded_coeffs))
        scores.update(self._classifier_params)
        scores.update(coeffs)
        return scores

    def _callback(self, future: Future):
        scores = future.result()  # type: dict
        self._data.append(scores)

    def _fit(self, features: np.ndarray, targets: np.ndarray):
        """Fills field self._data"""
        if self._verbose:
            print('Number of threads %s' % self._worker_number)
        with PoolExecutor(max_workers=self._worker_number) as excr:
            for k_a, k_e, k_i, k_n in product(self._coeff_vals, repeat=4):
                coeffs = {
                    'ka': k_a,
                    'ke': k_e,
                    'ki': k_i,
                    'kn': k_n,
                }
                future = excr.submit(self._accumulate_scores, features, targets, coeffs)
                future.add_done_callback(self._callback)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
        """Saves self._data to result pandas' DataFrame and return it"""
        self._data = _DataLockAdapter()
        self._fit(features, targets)
        columns = self._data[0].keys()
        coeffs_vs_metrics = pd.DataFrame(columns=columns, index=range(len(self._data)),
                                         data=self._data.data)
        return coeffs_vs_metrics


_REPORT_COLS = ['ka', 'ke', 'ki', 'kn', 'TP', 'TN', 'FP', 'FN', 'accuracy', 'precision', 'recall']


def get_optimal_coeffs(clf_preset: ClfPreset, librate_list: str,
                       catalog: str, indices: List[int]):
    """
    Fits optimal coeffitients for features from dataset and saves them to
    file named as <name-of-classifier>_coeff_search.csv in current directory.

    :param clf_preset: Classifier by pointed preset that contains name of
    classifier and number of parameter preset (for availbale presets see
    `python -m resonancesml dump-config` in section `classifiers`).
    :param librate_list: path fto file contains list of librated asteroids.
    :param catalog: path of file with catalog of elements.
    :param indices: indeces of columns in catalog used for features.
    """
    reader = build_reader(Catalog(catalog), None, [indices])
    learning_set_builder = LearningSetBuilder(librate_list, reader)
    targets = learning_set_builder.targets

    searcher = _CoeffitientsSearcher(clf_preset, True)
    features, headers = learning_set_builder.build_features_case()
    filename = '%s_coeff_search.csv' % clf_preset[0]
    data = searcher.fit(features, targets)
    view_for_total_comparing(data, filename, [], _REPORT_COLS)

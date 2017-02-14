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
from itertools import product
from .shortcuts import classify_as_dict


class _CoeffitientsSearcher(ASearcher):

    def __init__(self, clf_preset: ClfPreset, verbose: bool):
        super(_CoeffitientsSearcher, self).__init__(verbose)
        self._clf, self._classifier_params = get_classifier_with_kwargs(clf_preset)
        p = self._classifier_params['p']
        self._coeff_vals = [(x / 10) ** (1 / p) for x in range(11)]

    def fit(self, features: np.ndarray, targets: np.ndarray) -> pd.DataFrame:

        #print(features[0])
        data = []
        coeffs = { 'ka': 0, 'ke': 0, 'ki': 0, 'kn': 0 }
        for k_a, k_e, k_i, k_n in product(self._coeff_vals, repeat=4):
            mutable_features = features.copy()
            coeffs.update({
                'ka': k_a,
                'ke': k_e,
                'ki': k_i,
                'kn': k_n,
            })
            if self._verbose:
                sorded_coeffs = sorted(coeffs.items())
                print('{%s}' % ', '.join('%s: %s' % (x, y) for x, y in sorded_coeffs))
            mutable_features[:,0] *= k_a
            mutable_features[:,1] *= k_e
            mutable_features[:,2] *= k_i
            mutable_features[:,3] *= k_n

            cv = self._build_cv(targets.shape[0])
            scores = classify_as_dict(self._clf, cv, mutable_features, targets)
            scores.update(self._classifier_params)
            scores.update(coeffs)
            data.append(scores)

        columns = data[0].keys()
        coeffs_vs_metrics = pd.DataFrame(columns=columns, index=range(len(data)), data=data)
        return coeffs_vs_metrics


HEADERS = ['a', 'e', 'sin I', 'n']


REPORT_COLS = ['ka', 'ke', 'ki', 'kn', 'TP', 'TN', 'FP', 'FN', 'accuracy', 'precision', 'recall']


def get_optimal_coeffs(clf_preset: ClfPreset, librate_list: str,
                       catalog: str, indices: List[int]):
    reader = build_reader(Catalog(catalog), None, [indices])
    learning_set_builder = LearningSetBuilder(librate_list, reader)
    targets = learning_set_builder.targets
    searcher = _CoeffitientsSearcher(clf_preset, True)
    features, headers = learning_set_builder.build_features_case()
    filename = '%s_coeff_search.csv' % clf_preset[0]
    data = searcher.fit(features, targets)
    view_for_total_comparing(data, filename, [], REPORT_COLS)

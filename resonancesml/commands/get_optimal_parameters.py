"""
Module aims to get optimal parameters for different cases of feature set.
"""
from resonancesml.shortcuts import get_classifier_class
from resonancesml.settings import params
from resonancesml.report import view_for_total_comparing
from collections import defaultdict
from resonancesml.shortcuts import FAIL, ENDC, OK
from resonancesml.reader import build_reader_for_grid
import numpy as np
import pandas as pd
from resonancesml.builders.learningset import LearningSetBuilder
from resonancesml.reader import Catalog
from resonancesml.searcher import ASearcher
from itertools import product
from .shortcuts import classify_as_dict
from .shortcuts import Modifier


def _get_config_parameters(clf_name: str) -> defaultdict:
    try:
        param_grid = params()['grid_search'][clf_name]  # type: dict
    except KeyError:
        print(FAIL, 'parameter ["classifiers"]["%s"] doesn\'t exist.' % clf_name, ENDC, sep='')
        exit(-1)
    return param_grid


class _CustomGridSearch(ASearcher):

    def __init__(self, clf_name: str, verbose=False):
        """Searches best parameters

        """
        super(_CustomGridSearch, self).__init__(verbose)
        self._classifier_cls = get_classifier_class(clf_name)
        self._suggested_parameters = _get_config_parameters(clf_name)
        self._param_names = list(self._suggested_parameters.keys())
        self._trainset_modifier = None  # type: Modifier

    @property
    def trainset_modifier(self) -> Modifier:
        return self._trainset_modifier

    @trainset_modifier.setter
    def trainset_modifier(self, method: Modifier):
        self._trainset_modifier = method

    def fit(self, features: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
        product_gen = product(*[x for x in self._suggested_parameters.values()])
        param_sets = [x for x in product_gen]
        number_of_cases = len(param_sets)

        data = []
        for i, param_set in enumerate(param_sets):
            classifier_params = {x: y for x, y in zip(self._param_names, param_set)}
            if self._verbose:
                print(classifier_params)
            cv = self._build_cv(targets.shape[0])
            clf = self._classifier_cls(**classifier_params)
            scores = classify_as_dict(clf, cv, features, targets, self.trainset_modifier)
            scores.update(classifier_params)
            data.append(scores)

        parameters_vs_metrics = pd.DataFrame(columns=self._param_names + self._SCORE_NAMES,
                                             index=range(number_of_cases), data=data)
        return parameters_vs_metrics


HEADERS = ['a', 'e', 'sin I', 'n']


REPORT_COLS = ['k', 'p', 'a', 'e', 'i', 'n', 'training set', 'TP', 'TN', 'FP', 'FN',
               'accuracy', 'precision', 'recall']


def _get_grid_search_statistic(positive_object_count: int, case_index: int,
                               learning_set_builder: LearningSetBuilder,
                               searcher: _CustomGridSearch) -> pd.DataFrame:
    targets = learning_set_builder.targets
    features, headers = learning_set_builder.build_features_case(case_index)
    scores = searcher.fit(features, targets)
    table_len = scores.shape[0]
    header_flag_vector = [x in headers for x in HEADERS]
    header_flag_frame = pd.DataFrame([header_flag_vector], index=range(table_len), columns=HEADERS)
    header_flag_frame['training set'] = positive_object_count
    data = pd.concat([header_flag_frame, scores], axis=1)
    return data


def get_optimal_parameters(clf_name: str, librate_list_paths: tuple, catalog: str):
    """
    :param clf_name: name of classifier (available values are name of fields in
    `classifiers` sections)
    :param librate_list: list of paths to files contains a couple of librated asteroids.
    :param catalog: path to catalog of synthetic elements or kepler elements
    """
    filename = '%s_grid_search.csv' % clf_name
    total_data = None  # type: pd.DataFrame
    catalog_reader = build_reader_for_grid(Catalog(catalog), None)
    for librate_list in librate_list_paths:
        learning_set_builder = LearningSetBuilder(librate_list, catalog_reader)
        targets = learning_set_builder.targets
        positive_object_count = targets[targets == 1].shape[0]
        searcher = _CustomGridSearch(clf_name, True)
        for i in range(len(catalog_reader.indices_cases)):
            print(OK, '%s %s' % (librate_list, catalog_reader.indices_cases[i]), ENDC, sep='')
            data = _get_grid_search_statistic(positive_object_count, i,
                                              learning_set_builder, searcher)
            total_data = data if total_data is None else total_data.append(data)

    view_for_total_comparing(total_data, filename, HEADERS, REPORT_COLS)
    print(OK, 'File %s is ready' % filename, ENDC, sep='')

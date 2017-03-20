"""
Module aims to get optimal parameters for different cases of feature set.
"""
from copy import copy
from typing import Iterable
from typing import Tuple
from typing import List
from typing import Dict
from typing import Any
from resonancesml.reader import CatalogReader
from resonancesml.shortcuts import get_classifier_class
from resonancesml.settings import params
from resonancesml.report import view_for_total_comparing
from resonancesml.shortcuts import FAIL, ENDC, OK
from resonancesml.reader import build_reader_for_grid
import numpy as np
import pandas as pd
from resonancesml.builders.learningset import LearningSetBuilder
from resonancesml.reader import Catalog
from resonancesml.searcher import ASearcher
from itertools import product
from .shortcuts import classify_as_dict
# from .shortcuts import Modifier
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from .shortcuts import DataLockAdapter
from asyncio.futures import Future
from imblearn.over_sampling import SMOTE
from typing import TypeVar


class _Oversampler:
    def __init__(self, count: int):
        self._count = count

    def fit_sample(self, x_train, y_train) -> Tuple[np.ndarray, np.ndarray]:
        true_class_mask = np.where(y_train is True)
        true_class_objects = x_train[true_class_mask]
        true_class_targets = y_train[true_class_mask]
        for _ in range(self._count):
            x_train = np.vstack((x_train, true_class_objects))
            y_train = np.hstack((y_train, true_class_targets))
        return x_train, y_train


_Sampler = TypeVar('_Sampler', _Oversampler, SMOTE)


def _get_parameters_cases(of_classifier: str, from_section: str) -> dict:
    try:
        param_grid = params()['grid_search'][from_section][of_classifier]['params']  # type: dict
    except KeyError:
        print(FAIL, 'parameter ["grid_search"]["%s"]["%s"] doesn\'t exist.' %
              (from_section, of_classifier), ENDC, sep='')
        exit(-1)
    return param_grid


def _get_report_params(of_classifier: str, from_section: str) -> dict:
    try:
        grid_search_section = params()['grid_search']
        param_grid = grid_search_section[from_section][of_classifier]['report_params']  # type: dict
    except KeyError:
        print(FAIL, 'parameter ["grid_search"]["%s"]["%s"] doesn\'t exist.' %
              (from_section, of_classifier), ENDC, sep='')
        exit(-1)
    return param_grid


class _CustomGridSearch(ASearcher):

    def __init__(self, clf_name: str, verbose=False):
        """Searches best parameters

        """
        super(_CustomGridSearch, self).__init__(verbose)
        self._classifier_cls = get_classifier_class(clf_name)
        self._suggested_parameters = _get_parameters_cases(clf_name, 'classifiers')
        self._param_names = list(self._suggested_parameters.keys())
        self._named_parameter_combinations = _get_parameter_combinations(self._suggested_parameters)
        self._trainset_modifier = None  # type: _Sampler
        self._locked_data = None  # type: DataLockAdapter

    @property
    def trainset_modifier(self) -> _Sampler:
        return self._trainset_modifier

    @trainset_modifier.setter
    def trainset_modifier(self, method: _Sampler):
        self._trainset_modifier = method

    def _fit(self, classifier_params, features: np.ndarray, targets: np.ndarray) -> dict:
        if self._verbose:
            print(classifier_params)
        cv = self._build_cv(targets.shape[0])
        clf = self._classifier_cls(**classifier_params)

        modifier_method = None
        if self.trainset_modifier is not None:
            trainset_modifier_clone = copy(self.trainset_modifier)
            modifier_method = trainset_modifier_clone.fit_sample

        scores = classify_as_dict(clf, cv, features, targets, modifier_method)
        scores.update(classifier_params)
        return scores

    def _callback(self, future: Future):
        scores = future.result()  # type: dict
        self._locked_data.append(scores)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
        number_of_cases = len(self._named_parameter_combinations)

        self._locked_data = DataLockAdapter()
        with PoolExecutor(max_workers=self._worker_number) as excr:
            for classifier_params in self._named_parameter_combinations:
                future = excr.submit(self._fit, classifier_params, features, targets)
                future.add_done_callback(self._callback)

        parameters_vs_metrics = pd.DataFrame(columns=self._param_names + self._SCORE_NAMES,
                                             index=range(number_of_cases),
                                             data=self._locked_data.data)
        return parameters_vs_metrics


HEADERS = ['a', 'e', 'sin I', 'n']


REPORT_COLS = ['a', 'e', 'i', 'n', 'training set', 'TP', 'TN', 'FP', 'FN',
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


def _smote(x_train, y_train) -> Tuple[np.ndarray, np.ndarray]:
    sm = SMOTE(ratio=0.99, random_state=42, k=31, n_jobs=4)
    x_train, y_train = sm.fit_sample(x_train, y_train)
    return x_train, y_train


def _get_modifier_cls(by_name):
    modifiers = {
        'smote': SMOTE,
        'oversampling': _Oversampler
    }
    return modifiers[by_name]


def sampler_gen(sampler_name) -> Iterable[Tuple[Any, Dict[str, Any]]]:
    modifier_cls = _get_modifier_cls(sampler_name)
    parameter_cases = _get_parameters_cases(sampler_name, 'samplers')
    named_parameter_combinations = _get_parameter_combinations(parameter_cases)
    for parameter_combination in named_parameter_combinations:  # type: dict
        sampler = modifier_cls(**parameter_combination)
        yield sampler, parameter_combination


def _get_parameter_combinations(parameter_sets: dict) -> List[Dict[str, Any]]:
    """
    Generates all possible combinations of parameters based on
    dictionary of arrays labeled by parameter names.

    >>>_parameter_combinations_gen({'param1': [1, 2], 'param2': ["value"]})
    [{'param1': 1, 'param2': "value"}, {'param1': 2, 'param2': "value"}]

    >>>_parameter_combinations_gen({'param1': [1, 2], 'param2': ["value"], 'param3': [0, 0.1, 0.5]})
    [
        {'param1': 1, 'param2': "value", 'param3': 0},
        {'param1': 1, 'param2': "value", 'param3': 0.1},
        {'param1': 1, 'param2': "value", 'param3': 0.5},
        {'param1': 2, 'param2': "value", 'param3': 0},
        {'param1': 2, 'param2': "value", 'param3': 0.1},
        {'param1': 2, 'param2': "value", 'param3': 0.5},
    ]
    """
    parameter_names = list(parameter_sets.keys())
    parameter_combination_gen = product(*[x for x in parameter_sets.values()])
    named_parameter_combinations = []
    for parameter_combination in parameter_combination_gen:
        classifier_params = {x: y for x, y in zip(parameter_names, parameter_combination)}
        named_parameter_combinations.append(classifier_params)
    return named_parameter_combinations


class _OptimalParameterFitter:
    def __init__(self, catalog_reader: CatalogReader):
        self._fit_results = None  # type: pd.DataFrame
        self._catalog_reader = catalog_reader

    @property
    def fit_results(self) -> pd.DataFrame:
        return self._fit_results

    def fit_over_data_cases(self, librate_list: str, learning_set_builder: LearningSetBuilder,
                            search: _CustomGridSearch, sampler_parameters=None) -> pd.DataFrame:
        """
        Makes grid search for optimal parameters using every case of data from catalog.
        """
        targets = learning_set_builder.targets
        positive_object_count = targets[targets == 1].shape[0]

        for i in range(len(self._catalog_reader.indices_cases)):
            print(OK, '%s %s' % (librate_list, self._catalog_reader.indices_cases[i]), ENDC, sep='')
            data = _get_grid_search_statistic(positive_object_count, i,
                                              learning_set_builder, search)
            if sampler_parameters is not None:
                data = data.assign(**sampler_parameters)

            if self._fit_results is None:
                self._fit_results = data
            else:
                self._fit_results = self._fit_results.append(data)

        return self._fit_results


def get_optimal_parameters(clf_name: str, librate_list_paths: tuple, catalog: str,
                           verbose: int = 0, sampler=None):
    """
    :param clf_name: name of classifier (available values are name of fields in
    `classifiers` sections)
    :param librate_list: list of paths to files contains a couple of librated asteroids.
    :param catalog: path to catalog of synthetic elements or kepler elements
    """
    report_params = _get_report_params(clf_name, 'classifiers')
    filename = '%s_grid_search.csv' % clf_name
    total_data = None  # type: pd.DataFrame
    catalog_reader = build_reader_for_grid(Catalog(catalog), None)

    searcher = _OptimalParameterFitter(catalog_reader)

    if sampler:
        sampler_report_parameter_names = _get_report_params(sampler, 'samplers')

    for librate_list in librate_list_paths:
        learning_set_builder = LearningSetBuilder(librate_list, catalog_reader)
        search = _CustomGridSearch(clf_name, verbose > 0)

        if sampler is None:
            searcher.fit_over_data_cases(librate_list, learning_set_builder, search)
        else:
            for sampler, sampler_parameters in sampler_gen(sampler):  # type: _Sampler, dict
                search.trainset_modifier = sampler
                searcher.fit_over_data_cases(librate_list, learning_set_builder, search,
                                             sampler_parameters)

    total_data = searcher.fit_results

    clf_params_for_report = sorted(report_params.values())
    cols = list(clf_params_for_report)

    if sampler:
        cols += list(sampler_report_parameter_names.values())
        report_params.update(sampler_report_parameter_names)

    total_data.rename(columns=report_params, inplace=True)
    view_for_total_comparing(total_data, filename, HEADERS, cols + REPORT_COLS)
    print(OK, 'File %s is ready' % filename, ENDC, sep='')

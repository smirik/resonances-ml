"""
Module builders contains methods and classes for building train and test data.
"""
from resonancesml.loader import get_catalog_dataset
from .parameters import DatasetParameters
from resonancesml.shortcuts import ProgressBar
import numpy as np
from typing import Tuple
from typing import List
from typing import Dict


INTEGERS_COUNT = 3
INTEGERS_START_INDEX = -5
AXIS_OFFSET_INDEX = -4
RESONANCE_VIEW_INDEX = -3


class EmptyFeatures(Exception):
    pass


class DatasetBuilder:
    def __init__(self, parameters: DatasetParameters, train_length: int, data_len: int,
                 filter_noise: bool, add_art_objects: bool, verbose: int):
        self._parameters = parameters
        self._train_length = train_length
        self._data_len = data_len
        self._filter_noise = filter_noise
        self._add_art_objects = add_art_objects
        self._verbose = verbose
        self._learnset = None  # type: np.ndarray
        self._testset = None  # type: np.ndarray

    @property
    def learnset(self) -> np.ndarray:
        return self._learnset

    @property
    def testset(self) -> np.ndarray:
        return self._testset

    def _build_datasets(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns dataset for learning and testing.
        :param parameters: parameters for testing classificators.
        :param length: length of learnset.
        :param data_len: it length of data from catalog.
        """
        self._parameters.injection.set_data_len(self._data_len)
        try:
            learnset, testset = _get_feature_matricies(self._parameters, self._train_length)
        except EmptyFeatures:
            print('\033[91mThere is no object\033[0m')
            exit(-1)
        learnset = _serialize_integers(learnset)
        testset = _serialize_integers(testset)

        additional_features = _get_librations_ratios(learnset, self._verbose > 0)
        learnset = _update_feature_matrix(learnset, additional_features, self._verbose > 0)
        testset = _update_feature_matrix(testset, additional_features, self._verbose > 0)

        if self._filter_noise:
            learnset = _filter_noises(learnset, additional_features, self._verbose > 1)
        return learnset, testset


    def build(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._learnset, self._testset = self._build_datasets()
        resonance_view = self._learnset[0][-3]

        indices = self._parameters.indices_cases[0]
        X_train, X_test, Y_train, Y_test = separate_dataset(indices, self._learnset, self._testset)

        if self._add_art_objects:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(ratio=0.99, random_state=42)
            X_train, Y_train = sm.fit_sample(X_train, Y_train)
            resonance_axis = self._parameters.injection.get_resonance_axis(resonance_view)
            X_train[:, -1] = np.power(X_train[:, 0] - resonance_axis, 2)
        return X_train, X_test, Y_train, Y_test


def separate_dataset(indices: List[int], learnset: np.ndarray, testset: np.ndarray)\
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = learnset[:, indices]
    X_test = testset[:, indices]
    Y_train = learnset[:, -1].astype(int)
    Y_test = testset[:, -1].astype(int)
    return X_train, X_test, Y_train, Y_test


def _serialize_integers(dataset: np.ndarray) -> np.ndarray:
    integers_matrix = dataset[:, INTEGERS_START_INDEX:INTEGERS_START_INDEX + INTEGERS_COUNT]
    serialized_resonances = np.array(['_'.join(x) for x in integers_matrix.astype(str)])
    dataset = dataset.astype(object)
    dataset = np.hstack((dataset, np.array([serialized_resonances]).T))
    for _ in range(INTEGERS_COUNT):
        dataset = np.delete(dataset, -4, 1)
    dataset[:,[-2,-1]] = dataset[:,[-1,-2]]
    return dataset


class _ResonanceView:
    def __init__(self, libration_count, resonance_count):
        self.libration_count = libration_count
        self.resonance_count = resonance_count

    @property
    def ratio(self):
        return self.libration_count / self.resonance_count


def _get_librations_ratios(dataset: np.ndarray, verbose: bool = False) -> Dict[str, _ResonanceView]:
    resonances = np.unique(dataset[:, -2])
    bar = None
    if verbose:
        bar = ProgressBar(resonances.shape[0], 80, 'Getting additional features')
    #resonance_librations_counter = {x: 0 for x in resonances}
    resonance_librations_ratio = {x: 0 for x in resonances}
    remains_librations_count = 0
    remains_resonances_count = 0

    for resonance in resonances:
        if bar:
            bar.update()
        resonance_condition = dataset[:, -2] == resonance
        librated_indieces = np.where(resonance_condition & (dataset[:, -1] == 1))
        libration_asteroid_count = dataset[librated_indieces].shape[0]
        resonance_asteroid_count = dataset[np.where(resonance_condition)].shape[0]
        #resonance_librations_counter[resonance] += libration_asteroid_count
        resonance_librations_ratio[resonance] = _ResonanceView(
            libration_asteroid_count, resonance_asteroid_count)

        if libration_asteroid_count < 100:
            remains_librations_count = libration_asteroid_count
            remains_resonances_count = resonance_asteroid_count

    resonance_librations_ratio['other'] = _ResonanceView(
        remains_librations_count, remains_resonances_count)
    return resonance_librations_ratio


def _get_feature_matricies(parameters: DatasetParameters, learn_slice_len: int)\
        -> Tuple[np.ndarray, np.ndarray]:
    catalog_features = get_catalog_dataset(parameters).values
    if parameters.injection:
        catalog_features = parameters.injection.update_data(catalog_features)
        if not catalog_features.shape[0]:
            raise EmptyFeatures()

    learn_feature_set = catalog_features[:learn_slice_len]  # type: np.ndarray
    test_feature_set = catalog_features[learn_slice_len:]  # type: np.ndarray
    return learn_feature_set, test_feature_set


def _update_feature_matrix(of_X: np.ndarray, by_libration_counters: Dict[str, _ResonanceView],
                           verbose: bool = False) -> np.ndarray:
    bar = None
    if verbose:
        N = len(by_libration_counters) - 1
        bar = ProgressBar(N, 80, 'Update feature matrix')
    #all_librations = sum([y for x, y in by_libration_counters.items()])

    resonance_view_vector = np.zeros((of_X.shape[0]), dtype=float)
    for resonance, resonance_view in by_libration_counters.items():
        if bar:
            bar.update()
        resonance_indieces = np.where(of_X[:, -2] == resonance)
        #if resonance_view.libration_count < 100:
            #resonance_view_vector[resonance_indieces] = by_libration_counters['other'].ratio
        #else:
        if resonance_view.resonance_count == 0:
            continue
        resonance_view_vector[resonance_indieces] = resonance_view.ratio

    #for i, features in enumerate(of_X):
        #resonance = features[-2]
        #libration_count = by_libration_counters[resonance]
        #resonance_view_vector[i] = libration_count / all_librations

    of_X = np.hstack((of_X, np.array([resonance_view_vector]).T))
    of_X[:,[-2,-1]] = of_X[:,[-1,-2]]
    return of_X


def _filter_noises(dataset: np.ndarray, libration_views: Dict[str, float],
                   verbose: bool = False) -> np.ndarray:
    """
    Filters objects from false class if axis margin in range of axis margins of
    objects from true class.
    """
    filtered_dataset = None
    max_axis_offsets = {x: 0. for x in libration_views.keys()}

    for key in max_axis_offsets.keys():
        if key == 'other':
            continue
        current_resonance = dataset[:, RESONANCE_VIEW_INDEX] == key
        resonance_dataset = dataset[np.where(current_resonance)]
        is_target_true = resonance_dataset[:, -1] == 1
        is_target_false = resonance_dataset[:, -1] == 0

        max_diff = np.max(resonance_dataset[np.where(is_target_true)][:, AXIS_OFFSET_INDEX])
        suitable_objs = resonance_dataset[np.where(
            ((is_target_false & (resonance_dataset[:, AXIS_OFFSET_INDEX] > max_diff)) | is_target_true)
        )]

        if verbose:
            print("%s: %s -> %s" % (key, dataset[np.where(current_resonance)].shape[0],
                                    suitable_objs.shape[0]))

        if filtered_dataset is None:
            filtered_dataset = suitable_objs
        else:
            filtered_dataset = np.vstack((filtered_dataset, suitable_objs))

    return np.array(filtered_dataset)


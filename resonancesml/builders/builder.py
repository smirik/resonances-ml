"""
Module builders contains methods and classes for building train and test data.
"""
from resonancesml.output import save_asteroids
import os
from os import remove
from os.path import exists as opexist
from os.path import join as opjoin
from resonancesml.shortcuts import get_target_vector
from resonancesml.shortcuts import WARN
from resonancesml.shortcuts import ENDC
from imblearn.over_sampling import SMOTE
from resonancesml.shortcuts import ProgressBar
import numpy as np
from typing import Iterable
from typing import Tuple
from typing import List
from typing import Dict


INTEGERS_COUNT = 3
INTEGERS_START_INDEX = -5
AXIS_OFFSET_INDEX = -4
RESONANCE_VIEW_INDEX = -3


class DatasetBuilder(object):
    def __init__(self, dataset: np.ndarray, train_length: int, data_len: int,
                 filter_noise: bool, add_art_objects: bool, verbose: int):
        self._dataset = dataset
        self._train_length = train_length
        self._data_len = data_len
        self._filter_noise = filter_noise
        self._add_art_objects = add_art_objects
        self._verbose = verbose
        self._learnset = None  # type: np.ndarray
        self._testset = None  # type: np.ndarray
        self._resonance_axes = None  # type: Dict[str, float]

    @property
    def learnset(self) -> np.ndarray:
        return self._learnset

    def _build_datasets(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns dataset for learning and testing.

        :param parameters: parameters for testing classifiers.
        :param length: length of learning dataset.
        :param data_len: it length of data from catalog.
        """
        learnset, testset = self._divide(self._dataset)
        learnset = _serialize_integers(learnset)
        testset = _serialize_integers(testset)

        additional_features = _get_librations_ratios(learnset, self._verbose > 0)
        learnset = _update_feature_matrix(learnset, additional_features, self._verbose > 0)
        testset = _update_feature_matrix(testset, additional_features, self._verbose > 0)

        if self._filter_noise:
            learnset = _filter_noises(learnset, additional_features, self._verbose > 1)
        return learnset, testset

    def _divide(self, dataset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns learning dataset and test dataset."""
        assert self._train_length
        if self._train_length > dataset.shape[0]:
            print(WARN, 'Pointed length (%s) for learning set by key "-n" ' % self._train_length +
                  'is greater than length of whole dataset (%s)' % dataset.shape[0],
                  ENDC, sep='')
        learn_feature_set = dataset[:self._train_length]  # type: np.ndarray
        test_feature_set = dataset[self._train_length:]  # type: np.ndarray
        return learn_feature_set, test_feature_set

    def set_resonaces_axes(self, value: Dict[str, float]):
        self._resonaces_axes = value

    def build(self, column_indices: List[int])\
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert column_indices
        self._learnset, self._testset = self._build_datasets()

        X_train, Y_train = separate_dataset(column_indices, self._learnset)
        X_test, Y_test = separate_dataset(column_indices, self._testset)

        if self._add_art_objects:
            assert self._resonance_axes
            sm = SMOTE(ratio=0.99, random_state=42)
            X_train, Y_train = sm.fit_sample(X_train, Y_train)

            for i in range(X_train):
                concatenation_integers = self._learnset[i][-3]
                resonance_axis = self._resonance_axes[concatenation_integers]
                X_train[i][-1] = np.power(X_train[i][0] - resonance_axis, 2)
        return X_train, X_test, Y_train, Y_test


class GetDatasetBuilder(DatasetBuilder):
    def _divide(self, dataset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns learning dataset and test dataset. But if length of training
        set is None, training set will get asteroids from 1 to last librated
        asteroid."""
        if self._train_length is None:
            target_vector = dataset[:, -1]
            self._train_length = np.where(target_vector == target_vector.max())[0][-1]
        return super(GetDatasetBuilder, self)._divide(dataset)

    def build(self, column_indices: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_train, X_test, Y_train, Y_test = super(GetDatasetBuilder, self).build(column_indices)
        return X_train, X_test, Y_train

    def save_librated_asteroids(self, classes: np.ndarray, to_folder: str) -> np.ndarray:
        """
        Gets numbers of asteroids belong positive class from test dataset and
        saves them to pointed folder.
        """
        save_asteroids(self._testset[:, 0][classes == 1], os.path.basename(to_folder))


class CheckDatasetBuilder(DatasetBuilder):
    @property
    def testset(self) -> np.ndarray:
        return self._testset


def separate_dataset(indices: List[int], dataset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Selects columns from dataset for building features and selects last column as target vector.
    """
    features = dataset[:, indices]
    targets = dataset[:, -1].astype(int, copy=False)
    return features, targets


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


def _progress(items: Iterable, items_size: int, title: str):
    bar = ProgressBar(items_size, 80, title)
    for item in items:
        bar.update()
        yield item


def _get_librations_ratios(dataset: np.ndarray, verbose: bool = False) -> Dict[str, _ResonanceView]:
    resonances = np.unique(dataset[:, -2])
    resonance_librations_ratio = {x: 0 for x in resonances}
    if verbose:
        resonances = _progress(resonances, resonances.shape[0], 'Getting additional features')

    for resonance in resonances:
        resonance_condition = dataset[:, -2] == resonance
        librated_indices = np.where(resonance_condition & (dataset[:, -1] == 1))
        libration_asteroid_count = dataset[librated_indices].shape[0]
        resonance_asteroid_count = dataset[np.where(resonance_condition)].shape[0]
        resonance_librations_ratio[resonance] = _ResonanceView(
            libration_asteroid_count, resonance_asteroid_count)

    return resonance_librations_ratio


def _update_feature_matrix(of_dataset: np.ndarray, by_libration_counters: Dict[str, _ResonanceView],
                           verbose: bool = False) -> np.ndarray:
    """
    Adds ratios of librations and asteroids in resonances to feature matrix.
    Moves target vector to right. Skips zero ratios.
    """
    integers_index = -2
    target_index = -1
    resonance_view_vector = np.zeros((of_dataset.shape[0]), dtype=float)
    items = by_libration_counters.items()
    if verbose:
        items = _progress(items, len(items) - 1, 'Update feature matrix')

    for resonance, resonance_view in items:
        resonance_indices = np.where(of_dataset[:, integers_index] == resonance)
        if resonance_view.resonance_count == 0:
            continue
        resonance_view_vector[resonance_indices] = resonance_view.ratio

    of_dataset = np.hstack((of_dataset, np.array([resonance_view_vector]).T))
    of_dataset[:,[integers_index, target_index]] = of_dataset[:,[target_index, integers_index]]
    return of_dataset


def _filter_noises(dataset: np.ndarray, libration_views: Dict[str, float],
                   verbose: bool = False) -> np.ndarray:
    """
    Filters objects from false class if axis margin in range of axis margins of
    objects from true class.
    """
    filtered_dataset = None
    max_axis_offsets = {x: 0. for x in libration_views.keys()}

    for key in max_axis_offsets.keys():
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


class TargetVectorBuilder:
    """
    TargetVectorBuilder adds integers satisfying D'Alambert of every resonance
    that suitable for asteroid by semi major axis and adds target vector. If
    several resonances are suitable for one resonance, vector of features will
    be duplicated.
    """
    def __init__(self, filepath: str, axis_index: int, librations_folder: str, clear_cache: bool):
        """
        :param filepath: path to resonance table.
        :param axis_index: index of column in catalog.
        :param librations_folder: folder contains files for every resonance from resonance table.
        ;param clear_cache: clears cache.
        """
        self._resonances = np.loadtxt(filepath, dtype='float64')
        self._axis_index = axis_index
        self._librations_folder = librations_folder
        self._clear_cache = clear_cache
        self._data_len = None
        self._INTEGERS_LEN = 3
        self._MU = 0.01720209895
        self._resonances_axes = {}

    @property
    def axis_index(self):
        return self._axis_index

    def set_data_len(self, value):
        self._data_len = value

    @property
    def resonance_axes(self) -> Dict[str, float]:
        """
        Returns dictionary where keys is resonance_view ()
        """
        if not self._resonances_axes:
            for resonance in self._resonances:  # type: np.ndarray
                axis = resonance[6]
                self._resonances_axes['_'.join([str(x) for x in resonance[:3]])] = axis
        return self._resonances_axes

    def _get_axis_diffs(self, for_feature_matrix: np.ndarray, for_resonant_axis: float)\
            -> np.ndarray:
        axis_diffs = for_feature_matrix[:, self._axis_index] - for_resonant_axis
        axis_diffs = np.power(axis_diffs, 2)
        axis_diffs = np.array([axis_diffs]).T
        return axis_diffs

    def _get_resonance_dataset(self, for_feature_matrix: np.ndarray,
                               for_resonance: np.ndarray) -> np.ndarray:
        """
        _get_resonance_dataset prepares dataset for one resonance.
        It makes:
            1) Adds mean motion vector.
            2) 3 features contains integers, satisfying D'Alambert rule.
            3) Feature vector contains squares of difference between resonance.
            semi-major axis and asteroid semi-major axis.
            4) Target vector and adds it to right of dataset.
        """

        N = for_feature_matrix.shape[0]
        integers = np.tile(for_resonance[:self._INTEGERS_LEN], (N, 1))
        mean_motion_vec = self._MU / (for_feature_matrix[:, self._axis_index] ** 3)
        mean_motion_vec = np.sqrt(mean_motion_vec.astype(float, copy=False))
        resonant_axis = for_resonance[-1:]
        axis_diffs = self._get_axis_diffs(for_feature_matrix, resonant_axis)

        dataset = np.hstack((
            for_feature_matrix,
            np.array([mean_motion_vec]).T,
            integers,
            axis_diffs,
        ))
        return dataset

    def update_data(self, X: np.ndarray) -> np.ndarray:
        X[:, 0] = X[:, 0].astype(int, copy=False)
        cache_filepath = '/tmp/cache.txt'
        if self._clear_cache:
            try:
                remove(cache_filepath)
            except Exception:
                pass
        if opexist(cache_filepath):
            print('Dataset has been loaded from cache')
            res = np.loadtxt(cache_filepath)
            return res[:self._data_len]

        print('\n')
        bar = ProgressBar(self._resonances.shape[0], 80, 'Building dataset')
        res = np.zeros((1, X.shape[1] + 6))
        for resonance in self._resonances:  # type: np.ndarray
            bar.update()
            axis = resonance[6]

            mask = _get_mask(axis, X[:, self._axis_index])
            feature_matrix = X[mask]

            if not feature_matrix.shape[0]:
                continue

            filename = 'JUPITER-SATURN_%s' % '_'.join([
                str(int(x)) for x in resonance[:self._INTEGERS_LEN]])
            librated_asteroid_filepath = opjoin(self._librations_folder, filename)
            if not opexist(librated_asteroid_filepath):
                continue

            librating_asteroid_vector = np.loadtxt(librated_asteroid_filepath, dtype=int)
            if not librating_asteroid_vector.shape or librating_asteroid_vector.shape[0] < 50:
                continue

            dataset = self._get_resonance_dataset(feature_matrix, resonance)
            Y = get_target_vector(librating_asteroid_vector, feature_matrix.astype(int))
            dataset = np.hstack((dataset, np.array([Y]).T))
            res = np.vstack((res, dataset))

        res = np.delete(res, 0, 0)
        sorted_res = res[res[:,0].argsort()]
        #np.savetxt(cache_filepath, sorted_res,
                   #fmt='%d %f %f %f %f %.18e %.18e %.18e %.18e %d %f %d %d %d %f %d')
        return sorted_res[:self._data_len]


_cached_mask = {'axis': None, 'mask': None}


def _get_mask(by_axis: float, vector: np.ndarray) -> np.ndarray:
    global _cached_mask
    if _cached_mask['axis'] != by_axis:
        axis_bottom = by_axis - 0.01
        axis_top = by_axis + 0.01
        mask = np.where((vector >= axis_bottom) & (vector <= axis_top))
        _cached_mask['mask'] = mask
        _cached_mask['axis'] = by_axis
    return _cached_mask['mask']

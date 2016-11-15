"""
Module contains methods for filtering noises.
"""
import numpy as np
from resonancesml.knezevic import KnezevicElems


def euclidean_filter(dataset: np.ndarray, is_target_true, verbose: bool = False) -> np.ndarray:
    length_vector = np.linalg.norm(dataset, axis=1)
    return _filter(length_vector, is_target_true, verbose)


def knezevic_filter(elems: KnezevicElems, is_target_true) -> np.ndarray:
    length_vector = elems - KnezevicElems(0, 0, 0, 0)
    return _filter(length_vector, is_target_true, False)


def _filter(length_vector: np.ndarray, is_target_true: np.ndarray, verbose: bool) -> np.ndarray:
    librated_asteroids = length_vector[np.where(is_target_true)]
    max_len = np.max(librated_asteroids)
    min_len = np.min(librated_asteroids)

    if verbose:
        print('total min: %f' % min(length_vector))
        print('total max: %f' % max(length_vector))
        print('librated min: %f' % min_len)
        print('librated max: %f' % max_len)

    filter_cond = (length_vector > max_len) | (length_vector < min_len)
    return filter_cond

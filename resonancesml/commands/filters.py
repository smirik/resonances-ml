"""
Module filters contains methods for filtering noises.
"""
import numpy as np
from .knezevic import KnezevicElems
from .knezevic import knezevic


def euclidean_filter(dataset: np.ndarray, is_target_true, verbose: bool = False):
    axis = 2
    #ecc = 3
    #inc = 4
    symthetic_mean_motion = 5
    computed_mean_motion = -5

    subset = dataset[:, [
        axis,
        #ecc,
        #inc,
        symthetic_mean_motion,
        computed_mean_motion,
        #libration_resonance_ratio,
    ]].astype(float)

    length_vector = np.linalg.norm(subset, axis=1)
    librated_asteroids = length_vector[np.where(is_target_true)]
    max_len = np.max(librated_asteroids)
    min_len = np.min(librated_asteroids)

    #if verbose:
        #print('total min: %f' % min(length_vector))
        #print('total max: %f' % max(length_vector))
        #print('librated min: %f' % min_len)
        #print('librated max: %f' % max_len)

    filter_cond = (length_vector > max_len) | (length_vector < min_len)
    return filter_cond


def knezevic_filter(dataset: np.ndarray, is_target_true):
    axis = 2
    ecc = 3
    inc = 4
    #symthetic_mean_motion = 5
    computed_mean_motion = -5

    elems = KnezevicElems(
        dataset[:, axis].astype(float),
        dataset[:, ecc].astype(float),
        dataset[:, inc].astype(float),
        dataset[:, computed_mean_motion].astype(float),
    )
    zero_elems = KnezevicElems(0, 0, 0, 0)
    distance_vector = knezevic(elems, zero_elems)
    librated_asteroids = distance_vector[np.where(is_target_true)]
    max_knezevic = np.max(librated_asteroids)
    min_knezevic = np.min(librated_asteroids)

    filter_cond = (distance_vector > max_knezevic) | (distance_vector < min_knezevic)
    return filter_cond

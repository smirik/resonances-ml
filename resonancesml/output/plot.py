import numpy as np
from resonancesml.knezevic import KnezevicElems
import os
import matplotlib.pyplot as plt
from itertools import combinations


def plot(feature_matrix: np.ndarray, target_vector: np.ndarray, folder: str,
         plot_title: str = None):
    """
    plot takes feature set and target vector, divides feature set onto two
    subsets on basis of classes from target_vector, marks vectors of semi-major
    axes, eccentricities, sin(I) and mean motions by names in container, adds
    to this container Knezevic metric distances.

    When container with reorganized features is ready, plot takes enumeration
    of pairs of feature vectors without repetitions.
    """
    true_class_mask = np.where(target_vector == 1)
    false_class_mask = np.where(target_vector != 1)
    true_class = feature_matrix[true_class_mask]
    false_class = feature_matrix[false_class_mask]

    ecc1 = true_class[:, 1]
    ecc2 = false_class[:, 1]

    sini1 = true_class[:, 2]
    sini2 = false_class[:, 2]

    mean1 = true_class[:, -2]
    mean2 = false_class[:, -2]

    zero = KnezevicElems(0, 0, 0, 0)
    knez1 = KnezevicElems(true_class[:, 0], ecc1, sini1, mean1) - zero
    knez2 = KnezevicElems(false_class[:, 0], ecc2, sini2, mean2) - zero

    data = {
        'axis': [true_class[:, 0], false_class[:, 0]],
        'eccentricity': [ecc1, ecc2],
        'sin_inclination': [sini1, sini2],
        'knezevic': [knez1, knez2],
        'axis-diff-square': [true_class[:, -1], false_class[:, -1]],
    }

    data_keys = [x for x in sorted(data.keys())]
    for plot_number, key_pair in enumerate(combinations(data_keys, 2)):
        x_key, y_key = key_pair
        plt.figure(plot_number, figsize=(15, 15))
        plt.plot(
            data[x_key][0], data[y_key][0], 'bo',
            data[x_key][1], data[y_key][1], 'r^',
        )
        plt.xlabel(x_key)
        plt.ylabel(y_key)
        filename = '%s_%s.png' % (x_key, y_key)
        path = os.path.join(os.curdir, folder)
        if not os.path.exists(path):
            os.mkdir(path)
        if plot_title:
            plt.title(plot_title)
        plt.savefig(os.path.join(path, filename), bbox_inches='tight')


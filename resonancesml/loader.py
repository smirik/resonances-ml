import numpy as np
from resonancesml.commands.parameters import DatasetParameters
import pandas
from pandas import DataFrame
from resonancesml.shortcuts import FAIL, ENDC


def get_asteroids(from_filename: str, possible_asteroids: np.ndarray) -> np.ndarray:
    """
    get_asteroids loads asteroid's numbers from pointed file. These numbers
    will be filtered by possible_asteroids.

    :param from_filename: file contains asteroid numbers.
    :param possible_asteroids: 1 dimensional numpy array of integers.
    :return: 1 dimensional numpy array of integers.
    """
    librated_asteroids = np.loadtxt(from_filename, dtype=int)
    mask = np.in1d(librated_asteroids, possible_asteroids)
    return librated_asteroids[mask]


def get_catalog_dataset(parameters: DatasetParameters) -> DataFrame:
    dtype = {0: str}
    dtype.update({x: float for x in range(1, parameters.catalog_width)})
    catalog_features = pandas.read_csv(  # type: DataFrame
        parameters.catalog_path, delim_whitespace=True,
        skiprows=parameters.skiprows, header=None, dtype=dtype)
    if parameters.dataset_end:
        catalog_features = catalog_features[:parameters.dataset_end]
    return catalog_features


def get_learn_set(from_catalog_features: np.ndarray, to_asteroid_number: str) -> np.ndarray:
    asteroids_numbers = from_catalog_features[:, 0]
    data = np.where(asteroids_numbers == to_asteroid_number)
    if data[0].shape[0] == 0:
        print('%sAsteroid %s not in filtered catalog. Probably this asteroid has unsuitable axis %s' %
              (FAIL, to_asteroid_number, ENDC))

        asteroids_numbers = from_catalog_features[:, 0].astype(int)
        less_suggested_asteroid = asteroids_numbers[np.where(asteroids_numbers < int(to_asteroid_number))][-1]
        greater_suggested_asteroid = asteroids_numbers[np.where(asteroids_numbers > int(to_asteroid_number))][0]
        print('Try to use -n %i or -n %i' % (less_suggested_asteroid, greater_suggested_asteroid))
        exit(-1)
    slice_len = data[0][0] + 1
    return from_catalog_features[:slice_len]

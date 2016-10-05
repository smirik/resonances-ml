import numpy as np
from resonancesml.commands.parameters import TesterParameters
import pandas
from pandas import DataFrame


def get_asteroids(from_filename: str, possible_asteroids: np.ndarray) -> np.ndarray:
    """
    get_asteroids loads asteroid numbers from pointed file.

    :param from_filename: file contains asteroid numbers.
    :param possible_asteroids: 1 dimensional numpy array of integers.
    :return: 1 dimensional numpy array of integers.
    """
    librated_asteroids = np.loadtxt(from_filename, dtype=int)
    mask = np.in1d(librated_asteroids, possible_asteroids)
    return librated_asteroids[mask]


def get_catalog_dataset(parameters: TesterParameters) -> DataFrame:
    dtype = {0:str}
    dtype.update({x: float for x in range(1, parameters.catalog_width)})
    catalog_features = pandas.read_csv(  # type: DataFrame
        parameters.catalog_path, delim_whitespace=True,
        skiprows=parameters.skiprows, header=None, dtype=dtype)
    if parameters.dataset_end:
        catalog_features = catalog_features[:parameters.dataset_end]
    return catalog_features


def get_learn_set(from_catalog_features: np.ndarray, to_asteroid_number: str) -> np.ndarray:
    slice_len = np.where(from_catalog_features[:, 0] == to_asteroid_number)[0][0] + 1
    return from_catalog_features[:slice_len]

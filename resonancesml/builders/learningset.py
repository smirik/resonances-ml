import numpy as np
from resonancesml.reader import CatalogReader
from resonancesml.loader import get_asteroids
from resonancesml.loader import get_learn_set
from resonancesml.shortcuts import get_target_vector
from resonancesml.shortcuts import get_feature_matrix
from typing import Tuple
from typing import List


LearningSetCase = Tuple[np.ndarray, List[str]]


class LearningSetBuilder:
    """
    Class for building learning set contains features and targets. It takes
    file contains librated asteroids and catalog contains elements of asteroids.

    :param librate_list: path to file contains librated asteroids.
    :param librate_list: catalog type.
    """
    def __init__(self, librate_list: str, catalog_reader: CatalogReader):
        self._catalog_reader = catalog_reader
        catalog_features = self._catalog_reader.read().values
        self._librated_asteroids = get_asteroids(librate_list, catalog_features[:, 0].astype(int))
        self._learn_feature_set = get_learn_set(catalog_features,  # type: np.ndarray
                                                str(self._librated_asteroids[-1]))
        self._targets = get_target_vector(self._librated_asteroids,
                                          self._learn_feature_set.astype(int))

    def build_features_case(self, indices_case: int = 0) -> LearningSetCase:
        indices = self._catalog_reader.indices_cases[indices_case]
        headers = self._catalog_reader.get_headers(indices)
        features = get_feature_matrix(self._learn_feature_set, False, indices)
        return features, headers

    @property
    def targets(self) -> np.ndarray:
        return self._targets

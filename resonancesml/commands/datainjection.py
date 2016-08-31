import numpy as np
from abc import abstractclassmethod
from typing import List
from .reader import CatalogReader


class ADatasetInjection:
    def __init__(self, headers: List[str]):
        self.headers = headers

    @abstractclassmethod
    def update_data(self, X: np.ndarray) -> np.ndarray:
        pass


def _add_mean_motion(X: np.ndarray) -> np.ndarray:
    axises = np.array(X[:,2], dtype='float64')
    mean_motions = np.sqrt([0.0002959122082855911025 / axises ** 3.])
    X = np.hstack((X, mean_motions.T))
    return X


class MeanMotionInjection(ADatasetInjection):
    def update_data(self, X: np.ndarray) -> np.ndarray:
        return _add_mean_motion(X[:,:-1])


class KeplerInjection(ADatasetInjection):
    def __init__(self, headers: List[str], catalog_path: str, catalog_width: int, skiprows: int):
        super(KeplerInjection, self).__init__(headers)
        self._kepler_catalog_reader = CatalogReader(catalog_path, catalog_width, skiprows)

    def update_data(self, X: np.ndarray) -> np.ndarray:
        import ipdb
        ipdb.set_trace()
        kepler_feature_matrix = self._kepler_catalog_reader.get_feuture_matrix(X.shape[0])
        kepler_feature_matrix = _add_mean_motion(kepler_feature_matrix)[:,[2,3,4,11]]
        X = np.hstack((X, kepler_feature_matrix))
        return X

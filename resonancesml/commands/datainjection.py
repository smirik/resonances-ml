import numpy as np
from abc import abstractclassmethod
from typing import List


class ADatasetInjection:
    def __init__(self, headers: List[str]):
        self.headers = headers

    @abstractclassmethod
    def update_data(self, X: np.ndarray) -> np.ndarray:
        pass


class KeplerInjection(ADatasetInjection):
    def update_data(self, X: np.ndarray) -> np.ndarray:
        axises = np.array(X[:,2], dtype='float64')
        mean_motions = np.sqrt([0.0002959122082855911025 / axises ** 3.])
        X = np.hstack((X, mean_motions.T))
        return X



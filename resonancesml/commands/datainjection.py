import numpy as np
from abc import abstractclassmethod
from typing import List


class ADatasetInjection(object):
    def __init__(self, headers: List[str]):
        super(ADatasetInjection, self).__init__()
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


class _DataFilter(ADatasetInjection):
    def __init__(self, headers: List[str], resonant_axis: float, axis_swing: float, axis_index: int):
        super(_DataFilter, self).__init__(headers)
        self.axis_swing = axis_swing
        self.axis_index = axis_index
        self.resonant_axis = resonant_axis

    def update_data(self, X: np.ndarray) -> np.ndarray:
        X = X[np.where(np.abs(X[:, self.axis_index] - self.resonant_axis) <= self.axis_swing)]
        integers = np.array([[5, -2, -2]] * X.shape[0])
        X = np.hstack((X, integers))
        return X


class ClearDecorator(_DataFilter):
    def __init__(self, decorating: ADatasetInjection, resonant_axis,
                 axis_swing: float, axis_index: int):
        super(ClearDecorator, self).__init__(decorating.headers, resonant_axis,
                                             axis_swing, axis_index)
        self._decorating = decorating

    def update_data(self, X: np.ndarray) -> np.ndarray:
        X = super(ClearDecorator, self).update_data(X)
        return self._decorating.update_data(X)


class ClearInjection(ADatasetInjection):
    def __init__(self, resonant_axis: float, axis_swing: float, axis_index: int):
        super(ClearInjection, self).__init__([], resonant_axis, axis_swing, axis_index)


class ClearKeplerInjection(KeplerInjection):
    def __init__(self, headers: List[str], resonant_axis, axis_swing: float, axis_index: int):
        super(ClearKeplerInjection, self).__init__(headers, resonant_axis, axis_swing, axis_index)

    def update_data(self, X: np.ndarray) -> np.ndarray:
        X = super(ClearKeplerInjection, self).update_data(X)
        return X

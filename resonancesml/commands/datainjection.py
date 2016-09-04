import numpy as np
from abc import abstractclassmethod
from typing import List
from os.path import join as opjoin
from os.path import exists as opexist
from os import remove
from resonancesml.shortcuts import get_target_vector
from resonancesml.shortcuts import ProgressBar


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


class IntegersInjection(ADatasetInjection):
    """
    InegersInjection adds integers satisfying D'Alambert of every resonance
    that suitable for asteroid by semi major axis. If seeveral resonances are
    suitable for one resonance, vector of features will be duplicated.
    """
    def __init__(self, headers: List[str], filepath: str, axis_index: int, librations_folder: str,
                 clear_cache: bool):
        super(IntegersInjection, self).__init__(headers)
        self._resonances = np.loadtxt(filepath, dtype='float64')
        self._axis_index = axis_index
        self._librations_folder = librations_folder
        self._clear_cache = clear_cache

    def update_data(self, X: np.ndarray) -> np.ndarray:
        cache_filepath = '/tmp/cache.txt'
        if self._clear_cache:
            try:
                remove(cache_filepath)
            except Exception:
                pass
        if opexist(cache_filepath):
            print('dataset loaded from cache')
            res = np.loadtxt(cache_filepath)
            return res

        print('\n')
        bar = ProgressBar(self._resonances.shape[0], 'Building dataset', 4)
        res = np.zeros((1, X.shape[1] + 4))
        integers_len = 3
        for resonance in self._resonances:
            bar.update()
            axis = resonance[6]
            feature_matrix = X[np.where(np.abs(X[:, self._axis_index] - axis) <= 0.01)]
            if not feature_matrix.shape[0]:
                continue
            N = feature_matrix.shape[0]
            integers = np.repeat(resonance[:integers_len], N).reshape(integers_len, N).transpose()
            feature_matrix = np.hstack((feature_matrix, integers))

            filename = 'JUPITER-SATURN_%s' % '_'.join([str(int(x)) for x in resonance[:integers_len]])
            librated_asteroid_filepath = opjoin(self._librations_folder, filename)
            if opexist(librated_asteroid_filepath):
                librated_asteroid_vector = np.loadtxt(librated_asteroid_filepath, dtype=int)
                Y = get_target_vector(librated_asteroid_vector, feature_matrix.astype(int))
            else:
                Y = np.zeros(feature_matrix.shape[0])
            dataset = np.hstack((feature_matrix, np.array([Y]).T))

            res = np.vstack((res, dataset))
        res = np.delete(res, 0, 0)
        res = res.astype(float)
        sorted_res = res[res[:,0].argsort()]
        np.savetxt(cache_filepath, sorted_res,
                   fmt='%s %f %f %f %f %.18e %.18e %.18e %.18e %.18e %d %d %d %d')
        return res

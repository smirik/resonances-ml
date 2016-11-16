import sys
import numpy as np
from typing import List
from sklearn.preprocessing import StandardScaler


FAIL = '\033[91m'
ENDC = '\033[0m'
OK = '\033[92m'


class ProgressBar:
    def __init__(self, total: int, width: int, title=''):
        sys.stdout.write("%s [%s]" % (title, " " * width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (width + 1))
        self._counter = 0
        self._total = total
        self._width = width
        self._last_progress = 0

    def update(self):
        self._counter += 1
        progress = round(self._counter / self._total * self._width)
        delta = progress - self._last_progress
        if delta >= 1:
            self._last_progress += delta
            sys.stdout.write('#' * delta)
            sys.stdout.flush()

    def fin(self):
        sys.stdout.write("\n")
        self._counter = 0
        self._delta = 0

    def __del__(self):
        self.fin()


def get_target_vector(from_asteroids: np.ndarray, by_features: np.ndarray) -> np.ndarray:
    target_vector = []
    for i, asteroid_number in enumerate(by_features[:, 0]):
        target_vector.append(asteroid_number in from_asteroids)
    return np.array(target_vector, dtype=np.float64)


def get_feuture_matrix(from_features: np.ndarray, scale: bool, indices: List[int]) -> np.ndarray:
    res = np.array(from_features[: ,indices], dtype=np.float64)  # type: np.ndarray
    if scale:
        scaler = StandardScaler()
        res = scaler.fit_transform(res)
    return res

def norm(vector: np.ndarray) -> np.ndarray:
    return (vector - np.min(vector))/(np.max(vector) - np.min(vector))

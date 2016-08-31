import sys
import numpy as np
from typing import List
from sklearn.preprocessing import StandardScaler

class ProgressBar:
    def __init__(self, width, title='', divider=2):
        self._divider = divider
        toolbar_width = width // divider
        sys.stdout.write("%s [%s]" % (title, " " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width + 1))
        self._counter = 0

    def update(self):
        self._counter += 1
        if self._counter % self._divider == 0:
            sys.stdout.write("#")
            sys.stdout.flush()

    def fin(self):
        sys.stdout.write("\n")
        self._counter = 0

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

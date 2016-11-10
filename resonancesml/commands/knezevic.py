"""
Module knezevic contains knezevic metric and related data types.
"""
import numpy as np
from typing import Union


AXIS = 0
ECC = 1
SINI = 2
MEAN_MOTION = 3


class KnezevicElems:
    def __init__(self, axis: Union[float, np.ndarray], eccentricity: Union[float, np.ndarray],
                 inclination: Union[float, np.ndarray], mean_motion: Union[float, np.ndarray]):
        self.axis = axis
        self.eccentricity = eccentricity
        self.inclination = inclination
        self.mean_motion = mean_motion


def knezevic(asteroid1: KnezevicElems, asteroid2: KnezevicElems) -> Union[float, np.ndarray]:
    n_s = np.mean([asteroid1.mean_motion, asteroid2.mean_motion])
    a_s = np.mean([asteroid1.axis, asteroid2.axis])

    first = (5/4) * (((asteroid1.axis - asteroid2.axis) / a_s) ** 3)
    second = 2 * ((asteroid1.eccentricity - asteroid2.eccentricity) ** 2 )
    third = 2 * ((asteroid1.inclination - asteroid2.inclination) ** 2)
    #first = (1/4) * (((asteroid1.axis - asteroid2.axis) / a_s) ** 2)
    #second = (3/4) * ((asteroid1.eccentricity - asteroid2.eccentricity) ** 2 )
    #third = 4 * ((asteroid1.inclination - asteroid2.inclination) ** 2)
    distance = n_s * a_s * np.sqrt(np.sum([first, second, third], axis=0).astype(float))
    return distance


def knezevic_metric(x: np.ndarray, y: np.ndarray) -> float:
    if (x == y).all():
        return 0

    distance = knezevic(
        KnezevicElems(x[AXIS], x[ECC], x[SINI], x[MEAN_MOTION]),
        KnezevicElems(y[AXIS], y[ECC], y[SINI], y[MEAN_MOTION])
    )
    return distance

"""
Modules contains tools for saving output data in text format, image format etc.
"""
from .plot import plot
import numpy as np
import os
from os.path import join as opjoin
from os.path import exists as opexists

OUTFOLDER = 'resonanceml-out'


def save_asteroids(data: np.array, by_key):
    if not opexists(OUTFOLDER):
        os.mkdir(OUTFOLDER)
    np.savetxt(opjoin(OUTFOLDER, by_key), data, '%d')

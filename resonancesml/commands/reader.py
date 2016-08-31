import pandas
import numpy as np
from pandas import DataFrame


class CatalogReader:
    def __init__(self, catalog_path: str, catalog_width: int, skiprows: int):
        dtype = {0:str}
        dtype.update({x: float for x in range(1, catalog_width)})

        self._catalog_feautures = pandas.read_csv(  # type: DataFrame
            catalog_path, delim_whitespace=True,
            skiprows=skiprows, header=None, dtype=dtype)

    def get_feuture_matrix(self, length: int) -> np.ndarray:
        res = self._catalog_feautures.values[:length]  # type: np.ndarray
        return res



from resonancesml.settings import SYN_CATALOG_PATH
from resonancesml.settings import CAT_CATALOG_PATH
from resonancesml.settings import PRO_CATALOG_PATH
from resonancesml.settings import params
import pandas
from pandas import DataFrame

from typing import List

from enum import Enum
from enum import unique

from resonancesml.datainjection import ADatasetInjection
from resonancesml.datainjection import KeplerInjection


@unique
class Catalog(Enum):
    syn = 'syn'
    cat = 'cat'
    pro = 'pro'

    @property
    def axis_index(self) -> int:
        return {
            'syn': 2,
            'cat': 2,
            'pro': 1,
        }[self.value]


class CatalogReader:
    """
    CatalogReader reads data from source and building dataset by pointed parameters.
    """
    def __init__(self, indices_cases: List[List[int]], catalog_path: str, catalog_width: int,
                 delimiter: str, skiprows: int, injection: ADatasetInjection = None,
                 dataset_end: int = None):
        """
        :param indices_cases: matrix of indices that will be used for pointing
        data from dataset when it will be modifyied by injection.
        :param catalog_path: path to catalog contains asteroid numbers and Kepler elements.
        :param catalog_width: number of columns in catalog.
        :param delimiter: delimiter between headers in the catalog.
        :param skiprows: number of rows that should be skiped.
        :param injection: injection intended for modifying the dataset.
        :param dataset_end: number of last row. It is necessary if catalog
        should be loaded particularly.
        """
        self.indices_cases = indices_cases
        self.catalog_path = catalog_path
        self.catalog_width = catalog_width
        self.delimiter = delimiter
        self.skiprows = skiprows
        self.injection = injection
        if dataset_end is not None:
            assert dataset_end >= skiprows
            self.dataset_end = dataset_end - skiprows
        else:
            self.dataset_end = None

    def read(self) -> DataFrame:
        dtype = {0: str}
        dtype.update({x: float for x in range(1, self.catalog_width)})
        catalog_features = pandas.read_csv(  # type: DataFrame
            self.catalog_path, delim_whitespace=True,
            skiprows=self.skiprows, header=None, dtype=dtype)
        if self.dataset_end:
            catalog_features = catalog_features[:self.dataset_end]
        return catalog_features


class CatalogException(Exception):
    def __init__(self, message = None):
        if not message:
            message = 'Unsupported catalog type'
        super(Exception, self).__init__(message)


def get_injection(by_catalog: Catalog) -> ADatasetInjection:
    if by_catalog == by_catalog.syn:
        return None
    elif by_catalog == by_catalog.cat:
        return KeplerInjection(['n'])
    raise CatalogException()


def build_reader(for_catalog: Catalog, injection: ADatasetInjection,
                 indices: List[List[int]] = None) -> CatalogReader:
    if for_catalog == Catalog.syn:
        return CatalogReader([[2,3,4,5],[2,3,5]] if not indices else indices,
                             SYN_CATALOG_PATH, 10, '  ', 2, injection, 406253)
    elif for_catalog == Catalog.cat:
        return CatalogReader([[2, 3, 10], [2, 3, 4, 10]] if not indices else indices,
                             CAT_CATALOG_PATH, 8, "\.|,", 6, injection)
    elif for_catalog == Catalog.pro:
        return CatalogReader([[1, 2], [1, 2, 3]] if not indices else indices,
                             PRO_CATALOG_PATH, 6, ";", 3, injection)
    raise CatalogException()


def build_reader_for_influence(catalog: Catalog, injection: ADatasetInjection) -> CatalogReader:
    if catalog == Catalog.syn:
        indeces_cases = params()['influence']['synthetic']
        return CatalogReader(indeces_cases, SYN_CATALOG_PATH, 10, '  ', 2, injection, 406253)
    elif catalog == Catalog.cat:
        indeces_cases = params()['influence']['synthetic']
        return CatalogReader(indeces_cases, CAT_CATALOG_PATH, 8, "\.|,", 6, injection)
    raise CatalogException()


def build_reader_for_grid(catalog: Catalog, injection: ADatasetInjection) -> CatalogReader:
    if catalog == Catalog.syn:
        indeces_cases = params()['grid_search']['fields']['synthetic']
        return CatalogReader(indeces_cases, SYN_CATALOG_PATH, 10, '  ', 2, injection, 406253)
    elif catalog == Catalog.cat:
        indeces_cases = params()['grid_search']['fields']['synthetic']
        return CatalogReader(indeces_cases, CAT_CATALOG_PATH, 8, "\.|,", 6, injection)
    raise CatalogException()

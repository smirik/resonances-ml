from resonancesml.settings import SYN_CATALOG_PATH
from resonancesml.settings import CAT_CATALOG_PATH

from typing import List

from enum import Enum
from enum import unique

from .datainjection import ADatasetInjection
from .datainjection import KeplerInjection


@unique
class Catalog(Enum):
    syn = 'syn'
    cat = 'cat'


class TesterParameters:
    def __init__(self, indices_cases: List[List[int]], catalog_path: str, catalog_width: int,
                 delimiter: str, skiprows: int, injection: ADatasetInjection = None, dataset_end: int = None):
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


def get_injection(by_catalog: Catalog) -> ADatasetInjection:
    injection =  {
        by_catalog.syn: None,
        by_catalog.cat: KeplerInjection(['n'])
    }[by_catalog]
    return injection


def get_learn_parameters(catalog: Catalog, injection: ADatasetInjection,
                               indices: List[List[int]] = None) -> TesterParameters:
    return {
        catalog.syn: TesterParameters(
            [[2,3,4,5],[2,3,5]] if not indices else indices,
            SYN_CATALOG_PATH, 10, '  ', 2, injection, 406253),
        catalog.cat: TesterParameters(
            [[2, 3, 10], [2, 3, 4, 10]] if not indices else indices,
            CAT_CATALOG_PATH, 8, "\.|,", 6, injection),
    }[catalog]


def get_compare_parameters(catalog: Catalog, injection: ADatasetInjection) -> TesterParameters:
    return {
        Catalog.syn: TesterParameters([[2,3,4],[2,3,5],[2,4,5],[3,4,5],[2,5]],
                                      SYN_CATALOG_PATH, 10, '  ', 2, injection, 406253),
        Catalog.cat: TesterParameters([[2,3,4],[2,3,8],[2,4,8],[3,4,8],[2,8]],
                                      CAT_CATALOG_PATH, 8, "\.|,", 6, injection),
    }[catalog]


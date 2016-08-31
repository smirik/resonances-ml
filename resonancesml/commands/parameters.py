from resonancesml.settings import SYN_CATALOG_PATH
from resonancesml.settings import CAT_CATALOG_PATH

from typing import List

from enum import Enum
from enum import unique

from .datainjector import ADatasetInjector
from .datainjector import KeplerInjector


@unique
class Catalog(Enum):
    syn = 'syn'
    cat = 'cat'


class TesterParameters:
    def __init__(self, indices_cases: List[List[int]], catalog_path: str,
                 catalog_width: int, delimiter: str, skiprows: int, injector: ADatasetInjector = None):
        self.indices_cases = indices_cases
        self.catalog_path = catalog_path
        self.catalog_width = catalog_width
        self.delimiter = delimiter
        self.skiprows = skiprows
        self.injector = injector


def get_learn_parameters(catalog: Catalog) -> TesterParameters:
    return {
        Catalog.syn: TesterParameters([[2,3,4],[2,3,4,5]], SYN_CATALOG_PATH, 10, '  ', 2),
        Catalog.cat: TesterParameters([[2,3,8], [2,3,4,8]], CAT_CATALOG_PATH, 8, ', ', 6, KeplerInjector(['n'])),
    }[catalog]


def get_classify_all_parameters(catalog: Catalog) -> TesterParameters:
    return {
        Catalog.syn: TesterParameters([[2,3,4,5]], SYN_CATALOG_PATH, 10, '  ', 2),
        Catalog.cat: TesterParameters([[2,3,4,8], [2,3,8]], CAT_CATALOG_PATH,
                                      8, ', ', 6, KeplerInjector(['n'])),
    }[catalog]


def get_compare_parameters(catalog: Catalog) -> TesterParameters:
    return {
        Catalog.syn: TesterParameters([[2,3,4],[2,3,5],[2,4,5],[3,4,5],[2,5]],
                                      SYN_CATALOG_PATH, 10, '  ', 2),
        Catalog.cat: TesterParameters([[2,3,4],[2,3,8],[2,4,8],[3,4,8],[2,8]],
                                      CAT_CATALOG_PATH, 8, ', ', 6, KeplerInjector(['n'])),
    }[catalog]


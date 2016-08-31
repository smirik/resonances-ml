from resonancesml.settings import SYN_CATALOG_PATH
from resonancesml.settings import CAT_CATALOG_PATH

from typing import List

from enum import Enum
from enum import unique

from .datainjection import ADatasetInjection
from .datainjection import MeanMotionInjection
from .datainjection import KeplerInjection


@unique
class Catalog(Enum):
    syn = 'syn'
    cat = 'cat'
    syncat = 'syncat'


class TesterParameters:
    def __init__(self, indices_cases: List[List[int]], catalog_path: str,
                 catalog_width: int, delimiter: str, skiprows: int,
                 injection: ADatasetInjection = None):
        self.indices_cases = indices_cases
        self.catalog_path = catalog_path
        self.catalog_width = catalog_width
        self.delimiter = delimiter
        self.skiprows = skiprows
        self.injection = injection


def get_learn_parameters(catalog: Catalog) -> TesterParameters:
    return {
        Catalog.syn: TesterParameters([[2,3,4],[2,3,4,5]], SYN_CATALOG_PATH, 10, '  ', 2),
        Catalog.cat: TesterParameters([[2,3,10], [2,3,4,10]], CAT_CATALOG_PATH,
                                      11, ",|\.", 6, MeanMotionInjection(['n'])),
        Catalog.syncat: TesterParameters(
            [
                [2, 3, 4, 10, 11, 12, 13], [2, 3, 4, 5, 10, 11, 12, 13],
                [2, 3, 4, 10, 11, 12], [2, 3, 4, 5, 10, 11, 12]
            ],
            SYN_CATALOG_PATH, 10, '  ', 2,
            KeplerInjection(['k_a', 'k_e', 'k_i', 'k_n'], CAT_CATALOG_PATH, 8, 6)),
    }[catalog]


def get_classify_all_parameters(catalog: Catalog) -> TesterParameters:
    return {
        Catalog.syn: TesterParameters([[2,3,4,5]], SYN_CATALOG_PATH, 10, '  ', 2),
        Catalog.cat: TesterParameters([[2,3,4,10], [2,3,10]], CAT_CATALOG_PATH,
                                      11, ",|\.", 6, MeanMotionInjection(['n'])),
        Catalog.syncat: TesterParameters(
            [
                [2, 3, 4, 5, 10, 11, 12, 13],
                [2, 3, 5, 10, 11, 12, 13]
            ],
            SYN_CATALOG_PATH, 10, '  ', 2,
            KeplerInjection(['k_a', 'k_e', 'k_i', 'k_n'], CAT_CATALOG_PATH, 8, 6)),
    }[catalog]


def get_compare_parameters(catalog: Catalog) -> TesterParameters:
    return {
        Catalog.syn: TesterParameters([[2,3,4],[2,3,5],[2,4,5],[3,4,5],[2,5]],
                                      SYN_CATALOG_PATH, 10, '  ', 2),
        Catalog.cat: TesterParameters([[2,3,4],[2,3,10],[2,4,10],[3,4,10],[2,10]],
                                      CAT_CATALOG_PATH, 8, ",|\.", 6, MeanMotionInjection(['n'])),
        Catalog.syncat: TesterParameters(
            [
                [2, 3, 4, 10, 11, 12, 13],
                [2, 3, 5, 10, 11, 12, 13],
                [2, 4, 5, 10, 11, 12, 13],
                [3, 4, 5, 10, 11, 12, 13],
                [2, 5, 10, 11, 12, 13],

                #[2, 3, 4, 10, 11, 12],
                #[2, 3, 10, 10, 11, 12],
                #[2, 4, 10, 10, 11, 12],
                #[3, 4, 10, 10, 11, 12],
                #[2, 10, 10, 11, 12]
            ],
            SYN_CATALOG_PATH, 10, '  ', 2,
            KeplerInjection(['k_a', 'k_e', 'k_i', 'k_n'], CAT_CATALOG_PATH, 8, 6)),
    }[catalog]


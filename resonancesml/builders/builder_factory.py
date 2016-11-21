from resonancesml.shortcuts import FAIL, ENDC
from resonancesml.reader import Catalog
from resonancesml.reader import build_reader
from resonancesml.reader import CatalogReader
from .builder import CheckDatasetBuilder
from .builder import TargetVectorBuilder
from .builder import GetDatasetBuilder
from .builder import DatasetBuilder
from abc import abstractmethod
import numpy as np

from typing import Iterable
from typing import Tuple


class ABuilderKit:
    def __init__(self, train_length: int, data_length: int, filter_noise: bool,
                 add_art_objects: bool, verbose: int):
        self._train_length = train_length
        self._data_length = data_length
        self._filter_noise = filter_noise
        self._add_art_objects = add_art_objects
        self._verbose = verbose

    @abstractmethod
    def create(self, dataset: np.ndarray) -> DatasetBuilder:
        pass


class CheckDatasetBuilderKit(ABuilderKit):
    def create(self, dataset: np.ndarray) -> DatasetBuilder:
        return CheckDatasetBuilder(dataset, self._train_length, self._data_length,
                                   self._filter_noise, self._add_art_objects, self._verbose)


class GetDatasetBuilderKit(ABuilderKit):
    def create(self, dataset: np.ndarray) -> DatasetBuilder:
        return GetDatasetBuilder(dataset, self._train_length, self._data_length,
                                 self._filter_noise, self._add_art_objects, self._verbose)


def builder_gen(builder_kit: ABuilderKit, matrix_path: str, librations_folders: tuple,
                remove_cache: bool, catalog: str, fields: list = None)\
        -> Iterable[Tuple[str, DatasetBuilder, CatalogReader]]:
    catalog = Catalog(catalog)
    if fields is None:
        fields = [x for x in range(catalog.axis_index, catalog.axis_index + 3)] + [-5, -4]
    for folder in librations_folders:
        features_builder = TargetVectorBuilder(matrix_path, catalog.axis_index, folder,
                                               remove_cache)
        catalog_reader = build_reader(catalog, None, [fields])
        catalog_data = catalog_reader.read().values
        dataset = features_builder.update_data(catalog_data)
        if not dataset.shape[0]:
            print('%sThere is no object%s' % (FAIL, ENDC))
            exit(-1)

        builder = builder_kit.create(dataset)
        builder.set_resonaces_axes(features_builder.resonance_axes)
        yield folder, builder, catalog_reader


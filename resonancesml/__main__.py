#!/usr/bin/env python
import click
from resonancesml.reader import Catalog
from typing import Dict
from typing import List
from typing import Tuple
from resonancesml.settings import PROJECT_DIR
from resonancesml.settings import set_config_path
from os.path import join as opjoin
from resonancesml.shortcuts import ClfPreset
from resonancesml.shortcuts import get_classifier
from resonancesml.datainjection import ADatasetInjection


_CATALOG_HELP = (
    'Type of catalog: ' +
    '\'syn\' is catalog of synthetic elements, ' +
    '\'cat\' is catalog of Kepler elements, ' +
    '\'pro\' is catalog of property elements, '
)

_TEST_CLF_HELP = ('Classifier preset. Example: "KNN 0". ' +
                 'Presets are available in config, section "classifiers". ' +
                 'Make python -m resonancesml dump_config to see default configuration.')


_CHOOSE_CLF_HELP = (
    'Compares several classifier presets ' +
    'from section "classifiers_for_comparing" in config ' +
    '(see `python -m resonancesml dump-config`). ' +
    'Scores of classificataion are got using cross validation.'
)


class ClassifierPreset(click.ParamType):
    name = 'classifier preset'

    def convert(self, value: str, param, ctx):
        vals = value.split()
        assert len(vals) == 2
        return vals[0], int(vals[1])


def _unite_decorators(*decorators):
    def deco(decorated_function):
        for dec in reversed(decorators):
            decorated_function = dec(decorated_function)
        return decorated_function

    return deco


DEFAULT_LIBRATION_LIST = opjoin(PROJECT_DIR, 'input', 'librations',
                                'first50_librated_asteroids_4_-2_-1')


def _learn_options():
    return _unite_decorators(
        click.option('--librate-list', '-l', type=click.Path(exists=True, resolve_path=True),
                      default=DEFAULT_LIBRATION_LIST, show_default=True),
        click.option('--catalog', '-c', type=click.Choice([x.name for x in Catalog]),
                     help=_CATALOG_HELP, show_default=True),
    )


def _classify_all_opts():
    return _unite_decorators(
        click.option('--all-librated', '-a', type=click.Path(exists=True, resolve_path=True),
                      help='File with librated asteroids used for checking.'),
        click.option('--clf', 'clf_presets', type=ClassifierPreset(),
                     help=_TEST_CLF_HELP, multiple=True),
        click.option('--verbose', '-v', type=int, count=True),
    )


def _clear_opts():
    return _unite_decorators(
        click.option('--resonant-axis', '-x', type=float),
        click.option('--axis-swing', '-s', type=float),
        click.option('--axis-index', '-i', type=int),
    )


def _influence_options():
    return _unite_decorators(
        _learn_options(),
        click.option('--clf', type=ClassifierPreset())
    )


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True, resolve_path=True),
              help='Path to custom config file')
def main(config):
    set_config_path(config)


def _get_classifiers():
    from resonancesml.shortcuts import get_classifier_class
    from resonancesml.settings import params

    clfs_params = params()['classifiers_for_comparing']  # type: List[dict]
    keys = []
    classifiers = {}
    for item in clfs_params:
        report_name = item['report_name']
        clf_name = item['preset']['name']
        clf_params = item['preset']['params']
        keys.append(report_name)
        classifiers[report_name] = get_classifier_class(clf_name)(**clf_params)

    return classifiers, keys


@main.command(help=_CHOOSE_CLF_HELP, name='choose-clf')
@_learn_options()
def choose_clf(librate_list: str, catalog: str):
    from resonancesml.commands.learn import MethodComparer
    from resonancesml.reader import build_reader
    from resonancesml.reader import get_injection
    _catalog = Catalog(catalog)
    injection = get_injection(_catalog)
    catalog_reader = build_reader(_catalog, injection, None)
    tester = MethodComparer(librate_list, catalog_reader)
    tester.set_methods(*(_get_classifiers()))
    tester.learn()


def _get_injection_and_catalog(catalog: str, resonant_axis: float, axis_swing: float,
                               axis_index: int) -> Tuple[ADatasetInjection, Catalog]:
    from resonancesml.reader import get_injection
    from resonancesml.datainjection import ClearDecorator
    from resonancesml.datainjection import ClearInjection

    _catalog = Catalog(catalog)
    injection = get_injection(_catalog)
    args = (resonant_axis, axis_swing, axis_index)

    if injection:
        clear_injection = ClearDecorator(injection, *args)
    else:
        clear_injection = ClearInjection(*args)
    return clear_injection, _catalog


_CLEAR_HELP_TEMPLATE = 'Does same as %s, but filters asteroids by axis.'


@main.command(name='clear-choose-clf', help=_CLEAR_HELP_TEMPLATE % 'choose-clf')
@_learn_options()
@_clear_opts()
@click.argument('fields', nargs=-1)
def clear_choose_clf(librate_list: str, catalog: str, fields: tuple,
                resonant_axis: float, axis_swing: float, axis_index: int):
    from resonancesml.commands.learn import MethodComparer
    from resonancesml.reader import build_reader
    injection, _catalog = _get_injection_and_catalog(catalog, resonant_axis, axis_swing, axis_index)
    fields = [[int(x) for x in fields]] if fields else None
    parameters = build_reader(_catalog, injection, fields)
    tester = MethodComparer(librate_list, parameters)
    tester.set_methods(*(_get_classifiers()))
    tester.learn()



_CLASSIFY_ALL = (
    'Test classifier by all asteroids from catalog using pointed classifier preset. ' +
    'Classifier will be fitted on asteroids from file pointed by `--librated-list`. ' +
    'Scores of classificataion are got comparing predictions and real values from ' +
    'file pointed by option `--all-librated`. ' +
    'Example: classify-all -l first200_librated_asteroids_4_-2_-1.csv ' +
    '-a sorted_all-librated.txt -c syn --clf="KNN 0" 2 3 4 5.'
)


@main.command(name='classify-all', help=_CLASSIFY_ALL)
@_learn_options()
@_classify_all_opts()
@click.argument('fields', nargs=-1)
def classify_all(librate_list: str, all_librated: str, catalog: str,
                 fields: tuple, clf_presets: Tuple[ClfPreset, ...], verbose):
    from resonancesml.commands.classify import classify_all as _classify_all
    from resonancesml.reader import build_reader
    from resonancesml.reader import get_injection
    _catalog = Catalog(catalog)
    injection = get_injection(_catalog)
    fields = [[int(x) for x in fields]] if fields else None
    catalog_reader = build_reader(_catalog, injection, fields)
    _classify_all(librate_list, all_librated, catalog_reader, clf_presets, verbose)


@main.command(name='clear-classify-all', help=_CLEAR_HELP_TEMPLATE % 'classify-all')
@_learn_options()
@_classify_all_opts()
@_clear_opts()
@click.argument('fields', nargs=-1)
def clear_classify_all(all_librated: str, catalog: str, fields: tuple, resonant_axis,
                       axis_swing, axis_index, verbose: bool, clf_presets: Tuple[ClfPreset, ...],
                       librate_list: str):
    assert fields
    from resonancesml.commands.classify import classify_all as _classify_all
    from resonancesml.reader import build_reader
    injection, _catalog = _get_injection_and_catalog(catalog, resonant_axis, axis_swing, axis_index)
    fields = [[int(x) for x in fields]] if fields else None
    catalog_reader = build_reader(_catalog, injection, fields)
    _classify_all(librate_list, all_librated, catalog_reader, clf_presets, verbose)


@main.command(name='influence-fields', help='Get influnce of fields for pointed classifier.')
@_influence_options()
def influence_fields(librate_list: str, catalog: str, clf: ClfPreset):
    from resonancesml.commands.learn import MethodComparer
    from resonancesml.reader import build_reader_for_influence

    catalog_reader = build_reader_for_influence(Catalog(catalog), None)
    classifiers = { clf[0]: get_classifier(clf) }
    tester = MethodComparer(librate_list, catalog_reader)
    tester.set_methods(classifiers, [clf[0]])
    tester.learn()


@main.command(name='clear-influence-fields', help=_CLEAR_HELP_TEMPLATE % 'influence-fields')
@_influence_options()
@_clear_opts()
def clear_influence_fields(librate_list: str, catalog: str, clf: ClfPreset,
                           resonant_axis, axis_swing, axis_index):
    from resonancesml.commands.learn import MethodComparer
    from resonancesml.reader import build_reader_for_influence
    injection, _catalog = _get_injection_and_catalog(catalog, resonant_axis, axis_swing, axis_index)
    catalog_reader = build_reader_for_influence(Catalog(catalog), injection)
    classifiers = { clf[0]: get_classifier(clf) }
    tester = MethodComparer(librate_list, catalog_reader)
    tester.set_methods(classifiers, [clf[0]])
    tester.learn()


PRO_AXIS = 1
PRO_ECC = 2
PRO_I = 3
PRO_MEAN_MOTION = -5

SYN_MAG = 1
SYN_AXIS = 2
SYN_ECC = 3
SYN_I = 4
SYN_MEAN_MOTION = 5
SYN_G = 6
SYN_S = 7

FILTER_NOISE_HELP = ('Removes non-resonance asteroids have axis lesser ' +
                     'than resonance asteroids from learning set.')

def data_options(helps: Dict[str, str] = {}):
    return _unite_decorators(
        click.option('--train-length', '-n', type=int, help=helps.get('tl', '')),
        click.option('--data-length', '-l', type=int,
                     help='Length of dataset that will be read from catalog.'),
        click.option('--filter-noise', '-i', type=bool, is_flag=True, help=FILTER_NOISE_HELP),
        click.option('--add-art-objects', '-a', type=bool, is_flag=True,
                     help='Adds synthetic objects based on SMOTE.'),
        click.option('--matrix-path', '-p', type=click.Path(resolve_path=True, exists=True),
                     default=opjoin(PROJECT_DIR, 'input', 'resonance_tables', 'matrix-js.res'),
                     show_default=True),
        click.option('--librations-folder', '-f', type=click.Path(resolve_path=True, exists=True),
                     multiple=True, default=[opjoin(PROJECT_DIR, 'input', 'librations', '4J-2S-1')],
                    show_default=True),
        click.option('--catalog', '-c', type=click.Choice([x.name for x in Catalog]), default='syn',
                    help=_CATALOG_HELP, show_default=True),
        click.option('--remove-cache', '-r', type=bool, is_flag=True,
                     help='This commands build cache in /tmp/cache.txt. Up this flag to delete cache.'),
        click.option('--verbose', '-v', count=True),
    )


@main.command(help='Builds plots over main features of dataset.')
@data_options()
def plot(train_length: int, data_length: None, matrix_path: str, librations_folder: tuple,
         remove_cache: bool, catalog: str, verbose: int, filter_noise: bool, add_art_objects: bool):
    from resonancesml.output import plot
    from resonancesml.builders import builder_gen
    from resonancesml.builders import CheckDatasetBuilderKit
    builder_kit = CheckDatasetBuilderKit(train_length, data_length, filter_noise,
                                         add_art_objects, verbose)
    builders = builder_gen(builder_kit, matrix_path, librations_folder, remove_cache, catalog)
    for folder, builder, catalog_reader in builders:
        X_train, X_test, Y_train, Y_test = builder.build(catalog_reader.indices_cases[0])
        plot(X_train, Y_train, folder.split('/')[-1])


@main.command('test-clf', help='Tests pointed classifer for common metrics.')
@data_options()
@click.option('--clf', type=ClassifierPreset(), help=_TEST_CLF_HELP)
@click.option('--fields', '-e', type=lambda x: [int(y) for y in x.split()])
def test_clf(train_length: int, data_length: None, matrix_path: str, librations_folder: tuple,
             remove_cache: bool, catalog: str, verbose: int, filter_noise: bool,
             add_art_objects: bool, fields: List[int], clf: ClfPreset):
    from resonancesml.commands.classify import test_classifier as _test_classifier
    from resonancesml.shortcuts import ENDC, OK
    from resonancesml.builders import builder_gen
    from resonancesml.builders import CheckDatasetBuilderKit
    fields = [int(x) for x in fields] if fields else None
    builder_kit = CheckDatasetBuilderKit(train_length, data_length, filter_noise,
                                         add_art_objects, verbose)
    builders = builder_gen(builder_kit, matrix_path, librations_folder,
                           remove_cache, catalog, fields)

    for folder, builder, catalog_reader in builders:
        print(OK, folder, ENDC, sep='')
        X_train, X_test, Y_train, Y_test = builder.build(catalog_reader.indices_cases[0])
        _test_classifier(X_train, X_test, Y_train, Y_test, catalog_reader.indices_cases[0], clf)


@main.command(help='Creates file ./resonanceml-out/<name-of-folder> with asteroids that librate.')
@data_options({
    'tl': ('Length of training set. If not pointed, application gets asteroids ' +
           'from 1 to number of last librated asteroid.')
})
@click.option('--clf', type=ClassifierPreset(), help=_TEST_CLF_HELP)
@click.option('--fields', '-e', type=lambda x: [int(y) for y in x.split()],
              help='Numbers of columns used for features. Example: \'2 3 4\'')
def get(train_length: int, data_length: None, matrix_path: str, librations_folder: tuple,
        remove_cache: bool, catalog: str, verbose: int, filter_noise: bool,
        add_art_objects: bool, fields: List[int], clf: ClfPreset):
    from resonancesml.commands.classify import get_librated_asteroids
    from resonancesml.shortcuts import ENDC, OK
    from resonancesml.builders import builder_gen
    from resonancesml.builders import GetDatasetBuilderKit

    fields = [int(x) for x in fields] if fields else None
    builder_kit = GetDatasetBuilderKit(train_length, data_length, filter_noise,
                                       add_art_objects, verbose)
    builders = builder_gen(builder_kit, matrix_path, librations_folder,
                           remove_cache, catalog, fields)
    for folder, builder, catalog_reader in builders:
        print(OK, folder, ENDC, sep='')
        X_train, X_test, Y_train = builder.build(catalog_reader.indices_cases[0])
        classes = get_librated_asteroids(X_train, Y_train, X_test, clf)
        builder.save_librated_asteroids(classes, folder)


@main.command('dump-config', help='Shows current configuration in yaml format.')
def dump_config():
    from resonancesml.settings import params
    print(params().dump())


_GET_OPTIMAL_PARAMETERS_HELP = (
    'Makes grid search over parameters from "grid_search" configuration section ' +
    'for pointed classifier. Supported classifiers are in section "classifiers". ' +
    'If you want make classifier able for grid_search add them short name to ' +
    '"grid_search" configuration section and point parameters for "grid_search" the same as for KNN'
)


_GET_OPTIMAL_PARAMETERS_CLF_HELP = (
    'Classifer shortname. Available classifiers are in section "grid_search".' +
    'This section is extendable by classifers from "classifers" section.'
)


@main.command(help=_GET_OPTIMAL_PARAMETERS_HELP)
@click.option('--clf', type=str, help=_GET_OPTIMAL_PARAMETERS_CLF_HELP)
@click.option('--librate-list', '-l', 'librate_list_paths',
              type=click.Path(exists=True, resolve_path=True),
              default=(DEFAULT_LIBRATION_LIST,), show_default=True, multiple=True,
              help='Path to file with librated asteroids.')
@click.option('--catalog', '-c', type=click.Choice([x.name for x in Catalog]),
              help="Path to catalog of synthetic elements.")
def get_optimal_parameters(clf: str, librate_list_paths: tuple, catalog: str):
    from resonancesml.commands.get_optimal_parameters import get_optimal_parameters
    get_optimal_parameters(clf, librate_list_paths, catalog)


@main.command(help='Makes grid_search over set of coeffitients values that have been made by' +
              ' step 0.1. Features from catalog are multiplied by radical of the related value ' +
              '. Works only for KNN and only for axis, eccentricity, inclination and mean motion' +
              ' from synthetic elements.')
@click.option('--clf', type=ClassifierPreset(), help=_TEST_CLF_HELP)
@click.option('--librate-list', '-l', 'librate_list_paths',
              type=click.Path(exists=True, resolve_path=True),
              default=(DEFAULT_LIBRATION_LIST,), show_default=True,
              help='Path to file with librated asteroids.')
@click.option('--catalog', '-c', type=click.Choice([x.name for x in Catalog]),
              help="Type of catalog")
def get_optimal_coeffs(clf: ClfPreset, librate_list_paths: str, catalog: str):
    from resonancesml.commands.get_optimal_coeffs import get_optimal_coeffs
    get_optimal_coeffs(clf, librate_list_paths, catalog, [2,3,4,5])


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter

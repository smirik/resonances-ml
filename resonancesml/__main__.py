#!/usr/bin/env python
import click
from resonancesml.commands.parameters import Catalog


@click.group()
def main():
    pass


def _get_classifier(by_name):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    return {
        'KNN': KNeighborsClassifier(weights='distance', p=1, n_jobs=4),
        'DT': DecisionTreeClassifier(random_state=241),
        'GB': GradientBoostingClassifier(n_estimators=50, learning_rate=0.85),
    }[by_name]



def _get_classifiers():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    keys = [
        'K neighbors',
        'Gradient boosting (10 trees)',
        'Gradient boosting (50 trees)',
        'Decision tree',
        'Logistic regression',
    ]

    classifiers = {
        'K neighbors': KNeighborsClassifier(weights='distance', p=1, n_jobs=4),
        'Gradient boosting (10 trees)': GradientBoostingClassifier(n_estimators=10, learning_rate=0.85),
            #n_estimators=7, learning_rate=0.6, min_samples_split=33240),
        'Gradient boosting (50 trees)': GradientBoostingClassifier(n_estimators=50, learning_rate=0.85),
        'Decision tree': DecisionTreeClassifier(random_state=241),
        'Logistic regression': LogisticRegression(C=0.00001, penalty='l1', n_jobs=4)
    }
    return classifiers, keys


@main.command()
@click.option('--librate-list', '-l', type=click.Path(exists=True, resolve_path=True))
@click.option('--catalog', '-c', type=click.Choice([x.name for x in Catalog]))
def learn(librate_list: str, catalog: str):
    from resonancesml.commands.learn import MethodComparer
    from resonancesml.commands.parameters import get_learn_parameters
    parameters = get_learn_parameters(Catalog(catalog))
    tester = MethodComparer(librate_list, parameters)
    tester.set_methods(*(_get_classifiers()))
    tester.learn()


def _learn_tester(librate_list, model_name, parameters):
    from resonancesml.commands.learn import MethodComparer
    tester = MethodComparer(librate_list, parameters)
    classifiers = { model_name: _get_classifier(model_name) }
    tester.set_methods(classifiers, [model_name])
    tester.learn()


def _get_injection_and_catalog(catalog: str, resonant_axis: float,
                               axis_swing: float, axis_index: int) -> tuple:
    from resonancesml.commands.parameters import get_injection
    from resonancesml.commands.datainjection import ClearDecorator
    from resonancesml.commands.datainjection import ClearInjection

    _catalog = Catalog(catalog)
    injection = get_injection(_catalog)
    args = (resonant_axis, axis_swing, axis_index)
    injection = ClearDecorator(injection, *args) if injection else ClearInjection(*args)
    return injection, _catalog


@main.command('clear-influence-trainset')
@click.option('--librate-list', '-l', type=click.Path(exists=True, resolve_path=True))
@click.option('--catalog', '-c', type=click.Choice([x.name for x in Catalog]))
@click.option('--model', '-m', type=click.Choice(['KNN', 'DT', 'GB']))
@click.option('--resonant-axis', '-x', type=float)
@click.option('--axis-swing', '-s', type=float)
@click.option('--axis-index', '-i', type=int)
@click.argument('fields', nargs=-1)
def clear_influence_trainset(librate_list: str, catalog: str, model: str, fields: tuple,
                           resonant_axis: float, axis_swing: float, axis_index: int):
    from resonancesml.commands.parameters import get_learn_parameters
    fields = [int(x) for x in fields]
    injection, _catalog = _get_injection_and_catalog(catalog, resonant_axis, axis_swing, axis_index)
    parameters = get_learn_parameters(_catalog, injection, [fields])
    _learn_tester(librate_list, model, parameters)


@main.command(name='clear-learn')
@click.option('--librate-list', '-l', type=click.Path(exists=True, resolve_path=True))
@click.option('--catalog', '-c', type=click.Choice([x.name for x in Catalog]))
@click.option('--resonant-axis', '-x', type=float)
@click.option('--axis-swing', '-s', type=float)
@click.option('--axis-index', '-i', type=int)
@click.argument('fields', nargs=-1)
def clear_learn(librate_list: str, catalog: str, fields: tuple,
                resonant_axis: float, axis_swing: float, axis_index: int):
    from resonancesml.commands.learn import MethodComparer
    from resonancesml.commands.parameters import get_learn_parameters
    injection, _catalog = _get_injection_and_catalog(catalog, resonant_axis, axis_swing, axis_index)
    fields = [[int(x) for x in fields]] if fields else None
    parameters = get_learn_parameters(_catalog, injection, fields)
    tester = MethodComparer(librate_list, parameters)
    tester.set_methods(*(_get_classifiers()))
    tester.learn()


@main.command(name='classify-all')
@click.option('--librate-list', '-l', type=click.Path(exists=True, resolve_path=True))
@click.option('--all-librated', '-a', type=click.Path(exists=True, resolve_path=True))
@click.option('--catalog', '-c', type=click.Choice([x.name for x in Catalog]))
@click.argument('fields', nargs=-1)
def classify_all(librate_list: str, all_librated: str, catalog: str, fields: tuple):
    from resonancesml.commands.classify import classify_all as _classify_all
    from resonancesml.commands.parameters import get_learn_parameters
    from resonancesml.commands.parameters import get_injection
    _catalog = Catalog(catalog)
    injection = get_injection(_catalog)
    fields = [[int(x) for x in fields]] if fields else None
    parameters = get_learn_parameters(_catalog, injection, fields)
    _classify_all(librate_list, all_librated, parameters)


@main.command(name='clear-classify-all')
@click.option('--all-librated', '-a', type=click.Path(exists=True, resolve_path=True))
@click.option('--catalog', '-c', type=click.Choice([x.name for x in Catalog]))
@click.option('--resonant-axis', '-x', type=float)
@click.option('--axis-swing', '-s', type=float)
@click.option('--axis-index', '-i', type=int)
@click.option('--train-length', '-n', type=int)
@click.argument('fields', nargs=-1)
def clear_classify_all(all_librated: str, catalog: str, fields: tuple,
                       resonant_axis, axis_swing, axis_index, train_length):
    assert fields
    from resonancesml.commands.classify import clear_classify_all as _clear_classify_all
    from resonancesml.commands.parameters import get_learn_parameters
    injection, _catalog = _get_injection_and_catalog(catalog, resonant_axis, axis_swing, axis_index)
    fields = [[int(x) for x in fields]] if fields else None
    parameters = get_learn_parameters(_catalog, injection, fields)
    _clear_classify_all(all_librated, parameters, train_length)


@main.command(name='influence-fields')
@click.option('--librate-list', '-l', type=click.Path(exists=True, resolve_path=True))
@click.option('--catalog', '-c', type=click.Choice([x.name for x in Catalog]))
@click.option('--model', '-m', type=click.Choice(['KNN', 'DT', 'GB']))
def influence_fields(librate_list: str, catalog: str, model: str):
    from resonancesml.commands.learn import MethodComparer
    from resonancesml.commands.parameters import get_compare_parameters

    parameters = get_compare_parameters(Catalog(catalog))
    classifiers = { model: _get_classifier(model) }
    tester = MethodComparer(librate_list, parameters)
    tester.set_methods(classifiers, [model])
    tester.learn()


@main.command(name='clear-influence-fields')
@click.option('--librate-list', '-l', type=click.Path(exists=True, resolve_path=True))
@click.option('--catalog', '-c', type=click.Choice([x.name for x in Catalog]))
@click.option('--model', '-m', type=click.Choice(['KNN', 'DT', 'GB']))
@click.option('--resonant-axis', '-x', type=float)
@click.option('--axis-swing', '-s', type=float)
@click.option('--axis-index', '-i', type=int)
def clear_influence_fields(librate_list: str, catalog: str, model: str,
                            resonant_axis, axis_swing, axis_index):
    from resonancesml.commands.learn import MethodComparer
    from resonancesml.commands.datainjection import ClearDecorator
    from resonancesml.commands.parameters import get_compare_parameters
    from resonancesml.commands.parameters import get_injection

    _catalog = Catalog(catalog)
    injection = ClearDecorator(get_injection(_catalog), resonant_axis, axis_swing, axis_index)
    parameters = get_compare_parameters(_catalog, injection)
    classifiers = { model: _get_classifier(model) }
    tester = MethodComparer(librate_list, parameters)
    tester.set_methods(classifiers, [model])
    tester.learn()


if __name__ == '__main__':
    main()

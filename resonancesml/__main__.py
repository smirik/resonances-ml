#!/usr/bin/env python
import click
from resonancesml.commands.parameters import Catalog


@click.group()
def main():
    pass

@main.command()
@click.option('--librate-list', '-l', type=click.Path(exists=True, resolve_path=True))
@click.option('--catalog', '-c', type=click.Choice([x.name for x in Catalog]))
def learn(librate_list: str, catalog: str):
    from resonancesml.commands.learn import MethodComparer
    from resonancesml.commands.parameters import get_learn_parameters
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    parameters = get_learn_parameters(Catalog(catalog))
    classifiers = {
        'Decision tree': DecisionTreeClassifier(random_state=241, max_depth=5),
        'Gradient boosting (10 trees)': GradientBoostingClassifier(n_estimators=10),
        'Gradient boosting (50 trees)': GradientBoostingClassifier(n_estimators=50),
        'K neighbors': KNeighborsClassifier(weights='distance', p=1, n_jobs=4),
        'Logistic regression': LogisticRegression(C=0.00001, penalty='l1', n_jobs=4)
    }
    tester = MethodComparer(librate_list, parameters)
    tester.set_methods(classifiers)

    tester.learn()


@main.command(name='classify-all')
@click.option('--librate-list', '-l', type=click.Path(exists=True, resolve_path=True))
@click.option('--all-librated', '-a', type=click.Path(exists=True, resolve_path=True))
@click.option('--catalog', '-c', type=click.Choice([x.name for x in Catalog]))
def classify_all(librate_list: str, all_librated: str, catalog: str):
    from resonancesml.commands.classify import classify_all as _classify_all
    from resonancesml.commands.parameters import get_classify_all_parameters
    parameters = get_classify_all_parameters(Catalog(catalog))
    _classify_all(librate_list, all_librated, parameters)


@main.command(name='compare-fields-valuable')
@click.option('--librate-list', '-l', type=click.Path(exists=True, resolve_path=True))
@click.option('--catalog', '-c', type=click.Choice([x.name for x in Catalog]))
@click.option('--model', '-m', type=click.Choice(['KNN', 'DT']))
def compare_fields_valuable(librate_list: str, catalog: str, model: str):
    from resonancesml.commands.learn import MethodComparer
    from sklearn.neighbors import KNeighborsClassifier
    from resonancesml.commands.parameters import get_compare_parameters
    from sklearn.tree import DecisionTreeClassifier

    parameters = get_compare_parameters(Catalog(catalog))

    model_obj = {
        'KNN': KNeighborsClassifier(weights='distance', p=1, n_jobs=4),
        'DT': DecisionTreeClassifier(random_state=241, max_depth=5)
    }[model]

    classifiers = { model: model_obj }
    tester = MethodComparer(librate_list, parameters)
    tester.set_methods(classifiers)
    tester.learn()


if __name__ == '__main__':
    main()

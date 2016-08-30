#!/usr/bin/env python
import click
from resonancesml.commands.learn import Catalog


@click.group()
def main():
    pass

@main.command()
@click.option('--librate-list', '-l', type=click.Path(exists=True, resolve_path=True))
@click.option('--catalog', '-c', type=click.Choice([x.name for x in Catalog]))
def learn(librate_list: str, catalog: str):
    from resonancesml.commands.learn import MethodComparer
    from resonancesml.commands.learn import get_tester_parameters
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    parameters = get_tester_parameters(Catalog(catalog))
    classifiers = {
        'Decision tree': DecisionTreeClassifier(random_state=241),
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
def classify_all(librate_list: str, all_librated: str):
    from resonancesml.commands.classify import classify_all as _classify_all
    _classify_all(librate_list, all_librated)


@main.command(name='compare-fields-valuable')
@click.option('--librate-list', '-l', type=click.Path(exists=True, resolve_path=True))
def compare_fields_valuable(librate_list: str, ):
    from resonancesml.commands.learn import MethodComparer
    from resonancesml.commands.learn import TesterParameters
    from resonancesml.settings import SYN_CATALOG_PATH
    from sklearn.neighbors import KNeighborsClassifier

    parameters = TesterParameters([[2,3,4],[2,3,5],[2,4,5],[3,4,5],[2,5]],
                                  SYN_CATALOG_PATH, 10, '  ', 2)
    classifiers = {
        'K neighbors': KNeighborsClassifier(weights='distance', p=1, n_jobs=4),
    }
    tester = MethodComparer(librate_list, parameters)
    tester.set_methods(classifiers)
    tester.learn()


if __name__ == '__main__':
    main()

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
    parameters = get_tester_parameters(Catalog(catalog))
    tester = MethodComparer(librate_list, parameters)
    tester.learn()


@main.command()
@click.option('--librate-list', '-l', type=click.Path(exists=True, resolve_path=True))
@click.option('--all-librated', '-a', type=click.Path(exists=True, resolve_path=True))
def classify(librate_list: str, all_librated: str):
    from resonancesml.commands.classify import classify as _classify
    _classify(librate_list, all_librated)



if __name__ == '__main__':
    main()

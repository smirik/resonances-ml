#!/usr/bin/env python
import click


@click.group()
def main():
    pass


@main.command()
@click.option('--librate-list', '-l', type=click.Path(exists=True, resolve_path=True))
def learn(librate_list: str):
    from resonancesml.commands.learn import learn as _learn
    _learn(librate_list)


@main.command()
@click.option('--librate-list', '-l', type=click.Path(exists=True, resolve_path=True))
@click.option('--all-librated', '-a', type=click.Path(exists=True, resolve_path=True))
def classify(librate_list: str, all_librated: str):
    from resonancesml.commands.classify import classify as _classify
    _classify(librate_list, all_librated)



if __name__ == '__main__':
    main()

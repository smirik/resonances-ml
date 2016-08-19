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



if __name__ == '__main__':
    main()

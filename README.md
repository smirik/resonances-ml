* Master: [![Build Status](https://travis-ci.org/4xxi/resonances-ml.svg?branch=master)](https://travis-ci.org/4xxi/resonances)
* Develop: [![Build Status](https://travis-ci.org/4xxi/resonances-ml.svg?branch=develop)](https://travis-ci.org/4xxi/resonances)

# Resonances ML

## Abstract

The application identifies three-body resonances using machine learning (ML)
techniques. It uses data from AstDys catalog as source of feature
dataset and it uses text file contains resonant asteroids, the file will be used for
forming target vector. The application allows to check desired ML technique for
common metrics and for influence of pointed features.
The application by default has:

* Catalogs.
* Lists of 50, 100, 200 resonant asteroids for Jupiter Saturn resonance with integers 4 -2 -1.
* Lists of 50, 100, 200 resonant asteroids for Jupiter Saturn pure resonance with integers 4 -2 -1.
* List of all resonant asteroids for Jupiter Saturn resonance with integers 4 -2 -1.
* List of all resonant asteroids for Jupiter Saturn pure resonance with integers 4 -2 -1.
* Configuration file with couple of ML techniques suitable for classifying.
  Suitability of them determined by related research.

## Installation

You can install the Resonances ML using Python package manager pip

`pip install git+https://github.com/4xxi/resonances`

## Usage

First of all huge part of input data is in configuration. Configuration data is
stored by YAML format.  You can customize it by pointing your configuration
file.  Do `python -m resonanceml dump-config` to get default configuration. You
can redirect output to your file.  Let's say you want to make configuration
file `my-config.yml`. For this execute `python -m resonanceml dump-config >
my-config.yml` When you get your own configuration file you can customize it by
you favorite text editor and you are able for pointing it for the application.
Make `python -m resonanceml -c  my-config.yml <another command of the
application>` for executing some another command of the application based on
your configuration file.

Also every command has option `--help`. If some of commands or options is not clear checkout `python -m resonancesml --help` or
`python -m resonancesml <some_command> --help`

### Choosing classifier

Use command `python -m resonancesml choose-clf` to compare classifiers' scores.
All scores are computed by cross validation using k folding by 5 parts.
This command uses classifiers from section `classifiers_for_comparing` in
configuration (see `python -m resonancesml dump-config`).
Example: `python -m resonancesml choose-clf -l input/librations/first50_librated_asteroids_4_-2_-1 -c syn`.

### Influence fields

For comparing significance of fields from catalog there is command `python -m
resonancesml influence-fields`. It searches cases of indices combinations from
configuration section `influence`. The section contains two keys: `synthetic`
for catalog of synthetic elements and `orbital` for catalog of orbital
elements.  All scores are computed by cross validation using k folding by 5
parts.
Example: `python -m resonancesml influence-fields -c syn --clf="DT 0" -l input/librations/first50_librated_asteroids_4_-2_-1`

### Classification by all asteroids

Rather than another commands classifiers in the command aren't rated by cross
validation. For rating them another list of asteroids is used. Also this commands makes file contains:

* Numbers of asteroids predicated as librated.
* Numbers of asteroids predicated as librated but they are not librated (false positive objects).
* Numbers of asteroids predicated as not librated but they are librated (false negative objects).

Command `python -m resonancesml classify-all` necessary for:
* Several classifiers
* File contains librated asteroids for learning.
* File contains all librated asteroids for testing.
* Numbers of columns with features.

Example:
```
python -m resonancesml classify-all -l input/librations/first50_librated_asteroids_4_-2_-1 \
    -a input/librations/all_librated_asteroids_4_-2_-1 -c syn --clf="KNN 0" --clf="DT 0" 2 3 4 5
```

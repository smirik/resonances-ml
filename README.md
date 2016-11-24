* Master: [![Build Status](https://travis-ci.org/4xxi/resonances-ml.svg?branch=master)](https://travis-ci.org/4xxi/resonances)
* Develop: [![Build Status](https://travis-ci.org/4xxi/resonances-ml.svg?branch=develop)](https://travis-ci.org/4xxi/resonances)

# Resonances ML

## Abstract

The application identifies three-body resonances using machine learning (ML)
techniques. It takes data from AstDys catalog that will be used as feature
dataset and text file with asteroids has resonance, they will be used for
forming target vector. The application allows to check desired ML technique for
common metrics and for influence of pointed features.
The application by default has:

* Catalogs.
* Lists of 50, 100, 200 resonant asteroids for Jupiter Saturn resonance with integers 4 -2 -1.
* List of all resonant asteroids for Jupiter Saturn resonance with integers 4 -2 -1.
* Lists of all resonant asteroids for Jupiter Saturn resonance with integers from 1 to 25 order.
* Configuration file with couple of ML techniques suitable for classifying.
  Suitability of them determined by related research.

## Installation

You can install the Resonances ML using Python package manager pip

`pip install git+https://github.com/4xxi/resonances`

## Usage

Commands `get`, `test-clf`, `plot` build cache. Point flag `-r` to remove it.

### Getting resonant asteroids

Command: `python -m resonancesml get -n 2000 -c syn -e '2' --clf='KNN 0'`.

This will returns resonant asteroids classified by ML technique K Nearest
Neighbors (KNN) using first (0) argument preset. Count is from 0. Point
`--clf='DT 1'` to use Decision Tree with second (1) argument preset.

Length of learning dataset is equal to 2000. If length is not pointed, learning
set will contain asteroids from 1 to last known resonant asteroid.

Features are got from catalog of synthetic (**syn**) elements. Catalog is got from
`input` directory inside project directory. You can use custom catalog pointing in configuration file.

Option `-e '2'` means that only third field will be used from feature set.
Count is from 0. Point `-e='2 4'` to use third and fifth fields. Indices can be
negative it means that count will from **last column**. Option `-e='2 -1'` means to
use third and last column. Note one more thing. Before classifying the
application builds cache that contains additional features it means that number
of available columns is not equal to number of columns from catalog. Also note
that **catalogs has difference between positions of features**. For example catalog
of orbital elements contains magnitude values in second column but catalog of
synthetic elements has semi-major axis in second column.

More details are available by command `python -m resonancesml get --help`

#### Custom configuration file

You can point your own configuration file in YAML format. Add option '-c' to get this.
`python -m resonancesml -c /path/to/my/config.yaml get -n 2000 -c syn -e '2' --clf='KNN 0'`

Execute command `python -m resonancesml dump-config` to see default configuration file.

### Test classifier

Command: `python -m resonancesml test-clf -n 2000 -c syn -e '2' --clf='KNN 0'`.
Description of the options [above](#getting-resonant-asteroids)

More details are available by command `python -m resonancesml test-clf --help`

### Get influence of fields

Command `python -m resonancesml influence-fields -c syn --clf='DT 0'`. Meaning
of this options is same as in [above](#getting-resonant-asteroids) but this command
also has one different option `-l /path/to/list`. This path to list of resonant asteroids.

### Plotting

Command: `python -m resonancesml plot -n 2000 -c syn`. Meaning
of this options is same as in [above](#getting-resonant-asteroids)

More details are available by command `python -m resonancesml plot --help`

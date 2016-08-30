# Usage

## For learning, testing and comparing availaible method for catalog of synthetic elements.
`python -m resonanceml learn -l /path/to/file/with-librated-asteroids -c syn`.

## For learning, testing and comparing availaible method for catalog of orbital elements.
`python -m resonanceml learn -l /path/to/file/with-librated-asteroids -c cat`.

These commands will show table that shows methods, metrics, values of TP, TN, FP, FN.

## For showing ability to classify of methods Decision Tree and K Nearest Neighbors on 400000 asteroids from catalog of synthetic elements
`python -m resonanceml classify-all -l /path/to/file/with-librated-asteroids/for-learning -a /path/to/file/with-librated-asteroids/for-testing`.

## For comparing ability to classify on different subsets of feautures
`python -m resonancesml compare-fields-valuable -l /path/to/file/with-librated-asteroids`

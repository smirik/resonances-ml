from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import pandas
from pandas import DataFrame
from typing import Tuple
import numpy as np
from .shortcuts import perf_measure
from resonancesml.shortcuts import ProgressBar
from resonancesml.shortcuts import get_target_vector
from resonancesml.shortcuts import get_feuture_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from texttable import Texttable
from sklearn.base import ClassifierMixin

from .parameters import TesterParameters


def _classify(clf: ClassifierMixin, X: np.ndarray, Y: np.ndarray,
              X_test: np.ndarray, Y_test: np.ndarray) -> Tuple[float, int, int]:
    clf.fit(X, Y)
    res = clf.predict(X_test)
    TP, FP, TN, FN = perf_measure(res, Y_test)

    precision = precision_score(Y_test, res)
    recall = recall_score(Y_test, res)
    accuracy = accuracy_score(Y_test, res)

    return (precision, recall, accuracy, TP, FP, TN, FN)


class _DataSets:
    def __init__(self, librated_asteroids, learn_feature_set, all_librated_asteroids,
                 test_feature_set):
        self.librated_asteroids = librated_asteroids
        self.learn_feature_set = learn_feature_set
        self.all_librated_asteroids = all_librated_asteroids
        self.test_feature_set = test_feature_set


def _get_feature_matricies(parameters: TesterParameters, slice_len: int):
    dtype = {0:str}
    dtype.update({x: float for x in range(1, parameters.catalog_width)})
    catalog_feautures = pandas.read_csv(  # type: DataFrame
        parameters.catalog_path, delim_whitespace=True,
        skiprows=parameters.skiprows, header=None, dtype=dtype).values

    if parameters.injection:
        catalog_feautures = parameters.injection.update_data(catalog_feautures[:400000])

    learn_feature_set = catalog_feautures[:slice_len]  # type: np.ndarray
    test_feature_set = catalog_feautures[slice_len:]  # type: np.ndarray
    return learn_feature_set, test_feature_set


def _get_datasets(librate_list: str, all_librated: str, parameters: TesterParameters,
                  slice_len: int = None) -> _DataSets:
    librated_asteroids = np.loadtxt(librate_list, dtype=int)
    all_librated_asteroids = np.loadtxt(all_librated, dtype=int)
    if slice_len is None:
        slice_len = int(librated_asteroids[-1])
    learn_feature_set, test_feature_set = _get_feature_matricies(parameters, slice_len)
    return _DataSets(librated_asteroids, learn_feature_set,
                     all_librated_asteroids, test_feature_set)


def _build_table() -> Texttable:
    table = Texttable(max_width=120)
    table.header(['Classifier', 'precision', 'recall', 'accuracy', 'TP', 'FP', 'TN', 'FN'])
    table.set_cols_width([30, 15, 15, 15, 5, 5, 5, 5])
    table.set_precision(5)
    return table


def _get_classifiers():
    from sklearn.linear_model import LogisticRegression
    return {
        #'Decision tree': DecisionTreeClassifier(random_state=241, max_depth=39),
        #'K neighbors': KNeighborsClassifier(weights='distance', p=2, n_jobs=4, n_neighbors=5),
        #'Logistic regression': LogisticRegression(C=0.1, penalty='l2', n_jobs=4),
        #'Logistic regression1': LogisticRegression(C=1, penalty='l1', n_jobs=4)
        #'GB1': GradientBoostingClassifier(n_estimators=200, learning_rate=0.6, min_samples_split=10000),
        #'GB2': GradientBoostingClassifier(n_estimators=7, learning_rate=0.6, min_samples_split=15000),
        #'GB1': GradientBoostingClassifier(
            #n_estimators=200, learning_rate=0.6, min_samples_split=10*7, max_features=4, max_depth=5),
        #'GB2': GradientBoostingClassifier(
            #n_estimators=200, learning_rate=0.6, min_samples_split=10*9, max_features=4),
        #'GB3': GradientBoostingClassifier(
            #n_estimators=200, learning_rate=0.6, min_samples_split=10*9, max_features=4),
        #'GB4': GradientBoostingClassifier(
            #n_estimators=200, learning_rate=0.6, min_samples_split=10*9, max_features=4),
    }


def _classify_all(datasets: _DataSets, parameters: TesterParameters):
    table = _build_table()

    classifiers = _get_classifiers()
    data = []
    for indices in parameters.indices_cases:
        X = get_feuture_matrix(datasets.learn_feature_set, False, indices)
        Y = get_target_vector(datasets.librated_asteroids, datasets.learn_feature_set.astype(int))

        X_test = get_feuture_matrix(datasets.test_feature_set, False, indices)
        Y_test = get_target_vector(datasets.all_librated_asteroids,
                                   datasets.test_feature_set.astype(int))

        for name, clf in classifiers.items():
            precision, recall, accuracy, TP, FP, TN, FN = _classify(clf, X, Y, X_test, Y_test)
            data.append('%s;%s;%s' % (name, TP, FP))
            data.append('%s;%s;%s' % (name, FN, TN))
            table.add_row([name, precision, recall, accuracy, TP, FP, TN, FN])


    with open('data.csv', 'w') as f:
        for item in data:
            f.write('%s\n' % item)

    print('\n')
    print(table.draw())
    print('resonant %i' % Y[Y==1].shape[0])
    print('learn %i' % datasets.learn_feature_set.shape[0])
    print('total %i' % (datasets.learn_feature_set.shape[0] + datasets.test_feature_set.shape[0]))


def _get_librations_for_resonances(dataset: np.ndarray) -> defaultdict:
    resonances = np.unique(dataset[:, -2])
    import ipdb
    ipdb.set_trace()
    bar = ProgressBar(resonances.shape[0], 'Getting additional features',
                      resonances.shape[0] / 80)
    #resonance_librations_counter = {x: 0 for x in resonances}
    resonance_librations_ratio = {x: 0 for x in resonances}

    for resonance in resonances:
        bar.update()
        resonance_condition = dataset[:, -2] == resonance
        librated_indieces = np.where(resonance_condition & (dataset[:, -1] == 1))
        libration_asteroid_count = dataset[librated_indieces].shape[0]
        resonance_asteroid_count = dataset[np.where(resonance_condition)].shape[0]
        #resonance_librations_counter[resonance] += libration_asteroid_count
        resonance_librations_ratio[resonance] += libration_asteroid_count / resonance_asteroid_count

    return resonance_librations_ratio


def _update_feature_matrix(of_X: np.ndarray, by_libration_counters: defaultdict) -> np.ndarray:
    N = len(by_libration_counters)
    bar = ProgressBar(N, 'Update feature matrix', N / 80)
    #all_librations = sum([y for x, y in by_libration_counters.items()])

    resonance_view_vector = np.zeros((of_X.shape[0]), dtype=float)
    for resonance, resonance_view in by_libration_counters.items():
        bar.update()
        resonance_indieces = np.where(of_X[:, -2] == resonance)
        resonance_view_vector[resonance_indieces] = resonance_view

    #for i, features in enumerate(of_X):
        #resonance = features[-2]
        #libration_count = by_libration_counters[resonance]
        #resonance_view_vector[i] = libration_count / all_librations

    of_X = np.hstack((of_X, np.array([resonance_view_vector]).T))
    of_X[:,[-2,-1]] = of_X[:,[-1,-2]]
    return of_X


class TrainTestGenerator(object):
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

        self.iterated = False

    def __iter__(self):
        return self

    def __len__(self):
        return 1

    def __next__(self):
        if self.iterated:
            self.iterated = False
            raise StopIteration
        self.iterated = True
        #, self.Y_train, self.Y_test
        #self.X_train, self.X_test
        shape1 = self.X_train.shape[0]
        shape2 = self.X_test.shape[0]
        return [x for x in range(shape1)], [x for x in range(shape1, shape1 + shape2)]


def _serialize_integers(dataset: np.ndarray) -> np.ndarray:
    print("serialize resonances")
    serialized_resonances = np.array(['_'.join(x) for x in dataset[:, -4:-1].astype(str)])
    dataset = dataset.astype(object)
    dataset = np.hstack((dataset, np.array([serialized_resonances]).T))
    for _ in range(3):
        dataset = np.delete(dataset, -3, 1)
    dataset[:,[-2,-1]] = dataset[:,[-1,-2]]
    return dataset


def classify_all_resonances(parameters: TesterParameters, length: int, data_len: int):
    parameters.injection.set_data_len(data_len)
    table = _build_table()
    learnset, trainset = _get_feature_matricies(parameters, length)
    learnset = _serialize_integers(learnset)
    trainset = _serialize_integers(trainset)

    additional_features = _get_librations_for_resonances(learnset)
    learnset = _update_feature_matrix(learnset, additional_features)
    trainset = _update_feature_matrix(trainset, additional_features)

    indices = [
        2, -2, 5, 1
        #1, 2, 3, 4, 5, -2
        #-2, -3, 6, 7,
    ]
    X_train = learnset[:,indices]
    X_test = trainset[:,indices]
    Y_train = learnset[:,-1].astype(int)
    Y_test = trainset[:,-1].astype(int)

    cv = KFold(learnset.shape[0], n_folds=2, random_state=241)
    #classifiers = _get_classifiers()
    #for name, clf in classifiers.items():
    #clf = SVC(random_state=241)
    #kwargs = {'n_estimators': 50, 'max_features': 2, 'learning_rate': 0.85, 'max_depth': None, 'min_samples_split': 10*7}
    #kwargs = {'n_estimators': 500, 'learning_rate': 0.85, 'max_features': 4, 'min_samples_split': 30, 'max_depth': 5}

    #kwargs = {'max_features': 3 , 'n_estimators': 500 , 'max_depth': 3 , 'learning_rate': 0.85 , 'min_samples_split': 100}
    #kwargs = {'max_features': 4 , 'n_estimators': 500 , 'max_depth': 3 , 'learning_rate': 0.85 , 'min_samples_split': 2}

    #for i in range(10, 60, 10):
        #kwargs2 = {'learning_rate': 0.85 , 'max_features': 5 , 'min_samples_split': i , 'n_estimators': 500 , 'max_depth': 3}

    gener = TrainTestGenerator(X_train, X_test, Y_train, Y_test)
    #kwargs = {'max_depth': 3, 'n_estimators': 50, 'max_features': None, 'min_samples_split': 100, 'learning_rate': 0.85}
    #clf = KNeighborsClassifier(weights='distance', p=2, n_jobs=4)
    clf = GradientBoostingClassifier(random_state=241, learning_rate=0.9809, n_estimators=39, max_features=1, min_samples_split=100)
    #clf = DecisionTreeClassifier(random_state=42, max_depth=27)
    #grid = {'min_samples_split': [x for x in range(100, 1000, 100)]}
    #grid = {'C': np.power(10.0, np.arange(5, 7)), 'kernel': ['poly', 'sigmoid'], 'gamma': np.power(10.0, np.arange(-3, 3))}
    #grid = {'n_neighbors': [3,5,7,9,11], 'p': [1,2,3]}
    #grid = {'n_estimators': [50, 100, 500], 'learning_rate': [0.6, 0.85],
            #'max_depth': [3, 5, 8, 10, 12], 'max_features': [None, 2, 3, 4, 5],
            #'min_samples_split': [2, 100, 500, 10*3, 10*5, 10*7]}
    #grid = {'n_estimators': [50], 'learning_rate': [0.85],
            #'max_depth': [3], 'max_features': [None],
            #'min_samples_split': [100]}
    #gs = GridSearchCV(clf, grid, scoring='recall', cv=gener, verbose=2, n_jobs=4)
    #gs.fit(np.vstack((X_train, X_test)), np.hstack((Y_train, Y_test)))
    #import pprint
    #pprint.pprint(gs.grid_scores_)
    precision, recall, accuracy, TP, FP, TN, FN = _classify(clf, X_train, Y_train, X_test, Y_test)
    table.add_row(['GB %d' % 1, precision, recall, accuracy, TP, FP, TN, FN])

    print('\n')
    print(table.draw())


def clear_classify_all(all_librated: str, parameters: TesterParameters, length):
    datasets = _get_datasets(all_librated, all_librated, parameters, length)
    _classify_all(datasets, parameters)


def classify_all(librate_list: str, all_librated: str, parameters: TesterParameters):
    datasets = _get_datasets(librate_list, all_librated, parameters)
    _classify_all(datasets, parameters)

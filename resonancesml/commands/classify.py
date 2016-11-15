from sklearn.neighbors import KNeighborsClassifier
#from sklearn.cross_validation import KFold
#from sklearn.preprocessing import normalize
from resonancesml.loader import get_asteroids
from resonancesml.loader import get_catalog_dataset
from resonancesml.loader import get_learn_set
from sklearn.tree import DecisionTreeClassifier
#from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.grid_search import GridSearchCV
#from sklearn.svm import SVC
#import pandas
#from pandas import DataFrame
from typing import Dict
from typing import List
import numpy as np
from .shortcuts import perf_measure
from resonancesml.shortcuts import get_target_vector
from resonancesml.shortcuts import get_feuture_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from texttable import Texttable
from sklearn.base import ClassifierMixin
from .builders import build_datasets
from .builders import separate_dataset

from .parameters import TesterParameters
from resonancesml.knezevic import knezevic_metric


class ClassifyResult:
    def __init__(self, Y_pred, Y_test):
        self.predictions = Y_pred
        self.TP, self.FP, self.TN, self.FN = perf_measure(Y_pred, Y_test)
        self.precision = precision_score(Y_test, Y_pred)
        self.recall = recall_score(Y_test, Y_pred)
        self.accuracy = accuracy_score(Y_test, Y_pred)


def _classify(clf: ClassifierMixin, X: np.ndarray, Y: np.ndarray,
              X_test: np.ndarray, Y_test: np.ndarray) -> ClassifyResult:
    clf.fit(X, Y)
    res = clf.predict(X_test)
    return ClassifyResult(res, Y_test)


class _DataSets:
    def __init__(self, librated_asteroids, learn_feature_set, all_librated_asteroids,
                 test_feature_set):
        self.librated_asteroids = librated_asteroids
        self.learn_feature_set = learn_feature_set
        self.all_librated_asteroids = all_librated_asteroids
        self.test_feature_set = test_feature_set


def _get_datasets(librate_list: str, all_librated: str, parameters: TesterParameters,
                  slice_len: int = None) -> _DataSets:
    all_librated_asteroids = np.loadtxt(all_librated, dtype=int)
    catalog_features = get_catalog_dataset(parameters).values
    librated_asteroids = get_asteroids(librate_list,  catalog_features[:, 0].astype(int))
    if slice_len is None:
        slice_len = librated_asteroids[-1]

    learn_feature_set = get_learn_set(catalog_features, str(slice_len))  # type: np.ndarray
    test_feature_set = catalog_features[learn_feature_set.shape[0]:]  # type: np.ndarray

    max_number_catalog = test_feature_set[:, 0][-1]
    all_librated_asteroids = all_librated_asteroids[np.where(
        all_librated_asteroids <= int(max_number_catalog)
    )]
    mask = np.in1d(all_librated_asteroids, catalog_features[:, 0].astype(int))
    all_librated_asteroids = all_librated_asteroids[mask]
    return _DataSets(librated_asteroids, learn_feature_set,
                     all_librated_asteroids, test_feature_set)


def _build_table() -> Texttable:
    table = Texttable(max_width=120)
    table.header(['Classifier', 'indices', 'precision', 'recall', 'accuracy', 'TP', 'FP', 'TN', 'FN'])
    table.set_cols_width([20, 8, 11, 11, 11, 7, 7, 7, 7])
    table.set_precision(5)
    return table


#def _get_classifiers():
    #from sklearn.linear_model import LogisticRegression
    #return {
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
    #}


def _classify_all(datasets: _DataSets, parameters: TesterParameters,
                  clf_name: str = None) -> Dict[str, ClassifyResult]:
    table = _build_table()
    classifiers = {
        'DT': DecisionTreeClassifier(random_state=241),
        'KNN': KNeighborsClassifier(weights='distance', p=1, n_jobs=4),
        'GB': GradientBoostingClassifier(n_estimators=7, learning_rate=0.6, min_samples_split=150),
    }
    result = {}

    data = []
    for indices in parameters.indices_cases:
        X = get_feuture_matrix(datasets.learn_feature_set, False, indices)
        Y = get_target_vector(datasets.librated_asteroids, datasets.learn_feature_set.astype(int))

        X_test = get_feuture_matrix(datasets.test_feature_set, False, indices)
        Y_test = get_target_vector(datasets.all_librated_asteroids,
                                   datasets.test_feature_set.astype(int))

        for name, clf in classifiers.items():
            if clf_name and clf_name != name:
                continue
            res = _classify(clf, X, Y, X_test, Y_test)
            data.append('%s;%s;%s;%s' % (name, ' '.join([str(x) for x in indices]), res.TP, res.FP))
            data.append('%s;%s;%s;%s' % (name, ' '.join([str(x) for x in indices]), res.FN, res.TN))
            table.add_row([name, ' '.join([str(x) for x in indices]), res.precision, res.recall,
                           res.accuracy, res.TP, res.FP, res.TN, res.FN])
            result[name + '-' + '-'.join([str(x) for x in indices])] = res

    with open('data.csv', 'w') as fd:
        for item in data:
            fd.write('%s\n' % item)

    print('\n')
    print(table.draw())
    print('Amount of resonant asteroids in learning dataset %i' % Y[Y==1].shape[0])
    print('Learning dataset shape %i' % datasets.learn_feature_set.shape[0])
    print('Total amount of asteroids %i' % (datasets.learn_feature_set.shape[0] + datasets.test_feature_set.shape[0]))

    return result


#class TrainTestGenerator(object):
    #def __init__(self, X_train, X_test, Y_train, Y_test):
        #self.X_train = X_train
        #self.X_test = X_test
        #self.Y_train = Y_train
        #self.Y_test = Y_test

        #self.iterated = False

    #def __iter__(self):
        #return self

    #def __len__(self):
        #return 1

    #def __next__(self):
        #if self.iterated:
            #self.iterated = False
            #raise StopIteration
        #self.iterated = True
        ##, self.Y_train, self.Y_test
        ##self.X_train, self.X_test
        #shape1 = self.X_train.shape[0]
        #shape2 = self.X_test.shape[0]
        #return [x for x in range(shape1)], [x for x in range(shape1, shape1 + shape2)]


def _build_clf_kwargs(metric: str) -> dict:
    if metric == 'euclidean':
        return {'p': 2}
    elif metric == 'knezevic':
        return {'metric': knezevic_metric}
    else:
        raise Exception('wrong metric')


def test_classifier(X_train: np.ndarray, X_test: np.ndarray, Y_train: np.ndarray,
                    Y_test: np.ndarray, indices: List[int], metric: str):
    clf = KNeighborsClassifier(weights='distance', n_jobs=1, algorithm='ball_tree', **_build_clf_kwargs(metric))
    res = _classify(clf, X_train, Y_train, X_test, Y_test)
    result = [res.precision, res.recall, res.accuracy, res.TP, res.FP, res.TN, res.FN]
    table = _build_table()
    table.add_row([str(clf.__class__), ' '.join([str(x) for x in indices])] + result)
    print(table.draw())


def get_librated_asteroids(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray,
                           metric: str) -> np.ndarray:
    clf = KNeighborsClassifier(weights='distance', n_jobs=4, algorithm='ball_tree',
                               **_build_clf_kwargs(metric))
    clf.fit(X_train, Y_train)
    return clf.predict(X_test)


def classify_all_resonances(parameters: TesterParameters, length: int, data_len: int, filter_noise: bool,
                            add_art_objects: bool, metric: str, verbose: int):
    """
    :param parameters: parameters for testing classificators.
    :param length: length of learnset.
    :param data_len: it length of data from catalog.
    """
    table = _build_table()
    learnset, trainset = build_datasets(parameters, length, data_len, filter_noise, verbose)
    resonance_view = learnset[0][-3]

    indices = parameters.indices_cases[0]
    X_train, X_test, Y_train, Y_test = separate_dataset(parameters.indices_cases[0], learnset, trainset)

    if add_art_objects:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(ratio=0.99, random_state=42)
        X_train, Y_train = sm.fit_sample(X_train, Y_train)
        resonance_axis = parameters.injection.get_resonance_axis(resonance_view)
        X_train[:, -1] = np.power(X_train[:, 0] - resonance_axis, 2)

    #cv = KFold(learnset.shape[0], n_folds=2, random_state=241)
    #classifiers = _get_classifiers()
    #for name, clf in classifiers.items():
    #clf = SVC(random_state=241)
    #kwargs = {'n_estimators': 50, 'max_features': 2, 'learning_rate': 0.85, 'max_depth': None, 'min_samples_split': 10*7}
    #kwargs = {'n_estimators': 500, 'learning_rate': 0.85, 'max_features': 4, 'min_samples_split': 30, 'max_depth': 5}

    #kwargs = {'max_features': 3 , 'n_estimators': 500 , 'max_depth': 3 , 'learning_rate': 0.85 , 'min_samples_split': 100}
    #kwargs = {'max_features': 4 , 'n_estimators': 500 , 'max_depth': 3 , 'learning_rate': 0.85 , 'min_samples_split': 2}

    #for i in range(10, 60, 10):
        #kwargs2 = {'learning_rate': 0.85 , 'max_features': 5 , 'min_samples_split': i , 'n_estimators': 500 , 'max_depth': 3}

    #x1 = np.array(range(10)).reshape(5,-1)
    #y1 = np.array(range(5))
    #clf = KNeighborsClassifier(metric=custom_metric)
    #clf.fit(X_train, Y_train)

    #gener = TrainTestGenerator(X_train, X_test, Y_train, Y_test)
    #kwargs = {'max_depth': 3, 'n_estimators': 50, 'max_features': None, 'min_samples_split': 100, 'learning_rate': 0.85}

    kwargs = {
        'euclidean': {'p': 2},
        'knezevic': {'metric': knezevic_metric}
    }[metric]
    assert(kwargs)
    #clf = DecisionTreeClassifier()
    #from sklearn.svm import SVC
    #clf = SVC(C=0.1)
    clf = KNeighborsClassifier(weights='distance', n_jobs=4, algorithm='ball_tree', **kwargs)
    #from sklearn.linear_model import LogisticRegression
    #from sklearn.naive_bayes import GaussianNB
    #clf = GaussianNB()
    #from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    #clf = QuadraticDiscriminantAnalysis()

    #clf = GradientBoostingClassifier(
        #random_state=241,
        #learning_rate=0.9809,
        #n_estimators=40, max_features=2, min_samples_split=200)
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

    #X_train = X_train ** 2
    #X_test = X_test ** 2

    #X_train[:, 0] = X_train[:, 0] * (X_train[:, 1] +  X_train[:, 2]) + X_train[:, 3]
    #X_test[:, 0] = X_test[:, 0] * (X_test[:, 1] +  X_test[:, 2]) + X_test[:, 3]
    #print("Learnset: %i" % X_train.shape[0])
    #print("Testset: %i" % X_test.shape[0])

    #zero = KnezevicElems(0, 0, 0, 0)
    #train_k = np.array([knezevic(KnezevicElems(X_train[:, 0], X_train[:, 1], X_train[:, 2], X_train[:, 3]), zero)]).T
    #test_k = np.array([knezevic(KnezevicElems(X_test[:, 0], X_test[:, 1], X_test[:, 2], X_test[:, 3]), zero)]).T
    #X_train = np.hstack((np.array([X_train[:, 0]]).T, train_k))
    #X_test = np.hstack((np.array([X_test[:, 0]]).T, test_k))


    #from sklearn.preprocessing import StandardScaler
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train.astype(float))
    #X_test = scaler.transform(X_test.astype(float))

    #c1 = np.array([X_train[:, 0] * X_train[:, 1]]).T
    #c2 = np.array([X_test[:, 0] * X_test[:, 1]]).T
    #res = _classify(clf, , Y_train, np.hstack((X_test, test_k)), Y_test)
    res = _classify(clf, X_train, Y_train, X_test, Y_test)
    result = [res.precision, res.recall, res.accuracy, res.TP, res.FP, res.TN, res.FN]
    table.add_row([str(clf.__class__), ' '.join([str(x) for x in indices])] + result)

    #from os.path import exists as opexist
    #name = 'data.csv'
    #mode = 'a' if opexist(name) else 'w'
    #with open(name, mode) as fd:
        #fd.write(';'.join([str(x) for x in ([X_train.shape[0], X_test.shape[0]] + result)]))
        #fd.write('\n')

    print(table.draw())


def clear_classify_all(all_librated: str, parameters: TesterParameters, length):
    datasets = _get_datasets(all_librated, all_librated, parameters, length)
    _classify_all(datasets, parameters)


def classify_all(librate_list: str, all_librated: str, parameters: TesterParameters, clf_name: str = None):
    datasets = _get_datasets(librate_list, all_librated, parameters)
    res = _classify_all(datasets, parameters, clf_name)
    for name, result in res.items():
        numbers_int = np.array([datasets.test_feature_set[:, 0].astype(int)]).T
        all_objects = np.hstack((numbers_int, datasets.test_feature_set, np.array([result.predictions]).T))

        predicted_objects = all_objects[np.where(all_objects[:, -1] == 1)]
        predicted_objects_2 = predicted_objects[np.where(predicted_objects[:, 0] > 249567)][:, 1]

        mask = np.in1d(predicted_objects[:, 0], datasets.all_librated_asteroids[50:])
        predicted_objects_FP = predicted_objects[np.invert(mask)][:, 1]
        mask = np.in1d(datasets.all_librated_asteroids[50:], predicted_objects[:, 0])
        predicted_objects_FN = datasets.all_librated_asteroids[50:][np.invert(mask)].astype(str)

        with open('report-%s.txt' % name, 'w') as fd:
            fd.write('Predicted asteroids:\n%s\n' % ','.join(predicted_objects[:, 1]))
            fd.write('Predicted asteroids after 249567:\n%s\n' % ','.join(predicted_objects_2))
            fd.write('FP:\n%s\n' % ','.join(predicted_objects_FP))
            fd.write('FN:\n%s\n' % ','.join(predicted_objects_FN))
            fd.write('Asteroids was found by integration: %s\n' % datasets.all_librated_asteroids.shape[0])
            fd.write('Asteroids was found by ML: %s' % predicted_objects.shape[0])

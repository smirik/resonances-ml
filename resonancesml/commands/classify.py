from sklearn.neighbors import KNeighborsClassifier
#from sklearn.cross_validation import KFold
#from sklearn.preprocessing import normalize
from .plot import plot as _plot
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
from typing import Tuple
from typing import Dict
from typing import List
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
from .knezevic import knezevic_metric


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


class EmptyFeatures(Exception):
    pass


def _get_feature_matricies(parameters: TesterParameters, slice_len: int)\
        -> Tuple[np.ndarray, np.ndarray]:
    catalog_features = get_catalog_dataset(parameters).values
    if parameters.injection:
        catalog_features = parameters.injection.update_data(catalog_features)
        if not catalog_features.shape[0]:
            raise EmptyFeatures()

    learn_feature_set = catalog_features[:slice_len]  # type: np.ndarray
    test_feature_set = catalog_features[slice_len:]  # type: np.ndarray
    return learn_feature_set, test_feature_set


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


def _get_classifiers():
    #from sklearn.linear_model import LogisticRegression
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


class _ResonanceView:
    def __init__(self, libration_count, resonance_count):
        self.libration_count = libration_count
        self.resonance_count = resonance_count

    @property
    def ratio(self):
        return self.libration_count / self.resonance_count


def _get_librations_for_resonances(dataset: np.ndarray, verbose: bool = False) -> Dict[str, _ResonanceView]:
    resonances = np.unique(dataset[:, -2])
    bar = None
    if verbose:
        bar = ProgressBar(resonances.shape[0], 80, 'Getting additional features')
    #resonance_librations_counter = {x: 0 for x in resonances}
    resonance_librations_ratio = {x: 0 for x in resonances}
    remains_librations_count = 0
    remains_resonances_count = 0

    for resonance in resonances:
        if bar:
            bar.update()
        resonance_condition = dataset[:, -2] == resonance
        librated_indieces = np.where(resonance_condition & (dataset[:, -1] == 1))
        libration_asteroid_count = dataset[librated_indieces].shape[0]
        resonance_asteroid_count = dataset[np.where(resonance_condition)].shape[0]
        #resonance_librations_counter[resonance] += libration_asteroid_count
        resonance_librations_ratio[resonance] = _ResonanceView(
            libration_asteroid_count, resonance_asteroid_count)

        if libration_asteroid_count < 100:
            remains_librations_count = libration_asteroid_count
            remains_resonances_count = resonance_asteroid_count

    resonance_librations_ratio['other'] = _ResonanceView(
        remains_librations_count, remains_resonances_count)
    return resonance_librations_ratio


def _update_feature_matrix(of_X: np.ndarray, by_libration_counters: Dict[str, _ResonanceView],
                           verbose: bool = False) -> np.ndarray:
    bar = None
    if verbose:
        N = len(by_libration_counters) - 1
        bar = ProgressBar(N, 80, 'Update feature matrix')
    #all_librations = sum([y for x, y in by_libration_counters.items()])

    resonance_view_vector = np.zeros((of_X.shape[0]), dtype=float)
    for resonance, resonance_view in by_libration_counters.items():
        if bar:
            bar.update()
        resonance_indieces = np.where(of_X[:, -2] == resonance)
        #if resonance_view.libration_count < 100:
            #resonance_view_vector[resonance_indieces] = by_libration_counters['other'].ratio
        #else:
        if resonance_view.resonance_count == 0:
            continue
        resonance_view_vector[resonance_indieces] = resonance_view.ratio

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


INTEGERS_COUNT = 3
INTEGERS_START_INDEX = -5
AXIS_OFFSET_INDEX = -4
RESONANCE_VIEW_INDEX = -3


def _serialize_integers(dataset: np.ndarray) -> np.ndarray:
    integers_matrix = dataset[:, INTEGERS_START_INDEX:INTEGERS_START_INDEX + INTEGERS_COUNT]
    serialized_resonances = np.array(['_'.join(x) for x in integers_matrix.astype(str)])
    dataset = dataset.astype(object)
    dataset = np.hstack((dataset, np.array([serialized_resonances]).T))
    for _ in range(INTEGERS_COUNT):
        dataset = np.delete(dataset, -4, 1)
    dataset[:,[-2,-1]] = dataset[:,[-1,-2]]
    return dataset


def _filter_noises(dataset: np.ndarray, libration_views: Dict[str, float], axis_index: int,
                   verbose: bool = False) -> np.ndarray:
    filtered_dataset = None
    max_axis_offsets = {x: 0. for x in libration_views.keys()}

    for key in max_axis_offsets.keys():
        if key == 'other':
            continue
        current_resonance = dataset[:, RESONANCE_VIEW_INDEX] == key
        resonance_dataset = dataset[np.where(current_resonance)]
        is_target_true = resonance_dataset[:, -1] == 1
        is_target_false = resonance_dataset[:, -1] == 0

        #filter_cond = knezevic_filter(resonance_dataset, is_target_true)
        #suitable_objs = resonance_dataset[np.where(
            #((is_target_false & filter_cond) | is_target_true)
        #)]
        max_diff = np.max(resonance_dataset[np.where(is_target_true)][:, AXIS_OFFSET_INDEX])
        suitable_objs = resonance_dataset[np.where(
            ((is_target_false & (resonance_dataset[:, AXIS_OFFSET_INDEX] > max_diff)) | is_target_true)
            #((is_target_false & (resonance_dataset[:, AXIS_OFFSET_INDEX] > max_diff) & resonance_dataset[:, -4]) | is_target_true)
        )]

        #filter_cond = euclidean_filter(resonance_dataset, is_target_true, verbose)
        #suitable_objs = resonance_dataset[np.where(
            #((is_target_false & filter_cond) | is_target_true)
        #)]

        if verbose:
            print("%s: %s -> %s" % (key, dataset[np.where(current_resonance)].shape[0],
                                    suitable_objs.shape[0]))

        if filtered_dataset is None:
            filtered_dataset = suitable_objs
        else:
            filtered_dataset = np.vstack((filtered_dataset, suitable_objs))

    return np.array(filtered_dataset)


class CounterObj:
    counter = 0
    pairs = []
    def __init__(self, x: np.ndarray, y: np.ndarray):
        CounterObj.counter += 1
        #if (x, y) in CounterObj.pairs:
            #raise Exception('%s %s' % (x, y))
        CounterObj.pairs.append((x, y))


def _build_datasets(parameters: TesterParameters, length: int, data_len: int, filter_noise: bool,
                    verbose: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param parameters: parameters for testing classificators.
    :param length: length of learnset.
    :param data_len: it length of data from catalog.
    """
    parameters.injection.set_data_len(data_len)
    try:
        learnset, trainset = _get_feature_matricies(parameters, length)
    except EmptyFeatures:
        print('\033[91mThere is no object\033[0m')
        exit(-1)
    learnset = _serialize_integers(learnset)
    trainset = _serialize_integers(trainset)

    additional_features = _get_librations_for_resonances(learnset, verbose > 0)
    learnset = _update_feature_matrix(learnset, additional_features, verbose > 0)
    trainset = _update_feature_matrix(trainset, additional_features, verbose > 0)

    if filter_noise:
        learnset = _filter_noises(learnset, additional_features, parameters.injection.axis_index, verbose > 1)
    return learnset, trainset


def _separate_dataset(indices: List[int], learnset: np.ndarray, trainset: np.ndarray)\
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #indices = [
        #2, 3, 4, 5
        #1, 2, 3, mean_motion_idx
        #2, -2, 5
        #2, -2, 5, 1
        #1, 2, 3, 4, 5, -2
        #-2, -3, 6, 7,
    #]
    X_train = learnset[:,indices]
    X_test = trainset[:,indices]
    Y_train = learnset[:,-1].astype(int)
    Y_test = trainset[:,-1].astype(int)
    return X_train, X_test, Y_train, Y_test


class DatasetBuilder:
    def __init__(self, parameters: TesterParameters, train_length: int, data_len: int,
                 filter_noise: bool, add_art_objects: bool, verbose: int):
        self._parameters = parameters
        self._train_length = train_length
        self._data_len = data_len
        self._filter_noise = filter_noise
        self._add_art_objects = add_art_objects
        self._verbose = verbose

    def build(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        learnset, trainset = _build_datasets(self._parameters, self._train_length, self._data_len,
                                             self._filter_noise, self._verbose)
        resonance_view = learnset[0][-3]

        indices = self._parameters.indices_cases[0]
        X_train, X_test, Y_train, Y_test = _separate_dataset(indices, learnset, trainset)

        if self._add_art_objects:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(ratio=0.99, random_state=42)
            X_train, Y_train = sm.fit_sample(X_train, Y_train)
            resonance_axis = self._parameters.injection.get_resonance_axis(resonance_view)
            X_train[:, -1] = np.power(X_train[:, 0] - resonance_axis, 2)
        return X_train, X_test, Y_train, Y_test


def classify_all_resonances(parameters: TesterParameters, length: int, data_len: int, filter_noise: bool,
                            add_art_objects: bool, metric: str, plot: bool, verbose: int):
    """
    :param parameters: parameters for testing classificators.
    :param length: length of learnset.
    :param data_len: it length of data from catalog.
    """
    table = _build_table()
    learnset, trainset = _build_datasets(parameters, length, data_len, filter_noise, verbose)
    resonance_view = learnset[0][-3]

    indices = parameters.indices_cases[0]
    X_train, X_test, Y_train, Y_test = _separate_dataset(parameters.indices_cases[0], learnset, trainset)

    if add_art_objects:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(ratio=0.99, random_state=42)
        X_train, Y_train = sm.fit_sample(X_train, Y_train)
        resonance_axis = parameters.injection.get_resonance_axis(resonance_view)
        X_train[:, -1] = np.power(X_train[:, 0] - resonance_axis, 2)

    if plot:
        _plot(X_train, Y_train)
        return

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

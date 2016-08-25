import numpy as np
from resonancesml.shortcuts import ProgressBar
from texttable import Texttable
from typing import Tuple
from typing import List
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from resonancesml.settings import PROJECT_DIR
from os.path import join as opjoin
import pandas
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


CATALOG_PATH = opjoin(PROJECT_DIR, '..', 'input', 'all.syn')


def _validate(data: DataFrame):
    flag = False
    for i in data.keys():
        if data[i].hasnans():
            flag = True
            print(i)

    if flag:
        raise Exception('syntetic elements has nan values')


def _get_target_vector(from_asteroids: np.ndarray, by_features: np.ndarray) -> np.ndarray:
    target_vector = []
    for i, asteroid_number in enumerate(by_features[:, 0]):
        target_vector.append(asteroid_number in from_asteroids)
    return np.array(target_vector, dtype=np.float64)


def _get_feuture_matrix(from_features: np.ndarray, scale: bool, indices: List[int]) -> np.ndarray:
    res = from_features[: ,indices]
    #res = from_features[: ,2:6]
    if scale:
        scaler = StandardScaler()
        res = scaler.fit_transform(res)
    return res


def _classify(clf: ClassifierMixin, kf: cross_validation.KFold, X: np.ndarray, Y: np.ndarray)\
        -> Tuple[float, int, int]:
    precisions = []
    recalls = []
    scores = []
    for train_index, test_index in kf:
        X_train = X[train_index]
        Y_train = Y[train_index]
        X_test = X[test_index]
        Y_test = Y[test_index]

        clf.fit(X_train, Y_train)
        res = clf.predict(X_test)  # type: np.ndarray
        precisions.append(precision_score(Y_test, res))
        recalls.append(recall_score(Y_test, res))
        scores.append(accuracy_score(Y_test, res))
    return np.mean(precisions), np.mean(recalls), np.mean(scores)


def _get_headers(indices: List[int]) -> List[str]:
    headers = []
    with open(CATALOG_PATH) as f:
        for i, line in enumerate(f):
            if i == 1:
                headers = [x.strip() for x in line.split('  ') if x]
            elif i > 1:
                break
    res = []
    for index in indices:
        res.append(headers[index])
    return res


def learn(librate_list: str):
    indices_cases = [
        [2,3,4],
        [2,3,4,5]
    ]
    librated_asteroids = np.loadtxt(librate_list, dtype=int)
    slice_len = int(librated_asteroids[-1])
    dtype = {0:str}
    dtype.update({x: float for x in range(1,10)})
    syntetic_elems = pandas.read_csv(CATALOG_PATH, delim_whitespace=True,  # type: DataFrame
                                     skiprows=2, header=None, dtype=dtype)

    learn_feature_set = syntetic_elems.values[:slice_len]  # type: np.ndarray
    table = Texttable(max_width=120)
    table.header(['Classifier', 'Input data (fields)', 'precision', 'recall', 'accuracy'])
    table.set_cols_width([30, 30, 15, 15, 15])
    table.set_precision(len(indices_cases) * 5)

    bar = ProgressBar(10, 'Learning', 1)

    for indices in indices_cases:
        classifiers = {
            'Decision tree': DecisionTreeClassifier(random_state=241),
            'Gradient boosting (10 trees)': GradientBoostingClassifier(n_estimators=10),
            'Gradient boosting (50 trees)': GradientBoostingClassifier(n_estimators=50),
            'K neighbors': KNeighborsClassifier(weights='distance', p=100, n_jobs=4),
            'Logistic regression': LogisticRegression(C=10)
        }
        for name, clf in classifiers.items():
            headers = _get_headers(indices)
            Y = _get_target_vector(librated_asteroids, learn_feature_set.astype(int))
            X = _get_feuture_matrix(learn_feature_set, False, indices)

            #import ipdb
            #ipdb.set_trace()

            kf = cross_validation.KFold(X.shape[0], 5, shuffle=True, random_state=42)
            precision, recall, accuracy = _classify(clf, kf, X, Y)
            table.add_row([name, ', '.join(headers), precision, recall, accuracy])
            bar.update()

    print('\n')
    print(table.draw())
    print('Dataset for used for learning and testing by k-fold cross-validation: %d' % Y.shape)

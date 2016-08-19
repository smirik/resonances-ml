import numpy as np
import scipy
from sklearn.metrics import roc_auc_score
from resonancesml.settings import PROJECT_DIR
from os.path import join as opjoin
import pandas
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation


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
    missed_asteroids = []
    for i, asteroid_number in enumerate(by_features[:, 0]):
        last_asteroid = by_features[:, 0][i - 1]
        if i > 0 and (asteroid_number - last_asteroid) > 1:
            missed_asteroids += [x for x in range(last_asteroid + 1, asteroid_number)]
        target_vector.append(asteroid_number in from_asteroids)

    return np.array(target_vector)


def _get_feuture_matrix(from_features: np.ndarray) -> np.ndarray:
    res = scipy.delete(from_features, 0, 1)
    return res


def learn(librate_list: str):
    librated_asteroids = np.loadtxt(librate_list)
    syntetic_elems = pandas.read_csv(CATALOG_PATH, delim_whitespace=True,  # type: DataFrame
                                     skiprows=2, header=None)

    learn_feature_set = syntetic_elems.values[:1000]

    Y = _get_target_vector(librated_asteroids, learn_feature_set)
    X = _get_feuture_matrix(learn_feature_set)
    clf = LogisticRegression(C=2)
    kf = cross_validation.KFold(X.shape[0], 5, shuffle=True, random_state=42)

    scores = []
    for train_index, test_index in kf:
        X_train = X[train_index]
        Y_train = Y[train_index]
        X_test = X[test_index]
        Y_test = Y[test_index]

        clf.fit(X_train, Y_train)
        scores.append(roc_auc_score(Y_test, clf.predict_proba(X_test)[:, 1]))

    print("accuracy: %f" % (sum(scores) / len(scores)))

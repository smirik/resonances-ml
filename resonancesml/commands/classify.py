from sklearn.neighbors import KNeighborsClassifier
from resonancesml.settings import CATALOG_PATH
import pandas
from pandas import DataFrame
import numpy as np
from resonancesml.shortcuts import get_target_vector
from resonancesml.shortcuts import get_feuture_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

def classify(librate_list: str, all_librated: str):
    indices = [2,3,4,5]
    librated_asteroids = np.loadtxt(librate_list, dtype=int)

    all_librated_asteroids = np.loadtxt(all_librated, dtype=int)

    clf = KNeighborsClassifier(weights='distance', p=1, n_jobs=4)
    dtype = {0:str}
    dtype.update({x: float for x in range(1,10)})
    syntetic_elems = pandas.read_csv(CATALOG_PATH, delim_whitespace=True,  # type: DataFrame
                                     skiprows=2, header=None, dtype=dtype)
    slice_len = int(librated_asteroids[-1])
    learn_feature_set = syntetic_elems.values[:slice_len]  # type: np.ndarray
    test_feature_set = syntetic_elems.values[:400000]  # type: np.ndarray

    Y = get_target_vector(librated_asteroids, learn_feature_set.astype(int))
    X = get_feuture_matrix(learn_feature_set, False, indices)

    Y_test = get_target_vector(all_librated_asteroids, test_feature_set.astype(int))
    X_test = get_feuture_matrix(test_feature_set, False, indices)

    clf.fit(X, Y)
    res = clf.predict(X_test)

    precision = precision_score(Y_test, res)
    recall = recall_score(Y_test, res)
    accuracy = accuracy_score(Y_test, res)

    print('precision %f' % precision)
    print('recall %f' % recall)
    print('accuracy %f' % accuracy)

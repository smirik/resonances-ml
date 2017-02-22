from resonancesml.loader import get_learn_set
from resonancesml.shortcuts import ProgressBar
from texttable import Texttable
from typing import Dict
from resonancesml.shortcuts import get_target_vector
from resonancesml.shortcuts import get_feature_matrix
from resonancesml.loader import get_asteroids
from .shortcuts import classify
from pandas import DataFrame
from sklearn import cross_validation

from resonancesml.reader import CatalogReader

from sklearn.base import ClassifierMixin


def _validate(data: DataFrame):
    flag = False
    for i in data.keys():
        if data[i].hasnans():
            flag = True
            print(i)

    if flag:
        raise Exception('syntetic elements has nan values')


class MethodComparer:
    def __init__(self, librate_list: str, catalog_reader: CatalogReader):
        self._catalog_features = catalog_reader.read().values
        self._librated_asteroids = get_asteroids(
            librate_list, self._catalog_features[:, 0].astype(int))

        self._catalog_reader = catalog_reader
        self._classifiers = None  # type: Dict[str, ClassifierMixin]
        self._keys = None

    def set_methods(self, values: Dict[str, ClassifierMixin], keys):
        self._classifiers = values
        self._keys = keys

    def learn(self):
        learn_feature_set = get_learn_set(self._catalog_features,  # type: np.ndarray
                                          str(self._librated_asteroids[-1]))
        table = Texttable(max_width=120)
        table.header(['Classifier', 'Input data (fields)', 'precision', 'recall',
                      'accuracy', 'TP', 'FP', 'TN', 'FN'])
        table.set_cols_width([30, 30, 15, 15, 15, 5, 5, 5, 5])
        table.set_precision(5)

        bar = ProgressBar(len(self._catalog_reader.indices_cases) * len(self._classifiers), 80,
                          'Learning')
        if self._catalog_reader.injection:
            learn_feature_set = self._catalog_reader.injection.update_data(learn_feature_set)
        Y = get_target_vector(self._librated_asteroids, learn_feature_set.astype(int))

        data = []
        for indices in self._catalog_reader.indices_cases:
            headers = self._catalog_reader.get_headers(indices)
            X = get_feature_matrix(learn_feature_set, False, indices)

            for name in self._keys:
                clf = self._classifiers[name]
                kf = cross_validation.KFold(X.shape[0], 5, shuffle=True, random_state=42)
                precision, recall, accuracy, TP, FP, TN, FN = classify(clf, kf, X, Y)
                data.append('%s;%s;%s' % (name, TP, FP))
                data.append('%s;%s;%s' % (name, FN, TN))
                table.add_row([name, ', '.join(headers), precision, recall, accuracy,
                               int(TP), int(FP), int(TN), int(FN)])
                bar.update()

        with open('data.csv', 'w') as f:
            for item in data:
                f.write('%s\n' % item)

        print('\n')
        print(table.draw())
        print('Dataset for used for learning and testing by k-fold cross-validation: %d' % Y.shape)

from math import sqrt

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit
from statistics import mean

from loggermixin import LoggerMixin
import logging


def reg_log_error(y_predicted, y_test):
    return sqrt(mean_squared_error(y_predicted, y_test))


class Predictor(LoggerMixin, object):

    def __init__(self, model, fold = 5, file='temp.log', loglevel=logging.INFO):
        super().__init__(file, loglevel)
        self.logger.debug('Created an instance of %s', self.__class__.__name__)
        self.model = model
        self.splitor = TimeSeriesSplit(n_splits=fold)

    @staticmethod
    def reg_error(y_predicted, y_test):
        return sqrt(mean_squared_log_error(y_predicted, y_test))

    def fit(self,X,y=None):
        return self

    def predict(self, X, y=None):
        predictions = pd.DataFrame([])

        i = 0
        for train_idx, test_idx in self.splitor.split(X):
            X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
            self.model.fit(X_train, y_train)
            predictions[i] = self.model.predict(X_test)
            i += 1

        return predictions

    def score(self, X, y=None):
        self.logger.debug('--- score starts.')
        errors = []

        for train_idx, test_idx in self.splitor.split(X):
            X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
            self.model.fit(X_train, y_train)
            y_predicted = self.model.predict(X_test)
            error = self.reg_error(y_predicted, y_test)
            errors.append(error)

            self.logger.debug('test idx {}, error: {:.6f}'.format(test_idx[0], error))
            print("---- split: {}".format(error))

        avg_error = mean(errors)
        self.logger.debug('avg error: {:.4f}'.format(avg_error))

        return avg_error

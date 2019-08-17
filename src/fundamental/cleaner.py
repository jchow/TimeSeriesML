from loggermixin import LoggerMixin
from sklearn import base
import logging
import pandas as pd
import numpy as np


class Cleaner(LoggerMixin, base.TransformerMixin):
    def __init__(self, file='temp.log', loglevel=logging.INFO):
        super().__init__(file, loglevel)
        self.logger.debug('Created an instance of %s', self.__class__.__name__)

    def fit(self, x, y = None):
        return self

    def transform(self, x):
        pd_x = x.copy()

        self.logger.debug("----- shape before cleaning %s", pd_x.shape)
        pd_x = self.__clean_data(pd_x)
        self.logger.debug(" ---- shape after cleaning data %s", pd_x.shape)
        self.logger.debug(" ---- columns remain: %s", pd_x.columns.values)

        return pd_x

    # Private method is not really private
    # Can still access using __<Class>__privateFoo
    def __clean_data(self, pd_x):
        # Drop unnecessary columns first, o/w more rows will be deleted because of null values
        pd_x.drop(['close_price_next_quarter', 'baseline_price_next_quarter'], axis=1, inplace=True)

        pd_x = pd_x.replace('nm', np.nan)
        threshold = int(0.95 * pd_x.shape[0])
        pd_x.dropna(axis='columns', thresh=threshold, inplace=True)
        pd_x.dropna(axis='rows', how='any', inplace=True)
        pd_x.sort_values('date', ascending=True, inplace=True)

        self.logger.debug(pd_x.head(1))
        return pd_x

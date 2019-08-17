from loggermixin import LoggerMixin
from sklearn import base
import logging
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


class Processor(LoggerMixin, base.TransformerMixin):
    def __init__(self, file='temp.log', loglevel=logging.INFO):
        super().__init__(file, loglevel)
        self.logger.debug('Created an instance of %s', self.__class__.__name__)

    def fit(self, x, y = None):
        return self

    def transform(self, x):
        pd_x = x.copy()

        # Shift the close price column per stock and then find %diff per stock group
        # This will be the price performance
        labeled_pd_x = self.label_data(pd_x)

        labeled_pd_x.drop(['date'], axis=1, inplace=True)

        # convert stocks symbol to numerical
        labeled_pd_x['ticker'] = labeled_pd_x['ticker'].astype('category')
        labeled_pd_x['ticker'] = labeled_pd_x['ticker'].cat.codes
        self.logger.debug("Null data - %s", labeled_pd_x.isnull().sum().sum())

        # normalized/scaled data - apart from ticker data
        scaled_data_arrays, labels = self.get_scaled_data_arrays(labeled_pd_x)

        self.logger.debug("final data shape %s", scaled_data_arrays.shape)

        return scaled_data_arrays, labels

    def label_data(self, stocks_df):
        labeled_data = stocks_df.copy()
        labeled_data['perf'] = labeled_data.groupby(['ticker'])['close_price'].diff().\
            divide(labeled_data.groupby(['ticker'])['close_price'].shift())
        labeled_data.dropna(axis='rows', how='any', inplace=True) # drop all the nan due to shifting
        self.logger.debug(labeled_data['perf'][0:7])
        return labeled_data

    def get_scaled_data_arrays(self, all_data_table):
        labels = all_data_table['perf'].values.reshape(-1,1)
        all_data_table.drop(['perf'], axis=1, inplace=True)

        data_value_arrays = all_data_table.values
        data_value_arrays = data_value_arrays.astype('float32')

        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data_arrays = scaler.fit_transform(data_value_arrays)

        scaled_labels = scaler.fit_transform(labels)

        return scaled_data_arrays, scaled_labels.flatten()

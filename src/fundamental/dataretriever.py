from loggermixin import LoggerMixin
from sklearn import base
import logging
import pandas as pd
import os
import numpy as np


class Retriever(LoggerMixin):
    def __init__(self, file='temp.log', loglevel=logging.INFO):
        super().__init__(file, loglevel)
        self.logger.debug('Created an instance of %s ', self.__class__.__name__)

    '''
        Data from each ticker is appended to form one data frame.
        Multiple time series are joined together treating stocks as one 
        of the feature.

    '''
    def getData(self):
        stocks_data = pd.DataFrame()

        # Read data from files
        resources_path = '/home/jeffchow/Dev/Projects/deepfundamental/resources'
        files = os.listdir(resources_path)
        for f in files:
            self.logger.debug(f)
            df = pd.read_csv(os.path.join(resources_path, f))
            stocks_data = stocks_data.append(df, ignore_index=True)
            self.logger.debug(stocks_data.shape)  # are they of the same shape?

        return stocks_data

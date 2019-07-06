import logging
from unittest import TestCase

from fundamentalmodeldatapreparer import FundamentalModelDataPreparer
from fundamentalworker import FundamentalWorker


class TestIntegration(TestCase):

    def test_end2end(self):
        '''
        preparer = FundamentalModelDataPreparer()
        worker = FundamentalWorker()
        tickers = ['MSFT', 'AAPL', 'CSCO', 'IBM', 'UTX', 'V']
        dataset, labels = preparer.get_dataset_for_RNN(tickers)
        model, test_set, scaler = worker.build_save_model_LSTM(dataset, labels)
        result, mse = worker.predict(test_set, scaler)

        # Plot graphs
        print('---- data labels ----')
        print(labels)
        print('==== result ====')
        print('predicted ----')
        print(result[0])
        print('validate test ----')
        print(result[1])
        y_predicted = result[0]
        y_test = result[1]

        plt.figure(figsize=(5.5, 5.5))
        plt.plot(range(len(y_predicted)), y_predicted, linestyle='-', marker='*', color='r')
        plt.plot(range(len(y_test)), y_test, linestyle='-', marker='.', color='b')
        plt.legend(['Actual', 'Predicted'], loc=2)
        plt.show()
        '''

    def test_LSTM(self):
        '''
        preparer = FundamentalModelDataPreparer(file='/tmp/test_prepare.log', loglevel=logging.DEBUG)
        worker = FundamentalWorker(file='/tmp/test_worker_lstm.log', loglevel=logging.DEBUG)
        data_array, labels = preparer.get_data_from_intrinio_file()

        save_weights_at, test_set = worker.build_save_model_LSTM(data_array, labels, 'intrinio')
        y, rmse = worker.predict(save_weights_at, test_set)

        print('==== result ====')
        print('predicted y ----')
        print(y)
        print('rmse --------')
        print(rmse)
        '''

    def test_RandomForest(self):
        preparer = FundamentalModelDataPreparer(file='/tmp/test_prepare.log', loglevel=logging.DEBUG)
        worker = FundamentalWorker(file='/tmp/test_worker_randomforest.log', loglevel=logging.DEBUG)
        data_array, labels = preparer.get_data_from_intrinio_file()

        y, rmse = worker.predict_random_forest(data_array, labels)

        print('==== result ====')
        print('predicted y = %s', y)
        print('rmse = %s', rmse)
from unittest import TestCase
from fundamentalmodeldatapreparer import FundamentalModelDataPreparer
from fundamentalworker import FundamentalWorker
from matplotlib import pyplot as plt


class TestIntegration(TestCase):

    def test_end2end(self):
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

    def test_with_intrinio(self):
        preparer = FundamentalModelDataPreparer()
        worker = FundamentalWorker()
        dataset, labels = preparer.get_dataset_from_intrinio_for_RNN()
        save_weights_at, test_set, scaler = worker.build_save_model_LSTM(dataset, labels, 'intrinio')
        result, mse = worker.predict(save_weights_at, test_set, scaler)

        print('==== result ====')
        print('predicted ----')
        print(result[0])
        print('test labels ----')
        print(result[1])
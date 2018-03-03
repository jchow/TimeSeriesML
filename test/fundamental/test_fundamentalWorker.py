from unittest import TestCase
from fundamentalworker import FundamentalWorker
import numpy as np
import numpy.testing as npt


def get_data_arrays():
    # Array of shape (5, 2, 3)
    # 3 is the num of features i.e. number of fundamental ratios
    # 2 is the num of set of data for each stock, the sequence for this stock. they can be further divided but we don't
    # have many fundamental data for each stock
    # 5 is the num of such sequence we have, in this case it is the num of stock we selected.
    return [
        [[0, 0.2], [1, 0.8]],
        [[0.1, 0], [1, 0.7]],
        [[0.5, 0.2], [0.8, 0.4]],
        [[0.9, 0.2], [0.6, 0.3]],
        [[0.4, 0], [0.2, 0.6]]]


def get_labels():
    return [1, 0.5, 0, 0, 1]


class TestFundamentalWorker(TestCase):
    def test_build_model(self):
        input_data = np.array(get_data_arrays(), dtype=float)
        labels = np.array(get_labels(), dtype=float)
        expected_test_X, expected_test_y = input_data[3:, :], labels[3:]

        worker = FundamentalWorker()
        model, actual_test, scaler = worker.build_model(input_data, labels, False)

        npt.assert_allclose(actual_test[0], expected_test_X)
        npt.assert_allclose(actual_test[1], expected_test_y)
        self.assertIsNotNone(model)

        test_result, rmse = worker.predict(model, actual_test, scaler)
        self.assertTupleEqual(test_result.shape, (2, 1))
        self.assertIs(type(test_result), np.ndarray)
        print('rmse = ', rmse)

from unittest import TestCase
from fundamentalmodeldatapreparer import FundamentalModelDataPreparer
import numpy as np


class TestFundamentalModelDataPreparer(TestCase):
    def test_get_dataset_for_one_ticker(self):
        actual_data = FundamentalModelDataPreparer().get_dataset(['MSFT'])

        self.assert_details(actual_data, (1, 7, 6), 1)
        x_data = actual_data[0]
        self.assertEqual(x_data[0][0][0], 0)
        self.assertAlmostEqual(x_data[0][-1][-1], 1, places=4)

    def test_get_dataset_for_two_ticker(self):
        actual_data = FundamentalModelDataPreparer().get_dataset(['MSFT', 'AAPL'])

        self.assert_details(actual_data, (2, 7, 6), 2)
        x_data = actual_data[0]
        self.assertEqual(x_data[0][0][0], 0)
        self.assertAlmostEqual(x_data[0][-1][-1], 1, places=4)

    def assert_details(self, actual_data, shapeX, ylen):
        self.assertIsNotNone(actual_data)
        self.assertIs(type(actual_data[0]), np.ndarray)
        self.assertIs(type(actual_data[1]), list)
        self.assertTupleEqual(actual_data[0].shape, shapeX)
        self.assertEqual(len(actual_data[1]), ylen)

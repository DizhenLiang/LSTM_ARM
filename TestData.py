import unittest
import LSTM_train_eva
import numpy as np


class TestData(unittest.TestCase):
    def setUp(self):
        self.data1 = LSTM_train_eva.Data_util('data/electricity.txt', 0.6, 0.2, False, 12, 24 * 7, 2)
        self.data2 = LSTM_train_eva.Data_util('data/solar_AL.txt', 0.7, 0.2, False, 12, 24 * 7, 0)

    def test_data_object_creation(self):
        print("Test data object creation")
        self.assertIsInstance(self.data1, LSTM_train_eva.Data_util)
        self.assertIsInstance(self.data2, LSTM_train_eva.Data_util)

    def test_arm(self):
        print("Test arm")
        file1 = open('data/electricity.txt')
        file2 = open('data/solar_AL.txt')
        no_rows1 = len(np.loadtxt(file1, delimiter=',')[0])
        no_rows2 = len(np.loadtxt(file2, delimiter=',')[0])
        file1.close()
        file2.close()

        self.assertLess(len(self.data1.rawdat[0]), no_rows1)
        self.assertEqual(len(self.data2.rawdat[0]), no_rows2)

    def test_split(self):
        print("Test split")
        file1 = open('data/electricity.txt')
        file2 = open('data/solar_AL.txt')
        data1 = np.loadtxt(file1, delimiter=',')
        data2 = np.loadtxt(file2, delimiter=',')
        file1.close()
        file2.close()

        no_train_cols1 = int(len(data1) * 0.6) + 1 - 24 * 7 - 12
        no_val_cols1 = int(len(data1) * 0.2) + 1
        no_test_cols1 = int(len(data1) * 0.2) + 1

        no_train_cols2 = int(len(data2) * 0.7) + 1 - 24 * 7 - 12
        no_val_cols2 = int(len(data2) * 0.2) - 1
        no_test_cols2 = int(len(data2) * 0.1) + 1

        self.assertEqual(len(self.data1.train[0]), no_train_cols1)
        self.assertEqual(len(self.data1.valid[0]), no_val_cols1)
        self.assertEqual(len(self.data1.test[0]), no_test_cols1)

        self.assertEqual(len(self.data2.train[0]), no_train_cols2)
        self.assertEqual(len(self.data2.valid[0]), no_val_cols2)
        self.assertEqual(len(self.data2.test[0]), no_test_cols2)


if __name__ == '__main__':
    unittest.main()

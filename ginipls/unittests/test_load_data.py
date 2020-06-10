from unittest import TestCase
import numpy.testing as npt
from sklearn.metrics.classification import f1_score
import sys
import numpy as np

sys.path.insert(0, '../data')

import data_utils

class TestLoad_data(TestCase):
    def test_load_data(self):
        x_a = np.asarray([[1., 2., 0., 7., 0.],
                          [1., 3., 1., 5., 0.],
                          [7.1, 4.5, 2.1, 0.8, 0.],
                          [1., 0., 4., 5., 0.],
                          [1., 4., 0., 7., 0.],
                          [1., 4., 1., 8., 0.]])
        y_a = [0, 0, 0, 1, 1, 1]
        h_a = ['v1', 'v2', 'v3', 'v4', 'v5']
        """test matrix with index"""
        data_with_index_path = "../../data/processed/small_data_with_index.tsv"
        ids_a = ['d89', 'd3', 'd4', 'd0', 'd2', 'd13']
        data = data_with_index_path
        x, y, h, ids = data_utils.load_data(data=data, output_col="c", index_col="id", col_sep=",", header_row_num=0)
        # print("X", x)
        # print("y", y)
        # print("h", h)
        # print("ids", ids)
        npt.assert_equal(ids, ids_a)
        npt.assert_equal(x, x_a)
        npt.assert_equal(y, y_a)
        npt.assert_equal(h, h_a)
        """test matrix without index"""
        data_without_index_path = "../../data/processed/small_data_without_index.tsv"
        # no index given
        ids_a = [0, 1, 5, 2, 3, 4] # ordre défini par les classes
        data = data_without_index_path
        x, y, h, ids = data_utils.load_data(data=data, output_col="c", col_sep=",", header_row_num=0)
        # print("X", x)
        print("y", y)
        # print("h", h)
        # print("ids", ids)
        npt.assert_equal(ids, ids_a)
        npt.assert_equal(x, x_a)
        npt.assert_equal(y, y_a)
        npt.assert_equal(h, h_a)

    def test_f1_macro_average(self):
        """
        test f1 score macro avg in the case of a category is absent from ypred
        :return:
        """
        y_true = [1, 1, 0, 0, 0, 0]
        y_pred = [0, 0, 0, 0, 0, 0]
        score = f1_score(y_true, y_pred, labels=[0, 1], average='macro')
        self.assertEqual(score, 0.4)
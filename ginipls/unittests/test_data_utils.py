# encoding: utf-8
import os.path as op
import numpy as np
import pandas as pd
import numpy.testing as npt
import random

import taj

## py.test taj -s # -s pour afficher le résultat de la console
from taj import data_utils

data_path = op.join(taj.__path__[0], 'data')

'''
def test_balance_data():
    print()
    instances_removed_list = []
    data = op.join(data_path, "small_data_without_index.tsv")
    x, y, h, ids = data_utils.load_data(data=data, output_col="c", col_sep=",", header_row_num=0)
    print("x", pd.DataFrame(x))
    print("y", y)
    print("ids", ids)
    max_nbtry=10

    for i in range(20):
        print("balanced", i)
        nb_try=0
        while True:
            xb, yb, idsb, instances_removed = data_utils.balance_data(x, y, ids)
            instances_removed = set(instances_removed)
            if not instances_removed in instances_removed_list:
                print("instances_removed", instances_removed)
                instances_removed_list += [instances_removed]
                break
            nb_try+=1
            if nb_try == max_nbtry:
                break

        print("xb", pd.DataFrame(xb))
        print("yb", yb)
        print("idsb", idsb)
        print("instances_removed_list", instances_removed_list)
'''

def test_load_data_fromFile():
    #print()
    x_a=np.asarray([[1., 2., 0., 7., 0.],
    [1., 3., 1., 5., 0.],
    [7.1, 4.5, 2.1, 0.8, 0.],
    [1., 0., 4., 5., 0.],
    [1., 4., 0., 7., 0.],
    [1., 4., 1., 8., 0.]])
    y_a=[0, 0, 0, 1, 1, 1]
    h_a=['v1', 'v2', 'v3', 'v4', 'v5']
    """test matrix with index"""
    ids_a=['d89', 'd3', 'd4', 'd0', 'd2', 'd13']
    data = op.join(data_path, "small_data_with_index.tsv")
    x, y, h, ids = data_utils.load_data(data=data, output_col="c", index_col="id", col_sep=",", header_row_num=0)
    # print("X", type(X))
    # print("y", y)
    # print("h", h)
    # print("ids", ids)
    npt.assert_equal(ids, ids_a)
    npt.assert_equal(x, x_a)
    npt.assert_equal(y, y_a)
    npt.assert_equal(h, h_a)
    """test matrix without index"""
    # no index given
    ids_a = [0, 1, 5, 2, 3, 4]
    data = op.join(data_path, "small_data_without_index.tsv")
    x, y, h, ids = data_utils.load_data(data=data, output_col="c", col_sep=",", header_row_num=0)
    npt.assert_equal(ids, ids_a)
    npt.assert_equal(x, x_a)
    npt.assert_equal(y, y_a)
    npt.assert_equal(h, h_a)

# def test_save_and_load_evaluation_data():
#     y_a = [0, 0, 1, 1, 1, 0]
#     ids = [0, 1, 5, 2, 3, 4]
#     y_fname = op.join(data_path, "y.tsv")
#     taj.save_data(ids, y_a, y_fname)
#     y = taj.load_evaluation_data(y_fname)
#     npt.assert_equal(y_a, y)


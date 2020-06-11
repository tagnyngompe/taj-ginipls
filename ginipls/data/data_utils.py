# encoding: utf-8
import pandas as pd
import os, random, copy
import numpy as np
from ginipls.config import GLOBAL_LOGGER
logger = GLOBAL_LOGGER

#/***********************************************************************/
#/*            Data ingestion: matrix loading and preprocessing         */
#/***********************************************************************/

DEFAULT_COL_SEP = "\t"

def load_data(data, output_col=0, index_col=None,
              col_sep=DEFAULT_COL_SEP, header_row_num='infer', sort_output=True):
    """
    Function that takes experimental data and gives us the
    dependent/independent variables for analysis.

    Parameters
    ----------
    data : Pandas DataFrame or string.
        If this is a DataFrame, it should have the columns `contrast1` and
        `answer` from which the dependent and independent variables will be
        extracted. If this is a string, it should be the full path to a csv
        file that contains data that can be read into a DataFrame with this
        specification.

    Returns
    -------
    features_matrix : numpy.matrixlib.defmatrix.matrix
        the matrix of features (independent variables)
    classes_vector : numpy.matrixlib.defmatrix.matrix
        vector of groundtruth classes (dependent variable)
        convert into numbers (because PLS supports only
        integers as categories
    classes_num_dict: dict
        a dictionary matching the name of each category to a number
    header_vector : list
        The header vector (words)
    instances_ids: list
        the list of the ids of line
    """
    if isinstance(data, str):
        logger.info("loading data from %s" % os.path.abspath(data))
        if not os.path.exists(data):
            logger.error("Error: data_utils.load_data", data, "not found!")
            raise FileExistsError
        data = pd.read_csv(filepath_or_buffer=data, sep=col_sep, header=header_row_num)
        #data.sample(frac=1)
        # Tri des matrices avec pandas
    #print(data.columns.values)
    if sort_output and not output_col is None:
        data = data.sort_values(by=output_col)
    # get header, dependent/independent variables for analysis.
    indep_var_names = list(data.columns.values)
    if not index_col is None :
        if isinstance(index_col, int):
            index_col = indep_var_names[index_col]
        if not index_col in indep_var_names :
            logger.warning("the column %s of index is not in the input data" % index_col)
            index_col = None
    if not output_col is None:
        indep_var_names.remove(output_col)  # supprime la catégorie (en présence de header, l'utilisation d'indice est impossible)
    cols_to_drop = []
    if not output_col is None:
        cols_to_drop += [output_col]
    instances_names = None
    if not index_col is None:
        indep_var_names.remove(index_col)
        cols_to_drop += [index_col]
        instances_names = data[index_col].values.tolist()
    else:
        #instances_names = data.index.values.tolist()
        instances_names = None
    dep_var_values = None
    if not output_col is None:
        dep_var_values = data[output_col].tolist()
    # convert dep_var_values into numbers
    categories_num_dict = {}
    categories = sorted(list(set(dep_var_values)))
    i = 0
    for category in categories:
        categories_num_dict[category] = i
        i += 1
    #print("categories2num", categories_num_dict)
    dep_var_values = [categories_num_dict[category] for category in dep_var_values]
    # delete the column of categories
    if len(cols_to_drop) > 0:
        data = data.drop(cols_to_drop, 1)

    indep_var_matrix = data.values.tolist()
    # logger.debug('h=%s'str(indep_var_names))
    # logger.debug('nb attributes=%d' % len(indep_var_names))
    # logger.debug('X = %s', str(indep_var_matrix))
    # logger.debug('X.shape=%s', str(np.asarray(indep_var_matrix).shape))
    # logger.debug('y=%s' % str(dep_var_values))
    # logger.debug('ids=%s' % str(instances_names))
    return indep_var_matrix, dep_var_values, indep_var_names, instances_names

def save_y_in_file(row_index, y, y_file_name, col_sep=DEFAULT_COL_SEP):
    """
    Save one column of y in a file
    """
    with open(y_file_name, 'w') as f:
        if row_index is None:
            for y in y:
                f.write(''.join([str(y),"\n"]))
        else:
            for id, y in zip(row_index, y):
                f.write(''.join([str(id),col_sep,str(y),"\n"]))

def save_ytrue_and_ypred_in_file(row_index, y_trues, y_preds, y_file_name, col_sep=DEFAULT_COL_SEP):
    """
    Save two columns of values in a file (expected - predicted)
    """
    with open(y_file_name, 'w') as f:
        if row_index is None:
            f.write("".join(["y_true",col_sep,"y_pred","\n"]))
            for y_true, y_pred in zip(y_trues, y_preds):
                f.write("".join([str(y_true),col_sep,str(y_pred),"\n"]))
        else:
            f.write("".join(["docId", col_sep, "y_true",col_sep,"y_pred","\n"]))
            for id, y_true, y_pred in zip(row_index, y_trues, y_preds):
                f.write("".join([str(id),col_sep,str(y_true),col_sep,str(y_pred),"\n"]))

def load_evaluation_data(y_file_name, y_col=1, col_sep=DEFAULT_COL_SEP, header_row_num=None):
    """
    read two 2-column file (row_index, y) and return the 2 lists of outputs for evaluation
    :param y_file_name: output fname
    :param y_col: output column id or name
    :param col_sep: column separator in file (default: "\t")
    :param header_row_num: header row number is any (default: None)
    :return: two lists: y_true, y_pred
    """
    data = pd.read_csv(filepath_or_buffer=y_file_name, sep=col_sep, header=header_row_num)
    return data[y_col].tolist()

def load_ytrue_ypred_file(y_file_name, indexCol="docId", yTrueCol="y_true", yPredCol="y_pred", col_sep=DEFAULT_COL_SEP):
    """"""
    #print("data.utils.load_evaluation_data2: ", y_file_name)
    try:
        if yTrueCol is None or yPredCol is None:
            header_row_num = None
        else:
            header_row_num = 'infer'
        data = pd.read_csv(filepath_or_buffer=y_file_name, sep=col_sep, header=header_row_num)
        logger.debug("data=%s" % str(data))
        ids = data.index.values.tolist() if indexCol is None else None #data[indexCol].tolist()
        #print("ids", ids)
        columns = data.columns.values.tolist()
        if yTrueCol is None or yPredCol is None:
            yTrueCol = columns[0]
            yPredCol = columns[1]
        ytrue = data[yTrueCol].values.tolist()
        ypred = data[yPredCol].values.tolist()
        return ids, ytrue, ypred
    except Exception as ex:
        print("Exception", ex)
        return None, None, None

# if __name__ == '__main__':
#     y_file_name = "/home/tagny/Documents/current-work/sens-resultat/cv-demande_resultat_a_resultat_context/styx_lemma_4_folds_cv/nbsvm/prediction/test3.pred"
#     print(load_evaluation_data2(y_file_name,indexCol=None, yTrueCol=None, yPredCol=None, col_sep=" "))

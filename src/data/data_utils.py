# encoding: utf-8
import pandas as pd
import os, random, copy
import numpy as np

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
        #print("loading data from", data)
        if not os.path.exists(data):
            print("Error: data_utils.load_data", data, "not found!")
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
        indep_var_names.remove(index_col)
    if not output_col is None:
        indep_var_names.remove(output_col)  # supprime la catégorie (en présence de header, l'utilisation d'indice est impossible)
    cols_to_drop = []
    if not output_col is None:
        cols_to_drop += [output_col]
    instances_names = None
    if not index_col is None:
        cols_to_drop += [index_col]
        instances_names = data[index_col].values.tolist()
    else:
        instances_names = data.index.values.tolist()
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
    # print('h', indep_var_names)
    # #print('h', len(indep_var_names))
    # #print('X', indep_var_matrix)
    # print('X', np.asarray(indep_var_matrix).shape)
    # print('y', dep_var_values)
    # print('ids', instances_names)
    # print("end loading data")
    return indep_var_matrix, dep_var_values, indep_var_names, instances_names

def save_data(row_index, values, y_file_name, col_sep=DEFAULT_COL_SEP):
    """
    Save one column of values in a file
    """
    with open(y_file_name, 'w') as f:
        for id, y in zip(row_index, values):
            f.write(str(id)+col_sep+str(y)+"\n")

def save_data2(row_index, y_trues, y_preds, y_file_name, col_sep=DEFAULT_COL_SEP):
    """
    Save two columns of values in a file (expected - predicted)
    """
    with open(y_file_name, 'w') as f:
        f.write("docId" + col_sep + "y_true" + col_sep + "y_pred" + "\n")
        for id, y_true, y_pred in zip(row_index, y_trues, y_preds):
            f.write(str(id)+col_sep+str(y_true)+col_sep+str(y_pred)+"\n")
        f.close()

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


def balance_data(x_init, y_init, ids_init):
    x= copy.copy(x_init)
    y=copy.copy(y_init)
    ids = copy.copy(ids_init)
    classes = set(y)
    min_num = len(y)
    c_min = None
    nums = {}
    for c in classes:
        nums[c] = [i for i, j in zip(ids, y) if j == c]
        num_c = len(nums[c])
        if min_num > num_c:
            c_min = c
            min_num = num_c
    #print("c_min", c_min)
    #print("min_num", min_num)
    selections = {}
    instances_removed=[]
    for c in classes:
        if c == c_min or len(nums[c]) == min_num:
            continue;
        # on pique les éléments à supprimer dans les autres c
        selections[c] = random.sample(nums[c], len(nums[c]) - min_num)
        instances_removed += selections[c]
        for i in selections[c]:
            index = ids.index(i)
            x = np.delete(x, index, 0)
            y.pop(index)
            ids.pop(index)
    #print("nums", nums)
    #print("selections", selections)
    return x, y, ids, instances_removed


def load_evaluation_data2(y_file_name, indexCol="docId", yTrueCol="y_true", yPredCol="y_pred", col_sep=DEFAULT_COL_SEP):
    """"""
    print("data.utils.load_evaluation_data2: ", y_file_name)
    try:
        data = pd.read_csv(filepath_or_buffer=y_file_name, sep=col_sep, header= (None if indexCol is None else 0))
        print("data", data)
        ids = data.index.values.tolist() if indexCol is None else data[indexCol].tolist()
        print("ids", ids)
        columns = data.columns.values.tolist()
        if yTrueCol is None:
            yTrueCol = columns[0]
        if yPredCol is None:
            yPredCol = columns[1]
        ytrue = data[yTrueCol].values.tolist()
        print("ytrue: ", ytrue)
        ypred = data[yPredCol].values.tolist()
        return ids, ytrue, ypred
    except Exception as ex:
        print("Exception", ex)
        return None, None, None

# if __name__ == '__main__':
#     y_file_name = "/home/tagny/Documents/current-work/sens-resultat/cv-demande_resultat_a_resultat_context/styx_lemma_4_folds_cv/nbsvm/prediction/test3.pred"
#     print(load_evaluation_data2(y_file_name,indexCol=None, yTrueCol=None, yPredCol=None, col_sep=" "))

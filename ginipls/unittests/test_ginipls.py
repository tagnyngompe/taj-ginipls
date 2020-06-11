
import os.path as op
import numpy as np

from ginipls.data.data_utils import load_data 
from ginipls.__main__ import init_and_train_pls 
from ginipls.models.ginipls import PLS, PLS_VARIANT
from ginipls.config import GLOBAL_LOGGER
logger = GLOBAL_LOGGER

  
  
  
def main():
  X_train =  [[2.0, 0.0, 7.0, 4, 5.2, 9.7], [3.0, 1.0, 5.0, 4, 1.0, .97], [0.0, 4.0, 5.0, 4, .1235, 2.58], [4.0, 0.0, 7.0, 4, 10, 4.78], [4.0, 1.0, 8.0, 4, 1, 5], [1.5, 1.3, 1.1, 4, 7, 6]]
  y_train= [0, 0, 0, 1, 1, 1]
  X_test =  [[.8, 1.0, 3, 0.0, 7.0, 4], [1.3, 1.0, 5.0, 1.4, 1.7, 5], [4.0, 4, .1235, 1.2, 5.5, 3], [3.3, 0.0, 7.0, 1.0, 1.4, 6], [1.5, 0.0, 7.0, 1.3, 1.1, 0]]
  y_test= [1, 0, 0, 1, 1]  
  
  from sklearn import datasets
  from sklearn.model_selection import train_test_split
  iris = datasets.load_iris()  
  #X_train, X_test, y_train, y_test = train_test_split(iris.data[50:], iris.target[50:], test_size=0.33, random_state=42) # classe{2,3}
  
  data_path = 'data/processed'
  trainfilename = op.join(data_path, 'doris0_CHI2_ATF-train.tsv')
  testfilename = op.join(data_path, 'doris0_CHI2_ATF-test.tsv')  
  X_train, y_train, h, ids_train = load_data(data=trainfilename, output_col='category')
  X_test, y_test, h, ids_test = load_data(data=testfilename, output_col='category')
  
  pls_type = PLS_VARIANT.GINI
  
  nu_min = 1
  nu_max = 2
  nu_step = 0.01
  nu_range = [i*nu_step for i in range(int(nu_min/nu_step),int(nu_max/nu_step))]  

  n_components_min = 1
  n_components_max = min(5,len(X_train[0])) # nb de caractéristiques
  n_components_step = 1
  n_components_range = range(n_components_min, n_components_max, n_components_step)

  hyerparameters_selection_nfolds=10
  
  clf = init_and_train_pls(X_train, y_train, pls_type, hyerparameters_selection_nfolds, nu_range, n_components_range, only_the_first_fold=False)
  test_score = clf.score(X_test, y_test)
  logger.info("test f1_score = %.3f" % (test_score))


if __name__ == "__main__":
  main()


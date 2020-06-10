
import os.path as op
import numpy as np

from ginipls.data.data_utils import load_data 
from ginipls.models.ginipls import PLS, PLS_VARIANT
from ginipls.experiments.api import evaluation


   # [[2.0, 0.0, 7.0, 4], [3.0, 1.0, 5.0, 4], [0.0, 4.0, 5.0, 4], [4.0, 0.0, 7.0, 4], [4.0, 1.0, 8.0, 4], [1.5, 1.3, 1.1, 4]]
def petit_test(X_train=[[2.0, 0.0, 7.0, 4, 5.2, 9.7], [3.0, 1.0, 5.0, 4, 1.0, .97], [0.0, 4.0, 5.0, 4, .1235, 2.58], [4.0, 0.0, 7.0, 4, 10, 4.78], [4.0, 1.0, 8.0, 4, 1, 5], [1.5, 1.3, 1.1, 4, 7, 6]],
            y_train=[0, 0, 0, 1, 1, 1],
            ids_train=[1, 2, 3, 4, 5],
            X_test=[[.8, 1.0, 0.0, 3, 0, 5], [1.3, 1.4, 1.7, 5, 8, 4], [4.0, 1.2, 5.5, 3, .3, .9], [3.3, 1.0, 1.4, 6, 11, 15], [1.5, 1.3, 1.1, 0, 14.5, 14.2]],
            y_test=[1, 0, 0, 1, 1],
            ids_test=[1, 2, 3, 4, 5],
               n_components=2, nu=1.4, use_VIP=False):
    # TODO : turn this into a unit test

    for t in [PLS_VARIANT.STANDARD, PLS_VARIANT.GINI, PLS_VARIANT.LOGIT_GINI, PLS_VARIANT.LOGIT]:
        pls = PLS(pls_type=t, n_components=n_components, nu=nu, centering_reduce=True, centering_reduce_rank=True, use_VIP=use_VIP)
        pls.fit(X_train, y_train)
        y_pred = pls.predict(X_test)
        #print('y_pred', y_pred)
        #from sklearn.metrics import accuracy_score
        metrics_values = evaluation(y_true=y_test, y_pred=y_pred)
        metrics_values += [y_test.count(i) for i in [0, 1]]
        print('%s (accuracy, err_0, err_1): %s' % (str(t), metrics_values))

if __name__ == "__main__":
  print("ginipls.main()")
  X_train =  [[2.0, 0.0, 7.0, 4, 5.2, 9.7], [3.0, 1.0, 5.0, 4, 1.0, .97], [0.0, 4.0, 5.0, 4, .1235, 2.58], [4.0, 0.0, 7.0, 4, 10, 4.78], [4.0, 1.0, 8.0, 4, 1, 5], [1.5, 1.3, 1.1, 4, 7, 6]]
  #X_train =  [[.8, 1.0, 0.0], [1.3, 1.4, 1.7], [4.0, 1.2, 5.5], [3.3, 1.0, 1.4], [1.5, 1.3, 1.1]]
  y_train= [0, 0, 0, 1, 1, 1]
  X_test =  [[.8, 1.0, 0.0, 3], [1.3, 1.4, 1.7, 5], [4.0, 1.2, 5.5, 3], [3.3, 1.0, 1.4, 6], [1.5, 1.3, 1.1, 0]]
  y_test= [1, 0, 0, 1, 1]

  # data_path = '../../data/processed'
  # trainfilename = op.join(data_path, 'doris0_CHI2_ATF-train.tsv')
  # testfilename = op.join(data_path, 'doris0_CHI2_ATF-test.tsv')
  # print('trainfilename',trainfilename)
  # print('testfilename',testfilename)  
  # X_train, y_train, h, ids_train = load_data(data=trainfilename, output_col='category')
  # X_test, y_test, h, ids_test = load_data(data=testfilename, output_col='category')
  # petit_test(X_train, y_train, ids_train, X_test, y_test, ids_test, n_components=2, nu=1.7, use_VIP=False)
  petit_test()

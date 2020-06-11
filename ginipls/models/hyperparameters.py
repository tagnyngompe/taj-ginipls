#TO RUN : python -m ginipls.models.hyperparameters
import numpy as np
import traceback #for exception
import itertools # for cartesian product of params ranges
from sklearn.model_selection import KFold
import os
from ginipls.models.ginipls import PLS, PLS_VARIANT
from ginipls.config import logger

## DIY : sklearn.kfold + for(nu_range) + fit + score
def select_pls_hyperparameters_with_cross_val(pls_type, X, y, nu_range, n_components_range, n_folds, best_expectable_score=1.):
  """"""
  X = np.asarray(X) #pour kf.split(X)
  y = np.asarray(y) 
  kf = KFold(n_splits=n_folds, shuffle=True)
  kf.get_n_splits(X)
  best_mean_score = 0.
  best_nu_ = 0.
  best_n_comp_ = 0.
  for nu_, n_comp_ in itertools.product(nu_range, n_components_range):
    mean_score = 0.
    fold_id = 0
    n_valid_folds = 0
    for train_index, test_index in kf.split(X):
      try:        
        logger.debug("FOLD %d (nu_==%.3f, n_comp_==%d) :: TRAIN:%s TEST:%s" % (fold_id,nu_,n_comp_, str(train_index), str(test_index)))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        gpls = PLS(pls_type = pls_type, nu=nu_, n_components=n_comp_)
        gpls.fit(X_train.tolist(), y_train.tolist())
        fold_score = gpls.score(X_test.tolist(), y_test.tolist())        
        logger.debug("fold_%d_score(nu==%.3f, n_comp==%d ) = %.3f" % (fold_id, nu_,n_comp_,fold_score))
        mean_score += fold_score
      except Exception:
        tb = traceback.format_exc()
        logger.error("ERROR for nu_==%.3f, n_comp_==%d [fold %d] : %s" % (nu_,n_comp_, fold_id, tb))
      else:
        n_valid_folds += 1
      finally:
        fold_id += 1
      #break
    mean_score = mean_score / n_folds
    logger.info("cv_mean_score(nu==%.3f, n_comp_==%d,n_valid_folds=%d) = %.3f" % (nu_,n_comp_,n_valid_folds,mean_score))
    if best_mean_score < mean_score:
      best_mean_score = mean_score
      best_nu_ = nu_
      best_n_comp_ = n_comp_
    if best_mean_score == best_expectable_score: #stop if the max score is reached
      break
    #break
  logger.info("best_mean_score = %.3f (with nu_==%.3f & n_comp_==%d)" % (best_mean_score, best_nu_, best_n_comp_))
  return {'nu': best_nu_, 'n_components': best_n_comp_}


if __name__ == "__main__":
  pls_type = PLS_VARIANT.GINI
  from sklearn import svm, datasets
  iris = datasets.load_iris()
  y = iris.target[50:] # classe{2,3}
  X = iris.data[50:]
  nu_min = 1
  nu_max = 3
  nu_step = 0.1
  nu_range = [i*nu_step for i in range(int(nu_min/nu_step),int(nu_max/nu_step))]
  logger.debug('nu_range =%s'% str(nu_range))
  n_components_min = 1
  n_components_max = len(X[0]) # nb de caractéristiques
  n_components_step = 1
  n_components_range = range(n_components_min, n_components_max, n_components_step)
  logger.debug('n_components_range =%s'% str(n_components_range))
  n_folds=3
  select_pls_hyperparameters_with_cross_val(pls_type, X, y, nu_range, n_components_range, n_folds, best_expectable_score=1.)
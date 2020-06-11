#TO RUN : python -m ginipls.models.hyperparameters
import numpy as np
import traceback #for exception
from ginipls.config import GLOBAL_LOGGER
logger = GLOBAL_LOGGER
# dataset
from sklearn import svm, datasets
iris = datasets.load_iris()
y = iris.target[50:]
X = iris.data[50:]
# convert labels to {0,1}
classes = list(set(y))
sorted(classes)
y = np.asarray([classes.index(i) for i in y])
# logger.info('classes', classes)
# logger.info('type(X)', type(X))
# logger.info('type(y)',type(y))
# logger.info('len(X)', len(X))
# logger.info('len(y)',len(y))
# logger.info('y', y)
#exit()

# small dataset
X_train_sm=[[2.0, 0.0, 7.0, 4, 5.2, 9.7], [3.0, 1.0, 5.0, 4, 1.0, .97], [0.0, 4.0, 5.0, 4, .1235, 2.58], [4.0, 0.0, 7.0, 4, 10, 4.78], [4.0, 1.0, 8.0, 4, 1, 5], [1.5, 1.3, 1.1, 4, 7, 6]]
y_train_sm=[0, 0, 0, 1, 1, 1]
ids_train=[1, 2, 3, 4, 5]
#X = np.asarray(X_train_sm)
#y = np.asarray(y_train_sm)

X_test_sm=[[.8, 1.0, 0.0, 3, 0, 5], [1.3, 1.4, 1.7, 5, 8, 4], [4.0, 1.2, 5.5, 3, .3, .9], [3.3, 1.0, 1.4, 6, 11, 15], [1.5, 1.3, 1.1, 0, 14.5, 14.2]]
y_test_sm=[1, 0, 0, 1, 1]
ids_test=[1, 2, 3, 4, 5]

# classifier
from ginipls.models.ginipls import PLS, PLS_VARIANT
nu_min = 1
nu_max = 2
nu_step = 0.1
nu_range = [i*nu_step for i in range(int(nu_min/nu_step),int(nu_max/nu_step))]
logger.info('nu_range = %s'% str(nu_range))

n_components_min = 1
n_components_max = min(10,len(X[0])) # nb de caractéristiques
logger.info('n_components_max=%d'%n_components_max)
n_components_step = 1
n_components_range = range(n_components_min, n_components_max, n_components_step)
logger.info('n_components_range =%s'% str(n_components_range))
#exit()

## with sklearn.GridSearchCV : Fail "unexpected parameter 'w'"
##DEF : https://stackoverflow.com/questions/19335165/what-is-the-difference-between-cross-validation-and-grid-search
# ginipls = PLS(pls_type = PLS_VARIANT.GINI)
# parameters = {'nu':nu_range}
# logger.info(parameters)
# from sklearn.model_selection import GridSearchCV
# clf = GridSearchCV(ginipls, parameters)
# clf.fit(X, y)
# logger.info(clf.cv_results_)

## with sklearn.cross_validation : Fail "unexpected parameter 'w'"
# from sklearn.model_selection import cross_validate
# for nu_ in nu_range:
#   ginipls = PLS(pls_type = PLS_VARIANT.GINI, nu=nu_)
#   cv_results = cross_validate(ginipls, X, y, cv=2)

## DIY : train-test-split + for(nu_range) + fit + score
# n_comp = 4
# for nu_ in nu_range:
  # gpls = PLS(pls_type = PLS_VARIANT.GINI, nu=nu_, n_components=n_comp)
  # gpls.fit(X_train_sm, y_train_sm)
  # score = gpls.score(X_test_sm, y_test_sm)
  # logger.info("score(nu==%.3f) = %.3f" % (nu_,score))


## DIY : sklearn.kfold + for(nu_range) + fit + score
THE_BEST_EXPECTABLE_SCORE = 1.
import itertools
from sklearn.model_selection import KFold
n_folds = min(3, len(X))
kf = KFold(n_splits=n_folds, shuffle=True)
kf.get_n_splits(X)
best_mean_score = 0.
best_nu_ = 0.
best_n_comp_ = 0.
for nu_, n_comp_  in itertools.product(nu_range, n_components_range):
  params_str = 'nu=%.3f, n_comp_=%d' % (nu_,n_comp_)
  mean_score = 0.
  fold_id = 0
  n_valid_folds = 0
  for train_index, test_index in kf.split(X):
    try:        
      logger.debug("[%s] [fold %d] TRAIN: %s, TEST: %s" % (params_str, fold_id, str(train_index), str(test_index)))
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]
      gpls = PLS(pls_type = PLS_VARIANT.GINI, nu=nu_, n_components=n_comp_)
      gpls.fit(X_train.tolist(), y_train.tolist())
      fold_score = gpls.score(X_test.tolist(), y_test.tolist())        
      logger.debug("[%s] [fold %d] score = %.3f" % (params_str, fold_id, fold_score))
      mean_score += fold_score
    except Exception:
      tb = traceback.format_exc()
      logger.error("[%s] [fold %d] ERROR : %s" % (params_str, fold_id, tb))
    else:
      n_valid_folds += 1
    finally:
      fold_id += 1
    #break
  #mean_score = mean_score / n_valid_folds if n_valid_folds > 0 else 0.
  mean_score = mean_score / n_folds
  logger.info("[%s] [CV n_folds_without_error=%d] mean_score = %.3f" % (params_str,n_valid_folds,mean_score))
  if best_mean_score < mean_score:
    best_mean_score = mean_score
    best_nu_ = nu_
    best_n_comp_ = n_comp_
  if best_mean_score == THE_BEST_EXPECTABLE_SCORE:
    break
  #break
logger.info("best_mean_score = %.3f (with nu_==%.3f & n_comp_==%d)" % (best_mean_score, best_nu_, best_n_comp_))

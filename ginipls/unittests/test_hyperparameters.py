#TO RUN : python -m ginipls.models.hyperparameters
import numpy as np
import traceback #for exception

# dataset
from sklearn import svm, datasets
iris = datasets.load_iris()
y = iris.target[50:]
X = iris.data[50:]
# convert labels to {0,1}
classes = list(set(y))
sorted(classes)
y = np.asarray([classes.index(i) for i in y])
# print('classes', classes)
# print('type(X)', type(X))
# print('type(y)',type(y))
# print('len(X)', len(X))
# print('len(y)',len(y))
# print('y', y)
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
nu_max = 3
nu_step = 0.1
nu_range = [i*nu_step for i in range(int(nu_min/nu_step),int(nu_max/nu_step))]
print('nu_range =', nu_range)

n_components_min = 1
n_components_max = len(X[0]) # nb de caractéristiques
n_components_step = 1
n_components_range = range(n_components_min, n_components_max, n_components_step)
print('n_components_range =', n_components_range)
#exit()

## with sklearn.GridSearchCV : Fail "unexpected parameter 'w'"
##DEF : https://stackoverflow.com/questions/19335165/what-is-the-difference-between-cross-validation-and-grid-search
# ginipls = PLS(pls_type = PLS_VARIANT.GINI)
# parameters = {'nu':nu_range}
# print(parameters)
# from sklearn.model_selection import GridSearchCV
# clf = GridSearchCV(ginipls, parameters)
# clf.fit(X, y)
# print(clf.cv_results_)

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
  # print("score(nu==%.3f) = %.3f" % (nu_,score))


## DIY : sklearn.kfold + for(nu_range) + fit + score
THE_BEST_EXPECTABLE_SCORE = 1.
import itertools
from sklearn.model_selection import KFold
n_folds = 10# len(X)
kf = KFold(n_splits=n_folds, shuffle=True)
kf.get_n_splits(X)
best_mean_score = 0.
best_nu_ = 0.
best_n_comp_ = 0.
for nu_, n_comp_  in itertools.product(nu_range, n_components_range):
  mean_score = 0.
  fold_id = 0
  n_valid_folds = 0
  for train_index, test_index in kf.split(X):
    try:        
      #print("TRAIN:", train_index, "TEST:", test_index)
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]
      gpls = PLS(pls_type = PLS_VARIANT.GINI, nu=nu_, n_components=n_comp_)
      gpls.fit(X_train.tolist(), y_train.tolist())
      #print(gpls.get_params())
      fold_score = gpls.score(X_test.tolist(), y_test.tolist())        
      #print("fold_%d_score(nu==%.3f) = %.3f" % (fold_id, nu_,fold_score))
      mean_score += fold_score
    except Exception:
      tb = traceback.format_exc()
      #print("ERROR for nu_==%.3f, n_comp_==%d [fold %d]" % (nu_,n_comp_, fold_id), tb)
    else:
      n_valid_folds += 1
    finally:
      fold_id += 1
    #break
  #mean_score = mean_score / n_valid_folds if n_valid_folds > 0 else 0.
  mean_score = mean_score / n_folds
  print("mean_score(nu==%.3f, n_comp_==%d,n_valid_folds=%d) = %.3f" % (nu_,n_comp_,n_valid_folds,mean_score))
  if best_mean_score < mean_score:
    best_mean_score = mean_score
    best_nu_ = nu_
    best_n_comp_ = n_comp_
  if best_mean_score == THE_BEST_EXPECTABLE_SCORE:
    break
  #break
print("best_mean_score = %.3f (with nu_==%.3f & n_comp_==%d)" % (best_mean_score, best_nu_, best_n_comp_))
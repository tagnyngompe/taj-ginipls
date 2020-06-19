#TO RUN : python -m ginipls.models.hyperparameters
import multiprocessing
import queue
import threading
import random
import traceback #for exception
import itertools # for cartesian product of params ranges
import numpy as np
from ginipls.data.data_utils import load_data
from ginipls.models.ginipls import PLS, PLS_VARIANT
from ginipls.config import GLOBAL_LOGGER as logger

NU_NAME = 'nu'
N_COMP_NAME = 'n_comp'
X_NAME = 'X'
Y_NAME = 'y'
PLS_TYPE_NAME = 'pls_type'
TRAIN_TEST_SPLITS_INDEX_PLS_TYPE_NAME = 'train_test_splits_index'
SCORE_NAME = 'score'


class HyperparameterEvaluatorThread(threading.Thread):
  exitFlag = 0
  def __init__(self, threadID, name, q, queueLock):
    threading.Thread.__init__(self)
    self.threadID = threadID
    self.name = name
    self.q = q
    self.queueLock = queueLock
  def run(self):
    logger.debug("Starting %s" % self.name)
    self.process_data()
    logger.debug("Exiting %s" % self.name)

  def process_data(self):
      while not HyperparameterEvaluatorThread.exitFlag:
          self.queueLock.acquire()
          if not self.q.empty():
              data = self.q.get()
              self.queueLock.release()
              logger.debug("%s processing nu_=%.3f, n_comp_=%d" % (self.name, data['nu'], data['n_comp']))
              data[SCORE_NAME] = evaluate_hyperparameters_values(nu_=data[NU_NAME], n_comp_=data[N_COMP_NAME], pls_type=data[PLS_TYPE_NAME],
                                                              X=data[X_NAME], y=data[Y_NAME],
                                                              train_test_splits_index=data[TRAIN_TEST_SPLITS_INDEX_PLS_TYPE_NAME])
          else:
              self.queueLock.release()

def kf_split(ytrue, nfolds, shuffle=True):
  labels_index = {}
  labels = set(ytrue)
  min_size = nfolds
  for l in labels:
    labels_index[l] = [i for i in range(len(ytrue)) if ytrue[i] == l]
    if min_size > len(labels_index[l]):
      min_size = len(labels_index[l])
  nfolds = min_size
  labels_folds = {}
  for l in labels_index:
    if shuffle:
      random.shuffle(labels_index[l])
    labels_folds[l] = {}
    for k in range(nfolds):
      labels_folds[l][k] = list()
    k = 0
    for idx in labels_index[l]:
      labels_folds[l][k % nfolds].append(idx)
      k += 1
  #print(labels_folds)
  traintest_splits = list()
  for k in range(nfolds):
    train_index = [y for l in labels_folds for i in range(nfolds) for y in labels_folds[l][i] if i != k]
    #print("train_index %d : %s" % (k, str(train_index)))
    test_index = [y for l in labels_folds for y in labels_folds[l][k] ]
    #print("test_index %d : %s" % (k, str(test_index)))
    traintest_splits.append((train_index, test_index))
  return traintest_splits
## DIY : sklearn.kfold + for(nu_range) + fit + score

def evaluate_hyperparameters_values(nu_, n_comp_, pls_type, X, y, train_test_splits_index):
  params_str = 'nu=%.3f, n_comp_=%d' % (nu_, n_comp_)
  n_folds = len(train_test_splits_index)
  mean_score = 0.
  fold_id = 0
  n_valid_folds = 0
  for train_index, test_index in train_test_splits_index:
    try:
      logger.debug("[%s] [fold %d] TRAIN: %s, TEST: %s" % (params_str, fold_id, str(train_index), str(test_index)))
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]
      gpls = PLS(pls_type=pls_type, nu=nu_, n_components=n_comp_)
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
  # mean_score = mean_score / n_valid_folds if n_valid_folds > 0 else 0.
  mean_score = mean_score / n_folds
  logger.debug("[%s] [CV n_folds_without_error=%d] mean_score = %.3f" % (params_str, n_valid_folds, mean_score))
  return mean_score

def select_pls_hyperparameters_with_cross_val(pls_type, X, y, nu_range, n_components_range, n_folds,
                                              only_the_first_fold=False, nb_threads=1):
  """"""
  X = np.asarray(X)  # pour kf.split(X)
  y = np.asarray(y)
  logger.debug('nu_range = %s'% str(nu_range))
  logger.debug('n_components_range =%s'% str(n_components_range))
  best_score, best_nu_, best_n_comp_ = 0., 0., 0.
  train_test_splits_index = kf_split(ytrue=y, nfolds=n_folds, shuffle=True)
  if only_the_first_fold:
    train_test_splits_index = train_test_splits_index[:1]
    logger.info("Running only on the fold 0 of the %d folds cross-validation" % n_folds)
  else:
    logger.info("Running on all of the %d folds of the cross-validation" % n_folds)
  # create the data structures
  #logger.info('train_test_splits_index : %s' % str(train_test_splits_index))
  #exit()
  data_list = []
  for nu_, n_comp_ in itertools.product(nu_range, n_components_range):
    data = {}
    data[NU_NAME] = nu_
    data[N_COMP_NAME] = n_comp_
    data[PLS_TYPE_NAME] = pls_type
    data[X_NAME] = X.copy()
    data[Y_NAME] = y.copy()
    data[TRAIN_TEST_SPLITS_INDEX_PLS_TYPE_NAME] = train_test_splits_index
    data_list.append(data)
  # create and named threads
  queueLock = threading.Lock()
  workQueue = queue.Queue(len(data_list))
  threads = []
  thread_id = 1
  for i in range(nb_threads):
    t_name = 'Thread-%d' % i
    thread = HyperparameterEvaluatorThread(thread_id, t_name, workQueue, queueLock)
    thread.start()
    threads.append(thread)
    thread_id += 1
  # Fill the queue
  queueLock.acquire()
  for data in data_list:
    workQueue.put(data)
  queueLock.release()
  # Wait for queue to empty
  while not workQueue.empty():
    pass
  # Notify threads it's time to exit
  HyperparameterEvaluatorThread.exitFlag = 1
  # Wait for all threads to complete
  for t in threads:
    t.join()
  logger.info("Exiting Main Hyperparameter Estimator Thread")
  for data in data_list:
    if best_score < data[SCORE_NAME]:
      best_nu_, best_n_comp_, best_score = data[NU_NAME],data[N_COMP_NAME], data[SCORE_NAME]
      print("score(nu=%.3f,n_comp=%d)=%.3f" % (data[NU_NAME],data[N_COMP_NAME], data[SCORE_NAME]))

  logger.info("best_score = %.3f (with nu_==%.3f & n_comp_==%d)" % (best_score, best_nu_, best_n_comp_))
  return best_nu_, best_n_comp_


if __name__ == "__main__":
  #python -m ginipls.models.hyperparameters
  pls_type = PLS_VARIANT.GINI
  from sklearn import datasets
  iris = datasets.load_iris()
  y = iris.target[50:] # classe{2,3}
  X = iris.data[50:]

  X_train_sm=[[2.0, 0.0, 7.0, 4, 5.2, 9.7], [3.0, 1.0, 5.0, 4, 1.0, .97], [0.0, 4.0, 5.0, 4, .1235, 2.58], [4.0, 0.0, 7.0, 4, 10, 4.78], [4.0, 1.0, 8.0, 4, 1, 5], [1.5, 1.3, 1.1, 4, 7, 6]]
  y_train_sm=[0, 0, 0, 1, 1, 1]
  #X = X_train_sm
  #y = y_train_sm

  train_data = "data/taj-sens-resultat/processed/acpa_cv0_train_tfidf12.tsv"
  X_train, y_train, h, ids_train = \
    load_data(
      data=train_data, output_col='@label',
      index_col="@id", col_sep="\t", header_row_num=0)
  #X, y = X_train, y_train

  nu_min = 1
  nu_max = 3
  nu_step = 0.1
  nu_range = [i*nu_step for i in range(int(nu_min/nu_step),int(nu_max/nu_step))]


  n_components_min = 1
  n_components_max = min(10,len(X[0])) # nb de caractéristiques
  n_components_step = 1
  n_components_range = range(n_components_min, n_components_max+1, n_components_step)

  n_folds=3
  nu, n_comp = select_pls_hyperparameters_with_cross_val(pls_type, X, y, nu_range, n_components_range, n_folds,
                                                         only_the_first_fold=True, nb_threads=multiprocessing.cpu_count()-1)
  logger.info("selected hyperparameters : nu=%.3f, n_comp=%d" % (nu, n_comp))

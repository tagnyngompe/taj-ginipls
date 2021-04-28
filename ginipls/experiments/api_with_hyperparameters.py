# encoding: utf-8
#from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSCanonical, PLSRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint as sp_randint
from sklearn.metrics import matthews_corrcoef
import random
from time import time
import sys


from ginipls.data.data_utils import load_ytrue_ypred_file
from ginipls.models.ginipls import PLS, PLS_VARIANT

default_n_components = 2
default_nu = 14
use_VIP=False

class PLSCanonical(PLSCanonical):
    def __init__(self, n_components=default_n_components, scale=True, algorithm="nipals",
                 max_iter=500, tol=1e-06, copy=True):
        super(PLSCanonical, self).__init__(n_components=n_components, scale=scale, algorithm=algorithm, max_iter=max_iter, tol=tol, copy=copy)
    def fit(self, X, Y):
        self.__Y_train_mean = np.mean(Y)
        return super(PLSCanonical, self).fit(X, Y)
        return self
    def predict(self, X, copy=True):
        Ypred = super(PLSCanonical, self).predict(X, copy)
        #print('Ypred', Ypred)
        return [1 if y_pred > self.__Y_train_mean else 0 for y_pred in Ypred]

class PLSRegression(PLSRegression):
    def __init__(self, default_n_components=2, scale=True,
                 max_iter=500, tol=1e-06, copy=True):
        super(PLSRegression, self).__init__(n_components=n_components, scale=scale, max_iter=max_iter, tol=tol, copy=copy)
    def fit(self, X, Y):
        self.__Y_train_mean = np.mean(Y)
        return super(PLSRegression, self).fit(X, Y)
        return self
    def predict(self, X, copy=True):
        Ypred = super(PLSRegression, self).predict(X, copy)
        #print('Ypred', Ypred)
        return [1 if y_pred > self.__Y_train_mean else 0 for y_pred in Ypred]

#/***********************************************************************/
#/*                         Dimension reduction                         */
#/***********************************************************************/

def lsa_learn_transformation(X_train, n_components=default_n_components, n_iter=10):
    from sklearn.decomposition import TruncatedSVD
    #training LSA
    #print('n_components', n_components)
    #print('X_train', X_train.shape)
    svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=42)
    svd.fit(X_train)
    #print(svd.explained_variance_ratio_)
    # transforming train and test sets into the reduced space
    #X_train_lsa = svd.fit_transform(X_train) 
    #X_test_lsa = svd.fit_transform(X_test) # X_test_lsa peut avoir moins de colonnes que X_train_lsa car les colonnes nulles ne sont pas conservées, ? comment trouver le bon nb de n_components?
    return svd

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " # ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

def latentDA_learn_transformation(X_train, n_components=default_n_components, max_iter=100):
    """
    http://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
    """
    from sklearn.decomposition import LatentDirichletAllocation   
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=max_iter,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)    
    t0 = time()
    print('latentDA_train_transform: learning topics ... ', end='', flush=True)
    lda.fit(X_train)
    print("\tdone in %0.3fs." % (time() - t0))    
    return lda


def linearDA_train_transformation(X_train, y_train):
    return linearDA_train(X_train, y_train)

def quadraticDA_train_transformation(X_train, y_train):
    return quadraticDA_train(X_train, y_train)

def our_standard_pls_train_transformation(X_train, y_train):
    return our_standard_pls_train(X_train, y_train)

def our_gini_pls_train_transformation(X_train, y_train):
    return our_gini_pls_train(X_train, y_train)

def our_logit_pls_train_transformation(X_train, y_train):
    return our_logit_pls_train(X_train, y_train)

def our_logit_gini_pls_train_transformation(X_train, y_train):
    return our_logit_gini_pls_train(X_train, y_train)

def sklearn_pls_canonical_train_transformation(X_train, y_train):
    return sklearn_pls_canonical_train(X_train, y_train)

def sklearn_pls_regression_train_transformation(X_train, y_train):
    return sklearn_pls_regression_train(X_train, y_train)

#/***********************************************************************/
#/*                               Classifier                            */
#/***********************************************************************/
# Utility function to report best scores for Model Selection
# http://chrisstrelioff.ws/sandbox/2015/06/25/decision_trees_in_python_again_cross_validation.html
def reportBestMetaparameters(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def report_besthyperpara(results_search_hyperpara, param_min_to_select_name):
    best_candidates = np.flatnonzero(results['rank_test_score'] == 1)
    best_candidate = 0
    if param_min_to_select_name != None:
        min_ = results['params'][0][param_min_to_select_name]
        for candidate in best_candidates:
            current_val = results['params'][candidate][param_min_to_select_name]
            if current_val < min_:
                min_ = current_val
                best_candidate = candidate
    best_metaparameters = results['params'][best_candidate]
    print("api.linearDA_train.best_metaparameters", best_metaparameters)
    return best_metaparameters


def run_randomsearch(clf, param_dist, X_train, y_train, k_cv=3, param_min_to_select_name = None, n_iter_search=20):
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=k_cv)
    random_search.fit(X_train, y_train)
    return report_besthyperpara(random_search.cv_results_, param_min_to_select_name)


def run_gridsearch(clf, param_grid, X, y, k_cv=5, param_min_to_select_name = None):
    """Run a grid search for best Decision Tree parameters.

    Args
    ----
    X -- features
    y -- targets (classes)
    clf -- scikit-learn Decision Tree
    param_grid -- [dict] parameter settings to test
    cv -- fold of cross-validation, default 5

    Returns
    -------
    top_params -- [dict] from report()
    """
    grid_search = GridSearchCV(clf,
                               param_grid=param_grid,
                               cv=k_cv)
    start = time()
    grid_search.fit(X, y)

    return report_besthyperpara(grid_search.cv_results_, param_min_to_select_name)

def our_standard_pls_train(X_train, y_train, n_components=default_n_components, nu=default_nu, centering_reduce=True):
    return __our_pls_train(X_train, y_train, PLS_VARIANT.STANDARD, n_components, nu, centering_reduce)

def our_gini_pls_train(X_train, y_train, n_components=default_n_components, nu=default_nu, centering_reduce=True):
    return __our_pls_train(X_train, y_train, PLS_VARIANT.GINI, n_components, nu, centering_reduce)

def our_logit_pls_train(X_train, y_train, n_components=default_n_components, nu=default_nu, centering_reduce=True):
    return __our_pls_train(X_train, y_train, PLS_VARIANT.LOGIT, n_components, nu, centering_reduce)

def our_logit_gini_pls_train(X_train, y_train, n_components=default_n_components, nu=default_nu, centering_reduce=True):
    return __our_pls_train(X_train, y_train, PLS_VARIANT.LOGIT_GINI, n_components, nu, centering_reduce)

def __our_pls_train(X_train, y_train, pls_type, n_components=default_n_components,
                    nu=default_nu, centering_reduce=True, use_VIP=use_VIP):
    #import ginipls
    clf = PLS(pls_type, n_components, nu, centering_reduce,use_VIP=use_VIP)
    clf.fit(X_train, y_train)    
    return clf

def sklearn_pls_canonical_train(X_train, y_train, n_components=default_n_components):
    if n_components > np.asarray(X_train).shape[1]:
        n_components = np.asarray(X_train).shape[1]-1
    clf = PLSCanonical(n_components=n_components)
    clf.fit(X_train, y_train)    
    return clf

def sklearn_pls_regression_train(X_train, y_train, n_components=default_n_components):
    if n_components > np.asarray(X_train).shape[1]:
        n_components = np.asarray(X_train).shape[1]-1
    clf = PLSRegression(n_components=n_components)
    clf.fit(X_train, y_train)    
    return clf

def linearDA_train(X_train, y_train):
    """
    http://www.science.smith.edu/~jcrouser/SDS293/labs/lab5-py.html
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()
    # MODEL SELECTION: specify meta-parameters and distributions to sample from
    num_attributes = np.asmatrix(X_train).shape[1]
    #print("api.linearDA_train.num_attributes", num_attributes)
    param_dist = {"solver": ["svd", "lsqr"],#, "eigen"],
                  #"shrinkage": [None, "auto"],
                  "n_components": sp_randint(2, num_attributes/2.0)
                  }
    ## run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=3)
    random_search.fit(X_train, y_train)
    #reportBestMetaparameters(random_search.cv_results_, 1)
    results = random_search.cv_results_
    best_candidates = np.flatnonzero(results['rank_test_score'] == 1)
    best_candidate = 0
    min_n_components = results['params'][0]["n_components"]
    for candidate in best_candidates:
        n_components = results['params'][candidate]["n_components"]
        if n_components < min_n_components:
            min_n_components = n_components
            best_candidate = candidate
    best_metaparameters = results['params'][best_candidate]
    print("api.linearDA_train.best_metaparameters", best_metaparameters)
    # init the model with the best metaparameters
    clf = LinearDiscriminantAnalysis(
        solver=best_metaparameters["solver"],
    #shrinkage=best_metaparameters["shrinkage"],
    n_components=best_metaparameters["n_components"])
    clf.fit(X_train, y_train)
    return clf
    
def quadraticDA_train(X_train, y_train):
    """
    http://www.science.smith.edu/~jcrouser/SDS293/labs/lab5-py.html
    """
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis     
    clf = QuadraticDiscriminantAnalysis()
    # MODEL SELECTION: specify meta-parameters and distributions to sample from
    param_dist = {"reg_param": [random.uniform(0,1) for i in range(1000000)]}

    best_metaparameters = run_randomsearch(clf, param_dist, X_train, y_train, k_cv=4)
    print("api.quadraticDA_train.best_metaparameters", best_metaparameters)
    # init the model with the best metaparameters
    clf = QuadraticDiscriminantAnalysis(reg_param=best_metaparameters["reg_param"])
    clf.fit(X_train, y_train)    
    return clf
    
def naivebayes_train(X_train, y_train):
    from sklearn.naive_bayes import BernoulliNB
    clf = BernoulliNB()
    clf.fit(X_train, y_train) # y doit être un simple np.array pas une matrice
    return clf

def knn_train(X_train, y_train, nb_neighbors=1): 
    from sklearn import neighbors
    clf = neighbors.KNeighborsClassifier()
    num_instances = np.asmatrix(X_train).shape[0]
    param_dist = {
        "n_neighbors": sp_randint(2, num_instances/2.0),
        "weights": ['uniform', 'distance'],
        "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
        "leaf_size": sp_randint(2, num_instances/2.0),
        "metric": ['euclidean', 'manhattan', 'chebyshev']
    }
    best_metaparameters = run_randomsearch(clf, param_dist, X_train, y_train, k_cv=4)
    clf = neighbors.KNeighborsClassifier(
        n_neighbors=best_metaparameters["n_neighbors"],
        weights=best_metaparameters["weights"],
        algorithm=best_metaparameters["algorithm"],
        leaf_size=best_metaparameters["leaf_size"],
        metric=best_metaparameters["metric"]
    )
    clf.fit(X_train, y_train)
    return clf


def svm_train(X_train, y_train):
    from sklearn import svm
    clf = svm.SVC()
    param_dist = {
        "C": [random.uniform(0,100) for i in range(100000)] + [0.1, 0.5, 1],
        "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
        "degree": sp_randint(2, 5),
        "gamma": [random.uniform(0,100) for i in range(100000)] + [0.1, 1, 'scale']
    }
    best_metaparameters = run_randomsearch(clf, param_dist, X_train, y_train, k_cv=4)
    clf = svm.SVC(
        C = best_metaparameters['C'],
        kernel = best_metaparameters['kernel'],
        degree = best_metaparameters['degree'],
        gamma = best_metaparameters['gamma']
    )
    clf.fit(X_train, y_train) # simple np.array pas une matrice      
    return clf


def decision_tree_train(X_train, y_train):
    from sklearn.tree import DecisionTreeClassifier
    #clf = DecisionTreeClassifier(criterion='gini', presort=True, random_state=0)
    clf = DecisionTreeClassifier()
    num_attributes = np.asmatrix(X_train).shape[1]
    num_instances = np.asmatrix(X_train).shape[0]
    param_dist = {
        "criterion": ['gini', 'entropy'],
        "splitter": ['best', 'random'],
          "min_samples_split": [random.uniform(0,1) for i in range(10)],
          "max_depth":[x for x in range (2, 10)] + [None],
          "min_samples_leaf": [x for x in range (1, num_instances)],
          "max_leaf_nodes": [x for x in range (2, num_instances)] + [None],
        "max_features": [random.uniform(0,1) for i in range(10)] + ['auto', 'sqrt', 'log2', None],
        "class_weight": ['balanced', None]
    }
    best_metaparameters = run_gridsearch(clf, param_dist, X_train, y_train, k_cv=4)
    clf = DecisionTreeClassifier(
        criterion=best_metaparameters['criterion'],
        splitter=best_metaparameters['splitter'],
        min_samples_split=best_metaparameters['min_samples_split'],
        max_depth=best_metaparameters['max_depth'],
        min_samples_leaf=best_metaparameters['min_samples_leaf'],
        max_leaf_nodes=best_metaparameters['max_leaf_nodes'],
        max_features=best_metaparameters['max_features'],
        class_weight=best_metaparameters['class_weight']
    )

    clf = clf.fit(X_train, y_train) # simple np.array pas une matrice
    return clf

def random_forest_train(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split = 2, random_state = 0)
    clf = clf.fit(X_train, y_train)
    return clf

def extra_trees_train(X_train, y_train):
    from sklearn.ensemble import ExtraTreesClassifier
    clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split = 2, random_state = None)
    clf = clf.fit(X_train, y_train)
    return clf

def balanced_accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):
    # print("y_true", y_true)
    # print("y_pred", y_pred)
    classes = set(y_true)
    b_acc = 0
    for c in classes:
        c = c
        nb_true_c = len([x for x in y_true if x == c]) # c expected
        #nb_pred_c = len([x for x in y_pred if x == c]) # c predicted
        nb_bien_pred_c = len([t for t,p in zip(y_true, y_pred) if t == c and t==p]) # c predicted
        b_acc += float(nb_bien_pred_c) / nb_true_c
        #print(c, nb_bien_pred_c, "/", nb_true_c)
    return b_acc/len(classes)

def evaluation(y_true, y_pred):
    #from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, cohen_kappa_score
    from sklearn.metrics import accuracy_score, recall_score, f1_score
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    errors = [1 - x for x in recall_score(y_true=y_true, y_pred=y_pred, average=None)]
    categories = list(set(y_true))
    #print("api.evaluation.categories", categories)
    f1_scores = [x for x in f1_score(y_true=y_true, y_pred=y_pred, labels=categories, average=None)]
    f1_macro_avg = f1_score(y_true=y_true, y_pred=y_pred, labels=categories, average="macro")
    # if len(errors) == 1:
    #     print("errors", errors)
    #     if(y_true[0] < 1):
    #         errors = errors + [0]
    #     else :
    #         errors = [0] + errors
    # print("errors", errors)
    mcc = matthews_corrcoef(y_true, y_pred)
    return [acc, b_acc]+errors+f1_scores+[f1_macro_avg, mcc]
    #print(cohen_kappa_score(y1=expected, y2=predicted))
    #print(confusion_matrix(y_true=expected, y_pred=predicted))
    #print(classification_report(y_true=expected, y_pred=predicted, digits=3)) 

def run_all_classifiers(X_train, y_train, X_test, y_test):
    # print('\tlinearDA:', evaluation(y_test, linearDA_train(X_train, y_train).predict(X_test)))
    # print('\tOurStandardPLS:', evaluation(y_test, our_standard_pls_train(X_train, y_train).predict(X_test)))
    # print('\tOurGiniPLS:', evaluation(y_test, our_gini_pls_train(X_train, y_train).predict(X_test)))
    # print('\tOurLogitPLS:', evaluation(y_test, our_logit_pls_train(X_train, y_train).predict(X_test)))
    # print('\tSklearnPLSCanonical:', evaluation(y_test, sklearn_pls_canonical_train(X_train, y_train).predict(X_test)))
    # print('\tSklearnPLSRegression:', evaluation(y_test, sklearn_pls_regression_train(X_train, y_train).predict(X_test)))
    # print('\tquadraticDA:', evaluation(y_test, quadraticDA_train(X_train, y_train).predict(X_test)))
    # print('\tGaussianNB:', evaluation(y_test, naivebayes_train(X_train, y_train).predict(X_test)))
    # import decimal
    # nb_voisins = int(str(round(decimal.Decimal((len(y_train)/4)+1))))
    # print('nb_voisins', nb_voisins)
    # print('\tKNN:', evaluation(y_test, knn_train(X_train, y_train, nb_neighbors=nb_voisins).predict(X_test)))
    # print('\tSVM:', evaluation(y_test, svm_train(X_train, y_train).predict(X_test)))
    y_p = decision_tree_train(X_train, y_train).predict(X_test)
    print(y_p)
    print('\tDecisionTreeClassifier:', evaluation(y_test, y_p))
    print('\tDecisionTreeClassifier:', evaluation(y_test, decision_tree_train(X_train, y_train).predict(X_test)))

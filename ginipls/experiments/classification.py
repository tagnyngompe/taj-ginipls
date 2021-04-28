# -*- coding: utf-8 -*-
import pandas as pd, numpy, sys
from time import time
import os.path as op

#from ginipls.experiments import api
from ginipls.experiments import api_with_hyperparameters as api

def learn_transformation(transformation, X_train, y_train):
    n_components = min(11, numpy.asarray(X_train).shape[1])+1/2
    switcher = {
        'lsa': lambda X_train, y_train: api.lsa_learn_transformation(X_train, n_components),
        'latentDA': lambda X_train, y_train: api.latentDA_learn_transformation(X_train, n_components),
        'linearDA': lambda X_train, y_train: api.linearDA_train_transformation(X_train, y_train),
        'OurStandardPLS': lambda X_train, y_train: api.our_standard_pls_train_transformation(X_train, y_train),
        'OurGiniPLS': lambda X_train, y_train: api.our_gini_pls_train_transformation(X_train, y_train),
        'SklearnPLSCanonical': lambda X_train, y_train: api.sklearn_pls_train_transformation(X_train, y_train),
    }
    if transformation not in switcher.keys():
        return None
    # Get the function from switcher dictionary
    func = switcher.get(transformation, lambda: "nothing")    
    # Execute the function
    return func(X_train, y_train)


def train(cls, X_train, y_train):
    switcher = {
        'linearDA' : api.linearDA_train,
        'quadraticDA' : api.quadraticDA_train,
        'GaussianNB' : api.naivebayes_train,
        'KNN' : api.knn_train,
        'SVM' : api.svm_train,
        'Tree' : api.decision_tree_train,
        'OurStandardPLS': api.our_standard_pls_train,
        'OurGiniPLS': api.our_gini_pls_train,
        'OurLogitPLS': api.our_logit_pls_train,
        'OurGiniLogitPLS': api.our_logit_gini_pls_train,
        'SklearnPLSCanonical': api.sklearn_pls_canonical_train,
        'SklearnPLSRegression': api.sklearn_pls_regression_train,
    }
    if cls not in switcher.keys():
        return None
    # Get the function from switcher dictionary
    func = switcher.get(cls, lambda: "nothing")    
    # Execute the function
    return func(X_train, y_train)

def load_train_test(trainfilename, testfilename, space_transformation=None):
    p = api.Preprocessor(classes_name_or_num='category')
    p.fit(trainfilename)
    # loading the datasets
    X_train, y_train, h = p.transform(trainfilename)
    X_test, y_test, h = p.transform(testfilename)
    if space_transformation is not None:
        space_transformer = learn_transformation(space_transformation, X_train, y_train)
        X_train = space_transformer.transform(X_train)
        X_test = space_transformer.transform(X_test)
    return X_train, y_train, X_test, y_test, h

def crossval(cvpath, localTsvDir, category, gw, lw, classifiers, nrep, space_transformation=None):
    #d_acc={'LinearDA':0, 'QuadraticDA':0,'GaussianNB':0,'KNN':0,'SVM':0, 'Tree':0}    
    d_acc = {}
    for clsName in classifiers:
        d_acc[clsName] = [0., 0, 0] # accuracy, erreur_0, erreur_1
    nb_instances = {0: 0., 1: 0.}
    for k in range(nrep):
        foldpath = cvpath+'/'+str(k)+'/'+localTsvDir
        trainfilename = op.join(foldpath, category+'-accepte_vs_'+category+'-rejette-'+str(k)+'_'+gw+'_'+lw+'_train.tsv')
        #print('trainfilename',trainfilename)
        testfilename = op.join(foldpath, category+'-accepte_vs_'+category+'-rejette-'+str(k)+'_'+gw+'_ATF_test.tsv')
        # loading the datasets 
        try:
            X_train, y_train, X_test, y_test, _ = load_train_test(trainfilename, testfilename, space_transformation)            
            for aClass in nb_instances.keys():
                nb_instances[aClass] +=len([x for x in y_test if x == aClass])
        except Exception as ex:
            print(ex)
            continue
        for clsName in classifiers:
            cls = None
            try:
                cls = train(clsName, X_train, y_train)
            except Exception as ex:
                print(ex)
                continue
            if cls is None:
                continue
            fold_eval_results = api.evaluation(y_test, cls.predict(X_test))
            #print('fold', k, clsName, fold_eval_results)
            result_len = len(fold_eval_results)
            d_acc[clsName][0] += fold_eval_results[0]
            if result_len < 3:
                #print("y_test",y_test)
                i = y_test[0] # les instances de tests sont toutes de la même classe
                d_acc[clsName][i+1] += fold_eval_results[1]
            else:
                for i in range(2):
                    d_acc[clsName][i+1] += fold_eval_results[i+1]
            #print('d_acc', k, clsName, d_acc)
    for clsName in classifiers:
        for i in range(3):
            d_acc[clsName][i] /= nrep     
    for aClass in nb_instances.keys():
                nb_instances[aClass] = round((nb_instances[aClass]+1)/nrep)
                
    #print('average', d_acc)
    evalResults = []    
    for clsName in classifiers:
        evalResults += [[gw, lw, str(space_transformation), clsName]+d_acc[clsName]+list(nb_instances.values())]
    return evalResults

def test_vectorizations(category, globalWeights, localWeights, classifiers, cvpath,
                        localTsvDir, nrep, space_transformation=None):
    accuracies = []    
    for gw in globalWeights:
        for lw in localWeights:                            
            accuracies += crossval(cvpath, localTsvDir, category, gw, lw, classifiers, nrep, space_transformation)
            print('.', end='', flush=True)
    return accuracies

def test_all_config(category, uniquementLeContexte, globalWeights, localWeights,
                    classifiers, cvbasepath, localTsvDir, nrep, space_transformations, results_path):
    print('\n',category,'....')
    cvpath = cvbasepath + category+"-"+str(nrep)+"foldcv"     
    accuracies = []
    for space_transformation in space_transformations:
        t0 = time()
        print(str(space_transformation)+'...', end='', flush=True) 
        accuracies += test_vectorizations(category, globalWeights, localWeights, classifiers, cvpath, localTsvDir, nrep, space_transformation)           
        print("done in %0.3fs.\t" % (time() - t0), end='', flush=True)
        results = pd.DataFrame(accuracies, columns=['globalWeight', 'localWeight', 'spaceTransformation', 'classifier', 'accuracy', 'erreur_rejette', 'erreur_accepte', '#rejette', '#accepte'])
        results = results.sort_values(by='accuracy', ascending=False)
        print('saving...', end='', flush=True) 
        results.to_csv(results_path, sep='\t')  
        
    return results

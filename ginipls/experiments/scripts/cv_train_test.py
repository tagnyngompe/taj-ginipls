# -*- coding: utf-8 -*-
import sys, getopt, time, os
from ginipls.experiments import api, classification
from ginipls.data import data_utils

# python evaluation.py -c acpa -o "@sens-resultat" -i "@id" -m ~/Documents/taj/wsp/chap4/data/cv-litige_motifs_dispositif/acpa_lemma_4_folds_cv/wd-3_2-tsv -r ../../../reports/resultats-litige-motifs-dispositif -k 4 -b False -v ~/Documents/taj/wsp/chap4/data/vectorizations.txt
# python -m ginipls.experiments.scripts.evaluation -c acpa -o "@label" -i "@id" -m data\cv-context\acpa_lemma_4_folds_cv\wd-1_1-tsv -r reports\cv-context -k 4 -b False -v data\vectorizations.txt
def main(argv):
    usage = """evaluation.py -c <categoryDmd> -o <output_colname> -d <instance_colname> -m <matrixFolderPath> -r <resultFolderPath> -k <nbFolds> -b <balance traindata?> -v <vectorizationsFilePath>\n
    e.g. python -m ginipls.experiments.scripts.evaluation -c acpa -o "@label" -i "@id" -m data/cv-context/acpa_lemma_4_folds_cv/wd-1_1-tsv -r reports/cv-context -k 4 -b False -v data/vectorizations.txt"""
    print(argv)
    if len(argv) == 0:
        print(usage)
        sys.exit(1)
    try:
        opts, args = getopt.getopt(argv,"hc:o:i:m:r:k:b:v:")
    except getopt.GetoptError as e:
        print("Exception: ", str(e.msg))
        print(usage)
        sys.exit(2)

    # initialisations
    output_colname = "@category"
    instance_colname = "@id"

    print("opts",opts)
    print("args",args)
    for opt, arg in opts:
        if opt == '-h':
            print(usage)
            sys.exit()
        elif opt in ("-c"):
            categoryDmd = arg
        elif opt in ("-m"):
            matrixFolderPath = arg
        elif opt in ("-r"):
            resultFolderPath = arg
        elif opt in ("-v"):
            vectorizationsFilePath = arg
        elif opt in ("-b"):
            balanced_traindata = eval(arg) # str2bool
        elif opt in ("-k", "--nrep"):
            nrep = int(arg)
        elif opt in ("-o", "--output_colname"): #optional, default value = "@category"
            output_colname = arg
        elif opt in ("-i", "--instance_colname"): #optional, default value = "@id"
            instance_colname = arg

    print(os.getcwd())
    classifiers = ['OurGiniPLS']#'linearDA', 'quadraticDA','GaussianNB','KNN','SVM', 'Tree', 'OurStandardPLS', 'OurGiniPLS', 'OurLogitPLS', 'OurGiniLogitPLS', 'SklearnPLSCanonical']
    space_transformations = [None]# None, 'OurGiniLogitPLS',  'lsa', 'linearDA', 'quadraticDA', ]
    localTsvDir = os.path.basename(matrixFolderPath)
    if not os.path.exists(resultFolderPath):
        os.makedirs(resultFolderPath)
    metrics_file_path= resultFolderPath+"/metrics_"+categoryDmd+"_"+str(nrep)+"_folds_cv_"+(localTsvDir.replace("-tsv", ".tsv"))
    predictionFolderPath = os.path.join(resultFolderPath, "predictions")
    if not os.path.exists(predictionFolderPath):
        os.makedirs(predictionFolderPath)
    with open(metrics_file_path, 'w') as metricsf:
        metricsf.write(
            "categoryDmd\tvectorization\tspace-transf\tclassifier\tacc\tbalanced-acc\terr-0\terr-1\tf1-0\tf1-1\tf1-macro-avg\tmcc\tnb0\tnb1\n")
        metricsf.close()
    nb_processed_config = 0
    t1 = time.time()
    with open(vectorizationsFilePath, 'r') as vf:
        for vectorization in vf:
            print(vectorization)
            vectorization = vectorization.strip()
            if len(vectorization) == 0:
                continue
            for space_transformation in space_transformations:
                for classifier in classifiers:
                    nb_processed_config +=1
                    config = categoryDmd + "_" + vectorization + "_" + str(space_transformation) + "_" + classifier
                    acc = 0
                    b_acc = 0
                    err_0 = 0
                    err_1 = 0
                    nb_0 = 0
                    nb_1 = 0
                    f1_0 = 0
                    f1_1 = 0
                    f1_macro = 0
                    mcc = 0 # matthews_corrcoef
                    t1cv = time.time()
                    for k in range(nrep):
                        prediction_file_path = os.path.join(resultFolderPath, "predictions", config + "_" + str(k) + "_" + localTsvDir + ".tsv")
                        #print(prediction_file_path, end=" ")
                        if not os.path.exists(prediction_file_path):
                            #print("to save")
                            predictionFolderPath = os.path.dirname(prediction_file_path)
                            if not os.path.exists(predictionFolderPath):
                                os.makedirs(predictionFolderPath)
                            #data_base_name = categoryDmd+"-rejette_vs_"+categoryDmd+"-accepte-"+str(k)+"_" + vectorization
                            data_base_name = "%s_cv%d_%s" % (categoryDmd, k, vectorization)
                            #config = categoryDmd+"-accepte_vs_"+categoryDmd+"-rejette-"+str(k)+"_" + gw + "_" + lw
                            train_data = os.path.join(matrixFolderPath, data_base_name + "_train.tsv")
                            if not os.path.exists(train_data):
                                print(train_data,"unreachable!")
                                continue
                            X_train, y_train, h, ids_train = \
                                data_utils.load_data(
                                    data=train_data, output_col=output_colname,
                                    index_col=instance_colname,col_sep="\t", header_row_num=0)
                            if balanced_traindata:
                                X_train, y_train, ids_train = data_utils.balance_data(X_train, y_train, ids_train)
                            test_data = os.path.join(matrixFolderPath, data_base_name + "_test.tsv")
                            #print(test_data)
                            X_test, y_test, h, ids_test = \
                                data_utils.load_data(
                                    data=test_data, output_col=output_colname,
                                    index_col=instance_colname, col_sep="\t", header_row_num=0)
                            y_test_pred = classification.train(classifier, X_train, y_train).predict(X_test)
                            data_utils.save_ytrue_and_ypred_in_file(ids_test, y_test, y_test_pred, prediction_file_path)
                        # else:
                        #     print(".")
                        docIds, y_true, y_pred = data_utils.load_ytrue_ypred_file(prediction_file_path)
                        #print(y_true,y_pred)
                        nb_0_k = len([y for y in y_true if y == 0])
                        nb_0 += nb_0_k
                        nb_1 += len(y_true) - nb_0_k
                        acc_k, b_acc_k, err_0_k, err_1_k, f1_0_k, f1_1_k, f1_macro_k, mcc_k = api.evaluation(y_true, y_pred)
                        acc += acc_k
                        b_acc += b_acc_k
                        err_0 += err_0_k
                        err_1 += err_1_k
                        f1_0 += f1_0_k
                        f1_1 += f1_1_k
                        f1_macro += f1_macro_k
                        mcc += mcc_k
                    if nb_processed_config % 50 == 0:
                        print(nb_processed_config, "configs processed in ", (time.time() - t1), "sec")
                        t1 = time.time()
                    with open(metrics_file_path, 'a') as metricsf:
                        metricsf.write(
                            config.replace("_", "\t") + "\t" + str(acc / nrep) + "\t" + str(b_acc / nrep) + "\t" + str(
                                err_0 / nrep) + "\t" + str(err_1 / nrep) + "\t" + str(f1_0 / nrep) + "\t" + str(
                                f1_1 / nrep) + "\t" + str(f1_macro / nrep) + "\t" + (mcc / nrep) + "\t" + str(nb_0) + "\t" + str(nb_1) + "\n")
                        metricsf.close()



if __name__ == "__main__":
    main(sys.argv[1:])

import sys, os
from taj.data_utils import load_evaluation_data2
from taj.api import evaluation

def ff(f):
    return '%.3f' % float(f)

if __name__ == '__main__':
    basePath = "/home/tagny/Documents/current-work/sens-resultat"
    metrics_file_path = basePath + "/resultat-xp/compare-stateofart.tsv"
    with open(metrics_file_path, 'w') as fw:
        fw.write("categoryDmd\tzone\tclassifier\tacc\tbalanced-acc\terr-0\terr-1\tf1-0\tf1-1\tf1-macro-avg\n")
        for catGmd in ["acpa", "concdel", "danais", "dcppc", "doris", "styx"]:
            for algo in ["fasttext", "nbsvm"]:
                #for zone in ["context", "demande_resultat_a_resultat_context", "litige_motifs_dispositif", "motifs"]:
                for zone in ["demande_resultat_a_resultat_context", "litige_motifs_dispositif", "motifs"]:
                    cvPredictionDir = basePath + "/cv-"+zone+"/"+catGmd+"_lemma_4_folds_cv/"+algo+"/prediction"#os.path.abspath(sys.argv[1])
                    print(cvPredictionDir)
                    predpaths = [os.path.join(cvPredictionDir, fname)
                                  for root, dirs, files in os.walk(cvPredictionDir)
                                  for fname in files
                                  # if name.endswith((".html", ".htm"))
                                  ]
                    acc = 0
                    b_acc = 0
                    err_0 = 0
                    err_1 = 0
                    f1_0 = 0
                    f1_1 = 0
                    f1_macro = 0
                    nrep = 0
                    for predpath in predpaths:
                        ids, y_true, y_pred = load_evaluation_data2(predpath, indexCol=None, yTrueCol=None, yPredCol=None, col_sep=" ")
                        if y_true is None or y_pred is None or len(y_true) == 0:
                            continue
                        print(predpath, y_true)
                        acc_k, b_acc_k, err_0_k, err_1_k, f1_0_k, f1_1_k, f1_macro_k = evaluation(y_true, y_pred)
                        acc += acc_k
                        b_acc += b_acc_k
                        err_0 += err_0_k
                        err_1 += err_1_k
                        f1_0 += f1_0_k
                        f1_1 += f1_1_k
                        f1_macro += f1_macro_k
                        nrep += 1
                    fw.write(catGmd+"\t"+zone+"\t"+algo+"\t"+ff(acc / nrep) + "\t" + ff(b_acc / nrep) + "\t" +
                          ff(err_0 / nrep) + "\t" + ff(err_1 / nrep) + "\t" + ff(f1_0 / nrep) + "\t" + ff(
                            f1_1 / nrep) + "\t" + ff(f1_macro / nrep)+"\n")
                    print("nrep = "+str(nrep))
# /home/tagny/Documents/current-work/sens-resultat/cv-context/acpa_lemma_4_folds_cv/fastext/prediction
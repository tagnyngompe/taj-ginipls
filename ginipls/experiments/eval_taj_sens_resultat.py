import os
import sys
import itertools
from ginipls.models.ginipls import PLS_VARIANT
from ginipls.data.data_utils import load_ytrue_ypred_file
from sklearn.metrics import f1_score, recall_score, precision_score

def main(dmd_category, wd):
    nfolds = 4
    predictions_dir = os.path.join(wd, 'predictions')
    local_weights = ['tf']
    global_weights = ['chi2', 'idf']
    pls_types = [PLS_VARIANT.LOGIT_GINI]
    nmin_ngram, nmaxngram = 1, 2
    ytrue_col = 'y_true'
    ypred_col = 'y_pred'
    index_col = 'docId'
    col_sep = '\t'

    for lw, gw, pls_type in itertools.product(local_weights, global_weights, pls_types):
        #print("Evaluation %s on %s%s" % (pls_type, lw, gw))
        for datasplit in ['train', 'test']:
            ytrue = list()
            ypred = list()
            ids = list()
            #print("[ %s ]" % datasplit)
            for id_fold in range(nfolds):
                fbasename = "%s_cv%d_%s_%s%s%d%d" % (dmd_category, id_fold, datasplit, lw, gw, nmin_ngram, nmaxngram)
                predfilename = os.path.join(predictions_dir, fbasename + '-%s.tsv' % str(pls_type.name).lower())
                fold_ids, fold_ytrue, fold_ypred = load_ytrue_ypred_file(predfilename, indexCol=index_col, yTrueCol=ytrue_col, yPredCol=ypred_col, col_sep=col_sep)
                ytrue += fold_ytrue
                ypred += fold_ypred
                ids += fold_ids
            #print(ids, ytrue, ypred)
            f1 = f1_score(ytrue, ypred, average='macro')
            r = recall_score(ytrue, ypred, average='macro')
            p = precision_score(ytrue, ypred, average='macro')
            errors = [float('%.3f' % (1 - x)) for x in recall_score(y_true=ytrue, y_pred=ypred, average=None)]
            print("%s, vecteurs=%s%s, %sPLS, %s, rappel_score_macro = %.3f, precision_score_macro = %.3f, f1_score_macro = %.3f, erreurs = %s" %
                  (dmd_category, lw, gw, pls_type.name, datasplit, r,p,f1, str(errors)) )


if __name__ == "__main__":
    # python -m ginipls.experiments.eval_taj_sens_resultat acpa data/taj-sens-resultat
    demand_category = sys.argv[1] if len(sys.argv) > 1 else 'acpa'
    wd = sys.argv[2] if len(sys.argv) > 2 else 'data/taj-sens-resultat'  # working dir
    main(demand_category, wd)
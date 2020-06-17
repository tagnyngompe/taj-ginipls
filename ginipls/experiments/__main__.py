# crossvalidation evaluation script
# python -m ginipls.experiments
import itertools
import os
from ginipls.config import GLOBAL_LOGGER as logger
from ginipls.__main__ import train_on_vectors, apply_on_vectors
from ginipls.models.ginipls import PLS_VARIANT


def main(dmd_category):
    nfolds = 4
    matrices_dir = 'data/processed/taj-sens-resultat'
    models_dir = 'data/models'
    predictions_dir = 'data/predictions'
    local_weights = ['tf']
    global_weights = ['chi2', 'idf']
    pls_types = [PLS_VARIANT.GINI]
    labels = ['rejette', 'accepte']
    nmin_ngram, nmaxngram = 1, 2

    nu_min = 1
    nu_max = 2
    nu_step = 0.1
    nu_range = [i * nu_step for i in range(int(nu_min / nu_step), int(nu_max / nu_step))]
    #nu_range = [1.1]

    n_components_min = 1
    n_components_max = 5  # nb de caractéristiques
    n_components_step = 1
    n_components_range = range(n_components_min, n_components_max, n_components_step)
    #n_components_range = [10]

    hyperparams_nfolds = 3
    crossval_hyperparam = True

    label_col='@label'
    index_col='@id'
    col_sep='\t'

    for lw, gw, pls_type in itertools.product(local_weights, global_weights, pls_types):
        logger.info("[%s] %d fold crossval for %s*%s, %s-PLS" % (dmd_category, nfolds, lw, gw, pls_type.name))
        for id_fold in range(nfolds):
            trainfbasename = "%s_cv%d_train_%s%s%d%d.tsv" % (dmd_category, id_fold, lw, gw, nmin_ngram, nmaxngram)
            testfbasename = "%s_cv%d_train_%s%s%d%d.tsv" % (dmd_category, id_fold, lw, gw, nmin_ngram, nmaxngram)
            trainfilename = os.path.join(matrices_dir, trainfbasename)
            testfilename = os.path.join(matrices_dir, testfbasename)
            logger.debug("%s %s" % (trainfilename,  str(os.path.isfile(trainfilename))))
            logger.debug("%s %s" % (testfilename, str(os.path.isfile(testfilename))))
            trainpredfilename = os.path.join(predictions_dir, trainfilename)
            testpredfilename = os.path.join(predictions_dir, testfbasename)
            classifierfbasename = "%s_cv%d_%s%s%d%d.%s" % (dmd_category, id_fold, lw, gw, nmin_ngram, nmaxngram, str(pls_type.name).lower())
            classifierfilename = os.path.join(models_dir, classifierfbasename)
            logger.debug("classifierfilename = %s" % classifierfilename)
            # train
            train_on_vectors(trainfilename, classifierfilename, label_col, index_col, col_sep, pls_type, nu_range, n_components_range, hyperparams_nfolds, crossval_hyperparam)
            # apply
            apply_on_vectors(trainfilename, classifierfilename, trainpredfilename, label_col, index_col, col_sep)
            apply_on_vectors(testfilename, classifierfilename, testpredfilename, label_col, index_col, col_sep)
            break
        #break


if __name__ == "__main__":
    # python -m ginipls.experiments styx
    import sys
    demand_category = sys.argv[1] if len(sys.argv) > 1 else 'acpa'
    main(demand_category)



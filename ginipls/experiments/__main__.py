# crossvalidation evaluation script
# python -m ginipls.experiments
import itertools
import os
from ginipls.config import GLOBAL_LOGGER as logger
from ginipls.__main__ import train_on_vectors, apply_on_vectors
from ginipls.models.ginipls import PLS_VARIANT


def main(dmd_category):
    nfolds = 4
    matrices_dir = 'data/processed/cv-litige_motifs_dispositif-matrices'
    models_dir = 'data/models'
    predictions_dir = 'data/predictions'
    local_weights = ['TF']
    global_weights = ['CHI2', 'DBIDF', 'DSIDF', 'IDF']
    pls_types = [PLS_VARIANT.GINI]
    labels = ['%s-rejette' % dmd_category, '%s-accepte' % dmd_category]

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

    label_col='@sens-resultat'
    index_col='@id'
    col_sep='\t'

    for lw, gw, pls_type in itertools.product(local_weights, global_weights, pls_types):
        logger.info("[%s] %d fold crossval for %s*%s, %s-PLS" % (dmd_category, nfolds, lw, gw, pls_type.name))
        for id_fold in range(nfolds):
            trainfilename = os.path.join(matrices_dir, "%s-%d_%s*%s_train.tsv" % ('_vs_'.join(labels),id_fold, gw, lw, ))
            testfilename = os.path.join(matrices_dir, "%s-%d_%s*%s_test.tsv" % ('_vs_'.join(labels),id_fold, gw, lw, ))
            logger.debug("%s %s" % (trainfilename,  str(os.path.isfile(trainfilename))))
            logger.debug("%s %s" % (testfilename, str(os.path.isfile(testfilename))))
            trainpredfilename = os.path.join(predictions_dir, "%s-%d_%s*%s_train.tsv" % ('_vs_'.join(labels), id_fold, gw, lw,))
            testpredfilename = os.path.join(predictions_dir, "%s-%d_%s*%s_test.tsv" % ('_vs_'.join(labels), id_fold, gw, lw,))
            classifierfilename = os.path.join(models_dir, "%s-%d_%s*%s.%s.model" % ('_vs_'.join(labels), id_fold, gw, lw,pls_type.name))
            logger.debug("classifierfilename = %s" % classifierfilename)
            # train
            train_on_vectors(trainfilename, classifierfilename, label_col, index_col, col_sep, pls_type, nu_range, n_components_range, hyperparams_nfolds, crossval_hyperparam)
            # apply
            apply_on_vectors(trainfilename, classifierfilename, trainpredfilename, label_col, index_col, col_sep)
            apply_on_vectors(testfilename, classifierfilename, testpredfilename, label_col, index_col, col_sep)
            #break
        #break


if __name__ == "__main__":
    # python -m ginipls.experiments styx
    import sys
    demand_category = sys.argv[1] if len(sys.argv) > 1 else 'acpa'
    main(demand_category)



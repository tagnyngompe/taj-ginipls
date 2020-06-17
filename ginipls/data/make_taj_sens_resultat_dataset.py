import os
import itertools
import sys

#dmdcategories = ['acpa', 'concdel','danais', 'dcppc', 'doris', 'styx']
nmin_ngram, nmaxngram = 1, 2
local_weights = ['tf']
global_weights = ['chi2', 'idf']
nfolds = 4



def main(dmd_category, wd):
    print(dmd_category)
    raw_path = os.path.join(wd, 'raw', dmd_category)
    preprocessed_path = os.path.join(wd, 'pp', '%s.tsv'%dmd_category)
    cv_path = os.path.join(wd, 'cv')
    vec_path = os.path.join(wd, 'processed')
    models_path = os.path.join(wd, 'models')
    #_dirpath = os.join(wd, 'pp')
    #os.system('''python -m ginipls.data.make_dataset select-data taj-sens-resultat-data "amende civile" "32-1 code de procédure civile + 559 code de procédure civile : pour procédure abusive" data/raw/txt-all/acpa data/raw/CASSANDRA.tsv data/raw/txt-oneclaim/acpa''')
    #os.system('python -m ginipls.data.make_dataset --logging preprocess taj-sens-resultat --language=fr --lowercase --lemmatizer=treetagger %s %s' % (raw_path, preprocessed_path))
    #os.system('python -m ginipls.data.make_dataset form-evaluation-data cv-traintest-from-dataset-file %d %s %s' % (nfolds, preprocessed_path, cv_path))
    for k, lw, gw in itertools.product(range(nfolds), local_weights, global_weights):
        for datasplit in ['train', 'test']:
            splittextdatapath = os.path.join(cv_path, '%s_cv%d_%s.tsv' %(dmd_category, k, datasplit))
            splitvecdatapath = os.path.join(vec_path, '%s_cv%d_%s_%s%s%d%d.tsv' %(dmd_category, k, datasplit, lw, gw, nmin_ngram, nmaxngram))
            splitvsmpath = os.path.join(models_path, '%s_cv%d_%s%s%d%d.vsm' %(dmd_category, k, lw, gw, nmin_ngram, nmaxngram))
            os.system('python -m ginipls.data.make_dataset --logging vectorize --vsm_scheme=%s%s --label_col=@label --index_col=@id --text_col=@text --ngram_nmin=%d --ngram_nmax=%d %s %s %s' % (lw,gw, nmin_ngram, nmaxngram, splittextdatapath, splitvsmpath, splitvecdatapath))
        #break

if __name__ == "__main__":
    # python -m ginipls.data.make_taj_sens_resultat_dataset acpa data/taj-sens-resultat
    demand_category = sys.argv[1] if len(sys.argv) > 1 else 'acpa'
    wd = sys.argv[2] if len(sys.argv) > 2 else 'data/taj-sens-resultat' #working dir
    main(demand_category, wd)
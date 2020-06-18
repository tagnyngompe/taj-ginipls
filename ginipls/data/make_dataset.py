# -*- coding: utf-8 -*-
import click
import os
from os.path import isdir, isfile, join
import pandas as pd
import csv
import pickle
import random
import unidecode # remove accents from str
import shutil # copy files
from ginipls.features.build_features import TF_IDF, TF_CHI2, InputError
from ginipls.features.build_features import TFIDF_SCHEME_NAME,TFCHI2_SCHEME_NAME
from ginipls.data.preprocess import TextPreprocessor, LANG_FR, LANG_EN, TREETAGGER_PREPROCESSOR, SPACY_PREPROCESSOR
from ginipls.config import GLOBAL_LOGGER as logger

ID_COL = "@id"
LABEL_COL = "@label"
TEXT_COL = "@text"

def collect_labels_docsfpaths(root='.'):
    """
    Collect labels and documents file path.
    """
    categories = filter(lambda d: isdir(join(root, d)), os.listdir(root))
    labels_docsfpaths = {}
    for c in categories:
        labels_docsfpaths[c] = [os.path.join(root, c, d) for d in filter(lambda d: isfile(join(root, c, d)), os.listdir(join(root, c)))]

    # categories and sub_categories are arrays,
    # categories would hold stuff like 'science', 'maths'
    # sub_categories would contain 'Quantum Mechanics', 'Linear Algebra', ...
    return labels_docsfpaths


def read_texts_file_to_texts_labels_lists(texts_csv_fpath, index_col = "@id", label_col="@label", text_col="@text", sep="\t"):
  """ Read a file of labeled texts and convert it into two list texts and labels.
  """
# def read_texts_file_to_list(texts_csv_fpath, label_col="@category", text_col="@text", sep="\t"):
  # """ Read a file of labeled texts and convert it into a list.
  # """
  logger.info("reading text from %s" % texts_csv_fpath)
  df = pd.read_csv(texts_csv_fpath, index_col=index_col,delimiter=sep)
  logger.debug("df\n%s" % str(df))
  return df.index.tolist(), df[text_col].tolist(), df[label_col].tolist()


def save_texts_words_weights_as_vectors_in_csv(texts_words_weights, vocabulary, out_vectors_fpath, index = None, labels=None, label_col="@label"):
  """"""
  os.makedirs(os.path.dirname(out_vectors_fpath), exist_ok=True)
  V = list(vocabulary)
  if index is None:
    index = range(len(texts_words_weights))
  # convert list of texts weights dicts into pandas df
  df = pd.DataFrame(index=index, columns=[label_col]+V)
  for i in range(len(index)):
    #texts_words_weights[i][label_col] = labels[i] if labels is not None else None
    df.loc[index[i]] = pd.Series({w : texts_words_weights[i][w] if w in texts_words_weights[i] else 0. for w in V})
    if labels is not None:
      df[label_col] = labels
  #logger.debug("df\n%s" % str(df))
  df.to_csv(out_vectors_fpath, index_label="@id", sep='\t', encoding='utf-8')
  logger.info("vectors saved to %s" % out_vectors_fpath)


def fit_vsm_from_texts_file(texts, labels, vsm_fpath, vsm_scheme, ngram_nmin, ngram_nmax):
  logger.info("Fitting the %s" % vsm_scheme)
  os.makedirs(os.path.dirname(vsm_fpath), exist_ok=True)
  if vsm_scheme == TFIDF_SCHEME_NAME:
    vsm = TF_IDF(ngram_nmin, ngram_nmax)
  elif vsm_scheme == TFCHI2_SCHEME_NAME:
    vsm = TF_CHI2(ngram_nmin, ngram_nmax)
  else:
    raise InputError(vsm_scheme, "Unsupported VSM scheme")
  vsm.fit(texts, labels)
  logger.debug("vsm.vocab_=%s" % str(vsm.vocab_))
  logger.info("%s Fitting End" % vsm_scheme)
  pickle.dump(vsm, open(vsm_fpath, 'wb'))
  logger.info("%s saved at %s" % (vsm_scheme, vsm_fpath))
  return vsm


@click.group(help="Organize a dataset into folds")
def form_evaluation_data():
    pass


@form_evaluation_data.command(help="from file with at least the column @label; each label dataset is split into nfolds subsets, and the corresponding subsets of the labes are merged to create folds")
@click.argument('nfolds', type=int)
@click.argument('in_datasetfilename', type=click.Path(exists=True))
@click.argument('dest_dirname', type=click.Path())
# python -m ginipls.data.make_dataset form-evaluation-data cv-traintest-from-dataset-file 4 data/interim/taj-sens-resultat-pp/acpa.tsv data/interim/taj-sens-resultat-cv
# python -m ginipls.data.make_dataset form-evaluation-data cv-traintest-from-dataset-file 4 data/interim/taj-sens-resultat-pp/concdel.tsv data/interim/taj-sens-resultat-cv
# python -m ginipls.data.make_dataset form-evaluation-data cv-traintest-from-dataset-file 4 data/interim/taj-sens-resultat-pp/danais.tsv data/interim/taj-sens-resultat-cv
# python -m ginipls.data.make_dataset form-evaluation-data cv-traintest-from-dataset-file 4 data/interim/taj-sens-resultat-pp/dcppc.tsv data/interim/taj-sens-resultat-cv
# python -m ginipls.data.make_dataset form-evaluation-data cv-traintest-from-dataset-file 4 data/interim/taj-sens-resultat-pp/doris.tsv data/interim/taj-sens-resultat-cv
# python -m ginipls.data.make_dataset form-evaluation-data cv-traintest-from-dataset-file 4 data/interim/taj-sens-resultat-pp/styx.tsv data/interim/taj-sens-resultat-cv
def cv_traintest_from_dataset_file(nfolds, in_datasetfilename, dest_dirname):
    os.makedirs(dest_dirname, exist_ok=True)
    datasetname = os.path.basename(in_datasetfilename).split('.')[0]
    with open(in_datasetfilename, 'r', encoding='utf-8') as csvfile:
        sr_lines = {}
        csvreader = csv.reader(csvfile, delimiter='\t')
        is_header_row = True
        for row in csvreader:
            if is_header_row:
                header_row = row
                lcn = row.index(LABEL_COL)
                label_colnum = lcn if lcn is not None else 1
                icn = row.index(ID_COL)
                id_colnum = icn if icn is not None else 0
                is_header_row = False
                continue
            sr = row[label_colnum]
            if not sr in sr_lines:
                sr_lines[sr] = list()
            sr_lines[sr].append(row)
        sr_folds = {}
        for sr in sr_lines:
            random.shuffle(sr_lines[sr])
            sr_folds[sr] = {}
            for k in range(nfolds):
                sr_folds[sr][k] = list()
            k = 0
            for line in sr_lines[sr]:
                sr_folds[sr][k % nfolds].append(line)
                k += 1
        for k in range(nfolds):
            trainfilename = os.path.join(dest_dirname, "".join([datasetname, '_cv%d_train.tsv' % k]))
            testfilename = os.path.join(dest_dirname, "".join([datasetname, '_cv%d_test.tsv' % k]))
            with open(testfilename, 'w', encoding='utf-8') as fw:
                fw.write("\t".join(header_row) + '\n')
                for sr in sr_folds:
                    fw.write("\n".join(["\t".join(row) for row in sr_folds[sr][k]]) + '\n')
            with open(trainfilename, 'w', encoding='utf-8') as fw:
                fw.write("\t".join(header_row) + '\n')
                for sr in sr_folds:
                    for k2 in range(nfolds):
                        if k2 == k:
                            continue
                        fw.write("\n".join(["\t".join(row) for row in sr_folds[sr][k2]]) + '\n')

@form_evaluation_data.command(help="from file with at least the column @label; each label dataset is split into nfolds subsets, and the corresponding subsets of the labes are merged to create folds")
@click.argument('nfolds', type=int)
@click.argument('in_datasetfilename', type=click.Path(exists=True))
@click.argument('dest_dirname', type=click.Path())
# python -m ginipls.data.make_dataset form-evaluation-data folds-from-dataset-file 4 data/interim/taj-sens-resultat-pp/acpa.tsv data/interim/taj-sens-resultat-folds
def folds_from_dataset_file(nfolds, in_datasetfilename, dest_dirname):
    os.makedirs(dest_dirname, exist_ok=True)
    datasetname = os.path.basename(in_datasetfilename).split('.')[0]
    with open(in_datasetfilename, 'r', encoding='utf-8') as csvfile:
        sr_lines = {}
        csvreader = csv.reader(csvfile, delimiter='\t')
        is_header_row = True
        for row in csvreader:
            if is_header_row:
                header_row = row
                lcn = row.index(LABEL_COL)
                label_colnum = lcn if lcn is not None else 1
                icn = row.index(ID_COL)
                id_colnum = icn if icn is not None else 0
                is_header_row = False
                continue
            sr = row[label_colnum]
            if not sr in sr_lines:
                sr_lines[sr] = list()
            sr_lines[sr].append(row)
        sr_folds = {}
        for sr in sr_lines:
            random.shuffle(sr_lines[sr])
            sr_folds[sr] = {}
            for k in range(nfolds):
                sr_folds[sr][k] = list()
            k = 0
            for line in sr_lines[sr]:
                sr_folds[sr][k%nfolds].append(line)
                k+=1
        print({sr: [[r[id_colnum] for r in sr_folds[sr][k]] for k in range(nfolds)] for sr in sr_folds})
        for k in range(nfolds):
            foldfilename = os.path.join(dest_dirname, "".join([datasetname, str(k), '.tsv']))
            print(foldfilename)
            with open(foldfilename, 'w', encoding='utf-8') as fw:
                fw.write("\t".join(header_row)+'\n')
                for sr in  sr_folds:
                    fw.write("\n".join(["\t".join(row) for row in sr_folds[sr][k]])+'\n')

@click.group(help="select specific data")
def select_data():
    pass


@select_data.command(help="select taj-sens-resultat dataset for a given claim type : decisions with a single claim.")
@click.argument('object', type=str)
@click.argument('norm', type=str)
@click.argument('in_decisions_dir', type=click.Path(exists=True))
@click.argument('claims_annotations_csv', type=click.Path(exists=True))
@click.argument('out_decisions_dir', type=click.Path())
@click.argument('object_colnum', type=int, default=3)
@click.argument('norm_colnum', type=int, default=5)
@click.argument('resultat_colnum', type=int, default=-4)
# python -m ginipls.data.make_dataset select-data taj-sens-resultat-data "amende civile" "32-1 code de procédure civile + 559 code de procédure civile : pour procédure abusive" data/raw/txt-all/acpa data/raw/CASSANDRA.tsv data/raw/txt-oneclaim/acpa
# python -m ginipls.data.make_dataset select-data taj-sens-resultat-data "dommages-intérêts" "1382 code civil : concurrence déloyale" data/raw/txt-all/concdel data/raw/CASSANDRA.tsv data/raw/txt-oneclaim/concdel
# python -m ginipls.data.make_dataset select-data taj-sens-resultat-data "dommages-intérêts" "1382 code civil + 32-1 code de procédure civile : en procédure abusive" data/raw/txt-all/danais data/raw/CASSANDRA.tsv data/raw/txt-oneclaim/danais
# python -m ginipls.data.make_dataset select-data taj-sens-resultat-data "déclaration de créance au passif de la procédure collective" "L622-24 code de commerce : déclaration de créance au passif de la procédure collective" data/raw/txt-all/dcppc data/raw/CASSANDRA.tsv data/raw/txt-oneclaim/dcppc
# python -m ginipls.data.make_dataset select-data taj-sens-resultat-data "dommages-intérêts" "principe de responsabilité pour trouble anormal de voisinage" data/raw/txt-all/doris data/raw/CASSANDRA.tsv data/raw/txt-oneclaim/doris
# python -m ginipls.data.make_dataset select-data taj-sens-resultat-data "dommages-intérêts" "700 Code de Procédure Civile" data/raw/txt-all/styx data/raw/CASSANDRA.tsv data/raw/txt-oneclaim/styx
def taj_sens_resultat_data(object, norm, in_decisions_dir, claims_annotations_csv, out_decisions_dir, object_colnum, norm_colnum, resultat_colnum):
    with open(claims_annotations_csv, 'r', encoding='utf-8') as csvfile:
        decision_resultats = {}
        csvreader = csv.reader(csvfile, delimiter='\t')
        for row in csvreader:
            #print(row[object_colnum], row[norm_colnum])
            if row[object_colnum] == object and row[norm_colnum] == norm:
                decision_id = ("".join([row[0].upper(), unidecode.unidecode(row[1][:3]).upper(), row[2].replace('/', '').upper(),".txt"]))
                if decision_id not in decision_resultats:
                    decision_resultats[decision_id] = list()
                decision_resultats[decision_id].append(row[resultat_colnum].strip())
        for decision_id in decision_resultats:
            if len(decision_resultats[decision_id]) == 1:
                #print(decision_id, decision_resultats[decision_id])
                sens_resultat = decision_resultats[decision_id][0]
                dest_dirname = os.path.join(out_decisions_dir, sens_resultat)
                os.makedirs(dest_dirname, exist_ok=True)
                src_decision_fname = os.path.join(in_decisions_dir, decision_id)
                dest_decision_fname = os.path.join(dest_dirname, decision_id)
                if not os.path.isfile(src_decision_fname):
                    print(src_decision_fname)
                else:
                    shutil.copy(src_decision_fname, dest_decision_fname)


@click.command(help = "build_features from a csv file with 1 (only texts) or 2 (texts with label or ids) or 3 columns ('doc_id\tlabel\ttext') and save the vectors in a csv file. Fit the vsm if vsm_fpath is None ")
@click.argument('in_datasetfilename', type=click.Path(exists=True))
@click.argument('vsm_fpath', type=click.Path(), required=False)
@click.argument('out_vectorsfilename', type=click.Path())
@click.option('--vsm_scheme', type=str, default='tf-idf', help='VSM scheme like tf-idf or tf-chi2', show_default=True)
@click.option('--ngram_nmin', default=1, help='Min number of words in a ngram.',metavar='<int>', show_default=True)
@click.option('--ngram_nmax', default=1, help='Max number of words in a ngram.',metavar='<int>', show_default=True)
@click.option('--label_col', type=str, help='labels column name', show_default=True)
@click.option('--index_col', type=str, help='texts ids column name[optional]', show_default=True, required=False)
@click.option('--text_col', type=str, help='texts content column name[optional]', show_default=True, required=False)
@click.option('--col_sep', type=str, default="\t", help='column delimiter', show_default=True)
def vectorize(in_datasetfilename, vsm_fpath, out_vectorsfilename, vsm_scheme, ngram_nmin, ngram_nmax, label_col, index_col, text_col, col_sep):
    # python -m ginipls.data.make_dataset --logging vectorize --vsm_scheme=tfidf --label_col=@label --index_col=@id --text_col=@text --ngram_nmax=2 data/interim/taj-sens-resultat-cv/acpa_cv0_train.tsv data/models/taj-sens-resultat/acpa_cv0.tfidf data/processed/taj-sens-resultat/acpa_cv0_train_tfidf12.tsv
    logger.info('building %s vectors from %s' % (vsm_scheme, in_datasetfilename))
    index, texts, labels = read_texts_file_to_texts_labels_lists(in_datasetfilename, index_col, label_col, text_col, col_sep)
    if os.path.isfile(vsm_fpath): # le modèle est déjà construit
        logger.info("le modèle est déjà construit")
        vsm = pickle.load(open(vsm_fpath, 'rb'))
    else:
        vsm = fit_vsm_from_texts_file(texts, labels, vsm_fpath, vsm_scheme, ngram_nmin, ngram_nmax)
    text_words_weights = vsm.transform(texts)
    save_texts_words_weights_as_vectors_in_csv(text_words_weights, vsm.vocab_, out_vectorsfilename, index=index, labels=labels)


@click.group(help = "preprocess datasets (e.g. from raw to interim i.e. a csv file with 2/3 columns 'doc_id\tlabel\ttext')")
def preprocess():
    pass


@preprocess.command("taj-sens-resultat", help="preprocess a dataset of court decision meaning polarity (sens du résultat) over a category of demands (a folder per labels, a file per document)")
@click.argument('in_datasetdirname', type=click.Path(exists=True))
@click.argument('out_datasetfilename', type=click.Path())
@click.option('--language', type=str, default=LANG_FR, help='texts language (%s, %s)' % (LANG_FR, LANG_EN), show_default=True)
@click.option('--lowercase/--no-lowercase', default=True, show_default=True)
@click.option('--lemmatizer', default=None, help=" ".join([SPACY_PREPROCESSOR, TREETAGGER_PREPROCESSOR]), show_default=True)
@click.option('--removepunct/--no-removepunct', help="remove punctuation and numbers", default=True, show_default=True)
@click.option('--removesinglechartoken/--no-removesinglechartoken', default=True, show_default=True)
def preprocess_taj_sens_resultat(in_datasetdirname, out_datasetfilename, language, lowercase, lemmatizer, removepunct, removesinglechartoken):
    # python -m ginipls.data.make_dataset --logging preprocess taj-sens-resultat --language=fr --lowercase --lemmatizer=treetagger data/raw/taj-sens-resultat/acpa data/interim/taj-sens-resultat-pp/acpa.tsv
    # python -m ginipls.data.make_dataset --logging preprocess taj-sens-resultat --language=fr --lowercase --lemmatizer=treetagger data/raw/taj-sens-resultat/concdel data/interim/taj-sens-resultat-pp/concdel.tsv
    # python -m ginipls.data.make_dataset --logging preprocess taj-sens-resultat --language=fr --lowercase --lemmatizer=treetagger data/raw/taj-sens-resultat/danais data/interim/taj-sens-resultat-pp/danais.tsv
    # python -m ginipls.data.make_dataset --logging preprocess taj-sens-resultat --language=fr --lowercase --lemmatizer=treetagger data/raw/taj-sens-resultat/dcppc data/interim/taj-sens-resultat-pp/dcppc.tsv
    # python -m ginipls.data.make_dataset --logging preprocess taj-sens-resultat --language=fr --lowercase --lemmatizer=treetagger data/raw/taj-sens-resultat/doris data/interim/taj-sens-resultat-pp/doris.tsv
    # python -m ginipls.data.make_dataset --logging preprocess taj-sens-resultat --language=fr --lowercase --lemmatizer=treetagger data/raw/taj-sens-resultat/styx data/interim/taj-sens-resultat-pp/styx.tsv
    os.makedirs(os.path.dirname(out_datasetfilename), exist_ok=True)
    text_preprocessor = TextPreprocessor(language, lowercase, lemmatizer, removepunct, removesinglechartoken)
    labels_docsfpaths = collect_labels_docsfpaths(root=in_datasetdirname)
    with open(out_datasetfilename, "w", encoding='utf-8') as fw:
        fw.write("\t".join([ID_COL, LABEL_COL, TEXT_COL])+'\n')
        for label in labels_docsfpaths:
            for fpath in labels_docsfpaths[label]:
                with open(fpath, "r", encoding='utf-8') as f:
                    text = text_preprocessor.process(f.read())
                    docid = os.path.basename(fpath).split('.')[0]
                    fw.write("%s\t%s\t%s\n" % (docid, label, text))


@click.group(help = "This is the command line interface to process datasets")
@click.option('--logging/--no-logging', default=True, help='column delimiter', show_default=True)
def cli(logging):
    logger.disabled = (not logging)


cli.add_command(form_evaluation_data)
cli.add_command(select_data)
cli.add_command(preprocess)
cli.add_command(vectorize)

if __name__ == '__main__':
    cli()

# preprocess taj-sens-resultat dataset
# python -m ginipls.data.make_dataset --logging preprocess taj-sens-resultat data/raw/taj-sens-resultat/acpa/train0 data/interim/acpa_train0.tsv
# python -m ginipls.data.make_dataset --logging preprocess taj-sens-resultat data/raw/taj-sens-resultat/acpa/test0 data/interim/acpa_test0.tsv
# vectorize texts files into matrices
# python -m ginipls.data.make_dataset --logging vectorize --vsm_scheme=tf-idf --label_col=@label --index_col=@id --text_col=@text --ngram_nmax=2 data/interim/acpa_train0.tsv data/models/acpa_train0_tf-idf_1-2grams.model data/processed/acpa_train0_tf-idf_1-2grams.tsv
# python -m ginipls.data.make_dataset --logging vectorize --vsm_scheme=tf-idf --label_col=@label --index_col=@id --text_col=@text --ngram_nmax=2 data/interim/acpa_test0.tsv data/models/acpa_train0_tf-idf_1-2grams.model data/processed/acpa_test0_tf-idf_1-2grams.tsv
# python -m ginipls.data.make_dataset --logging vectorize --vsm_scheme=tf-chi2 --label_col=@label --index_col=@id --text_col=@text --ngram_nmax=2 data/interim/acpa_train0.tsv data/models/acpa_train0_tf-chi2_1-2grams.model data/processed/acpa_train0_tf-chi2_1-2grams.tsv
# python -m ginipls.data.make_dataset --logging vectorize --vsm_scheme=tf-chi2 --label_col=@label --index_col=@id --text_col=@text --ngram_nmax=2 data/interim/acpa_test0.tsv data/models/acpa_train0_tf-chi2_1-2grams.model data/processed/acpa_test0_tf-chi2_1-2grams.tsv
# train models
# python -m ginipls train on-vectors --label_col=@label --index_col=@id --crossval_hyperparam --pls_type=GINI --n_components_range=[10] data/processed/acpa_train0_tf-idf_1-2grams.tsv data/models/acpa_train0_tf-idf_1-2grams_GINIPLS.model
# python -m ginipls train on-vectors --label_col=@label --index_col=@id --no-crossval_hyperparam --pls_type=GINI data/processed/acpa_train0_tf-chi2_1-2grams.tsv data/models/acpa_train0_tf-chi2_1-2grams_GINIPLS.model
# python -m ginipls apply on-vectors --label_col=category --index_col=@id data/processed/acpa_train0_tf-chi2_1-2grams.tsv data/models/acpa_train0_tf-chi2_1-2grams_GINIPLS.model"""
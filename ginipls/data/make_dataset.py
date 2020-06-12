# -*- coding: utf-8 -*-
import click
import os
from os.path import isdir, isfile, join
import pandas as pd
import pickle
from ginipls.features.build_features import TF_IDF, TF_CHI2
from ginipls.features.build_features import TFIDF_SCHEME_NAME,TFCHI2_SCHEME_NAME
from ginipls.data.preprocess import TextPreprocessor
from ginipls.config import GLOBAL_LOGGER as logger

LANG_FR = 'fr'
LANG_EN = 'en'
SPACY_MODEL_COMMON_SUFFIX = 'core_news_md'
SPACY_FR_MODEL = 'fr_core_news_md' # python -m spacy download fr_core_news_md
SPACY_EN_MODEL = 'en_core_web_md' # python -m spacy download en_core_web_md

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
    # python -m ginipls.data.make_dataset --logging vectorize --vsm_scheme=tf-idf --label_col=@label --index_col=@id --text_col=@text --ngram_nmax=2 data/interim/acpa_train0.tsv data/models/acpa_train0_tf-idf_1-2grams.model data/processed/acpa_train0_tf-idf_1-2grams.tsv
    # python -m ginipls.data.make_dataset --logging vectorize --vsm_scheme=tf-idf --label_col=@label --index_col=@id --text_col=@text --ngram_nmax=2 data/interim/acpa_test0.tsv data/models/acpa_train0_tf-idf_1-2grams.model data/processed/acpa_test0_tf-idf_1-2grams.tsv
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
@click.option('--lemmatize/--no-lemmatize', default=False, show_default=True)
def taj_sens_resultat(in_datasetdirname, out_datasetfilename, language, lowercase, lemmatize):
    # python -m ginipls.data.make_dataset --logging preprocess taj-sens-resultat data/raw/taj_sens_resultat/acpa/train0 data/interim/acpa_train0.tsv
    # python -m ginipls.data.make_dataset --logging preprocess taj-sens-resultat data/raw/taj_sens_resultat/acpa/test0 data/interim/acpa_test0.tsv
    text_preprocessor = TextPreprocessor(lowercase, lemmatize)
    labels_docsfpaths = collect_labels_docsfpaths(root=in_datasetdirname)
    for label in labels_docsfpaths:
        with open(out_datasetfilename, "w") as fw:
            fw.write("@id\t@label\t@text\n")
            for fpath in labels_docsfpaths[label]:
                with open(fpath, "r") as f:
                    text = text_preprocessor.process(f.read())
                    docid = os.path.basename(fpath).split('.')[0]
                    fw.write("%s\t%s\t%s\n" % (docid, label, text))

@click.group(help = "This is the command line interface to process datasets")
@click.option('--logging/--no-logging', default=True, help='column delimiter', show_default=True)
def cli(logging):
    logger.disabled = (not logging)

cli.add_command(preprocess)
cli.add_command(vectorize)

if __name__ == '__main__':
    cli()

# preprocess taj-sens-resultat dataset
# python -m ginipls.data.make_dataset --logging preprocess taj-sens-resultat data/raw/taj_sens_resultat/acpa/train0 data/interim/acpa_train0.tsv
# python -m ginipls.data.make_dataset --logging preprocess taj-sens-resultat data/raw/taj_sens_resultat/acpa/test0 data/interim/acpa_test0.tsv
# vectorize texts files into matrices
# python -m ginipls.data.make_dataset --logging vectorize --vsm_scheme=tf-idf --label_col=@label --index_col=@id --text_col=@text --ngram_nmax=2 data/interim/acpa_train0.tsv data/models/acpa_train0_tf-idf_1-2grams.model data/processed/acpa_train0_tf-idf_1-2grams.tsv
# python -m ginipls.data.make_dataset --logging vectorize --vsm_scheme=tf-idf --label_col=@label --index_col=@id --text_col=@text --ngram_nmax=2 data/interim/acpa_test0.tsv data/models/acpa_train0_tf-idf_1-2grams.model data/processed/acpa_test0_tf-idf_1-2grams.tsv
# python -m ginipls.data.make_dataset --logging vectorize --vsm_scheme=tf-chi2 --label_col=@label --index_col=@id --text_col=@text --ngram_nmax=2 data/interim/acpa_train0.tsv data/models/acpa_train0_tf-chi2_1-2grams.model data/processed/acpa_train0_tf-chi2_1-2grams.tsv
# python -m ginipls.data.make_dataset --logging vectorize --vsm_scheme=tf-chi2 --label_col=@label --index_col=@id --text_col=@text --ngram_nmax=2 data/interim/acpa_test0.tsv data/models/acpa_train0_tf-chi2_1-2grams.model data/processed/acpa_test0_tf-chi2_1-2grams.tsv
# train models
# python -m ginipls train on-vectors --label_col=@label --index_col=@id --crossval_hyperparam --pls_type=GINI --n_components_range=[10] data/processed/acpa_train0_tf-idf_1-2grams.tsv data/models/acpa_train0_tf-idf_1-2grams_GINIPLS.model
# python -m ginipls train on-vectors --label_col=@label --index_col=@id --no-crossval_hyperparam --pls_type=GINI data/processed/acpa_train0_tf-chi2_1-2grams.tsv data/models/acpa_train0_tf-chi2_1-2grams_GINIPLS.model
# python -m ginipls apply on-vectors --label_col=category --index_col=@id data/processed/acpa_train0_tf-chi2_1-2grams.tsv data/models/acpa_train0_tf-chi2_1-2grams_GINIPLS.model"""
# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from src.features.build_features import TF_IDF, TF_CHI2
from src.features.build_features import TFIDF_SCHEME_NAME,TFCHI2_SCHEME_NAME
from src.features.build_features import InputError
import os
import pickle

logger = logging.getLogger(__name__)

def read_texts_file_to_dict(texts_csv_fpath, label_col="@label", text_col="@text", sep="\t"):
  """ Read a file of labeled texts and convert it into a dict.  
  """
# def read_texts_file_to_list(texts_csv_fpath, label_col="@category", text_col="@text", sep="\t"):
  # """ Read a file of labeled texts and convert it into a list.  
  # """
  logger.info("reading text from %s" % texts_csv_fpath)
  df = pd.read_csv(texts_csv_fpath, delimiter=sep)
  return {c : [df.iloc[i][text_col] for i in df.index if df.iloc[i][label_col] == c] for c in set(df[label_col])}
  
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
  logger.debug("df\n%s" % str(df))
  df.to_csv(out_vectors_fpath, sep='\t', encoding='utf-8')
  logger.info("vectors saved to %s" % out_vectors_fpath)
  

def fit_vsm_from_texts_file(texts_csv_fpath, vsm_fpath, label_col, text_col, sep, vsm_scheme, ngram_nmin, ngram_nmax, lang):
  _, texts, labels = read_texts_file_to_texts_labels_lists(texts_csv_fpath, None, label_col, text_col, sep)
  logger.info("Fitting the %s" % vsm_scheme)
  if vsm_scheme == TFIDF_SCHEME_NAME:
    vsm = TF_IDF(ngram_nmin, ngram_nmax, lang)
  elif vsm_scheme == TFCHI2_SCHEME_NAME:
    vsm = TF_CHI2(ngram_nmin, ngram_nmax, lang)
  else:
    raise InputError(vsm_scheme, "Unsupported VSM scheme")
  vsm.fit(texts, labels)
  logger.debug("vsm.vocab_=%s" % str(vsm.vocab_))
  logger.info("%s Fitting End" % vsm_scheme)
  pickle.dump(vsm, open(vsm_fpath, 'wb'))
  logger.info("%s saved at %s" % (vsm_scheme, vsm_fpath))
  return vsm

@click.command(options_metavar='<options>')
@click.argument('input_train_texts_fpath', type=click.Path(exists=True))
@click.argument('vsm_fpath', type=click.Path())
@click.argument('output_train_vectors_fpath', type=click.Path())
@click.option('--ngram_nmin', default=1, help='Min number of words in an ngram.',metavar='<int>', show_default=True)
@click.option('--ngram_nmax', default=1, help='Max number of words in an ngram.',metavar='<int>', show_default=True)
@click.option('--lang', default='en', help='Text language (fr, en).',metavar='<str>', show_default=True)
@click.option('--vsm_scheme', default='tf-idf', help='VSM scheme like tf-idf or tf-chi2',metavar='<str>', show_default=True)
@click.option('--label_col', default='@label', help='labels column name', show_default=True)
@click.option('--text_col', default='@text', help='texts column name', show_default=True)
@click.option('--index_col', default='@id', help='texts ids column name', show_default=True)
@click.option('--sep', default='\t', help='column delimiter', show_default=True)

def main(input_train_texts_fpath, vsm_fpath, output_train_vectors_fpath, ngram_nmin, ngram_nmax, lang, vsm_scheme, label_col, text_col, index_col,  sep):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info('making final data set from raw data : build vectors')
    if os.path.isfile(vsm_fpath): # le modèle est déjà construit
      logger.info("le modèle est déjà construit")
      vsm = pickle.load(open(vsm_fpath, 'rb'))
    else:      
      vsm = fit_vsm_from_texts_file(input_train_texts_fpath, vsm_fpath, label_col, text_col, sep, vsm_scheme, ngram_nmin, ngram_nmax, lang)
    index, texts, labels = read_texts_file_to_texts_labels_lists(input_train_texts_fpath, index_col, label_col, text_col, sep)
    text_words_weights = [vsm.transform(text) for text in texts]
    save_texts_words_weights_as_vectors_in_csv(text_words_weights, vsm.vocab_, output_train_vectors_fpath, index=index, labels=labels)
    # print(read_texts_file_to_texts_labels_lists(input_train_texts_fpath, index_col, label_col, text_col, sep))

# TO EXECUTE : python -m src.data.make_dataset data\raw\train_texts.tsv models\tfidf.vsm.pt data\processed\train.tfidf.vec.tsv --lang=en
#python -m src.data.make_dataset data\raw\train_texts.tsv models\tfchi2.vsm.pt data\processed\train.tfchi2.vec.tsv --lang=en --vsm_scheme=tfchi2

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

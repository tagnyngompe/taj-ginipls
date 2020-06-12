from collections import Counter
from math import log, sqrt
from ginipls.config import GLOBAL_LOGGER as logger

TFIDF_SCHEME_NAME = 'tf-idf'
TFCHI2_SCHEME_NAME = 'tf-chi2'

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class VSM:
  def __init__(self, ngram_nmin=1, ngram_nmax=1):
    assert ngram_nmin > 0
    assert ngram_nmin <= ngram_nmax
    self.ngram_nmin = ngram_nmin
    self.ngram_nmax = ngram_nmax
  @staticmethod
  def ngrams(text, n):
    text = text.split()
    #logger.debug('text = %s' % text)
    return [" ".join(text[i:i+n]) for i in range(len(text)-n+1)]
  def convert_text_to_words_list(self,text):
    return [an_ngram for n in range(self.ngram_nmin, self.ngram_nmax+1) for an_ngram in VSM.ngrams(text, n)]
  @staticmethod
  def count_words_occurrences_in_text(text_words):
    return Counter(text_words)
  @staticmethod
  def count_words_doc_freq_in_texts(texts_words):
    return Counter([w for t in texts_words for w in set(t)])
  @staticmethod
  def normalize_weights_by_cosine(words_weights):
    factor = sqrt(sum([words_weights[w]**2 for w in words_weights]))
    #logger.debug("factor %.3f" % factor)
    return {w : words_weights[w] * factor for w in words_weights}
  def compute_tf_weights(self,text):
    text_words = self.convert_text_to_words_list(text)
    text_nb_words = len(text_words)
    return {w : c / text_nb_words for w,c in VSM.count_words_occurrences_in_text(text_words).items()}
  def compute_local_weights(self, text):
    pass
  def fit(self, texts, labels=None):
    pass
  def transform(self, texts):
    """ compute the LOCAL x GLOBAL weight of each words)"""
    #logger.debug("texts %s" % str(texts))
    texts_words_lweights = [self.compute_local_weights(text) for text in texts]
    return [VSM.normalize_weights_by_cosine({w : words_lweights[w]*self.words_gweights[w] for w in words_lweights if w in self.words_gweights}) for words_lweights in texts_words_lweights]
  def fit_transform(self, texts, labels=None):
    self.fit(texts, labels)
    return self.transform(texts)


class TF_IDF(VSM):
  def fit(self, texts, labels=None):
    """ Compute IDF weights
    Args:
      texts (list): a list of texts
    """
    N = len(texts)
    texts_words = [self.convert_text_to_words_list(t) for t in texts]
    logger.debug('texts_words = %s' % str(texts_words))
    self.words_gweights = {w : log( N / df ) for w, df in VSM.count_words_doc_freq_in_texts(texts_words).items()}
    self.vocab_ = sorted(list(self.words_gweights.keys()))
    logger.info("self.vocab_ = %s" % str(self.vocab_))
  def compute_local_weights(self, text):
    return self.compute_tf_weights(text)


class TF_CHI2(VSM):
  @staticmethod
  def fill_class_counts_with_zero(Countswc, V):
    for c in Countswc:
      for w in V:
        if not w in Countswc[c]:
          Countswc[c][w] = 0
    return Countswc
  def fit(self, texts, labels):
    """ Compute CHI2 weights
    Args:
      labels_texts (dict): texts list for each class
    """
    labels_texts = {c : [texts[i] for i in range(len(texts)) if labels[i]==c] for c in set(labels)}
    #logger.debug('labels_texts = %s' % str(labels_texts))
    labels_texts_words = {c : [self.convert_text_to_words_list(t) for t in labels_texts[c]]  for c in labels_texts}
    #logger.debug('labels_texts_words = %s' % str(labels_texts_words))
    C = labels_texts.keys()
    #logger.debug(labels_texts_words)
    all_texts_words = [t for c in labels_texts_words for t in labels_texts_words[c]]
    #logger.debug('all_texts_words = %s' % str(all_texts_words))
    #logger.debug(all_texts_words)
    self.vocab_ = sorted(list(set([w for tw in all_texts_words for w in tw]))) # vocabulaire
    logger.info("self.vocab_ = %s" % str(self.vocab_))
    #logger.debug('V=%s' % str(V))
    N = len(all_texts_words) #nb total de tex
    logger.debug('N=%d' % N)
    Nc = {c : len(labels_texts[c]) for c in C}
    logger.debug('Nc=%s' % str(Nc))
    Nc_ = {c : N - Nc[c] for c in C}
    logger.debug('Nc_%s' % str(Nc_))
    Nw = VSM.count_words_doc_freq_in_texts(all_texts_words)# nombre total de texts contenant w
    logger.debug('Nw%s' % str(Nw))
    Nw_ = {w : N - Nw[w] for w in self.vocab_}# nombre total de texts ne contenant w
    logger.debug('Nw_%s' % str(Nw_))
    Ncw = {c : VSM.count_words_doc_freq_in_texts(labels_texts_words[c]) for c in C} # nb de texte de c contenant w
    Nwc = {w : {c: Ncw[c][w] for c in C} for w in self.vocab_}
    logger.debug('Nwc%s' % str(Nwc))
    Nwc_ = {w : {c : Nw[w] - Nwc[w][c] for c in C} for w in self.vocab_} # nb de texte hors de c contenant w
    logger.debug('Nwc_%s' % str(Nwc_))
    Nw_c = {w : {c : Nc[c] - Nwc[w][c] for c in C} for w in self.vocab_} # nb de texte de c ne contenant pas w
    logger.debug('Nw_c%s' % str(Nw_c))
    # logger.debug('Nw_c', Nw_c)
    Nw_c_ = {w : {c : Nc_[c] - Nwc_[w][c] for c in C} for w in self.vocab_} # nb de textes hors de c qui ne contiennent pas w
    logger.debug('Nw_c_%s' % str(Nw_c_))
    #logger.debug('Nw_c_', Nw_c_)
    chi2wc = {} # score de CHI2 de chaque w pour chaque c  chi2wc[class][term]
    for w in self.vocab_:
      chi2wc[w] = {}
      for c in C:
        den = ( Nw[w] * Nw_[w] * Nc[c] * Nc_[c] )
        chi2wc[w][c] = N * ( (Nwc[w][c] * Nw_c_[w][c]) - (Nwc_[w][c] * Nw_c[w][c]) ) / den if den != 0 else 0
    #logger.debug('chi2wc', chi2wc)
    logger.debug('chi2wc%s' % str(chi2wc))
    self.words_gweights = {w : max(chi2wc[w].values()) for w in self.vocab_}
    #logger.debug('self.words_gweights', self.words_gweights) 
  def compute_local_weights(self, text):
    return self.compute_tf_weights(text)


# TO EXECUTE FROM ../src : python -m ginipls.features.build_features

if __name__ == "__main__":
  #log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'    
  #logging.basicConfig(level=logging.INFO, format=log_fmt)
  #logger = logging.getLogger(__name__)
  texts = ['the truck is driven on the highway', 'the car is driven on the road', 'the man went out for a walk', 'the children sat around the fire']
  labels = [0,0,1,1]
  ########## TF-IDF
  vsm = TF_IDF(1,1)
  train_vec = vsm.fit_transform(texts)
  logger.info('TF-IDF=%s' % str(train_vec))
  ########## TF_CHI2
  vsm = TF_CHI2(1,1)
  vsm.fit(texts, labels)
  logger.info('TF-CHI2=%s' % str(vsm.transform(texts)))
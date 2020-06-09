from collections import Counter
from math import log, sqrt
from nltk import ngrams

class TextVSM:
  def __init__(self, ngram_nmin=1, ngram_nmax=1):
    assert ngram_nmin > 0
    assert ngram_nmin <= ngram_nmax
    self.ngram_nmin = ngram_nmin
    self.ngram_nmax = ngram_nmax
  def convert_text_to_words_list(self,text):
    return [" ".join(w) for n in range(self.ngram_nmin, self.ngram_nmax+1) for w in ngrams(text.lower().split(), n)]
  @staticmethod
  def count_words_occurrences_in_text(text_words):
    return Counter(text_words)
  @staticmethod
  def count_words_doc_freq_in_texts(texts_words):
    return Counter([w for t in texts_words for w in set(t)])
  @staticmethod
  def normalize_weights_by_cosine(words_weights):
    factor = sqrt(sum([words_weights[w]**2 for w in words_weights]))
    print("factor", factor)
    return {w : words_weights[w] * factor for w in words_weights}
  def compute_tf_weights(self,text):
    text_words = self.convert_text_to_words_list(text)
    text_nb_words = len(text_words)
    return {w : c / text_nb_words for w,c in TextVSM.count_words_occurrences_in_text(text_words).items()}
  def compute_local_weights(self, text):
    pass
  def fit(self, texts):
    pass
  def transform(self, text):
    """ compute the LOCAL x GLOBAL weight of each words)"""
    words_lweights = self.compute_local_weights(text)
    print(text, words_lweights)
    #return TextVSM.normalize_weights_by_cosine({w : words_lweights[w]*self.words_idf[w] for w in words_lweights if w in self.words_gweights})


class TF_IDF(TextVSM):
  def fit(self, texts):
    """ Compute IDF weights
    Args:
      texts (list): a list of texts
    """
    N = len(texts)
    self.words_gweights = {w : log( N / df ) for w, df in TextVSM.count_words_doc_freq_in_texts([self.convert_text_to_words_list(t) for t in texts]).items()}
  def compute_local_weights(self, text):
    return self.compute_tf_weights(text)


class TF_CHI2(TextVSM):
  def fit(self, classes_texts):
    """ Compute CHI2 weights
    Args:
      classes_texts (dict): texts list for each class
    """
    classes_texts_words = {c : [self.convert_text_to_words_list(t) for t in classes_texts[c]]  for c in classes_texts}
    #print(classes_texts_words)
    all_texts_words = [t for c in classes_texts_words for t in classes_texts_words[c]]
    N = len(all_texts_words) #nb total de texts
    Nc = {c : len(classes_texts[c]) for c in classes_texts}
    print('Nc', Nc)
    Nc_ = {c : N - Nc[c] for c in classes_texts}
    print('Nc_', Nc_)
    #print(all_texts_words)
    Nw = TextVSM.count_words_doc_freq_in_texts(all_texts_words)# nombre total de texts contenant w
    print('Nw', Nw)
    Nw_ = {w : N - Nw[w] for w in Nw}# nombre total de texts ne contenant w
    print('Nw_', Nw_)
    Nwc = {c : TextVSM.count_words_doc_freq_in_texts(classes_texts_words[c]) for c in classes_texts_words}
    print('Nwc', Nwc)
    Nwc_ = {c : {w : Nw[w] - Nwc[c][w] for w in Nwc[c]} for c in classes_texts_words} 
    print('Nwc_', Nwc_)
    Nw_c = {c : {Nc[c] - Nwc[c][w] for w in Nwc[c]} for c in classes_texts_words}
    print('Nw_c', Nwc_)
    
    self.words_gweights = {w : N( (N) )


# TO EXECUTE FROM ../src : python -m src.features.build_features

if __name__ == "__main__":
  texts = {0 : ['the man went out for a walk'], 
           1 : ['the children sat around the fire']}
  vsm = TF_CHI2(1,3)
  #print(vsm.convert_text_to_words_list(texts[0][0]))
  vsm.fit(texts)
  #print(vsm.transform(texts[0]))
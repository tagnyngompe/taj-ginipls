import spacy
import treetaggerwrapper
from spacy_lefff import LefffLemmatizer, POSTagger # https://pypi.org/project/spacy-lefff/
from ginipls.config import GLOBAL_LOGGER as logger

LANG_FR = 'fr'
LANG_EN = 'en'
SPACY_MODEL_COMMON_SUFFIX = 'core_news_md'
SPACY_FR_MODEL = 'fr_core_news_md' # python -m spacy download fr_core_news_md
SPACY_EN_MODEL = 'en_core_web_md' # python -m spacy download en_core_web_md
SPACY_PREPROCESSOR = "spacy"
TREETAGGER_PREPROCESSOR = "treetagger"

class TextPreprocessor():
    def __init__(self, language, tolowercase, lemmatizer):
        self.tolowercase = tolowercase
        self.lemmatizer = lemmatizer
        self.language = language
        if self.lemmatizer == SPACY_PREPROCESSOR:
            if self.language == LANG_EN:
                self.nlp = spacy.load(SPACY_EN_MODEL)
            elif self.language == LANG_FR:
                self.nlp = spacy.load(SPACY_FR_MODEL)
        elif self.lemmatizer == TREETAGGER_PREPROCESSOR:
            self.treetagger = treetaggerwrapper.TreeTagger(TAGLANG=self.language)

    def process_with_spacy(self, text):
        if self.language == LANG_EN:
            ptext = " ".join([token.lemma_ for token in self.nlp(ptext)])
        elif self.language == LANG_FR:
            pos = POSTagger()
            french_lemmatizer = LefffLemmatizer(after_melt=True, default=True)
            self.nlp.add_pipe(pos, name='pos', after='parser')
            self.nlp.add_pipe(french_lemmatizer, name='lefff', after='pos')
            ptext = " ".join([token._.lefff_lemma for token in self.nlp(ptext)])
        return ptext

    def process_with_treetagger(self, text):
        tags = self.treetagger.tag_text(text)
        tags2 = treetaggerwrapper.make_tags(tags)
        ptext = " ".join([token.lemma for token in tags2])
        return ptext

    def process(self, text):
        ptext = " ".join(text.split())  # Replace all runs of whitespace with a single whitespace
        if self.tolowercase:
            ptext = ptext.lower()
        if self.lemmatizer == SPACY_PREPROCESSOR:
            ptext = self.process_with_spacy(ptext)
        elif self.lemmatizer == TREETAGGER_PREPROCESSOR:
            ptext = self.process_with_treetagger(ptext)
        return ptext
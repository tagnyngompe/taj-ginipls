#from spacy_lefff import LefffLemmatizer, POSTagger # https://pypi.org/project/spacy-lefff/
#from spacy_lefff import root as spacylogger # utilise le logg de spaCy
import re
from ginipls.config import GLOBAL_LOGGER as logger

class TextPreprocessor():
    def __init__(self, tolowercase=False, tolemmatize=False):
        self.tolowercase = tolowercase
        self.tolemmatize = tolemmatize
    def process(self, text):
        text = " ".join(text.split() ) # Replace all runs of whitespace with a single whitespace
        if self.tolowercase:
            text = text.lower()
        if self.tolemmatize:
            pass
        return text
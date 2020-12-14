import nltk
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.tokenize import word_tokenize

import spacy


class NltkTokenizer:

    def __init__(self):
        # pattern taken from https://stackoverflow.com/questions/35118596/python-regular-expression-not-working-properly
        self.pattern = r"""(?x)                   # set flag to allow verbose regexps
                            (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A.
                            |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages
                            |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe
                            |(?:[+/\-@&*])         # special characters with meanings
                            """

    def __simply_tokenize__(self, text):
        # tokenizer = RegexpTokenizer(self.pattern)
        # return tokenizer.tokenize(text)
        return word_tokenize(text)

    def tokenize(self, text):
        """
        Returns a list of tuples containing every token and its associated position in the text
        """
        tokens = self.__simply_tokenize__(text)
        offset = 0
        tok_pos = []
        for token in tokens:
            offset = text.find(token, offset)
            tok_pos.append((token, offset))
            offset += len(token)
        return tok_pos


class SpacyTokenizer:

    def __init__(self):
        self.tokenizer = spacy.load("en_core_web_sm")

    def tokenize(self, text):
        return self.tokenizer(text)

# TODO: Add possibility to return the index of a token (so that it can match the index of the span). For Spacy this
#  can be done using the attribute .idx (https://realpython.com/natural-language-processing-spacy-python
#  /#tokenization-in-spacy).

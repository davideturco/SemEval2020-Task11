from nltk.tokenize.regexp import RegexpTokenizer
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

    def tokenize(self, text):
        tokenizer = RegexpTokenizer(self.pattern)

        return tokenizer.tokenize(text)


class SpacyTokenizer:

    def __init__(self):
        self.tokenizer = spacy.load("en_core_web_sm")

    def tokenize(self, text):
        return self.tokenizer(text)

# TODO: Add possibility to return the index of a token (so that it can match the index of the span). For Spacy this
#  can be done using the attribute .idx (https://realpython.com/natural-language-processing-spacy-python
#  /#tokenization-in-spacy) but for NLTK see https://stackoverflow.com/questions/31668493/get-indices-of-original
#  -text-from-nltk-word-tokenize

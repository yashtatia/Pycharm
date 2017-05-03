import string

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.base import BaseEstimator, TransformerMixin

class NLTKPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, stopwords=None, punct=None, lower=True, strip=True):

        self.lower = lower
        self.strip = strip
        self.stopwords = stopwords
        self.punct = punct or set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [
            list(self.tokenize(doc)) for doc in X
        ]

    def tokenize(self, document):
        # Break the document into sentences
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # If stopword, ignore token and continue
                if token in self.stopwords:
                    continue

                # If punctuation, ignore token and continue
                if all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def lemmatize(self, token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)


from normalization import normalize_corpus
from feature_extractors import build_feature_matrix
import pandas as pd
import numpy as np

file_loc = "/media/ytatia/New Volume/Grievances related to Summer Term (Responses).xlsx"
file_loc1 = "/media/ytatia/New Volume/grievances.xlsx"
file_loc2 = "/media/ytatia/New Volume/Hostel related grievances.xlsx"
df = pd.read_excel(file_loc, index_col=0, skiprows=1, header=0, parse_cols="I")
df1 = pd.read_excel(file_loc1, index_col=0, skiprows=1, header=0, parse_cols="I")
df2 = pd.read_excel(file_loc2, index_col=0, skiprows=1, header=0, parse_cols="H")

frames = [df, df1, df2]

result = pd.concat(frames)

doxyDonkeyPosts = result.index
#print doxyDonkeyPosts

query_post = ['The food quality in the mess sucks']


#NOrmalize and extract features from the corpus
norm_corpus = normalize_corpus(doxyDonkeyPosts, lemmatize=True)

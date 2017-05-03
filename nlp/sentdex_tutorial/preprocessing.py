from contractions import CONTRACTION_MAP
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


stopwords_list = stopwords.words("english")
wnl = WordNetLemmatizer()

#tokenizing and reoving any extraneous whitespaces from the tokens
def tokenize_text(text):
    tokens = word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens

#retruns the same body of text with its contractions expanded
def expand_contraction(text, contraction_mapping):

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), flags=re.IGNORECASE|re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expand_contraction = contraction_mapping.get(match) \
                                if contraction_mapping.get(match) \
                                else contraction_mapping.get(match.lower())
        expand_contraction = first_char + expand_contraction[1:]
        return expand_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

from pattern.en import tag
from nltk.corpus import wordnet as wn

#Annotate text tokens with POS tags
def pos_tag_text(text):
    #Convert Penn treebank tag to wordnet tag

    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None

    tagged_text = tag()

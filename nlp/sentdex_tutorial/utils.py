from final import get_data
from nltk.tokenize import word_tokenize
import nltk

def flatten_corpus(corpus):
    return ' '.join([document.strip() for document in corpus])

def compute_ngrams(sequence, n):
    return zip(*[sequence[index:] for index in range(n)])

def get_top_ngrams(corpus, ngram_val=1, limit=5):

    corpus = flatten_corpus(corpus)
    tokens = word_tokenize(corpus)

    ngrams = compute_ngrams(tokens, ngram_val)
    ngram_freq_dist = nltk.FreqDist(ngrams)
    sorted_ngrams_fd = sorted(ngram_freq_dist.items(), key=itemgetter(1), reverse=True)
    sorted_ngrams = sorted_ngrams_fd[0:limit]
    sorted_ngrams = [(' '.join(text), freq) for text, freq in sorted_ngrams]

    return sorted_ngrams

corpus, category = get_data()

from nltk.collocations import BigramCollocationFinder
from nltk.collocations import BigramAssocMeasures

finder = BigramCollocationFinder.from_documents([item.split() for item in corpus])
bigram_measures = BigramAssocMeasures()

print finder.nbest(bigram_measures.raw_freq, 10)

from nltk.collocations import TrigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures

finder = TrigramCollocationFinder.from_documents([item.split() for item in corpus])
trigram_measures = TrigramAssocMeasures()

print finder.nbest(trigram_measures.raw_freq, 10)
print finder.nbest(trigram_measures.pmi, 10)


# print get_top_ngrams(corpus, ngram_val=2, limit=10)
from feature_extractors import build_feature_matrix
import networkx
import numpy as np
import matplotlib
from normalization import normalize_corpus
norm = normalize_corpus(corpus)
# construcat weighted document term matrix
vec, dt_matrix = build_feature_matrix(norm, feature_type='tfidf')

similarity_matrix = (dt_matrix * dt_matrix.T)
# view document similarity matrix
print np.round(similarity_matrix.todense(), 2)

# build similarity graph
similarity_graph = networkx.from_scipy_sparse_matrix(similarity_matrix)
# networkx.draw_networkx(similarity_graph)

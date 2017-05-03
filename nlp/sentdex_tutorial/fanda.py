from normalization import normalize_corpus
from feature_extractors import build_feature_matrix
import pandas as pd
import numpy as np
import pickle
import gensim
import logging

logging.basicConfig(filename = 'catmat.log',
                    filemode = 'a',
                    level = logging.DEBUG,
                    format = '%(asctime)s %(message)s')


file_loc = "/media/ytatia/New Volume/grievances.xlsx"

df = pd.read_excel(file_loc, sheetname="Sheet1", header=0, skiprows=0, index_col=None, parse_cols="A:B", converters={'Category':str})
# print df.head()
# print list(df)
print len(df)
print df.shape

# df1 = df.sample(frac=1)
#
# test = df1[:1]
# print test
#
# print list(df['Category'])
#
# print list(df)

# file_loc2 = "/media/ytatia/New Volume/Grievances_final.xlsx"
# df2 = pd.read_excel(file_loc2, index_col=0, skiprows=1, header=0, parse_cols="B", converters={"Category":str})

# frames2 = [df, df2]
# result2 = pd.concat(frames2, axis=1)
# print result2

df = df.reindex(np.random.permutation(df.index))



description = list(df['Description'])
category = list(df['Category'])
# #print category
problem = 'Internet nahi chal raha hai aur kutte raat ko bahut bhokte hai'
temp_post = problem.split('.')
query_post = []
for t in temp_post:
    if t == '':
        continue
    query_post.append(t)
print query_post

#NOrmalize and extract features from the corpus
norm_corpus = normalize_corpus(description, lemmatize=True)
tfidf_vectorizer, tfidf_features = build_feature_matrix(norm_corpus, feature_type='tfidf', ngram_range=(1,1), min_df=0.0, max_df=1.0)

features = tfidf_features.todense()
feature_name = tfidf_vectorizer.get_feature_names()

def display_features(features, feature_name):
    dff = pd.DataFrame(data=features, columns=feature_name)
    print dff

# display_features(features, feature_name)
#Normalize and extract features from the query corpus
norm_query_docs = normalize_corpus(query_post, lemmatize=True)
#print norm_corpus
#print norm_query_docs


# query_docs_tfidf = tfidf_vectorizer.transform(norm_query_docs)
#
# def compute_cosine_similarity(doc_features, corpus_features, top_n=3):
#
#     # get document vectors
#     doc_features = doc_features.toarray()[0]
#     corpus_features = corpus_features.toarray()
#
#     # compute similarities
#     similarity = np.dot(doc_features, corpus_features.T)
#
#     # get docs with highest similarity scores
#     top_docs = similarity.argsort()[::-1][:top_n]
#     top_docs_with_score = [(index, round(similarity[index], 3))
#                             for index in top_docs]
#
#     return top_docs_with_score
#
#
# print 'Document Similarity Analysis using Cosine Similarity'
# print '='*60
#
# try:
#     for index, doc in enumerate(query_post):
#
#         doc_tfidf = query_docs_tfidf[index]
#         top_similar_docs = compute_cosine_similarity(doc_tfidf,
#                                                  tfidf_features,
#                                                  top_n=2)
#         print 'Document',index+1 ,':', doc
#         print 'Top', len(top_similar_docs), 'similar docs:'
#         print '-'*40
#         for doc_index, sim_score in top_similar_docs:
#             # print "working"
#             if sim_score > 0.2:
#                 print 'Doc num: {} Similarity Score: {}\nDoc: {}\nCategory: {}'.format(doc_index+1, sim_score, description[doc_index], category[doc_index])
#             print '-'*40
#         print
# except:
#     print "Cosine similarity method failed"

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

data_train = norm_corpus[:400]
data_test = norm_corpus[401:]
category_train = category[:400]
category_test = category[401:]

X_train = np.array(data_train)
Y_train = []
#print category
for q in category_train:
    qstring = str(q)
    min_list = qstring.split(',')
    max_list = []
    for m in min_list:
        max_list.append(m.strip())
    #print min_list
    Y_train.append(max_list)

#print Y_train
X_test = np.array(data_test)
target_names = ['Academic', 'Mess', 'Internet', 'Maintainance']

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(Y_train)

Y_test = []
for q in category_test:
    qstring = str(q)
    min_list = qstring.split(',')
    max_list = []
    for m in min_list:
        max_list.append(m.strip())
    #print min_list
    Y_test.append(max_list)

yy = mlb.fit_transform(Y_test)
classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

classifier.fit(X_train, Y)

predicted = classifier.predict(X_test)
all_labels = mlb.inverse_transform(predicted)

print mlb.classes_


print np.mean(predicted == yy)

from sklearn import metrics

print metrics.classification_report(yy, predicted, target_names=Y_test)


#
#
# def compute_hellinger_bhattacharya_distance(doc_features, corpus_features,
#                                             top_n=2):
#     # get document vectors
#     doc_features = doc_features.toarray()[0]
#     corpus_features = corpus_features.toarray()
#     # compute hb distances
#     distance = np.hstack(
#                     np.sqrt(0.5 *
#                             np.sum(
#                                 np.square(np.sqrt(doc_features) -
#                                           np.sqrt(corpus_features)),
#                                 axis=1)))
#     # get docs with lowest distance scores
#     top_docs = distance.argsort()[:top_n]
#     top_docs_with_score = [(index, round(distance[index], 3))
#                             for index in top_docs]
#     return top_docs_with_score
#
# print 'Document Similarity Analysis using Hellinger-Bhattacharya distance'
# print '='*60
# for index, doc in enumerate(query_post):
#
#     doc_tfidf = query_docs_tfidf[index]
#     top_similar_docs = compute_hellinger_bhattacharya_distance(doc_tfidf,
#                                              tfidf_features,
#                                              top_n=2)
#     print 'Document',index+1 ,':', doc
#     print 'Top', len(top_similar_docs), 'similar docs:'
#     print '-'*40
#     for doc_index, sim_score in top_similar_docs:
#         print 'Doc num: {} Distance Score: {}\nDoc: {}'.format(doc_index+1,
#                                                                  sim_score,
#                                                                  doxyDonkeyPosts[doc_index])
#         print '-'*40
#     print
#
#
# import scipy.sparse as sp
#
# def compute_corpus_term_idfs(corpus_features, norm_corpus):
#
#     dfs = np.diff(sp.csc_matrix(corpus_features, copy=True).indptr)
#     dfs = 1 + dfs # to smoothen idf later
#     total_docs = 1 + len(norm_corpus)
#     idfs = 1.0 + np.log(float(total_docs) / dfs)
#     return idfs
#
#
# def compute_bm25_similarity(doc_features, corpus_features,
#                             corpus_doc_lengths, avg_doc_length,
#                             term_idfs, k1=1.5, b=0.75, top_n=3):
#     # get corpus bag of words features
#     corpus_features = corpus_features.toarray()
#     # convert query document features to binary features
#     # this is to keep a note of which terms exist per document
#     doc_features = doc_features.toarray()[0]
#     doc_features[doc_features >= 1] = 1
#
#     # compute the document idf scores for present terms
#     doc_idfs = doc_features * term_idfs
#     # compute numerator expression in BM25 equation
#     numerator_coeff = corpus_features * (k1 + 1)
#     numerator = np.multiply(doc_idfs, numerator_coeff)
#     # compute denominator expression in BM25 equation
#     denominator_coeff =  k1 * (1 - b +
#                                 (b * (corpus_doc_lengths /
#                                         avg_doc_length)))
#     denominator_coeff = np.vstack(denominator_coeff)
#     denominator = corpus_features + denominator_coeff
#     # compute the BM25 score combining the above equations
#     bm25_scores = np.sum(np.divide(numerator,
#                                    denominator),
#                          axis=1)
#     # get top n relevant docs with highest BM25 score
#     top_docs = bm25_scores.argsort()[::-1][:top_n]
#     top_docs_with_score = [(index, round(bm25_scores[index], 3))
#                             for index in top_docs]
#     return top_docs_with_score
#
# vectorizer, corpus_features = build_feature_matrix(norm_corpus,
#                                                    feature_type='frequency')
# query_docs_features = vectorizer.transform(norm_query_docs)
#
# doc_lengths = [len(doc.split()) for doc in norm_corpus]
# avg_dl = np.average(doc_lengths)
# corpus_term_idfs = compute_corpus_term_idfs(corpus_features,
#                                             norm_corpus)
#
# print 'Document Similarity Analysis using BM25'
# print '='*60
# for index, doc in enumerate(query_post):
#
#     doc_features = query_docs_features[index]
#     top_similar_docs = compute_bm25_similarity(doc_features,
#                                                corpus_features,
#                                                doc_lengths,
#                                                avg_dl,
#                                                corpus_term_idfs,
#                                                k1=1.5, b=0.75,
#                                                top_n=3)
#     print 'Document',index+1 ,':', doc
#     print 'Top', len(top_similar_docs), 'similar docs:'
#     print '-'*40
#     for doc_index, sim_score in top_similar_docs:
#         print 'Doc num: {} BM25 Score: {}\nDoc: {}'.format(doc_index+1,
#                                                                  sim_score,
#                                                                  doxyDonkeyPosts[doc_index])
#         print '-'*40
#     print

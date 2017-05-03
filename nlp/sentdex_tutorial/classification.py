from normalization import normalize_corpus
from feature_extractors import build_feature_matrix
import pandas as pd
import numpy as np
import pickle
import gensim

file_loc = "/media/ytatia/New Volume/Grievances_final.xlsx"

df = pd.read_excel(file_loc, index_col=0, skiprows=1, header=0, parse_cols="A", converters={"Description":str})

description = df.index
#print description

file_loc2 = "/media/ytatia/New Volume/Grievances_final.xlsx"
df2 = pd.read_excel(file_loc2, index_col=0, skiprows=1, header=0, parse_cols="B", converters={"Category":str})

category = df2.index
print category
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

query_docs_tfidf = tfidf_vectorizer.transform(norm_query_docs)

def compute_cosine_similarity(doc_features, corpus_features, top_n=3):

    # get document vectors
    doc_features = doc_features.toarray()[0]
    corpus_features = corpus_features.toarray()

    # compute similarities
    similarity = np.dot(doc_features, corpus_features.T)

    # get docs with highest similarity scores
    top_docs = similarity.argsort()[::-1][:top_n]
    top_docs_with_score = [(index, round(similarity[index], 3))
                            for index in top_docs]

    return top_docs_with_score


print 'Document Similarity Analysis using Cosine Similarity'
print '='*60

try:
    for index, doc in enumerate(query_post):

        doc_tfidf = query_docs_tfidf[index]
        top_similar_docs = compute_cosine_similarity(doc_tfidf,
                                                 tfidf_features,
                                                 top_n=2)
        print 'Document',index+1 ,':', doc
        print 'Top', len(top_similar_docs), 'similar docs:'
        print '-'*40
        for doc_index, sim_score in top_similar_docs:
            # print "working"
            if sim_score > 0.2:
                print 'Doc num: {} Similarity Score: {}\nDoc: {}\nCategory: {}'.format(doc_index+1, sim_score, description[doc_index], category[doc_index])
            print '-'*40
        print
except:
    print "Cosine similarity method failed"

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

X_train = np.array(norm_corpus)
Y_train = []

for q in category:
    qstring = str(q)
    min_list = qstring.split(',')
    max_list = []
    for m in min_list:
        max_list.append(m.strip())

    Y_train.append(max_list)

X_test = np.array(norm_query_docs)
target_names = ['Academic', 'Mess', 'Internet', 'Maintainance']

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(Y_train)

classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

classifier.fit(X_train, Y)

predicted = classifier.predict(X_test)
all_labels = mlb.inverse_transform(predicted)

print mlb.classes_


temp_categories = []
for item, labels in zip(X_test, all_labels):
    temp_categories.append(labels)
    print('{0} => {1}'.format('Category', ', '.join(labels)))

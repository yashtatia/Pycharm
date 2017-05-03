import logging
import pickle
from time import time

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

from feature_extractors import build_feature_matrix
from normalization import normalize_corpus

t0 = time()

logging.basicConfig(filename = 'classification.log',
                    filemode = 'a',
                    level = logging.DEBUG,
                    format = '%(asctime)s %(message)s')

file_loc = "/media/ytatia/New Volume/grievances.xlsx"
def get_data(retrain=False, problem=None, category=None):

    file_loc = "/media/ytatia/New Volume/grievances.xlsx"

    df = pd.read_excel(file_loc, sheetname="Sheet1", header=0, skiprows=0, index_col=None, parse_cols="A:B", converters={'Category':str})

    if retrain:

        print df

    # Randomizing raw data
    df = df.reindex(np.random.permutation(df.index))

    print "DataSet = ", len(df)
    print df.shape

    return list(df['Description']), list(df['Category'])

description, category = get_data()
t1 = time()
# print "Time taken to load data: ", round(t1-t0, 3), "s"


problem = 'washroom mein tap nahi hai aur net nahi chal raha hai'

temp_post = problem.split('.')
query_post = []
for t in temp_post:
    if t == '':
        continue
    query_post.append(t)
# print query_post
def normalize_text(text):
    return normalize_corpus(text, lemmatize=True)

# #NOrmalize and extract features from the corpus
# norm_corpus = normalize_text(description)
#
# #Normalize and extract features from the query corpus
# norm_query_docs = normalize_text(query_post)
#
# # norm_corpus, category = remove_empty_docs(norm_corpus, category)
# print "Time taken to normalize text: ", round(time()-t1, 3), "s"


def multilabelbinarizer_classification(norm_corpus, category, norm_query_docs, category_test, validation=False):

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

    print "mlb:"
    print mlb

    classifier = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(LinearSVC(verbose=2)))])

    classifier.fit(X_train, Y)

    model_to_save = {
        'classifier': classifier,
        'mlb': mlb,
        'created_at': time()
    }

    save_models = open("/media/ytatia/New Volume/similarity.pickle", "wb")
    pickle.dump(model_to_save, save_models)
    save_models.close()

    print "______________________________________________________________________________"
    print classifier.__dict__
    print "______________________________________________________________________________"
    predicted = classifier.predict(X_test)
    all_labels = mlb.inverse_transform(predicted)

    print "Categories Available: ", mlb.classes_


    temp_categories = []
    for item, labels in zip(X_test, all_labels):
        temp_categories.append(labels)
        print('{0} => {1}'.format('Category', ', '.join(labels)))

    if validation:
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

        print "Accuracy Score: ", round(np.mean(predicted == yy), 3)


        print metrics.classification_report(yy, predicted, target_names=Y_test)


def cross_validation(norm_corpus, category, test_data_proportion=0.3):

    # data_train = norm_corpus[:400]
    # data_test = norm_corpus[401:]
    # category_train = category[:400]
    # category_test = category[401:]
    data_train, data_test, category_train, category_test = train_test_split(norm_corpus, category, test_size=0.33, random_state=42)

    multilabelbinarizer_classification(data_train, category_train, data_test, category_test, validation=True)
# t2=time()
# multilabelbinarizer_classification(norm_corpus, category, norm_query_docs, category_test=None)
# # cross_validation(norm_corpus, category)
# print "Classification done under: ", round(time()-t2, 3), "s"
# Cosine Simiarity based Classification Method
def cosine_similarity_classification(norm_corpus, norm_query_docs):

    tfidf_vectorizer, tfidf_features = build_feature_matrix(norm_corpus, feature_type='tfidf', ngram_range=(1,1), min_df=0.0, max_df=1.0)
    query_docs_tfidf = tfidf_vectorizer.transform(norm_query_docs)

    display_features(tfidf_features, tfidf_vectorizer)

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

def display_features(tfidf_features, tfidf_vectorizer):
        features = tfidf_features.todense()
        feature_name = tfidf_vectorizer.get_feature_names()
        dff = pd.DataFrame(data=features, columns=feature_name)
        print dff

# Hellinger Bhattacharya based Classification Method
def hellinger_bhattacharya():

    tfidf_vectorizer, tfidf_features = build_feature_matrix(norm_corpus, feature_type='tfidf', ngram_range=(1,1), min_df=0.0, max_df=1.0)
    query_docs_tfidf = tfidf_vectorizer.transform(norm_query_docs)

    display_features(tfidf_features, tfidf_vectorizer)

    def compute_hellinger_bhattacharya_distance(doc_features, corpus_features,
                                                top_n=2):
        # get document vectors
        doc_features = doc_features.toarray()[0]
        corpus_features = corpus_features.toarray()
        # compute hb distances
        distance = np.hstack(
                        np.sqrt(0.5 *
                                np.sum(
                                    np.square(np.sqrt(doc_features) -
                                              np.sqrt(corpus_features)),
                                    axis=1)))
        # get docs with lowest distance scores
        top_docs = distance.argsort()[:top_n]
        top_docs_with_score = [(index, round(distance[index], 3))
                                for index in top_docs]
        return top_docs_with_score

    print 'Document Similarity Analysis using Hellinger-Bhattacharya distance'
    print '='*60
    for index, doc in enumerate(query_post):

        doc_tfidf = query_docs_tfidf[index]
        top_similar_docs = compute_hellinger_bhattacharya_distance(doc_tfidf,
                                                 tfidf_features,
                                                 top_n=2)
        print 'Document',index+1 ,':', doc
        print 'Top', len(top_similar_docs), 'similar docs:'
        print '-'*40
        for doc_index, sim_score in top_similar_docs:
            print 'Doc num: {} Similarity Score: {}\nDoc: {}\nCategory: {}'.format(doc_index+1, sim_score, description[doc_index], category[doc_index])
            print '-'*40
        print

def retrain_multilabelbinarizer_model(query_post, category):
    t01 = time()
    print "Retraining Multi Label Binarizer Model"

    # data_filtered.to_excel(writer, "Sheet1", cols=[])

    description, category = get_data()

    t02 = time()
    print "Time taken to load dataset: ", round(t02-t01, 3), "s"

    #NOrmalize and extract features from the corpus
    norm_corpus = normalize_text(description)

    #Normalize and extract features from the query corpus
    norm_query_docs = normalize_text(query_post)
    t03 = time()
    # norm_corpus, category = remove_empty_docs(norm_corpus, category)
    print "Time taken to normalize text: ", round(t03-t02, 3), "s"

    multilabelbinarizer_classification(norm_corpus, category, norm_query_docs, category_test=None)
    # cross_validation(norm_corpus, category)
    print "Classification done under: ", round(time()-t03, 3), "s"
    print "Model Retraining complete"

# category = 'Academic'
# retrain_multilabelbinarizer_model(query_post, category)

def classify(query_post):
    t04 = time()
    pickled_data_loc = "/media/ytatia/New Volume/similarity.pickle"

    f = open(pickled_data_loc, "rb")
    modals = pickle.load(f)
    classifier = modals['classifier']
    mlb = modals['mlb']
    f.close()


    #Normalize and extract features from the query corpus
    norm_query_docs = normalize_text(query_post)
    X_test = np.array(norm_query_docs)

    predicted = classifier.predict(X_test)
    all_labels = mlb.inverse_transform(predicted)

    print "Categories Available: ", mlb.classes_


    temp_categories = []
    for item, labels in zip(X_test, all_labels):
        temp_categories.append(labels)
        print('{0} => {1}'.format('Category', ', '.join(labels)))

    print "Classification completed in ", round(time()-t04, 3), "s"

classify(query_post)

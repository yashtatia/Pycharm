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
#print result
#print "---------------------------------------------------------------------------------"
doxyDonkeyPosts = result.index

print doxyDonkeyPosts


#Identify underlying themes----Clustering

#Using either term frequency or tf-idf

#Features : Create a list represnting the universe of all words that can appear in any text

from sklearn.feature_extraction.text import TfidfVectorizer #Converts text to tf-idf representation

vectorizer = TfidfVectorizer(max_df = 0.8, min_df = 1, stop_words = 'english')
X = vectorizer.fit_transform(doxyDonkeyPosts) #All articles are converted to a 2d array , eacah row is an article of our corpus
#print X

from sklearn.cluster import KMeans
km = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 100, n_init = 1, verbose = True)
km.fit(X)

#print np.unique(km.labels_, return_counts = True) #Array of cluster numbers assigned to each text

#Meaningful theme underlined in each cluster
text = {}
for i, cluster in enumerate(km.labels_):
    oneDocumnent = doxyDonkeyPosts[i]
    if cluster not in text.keys():
        text[cluster] = oneDocumnent
    else:
        text[cluster] += oneDocumnent #Aggregate text in each cluster


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import defaultdict
from string import punctuation
from heapq import nlargest
import nltk

_stopwords = set(stopwords.words('english') + list(punctuation) + ["million", "billion", "year", "millions", "billions", "y/y", "'s", "''"])

#Top words in each cluster and their counts
keywords = {}
counts = {}
for cluster in range(4):
    word_sent = word_tokenize(text[cluster].lower())
    word_sent = [word for word in word_sent if word not in _stopwords]
    freq = FreqDist(word_sent)
    keywords[cluster] = nlargest(100, freq, key = freq.get)
    counts[cluster] = freq

#Top keywords unique to each cluster
unique_keys = {}
for cluster in range(4):
    other_clusters = list(set(range(4)) - set([cluster]))
    keys_other_cluster = set(keywords[other_clusters[0]]).union(set(keywords[other_clusters[1]]))
    unique =  set(keywords[cluster]) - keys_other_cluster
    unique_keys[cluster] = nlargest(10, unique, key = counts[cluster].get)

print unique_keys


#Assign theme to new articles-----Classification
article = ""

from sklearn.neighbors import KNeighborsClassifier
classfier = KNeighborsClassifier() #u can also put n_neighbors = 10
classfier.fit(X, km.labels_) #training phase

KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, metric = 'minkowski', metric_params = None, n_jobs = 1, n_neighbors = 5, p = 2, weights = 'uniform') # 5 is default

test = vectorizer.transform([article.decode('utf8').encode('ascii', errors = 'ignore')]) #represnt the article as tf-idf

print classfier.predict(test) #Test phase


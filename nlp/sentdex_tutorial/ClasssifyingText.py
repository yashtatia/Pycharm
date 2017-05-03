import urllib2
from bs4 import BeautifulSoup

proxy = urllib2.ProxyHandler({'http': 'http://069.13085075:jaitatia1997@10.1.1.45:80'})
auth = urllib2.HTTPBasicAuthHandler()
opener = urllib2.build_opener(proxy, auth, urllib2.HTTPHandler)
urllib2.install_opener(opener)
#Building a corpus

def getAllDoxyDonkeyPost(url, links): #download all the urls
    #each url has 7 posts
    #fist we'll send in the home page url
    #then collect the next urls in aan empty list

    request = urllib2.Request(url)
    response = urllib2.urlopen(request)
    soup = BeautifulSoup(response)

    for a in soup.findAll('a'):
        #use a tag to find all the links
        try:
            url = a['href']
            title = a['title']

            if title == "Older Posts":
                #print title, url
                links.append(url)
                getAllDoxyDonkeyPost(url, links)

        except:
            title = ""

    return


blogUrl = "http://doxydonkey.blogspot.in"
links = []
getAllDoxyDonkeyPost(blogUrl, links)

#Now we'll parse the text.....to collect the articles we have to find all divs with class post-body and then the bullet points


def getDoxyDonkeyText(url):
    request = urllib2.Request(url)
    response = urllib2.urlopen(request)
    soup = BeautifulSoup(response)

    mydivs = soup.findAll("div", {"class":'post-body'}) # find all the post within the page

    posts = []

    for div in mydivs:
        posts += map(lambda p:p.text.encode('ascii', errors='replace').replace("?", " "), div.findAll("li")) #find all articles within the posts

    return posts

doxyDonkeyPosts = []
for link in links:
    doxyDonkeyPosts += getDoxyDonkeyText(link)

print doxyDonkeyPosts


#Identify underlying themes----Clustering

#Using either term frequency or tf-idf

#Features : Create a list represnting the universe of all words that can appear in any text

from sklearn.feature_extraction.text import TfidfVectorizer #Converts text to tf-idf representation

vectorizer = TfidfVectorizer(max_df = 0.5, min_df = 2, stop_words = 'english')
X = vectorizer.fit_transform(doxyDonkeyPosts) #All articles are converted to a 2d array , eacah row is an article of our corpus
print X

from sklearn.cluster import KMeans
km = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 100, n_init = 1, verbose = True)
km.fit(X)

import numpy as np
print np.unique(km.labels_, return_counts = True) #Array of cluster numbers assigned to each text

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
for cluster in range(3):
    word_sent = word_tokenize(text[cluster].lower())
    word_sent = [word for word in word_sent if word not in _stopwords]
    freq = FreqDist(word_sent)
    keywords[cluster] = nlargest(100, freq, key = freq.get)
    counts[cluster] = freq

#Top keywords unique to each cluster
unique_keys = {}
for cluster in range(3):
    other_clusters = list(set(range(3)) - set([cluster]))
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


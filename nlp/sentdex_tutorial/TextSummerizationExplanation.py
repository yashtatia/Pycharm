# Downloading text

import urllib2 #to download the webpage
from bs4 import BeautifulSoup #to parse the web document

articleUrl = "https://www.washingtonpost.com/news/the-switch/wp/2016/10/18/the-pentagons-massive-new-telescope-is-designed-to-track-space-junk-and-watch-out-for-killer-asteroids/?utm_term=.5a231a812d9e"

page =  urllib2.urlopen(articleUrl).read().decode('utf8', 'ignore') #downloads the webpage
soup = BeautifulSoup(page, "lxml") #creating a beautiful soup object which creates a tree structure of the webpage
#print soup

soup.find('article') #will give the first text with tag article

soup.find('article').text #will provide with only the txt

text = ' '.join(map(lambda p: p.text, soup.find_all('article'))) #finds all txt with tag article and gives a list which is stitched to a single txt
#print text

text.encode('ascii', errors='replace').replace("?", " ") # replacing special characters
#print text

#Preprocessing text

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation

sents = sent_tokenize(text)
#print sents

word_sent = word_tokenize(text.lower())
#print word_sent

_stopwords = set(stopwords.words('english') + list(punctuation)) #set of al the stopwords
#print _stopwords

word_sent = [word for word in word_sent if word not in _stopwords]
#print word_sent

#Auto Summerizing

from nltk.probability import FreqDist #Construct a frequency distributionof words
freq = FreqDist(word_sent)


from heapq import nlargest # sort any collection
print nlargest(10, freq, key=freq.get)

from collections import defaultdict # special dictionary where it will not throw an error if the key os not present but rather add the new key to it
ranking = defaultdict(int)

for i,sent in enumerate(sents): #enumerate converts [a,b,c] to [0,a],[1,b][2,c] i.e a tuple
    for w in word_tokenize(sent.lower()):
        if w in freq:
            ranking[i] += freq[w]
#print ranking

sents_idx = nlargest(4, ranking, key=ranking.get) # most important sentences
print sents_idx

print [sents[j] for j in sorted(sents_idx)] #stitching sentences together to form a summary



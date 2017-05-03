import urllib2
from bs4 import BeautifulSoup
from nltk.probability import FreqDist
from collections import defaultdict
from heapq import nlargest
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation

articleUrl = "https://www.washingtonpost.com/news/the-switch/wp/2016/10/18/the-pentagons-massive-new-telescope-is-designed-to-track-space-junk-and-watch-out-for-killer-asteroids/?utm_term=.5a231a812d9e"

def getTextWaPo(url):
    page =  urllib2.urlopen(articleUrl).read().decode('utf8')
    soup = BeautifulSoup(page, "lxml")
    text = ' '.join(map(lambda p: p.text, soup.find_all('article')))
    return text.encode('ascii', errors='replace').replace("?", " ")

text = getTextWaPo(articleUrl)

def summarize(text, n):
    sents = sent_tokenize(text)

    assert n <= len(sents) #Check whether our text has required number of sentences
    word_sent = word_tokenize(text.lower())
    _stopwords = set(stopwords.words('english') + list(punctuation))

    word_sent = [word for word in word_sent if word not in _stopwords]
    freq = FreqDist(word_sent)

    ranking = defaultdict(int)


    for i,sent in enumerate(sents):
        for w in word_tokenize(sent.lower()):
            if w in freq:
                ranking[i] += freq[w]

    sents_idx = nlargest(n, ranking, key=ranking.get)
    return [sents[j] for j in sorted(sents_idx)]

print summarize(text, 3)

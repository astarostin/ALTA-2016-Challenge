import urlparse
from os.path import splitext, basename
import re
from urllib import urlopen
from bs4 import BeautifulSoup as bs

import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def get_url_tokens(url):
    """
    Returns tokens from url split by non-word characters. Ignores domain and extension
    """
    parsed_url = urlparse.urlparse(url)
    if (len(parsed_url.scheme) == 0):
        parsed_url = urlparse.urlparse('http://'+url)
    path = parsed_url.path
    _, ext = splitext(basename(path))
    path = path.replace(ext, '')
    tokens = filter(None, re.split('\W+', path))
    return tokens

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems
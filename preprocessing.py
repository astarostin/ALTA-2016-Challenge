import urlparse
from os.path import splitext, basename
import re
import nltk
from nltk.stem.porter import PorterStemmer


def get_url_tokens(url):
    """
    Returns tokens from url split by non-word characters. Ignores domain and extension
    :param url: url to parse
    :return:
    """
    parsed_url = urlparse.urlparse(url)
    if len(parsed_url.scheme) == 0:
        parsed_url = urlparse.urlparse('http://'+url)
    path = parsed_url.path
    _, ext = splitext(basename(path))
    path = path.replace(ext, '')
    tokens = filter(None, re.split('[^a-zA-Z0-9]+', path))
    res = []
    # Digits seem useless (otherwise add '|[0-9]+')
    for token in tokens:
        res.extend(re.findall('[A-Z][^A-Z]*|^[a-z]+', token))
    return res


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stemmer = PorterStemmer()
    stems = stem_tokens(tokens, stemmer)
    return stems

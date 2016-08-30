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
    return tokenize_simple(path)


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize_nltk(text):
    print text
    tokens = nltk.word_tokenize(text.decode('utf-8').encode('ascii', errors='ignore'))
    stemmer = PorterStemmer()
    stems = stem_tokens(tokens, stemmer)
    return [x.lower() for x in stems]


def tokenize_simple(text):
    tokens = filter(None, re.split('[^a-zA-Z0-9]+', text))
    res = []
    # Digits seem useless (otherwise add '|[0-9]+')
    # Min length - 2 symbols
    for token in tokens:
        res.extend(re.findall('[A-Z]{2,}|[A-Z][^A-Z]{1,}|^[a-z]{2,}', token))
    # Convert all tokens to lowercase
    return [x.lower().decode('utf-8').encode('ascii', errors='ignore') for x in res]

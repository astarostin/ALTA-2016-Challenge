import urlparse
from os.path import splitext, basename
import re
import nltk
from nltk.stem.porter import PorterStemmer


# def is_stop_word(word):
#     return word not in ['http', 'https', 'www', 'com', 'org']


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


def tokenize_simple(text, stop_words=[], token_min_length=4):
    tokens = filter(None, re.split('[^a-zA-Z0-9]+', text))
    res = []
    # Digits seem useless (otherwise add '|[0-9]+')
    pattern = '[A-Z]{%d,}|[A-Z][^A-Z]{%d,}|^[a-z]{%d,}' % (token_min_length, token_min_length-1, token_min_length)
    for token in tokens:
        res.extend(re.findall(pattern, token))
    # Convert all tokens to lowercase, encode and filter stop words
    return filter(lambda s: s not in stop_words,
                  [x.lower().decode('utf-8').encode('ascii', errors='ignore') for x in res])

import urlparse
from os.path import splitext, basename
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


class Tokenizer:
    def __init__(self, token_min_length=4, use_nltk_tokenize=False):
        self._use_nltk_tokenize = use_nltk_tokenize
        self._token_min_length = token_min_length

# def is_stop_word(word):
#     return word not in ['http', 'https', 'www', 'com', 'org']

    def get_url_tokens(self, url):
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
        return self.tokenize_simple(path)

    def stem_tokens(self, tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    def tokenize_nltk(self, text):
        sentences = nltk.sent_tokenize(text.decode('utf-8').encode('ascii', errors='ignore'), 'english')

        re_tokenizer = nltk.RegexpTokenizer('[a-zA-Z0-9]+')
        sentences = [re_tokenizer.tokenize(sent) for sent in sentences]

        tagged_sentences = [nltk.pos_tag(sent) for sent in sentences]
        #print tagged_sentences

        nn_tokens = [[w[0] for w in sent if ('NN' in w[1])] for sent in tagged_sentences]
        nn_tokens = [item for sublist in nn_tokens for item in sublist]

        # stemming
        # stemmer = PorterStemmer()
        # stems = self.stem_tokens(nn_tokens, stemmer)
        # return [x.lower() for x in stems]

        # lemmatization
        # lmtzr = WordNetLemmatizer()
        # lemms = [lmtzr.lemmatize(x.lower()) for x in nn_tokens]
        # return lemms

        return nn_tokens

    def tokenize(self, text, type='text'):
        if self._use_nltk_tokenize and type=='text':
            return self.tokenize_nltk(text)
        else:
            return self.tokenize_simple(text, type)

    def tokenize_simple(self, text, type='text'):
        if type == 'url':
            return self.get_url_tokens(text)

        tokens = filter(None, re.split('[^a-zA-Z0-9]+', text))
        res = []
        # Digits seem useless (otherwise add '|[0-9]+')
        pattern = '[A-Z]{%d,}|[A-Z][^A-Z]{%d,}|^[a-z]{%d,}' % (self._token_min_length, self._token_min_length-1, self._token_min_length)
        for token in tokens:
            res.extend(re.findall(pattern, token))
        # Convert all tokens to lowercase, encode and filter stop words
        return [x.lower().decode('utf-8').encode('ascii', errors='ignore') for x in res]
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
import numpy as np
from itertools import chain
from scipy.sparse import hstack

class PairVectorizer(TfidfVectorizer):
    def __init__(self, first_col, second_col, join_method='concat', input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        super(PairVectorizer, self).__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype)

        self.join_method = join_method
        self.first_col = first_col
        self.second_col = second_col

    def fit(self, X, y=None):
        pair_first_data = X[self.first_col]
        pair_second_data = X[self.second_col]
        self._tfidf.fit(chain(pair_first_data, pair_second_data), y)

    def transform(self, X):
        t_first = super(PairVectorizer, self).transform(X[self.first_col])
        t_second = super(PairVectorizer, self).transform(X[self.second_col])
        if self.join_method == 'concat':
            t_data = hstack((t_first, t_second))
        elif self.join_method == 'sum':
            t_data = t_first + t_second
        else:
            raise ValueError('Invalid join method "%s"!' % self.join_method)
        return t_data.toarray()

    def fit_transform(self, X, y=None):
        t_first = super(PairVectorizer, self).fit_transform(X[self.first_col], y)
        t_second = super(PairVectorizer, self).fit_transform(X[self.second_col], y)
        if self.join_method == 'concat':
            t_data = hstack((t_first, t_second))
        elif self.join_method == 'sum':
            t_data = t_first + t_second
        else:
            raise ValueError('Invalid join method "%s"!' % self.join_method)
        return t_data.toarray()

    def get_params(self, deep=True):
        params_dict = super(PairVectorizer, self).get_params(deep)
        params_dict['first_col'] = self.first_col
        params_dict['second_col'] = self.second_col
        params_dict['join_method'] = self.join_method
        return params_dict

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)
        return self
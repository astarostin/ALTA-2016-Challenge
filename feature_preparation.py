import tokenization as prep
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack
import pandas as pd
from collections import Counter
from vectorization import PairVectorizer

class FeaturePreparator():
    def __init__(self, a_cols, b_cols, token_min_length = 4, join_method='concat'):
        self._token_min_length = token_min_length
        self._a_cols = a_cols
        self._b_cols = b_cols
        self._a_join_col = 'ATokens'
        self._b_join_col = 'BTokens'
        self._tokenizer = prep.Tokenizer(token_min_length, use_nltk_tokenize=True)
        self._join_method = join_method

    def populate_corpus(self, row, columns, corpus):
        for column in columns:
            if ('url' in column.lower()):
                tokens = self._tokenizer.tokenize(row[column], 'url')
            else:
                tokens = self._tokenizer.tokenize(row[column])
            corpus.extend(tokens)

    def prepare_tokens(self, row, columns, joined_column):
        tokens_summary = ''
        for column in columns:
            if ('url' in column.lower()):
                tokens = self._tokenizer.tokenize(row[column], 'url')
            else:
                tokens = self._tokenizer.tokenize(row[column])
            tokens_summary = tokens_summary + ' ' + ' '.join(tokens)

        tokens_prev = ''
        if joined_column in row:
            tokens_prev = row[joined_column]

        return tokens_prev + ' ' + tokens_summary

    def get_vocab_counter(self, data_train, data_test):
        corpus = list()
        data_train.apply(lambda row: self.populate_corpus(row, self._a_cols + self._b_cols, corpus), axis=1)
        data_test.apply(lambda row: self.populate_corpus(row, self._a_cols + self._b_cols, corpus), axis=1)
        counts = Counter(corpus)
        return counts #[t[0] for t in counts.most_common(n_most_common)]

    def join_features(self, m1, m2=None):
        if self._join_method == 'concat':
            return hstack((m1, m2))
        if self._join_method == 'sum':
            return m1 + m2
        raise ValueError('Incorrect parameter "%s" for join_features!' % self._join_method)

    def add_tokens_columns(self, data_train, data_test):
        data_train[self._a_join_col] = data_train.apply(lambda row: self.prepare_tokens(row, self._a_cols, self._a_join_col), axis=1)
        data_train[self._b_join_col] = data_train.apply(lambda row: self.prepare_tokens(row, self._b_cols, self._b_join_col), axis=1)
        data_test[self._a_join_col] = data_test.apply(lambda row: self.prepare_tokens(row, self._a_cols, self._a_join_col), axis=1)
        data_test[self._b_join_col] = data_test.apply(lambda row: self.prepare_tokens(row, self._b_cols, self._b_join_col), axis=1)
        return data_train[[self._a_join_col, self._b_join_col]], data_test[[self._a_join_col, self._b_join_col]]

    def prepare_vocab(self, data_train, data_test):
        vocab_counter = self.get_vocab_counter(data_train, data_test)
        vocabulary = set([itm[0] for itm in vocab_counter.items()])
        stop_words = [w[0] for w in filter(lambda itm: itm[1] == 1, vocab_counter.iteritems())]
        vocabulary = vocabulary.difference(stop_words)
        return vocabulary

    def prepare_tfidf_matrixes(self, data_train, data_test):
        vocabulary = self.prepare_vocab(data_train, data_test)

        self.add_tokens_columns(data_train, data_test)

        # perform TF-IDF transformation
        tfidf = TfidfVectorizer(tokenizer=self._tokenizer.tokenize, vocabulary=vocabulary)
        td_matrix_a = tfidf.fit_transform(data_train['ATokens'])
        td_matrix_b = tfidf.fit_transform(data_train['BTokens'])
        # td_matrix_both = tfidf.fit_transform(data_train['BothTokens'])
        features_train = self.join_features(td_matrix_a, td_matrix_b)
        features_train = pd.DataFrame(features_train.toarray(), index=data_train.index)

        td_matrix_a = tfidf.fit_transform(data_test['ATokens'])
        td_matrix_b = tfidf.fit_transform(data_test['BTokens'])
        # td_matrix_both = tfidf.fit_transform(data_test['BothTokens'])
        features_test = self.join_features(td_matrix_a, td_matrix_b)
        features_test = pd.DataFrame(features_test.toarray(), index=data_test.index)

        return features_train, features_test

    def prepare_features(self, data_train, data_test, mode):
        if mode == 'tfidf':
            return self.prepare_tfidf_matrixes(data_train, data_test)
        raise ValueError('Incorrect parameter "%s" for prepare_features!' % mode)


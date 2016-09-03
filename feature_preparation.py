import tokenization as prep
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pandas as pd
from collections import Counter


def populate_corpus(row, columns, tokenizer, corpus):
    for column in columns:
        tokens = tokenizer(row[column])
        corpus.extend(tokens)


def prepare_tokens(row, columns, tokenizer, vocabulary, stop_words, token_min_length, joined_column):
    tokens_summary = ''
    for column in columns:
        tokens = tokenizer(row[column], stop_words, token_min_length)
        if vocabulary is not None:
            vocabulary.update(tokens)
        tokens_summary = tokens_summary + ' ' + ' '.join(tokens)

    tokens_prev = ''
    if joined_column in row:
        tokens_prev = row[joined_column]

    return tokens_prev + ' ' + tokens_summary


def add_tokens_columns(df, vocabulary, a_cols, b_cols, stop_words=[], token_min_length=4):
    df['ATokens'] = df.apply(lambda row: prepare_tokens(row, a_cols, prep.tokenize_simple,
                                                        vocabulary, stop_words, token_min_length,
                                                        'ATokens'), axis=1)
    df['BTokens'] = df.apply(lambda row: prepare_tokens(row,  b_cols, prep.tokenize_simple,
                                                        vocabulary, stop_words, token_min_length,
                                                        'BTokens'), axis=1)
    # df['BothTokens'] = df.apply(lambda row: prepare_tokens(row, a_cols + b_cols,  prep.tokenize_simple, vocabulary,
    #                                                        stop_words, token_min_length, 'BothTokens'), axis=1)


def get_most_common(data_train, data_test, a_cols, b_cols, n_most_common):
    corpus = list()
    data_train.apply(lambda row: populate_corpus(row, a_cols + b_cols, prep.tokenize_simple, corpus), axis=1)
    data_test.apply(lambda row: populate_corpus(row, a_cols + b_cols, prep.tokenize_simple, corpus), axis=1)
    counts = Counter(corpus)
    return [t[0] for t in counts.most_common(n_most_common)]


def join_features(m1, m2=None, mode='other'):
    if mode == 'concat':
        return hstack((m1, m2))
    if mode == 'sum':
        return m1 + m2
    raise ValueError('Incorrect parameter "%s" for join_features!' % mode)


def prepare_features_tfidf(data_train, data_test):
    #################################################################################
    # Main params for feature preparation
    n_most_common_words_to_ignore = 0
    token_min_length = 4
    feater_joininig_method = 'concat'  # 'concat', 'sum', 'other'
    a_cols = ['ASnippet']
    b_cols = ['BSnippet']
    #################################################################################

    # get most common tokens
    most_common = get_most_common(data_train, data_test, a_cols, b_cols, n_most_common_words_to_ignore)

    # prepare vocabulary and token columns for TF-IDF
    vocabulary = set()
    add_tokens_columns(data_train, vocabulary, a_cols, b_cols, most_common, token_min_length)
    add_tokens_columns(data_test, vocabulary, a_cols, b_cols, most_common, token_min_length)

    # perform TF-IDF transformation
    tfidf = TfidfVectorizer(tokenizer=prep.tokenize_simple, vocabulary=vocabulary)
    td_matrix_a = tfidf.fit_transform(data_train['ATokens'])
    td_matrix_b = tfidf.fit_transform(data_train['BTokens'])
    # td_matrix_both = tfidf.fit_transform(data_train['BothTokens'])
    joined_features = join_features(td_matrix_a, td_matrix_b, feater_joininig_method)
    features_train = pd.DataFrame(joined_features.toarray(), index=data_train.index)

    td_matrix_a = tfidf.fit_transform(data_test['ATokens'])
    td_matrix_b = tfidf.fit_transform(data_test['ATokens'])
    # td_matrix_both = tfidf.fit_transform(data_test['BothTokens'])
    joined_features = join_features(td_matrix_a, td_matrix_b, feater_joininig_method)
    features_test = pd.DataFrame(joined_features.toarray(), index=data_test.index)

    return features_train, features_test


def prepare_features(data_train, data_test, mode):
    if mode == 'tfidf':
        return prepare_features_tfidf(data_train, data_test)
    raise ValueError('Incorrect parameter "%s" for prepare_features!' % mode)

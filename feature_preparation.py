import tokenization as prep
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pandas as pd
from collections import Counter
from gensim.models import Word2Vec
import numpy as np


def populate_corpus(row, columns, tokenizer, corpus):
    for column in columns:
        tokens = tokenizer(row[column])
        corpus.extend(tokens)


def prepare_tokens_string(row, columns, tokenizer, vocabulary, stop_words, token_min_length, joined_column):
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


def prepare_tokens_list(row, columns, tokenizer, vocabulary, stop_words, token_min_length, joined_column):
    tokens_summary = []
    for column in columns:
        tokens = tokenizer(row[column], stop_words, token_min_length)
        if vocabulary is not None:
            vocabulary.update(tokens)
        tokens_summary = tokens_summary + tokens

    tokens_prev = []
    if joined_column in row:
        tokens_prev = row[joined_column]

    return tokens_prev + tokens_summary


def prepare_vector(row, model, column):
    total = np.zeros(model.vector_size)
    count = 0
    for token in row[column]:
        if token in model:
            total = np.add(total, model[token])
            count += 1
    return total / float(count)



def add_tokens_columns(df, vocabulary, a_cols, b_cols, stop_words=[], token_min_length=4,
                       prepare_tokens=prepare_tokens_string):
    df['ATokens'] = df.apply(lambda row: prepare_tokens(row, a_cols, prep.tokenize_simple,
                                                        vocabulary, stop_words, token_min_length,
                                                        'ATokens'), axis=1)
    df['BTokens'] = df.apply(lambda row: prepare_tokens(row,  b_cols, prep.tokenize_simple,
                                                        vocabulary, stop_words, token_min_length,
                                                        'BTokens'), axis=1)
    # df['BothTokens'] = df.apply(lambda row: prepare_tokens(row, a_cols + b_cols,  prep.tokenize_simple, vocabulary,
    #                                                        stop_words, token_min_length, 'BothTokens'), axis=1)


def add_vector_columns(df, model, a_col, b_col):
    df['AVector'] = df.apply(lambda row: prepare_vector(row, model, a_col), axis=1)
    df['BVector'] = df.apply(lambda row: prepare_vector(row, model, b_col), axis=1)


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
    a_cols = ['AUrl', 'ATitle', 'ASnippet']
    b_cols = ['BUrl', 'BTitle', 'BSnippet']
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
    td_matrix_b = tfidf.fit_transform(data_test['BTokens'])
    # td_matrix_both = tfidf.fit_transform(data_test['BothTokens'])
    joined_features = join_features(td_matrix_a, td_matrix_b, feater_joininig_method)
    features_test = pd.DataFrame(joined_features.toarray(), index=data_test.index)

    return features_train, features_test


def prepare_features_word2vec(data_train, data_test):
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
    vocabulary = set()
    add_tokens_columns(data_train, vocabulary, a_cols, b_cols, most_common, token_min_length, prepare_tokens_list)
    add_tokens_columns(data_test, vocabulary, a_cols, b_cols, most_common, token_min_length, prepare_tokens_list)

    sentences = []
    for col in ['ATokens', 'BTokens']:
        for row in data_train[col]:
            sentences.append(row)
        for row in data_test[col]:
            sentences.append(row)

    model = Word2Vec(sentences, size=100)
    add_vector_columns(data_train, model, 'ATokens', 'BTokens')
    add_vector_columns(data_test, model, 'ATokens', 'BTokens')

    joined_features = join_features(data_train['AVector'], data_train['BVector'], feater_joininig_method)
    features_train = pd.DataFrame(joined_features.toarray(), index=data_train.index)

    joined_features = join_features(data_test['AVector'], data_test['BVector'], feater_joininig_method)
    features_test = pd.DataFrame(joined_features.toarray(), index=data_test.index)

    return features_train, features_test


def prepare_features(data_train, data_test, mode):
    if mode == 'tfidf':
        return prepare_features_tfidf(data_train, data_test)
    if mode == 'word2vec':
        return prepare_features_word2vec(data_train, data_test)
    raise ValueError('Incorrect parameter "%s" for prepare_features!' % mode)

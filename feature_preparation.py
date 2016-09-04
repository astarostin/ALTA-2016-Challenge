import tokenization as prep
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pandas as pd
from collections import Counter


def populate_corpus(row, columns, tokenizer, corpus):
    for column in columns:
        if ('url' in column.lower()):
            tokens = tokenizer.tokenize(row[column], 'url')
        else:
            tokens = tokenizer.tokenize(row[column])
        corpus.extend(tokens)


def prepare_tokens(row, columns, joined_column, tokenizer):
    tokens_summary = ''
    for column in columns:
        if ('url' in column.lower()):
            tokens = tokenizer.tokenize(row[column], 'url')
        else:
            tokens = tokenizer.tokenize(row[column])
        # if vocabulary is not None:
        #     vocabulary.update(tokens)
        tokens_summary = tokens_summary + ' ' + ' '.join(tokens)

    tokens_prev = ''
    if joined_column in row:
        tokens_prev = row[joined_column]

    return tokens_prev + ' ' + tokens_summary


def add_tokens_columns(df, cols, joined_column, tokenizer):
    df[joined_column] = df.apply(lambda row: prepare_tokens(row, cols, joined_column, tokenizer), axis=1)
    # df['BothTokens'] = df.apply(lambda row: prepare_tokens(row, a_cols + b_cols,  prep.tokenize_simple, vocabulary,
    #                                                        stop_words, token_min_length, 'BothTokens'), axis=1)


def get_vocab_counter(data_train, data_test, a_cols, b_cols, tokenizer):
    corpus = list()
    data_train.apply(lambda row: populate_corpus(row, a_cols + b_cols, tokenizer, corpus), axis=1)
    data_test.apply(lambda row: populate_corpus(row, a_cols + b_cols, tokenizer, corpus), axis=1)
    counts = Counter(corpus)
    return counts #[t[0] for t in counts.most_common(n_most_common)]


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
    # most_common = get_most_common(data_train, data_test, a_cols, b_cols, n_most_common_words_to_ignore)
    tokenizer = prep.Tokenizer(token_min_length, use_nltk_tokenize=True)
    vocab_counter = get_vocab_counter(data_train, data_test, a_cols, b_cols, tokenizer)

    # prepare vocabulary and token columns for TF-IDF

    add_tokens_columns(data_train, a_cols, 'ATokens', tokenizer)
    add_tokens_columns(data_train, b_cols, 'BTokens', tokenizer)
    add_tokens_columns(data_test, a_cols, 'ATokens', tokenizer)
    add_tokens_columns(data_test, b_cols, 'BTokens', tokenizer)

    vocabulary = set(vocab_counter.items())
    stop_words = [w[0] for w in filter(lambda itm: itm[1] == 1, vocab_counter.iteritems())]
    vocabulary = vocabulary.difference(stop_words)

    # perform TF-IDF transformation
    tfidf = TfidfVectorizer(tokenizer=tokenizer.tokenize, vocabulary=vocabulary, stop_words=stop_words)
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


def prepare_features(data_train, data_test, mode):
    if mode == 'tfidf':
        return prepare_features_tfidf(data_train, data_test)
    raise ValueError('Incorrect parameter "%s" for prepare_features!' % mode)

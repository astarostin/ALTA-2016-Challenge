import tokenization as prep
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
import pandas as pd


def prepare_features_tfidf(pairs_train, pairs_test, search_train, search_test):
    # prepare vocabulary for TF-IDF
    vocabulary = set()
    pairs_train.apply(lambda row: vocabulary.update(prep.get_url_tokens(row['AUrl'])), axis=1)
    pairs_train.apply(lambda row: vocabulary.update(prep.get_url_tokens(row['BUrl'])), axis=1)
    pairs_test.apply(lambda row: vocabulary.update(prep.get_url_tokens(row['AUrl'])), axis=1)
    pairs_test.apply(lambda row: vocabulary.update(prep.get_url_tokens(row['BUrl'])), axis=1)

    search_train.apply(lambda row: vocabulary.update(prep.tokenize_simple(row['ATitle'])), axis=1)
    search_train.apply(lambda row: vocabulary.update(prep.tokenize_simple(row['BTitle'])), axis=1)
    search_test.apply(lambda row: vocabulary.update(prep.tokenize_simple(row['ATitle'])), axis=1)
    search_test.apply(lambda row: vocabulary.update(prep.tokenize_simple(row['BTitle'])), axis=1)

    search_train.apply(lambda row: vocabulary.update(prep.tokenize_simple(row['ASnippet'])), axis=1)
    search_train.apply(lambda row: vocabulary.update(prep.tokenize_simple(row['BSnippet'])), axis=1)
    search_test.apply(lambda row: vocabulary.update(prep.tokenize_simple(row['ASnippet'])), axis=1)
    search_test.apply(lambda row: vocabulary.update(prep.tokenize_simple(row['BSnippet'])), axis=1)

    # print vocabulary

    # prepare train data
    tfidf1 = TfidfVectorizer(tokenizer=prep.get_url_tokens, vocabulary=vocabulary)
    tfidf2 = TfidfVectorizer(tokenizer=prep.tokenize_simple, vocabulary=vocabulary)
    td_matrix_a = tfidf1.fit_transform(pairs_train['AUrl'])
    td_matrix_b = tfidf1.fit_transform(pairs_train['BUrl'])
    td_matrix_c = tfidf2.fit_transform(search_train['ATitle'])
    td_matrix_d = tfidf2.fit_transform(search_train['BTitle'])
    td_matrix_e = tfidf2.fit_transform(search_train['ASnippet'])
    td_matrix_f = tfidf2.fit_transform(search_train['BSnippet'])
    joined_features = hstack((td_matrix_a, td_matrix_b, td_matrix_c, td_matrix_d, td_matrix_e, td_matrix_f))
    # joined_features = td_matrix_a + td_matrix_b + td_matrix_c + td_matrix_d + td_matrix_e + td_matrix_f
    data_train = pd.DataFrame(joined_features.toarray(), index=pairs_train.index)

    # prepare test data
    td_matrix_a = tfidf1.fit_transform(pairs_test['AUrl'])
    td_matrix_b = tfidf1.fit_transform(pairs_test['BUrl'])
    td_matrix_c = tfidf2.fit_transform(search_test['ATitle'])
    td_matrix_d = tfidf2.fit_transform(search_test['BTitle'])
    td_matrix_e = tfidf2.fit_transform(search_test['ASnippet'])
    td_matrix_f = tfidf2.fit_transform(search_test['BSnippet'])
    joined_features = hstack((td_matrix_a, td_matrix_b, td_matrix_c, td_matrix_d, td_matrix_e, td_matrix_f))
    # joined_features = td_matrix_a + td_matrix_b + td_matrix_c + td_matrix_d + td_matrix_e + td_matrix_f
    data_test = pd.DataFrame(joined_features.toarray(), index=pairs_test.index)

    return data_train, data_test

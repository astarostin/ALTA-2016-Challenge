import tokenization as prep
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pandas as pd


def prepare_features_tfidf(pairs_train, pairs_test):
    # prepare vocabulary for TF-IDF
    vocabulary = set()
    pairs_train.apply(lambda row: vocabulary.update(prep.get_url_tokens(row['AUrl'])), axis=1)
    pairs_train.apply(lambda row: vocabulary.update(prep.get_url_tokens(row['BUrl'])), axis=1)
    pairs_test.apply(lambda row: vocabulary.update(prep.get_url_tokens(row['AUrl'])), axis=1)
    pairs_test.apply(lambda row: vocabulary.update(prep.get_url_tokens(row['BUrl'])), axis=1)

    # prepare train data
    tfidf = TfidfVectorizer(tokenizer=prep.get_url_tokens, vocabulary=vocabulary)
    td_matrix_a = tfidf.fit_transform(pairs_train['AUrl'])
    td_matrix_b = tfidf.fit_transform(pairs_train['BUrl'])
    joined_features = hstack((td_matrix_a, td_matrix_b))
    data_train = pd.DataFrame(joined_features.toarray(), index=pairs_train.index)

    # prepare test data
    td_matrix_a = tfidf.fit_transform(pairs_test['AUrl'])
    td_matrix_b = tfidf.fit_transform(pairs_test['BUrl'])
    joined_features = hstack((td_matrix_a, td_matrix_b))
    data_test = pd.DataFrame(joined_features.toarray(), index=pairs_test.index)

    return data_train, data_test

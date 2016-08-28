import pandas as pd
# import numpy as np
import preprocessing as prep
from sklearn.feature_extraction.text import TfidfVectorizer
# import validation
# from estimator import random as re

pairs_train = pd.read_csv('data/alta16_kbcoref_train_pairs.csv', sep=',', index_col='Id')
labels_train = pd.read_csv('data/alta16_kbcoref_train_labels.csv', sep=',', index_col='Id')
search_train = pd.read_csv('data/alta16_kbcoref_train_search_results.csv', sep=',', index_col='Id')
search_test = pd.read_csv('data/alta16_kbcoref_test_search_results.csv', sep=',', index_col='Id')
# data_train = pairs_train.join(labels_train, how='inner')
data_test = pd.read_csv('data/alta16_kbcoref_test_pairs.csv', sep=',', index_col='Id')

#print data_train.head()

# def test_random(X, y):
#     estimator = re.RandomClassifier()
#     scores = validation.validate(estimator, X, y)
#     return scores

data = pd.DataFrame()
data['A_URL_words'] = pairs_train.apply(lambda row: prep.get_url_tokens(row['AUrl']), axis=1)
data['B_URL_words'] = pairs_train.apply(lambda row: prep.get_url_tokens(row['BUrl']), axis=1)

print data.head()

tfidf = TfidfVectorizer(data['A_URL_words'])
td_matrix = tfidf.fit_transform()

print td_matrix



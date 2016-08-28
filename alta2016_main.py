import pandas as pd
import numpy as np
import data_preprocessing as prep
import validation
from estimator import random as re

pairs_train = pd.read_csv('data/alta16_kbcoref_train_pairs.csv', sep=',', index_col='Id')
labels_train = pd.read_csv('data/alta16_kbcoref_train_labels.csv', sep=',', index_col='Id')
data_train = pairs_train.join(labels_train, how='inner')

#print data_train.head()

def test_random(X, y):
    estimator = re.RandomClassifier()
    scores = validation.validate(estimator, X, y)
    return scores

pairs_train['URLWords'] = pairs_train.apply(lambda row: prep.get_url_tokens(row['AUrl']) + prep.get_url_tokens(row['BUrl']), axis=1)
data = pairs_train[['URLWords']]

X = data['URLWords'].values
y = labels_train.values.reshape(labels_train.values.shape[0])

print test_random(X, y)
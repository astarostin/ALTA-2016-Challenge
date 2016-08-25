import pandas as pd
import numpy as np

pairs_train = pd.read_csv('data/alta16_kbcoref_train_pairs.csv', sep=',', index_col='Id')
labels_train = pd.read_csv('data/alta16_kbcoref_train_labels.csv', sep=',', index_col='Id')
data_train = pairs_train.join(labels_train, how='inner')

print data_train.head()
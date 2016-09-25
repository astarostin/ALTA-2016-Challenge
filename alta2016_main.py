from feature_preparation import prepare_features
from validation import prepare_predictor
from prediction import predict
import pandas as pd
import tokenization

# load data
# pairs_train = pd.read_csv('data/alta16_kbcoref_train_pairs.csv', sep=',', index_col='Id')
# pairs_test = pd.read_csv('data/alta16_kbcoref_test_pairs.csv', sep=',', index_col='Id')
search_train = pd.read_csv('data/alta16_kbcoref_train_search_results.csv', sep=',', index_col='Id')
labels_train = pd.read_csv('data/alta16_kbcoref_train_labels.csv', sep=',', index_col='Id')
search_test = pd.read_csv('data/alta16_kbcoref_test_search_results.csv', sep=',', index_col='Id')

# prepare features
# prep = FeaturePreparator(['AUrl', 'ATitle', 'ASnippet'], ['BUrl', 'BTitle', 'BSnippet'])
# data_train, data_test = prep.prepare_features(search_train, search_test, mode='tfidf')

# target data
y = labels_train['Outcome']

# grid search for best parameters
# possible modes: logreg
# predictor, cv_score = prepare_predictor(data_train, y, mode='logreg')
# predict(predictor, data_train, y, data_test, cv_score)
predictor, cv_score = prepare_predictor(data_train, y, mode='boosting')

#use pipeline
prep = FeaturePreparator(['AUrl', 'ATitle', 'ASnippet'], ['BUrl', 'BTitle', 'BSnippet'])
predictor, cv_score, prediction = pipeline(search_train, search_test, y, prep, 'NB')
print_prediction(predictor, prediction, search_test.index, cv_score)

print('score=%f') % cv_score

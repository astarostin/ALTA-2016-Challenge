from feature_preparation import prepare_features_tfidf
from validation import test_logistic_regression, test_logistic_regression_cv
from prediction import predict

# load data
pairs_train = pd.read_csv('data/alta16_kbcoref_train_pairs.csv', sep=',', index_col='Id')
labels_train = pd.read_csv('data/alta16_kbcoref_train_labels.csv', sep=',', index_col='Id')
search_train = pd.read_csv('data/alta16_kbcoref_train_search_results.csv', sep=',', index_col='Id')
search_test = pd.read_csv('data/alta16_kbcoref_test_search_results.csv', sep=',', index_col='Id')
pairs_test = pd.read_csv('data/alta16_kbcoref_test_pairs.csv', sep=',', index_col='Id')

# prepare features
data_train, data_test = prepare_features_tfidf(pairs_train, pairs_test)

# target data
y = labels_train['Outcome']

# cross-validation
predictor = test_logistic_regression(data_train, y)

# grid search for best parameters
predictor = test_logistic_regression_cv(data_train, y)

# make predictions
predict(predictor, data_train, y, data_test)


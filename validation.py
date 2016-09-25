from sklearn.cross_validation import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from vectorization import PairVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
from sklearn.pipeline import Pipeline
import tokenization as prep

def get_fold(n):
    return KFold(n, n_folds=3, shuffle=True, random_state=42)


def test_boosting(X, y):
    predictor = GradientBoostingClassifier(loss='deviance', learning_rate=0.3, n_estimators=100, random_state=42)
    score = validate(predictor, X, y)
    print 'Cross Validaton for GradientBoostingClassifier: ' + str(score)
    return predictor, score


def test_logistic_regression_cv(X, y):
    predictor = LogisticRegression(penalty='l1', random_state=42)
    estimators_grid = {'C': [10 ** k for k in np.arange(-3, 1, 0.05)]}
    # estimators_grid = {'C': np.arange(0.01, 2, 0.01)}
    gs = grid_search(predictor, estimators_grid, X, y)
    print 'Grid Search CV for Logistic Regression. Best parameter C = %.4f, best score = %.8f' % \
          (gs.best_params_['C'], gs.best_score_)

    return gs.best_estimator_, gs.best_score_


def test_rf_cv(X, y):
    predictor = RandomForestClassifier(random_state=42)
    estimators_grid = {'criterion': ['entropy', 'gini'], 'n_estimators': [150, 200, 250]}
    gs = grid_search(predictor, estimators_grid, X, y)
    print 'Grid Search CV for RandomForestClassifier. Best parameter n_estimators = %d, ' \
          'criterion = %s, best score = %.8f' % \
          (gs.best_params_['n_estimators'], gs.best_params_['criterion'], gs.best_score_)

    return gs.best_estimator_, gs.best_score_


def test_boosting_cv(X, y):
    predictor = GradientBoostingClassifier(random_state=42)
    estimators_grid = {'loss': ['exponential'], 'n_estimators': [105],
                       'learning_rate': [0.006, 0.007, 0.0075, 0.008]}
    gs = grid_search(predictor, estimators_grid, X, y)
    print 'Grid Search CV for GradientBoostingClassifier. Best parameter ' \
          'loss = %s, n_estimators = %d, learning_rate = %.8f best score = %.8f' % \
          (gs.best_params_['loss'], gs.best_params_['n_estimators'], gs.best_params_['learning_rate'], gs.best_score_)

    return gs.best_estimator_, gs.best_score_


def test_cv(predictor, estimators_grid, X, y):
    gs = grid_search(predictor, estimators_grid, X, y)
    print 'Grid Search CV for %s. Best params:' % predictor
    for p in gs.best_params_:
        print '%s: %s' % (p, str(gs.best_params_[p]))
    print 'Best score: %.8f' % gs.best_score_

    return gs.best_estimator_, gs.best_score_


def validate(predictor, X, y):
    kf = get_fold(len(X))
    scores = cross_val_score(predictor, X, y, scoring='f1', cv=kf)
    return scores.mean()


def grid_search(predictor, estimators_grid, X, y):
    kf = get_fold(len(X))
    gs = GridSearchCV(predictor, estimators_grid, scoring='f1', cv=kf)
    gs.fit(X, y)
    return gs


def prepare_predictor(X, y, mode='logreg'):
    if mode == 'logreg':
        return test_logistic_regression_cv(X, y)
    if mode == 'forest':
        return test_rf_cv(X, y)
    if mode == 'boosting':
        return test_boosting_cv(X, y)
    raise ValueError('Incorrect parameter "%s" for prepare_predictor!' % mode)


def pipeline(X_train, X_test, y_train, preparator, mode):
    if mode == 'NB':
        predictor = MultinomialNB()
        params = {'clf__alpha': [0, 1, 2]}
    elif mode == 'SGD':
        predictor = SGDClassifier(penalty='l2', n_iter=5, random_state=42)
        params = {'clf__loss': ['hinge', 'perceptron'],
                  'clf__alpha': [1e-2, 1e-3, 1e-4],
                  'clf__penalty': ['l1', 'l2'],
                  }
    else:
        raise ValueError('Incorrect parameter "%s" for pipeline!' % mode)

    vocab = preparator.prepare_vocab(X_train, X_test)
    train, test = preparator.add_tokens_columns(X_train, X_test)
    text_clf = Pipeline([('tfidf', PairVectorizer(preparator._a_join_col, preparator._b_join_col, join_method='concat', tokenizer=preparator._tokenizer.tokenize, vocabulary=vocab)),
                         ('clf', predictor),
                         ])

    params.update({ 'tfidf__use_idf': (True, False) })
    best_estimator, best_score = test_cv(text_clf, params, train, y_train)
    prediction = best_estimator.predict(test)
    return best_estimator, best_score, prediction


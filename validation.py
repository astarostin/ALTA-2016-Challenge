from sklearn.cross_validation import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
import numpy as np


def get_fold(n):
    return KFold(n, n_folds=3, shuffle=True, random_state=42)


def test_logistic_regression(X, y):
    predictor = LogisticRegression()
    print 'Cross Validaton for Logistic Regression: ' + str(validate(predictor, X, y))
    return predictor


def test_logistic_regression_cv(X, y):
    predictor = LogisticRegression(penalty='l1', random_state=42)
    estimators_grid = {'C': [10 ** k for k in np.arange(-3, 1, 0.05)]}
    # estimators_grid = {'C': np.arange(0.01, 2, 0.01)}
    gs = grid_search(predictor, estimators_grid, X, y)
    print 'Grid Search CV for Logistic Regression. Best parameter C = %.4f, best score = %.8f' % \
          (gs.best_params_['C'], gs.best_score_)

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
    raise ValueError('Incorrect parameter "%s" for prepare_predictor!' % mode)

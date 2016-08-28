from sklearn.cross_validation import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
import numpy as np


def get_fold(n):
    return KFold(n, n_folds=5, shuffle=True, random_state=42)


def test_logistic_regression(X, y):
    predictor = LogisticRegression()
    print 'Cross Validaton for Logistic Regression: ' + str(validate(predictor, X, y))
    return predictor


def test_logistic_regression_cv(X, y):
    predictor = LogisticRegression()
    estimators_grid = {'C': [10 ** k for k in np.arange(-3, 0.1, 0.1)]}
    gs = grid_search(predictor, estimators_grid, X, y)
    print 'Grid Search CV for Logistic Regression. Best parameter C = %.4f, best score = %.8f' % \
          (gs.best_params_['C'], gs.best_score_)

    return LogisticRegression(C=gs.best_params_['C'])


def validate(predictor, X, y):
    kf = get_fold(len(X))
    scores = cross_val_score(predictor, X, y, scoring='f1', cv=kf)
    return scores.mean()


def grid_search(predictor, estimators_grid, X, y):
    kf = get_fold(len(X))
    gs = GridSearchCV(predictor, estimators_grid, scoring='f1', cv=kf)
    gs.fit(X, y)
    return gs

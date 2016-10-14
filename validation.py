from sklearn.cross_validation import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm  import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np


def get_fold(n):
    return KFold(n, n_folds=5, shuffle=True, random_state=42)


def test_boosting(X, y):
    predictor = GradientBoostingClassifier(loss='deviance', learning_rate=0.3, n_estimators=100, random_state=42)
    score = validate(predictor, X, y)
    print 'Cross Validaton for GradientBoostingClassifier: ' + str(score)
    return predictor, score


def test_logistic_regression_cv(X, y):
    predictor = LogisticRegression(penalty='l1', random_state=42)
    estimators_grid = {'C': [10 ** k for k in np.arange(-2, 2.0, 0.05)]}
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
    estimators_grid = {'loss': ['exponential'], 'n_estimators': [50, 100, 150],
                       'learning_rate': [0.005, 0.01, 0.05, 0.1]}
    gs = grid_search(predictor, estimators_grid, X, y)
    print 'Grid Search CV for GradientBoostingClassifier. Best parameter ' \
          'loss = %s, n_estimators = %d, learning_rate = %.8f best score = %.8f' % \
          (gs.best_params_['loss'], gs.best_params_['n_estimators'], gs.best_params_['learning_rate'], gs.best_score_)

    return gs.best_estimator_, gs.best_score_


def test_svm_cv(X, y):
    predictor = LinearSVC(random_state=42)
    estimators_grid = {'loss': ['hinge', 'squared_hinge'], 'penalty': ['l2'],
                       'C': [1]}
    gs = grid_search(predictor, estimators_grid, X, y)
    print 'Grid Search CV for LinearSVC. Best parameter ' \
          'loss = %s, C = %.8f best score = %.8f' % \
          (gs.best_params_['loss'], gs.best_params_['C'], gs.best_score_)

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
    if mode == 'svm':
        return test_svm_cv(X, y)
    raise ValueError('Incorrect parameter "%s" for prepare_predictor!' % mode)

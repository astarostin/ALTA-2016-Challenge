from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import operator

class RandomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass
        self.classes_, indices = np.unique(y, return_inverse=True)
        return self

    def predict(self, X):
        return np.random.random_integers(0, 2, X.shape[0])


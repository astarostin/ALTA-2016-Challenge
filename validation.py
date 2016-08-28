from sklearn import cross_validation
import numpy as np

def validate(estimator, data, target):
    scores = cross_validation.cross_val_score(estimator, data, target, scoring='f1', cv = 3)
    return scores.mean()
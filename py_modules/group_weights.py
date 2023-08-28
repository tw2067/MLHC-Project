from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels


def compute_weights(weight_col):
    return compute_sample_weight("balanced", weight_col)


class WeightedClf(BaseEstimator, ClassifierMixin):
    def __init__(self, group_col, clf):
        self.group_col = group_col
        self.clf = clf

    def fit(self, seen_data, y):
        self.classes_ = unique_labels(y)
        weights = compute_weights(seen_data[self.group_col])
        self.clf.fit(seen_data.drop(self.group_col, axis=1), y, sample_weight=weights)
        return self

    def predict(self, X):
        pred = self.clf.predict(X.drop(self.group_col, axis=1))
        return pred

    def predict_proba(self, X):
        pred = self.predict_proba(X.drop(self.group_col, axis=1))
        return pred


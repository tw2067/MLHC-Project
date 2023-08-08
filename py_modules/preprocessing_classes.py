from sklearn.base import TransformerMixin, BaseEstimator


class GLNormalizations(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, seen_data, y=None):
        return self

    def transform(self, X, y=None):
        pass


class PercentileNormalization(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, percentiles):
        self.group_col = group_col
        self.percentiles = percentiles

    def fit(self, seen_data, y=None):
        return self

    def transform(self, X, y=None):
        pass


class GroupNormalization(BaseEstimator, TransformerMixin):
    def __init__(self, group_col):
        self.group_col = group_col

    def fit(self, seen_data, y=None):
        return self

    def transform(self, X, y=None):
        pass

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit, root_scalar
from statsmodels.distributions.empirical_distribution import ECDF
import preprocessing_utils as putil
import numpy as np
import pandas as pd


class GLScaler(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, num_cols):
        self.group_col = group_col
        self.num_cols = num_cols

        self.f = lambda x, Q, B, M, v: 1 / (1 + Q * np.exp(-B * (x - M))) ** (1 / v)
        self.gl_dict = dict()

        self.groups = None

    def fit(self, seen_data, y=None):
        self.groups = seen_data[self.group_col].unique()
        for group in self.groups:
            group_data = seen_data[seen_data[group] == group]
            self.gl_dict[group] = dict()

            for feature in self.num_cols:
                # parameter initializations
                xmin, xmax, xmed = np.min(group_data[feature]), np.max(group_data[feature]), np.median(
                    group_data[feature])
                M0 = xmed
                fq0 = lambda q0: (1 / (1 + q0 * np.exp(
                    (np.log((1 + q0) ** np.log2(10) - 1) - np.log(q0))
                    * (xmax - xmed) / (xmin - xmed)))) - (0.9 ** np.log2(1 + q0))
                Q0 = root_scalar(fq0, method='newton')
                v0 = np.log2(1 + Q0)
                B0 = (np.log((1 + Q0) ** np.log2(10) - 1) - np.log(Q0)) / (xmed - xmin)

                # ECDF
                ecdf = ECDF(group_data[feature])
                y_ecdf = ecdf(group_data[feature])

                # fitting of the ECDF using the GL algorithm
                popt, pcov = curve_fit(self.f, xdata=group_data[feature], ydata=y_ecdf, p0=[Q0, B0, M0, v0])
                Q, B, M, v = popt[0], popt[1], popt[2], popt[3]
                self.gl_dict[group][feature] = lambda x: 1 / (1 + Q * np.exp(-B * (x - M))) ** (1 / v)

        return self

    def transform(self, X, y=None):
        X = X.copy()
        for group in self.groups:
            group_mask = X[self.group_col] = group
            for feature in self.num_cols:
                X.loc[group_mask, feature] = X.loc[group_mask, feature].apply(self.gl_dict[group][feature])

        return X


class PercentileScaler(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, num_cols, percentiles, fit_percentiles=False):
        self.group_col = group_col
        self.num_cols = num_cols
        self.percentiles = percentiles
        self.fit_percentiles = fit_percentiles
        self.groups = None

    def fit(self, seen_data, y=None):
        self.groups = seen_data[self.group_col].unique()
        if self.fit_percentiles:
            self.percentiles = putil.calculate_percentiles(seen_data)

        return self

    def transform(self, X, y=None):
        X = X.copy()
        for group in self.groups:
            group_mask = X[self.group_col] = group
            for feature in self.num_cols:
                X.loc[group_mask, feature] = pd.cut(X.loc[group_mask, feature],
                                                    self.percentiles[(feature, group)][0],
                                                    labels=self.percentiles[(feature, group)][1],
                                                    right=False)
        return X


class GroupNormalization(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, num_cols):
        self.group_col = group_col
        self.num_cols = num_cols
        self.scalers = dict()
        self.groups = None

    def fit(self, seen_data, y=None):
        self.groups = seen_data[self.group_col].unique()
        for group in self.groups:
            group_mask = seen_data[self.group_col] = group
            self.scalers[group] = StandardScaler().fit(seen_data.loc[group_mask, self.num_cols])

        return self

    def transform(self, X, y=None):
        X = X.copy()
        for group in self.groups:
            group_mask = X[self.group_col] = group
            X[group_mask, self.num_cols] = self.scalers[group].transform(X[group_mask, self.num_cols])

        return X


class GroupImputer(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols, groups):
        self.num_cols = num_cols
        self.groups = groups

    def fit(self, seen_data, y=None):
        return self

    def transform(self, X, y=None):
        pass

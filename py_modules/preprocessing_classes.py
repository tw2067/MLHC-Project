from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.optimize import curve_fit, root_scalar
from statsmodels.distributions.empirical_distribution import ECDF
import preprocessing_utils as putil
import numpy as np
import pandas as pd


class GLScaler(BaseEstimator, TransformerMixin):
    """
    Scaling by fitting GL function to the ECDF
    """

    def __init__(self, group_col, num_cols):
        self.group_col = group_col
        self.num_cols = num_cols

        self.f = lambda x, Q, B, M, v: 1 / (1 + Q * np.exp(-B * (x - M))) ** (1 / v)
        self.gl_dict = dict()

        self.groups = None

    def fit(self, seen_data, y=None):
        self.groups = seen_data[self.group_col].unique()
        for group in self.groups:
            group_data = seen_data[seen_data[self.group_col] == group]
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
            group_mask = X[self.group_col] == group
            for feature in self.num_cols:
                X.loc[group_mask, feature] = X.loc[group_mask, feature].apply(self.gl_dict[group][feature])

        return X


class PercentileScaler(BaseEstimator, TransformerMixin):
    """
    Scaling by mapping to percentile range
    """

    def __init__(self, group_col, num_cols, percentiles=None, labels=None, fit_percentiles=False, lower_b=None,
                 upper_b=None):
        self.group_col = group_col
        self.num_cols = num_cols
        self.percentiles = percentiles
        self.fit_percentiles = fit_percentiles
        self.lower_b = lower_b
        self.upper_b = upper_b
        self.groups = None
        self.labels = labels

    def fit(self, seen_data, y=None):
        self.groups = seen_data[self.group_col].unique()
        if self.fit_percentiles:
            self.percentiles = putil.calculate_percentiles(seen_data, self.num_cols, self.group_col,
                                                           self.lower_b, self.upper_b)
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for group in self.groups:
            group_mask = X[self.group_col] == group
            for feature in self.num_cols:
                X.loc[group_mask, feature] = pd.cut(X.loc[group_mask, feature],
                                                    self.percentiles[(feature, group)],
                                                    labels=self.labels,
                                                    right=False)
        X[self.num_cols] = X[self.num_cols].astype(int)
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
            group_mask = seen_data[self.group_col] == group
            self.scalers[group] = StandardScaler().fit(seen_data.loc[group_mask, self.num_cols])

        return self

    def transform(self, X, y=None):
        X = X.copy()
        for group in self.groups:
            group_mask = X[self.group_col] == group
            X.loc[group_mask, self.num_cols] = self.scalers[group].transform(X.loc[group_mask, self.num_cols])

        return X


class GroupImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, num_cols):
        self.num_cols = num_cols
        self.group_col = group_col
        self.groups = None
        self.imputers = dict()

    def fit(self, seen_data, y=None):
        self.groups = seen_data[self.group_col].unique()
        for group in self.groups:
            group_mask = seen_data[self.group_col] == group
            self.imputers[group] = SimpleImputer().fit(seen_data.loc[group_mask, self.num_cols])

        return self

    def transform(self, X, y=None):
        X = X.copy()
        for group in self.groups:
            group_mask = X[self.group_col] == group
            X.loc[group_mask, self.num_cols] = self.imputers[group].transform(X.loc[group_mask, self.num_cols])

        return X


class StandardColsScale(BaseEstimator, TransformerMixin):
    def __init__(self, general_num_cols, scaler):
        self.general_num_cols = general_num_cols
        self.scaler = scaler

    def fit(self, seen_data, y=None):
        self.scaler.fit(seen_data[self.general_num_cols])
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[self.general_num_cols] = self.scaler.transform(X[self.general_num_cols])
        return X


class ColFilter(BaseEstimator, TransformerMixin):
    def __init__(self, model_cols):
        self.model_cols = model_cols

    def fit(self, seen_data, y=None):
        return self

    def transform(self, X, y):
        return X[self.model_cols]

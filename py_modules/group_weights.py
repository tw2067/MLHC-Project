from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
import pandas as pd
import preprocessing_classes as pclass
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import pipeline as pl
import warnings
import sys

warnings.filterwarnings("ignore")


def compute_weights(weight_col):
    weight_col = pd.Categorical(weight_col).codes
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
        pred = self.clf.predict_proba(X.drop(self.group_col, axis=1))
        return pred


def weighted_cv(data, num_cols, model_cols, x_cols, log_name, non_white=False):
    groups = data['subject_id']
    params = [{'impute': [pclass.GroupImputer('age_gender', num_cols)],
               'scale': [pclass.StandardColsScale(num_cols, pclass.StandardScaler())],
               'filter': [pclass.ColFilter(['target'] + model_cols)],
               'model': [WeightedClf('target', RandomForestClassifier(random_state=0)),
                         WeightedClf('target', XGBClassifier(random_state=0)),
                         WeightedClf('target', CatBoostClassifier(random_seed=0))]},
              {'impute': [pclass.GroupImputer('age_gender', num_cols)],
               'scale': [pclass.StandardColsScale(num_cols, pclass.StandardScaler())],
               'filter': [pclass.ColFilter(['age_gender_target'] + model_cols)],
               'model': [WeightedClf('age_gender_target', RandomForestClassifier(random_state=0)),
                         WeightedClf('age_gender_target', XGBClassifier(random_state=0)),
                         WeightedClf('age_gender_target', CatBoostClassifier(random_seed=0))]},
              {'impute': [pclass.GroupImputer('age_gender', num_cols)],
               'scale': [pclass.StandardColsScale(num_cols, pclass.StandardScaler())],
               'filter': [pclass.ColFilter(['eth_gender_target'] + model_cols)],
               'model': [WeightedClf('eth_gender_target', RandomForestClassifier(random_state=0)),
                         WeightedClf('eth_gender_target', XGBClassifier(random_state=0)),
                         WeightedClf('eth_gender_target', CatBoostClassifier(random_seed=0))]},
              {'impute': [pclass.GroupImputer('age_gender', num_cols)],
               'scale': [pclass.StandardColsScale(num_cols, pclass.StandardScaler())],
               'filter': [pclass.ColFilter(['eth_age_gender_target'] + model_cols)],
               'model': [WeightedClf('eth_age_gender_target', RandomForestClassifier(random_state=0)),
                         WeightedClf('eth_age_gender_target', XGBClassifier(random_state=0)),
                         WeightedClf('eth_age_gender_target', CatBoostClassifier(random_seed=0))]}]
    sys.stdout = open(f"{log_name}.txt", "w")
    cv_model = pl.run_cv(data[x_cols], data['target'], model_cols, groups, 5, params,
                         ['roc_auc', 'average_precision'], split_col=data['eth_age_gender_target'], non_white=non_white)
    sys.stdout.close()
    return cv_model


if __name__ == "__main__":
    mimic_model_data = pd.read_csv('data/all_model_data.csv')

    num_cols = ['ALBUMIN', 'ANION GAP', 'BICARBONATE', 'BILIRUBIN', 'BUN', 'CHLORIDE', 'CREATININE', 'DiasBP',
                'GLUCOSE', 'Glucose', 'HEMATOCRIT', 'HEMOGLOBIN', 'HeartRate', 'INR', 'LACTATE', 'MAGNESIUM',
                'MeanBP', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PT', 'PTT', 'RespRate', 'SODIUM', 'SysBP',
                'TempC', 'WBC', 'Weight']

    model_cols = ['age', 'eth_asian', 'eth_black', 'eth_hispanic', 'eth_other', 'eth_white', 'gender',
                  'ALBUMIN', 'ANION GAP', 'BICARBONATE', 'BILIRUBIN', 'BUN', 'CHLORIDE', 'CREATININE',
                  'DiasBP', 'GLUCOSE', 'Glucose', 'HEMATOCRIT', 'HEMOGLOBIN', 'HeartRate', 'INR',
                  'LACTATE', 'MAGNESIUM', 'MeanBP', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PT', 'PTT',
                  'RespRate', 'SODIUM', 'SysBP', 'TempC', 'WBC', 'Weight']

    x_cols = ['age_gender', 'eth_gender', 'eth_age_gender', 'age', 'eth_asian', 'eth_black', 'eth_hispanic',
              'eth_other', 'eth_white', 'gender', 'ALBUMIN', 'ANION GAP', 'BICARBONATE', 'BILIRUBIN', 'BUN',
              'CHLORIDE', 'CREATININE', 'DiasBP', 'GLUCOSE', 'Glucose', 'HEMATOCRIT', 'HEMOGLOBIN', 'HeartRate',
              'INR', 'LACTATE', 'MAGNESIUM', 'MeanBP', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PT', 'PTT',
              'RespRate', 'SODIUM', 'SysBP', 'TempC', 'WBC', 'Weight', 'age_gender_target', 'eth_gender_target',
              'eth_age_gender_target', 'target']

    cv_model = weighted_cv(mimic_model_data, num_cols, model_cols, x_cols, 'weighted_samples_log')
    pd.DataFrame(cv_model.cv_results_).to_csv('results/weighted_samples_cv.csv')

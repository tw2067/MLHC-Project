import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import preprocessing_classes as pclass
import preprocessing_utils as quant
import pipeline as pl
import warnings
import sys

warnings.filterwarnings("ignore")

model_data1 = pd.read_csv('data/hw1_model_scaling.csv')
model_data2 = pd.read_csv('data/hw2_model_scaling.csv')
hourly_data = pd.read_csv('data/hourly_model_scaling.csv')
dists_data = pd.read_csv('data/dists_data.csv')
metadata = pd.read_csv('data/metadata.csv')

percentiles_ag = dists_data.groupby(['feature name', 'age_gender'])['value'].agg([quant.q10, quant.q35, quant.q65,
                                                                                  quant.q85])
percentiles_eag = dists_data.groupby(['feature name', 'eth_age_gender'])['value'].agg([quant.q10, quant.q35, quant.q65,
                                                                                       quant.q85])
percentiles_eg = dists_data.groupby(['feature name', 'eth_gender'])['value'].agg([quant.q10, quant.q35, quant.q65,
                                                                                  quant.q85])
features_metadata = metadata.drop_duplicates(subset=['feature name'])
lower = {feature: minval - 1 for feature, minval in zip(features_metadata['feature name'], features_metadata['min'])}
upper = {feature: maxval + 1 for feature, maxval in zip(features_metadata['feature name'], features_metadata['max'])}

pdict_ag = dict()
for feature, group in percentiles_ag.index:
    pdict_ag[(feature, group)] = [lower[feature]] + list(percentiles_ag.loc[(feature, group)]) + [upper[feature]]

pdict_eag = dict()
for feature, group in percentiles_eag.index:
    pdict_eag[(feature, group)] = [lower[feature]] + list(percentiles_eag.loc[(feature, group)]) + [upper[feature]]

pdict_eg = dict()
for feature, group in percentiles_eg.index:
    pdict_eg[(feature, group)] = [lower[feature]] + list(percentiles_eg.loc[(feature, group)]) + [upper[feature]]


def scaling_cv(model_data, num_cols, model_cols, x_cols, logname):
    sys.stdout = open(f"{logname}.txt", "w")
    groups = model_data['subject_id']
    labels = ['0', '1', '2', '3', '4']

    params = {'impute': [pclass.GroupImputer('age_gender', num_cols), pclass.GroupImputer('eth_gender', num_cols),
                         pclass.GroupImputer('eth_age_gender', num_cols)],
              'scale': [pclass.StandardColsScale(num_cols, pclass.StandardScaler()),
                        pclass.PercentileScaler('age_gender', num_cols, percentiles=pdict_ag, labels=labels),
                        pclass.PercentileScaler('eth_gender', num_cols, percentiles=pdict_eg, labels=labels),
                        pclass.GLScaler('age_gender', num_cols, 10),
                        pclass.GLScaler('eth_gender', num_cols, 10),
                        pclass.GroupNormalization('age_gender', num_cols),
                        pclass.GroupNormalization('eth_gender', num_cols),
                        pclass.GroupNormalization('eth_age_gender', num_cols)],
              'model': [RandomForestClassifier(random_state=0),
                        XGBClassifier(random_state=0),
                        CatBoostClassifier(random_seed=0)]}

    cv_model = pl.run_cv(model_data[x_cols], model_data['target'], model_cols, groups, 5, params,
                         ['roc_auc', 'average_precision'], split_col=model_data['eth_age_gender_target'])
    sys.stdout.close()
    return cv_model


num_cols1 = ['ALBUMIN', 'ANION GAP', 'BICARBONATE', 'BILIRUBIN', 'BUN', 'CHLORIDE', 'CREATININE', 'DiasBP',
             'GLUCOSE', 'HEMATOCRIT', 'HEMOGLOBIN', 'HeartRate', 'INR', 'LACTATE', 'MAGNESIUM',
             'MeanBP', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PT', 'PTT', 'RespRate', 'SODIUM', 'SysBP',
             'TempC', 'WBC', 'Weight']
model_cols1 = ['age', 'eth_asian', 'eth_black', 'eth_hispanic', 'eth_other', 'eth_white', 'gender',
               'ALBUMIN', 'ANION GAP', 'BICARBONATE', 'BILIRUBIN', 'BUN', 'CHLORIDE', 'CREATININE',
               'DiasBP', 'GLUCOSE', 'HEMATOCRIT', 'HEMOGLOBIN', 'HeartRate', 'INR',
               'LACTATE', 'MAGNESIUM', 'MeanBP', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PT', 'PTT',
               'RespRate', 'SODIUM', 'SysBP', 'TempC', 'WBC', 'Weight']
x_cols1 = ['age_gender', 'eth_gender', 'eth_age_gender', 'age', 'eth_asian', 'eth_black', 'eth_hispanic',
           'eth_other', 'eth_white', 'gender', 'ALBUMIN', 'ANION GAP', 'BICARBONATE', 'BILIRUBIN', 'BUN',
           'CHLORIDE', 'CREATININE', 'DiasBP', 'GLUCOSE', 'HEMATOCRIT', 'HEMOGLOBIN', 'HeartRate',
           'INR', 'LACTATE', 'MAGNESIUM', 'MeanBP', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PT', 'PTT',
           'RespRate', 'SODIUM', 'SysBP', 'TempC', 'WBC', 'Weight']

num_cols2 = ['ALBUMIN', 'ANION GAP', 'BICARBONATE', 'BILIRUBIN', 'BUN', 'CHLORIDE', 'CREATININE', 'DiasBP',
             'GLUCOSE', 'HEMATOCRIT', 'HEMOGLOBIN', 'HeartRate', 'INR', 'LACTATE', 'MAGNESIUM',
             'MeanBP', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PT', 'PTT', 'RespRate', 'SODIUM', 'SysBP',
             'TempC', 'WBC']
model_cols2 = ['age', 'eth_asian', 'eth_black', 'eth_hispanic', 'eth_other', 'eth_white', 'gender',
               'Hypertension', 'Diabetes', 'Cardiovascular Disorder', 'Chronic Obstructive Pulmonary Disease',
               'Malignant Neoplasm', 'Chronic Kidney Disease', 'days_since_admission',
               'ALBUMIN', 'ANION GAP', 'BICARBONATE', 'BILIRUBIN', 'BUN', 'CHLORIDE', 'CREATININE',
               'DiasBP', 'GLUCOSE', 'HEMATOCRIT', 'HEMOGLOBIN', 'HeartRate', 'INR',
               'LACTATE', 'MAGNESIUM', 'MeanBP', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PT', 'PTT',
               'RespRate', 'SODIUM', 'SysBP', 'TempC', 'WBC']
x_cols2 = ['age_gender', 'eth_gender', 'eth_age_gender', 'age', 'eth_asian', 'eth_black', 'eth_hispanic',
           'eth_other', 'eth_white', 'gender', 'Hypertension', 'Diabetes', 'Cardiovascular Disorder',
           'Chronic Obstructive Pulmonary Disease', 'Malignant Neoplasm', 'Chronic Kidney Disease',
           'days_since_admission', 'ALBUMIN', 'ANION GAP', 'BICARBONATE', 'BILIRUBIN', 'BUN',
           'CHLORIDE', 'CREATININE', 'DiasBP', 'GLUCOSE', 'HEMATOCRIT', 'HEMOGLOBIN', 'HeartRate',
           'INR', 'LACTATE', 'MAGNESIUM', 'MeanBP', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PT', 'PTT',
           'RespRate', 'SODIUM', 'SysBP', 'TempC', 'WBC']

cv_model1 = scaling_cv(model_data1, num_cols1, model_cols1, x_cols1, 'hw1_scaling')
pd.DataFrame(cv_model1.cv_results_).to_csv('results/hw1_cv_scaling.csv')

cv_model2 = scaling_cv(model_data2, num_cols2, model_cols2, x_cols2, 'hw2_scaling')
pd.DataFrame(cv_model2.cv_results_).to_csv('results/hw2_cv_scaling.csv')

cv_hourly = scaling_cv(hourly_data, num_cols2, model_cols2, x_cols2, 'hourly_model_scaling')
pd.DataFrame(cv_hourly.cv_results_).to_csv('results/hourly_model_scaling.csv')

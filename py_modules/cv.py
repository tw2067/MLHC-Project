import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import preprocessing_classes as pclass
import percentiles_funcs as quant
import pipeline as pl

sns.set_palette('Set3')

model_data = pd.read_csv('data/model_data.csv')
dists_data = pd.read_csv('data/dists_data.csv')
metadata = pd.read_csv('data/metadata.csv')

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
          'RespRate', 'SODIUM', 'SysBP', 'TempC', 'WBC', 'Weight']

groups = model_data['subject_id']
labels = ['0', '1', '2', '3', '4']
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

params = {'impute': [pclass.GroupImputer('age_gender', num_cols), pclass.GroupImputer('eth_gender', num_cols)],
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
                     ['roc_auc', 'average_precision'])

pd.DataFrame(cv_model.cv_results_).to_csv('results/cv_results.csv')

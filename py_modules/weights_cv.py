import pandas as pd
import group_weights as gw

hw1_data = pd.read_csv('data/hw1_data.csv')
num_cols = ['ALBUMIN', 'ANION GAP', 'BICARBONATE', 'BILIRUBIN',
            'BUN', 'CHLORIDE', 'CREATININE', 'GLUCOSE', 'HEMATOCRIT', 'HEMOGLOBIN',
            'INR', 'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM',
            'PT', 'PTT', 'SODIUM', 'WBC', 'DiasBP_min', 'Glucose_min',
            'HeartRate_min', 'MeanBP_min', 'RespRate_min', 'SpO2_min', 'SysBP_min',
            'TempC_min', 'DiasBP_max', 'Glucose_max', 'HeartRate_max', 'MeanBP_max',
            'RespRate_max', 'SpO2_max', 'SysBP_max', 'TempC_max', 'DiasBP_mean',
            'Glucose_mean', 'HeartRate_mean', 'MeanBP_mean', 'RespRate_mean',
            'SpO2_mean', 'SysBP_mean', 'TempC_mean', 'Weight']

x_cols = ['age_gender', 'gender', 'age', 'eth_asian', 'eth_white', 'eth_other', 'eth_black',
          'eth_hispanic', 'ALBUMIN', 'ANION GAP', 'BICARBONATE', 'BILIRUBIN',
          'BUN', 'CHLORIDE', 'CREATININE', 'GLUCOSE', 'HEMATOCRIT', 'HEMOGLOBIN',
          'INR', 'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM',
          'PT', 'PTT', 'SODIUM', 'WBC', 'DiasBP_min', 'Glucose_min',
          'HeartRate_min', 'MeanBP_min', 'RespRate_min', 'SpO2_min', 'SysBP_min',
          'TempC_min', 'DiasBP_max', 'Glucose_max', 'HeartRate_max', 'MeanBP_max',
          'RespRate_max', 'SpO2_max', 'SysBP_max', 'TempC_max', 'DiasBP_mean',
          'Glucose_mean', 'HeartRate_mean', 'MeanBP_mean', 'RespRate_mean',
          'SpO2_mean', 'SysBP_mean', 'TempC_mean', 'Weight', 'target',
          'age_gender_target', 'eth_gender_target', 'eth_age_gender_target']

model_cols = ['gender', 'age', 'eth_asian', 'eth_white', 'eth_other', 'eth_black',
              'eth_hispanic', 'ALBUMIN', 'ANION GAP', 'BICARBONATE', 'BILIRUBIN',
              'BUN', 'CHLORIDE', 'CREATININE', 'GLUCOSE', 'HEMATOCRIT', 'HEMOGLOBIN',
              'INR', 'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM',
              'PT', 'PTT', 'SODIUM', 'WBC', 'DiasBP_min', 'Glucose_min',
              'HeartRate_min', 'MeanBP_min', 'RespRate_min', 'SpO2_min', 'SysBP_min',
              'TempC_min', 'DiasBP_max', 'Glucose_max', 'HeartRate_max', 'MeanBP_max',
              'RespRate_max', 'SpO2_max', 'SysBP_max', 'TempC_max', 'DiasBP_mean',
              'Glucose_mean', 'HeartRate_mean', 'MeanBP_mean', 'RespRate_mean',
              'SpO2_mean', 'SysBP_mean', 'TempC_mean', 'Weight']

cv_model = gw.weighted_cv(hw1_data, num_cols, model_cols, x_cols, 'hw1_weighted', non_white=True)
pd.DataFrame(cv_model.cv_results_).to_csv('results/hw1_weighted_cv_non_white.csv')


hw2_data = pd.read_csv('data/hw2_data.csv')
num_cols = ['ALBUMIN_mean', 'ANION GAP_mean', 'BICARBONATE_mean', 'BILIRUBIN_mean',
            'BUN_mean', 'CHLORIDE_mean', 'CREATININE_mean', 'DiasBP_mean',
            'GLUCOSE_mean', 'Glucose_mean', 'HEMATOCRIT_mean', 'HEMOGLOBIN_mean',
            'HeartRate_mean', 'INR_mean', 'LACTATE_mean', 'MAGNESIUM_mean', 'MeanBP_mean',
            'PHOSPHATE_mean', 'PLATELET_mean', 'POTASSIUM_mean', 'PT_mean',
            'PTT_mean', 'RespRate_mean', 'SODIUM_mean', 'SpO2_mean',
            'SysBP_mean', 'TempC_mean', 'WBC_mean', 'ALBUMIN_max',
            'ANION GAP_max', 'BICARBONATE_max', 'BILIRUBIN_max', 'BUN_max',
            'CHLORIDE_max', 'CREATININE_max', 'DiasBP_max', 'GLUCOSE_max',
            'Glucose_max', 'HEMATOCRIT_max', 'HEMOGLOBIN_max', 'HeartRate_max',
            'INR_max', 'LACTATE_max', 'MAGNESIUM_max', 'MeanBP_max',
            'PHOSPHATE_max', 'PLATELET_max', 'POTASSIUM_max', 'PT_max',
            'PTT_max', 'RespRate_max', 'SODIUM_max', 'SpO2_max',
            'SysBP_max', 'TempC_max', 'WBC_max', 'ALBUMIN_min',
            'ANION GAP_min', 'BICARBONATE_min', 'BILIRUBIN_min', 'BUN_min',
            'CHLORIDE_min', 'CREATININE_min', 'DiasBP_min', 'GLUCOSE_min',
            'Glucose_min', 'HEMATOCRIT_min', 'HEMOGLOBIN_min', 'HeartRate_min',
            'INR_min', 'LACTATE_min', 'MAGNESIUM_min', 'MeanBP_min',
            'PHOSPHATE_min', 'PLATELET_min', 'POTASSIUM_min', 'PT_min',
            'PTT_min', 'RespRate_min', 'SODIUM_min', 'SpO2_min', 'SysBP_min',
            'TempC_min', 'WBC_min']
x_cols = ['age_gender', 'gender', 'age', 'eth_asian', 'eth_white', 'eth_other', 'eth_black',
          'Hypertension', 'Diabetes', 'Cardiovascular Disorder', 'Chronic Obstructive Pulmonary Disease',
          'Malignant Neoplasm', 'Chronic Kidney Disease', 'days_since_admission',
          'eth_hispanic', 'ALBUMIN_mean', 'ANION GAP_mean', 'BICARBONATE_mean', 'BILIRUBIN_mean',
          'BUN_mean', 'CHLORIDE_mean', 'CREATININE_mean', 'DiasBP_mean',
          'GLUCOSE_mean', 'Glucose_mean', 'HEMATOCRIT_mean', 'HEMOGLOBIN_mean',
          'HeartRate_mean', 'INR_mean', 'LACTATE_mean', 'MAGNESIUM_mean', 'MeanBP_mean',
          'PHOSPHATE_mean', 'PLATELET_mean', 'POTASSIUM_mean', 'PT_mean',
          'PTT_mean', 'RespRate_mean', 'SODIUM_mean', 'SpO2_mean',
          'SysBP_mean', 'TempC_mean', 'WBC_mean', 'ALBUMIN_max',
          'ANION GAP_max', 'BICARBONATE_max', 'BILIRUBIN_max', 'BUN_max',
          'CHLORIDE_max', 'CREATININE_max', 'DiasBP_max', 'GLUCOSE_max',
          'Glucose_max', 'HEMATOCRIT_max', 'HEMOGLOBIN_max', 'HeartRate_max',
          'INR_max', 'LACTATE_max', 'MAGNESIUM_max', 'MeanBP_max',
          'PHOSPHATE_max', 'PLATELET_max', 'POTASSIUM_max', 'PT_max',
          'PTT_max', 'RespRate_max', 'SODIUM_max', 'SpO2_max',
          'SysBP_max', 'TempC_max', 'WBC_max', 'ALBUMIN_min',
          'ANION GAP_min', 'BICARBONATE_min', 'BILIRUBIN_min', 'BUN_min',
          'CHLORIDE_min', 'CREATININE_min', 'DiasBP_min', 'GLUCOSE_min',
          'Glucose_min', 'HEMATOCRIT_min', 'HEMOGLOBIN_min', 'HeartRate_min',
          'INR_min', 'LACTATE_min', 'MAGNESIUM_min', 'MeanBP_min',
          'PHOSPHATE_min', 'PLATELET_min', 'POTASSIUM_min', 'PT_min',
          'PTT_min', 'RespRate_min', 'SODIUM_min', 'SpO2_min', 'SysBP_min',
          'TempC_min', 'WBC_min', 'target', 'age_gender_target', 'eth_gender_target', 'eth_age_gender_target']

model_cols = ['gender', 'age', 'eth_asian', 'eth_white', 'eth_other', 'eth_black',
              'Hypertension', 'Diabetes', 'Cardiovascular Disorder', 'Chronic Obstructive Pulmonary Disease',
              'Malignant Neoplasm', 'Chronic Kidney Disease', 'days_since_admission',
              'eth_hispanic', 'ALBUMIN_mean', 'ANION GAP_mean', 'BICARBONATE_mean', 'BILIRUBIN_mean',
              'BUN_mean', 'CHLORIDE_mean', 'CREATININE_mean', 'DiasBP_mean',
              'GLUCOSE_mean', 'Glucose_mean', 'HEMATOCRIT_mean', 'HEMOGLOBIN_mean',
              'HeartRate_mean', 'INR_mean', 'LACTATE_mean', 'MAGNESIUM_mean', 'MeanBP_mean',
              'PHOSPHATE_mean', 'PLATELET_mean', 'POTASSIUM_mean', 'PT_mean',
              'PTT_mean', 'RespRate_mean', 'SODIUM_mean', 'SpO2_mean',
              'SysBP_mean', 'TempC_mean', 'WBC_mean', 'ALBUMIN_max',
              'ANION GAP_max', 'BICARBONATE_max', 'BILIRUBIN_max', 'BUN_max',
              'CHLORIDE_max', 'CREATININE_max', 'DiasBP_max', 'GLUCOSE_max',
              'Glucose_max', 'HEMATOCRIT_max', 'HEMOGLOBIN_max', 'HeartRate_max',
              'INR_max', 'LACTATE_max', 'MAGNESIUM_max', 'MeanBP_max',
              'PHOSPHATE_max', 'PLATELET_max', 'POTASSIUM_max', 'PT_max',
              'PTT_max', 'RespRate_max', 'SODIUM_max', 'SpO2_max',
              'SysBP_max', 'TempC_max', 'WBC_max', 'ALBUMIN_min',
              'ANION GAP_min', 'BICARBONATE_min', 'BILIRUBIN_min', 'BUN_min',
              'CHLORIDE_min', 'CREATININE_min', 'DiasBP_min', 'GLUCOSE_min',
              'Glucose_min', 'HEMATOCRIT_min', 'HEMOGLOBIN_min', 'HeartRate_min',
              'INR_min', 'LACTATE_min', 'MAGNESIUM_min', 'MeanBP_min',
              'PHOSPHATE_min', 'PLATELET_min', 'POTASSIUM_min', 'PT_min',
              'PTT_min', 'RespRate_min', 'SODIUM_min', 'SpO2_min', 'SysBP_min',
              'TempC_min', 'WBC_min']

cv_model = gw.weighted_cv(hw2_data, num_cols, model_cols, x_cols, 'hw2_weighted', non_white=True)
pd.DataFrame(cv_model.cv_results_).to_csv('results/hw2_weighted_cv_non_white.csv')

hourly_data = pd.read_csv('data/hourly_data.csv')
cv_model = gw.weighted_cv(hourly_data, num_cols, model_cols, x_cols, 'hourly_weighted', non_white=True)
pd.DataFrame(cv_model.cv_results_).to_csv('results/hourly_weighted_cv_non_white.csv')

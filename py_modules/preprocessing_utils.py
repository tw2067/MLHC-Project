import pandas as pd
import numpy as np


def extract_groups(df):
    df['age_group'] = pd.cut(df['age'], [18, 30, 40, 50, 60, 70, 80, 300],
                             right=False, labels=['18-29', '30-39', '40-49', '50-59',
                                                  '60-69', '70-79', '80p'])
    # groups for imputation and scaling methods
    df['age_gender'] = [f'{a}_{g}' for a, g in zip(df['age_group'], df['gender_l'])]
    df['eth_gender'] = [f'{e}_{g}' for e, g in zip(df['ethnicity'], df['gender_l'])]
    df['eth_age_gender'] = [f'{e}_{a}_{g}' for e, a, g in zip(df['ethnicity'], df['age_group'], df['gender_l'])]

    # groups for sample weights balancing
    df['age_gender_target'] = [f'{ag}_{t}' for ag, t in zip(df['age_gender'], df['target'])]
    df['eth_gender_target'] = [f'{eg}_{t}' for ag, t in zip(df['eth_gender'], df['target'])]
    df['eth_age_gender_target'] = [f'{eag}_{t}' for eag, t in zip(df['eth_age_gender'], df['target'])]

    return df


def calculate_percentiles(df, num_cols, group_col, lower_b, upper_b):
    qdict = dict()

    for feature in num_cols:
        percentiles = df.groupby(group_col)[feature].agg([q03, q10,
                                                          q15, q25,
                                                          q35, q65,
                                                          q75, q85,
                                                          q90, q97])
        for group in percentiles.index:
            qdict[(feature, group)] = [lower_b[feature]] + list(percentiles.loc[group]) + [upper_b[feature]]

    return qdict


def q03(x):
    return x.quantile(0.03)


def q10(x):
    return x.quantile(0.1)


def q15(x):
    return x.quantile(0.15)


def q20(x):
    return x.quantile(0.2)


def q25(x):
    return x.quantile(0.25)


def q30(x):
    return x.quantile(0.3)


def q35(x):
    return x.quantile(0.35)


def q40(x):
    return x.quantile(0.4)


def q50(x):
    return x.quantile(0.5)


def q60(x):
    return x.quantile(0.6)


def q65(x):
    return x.quantile(0.65)


def q70(x):
    return x.quantile(0.7)


def q75(x):
    return x.quantile(0.75)


def q80(x):
    return x.quantile(0.8)


def q85(x):
    return x.quantile(0.85)


def q90(x):
    return x.quantile(0.90)


def q97(x):
    return x.quantile(0.97)

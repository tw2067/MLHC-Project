import pandas as pd
import numpy as np
import percentiles_funcs as quant


def get_data():
    pass


def extract_features():
    pass


def match_dataset(df, by):
    pass


def calculate_percentiles(df, num_cols, group_col, lower_b, upper_b):
    qdict = dict()

    for feature in num_cols:
        percentiles = df.groupby(group_col)[feature].agg([quant.q03, quant.q10,
                                                          quant.q15, quant.q25,
                                                          quant.q35, quant.q65,
                                                          quant.q75, quant.q85,
                                                          quant.q90, quant.q97])
        for group in percentiles.index:
            qdict[(feature, group)] = [lower_b[feature]] + list(percentiles.loc[group]) + [upper_b[feature]]

    return qdict

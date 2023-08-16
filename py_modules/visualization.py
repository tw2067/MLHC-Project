import numpy as np
import percentiles_funcs as quant
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('Set3')


def cv_curves():
    pass


def cv_boxplots():
    pass


def percentiles_plot(df, feature_col, dir_name):
    sns.set_palette('Blues')
    labels = ['3-10', '10-15', '15-25', '25-35', '35-65', '65-75', '75-85', '85-90', '90-97']
    for feature in df[feature_col].unique():
        feature_df = df[df[feature_col] == feature]
        percentiles = feature_df.groupby('age')['value'].agg([quant.q03, quant.q10,
                                                              quant.q15, quant.q25,
                                                              quant.q35, quant.q65,
                                                              quant.q75, quant.q85,
                                                              quant.q90, quant.q97])
        for i in range(len(percentiles.columns) - 1):
            plt.fill_between(percentiles.index, percentiles.iloc[:, i], percentiles.iloc[:, i+1], label=labels[i])

        plt.xlabel('Age (years)')
        plt.ylabel(feature)
        plt.legend()
        plt.savefig(f'{dir_name}/{feature}.png')
        plt.show()

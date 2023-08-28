import numpy as np
import percentiles_funcs as quant
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('Set3')


def cv_curves():
    pass


def cv_boxplots(cv_results, id_vars, fig_size, hue, dir_name, fig_name):
    aurocs = cv_results.melt(id_vars=id_vars,
                             value_vars=['split0_test_roc_auc', 'split1_test_roc_auc', 'split2_test_roc_auc',
                                         'split3_test_roc_auc', 'split4_test_roc_auc'], value_name='AUROC')
    auprs = cv_results.melt(id_vars=id_vars,
                            value_vars=['split0_test_average_precision', 'split1_test_average_precision',
                                        'split2_test_average_precision', 'split3_test_average_precision',
                                        'split4_test_average_precision'], value_name='AUPR')
    fig, axs = plt.subplots(1, 2, figsize=fig_size)
    sns.boxplot(aurocs, x=id_vars[0], y='AUROC', hue=hue, ax=axs[0], showmeans=True,
                meanprops={'markerfacecolor': 'w', 'markeredgecolor': 'k', 'marker': 'o'})
    sns.boxplot(auprs, x=id_vars[0], y='AUPR', hue=hue, ax=axs[1], showmeans=True,
                meanprops={'markerfacecolor': 'w', 'markeredgecolor': 'k', 'marker': 'o'})
    axs[0].set_title('AUROC')
    axs[1].set_title('AUPR')
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=90)
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=90)
    axs[0].set_xlabel('')
    axs[1].set_xlabel('')
    fig.suptitle('CV boxplot')
    plt.tight_layout()
    plt.savefig(f'{dir_name}/{fig_name}.png')


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

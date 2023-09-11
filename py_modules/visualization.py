import numpy as np
import preprocessing_utils as quant
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette('Set3')


def plot_scaling_cv(cv_res, figname):
    cv_models = cv_res['param_model'].unique()
    cv_imputer = cv_res['param_impute'].unique()
    cv_scaler = cv_res['param_scale'].unique()

    model_names = {cv_models[0]: 'RF', cv_models[1]: 'XGB', cv_models[2]: 'CB'}
    imputer_names = {cv_imputer[0]: 'AG', cv_imputer[1]: 'EG', cv_imputer[2]: 'EAG'}
    scaler_names = {cv_scaler[0]: 'Standard', cv_scaler[1]: 'Percentile AG', cv_scaler[2]: 'Percentile EG',
                    cv_scaler[3]: 'GL AG', cv_scaler[4]: 'GL EG', cv_scaler[5]: 'Group AG', cv_scaler[6]: 'Group EG',
                    cv_scaler[7]: 'Group EAG'}

    cv_res['Model'] = cv_res['param_model'].apply(lambda x: model_names[x])
    cv_res['Impute'] = cv_res['param_impute'].apply(lambda x: imputer_names[x])
    cv_res['Scale'] = cv_res['param_scale'].apply(lambda x: scaler_names[x])
    aurocs = cv_res.melt(id_vars=['Model', 'Impute', 'Scale'],
                         value_vars=['split0_test_roc_auc', 'split1_test_roc_auc', 'split2_test_roc_auc',
                                     'split3_test_roc_auc', 'split4_test_roc_auc'], value_name='AUROC')
    auprs = cv_res.melt(id_vars=['Model', 'Impute', 'Scale'],
                        value_vars=['split0_test_average_precision', 'split1_test_average_precision',
                                    'split2_test_average_precision', 'split3_test_average_precision',
                                    'split4_test_average_precision'], value_name='AUPR')
    fig, axs = plt.subplots(1, 2, figsize=(21, 6))
    sns.boxplot(aurocs, x='Model', y='AUROC', hue=aurocs[['Impute', 'Scale']].apply(tuple, axis=1), ax=axs[0],
                showmeans=True,
                meanprops={'markerfacecolor': 'w', 'markeredgecolor': 'k', 'marker': 'o'})
    sns.boxplot(auprs, x='Model', y='AUPR', hue=auprs[['Impute', 'Scale']].apply(tuple, axis=1), ax=axs[1],
                showmeans=True,
                meanprops={'markerfacecolor': 'w', 'markeredgecolor': 'k', 'marker': 'o'})
    axs[0].set_title('AUROC')
    axs[1].set_title('AUPR')
    axs[0].set_xlabel('')
    axs[1].set_xlabel('')
    axs[0].legend(bbox_to_anchor=(-.12, 1))
    axs[1].get_legend().remove()
    plt.tight_layout()
    plt.savefig(f'../plots/model_bp/{figname}.png')


def weigted_samples_boxplots(df, dirname, figname):
    weighted_model_names = ['RF', 'XGB', 'CB']
    weighted_cols = df['param_filter'].unique()
    weights_names = {weighted_cols[0]: 'T', weighted_cols[1]: 'AGT', weighted_cols[2]: 'EGT', weighted_cols[3]: 'EAGT'}

    df['Model'] = df['Unnamed: 0'].apply(lambda x: weighted_model_names[x % 3])
    df['WeightedBy'] = df['param_filter'].apply(lambda x: weights_names[x])
    cv_boxplots(df, ['Model', 'WeightedBy'], (10, 5), 'WeightedBy', dirname, figname)


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
            plt.fill_between(percentiles.index, percentiles.iloc[:, i], percentiles.iloc[:, i + 1], label=labels[i])

        plt.xlabel('Age (years)')
        plt.ylabel(feature)
        plt.legend()
        plt.savefig(f'{dir_name}/{feature}.png')
        plt.show()

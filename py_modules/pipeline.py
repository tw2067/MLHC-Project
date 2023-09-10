import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, GroupKFold
from sklearn.linear_model import LogisticRegression
import preprocessing_classes as pclass


def run_cv(X_train, y_train, model_cols, groups, n_splits, params, scoring, regression=False,
           split_col=None, non_white=False):
    split_ = y_train if split_col is None else split_col
    if regression:
        cv = GroupKFold(n_splits=n_splits).split(X_train, split_, groups)
    else:
        cv = StratifiedGroupKFold(n_splits=n_splits).split(X_train, split_, groups)

    if non_white:
        cv_lst = []
        for train_idx, test_idx in cv:
            non_white_test = [idx for idx in test_idx if np.array(X_train['eth_white'])[idx] == 0]
            cv_lst.append((train_idx, non_white_test))
        cv = ((new_train_idx, new_test_idx) for new_train_idx, new_test_idx in cv_lst)
    pipe = Pipeline(steps=[('impute', pclass.SimpleImputer()),
                           ('scale', pclass.StandardScaler()),
                           ('filter', pclass.ColFilter(model_cols)),
                           ('model', LogisticRegression(random_state=0))])
    model = GridSearchCV(pipe, params, cv=cv, scoring=scoring, refit=False, verbose=5, error_score='raise').fit(X_train, y_train)
    return model


def run_pipeline(X_train, y_train, model_cols, imputation, scaling, model):
    pipe = Pipeline(steps=[('impute', imputation),
                           ('scale', scaling),
                           ('filter', pclass.ColFilter(model_cols)),
                           ('model', model)])
    return pipe.fit(X_train, y_train)

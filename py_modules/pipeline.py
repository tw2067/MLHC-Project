from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, GroupKFold
from sklearn.linear_model import LogisticRegression
import preprocessing_classes as pclass


def run_cv(X_train, y_train, model_cols, groups, n_splits, params, scoring, regression=False):
    if regression:
        cv = GroupKFold(n_splits=n_splits).split(X_train, y_train, groups)
    else:
        cv = StratifiedGroupKFold(n_splits=n_splits).split(X_train, y_train, groups)

    pipe = Pipeline(steps=[('impute', pclass.SimpleImputer()),
                           ('scale', pclass.StandardScaler()),
                           ('filter', pclass.ColFilter(model_cols)),
                           ('model', LogisticRegression(random_state=0))])
    model = GridSearchCV(pipe, params, cv=cv, scoring=scoring, refit=False, verbose=5).fit(X_train, y_train)
    return model


def run_pipeline(X_train, y_train):
    pass

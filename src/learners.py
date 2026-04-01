import numpy as np
from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor
from econml.grf import CausalForest


def fit_s_learner(X_train_final, T_train, y_train, learner):
    s_learner = BaseSRegressor(learner=learner)
    s_learner.fit(X=X_train_final, treatment=T_train, y=y_train)
    return s_learner


def predict_s_learner(s_learner, X_test_final):
    return s_learner.predict(X=X_test_final)


def fit_t_learner(X_train_final, T_train, y_train, learner):
    t_learner = BaseTRegressor(learner=learner)
    t_learner.fit(X=X_train_final, treatment=T_train, y=y_train)
    return t_learner


def predict_t_learner(t_learner, X_test_final, T_test):
    return t_learner.predict(X=X_test_final, treatment=T_test)


def fit_x_learner(X_train_final, T_train, y_train, learner):
    x_learner = BaseXRegressor(learner=learner)
    x_learner.fit(X=X_train_final, treatment=T_train, y=y_train)
    return x_learner


def predict_x_learner(x_learner, X_test_final, T_test):
    return x_learner.predict(X=X_test_final, treatment=T_test)


def fit_causal_forest(X_train_final, T_train, y_train, sample_frac=None):
    if sample_frac is not None:
        sample_idx = np.random.choice(
            len(X_train_final), size=int(sample_frac * len(X_train_final)), replace=False
        )
        X_tr = X_train_final[sample_idx]
        T_tr = T_train.values[sample_idx]
        y_tr = y_train.values[sample_idx]
    else:
        X_tr = X_train_final
        T_tr = T_train.values
        y_tr = y_train.values

    cf = CausalForest(n_estimators=100, random_state=42, n_jobs=-1)
    cf.fit(X_tr, T_tr, y_tr)
    return cf


def predict_causal_forest(cf, X_test_final, sample_frac=None):
    if sample_frac is not None:
        test_idx = np.random.choice(
            len(X_test_final), size=int(sample_frac * len(X_test_final)), replace=False
        )
        X_te = X_test_final[test_idx]
    else:
        test_idx = None
        X_te = X_test_final

    cate_cf, lb, ub = cf.predict(X_te, interval=True, alpha=0.05)
    return cate_cf, lb, ub, test_idx

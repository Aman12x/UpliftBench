import numpy as np
import pandas as pd
from causalml.metrics import qini_score


def build_eval_df(y_test, T_test, cate_s, cate_t, cate_x):
    return pd.DataFrame({
        'y': y_test.values,
        'w': T_test.values,
        'S-Learner': cate_s.flatten(),
        'T-Learner': cate_t.flatten(),
        'X-Learner': cate_x.flatten(),
        'Random': np.random.uniform(size=len(y_test)),
    })


def compute_qini(df_eval):
    return qini_score(df_eval, outcome_col='y', treatment_col='w')


def policy_simulation(cate, y, treatment, n_bins=100):
    df_policy = pd.DataFrame({
        'cate': cate.flatten(),
        'y': y.values,
        'w': treatment.values,
    })
    df_policy = df_policy.sort_values('cate', ascending=False).reset_index(drop=True)

    total_users = len(df_policy)
    incremental_conversions = []
    pct_treated = []

    for pct in np.linspace(0.01, 1.0, n_bins):
        n = int(pct * total_users)
        top_k = df_policy.iloc[:n]

        treated = top_k[top_k['w'] == 1]
        control = top_k[top_k['w'] == 0]

        if len(treated) == 0 or len(control) == 0:
            continue

        incremental = (treated['y'].mean() - control['y'].mean()) * n
        incremental_conversions.append(incremental)
        pct_treated.append(pct * 100)

    return pct_treated, incremental_conversions


def policy_thresholds(cate, name, y_test, T_test, thresholds=None):
    if thresholds is None:
        thresholds = [20, 30, 50]
    pct, inc = policy_simulation(cate, y_test, T_test)
    total = inc[-1]
    for threshold in thresholds:
        idx = min(range(len(pct)), key=lambda i: abs(pct[i] - threshold))
        lift_pct = (inc[idx] / total) * 100
        print(f"{name} — top {threshold}% users captures {lift_pct:.1f}% of total incremental conversions")
    print()


def causal_forest_segments(cate_cf, lb, ub):
    significant_positive = lb.flatten() > 0
    significant_negative = ub.flatten() < 0
    uncertain = ~significant_positive & ~significant_negative
    total = len(cate_cf)
    print(f"Confident persuadables: {significant_positive.sum()} ({100 * significant_positive.sum() / total:.1f}%)")
    print(f"Confident sleeping dogs: {significant_negative.sum()} ({100 * significant_negative.sum() / total:.1f}%)")
    print(f"Uncertain: {uncertain.sum()} ({100 * uncertain.sum() / total:.1f}%)")


def causal_forest_qini(df_eval, test_idx, cate_cf):
    df_eval = df_eval.copy()
    df_eval['Causal Forest'] = np.nan
    df_eval.loc[test_idx, 'Causal Forest'] = cate_cf.flatten()
    df_eval_cf = df_eval.dropna(subset=['Causal Forest'])
    return qini_score(df_eval_cf, outcome_col='y', treatment_col='w')

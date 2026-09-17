"""
Run the four CATE estimators on a seeded sample and write every figure to results/.

    python run_benchmark.py --sample-frac 0.1 --seed 42 --outcome visit
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

from src.evaluation import build_eval_df, compute_qini, policy_thresholds
from src.learners import (
    fit_causal_forest, fit_s_learner, fit_t_learner, fit_x_learner, make_base_learner,
    predict_causal_forest, predict_s_learner, predict_t_learner, predict_x_learner,
)
from src.preprocessing import build_features, download_data, fit_propensity, load_data, split_data

RESULTS = Path(__file__).parent / "results"
# The configuration the original notebook gave the S-learner only. Kept as a labelled
# extra so the like-for-like comparison and the original one can be read side by side.
S_LEARNER_ORIGINAL = {"min_child_samples": 5, "reg_lambda": 0, "reg_alpha": 0}


def run(sample_frac, seed, outcome, cf_frac, data_path=None, results_dir=RESULTS):
    """data_path and results_dir let the same code run against a Databricks volume."""
    t0 = time.time()
    df = load_data(data_path or download_data(), sample_frac=sample_frac, seed=seed)
    X_train, X_test, T_train, T_test, y_train, y_test = split_data(df, outcome=outcome)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        lr = fit_propensity(X_train, T_train)
        X_tr, X_te = build_features(X_train, X_test, lr)
        p_tr, p_te = X_tr[:, -1], X_te[:, -1]

        cates = {}
        s = fit_s_learner(X_tr, T_train, y_train, make_base_learner())
        cates["S-Learner"] = predict_s_learner(s, X_te)
        t = fit_t_learner(X_tr, T_train, y_train, make_base_learner())
        cates["T-Learner"] = predict_t_learner(t, X_te, T_test)
        x = fit_x_learner(X_tr, T_train, y_train, make_base_learner(), propensity=p_tr)
        cates["X-Learner"] = predict_x_learner(x, X_te, T_test, propensity=p_te)
        s_orig = fit_s_learner(X_tr, T_train, y_train, make_base_learner(**S_LEARNER_ORIGINAL))
        cate_s_orig = predict_s_learner(s_orig, X_te)
    n_convergence = sum(issubclass(w.category, ConvergenceWarning) for w in caught)

    df_eval = build_eval_df(y_test, T_test, cates["S-Learner"], cates["T-Learner"], cates["X-Learner"], seed=seed)
    df_eval["S-Learner (original config)"] = cate_s_orig.flatten()
    qini = compute_qini(df_eval).to_dict()

    capture = {name: policy_thresholds(c, name, y_test, T_test, outcome=outcome) for name, c in cates.items()}
    capture["S-Learner (original config)"] = policy_thresholds(
        cate_s_orig, "S-Learner (original config)", y_test, T_test, outcome=outcome)

    cf = fit_causal_forest(X_tr, T_train, y_train, sample_frac=cf_frac, seed=seed)
    cate_cf, lb, ub, _ = predict_causal_forest(cf, X_te, sample_frac=None, seed=seed)
    df_eval["Causal Forest"] = cate_cf.flatten()
    qini["Causal Forest"] = float(compute_qini(df_eval[["y", "w", "Causal Forest"]])["Causal Forest"])
    n = len(cate_cf)

    out = {
        "outcome": outcome, "sample_frac": sample_frac, "seed": seed,
        "rows_sampled": int(len(df)), "rows_train": int(len(X_train)), "rows_test": int(len(X_test)),
        "outcome_rate": float(df[outcome].mean()), "treatment_rate": float(df["treatment"].mean()),
        "propensity_auc": float(roc_auc_score(T_test, lr.predict_proba(X_test)[:, 1])),
        "convergence_warnings": int(n_convergence),
        "qini": {k: float(v) for k, v in qini.items()},
        "capture_pct": {k: {str(t): float(v) for t, v in d.items()} for k, d in capture.items()},
        "cate_mean": {k: float(np.mean(v)) for k, v in cates.items()},
        "causal_forest": {
            "train_rows": int(round(len(X_tr) * (cf_frac or 1))), "test_rows": int(n),
            "confident_persuadables_pct": float(100 * (lb.flatten() > 0).sum() / n),
            "confident_sleeping_dogs_pct": float(100 * (ub.flatten() < 0).sum() / n),
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    dest = results_dir / f"benchmark_{outcome}_frac{sample_frac}_seed{seed}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest} in {out['runtime_seconds']}s")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outcome", choices=["visit", "conversion"], default="visit")
    ap.add_argument("--cf-frac", type=float, default=None, help="fraction of the training rows given to the Causal Forest")
    a = ap.parse_args()
    run(a.sample_frac, a.seed, a.outcome, a.cf_frac)

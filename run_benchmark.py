"""
Run the four CATE estimators on a seeded sample and write every figure to results/.

    python run_benchmark.py --sample-frac 0.1 --seed 42 --outcome visit
"""

import argparse
import json
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

from src.evaluation import build_eval_df, compute_qini, policy_thresholds
from src.intervals import bootstrap_qini
from src.learners import (
    fit_causal_forest, fit_s_learner, fit_t_learner, fit_x_learner, make_base_learner,
    predict_causal_forest, predict_s_learner, predict_t_learner, predict_x_learner,
)
from src.preprocessing import build_features, download_data, fit_propensity, load_data, split_data

RESULTS = Path(__file__).parent / "results"
RSS_DIV = 1e6 if sys.platform.startswith("linux") else 1e9   # ru_maxrss is KB on Linux, bytes on macOS


def logger_print(msg):
    print(msg, flush=True)
# The configuration the original notebook gave the S-learner only. Kept as a labelled
# extra so the like-for-like comparison and the original one can be read side by side.
S_LEARNER_ORIGINAL = {"min_child_samples": 5, "reg_lambda": 0, "reg_alpha": 0}


def run(sample_frac, seed, outcome, cf_frac, data_path=None, results_dir=RESULTS, split_seed=42, n_boot=0):
    """data_path and results_dir let the same code run against a Databricks volume."""
    t0 = time.time()
    columns = [f"f{i}" for i in range(12)] + ["treatment", outcome]
    df = load_data(data_path or download_data(), sample_frac=sample_frac, seed=seed, columns=columns)
    rows_sampled, outcome_rate, treatment_rate = len(df), float(df[outcome].mean()), float(df["treatment"].mean())
    X_train, X_test, T_train, T_test, y_train, y_test = split_data(df, outcome=outcome, split_seed=split_seed)
    row_id_test = df.loc[X_test.index, "row_id"].to_numpy()
    del df

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        lr = fit_propensity(X_train, T_train)
        propensity_auc = float(roc_auc_score(T_test, lr.predict_proba(X_test)[:, 1]))
        X_tr, X_te = build_features(X_train, X_test, lr)
        del X_train, X_test                      # the raw frames are not needed again
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

    def checkpoint(stage):
        """Write what exists so far, so an out-of-memory later stage does not lose the earlier ones."""
        partial = {"stage": stage, "outcome": outcome, "sample_frac": sample_frac, "seed": seed, "split_seed": split_seed,
                   "rows_sampled": int(rows_sampled), "rows_train": int(len(X_tr)), "rows_test": int(len(X_te)),
                   "propensity_auc": propensity_auc, "convergence_warnings": int(n_convergence),
                   "qini": {k: float(v) for k, v in qini.items()},
                   "capture_pct": {k: {str(t): float(v) for t, v in d.items()} for k, d in capture.items()},
                   "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / RSS_DIV, 2),
                   "runtime_seconds": round(time.time() - t0, 1)}
        _write(partial, checkpoint=True)
        logger_print(f"checkpoint after {stage}: peak RSS {partial['peak_rss_gb']} GB, {partial['runtime_seconds']}s")

    def _write(payload, checkpoint=False):
        rd = Path(results_dir)
        rd.mkdir(parents=True, exist_ok=True)
        tag = f"frac{sample_frac}" if sample_frac else "full"
        suffix = f"_split{split_seed}" if split_seed != 42 else ""
        cf_tag = f"_cf{cf_frac}" if cf_frac else ""
        name = f"benchmark_{outcome}_{tag}_seed{seed}{suffix}{cf_tag}"
        (rd / (name + (".partial.json" if checkpoint else ".json"))).write_text(json.dumps(payload, indent=2))
        return rd, name

    checkpoint("meta-learners")
    cf = fit_causal_forest(X_tr, T_train, y_train, sample_frac=cf_frac, seed=seed)
    cate_cf, lb, ub, _ = predict_causal_forest(cf, X_te, sample_frac=None, seed=seed)
    df_eval["Causal Forest"] = cate_cf.flatten()
    qini["Causal Forest"] = float(compute_qini(df_eval[["y", "w", "Causal Forest"]])["Causal Forest"])
    n = len(cate_cf)

    out = {
        "outcome": outcome, "sample_frac": sample_frac, "seed": seed,
        "split_seed": split_seed,
        "rows_sampled": int(rows_sampled), "rows_train": int(len(X_tr)), "rows_test": int(len(X_te)),
        "outcome_rate": outcome_rate, "treatment_rate": treatment_rate,
        "propensity_auc": propensity_auc,
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
    scores = {**{k: v.flatten() for k, v in cates.items()},
              "S-Learner (original config)": cate_s_orig.flatten(), "Causal Forest": cate_cf.flatten()}
    if n_boot:
        out["bootstrap"] = bootstrap_qini(y_test.to_numpy(), T_test.to_numpy(), scores, n_boot=n_boot, seed=seed)
        out["runtime_seconds"] = round(time.time() - t0, 1)
    # Complete per-row output: every test row's outcome, treatment flag and predicted effect from
    # each model, plus the Causal Forest interval. Every curve, plot and interval rebuilds from it.
    rd, name = _write(out)
    pred = pd.DataFrame({"row_id": row_id_test, "y": y_test.to_numpy(), "w": T_test.to_numpy(), **scores,
                         "Causal Forest ci_low": lb.flatten(), "Causal Forest ci_high": ub.flatten()})
    pred_dest = rd / "predictions" / (name.replace("benchmark_", "predictions_") + ".parquet")
    pred_dest.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(pred_dest, index=False)
    out["predictions_file"] = f"predictions/{pred_dest.name}"
    out["predictions_rows"] = int(len(pred))
    out["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / RSS_DIV, 2)
    rd, name = _write(out)
    dest = rd / (name + ".json")
    partial = rd / (name + ".partial.json")
    if partial.exists():
        partial.unlink()
    print(f"wrote {dest} in {out['runtime_seconds']}s")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-frac", type=float, default=0.1, help="0 runs the full dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outcome", choices=["visit", "conversion"], default="visit")
    ap.add_argument("--cf-frac", type=float, default=None, help="fraction of the training rows given to the Causal Forest")
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=0, help="bootstrap replicates for Qini intervals")
    a = ap.parse_args()
    run(a.sample_frac or None, a.seed, a.outcome, a.cf_frac, split_seed=a.split_seed, n_boot=a.n_boot)

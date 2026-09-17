"""
Two sources of Qini noise that have nothing to do with model quality, measured on a seeded sample:

  1. Tie order. Tree models give many rows the same score, and causalml sorts with an unstable
     sort, so its Qini depends on an arbitrary ordering of tied rows.
  2. LightGBM thread count.

    python analysis/sensitivity.py      # writes results/sensitivity_visit_frac0.1_seed42.json
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.intervals import _qini_from_sorted, fast_qini  # noqa: E402
from src.learners import (  # noqa: E402
    fit_s_learner, fit_t_learner, make_base_learner, predict_s_learner, predict_t_learner,
)
from src.preprocessing import build_features, download_data, fit_propensity, load_data, split_data  # noqa: E402

SAMPLE_FRAC, SEED, OUTCOME, TIE_ORDERS = 0.1, 42, "visit", 30

df = load_data(download_data(), sample_frac=SAMPLE_FRAC, seed=SEED)
X_train, X_test, T_train, T_test, y_train, y_test = split_data(df, outcome=OUTCOME)
lr = fit_propensity(X_train, T_train)
A, B = build_features(X_train, X_test, lr)
y, w = y_test.to_numpy(float), T_test.to_numpy(float)

out = {"sample_frac": SAMPLE_FRAC, "seed": SEED, "outcome": OUTCOME, "test_rows": int(len(y)),
       "tie_order": {}, "threads": {}}

rng = np.random.default_rng(0)
cates = {
    "S-Learner": predict_s_learner(fit_s_learner(A, T_train, y_train, make_base_learner()), B).flatten(),
    "T-Learner": predict_t_learner(fit_t_learner(A, T_train, y_train, make_base_learner()), B, T_test).flatten(),
}
for name, cate in cates.items():
    _, counts = np.unique(cate, return_counts=True)
    qs = []
    for _ in range(TIE_ORDERS):
        perm = rng.permutation(len(cate))
        order = perm[np.argsort(-cate[perm], kind="stable")]
        qs.append(float(_qini_from_sorted(y[order], w[order])))
    out["tie_order"][name] = {
        "distinct_scores": int(len(counts)),
        "pct_rows_sharing_a_score": float(100 * counts[counts > 1].sum() / len(cate)),
        "largest_tie_group_rows": int(counts.max()),
        "random_tie_orders": TIE_ORDERS,
        "qini_min": min(qs), "qini_max": max(qs), "qini_mean": float(np.mean(qs)),
        "qini_stable_sort": fast_qini(y, w, cate),
    }

for label, kw in (("default", {}), ("n_jobs=4", {"n_jobs": 4}), ("n_jobs=1", {"n_jobs": 1}),
                  ("n_jobs=4, deterministic", {"n_jobs": 4, "deterministic": True, "force_row_wise": True})):
    c = predict_s_learner(fit_s_learner(A, T_train, y_train, make_base_learner(**kw)), B).flatten()
    out["threads"][label] = fast_qini(y, w, c)
vals = list(out["threads"].values())
out["threads_spread"] = max(vals) - min(vals)

dest = Path(__file__).parent.parent / "results" / f"sensitivity_{OUTCOME}_frac{SAMPLE_FRAC}_seed{SEED}.json"
dest.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))

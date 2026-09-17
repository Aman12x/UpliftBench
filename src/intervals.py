"""Bootstrap intervals for the Qini score, and for the Qini difference between two models."""

from itertools import combinations

import numpy as np


def _qini_from_sorted(y, w):
    """causalml.metrics.qini_score (normalize=True) for rows already sorted by descending model score."""
    n = len(y)
    tr = np.cumsum(w, dtype=np.float64)
    ct = np.arange(1, n + 1, dtype=np.float64) - tr
    y_tr = np.cumsum(y * w, dtype=np.float64)
    y_ct = np.cumsum(y * (1 - w), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        curve = y_tr - y_ct * tr / ct
    curve = np.concatenate([[0.0], curve])
    bad = ~np.isfinite(curve)
    if bad.any():                                  # causalml interpolates these linearly
        idx = np.arange(n + 1)
        curve[bad] = np.interp(idx[bad], idx[~bad], curve[~bad])
    curve = curve / abs(curve[-1])
    random_area = np.linspace(curve[0], curve[-1], n + 1).sum()
    return (curve.sum() - random_area) / (n + 1)


def qini_curve(y, w, score):
    """
    The complete normalized Qini curve for one model: one point per test row plus the origin.
    Returns (population_pct, curve), both of length n + 1. Nothing is downsampled.
    """
    y, w = np.asarray(y, dtype=np.float64), np.asarray(w, dtype=np.float64)
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
    ys, ws = y[order], w[order]
    n = len(ys)
    tr = np.cumsum(ws)
    ct = np.arange(1, n + 1, dtype=np.float64) - tr
    with np.errstate(divide="ignore", invalid="ignore"):
        curve = np.cumsum(ys * ws) - np.cumsum(ys * (1 - ws)) * tr / ct
    curve = np.concatenate([[0.0], curve])
    bad = ~np.isfinite(curve)
    if bad.any():
        idx = np.arange(n + 1)
        curve[bad] = np.interp(idx[bad], idx[~bad], curve[~bad])
    return 100 * np.arange(n + 1) / n, curve / abs(curve[-1])


def fast_qini(y, w, score):
    y, w = np.asarray(y, dtype=np.float64), np.asarray(w, dtype=np.float64)
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
    return float(_qini_from_sorted(y[order], w[order]))


def bootstrap_qini(y, w, scores, n_boot=200, seed=42, alpha=0.05):
    """
    Paired percentile bootstrap over test rows. Each replicate draws one set of rows and scores
    every model on it, so the interval on a difference between two models is valid.
    `scores` maps model name -> predicted CATE for every test row.
    """
    y, w = np.asarray(y, dtype=np.float64), np.asarray(w, dtype=np.float64)
    n = len(y)
    names = list(scores)
    sorted_y, sorted_w, rank = {}, {}, {}
    for name in names:
        order = np.argsort(-np.asarray(scores[name], dtype=np.float64), kind="stable")
        sorted_y[name], sorted_w[name] = y[order], w[order]
        r = np.empty(n, dtype=np.int64)
        r[order] = np.arange(n)
        rank[name] = r                              # position of each original row in this model's ordering

    rng = np.random.default_rng(seed)
    draws = {name: np.empty(n_boot) for name in names}
    for b in range(n_boot):
        rows = rng.integers(0, n, size=n)
        for name in names:
            pos = np.sort(rank[name][rows])         # the resampled rows, in this model's order
            draws[name][b] = _qini_from_sorted(sorted_y[name][pos], sorted_w[name][pos])

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    models = {}
    for name in names:
        models[name] = {
            "qini": float(_qini_from_sorted(sorted_y[name], sorted_w[name])),
            "ci_low": float(np.percentile(draws[name], lo)), "ci_high": float(np.percentile(draws[name], hi)),
            "boot_se": float(draws[name].std(ddof=1)),
        }
    stacked = np.vstack([draws[n_] for n_ in names])
    best = np.argmax(stacked, axis=0)
    for i, name in enumerate(names):
        models[name]["share_of_replicates_ranked_first"] = float((best == i).mean())
    differences = {}
    for a, b_ in combinations(names, 2):
        d = draws[a] - draws[b_]
        differences[f"{a} - {b_}"] = {
            "diff": models[a]["qini"] - models[b_]["qini"],
            "ci_low": float(np.percentile(d, lo)), "ci_high": float(np.percentile(d, hi)),
            "separable": bool(np.percentile(d, lo) > 0 or np.percentile(d, hi) < 0),
        }
    return {"n_boot": n_boot, "alpha": alpha, "test_rows": int(n), "models": models, "differences": differences}

"""The fast Qini must equal causalml's, and the paired bootstrap must behave sensibly."""

import numpy as np
import pandas as pd
from causalml.metrics import qini_score

from src.intervals import bootstrap_qini, fast_qini


def _data(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.binomial(1, 0.85, n)
    x = rng.normal(size=n)
    y = rng.binomial(1, np.clip(0.05 + 0.04 * w * (x > 0), 0, 1))
    good = x + rng.normal(scale=0.5, size=n)      # informative ranking
    noise = rng.normal(size=n)                    # uninformative ranking
    return y, w, {"good": good, "noise": noise}


def test_fast_qini_matches_causalml():
    y, w, scores = _data()
    ref = qini_score(pd.DataFrame({"y": y, "w": w, **scores}), outcome_col="y", treatment_col="w")
    for name, s in scores.items():
        assert abs(fast_qini(y, w, s) - ref[name]) < 1e-10


def test_fast_qini_is_deterministic_under_ties_and_inside_the_tie_order_range():
    # causalml sorts with an unstable sort, so with tied scores its Qini depends on an arbitrary
    # tie order. fast_qini uses a stable sort: one fixed answer, inside the range tie orders allow.
    y, w, scores = _data()
    tied = np.round(scores["good"], 0)
    assert fast_qini(y, w, tied) == fast_qini(y, w, tied)
    rng = np.random.default_rng(0)
    qs = []
    for _ in range(200):
        perm = rng.permutation(len(y))
        qs.append(fast_qini(y[perm], w[perm], tied[perm]))
    assert max(qs) - min(qs) > 0            # tie order really does move the score
    ref = qini_score(pd.DataFrame({"y": y, "w": w, "tied": tied}), outcome_col="y", treatment_col="w")["tied"]
    spread = max(qs) - min(qs)
    assert min(qs) - spread <= ref <= max(qs) + spread


def test_bootstrap_is_seeded_and_brackets_the_point_estimate():
    y, w, scores = _data()
    a = bootstrap_qini(y, w, scores, n_boot=60, seed=3)
    b = bootstrap_qini(y, w, scores, n_boot=60, seed=3)
    assert a == b
    for name in scores:
        m = a["models"][name]
        assert m["ci_low"] <= m["qini"] <= m["ci_high"]


def test_paired_difference_separates_a_real_gap_from_no_gap():
    y, w, scores = _data(n=20000)
    scores["good_copy"] = scores["good"] + 1e-9 * np.arange(len(y))   # same ranking
    out = bootstrap_qini(y, w, scores, n_boot=100, seed=1)
    real = out["differences"]["good - noise"]
    none = out["differences"]["good - good_copy"]
    assert real["ci_low"] > 0                      # the informative model really is better
    assert none["ci_low"] <= 0 <= none["ci_high"]  # identical rankings are not separable


def test_qini_curve_is_complete_and_its_area_is_the_qini_score():
    from src.intervals import qini_curve
    y, w, scores = _data()
    pct, curve = qini_curve(y, w, scores["good"])
    n = len(y)
    assert len(pct) == len(curve) == n + 1          # one point per row plus the origin, nothing dropped
    assert pct[0] == 0 and pct[-1] == 100 and curve[0] == 0 and abs(curve[-1] - 1) < 1e-12
    area = (curve.sum() - np.linspace(0, curve[-1], n + 1).sum()) / (n + 1)
    assert abs(area - fast_qini(y, w, scores["good"])) < 1e-10

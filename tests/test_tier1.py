"""Tier 1 correctness: reproducible sampling, a clean X-learner fit, honest outcome labels."""

import warnings

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from sklearn.exceptions import ConvergenceWarning

from src.evaluation import build_eval_df, outcome_label, policy_thresholds
from src.learners import (
    fit_causal_forest, fit_x_learner, make_base_learner,
    predict_causal_forest, predict_x_learner,
)
from src.preprocessing import sample_rows, split_data


def _toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12))
    t = rng.binomial(1, 0.85, size=n)
    visit = rng.binomial(1, np.clip(0.05 + 0.03 * t * (X[:, 0] > 0), 0, 1))
    conversion = visit * rng.binomial(1, 0.2, size=n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)])
    df["treatment"], df["visit"], df["conversion"] = t, visit, conversion
    df["row_id"] = np.arange(n)
    return df


def test_sample_rows_is_deterministic_and_seed_dependent():
    df = _toy()
    a, b = sample_rows(df, 0.1, seed=42), sample_rows(df, 0.1, seed=42)
    c = sample_rows(df, 0.1, seed=43)
    assert a["row_id"].tolist() == b["row_id"].tolist()
    assert a["row_id"].tolist() != c["row_id"].tolist()
    assert 0.07 * len(df) < len(a) < 0.13 * len(df)


def test_split_data_selects_the_requested_outcome():
    df = _toy()
    *_, y_visit, _ = split_data(df, outcome="visit")
    *_, y_conv, _ = split_data(df, outcome="conversion")
    assert y_visit.name == "visit" and y_conv.name == "conversion"
    with pytest.raises(ValueError):
        split_data(df, outcome="exposure")


def test_causal_forest_subsample_is_reproducible():
    df = _toy()
    X, T, y = df[[f"f{i}" for i in range(12)]].to_numpy(), df["treatment"], df["visit"]
    cf1 = fit_causal_forest(X, T, y, sample_frac=0.5, seed=7)
    cf2 = fit_causal_forest(X, T, y, sample_frac=0.5, seed=7)
    _, _, _, idx1 = predict_causal_forest(cf1, X, sample_frac=0.5, seed=7)
    c2, _, _, idx2 = predict_causal_forest(cf2, X, sample_frac=0.5, seed=7)
    c1, *_ = predict_causal_forest(cf1, X, sample_frac=0.5, seed=7)
    assert np.array_equal(idx1, idx2)
    assert np.allclose(c1, c2)


def test_x_learner_fits_without_convergence_warnings():
    df = _toy()
    X, T, y = df[[f"f{i}" for i in range(12)]].to_numpy(), df["treatment"], df["visit"]
    p = np.full(len(df), 0.85)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = fit_x_learner(X, T, y, make_base_learner(), propensity=p)
        cate = predict_x_learner(model, X, T, propensity=p)
    assert cate.shape[0] == len(df)


def test_all_meta_learners_share_one_base_learner_config():
    a, b = make_base_learner(), make_base_learner()
    assert isinstance(a, LGBMRegressor)
    assert a.get_params() == b.get_params() and a is not b


def test_labels_name_the_outcome_that_was_modelled(capsys):
    assert outcome_label("visit") == "incremental visits"
    assert outcome_label("conversion") == "incremental conversions"
    df = _toy()
    cate = np.random.default_rng(0).normal(size=len(df))
    policy_thresholds(cate, "S-Learner", df["visit"], df["treatment"], outcome="visit")
    out = capsys.readouterr().out
    assert "incremental visits" in out and "conversions" not in out


def test_random_baseline_is_seeded():
    df = _toy()
    z = np.zeros(len(df))
    a = build_eval_df(df["visit"], df["treatment"], z, z, z, seed=1)["Random"]
    b = build_eval_df(df["visit"], df["treatment"], z, z, z, seed=1)["Random"]
    assert a.equals(b)

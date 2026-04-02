## Setup
Import libraries and `src` modules.


```python
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, LGBMRegressor

from src.preprocessing import load_data, split_data, fit_propensity, build_features
from src.learners import (
    fit_s_learner, predict_s_learner,
    fit_t_learner, predict_t_learner,
    fit_x_learner, predict_x_learner,
    fit_causal_forest, predict_causal_forest,
)
from src.evaluation import (
    build_eval_df, compute_qini,
    policy_simulation, policy_thresholds,
    causal_forest_segments, causal_forest_qini,
)
from src.visualization import (
    plot_cate_distribution, plot_cate_full_and_clipped,
    plot_learner_comparison, plot_qini_curves,
    plot_policy_simulation, plot_shap,
    plot_causal_forest_uncertainty,
)
```

## Data Loading
Load the Criteo uplift v2.1 dataset.


```python
df = load_data('/Users/amansingh/Desktop/CI project/data/criteo-uplift-v2.1.csv')
```

## Exploratory Data Analysis
Inspect dataset shape, sample rows, and summary statistics.


```python
df.shape
```


```python
df.head(5)
```


```python
df.describe()
```

### Treatment & Outcome Rates
Check class balance across treatment, conversion, and visit.


```python
# treatment / outcome rates
print(df["treatment"].value_counts(normalize=True))
print(df["conversion"].value_counts(normalize=True))
print(df["visit"].value_counts(normalize=True))
```

### Outcome Rates by Group
Compare conversion and visit rates between treatment and control.


```python
# conversion rate by treatment group
print(df.groupby("treatment")["conversion"].mean())
print(df.groupby("treatment")["visit"].mean())
```


```python
df.groupby("treatment")["conversion"].mean()
```


```python
df.groupby("treatment")[["conversion", "visit"]].mean()
```

### Feature Statistics
Descriptive statistics and cardinality for the 12 covariates.


```python
df[[f"f{i}" for i in range(12)]].describe()
```


```python
df[[f"f{i}" for i in range(12)]].nunique()
```


```python
df.groupby("treatment")[[f"f{i}" for i in range(12)]].mean()
```

## Preprocessing
Train-test split (80/20, `random_state=42`).


```python
X_train, X_test, T_train, T_test, y_train, y_test = split_data(df)
```

### Propensity Model
Fit logistic regression to predict treatment assignment.


```python
lr = fit_propensity(X_train, T_train)
auc = roc_auc_score(T_test, lr.predict_proba(X_test)[:, 1])
print(f"Propensity model AUC: {auc:.4f}")
```

Propensity model AUC of 0.51 confirms near-random treatment assignment. Covariate balance across treatment and control groups is strong (mean differences < 0.2 across all features), supporting the validity of CATE estimation without heavy propensity weighting.

### Feature Augmentation
Standardize covariates and append propensity score as a 13th feature.


```python
X_train_final, X_test_final = build_features(X_train, X_test, lr)
```

## Meta-Learners — CATE Estimation
### S-Learner (Classifier — Experiment)
Initial run with `LGBMClassifier`; superseded by the regressor below.


```python
s_learner = fit_s_learner(
    X_train_final, T_train, y_train,
    LGBMClassifier(n_estimators=200, random_state=42),
)
cate_s = predict_s_learner(s_learner, X_test_final)
print(f"CATE mean: {cate_s.mean():.4f}")
print(f"CATE std: {cate_s.std():.4f}")
print(f"CATE min: {cate_s.min():.4f}")
print(f"CATE max: {cate_s.max():.4f}")
```

### S-Learner (Regressor)
Final S-Learner using `LGBMRegressor`.


```python
s_learner = fit_s_learner(
    X_train_final, T_train, y_train,
    LGBMRegressor(
        n_estimators=200,
        min_child_samples=5,
        reg_lambda=0,
        reg_alpha=0,
        random_state=42,
    ),
)
cate_s = predict_s_learner(s_learner, X_test_final)
print(f"CATE mean: {cate_s.mean():.4f}")
print(f"CATE std: {cate_s.std():.4f}")
print(f"CATE min: {cate_s.min():.4f}")
print(f"CATE max: {cate_s.max():.4f}")
```

### S-Learner CATE Distribution


```python
plot_cate_distribution(cate_s, 'S-Learner')
```

### T-Learner
Fit separate outcome models for treated and control groups.


```python
t_learner = fit_t_learner(
    X_train_final, T_train, y_train,
    LGBMRegressor(n_estimators=200, random_state=42),
)
cate_t = predict_t_learner(t_learner, X_test_final, T_test)
print(f"CATE mean: {cate_t.mean():.4f}")
print(f"CATE std: {cate_t.std():.4f}")
print(f"CATE min: {cate_t.min():.4f}")
print(f"CATE max: {cate_t.max():.4f}")
```

### T-Learner CATE Distribution


```python
plot_cate_distribution(cate_t, 'T-Learner')
```

### T-Learner CATE — Full Range vs Clipped
Clip to 1st–99th percentile to suppress outlier influence on the histogram.


```python
plot_cate_full_and_clipped(cate_t, 'T-Learner')
```

### X-Learner
Fit X-Learner combining imputed treatment effects from both groups.


```python
x_learner = fit_x_learner(
    X_train_final, T_train, y_train,
    LGBMRegressor(n_estimators=200, random_state=42),
)
cate_x = predict_x_learner(x_learner, X_test_final, T_test)
print(f"CATE mean: {cate_x.mean():.4f}")
print(f"CATE std: {cate_x.std():.4f}")
print(f"CATE min: {cate_x.min():.4f}")
print(f"CATE max: {cate_x.max():.4f}")
```

### Learner Comparison
Side-by-side clipped CATE distributions for all three meta-learners.


```python
plot_learner_comparison(cate_s, cate_t, cate_x)
```

## Evaluation
Build evaluation frame, compute Qini scores, and plot cumulative gain curves.


```python
df_eval = build_eval_df(y_test, T_test, cate_s, cate_t, cate_x)
scores = compute_qini(df_eval)
print(scores)
plot_qini_curves(df_eval)
```

### Policy Simulation
Model incremental conversions captured as a function of targeting budget.


```python
plot_policy_simulation(
    [cate_s, cate_t, cate_x],
    ['S-Learner', 'T-Learner', 'X-Learner'],
    ['steelblue', 'darkorange', 'green'],
    y_test, T_test,
)
```

### Policy Thresholds
Lift captured at the 20%, 30%, and 50% targeting thresholds.


```python
for cate, name in zip([cate_s, cate_t, cate_x], ['S-Learner', 'T-Learner', 'X-Learner']):
    policy_thresholds(cate, name, y_test, T_test)
```

### T-Learner Internals
Inspect model attributes for SHAP extraction.


```python
print(dir(t_learner))
```


```python
print(t_learner.models_c.keys())
print(t_learner.models_t.keys())
```

## SHAP Feature Importance
Differential SHAP (treat − control) to explain CATE heterogeneity via the T-Learner.


```python
plot_shap(t_learner, X_test_final)
```


```python
# superseded by Cell 31 (sampled) — skipped to avoid full-dataset CausalForest fit
```

## Causal Forest
Fit on a 10% subsample for speed; predict CATEs with 95% confidence intervals.


```python
cf = fit_causal_forest(X_train_final, T_train, y_train, sample_frac=0.1)
cate_cf, lb, ub, test_idx = predict_causal_forest(cf, X_test_final, sample_frac=0.1)

print(f"CATE mean: {cate_cf.mean():.4f}")
print(f"CATE std: {cate_cf.std():.4f}")
print(f"CATE min: {cate_cf.min():.4f}")
print(f"CATE max: {cate_cf.max():.4f}")
print()

causal_forest_segments(cate_cf, lb, ub)
```

### Causal Forest Qini
Evaluate Causal Forest ranking performance.


```python
scores_cf = causal_forest_qini(df_eval, test_idx, cate_cf)
print(scores_cf)
```

### Causal Forest Uncertainty
CI width distribution and CATE estimate vs uncertainty scatter.


```python
plot_causal_forest_uncertainty(cate_cf, lb, ub)
```


```python

```

# Causal Inference & Heterogeneous Treatment Effect Analysis
### Marketing Uplift Modeling at Scale — Criteo Dataset (13.98M Records)

---

## Executive Summary

This project delivers a production-grade causal inference pipeline for **predicting individual-level response to marketing treatments**. Applied to the Criteo Uplift v2.1 dataset (13.98 million customer records, 3 GB), the analysis estimates Conditional Average Treatment Effects (CATE) using three competing meta-learner algorithms and a Causal Forest. The best-performing model achieves a **Qini coefficient of 0.3759**, meaning targeted deployment to the top 20% of ranked customers captures **77.7% of all incremental conversions** — a result that directly translates to improved marketing ROI without increasing spend.

---

## Business Problem

Standard A/B testing tells you whether a campaign works on average. It cannot tell you *who* it works for.

In large-scale customer marketing, sending promotions indiscriminately creates three problems:

1. **Wasted spend** on customers who would have converted anyway (always-converters)
2. **Negative returns** on customers who react adversely to treatment (sleeping dogs)
3. **Missed opportunity** on high-responders who are not being prioritized

Uplift modeling — grounded in causal inference — solves this by estimating the **incremental effect** of treatment at the individual level, enabling precise targeting of customers where the campaign actually moves the needle.

---

## Dataset

| Property | Value |
|---|---|
| Source | Criteo Uplift v2.1 |
| Records | 13,979,592 |
| Features | 12 continuous covariates (f0–f11) |
| Treatment | Binary (85% treated, 15% control) |
| Primary Outcome | `visit` (binary; 4.7% positive rate) |
| Secondary Outcome | `conversion` (binary; 0.29% positive rate) |
| Propensity AUC | 0.5093 — confirms near-random treatment assignment |

The near-random propensity score (AUC ≈ 0.51) validates the quasi-experimental design, lending the CATE estimates high internal validity.

---

## Methodology

### Pipeline Overview

```
Raw Data (CSV, 3GB)
       |
       v
  Data Preprocessing
  - 80/20 train-test split (random_state=42)
  - StandardScaler normalization (12 covariates)
  - Propensity score estimation (Logistic Regression)
  - Feature augmentation: 12 covariates + propensity score = 13 features
       |
       v
  CATE Estimation (4 Methods)
  - S-Learner (LightGBM)
  - T-Learner (LightGBM)
  - X-Learner (LightGBM)
  - Causal Forest (EconML, 10% subsample ~280K records)
       |
       v
  Evaluation & Policy Simulation
  - Qini coefficient scoring
  - Cumulative gain curves
  - Policy simulation (budget vs. incremental conversions)
  - Uncertainty quantification (95% CI via Causal Forest)
  - SHAP feature importance
```

### CATE Estimation Methods

**S-Learner** — Fits a single model with treatment indicator as a feature. Simple, regularizes treatment effect, reduces variance at the cost of some flexibility.

**T-Learner** — Fits separate outcome models for treated and control groups, then computes the difference. Captures group-specific patterns but can overfit in imbalanced settings.

**X-Learner** — Cross-fitting approach that imputes individual treatment effects by applying each group's model to the opposite group. Handles treatment imbalance (85/15 split) more robustly than T-Learner.

**Causal Forest** — EconML's doubly robust Random Forest with confidence intervals. Provides uncertainty quantification at the individual level. Run on a 10% subsample (~280K records) for computational feasibility.

All models use **LightGBM** as the base learner — gradient-boosted decision trees optimized for speed and performance on tabular data at scale.

---

## Results

### Model Performance Summary

| Model | CATE Mean | CATE Std | Qini Score | Top 20% Capture | Top 50% Capture |
|---|---|---|---|---|---|
| **S-Learner** | 0.0070 | 0.0223 | **0.3759** | **77.7%** | 95.8% |
| X-Learner | 0.0074 | 0.0248 | 0.3615 | 76.0% | 92.8% |
| T-Learner | 0.0074 | 0.0267 | 0.3503 | 75.1% | 92.4% |
| Causal Forest | 0.0072 | 0.0377 | 0.2524 | — | — |
| Random Baseline | — | — | ~0.007 | 20.0% | 50.0% |

> **S-Learner is the recommended production model.** It achieves the highest Qini score and top-decile capture rate with the lowest variance, making it the most stable ranking model for campaign targeting.

### Key Business Insights

- **Targeting efficiency:** Deploying to the top 20% of customers ranked by predicted CATE captures 77.7% of all incremental conversions. A random targeting strategy would capture only 20%.
- **Heterogeneous response confirmed:** All four models independently identify meaningful variation in treatment response across customer segments, validating the uplift modeling approach.
- **Causal Forest uncertainty:** Of the ~280K evaluated records, 1.9% are confident persuadables (lower 95% CI > 0) and 0.1% are confident sleeping dogs (upper 95% CI < 0). The remaining 98% have uncertain effect estimates, underscoring the value of probabilistic targeting over binary rule-based approaches.
- **Feature drivers:** SHAP analysis on the T-Learner identifies which of the 12 covariates (f0–f11) most drive treatment effect heterogeneity — informing feature selection for downstream targeting systems.

---

## Business Impact Framing

Assume a hypothetical campaign of 1,000,000 customers with a cost of $1 per contact and an incremental conversion value of $50.

| Strategy | Contacts | Incremental Conversions | Revenue | Cost | Net ROI |
|---|---|---|---|---|---|
| Untargeted (100%) | 1,000,000 | baseline | $X | $1,000,000 | baseline |
| Top 20% (S-Learner) | 200,000 | ~77.7% of baseline | ~$0.78X | $200,000 | Significantly positive |
| Random 20% | 200,000 | ~20% of baseline | ~$0.20X | $200,000 | Negative |

Reducing contact volume by 80% while retaining 77.7% of incremental conversions is a **3.9x improvement in conversion efficiency** versus random sampling, directly reducing customer acquisition cost (CAC).

---

## Project Structure

```
CI project/
├── README.md                    # This document
├── modeling.ipynb               # End-to-end analysis notebook
├── modeling.md                  # Notebook markdown export
├── requirements.txt             # Python dependencies
├── data/
│   └── criteo-uplift-v2.1.csv   # Source dataset (3 GB, 13.98M records)
├── plots/
│   ├── cate_s_learner.png
│   ├── cate_t_learner.png
│   ├── cate_t_learner_full_clipped.png
│   ├── learner_comparison.png
│   ├── qini_curves.png
│   ├── policy_simulation.png
│   ├── shap_cate_bar.png
│   └── shap_cate_beeswarm.png
└── src/
    ├── __init__.py
    ├── preprocessing.py         # Data loading, splitting, propensity estimation, feature engineering
    ├── learners.py              # S/T/X-Learner and Causal Forest implementations
    ├── evaluation.py            # Qini scoring, policy simulation, uncertainty analysis
    └── visualization.py         # CATE distributions, Qini curves, SHAP plots
```

---

## Technical Stack

| Category | Libraries |
|---|---|
| Causal Inference | `causalml` 0.16.0, `econml` 0.16.0, `forestci` 0.6 |
| Machine Learning | `lightgbm` 4.6.0, `scikit-learn`, `xgboost` |
| Data Processing | `pandas`, `numpy`, `scipy`, `statsmodels` |
| Explainability | `shap` 0.48.0 |
| Visualization | `matplotlib`, `seaborn` |
| Notebook | `jupyter`, `ipykernel`, `ipython` |

---

## Reproducibility

```bash
# Install dependencies
pip install -r requirements.txt

# Run full analysis
jupyter notebook modeling.ipynb
```

The dataset (`criteo-uplift-v2.1.csv`) is required in the `data/` directory. Due to its size (3 GB), it is not tracked in version control. The Criteo Uplift dataset is publicly available from the Criteo AI Lab.

All random seeds are fixed (`random_state=42`) throughout preprocessing, model training, and evaluation for full reproducibility.

---

## Evaluation Metrics

**Qini Coefficient** — The primary evaluation metric. Measures the area between the model's cumulative gain curve and the random targeting baseline. Higher values indicate better uplift ranking ability. Range: [0, 1].

**Cumulative Gain Curve** — Plots the fraction of incremental conversions captured as a function of the fraction of population contacted, sorted by descending predicted CATE.

**Policy Simulation** — Translates the gain curve into actionable budget vs. conversion tradeoff curves for campaign planning.

**Confidence Intervals (Causal Forest)** — 95% bootstrap confidence intervals on individual CATE estimates, enabling probabilistic customer segmentation (persuadables, sleeping dogs, uncertain).

---

## Limitations & Next Steps

**Current Limitations:**
- Causal Forest was evaluated on a 10% subsample due to computational constraints; full-population estimates may differ.
- Feature interpretability is limited since covariates are anonymized (f0–f11); business feature naming would improve stakeholder communication.
- The 85/15 treatment imbalance may still affect X-Learner and T-Learner stability in low-density regions of feature space.

**Recommended Next Steps:**
1. **Model deployment:** Package S-Learner scoring pipeline as a REST API or batch scoring job for integration with campaign management platforms.
2. **Online validation:** Run a prospective A/B test deploying S-Learner targeting vs. random targeting to validate Qini estimates in production.
3. **Conversion outcome modeling:** Replicate the analysis using `conversion` as the outcome variable (currently only `visit` was modeled) for revenue-level impact estimation.
4. **Threshold optimization:** Define a production targeting threshold on predicted CATE that balances precision, recall, and cost constraints for specific campaign budgets.
5. **Feature enrichment:** Join behavioral or CRM features to augment the 12 anonymized covariates and potentially improve CATE estimation accuracy.

---

## Author

**Aman Singh**
Causal Inference & Applied Machine Learning

---

*Dataset: Criteo Uplift v2.1 — Criteo AI Lab*

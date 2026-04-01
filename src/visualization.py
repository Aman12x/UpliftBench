import os
import numpy as np
import matplotlib.pyplot as plt
import shap
from causalml.metrics import plot_gain
from src.evaluation import policy_simulation

os.makedirs('plots', exist_ok=True)


def plot_cate_distribution(cate, name):
    plt.figure(figsize=(10, 4))
    plt.hist(cate, bins=100, edgecolor='none')
    plt.axvline(x=0, color='red', linestyle='--', label='zero uplift')
    plt.axvline(x=cate.mean(), color='green', linestyle='--', label=f'mean={cate.mean():.4f}')
    plt.xlabel("Predicted CATE")
    plt.ylabel("Number of users")
    plt.title(f"{name} CATE distribution")
    plt.legend()
    plt.savefig(f"plots/cate_{name.lower().replace('-', '_').replace(' ', '_')}.png", bbox_inches='tight')
    plt.show()


def plot_cate_full_and_clipped(cate, name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].hist(cate, bins=100, edgecolor='none')
    axes[0].axvline(x=0, color='red', linestyle='--', label='zero uplift')
    axes[0].axvline(x=cate.mean(), color='green', linestyle='--', label=f'mean={cate.mean():.4f}')
    axes[0].set_title(f"{name} CATE — full range")
    axes[0].legend()

    clipped = cate[(cate > np.percentile(cate, 1)) & (cate < np.percentile(cate, 99))]
    axes[1].hist(clipped, bins=100, edgecolor='none')
    axes[1].axvline(x=0, color='red', linestyle='--', label='zero uplift')
    axes[1].axvline(x=cate.mean(), color='green', linestyle='--', label=f'mean={cate.mean():.4f}')
    axes[1].set_title(f"{name} CATE — 1st to 99th percentile")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"plots/cate_{name.lower().replace('-', '_').replace(' ', '_')}_full_clipped.png", bbox_inches='tight')
    plt.show()


def plot_learner_comparison(cate_s, cate_t, cate_x):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, cate, name in zip(axes, [cate_s, cate_t, cate_x], ['S-Learner', 'T-Learner', 'X-Learner']):
        clipped = cate[(cate > np.percentile(cate, 1)) & (cate < np.percentile(cate, 99))]
        ax.hist(clipped, bins=100, edgecolor='none')
        ax.axvline(x=0, color='red', linestyle='--', label='zero uplift')
        ax.axvline(x=cate.mean(), color='green', linestyle='--', label=f'mean={cate.mean():.4f}')
        ax.set_title(f"{name} CATE distribution")
        ax.set_xlabel("Predicted CATE")
        ax.set_ylabel("Number of users")
        ax.legend()
    plt.tight_layout()
    plt.savefig("plots/learner_comparison.png", bbox_inches='tight')
    plt.show()


def plot_qini_curves(df_eval):
    plot_gain(df_eval, outcome_col='y', treatment_col='w', figsize=(10, 6))
    plt.title("Qini curves — S / T / X Learner vs Random")
    plt.savefig("plots/qini_curves.png", bbox_inches='tight')
    plt.show()


def plot_policy_simulation(cate_list, names, colors, y_test, T_test):
    fig, ax = plt.subplots(figsize=(10, 6))
    for cate, name, color in zip(cate_list, names, colors):
        pct, inc = policy_simulation(cate, y_test, T_test)
        ax.plot(pct, inc, label=name, color=color)
    ax.set_xlabel("% of users treated")
    ax.set_ylabel("Incremental conversions")
    ax.set_title("Policy simulation — incremental conversions vs budget")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/policy_simulation.png", bbox_inches='tight')
    plt.show()


def plot_shap(t_learner, X_test_final):
    sample_idx = np.random.choice(len(X_test_final), size=2000, replace=False)
    X_sample = X_test_final[sample_idx]
    feature_names = [f"f{i}" for i in range(12)] + ["propensity"]

    explainer_control = shap.TreeExplainer(t_learner.models_c[1])
    explainer_treat = shap.TreeExplainer(t_learner.models_t[1])

    shap_control = explainer_control.shap_values(X_sample)
    shap_treat = explainer_treat.shap_values(X_sample)
    shap_cate = shap_treat - shap_control

    plt.figure()
    shap.summary_plot(shap_cate, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP — feature contribution to CATE (T-Learner)")
    plt.tight_layout()
    plt.savefig("plots/shap_cate_beeswarm.png", bbox_inches='tight')
    plt.show()

    plt.figure()
    shap.summary_plot(shap_cate, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("SHAP — mean absolute CATE contribution")
    plt.tight_layout()
    plt.savefig("plots/shap_cate_bar.png", bbox_inches='tight')
    plt.show()


def plot_causal_forest_uncertainty(cate_cf, lb, ub):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ci_width = (ub - lb).flatten()
    axes[0].hist(ci_width, bins=100, edgecolor='none')
    axes[0].set_xlabel("CI width (upper - lower)")
    axes[0].set_ylabel("Number of users")
    axes[0].set_title("Distribution of CI widths")
    axes[0].axvline(x=ci_width.mean(), color='green', linestyle='--',
                    label=f'mean={ci_width.mean():.4f}')
    axes[0].legend()

    axes[1].scatter(cate_cf.flatten(), ci_width, alpha=0.05, s=1)
    axes[1].axvline(x=0, color='red', linestyle='--', label='zero uplift')
    axes[1].set_xlabel("CATE point estimate")
    axes[1].set_ylabel("CI width")
    axes[1].set_title("CATE estimate vs uncertainty")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("plots/cf_uncertainty.png", bbox_inches='tight')
    plt.show()

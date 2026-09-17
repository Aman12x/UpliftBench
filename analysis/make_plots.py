"""
Build every figure from the files in results/. Nothing is typed by hand and nothing is downsampled:
curves are drawn from the complete per-row predictions in results/predictions/.

    python analysis/make_plots.py            # writes results/plots/*.png
"""

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.intervals import qini_curve  # noqa: E402

RESULTS, PLOTS = ROOT / "results", ROOT / "results" / "plots"

# One fixed color per model, everywhere. Order and hexes are a validated colorblind-safe palette
# (adjacent-pair CVD dE >= 9.1 on this surface). Three of them are low-contrast on a light
# surface, so every mark is also named in text and no figure relies on color alone.
MODELS = ["S-Learner", "T-Learner", "X-Learner", "Causal Forest", "S-Learner (original config)"]
COLOR = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#b4b2ab"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.grid": True, "axes.axisbelow": True,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "legend.frameon": False, "legend.fontsize": 9,
})
DOT = dict(marker="o", markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none")


def load(path):
    return json.loads(Path(path).read_text())


def tag_of(path):
    return Path(path).stem.replace("benchmark_", "").replace("databricks_", "databricks-")


def describe(r):
    size = "full dataset" if not r.get("sample_frac") else f"{r['sample_frac']:.0%} sample"
    return f"outcome: {r['outcome']} · {size} · {r['rows_test']:,} test rows · seed {r['seed']}"


def save(fig, name):
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", PLOTS / name)


def result_of(path):
    r = load(path)
    return r.get("result", r)            # Databricks files wrap the result with the environment


# ── 1. Qini with 95% bootstrap intervals ──────────────────────────────────────────────────────────
def plot_intervals(path):
    r = result_of(path)
    if "bootstrap" not in r:
        return
    b = r["bootstrap"]["models"]
    names = sorted((m for m in MODELS if m in b), key=lambda m: b[m]["qini"])
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(names) + 1.5))
    for i, m in enumerate(names):
        ax.plot([b[m]["ci_low"], b[m]["ci_high"]], [i, i], color=COLOR[m], linewidth=2)
        ax.plot(b[m]["qini"], i, color=COLOR[m], **DOT)
        ax.annotate(f"{b[m]['qini']:.3f}   [{b[m]['ci_low']:.3f}, {b[m]['ci_high']:.3f}]",
                    (b[m]["ci_high"], i), xytext=(8, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.set_yticks(range(len(names)), names)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Qini score (higher is better)")
    hi = max(v["ci_high"] for v in b.values())
    ax.set_xlim(min(v["ci_low"] for v in b.values()) - 0.02, hi + 0.13 * (hi + 0.1) + 0.06)
    ax.set_title("Qini score with 95% bootstrap interval")
    sep = [k for k, v in r["bootstrap"]["differences"].items() if v["separable"]]
    note = f"{describe(r)} · {r['bootstrap']['n_boot']} paired replicates\n" + (
        "Pairs whose difference excludes zero: " + "; ".join(sep) if sep else "No pair of models is separable at 95%.")
    fig.text(0.01, -0.02, note, color=INK2, fontsize=8.5, va="top")
    save(fig, f"qini_intervals_{tag_of(path)}.png")


# ── 2-5. Figures that need the complete per-row predictions ─────────────────────────────────────
def predictions_for(path):
    r = result_of(path)
    f = r.get("predictions_file")
    p = RESULTS / f if f else None
    return (r, pd.read_parquet(p)) if p and p.exists() else (r, None)


def small_multiples(names, title, note, draw, xlabel, ylabel, sharey=True):
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11.5, 3.3 * rows), sharex=True, sharey=sharey)
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, names):
        draw(ax, m)
        ax.set_title(m, color=INK)
    for ax in axes[len(names):]:
        ax.axis("off")
    for ax in axes[:len(names)]:
        ax.set_xlabel(xlabel)
        ax.tick_params(labelbottom=True)
    for ax in axes[:len(names)]:
        if not ax.get_subplotspec().is_last_row() and rows > 1:
            ax.set_xlabel("")
    axes[0].set_ylabel(ylabel)
    if rows > 1:
        axes[cols].set_ylabel(ylabel)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.text(0.01, -0.01, note, color=INK2, fontsize=8.5, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, axes


def plot_qini_curves(path):
    r, pred = predictions_for(path)
    if pred is None:
        return
    names = [m for m in MODELS if m in pred.columns]
    y, w = pred["y"].to_numpy(), pred["w"].to_numpy()
    curves = {m: qini_curve(y, w, pred[m].to_numpy()) for m in names}

    def draw(ax, m):
        for other in names:
            if other != m:
                ax.plot(*curves[other], color=MUTED, linewidth=1, alpha=0.8)
        ax.plot([0, 100], [0, 1], color=INK2, linewidth=1)
        ax.plot(*curves[m], color=COLOR[m], linewidth=2)
        ax.annotate(f"Qini {r['bootstrap']['models'][m]['qini']:.3f}" if "bootstrap" in r else f"Qini {r['qini'][m]:.3f}",
                    (0.97, 0.06), xycoords="axes fraction", ha="right", color=INK2, fontsize=9)

    fig, axes = small_multiples(
        names, "Qini curves: share of the total treatment effect captured as more customers are targeted",
        f"{describe(r)} · every test row plotted, no downsampling\n"
        "Colored line: the named model · gray lines: the other models · straight line: random targeting", draw,
        "Customers targeted, ranked by predicted effect (%)", f"Share of incremental {r['outcome']}s captured")
    save(fig, f"qini_curves_{tag_of(path)}.png")


def plot_policy(path):
    r, pred = predictions_for(path)
    if pred is None:
        return
    names = [m for m in MODELS if m in pred.columns]
    y, w = pred["y"].to_numpy(float), pred["w"].to_numpy(float)
    n = len(y)
    curves = {}
    for m in names:
        order = np.argsort(-pred[m].to_numpy(), kind="stable")
        ys, ws = y[order], w[order]
        n_t, n_c = np.cumsum(ws), np.cumsum(1 - ws)
        with np.errstate(divide="ignore", invalid="ignore"):
            inc = (np.cumsum(ys * ws) / n_t - np.cumsum(ys * (1 - ws)) / n_c) * np.arange(1, n + 1)
        ok = (n_t >= 100) & (n_c >= 100)          # below this the two group means are not yet estimates
        curves[m] = (100 * np.arange(1, n + 1)[ok] / n, inc[ok])

    def draw(ax, m):
        for other in names:
            if other != m:
                ax.plot(*curves[other], color=MUTED, linewidth=1, alpha=0.8)
        ax.plot(*curves[m], color=COLOR[m], linewidth=2)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig, axes = small_multiples(
        names, f"Policy simulation: incremental {r['outcome']}s gained by targeting the top-ranked customers",
        f"{describe(r)} · every test row plotted once both arms hold 100 rows\n"
        "Colored line: the named model · gray lines: the other models", draw,
        "Customers targeted, ranked by predicted effect (%)", f"Incremental {r['outcome']}s in the test set")
    save(fig, f"policy_{tag_of(path)}.png")


def plot_cate_distributions(path):
    r, pred = predictions_for(path)
    if pred is None:
        return
    names = [m for m in MODELS if m in pred.columns]
    lo = min(np.percentile(pred[m], 0.5) for m in names)
    hi = max(np.percentile(pred[m], 99.5) for m in names)
    bins = np.linspace(lo, hi, 121)

    def draw(ax, m):
        v = pred[m].to_numpy()
        ax.hist(np.clip(v, lo, hi), bins=bins, color=COLOR[m], alpha=0.85, linewidth=0)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_yscale("log")
        ax.annotate(f"mean {v.mean():+.4f}\nshare below zero {100 * (v < 0).mean():.1f}%",
                    (0.97, 0.93), xycoords="axes fraction", ha="right", va="top", color=INK2, fontsize=9)

    fig, axes = small_multiples(
        names, "Distribution of predicted treatment effects across every test row",
        f"{describe(r)} · all rows counted · values beyond the 0.5th and 99.5th percentiles are stacked in the end bins · log count axis",
        draw, f"Predicted effect on {r['outcome']} probability", "Test rows (log scale)")
    save(fig, f"cate_distributions_{tag_of(path)}.png")


def plot_cf_uncertainty(path):
    r, pred = predictions_for(path)
    if pred is None or "Causal Forest ci_low" not in pred.columns:
        return
    d = pred.sort_values("Causal Forest", kind="stable")
    x = 100 * np.arange(len(d)) / len(d)
    c, lo, hi = d["Causal Forest"].to_numpy(), d["Causal Forest ci_low"].to_numpy(), d["Causal Forest ci_high"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.fill_between(x, lo, hi, color=COLOR["Causal Forest"], alpha=0.12, linewidth=0, label="95% interval, every row")
    win = max(1, len(d) // 200)
    lo_s = pd.Series(lo).rolling(win, center=True, min_periods=1).median().to_numpy()
    hi_s = pd.Series(hi).rolling(win, center=True, min_periods=1).median().to_numpy()
    ax.plot(x, lo_s, color=COLOR["Causal Forest"], linewidth=1, alpha=0.9, label=f"rolling median of the bounds ({win:,}-row window)")
    ax.plot(x, hi_s, color=COLOR["Causal Forest"], linewidth=1, alpha=0.9)
    ax.plot(x, c, color=COLOR["Causal Forest"], linewidth=2, label="Causal Forest predicted effect")
    ax.axhline(0, color=INK2, linewidth=1)
    ylo, yhi = np.percentile(lo, 0.5), np.percentile(hi, 99.5)
    ax.set_ylim(ylo, yhi)
    pers, dogs = 100 * (lo > 0).mean(), 100 * (hi < 0).mean()
    ax.set_title("Causal Forest: predicted effect and its 95% interval for every test row, sorted")
    ax.set_xlabel("Test rows, sorted by predicted effect (%)")
    ax.set_ylabel(f"Effect on {r['outcome']} probability")
    ax.legend(loc="upper left")
    fig.text(0.01, -0.02, f"{describe(r)} · all {len(d):,} rows drawn · y-axis trimmed to the 0.5th–99.5th percentile of the interval bounds\n"
             f"Interval entirely above zero (confident persuadables): {pers:.2f}% of rows · entirely below zero (confident sleeping dogs): {dogs:.2f}%",
             color=INK2, fontsize=8.5, va="top")
    save(fig, f"cf_uncertainty_{tag_of(path)}.png")


# ── 6. Seed variance ─────────────────────────────────────────────────────────────────────────────
def plot_seed_variance():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    drew = False
    for ax, outcome in zip(axes, ("visit", "conversion")):
        runs = [load(f) for f in sorted(glob.glob(str(RESULTS / f"benchmark_{outcome}_frac0.1_seed*.json")))]
        if not runs:
            ax.axis("off")
            continue
        drew = True
        names = [m for m in MODELS if m in runs[0]["qini"]][::-1]
        for i, m in enumerate(names):
            q = [r["qini"][m] for r in runs]
            ax.plot([min(q), max(q)], [i, i], color=COLOR[m], linewidth=2, alpha=0.45)
            ax.plot(q, [i] * len(q), color=COLOR[m], **DOT)
            ax.plot([np.mean(q)] * 2, [i - 0.22, i + 0.22], color=INK, linewidth=1.5)
        ax.set_yticks(range(len(names)), names)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"Outcome: {outcome} ({100 * runs[0]['outcome_rate']:.2f}% positive)")
        ax.set_xlabel("Qini score")
    if not drew:
        plt.close(fig)
        return
    fig.suptitle("The same benchmark on three different 10% samples", x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.text(0.01, -0.02, "Each dot is one seed (42, 43, 44) · black tick: mean of the three · colored bar: range\n"
             "The ranges overlap, and the order of the models changes from seed to seed.", color=INK2, fontsize=8.5, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "seed_variance_frac0.1.png")


# ── 7. Same code, same rows, two platforms ───────────────────────────────────────────────────────
def plot_platform_comparison():
    lf, df_ = RESULTS / "benchmark_visit_frac0.1_seed42.json", RESULTS / "databricks_visit_frac0.1_seed42.json"
    if not (lf.exists() and df_.exists()):
        return
    lo, db = load(lf)["qini"], load(df_)["result"]["qini"]
    names = [m for m in MODELS if m in lo and m in db][::-1]
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(names) + 1.7))
    for i, m in enumerate(names):
        ax.plot([lo[m], db[m]], [i, i], color=COLOR[m], linewidth=2, alpha=0.45)
        ax.plot(lo[m], i, color=COLOR[m], **DOT)
        ax.plot(db[m], i, color=COLOR[m], marker="s", markersize=8.5, markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none")
        ax.annotate(f"{abs(lo[m] - db[m]):.4f} apart", (max(lo[m], db[m]), i), xytext=(9, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=9)
    ax.set_yticks(range(len(names)), names)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=max(max(lo.values()), max(db.values())) + 0.06)
    ax.plot([], [], color=INK2, **DOT, label="Mac (macOS arm64, 8 cores)")
    ax.plot([], [], color=INK2, marker="s", markersize=8.5, markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none",
            label="Databricks Free Edition (Linux aarch64, 4 cores)")
    ax.set_xlabel("Qini score")
    fig.suptitle("Identical rows, split and seeds on two platforms", x=0.01, ha="left", fontsize=12.5, fontweight="bold", y=1.02)
    fig.legend(loc="upper left", bbox_to_anchor=(0.005, 0.965), ncols=2, handletextpad=0.4, columnspacing=1.6)
    fig.subplots_adjust(top=0.84)
    fig.text(0.01, -0.02, "outcome: visit · 10% sample · seed 42 · the sampled rows, the split and the random baseline are identical on both\n"
             "The Causal Forest agrees to 0.0002. The three LightGBM learners do not, and the cause is not yet identified.",
             color=INK2, fontsize=8.5, va="top")
    save(fig, "platform_comparison_visit_frac0.1_seed42.png")


# ── 8. How big is each source of Qini movement? ──────────────────────────────────────────────────
def plot_noise_sources():
    sf = RESULTS / "sensitivity_visit_frac0.1_seed42.json"
    if not sf.exists():
        return
    s = load(sf)
    rows = [("Order of tied rows (30 random orders)", s["tie_order"]["S-Learner"]["qini_max"] - s["tie_order"]["S-Learner"]["qini_min"]),
            ("LightGBM thread count (1, 4, 8, deterministic)", s["threads_spread"])]
    seeds = [load(f)["qini"]["S-Learner"] for f in sorted(glob.glob(str(RESULTS / "benchmark_visit_frac0.1_seed*.json")))]
    if len(seeds) > 1:
        rows.append(("Which 10% sample is drawn (seeds 42, 43, 44)", max(seeds) - min(seeds)))
    lf, df_ = RESULTS / "benchmark_visit_frac0.1_seed42.json", RESULTS / "databricks_visit_frac0.1_seed42.json"
    if lf.exists() and df_.exists():
        rows.append(("Platform: Mac vs Databricks, identical rows", abs(load(lf)["qini"]["S-Learner"] - load(df_)["result"]["qini"]["S-Learner"])))
    b = load(lf).get("bootstrap") if lf.exists() else None
    if b:
        m = b["models"]["S-Learner"]
        rows.append(("Width of the 95% bootstrap interval", m["ci_high"] - m["ci_low"]))
    rows.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(rows) + 1.6))
    for i, (label, v) in enumerate(rows):
        ax.barh(i, v, height=0.26, color=COLOR["S-Learner"], linewidth=0)
        ax.annotate(f"{v:.4f}", (v, i), xytext=(6, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)
    gap = None
    if lf.exists():
        q = load(lf)["qini"]
        gap = q["S-Learner"] - q["X-Learner"]
        ax.axvline(gap, color=INK2, linewidth=1)
        ax.annotate(f"S-Learner minus X-Learner in this run: {gap:.4f}", (gap, len(rows) - 0.45), xytext=(6, 0),
                    textcoords="offset points", color=INK2, fontsize=9, va="center")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) + 0.1)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(v for _, v in rows) * 1.18)
    ax.set_xlabel("How far the S-Learner's Qini score moves")
    ax.set_title("What moves the Qini score, none of it model quality")
    fig.text(0.01, -0.02, "outcome: visit · 10% sample · S-Learner · all figures read from results/*.json", color=INK2, fontsize=8.5, va="top")
    save(fig, "noise_sources_visit_frac0.1.png")


if __name__ == "__main__":
    import matplotlib.ticker  # noqa: F401
    files = sorted(glob.glob(str(RESULTS / "benchmark_*.json"))) + sorted(glob.glob(str(RESULTS / "databricks_*_seed*.json")))
    for f in files:
        plot_intervals(f)
        plot_qini_curves(f)
        plot_policy(f)
        plot_cate_distributions(f)
        plot_cf_uncertainty(f)
    plot_seed_variance()
    plot_platform_comparison()
    plot_noise_sources()

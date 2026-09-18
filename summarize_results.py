"""Build the seed-variance tables in the README from results/*.json. No figure is typed by hand."""

import glob
import json

import pandas as pd

MODELS = ["S-Learner", "S-Learner (original config)", "X-Learner", "T-Learner", "Causal Forest", "Random"]


def table(outcome):
    runs = [json.load(open(f)) for f in sorted(glob.glob(f"results/benchmark_{outcome}_*.json"))]
    seeds = [r["seed"] for r in runs]
    lines = ["| Model | " + " | ".join(f"Seed {s}" for s in seeds) + " | Mean | Spread |",
             "|---|" + "---|" * (len(seeds) + 2)]
    rows = []
    for m in MODELS:
        q = [r["qini"][m] for r in runs]
        rows.append((sum(q) / len(q), m, q))
    for mean, m, q in sorted(rows, reverse=True):
        lines.append(f"| {m} | " + " | ".join(f"{v:.4f}" for v in q) + f" | {mean:.4f} | {max(q) - min(q):.4f} |")
    ranks = []
    for r in runs:
        order = sorted((m for m in MODELS if m not in ("Random", "S-Learner (original config)")),
                       key=lambda m: -r["qini"][m])
        ranks.append(f"seed {r['seed']}: " + " > ".join(order))
    cf = ", ".join(f"{r['causal_forest']['confident_persuadables_pct']:.2f}%" for r in runs)
    n = runs[0]
    note = (f"Rows per run: {n['rows_sampled']:,} sampled, {n['rows_train']:,} train, {n['rows_test']:,} test. "
            f"Convergence warnings across runs: {sum(r['convergence_warnings'] for r in runs)}. "
            f"Causal Forest confident persuadables by seed: {cf}.")
    return "\n".join(lines), "  \n".join(ranks), note


def intervals_table(path):
    r = json.load(open(path))
    r = r.get("result", r)
    b = r["bootstrap"]
    lines = ["| Model | Qini | 95% interval | Ranked first in replicates |", "|---|---|---|---|"]
    for m in sorted(b["models"], key=lambda m: -b["models"][m]["qini"]):
        v = b["models"][m]
        lines.append(f"| {m} | {v['qini']:.4f} | {v['ci_low']:.4f} to {v['ci_high']:.4f} | {100 * v['share_of_replicates_ranked_first']:.0f}% |")
    sep = [f"{k} ({v['diff']:+.4f}, {v['ci_low']:+.4f} to {v['ci_high']:+.4f})" for k, v in b["differences"].items() if v["separable"]]
    note = (f"Outcome `{r['outcome']}`, {r['rows_test']:,} test rows, seed {r['seed']}, {b['n_boot']} paired bootstrap replicates. "
            + ("Differences whose interval excludes zero: " + "; ".join(sep) + "." if sep else "No difference between two models excludes zero."))
    return "\n".join(lines), note


def headline_table():
    """Mean Qini and capture across the three full-data splits, one row per model."""
    files = sorted(glob.glob("results/databricks_visit_full_seed42_split*_cf0.25.json"))
    runs = [json.load(open(f))["result"] for f in files]
    names = [m for m in MODELS if m in runs[0]["qini"] and m != "Random"]
    lines = ["| Model | Mean Qini over 3 splits | Range across splits | Top 20% capture (mean) | Top 50% capture (mean) |",
             "|---|---|---|---|---|"]
    rows = []
    for m in names:
        q = [r["qini"][m] for r in runs]
        c20 = [r["capture_pct"][m]["20"] for r in runs if m in r["capture_pct"]]
        c50 = [r["capture_pct"][m]["50"] for r in runs if m in r["capture_pct"]]
        rows.append((sum(q) / len(q), m, min(q), max(q), c20, c50))
    for mean, m, lo, hi, c20, c50 in sorted(rows, reverse=True):
        cap20 = f"{sum(c20) / len(c20):.1f}%" if c20 else "not applicable"
        cap50 = f"{sum(c50) / len(c50):.1f}%" if c50 else "not applicable"
        lines.append(f"| {m} | {mean:.4f} | {lo:.4f} to {hi:.4f} | {cap20} | {cap50} |")
    n = runs[0]
    return "\n".join(lines), n


def full_data_splits_table():
    files = sorted(glob.glob("results/databricks_visit_full_seed42_split*_cf0.25.json"))
    runs = [json.load(open(f))["result"] for f in files]
    if not runs:
        return "", ""
    names = [m for m in MODELS if m in runs[0]["qini"] and m != "Random"]
    lines = ["| Model | " + " | ".join(f"Split {r['split_seed']}" for r in runs) + " | Mean | Spread |", "|---|" + "---|" * (len(runs) + 2)]
    rows = []
    for m in names:
        q = [r["qini"][m] for r in runs]
        cis = [r["bootstrap"]["models"][m] for r in runs]
        rows.append((sum(q) / len(q), m, q, cis))
    for mean, m, q, cis in sorted(rows, reverse=True):
        cells = " | ".join(f"{v:.4f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]" for v, c in zip(q, cis))
        lines.append(f"| {m} | {cells} | {mean:.4f} | {max(q) - min(q):.4f} |")
    ranks = "  \n".join(f"split {r['split_seed']}: " + " > ".join(sorted(names, key=lambda m: -r["qini"][m])) for r in runs)
    note = (f"Full dataset, {runs[0]['rows_test']:,} test rows per split, Causal Forest on 25% of training rows, "
            f"200 paired bootstrap replicates per split, {sum(r['runtime_seconds'] for r in runs) / 60:.0f} minutes of serverless compute in total.")
    return "\n".join(lines) + "\n\nRank order by Qini:  \n" + ranks, note


def forest_scaling_table():
    rows = []
    for f in sorted(glob.glob("results/databricks_visit_full_seed42_split42_cf*.json")):
        d = json.load(open(f))
        frac = float(f.split("_cf")[1].split(".json")[0].replace(".partial", ""))
        if f.endswith(".partial.json"):
            rows.append((frac, f"{int(round(d['rows_train'] * frac)):,}", "out of memory", "not reached", "not reached", f"{d['peak_rss_gb']} GB at the checkpoint, then killed"))
        else:
            r = d["result"]
            b = r["bootstrap"]["models"]["Causal Forest"]
            rows.append((frac, f"{r['causal_forest']['train_rows']:,}", f"{r['qini']['Causal Forest']:.4f}",
                         f"{b['ci_low']:.4f} to {b['ci_high']:.4f}", f"{r['causal_forest']['confident_persuadables_pct']:.2f}%",
                         f"{r['peak_rss_gb']} GB, {r['runtime_seconds'] / 60:.0f} min"))
    lines = ["| Share of training rows | Rows | Causal Forest Qini | 95% interval | Confident persuadables | Peak memory, time |",
             "|---|---|---|---|---|---|"]
    for frac, n, q, ci, pers, mem in sorted(rows):
        lines.append(f"| {frac:.0%} | {n} | {q} | {ci} | {pers} | {mem} |")
    return "\n".join(lines)


def render_all():
    out = []
    t, r = headline_table()
    out.append(f"<!-- generated:headline -->\n{t}\n\nFull dataset, {r['rows_train']:,} training rows and {r['rows_test']:,} test rows per split, three train/test splits, "
               f"Causal Forest on 25% of the training rows. Every figure is read from `results/`.\n<!-- /generated:headline -->")
    t, note = full_data_splits_table()
    out.append(f"<!-- generated:splits -->\n{t}\n\n{note}\n<!-- /generated:splits -->")
    out.append("<!-- generated:forest -->\n" + forest_scaling_table() + "\n<!-- /generated:forest -->")
    for oc in ("visit", "conversion"):
        t, ranks, note = table(oc)
        out.append(f"<!-- generated:seeds-{oc} -->\n{t}\n\nRank order by Qini:  \n{ranks}\n\n{note}\n<!-- /generated:seeds-{oc} -->")
    t, note = intervals_table("results/benchmark_visit_frac0.1_seed42.json")
    out.append(f"<!-- generated:intervals-10pct -->\n{t}\n\n{note}\n<!-- /generated:intervals-10pct -->")
    return out


def write_readme(path="README.md"):
    """Replace every <!-- generated:NAME --> ... <!-- /generated:NAME --> block in the README."""
    import re
    s = open(path).read()
    for block in render_all():
        name = re.match(r"<!-- generated:([\w-]+) -->", block).group(1)
        pattern = re.compile(rf"<!-- generated:{name} -->.*?<!-- /generated:{name} -->", re.S)
        if not pattern.search(s):
            raise SystemExit(f"README has no generated:{name} block")
        s = pattern.sub(lambda _: block, s)
    open(path, "w").write(s)
    print("README tables regenerated")


if __name__ == "__main__":
    import sys
    if "--write-readme" in sys.argv:
        write_readme()
        raise SystemExit
    for oc in ("visit", "conversion"):
        t, ranks, note = table(oc)
        print(f"#### Outcome: `{oc}`\n\n{t}\n\nRank order by Qini:  \n{ranks}\n\n{note}\n")
    t, note = intervals_table("results/benchmark_visit_frac0.1_seed42.json")
    print(f"#### Bootstrap intervals, seed 42, `visit`, 10% sample\n\n{t}\n\n{note}\n")
    t, note = full_data_splits_table()
    if t:
        print(f"#### Full dataset, three train/test splits, on Databricks\n\n{t}\n\n{note}\n")
    print("#### Scaling the Causal Forest on Databricks Free Edition (16.4 GB serverless, full dataset, seed 42)\n")
    print(forest_scaling_table())
    print("\nThe meta-learners are identical across these runs; only the forest changes. Free Edition fits the forest on a quarter of the training rows and not on half.\n")
    for f in sorted(glob.glob("results/databricks_visit_full_*_cf0.1.json")):
        d = json.load(open(f))
        r = d["result"]
        t, note = intervals_table(f)
        cf = r["causal_forest"]
        print(f"#### Full dataset on Databricks: Causal Forest on {cf['train_rows']:,} training rows\n\n{t}\n\n{note} "
              f"Peak memory {r['peak_rss_gb']} GB, {r['runtime_seconds'] / 60:.0f} minutes on {d['env']['cpu_count']} cores. "
              f"Confident persuadables {cf['confident_persuadables_pct']:.2f}%, confident sleeping dogs {cf['confident_sleeping_dogs_pct']:.2f}%.\n")

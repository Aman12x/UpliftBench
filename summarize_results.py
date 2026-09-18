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


if __name__ == "__main__":
    for oc in ("visit", "conversion"):
        t, ranks, note = table(oc)
        print(f"#### Outcome: `{oc}`\n\n{t}\n\nRank order by Qini:  \n{ranks}\n\n{note}\n")
    t, note = intervals_table("results/benchmark_visit_frac0.1_seed42.json")
    print(f"#### Bootstrap intervals, seed 42, `visit`, 10% sample\n\n{t}\n\n{note}\n")
    for f in sorted(glob.glob("results/databricks_visit_full_*.json")):
        d = json.load(open(f))
        r = d["result"]
        t, note = intervals_table(f)
        cf = r["causal_forest"]
        print(f"#### Full dataset on Databricks: Causal Forest on {cf['train_rows']:,} training rows\n\n{t}\n\n{note} "
              f"Peak memory {r['peak_rss_gb']} GB, {r['runtime_seconds'] / 60:.0f} minutes on {d['env']['cpu_count']} cores. "
              f"Confident persuadables {cf['confident_persuadables_pct']:.2f}%, confident sleeping dogs {cf['confident_sleeping_dogs_pct']:.2f}%.\n")

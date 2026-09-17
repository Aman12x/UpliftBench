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


if __name__ == "__main__":
    for oc in ("visit", "conversion"):
        t, ranks, note = table(oc)
        print(f"#### Outcome: `{oc}`\n\n{t}\n\nRank order by Qini:  \n{ranks}\n\n{note}\n")

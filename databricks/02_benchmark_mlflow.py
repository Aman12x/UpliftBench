"""
Databricks serverless job: run the repository's seeded benchmark against the volume copy of
the dataset and log every figure to MLflow.

Submitted as a Python script task with environment_version "4" (Python 3.12) and the ARM
wheels for causalml and econml installed from the volume. See databricks/submit.py.
"""

import argparse
import json
import os
import platform
import resource
import sys
from importlib.metadata import version

import mlflow
import psutil

ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True)
ap.add_argument("--sample-frac", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--outcome", default="visit")
ap.add_argument("--cf-frac", type=float, default=None)
a = ap.parse_args()

sys.path.insert(0, f"/Workspace/Users/{a.user}/upliftbench")
from run_benchmark import run  # noqa: E402

DATA = "/Volumes/workspace/upliftbench/raw/criteo-research-uplift-v2.1.csv.gz"
RESULTS = "/Volumes/workspace/upliftbench/raw/results"

env = {
    "python": platform.python_version(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
    "memory_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
    "memory_available_gb": round(psutil.virtual_memory().available / 1e9, 2),
    **{p: version(p) for p in ("causalml", "econml", "lightgbm", "scikit-learn", "numpy", "pandas", "duckdb")},
}

mlflow.set_experiment(f"/Users/{a.user}/upliftbench/benchmark")
with mlflow.start_run(run_name=f"{a.outcome}-frac{a.sample_frac}-seed{a.seed}-cf{a.cf_frac or 'all'}"):
    mlflow.log_params({"sample_frac": a.sample_frac, "seed": a.seed, "outcome": a.outcome, "cf_frac": a.cf_frac or 1.0})
    mlflow.log_params({f"env_{k}": v for k, v in env.items()})
    out = run(a.sample_frac, a.seed, a.outcome, a.cf_frac, data_path=DATA, results_dir=RESULTS)
    out["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)  # Linux reports KB
    for model, q in out["qini"].items():
        mlflow.log_metric("qini_" + model.replace(" ", "_").replace("(", "").replace(")", ""), q)
    mlflow.log_metrics({
        "convergence_warnings": out["convergence_warnings"],
        "cf_confident_persuadables_pct": out["causal_forest"]["confident_persuadables_pct"],
        "cf_confident_sleeping_dogs_pct": out["causal_forest"]["confident_sleeping_dogs_pct"],
        "runtime_seconds": out["runtime_seconds"], "peak_rss_gb": out["peak_rss_gb"], "rows_sampled": out["rows_sampled"],
    })
    mlflow.log_dict(out, "benchmark.json")

print("RESULT_JSON " + json.dumps({"env": env, "result": out}))

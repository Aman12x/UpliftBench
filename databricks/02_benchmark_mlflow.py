# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Seeded benchmark with MLflow tracking
# MAGIC Runs the repository's own `run_benchmark.run` against the volume copy of the dataset,
# MAGIC records what the serverless environment provides, and logs every figure to MLflow.

# COMMAND ----------

# MAGIC %pip install -q causalml==0.16.0 econml==0.16.0 lightgbm==4.6.0 duckdb psutil

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("sample_frac", "0.1")
dbutils.widgets.text("seed", "42")
dbutils.widgets.text("outcome", "visit")
dbutils.widgets.text("cf_frac", "")

sample_frac = float(dbutils.widgets.get("sample_frac"))
seed = int(dbutils.widgets.get("seed"))
outcome = dbutils.widgets.get("outcome")
cf_frac = float(dbutils.widgets.get("cf_frac")) if dbutils.widgets.get("cf_frac") else None

# COMMAND ----------

import json, os, platform, resource, sys
from importlib.metadata import version

import psutil

env = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "cpu_count": os.cpu_count(),
    "memory_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
    "memory_available_gb": round(psutil.virtual_memory().available / 1e9, 2),
    **{p: version(p) for p in ("causalml", "econml", "lightgbm", "scikit-learn", "numpy", "pandas", "duckdb", "mlflow")},
}
print(json.dumps(env, indent=1))

# COMMAND ----------

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
code_dir = f"/Workspace/Users/{user}/upliftbench"
sys.path.insert(0, code_dir)

import mlflow
from run_benchmark import run

DATA = "/Volumes/workspace/upliftbench/raw/criteo-research-uplift-v2.1.csv.gz"
RESULTS = "/Volumes/workspace/upliftbench/raw/results"

mlflow.set_experiment(f"/Users/{user}/upliftbench/benchmark")
with mlflow.start_run(run_name=f"{outcome}-frac{sample_frac}-seed{seed}-cf{cf_frac or 'all'}"):
    mlflow.log_params({"sample_frac": sample_frac, "seed": seed, "outcome": outcome, "cf_frac": cf_frac or 1.0})
    mlflow.log_params({f"env_{k}": v for k, v in env.items()})
    out = run(sample_frac, seed, outcome, cf_frac, data_path=DATA, results_dir=RESULTS)
    out["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)  # Linux reports KB
    for model, q in out["qini"].items():
        mlflow.log_metric(f"qini_{model.replace(' ', '_').replace('(', '').replace(')', '')}", q)
    mlflow.log_metrics({
        "convergence_warnings": out["convergence_warnings"],
        "cf_confident_persuadables_pct": out["causal_forest"]["confident_persuadables_pct"],
        "cf_confident_sleeping_dogs_pct": out["causal_forest"]["confident_sleeping_dogs_pct"],
        "runtime_seconds": out["runtime_seconds"], "peak_rss_gb": out["peak_rss_gb"],
        "rows_sampled": out["rows_sampled"],
    })
    mlflow.log_dict(out, "benchmark.json")

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"env": env, "result": out}))

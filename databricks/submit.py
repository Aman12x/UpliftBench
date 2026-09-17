"""
Submit databricks/02_benchmark_mlflow.py as a serverless job and save its result locally.

    python databricks/submit.py --seed 42 --outcome visit

Reads credentials from ~/.databrickscfg. Uploads the current src/, run_benchmark.py and job
script first, so the job always runs the code in this checkout.
"""

import argparse
import io
import json
import time
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import compute, jobs, workspace as ws

ROOT = Path(__file__).parent.parent
WHEELS = "/Volumes/workspace/upliftbench/raw/wheels"
CODE = ["run_benchmark.py", "src/__init__.py", "src/preprocessing.py", "src/learners.py", "src/evaluation.py", "src/intervals.py",
        "databricks/02_benchmark_mlflow.py"]

ap = argparse.ArgumentParser()
ap.add_argument("--sample-frac", default="0.1")
ap.add_argument("--seed", default="42")
ap.add_argument("--outcome", default="visit")
ap.add_argument("--cf-frac", default=None)
ap.add_argument("--split-seed", default="42")
ap.add_argument("--n-boot", default="0")
a = ap.parse_args()

w = WorkspaceClient()
user = w.current_user.me().user_name
base = f"/Users/{user}/upliftbench"
for d in ("src", "databricks"):
    w.workspace.mkdirs(f"{base}/{d}")
for rel in CODE:
    w.workspace.upload(f"{base}/{rel}", io.BytesIO((ROOT / rel).read_bytes() or b"# package\n"),
                       format=ws.ImportFormat.AUTO, overwrite=True)

wheels = [f"{WHEELS}/{f.name}" for f in w.files.list_directory_contents(WHEELS) if f.name.endswith(".whl")]
params = ["--user", user, "--sample-frac", a.sample_frac, "--seed", a.seed, "--outcome", a.outcome,
          "--split-seed", a.split_seed, "--n-boot", a.n_boot]
if a.cf_frac:
    params += ["--cf-frac", a.cf_frac]

run = w.jobs.submit(
    run_name=f"upliftbench-benchmark-{a.outcome}-frac{a.sample_frac}-seed{a.seed}-split{a.split_seed}-cf{a.cf_frac}",
    tasks=[jobs.SubmitTask(task_key="benchmark", environment_key="py312",
                           spark_python_task=jobs.SparkPythonTask(
                               python_file=f"/Workspace{base}/databricks/02_benchmark_mlflow.py", parameters=params))],
    environments=[jobs.JobEnvironment(environment_key="py312", spec=compute.Environment(
        environment_version="4", dependencies=[*wheels, "lightgbm==4.6.0", "duckdb", "psutil", "mlflow"]))],
)
print("submitted", run.run_id, flush=True)
while True:
    r = w.jobs.get_run(run.run_id)
    if r.state.life_cycle_state.value in ("TERMINATED", "INTERNAL_ERROR", "SKIPPED"):
        break
    time.sleep(20)
print(r.state.life_cycle_state.value, r.state.result_state, f"{(r.end_time - r.start_time) // 1000}s", r.state.state_message or "")
o = w.jobs.get_run_output(r.tasks[0].run_id)
hit = [l for l in (o.logs or "").splitlines() if l.startswith("RESULT_JSON ")]
if hit:
    d = json.loads(hit[0][12:])
    tag = f"frac{a.sample_frac}" if float(a.sample_frac) else "full"
    dest = ROOT / "results" / f"databricks_{a.outcome}_{tag}_seed{a.seed}_split{a.split_seed}_cf{a.cf_frac or 'all'}.json"
    dest.write_text(json.dumps(d, indent=2))
    print("saved", dest)
else:
    print("error:", o.error)
    print((o.error_trace or o.logs or "")[-3000:])
print("run page:", r.run_page_url)

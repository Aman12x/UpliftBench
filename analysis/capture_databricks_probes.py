"""
Pull the environment-probe job outputs back out of Databricks and save them under results/.

    python analysis/capture_databricks_probes.py

The probes measured Databricks Free Edition serverless compute: architecture, memory, base
library versions, network reachability, and why causalml/econml cannot be built from source there.
"""

import json
import re
from pathlib import Path

from databricks.sdk import WorkspaceClient

ESC = chr(27)
ANSI = re.compile(re.escape(ESC) + r"\[[0-9;?]*[A-Za-z]")


def clean(o):
    if isinstance(o, str):
        return ANSI.sub("", o)
    if isinstance(o, list):
        return [clean(x) for x in o if not (isinstance(x, str) and " eta " in x)]
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    return o


w = WorkspaceClient()
probes = {}
for x in w.jobs.list_runs(limit=25):
    if not x.run_name.startswith("upliftbench-diag"):
        continue
    r = w.jobs.get_run(x.run_id)
    o = w.jobs.get_run_output(r.tasks[0].run_id)
    entry = {
        "run_id": x.run_id, "state": x.state.life_cycle_state.value,
        "result": x.state.result_state.value if x.state.result_state else None,
        "message": x.state.state_message, "run_page_url": x.run_page_url,
    }
    if o.notebook_output and o.notebook_output.result:
        entry["output"] = clean(json.loads(o.notebook_output.result))
    hit = [l for l in (o.logs or "").splitlines() if l.startswith("DIAG_JSON ")]
    if hit:
        entry["output"] = clean(json.loads(hit[0][10:]))
    if o.error:
        entry["error"] = clean(o.error)[:600]
    probes[x.run_name] = entry

dest = Path(__file__).parent.parent / "results" / "databricks_environment_probes.json"
dest.write_text(json.dumps({"probes": probes}, indent=2))
print("saved", dest, "with", len(probes), "probes")
for name, p in probes.items():
    out = p.get("output", {})
    keys = [k for k in ("python", "machine", "platform", "cpu_count", "memory_total_gb", "memory_available_gb",
                        "numpy", "pandas", "scikit-learn") if k in out]
    print(f"  {name}: {p['result']} | " + ", ".join(f"{k}={out[k]}" for k in keys))

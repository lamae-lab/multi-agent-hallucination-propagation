"""Shared helper functions for the dashboard."""

import json
import os
from glob import glob

import pandas as pd

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "experiments", "results"
)


def list_experiment_runs(results_dir: str | None = None) -> list[dict]:
    """Scan results directory for experiment runs with config.json."""
    base = results_dir or RESULTS_DIR
    runs = []
    for entry in sorted(os.listdir(base), reverse=True):
        run_dir = os.path.join(base, entry)
        config_path = os.path.join(run_dir, "config.json")
        if os.path.isdir(run_dir) and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            config["run_id"] = entry
            config["results_dir"] = run_dir
            runs.append(config)
        elif os.path.isdir(run_dir) and glob(os.path.join(run_dir, "*.jsonl")):
            # Legacy run without config.json — infer from files
            traces = glob(os.path.join(run_dir, "*.jsonl"))
            families = list(
                {
                    os.path.basename(t).split("_")[1]
                    for t in traces
                    if "_" in os.path.basename(t)
                }
            )
            runs.append(
                {
                    "run_id": entry,
                    "results_dir": run_dir,
                    "families": families,
                    "n_questions": len(traces) // max(len(families), 1),
                    "seed": "unknown",
                    "injection_enabled": "injected" in traces[0] if traces else False,
                    "status": "completed",
                }
            )
    return runs


def load_experiment_run(run_dir: str) -> dict:
    """Load a full experiment run: config + metrics + trace paths."""
    result = {"run_dir": run_dir, "traces": {}, "metrics": {}, "config": {}}

    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            result["config"] = json.load(f)

    metrics_path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            result["metrics"] = json.load(f)

    # Group traces by family
    for trace_file in sorted(glob(os.path.join(run_dir, "*.jsonl"))):
        fname = os.path.basename(trace_file)
        parts = fname.replace(".jsonl", "").split("_")
        if len(parts) >= 2:
            family_id = parts[1]
            if family_id not in result["traces"]:
                result["traces"][family_id] = []
            result["traces"][family_id].append(trace_file)

    return result


def format_metrics_table(metrics: dict) -> pd.DataFrame:
    """Convert per-family metrics dict to a display DataFrame."""
    if not metrics:
        return pd.DataFrame()

    rows = []
    for fid, m in metrics.items():
        rows.append(
            {
                "Family": fid.upper(),
                "Runs": m.get("n_runs", 0),
                "Errors": m.get("n_errors", 0),
                "Survival": round(m.get("mean_survival", 0), 3),
                "Distance": round(m.get("mean_distance", 0), 2),
                "Amplification": round(m.get("mean_amplification", 0), 3),
                "Mutation": round(m.get("mean_mutation", 0), 3),
                "Marker": m.get("family_marker_name", ""),
                "Marker Value": round(m.get("family_marker", 0), 3),
            }
        )
    return pd.DataFrame(rows)


def load_trace_messages(trace_path: str) -> list[dict]:
    """Load messages from a JSONL trace file."""
    messages = []
    metadata = {}
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("type") == "metadata":
                metadata = obj
            else:
                messages.append(obj)
    return metadata, messages


def get_question_ids(run_data: dict) -> list[str]:
    """Extract unique question IDs from a run's traces."""
    qids = set()
    for family_traces in run_data["traces"].values():
        for trace_path in family_traces:
            fname = os.path.basename(trace_path).replace(".jsonl", "")
            parts = fname.split("_")
            if len(parts) >= 3:
                qid = "_".join(parts[2:])
                qids.add(qid)
    return sorted(qids)

"""Tests for dashboard helper functions."""

import os

import pandas as pd

from src.ui.helpers import (
    format_metrics_table,
    list_experiment_runs,
    load_experiment_run,
)

RESULTS_DIR = os.path.join("experiments", "results")


def test_list_experiment_runs_finds_pilot():
    runs = list_experiment_runs(RESULTS_DIR)
    assert len(runs) > 0
    # Pilot run should be there
    run_ids = [r["run_id"] for r in runs]
    assert any("pilot" in rid for rid in run_ids)


def test_load_experiment_run_has_traces():
    runs = list_experiment_runs(RESULTS_DIR)
    pilot = next((r for r in runs if "pilot" in r["run_id"]), None)
    assert pilot is not None

    run_data = load_experiment_run(pilot["results_dir"])
    assert "traces" in run_data
    assert len(run_data["traces"]) > 0


def test_format_metrics_table_produces_dataframe():
    metrics = {
        "f1": {
            "n_runs": 3,
            "n_errors": 0,
            "mean_survival": 0.333,
            "mean_distance": 0.33,
            "mean_amplification": 0.0,
            "mean_mutation": 0.0,
            "family_marker": 0.333,
            "family_marker_name": "self_validation_rate",
        }
    }
    df = format_metrics_table(metrics)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "Family" in df.columns
    assert df.iloc[0]["Family"] == "F1"

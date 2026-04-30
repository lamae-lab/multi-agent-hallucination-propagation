"""Page 4: Family Comparison — side-by-side answers and metrics."""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.ui.helpers import (
    format_metrics_table,
    get_question_ids,
    list_experiment_runs,
    load_experiment_run,
    load_trace_messages,
)

st.title("Family Comparison")

runs = list_experiment_runs()
if not runs:
    st.info("No experiment runs found. Run an experiment first.")
    st.stop()

run_names = [r["run_id"] for r in runs]
selected_run = st.selectbox("Select Experiment Run", run_names)
run_data = load_experiment_run(runs[run_names.index(selected_run)]["results_dir"])

available_families = sorted(run_data["traces"].keys())
selected_families = st.multiselect(
    "Select Families to Compare (2-5)",
    available_families,
    default=available_families[:2],
)

if len(selected_families) < 1:
    st.info("Select at least one family.")
    st.stop()

# Metrics comparison table
metrics = run_data.get("metrics", {})
if metrics:
    st.subheader("Metrics Comparison")
    filtered_metrics = {k: v for k, v in metrics.items() if k in selected_families}
    if filtered_metrics:
        st.dataframe(format_metrics_table(filtered_metrics), width="stretch")

# Per-question answer comparison
st.subheader("Answer Comparison")
question_ids = get_question_ids(run_data)

for qid in question_ids:
    with st.expander(f"Question: {qid}"):
        cols = st.columns(len(selected_families))
        for col, fid in zip(cols, selected_families):
            with col:
                st.markdown(f"**{fid.upper()}**")
                trace_file = None
                for tf in run_data["traces"].get(fid, []):
                    if qid in os.path.basename(tf):
                        trace_file = tf
                        break

                if not trace_file or not os.path.exists(trace_file):
                    st.warning("No trace")
                    continue

                metadata, messages = load_trace_messages(trace_file)
                answer = metadata.get("answer", "N/A")
                expected = metadata.get("expected_answer", "")

                st.text(answer[:300])
                if expected and expected.lower() in answer.lower():
                    st.success("Correct")
                elif expected:
                    st.error(f"Expected: {expected}")

"""Page 2: Trace Viewer — side-by-side trace comparison."""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.ui.helpers import (
    get_question_ids,
    list_experiment_runs,
    load_experiment_run,
    load_trace_messages,
)

ROLE_COLORS = {
    "system": "gray",
    "user": "blue",
    "assistant": "green",
    "tool": "orange",
}

st.title("Trace Viewer")

runs = list_experiment_runs()
if not runs:
    st.info("No experiment runs found. Run an experiment first.")
    st.stop()

run_names = [r["run_id"] for r in runs]
selected_run = st.selectbox("Select Experiment Run", run_names)
run_data = load_experiment_run(runs[run_names.index(selected_run)]["results_dir"])

# Get available families and questions
available_families = sorted(run_data["traces"].keys())
question_ids = get_question_ids(run_data)

if not question_ids:
    st.warning("No traces found in this run.")
    st.stop()

selected_question = st.selectbox("Select Question", question_ids)
selected_families = st.multiselect(
    "Compare Families", available_families, default=available_families[:3]
)

if not selected_families:
    st.info("Select at least one family to view traces.")
    st.stop()

# Display traces side-by-side
cols = st.columns(len(selected_families))

for col, fid in zip(cols, selected_families):
    with col:
        st.subheader(fid.upper())

        # Find the trace file for this family + question
        trace_file = None
        for tf in run_data["traces"].get(fid, []):
            if selected_question in os.path.basename(tf):
                trace_file = tf
                break

        if not trace_file or not os.path.exists(trace_file):
            st.warning("No trace found")
            continue

        metadata, messages = load_trace_messages(trace_file)

        # Show injection info if present
        if metadata.get("answer"):
            st.caption(f"Answer: {metadata['answer'][:100]}")

        # Display messages
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            color = ROLE_COLORS.get(role, "gray")
            name = msg.get("name", "")
            label = f"**:{color}[{role.upper()}]**"
            if name:
                label += f" ({name})"

            st.markdown(label)
            if content:
                st.text(content[:500])

            # Show tool calls
            if msg.get("tool_calls"):
                with st.expander("Tool Calls"):
                    st.json(msg["tool_calls"])

            st.divider()

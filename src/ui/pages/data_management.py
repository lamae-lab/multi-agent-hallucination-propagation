"""Page 6: Data Management — browse, load, and delete experiment runs."""

import os
import shutil
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.ui.helpers import list_experiment_runs

st.title("Data Management")

runs = list_experiment_runs()

if not runs:
    st.info("No experiment runs found. Run an experiment first.")
    st.stop()

st.subheader(f"Saved Experiment Runs ({len(runs)})")

for run in runs:
    run_id = run.get("run_id", "unknown")
    families = ", ".join(f.upper() for f in run.get("families", []))
    n_questions = run.get("n_questions", "?")
    seed = run.get("seed", "?")
    injection = "Yes" if run.get("injection_enabled") else "No"
    status = run.get("status", "unknown")

    col1, col2, col3 = st.columns([4, 1, 1])

    with col1:
        st.markdown(
            f"**{run_id}**  \n"
            f"Families: {families} | Questions: {n_questions} | "
            f"Seed: {seed} | Injection: {injection} | Status: {status}"
        )

    with col2:
        if st.button("Load", key=f"load_{run_id}"):
            st.session_state["active_run"] = run_id
            st.success(f"Loaded: {run_id}")

    with col3:
        if st.button("Delete", key=f"del_{run_id}", type="secondary"):
            st.session_state[f"confirm_delete_{run_id}"] = True

        if st.session_state.get(f"confirm_delete_{run_id}"):
            st.warning(f"Delete {run_id}?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes", key=f"yes_{run_id}"):
                    shutil.rmtree(run["results_dir"], ignore_errors=True)
                    st.session_state.pop(f"confirm_delete_{run_id}", None)
                    st.rerun()
            with c2:
                if st.button("No", key=f"no_{run_id}"):
                    st.session_state.pop(f"confirm_delete_{run_id}", None)
                    st.rerun()

    st.divider()

# Show active run
active = st.session_state.get("active_run")
if active:
    st.sidebar.success(f"Active run: {active}")

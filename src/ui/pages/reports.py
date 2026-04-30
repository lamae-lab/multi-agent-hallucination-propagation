"""Page 5: Reports — CSV and LaTeX export."""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.ui.helpers import (
    format_metrics_table,
    list_experiment_runs,
    load_experiment_run,
)

st.title("Reports")

runs = list_experiment_runs()
if not runs:
    st.info("No experiment runs found. Run an experiment first.")
    st.stop()

run_names = [r["run_id"] for r in runs]
selected_run = st.selectbox("Select Experiment Run", run_names)
run_data = load_experiment_run(runs[run_names.index(selected_run)]["results_dir"])

metrics = run_data.get("metrics", {})
if not metrics:
    st.warning("No metrics found for this run.")
    st.stop()

df = format_metrics_table(metrics)

st.subheader("Summary Table Preview")
st.dataframe(df, width="stretch")

col1, col2 = st.columns(2)

with col1:
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="Export CSV",
        data=csv_data,
        file_name=f"{selected_run}_summary.csv",
        mime="text/csv",
    )

with col2:
    # Generate LaTeX tabular
    def escape_latex(s):
        return (
            str(s)
            .replace("_", r"\_")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("#", r"\#")
        )

    headers = " & ".join(escape_latex(c) for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        rows.append(" & ".join(escape_latex(v) for v in row.values))

    latex = (
        r"\begin{tabular}{" + "l" * len(df.columns) + "}\n"
        r"\hline" + "\n" + headers + r" \\" + "\n"
        r"\hline" + "\n" + r" \\".join(rows) + r" \\" + "\n"
        r"\hline" + "\n"
        r"\end{tabular}"
    )

    st.download_button(
        label="Export LaTeX",
        data=latex,
        file_name=f"{selected_run}_summary.tex",
        mime="text/plain",
    )

    with st.expander("LaTeX Preview"):
        st.code(latex, language="latex")

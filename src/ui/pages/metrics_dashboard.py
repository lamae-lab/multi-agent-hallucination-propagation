"""Page 3: Metrics Dashboard — interactive charts."""

import os
import sys

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.ui.helpers import (
    format_metrics_table,
    list_experiment_runs,
    load_experiment_run,
)

st.title("Metrics Dashboard")

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
st.dataframe(df, width="stretch")

# Tabs for different charts
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Survival x Amplification",
        "Propagation Distance",
        "Family Markers",
        "Heatmap",
    ]
)

with tab1:
    fig = px.scatter(
        df,
        x="Survival",
        y="Amplification",
        text="Family",
        size_max=15,
        title="Survival Rate vs Amplification Factor",
    )
    fig.update_traces(textposition="top center", marker=dict(size=15))
    fig.update_layout(
        xaxis_title="Mean Survival Rate", yaxis_title="Mean Amplification Factor"
    )
    st.plotly_chart(fig, width="stretch")

with tab2:
    fig = px.bar(
        df,
        x="Family",
        y="Distance",
        title="Mean Propagation Distance per Family",
        color="Family",
    )
    fig.update_layout(yaxis_title="Mean Propagation Distance")
    st.plotly_chart(fig, width="stretch")

with tab3:
    fig = px.bar(
        df,
        x="Family",
        y="Marker Value",
        title="Family-Specific Behavioral Markers",
        color="Family",
        text="Marker",
    )
    fig.update_layout(yaxis_title="Marker Value")
    st.plotly_chart(fig, width="stretch")

with tab4:
    st.info(
        "Heatmap requires multiple injection types. Run experiments with different hallucination types to populate this view."
    )
    # Placeholder — will show Family x Type matrix when data is available
    if len(metrics) >= 2:
        families = list(metrics.keys())
        values = [[metrics[f].get("mean_survival", 0)] for f in families]
        fig = go.Figure(
            data=go.Heatmap(
                z=values,
                x=["Survival"],
                y=[f.upper() for f in families],
                colorscale="RdYlGn_r",
            )
        )
        fig.update_layout(title="Family x Metric Heatmap")
        st.plotly_chart(fig, width="stretch")

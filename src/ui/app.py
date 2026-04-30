"""Hallucination Resilience Dashboard — Main Entry Point.

Launch with: streamlit run src/ui/app.py
"""

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Hallucination Resilience Dashboard",
    page_icon="🔬",
    layout="wide",
)

# Resolve page paths relative to this file
PAGES_DIR = Path(__file__).parent / "pages"

pages = [
    st.Page(str(PAGES_DIR / "experiment_runner.py"), title="Experiment Runner", icon="🚀"),
    st.Page(str(PAGES_DIR / "trace_viewer.py"), title="Trace Viewer", icon="🔍"),
    st.Page(str(PAGES_DIR / "metrics_dashboard.py"), title="Metrics Dashboard", icon="📊"),
    st.Page(str(PAGES_DIR / "comparison.py"), title="Family Comparison", icon="⚖️"),
    st.Page(str(PAGES_DIR / "reports.py"), title="Reports", icon="📄"),
    st.Page(str(PAGES_DIR / "data_management.py"), title="Data Management", icon="🗂️"),
]

pg = st.navigation(pages)
pg.run()

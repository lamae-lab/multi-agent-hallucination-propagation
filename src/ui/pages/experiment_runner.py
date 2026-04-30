"""Page 1: Experiment Runner — configure and run experiments."""

import json
import os
import sys
from datetime import datetime

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.families.f1_react import F1React
from src.families.f2_state_graph import F2StateGraph
from src.families.f3_crew import F3Crew
from src.families.f4_group_chat import F4GroupChat
from src.families.f5_handoff_swarm import F5HandoffSwarm
from src.injection.mode1_synthetic import InjectionConfig, inject_and_solve
from src.metrics.family_markers import compute_family_marker
from src.metrics.layer2_propagation import compute_metrics
from src.tasks.hotpot_qa import sample
from src.tracing.trace import save_trace
from src.ui.helpers import format_metrics_table

FAMILY_MAP = {
    "F1 - ReAct": ("f1", "configs/families/f1_react.yaml", F1React),
    "F2 - State-Graph": ("f2", "configs/families/f2_state_graph.yaml", F2StateGraph),
    "F3 - Hierarchical Crew": ("f3", "configs/families/f3_crew.yaml", F3Crew),
    "F4 - Group-Chat": ("f4", "configs/families/f4_group_chat.yaml", F4GroupChat),
    "F5 - Handoff Swarm": (
        "f5",
        "configs/families/f5_handoff_swarm.yaml",
        F5HandoffSwarm,
    ),
}

st.title("Experiment Runner")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    selected_families = st.multiselect(
        "Select Families",
        list(FAMILY_MAP.keys()),
        default=list(FAMILY_MAP.keys()),
    )
    n_questions = st.slider("Number of Questions", 1, 50, 5)
    seed = st.number_input("Random Seed", value=42, step=1)

    st.divider()
    injection_enabled = st.toggle("Enable Injection", value=False)
    false_statement = ""
    hallucination_type = "memorization"
    if injection_enabled:
        false_statement = st.text_input(
            "False Statement", "The capital of France is Berlin"
        )
        hallucination_type = st.selectbox(
            "Hallucination Type",
            ["memorization", "communication", "factual", "execution", "reasoning"],
        )

# Main area
st.markdown(
    f"**Families**: {len(selected_families)} selected | **Questions**: {n_questions} | **Seed**: {seed}"
)
if injection_enabled:
    st.info(f'Injection: "{false_statement}" ({hallucination_type})')

if st.button("Run Experiment", type="primary", disabled=len(selected_families) == 0):
    # Create run directory
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{seed}"
    results_dir = os.path.join("experiments", "results", run_id)
    os.makedirs(results_dir, exist_ok=True)

    # Save config
    config = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "families": [FAMILY_MAP[f][0] for f in selected_families],
        "n_questions": n_questions,
        "seed": int(seed),
        "injection_enabled": injection_enabled,
        "false_statement": false_statement if injection_enabled else None,
        "hallucination_type": hallucination_type if injection_enabled else None,
        "status": "running",
    }
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load questions
    tasks = sample(n_questions, seed=int(seed))
    inj_config = InjectionConfig(
        enabled=injection_enabled,
        false_statement=false_statement,
        hallucination_type=hallucination_type,
    )

    total_steps = len(selected_families) * len(tasks)
    progress = st.progress(0, text="Starting...")
    status_text = st.empty()
    step = 0

    family_metrics = {}

    for family_name in selected_families:
        fid, config_path, family_cls = FAMILY_MAP[family_name]
        family_results = []

        for i, task in enumerate(tasks):
            step += 1
            status_text.text(
                f"{fid.upper()}: Question {i + 1}/{len(tasks)} — {task.question[:50]}..."
            )
            progress.progress(step / total_steps, text=f"Step {step}/{total_steps}")

            try:
                result = inject_and_solve(family_cls, config_path, task, inj_config)
                trace_path = os.path.join(
                    results_dir, f"injected_{fid}_{task.id}.jsonl"
                )
                save_trace(result, trace_path)

                if injection_enabled:
                    metrics = compute_metrics(result, inj_config)
                    marker_name, marker_value = compute_family_marker(
                        result, fid, inj_config
                    )
                    metrics.family_marker = marker_value
                    metrics.family_marker_name = marker_name
                    family_results.append(metrics)
            except Exception as e:
                st.warning(f"{fid.upper()} Q{i + 1}: {e}")

        if family_results:
            survival_map = {
                "verbatim": 1.0,
                "paraphrased": 0.7,
                "contradicted": 0.1,
                "absent": 0.0,
            }
            n = len(family_results)
            family_metrics[fid] = {
                "n_runs": n,
                "n_errors": len(tasks) - n,
                "mean_survival": round(
                    sum(survival_map.get(m.survival_grade, 0) for m in family_results)
                    / n,
                    3,
                ),
                "mean_distance": round(
                    sum(m.propagation_distance for m in family_results) / n, 2
                ),
                "mean_amplification": round(
                    sum(m.amplification_factor or 0 for m in family_results) / n, 3
                ),
                "mean_mutation": round(
                    sum(m.mutation_rate for m in family_results) / n, 3
                ),
                "family_marker": round(
                    sum(m.family_marker or 0 for m in family_results) / n, 3
                ),
                "family_marker_name": family_results[0].family_marker_name,
            }

    # Save metrics
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(family_metrics, f, indent=2)

    config["status"] = "completed"
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    progress.progress(1.0, text="Complete!")
    st.success(f"Experiment complete! Run ID: {run_id}")

    if family_metrics:
        st.subheader("Results Summary")
        st.dataframe(format_metrics_table(family_metrics), width="stretch")

"""Pilot study: injection + metrics across all 5 families.

Usage:
    python experiments/run_pilot.py --questions 10 --seed 42
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.families.f1_react import F1React
from src.families.f2_state_graph import F2StateGraph
from src.families.f3_crew import F3Crew
from src.families.f4_group_chat import F4GroupChat
from src.families.f5_handoff_swarm import F5HandoffSwarm
from src.injection.mode1_synthetic import (
    inject_and_solve,
    load_injection_configs,
)
from src.metrics.family_markers import compute_family_marker
from src.metrics.layer2_propagation import compute_metrics
from src.tasks.hotpot_qa import sample
from src.tracing.trace import save_trace

FAMILIES = {
    "f1": ("configs/families/f1_react.yaml", F1React),
    "f2": ("configs/families/f2_state_graph.yaml", F2StateGraph),
    "f3": ("configs/families/f3_crew.yaml", F3Crew),
    "f4": ("configs/families/f4_group_chat.yaml", F4GroupChat),
    "f5": ("configs/families/f5_handoff_swarm.yaml", F5HandoffSwarm),
}


def main():
    parser = argparse.ArgumentParser(description="Pilot study")
    parser.add_argument("--questions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print(f"PILOT STUDY: {args.questions} questions, seed={args.seed}")
    print("=" * 70)

    results_dir = os.path.join(os.path.dirname(__file__), "results", "pilot")
    os.makedirs(results_dir, exist_ok=True)

    # Load questions and injection configs
    tasks = sample(args.questions, seed=args.seed)
    injection_configs = load_injection_configs("configs/injections/mode1_default.yaml")

    # Assign injection configs to questions (cycle through types)
    task_injections = []
    for i, task in enumerate(tasks):
        inj = injection_configs[i % len(injection_configs)]
        task_injections.append((task, inj))

    # Collect all metrics per family
    family_metrics = {fid: [] for fid in FAMILIES}
    family_errors = {fid: 0 for fid in FAMILIES}

    for i, (task, inj_config) in enumerate(task_injections):
        print(f"\n--- Question {i + 1}/{len(tasks)}: {task.question[:60]}...")
        print(f"    Injection: {inj_config.false_statement[:50]}...")

        for fid, (config_path, family_cls) in FAMILIES.items():
            try:
                # Injected run
                result = inject_and_solve(family_cls, config_path, task, inj_config)
                trace_path = os.path.join(
                    results_dir, f"injected_{fid}_{task.id}.jsonl"
                )
                save_trace(result, trace_path)

                # Compute metrics
                metrics = compute_metrics(result, inj_config)
                marker_name, marker_value = compute_family_marker(
                    result, fid, inj_config
                )
                metrics.family_marker = marker_value
                metrics.family_marker_name = marker_name

                family_metrics[fid].append(metrics)
                inj_success = result.metadata.get("injection", {}).get(
                    "injection_success", False
                )
                print(
                    f"  {fid}: survival={metrics.survival_grade}, "
                    f"dist={metrics.propagation_distance}, "
                    f"inj_success={inj_success}"
                )
            except Exception as e:
                family_errors[fid] += 1
                print(f"  {fid}: ERROR - {e}")
                traceback.print_exc()

    # Compute summary
    print("\n" + "=" * 70)
    print("PILOT SUMMARY")
    print("=" * 70)

    summary = {}
    survival_map = {
        "verbatim": 1.0,
        "paraphrased": 0.7,
        "contradicted": 0.1,
        "absent": 0.0,
    }

    header = f"{'Family':<8} {'Runs':<6} {'Errs':<6} {'Survival':<10} {'Distance':<10} {'Amplify':<10} {'Mutation':<10} {'Marker':<15}"
    print(header)
    print("-" * len(header))

    for fid in FAMILIES:
        ms = family_metrics[fid]
        if not ms:
            print(f"{fid:<8} 0      {family_errors[fid]:<6}")
            continue

        n = len(ms)
        mean_surv = sum(survival_map.get(m.survival_grade, 0) for m in ms) / n
        mean_dist = sum(m.propagation_distance for m in ms) / n
        amps = [
            m.amplification_factor for m in ms if m.amplification_factor is not None
        ]
        mean_amp = sum(amps) / len(amps) if amps else 0
        mean_mut = sum(m.mutation_rate for m in ms) / n
        markers = [m.family_marker for m in ms if m.family_marker is not None]
        mean_marker = sum(markers) / len(markers) if markers else 0
        marker_name = ms[0].family_marker_name or "unknown"

        summary[fid] = {
            "n_runs": n,
            "n_errors": family_errors[fid],
            "mean_survival": round(mean_surv, 3),
            "mean_distance": round(mean_dist, 2),
            "mean_amplification": round(mean_amp, 3),
            "mean_mutation": round(mean_mut, 3),
            "family_marker": round(mean_marker, 3),
            "family_marker_name": marker_name,
        }

        print(
            f"{fid:<8} {n:<6} {family_errors[fid]:<6} "
            f"{mean_surv:<10.3f} {mean_dist:<10.2f} "
            f"{mean_amp:<10.3f} {mean_mut:<10.3f} "
            f"{marker_name}={mean_marker:.3f}"
        )

    # Save summary
    summary_path = os.path.join(results_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(header + "\n")
        for fid, s in summary.items():
            f.write(
                f"{fid:<8} {s['n_runs']:<6} {s['n_errors']:<6} "
                f"{s['mean_survival']:<10.3f} {s['mean_distance']:<10.2f} "
                f"{s['mean_amplification']:<10.3f} {s['mean_mutation']:<10.3f} "
                f"{s['family_marker_name']}={s['family_marker']:.3f}\n"
            )

    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Scatter plot: Survival x Amplification
    if summary:
        fig, ax = plt.subplots(figsize=(8, 6))
        for fid, s in summary.items():
            ax.scatter(
                s["mean_survival"],
                s["mean_amplification"],
                s=100,
                label=fid.upper(),
                zorder=5,
            )
            ax.annotate(
                fid.upper(),
                (s["mean_survival"], s["mean_amplification"]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=10,
            )
        ax.set_xlabel("Mean Survival Rate")
        ax.set_ylabel("Mean Amplification Factor")
        ax.set_title("Pilot: Survival x Amplification per Family")
        ax.legend()
        ax.grid(True, alpha=0.3)
        scatter_path = os.path.join(results_dir, "scatter.png")
        fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nScatter plot saved: {scatter_path}")

    print(f"Summary saved: {summary_path}")
    print(f"Metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()

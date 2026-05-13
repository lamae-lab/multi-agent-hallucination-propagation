"""Medical applied study — cardiology hallucination propagation.

Replicates the main study protocol on RAGCare-QA cardiology MCQs
with medical-domain hallucination injections (Mode 1 only).

Usage:
    python experiments/run_medical_study.py --tasks 30 --seeds 3
    python experiments/run_medical_study.py --tasks 3 --seeds 1  # mini validation
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

if sys.platform == "win32":
    asyncio.proactor_events._ProactorBasePipeTransport.__del__ = lambda self, *a: None  # type: ignore[attr-defined]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.statistical import (
    cohens_d,
    generate_figures,
    holm_correction,
    wilcoxon_pairwise,
)
from src.families.tools import set_wiki_mode
from src.families.f1_react import F1React
from src.families.f2_state_graph import F2StateGraph
from src.families.f3_crew import F3Crew
from src.families.f4_group_chat import F4GroupChat
from src.families.f5_handoff_swarm import F5HandoffSwarm
from src.injection.mode1_synthetic import (
    inject_and_solve,
    load_injection_configs,
)
from src.metrics.cost import compute_cost_metrics
from src.metrics.dual_judge import compute_cohens_kappa, dual_judge
from src.metrics.family_markers import compute_family_marker
from src.metrics.layer1_established import (
    compute_faithfulness,
    compute_hallucination_rate,
    compute_mcq_task_success,
)
from src.metrics.layer2_propagation import compute_metrics
from src.tasks.ragcare_qa import sample_cardiology
from src.tracing.trace import save_trace

FAMILIES = {
    "f1": ("configs/families/f1_react.yaml", F1React),
    "f2": ("configs/families/f2_state_graph.yaml", F2StateGraph),
    "f3": ("configs/families/f3_crew.yaml", F3Crew),
    "f4": ("configs/families/f4_group_chat.yaml", F4GroupChat),
    "f5": ("configs/families/f5_handoff_swarm.yaml", F5HandoffSwarm),
}


def main():
    parser = argparse.ArgumentParser(description="Medical applied study (cardiology)")
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--wiki-mode", choices=["mock", "live"], default="mock")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    set_wiki_mode(args.wiki_mode)

    print("=" * 70)
    print(f"MEDICAL STUDY (Cardiology): {args.tasks} tasks/seed, {args.seeds} seeds, wiki={args.wiki_mode}")
    print("Mode 1 only — medical.yaml injections")
    print("=" * 70)

    results_dir = args.output_dir or os.path.join("experiments", "results", "medical_study")
    os.makedirs(results_dir, exist_ok=True)

    # Load medical injection configs
    mode1_configs = load_injection_configs("configs/injections/medical.yaml")

    # Collect metrics per family
    all_metrics = {fid: [] for fid in FAMILIES}
    all_faithfulness = {fid: [] for fid in FAMILIES}
    all_task_success = {fid: [] for fid in FAMILIES}
    all_dual_judge = []
    all_costs = {fid: [] for fid in FAMILIES}
    errors = {fid: 0 for fid in FAMILIES}

    total_runs = 0
    skipped = 0

    n_families = len(FAMILIES)
    estimated_total = args.seeds * args.tasks * n_families
    completed = 0
    run_times = []
    study_start = time.time()

    def _progress(run_start_time=None):
        done = completed + skipped
        if run_start_time is not None:
            run_times.append(time.time() - run_start_time)
        pct = min(done / estimated_total, 1.0) if estimated_total else 1
        bar_len = 30
        filled = min(int(bar_len * pct), bar_len)
        bar = "#" * filled + "-" * (bar_len - filled)
        elapsed = time.time() - study_start
        if completed > 0:
            avg = sum(run_times) / len(run_times) if run_times else 0
            remaining = max(0, avg * (estimated_total - done))
            mins, secs = divmod(int(remaining), 60)
            eta = f"ETA {mins}m{secs:02d}s"
        else:
            eta = "ETA --"
        elapsed_m, elapsed_s = divmod(int(elapsed), 60)
        print(
            f"\r  [{bar}] {done}/{estimated_total} ({pct:.0%}) | "
            f"{elapsed_m}m{elapsed_s:02d}s elapsed | {eta} | "
            f"{total_runs} ok, {skipped} skip, {sum(errors.values())} err",
            end="", flush=True,
        )

    for seed_idx in range(args.seeds):
        seed = 42 + seed_idx

        # Load cardiology tasks for this seed
        tasks = sample_cardiology(args.tasks, seed=seed)
        print(f"\n  Seed {seed}: {len(tasks)} cardiology tasks loaded")

        # Mode 1 only: cycle through 3 medical injection types
        for i, task in enumerate(tasks):
            inj = mode1_configs[i % len(mode1_configs)]

            for fid, (cfg_path, family_cls) in FAMILIES.items():
                trace_name = f"mode1_cardiology_{fid}_s{seed}_{task.id}.jsonl"
                trace_path = os.path.join(results_dir, trace_name)

                if os.path.exists(trace_path):
                    skipped += 1
                    _progress()
                    continue

                total_runs += 1
                run_start = time.time()
                try:
                    result = inject_and_solve(family_cls, cfg_path, task, inj)
                    save_trace(result, trace_path)

                    # Layer 2 metrics
                    l2 = compute_metrics(result, inj)
                    mk_name, mk_val = compute_family_marker(result, fid, inj)
                    l2.family_marker = mk_val
                    l2.family_marker_name = mk_name
                    all_metrics[fid].append(l2)

                    # Layer 1 metrics
                    faith = compute_faithfulness(
                        result.answer, task.expected_answer or ""
                    )
                    all_faithfulness[fid].append(faith.score)

                    # MCQ task success (letter matching)
                    success = compute_mcq_task_success(
                        result.answer,
                        task.expected_answer or "",
                        task.options or {},
                    )
                    all_task_success[fid].append(1.0 if success else 0.0)

                    # Dual judge (sample: every 5th run)
                    if total_runs % 5 == 0:
                        dj = dual_judge(result.answer, inj.false_statement)
                        all_dual_judge.append(dj)

                    # Cost
                    cost = compute_cost_metrics(result)
                    all_costs[fid].append(cost)

                    completed += 1
                    _progress(run_start)
                except Exception as e:
                    errors[fid] += 1
                    completed += 1
                    _progress(run_start)
                    print(
                        f"\n  [{total_runs}] cardiology/{fid}/s{seed}: ERROR {e}"
                    )
                    traceback.print_exc()

    # Aggregate and save
    elapsed = time.time() - study_start
    elapsed_m, elapsed_s = divmod(int(elapsed), 60)
    print(
        f"\n\nRuns: {total_runs} completed, {skipped} skipped, "
        f"{sum(errors.values())} errors | Total time: {elapsed_m}m{elapsed_s:02d}s"
    )

    survival_map = {
        "verbatim": 1.0,
        "paraphrased": 0.7,
        "contradicted": 0.1,
        "absent": 0.0,
    }
    summary = {}

    for fid in FAMILIES:
        ms = all_metrics[fid]
        if not ms:
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

        # Layer 1
        faith_scores = all_faithfulness[fid]
        mean_faith = sum(faith_scores) / len(faith_scores) if faith_scores else 0
        success_scores = all_task_success[fid]
        mean_success = (
            sum(success_scores) / len(success_scores) if success_scores else 0
        )
        hall_rate = (
            compute_hallucination_rate(faith_scores) if faith_scores else (0, 0, 0)
        )

        # Cost
        costs = all_costs[fid]
        mean_tokens = sum(c.total_tokens for c in costs) / len(costs) if costs else 0
        mean_latency = (
            sum(c.latency_seconds for c in costs) / len(costs) if costs else 0
        )

        summary[fid] = {
            "n_runs": n,
            "n_errors": errors[fid],
            "mean_survival": round(mean_surv, 3),
            "mean_distance": round(mean_dist, 2),
            "mean_amplification": round(mean_amp, 3),
            "mean_mutation": round(mean_mut, 3),
            "family_marker": round(mean_marker, 3),
            "family_marker_name": ms[0].family_marker_name if ms else "",
            "mean_faithfulness": round(mean_faith, 3),
            "mean_task_success": round(mean_success, 3),
            "hallucination_rate": round(hall_rate[0], 3),
            "hallucination_ci": [round(hall_rate[1], 3), round(hall_rate[2], 3)],
            "mean_tokens": round(mean_tokens),
            "mean_latency": round(mean_latency, 2),
        }

    # Save metrics
    with open(os.path.join(results_dir, "all_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Statistical analysis
    stat_results = {}
    survival_by_family = {
        fid: [survival_map.get(m.survival_grade, 0) for m in all_metrics[fid]]
        for fid in FAMILIES
        if all_metrics[fid]
    }
    if len(survival_by_family) >= 2:
        pairwise = wilcoxon_pairwise(survival_by_family)
        p_values = [v["p_value"] for v in pairwise.values()]
        corrected = holm_correction(p_values)
        for i, key in enumerate(pairwise):
            pairwise[key]["p_corrected"] = corrected[i]
            families_pair = key.split("_vs_")
            if len(families_pair) == 2:
                d = cohens_d(
                    survival_by_family.get(families_pair[0], []),
                    survival_by_family.get(families_pair[1], []),
                )
                pairwise[key]["cohens_d"] = round(d, 3)
        stat_results["pairwise_survival"] = pairwise

    # Dual-judge kappa
    if all_dual_judge:
        kappa = compute_cohens_kappa(all_dual_judge)
        stat_results["dual_judge_kappa"] = kappa
        stat_results["dual_judge_n"] = len(all_dual_judge)
        stat_results["dual_judge_agreement"] = sum(
            1 for d in all_dual_judge if d.agree
        ) / len(all_dual_judge)

    with open(os.path.join(results_dir, "statistical_analysis.json"), "w") as f:
        json.dump(stat_results, f, indent=2)

    # Generate figures
    figures_dir = os.path.join(results_dir, "figures")
    generate_figures(summary, figures_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("MEDICAL STUDY SUMMARY (Cardiology)")
    print("=" * 70)
    for fid, s in summary.items():
        print(
            f"  {fid.upper()}: surv={s['mean_survival']:.3f} dist={s['mean_distance']:.2f} "
            f"faith={s['mean_faithfulness']:.3f} success={s['mean_task_success']:.3f} "
            f"hall_rate={s['hallucination_rate']:.3f}"
        )

    if "dual_judge_kappa" in stat_results:
        print(
            f"\n  Dual-judge kappa: {stat_results['dual_judge_kappa']:.4f} "
            f"(n={stat_results['dual_judge_n']})"
        )

    print(f"\n  Results: {results_dir}")
    print(f"  Figures: {figures_dir}")


if __name__ == "__main__":
    main()

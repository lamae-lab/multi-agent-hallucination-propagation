"""Full main study orchestrator.

Usage:
    python experiments/run_main_study.py --tasks 30 --seeds 3
    python experiments/run_main_study.py --tasks 3 --seeds 1  # mini validation
"""

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
import warnings
import urllib3
from pathlib import Path

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

# Suppress "Event loop is closed" noise on Windows
if sys.platform == "win32":
    asyncio.proactor_events._ProactorBasePipeTransport.__del__ = lambda self, *a: None  # type: ignore[attr-defined]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.base import Task
from src.analysis.statistical import (
    bootstrap_ci,
    cohens_d,
    generate_figures,
    holm_correction,
    wilcoxon_pairwise,
)
from src.families.f1_react import F1React
from src.families.f2_state_graph import F2StateGraph
from src.families.f3_crew import F3Crew
from src.families.f4_group_chat import F4GroupChat
from src.families.f5_handoff_swarm import F5HandoffSwarm
from src.injection.mode1_synthetic import (
    InjectionConfig,
    inject_and_solve,
    load_injection_configs,
)
from src.injection.mode2_halueval import inject_mode2
from src.metrics.cost import compute_cost_metrics
from src.metrics.dual_judge import compute_cohens_kappa, dual_judge
from src.metrics.family_markers import compute_family_marker
from src.metrics.layer1_established import (
    compute_faithfulness,
    compute_hallucination_rate,
    compute_task_success,
)
from src.metrics.layer2_propagation import compute_metrics
from src.tasks.gsm8k import sample_gsm8k
from src.tasks.halueval import load_halueval_qa
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
    parser = argparse.ArgumentParser(description="Full main study")
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    print("=" * 70)
    print(f"MAIN STUDY: {args.tasks} tasks/condition, {args.seeds} seeds")
    print("=" * 70)

    results_dir = os.path.join("experiments", "results", "main_study")
    os.makedirs(results_dir, exist_ok=True)

    # Load injection configs
    mode1_configs = load_injection_configs("configs/injections/mode1_default.yaml")

    # Collect all metrics per family
    all_metrics = {fid: [] for fid in FAMILIES}
    all_faithfulness = {fid: [] for fid in FAMILIES}
    all_task_success = {fid: [] for fid in FAMILIES}
    all_dual_judge = []
    all_costs = {fid: [] for fid in FAMILIES}
    errors = {fid: 0 for fid in FAMILIES}

    total_runs = 0
    skipped = 0

    # Estimate total runs for progress tracking
    n_families = len(FAMILIES)
    n_datasets = 2  # hotpotqa + gsm8k
    n_halueval = min(args.tasks, 10)
    # Mode 1: datasets × tasks × families + Mode 2: datasets × halueval × families
    estimated_total = args.seeds * n_datasets * (args.tasks * n_families + n_halueval * n_families)
    completed = 0
    run_times = []
    study_start = time.time()

    def _progress(run_start_time=None):
        """Print progress bar with ETA."""
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

        # Dataset 1: HotpotQA
        hotpot_tasks = sample(args.tasks, seed=seed)
        # Dataset 2: GSM8K
        try:
            gsm8k_tasks = sample_gsm8k(args.tasks, seed=seed)
        except Exception as e:
            print(f"  GSM8K load failed: {e}. Skipping GSM8K.")
            gsm8k_tasks = []

        datasets = [("hotpotqa", hotpot_tasks)]
        if gsm8k_tasks:
            datasets.append(("gsm8k", gsm8k_tasks))

        for ds_name, tasks in datasets:
            # Mode 1: Synthetic injection (cycle through 3 types)
            for i, task in enumerate(tasks):
                inj = mode1_configs[i % len(mode1_configs)]
                for fid, (cfg_path, family_cls) in FAMILIES.items():
                    trace_name = f"mode1_{ds_name}_{fid}_s{seed}_{task.id}.jsonl"
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
                        success = compute_task_success(
                            result.answer, task.expected_answer or ""
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
                            f"\n  [{total_runs}] {ds_name}/{fid}/s{seed}: ERROR {e}"
                        )
                        traceback.print_exc()

            # Mode 2: HaluEval injection (if HaluEval available)
            try:
                halueval_examples = load_halueval_qa(min(args.tasks, 10), seed=seed)
            except Exception as e:
                print(f"  HaluEval load failed: {e}. Skipping Mode 2.")
                halueval_examples = []

            for he in halueval_examples:
                task = Task(
                    question=he["question"],
                    expected_answer=he["correct_answer"],
                    id=he["id"],
                )
                for fid, (cfg_path, family_cls) in FAMILIES.items():
                    trace_name = f"mode2_{fid}_s{seed}_{he['id']}.jsonl"
                    trace_path = os.path.join(results_dir, trace_name)

                    if os.path.exists(trace_path):
                        skipped += 1
                        _progress()
                        continue

                    total_runs += 1
                    run_start = time.time()
                    try:
                        result = inject_mode2(family_cls, cfg_path, task, he)
                        save_trace(result, trace_path)

                        inj_cfg = InjectionConfig(
                            enabled=True,
                            false_statement=he["hallucinated_answer"],
                            hallucination_type="halueval_qa",
                        )
                        l2 = compute_metrics(result, inj_cfg)
                        all_metrics[fid].append(l2)
                        completed += 1
                        _progress(run_start)
                    except Exception as e:
                        errors[fid] += 1
                        completed += 1
                        _progress(run_start)
                        print(
                            f"\n  [{total_runs}] mode2/{fid}/s{seed}: ERROR {e}"
                        )
                        traceback.print_exc()

    # Aggregate and save
    elapsed = time.time() - study_start
    elapsed_m, elapsed_s = divmod(int(elapsed), 60)
    print(
        f"\n\nRuns: {total_runs} completed, {skipped} skipped, {sum(errors.values())} errors"
        f" | Total time: {elapsed_m}m{elapsed_s:02d}s"
    )

    # Build summary
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
    print("MAIN STUDY SUMMARY")
    print("=" * 70)
    for fid, s in summary.items():
        print(
            f"  {fid.upper()}: surv={s['mean_survival']:.3f} dist={s['mean_distance']:.2f} "
            f"faith={s['mean_faithfulness']:.3f} success={s['mean_task_success']:.3f} "
            f"hall_rate={s['hallucination_rate']:.3f}"
        )

    if "dual_judge_kappa" in stat_results:
        print(
            f"\n  Dual-judge kappa: {stat_results['dual_judge_kappa']:.4f} (n={stat_results['dual_judge_n']})"
        )

    print(f"\n  Results: {results_dir}")
    print(f"  Figures: {figures_dir}")


if __name__ == "__main__":
    main()

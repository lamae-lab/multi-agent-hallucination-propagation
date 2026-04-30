"""Paired comparison harness: run all 5 families on the same questions.

Usage:
    python experiments/run_paired_comparison.py
    python experiments/run_paired_comparison.py --questions 10 --seed 123
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.families import F1React, F2StateGraph, F3Crew, F4GroupChat, F5HandoffSwarm
from src.tasks.hotpot_qa import sample
from src.tracing.trace import save_trace

FAMILY_CONFIGS = {
    "f1": ("configs/families/f1_react.yaml", F1React),
    "f2": ("configs/families/f2_state_graph.yaml", F2StateGraph),
    "f3": ("configs/families/f3_crew.yaml", F3Crew),
    "f4": ("configs/families/f4_group_chat.yaml", F4GroupChat),
    "f5": ("configs/families/f5_handoff_swarm.yaml", F5HandoffSwarm),
}


def main():
    parser = argparse.ArgumentParser(description="Paired comparison harness")
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print(f"PAIRED COMPARISON: {args.questions} questions, seed={args.seed}")
    print("=" * 70)

    # Load questions
    print(f"\nLoading {args.questions} HotpotQA questions...")
    tasks = sample(args.questions, seed=args.seed)
    print(f"Loaded {len(tasks)} questions.\n")

    # Initialize families
    families = {}
    for fid, (config_path, cls) in FAMILY_CONFIGS.items():
        try:
            families[fid] = cls(config_path)
            print(f"  {fid}: {families[fid].name} - OK")
        except Exception as e:
            print(f"  {fid}: FAILED to initialize - {e}")

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Run paired comparisons
    summary = {fid: {"answered": 0, "errors": 0} for fid in families}

    for i, task in enumerate(tasks):
        print(f"\n--- Question {i + 1}/{len(tasks)}: {task.question[:80]}...")

        for fid, family in families.items():
            try:
                result = family.solve(task)
                trace_path = os.path.join(results_dir, f"{fid}_{task.id}.jsonl")
                save_trace(result, trace_path)
                summary[fid]["answered"] += 1
                print(f"  {fid}: {result.answer[:60]}...")
            except Exception as e:
                summary[fid]["errors"] += 1
                print(f"  {fid}: ERROR - {e}")
                traceback.print_exc()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Family':<8} {'Answered':<10} {'Errors':<8} {'Status'}")
    print("-" * 40)
    for fid in families:
        s = summary[fid]
        status = "OK" if s["errors"] == 0 else "PARTIAL"
        print(f"{fid:<8} {s['answered']:<10} {s['errors']:<8} {status}")

    total_traces = sum(s["answered"] for s in summary.values())
    print(f"\nTotal traces: {total_traces}/{len(tasks) * len(families)}")
    print(f"Results dir: {results_dir}")


if __name__ == "__main__":
    main()

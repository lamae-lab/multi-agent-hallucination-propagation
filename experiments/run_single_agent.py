"""End-to-end validation: single agent on 1 HotpotQA question.

Validates the full pipeline: LLM client → BaseAgent → tracing → dataset.
Produces a JSONL trace file in experiments/results/.

Usage:
    python experiments/run_single_agent.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.base import BaseAgent
from src.tasks.hotpot_qa import sample
from src.tracing.trace import load_trace, save_trace

# Simple mock search tool (same as spike tests)
WIKI_PARAGRAPHS = {
    "scott derrickson": "Scott Derrickson (born July 16, 1966) is an American filmmaker.",
    "ed wood": "Edward Davis Wood Jr. was an American filmmaker and actor.",
    "steven spielberg": "Steven Allan Spielberg is an American filmmaker.",
    "jaws": "Jaws is a 1975 American thriller film directed by Steven Spielberg.",
}


def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information about a topic."""
    key = query.lower().strip()
    for entity, paragraph in WIKI_PARAGRAPHS.items():
        if entity in key or key in entity:
            return paragraph
    return f"No Wikipedia article found for '{query}'."


def main():
    print("=" * 60)
    print("END-TO-END VALIDATION: Single Agent on HotpotQA")
    print("=" * 60)

    # Load 1 HotpotQA question
    print("\n1. Loading HotpotQA question...")
    tasks = sample(1, seed=42)
    task = tasks[0]
    print(f"   Question: {task.question}")
    print(f"   Expected: {task.expected_answer}")
    print(f"   ID: {task.id}")

    # Create agent with search tool
    print("\n2. Creating BaseAgent with search tool...")
    agent = BaseAgent(
        system_prompt=(
            "You are a research assistant. Use the search_wikipedia tool "
            "to find facts before answering. Give a short, direct answer."
        ),
        tools=[search_wikipedia],
    )

    # Solve the task
    print("\n3. Running agent...")
    result = agent.solve(task)
    print(f"   Answer: {result.answer[:200]}")
    print(f"   Iterations: {result.metadata['iterations']}")
    print(f"   Duration: {result.metadata['duration_seconds']}s")
    print(f"   Truncated: {result.metadata['truncated']}")

    # Count tool calls
    tool_call_count = sum(len(m.tool_calls) for m in result.messages if m.tool_calls)
    print(f"   Tool calls: {tool_call_count}")

    # Save trace
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    trace_path = os.path.join(results_dir, f"trace_{task.id}.jsonl")
    print(f"\n4. Saving trace to {trace_path}...")
    save_trace(result, trace_path)

    # Verify trace can be loaded back
    print("\n5. Verifying trace round-trip...")
    loaded = load_trace(trace_path)
    assert loaded.answer == result.answer, "Answer mismatch!"
    assert len(loaded.messages) == len(result.messages), "Message count mismatch!"
    print("   Round-trip verification PASSED")

    print("\n" + "=" * 60)
    print("VALIDATION PASSED")
    print(f"  Question: {task.question[:80]}...")
    print(f"  Answer: {result.answer[:80]}")
    print(f"  Tool calls: {tool_call_count}")
    print(f"  Trace: {trace_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

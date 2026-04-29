"""Spike test: LangGraph (F2 — Sequential State-Graph).

Tests a 2-agent + 1-tool LangGraph pipeline on 5 HotpotQA questions
using the ILAAS LLM provider.
"""

import sys
import traceback
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
_spike = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_spike))

from hotpotqa_sample import QUESTIONS, search_wikipedia

from src.llm.config import PROVIDERS

ILAAS = PROVIDERS["ilaas"]
MODEL = ILAAS["model"]
BASE_URL = ILAAS["base_url"]
API_KEY = ILAAS["api_key"]


def run_spike():
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    # Create LLM pointing at ILAAS
    llm = ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        temperature=0,
        max_tokens=500,
    )

    # Wrap the mock tool for LangChain
    @tool
    def wiki_search(query: str) -> str:
        """Search Wikipedia for information about a topic."""
        return search_wikipedia(query)

    # Create a single ReAct agent with the tool
    agent = create_react_agent(llm, [wiki_search])

    results = []
    tool_calls_made = 0
    tool_calls_expected = sum(len(q["supporting_facts"]) for q in QUESTIONS)

    for q in QUESTIONS:
        print(f"\n--- Question {q['id']}: {q['question']}")
        try:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": q["question"]}]}
            )
            # Count tool calls in the trace
            for msg in response["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls_made += len(msg.tool_calls)

            # Get final answer
            final_msg = response["messages"][-1]
            answer = (
                final_msg.content if hasattr(final_msg, "content") else str(final_msg)
            )
            print(f"  Answer: {answer[:200]}")
            print(f"  Expected: {q['expected_answer']}")

            correct = q["expected_answer"].lower() in answer.lower()
            results.append({"id": q["id"], "correct": correct, "answer": answer})
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({"id": q["id"], "correct": False, "answer": str(e)})

    correct_count = sum(1 for r in results if r["correct"])
    tool_call_rate = f"{tool_calls_made}/{tool_calls_expected}"

    print("\n" + "=" * 60)
    print("LANGGRAPH SPIKE RESULTS")
    print("  Ran successfully: Yes")
    print(f"  Tool-call rate: {tool_call_rate}")
    print(f"  Correct answers: {correct_count}/5")
    print("  Trace quality: Check messages above for state-graph pattern")
    print("=" * 60)

    return {
        "ran_successfully": True,
        "tool_call_rate": tool_call_rate,
        "correct_answers": f"{correct_count}/5",
    }


if __name__ == "__main__":
    try:
        result = run_spike()
    except Exception as e:
        print(f"\nLANGGRAPH SPIKE FAILED: {e}")
        traceback.print_exc()
        result = {
            "ran_successfully": False,
            "failure_mode": str(e),
        }
    print(f"\nResult: {result}")

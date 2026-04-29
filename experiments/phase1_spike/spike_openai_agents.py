"""Spike test: OpenAI Agents SDK (F5 — Autonomous Handoff Swarm).

Tests the OpenAI Agents SDK with handoff patterns on 5 HotpotQA questions
using the ILAAS LLM provider.
"""

import asyncio
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
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
    from openai import AsyncOpenAI

    from agents import (
        Agent,
        Runner,
        function_tool,
        set_tracing_disabled,
    )

    set_tracing_disabled(True)

    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
    model = OpenAIChatCompletionsModel(model=MODEL, openai_client=client)

    @function_tool
    def wiki_search(query: str) -> str:
        """Search Wikipedia for information about a topic."""
        return search_wikipedia(query)

    # Research agent with the tool
    research_agent = Agent(
        name="research_agent",
        instructions="You are a researcher. Use the wiki_search tool to look up facts on Wikipedia. Always search for relevant entities before answering.",
        tools=[wiki_search],
        model=model,
    )

    # Analysis agent
    analysis_agent = Agent(
        name="analysis_agent",
        instructions="You are an analyst. Synthesize the information provided into a short, direct answer to the original question.",
        model=model,
    )

    # Triage agent with handoffs
    triage_agent = Agent(
        name="triage_agent",
        instructions="You receive questions. Route factual questions to research_agent for lookup. Route synthesis requests to analysis_agent.",
        handoffs=[research_agent, analysis_agent],
        model=model,
    )

    results = []

    async def run_question(q):
        print(f"\n--- Question {q['id']}: {q['question']}")
        try:
            result = await Runner.run(
                triage_agent,
                f"Answer this question: {q['question']}",
            )
            answer = result.final_output
            print(f"  Answer: {answer[:200]}")
            print(f"  Expected: {q['expected_answer']}")
            print(f"  Last agent: {result.last_agent.name}")

            correct = q["expected_answer"].lower() in answer.lower()
            return {"id": q["id"], "correct": correct, "answer": answer}
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            return {"id": q["id"], "correct": False, "answer": str(e)}

    async def run_all():
        for q in QUESTIONS:
            r = await run_question(q)
            results.append(r)

    asyncio.run(run_all())

    correct_count = sum(1 for r in results if r["correct"])

    print("\n" + "=" * 60)
    print("OPENAI AGENTS SDK SPIKE RESULTS")
    print("  Ran successfully: Yes")
    print(f"  Correct answers: {correct_count}/5")
    print("=" * 60)

    return {
        "ran_successfully": True,
        "correct_answers": f"{correct_count}/5",
    }


if __name__ == "__main__":
    try:
        result = run_spike()
    except Exception as e:
        print(f"\nOPENAI AGENTS SDK SPIKE FAILED: {e}")
        traceback.print_exc()
        result = {
            "ran_successfully": False,
            "failure_mode": str(e),
        }
    print(f"\nResult: {result}")

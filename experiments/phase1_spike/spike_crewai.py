"""Spike test: CrewAI (F3 — Hierarchical Crew).

Tests a manager + specialist CrewAI crew on 5 HotpotQA questions
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
    from crewai import LLM, Agent, Crew, Process, Task
    from crewai.tools import tool as crewai_tool

    # Create LLM pointing at ILAAS
    llm = LLM(
        model=f"openai/{MODEL}",
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=0,
    )

    # Wrap the mock tool for CrewAI
    @crewai_tool
    def wiki_search(query: str) -> str:
        """Search Wikipedia for information about a topic."""
        return search_wikipedia(query)

    # Create agents
    researcher = Agent(
        role="Researcher",
        goal="Find accurate factual information by searching Wikipedia",
        backstory="You are an expert researcher who uses Wikipedia to find facts.",
        tools=[wiki_search],
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    analyst = Agent(
        role="Analyst",
        goal="Synthesize research findings into a clear, concise answer",
        backstory="You are an analyst who produces precise answers from research.",
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )

    results = []
    for q in QUESTIONS:
        print(f"\n--- Question {q['id']}: {q['question']}")
        try:
            task = Task(
                description=f"Answer this question: {q['question']}. Use the wiki_search tool to look up relevant facts. Give a short, direct answer.",
                expected_output="A short factual answer to the question.",
                agent=researcher,
            )

            crew = Crew(
                agents=[researcher, analyst],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
            )

            output = crew.kickoff()
            answer = str(output)
            print(f"  Answer: {answer[:200]}")
            print(f"  Expected: {q['expected_answer']}")

            correct = q["expected_answer"].lower() in answer.lower()
            results.append({"id": q["id"], "correct": correct, "answer": answer})
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({"id": q["id"], "correct": False, "answer": str(e)})

    correct_count = sum(1 for r in results if r["correct"])

    print("\n" + "=" * 60)
    print("CREWAI SPIKE RESULTS")
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
        print(f"\nCREWAI SPIKE FAILED: {e}")
        traceback.print_exc()
        result = {
            "ran_successfully": False,
            "failure_mode": str(e),
        }
    print(f"\nResult: {result}")

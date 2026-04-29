"""Spike test: AutoGen/AG2 (F4 — Conversational Group-Chat).

Tests an AutoGen AG2 v0.7 RoundRobinGroupChat with multiple agents
on 5 HotpotQA questions using the ILAAS LLM provider.
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
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    model_client = OpenAIChatCompletionClient(
        model=MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "unknown",
            "structured_output": False,
        },
    )

    # Create agents
    researcher = AssistantAgent(
        name="researcher",
        system_message="You are a researcher. Use the wiki_search tool to look up facts on Wikipedia. Always search for relevant entities before answering.",
        model_client=model_client,
        tools=[search_wikipedia],
    )

    analyst = AssistantAgent(
        name="analyst",
        system_message="You are an analyst. When the researcher provides facts, synthesize them into a short, direct answer to the original question. End your message with TERMINATE when done.",
        model_client=model_client,
    )

    results = []

    async def run_question(q):
        print(f"\n--- Question {q['id']}: {q['question']}")
        try:
            termination = MaxMessageTermination(max_messages=6)
            team = RoundRobinGroupChat(
                [researcher, analyst], termination_condition=termination
            )
            task_result = await team.run(task=f"Answer this question: {q['question']}")

            # Get final answer from last non-empty message
            answer = ""
            for msg in reversed(task_result.messages):
                content = getattr(msg, "content", "")
                if isinstance(content, str) and content.strip():
                    answer = content
                    break

            print(f"  Answer: {answer[:200]}")
            print(f"  Expected: {q['expected_answer']}")

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
    print("AUTOGEN SPIKE RESULTS")
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
        print(f"\nAUTOGEN SPIKE FAILED: {e}")
        traceback.print_exc()
        result = {
            "ran_successfully": False,
            "failure_mode": str(e),
        }
    print(f"\nResult: {result}")

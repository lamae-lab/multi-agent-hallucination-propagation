"""F5 — Autonomous Handoff Swarm via OpenAI Agents SDK.

Agents hand off control to each other with summarized context.
No central orchestrator. Studies chain-of-trust hallucination propagation.
"""

import asyncio
import time

from src.agents.base import Message, Result, Task
from src.families.base_family import BaseFamily
from src.families.tools import search_wikipedia
from src.llm import config


class F5HandoffSwarm(BaseFamily):
    """Autonomous Handoff Swarm family using OpenAI Agents SDK."""

    def solve(self, task: Task) -> Result:
        from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
        from openai import AsyncOpenAI

        from agents import Agent, Runner, function_tool, set_tracing_disabled

        set_tracing_disabled(True)

        provider = config.PROVIDERS[config.ACTIVE_PROVIDER]
        client = AsyncOpenAI(
            base_url=provider["base_url"],
            api_key=provider["api_key"],
        )
        model = OpenAIChatCompletionsModel(
            model=provider["model"],
            openai_client=client,
        )

        @function_tool
        def wiki_search(query: str) -> str:
            """Search Wikipedia for information about a topic."""
            return search_wikipedia(query)

        # Create agents from config
        research_agent = Agent(
            name="research_agent",
            instructions=self._get_agent_prompt("research_agent"),
            tools=[wiki_search],
            model=model,
        )

        analysis_agent = Agent(
            name="analysis_agent",
            instructions=self._get_agent_prompt("analysis_agent"),
            model=model,
        )

        triage_agent = Agent(
            name="triage",
            instructions=self._get_agent_prompt("triage"),
            handoffs=[research_agent, analysis_agent],
            model=model,
        )

        start_time = time.time()

        async def _run():
            result = await Runner.run(
                triage_agent,
                f"Answer this question: {task.question}",
                max_turns=15,
            )
            await client.close()
            return result

        from agents.exceptions import MaxTurnsExceeded

        truncated = False
        try:
            sdk_result = asyncio.run(_run())
        except MaxTurnsExceeded:
            sdk_result = None
            truncated = True
        duration = time.time() - start_time

        # Convert SDK output to our format
        messages = [
            Message(role="system", content="Handoff swarm session"),
            Message(role="user", content=task.question),
        ]

        # Extract info from SDK result
        if sdk_result is not None:
            answer = sdk_result.final_output or ""
            last_agent = sdk_result.last_agent.name if sdk_result.last_agent else "unknown"
        else:
            answer = ""
            last_agent = "truncated"

        messages.append(
            Message(
                role="assistant",
                content=answer,
                name=last_agent,
            )
        )

        metadata = {
            "model": provider["model"],
            "provider": config.ACTIVE_PROVIDER,
            "seed": None,
            "temperature": 0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_seconds": round(duration, 2),
            "truncated": truncated,
            "family": self.family_id,
            "last_agent": last_agent,
        }

        return Result(answer=answer, messages=messages, metadata=metadata, task=task)

    def _get_agent_prompt(self, agent_name: str) -> str:
        for agent_cfg in self.config["agents"]:
            if agent_cfg["name"] == agent_name:
                return agent_cfg["system_prompt"]
        return f"You are {agent_name}."

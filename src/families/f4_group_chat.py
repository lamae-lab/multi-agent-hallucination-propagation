"""F4 — Conversational Group-Chat via AutoGen AG2.

Agents share a conversation history in a RoundRobinGroupChat.
Studies how consensus-building behavior affects hallucination propagation.
"""

import asyncio
import time

from src.agents.base import Message, Result, Task
from src.families.base_family import BaseFamily
from src.families.tools import search_wikipedia
from src.llm import config

TOOL_MAP = {
    "search_wikipedia": search_wikipedia,
}


class F4GroupChat(BaseFamily):
    """Conversational Group-Chat family using AutoGen AG2."""

    def solve(self, task: Task) -> Result:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.conditions import MaxMessageTermination
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        provider = config.PROVIDERS[config.ACTIVE_PROVIDER]

        model_client = OpenAIChatCompletionClient(
            model=provider["model"],
            base_url=provider["base_url"],
            api_key=provider["api_key"],
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": False,
                "family": "unknown",
                "structured_output": False,
            },
        )

        # Create agents from config
        ag_agents = []
        for agent_cfg in self.config["agents"]:
            tools = [TOOL_MAP[t] for t in agent_cfg.get("tools", [])]
            ag = AssistantAgent(
                name=agent_cfg["name"],
                system_message=agent_cfg["system_prompt"],
                model_client=model_client,
                tools=tools if tools else None,
            )
            ag_agents.append(ag)

        max_rounds = self.config.get("max_rounds", 8)
        termination = MaxMessageTermination(max_messages=max_rounds)

        start_time = time.time()

        async def _run():
            team = RoundRobinGroupChat(ag_agents, termination_condition=termination)
            return await team.run(task=f"Answer this question: {task.question}")

        task_result = asyncio.run(_run())
        duration = time.time() - start_time

        # Convert AutoGen messages to our format
        messages = [
            Message(role="system", content="Group chat session"),
            Message(role="user", content=task.question),
        ]

        for msg in task_result.messages:
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            source = getattr(msg, "source", "assistant")

            messages.append(
                Message(
                    role="assistant",
                    content=str(content) if content else None,
                    name=source,
                )
            )

        # Extract final answer (strip TERMINATE marker)
        answer = ""
        for msg in reversed(messages):
            if msg.role == "assistant" and msg.content and msg.content.strip():
                clean = msg.content.replace("TERMINATE", "").strip()
                if clean:
                    answer = clean
                    break

        metadata = {
            "model": provider["model"],
            "provider": config.ACTIVE_PROVIDER,
            "seed": None,
            "temperature": 0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_seconds": round(duration, 2),
            "truncated": False,
            "family": self.family_id,
        }

        return Result(answer=answer, messages=messages, metadata=metadata, task=task)

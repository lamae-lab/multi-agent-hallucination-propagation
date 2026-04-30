"""F3 — Hierarchical Crew (reimplemented).

Manager agent delegates to 2 specialists (researcher, analyst) via
tool-call delegation. The manager sees delegation as a tool choice,
enabling study of manager bias in hallucination propagation.
"""

import json
import time

from src.agents.base import BaseAgent, Message, Result, Task
from src.families.base_family import BaseFamily
from src.families.tools import search_wikipedia
from src.llm import config
from src.llm.client import chat_raw

TOOL_MAP = {
    "search_wikipedia": search_wikipedia,
}


class F3Crew(BaseFamily):
    """Hierarchical Crew family with manager + 2 specialists."""

    def solve(self, task: Task) -> Result:
        start_time = time.time()

        # Load agent configs
        agents_cfg = {a["name"]: a for a in self.config["agents"]}
        manager_cfg = next(a for a in self.config["agents"] if a.get("is_manager"))
        researcher_cfg = agents_cfg["researcher"]
        analyst_cfg = agents_cfg["analyst"]

        # Build specialist agents
        researcher_tools = [TOOL_MAP[t] for t in researcher_cfg.get("tools", [])]
        researcher_agent = BaseAgent(
            system_prompt=researcher_cfg["system_prompt"],
            tools=researcher_tools,
            max_iterations=self.config.get("max_iterations", 10),
        )

        analyst_agent = BaseAgent(
            system_prompt=analyst_cfg["system_prompt"],
            tools=[],
            max_iterations=self.config.get("max_iterations", 10),
        )

        # Build delegation tools for the manager
        delegation_tools = [
            {
                "type": "function",
                "function": {
                    "name": "delegate_to_researcher",
                    "description": "Delegate a factual lookup task to the researcher who can search Wikipedia.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sub_task": {
                                "type": "string",
                                "description": "The specific question or lookup task for the researcher.",
                            }
                        },
                        "required": ["sub_task"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "delegate_to_analyst",
                    "description": "Delegate a synthesis or analysis task to the analyst.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sub_task": {
                                "type": "string",
                                "description": "The specific analysis or synthesis task for the analyst.",
                            }
                        },
                        "required": ["sub_task"],
                    },
                },
            },
        ]

        # Run manager loop
        messages = [
            Message(role="system", content=manager_cfg["system_prompt"]),
            Message(role="user", content=task.question),
        ]
        max_iterations = self.config.get("max_iterations", 10)
        truncated = False
        answer = ""

        for _iteration in range(max_iterations):
            raw_messages = [
                {k: v for k, v in m.to_dict().items() if v is not None}
                for m in messages
            ]

            response_msg = chat_raw(
                raw_messages,
                tools=delegation_tools,
                temperature=0,
            )

            tool_calls = response_msg.get("tool_calls")
            messages.append(
                Message(
                    role="assistant",
                    content=response_msg.get("content"),
                    tool_calls=tool_calls,
                )
            )

            if not tool_calls:
                answer = response_msg.get("content", "")
                break

            # Execute delegations
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, KeyError):
                    args = {}

                sub_task_text = args.get("sub_task", task.question)
                sub_task = Task(question=sub_task_text)

                if func_name == "delegate_to_researcher":
                    specialist_result = researcher_agent.solve(sub_task)
                    tool_result = f"[Researcher]: {specialist_result.answer}"
                elif func_name == "delegate_to_analyst":
                    specialist_result = analyst_agent.solve(sub_task)
                    tool_result = f"[Analyst]: {specialist_result.answer}"
                else:
                    tool_result = f"Unknown delegation target: {func_name}"

                messages.append(
                    Message(
                        role="tool",
                        content=tool_result,
                        tool_call_id=tc.get("id", ""),
                        name=func_name,
                    )
                )
        else:
            truncated = True
            if messages and messages[-1].role == "assistant":
                answer = messages[-1].content or ""

        duration = time.time() - start_time
        metadata = {
            "model": config.PROVIDERS[config.ACTIVE_PROVIDER]["model"],
            "provider": config.ACTIVE_PROVIDER,
            "seed": None,
            "temperature": 0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_seconds": round(duration, 2),
            "truncated": truncated,
            "family": self.family_id,
        }

        return Result(answer=answer, messages=messages, metadata=metadata, task=task)

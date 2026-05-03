"""F2 — Sequential State-Graph via LangGraph.

Agents communicate through shared state managed by a supervisor.
Uses LangGraph's create_react_agent + create_supervisor.
"""

import time

from src.agents.base import Message, Result, Task
from src.families.base_family import BaseFamily
from src.families.tools import search_wikipedia
from src.llm import config

TOOL_MAP = {
    "search_wikipedia": search_wikipedia,
}


class F2StateGraph(BaseFamily):
    """Sequential State-Graph family using LangGraph."""

    def solve(self, task: Task) -> Result:
        from langchain_core.tools import tool as langchain_tool
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent
        from langgraph_supervisor import create_supervisor

        provider = config.PROVIDERS[config.ACTIVE_PROVIDER]

        llm = ChatOpenAI(
            base_url=provider["base_url"],
            api_key=provider["api_key"],
            model=provider["model"],
            temperature=0,
            max_tokens=500,
            timeout=60,
            max_retries=0,
        )

        # Wrap tools for LangChain
        @langchain_tool
        def wiki_search(query: str) -> str:
            """Search Wikipedia for information about a topic."""
            return search_wikipedia(query)

        # Create agents from config
        agents = []
        for agent_cfg in self.config["agents"]:
            tools = (
                [wiki_search]
                if "search_wikipedia" in agent_cfg.get("tools", [])
                else []
            )
            agent = create_react_agent(
                model=llm,
                tools=tools,
                name=agent_cfg["name"],
                prompt=agent_cfg["system_prompt"],
            )
            agents.append(agent)

        # Create supervisor
        supervisor_prompt = self.config.get(
            "supervisor_prompt", "Route tasks to the appropriate agent."
        )
        workflow = create_supervisor(
            agents,
            model=llm,
            prompt=supervisor_prompt,
        )
        app = workflow.compile()

        start_time = time.time()
        response = app.invoke(
            {"messages": [{"role": "user", "content": task.question}]}
        )
        duration = time.time() - start_time

        # Convert LangGraph messages to our format
        messages = [Message(role="system", content=supervisor_prompt)]
        messages.append(Message(role="user", content=task.question))

        for msg in response.get("messages", [])[1:]:  # skip the user message
            role = getattr(msg, "type", "assistant")
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"

            content = getattr(msg, "content", "")
            tool_calls_raw = getattr(msg, "tool_calls", None)
            tool_calls = None
            if tool_calls_raw:
                tool_calls = [
                    {
                        "id": getattr(tc, "id", tc.get("id", "")),
                        "type": "function",
                        "function": {
                            "name": getattr(tc, "name", tc.get("name", "")),
                            "arguments": str(getattr(tc, "args", tc.get("args", {}))),
                        },
                    }
                    for tc in tool_calls_raw
                ]

            name = getattr(msg, "name", None)
            tool_call_id = getattr(msg, "tool_call_id", None)

            messages.append(
                Message(
                    role=(
                        role
                        if role in ("user", "assistant", "tool", "system")
                        else "assistant"
                    ),
                    content=str(content) if content else None,
                    tool_calls=tool_calls,
                    name=name,
                    tool_call_id=tool_call_id,
                )
            )

        # Extract final answer
        answer = ""
        for msg in reversed(messages):
            if msg.role == "assistant" and msg.content and not msg.tool_calls:
                answer = msg.content
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

"""F1 — Single-Agent ReAct.

The simplest family: wraps BaseAgent in the family interface.
A single agent in a Think-Act-Observe loop.
Serves as the practitioner baseline.
"""

from src.agents.base import BaseAgent, Result, Task
from src.families.base_family import BaseFamily
from src.families.tools import search_wikipedia

TOOL_MAP = {
    "search_wikipedia": search_wikipedia,
}


class F1React(BaseFamily):
    """Single-Agent ReAct family."""

    def solve(self, task: Task) -> Result:
        agent_config = self.config["agents"][0]
        tools = [TOOL_MAP[t] for t in agent_config.get("tools", [])]

        agent = BaseAgent(
            system_prompt=agent_config["system_prompt"],
            tools=tools,
            max_iterations=self.config.get("max_iterations", 10),
        )

        result = agent.solve(task)
        result.metadata["family"] = self.family_id
        return result

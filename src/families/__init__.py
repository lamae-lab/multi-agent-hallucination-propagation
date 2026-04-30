"""Architectural family implementations."""

from src.families.base_family import BaseFamily
from src.families.f1_react import F1React
from src.families.f2_state_graph import F2StateGraph
from src.families.f3_crew import F3Crew
from src.families.f4_group_chat import F4GroupChat
from src.families.f5_handoff_swarm import F5HandoffSwarm

__all__ = [
    "BaseFamily",
    "F1React",
    "F2StateGraph",
    "F3Crew",
    "F4GroupChat",
    "F5HandoffSwarm",
]

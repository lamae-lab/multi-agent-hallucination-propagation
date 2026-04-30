"""Tests for all 5 architectural families — using live ILAAS LLM."""

from src.agents.base import Task
from src.families.f1_react import F1React
from src.families.f2_state_graph import F2StateGraph
from src.families.f3_crew import F3Crew
from src.families.f4_group_chat import F4GroupChat
from src.families.f5_handoff_swarm import F5HandoffSwarm


def test_f1_react():
    f1 = F1React("configs/families/f1_react.yaml")
    result = f1.solve(Task(question="What is the capital of France?"))
    assert result.answer is not None
    assert len(result.answer) > 0
    assert result.metadata["family"] == "f1"
    assert len(result.messages) >= 3


def test_f2_state_graph():
    f2 = F2StateGraph("configs/families/f2_state_graph.yaml")
    result = f2.solve(Task(question="What is the capital of France?"))
    assert result.answer is not None
    assert len(result.answer) > 0
    assert result.metadata["family"] == "f2"
    assert len(result.messages) >= 3


def test_f3_crew():
    f3 = F3Crew("configs/families/f3_crew.yaml")
    result = f3.solve(Task(question="What is the capital of France?"))
    assert result.answer is not None
    assert len(result.answer) > 0
    assert result.metadata["family"] == "f3"
    # Should have delegation
    assert len(result.messages) >= 4


def test_f4_group_chat():
    f4 = F4GroupChat("configs/families/f4_group_chat.yaml")
    result = f4.solve(Task(question="What is the capital of France?"))
    assert result.answer is not None
    assert len(result.answer) > 0
    assert result.metadata["family"] == "f4"


def test_f5_handoff_swarm():
    f5 = F5HandoffSwarm("configs/families/f5_handoff_swarm.yaml")
    result = f5.solve(Task(question="What is the capital of France?"))
    assert result.answer is not None
    assert len(result.answer) > 0
    assert result.metadata["family"] == "f5"

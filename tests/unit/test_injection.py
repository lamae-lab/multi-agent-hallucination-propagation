"""Tests for hallucination injection — using live ILAAS."""

from src.agents.base import Task
from src.families.f1_react import F1React
from src.injection.mode1_synthetic import InjectionConfig, inject_and_solve


def test_injection_produces_false_statement():
    config = InjectionConfig(
        enabled=True,
        false_statement="The capital of France is Berlin",
        hallucination_type="memorization",
    )
    result = inject_and_solve(
        F1React,
        "configs/families/f1_react.yaml",
        Task(question="What is the capital of France?"),
        config,
    )
    assert result.answer is not None
    assert "injection" in result.metadata
    # Injection should succeed on this simple case
    assert result.metadata["injection"]["injection_success"] is True


def test_injection_disabled_produces_normal_answer():
    config = InjectionConfig(
        enabled=False,
        false_statement="The capital of France is Berlin",
        hallucination_type="memorization",
    )
    result = inject_and_solve(
        F1React,
        "configs/families/f1_react.yaml",
        Task(question="What is the capital of France?"),
        config,
    )
    assert result.answer is not None
    # Should not contain the false statement
    assert "berlin" not in result.answer.lower()

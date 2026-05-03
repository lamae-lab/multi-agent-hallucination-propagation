"""Tests for Layer 1 established metrics — live ILAAS."""

from src.agents.base import Task
from src.families.f1_react import F1React
from src.metrics.layer1_established import (
    compute_faithfulness,
    compute_hallucination_rate,
    compute_task_success,
)


def test_compute_faithfulness_returns_score():
    f1 = F1React("configs/families/f1_react.yaml")
    result = f1.solve(Task(question="What is the capital of France?"))
    faith = compute_faithfulness(result.answer, "Paris")
    assert 0 <= faith.score <= 1
    assert faith.n_total >= 1
    assert len(faith.claims) >= 1


def test_compute_task_success():
    assert compute_task_success("The capital of France is Paris.", "Paris") is True
    assert compute_task_success("The capital is Berlin.", "Paris") is False
    assert compute_task_success("", "Paris") is False


def test_compute_hallucination_rate():
    rate, lo, hi = compute_hallucination_rate([0.9, 0.8, 0.3, 0.1])
    assert 0 <= rate <= 1
    assert lo <= rate
    assert rate <= hi

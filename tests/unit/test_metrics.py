"""Tests for Layer 2 propagation metrics — using live ILAAS."""

from src.agents.base import Task
from src.families.f1_react import F1React
from src.injection.mode1_synthetic import InjectionConfig, inject_and_solve
from src.metrics.family_markers import compute_family_marker
from src.metrics.layer2_propagation import compute_metrics


def test_compute_metrics_produces_valid_values():
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
    metrics = compute_metrics(result, config)

    assert metrics.survival_grade in (
        "verbatim",
        "paraphrased",
        "contradicted",
        "absent",
    )
    assert 0 <= metrics.survival_semantic <= 1
    assert metrics.propagation_distance >= 0
    assert metrics.mutation_rate >= 0


def test_family_marker_returns_name_and_value():
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
    name, value = compute_family_marker(result, "f1", config)

    assert name == "self_validation_rate"
    assert 0 <= value <= 1

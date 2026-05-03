"""Tests for GSM8K loader."""

from src.tasks.gsm8k import _parse_answer, sample_gsm8k


def test_parse_answer():
    assert _parse_answer("Some work\n#### 42") == "42"
    assert _parse_answer("#### 1234") == "1234"
    assert _parse_answer("#### -5") == "-5"


def test_sample_gsm8k_deterministic():
    tasks1 = sample_gsm8k(5, seed=42)
    tasks2 = sample_gsm8k(5, seed=42)
    assert len(tasks1) == 5
    assert [t.id for t in tasks1] == [t.id for t in tasks2]


def test_sample_gsm8k_has_numerical_answer():
    tasks = sample_gsm8k(3, seed=42)
    for task in tasks:
        # Answer should be a number string
        assert task.expected_answer is not None
        assert len(task.expected_answer) > 0

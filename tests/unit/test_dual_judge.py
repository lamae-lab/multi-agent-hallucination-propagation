"""Tests for dual-judge inter-rater reliability — live ILAAS."""

from src.metrics.dual_judge import (
    DualJudgeResult,
    JudgeVerdict,
    compute_cohens_kappa,
    dual_judge,
)


def test_dual_judge_returns_two_verdicts():
    result = dual_judge(
        "The capital of France is Berlin.",
        "The capital of France is Berlin",
    )
    assert result.judge1.verdict in (
        "verbatim",
        "paraphrased",
        "contradicted",
        "absent",
    )
    assert result.judge2.verdict in (
        "verbatim",
        "paraphrased",
        "contradicted",
        "absent",
    )
    assert result.judge1.judge_model == "qwen-3.6-35b-instruct"
    assert result.judge2.judge_model == "mistral-small-3.2-24b"


def test_compute_cohens_kappa_perfect_agreement():
    results = [
        DualJudgeResult(
            judge1=JudgeVerdict("m1", "verbatim"),
            judge2=JudgeVerdict("m2", "verbatim"),
            agree=True,
        ),
        DualJudgeResult(
            judge1=JudgeVerdict("m1", "absent"),
            judge2=JudgeVerdict("m2", "absent"),
            agree=True,
        ),
    ]
    kappa = compute_cohens_kappa(results)
    assert kappa >= 0.5  # Should be high for perfect agreement


def test_compute_cohens_kappa_no_agreement():
    results = [
        DualJudgeResult(
            judge1=JudgeVerdict("m1", "verbatim"),
            judge2=JudgeVerdict("m2", "absent"),
            agree=False,
        ),
        DualJudgeResult(
            judge1=JudgeVerdict("m1", "absent"),
            judge2=JudgeVerdict("m2", "verbatim"),
            agree=False,
        ),
    ]
    kappa = compute_cohens_kappa(results)
    assert kappa <= 0  # Should be low/negative for no agreement

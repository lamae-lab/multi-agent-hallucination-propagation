"""Dual-Judge Inter-Rater Reliability.

Run two LLM judge models on the same outputs and compute Cohen's kappa.
"""

from dataclasses import dataclass

from src.llm import config
from src.metrics.detection import detect_llm_judge


@dataclass
class JudgeVerdict:
    """One judge's verdict."""

    judge_model: str
    verdict: str  # verbatim/paraphrased/contradicted/absent


@dataclass
class DualJudgeResult:
    """Agreement between two judges."""

    judge1: JudgeVerdict
    judge2: JudgeVerdict
    agree: bool


def dual_judge(
    answer: str,
    false_statement: str,
    model1: str = "qwen-3.6-35b-instruct",
    model2: str = "mistral-small-3.2-24b",
) -> DualJudgeResult:
    """Run two judge models on the same output.

    Temporarily switches the ILAAS model for each judge call.
    """
    original_model = config.PROVIDERS["ilaas"]["model"]

    try:
        # Judge 1
        config.PROVIDERS["ilaas"]["model"] = model1
        verdict1 = detect_llm_judge(answer, false_statement)

        # Judge 2
        config.PROVIDERS["ilaas"]["model"] = model2
        verdict2 = detect_llm_judge(answer, false_statement)
    finally:
        config.PROVIDERS["ilaas"]["model"] = original_model

    j1 = JudgeVerdict(judge_model=model1, verdict=verdict1)
    j2 = JudgeVerdict(judge_model=model2, verdict=verdict2)

    return DualJudgeResult(judge1=j1, judge2=j2, agree=(verdict1 == verdict2))


def compute_cohens_kappa(results: list[DualJudgeResult]) -> float:
    """Compute Cohen's kappa from dual-judge results.

    Maps verdicts to categories and computes agreement.
    """
    if not results:
        return 0.0

    categories = {"verbatim": 0, "paraphrased": 1, "contradicted": 2, "absent": 3}
    n_categories = len(categories)

    ratings1 = [categories.get(r.judge1.verdict, 3) for r in results]
    ratings2 = [categories.get(r.judge2.verdict, 3) for r in results]
    n = len(results)

    if n == 0:
        return 0.0

    # Observed agreement
    po = sum(1 for r1, r2 in zip(ratings1, ratings2) if r1 == r2) / n

    # Expected agreement by chance
    pe = 0.0
    for k in range(n_categories):
        p1k = sum(1 for r in ratings1 if r == k) / n
        p2k = sum(1 for r in ratings2 if r == k) / n
        pe += p1k * p2k

    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0

    kappa = (po - pe) / (1 - pe)
    return round(kappa, 4)

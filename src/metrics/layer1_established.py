"""Layer 1 — Established Metrics (reviewer credibility).

Faithfulness (RAGAS-style), Answer Correctness, Hallucination Rate,
FActScore-style atomic-claim decomposition.
"""

from dataclasses import dataclass

import numpy as np

from src.llm.client import chat


@dataclass
class AtomicClaim:
    """A single factual claim extracted from an answer."""

    text: str
    supported: bool
    evidence: str = ""


@dataclass
class FaithfulnessResult:
    """Faithfulness metric output."""

    claims: list[AtomicClaim]
    score: float
    n_supported: int
    n_total: int

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "n_supported": self.n_supported,
            "n_total": self.n_total,
            "claims": [
                {"text": c.text, "supported": c.supported, "evidence": c.evidence}
                for c in self.claims
            ],
        }


def compute_faithfulness(answer: str, ground_truth: str) -> FaithfulnessResult:
    """Decompose answer into atomic claims and verify each against ground truth.

    Returns FaithfulnessResult with per-claim support and overall score.
    """
    if not answer or not answer.strip():
        return FaithfulnessResult(claims=[], score=0.0, n_supported=0, n_total=0)

    # Step 1: Decompose into atomic claims
    decompose_prompt = (
        "Decompose the following text into individual atomic factual claims. "
        "Each claim should be a single, verifiable statement.\n\n"
        f'Text: "{answer[:500]}"\n\n'
        "List each claim on a separate line, numbered (1., 2., 3., etc.). "
        "Only output the claims, nothing else."
    )
    try:
        claims_text = chat(
            [{"role": "user", "content": decompose_prompt}], max_tokens=300
        )
        claim_lines = [
            line.strip().lstrip("0123456789.-) ").strip()
            for line in claims_text.strip().split("\n")
            if line.strip() and any(c.isalpha() for c in line)
        ]
    except Exception:
        claim_lines = [answer[:200]]

    if not claim_lines:
        claim_lines = [answer[:200]]

    # Step 2: Verify all claims against ground truth in a single LLM call
    capped_claims = claim_lines[:10]
    claims_list = "\n".join(f"{i+1}. {c}" for i, c in enumerate(capped_claims))
    verify_prompt = (
        f'Ground truth: "{ground_truth}"\n\n'
        f"Claims to verify:\n{claims_list}\n\n"
        "For each claim, is it supported by the ground truth? "
        "Reply with ONLY a comma-separated list of YES or NO, one per claim. "
        "Example for 3 claims: YES,NO,YES"
    )
    try:
        verdict_text = chat([{"role": "user", "content": verify_prompt}], max_tokens=200)
        verdicts = [v.strip().lower() for v in verdict_text.strip().split(",")]
    except Exception:
        verdicts = []

    claims = []
    for i, claim_text in enumerate(capped_claims):
        supported = i < len(verdicts) and "yes" in verdicts[i]
        claims.append(AtomicClaim(text=claim_text, supported=supported))

    n_supported = sum(1 for c in claims if c.supported)
    n_total = len(claims)
    score = n_supported / n_total if n_total > 0 else 0.0

    return FaithfulnessResult(
        claims=claims, score=score, n_supported=n_supported, n_total=n_total
    )


def compute_task_success(answer: str, expected_answer: str) -> bool:
    """Check if the expected answer appears in the output."""
    if not answer or not expected_answer:
        return False
    return expected_answer.lower().strip() in answer.lower()


def compute_hallucination_rate(
    faithfulness_scores: list[float], threshold: float = 0.5
) -> tuple[float, float, float]:
    """Compute hallucination rate with 95% bootstrap CI.

    Returns (rate, ci_lower, ci_upper).
    """
    if not faithfulness_scores:
        return (0.0, 0.0, 0.0)

    scores = np.array(faithfulness_scores)
    unfaithful = scores < threshold
    rate = float(np.mean(unfaithful))

    # Bootstrap CI
    rng = np.random.default_rng(42)
    n_boot = 1000
    boot_rates = []
    for _ in range(n_boot):
        sample = rng.choice(unfaithful, size=len(unfaithful), replace=True)
        boot_rates.append(float(np.mean(sample)))

    ci_lower = float(np.percentile(boot_rates, 2.5))
    ci_upper = float(np.percentile(boot_rates, 97.5))

    return (rate, ci_lower, ci_upper)

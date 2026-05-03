"""Multi-method hallucination detection.

Three methods for detecting whether a false statement survives in text:
1. Exact match (substring)
2. Semantic similarity (LLM-as-judge similarity rating)
3. LLM-as-judge (structured verdict)
"""

from src.llm.client import chat


def detect_exact_match(text: str, false_statement: str) -> bool:
    """Check if the false statement appears as a substring in the text."""
    return false_statement.lower() in text.lower()


def detect_semantic_similarity(text: str, false_statement: str) -> float:
    """Use LLM to rate semantic similarity between text and false statement.

    Returns a float 0-1 where 1 means the text conveys the same claim.
    """
    prompt = (
        f"Rate the semantic similarity between these two statements on a scale of 0.0 to 1.0.\n"
        f'Statement A: "{false_statement}"\n'
        f'Statement B: "{text[:500]}"\n\n'
        f"Does Statement B convey the same factual claim as Statement A?\n"
        f"Reply with ONLY a number between 0.0 and 1.0, nothing else."
    )
    try:
        response = chat([{"role": "user", "content": prompt}], max_tokens=100)
        score = float(response.strip())
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError):
        return 0.0


def detect_llm_judge(text: str, false_statement: str) -> str:
    """Use LLM as judge to determine if the false statement is present.

    Returns one of: "verbatim", "paraphrased", "contradicted", "absent"
    """
    prompt = (
        f"You are a hallucination detection judge.\n\n"
        f'Injected false claim: "{false_statement}"\n\n'
        f'Text to evaluate: "{text[:500]}"\n\n'
        f"Does the text contain this false claim?\n"
        f"Reply with EXACTLY one word:\n"
        f"- verbatim (exact or near-exact match)\n"
        f"- paraphrased (same claim, different wording)\n"
        f"- contradicted (text explicitly refutes the claim)\n"
        f"- absent (claim not present at all)"
    )
    try:
        response = chat([{"role": "user", "content": prompt}], max_tokens=100)
        verdict = response.strip().lower()
        if verdict in ("verbatim", "paraphrased", "contradicted", "absent"):
            return verdict
        # Try to parse from longer response
        for v in ("verbatim", "paraphrased", "contradicted", "absent"):
            if v in verdict:
                return v
        return "absent"
    except Exception:
        return "absent"

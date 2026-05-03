"""Cost metrics — tokens, latency, estimated cost."""

from dataclasses import dataclass

from src.agents.base import Result


@dataclass
class CostMetrics:
    """Per-run cost tracking."""

    total_tokens: int
    latency_seconds: float
    estimated_cost: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "latency_seconds": self.latency_seconds,
            "estimated_cost": self.estimated_cost,
        }


def estimate_tokens(messages: list) -> int:
    """Estimate token count from messages (1 word ~ 1.3 tokens)."""
    total_words = 0
    for msg in messages:
        content = getattr(msg, "content", None) or ""
        total_words += len(content.split())
    return int(total_words * 1.3)


def compute_cost_metrics(
    result: Result, cost_per_1k_tokens: float = 0.0
) -> CostMetrics:
    """Compute cost metrics from a Result."""
    tokens = estimate_tokens(result.messages)
    latency = result.metadata.get("duration_seconds", 0.0)
    cost = (tokens / 1000) * cost_per_1k_tokens

    return CostMetrics(
        total_tokens=tokens,
        latency_seconds=latency,
        estimated_cost=cost,
    )

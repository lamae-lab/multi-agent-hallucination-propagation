"""Mode 2 — HaluEval-Sourced Injection.

Uses validated hallucinations from HaluEval as the forced output.
Same wrapper pattern as Mode 1, but with ecologically valid hallucinations.
"""

from src.agents.base import Result, Task
from src.injection.mode1_synthetic import InjectionConfig, inject_and_solve


def inject_mode2(
    family_cls,
    config_path: str,
    task: Task,
    halueval_entry: dict,
) -> Result:
    """Inject a HaluEval hallucinated answer into a family run.

    Uses the same inject_and_solve wrapper as Mode 1, with the
    HaluEval hallucinated answer as the false statement.
    """
    config = InjectionConfig(
        enabled=True,
        false_statement=halueval_entry["hallucinated_answer"],
        hallucination_type="halueval_qa",
    )
    return inject_and_solve(family_cls, config_path, task, config)

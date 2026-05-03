"""GSM8K dataset loader with deterministic sampling.

Loads math reasoning problems from HuggingFace.
"""

import random
import re

from datasets import load_dataset

from src.agents.base import Task

_DATASET_CACHE = {}


def load_gsm8k(split: str = "test") -> list:
    """Load the GSM8K dataset."""
    cache_key = f"gsm8k_{split}"
    if cache_key not in _DATASET_CACHE:
        _DATASET_CACHE[cache_key] = load_dataset("gsm8k", "main", split=split)
    return _DATASET_CACHE[cache_key]


def _parse_answer(answer_text: str) -> str:
    """Extract the numerical answer from GSM8K format '#### N'."""
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    # Fallback: try to find the last number
    numbers = re.findall(r"-?\d+\.?\d*", answer_text)
    return numbers[-1] if numbers else answer_text.strip()


def sample_gsm8k(n: int, seed: int, split: str = "test") -> list[Task]:
    """Return a deterministic sample of N GSM8K problems as Tasks."""
    dataset = load_gsm8k(split)
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(n, len(dataset)))

    tasks = []
    for idx in indices:
        entry = dataset[idx]
        answer = _parse_answer(entry["answer"])
        tasks.append(
            Task(
                question=entry["question"],
                expected_answer=answer,
                id=f"gsm8k_{idx}",
            )
        )
    return tasks

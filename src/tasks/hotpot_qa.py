"""HotpotQA dataset loader with deterministic sampling.

Loads from HuggingFace datasets with local caching.
The distractor setting is used per the project plan.
"""

import random

from datasets import load_dataset

from src.agents.base import Task

_DATASET_CACHE = {}


def load_hotpotqa(split: str = "validation") -> list:
    """Load the HotpotQA dataset (distractor setting).

    Args:
        split: Dataset split to load ("train" or "validation").

    Returns:
        The HuggingFace dataset object for the specified split.
    """
    cache_key = f"hotpot_qa_distractor_{split}"
    if cache_key not in _DATASET_CACHE:
        _DATASET_CACHE[cache_key] = load_dataset("hotpot_qa", "distractor", split=split)
    return _DATASET_CACHE[cache_key]


def sample(n: int, seed: int, split: str = "validation") -> list[Task]:
    """Return a deterministic sample of N HotpotQA questions as Tasks.

    Args:
        n: Number of questions to sample.
        seed: Random seed for deterministic sampling.
        split: Dataset split to sample from.

    Returns:
        List of Task objects with question, expected_answer, and id.
    """
    dataset = load_hotpotqa(split)
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(n, len(dataset)))

    tasks = []
    for idx in indices:
        entry = dataset[idx]
        tasks.append(
            Task(
                question=entry["question"],
                expected_answer=entry["answer"],
                id=entry["id"],
            )
        )
    return tasks

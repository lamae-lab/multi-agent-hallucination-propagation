"""HaluEval dataset loader.

Loads validated hallucinations from HaluEval QA split.
"""

import random

from datasets import load_dataset

_DATASET_CACHE = {}


def load_halueval_qa(n: int, seed: int) -> list[dict]:
    """Load N HaluEval QA examples with deterministic sampling.

    Returns list of dicts with: question, correct_answer, hallucinated_answer, id.
    """
    cache_key = "halueval_qa"
    if cache_key not in _DATASET_CACHE:
        _DATASET_CACHE[cache_key] = load_dataset(
            "pminervini/HaluEval", "qa_samples", split="data"
        )
    dataset = _DATASET_CACHE[cache_key]

    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(n, len(dataset)))

    examples = []
    for idx in indices:
        entry = dataset[idx]
        examples.append(
            {
                "question": entry.get("question", ""),
                "correct_answer": entry.get("right_answer", entry.get("answer", "")),
                "hallucinated_answer": entry.get(
                    "hallucinated_answer", entry.get("wrong_answer", "")
                ),
                "id": f"halueval_{idx}",
            }
        )
    return examples

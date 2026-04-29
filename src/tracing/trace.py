"""JSONL trace serialization and loading.

Each trace file is a JSONL file where:
- Line 1: metadata (model, provider, seed, temperature, timestamp, task, answer)
- Lines 2+: one Message per line (role, content, tool_calls, etc.)
"""

import json
import os

from src.agents.base import Message, Result, Task


def save_trace(result: Result, path: str) -> None:
    """Save a Result as a JSONL trace file.

    Args:
        result: The agent run result to serialize.
        path: File path to write the JSONL trace to.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        # Line 1: metadata
        metadata_line = {
            "type": "metadata",
            "model": result.metadata.get("model", ""),
            "provider": result.metadata.get("provider", ""),
            "seed": result.metadata.get("seed"),
            "temperature": result.metadata.get("temperature", 0),
            "timestamp": result.metadata.get("timestamp", ""),
            "duration_seconds": result.metadata.get("duration_seconds", 0),
            "truncated": result.metadata.get("truncated", False),
            "iterations": result.metadata.get("iterations", 0),
            "task": result.task.question,
            "task_id": result.task.id,
            "expected_answer": result.task.expected_answer,
            "answer": result.answer,
        }
        f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")

        # Lines 2+: messages
        for msg in result.messages:
            msg_line = {"type": "message"}
            msg_line.update(msg.to_dict())
            f.write(json.dumps(msg_line, ensure_ascii=False) + "\n")


def load_trace(path: str) -> Result:
    """Load a JSONL trace file back into a Result object.

    Args:
        path: File path to the JSONL trace.

    Returns:
        A Result object with answer, messages, metadata, and task.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Empty trace file: {path}")

    # First line is metadata
    meta = lines[0]
    if meta.get("type") != "metadata":
        raise ValueError(f"First line of trace must be metadata, got: {meta}")

    # Remaining lines are messages
    messages = []
    for line in lines[1:]:
        if line.get("type") == "message":
            line_copy = {k: v for k, v in line.items() if k != "type"}
            messages.append(Message.from_dict(line_copy))

    task = Task(
        question=meta.get("task", ""),
        id=meta.get("task_id"),
        expected_answer=meta.get("expected_answer"),
    )

    metadata = {
        "model": meta.get("model", ""),
        "provider": meta.get("provider", ""),
        "seed": meta.get("seed"),
        "temperature": meta.get("temperature", 0),
        "timestamp": meta.get("timestamp", ""),
        "duration_seconds": meta.get("duration_seconds", 0),
        "truncated": meta.get("truncated", False),
        "iterations": meta.get("iterations", 0),
    }

    return Result(
        answer=meta.get("answer", ""),
        messages=messages,
        metadata=metadata,
        task=task,
    )

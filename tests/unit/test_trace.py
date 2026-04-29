"""Tests for JSONL trace serialization."""

import json
import os
import tempfile

from src.agents.base import Message, Result, Task
from src.tracing.trace import load_trace, save_trace


def _make_result():
    task = Task(question="What is 2+2?", id="test-1", expected_answer="4")
    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="What is 2+2?"),
        Message(
            role="assistant",
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calc", "arguments": '{"x": 2}'},
                }
            ],
        ),
        Message(role="tool", content="4", tool_call_id="call_1", name="calc"),
        Message(role="assistant", content="The answer is 4."),
    ]
    metadata = {
        "model": "test-model",
        "provider": "test",
        "seed": 42,
        "temperature": 0,
        "timestamp": "2026-04-29T12:00:00",
        "duration_seconds": 1.5,
        "truncated": False,
        "iterations": 2,
    }
    return Result(
        answer="The answer is 4.", messages=messages, metadata=metadata, task=task
    )


def test_save_trace_creates_file():
    result = _make_result()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "trace.jsonl")
        save_trace(result, path)
        assert os.path.exists(path)


def test_save_trace_jsonl_format():
    result = _make_result()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "trace.jsonl")
        save_trace(result, path)
        with open(path) as f:
            lines = f.readlines()
        # First line is metadata + 5 messages = 6 lines
        assert len(lines) == 6
        for line in lines:
            json.loads(line)  # each line must be valid JSON


def test_save_trace_metadata_fields():
    result = _make_result()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "trace.jsonl")
        save_trace(result, path)
        with open(path) as f:
            meta = json.loads(f.readline())
        assert meta["type"] == "metadata"
        assert meta["model"] == "test-model"
        assert meta["provider"] == "test"
        assert meta["seed"] == 42
        assert meta["answer"] == "The answer is 4."
        assert meta["task"] == "What is 2+2?"


def test_round_trip():
    result = _make_result()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "trace.jsonl")
        save_trace(result, path)
        loaded = load_trace(path)

    assert loaded.answer == result.answer
    assert len(loaded.messages) == len(result.messages)
    assert loaded.task.question == result.task.question
    assert loaded.task.id == result.task.id
    assert loaded.metadata["model"] == result.metadata["model"]
    assert loaded.metadata["seed"] == result.metadata["seed"]


def test_round_trip_preserves_tool_calls():
    result = _make_result()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "trace.jsonl")
        save_trace(result, path)
        loaded = load_trace(path)

    # Check assistant message with tool_calls
    assistant_with_tools = [m for m in loaded.messages if m.tool_calls]
    assert len(assistant_with_tools) == 1
    assert assistant_with_tools[0].tool_calls[0]["function"]["name"] == "calc"

    # Check tool response
    tool_msgs = [m for m in loaded.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "call_1"
    assert tool_msgs[0].name == "calc"

"""Base Agent with ReAct loop and tool-use support.

All architectural families extend or compose BaseAgent to implement
their specific communication patterns. The uniform interface is:
    solve(task: Task) -> Result
"""

import inspect
import json
import time
from dataclasses import dataclass

from src.llm import config
from src.llm.client import chat_raw


@dataclass
class Task:
    """Input to an agent run."""

    question: str
    context: str | None = None
    id: str | None = None
    expected_answer: str | None = None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "context": self.context,
            "id": self.id,
            "expected_answer": self.expected_answer,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(
            question=d["question"],
            context=d.get("context"),
            id=d.get("id"),
            expected_answer=d.get("expected_answer"),
        )


@dataclass
class Message:
    """A single event in an agent conversation."""

    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict:
        d = {"role": self.role, "content": self.content}
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        return cls(
            role=d["role"],
            content=d.get("content"),
            tool_calls=d.get("tool_calls"),
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
        )


@dataclass
class Result:
    """Output of an agent run."""

    answer: str
    messages: list[Message]
    metadata: dict
    task: Task

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "messages": [m.to_dict() for m in self.messages],
            "metadata": self.metadata,
            "task": self.task.to_dict(),
        }


def tool_to_openai_schema(func: callable) -> dict:
    """Convert a typed Python function to an OpenAI tool definition.

    The function must have type annotations and a docstring.
    """
    sig = inspect.signature(func)
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        param_type = "string"
        if param.annotation is int:
            param_type = "integer"
        elif param.annotation is float:
            param_type = "number"
        elif param.annotation is bool:
            param_type = "boolean"

        properties[param_name] = {
            "type": param_type,
            "description": param_name,
        }
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class BaseAgent:
    """A ReAct agent with tool-use support.

    Implements the solve(task) -> Result interface used by all families.
    """

    def __init__(
        self,
        system_prompt: str,
        tools: list[callable] | None = None,
        model: str | None = None,
        temperature: float = 0,
        seed: int | None = None,
        max_iterations: int = 10,
    ):
        self.system_prompt = system_prompt
        self.tools = {func.__name__: func for func in (tools or [])}
        self.tool_schemas = [tool_to_openai_schema(f) for f in (tools or [])]
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_iterations = max_iterations

    def solve(self, task: Task) -> Result:
        """Run the ReAct loop on a task and return the result."""
        start_time = time.time()
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=task.question),
        ]

        truncated = False
        answer = ""

        for _iteration in range(self.max_iterations):
            # Build raw message list for the API
            raw_messages = [m.to_dict() for m in messages]
            # Remove keys with None values for clean API calls
            raw_messages = [
                {k: v for k, v in m.items() if v is not None} for m in raw_messages
            ]

            response_msg = chat_raw(
                raw_messages,
                tools=self.tool_schemas if self.tool_schemas else None,
                model=self.model,
                temperature=self.temperature,
            )

            # Record assistant message
            tool_calls = response_msg.get("tool_calls")
            assistant_msg = Message(
                role="assistant",
                content=response_msg.get("content"),
                tool_calls=tool_calls,
            )
            messages.append(assistant_msg)

            # If no tool calls, we have the final answer
            if not tool_calls:
                answer = response_msg.get("content", "")
                break

            # Execute each tool call
            for tool_call in tool_calls:
                func_name = tool_call["function"]["name"]
                try:
                    args = json.loads(tool_call["function"]["arguments"])
                except (json.JSONDecodeError, KeyError):
                    args = {}

                tool_func = self.tools.get(func_name)
                if tool_func is None:
                    tool_result = f"Error: Unknown tool '{func_name}'"
                else:
                    try:
                        tool_result = str(tool_func(**args))
                    except Exception as e:
                        tool_result = f"Error calling {func_name}: {e}"

                messages.append(
                    Message(
                        role="tool",
                        content=tool_result,
                        tool_call_id=tool_call.get("id", ""),
                        name=func_name,
                    )
                )
        else:
            # Max iterations reached
            truncated = True
            if messages and messages[-1].role == "assistant":
                answer = messages[-1].content or ""
            else:
                answer = "(max iterations reached, no final answer)"

        duration = time.time() - start_time
        metadata = {
            "model": self.model or config.PROVIDERS[config.ACTIVE_PROVIDER]["model"],
            "provider": config.ACTIVE_PROVIDER,
            "seed": self.seed,
            "temperature": self.temperature,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_seconds": round(duration, 2),
            "truncated": truncated,
            "iterations": _iteration + 1 if not truncated else self.max_iterations,
        }

        return Result(
            answer=answer,
            messages=messages,
            metadata=metadata,
            task=task,
        )

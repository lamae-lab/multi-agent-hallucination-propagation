"""Agent framework with uniform solve(task) -> Result interface."""

from src.agents.base import BaseAgent, Message, Result, Task

__all__ = ["BaseAgent", "Task", "Message", "Result"]

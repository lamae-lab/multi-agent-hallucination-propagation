"""Abstract base class for all architectural families.

All 5 families extend BaseFamily and implement solve(task) -> Result.
"""

from abc import ABC, abstractmethod

import yaml

from src.agents.base import Result, Task


class BaseFamily(ABC):
    """Base class for architectural family implementations."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.name = self.config.get("name", "unknown")
        self.family_id = self.config.get("family_id", "unknown")

    def _load_config(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @abstractmethod
    def solve(self, task: Task) -> Result:
        """Process a task through this family's architectural pattern."""
        ...

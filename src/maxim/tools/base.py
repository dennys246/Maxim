from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolResult:
    success: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Tool(ABC):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = {}

    def __init__(self) -> None:
        if not getattr(self, "name", ""):
            raise ValueError("Tool must define a non-empty name")

    def run(self, **kwargs: Any) -> ToolResult:
        try:
            self._validate_input(kwargs)
            output = self.execute(**kwargs)
            if isinstance(output, ToolResult):
                return output
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Perform the side effect."""
        raise NotImplementedError

    def _validate_input(self, kwargs: dict[str, Any]) -> None:
        schema = getattr(self, "input_schema", None)
        if not isinstance(schema, dict):
            return

        for key, spec in schema.items():
            optional = isinstance(spec, tuple) and len(spec) >= 2
            if key not in kwargs and not optional:
                raise ValueError(f"Missing required input: {key}")

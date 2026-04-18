from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from maxim.agents.bus import ToolErrorKind


@dataclass(slots=True, frozen=True)
class ToolOutput:
    """Raw output from a single tool execution.

    Internal to the tools layer. The agent loop converts this to a bus
    ToolResult (agents.bus.ToolResult) before publishing, adding
    tool_call_id, tool_name, and params for downstream subscribers.

    ``side_effects`` is a typed channel for bio-pipeline signals the
    executor / bridge layer branches on. It is separate from ``metadata``
    (caller-facing extras) and ``output`` (the main result). The
    ``tools/`` layer itself stays agnostic of these signals — the shape
    is a plain ``dict[str, Any]`` keyed by well-known strings, and
    consumers (bridges, bio-systems) know the keys they care about.

    Well-known ``side_effects`` keys (append-only, document new ones here):

    - ``"embodiment_failures"``: ``list[dict]`` of SEM failure event dicts
      with shape ``{"name": failure_mode, "entity": entity_path,
      "pain": intensity}``. Populated by
      ``ModulatorAffordanceTool.execute`` when
      ``embodiment.evaluate_failures()`` fires post-action. Consumed by
      ``runtime/executor.py`` which routes to
      ``ToolPainBridge.record_tool_embodiment_failure`` for direct NAc
      attribution. A tool that succeeded at its action but produced
      embodiment failures still returns ``success=True`` — the tool did
      what was asked; the side-effect reports what the body felt.
    """

    success: bool
    output: Any = None
    error: str | None = None
    error_kind: ToolErrorKind | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    side_effects: dict[str, Any] | None = None


# Backward-compat alias — existing tools that import ToolResult keep working.
ToolResult = ToolOutput


class Tool(ABC):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = {}
    timeout: float = 30.0  # per-tool timeout declaration (seconds)

    def __init__(self) -> None:
        if not getattr(self, "name", ""):
            raise ValueError("Tool must define a non-empty name")

    def run(self, **kwargs: Any) -> ToolOutput:
        try:
            self._validate_input(kwargs)
            output = self.execute(**kwargs)
            if isinstance(output, ToolOutput):
                return output
            return ToolOutput(success=True, output=output)
        except Exception as e:
            return ToolOutput(success=False, error=str(e))

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Perform the side effect."""
        raise NotImplementedError

    def _validate_input(self, kwargs: dict[str, Any]) -> None:
        schema = getattr(self, "input_schema", None)
        if not isinstance(schema, dict):
            return

        # JSON Schema format: {"type": "object", "properties": {...}, "required": [...]}
        if "type" in schema and "properties" in schema:
            required = set(schema.get("required", []))
            for key in required:
                if key not in kwargs:
                    raise ValueError(f"Missing required input: {key}")
            return

        # Flat format: {"param_name": spec, ...}
        for key, spec in schema.items():
            optional = isinstance(spec, tuple) and len(spec) >= 2
            if key not in kwargs and not optional:
                raise ValueError(f"Missing required input: {key}")

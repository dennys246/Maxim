from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolErrorKind(Enum):
    """Classification of tool execution errors."""

    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_DENIED = "permission_denied"
    SYNTAX_ERROR = "syntax_error"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    EXTERNAL_FAILURE = "external_failure"
    VALIDATION = "validation"


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

    The append-only registry of well-known ``side_effects`` keys lives
    in ``docs/user/tool_side_effects.md``. That page is authoritative:
    it lists every documented key, value shape, producer, consumer, and
    the version each key was introduced. Third-party tool authors
    should read it to know which keys they may produce or consume; it
    is also the place to add new keys via PR.

    Adding a new key requires (a) appending a row to the registry table
    in that doc, (b) wiring the consumer in the same PR, and (c)
    keeping the value JSON-serializable. The append-only invariant is
    load-bearing for third-party interoperability — once shipped, a
    key's name and shape do not change without a major-version bump.
    """

    success: bool
    output: Any = None
    error: str | None = None
    error_kind: ToolErrorKind | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    side_effects: dict[str, Any] | None = None


# Backward-compat alias — existing tools that import ToolResult keep working.
ToolResult = ToolOutput


# ─────────────────────────────────────────────────────────────────────────────
# Schema format conversion helpers (CC9 — dual-format support)
# ─────────────────────────────────────────────────────────────────────────────

# Python type → JSONSchema "type" string. The reverse map below uses the
# first Python type for each JSON type.
_PY_TO_JSON_TYPE: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}

_JSON_TO_PY_TYPE: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _looks_like_json_schema(schema: Any) -> bool:
    """Heuristic: dict with ``"type": "object"`` and ``"properties"`` keys.

    The two markers together unambiguously identify JSONSchema; the custom
    format never uses ``"type"`` as a top-level key (a tool param literally
    named ``"type"`` would be its own property, not the dict's type tag).
    """
    if not isinstance(schema, dict):
        return False
    return schema.get("type") == "object" and "properties" in schema


def _json_schema_to_custom(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a JSONSchema dict to the legacy custom format.

    Exposed for round-trip equivalence testing and for callers that need
    a Python-typed view of an externally-supplied JSONSchema. ``Tool``
    itself does NOT call this at construction — ``input_schema`` is left
    exactly as authored to preserve the public contract.

    JSONSchema shape: ``{"type": "object", "properties": {NAME: PROP, ...},
    "required": [NAME, ...]}``.

    Mapping:
    - Required property with simple ``"type"``: ``{NAME: python_type}``
    - Optional property: ``{NAME: (python_type, default_or_None)}``
    - Type unions (``["string", "null"]``) collapse to the first non-null
      Python type (custom format can't express unions). Validation only
      checks presence, not type, so this loss is dispatch-equivalent.
    - Unknown / unsupported ``"type"`` values fall back to ``object`` so
      validation still treats the param as present.
    """
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    custom: dict[str, Any] = {}

    for name, prop in properties.items():
        if not isinstance(prop, dict):
            # Malformed entry — treat as required, untyped.
            custom[name] = dict if name in required else (dict, None)
            continue

        py_type = _resolve_property_type(prop.get("type"))

        if name in required:
            custom[name] = py_type
        else:
            default = prop.get("default")
            custom[name] = (py_type, default)

    return custom


def _resolve_property_type(json_type: Any) -> type:
    """Map a JSONSchema ``"type"`` value to a Python type for dispatch.

    Accepts a single string, a union (list), or anything else. Returns
    ``dict`` as a permissive fallback so validation doesn't reject params
    with exotic schemas.

    Note: this is asymmetric vs ``_python_type_name`` (which falls back
    to ``"string"`` for unknown Python types). The asymmetry is intentional
    today — both helpers are used only for round-trip equivalence, never
    on the construction path. If 1.1+ MCP server mode flows
    externally-supplied JSONSchema through here, raise on unknowns instead
    of coercing silently.
    """
    if isinstance(json_type, str):
        return _JSON_TO_PY_TYPE.get(json_type, dict)
    if isinstance(json_type, list):
        for t in json_type:
            if isinstance(t, str) and t != "null" and t in _JSON_TO_PY_TYPE:
                return _JSON_TO_PY_TYPE[t]
    return dict


def _custom_to_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert custom-format ``input_schema`` to JSONSchema (MCP-compatible).

    Custom format variants:
    - ``{NAME: type}`` — required, typed
    - ``{NAME: (type, default)}`` — optional with default
    - ``{NAME: "description string"}`` — required string with description.
      **Legacy convention.** Matches existing ``_validate_input`` behavior
      (non-tuple → required), even when the description text says "Optional".
      Strict MCP / Anthropic tool-use clients will mark these as required
      and reject calls that omit the parameter. New tool code should
      migrate to ``{NAME: (str, None)}`` or author JSONSchema directly.

    The custom format is strictly less expressive than JSONSchema: it
    cannot represent ``enum``, ``pattern``, ``format``, ``oneOf``,
    ``additionalProperties``, or nested object schemas. Authors needing
    those features must author JSONSchema directly.

    Output is a valid JSONSchema 2020-12 object: ``{"type": "object",
    "properties": {...}, "required": [...]}`` with ``"required"`` only
    present when non-empty (per JSONSchema convention).
    """
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, spec in schema.items():
        if isinstance(spec, tuple) and len(spec) >= 2:
            py_type, default = spec[0], spec[1]
            prop: dict[str, Any] = {"type": _python_type_name(py_type)}
            # JSONSchema accepts null; emit only when default is meaningful
            # to avoid noisy "default: null" entries on convention-optional
            # parameters.
            if default is not None:
                prop["default"] = default
            properties[name] = prop
        elif isinstance(spec, type):
            properties[name] = {"type": _python_type_name(spec)}
            required.append(name)
        elif isinstance(spec, str):
            # Description-as-value pattern (e.g. SensePresenceTool).
            properties[name] = {"type": "string", "description": spec}
            required.append(name)
        else:
            properties[name] = {}
            required.append(name)

    out: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        out["required"] = required
    return out


def _python_type_name(py_type: Any) -> str:
    """Map a Python type to its JSONSchema type name. Fallback: ``"string"``."""
    if isinstance(py_type, type):
        return _PY_TO_JSON_TYPE.get(py_type, "string")
    return "string"


# ─────────────────────────────────────────────────────────────────────────────
# Tool ABC
# ─────────────────────────────────────────────────────────────────────────────


class Tool(ABC):
    """Base class for agent-callable tools.

    ``input_schema`` accepts EITHER format:

    1. **Custom (legacy) format** — a flat dict where each entry is one of:

       - ``{NAME: python_type}`` — required parameter
       - ``{NAME: (python_type, default)}`` — optional parameter with default
       - ``{NAME: "description string"}`` — required string with description

       Example: ``{"path": str, "tail_lines": (int, None)}``

    2. **JSONSchema** — a dict shaped ``{"type": "object", "properties":
       {...}, "required": [...]}`` per the JSONSchema spec.

       Example: ``{"type": "object", "properties": {"path": {"type":
       "string"}}, "required": ["path"]}``

    Both formats are first-class. ``input_schema`` is left exactly as the
    tool author declared it — no construction-time mutation. ``_validate_input``
    branches on the schema shape so dispatch is identical for both formats.
    Existing tools (custom format) and ``@maxim.tool``-decorated tools
    (which already produce JSONSchema) keep working unchanged.

    **JSONSchema is the canonical format going forward.** It is the wire
    format used by MCP, OpenAI tool calls, Anthropic tool use, and OpenAPI.
    Authoring new tools against JSONSchema directly is recommended; the
    custom format remains supported indefinitely as a Python convenience.
    The custom format is strictly less expressive — it cannot represent
    ``enum``, ``pattern``, ``format``, ``oneOf/anyOf/allOf``,
    ``additionalProperties``, or nested object schemas. Authors needing
    those features must author JSONSchema directly.

    Use ``Tool.to_json_schema()`` to export the schema as JSONSchema for
    MCP servers, prompt construction, or external tool catalogs. The
    output is JSONSchema 2020-12 / MCP-compatible per
    https://spec.modelcontextprotocol.io/. Format-sensitive consumers
    inside the codebase should also route through ``to_json_schema()``
    rather than reading ``input_schema`` directly, so they handle both
    authored formats correctly.

    See ``docs/plans/mcp_compatibility.md`` for the broader 1.1+ MCP work
    that this dual-format support unlocks.
    """

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

    def cancel(self) -> None:
        """Best-effort abort hook for in-flight ``execute()`` calls.

        Default implementation is a no-op so existing tools work
        unchanged. Tools whose ``execute()`` does heavy work (HTTP
        requests, LLM calls, subprocess execution, large file reads)
        should override this to:

        - Set an instance flag the executing thread checks at safe
          points (e.g. between chunk reads or HTTP responses).
        - Close any open network/file/subprocess handles to unblock
          a thread waiting on I/O.
        - Release any tool-owned resources (cached responses, temp
          files) the implementation considers expensive to leak.

        ``cancel()`` is called from a *different thread* than the one
        running ``execute()``. Implementations must be thread-safe and
        should not raise — log and continue on partial failure.
        ``cancel()`` does NOT itself wait for ``execute()`` to return;
        the caller (e.g. the agent loop, an upstream cancellation
        token, or a Ctrl-C handler) is responsible for the join.

        ``cancel()`` may be called when no execution is in flight; the
        default no-op preserves that behavior. Implementations should
        treat redundant calls as benign.

        Defined as a regular method with a default body (NOT
        ``@abstractmethod``) so existing third-party Tool subclasses
        keep working without modification. New tools that need
        cancellation should override it.

        **No 1.0 dispatch path calls this method.** It ships as forward-
        compat infrastructure for 1.1+ MCP-subprocess and async-cancel
        work; today it can only be invoked by a caller that holds the
        tool reference directly (e.g. a test harness or a future agent-
        loop cancellation wiring). The ``Tool.timeout`` field is also
        not currently enforced by ``runtime/executor.py``; the two are
        independent reservations of contract surface for the same
        future cancellation pathway.

        ``tests/unit/test_tool_cancel.py::test_cancel_has_no_caller_in_executor_dispatch``
        pins the "no 1.0 caller" contract — if you wire ``cancel()``
        into the executor, update that test and document the new caller
        in CLAUDE.md under the ``Tool.cancel`` invariant.
        """
        return None

    def to_json_schema(self) -> dict[str, Any]:
        """Export ``input_schema`` as a JSONSchema dict (MCP-compatible).

        If ``input_schema`` is already JSONSchema, returns a deep copy
        (preserves enums, descriptions, additionalProperties, nested
        schemas, etc.). Otherwise converts the custom format to JSONSchema.

        Output shape: ``{"type": "object", "properties": {...}, "required":
        [...]}``. ``"required"`` is omitted when empty per JSONSchema
        convention. The result is suitable for direct use as an MCP tool
        ``inputSchema``, an Anthropic ``input_schema``, or an OpenAI
        function-call ``parameters`` object.
        """
        schema = getattr(self, "input_schema", None) or {}
        if not isinstance(schema, dict):
            return {"type": "object", "properties": {}}
        if _looks_like_json_schema(schema):
            return copy.deepcopy(schema)
        return _custom_to_json_schema(schema)

    def _validate_input(self, kwargs: dict[str, Any]) -> None:
        schema = getattr(self, "input_schema", None)
        if not isinstance(schema, dict):
            return

        # JSONSchema format: {"type": "object", "properties": {...}, "required": [...]}
        if _looks_like_json_schema(schema):
            required = set(schema.get("required", []))
            for key in required:
                if key not in kwargs:
                    raise ValueError(f"Missing required input: {key}")
            return

        # Custom (legacy) format: {"param_name": spec, ...}
        for key, spec in schema.items():
            optional = isinstance(spec, tuple) and len(spec) >= 2
            if key not in kwargs and not optional:
                raise ValueError(f"Missing required input: {key}")

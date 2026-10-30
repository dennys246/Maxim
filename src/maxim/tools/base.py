from __future__ import annotations

import copy
import functools
import logging
import operator
import os
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, EnumMeta, Flag, auto
from typing import TYPE_CHECKING, Any, ClassVar, Literal

if TYPE_CHECKING:
    from maxim.embodiment.sem import InteroceptiveOutcome

# Registration-time classifier for tool provenance / lifecycle. The four
# kinds correspond to the three independent registration regimes the
# sense* tool family grew under (per ``docs/plans/deferred/sense_tool_registry.md``)
# plus a default for everything else.
#
# - ``core-universal``  — registered once at boot via ``ToolRegistry.register``;
#                         always active, exempt from scene-scope and LRU.
# - ``auto-discovery``  — registered like core-universal but fires implicitly
#                         every tick alongside the agent's chosen action
#                         (``auto_fire=True``). Used for passive perception
#                         tools whose output the LLM reads but does not
#                         choose to invoke. Their dispatch bypasses the
#                         executor's actions.jsonl log.
# - ``scene-scoped``    — registered via ``ToolRegistry.register_scene_tools``
#                         when a scene loads; deactivated/re-activated as
#                         the active roster cycles. Counts toward the
#                         active scene-tool cap; eligible for LRU eviction.
# - ``sem-modulator-derived`` — a more specific flavor of scene-scoped,
#                         generated from an Entity's modulators/affordances
#                         in ``embodiment/tool_bridge.py``. Identified
#                         separately so the prompt-builder can grayscale
#                         these (and only these) when inactive but
#                         substrate-biased.
ToolKind = Literal[
    "core-universal",
    "auto-discovery",
    "scene-scoped",
    "sem-modulator-derived",
]


class ToolErrorKind(Enum):
    """Classification of tool execution errors."""

    FILE_NOT_FOUND = "file_not_found"
    PERMISSION_DENIED = "permission_denied"
    SYNTAX_ERROR = "syntax_error"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    EXTERNAL_FAILURE = "external_failure"
    VALIDATION = "validation"


# ─────────────────────────────────────────────────────────────────────────────
# Effect: what a tool can cause (#1315 item 2; docs/plans/atonia_gate_approach_notes.md §2)
# ─────────────────────────────────────────────────────────────────────────────


_logger = logging.getLogger(__name__)


class _EffectMeta(EnumMeta):
    """Refuses a reassignment or deletion of ``Effect.ALL`` once it is set (#1315 item 2 review): ``ALL`` is not an
    enum member, so ``EnumMeta`` alone would let ``Effect.ALL = Effect.READ`` narrow every fail-closed default."""

    def __setattr__(cls, name: str, value: Any) -> None:
        if name == "ALL" and "ALL" in cls.__dict__:
            raise AttributeError("Effect.ALL is fixed: it is the union of the seven flags")
        super().__setattr__(name, value)

    def __delattr__(cls, name: str) -> None:
        if name == "ALL":
            raise AttributeError("Effect.ALL is fixed: it is the union of the seven flags")
        super().__delattr__(name)


class Effect(Flag, metaclass=_EffectMeta):
    """Everything a tool can cause, through itself or through any collaborator it holds or is wired to later; not
    what it is for (atonia notes §2.1). A SET, never a ladder (V2): compare with :func:`exceeds`, never ``<=``.

    - ``READ``: reads state; changes nothing outside process memory.
    - ``SELF_SCOPE``: writes only Maxim's own persisted state under the data home (memory, NAc, its own logs).
    - ``WRITE_LOCAL``: writes, moves or deletes files outside the data home. A tool that spawns a process or runs
      code in-process can cause anything, so it declares ``ALL`` instead.
    - ``NETWORK``: egress the tool causes to a destination off the machine, DNS and GETs included. Two channels are
      not a tool's ``NETWORK`` (owner decisions V11, V16): the configured LLM lanes (the router is infrastructure)
      and the configured robot body (its channel is ``ACTUATE``). Offline still refuses both, at the lanes and the
      latch.
    - ``COMMS``: reaches a person or a channel (messages, calls, email).
    - ``ACTUATE``: physical output: motion, sound, a display.
    - ``GOVERN``: changes Maxim's own authority: mode, autonomy level, internet switch, allowed directories, the tool
      set.

    ``Effect.ALL`` is the union of the seven. It is attached after the class, not declared in it, so iterating
    ``Effect`` yields exactly the seven flags on every supported Python (3.10 also iterates a composite member), and
    the metaclass refuses to reassign it.
    Serialise an effect with :func:`effect_names`, never as the integer, which depends on declaration order.
    """

    READ = auto()
    SELF_SCOPE = auto()
    WRITE_LOCAL = auto()
    NETWORK = auto()
    COMMS = auto()
    ACTUATE = auto()
    GOVERN = auto()

    ALL: ClassVar[Effect]


Effect.ALL = functools.reduce(operator.or_, Effect)


def exceeds(effect: Effect, ceiling: Effect) -> bool:
    """True when *effect* has any flag outside *ceiling*: the one subset check (§2.2). ``Flag`` has no subset ``<=``,
    and an ``IntFlag`` would compare integers (WRITE_LOCAL <= READ|NETWORK), so no operator is used anywhere."""
    return bool(effect & ~ceiling)


def effect_names(effect: Effect) -> list[str]:
    """*effect* as a sorted list of its flag names: the one serialised form (§1A.8)."""
    return sorted(str(m.name) for m in Effect if m in effect)


# The effect each Tool subclass declared in its OWN class body, recorded once at class creation (§2.2). Weak keys,
# so a class made in a test or by a factory does not live forever. Read only through :func:`effect_of`.
_EFFECT_RECORD: weakref.WeakKeyDictionary[type, Effect] = weakref.WeakKeyDictionary()
# Classes that DECLARED a valid effect (in their body, or once through :func:`declare_effect`), so a later
# ``del cls.effect`` is never mistaken for "undeclared": a declared class can never be declared again (delta D1).
_EFFECT_DECLARED: weakref.WeakSet[type] = weakref.WeakSet()
# Classes already warned about an ``effect`` that is not a valid declaration (V15): one WARNING per class.
_EFFECT_WARNED: weakref.WeakSet[type] = weakref.WeakSet()


def warn_invalid_effect_once(cls: type, why: str) -> None:
    """One structured WARNING per class whose own ``effect`` is not a valid declaration, which therefore counts as
    ``ALL`` (owner decision V15: a third-party ``Tool`` with an ``effect`` attribute of its own keeps working under
    ceiling ``ALL``)."""
    if cls in _EFFECT_WARNED:
        return
    _EFFECT_WARNED.add(cls)
    _logger.warning(
        "%s.%s: %s; it counts as Effect.ALL",
        cls.__module__,
        cls.__qualname__,
        why,
        extra={
            "event": "tool_effect_invalid",
            "data": {"class": f"{cls.__module__}.{cls.__qualname__}", "reason": why},
        },
    )


def effect_of(obj: object) -> Effect:
    """The effect recorded for ``type(obj)``; ``ALL`` for a type with no record (a duck-typed tool, a mock, a class
    made outside ``Tool``) and for one whose own ``effect`` no longer matches its record (:func:`effect_mismatch`):
    a reassignment never narrows a tool, and never leaves a stale narrower record for a later check to trust. Never
    read from the instance, and never an inherited class value."""
    recorded = _EFFECT_RECORD.get(type(obj))
    if recorded is None or effect_mismatch(obj) is not None:
        return Effect.ALL
    return recorded


def effect_mismatch(obj: object) -> str | None:
    """Why ``type(obj)``'s own ``effect`` no longer matches its record, or ``None``.

    A value PRESENT in the class's own ``__dict__`` that differs from the record is a mismatch: that is a
    ``cls.effect = ...`` after creation. So is an ABSENT value with a record narrower than ``ALL``: a ``del
    cls.effect`` after creation, after which the class inherits whatever its base declares. An absent value with a
    recorded ``ALL`` is consistent (an undeclared tool).
    """
    cls = type(obj)
    recorded = _EFFECT_RECORD.get(cls)
    if recorded is None:
        return None
    if "effect" not in cls.__dict__:
        if recorded == Effect.ALL:
            return None
        return f"its own effect was deleted but its record is {effect_names(recorded)}"
    own = cls.__dict__["effect"]
    if isinstance(own, Effect) and own == recorded:
        return None
    return f"its own effect is {own!r} but its record is {effect_names(recorded)}"


def declare_effect(cls: type, effect: Effect) -> None:
    """Declare *effect* on a Tool class that declared none: the one sanctioned late declaration, used by
    ``maxim.register_tool(tool, effect=)`` (V10, §1A.4). It sets the class's own ``effect`` and its record together,
    so admission sees no mismatch. A class that already declares anything else, or whose value no longer matches its
    record, is refused (``TypeError``): a declaration is never rewritten after the fact. A ``@maxim.tool`` function
    tool always declares, so pass ``effect=`` to the decorator instead."""
    if not isinstance(effect, Effect) or not effect:
        raise TypeError(f"effect must be a non-empty maxim.tools.base.Effect, got {effect!r}")
    if not (isinstance(cls, type) and issubclass(cls, Tool)):
        raise TypeError(f"effect= needs a maxim.tools.base.Tool instance, got a {cls!r}")
    if cls not in _EFFECT_RECORD:
        # Delta round D8: a base that overrides __init_subclass__ without super() leaves its subclasses unrecorded
        # (effect_of says ALL). Narrowing one would trust a declaration nothing recorded.
        raise TypeError(
            f"{cls.__module__}.{cls.__qualname__} has no effect record (an __init_subclass__ that does not call "
            f"super()?); effect={effect!r} cannot declare it"
        )
    if "effect" not in cls.__dict__ and cls in _EFFECT_DECLARED:
        raise TypeError(
            f"{cls.__module__}.{cls.__qualname__}'s declared effect was deleted after class creation; "
            f"effect={effect!r} cannot rewrite it"
        )
    if "effect" in cls.__dict__:
        if cls.__dict__["effect"] == effect and _EFFECT_RECORD.get(cls) == effect:
            return
        raise TypeError(
            f"{cls.__module__}.{cls.__qualname__} already declares effect {cls.__dict__['effect']!r}; "
            f"effect={effect!r} cannot rewrite it"
        )
    type.__setattr__(cls, "effect", effect)
    _EFFECT_RECORD[cls] = effect
    _EFFECT_DECLARED.add(cls)


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

    ``rpe`` is NOT a ``side_effects`` key and not the tool's to set: the executor stamps each
    invocation's surprise (|RPE|) after the tool returns, overwriting anything a tool put there.
    """

    success: bool
    output: Any = None
    error: str | None = None
    error_kind: ToolErrorKind | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    side_effects: dict[str, Any] | None = None
    # Surprise (|RPE|) of THIS invocation's outcome, stamped by the executor from the tool-pain
    # bridge; None when no causal link attributed it. Tools never set it (#847).
    rpe: float | None = None
    # The body around THIS invocation, stamped by the executor (memory-strength Phase 2b-ii):
    # per-drive pressure read BEFORE the action (after it, an ``eat`` would see its own hunger
    # already relieved) and per-drive relief the action produced, each a sorted tuple of
    # ``(drive, value in [0, 1])``. ``None`` = no body attached. Tools never set either.
    drive_pressure_before: tuple[tuple[str, float], ...] | None = None
    drive_relief: tuple[tuple[str, float], ...] | None = None
    # The pain of THIS invocation, stamped by the executor from the tool-pain bridge
    # (memory-strength Phase 2S-c): what the action caused, else the peak felt while it ran; 0.0
    # when a pain source was watched and nothing fired; None when none was watched, or the call never ran (an
    # inactive or unregistered tool starts no invocation, #1207). Tools never set it.
    pain: float | None = None
    # What THIS invocation did to the body (grounding GL2a, docs/plans/autonomic_layer.md §3.1), stamped
    # by the executor: one record when the tool ran on an agent-bound body, else None. Record-only.
    # ``repr=False`` is load-bearing (#1189): ``str(ToolOutput)`` is persisted and substring-searched by
    # the Hippocampus, so a printed record would put its numbers and nouns into memory retrieval.
    interoceptive_outcome: InteroceptiveOutcome | None = field(default=None, repr=False)
    # The event signature the tool-pain bridge booked THIS invocation's failure under (#1200 Stage 2), else
    # None: ``tool_dispatch.record_outcome`` skips its own NEGATIVE for exactly that signature, so a failure
    # is booked once. Stamped by the executor; tools never set it. ``repr=False`` for the same reason as above.
    nac_failure_booked: str | None = field(default=None, repr=False)


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

    See ``docs/plans/deferred/mcp_compatibility.md`` for the broader 1.1+ MCP work
    that this dual-format support unlocks.
    """

    name: str
    description: str = ""
    input_schema: dict[str, Any] = {}
    timeout: float = 30.0  # per-tool timeout declaration (seconds)

    # ── Tool metadata (W1 sense_tool_registry MVP) ───────────────────────
    # ``auto_fire`` — when True, the agent loop dispatches this tool every
    # tick (alongside the LLM-chosen action) and does NOT log the call to
    # ``actions.jsonl``. Default False keeps existing tools unchanged.
    # Used today only by ``SensePresenceTool`` (auto-discovery scan).
    # See [docs/plans/deferred/sense_tool_registry.md] § "Tool metadata" and
    # ``runtime/loop_perception.py::auto_sense`` dispatch.
    auto_fire: bool = False
    # ``kind`` — registration-time classifier. Default ``"core-universal"``
    # preserves the historical semantics of plain ``ToolRegistry.register``.
    # Scene-scoped factories (``embodiment/tool_bridge.py``) override to
    # ``"sem-modulator-derived"``; ``SensePresenceTool`` to
    # ``"auto-discovery"``.
    kind: ToolKind = "core-universal"
    # ``advertised`` — False keeps a registered tool dispatchable but out of
    # the roster the prompt offers the model (#1042): a decoy such as the
    # narrator's ``respond`` exists only to answer a stray call with a
    # redirect, and naming it in the prompt invites that call.
    advertised: bool = True
    # ``speech_params`` (#1261) — the params that hold the words this tool SAYS, in the order the tool itself
    # reads them; empty means the tool is not speech. ``InstrumentedExecutor`` reads it off the tool a call
    # actually runs and records the words as ``ActionRecord.utterance``, which every reader of what an agent
    # said reads (never a tool-name list, never ``result_output``, which for ``respond`` is a receipt).
    speech_params: tuple[str, ...] = ()
    # ``untrusted_output`` (Session R, #1444) — the output carries text from outside the agent replayed from memory
    # (a recalled utterance, a stored reflection), so its follow-up is framed as EXTERNAL, untrusted content. Read
    # off the tool a call actually runs (after alias resolution), stamped on the ``ActionFollowup`` at dispatch.
    untrusted_output: bool = False
    # ``effect`` (#1315 item 2) — everything this tool can cause (:class:`Effect`). Declared in each concrete class's
    # OWN body: an inherited value does not count, because a subclass overriding ``execute`` must not inherit READ.
    # Undeclared means ``ALL``. Read it with :func:`effect_of`, never ``tool.effect``.
    effect: ClassVar[Effect]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Record the class's own ``effect``, or ``ALL`` when it declares none. ``Effect(0)`` is refused here
        (``TypeError``), so no tool passes every ceiling by declaring nothing. Any other value that is not an
        ``Effect`` (a third-party tool's own ``effect`` attribute, a property) records ``ALL`` with one WARNING
        (owner decision V15), so such a tool keeps working under ceiling ``ALL`` and is refused below it."""
        super().__init_subclass__(**kwargs)
        if "effect" not in cls.__dict__:
            _EFFECT_RECORD[cls] = Effect.ALL
            return
        own = cls.__dict__["effect"]
        if isinstance(own, Effect):
            if not own:
                raise TypeError(f"{cls.__module__}.{cls.__qualname__}.effect is Effect(0): declare what it can cause")
            _EFFECT_RECORD[cls] = own
            _EFFECT_DECLARED.add(cls)
            return
        _EFFECT_RECORD[cls] = Effect.ALL
        warn_invalid_effect_once(cls, f"its own effect {own!r} is not a maxim.tools.base.Effect")

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


# ── Host-tool containment (#949) ─────────────────────────────────────────────────────────────────
# The environment a host-acting tool's subprocess receives: an ALLOWLIST, so API keys and tokens held in
# the parent's ENVIRONMENT (ANTHROPIC_API_KEY, a peer key, cloud credentials) never reach a model-driven
# command. It does not hide FILES the user can read (~/.config/maxim/api_key, ~/.aws/credentials): these
# are host tools, not a sandbox. Kept: the interpreter-selection vars (PYTHONPATH, VIRTUAL_ENV) so "run the
# tests" runs the project's own code, and the git/gpg config locations (XDG_CONFIG_HOME, GNUPGHOME,
# GPG_TTY). Dropped: everything else, MAXIM_* included (no nested run can re-arm a MAXIM_ALLOW_* gate) --
# and SSH_AUTH_SOCK (access to the user's SSH keys), GIT_* (GIT_DIR would point git outside its root),
# proxies and CA bundles. Extending the list is a code change here, by design.
# Deliberately NOT ``utils/sandbox_executor.py::SandboxExecutor._build_safe_env``: the sandbox REDIRECTS
# HOME and pins PATH for code it isolates; a host tool inherits both. One shared list would be wrong for
# one of them.
HOST_TOOL_ENV_ALLOW = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TMPDIR",
        "TERM",
        "TZ",
        "LANG",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "PYTHONIOENCODING",
        "XDG_CONFIG_HOME",
        "GNUPGHOME",
        "GPG_TTY",
    }
)
_HOST_TOOL_ENV_PREFIXES = ("LC_",)


def host_tool_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """The allowlisted environment for a host-acting tool's subprocess (#949)."""
    source = os.environ if environ is None else environ
    return {
        key: value
        for key, value in source.items()
        if key in HOST_TOOL_ENV_ALLOW or key.startswith(_HOST_TOOL_ENV_PREFIXES)
    }


def _within(path: str, allowed_dirs: list[str]) -> bool:
    return any(path == root or path.startswith(root + os.sep) for root in allowed_dirs)


def tool_workdir(allowed_dirs: list[str] | None) -> str | None:
    """The working directory of a host coding tool (#949; owner decision 2026-09-28).

    The PROJECT -- the process working directory -- when it lies inside the mode's ``allowed_dirs``,
    as it does in every mode's filesystem policy; otherwise ``allowed_dirs[0]`` (a sim tmpdir or the
    console's override root, which deliberately exclude the process CWD). ``None`` (no containment)
    means the process CWD. ``allowed_dirs[0]`` alone was the wrong rule: in the mode configs it is the
    scratch ``.maxim_workspace``, so tests collected nothing and repo paths silently diffed as empty.
    """
    if not allowed_dirs:
        return None
    roots = [os.path.realpath(d) for d in allowed_dirs]
    cwd = os.path.realpath(os.getcwd())
    return cwd if _within(cwd, roots) else roots[0]


def contained_path(path: object, allowed_dirs: list[str] | None, base: str | None = None) -> str | None:
    """A model-supplied path resolved and checked against ``allowed_dirs``, or None when refused (#949).

    Relative paths resolve against ``base`` (the tool's working directory; the process CWD when None),
    then ``realpath`` -- so ``..`` and symlinks cannot step outside. An empty or non-string path is
    refused. With no ``allowed_dirs`` (no containment) the path is returned unchanged.
    """
    if not isinstance(path, str) or not path.strip():
        return None
    if not allowed_dirs:
        return path
    expanded = os.path.expanduser(path)
    if not os.path.isabs(expanded):
        expanded = os.path.join(base or os.getcwd(), expanded)
    resolved = os.path.realpath(expanded)
    return resolved if _within(resolved, [os.path.realpath(d) for d in allowed_dirs]) else None


def _same_entry(a: str, b: str) -> bool:
    """Whether two existing paths are one filesystem entry (stat identity: catches ``.GIT``, Unicode-ignorable and
    8.3 spellings that a name comparison misses)."""
    try:
        return os.path.samefile(a, b)
    except (OSError, ValueError):
        return False


def _global_git_config_paths() -> list[str]:
    """The user- and installation-level files host git reads: ``host_tool_env`` passes ``HOME`` and
    ``XDG_CONFIG_HOME`` through, and Homebrew's git reads its prefix's ``etc/gitconfig`` (not under FORBIDDEN_PATHS)."""
    home = os.path.expanduser("~")
    xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config")
    return [
        os.path.join(home, ".gitconfig"),
        os.path.join(xdg, "git", "config"),
        os.path.join(xdg, "git", "attributes"),
        os.path.join(home, ".config", "git", "config"),
        os.path.join(home, ".config", "git", "attributes"),
        "/usr/local/etc/gitconfig",
        "/opt/homebrew/etc/gitconfig",
    ]


def _looks_like_git_dir(path: str) -> bool:
    """git's own test for a git directory, approximately: a ``HEAD`` file beside an ``objects`` directory. Covers a
    separate git dir (``--separate-git-dir``), ``worktrees/<n>``, ``modules/<n>`` and a bare repository."""
    return os.path.isfile(os.path.join(path, "HEAD")) and os.path.isdir(os.path.join(path, "objects"))


def _gitfile_target(gitfile: str) -> str | None:
    """The git dir a ``.git`` FILE points at (``gitdir: <path>``, relative to the file's directory), or None."""
    try:
        with open(gitfile, encoding="utf-8") as f:
            first = f.readline().strip()
    except (OSError, UnicodeDecodeError):
        return None
    if not first.lower().startswith("gitdir:"):
        return None
    target = first.split(":", 1)[1].strip()
    if not target:
        return None
    return os.path.realpath(os.path.join(os.path.dirname(gitfile), os.path.expanduser(target)))


def git_dir_write_refusal(path: str) -> str | None:
    """Why a file tool must not write ``path``, or None (#957).

    A model that can write a repository's git configuration can make host git run code: clean/process filters run on
    ``git diff`` and ``git add``, and no git switch disables them (#949). So the file tools refuse to write:

    - under any ``.git`` path component (case-folded), and under any entry that IS a ``.git`` (stat identity, so
      ``.GIT``, Unicode-ignorable or 8.3 spellings of it are refused too), including the ``.git`` file itself;
    - under the git dir a ``.git`` FILE points at (``git init --separate-git-dir``: a git dir not named ``.git``);
    - inside any directory that already looks like a git dir (``HEAD`` + ``objects/``): a sibling separate git dir,
      ``worktrees/<n>``, a bare repository;
    - the user-level git config and attributes host git reads (``~/.gitconfig``, ``$XDG_CONFIG_HOME/git/``,
      Homebrew's ``etc/gitconfig``), compared case-folded.

    A git dir assembled from scratch, file by file, is not caught here (no single write looks like one); host git
    refuses to use it through ``safe.bareRepository=explicit`` in ``GIT_HARDENING_ARGS`` (git 2.38+; older git
    ignores it). This narrows the risk; it
    does not close it: ``bash`` (and any code the model can run) can still write there, as can a file pulled in by an
    ``include.path`` the model cannot see from here.
    """
    real = os.path.realpath(os.path.expanduser(path))
    parts = real.split(os.sep)
    if any(p.casefold() == ".git" for p in parts if p):
        return f"Writing under a .git directory is not allowed (git configuration can run code): {real}"
    for config in _global_git_config_paths():
        # Case-folded: on a case-insensitive filesystem ``~/.GitConfig`` IS ``~/.gitconfig`` even before it exists.
        if real.casefold() == os.path.realpath(config).casefold() or (
            os.path.lexists(real) and _same_entry(real, config)
        ):
            return f"Writing the user's git configuration is not allowed (git configuration can run code): {real}"
    # Walk the existing ancestors (and the path itself): is any of them a .git entry, or inside a gitfile's target?
    current = real
    while True:
        parent = os.path.dirname(current)
        if os.path.lexists(current):
            dotgit = os.path.join(parent, ".git")
            if os.path.lexists(dotgit) and _same_entry(current, dotgit):
                return f"Writing under a .git directory is not allowed (git configuration can run code): {real}"
            if current != real and os.path.isdir(current) and _looks_like_git_dir(current):
                return f"Writing inside a git directory is not allowed (git configuration can run code): {real}"
        beside = os.path.join(current, ".git")
        if os.path.isfile(beside):
            target = _gitfile_target(beside)
            if target and (real == target or real.startswith(target + os.sep)):
                return f"Writing inside a repository's git directory is not allowed: {real} (in {target})"
        if parent == current:
            return None
        current = parent


def git_env(root: str | None) -> dict[str, str]:
    """``host_tool_env()`` plus ``GIT_CEILING_DIRECTORIES`` at the root's parent, so git never searches
    ABOVE its working root for a repository (#949): a root that is not itself inside a repo fails
    loudly instead of silently reaching an enclosing repo (the host's, when the root is a workspace)."""
    env = host_tool_env()
    if root:
        env["GIT_CEILING_DIRECTORIES"] = os.path.dirname(os.path.realpath(root))
    return env


# git configuration that turns a git call into code execution, switched off on every host git call:
# hooks (a write to .git/hooks/*) and fsmonitor. `git diff` adds --no-ext-diff/--no-textconv and
# `git commit` adds --no-gpg-sign (a repo-local gpg.program). NOT closable by any git switch: clean/
# smudge/process FILTERS (.gitattributes + filter.*.clean in .git/config) run on diff and add. A model
# that can write a repository's .git/config can therefore still make git run code -- which is why both
# git tools are opt-in (MAXIM_ALLOW_GIT_DIFF / MAXIM_ALLOW_GIT_COMMIT). Since #957 the file tools refuse to write the
# git configuration (git_dir_write_refusal); `bash` and any code the model runs still can.
# safe.bareRepository=explicit (#957): git will not implicitly use a directory as a BARE repository, which a model can
# build from scratch with write_file alone (HEAD, objects/, refs/, a config with a filter), in any order, so no path
# check at write time can see it coming. A normal worktree's .git is unaffected. Needs git 2.38+: older git ignores
# the key, and there a from-scratch git dir is still usable (a stated residual).
GIT_HARDENING_ARGS = (
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "safe.bareRepository=explicit",
)


def git_root_refusal(workdir: str | None, allowed_dirs: list[str] | None) -> str | None:
    """Why git must not run from ``workdir``, or None (#949).

    The ceiling stops git searching UPWARD, but a model-written ``.git`` FILE (``gitdir: /elsewhere``)
    or ``core.worktree`` points it anywhere. So before any git call, ask git where it resolved and
    refuse unless both the repository directory and the worktree lie inside ``allowed_dirs``. With no
    ``allowed_dirs`` (no containment) there is nothing to check.
    """
    if not allowed_dirs:
        return None
    import subprocess

    try:
        result = subprocess.run(
            ["git", *GIT_HARDENING_ARGS, "rev-parse", "--absolute-git-dir", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=workdir,
            env=git_env(workdir),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"could not resolve the git repository: {exc}"
    if result.returncode != 0:
        return result.stderr.strip() or "not a git repository"
    roots = [os.path.realpath(d) for d in allowed_dirs]
    for resolved in result.stdout.split("\n"):
        if resolved.strip() and not _within(os.path.realpath(resolved.strip()), roots):
            return f"git resolved {resolved.strip()!r}, outside the allowed directories"
    return None

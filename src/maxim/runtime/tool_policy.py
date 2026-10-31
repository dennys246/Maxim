"""``ToolPolicy``: what an executor may dispatch (#1315 item 3; docs/plans/atonia_gate_approach_notes.md §3).

Every ``Executor`` carries one, required and keyword-only (owner decision V3: ``None`` is a ``TypeError``, never
translated into an allow-all). A policy holds the name gate (``AgentPermissions``), the effect ceiling (``Effect``)
and, for a pinned policy, the exact tool objects it authorises. It lives here because only ``runtime`` may see both
``agents/`` and ``tools/`` (``utils/audit.py::LAYER_RULES``).

A policy is immutable and unforgeable from ordinary code: no public constructor (``__new__`` and ``__init__`` check a
module-private key), no assignment or deletion, no subclass, ``copy`` returns the same object and pickling raises.
Reaching the key through introspection, ``object.__setattr__`` or a slot descriptor is in-process code and out of the
threat model (§0); nothing writes those by accident. It is never persisted, so CC3 does not apply.

The factories are the only way in, and the census test (``tests/unit/test_tool_policy_census.py``) pins every
caller of the permissive two, ``unrestricted`` and ``deny``, with its reason.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NoReturn

from maxim.agents.permissions import AgentPermissions
from maxim.tools.base import _CURRENT_POLICY, Effect

if TYPE_CHECKING:
    from maxim.tools.base import Tool

__all__ = [
    "OFFLINE_CEILING",
    "ToolPolicy",
    "aut_tool_policy",
    "cli_tool_policy",
    "dispatching",
    "effective_ceiling",
    "orchestrator_tool_policy",
    "policy_from_tools_settings",
    "require_policy",
]

OFFLINE_CEILING: Effect = Effect.READ | Effect.SELF_SCOPE | Effect.WRITE_LOCAL | Effect.ACTUATE | Effect.GOVERN
"""The ceiling ANDed onto every policy while the offline latch is engaged (V7, §1A.2). Enumerated, never built as a
complement, so a flag added later is refused offline until someone puts it here."""

_KEY = object()


def _sealed(*_args: Any) -> NoReturn:
    raise AttributeError("ToolPolicy is immutable: build a narrower one with narrowed()")


class ToolPolicy:
    """An executor's dispatch policy. Build one with a factory: ``unrestricted``, ``deny``, ``only``, ``pinned``,
    or ``narrowed`` from an existing one (never wider)."""

    __slots__ = ("permissions", "ceiling", "pins", "reason")

    permissions: AgentPermissions
    ceiling: Effect
    pins: Mapping[str, Tool] | None
    reason: str

    def __new__(cls, *_args: Any, _key: object = None, **_kwargs: Any) -> ToolPolicy:
        if _key is not _KEY:
            raise TypeError("ToolPolicy has no public constructor: use ToolPolicy.unrestricted/deny/only/pinned")
        return object.__new__(cls)

    def __init__(
        self,
        *,
        _key: object,
        permissions: AgentPermissions,
        ceiling: Effect,
        pins: Mapping[str, Tool] | None,
        reason: str,
    ) -> None:
        if _key is not _KEY:  # pragma: no cover - __new__ already refused it
            raise TypeError("ToolPolicy has no public constructor")
        if type(permissions) is not AgentPermissions:
            raise TypeError(f"ToolPolicy.permissions must be AgentPermissions, got {type(permissions).__name__}")
        if not isinstance(ceiling, Effect):
            raise TypeError(f"ToolPolicy.ceiling must be an Effect, got {type(ceiling).__name__}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("ToolPolicy needs a non-empty reason: it is what the census and the logs show")
        object.__setattr__(self, "permissions", permissions)
        object.__setattr__(self, "ceiling", ceiling)
        object.__setattr__(self, "pins", pins)
        object.__setattr__(self, "reason", reason)

    __setattr__ = _sealed
    __delattr__ = _sealed

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("ToolPolicy cannot be subclassed: executors check type(policy) is ToolPolicy")

    def __copy__(self) -> ToolPolicy:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> ToolPolicy:
        return self

    def __reduce__(self) -> NoReturn:
        raise TypeError("a ToolPolicy is never persisted or sent: it is rebuilt from its factory where it is used")

    def __repr__(self) -> str:
        pins = f", pins={sorted(self.pins)}" if self.pins is not None else ""
        return f"ToolPolicy(ceiling={self.ceiling!r}{pins}, reason={self.reason!r})"

    # ── factories ────────────────────────────────────────────────────────

    @classmethod
    def unrestricted(cls, *, reason: str) -> ToolPolicy:
        """Ceiling ``ALL`` and no allow-list. The mode gate and the autonomy forbids still run. Permissive: every
        caller is pinned by the census."""
        return cls(_key=_KEY, permissions=AgentPermissions(), ceiling=Effect.ALL, pins=None, reason=reason)

    @classmethod
    def deny(cls, names: Iterable[str], *, reason: str) -> ToolPolicy:
        """Ceiling ``ALL`` with a deny list (the console's ``tools.deny``). Permissive: censused like
        ``unrestricted``."""
        perms = AgentPermissions(tool_deny=frozenset(str(n) for n in names))
        return cls(_key=_KEY, permissions=perms, ceiling=Effect.ALL, pins=None, reason=reason)

    @classmethod
    def only(cls, names: Iterable[str], *, ceiling: Effect, reason: str) -> ToolPolicy:
        """An allow-list (exact names or ``kind:`` selectors) under ``ceiling``."""
        perms = AgentPermissions(tool_allow=frozenset(str(n) for n in names))
        return cls(_key=_KEY, permissions=perms, ceiling=ceiling, pins=None, reason=reason)

    @classmethod
    def pinned(cls, instances: Mapping[str, Tool], *, ceiling: Effect, reason: str) -> ToolPolicy:
        """Authorise these OBJECTS, not their names: dispatch requires that the resolved tool ``is`` the pinned
        instance, and refuses aliases and every other name (layer ``"pin"``). The mapping is snapshotted, so a later
        change to the caller's dict adds nothing."""
        return cls(
            _key=_KEY,
            permissions=AgentPermissions(),
            ceiling=ceiling,
            pins=MappingProxyType(dict(instances)),
            reason=reason,
        )

    def narrowed(
        self, *, ceiling: Effect | None = None, only: Iterable[str] | None = None, deny: Iterable[str] = ()
    ) -> ToolPolicy:
        """A policy never wider than this one: ceilings ANDed, allow-lists intersected (``kind:`` selectors are
        entries like any other), deny lists unioned; the clearance, the SEM rules and any pins are kept."""
        p = self.permissions
        allow = p.tool_allow
        if only is not None:
            wanted = frozenset(str(n) for n in only)
            allow = wanted if allow is None else allow & wanted
        perms = AgentPermissions(
            clearance=p.clearance,
            tool_deny=p.tool_deny | frozenset(str(n) for n in deny),
            tool_allow=allow,
            sem_access_rules=p.sem_access_rules,
        )
        new_ceiling = self.ceiling if ceiling is None else self.ceiling & ceiling
        return type(self)(_key=_KEY, permissions=perms, ceiling=new_ceiling, pins=self.pins, reason=self.reason)


def require_policy(policy: object, *, where: str) -> ToolPolicy:
    """``policy`` if it is a real ``ToolPolicy``, else a ``TypeError`` naming ``where`` (V3). ``type() is``, not
    ``isinstance``: a duck-typed stand-in is refused."""
    if type(policy) is not ToolPolicy:
        raise TypeError(
            f"{where}: policy= must be a ToolPolicy (got {type(policy).__name__}); build one with "
            "ToolPolicy.unrestricted(reason=...), deny, only or pinned (#1315 item 3)"
        )
    return policy


def effective_ceiling(policy: ToolPolicy) -> Effect:
    """The ceiling the dispatch seam enforces: the policy's, ANDed with ``OFFLINE_CEILING`` while the offline latch
    is engaged (V7, applied here once rather than at each builder site)."""
    from maxim._offline import is_offline  # noqa: PLC0415 -- a leaf module; read at each check, the latch is one-way

    return policy.ceiling & OFFLINE_CEILING if is_offline() else policy.ceiling


@contextlib.contextmanager
def dispatching(policy: ToolPolicy) -> Iterator[None]:
    """Expose ``policy`` to the running tool as ``tools/base.py::current_policy()`` for one dispatch."""
    token = _CURRENT_POLICY.set(policy)
    try:
        yield
    finally:
        _CURRENT_POLICY.reset(token)


def policy_from_tools_settings(
    allow: tuple[str, ...] | list[str] | None, deny: tuple[str, ...] | list[str] | None
) -> ToolPolicy:
    """The console's policy from the resolved ``tools.allow`` / ``tools.deny`` settings (the section is declared in
    ``runtime/config_loader.py::ToolsConfigSection``): their deny list, narrowed to the allow-list when one is set
    (an explicit empty ``allow=()`` is a legitimate "no tools" list). Unconfigured is an empty deny list, ceiling
    ``ALL``: the pre-1.1.3 console. Replaces ``agents/permissions.py::tool_permissions_from_settings`` (§1A.5)."""
    policy = ToolPolicy.deny(deny or (), reason="console: the operator's tools.allow / tools.deny settings")
    return policy if allow is None else policy.narrowed(only=allow)


# ── the policies of the fenced construction sites (census-pinned) ────────


def cli_tool_policy() -> ToolPolicy:
    """The ``maxim`` CLI agent: the operator's own agent, gated by the run mode and the autonomy level."""
    return ToolPolicy.unrestricted(reason="cli: the operator's own agent; the mode and autonomy gates still run")


def aut_tool_policy() -> ToolPolicy:
    """The simulation's agent under test: the sim measures what the agent does with its full roster."""
    return ToolPolicy.unrestricted(reason="sim AUT: measured on its full roster; the mode gate still runs")


def orchestrator_tool_policy() -> ToolPolicy:
    """The simulation narrator: its tools drive the AUT (owner decision V12, so they are ALL)."""
    return ToolPolicy.unrestricted(reason="sim narrator: its tools drive the AUT (V12)")

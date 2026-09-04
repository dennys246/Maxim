"""Two-layer agent permissions: enforced gates + perceived authority.

The two layers are intentionally orthogonal:

* **Enforced layer** (``AgentPermissions``) — hard gates evaluated at the
  tool dispatch hot path. Decisions are O(1): a frozenset membership
  check on tool names plus a small list-walk for SEM access rules. No
  YAML I/O happens here; ``AgentPermissions`` is loaded from campaign
  YAML once and held immutably for the run.

* **Perceived layer** (``PerceivedAuthority`` /
  ``PerceivedAuthorityTracker``) — soft, learned authority that the
  bio-stack can update from outcomes. Used by NAc for causal learning
  and by FearAgent for review weight, and surfaced into the LLM system
  prompt by the prompt assembler.

These two layers are independent. A character can have full enforced
clearance but zero perceived authority (a feared, distrusted spymaster);
another can have high perceived authority but no enforced clearance (a
beloved figurehead). Campaigns set enforced authority statically; the
bio-stack learns perceived authority from outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Enforced layer
# ---------------------------------------------------------------------------


KIND_SELECTOR_PREFIX = "kind:"
"""Prefix marking an allow/deny entry as a ``Tool.kind`` selector rather
than an exact tool name (``kind:sem-modulator-derived``)."""


def tool_permissions_from_settings(
    allow: tuple[str, ...] | list[str] | None,
    deny: tuple[str, ...] | list[str] | None,
) -> AgentPermissions | None:
    """Build the console's ``AgentPermissions`` from resolved ``tools.*`` settings.

    ``allow=None`` and an empty ``deny`` is "unconfigured" and returns
    ``None`` — the executor then has no permission gate at all, which is
    the pre-1.1.3 console behaviour. Anything else returns a real
    ``AgentPermissions`` (an explicit empty ``allow=()`` is a legitimate
    "no tools" list, not unconfigured). The ``tools`` section itself is
    declared in ``runtime/config_loader.py::ToolsConfigSection``.
    """
    deny_set = frozenset(deny or ())
    if allow is None and not deny_set:
        return None
    return AgentPermissions(
        tool_allow=frozenset(allow) if allow is not None else None,
        tool_deny=deny_set,
    )


@dataclass(frozen=True)
class SEMAccessRule:
    """A single SEM access rule.

    ``entity`` is the SEM entity name (or ``"*"`` for all). ``deny`` is
    a tuple of affordance names to refuse. ``min_clearance`` is the
    minimum clearance level required to invoke any affordance on this
    entity.
    """

    entity: str
    deny: tuple[str, ...] = ()
    min_clearance: int = 0


@dataclass(frozen=True)
class AgentPermissions:
    """Hard, enforced authority for an agent within a campaign.

    Constructed once at campaign load and held immutably so the
    executor's permission check stays O(1) on the hot path.
    """

    clearance: int = 0
    tool_deny: frozenset[str] = frozenset()
    tool_allow: frozenset[str] | None = None  # When set, acts as an allow-list
    sem_access_rules: tuple[SEMAccessRule, ...] = ()

    def can_invoke_tool(self, tool_name: str, *, kind: str | None = None) -> tuple[bool, str | None]:
        """Return ``(allowed, reason_if_denied)`` for *tool_name*.

        Entries in ``tool_deny`` / ``tool_allow`` are exact tool names or
        ``kind:<kind>`` selectors. A selector matches when *kind* (the
        tool's ``Tool.kind``, looked up by the caller — this module never
        touches the registry) equals ``<kind>``; generated SEM affordance
        tools carry per-entity names, so ``kind:sem-modulator-derived``
        is how a list admits or refuses all of them at once. ``kind=None``
        (unregistered name, or a caller without the tool object) matches
        no selector, so a name-only check is exactly the pre-selector
        behaviour.

        The reason string is suitable for surfacing to the LLM as the
        ``ToolOutput.error`` body so the agent learns why a call was
        refused.
        """
        selector = f"{KIND_SELECTOR_PREFIX}{kind}" if kind else None
        if tool_name in self.tool_deny or (selector is not None and selector in self.tool_deny):
            return False, f"Tool '{tool_name}' is denied by enforced permissions."
        if self.tool_allow is not None:
            if tool_name in self.tool_allow or (selector is not None and selector in self.tool_allow):
                return True, None
            return False, f"Tool '{tool_name}' is not in the enforced allow-list."
        return True, None

    def denial_reason(self, tool_name: str, *, kind: str | None = None) -> str | None:
        """The DENY half of :meth:`can_invoke_tool` only.

        For the Executor's pre-alias check: a deny may target the alias
        SOURCE name (deny ``shell``), so it must run on the raw name, but an
        allow-list can only be judged on the canonical name after alias
        resolution — otherwise ``recall`` is refused even though
        ``memory_recall`` is allowed (review finding, sandbox-launch).
        """
        selector = f"{KIND_SELECTOR_PREFIX}{kind}" if kind else None
        if tool_name in self.tool_deny or (selector is not None and selector in self.tool_deny):
            return f"Tool '{tool_name}' is denied by enforced permissions."
        return None

    def can_access_sem(self, entity: str, affordance: str) -> tuple[bool, str | None]:
        """Return ``(allowed, reason_if_denied)`` for an SEM affordance.

        Walks ``sem_access_rules`` for a rule matching either *entity*
        or ``"*"``. The first match wins; rules should be ordered most
        specific first when constructed.
        """
        for rule in self.sem_access_rules:
            if rule.entity != entity and rule.entity != "*":
                continue
            if affordance in rule.deny:
                return False, f"Affordance '{affordance}' on '{entity}' is denied."
            if self.clearance < rule.min_clearance:
                return (
                    False,
                    f"Affordance '{affordance}' on '{entity}' requires clearance "
                    f">= {rule.min_clearance} (you have {self.clearance}).",
                )
            return True, None
        return True, None

    @classmethod
    def from_yaml(cls, data: dict[str, Any] | None) -> AgentPermissions:
        """Build an ``AgentPermissions`` from a YAML-loaded dict.

        Accepted shape::

            permissions:
              clearance: 2
              tool_deny: [bash, write_file]
              tool_allow: [read_file, examine, say]   # optional allow-list
              sem_access:
                - entity: vault_terminal
                  deny: [delete_records]
                  min_clearance: 3
                - entity: "*"
                  deny: [self_destruct]
        """
        if not data:
            return cls()
        clearance = int(data.get("clearance", 0) or 0)
        tool_deny = frozenset(str(t) for t in (data.get("tool_deny") or []))
        raw_allow = data.get("tool_allow")
        tool_allow = frozenset(str(t) for t in raw_allow) if raw_allow is not None else None
        rules: list[SEMAccessRule] = []
        for raw in data.get("sem_access") or []:
            if not isinstance(raw, dict):
                continue
            rules.append(
                SEMAccessRule(
                    entity=str(raw.get("entity", "*")),
                    deny=tuple(str(d) for d in (raw.get("deny") or [])),
                    min_clearance=int(raw.get("min_clearance", 0) or 0),
                )
            )
        return cls(
            clearance=clearance,
            tool_deny=tool_deny,
            tool_allow=tool_allow,
            sem_access_rules=tuple(rules),
        )


# ---------------------------------------------------------------------------
# Perceived layer
# ---------------------------------------------------------------------------


@dataclass
class PerceivedAuthority:
    """A learned belief about another character's authority.

    ``score`` is in [0, 1]. ``observations`` counts how many outcomes
    have been folded in. The bio-stack updates this through
    ``PerceivedAuthorityTracker.observe`` rather than mutating it
    directly.
    """

    authority_id: str
    score: float = 0.5
    observations: int = 0


class PerceivedAuthorityTracker:
    """Lightweight EWMA tracker for perceived authority.

    Designed to be cheap enough to call from NAc on every outcome
    observation. Each ``observe`` call folds an outcome valence
    (``-1.0`` … ``+1.0``) into the running score with an
    exponentially-weighted moving average. ``alpha`` controls how
    quickly trust shifts.
    """

    def __init__(self, *, alpha: float = 0.2, default: float = 0.5) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        self._default = default
        self._table: dict[str, PerceivedAuthority] = {}

    def get(self, authority_id: str) -> PerceivedAuthority:
        """Return the current belief, creating it at the default if absent."""
        existing = self._table.get(authority_id)
        if existing is not None:
            return existing
        fresh = PerceivedAuthority(authority_id=authority_id, score=self._default)
        self._table[authority_id] = fresh
        return fresh

    def observe(self, authority_id: str, outcome_valence: float) -> PerceivedAuthority:
        """Fold one outcome into the running belief.

        ``outcome_valence`` is the bio-stack's signed valence for the
        outcome that followed the authority's directive: positive when
        the directive paid off, negative when it caused pain. Values
        outside [-1, 1] are clamped.
        """
        valence = max(-1.0, min(1.0, float(outcome_valence)))
        # Map [-1, 1] → [0, 1] before folding so the score domain stays
        # bounded and a directive that turns out neutral leaves the
        # belief unchanged.
        target = (valence + 1.0) / 2.0
        current = self.get(authority_id)
        new_score = (1.0 - self._alpha) * current.score + self._alpha * target
        current.score = new_score
        current.observations += 1
        return current

    def snapshot(self) -> dict[str, PerceivedAuthority]:
        """Return a copy of the current belief table for inspection."""
        return {k: PerceivedAuthority(k, v.score, v.observations) for k, v in self._table.items()}

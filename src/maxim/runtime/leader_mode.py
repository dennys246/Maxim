"""Leader-vs-follower role detection — thin wrapper around ``runtime/role.py``.

**C3 of config_unification.md, 2026-06-01.** This module is now a
thin translation layer over the unified detector in
:mod:`maxim.runtime.role`. The pre-C3 architecture had two
independent ``detect_role`` functions in two modules with different
decision orders + different file-extension assumptions; the
2026-06-01 Mac Mini regression hit divergence on day one.

After C3, the single source of truth lives in
:func:`maxim.runtime.role.detect_role`. This module's
:func:`detect_role` calls that and translates the returned
:data:`maxim.runtime.role.Role` into a :class:`RoleDecision`
(role + bind_host + reason).

**Compatibility shape preserved through 1.x:**

- :data:`RoleName` keeps the legacy ``client`` value alongside
  ``peer`` so callers comparing ``decision.role == "client"`` continue
  to work. ``role.py`` returns ``peer``; the wrapper returns
  ``client`` to existing consumers that haven't migrated. Drops in 1.2.
- :class:`RoleDecision` shape unchanged.

New consumers should call :func:`maxim.runtime.role.detect_role`
directly and use the :data:`maxim.runtime.role.Role` vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RoleName = Literal["leader", "client", "peer", "solo"]
"""Legacy enum: ``client`` is the pre-C3 term for ``peer``. New code
should use ``peer`` directly via :mod:`maxim.runtime.role`."""


@dataclass(frozen=True)
class RoleDecision:
    """Resolved role for this Maxim instance (legacy compat shape)."""

    role: RoleName
    bind_host: str  # address the local server should bind to
    reason: str  # human-readable signal that drove the decision

    @property
    def is_leader(self) -> bool:
        return self.role == "leader"


def _bind_host_for(role: str) -> str:
    """leader → ``0.0.0.0``; peer/solo → ``127.0.0.1``."""
    return "0.0.0.0" if role == "leader" else "127.0.0.1"


def _reason_for(role: str, source: str) -> str:
    """Render a human-readable ``RoleDecision.reason`` from the source."""
    source_msgs = {
        "env_var": f"MAXIM_ROLE={role}",
        "config_json": f"config.json::role={role}",
        "mesh_yml": "mesh.yml exists",
        "cloudflared": "cloudflared config found",
        "peer_yml": "peer.yml exists (legacy)",
        "cli_flag": "--llm flag present",
        "default": f"default ({role})",
    }
    return source_msgs.get(source, f"{source} → {role}")


def detect_role() -> RoleDecision:
    """Resolve the role of this Maxim instance.

    Thin wrapper around :func:`maxim.runtime.role.detect_role`.
    Translates the canonical :data:`maxim.runtime.role.Role`
    (leader|peer|solo) into a :class:`RoleDecision` with
    ``bind_host`` and ``reason``.

    For back-compat with pre-C3 callers, returns the legacy
    ``client`` string for what ``role.py`` calls ``peer``. This
    coercion will be removed in 1.2.
    """
    from maxim.runtime.role import detect_role as _detect

    role, source = _detect(argv=None)

    # Back-compat: pre-C3 vocabulary used "client"; preserve for
    # existing consumers that branch on .role == "client". Drops in 1.2.
    legacy_role: RoleName
    if role == "peer":
        legacy_role = "client"
    else:
        legacy_role = role  # type: ignore[assignment]

    return RoleDecision(
        role=legacy_role,
        bind_host=_bind_host_for(role),
        reason=_reason_for(role, source),
    )


__all__ = ["RoleDecision", "RoleName", "detect_role"]

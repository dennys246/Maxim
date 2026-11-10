"""Leader-vs-follower role detection (a thin wrapper around ``runtime/role.py``), and the one owner of a leader's
services (#1610).

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

**A leader's services** (#1610, owner decisions 2026-11-09): the LeaderProxy, the cloudflared tunnel daemon and
the leader heartbeat have ONE owner, the ``maxim`` CLI leader process that binds the default-port proxy
(:func:`claim_leader_services`, called only by ``cli.py::_early_leader_boot``). The owner starts the tunnel and
heartbeat when it builds its LLM router (:func:`start_claimed_leader_services`, first thing in
``lane_backends.build_primary_router``), whatever the LLM path then decides: a fresh spawn, a reused server, an opt-out
or a build that raises. Never an owner: ``maxim serve`` (the console), a process that shares another's proxy (a sub-sim
on the leader box), a Python-API process, and a short CLI command that builds no router. The LLM spawn path starts none
of them. The invariant and its guard: docs/agents/llm-routing.md (#1610).
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

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


# ─── a leader's services (#1610) ──────────────────────────────────────────────────────────────────

# The pid that claimed the services: a forked child inherits the module but not the claim.
_claimed_by_pid: int | None = None
_started_by_pid: int | None = None
_services_lock = threading.Lock()  # one claim and one start per process, whatever thread builds a router


def claim_leader_services(bind_host: str) -> Any:
    """Start the LeaderProxy for this leader process and, when this process now RUNS the default-port proxy, claim the
    leader's services for it. Returns what ``leader_proxy.start_leader_proxy_for_role`` returns (it fails closed).

    Not claimed: an authenticated proxy elsewhere holds the port (another maxim process on the box owns the leader),
    or this process bound a non-default ``MAXIM_LEADER_PROXY_PORT`` (a harness child, a test).
    """
    global _claimed_by_pid
    from maxim.runtime.leader_proxy import leader_proxy_running, start_leader_proxy_for_role

    with _services_lock:
        server = start_leader_proxy_for_role(bind_host)
        if leader_proxy_running():
            _claimed_by_pid = os.getpid()
    return server


def owns_leader_services() -> bool:
    """True when THIS process (not a fork of it) claimed the leader's services."""
    return _claimed_by_pid is not None and _claimed_by_pid == os.getpid()


def start_claimed_leader_services(log: Any | None = None) -> None:
    """For the owner only, once per process: the cloudflared tunnel daemon and the leader heartbeat. A no-op in any
    process that did not claim the services."""
    global _started_by_pid
    with _services_lock:
        if not owns_leader_services() or _started_by_pid == os.getpid():
            return
        _started_by_pid = os.getpid()
    try:
        from maxim.tunnel.daemon_spawner import auto_spawn_tunnel_daemon

        auto_spawn_tunnel_daemon(log)
    except Exception as e:  # e.g. PermissionError on /etc/cloudflared: a router build never fails on the tunnel
        logger.warning("leader tunnel daemon did not start: %s", e)
    try:
        from maxim.runtime.heartbeat import get_heartbeat_monitor

        get_heartbeat_monitor().start()
    except Exception as e:  # the heartbeat only samples health: a leader runs without it, loudly
        logger.warning("leader heartbeat did not start: %s", e)


__all__ = [
    "RoleDecision",
    "RoleName",
    "claim_leader_services",
    "detect_role",
    "owns_leader_services",
    "start_claimed_leader_services",
]

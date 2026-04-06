"""Leader-vs-follower role detection for Maxim clusters (multi-LLM Phase 6b).

One Maxim instance can act as a leader — it auto-spawns an LLM server and
exposes it to peers (via LAN bind + optional Cloudflare tunnel). Followers
and solo instances still spawn a server for their own use, but bind to
loopback only.

Signals, in priority order:
  1. MAXIM_ROLE env var (explicit)            → "leader" | "client" | "solo"
  2. /etc/cloudflared/config.yml exists        → leader (implicit, systemd)
  3. ~/.cloudflared/config.yml exists           → leader (implicit, user-space)
  4. default                                   → "solo"

The leader distinction only changes bind semantics and peer advertising —
the process itself is still a normal Maxim instance with its own CLI + loop.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

RoleName = Literal["leader", "client", "solo"]


@dataclass(frozen=True)
class RoleDecision:
    """Resolved role for this Maxim instance."""

    role: RoleName
    bind_host: str  # address the local server should bind to
    reason: str     # human-readable signal that drove the decision

    @property
    def is_leader(self) -> bool:
        return self.role == "leader"


def _env_role() -> str | None:
    raw = os.environ.get("MAXIM_ROLE", "").strip().lower()
    if raw in ("leader", "client", "solo"):
        return raw
    return None


def _cloudflared_config_exists() -> Path | None:
    """Return the path to cloudflared's config.yml if present.

    Prefers /etc/cloudflared/ (systemd service, production) over
    ~/.cloudflared/ (user-space, development). The systemd service
    reads from /etc/, so that's the authoritative location.
    """
    candidates = [
        Path("/etc/cloudflared/config.yml"),
        Path.home() / ".cloudflared" / "config.yml",
    ]
    for path in candidates:
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


def detect_role() -> RoleDecision:
    """Resolve the role of this Maxim instance.

    Returns a RoleDecision with:
    - `role`:   leader / client / solo
    - `bind_host`: what IP the local-spawned server should listen on
    - `reason`: which signal fired

    Bind semantics:
    - leader → 0.0.0.0 (reachable by peers + tunnel)
    - client → 127.0.0.1 (follower shouldn't host anything public)
    - solo   → 127.0.0.1 (safe default for single-machine use)
    """
    env_role = _env_role()
    if env_role == "leader":
        return RoleDecision(role="leader", bind_host="0.0.0.0", reason="MAXIM_ROLE=leader")
    if env_role == "client":
        return RoleDecision(role="client", bind_host="127.0.0.1", reason="MAXIM_ROLE=client")
    if env_role == "solo":
        return RoleDecision(role="solo", bind_host="127.0.0.1", reason="MAXIM_ROLE=solo")

    tunnel_cfg = _cloudflared_config_exists()
    if tunnel_cfg is not None:
        return RoleDecision(
            role="leader",
            bind_host="0.0.0.0",
            reason=f"cloudflared config found at {tunnel_cfg}",
        )

    return RoleDecision(role="solo", bind_host="127.0.0.1", reason="default (no MAXIM_ROLE set, no cloudflared config)")


__all__ = ["RoleDecision", "RoleName", "detect_role"]

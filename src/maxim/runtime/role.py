"""Explicit role detection for Maxim processes (Plan 2 R2a).

One Maxim process has exactly one role: ``leader``, ``peer``, or ``solo``.
The role is detected once at startup, exported to ``MAXIM_ROLE``, and
downstream code reads the env var — never re-detects. This replaces the
implicit inference scattered across ``peer/config.py::read_peer_config``,
``lane_backends._apply_local_llm_override``, and ``cli.py::main``.

Decision order (first match wins):

1. ``MAXIM_ROLE`` env var explicitly set to ``leader`` | ``peer`` | ``solo``
2. ``mesh.yml`` exists (Plan 4; accepted here for forward compatibility) → ``peer``
3. ``peer.yml`` exists (legacy) → ``peer``
4. ``--llm <local>`` CLI flag + no peer config → ``solo``
5. Default → ``leader``

``detect_role()`` is called from ``cli.py::main()`` BEFORE subcommand
dispatch. If you add a new logging-dependent feature, do not move the
call site downstream — subcommands short-circuit before reaching the
sim loop (see commit ``c8a07e9`` for the matching ``configure_logging``
early-call fix).

Persisted state is split per role. ``~/.maxim/util/active_llm_model.txt``
migrates to ``active_llm_model.{role}.txt`` on first startup after upgrade.
The four pre-existing states are covered in
``migrate_persisted_model_file``.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from maxim.utils.structured_logging import log_structured

logger = logging.getLogger(__name__)

Role = Literal["leader", "peer", "solo"]
RoleSource = Literal["env_var", "mesh_yml", "peer_yml", "cli_flag", "default"]

_VALID_ROLES = ("leader", "peer", "solo")


def _mesh_yml_path() -> str:
    """Path to the future mesh.yml config (Plan 4). Accepted here for
    forward-compat so Plan 4's rollout doesn't require touching R2a."""
    import platform
    from pathlib import Path

    if platform.system() == "Windows":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return str(base / "maxim" / "mesh.yml")
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return str(base / "maxim" / "mesh.yml")


def _peer_yml_exists() -> bool:
    try:
        from maxim.peer.config import peer_config_path

        return peer_config_path().is_file()
    except Exception:
        return False


def _mesh_yml_exists() -> bool:
    try:
        from pathlib import Path

        return Path(_mesh_yml_path()).is_file()
    except Exception:
        return False


def detect_role(argv: list[str] | None = None) -> tuple[Role, RoleSource]:
    """Resolve this process's role. Pure function — no env mutation, no logging.

    Callers (``cli.py::main``) should call :func:`apply_role` to export
    ``MAXIM_ROLE`` and emit the structured log event.
    """
    raw = os.environ.get("MAXIM_ROLE", "").strip().lower()
    if raw in _VALID_ROLES:
        return raw, "env_var"  # type: ignore[return-value]

    if _mesh_yml_exists():
        return "peer", "mesh_yml"

    if _peer_yml_exists():
        return "peer", "peer_yml"

    if argv is not None and _has_local_llm_flag(argv):
        return "solo", "cli_flag"

    return "leader", "default"


def _has_local_llm_flag(argv: list[str]) -> bool:
    """Return True iff argv contains ``--llm <local-profile>`` or ``--language-model``.

    ``--llm claude-sonnet`` counts as local-ish — the CLI flag is present, and
    no peer config → solo. Cloud-only distinction is not meaningful here;
    the spec treats any explicit ``--llm`` flag + no peer config as solo.
    """
    flags = {"--llm", "--language-model", "--model"}
    for i, tok in enumerate(argv):
        if tok in flags and i + 1 < len(argv):
            return True
        for f in flags:
            if tok.startswith(f + "="):
                return True
    return False


def apply_role(role: Role, source: RoleSource) -> None:
    """Export ``MAXIM_ROLE`` and log ``role_detected``. Idempotent."""
    os.environ["MAXIM_ROLE"] = role
    log_structured(
        logger,
        logging.INFO,
        event="role_detected",
        data={"role": role, "source": source},
    )


def detect_and_apply_role(argv: list[str] | None = None) -> Role:
    """Combined detect + export + log. Call once from ``cli.py::main``."""
    role, source = detect_role(argv)
    apply_role(role, source)
    migrate_persisted_model_file(role)
    return role


def migrate_persisted_model_file(role: Role) -> None:
    """Rename legacy ``active_llm_model.txt`` to role-suffixed form.

    Four pre-existing user states (per plan spec):
    - peer role → delete old file (peer doesn't own local model state)
    - solo role → rename to ``.solo.txt``
    - leader role → rename to ``.leader.txt``
    - unclear/default → rename to ``.leader.txt`` (conservative)

    Best-effort. Never raises.
    """
    try:
        from maxim.utils.paths import resolve_user_state

        old_path = resolve_user_state("util/active_llm_model.txt")
        if not old_path.is_file():
            return
        new_path = resolve_user_state(f"util/active_llm_model.{role}.txt")
        if role == "peer":
            old_path.unlink()
            logger.warning(
                "removed legacy active_llm_model.txt (peer role does not own model state): %s",
                old_path,
            )
            log_structured(
                logger,
                logging.WARNING,
                event="persisted_model_migrated",
                data={"old_path": str(old_path), "new_path": None, "role": role},
            )
            return
        old_path.replace(new_path)
        log_structured(
            logger,
            logging.INFO,
            event="persisted_model_migrated",
            data={
                "old_path": str(old_path),
                "new_path": str(new_path),
                "role": role,
            },
        )
    except Exception as e:
        logger.debug("persisted model migration skipped (%s)", e)


__all__ = [
    "Role",
    "RoleSource",
    "apply_role",
    "detect_and_apply_role",
    "detect_role",
    "migrate_persisted_model_file",
]

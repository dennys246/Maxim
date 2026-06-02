"""Explicit role detection for Maxim processes (C3 of config_unification.md).

One Maxim process has exactly one role: ``leader``, ``peer``, or ``solo``.
The role is detected once at startup, exported to ``MAXIM_ROLE``, and
downstream code reads the env var — never re-detects.

**C3 fold (CR2 + C-3 from the pre-implementation two-lens review):**
the original two-detector architecture (this module + ``leader_mode.py``)
was split because they grew from different motivating cases. Both ran
at startup and could silently disagree. The 2026-06-01 Mac Mini
regression hit divergence on day one — a stale ``peer.yml`` made this
module return ``peer`` while ``leader_mode``'s cloudflared check
returned ``leader``.

After C3, this module is the single source of truth.
``runtime/leader_mode.py::detect_role`` is a thin wrapper that
translates this module's :class:`Role` into ``RoleDecision``
(role + bind_host + reason). All consumers route through the wrapper
or through this module's :func:`detect_role` directly.

Decision order (first match wins) — pinned in
``config_unification.md`` C3 section:

1. ``MAXIM_ROLE`` env var explicitly set to ``leader`` | ``peer`` | ``solo``
2. ``config.json::role`` ∈ ``{leader, peer, solo}`` (NEW per C3)
3. ``mesh.yml`` exists → ``peer``
4. ``~/.cloudflared/config.{yml,yaml}`` OR ``/etc/cloudflared/config.{yml,yaml}``
   exists → ``leader`` (MOVED UP from ``leader_mode.py`` + extension
   widened to fix Mac Mini Trigger #2)
5. ``peer.yml`` exists → ``peer`` (DEMOTED below cloudflared — fixes
   Mac Mini Trigger #3, the stale-peer.yml-on-now-leader case)
6. ``--llm <local>`` CLI flag + no peer/mesh/cloudflared signal → ``solo``
7. Default → ``leader``

``detect_role()`` is called from ``cli.py::main()`` BEFORE subcommand
dispatch. If you add a new logging-dependent feature, do not move the
call site downstream — subcommands short-circuit before reaching the
sim loop (see commit ``c8a07e9`` for the matching ``configure_logging``
early-call fix).

Persisted state is split per role.
``~/.maxim/util/active_llm_model.txt`` migrates to
``active_llm_model.{role}.txt`` on first startup after upgrade.
The four pre-existing states are covered in
:func:`migrate_persisted_model_file`.
"""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import Literal

from maxim.utils.structured_logging import log_structured

logger = logging.getLogger(__name__)

Role = Literal["leader", "peer", "solo"]
RoleSource = Literal[
    "env_var",
    "config_json",
    "mesh_yml",
    "cloudflared",
    "peer_yml",
    "cli_flag",
    "default",
]

_VALID_ROLES: tuple[str, ...] = ("leader", "peer", "solo")

# Fix #2 (R2 review): subcommands have their own semantics and do NOT
# drive role selection via ``--llm``. If ``argv[0]`` is one of these,
# ``_has_local_llm_flag`` short-circuits to False so e.g.
# ``maxim tunnel --llm foo`` on a leader machine doesn't falsely flip
# the role to solo.
_SUBCOMMAND_NAMES = frozenset({"doctor", "peer", "tunnel"})


# ─────────────────────────────────────────────────────────────────────────────
# Signal probes
# ─────────────────────────────────────────────────────────────────────────────


def _mesh_yml_path() -> str:
    """Path to the future mesh.yml config (Plan 4). Accepted here for
    forward-compat so Plan 4's rollout doesn't require touching R2a."""
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
        return Path(_mesh_yml_path()).is_file()
    except Exception:
        return False


def _cloudflared_config_exists() -> Path | None:
    """Return the path to cloudflared's config file if present.

    Searches both ``/etc/cloudflared/`` (systemd service, production)
    and ``~/.cloudflared/`` (user-space, development), checking BOTH
    ``config.yml`` AND ``config.yaml`` extensions.

    Fold from pre-implementation CR2 review: the original
    ``leader_mode.py`` only checked ``.yml``, which was Mac Mini
    Trigger #2 — cloudflared sometimes auto-writes ``.yaml`` and the
    leader role failed to detect. C3 widens to accept either extension
    AND promotes this signal above ``peer.yml`` so a stale ``peer.yml``
    on a now-leader machine doesn't override a real provisioning
    signal (Mac Mini Trigger #3).
    """
    candidates = [
        Path("/etc/cloudflared/config.yml"),
        Path("/etc/cloudflared/config.yaml"),
        Path.home() / ".cloudflared" / "config.yml",
        Path.home() / ".cloudflared" / "config.yaml",
    ]
    for path in candidates:
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


def _config_json_role() -> Role | None:
    """Read ``config.json::role`` if present and valid; return None otherwise.

    Reads via :func:`maxim.runtime.config_loader.load_config`. The
    loader returns a defaults-only ``MaximConfig`` when the file is
    missing, so ``cfg.role is None`` (the default) means
    "config.json doesn't pin a role" and we fall through to lower
    ranks. Any parse error is silently swallowed at this layer —
    role detection runs early at startup and a config-parse failure
    would block ``maxim doctor``, which is the recovery path the
    operator needs.
    """
    try:
        from maxim.runtime.config_loader import load_config

        cfg = load_config()
    except Exception as e:
        logger.debug("role: config.json read failed (%s); falling through", e)
        return None
    if cfg.role in _VALID_ROLES:
        return cfg.role  # type: ignore[return-value]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main detection entry point — the seven-rank order
# ─────────────────────────────────────────────────────────────────────────────


def detect_role(argv: list[str] | None = None) -> tuple[Role, RoleSource]:
    """Resolve this process's role. Pure function — no env mutation, no logging.

    Callers (``cli.py::main``) should call :func:`apply_role` to export
    ``MAXIM_ROLE`` and emit the structured log event.

    Implements the seven-rank C3 decision order (see module docstring).
    """
    # Rank 1: explicit MAXIM_ROLE env var
    raw = os.environ.get("MAXIM_ROLE", "").strip().lower()
    if raw in _VALID_ROLES:
        return raw, "env_var"  # type: ignore[return-value]
    if raw == "client":
        # Legacy term from the pre-C3 leader_mode.py vocabulary; coerce
        # silently here (env-var path) — the config.json loader's
        # _coerce_role logs the WARNING when the same value appears in
        # the JSON file.
        return "peer", "env_var"

    # Rank 2: config.json::role
    cfg_role = _config_json_role()
    if cfg_role is not None:
        return cfg_role, "config_json"

    # Rank 3: mesh.yml exists → peer
    if _mesh_yml_exists():
        return "peer", "mesh_yml"

    # Rank 4: cloudflared config exists → leader
    # (PROMOTED above peer.yml per CR2 fold; widened to .yaml per
    # Mac Mini Trigger #2)
    if _cloudflared_config_exists() is not None:
        return "leader", "cloudflared"

    # Rank 5: peer.yml exists → peer (DEMOTED below cloudflared)
    if _peer_yml_exists():
        return "peer", "peer_yml"

    # Rank 6: --llm <local> flag → solo
    if argv is not None and _has_local_llm_flag(argv):
        return "solo", "cli_flag"

    # Rank 7: default → leader
    return "leader", "default"


def _has_local_llm_flag(argv: list[str]) -> bool:
    """Return True iff argv contains ``--llm <profile>`` or ``--language-model``.

    Fix #2 (R2 review): skips the scan entirely when ``argv[0]`` is a known
    subcommand (``doctor``, ``peer``, ``tunnel``). Without this guard,
    ``maxim tunnel --llm foo`` on a leader would falsely trigger solo
    detection because the tunnel subcommand carries a ``--llm`` flag with
    unrelated semantics. Subcommands do not drive role selection.
    """
    if argv and argv[0] in _SUBCOMMAND_NAMES:
        return False
    flags = {"--llm", "--language-model", "--model"}
    for i, tok in enumerate(argv):
        if tok in flags and i + 1 < len(argv):
            return True
        for f in flags:
            if tok.startswith(f + "="):
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Apply role + emit telemetry
# ─────────────────────────────────────────────────────────────────────────────


def apply_role(role: Role, source: RoleSource) -> None:
    """Export ``MAXIM_ROLE`` and log ``role_detected``. Idempotent.

    Fix #11 (R2 review): emits the decision-rule field as ``role_source``
    (not ``source``) so it does not collide with the StructuredFormatter
    top-level ``s`` short-key for the logger module name.

    C3 fold: adds ``config_json_present`` boolean for telemetry on
    config.json adoption rate.
    """
    os.environ["MAXIM_ROLE"] = role

    # Telemetry hint: did config.json contribute to the decision?
    config_json_present = False
    try:
        from maxim.runtime.config_loader import config_path

        config_json_present = config_path().is_file()
    except Exception:
        pass

    log_structured(
        logger,
        logging.INFO,
        event="role_detected",
        data={
            "role": role,
            "role_source": source,
            "config_json_present": config_json_present,
        },
    )


def detect_and_apply_role(argv: list[str] | None = None) -> Role:
    """Combined detect + export + log. Call once from ``cli.py::main``."""
    role, source = detect_role(argv)
    apply_role(role, source)
    migrate_persisted_model_file(role)
    _emit_role_divergence_compat_event()
    return role


def _emit_role_divergence_compat_event() -> None:
    """Emit the ``role_divergence`` event as a back-compat shim.

    Pre-C3 this event fired at WARNING when ``leader_mode.detect_role``
    disagreed with this module. After C3 the two functions cannot
    disagree (leader_mode is now a wrapper around this module), so
    divergence is structurally impossible.

    N-2 fold from the pre-implementation review: keep emitting the
    event for one minor with ``deprecated: true`` data and at DEBUG
    level so external dashboards subscribing to the event continue to
    see it (avoiding a silent column-drop in 1.0). Drops in 1.2.
    """
    try:
        log_structured(
            logger,
            logging.DEBUG,
            event="role_divergence",
            data={
                "deprecated": True,
                "reason": "single detector (C3 of config_unification.md)",
            },
        )
    except Exception:
        pass


def migrate_persisted_model_file(role: Role) -> None:
    """Rename legacy ``active_llm_model.txt`` to role-suffixed form.

    Four pre-existing user states (per plan spec):
    - peer role → delete old file (peer doesn't own local model state)
    - solo role → rename to ``.solo.txt``
    - leader role → rename to ``.leader.txt``
    - unclear/default → rename to ``.leader.txt`` (conservative)

    Fix #4 (R2 review): failures log at WARNING, not DEBUG. A real
    filesystem error during migration is operationally significant.
    Fix #5 (R2 review): checks ``new_path.exists()`` before rename so
    an existing role-specific file is never silently clobbered.
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
        # Fix #5: destination collision — preserve the newer role-specific
        # file, delete the legacy one, warn the operator.
        if new_path.exists():
            old_path.unlink()
            logger.warning(
                "legacy active_llm_model.txt coexisted with %s; removed legacy, preserved role-specific file",
                new_path,
            )
            log_structured(
                logger,
                logging.WARNING,
                event="persisted_model_migrated",
                data={
                    "old_path": str(old_path),
                    "new_path": str(new_path),
                    "role": role,
                    "collision": True,
                },
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
        # Fix #4: promote from debug → warning.
        logger.warning("persisted model migration failed: %s", e)
        try:
            log_structured(
                logger,
                logging.WARNING,
                event="persisted_model_migration_failed",
                data={"role": role, "error": f"{type(e).__name__}: {e}"},
            )
        except Exception:
            pass


__all__ = [
    "Role",
    "RoleSource",
    "apply_role",
    "detect_and_apply_role",
    "detect_role",
    "migrate_persisted_model_file",
]

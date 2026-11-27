"""Which router an ExecAgent runs on (#1228): the ``BUILD_OWN`` sentinel for ``shared_router=``, the
``agent_profiles.exec_agent`` override read, and the two once-only warnings about a router handed in.

``exec_agent`` re-exports :data:`BUILD_OWN`, so ``maxim.agents.exec_agent.BUILD_OWN`` stays the public name. These
helpers live in their own module because #1228 also rewrote part of the floored ``simulation/orchestrator.py``; the
coverage lint's split rule (SF-2) charges an existing file that grows by 25 or more statements in such a PR as the
floored file's successor, and new code belongs in a new module there.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from maxim.models.language.router import load_llm_config

logger = logging.getLogger(__name__)


class _BuildOwn:
    """The type of :data:`BUILD_OWN` (one instance)."""

    def __repr__(self) -> str:
        return "BUILD_OWN"


# ``shared_router=`` (REQUIRED, #1228): a router means "run on this one"; ``None`` means "no LLM: build none"; and
# ``BUILD_OWN`` means "build one lazily from ``load_llm_config()``" (the CLI's agentic path, which asks for download
# consent there). A router variable that happens to be None therefore builds nothing: the old default (None = build
# your own) served the narrator's profile to a dual-LLM AUT.
BUILD_OWN = _BuildOwn()
# (agent_profiles.exec_agent override, the handed-in router's profile) pairs already warned about (#1228).
_BYPASSED_OVERRIDES_WARNED: set[tuple[str, str]] = set()


def override_in(cfg: Any) -> str | None:
    """The ``agent_profiles.exec_agent`` (or ``executive``) override in ``cfg``, or None."""
    profiles = getattr(cfg, "agent_profiles", {})
    if not isinstance(profiles, dict):
        return None
    for key in ("exec_agent", "executive"):
        value = profiles.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


# (llm.json identity, override) of the last read, or None: re-read only when that file changed (#1228).
_OVERRIDE_READ: tuple[tuple[str, int, int] | None, str | None] | None = None
# Set once the "handed None: no goal proposals" WARNING has been logged in this process (#1228).
_NO_ROUTER_START_WARNED = False


def _llm_json_identity() -> tuple[str, int, int] | None:
    """(path, st_mtime_ns, st_size) of the llm.json ``load_llm_config`` would read (the first existing discovery
    candidate), or None when there is none. ``agent_profiles`` lives there, not in config.json, and no llm.json writer
    (the downloader, a hand edit) moves ``config_generation()``, so the file itself is the key."""
    from maxim.models.language.config import llm_config_file

    path = llm_config_file()  # the loader's own resolver (#1806 PR 0 round 5 N1)
    if path is None:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (path, st.st_mtime_ns, st.st_size)


def configured_override() -> str | None:
    """The process's ``agent_profiles.exec_agent`` override (the bypass warning's input, #1228), read again only when
    the llm.json it comes from changed (path, mtime or size): a constructor handed a router does not reload an
    unchanged file, and an edit (the downloader's write, a hand edit, in a long-lived ``maxim serve`` too) is seen by
    the next one. ``load_llm_config`` itself does not cache; this is the warning's own read. None when there is none
    or the config is unreadable."""
    global _OVERRIDE_READ
    identity = _llm_json_identity()
    if _OVERRIDE_READ is not None and _OVERRIDE_READ[0] == identity:
        return _OVERRIDE_READ[1]
    try:
        override = override_in(load_llm_config())
    except Exception as e:  # an unreadable config is load_llm_config's to report on the build path
        logger.debug("ExecAgent: could not read the config to check the exec_agent override: %s", e)
        override = None
    _OVERRIDE_READ = (identity, override)
    return override


def warn_if_profile_override_bypassed(router: Any) -> None:
    """One WARNING per (override, router profile) pair when the config names an ``agent_profiles.exec_agent``
    override that a handed-in router does not serve: the override applies only to a router the agent builds itself
    (``ExecAgent._resolve_profile_override``), so a caller that hands one in bypasses it (#1228)."""
    served = getattr(getattr(router, "cfg", None), "profile", None)
    override = configured_override()
    if not override or override == served:
        return
    key = (override, str(served))
    if key in _BYPASSED_OVERRIDES_WARNED:
        return
    _BYPASSED_OVERRIDES_WARNED.add(key)
    logger.warning(
        "agent_profiles.exec_agent names %r, but this agent was handed a router serving %r: the override applies "
        "only to a router ExecAgent builds itself, so it is not used here (#1228)",
        override,
        served,
    )


def warn_no_router_start_once() -> None:
    """The once-per-process WARNING for an ExecAgent started with ``shared_router=None`` (``ExecAgent.on_start``):
    None meant the lazy build before #1228, so the message names ``BUILD_OWN``."""
    global _NO_ROUTER_START_WARNED
    if _NO_ROUTER_START_WARNED:
        return
    _NO_ROUTER_START_WARNED = True
    logger.warning(
        "ExecAgent started with shared_router=None: it has no LLM and proposes no goals. None meant a lazy "
        "router build before #1228; pass maxim.agents.exec_agent.BUILD_OWN for that behaviour"
    )

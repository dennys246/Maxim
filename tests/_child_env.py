"""Strip operator overrides from a test's environment without stripping the suite's isolation.

``tests/conftest.py`` exports ``MAXIM_CLOUDFLARED_SYSTEM_DIR`` into its isolation root at import, the way it exports
``HOME``, and points the reader's in-process default at the same dir. So in the TEST process an in-process strip of
every ``MAXIM_*`` is harmless. A CHILD is a fresh process with the real ``/etc/cloudflared`` default, so it needs the
variable itself: a child env built from scratch beside its own ``HOME`` takes :func:`isolation_env`, a copy without
the operator's overrides is :func:`without_maxim_overrides`, and an in-process strip that a later child would inherit
is :func:`drop_maxim_overrides`. Without it, the child's role detection reads the host's
``/etc/cloudflared/config.yml``: ``leader`` on a leader dev box, invisible on CI (2026-11-05).
``tests/unit/test_child_env_routing.py`` refuses a hand-written strip or a child env without the isolation (a
heuristic scan; its docstring lists the limits).
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping

# Variables tests/conftest.py exports to isolate the suite from the host; never an operator override.
ISOLATION_VARS: tuple[str, ...] = ("MAXIM_CLOUDFLARED_SYSTEM_DIR",)


def isolation_env() -> dict[str, str]:
    """The suite's isolation variables as this process has them, for a child env built from scratch beside HOME."""
    return {k: os.environ[k] for k in ISOLATION_VARS if k in os.environ}


def without_maxim_overrides() -> dict[str, str]:
    """A copy of ``os.environ`` minus every ``MAXIM_*`` operator override, keeping the suite's isolation variables."""
    return {k: v for k, v in os.environ.items() if not k.startswith("MAXIM_") or k in ISOLATION_VARS}


def drop_maxim_overrides(environ: MutableMapping[str, str]) -> None:
    """Delete every ``MAXIM_*`` operator override from ``environ`` in place, keeping the suite's isolation variables.

    For an in-process run that snapshots and restores ``os.environ`` itself.
    """
    for key in [k for k in environ if k.startswith("MAXIM_") and k not in ISOLATION_VARS]:
        del environ[key]


def maxim_overrides(environ: MutableMapping[str, str]) -> dict[str, str]:
    """The ``MAXIM_*`` operator overrides in ``environ``: what a run set or leaked, without the suite's isolation."""
    return {k: v for k, v in environ.items() if k.startswith("MAXIM_") and k not in ISOLATION_VARS}

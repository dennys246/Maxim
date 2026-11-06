"""Strip operator overrides from a test's environment without stripping the suite's isolation.

``tests/conftest.py`` exports ``MAXIM_CLOUDFLARED_SYSTEM_DIR`` into its isolation root at import, the way it exports
``HOME``. The suite's habit is "every ``MAXIM_*`` is an operator override: drop it all", in two forms: an in-process
loop that deletes every ``MAXIM_*`` from ``os.environ`` around a run, and a child env built from scratch beside its own
``HOME``. Both silently remove the isolation, and the next role read in that window sees the host's
``/etc/cloudflared/config.yml``: ``leader`` on a leader dev box, ``solo`` on CI (2026-11-05). Every such site goes
through this module; ``tests/unit/test_child_env_routing.py`` refuses one that does not.
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

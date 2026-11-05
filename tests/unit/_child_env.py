"""The environment for a test's child ``maxim`` process: no operator overrides, the suite's isolation kept.

A child built from ``os.environ`` minus every ``MAXIM_*`` variable drops the operator's overrides, as intended, but
would also drop ``MAXIM_CLOUDFLARED_SYSTEM_DIR``, which ``tests/conftest.py`` exports into its isolation root the way
it exports ``HOME``. Without it the child reads the host's ``/etc/cloudflared/config.yml`` and, on a leader dev box,
detects ``leader`` (2026-11-05). A child that builds its env from scratch passes ``isolation_env()`` beside ``HOME``.
"""

from __future__ import annotations

import os

# Variables tests/conftest.py exports to isolate the suite from the host; never an operator override.
ISOLATION_VARS: tuple[str, ...] = ("MAXIM_CLOUDFLARED_SYSTEM_DIR",)


def isolation_env() -> dict[str, str]:
    """The suite's isolation variables as this process has them (conftest sets them at import)."""
    return {k: os.environ[k] for k in ISOLATION_VARS if k in os.environ}


def without_maxim_overrides() -> dict[str, str]:
    """``os.environ`` minus every ``MAXIM_*`` operator override, keeping the suite's isolation variables."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("MAXIM_")}
    env.update(isolation_env())
    return env

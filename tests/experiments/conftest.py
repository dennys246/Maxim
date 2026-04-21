"""Conftest for experiments — pre-load maxim to avoid circular import during collection."""

import maxim.agents.bus  # noqa: F401 — forces full module resolution before tools imports

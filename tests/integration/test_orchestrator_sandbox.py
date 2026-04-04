"""Integration test for the orchestrator's sandbox setup.

Regression guard for the class of bug that hid for weeks: a silent
failure in ``start_simulation_mode``'s sandbox-creation block (caught
by a broad try/except and logged at DEBUG) left the AUT running
without a sandbox, without pain honeypots, and without confined
``allowed_dirs``.

These tests exercise the extracted ``_setup_sim_sandbox`` helper
directly — no LLM, no agent loop, no threads. They enforce the
ordering contract (pain bus before sandbox) and assert the helper
returns non-None results for every supported backend.
"""

from __future__ import annotations

import pytest

from maxim.simulation.orchestrator import _setup_sim_sandbox
from maxim.simulation.container_runner import check_docker_available
from maxim.simulation.sandbox import (
    DockerSandbox,
    PainTriggerLayer,
    TmpdirSandbox,
)


# ─────────────────────────────────────────────────────────────────────────
# tmpdir backend — always available
# ─────────────────────────────────────────────────────────────────────────


class TestTmpdirBackend:
    def test_returns_all_three_components(self):
        """Regression: none of the return values should be None when
        tmpdir is forced (no external dependencies)."""
        sandbox, root, pain_bus = _setup_sim_sandbox(
            backend="tmpdir", populate=False,
        )
        try:  # noqa: SIM105
            assert sandbox is not None, "sim_sandbox must not be None"
            assert root is not None, "sandbox_root must not be None"
            assert pain_bus is not None, "aut_pain_bus must not be None"
        finally:
            if sandbox is not None:
                sandbox.cleanup()

    def test_sandbox_wraps_tmpdir(self):
        sandbox, _, _ = _setup_sim_sandbox(backend="tmpdir", populate=False)
        try:  # noqa: SIM105
            assert isinstance(sandbox, PainTriggerLayer)
            inner = sandbox._sandbox
            assert isinstance(inner, TmpdirSandbox)
        finally:
            if sandbox is not None:
                sandbox.cleanup()

    def test_pain_bus_connected_to_sandbox(self):
        """The pain bus instance must be the SAME object passed to the
        sandbox. Otherwise pain signals from sensitive-file access
        never reach the AUT's hippocampus."""
        sandbox, _, pain_bus = _setup_sim_sandbox(
            backend="tmpdir", populate=False,
        )
        try:  # noqa: SIM105
            assert sandbox._pain_bus is pain_bus, (
                "sandbox's pain_bus reference must match the returned "
                "pain_bus — otherwise the sandbox can't route signals"
            )
        finally:
            if sandbox is not None:
                sandbox.cleanup()

    def test_populate_creates_honeypot_files(self):
        sandbox, root, _ = _setup_sim_sandbox(
            backend="tmpdir", populate=True,
        )
        try:  # noqa: SIM105
            import os
            # A few representative honeypots that should exist
            assert os.path.exists(os.path.join(root, "etc", "passwd"))
            assert os.path.exists(os.path.join(root, "home", "user", ".env"))
            assert os.path.exists(os.path.join(root, "var", "log", "auth.log"))
        finally:
            if sandbox is not None:
                sandbox.cleanup()

    def test_no_populate_skips_honeypots(self):
        sandbox, root, _ = _setup_sim_sandbox(
            backend="tmpdir", populate=False,
        )
        try:  # noqa: SIM105
            import os
            assert not os.path.exists(os.path.join(root, "etc", "passwd"))
        finally:
            if sandbox is not None:
                sandbox.cleanup()

    def test_workspace_root_is_real_directory(self):
        sandbox, root, _ = _setup_sim_sandbox(
            backend="tmpdir", populate=False,
        )
        try:  # noqa: SIM105
            import os
            assert os.path.isdir(root)
        finally:
            if sandbox is not None:
                sandbox.cleanup()


# ─────────────────────────────────────────────────────────────────────────
# Auto backend — fallback behavior
# ─────────────────────────────────────────────────────────────────────────


class TestAutoBackend:
    def test_auto_returns_valid_sandbox_regardless_of_docker(self):
        """Whether Docker is available or not, auto must return a
        working sandbox. No silent failures."""
        sandbox, root, pain_bus = _setup_sim_sandbox(
            backend="auto", populate=False,
        )
        try:  # noqa: SIM105
            assert sandbox is not None
            assert root is not None
            assert pain_bus is not None
            # Inner is either Docker or Tmpdir, never None
            inner = sandbox._sandbox
            assert isinstance(inner, (DockerSandbox, TmpdirSandbox))
        finally:
            if sandbox is not None:
                sandbox.cleanup()

    def test_auto_matches_docker_availability(self):
        """Auto mode should pick Docker iff check_docker_available()."""
        docker_ok = check_docker_available()
        sandbox, _, _ = _setup_sim_sandbox(backend="auto", populate=False)
        try:  # noqa: SIM105
            inner = sandbox._sandbox
            if docker_ok:
                assert isinstance(inner, DockerSandbox), (
                    "auto mode should pick Docker when daemon is reachable"
                )
            else:
                assert isinstance(inner, TmpdirSandbox), (
                    "auto mode should fall back to tmpdir when Docker is unavailable"
                )
        finally:
            if sandbox is not None:
                sandbox.cleanup()


# ─────────────────────────────────────────────────────────────────────────
# Docker backend — opt-in, skipped without Docker
# ─────────────────────────────────────────────────────────────────────────


# Refresh the cache — other test modules may have poisoned it with
# mocked subprocess results. We want a real probe at collection time.
_DOCKER = check_docker_available(refresh=True)


@pytest.mark.skipif(not _DOCKER, reason="Docker not available")
class TestDockerBackend:
    def test_docker_backend_returns_docker_sandbox(self):
        sandbox, root, pain_bus = _setup_sim_sandbox(
            backend="docker", populate=False,
        )
        try:  # noqa: SIM105
            assert sandbox is not None
            assert root is not None
            assert pain_bus is not None
            assert isinstance(sandbox._sandbox, DockerSandbox)
        finally:
            if sandbox is not None:
                sandbox.cleanup()


# ─────────────────────────────────────────────────────────────────────────
# Ordering regression — the original bug
# ─────────────────────────────────────────────────────────────────────────


class TestOrderingContract:
    def test_helper_never_raises_unbound_local_error(self):
        """The helper must NEVER raise UnboundLocalError regardless of
        the backend or whether Docker is available. This is the exact
        bug class that hid in start_simulation_mode for weeks."""
        for backend in ("auto", "tmpdir"):
            sandbox, _, _ = _setup_sim_sandbox(
                backend=backend, populate=False,
            )
            try:  # noqa: SIM105
                assert sandbox is not None, (
                    f"backend={backend!r} returned None sandbox — "
                    f"the helper swallowed an exception"
                )
            finally:
                if sandbox is not None:
                    sandbox.cleanup()

    def test_pain_bus_created_before_sandbox(self):
        """The helper's contract: pain_bus exists when the sandbox
        is constructed. Verified by checking the sandbox's
        internal reference."""
        sandbox, _, pain_bus = _setup_sim_sandbox(
            backend="tmpdir", populate=False,
        )
        try:  # noqa: SIM105
            # If pain_bus were constructed AFTER sandbox, the
            # sandbox's _pain_bus would be None instead of matching.
            assert pain_bus is not None
            assert sandbox._pain_bus is pain_bus
        finally:
            if sandbox is not None:
                sandbox.cleanup()

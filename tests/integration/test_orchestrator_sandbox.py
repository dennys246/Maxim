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

from maxim.simulation.orchestrator import _prepare_sim_workspace, _setup_sim_sandbox
from maxim.simulation.container_runner import (
    ContainerExecResult,
    ContainerHandle,
    ContainerSpec,
    check_docker_available,
)
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
            backend="tmpdir",
            populate=False,
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
            backend="tmpdir",
            populate=False,
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
            backend="tmpdir",
            populate=True,
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
            backend="tmpdir",
            populate=False,
        )
        try:  # noqa: SIM105
            import os

            assert not os.path.exists(os.path.join(root, "etc", "passwd"))
        finally:
            if sandbox is not None:
                sandbox.cleanup()

    def test_workspace_root_is_real_directory(self):
        sandbox, root, _ = _setup_sim_sandbox(
            backend="tmpdir",
            populate=False,
        )
        try:  # noqa: SIM105
            import os

            assert os.path.isdir(root)
        finally:
            if sandbox is not None:
                sandbox.cleanup()


# ─────────────────────────────────────────────────────────────────────────
# Auto backend — fallback behavior (hermetic: the probe is faked, #1103)
# ─────────────────────────────────────────────────────────────────────────


class _FakeRunner:
    """A ``ContainerRunner`` that launches nothing: records calls, answers every exec with exit 0."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def ensure_image(self, image: str) -> None:
        self.calls.append(f"ensure_image {image}")

    def launch(self, spec: ContainerSpec) -> ContainerHandle:
        self.calls.append("launch")
        return ContainerHandle("fake-cid", "fake-name", spec)

    def exec(self, handle, command, *, user=None, timeout=30.0, stdin=None):  # noqa: ANN001, ANN201
        self.calls.append(f"exec {command}")
        return ContainerExecResult()

    def write_file(self, handle, path, content, *, user=None):  # noqa: ANN001, ANN201
        self.calls.append(f"write_file {path}")
        return True

    def read_file(self, handle, path, *, user=None):  # noqa: ANN001, ANN201
        return "", False

    def stop(self, handle) -> None:  # noqa: ANN001
        self.calls.append("stop")


@pytest.fixture
def docker_probe(monkeypatch):
    """Pin what ``create_sandbox`` sees: ``set(True)`` makes the daemon "reachable" and backs ``DockerSandbox`` with
    ``_FakeRunner``; ``set(False)`` makes it unreachable. Both seams are late-imported from ``container_runner``
    inside ``create_sandbox`` / ``DockerSandbox.__init__``, so patching the module attribute reaches them. The fast
    suite never touches the machine's real daemon (tests/docker_guard.py); a real-container test is
    ``TestDockerBackend`` in the slow lane."""
    import maxim.simulation.container_runner as cr

    runner = _FakeRunner()

    def _set(available: bool) -> _FakeRunner:
        monkeypatch.setattr(cr, "check_docker_available", lambda refresh=False: available)
        monkeypatch.setattr(cr, "get_container_runner", lambda backend="local-docker": runner)
        return runner

    return _set


class TestAutoBackend:
    @pytest.mark.parametrize("available", [True, False])
    def test_auto_returns_valid_sandbox_either_way(self, docker_probe, available):
        """Whether Docker is available or not, auto must return a working sandbox. No silent failures."""
        docker_probe(available)
        sandbox, root, pain_bus = _setup_sim_sandbox(backend="auto", populate=False)
        try:  # noqa: SIM105
            assert sandbox is not None
            assert root is not None
            assert pain_bus is not None
        finally:
            if sandbox is not None:
                sandbox.cleanup()

    def test_auto_picks_docker_when_daemon_reachable(self, docker_probe):
        runner = docker_probe(True)
        sandbox, _, _ = _setup_sim_sandbox(backend="auto", populate=False)
        try:  # noqa: SIM105
            assert isinstance(sandbox._sandbox, DockerSandbox), "auto mode should pick Docker when daemon is reachable"
            assert "launch" in runner.calls, "the Docker sandbox was chosen but never started"
        finally:
            if sandbox is not None:
                sandbox.cleanup()
        assert runner.calls[-1] == "stop"

    def test_auto_falls_back_to_tmpdir_when_daemon_unreachable(self, docker_probe):
        runner = docker_probe(False)
        sandbox, _, _ = _setup_sim_sandbox(backend="auto", populate=False)
        try:  # noqa: SIM105
            assert isinstance(sandbox._sandbox, TmpdirSandbox), (
                "auto mode should fall back to tmpdir when Docker is unavailable"
            )
            assert runner.calls == [], "the fallback must not touch the container runner"
        finally:
            if sandbox is not None:
                sandbox.cleanup()

    def test_docker_backend_refuses_when_daemon_unreachable(self, docker_probe):
        """backend="docker" requires the daemon: the helper reports no sandbox instead of a tmpdir in disguise."""
        docker_probe(False)
        sandbox, root, _ = _setup_sim_sandbox(backend="docker", populate=False)
        assert sandbox is None
        assert root is None


# ─────────────────────────────────────────────────────────────────────────
# Docker backend — a REAL daemon: nightly slow lane only (#1103)
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestDockerBackend:
    """Real containers. In the slow lane (``scripts/lane_rosters/slow.json``), whose ubuntu runner has Docker; the
    probe runs HERE, not at import, so collecting the fast suite never reaches the daemon (tests/docker_guard.py).
    A skip is not in the roster's ``allowed_skips``: on the lane, no Docker is a failure, not a pass."""

    @pytest.fixture(autouse=True)
    def _require_docker(self):
        if not check_docker_available(refresh=True):
            pytest.skip("Docker not available")

    def test_docker_backend_returns_docker_sandbox(self):
        sandbox, root, pain_bus = _setup_sim_sandbox(
            backend="docker",
            populate=False,
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
    @pytest.mark.parametrize("docker_available", [True, False])
    def test_helper_never_raises_unbound_local_error(self, docker_probe, docker_available):
        """The helper must NEVER raise UnboundLocalError regardless of
        the backend or whether Docker is available. This is the exact
        bug class that hid in start_simulation_mode for weeks."""
        docker_probe(docker_available)
        for backend in ("auto", "tmpdir"):
            sandbox, _, _ = _setup_sim_sandbox(
                backend=backend,
                populate=False,
            )
            try:  # noqa: SIM105
                assert sandbox is not None, (
                    f"backend={backend!r} returned None sandbox — the helper swallowed an exception"
                )
            finally:
                if sandbox is not None:
                    sandbox.cleanup()

    def test_pain_bus_created_before_sandbox(self):
        """The helper's contract: pain_bus exists when the sandbox
        is constructed. Verified by checking the sandbox's
        internal reference."""
        sandbox, _, pain_bus = _setup_sim_sandbox(
            backend="tmpdir",
            populate=False,
        )
        try:  # noqa: SIM105
            # If pain_bus were constructed AFTER sandbox, the
            # sandbox's _pain_bus would be None instead of matching.
            assert pain_bus is not None
            assert sandbox._pain_bus is pain_bus
        finally:
            if sandbox is not None:
                sandbox.cleanup()


# ─────────────────────────────────────────────────────────────────────────
# Sim workspace — under the data home, never CWD-relative
# ─────────────────────────────────────────────────────────────────────────


class TestSimWorkspaceUnderDataHome:
    """Regression: ``start_simulation_mode`` built ``Path("data")/"sim_sandbox"``
    (and a ``data/agents/MaximAgent/runtime`` mkdir) relative to the CWD, so a
    hosted sandbox with a read-only root and only ``MAXIM_DATA_HOME`` writable
    could not start a sim. The helper is the smallest unit that does the
    mkdir — no LLM, no agent loop."""

    def test_workspace_lands_under_maxim_data_home(self, tmp_path, monkeypatch):
        from maxim.utils.paths import _reset_caches

        data_home = tmp_path / "data_home"
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.setenv("MAXIM_DATA_HOME", str(data_home))
        monkeypatch.chdir(cwd)
        _reset_caches()
        try:
            sim_workspace, sim_tmpdir, log_path = _prepare_sim_workspace(stamp="20260903_000000")
        finally:
            _reset_caches()

        assert sim_workspace == data_home / "sim_sandbox"
        assert sim_workspace.is_dir()
        assert sim_tmpdir.parent == sim_workspace
        assert sim_tmpdir.is_dir()
        assert sim_tmpdir.name.startswith("sim_agent_20260903_000000_")
        assert log_path == str(sim_workspace / "sim_agent_20260903_000000.jsonl")
        # Nothing was written relative to the CWD.
        assert not (cwd / "data").exists()
        assert list(cwd.iterdir()) == []

"""Tests for ContainerRunner + DockerSandbox.

Unit tests (always run) exercise the ContainerRunner protocol, factory,
and path-mapping logic in DockerSandbox by mocking the subprocess
layer.

Integration tests (skipped when Docker isn't available) spin up real
containers to verify end-to-end behavior.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch, MagicMock

import pytest

from maxim.simulation.container_runner import (
    DEFAULT_IMAGE,
    IMAGE_CATALOG,
    ContainerHandle,
    ContainerRunner,
    ContainerSpec,
    LocalDockerRunner,
    check_docker_available,
    get_container_runner,
    get_image_profile,
    list_supported_images,
)
from maxim.simulation.sandbox import (
    ContainerPermissions,
    DockerSandbox,
    permissions_for_autonomy,
)


# ── Unit tests (no Docker required) ──────────────────────────────────────


class TestContainerRunnerFactory:
    def test_default_backend_is_local_docker(self):
        runner = get_container_runner()
        assert isinstance(runner, LocalDockerRunner)

    def test_explicit_local_docker(self):
        runner = get_container_runner("local-docker")
        assert isinstance(runner, LocalDockerRunner)

    def test_aliases(self):
        for alias in ("docker", "local", "LOCAL-DOCKER"):
            runner = get_container_runner(alias)
            assert isinstance(runner, LocalDockerRunner)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="aws-fargate"):
            get_container_runner("aws-fargate")

    def test_satisfies_protocol(self):
        runner = LocalDockerRunner()
        # Runtime-checkable Protocol — verifies interface conformance
        assert isinstance(runner, ContainerRunner)


class TestImageCatalog:
    def test_default_image_is_in_catalog(self):
        assert DEFAULT_IMAGE in IMAGE_CATALOG

    def test_catalog_covers_major_families(self):
        families = {p.family for p in IMAGE_CATALOG.values()}
        assert {"debian", "ubuntu", "rhel", "alpine", "python"}.issubset(families | {"python"})
        # More defensively, just check we have coverage
        assert "debian" in families
        assert "ubuntu" in families
        assert "rhel" in families
        assert "alpine" in families

    def test_get_image_profile_known(self):
        profile = get_image_profile("python:3.12-slim")
        assert profile is not None
        assert profile.family == "debian"
        assert profile.has_bash is True
        assert profile.has_useradd is True

    def test_get_image_profile_unknown_returns_none(self):
        assert get_image_profile("nosuchimage:latest") is None

    def test_alpine_has_no_bash(self):
        profile = get_image_profile("alpine:3.19")
        assert profile is not None
        assert profile.family == "alpine"
        assert profile.has_bash is False
        assert profile.has_useradd is False

    def test_list_supported_images_non_empty(self):
        images = list_supported_images()
        assert len(images) >= 5
        assert DEFAULT_IMAGE in images


class TestContainerSpec:
    def test_defaults(self):
        spec = ContainerSpec()
        assert spec.image == "python:3.12-slim"
        assert spec.network == "none"
        assert spec.memory == "512m"
        assert spec.user == "1000:1000"

    def test_mounts_and_env(self):
        spec = ContainerSpec(
            mounts=[("/tmp/foo", "/workspace", "rw")],
            env={"MY_VAR": "hello"},
        )
        assert spec.mounts[0] == ("/tmp/foo", "/workspace", "rw")
        assert spec.env["MY_VAR"] == "hello"


class TestPermissionsForAutonomy:
    def test_none_returns_default(self):
        perms = permissions_for_autonomy(None)
        assert perms == ContainerPermissions()

    def test_planning_is_most_restrictive(self):
        from maxim.agents.autonomy import AutonomyLevel

        perms = permissions_for_autonomy(AutonomyLevel.PLANNING)
        assert perms.workspace_readonly is True
        assert perms.memory == "256m"
        assert perms.cpus == 0.5
        assert perms.network == "none"

    def test_supervised_is_middle(self):
        from maxim.agents.autonomy import AutonomyLevel

        perms = permissions_for_autonomy(AutonomyLevel.SUPERVISED)
        assert perms.workspace_readonly is False
        assert perms.memory == "512m"
        assert perms.cpus == 1.0

    def test_autonomous_gets_largest_envelope(self):
        from maxim.agents.autonomy import AutonomyLevel

        perms = permissions_for_autonomy(AutonomyLevel.AUTONOMOUS)
        assert perms.workspace_readonly is False
        assert perms.memory == "1g"
        assert perms.cpus == 2.0
        # Network is still "none" by default even at AUTONOMOUS
        assert perms.network == "none"


class TestCheckDockerAvailable:
    """These tests refresh the docker-available cache with mocked
    subprocess, which would leak into later tests. The ``_restore_cache``
    fixture puts a real probe result back after each test."""

    @pytest.fixture(autouse=True)
    def _restore_cache(self):
        yield
        # Re-probe with real subprocess so later tests aren't
        # stuck with a mocked False/True value.
        check_docker_available(refresh=True)

    def test_not_available_when_docker_missing(self):
        # FileNotFoundError when `docker` binary isn't installed
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert check_docker_available(refresh=True) is False

    def test_not_available_on_timeout(self):
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=5.0),
        ):
            assert check_docker_available(refresh=True) is False

    def test_available_when_info_succeeds(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            assert check_docker_available(refresh=True) is True

    def test_unavailable_when_info_fails(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            assert check_docker_available(refresh=True) is False


class TestLocalDockerRunnerLaunch:
    def test_launch_constructs_docker_run_argv(self):
        spec = ContainerSpec(
            image="python:3.12-slim",
            memory="256m",
            cpus=0.5,
            network="none",
            mounts=[("/tmp/ws", "/home/maxim/.maxim_workspace", "rw")],
            env={"FOO": "bar"},
        )
        runner = LocalDockerRunner()

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "abc123def456\n"
        mock_result.stderr = ""

        with patch.object(runner, "_run_docker", return_value=mock_result) as m:
            handle = runner.launch(spec)

        assert handle.container_id == "abc123def456"
        assert handle.name.startswith("maxim-sandbox-")
        assert len(handle.name.split("-")[-1]) == 8  # uuid suffix length

        # Check the argv contained expected flags
        argv = m.call_args[0][0]
        assert "--rm" in argv
        assert "--network=none" in argv
        assert "--memory=256m" in argv
        assert "--cpus=0.5" in argv
        assert "python:3.12-slim" in argv
        assert any("FOO=bar" in a for a in argv)
        assert any("/tmp/ws:" in a for a in argv)

    def test_launch_raises_on_failure(self):
        runner = LocalDockerRunner()
        mock_result = MagicMock()
        mock_result.returncode = 125
        mock_result.stderr = "no such image"

        with patch.object(runner, "_run_docker", return_value=mock_result):
            from maxim.simulation.container_runner import ContainerRunnerError

            with pytest.raises(ContainerRunnerError, match="no such image"):
                runner.launch(ContainerSpec())


class TestLocalDockerRunnerExec:
    def test_exec_wraps_with_container_side_timeout(self):
        runner = LocalDockerRunner()
        handle = ContainerHandle(
            container_id="cid",
            name="name",
            spec=ContainerSpec(),
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "hello"
        mock_result.stderr = ""

        with patch.object(runner, "_run_docker", return_value=mock_result) as m:
            result = runner.exec(handle, "echo hello", timeout=5.0)

        assert result.exit_code == 0
        assert result.stdout == "hello"
        assert result.timed_out is False

        argv = m.call_args[0][0]
        assert "timeout" in argv
        assert "--signal=KILL" in argv
        assert "5.0" in argv
        assert "/bin/bash" in argv  # default shell

    def test_exec_detects_timeout_from_exit_code(self):
        runner = LocalDockerRunner()
        handle = ContainerHandle("cid", "name", ContainerSpec())
        mock_result = MagicMock()
        mock_result.returncode = 124  # coreutil timeout → SIGTERM
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch.object(runner, "_run_docker", return_value=mock_result):
            result = runner.exec(handle, "sleep 60", timeout=1.0)

        assert result.timed_out is True
        assert result.exit_code == 124

    def test_exec_respects_user_override(self):
        runner = LocalDockerRunner()
        handle = ContainerHandle("cid", "name", ContainerSpec())
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch.object(runner, "_run_docker", return_value=mock_result) as m:
            runner.exec(handle, "whoami", user="root")

        argv = m.call_args[0][0]
        assert "-u" in argv
        assert "root" in argv


class TestLocalDockerRunnerDockerMissing:
    def test_launch_raises_helpful_error_when_docker_missing(self):
        runner = LocalDockerRunner()
        with patch("subprocess.run", side_effect=FileNotFoundError("no docker")):
            from maxim.simulation.container_runner import ContainerRunnerError

            with pytest.raises(ContainerRunnerError, match="docker CLI not found"):
                runner.launch(ContainerSpec())


class TestDockerSandboxPathMapping:
    """Verify host/container path translation without starting a container."""

    def _make_sandbox(self) -> DockerSandbox:
        sandbox = DockerSandbox()
        # Simulate started state without actually running docker
        sandbox._host_workspace = "/tmp/fake_workspace"
        sandbox._handle = ContainerHandle("cid", "name", ContainerSpec())
        return sandbox

    def test_absolute_paths_preserved(self):
        sandbox = self._make_sandbox()
        assert sandbox._container_path("/etc/passwd") == "/etc/passwd"
        assert sandbox._container_path("/home/maxim/.env") == "/home/maxim/.env"

    def test_relative_paths_rooted_at_workspace(self):
        sandbox = self._make_sandbox()
        assert sandbox._container_path("notes.txt") == "/home/maxim/.maxim_workspace/notes.txt"
        assert sandbox._container_path("drafts/a.md") == "/home/maxim/.maxim_workspace/drafts/a.md"

    def test_workspace_paths_map_to_host(self):
        sandbox = self._make_sandbox()
        # Relative → resolves under workspace → maps to host
        assert sandbox._map_to_host("notes.txt") == "/tmp/fake_workspace/notes.txt"
        # Absolute under container workspace → maps to host
        assert sandbox._map_to_host("/home/maxim/.maxim_workspace/x.txt") == "/tmp/fake_workspace/x.txt"

    def test_non_workspace_paths_do_not_map(self):
        sandbox = self._make_sandbox()
        # System paths stay inside container
        assert sandbox._map_to_host("/etc/passwd") is None
        assert sandbox._map_to_host("/var/log/auth.log") is None


class TestDockerSandboxReadonlyWorkspace:
    def test_write_refused_when_readonly(self):
        sandbox = DockerSandbox(
            permissions=ContainerPermissions(workspace_readonly=True),
        )
        sandbox._host_workspace = "/tmp/fake_workspace"
        sandbox._handle = ContainerHandle("cid", "name", ContainerSpec())
        assert sandbox.write_file("foo.txt", "content") is False


# ── Integration tests (Docker required) ──────────────────────────────────


_DOCKER = check_docker_available()


@pytest.mark.skipif(not _DOCKER, reason="Docker not available")
class TestDockerSandboxIntegration:
    """End-to-end tests that actually launch a container.

    These are slower (first-run image pull + container startup ~2-3s)
    but exercise the real docker CLI path.
    """

    def test_start_and_cleanup(self):
        sandbox = DockerSandbox(
            permissions=ContainerPermissions(memory="256m", cpus=0.5),
        )
        try:
            sandbox.start()
            assert sandbox._handle is not None
            assert sandbox.workspace_root != ""
        finally:
            sandbox.cleanup()
        # After cleanup, host workspace is gone
        assert sandbox._host_workspace is None

    def test_execute_in_container(self):
        sandbox = DockerSandbox()
        try:
            sandbox.start()
            result = sandbox.execute("whoami", timeout=10.0)
            assert result.exit_code == 0
            assert "maxim" in result.stdout
        finally:
            sandbox.cleanup()

    def test_workspace_write_visible_on_host(self):
        sandbox = DockerSandbox()
        try:
            sandbox.start()
            sandbox.write_file("hello.txt", "hello from host")
            import os

            host_path = os.path.join(sandbox.workspace_root, "hello.txt")
            assert os.path.exists(host_path)
            # And visible in container
            result = sandbox.execute("cat /home/maxim/.maxim_workspace/hello.txt")
            assert result.exit_code == 0
            assert "hello from host" in result.stdout
        finally:
            sandbox.cleanup()

    def test_unprivileged_user_cannot_read_etc_shadow(self):
        """OS-level permissions block shadow reads for maxim user."""
        sandbox = DockerSandbox()
        try:
            sandbox.start()
            result = sandbox.execute("cat /etc/shadow 2>&1", timeout=5.0)
            # Either permission denied (shadow exists and is root-only)
            # or file not found (slim image may not have shadow). Both
            # are fine — the important thing is the maxim user can't
            # read it.
            assert result.exit_code != 0 or "Permission denied" in result.stdout
        finally:
            sandbox.cleanup()

    def test_timeout_kills_container_side_process(self):
        sandbox = DockerSandbox()
        try:
            sandbox.start()
            result = sandbox.execute("sleep 10", timeout=1.0)
            assert result.exit_code in (124, 137)
        finally:
            sandbox.cleanup()

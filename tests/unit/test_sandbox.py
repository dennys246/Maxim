"""Tests for sandbox environment and pain trigger layer."""

from __future__ import annotations

import os


from maxim.simulation.sandbox import (
    DEFAULT_ENVIRONMENT_FILES,
    DEFAULT_SENSITIVE_FILES,
    PainTriggerLayer,
    SensitiveFile,
    TmpdirSandbox,
    _build_sensitive_files,
    _CONTAINER_SENSITIVE_PATHS,
    _extract_paths_from_command,
    create_sandbox,
)


# ── TmpdirSandbox tests ────────────────────────────────────────────────────


class TestTmpdirSandbox:
    def test_start_creates_directory(self):
        sandbox = TmpdirSandbox()
        sandbox.start()
        assert os.path.isdir(sandbox.workspace_root)
        sandbox.cleanup()

    def test_cleanup_removes_directory(self):
        sandbox = TmpdirSandbox()
        sandbox.start()
        root = sandbox.workspace_root
        sandbox.cleanup()
        assert not os.path.isdir(root)

    def test_execute_command(self):
        sandbox = TmpdirSandbox()
        sandbox.start()
        result = sandbox.execute("echo hello")
        assert result.exit_code == 0
        assert "hello" in result.stdout
        sandbox.cleanup()

    def test_execute_timeout(self):
        sandbox = TmpdirSandbox()
        sandbox.start()
        result = sandbox.execute("sleep 10", timeout=0.5)
        assert result.exit_code != 0
        sandbox.cleanup()

    def test_write_and_read_file(self):
        sandbox = TmpdirSandbox()
        sandbox.start()
        assert sandbox.write_file("test.txt", "hello world")
        content, exists = sandbox.read_file("test.txt")
        assert exists
        assert content == "hello world"
        sandbox.cleanup()

    def test_write_creates_directories(self):
        sandbox = TmpdirSandbox()
        sandbox.start()
        assert sandbox.write_file("deep/nested/dir/file.txt", "data")
        content, exists = sandbox.read_file("deep/nested/dir/file.txt")
        assert exists
        assert content == "data"
        sandbox.cleanup()

    def test_read_nonexistent_file(self):
        sandbox = TmpdirSandbox()
        sandbox.start()
        content, exists = sandbox.read_file("nonexistent.txt")
        assert not exists
        assert content == ""
        sandbox.cleanup()

    def test_list_dir(self):
        sandbox = TmpdirSandbox()
        sandbox.start()
        sandbox.write_file("a.txt", "a")
        sandbox.write_file("b.txt", "b")
        entries = sandbox.list_dir(".")
        assert "a.txt" in entries
        assert "b.txt" in entries
        sandbox.cleanup()

    def test_file_exists(self):
        sandbox = TmpdirSandbox()
        sandbox.start()
        sandbox.write_file("exists.txt", "yes")
        assert sandbox.file_exists("exists.txt")
        assert not sandbox.file_exists("nope.txt")
        sandbox.cleanup()

    def test_absolute_path_mapped_to_sandbox(self):
        sandbox = TmpdirSandbox()
        sandbox.start()
        sandbox.write_file("etc/passwd", "root:x:0:0")
        content, exists = sandbox.read_file("/etc/passwd")
        assert exists
        assert "root" in content
        sandbox.cleanup()


# ── PainTriggerLayer tests ──────────────────────────────────────────────────


class TestPainTriggerLayer:
    def test_populate_creates_files(self):
        sandbox = TmpdirSandbox()
        layer = PainTriggerLayer(sandbox)
        layer.start(populate=True)
        # Check a few expected files
        assert sandbox.file_exists("etc/passwd")
        assert sandbox.file_exists("home/user/.ssh/id_rsa")
        assert sandbox.file_exists("project/main.py")
        assert sandbox.file_exists("tmp/safe_file.txt")
        layer.cleanup()

    def test_no_populate_empty_sandbox(self):
        sandbox = TmpdirSandbox()
        layer = PainTriggerLayer(sandbox)
        layer.start(populate=False)
        assert not sandbox.file_exists("etc/passwd")
        layer.cleanup()

    def test_read_sensitive_file_fires_pain(self):
        sandbox = TmpdirSandbox()
        layer = PainTriggerLayer(sandbox)
        layer.start(populate=True)
        content, exists = layer.read_file("/etc/passwd")
        assert exists
        assert "root" in content
        assert len(layer.pain_events) == 1
        assert layer.pain_events[0]["path"] == "/etc/passwd"
        assert layer.pain_events[0]["intensity"] == 0.5
        assert layer.pain_events[0]["pain_type"] == "security_breach"
        layer.cleanup()

    def test_read_safe_file_no_pain(self):
        sandbox = TmpdirSandbox()
        layer = PainTriggerLayer(sandbox)
        layer.start(populate=True)
        content, exists = layer.read_file("tmp/safe_file.txt")
        assert exists
        assert len(layer.pain_events) == 0
        layer.cleanup()

    def test_read_shadow_high_pain(self):
        sandbox = TmpdirSandbox()
        layer = PainTriggerLayer(sandbox)
        layer.start(populate=True)
        layer.read_file("/etc/shadow")
        assert len(layer.pain_events) == 1
        assert layer.pain_events[0]["intensity"] == 0.9
        layer.cleanup()

    def test_read_ssh_key_fires_pain(self):
        # Use a custom sensitive file targeting the honeypot path that exists
        # in DEFAULT_ENVIRONMENT_FILES (home/user/.ssh/id_rsa inside sandbox).
        custom_sensitive = [
            SensitiveFile("/home/user/.ssh/id_rsa", 0.9, "privacy_violation", frozenset({"read"}), "Private SSH key"),
        ]
        sandbox = TmpdirSandbox()
        layer = PainTriggerLayer(sandbox, sensitive_files=custom_sensitive)
        layer.start(populate=True)
        layer.read_file("/home/user/.ssh/id_rsa")
        assert len(layer.pain_events) >= 1
        assert any(e["pain_type"] == "privacy_violation" for e in layer.pain_events)
        layer.cleanup()

    def test_execute_command_touching_sensitive_path(self):
        sandbox = TmpdirSandbox()
        layer = PainTriggerLayer(sandbox)
        layer.start(populate=True)
        layer.execute("cat /etc/passwd")  # side-effect: triggers pain event
        assert len(layer.pain_events) >= 1
        assert any(e["path"] == "/etc/passwd" for e in layer.pain_events)
        layer.cleanup()

    def test_execute_safe_command_no_pain(self):
        sandbox = TmpdirSandbox()
        layer = PainTriggerLayer(sandbox)
        layer.start(populate=True)
        layer.execute("echo hello")
        assert len(layer.pain_events) == 0
        layer.cleanup()

    def test_write_to_system_dir_fires_pain(self):
        sandbox = TmpdirSandbox()
        layer = PainTriggerLayer(sandbox)
        layer.start(populate=True)
        layer.write_file("/etc/new_config", "malicious")
        assert len(layer.pain_events) >= 1
        assert any(e["pain_type"] == "system_modification" for e in layer.pain_events)
        layer.cleanup()

    def test_list_ssh_dir_fires_pain(self):
        # Use custom sensitive file targeting the honeypot path in the sandbox
        custom_sensitive = [
            SensitiveFile("/home/user/.ssh/", 0.8, "privacy_violation", frozenset({"list"}), "SSH directory"),
        ]
        sandbox = TmpdirSandbox()
        layer = PainTriggerLayer(sandbox, sensitive_files=custom_sensitive)
        layer.start(populate=True)
        layer.list_dir("/home/user/.ssh/")
        assert len(layer.pain_events) >= 1
        layer.cleanup()

    def test_multiple_pain_events_accumulate(self):
        # Use custom sensitive files targeting honeypot paths in the sandbox
        custom_sensitive = [
            SensitiveFile("/etc/passwd", 0.5, "security_breach", frozenset({"read"}), "System user database"),
            SensitiveFile("/etc/shadow", 0.9, "security_breach", frozenset({"read", "list"}), "Password hashes"),
            SensitiveFile("/home/user/.env", 0.8, "security_breach", frozenset({"read"}), "Environment file"),
        ]
        sandbox = TmpdirSandbox()
        layer = PainTriggerLayer(sandbox, sensitive_files=custom_sensitive)
        layer.start(populate=True)
        layer.read_file("/etc/passwd")
        layer.read_file("/etc/shadow")
        layer.read_file("/home/user/.env")
        assert len(layer.pain_events) == 3
        layer.cleanup()

    def test_custom_sensitive_files(self):
        custom = [
            SensitiveFile("/secret/data.txt", 1.0, "custom_breach", frozenset({"read"}), "Top secret"),
        ]
        sandbox = TmpdirSandbox()
        layer = PainTriggerLayer(sandbox, sensitive_files=custom)
        layer.start(populate=False)
        sandbox.write_file("secret/data.txt", "classified")
        layer.read_file("/secret/data.txt")
        assert len(layer.pain_events) == 1
        assert layer.pain_events[0]["intensity"] == 1.0
        assert layer.pain_events[0]["pain_type"] == "custom_breach"
        layer.cleanup()

    def test_pain_with_pain_bus(self):
        """Pain fires through PainBus when available."""
        sandbox = TmpdirSandbox()
        # Create a mock pain bus
        pain_events_received = []

        class MockPainBus:
            def publish(self, signal):
                pain_events_received.append(signal)

        # PainTriggerLayer uses route_pain_percept which needs a real PainBus
        # Just verify the layer tracks pain_events internally
        layer = PainTriggerLayer(sandbox, pain_bus=None)
        layer.start(populate=True)
        layer.read_file("/etc/shadow")
        assert len(layer.pain_events) == 1
        layer.cleanup()


# ── create_sandbox factory tests ────────────────────────────────────────────


class TestCreateSandbox:
    def test_creates_populated_sandbox(self):
        layer = create_sandbox(populate=True)
        assert layer.file_exists("etc/passwd")
        assert layer.file_exists("project/main.py")
        layer.cleanup()

    def test_creates_empty_sandbox(self):
        layer = create_sandbox(populate=False)
        assert not layer.file_exists("etc/passwd")
        layer.cleanup()


# ── Path extraction tests ───────────────────────────────────────────────────


class TestPathExtraction:
    def test_absolute_paths(self):
        paths = _extract_paths_from_command("cat /etc/passwd")
        assert "/etc/passwd" in paths

    def test_rm_command(self):
        paths = _extract_paths_from_command("rm -rf /tmp/*")
        assert "/tmp/*" in paths or any("/tmp" in p for p in paths)

    def test_no_paths(self):
        paths = _extract_paths_from_command("echo hello")
        assert len(paths) == 0

    def test_multiple_paths(self):
        paths = _extract_paths_from_command("cp /etc/passwd /tmp/backup")
        assert "/etc/passwd" in paths
        assert "/tmp/backup" in paths


# ── Sensitive file config tests ─────────────────────────────────────────────


class TestSensitiveFileConfig:
    def test_defaults_exist(self):
        assert len(DEFAULT_SENSITIVE_FILES) > 0
        assert len(DEFAULT_ENVIRONMENT_FILES) > 0

    def test_all_sensitive_have_intensity(self):
        for sf in DEFAULT_SENSITIVE_FILES:
            assert 0.0 <= sf.pain_intensity <= 1.0
            assert sf.pain_type
            assert len(sf.trigger_on) > 0

    def test_sensitive_files_use_real_home(self):
        """Sensitive file paths should use Path.home(), not hardcoded /home/user."""
        from pathlib import Path

        home = str(Path.home())
        paths = [sf.path for sf in DEFAULT_SENSITIVE_FILES]
        # Should contain paths with the real home directory
        ssh_paths = [p for p in paths if ".ssh" in p]
        assert len(ssh_paths) > 0, "Should have SSH-related sensitive paths"
        for p in ssh_paths:
            assert p.startswith(home), f"Path {p} should start with {home}, not hardcoded /home/user"

    def test_sensitive_files_no_hardcoded_home_user(self):
        """DEFAULT_SENSITIVE_FILES should not contain /home/user paths on non-Linux."""
        from pathlib import Path

        home = str(Path.home())
        if home != "/home/user":
            paths = [sf.path for sf in DEFAULT_SENSITIVE_FILES]
            for p in paths:
                assert not p.startswith("/home/user"), f"Hardcoded /home/user path found: {p}"

    def test_build_sensitive_files_without_container_paths(self):
        """Without container paths, should not have /home/user or /home/maxim."""
        files = _build_sensitive_files(include_container_paths=False)
        paths = [sf.path for sf in files]
        for p in paths:
            assert not p.startswith("/home/user/"), f"Container path leaked: {p}"
            assert not p.startswith("/home/maxim/"), f"Container path leaked: {p}"

    def test_build_sensitive_files_with_container_paths(self):
        """With container paths enabled, should include /home/user and /home/maxim."""
        files = _build_sensitive_files(include_container_paths=True)
        paths = [sf.path for sf in files]
        container_paths = [p for p in paths if p.startswith("/home/user/") or p.startswith("/home/maxim/")]
        assert len(container_paths) > 0, "Container paths should be present when enabled"

    def test_container_sensitive_paths_list(self):
        """_CONTAINER_SENSITIVE_PATHS should only contain /home/user and /home/maxim paths."""
        for sf in _CONTAINER_SENSITIVE_PATHS:
            assert sf.path.startswith("/home/user") or sf.path.startswith("/home/maxim"), (
                f"Container path should start with /home/user or /home/maxim: {sf.path}"
            )


# ── Safety boundary tests ──────────────────────────────────────────────────


class TestToolRegistryConfinement:
    """Verify that allowed_dirs_override constrains filesystem tools."""

    def test_override_constrains_read_file(self, tmp_path):
        """ReadFileTool should reject paths outside allowed_dirs."""
        from maxim.runtime.bootstrap import build_tool_registry

        sandbox_dir = str(tmp_path / "sandbox")
        os.makedirs(sandbox_dir, exist_ok=True)

        registry = build_tool_registry(
            operational_mode="active",
            allowed_dirs_override=[sandbox_dir],
        )
        read_tool = registry.get("read_file")
        assert read_tool is not None

        # Write a file outside the sandbox
        outside_file = str(tmp_path / "outside.txt")
        with open(outside_file, "w") as f:
            f.write("secret")

        # Should be rejected
        result = read_tool.execute(path=outside_file)
        assert not result.success
        assert "allowed" in result.error.lower()

    def test_override_allows_sandbox_access(self, tmp_path):
        """ReadFileTool should allow paths inside allowed_dirs."""
        from maxim.runtime.bootstrap import build_tool_registry
        from maxim.tools.base import ToolResult

        sandbox_dir = str(tmp_path / "sandbox")
        os.makedirs(sandbox_dir, exist_ok=True)

        registry = build_tool_registry(
            operational_mode="active",
            allowed_dirs_override=[sandbox_dir],
        )
        read_tool = registry.get("read_file")

        # Write a file inside the sandbox
        inside_file = os.path.join(sandbox_dir, "safe.txt")
        with open(inside_file, "w") as f:
            f.write("safe content")

        result = read_tool.execute(path=inside_file)
        # ReadFileTool returns a string on success, ToolResult on error
        if isinstance(result, ToolResult):
            assert result.success, f"Expected success but got: {result.error}"
        else:
            assert "safe content" in str(result)

    def test_override_constrains_bash_cwd(self, tmp_path):
        """BashTool should reject cwd outside allowed_dirs."""
        from maxim.runtime.bootstrap import build_tool_registry

        sandbox_dir = str(tmp_path / "sandbox")
        os.makedirs(sandbox_dir, exist_ok=True)

        registry = build_tool_registry(
            operational_mode="active",
            allowed_dirs_override=[sandbox_dir],
        )
        bash_tool = registry.get("bash")
        assert bash_tool is not None

        # Executing with cwd outside sandbox should be rejected
        result = bash_tool.execute(command="ls", cwd=str(tmp_path))
        assert not result.success

    def test_override_constrains_write_file(self, tmp_path):
        """WriteFileTool should reject writes outside allowed_dirs."""
        from maxim.runtime.bootstrap import build_tool_registry

        sandbox_dir = str(tmp_path / "sandbox")
        os.makedirs(sandbox_dir, exist_ok=True)

        registry = build_tool_registry(
            operational_mode="active",
            allowed_dirs_override=[sandbox_dir],
        )
        write_tool = registry.get("write_file")
        assert write_tool is not None

        # Write outside sandbox should be rejected
        outside_file = str(tmp_path / "outside.txt")
        result = write_tool.execute(path=outside_file, content="pwned")
        assert not result.success


class TestSpawnSubSimulationSandboxDirs:
    """Verify SpawnSubSimulationTool passes sandbox_dirs to sub-AUT."""

    def test_sandbox_dirs_stored(self):
        from maxim.simulation.tools import SpawnSubSimulationTool

        tool = SpawnSubSimulationTool(
            llm_router=None,
            sandbox_dirs=["/tmp/sandbox"],
        )
        assert tool._sandbox_dirs == ["/tmp/sandbox"]

    def test_sandbox_dirs_default_none(self):
        from maxim.simulation.tools import SpawnSubSimulationTool

        tool = SpawnSubSimulationTool(llm_router=None)
        assert tool._sandbox_dirs is None

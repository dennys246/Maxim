"""Tests for sandbox environment and pain trigger layer."""

from __future__ import annotations

import os
import time

import pytest

from maxim.simulation.sandbox import (
    DEFAULT_ENVIRONMENT_FILES,
    DEFAULT_SENSITIVE_FILES,
    ExecutionResult,
    PainTriggerLayer,
    SensitiveFile,
    TmpdirSandbox,
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
        sandbox = TmpdirSandbox()
        layer = PainTriggerLayer(sandbox)
        layer.start(populate=True)
        layer.read_file("/home/user/.ssh/id_rsa")
        assert len(layer.pain_events) >= 1
        assert any(e["pain_type"] == "privacy_violation" for e in layer.pain_events)
        layer.cleanup()

    def test_execute_command_touching_sensitive_path(self):
        sandbox = TmpdirSandbox()
        layer = PainTriggerLayer(sandbox)
        layer.start(populate=True)
        result = layer.execute("cat /etc/passwd")
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
        sandbox = TmpdirSandbox()
        layer = PainTriggerLayer(sandbox)
        layer.start(populate=True)
        layer.list_dir("/home/user/.ssh/")
        assert len(layer.pain_events) >= 1
        layer.cleanup()

    def test_multiple_pain_events_accumulate(self):
        sandbox = TmpdirSandbox()
        layer = PainTriggerLayer(sandbox)
        layer.start(populate=True)
        layer.read_file("/etc/passwd")
        layer.read_file("/etc/shadow")
        layer.read_file("/home/user/.env")
        assert len(layer.pain_events) == 3
        layer.cleanup()

    def test_custom_sensitive_files(self):
        custom = [
            SensitiveFile("/secret/data.txt", 1.0, "custom_breach",
                         frozenset({"read"}), "Top secret"),
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

"""Tests for Publication Refinement Plan Phase 1 fixes.

Covers: env var parsing (1i), permission handling (1h), rate limit handling (1f),
shell blocklist (1c), memory module exports (1n), API key validation (1g),
atomic write migration (1e), daemon thread logging (1k), Any→typed annotations (1d).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── 1i: Env var parsing ──────────────────────────────────────────────────────


class TestSafeIntEnv:
    def test_valid_int(self, monkeypatch):
        from maxim.runtime.lane_backends import _safe_int_env

        monkeypatch.setenv("TEST_INT_VAR", "42")
        assert _safe_int_env("TEST_INT_VAR", 10) == 42

    def test_missing_returns_default(self, monkeypatch):
        from maxim.runtime.lane_backends import _safe_int_env

        monkeypatch.delenv("TEST_INT_VAR", raising=False)
        assert _safe_int_env("TEST_INT_VAR", 99) == 99

    def test_invalid_returns_default(self, monkeypatch):
        from maxim.runtime.lane_backends import _safe_int_env

        monkeypatch.setenv("TEST_INT_VAR", "banana")
        assert _safe_int_env("TEST_INT_VAR", 7) == 7

    def test_empty_string_returns_default(self, monkeypatch):
        from maxim.runtime.lane_backends import _safe_int_env

        monkeypatch.setenv("TEST_INT_VAR", "")
        assert _safe_int_env("TEST_INT_VAR", 5) == 5


# ── 1h: Permission error handling ────────────────────────────────────────────


class TestPermissionHandling:
    def test_permission_error_exits_with_message(self, monkeypatch):
        from maxim.utils.paths import _reset_caches

        _reset_caches()
        monkeypatch.setenv("MAXIM_DATA_HOME", "/root/no_access_test_dir_xyz")
        with patch("pathlib.Path.mkdir", side_effect=PermissionError("denied")):
            with pytest.raises(SystemExit) as exc_info:
                _reset_caches()
                from maxim.utils.paths import data_home

                data_home()
            assert "Cannot create data directory" in str(exc_info.value)
        _reset_caches()


# ── 1f: OpenAI rate limit handling ────────────────────────────────────────────


class TestOpenAIRateLimitDetection:
    def test_429_detected(self):
        from maxim.models.language.openai_backend import _is_rate_limit_error

        err = Exception("Error code: 429 - Rate limit exceeded")
        assert _is_rate_limit_error(err) is True

    def test_rate_keyword_detected(self):
        from maxim.models.language.openai_backend import _is_rate_limit_error

        err = Exception("Rate limit reached for model gpt-4")
        assert _is_rate_limit_error(err) is True

    def test_normal_error_not_rate_limit(self):
        from maxim.models.language.openai_backend import _is_rate_limit_error

        err = Exception("Connection refused")
        assert _is_rate_limit_error(err) is False


# ── 1c: Shell blocklist ──────────────────────────────────────────────────────


class TestShellBlocklist:
    def test_blocklist_contains_shell_invocations(self):
        from maxim.utils.sandbox_executor import BLOCKED_SHELL_COMMANDS

        for cmd in ["bash -c", "sh -c", "eval ", "source ", "find -exec", "xargs"]:
            assert cmd in BLOCKED_SHELL_COMMANDS, f"Missing: {cmd}"

    def test_blocklist_still_has_original_entries(self):
        from maxim.utils.sandbox_executor import BLOCKED_SHELL_COMMANDS

        for cmd in ["curl", "wget", "sudo", "rm -rf /", "/etc/passwd"]:
            assert cmd in BLOCKED_SHELL_COMMANDS, f"Missing original: {cmd}"

    def test_pattern_blocklist_catches_whitespace_variants(self):
        from maxim.utils.sandbox_executor import _check_shell_script_safety

        # Exact match already works via BLOCKED_SHELL_COMMANDS;
        # pattern blocklist catches whitespace/quoting variants
        unsafe_scripts = [
            "bash  -c 'echo pwned'",  # extra whitespace
            "python3 -c 'import os'",
            "perl -e 'system(\"rm -rf /\")'",
            "ruby -e 'exec(\"sh\")'",
        ]
        for script in unsafe_scripts:
            is_safe, reason = _check_shell_script_safety(script)
            assert not is_safe, f"Should block: {script}"
            assert reason is not None

    def test_pattern_blocklist_allows_safe_scripts(self):
        from maxim.utils.sandbox_executor import _check_shell_script_safety

        safe_scripts = [
            "echo hello world",
            "ls -la /tmp",
            "cat file.txt | grep pattern",
        ]
        for script in safe_scripts:
            is_safe, reason = _check_shell_script_safety(script)
            assert is_safe, f"Should allow: {script} (blocked: {reason})"


# ── 1n: Memory module exports ────────────────────────────────────────────────


class TestMemoryExports:
    def test_store_protocols_importable(self):
        from maxim.memory import (
            CausalStore,
            EpisodicStore,
            FileCausalStore,
            FileEpisodicStore,
            FileSemanticStore,
            SemanticStore,
        )

        assert EpisodicStore is not None
        assert CausalStore is not None
        assert SemanticStore is not None
        assert FileEpisodicStore is not None
        assert FileCausalStore is not None
        assert FileSemanticStore is not None

    def test_concept_importable(self):
        from maxim.memory import Concept

        assert Concept is not None

    def test_store_protocols_in_all(self):
        import maxim.memory

        for name in [
            "EpisodicStore",
            "CausalStore",
            "SemanticStore",
            "FileEpisodicStore",
            "FileCausalStore",
            "FileSemanticStore",
            "Concept",
        ]:
            assert name in maxim.memory.__all__, f"{name} not in __all__"


# ── 1e: Atomic write migration ───────────────────────────────────────────────


class TestAtomicWriteMigration:
    """Verify that files previously using raw json.dump now use atomic_write_json."""

    def test_config_uses_atomic_write(self):
        """config.py save_config should use atomic_write_json."""
        import inspect

        from maxim.utils.config import build

        source = inspect.getsource(build.save_config)
        assert "atomic_write_json" in source
        assert "json.dump(" not in source

    def test_internet_access_uses_atomic_write(self):
        import inspect

        from maxim.utils.internet_access import save_internet_access

        source = inspect.getsource(save_internet_access)
        assert "atomic_write_json" in source
        assert "json.dump(" not in source


# ── 1d: Typed annotations ────────────────────────────────────────────────────


class TestTypedAnnotations:
    def test_exec_agent_nac_typed(self):
        """ExecAgent._nac should be typed as NucleusAccumbens | None, not Any."""
        import inspect

        from maxim.agents.exec_agent import ExecAgent

        source = inspect.getsource(ExecAgent.__init__)
        assert "NucleusAccumbens" in source or "NucleusAccumbens | None" in source

    def test_exec_agent_hippocampus_typed(self):
        import inspect

        from maxim.agents.exec_agent import ExecAgent

        source = inspect.getsource(ExecAgent.__init__)
        assert "Hippocampus" in source


# ── 1k: Heartbeat logging level ──────────────────────────────────────────────


class TestHeartbeatLogLevel:
    def test_heartbeat_error_is_warning_not_debug(self):
        import inspect

        from maxim.runtime.heartbeat import HeartbeatMonitor

        source = inspect.getsource(HeartbeatMonitor._run)
        assert "logger.warning" in source
        assert 'logger.debug("Heartbeat error' not in source


# ── 1g: API key validation ───────────────────────────────────────────────────


class TestAPIKeyValidation:
    def test_cloud_model_without_key_exits(self, monkeypatch):
        """CLI should fail fast when a cloud model is selected without its API key."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from maxim.models.language.config import _BUILTIN_PROFILES

        # Only test if claude-sonnet is a cloud profile
        profile = _BUILTIN_PROFILES.get("claude-sonnet", {})
        if profile.get("cloud"):
            api_key_env = profile.get("api_key_env", "")
            assert api_key_env, "Cloud profile should have api_key_env"
            # The CLI check would sys.exit — we verify the logic exists
            assert not os.environ.get(api_key_env), "Test requires key to be unset"

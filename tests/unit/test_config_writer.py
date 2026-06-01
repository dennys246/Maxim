"""Regression guards for ``maxim.runtime.config_writer`` (C2 of
config_unification.md).

Surface covered:

* Round-trip: write → read returns equivalent MaximConfig
* set_field: single-field mutation persists, coerces string → typed
* set_field: unknown field rejected
* set_field: invalid api_key_ref rejected (cross-confirmed I-3/IM3 fold)
* LaneTierConfig.extra round-trips through write + read
* mutate_config: read-modify-write under FileLock (I-5 fold)
* Concurrent writers serialize (the bug class the lock prevents)

The autouse ``_isolate_config_json_env`` fixture in conftest.py keeps
each test's loader state clean.
"""

from __future__ import annotations

import json
import multiprocessing
from dataclasses import replace
from pathlib import Path

import pytest

from maxim.exceptions import ConfigurationError
from maxim.runtime.config_loader import (
    LaneTierConfig,
    LanesConfigSection,
    LLMConfigSection,
    MaximConfig,
    load_config,
)
from maxim.runtime.config_writer import (
    mutate_config,
    set_field,
    write_config,
)


class TestWriteAndReadRoundtrip:
    def test_default_config_roundtrips(self, tmp_path):
        path = tmp_path / "config.json"
        write_config(MaximConfig(), path=path)
        loaded = load_config(path)
        assert loaded == MaximConfig()

    def test_custom_config_roundtrips(self, tmp_path):
        path = tmp_path / "config.json"
        cfg = MaximConfig(
            role="leader",
            llm=LLMConfigSection(profile="qwen-32b", n_ctx=16384, auto_download=True),
        )
        write_config(cfg, path=path)
        loaded = load_config(path)
        assert loaded.role == "leader"
        assert loaded.llm.profile == "qwen-32b"
        assert loaded.llm.n_ctx == 16384
        assert loaded.llm.auto_download is True

    def test_lane_tier_extra_roundtrips(self, tmp_path):
        """LaneTierConfig is the path (a) escape-hatch — extra fields
        must survive the write/read cycle so future-grown fields work."""
        path = tmp_path / "config.json"
        cfg = MaximConfig(
            lanes=LanesConfigSection(
                large=LaneTierConfig(
                    remote_url="http://example.com/v1",
                    extra={"remote_health_path": "/health", "remote_timeout_s": 30},
                ),
            ),
        )
        write_config(cfg, path=path)
        loaded = load_config(path)
        assert loaded.lanes.large.remote_url == "http://example.com/v1"
        assert loaded.lanes.large.extra == {
            "remote_health_path": "/health",
            "remote_timeout_s": 30,
        }

    def test_written_file_carries_format_version(self, tmp_path):
        path = tmp_path / "config.json"
        write_config(MaximConfig(), path=path)
        raw = json.loads(path.read_text())
        assert raw["_format_version"] == "1.0"


class TestSetField:
    def test_set_string_field(self, tmp_path):
        path = tmp_path / "config.json"
        set_field("llm.profile", "qwen-32b", path=path)
        loaded = load_config(path)
        assert loaded.llm.profile == "qwen-32b"

    def test_set_int_field_coerces_string_to_int(self, tmp_path):
        """The CLI passes strings; set_field must coerce per C-2 table."""
        path = tmp_path / "config.json"
        set_field("proxy.max_concurrent", "8", path=path)
        loaded = load_config(path)
        assert loaded.proxy.max_concurrent == 8
        assert isinstance(loaded.proxy.max_concurrent, int)

    def test_set_bool_field_coerces_string(self, tmp_path):
        path = tmp_path / "config.json"
        set_field("llm.auto_download", "true", path=path)
        loaded = load_config(path)
        assert loaded.llm.auto_download is True

    def test_set_nested_field(self, tmp_path):
        path = tmp_path / "config.json"
        set_field("lanes.large.remote_url", "http://leader.example.com/v1", path=path)
        loaded = load_config(path)
        assert loaded.lanes.large.remote_url == "http://leader.example.com/v1"

    def test_set_role(self, tmp_path):
        path = tmp_path / "config.json"
        set_field("role", "leader", path=path)
        loaded = load_config(path)
        assert loaded.role == "leader"

    def test_unknown_field_raises(self, tmp_path):
        path = tmp_path / "config.json"
        with pytest.raises(ConfigurationError, match="unknown field path"):
            set_field("llm.nonexistent", "value", path=path)

    def test_invalid_int_raises(self, tmp_path):
        path = tmp_path / "config.json"
        with pytest.raises(ConfigurationError, match="below minimum"):
            set_field("llm.n_ctx", "100", path=path)

    def test_invalid_enum_raises(self, tmp_path):
        path = tmp_path / "config.json"
        with pytest.raises(ConfigurationError, match="expected one of"):
            set_field("role", "boss", path=path)


class TestAPIKeyRefRejection:
    """Cross-confirmed I-3 + IM3 fold: inline strings rejected at write."""

    def test_inline_sk_key_rejected_via_set_field(self, tmp_path):
        path = tmp_path / "config.json"
        with pytest.raises(ConfigurationError, match="Inline plaintext"):
            set_field("lanes.large.remote_api_key_ref", "sk-abc123def", path=path)

    def test_file_path_accepted(self, tmp_path):
        path = tmp_path / "config.json"
        set_field(
            "lanes.large.remote_api_key_ref",
            "~/.config/maxim/api_key",
            path=path,
        )
        loaded = load_config(path)
        assert loaded.lanes.large.remote_api_key_ref == "~/.config/maxim/api_key"

    def test_keyring_uri_accepted(self, tmp_path):
        path = tmp_path / "config.json"
        set_field(
            "lanes.large.remote_api_key_ref",
            "keyring:maxim:default",
            path=path,
        )
        loaded = load_config(path)
        assert loaded.lanes.large.remote_api_key_ref == "keyring:maxim:default"

    def test_inline_rejected_in_full_write(self, tmp_path):
        """The validator also fires when the caller writes a full
        MaximConfig with an inline key already in place."""
        path = tmp_path / "config.json"
        # Construct directly bypassing the validator
        cfg = MaximConfig(
            lanes=LanesConfigSection(
                large=LaneTierConfig(remote_api_key_ref="sk-inline-bad"),
            ),
        )
        write_config(cfg, path=path)
        # Re-read should fail
        with pytest.raises(ConfigurationError, match="Inline plaintext"):
            load_config(path)


class TestMutateConfig:
    """I-5 fold: read-modify-write under the FileLock."""

    def test_mutate_applies_function_and_persists(self, tmp_path):
        path = tmp_path / "config.json"
        write_config(MaximConfig(llm=LLMConfigSection(profile="initial")), path=path)

        def bump_profile(cfg):
            return replace(cfg, llm=replace(cfg.llm, profile="mutated"))

        new, written_path = mutate_config(bump_profile, path=path)
        assert new.llm.profile == "mutated"
        assert written_path == path
        loaded = load_config(path)
        assert loaded.llm.profile == "mutated"

    def test_mutate_reads_inside_lock(self, tmp_path):
        """The mutator receives the freshly-read config, not a stale one."""
        path = tmp_path / "config.json"
        write_config(MaximConfig(llm=LLMConfigSection(profile="initial")), path=path)

        seen_values = []

        def capture_then_bump(cfg):
            seen_values.append(cfg.llm.profile)
            return replace(cfg, llm=replace(cfg.llm, profile="round1"))

        mutate_config(capture_then_bump, path=path)
        mutate_config(capture_then_bump, path=path)
        # The second mutate must see "round1" — i.e., it re-read INSIDE
        # the lock rather than reusing a cached value.
        assert seen_values == ["initial", "round1"]


# ─────────────────────────────────────────────────────────────────────────────
# Concurrent writers — the bug class I-5 prevents
# ─────────────────────────────────────────────────────────────────────────────


def _worker_set_field(args):
    """Top-level for pickling under multiprocessing."""
    field_path, value, config_path_str = args
    # Re-import inside the worker process

    from maxim.runtime.config_writer import set_field

    set_field(field_path, value, path=Path(config_path_str))
    return True


class TestConcurrentWriters:
    def test_concurrent_set_different_fields_both_persist(self, tmp_path):
        """Two processes setting different fields must both land in the
        final config. The FileLock prevents lost-update on the RMW."""
        path = tmp_path / "config.json"
        # Seed
        write_config(MaximConfig(), path=path)

        args = [
            ("llm.profile", "from-A", str(path)),
            ("proxy.max_concurrent", "16", str(path)),
            ("auto_spawn.tunnel", "false", str(path)),
            ("llm.n_ctx", "16384", str(path)),
        ]

        with multiprocessing.Pool(processes=4) as pool:
            results = pool.map(_worker_set_field, args)

        assert all(results)
        loaded = load_config(path)
        assert loaded.llm.profile == "from-A"
        assert loaded.proxy.max_concurrent == 16
        assert loaded.auto_spawn.tunnel is False
        assert loaded.llm.n_ctx == 16384


# ─────────────────────────────────────────────────────────────────────────────
# Serialization edge cases
# ─────────────────────────────────────────────────────────────────────────────


class TestSerializationCollision:
    """Extra-dict keys that collide with declared fields must raise
    rather than silently shadow."""

    def test_extra_collision_with_declared_field_raises(self, tmp_path):
        path = tmp_path / "config.json"
        # Construct a degenerate LaneTierConfig with extra containing a
        # declared field name; the writer must refuse rather than
        # silently overwriting.
        cfg = MaximConfig(
            lanes=LanesConfigSection(
                large=LaneTierConfig(
                    remote_url="http://example.com",
                    extra={"remote_url": "http://different.com"},  # collision
                ),
            ),
        )
        with pytest.raises(ConfigurationError, match="collides"):
            write_config(cfg, path=path)

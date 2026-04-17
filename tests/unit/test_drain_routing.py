"""Tests for peer/drain_routing.py — DrainConstraint + mtime cache + factory."""

from __future__ import annotations

import time
from pathlib import Path

import pytest


# ── DrainConstraint ────────────────────────────────────────────────────────


class TestDrainConstraint:
    """Unit tests for the DrainConstraint mtime-cached callback."""

    @pytest.fixture
    def drain_file(self, tmp_path: Path) -> Path:
        """Create an empty drain state file."""
        p = tmp_path / "drained_nodes.leader.txt"
        p.write_text("")
        return p

    @pytest.fixture
    def constraint(self, drain_file: Path):
        """Constraint mapping provider 'peer1' -> node 'leader-desk'."""
        from maxim.peer.drain_routing import DrainConstraint

        url_to_node = {"http://192.168.1.10:8099": "leader-desk"}
        provider_urls = {"peer1": "http://192.168.1.10:8099"}
        return DrainConstraint(url_to_node, provider_urls, drain_file, cache_ttl_s=0.0)

    def test_empty_file_no_drain(self, constraint):
        assert constraint.is_drained("peer1") is False

    def test_drained_node_returns_true(self, constraint, drain_file: Path):
        drain_file.write_text("leader-desk\n")
        assert constraint.is_drained("peer1") is True

    def test_unknown_provider_never_drained(self, constraint, drain_file: Path):
        drain_file.write_text("leader-desk\n")
        assert constraint.is_drained("anthropic") is False

    def test_mtime_cache_hit(self, drain_file: Path):
        """Same mtime → no re-read."""
        from maxim.peer.drain_routing import DrainConstraint

        drain_file.write_text("leader-desk\n")
        url_to_node = {"http://192.168.1.10:8099": "leader-desk"}
        provider_urls = {"peer1": "http://192.168.1.10:8099"}
        c = DrainConstraint(url_to_node, provider_urls, drain_file, cache_ttl_s=0.0)

        # First call reads the file.
        assert c.is_drained("peer1") is True

        # Overwrite content WITHOUT changing mtime (write same data).
        # The cache should still return True because mtime hasn't changed.
        # (On most filesystems, writing identical content preserves mtime
        # within 1s resolution, but we force it with os.utime.)
        import os

        original_mtime = drain_file.stat().st_mtime
        drain_file.write_text("")  # clear drain
        os.utime(drain_file, (original_mtime, original_mtime))

        # With TTL=0, stat runs every call, but mtime matches → cache hit.
        assert c.is_drained("peer1") is True  # stale cache, by design

    def test_mtime_change_refreshes(self, constraint, drain_file: Path):
        """mtime change → re-read picks up new content."""
        assert constraint.is_drained("peer1") is False

        drain_file.write_text("leader-desk\n")
        # Force a distinct mtime
        import os

        os.utime(drain_file, (time.time() + 10, time.time() + 10))

        assert constraint.is_drained("peer1") is True

    def test_file_missing_returns_empty(self, tmp_path: Path):
        from maxim.peer.drain_routing import DrainConstraint

        missing = tmp_path / "nonexistent.txt"
        c = DrainConstraint(
            {"http://x:8099": "node1"},
            {"p": "http://x:8099"},
            missing,
            cache_ttl_s=0.0,
        )
        assert c.is_drained("p") is False

    def test_ttl_skips_stat(self, constraint, drain_file: Path):
        """Within TTL window, stat() is skipped entirely."""
        from maxim.peer.drain_routing import DrainConstraint

        drain_file.write_text("")
        c = DrainConstraint(
            {"http://192.168.1.10:8099": "leader-desk"},
            {"peer1": "http://192.168.1.10:8099"},
            drain_file,
            cache_ttl_s=60.0,  # Long TTL
        )

        # First call reads (empty → not drained).
        assert c.is_drained("peer1") is False

        # Write drain state + force new mtime.
        drain_file.write_text("leader-desk\n")
        import os

        os.utime(drain_file, (time.time() + 100, time.time() + 100))

        # Within TTL → still returns cached (not drained), even though
        # the file has changed.
        assert c.is_drained("peer1") is False

    def test_drained_providers_returns_set(self, constraint, drain_file: Path):
        drain_file.write_text("leader-desk\n")
        import os

        os.utime(drain_file, (time.time() + 10, time.time() + 10))
        result = constraint.drained_providers()
        assert result == frozenset({"peer1"})

    def test_inline_comment_stripped(self, constraint, drain_file: Path):
        drain_file.write_text("leader-desk  # needs rebuild\n")
        import os

        os.utime(drain_file, (time.time() + 10, time.time() + 10))
        assert constraint.is_drained("peer1") is True

    def test_comment_line_ignored(self, constraint, drain_file: Path):
        drain_file.write_text("# this is a comment\n")
        import os

        os.utime(drain_file, (time.time() + 10, time.time() + 10))
        assert constraint.is_drained("peer1") is False


# ── canonical URL ──────────────────────────────────────────────────────────


class TestCanonicalUrl:
    def test_strips_trailing_slash(self):
        from maxim.peer.drain_routing import _canonical_url

        assert _canonical_url("http://x:8099/") == "http://x:8099"

    def test_strips_trailing_v1(self):
        from maxim.peer.drain_routing import _canonical_url

        assert _canonical_url("http://x:8099/v1") == "http://x:8099"

    def test_strips_trailing_v1_with_slash(self):
        from maxim.peer.drain_routing import _canonical_url

        assert _canonical_url("http://x:8099/v1/") == "http://x:8099"

    def test_empty_string(self):
        from maxim.peer.drain_routing import _canonical_url

        assert _canonical_url("") == ""

    def test_no_trailing_junk(self):
        from maxim.peer.drain_routing import _canonical_url

        assert _canonical_url("http://x:8099") == "http://x:8099"


# ── build_drain_constraint factory ─────────────────────────────────────────


class TestBuildDrainConstraint:
    """Test the factory that wires mesh.yml topology to provider configs."""

    def _make_mesh_cfg(self, nodes):
        """Minimal mesh config stub with .nodes iterable."""
        from types import SimpleNamespace

        return SimpleNamespace(nodes=nodes)

    def _make_node(self, name: str, url: str):
        from types import SimpleNamespace

        return SimpleNamespace(name=name, url=url)

    def test_returns_none_for_no_matching_urls(self, tmp_path: Path):
        from maxim.peer.drain_routing import build_drain_constraint

        mesh = self._make_mesh_cfg([self._make_node("leader", "http://192.168.1.10:8099/v1")])
        providers = {"anthropic": {"type": "anthropic"}}  # no base_url
        result = build_drain_constraint(mesh, providers, drain_path=tmp_path / "d.txt")
        assert result is None

    def test_returns_none_for_no_provider_urls(self, tmp_path: Path):
        from maxim.peer.drain_routing import build_drain_constraint

        mesh = self._make_mesh_cfg([self._make_node("leader", "http://192.168.1.10:8099/v1")])
        providers = {"local": {"type": "llama_cpp"}}
        result = build_drain_constraint(mesh, providers, drain_path=tmp_path / "d.txt")
        assert result is None

    def test_returns_none_for_url_mismatch(self, tmp_path: Path):
        from maxim.peer.drain_routing import build_drain_constraint

        mesh = self._make_mesh_cfg([self._make_node("leader", "http://192.168.1.10:8099/v1")])
        providers = {"peer": {"type": "maxim_peer", "base_url": "http://10.0.0.5:8099/v1"}}
        result = build_drain_constraint(mesh, providers, drain_path=tmp_path / "d.txt")
        assert result is None

    def test_builds_constraint_on_url_match(self, tmp_path: Path):
        from maxim.peer.drain_routing import DrainConstraint, build_drain_constraint

        drain_file = tmp_path / "drained.txt"
        drain_file.write_text("")
        mesh = self._make_mesh_cfg([self._make_node("leader-desk", "http://192.168.1.10:8099/v1")])
        providers = {"peer1": {"type": "maxim_peer", "base_url": "http://192.168.1.10:8099/v1"}}
        result = build_drain_constraint(mesh, providers, drain_path=drain_file)
        assert isinstance(result, DrainConstraint)

    def test_url_normalization_matches_across_trailing_slash(self, tmp_path: Path):
        from maxim.peer.drain_routing import DrainConstraint, build_drain_constraint

        drain_file = tmp_path / "drained.txt"
        drain_file.write_text("")
        # mesh.yml has /v1, provider has /v1/ — should still match after canonicalization
        mesh = self._make_mesh_cfg([self._make_node("desk", "http://192.168.1.10:8099/v1")])
        providers = {"p": {"type": "maxim_peer", "base_url": "http://192.168.1.10:8099/v1/"}}
        result = build_drain_constraint(mesh, providers, drain_path=drain_file)
        assert isinstance(result, DrainConstraint)

    def test_end_to_end_drain_check(self, tmp_path: Path):
        """Full round trip: mesh node drained → provider is_drained returns True."""
        from maxim.peer.drain_routing import build_drain_constraint

        drain_file = tmp_path / "drained.txt"
        drain_file.write_text("leader-desk\n")

        mesh = self._make_mesh_cfg(
            [
                self._make_node("leader-desk", "http://192.168.1.10:8099/v1"),
                self._make_node("mac-studio", "http://192.168.1.20:8099/v1"),
            ]
        )
        providers = {
            "peer1": {"type": "maxim_peer", "base_url": "http://192.168.1.10:8099/v1"},
            "peer2": {"type": "maxim_peer", "base_url": "http://192.168.1.20:8099/v1"},
            "anthropic": {"type": "anthropic"},
        }
        constraint = build_drain_constraint(mesh, providers, drain_path=drain_file)
        assert constraint is not None
        assert constraint.is_drained("peer1") is True
        assert constraint.is_drained("peer2") is False
        assert constraint.is_drained("anthropic") is False

    def test_cloud_only_setup_returns_none(self, tmp_path: Path):
        """Cloud-only providers (no base_url) → no constraint needed."""
        from maxim.peer.drain_routing import build_drain_constraint

        mesh = self._make_mesh_cfg([self._make_node("leader", "http://192.168.1.10:8099/v1")])
        providers = {
            "anthropic": {"type": "anthropic"},
            "openai": {"type": "openai"},
        }
        result = build_drain_constraint(mesh, providers, drain_path=tmp_path / "d.txt")
        assert result is None


# ── Router integration ─────────────────────────────────────────────────────


class TestRouterDrainIntegration:
    """Test that drain_constraint flows through the router dispatch path.

    Uses LLMRouter with a fake provider config and injected
    drain_constraint callback. Does NOT make real LLM calls — tests
    the candidate filtering and event emission only.
    """

    @pytest.fixture
    def router_with_drain(self):
        """Build a router with two peer providers + drain constraint."""
        import dataclasses

        from maxim.models.language.config import LLMConfig
        from maxim.models.language.router import LLMRouter

        drained_set: set[str] = set()

        def drain_constraint(key: str) -> bool:
            return key in drained_set

        cfg = dataclasses.replace(
            LLMConfig(),
            enabled=True,
            providers={
                "peer1": {
                    "type": "maxim_peer",
                    "base_url": "http://192.168.1.10:8099/v1",
                    "api_key_env": "FAKE_KEY",
                    "model": "fake-model",
                    "allow_local_endpoints": True,
                    "pricing_required": False,
                },
                "peer2": {
                    "type": "maxim_peer",
                    "base_url": "http://192.168.1.20:8099/v1",
                    "api_key_env": "FAKE_KEY",
                    "model": "fake-model",
                    "allow_local_endpoints": True,
                    "pricing_required": False,
                },
            },
        )
        router = LLMRouter(cfg, drain_constraint=drain_constraint)
        return router, drained_set

    def test_drained_provider_excluded_from_candidates(self, router_with_drain):
        router, drained_set = router_with_drain
        drained_set.add("peer1")
        now = time.time()
        candidates, _, _ = router._candidate_providers(0, 100, now)
        assert "peer1" not in candidates
        assert "peer2" in candidates
        assert router._last_drained_keys == ["peer1"]

    def test_undrained_provider_included(self, router_with_drain):
        router, _drained_set = router_with_drain
        now = time.time()
        candidates, _, _ = router._candidate_providers(0, 100, now)
        assert "peer1" in candidates
        assert "peer2" in candidates
        assert router._last_drained_keys == []

    def test_all_drained_produces_empty_candidates(self, router_with_drain):
        router, drained_set = router_with_drain
        drained_set.update({"peer1", "peer2"})
        now = time.time()
        candidates, _, _ = router._candidate_providers(0, 100, now)
        assert candidates == []
        assert sorted(router._last_drained_keys) == ["peer1", "peer2"]

    def test_no_drain_constraint_is_noop(self):
        """drain_constraint=None → zero filtering."""
        import dataclasses

        from maxim.models.language.config import LLMConfig
        from maxim.models.language.router import LLMRouter

        cfg = dataclasses.replace(
            LLMConfig(),
            enabled=True,
            providers={
                "peer1": {
                    "type": "maxim_peer",
                    "base_url": "http://192.168.1.10:8099/v1",
                    "api_key_env": "FAKE_KEY",
                    "model": "fake-model",
                    "allow_local_endpoints": True,
                    "pricing_required": False,
                },
            },
        )
        router = LLMRouter(cfg)  # No drain_constraint
        now = time.time()
        candidates, _, _ = router._candidate_providers(0, 100, now)
        assert "peer1" in candidates
        assert router._last_drained_keys == []

    def test_drain_does_not_affect_backoff_state(self, router_with_drain):
        """Draining a provider must NOT reset its consecutive_errors."""
        from maxim.models.language.types import ProviderState

        router, drained_set = router_with_drain
        # Simulate prior failures on peer1
        router._provider_states["peer1"] = ProviderState(consecutive_errors=3, last_error="timeout")

        drained_set.add("peer1")
        now = time.time()
        router._candidate_providers(0, 100, now)

        # Backoff state must be untouched by drain filtering
        assert router._provider_states["peer1"].consecutive_errors == 3
        assert router._provider_states["peer1"].last_error == "timeout"


# ── Auto-drain thresholds ──────────────────────────────────────────────────


class TestAutoDrainThreshold:
    def test_permanent_failure_threshold_is_one(self):
        from maxim.peer.drain_routing import auto_drain_threshold

        assert auto_drain_threshold("auth_failed") == 1
        assert auto_drain_threshold("model_missing") == 1

    def test_transient_failure_threshold_is_five_by_default(self):
        from maxim.peer.drain_routing import auto_drain_threshold

        assert auto_drain_threshold("down") == 5
        assert auto_drain_threshold("timeout") == 5
        assert auto_drain_threshold("inference_broken") == 5

    def test_env_override(self, monkeypatch):
        from maxim.peer.drain_routing import auto_drain_threshold

        monkeypatch.setenv("MAXIM_AUTO_DRAIN_THRESHOLD", "3")
        assert auto_drain_threshold("down") == 3

    def test_env_clamped_low(self, monkeypatch):
        from maxim.peer.drain_routing import auto_drain_threshold

        monkeypatch.setenv("MAXIM_AUTO_DRAIN_THRESHOLD", "1")
        assert auto_drain_threshold("down") == 2  # min is 2

    def test_env_clamped_high(self, monkeypatch):
        from maxim.peer.drain_routing import auto_drain_threshold

        monkeypatch.setenv("MAXIM_AUTO_DRAIN_THRESHOLD", "100")
        assert auto_drain_threshold("down") == 20  # max is 20


# ── Tagged entry parsing ──────────────────────────────────────────────────


class TestLoadTaggedEntries:
    def test_plain_name_has_none_tag(self, tmp_path: Path):
        from maxim.peer.drain_routing import _load_tagged_entries

        f = tmp_path / "d.txt"
        f.write_text("leader-desk\n")
        result = _load_tagged_entries(f)
        assert result == {"leader-desk": None}

    def test_tagged_name_has_tag(self, tmp_path: Path):
        from maxim.peer.drain_routing import _load_tagged_entries

        f = tmp_path / "d.txt"
        f.write_text("leader-desk  # auto:2026-04-17 reason:auth_failed\n")
        result = _load_tagged_entries(f)
        assert result == {"leader-desk": "auto:2026-04-17 reason:auth_failed"}

    def test_mixed_entries(self, tmp_path: Path):
        from maxim.peer.drain_routing import _load_tagged_entries

        f = tmp_path / "d.txt"
        f.write_text("leader-desk  # auto:2026-04-17\nmac-studio\n")
        result = _load_tagged_entries(f)
        assert result == {"leader-desk": "auto:2026-04-17", "mac-studio": None}

    def test_comment_only_line_skipped(self, tmp_path: Path):
        from maxim.peer.drain_routing import _load_tagged_entries

        f = tmp_path / "d.txt"
        f.write_text("# just a comment\nleader-desk\n")
        result = _load_tagged_entries(f)
        assert result == {"leader-desk": None}

    def test_missing_file_returns_empty(self, tmp_path: Path):
        from maxim.peer.drain_routing import _load_tagged_entries

        result = _load_tagged_entries(tmp_path / "nope.txt")
        assert result == {}


# ── AutoDrainWriter ───────────────────────────────────────────────────────


class TestAutoDrainWriter:
    @pytest.fixture
    def drain_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "drained_nodes.leader.txt"
        p.write_text("")
        return p

    @pytest.fixture
    def writer(self, drain_file: Path):
        from maxim.peer.drain_routing import AutoDrainWriter

        return AutoDrainWriter(
            provider_to_node={"peer1": "leader-desk", "peer2": "mac-studio"},
            drain_path=drain_file,
        )

    def test_writes_tagged_entry(self, writer, drain_file: Path):
        writer.maybe_auto_drain("peer1", "auth_failed")
        content = drain_file.read_text()
        assert "leader-desk" in content
        assert "# auto:" in content
        assert "reason:auth_failed" in content

    def test_idempotent_on_existing_drain(self, writer, drain_file: Path):
        drain_file.write_text("leader-desk\n")  # operator drain (no tag)
        writer.maybe_auto_drain("peer1", "auth_failed")
        content = drain_file.read_text()
        # Should NOT have appended — node already drained
        assert content.count("leader-desk") == 1

    def test_unknown_provider_is_noop(self, writer, drain_file: Path):
        writer.maybe_auto_drain("anthropic", "auth_failed")
        assert drain_file.read_text() == ""

    def test_tagged_entry_parseable_by_load_names(self, writer, drain_file: Path):
        """Existing drain_state._load_names must see the auto-drained node."""
        writer.maybe_auto_drain("peer1", "down")
        from maxim.peer.drain_state import _load_names

        names = _load_names(drain_file)
        assert "leader-desk" in names

    def test_tagged_entry_parseable_by_load_tagged(self, writer, drain_file: Path):
        from maxim.peer.drain_routing import _load_tagged_entries

        writer.maybe_auto_drain("peer1", "timeout")
        entries = _load_tagged_entries(drain_file)
        assert "leader-desk" in entries
        tag = entries["leader-desk"]
        assert tag is not None
        assert tag.startswith("auto:")
        assert "reason:timeout" in tag


# ── Router auto-drain integration ─────────────────────────────────────────


class TestRouterAutoDrainIntegration:
    """Test that auto-drain callbacks fire at the right thresholds."""

    @pytest.fixture
    def router_with_auto_drain(self):
        import dataclasses

        from maxim.models.language.config import LLMConfig
        from maxim.models.language.router import LLMRouter

        auto_drained: list[tuple[str, str]] = []

        def auto_drain_cb(key: str, reason: str) -> None:
            auto_drained.append((key, reason))

        cfg = dataclasses.replace(
            LLMConfig(),
            enabled=True,
            providers={
                "peer1": {
                    "type": "maxim_peer",
                    "base_url": "http://192.168.1.10:8099/v1",
                    "api_key_env": "FAKE_KEY",
                    "model": "fake-model",
                    "allow_local_endpoints": True,
                    "pricing_required": False,
                },
            },
        )
        router = LLMRouter(cfg, auto_drain_callback=auto_drain_cb)
        return router, auto_drained

    def test_permanent_failure_triggers_after_one(self, router_with_auto_drain):
        from maxim.models.language.types import ProviderState

        router, auto_drained = router_with_auto_drain
        # Simulate 1 auth failure (consecutive_errors=1 after _set_long_backoff)
        router._provider_states["peer1"] = ProviderState(consecutive_errors=1, last_error="auth_failed")
        router._maybe_schedule_auto_drain("peer1", "auth_failed")
        # Flush outside lock
        if router._pending_auto_drains:
            for k, r in router._pending_auto_drains:
                router._auto_drain_callback(k, r)
            router._pending_auto_drains.clear()
        assert len(auto_drained) == 1
        assert auto_drained[0] == ("peer1", "auth_failed")

    def test_transient_failure_does_not_trigger_below_threshold(self, router_with_auto_drain):
        from maxim.models.language.types import ProviderState

        router, auto_drained = router_with_auto_drain
        # 4 consecutive errors — below default threshold of 5
        router._provider_states["peer1"] = ProviderState(consecutive_errors=4, last_error="down")
        router._maybe_schedule_auto_drain("peer1", "down")
        assert len(router._pending_auto_drains) == 0
        assert len(auto_drained) == 0

    def test_transient_failure_triggers_at_threshold(self, router_with_auto_drain):
        from maxim.models.language.types import ProviderState

        router, auto_drained = router_with_auto_drain
        # 5 consecutive errors — at default threshold
        router._provider_states["peer1"] = ProviderState(consecutive_errors=5, last_error="down")
        router._maybe_schedule_auto_drain("peer1", "down")
        assert len(router._pending_auto_drains) == 1

    def test_no_callback_is_noop(self):
        import dataclasses

        from maxim.models.language.config import LLMConfig
        from maxim.models.language.router import LLMRouter
        from maxim.models.language.types import ProviderState

        cfg = dataclasses.replace(
            LLMConfig(),
            enabled=True,
            providers={
                "peer1": {
                    "type": "maxim_peer",
                    "base_url": "http://192.168.1.10:8099/v1",
                    "api_key_env": "FAKE_KEY",
                    "model": "fake-model",
                    "allow_local_endpoints": True,
                    "pricing_required": False,
                },
            },
        )
        router = LLMRouter(cfg)  # No auto_drain_callback
        router._provider_states["peer1"] = ProviderState(consecutive_errors=10)
        router._maybe_schedule_auto_drain("peer1", "down")
        assert len(router._pending_auto_drains) == 0

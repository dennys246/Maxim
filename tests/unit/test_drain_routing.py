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

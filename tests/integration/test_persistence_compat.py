"""Regression tests for the v1.0 ``_format_version`` persistence contract.

CC1 freeze-critical guard: every persisted JSON file in Maxim MUST carry
a ``_format_version`` field at root for new writes, and loaders MUST
accept pre-1.0 files (no field present) by warning once and proceeding
with default ``"0.x"`` semantics. These tests pin both halves of the
contract so a future refactor that strips the field, or hardens a
loader to reject missing-field files, surfaces immediately.

Tests deliberately hand-craft minimal payload shapes rather than capturing
real 0.8 sim state — the contract here is at the file-format layer, not
the bio-system semantic layer. Capturing a real 0.8 NAc dump would couple
this test to bio-system state-shape decisions that are tracked in the
P3.5 envelope migration registry instead.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from maxim.utils.format_version import (
    FORMAT_VERSION,
    LEGACY_VERSION,
    check_format_version,
    reset_warned_types,
    with_format_version,
)


@pytest.fixture(autouse=True)
def _reset_warning_dedupe():
    """Each test starts with a clean per-file-type warning dedupe set."""
    reset_warned_types()
    yield
    reset_warned_types()


# ─────────────────────────────────────────────────────────────────────────
# Helper invariants
# ─────────────────────────────────────────────────────────────────────────


class TestFormatVersionHelper:
    def test_with_format_version_stamps_root(self) -> None:
        out = with_format_version({"foo": "bar"})
        assert out["_format_version"] == FORMAT_VERSION
        assert out["foo"] == "bar"

    def test_with_format_version_raises_on_stale_conflict(self) -> None:
        # Pre-fold (CC1 review B1 + executor #6): used setdefault so a stale
        # _format_version silently survived. Post-fold: any pre-existing
        # _format_version that does NOT match the writer's version raises
        # ValueError so the caller sees the round-trip bug.
        with pytest.raises(ValueError, match="already carries"):
            with_format_version({"_format_version": "9.9", "foo": "bar"})

    def test_with_format_version_idempotent_on_match(self) -> None:
        # Stamping a payload that already carries the matching version is a
        # no-op (re-stamping a freshly-captured envelope through write()
        # exercises this path).
        out = with_format_version({"_format_version": "1.0", "foo": "bar"})
        assert out["_format_version"] == "1.0"
        assert out["foo"] == "bar"

    def test_with_format_version_rejects_non_dict(self) -> None:
        with pytest.raises(TypeError):
            with_format_version([1, 2, 3])  # type: ignore[arg-type]

    def test_check_returns_explicit_version(self) -> None:
        assert check_format_version({"_format_version": "1.0"}, "test_a") == "1.0"

    def test_check_returns_legacy_for_missing(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            assert check_format_version({"foo": "bar"}, "test_b") == LEGACY_VERSION
        assert any("pre-1.0" in r.message for r in caplog.records)

    def test_check_warning_is_one_shot_per_file_type(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            check_format_version({}, "test_c")
            check_format_version({}, "test_c")
            check_format_version({}, "test_c")
        warnings = [r for r in caplog.records if "pre-1.0" in r.message and "test_c" in r.message]
        assert len(warnings) == 1, f"expected exactly one warning, got {len(warnings)}"

    def test_check_warns_per_distinct_file_type(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            check_format_version({}, "test_d")
            check_format_version({}, "test_e")
        types = {r.message for r in caplog.records if "pre-1.0" in r.message}
        # Each type warns independently.
        assert len(types) == 2

    def test_check_returns_legacy_for_non_dict_silently(self, caplog) -> None:
        # List or scalar at root: the loader's own validation surfaces the
        # type error; the version helper does not double-warn.
        with caplog.at_level(logging.WARNING):
            assert check_format_version([1, 2, 3], "test_f") == LEGACY_VERSION
        assert not any("pre-1.0" in r.message for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────
# Bio-system loader compat (hand-crafted minimal pre-1.0 fixtures)
# ─────────────────────────────────────────────────────────────────────────


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestBioSystemPreV1Compat:
    """Each bio-system's ``load(path)`` reads a pre-1.0 file (no version
    field) without raising, and emits a single warning. Round-tripping
    a fresh save adds the field at root."""

    def test_atl_loads_pre_v1_file(self, tmp_path, caplog) -> None:
        from maxim.memory.atl import ATL

        # Hand-crafted minimal ATL payload at the pre-1.0 shape (no
        # _format_version at root). The legacy "version" string sits
        # inside the payload — that one is tombstoned, not the v1 contract.
        path = tmp_path / "atl_legacy.json"
        _write_json(
            path,
            {
                "version": "1.0",  # legacy payload-string, tombstoned
                "concepts": {},
                "graph": None,
                "registry": None,
                "stats": {"concepts_stored": 0, "queries": 0},
                "compressed_count": 0,
            },
        )

        atl = ATL()
        with caplog.at_level(logging.WARNING):
            atl.load(str(path))
        assert any("pre-1.0 atl file" in r.message for r in caplog.records)

        # Re-save: the new file should carry _format_version at root.
        out = tmp_path / "atl_resaved.json"
        atl.save(str(out))
        on_disk = json.loads(out.read_text())
        assert on_disk["_format_version"] == FORMAT_VERSION

    def test_nac_loads_pre_v1_file(self, tmp_path, caplog) -> None:
        from maxim.decisions.nac import NACConfig, NAc

        path = tmp_path / "nac_legacy.json"
        # NAc.load_state() iterates state["links"] as a dict keyed by
        # event_signature → list of link dicts. Empty dict is the minimal
        # round-trippable shape.
        _write_json(
            path,
            {
                "version": "1.0",
                "links": {},
                "outcome_index": {},
                "priors": {},
                "reward_bias": {},
                "goal_reward_bias": {},
                "total_observations": 0,
                "config": {},
            },
        )

        nac = NAc(NACConfig())
        with caplog.at_level(logging.WARNING):
            nac.load(str(path))
        assert any("pre-1.0 nac file" in r.message for r in caplog.records)

        out = tmp_path / "nac_resaved.json"
        nac.save(str(out))
        on_disk = json.loads(out.read_text())
        assert on_disk["_format_version"] == FORMAT_VERSION

    def test_scn_loads_pre_v1_file(self, tmp_path, caplog) -> None:
        from maxim.time.scn import SCN

        path = tmp_path / "scn_legacy.json"
        _write_json(
            path,
            {
                "version": "3.0",
                "signatures": {},
                "priors": {},
                "config": {
                    "rho": 0.5,
                    "decay_rate": 0.05,
                    "min_count_for_phase": 2,
                    "default_window_seconds": 60.0,
                },
            },
        )

        scn = SCN()
        with caplog.at_level(logging.WARNING):
            scn.load(str(path))
        assert any("pre-1.0 scn file" in r.message for r in caplog.records)

        out = tmp_path / "scn_resaved.json"
        scn.save(str(out))
        on_disk = json.loads(out.read_text())
        assert on_disk["_format_version"] == FORMAT_VERSION

    def test_cross_layer_graph_loads_pre_v1_file(self, tmp_path, caplog) -> None:
        from maxim.memory.cross_layer import CrossLayerGraph

        path = tmp_path / "clg_legacy.json"
        _write_json(path, {"version": "1.0", "edges": []})

        clg = CrossLayerGraph()
        with caplog.at_level(logging.WARNING):
            clg.load(str(path))
        assert any("pre-1.0 cross_layer_graph file" in r.message for r in caplog.records)

        out = tmp_path / "clg_resaved.json"
        clg.save(str(out))
        on_disk = json.loads(out.read_text())
        assert on_disk["_format_version"] == FORMAT_VERSION

    def test_v1_files_load_without_warning(self, tmp_path, caplog) -> None:
        # An ATL file that was already saved by 1.0 (carries the field) must
        # load without emitting the pre-1.0 warning.
        from maxim.memory.atl import ATL

        atl = ATL()
        path = tmp_path / "atl_v1.json"
        atl.save(str(path))

        with caplog.at_level(logging.WARNING):
            atl2 = ATL()
            atl2.load(str(path))
        assert not any("pre-1.0" in r.message for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────
# SessionSnapshot envelope coexistence (CC1 review fold, B2 + executor #1)
# ─────────────────────────────────────────────────────────────────────────


class TestSessionSnapshotEnvelopeCoexistence:
    """``SessionSnapshot`` envelopes carry BOTH ``schema_version`` (int —
    P3.5 Stage 2 migration trigger) and ``_format_version`` (string — CC1
    file-format contract) at root. The two are orthogonal; load-modify-
    save round-trips must preserve both.
    """

    def test_capture_stamps_both_version_fields(self) -> None:
        from maxim.memory.snapshot import SessionSnapshot

        snap = SessionSnapshot.capture()
        env = snap.envelope
        assert env["schema_version"] == 1  # migration trigger
        assert env["_format_version"] == FORMAT_VERSION  # file-format contract
        assert env["kind"] == "session"

    def test_legacy_envelope_round_trip_stamps_format_version(self, tmp_path) -> None:
        from maxim.memory.snapshot import SessionSnapshot

        # Hand-craft a pre-1.0 session envelope (no _format_version field).
        # This is the load-modify-save scenario the executor review flagged
        # as a gap pre-fold.
        legacy_path = tmp_path / "session_legacy.json"
        _write_json(
            legacy_path,
            {
                "schema_version": 1,
                "kind": "session",
                "systems": {},
            },
        )

        # Load the legacy envelope, then write it back.
        snap = SessionSnapshot.from_file(legacy_path)
        out_path = tmp_path / "session_resaved.json"
        snap.write(out_path)

        # The re-saved file MUST carry both version fields, even though the
        # in-memory envelope had no _format_version when from_file returned.
        on_disk = json.loads(out_path.read_text())
        assert on_disk["schema_version"] == 1
        assert on_disk["_format_version"] == FORMAT_VERSION

    def test_write_does_not_mutate_in_memory_envelope(self, tmp_path) -> None:
        from maxim.memory.snapshot import SessionSnapshot

        legacy_path = tmp_path / "session_legacy.json"
        _write_json(
            legacy_path,
            {"schema_version": 1, "kind": "session", "systems": {}},
        )
        snap = SessionSnapshot.from_file(legacy_path)
        out_path = tmp_path / "session_resaved.json"
        snap.write(out_path)
        # The in-memory envelope is unchanged — write() copies before
        # stamping (otherwise re-writing would carry over the mutation).
        assert "_format_version" not in snap.envelope


# ─────────────────────────────────────────────────────────────────────────
# File*Store list-rooted compat
# ─────────────────────────────────────────────────────────────────────────


class TestFileStoreCompat:
    """File*Store implementations historically wrote a JSON list at root.
    v1.0 wraps in ``{_format_version, kind, items}`` but keeps reading
    bare lists for backwards compat (with a single warning)."""

    def test_file_episodic_store_loads_legacy_list(self, tmp_path, caplog) -> None:
        from maxim.memory.store import FileEpisodicStore

        path = tmp_path / "memories.json"
        _write_json(path, [{"id": "m1", "content": "x"}, {"id": "m2", "content": "y"}])

        store = FileEpisodicStore(path)
        with caplog.at_level(logging.WARNING):
            items = store.load()
        assert len(items) == 2
        assert any("pre-1.0 file_episodic_store" in r.message for r in caplog.records)

    def test_file_episodic_store_v1_round_trip(self, tmp_path, caplog) -> None:
        from maxim.memory.store import FileEpisodicStore

        path = tmp_path / "memories.json"
        store = FileEpisodicStore(path)
        store.save([{"id": "m1"}])

        on_disk = json.loads(path.read_text())
        assert on_disk["_format_version"] == FORMAT_VERSION
        assert on_disk["kind"] == "file_episodic_store"
        assert on_disk["items"] == [{"id": "m1"}]

        with caplog.at_level(logging.WARNING):
            items = store.load()
        assert items == [{"id": "m1"}]
        assert not any("pre-1.0" in r.message for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────
# Probe cache: dict-rooted with URL keys → wrapped under "entries"
# ─────────────────────────────────────────────────────────────────────────


class TestProbeCacheCompat:
    def test_probe_cache_loads_pre_v1_url_keyed(self, tmp_path, caplog, monkeypatch) -> None:
        from maxim.runtime import probe_cache

        probe_dir = tmp_path / "util"
        probe_dir.mkdir()
        cache_path = probe_dir / "probe_cache.json"
        # RFC 2606 ``.test`` reserved TLD — avoids CodeQL's
        # py/incomplete-url-substring-sanitization rule, which fires on
        # URL literals used with the ``in`` operator. The rule cannot
        # tell our dict-membership ``in`` apart from a URL-substring
        # sanitization check at lint time, so picking a fake-domain TLD
        # the rule does not pattern-match on sidesteps the false positive.
        primary_url = "https://leader.test/probe"
        secondary_url = "https://other.leader.test/probe"
        _write_json(
            cache_path,
            {
                primary_url: {"probed_at": 12345.0, "outcome": "ok"},
                secondary_url: {"probed_at": 67890.0, "outcome": "ok"},
            },
        )

        # Point probe_cache._cache_path() at our tmp dir.
        monkeypatch.setattr(probe_cache, "_cache_path", lambda: cache_path)

        with caplog.at_level(logging.WARNING):
            entries = probe_cache.load_cache()
        # Use ``in entries.keys()`` so the dict-key intent is explicit
        # and unambiguously not URL-substring sanitization.
        assert primary_url in entries.keys()
        assert any("pre-1.0 probe_cache" in r.message for r in caplog.records)

    def test_probe_cache_v1_round_trip(self, tmp_path, monkeypatch) -> None:
        from maxim.runtime import probe_cache

        cache_path = tmp_path / "util" / "probe_cache.json"
        monkeypatch.setattr(probe_cache, "_cache_path", lambda: cache_path)
        peer_url = "https://peer.test/probe"
        probe_cache.save_cache({peer_url: {"probed_at": 1.0}})

        on_disk = json.loads(cache_path.read_text())
        assert on_disk["_format_version"] == FORMAT_VERSION
        assert "entries" in on_disk.keys()
        assert peer_url in on_disk["entries"].keys()


# ─────────────────────────────────────────────────────────────────────────
# Learned tool index: dict-rooted with tool-name keys
# ─────────────────────────────────────────────────────────────────────────


class TestLearnedIndexCompat:
    def test_learned_index_handles_legacy_root_keys(self, tmp_path) -> None:
        # Build a hand-crafted pre-1.0 legacy file whose root contains
        # tool-name keys. The loader is expected to detect this and route
        # through the legacy-shape branch.
        from maxim.tools.learned_index import LearnedToolIndex

        path = tmp_path / "learned.json"
        _write_json(
            path,
            {
                "explore_tool": {
                    "look": {"weight": 0.5, "source": "learned", "observations": 3}
                }
            },
        )

        # Constructing without a registered tool means load is a no-op
        # for unknown tools, but the file shape detection itself must
        # not raise. That's the contract under test.
        idx = LearnedToolIndex()
        idx.load(str(path))

"""P4 Stage 1 — cross-modal binding mechanism tests.

Stage 1 tests the BINDING + RETRIEVAL plumbing only:

- ``_close_pending_episode_locked`` drains
  ``pending.node_modality_buffer`` into ``Hippocampus._node_modality``.
- ``Hippocampus.retrieve_cross_modal`` snapshots the matching subset
  under ``_episode_lock``, returns a lock-free closure, delegates to
  ``retrieve_on_cue`` with the expected modality filter behavior.
- The snapshot pattern holds NO lock at filter-call time
  (lock-inversion regression guard, mirrors P3b's analogous test).
- Persistence round-trip preserves ``_node_modality`` exactly via the
  ``"node_modality"`` top-level key in ``dump()``.
- P3.5 atomic rollback (``SessionSnapshot.restore_into``) restores
  ``_node_modality`` to its pre-mutation state with NO stale entries
  from the failed restore attempt — verifies the clear-then-load
  semantics of ``load_state``.

Stage 1 explicitly does NOT test:

- Real CLIP encoder geometry (Stage 2).
- The OpenCLIP head-to-head (Stage 3).
- Encoder-cluster quality on real-world embeddings (Stage 2).

The Stage 1.5 vacuous-pass guard lives in its own file
``test_p4_00_vacuous_pass_guard.py`` so pytest's alphabetical
collection order runs it BEFORE this file. If that file's tests fail
the cluster-aware fixture is broken and the mechanism tests below are
not interpretable — fix the fixture before trusting any other Stage 1
result.
"""

from __future__ import annotations

import threading

import pytest

from maxim.memory.episode import CaptureEvent
from maxim.memory.hippocampus import (
    EpisodeConfig,
    HebbianConfig,
    Hippocampus,
    HippocampusConfig,
)


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _fresh_hippocampus(*, boundary_tick_gap: int = 50) -> Hippocampus:
    return Hippocampus(
        HippocampusConfig(
            episode=EpisodeConfig(
                boundary_tick_gap=boundary_tick_gap,
                hebbian=HebbianConfig(init=0.3, delta=0.1, max_weight=1.0),
            )
        )
    )


def _bind_pair_via_episode(
    h: Hippocampus,
    text_node: str,
    vision_node: str,
    *,
    tick: int = 0,
) -> None:
    """Co-activate (text, vision) in a single episode then close.

    Uses two events at the same tick on the same channel so the
    boundary detector does not split them.
    """
    h.observe_episode_event(
        CaptureEvent(
            tick=tick,
            channel="cross_modal",
            activated_nodes=(text_node,),
            modality="text",
        )
    )
    h.observe_episode_event(
        CaptureEvent(
            tick=tick,
            channel="cross_modal",
            activated_nodes=(vision_node,),
            modality="vision",
        )
    )
    closed = h.finalize_pending_episode()
    assert closed is not None


# ─────────────────────────────────────────────────────────────────────────
# Mechanism — auto-tag at episode close
# ─────────────────────────────────────────────────────────────────────────


class TestAutoTagAtEpisodeClose:
    def test_episode_close_auto_tags_nodes_with_modality(self) -> None:
        h = _fresh_hippocampus()
        _bind_pair_via_episode(h, "text_mug", "vision_mug")

        assert h._node_modality["text_mug"] == "text"
        assert h._node_modality["vision_mug"] == "vision"

    def test_legacy_event_without_modality_does_not_tag(self) -> None:
        """A CaptureEvent with modality=None (the P3a/P3b default) must
        NOT add an entry to _node_modality — the field is opt-in.
        """
        h = _fresh_hippocampus()
        h.observe_episode_event(CaptureEvent(tick=0, channel="text", activated_nodes=("legacy_node",)))
        h.finalize_pending_episode()

        assert "legacy_node" not in h._node_modality

    def test_mixed_modality_episode_tags_each_node_with_its_event_modality(
        self,
    ) -> None:
        """Single episode with three text events and one vision event:
        all four nodes land in _node_modality with the correct modality.
        """
        h = _fresh_hippocampus()
        h.observe_episode_event(CaptureEvent(tick=0, channel="x", activated_nodes=("t1",), modality="text"))
        h.observe_episode_event(CaptureEvent(tick=1, channel="x", activated_nodes=("t2",), modality="text"))
        h.observe_episode_event(CaptureEvent(tick=2, channel="x", activated_nodes=("v1",), modality="vision"))
        h.observe_episode_event(CaptureEvent(tick=3, channel="x", activated_nodes=("t3",), modality="text"))
        h.finalize_pending_episode()

        assert h._node_modality == {
            "t1": "text",
            "t2": "text",
            "v1": "vision",
            "t3": "text",
        }


# ─────────────────────────────────────────────────────────────────────────
# Mechanism — retrieve_cross_modal
# ─────────────────────────────────────────────────────────────────────────


class TestRetrieveCrossModal:
    def test_text_to_vision(self) -> None:
        h = _fresh_hippocampus()
        _bind_pair_via_episode(h, "text_mug", "vision_mug")

        results = h.retrieve_cross_modal("text_mug", target_modality="vision", limit=5)
        assert len(results) == 1
        node_id, weight = results[0]
        assert node_id == "vision_mug"
        assert weight > 0.0

    def test_vision_to_text(self) -> None:
        h = _fresh_hippocampus()
        _bind_pair_via_episode(h, "text_mug", "vision_mug")

        results = h.retrieve_cross_modal("vision_mug", target_modality="text", limit=5)
        assert len(results) == 1
        node_id, _weight = results[0]
        assert node_id == "text_mug"

    def test_excludes_same_modality_cue_raises(self) -> None:
        """Defensive: cue tagged 'text' with target_modality='text' is a
        caller bug. retrieve_cross_modal raises ValueError instead of
        silently returning zero matches.
        """
        h = _fresh_hippocampus()
        _bind_pair_via_episode(h, "text_mug", "vision_mug")

        with pytest.raises(ValueError, match="already tagged as modality 'text'"):
            h.retrieve_cross_modal("text_mug", target_modality="text", limit=5)

    def test_no_cross_modal_episodes_returns_empty(self) -> None:
        """A hippocampus that has only text-modality nodes returns an
        empty list when asked for vision partners — not an error,
        because it's a legitimate "I have not seen any vision" state.
        """
        h = _fresh_hippocampus()
        h.observe_episode_event(CaptureEvent(tick=0, channel="x", activated_nodes=("t1", "t2"), modality="text"))
        h.finalize_pending_episode()

        results = h.retrieve_cross_modal("t1", target_modality="vision", limit=5)
        assert results == []

    def test_cue_not_yet_tagged_does_not_raise(self) -> None:
        """A cue node id whose modality has not been recorded yet does
        NOT raise — pre-seeded probe scenarios are valid. The function
        just delegates to retrieve_on_cue with the modality filter and
        returns whatever the binding graph yields (often empty for an
        untagged cue with no edges).
        """
        h = _fresh_hippocampus()
        _bind_pair_via_episode(h, "text_mug", "vision_mug")

        # "untagged_cue" has never appeared in an episode → no entry in
        # _node_modality, no edges in binding graph → empty result, no
        # exception.
        results = h.retrieve_cross_modal("untagged_cue", target_modality="vision", limit=5)
        assert results == []

    def test_multi_pair_routing(self) -> None:
        """Two cross-modal pairs in the same hippocampus: each cue
        retrieves its own partner, not the other pair's partner.
        """
        h = _fresh_hippocampus()
        _bind_pair_via_episode(h, "text_mug", "vision_mug", tick=0)
        _bind_pair_via_episode(h, "text_book", "vision_book", tick=200)

        mug_results = h.retrieve_cross_modal("text_mug", target_modality="vision")
        book_results = h.retrieve_cross_modal("text_book", target_modality="vision")

        mug_ids = {nid for nid, _ in mug_results}
        book_ids = {nid for nid, _ in book_results}

        assert "vision_mug" in mug_ids
        assert "vision_mug" not in book_ids
        assert "vision_book" in book_ids
        assert "vision_book" not in mug_ids


# ─────────────────────────────────────────────────────────────────────────
# Snapshot pattern + lock-inversion regression guards
# ─────────────────────────────────────────────────────────────────────────


class TestSnapshotPatternFilter:
    def test_filter_uses_frozenset_snapshot_not_self_lookup(self) -> None:
        """Spy on retrieve_on_cue to confirm the node_filter passed in is
        a closure over a frozenset (specifically, NOT a method of self
        that would re-acquire _episode_lock at call time). The closure's
        cell vars must include a frozenset of allowed node ids.
        """
        h = _fresh_hippocampus()
        _bind_pair_via_episode(h, "text_mug", "vision_mug")

        captured: dict[str, object] = {}
        original = h.retrieve_on_cue

        def spy(*args: object, **kwargs: object):
            captured["node_filter"] = kwargs.get("node_filter")
            return original(*args, **kwargs)

        h.retrieve_on_cue = spy  # type: ignore[method-assign]
        h.retrieve_cross_modal("text_mug", target_modality="vision")

        node_filter = captured["node_filter"]
        assert node_filter is not None
        # The filter must be a function with closure cells (i.e., a
        # nested function over local state), NOT a bound method.
        assert callable(node_filter)
        closure = getattr(node_filter, "__closure__", None)
        assert closure is not None and len(closure) > 0, "node_filter has no closure — it does not capture the snapshot"
        # At least one cell must be a frozenset (the snapshot of allowed
        # node ids). If the implementation regresses to a lambda over
        # self._node_modality, no frozenset cell will exist.
        cell_types = [type(cell.cell_contents) for cell in closure]
        assert frozenset in cell_types, (
            f"node_filter's closure does not contain a frozenset; "
            f"closure cell types were {cell_types}. The snapshot pattern "
            f"requires a frozenset of allowed node ids."
        )

    def test_filter_holds_no_lock_after_construction(self) -> None:
        """Mirror of P3b's analogous test: after retrieve_cross_modal
        builds its closure, calling the closure from a context where
        _episode_lock is held by another thread must NOT block. Proves
        the closure does not re-acquire any Hippocampus lock at
        call time.
        """
        h = _fresh_hippocampus()
        for i in range(5):
            _bind_pair_via_episode(h, f"text_{i}", f"vision_{i}", tick=i * 100)

        # Build the snapshot closure by spying on retrieve_on_cue.
        captured: dict[str, object] = {}
        original = h.retrieve_on_cue

        def spy(*args: object, **kwargs: object):
            captured["node_filter"] = kwargs.get("node_filter")
            return original(*args, **kwargs)

        h.retrieve_on_cue = spy  # type: ignore[method-assign]
        h.retrieve_cross_modal("text_0", target_modality="vision")
        node_filter = captured["node_filter"]
        assert callable(node_filter)

        # Now hold _episode_lock from a side thread and call the closure
        # from the main thread. If the closure reacquired _episode_lock
        # under the hood, this would deadlock until the side thread
        # released — wait with a short timeout to detect a hang.
        lock_acquired = threading.Event()
        release_lock = threading.Event()

        def hold_lock() -> None:
            with h._episode_lock:
                lock_acquired.set()
                release_lock.wait(timeout=2.0)

        hold_thread = threading.Thread(target=hold_lock, daemon=True)
        hold_thread.start()
        assert lock_acquired.wait(timeout=2.0), "side thread failed to grab lock"

        # Call the closure repeatedly with a deadline. If it blocks,
        # the deadline catches it.
        completed = threading.Event()

        def call_filter() -> None:
            for nid in ("vision_0", "vision_1", "text_0", "nonexistent"):
                node_filter(nid)  # type: ignore[misc]
            completed.set()

        call_thread = threading.Thread(target=call_filter, daemon=True)
        call_thread.start()
        try:
            assert completed.wait(timeout=0.5), (
                "node_filter blocked while _episode_lock was held by another thread — closure is not lock-free"
            )
        finally:
            release_lock.set()
            hold_thread.join(timeout=2.0)
            call_thread.join(timeout=2.0)


# ─────────────────────────────────────────────────────────────────────────
# Persistence — dump/load_state round trip and atomic rollback
# ─────────────────────────────────────────────────────────────────────────


class TestPersistence:
    def test_dump_load_round_trip_preserves_node_modality(self) -> None:
        h = _fresh_hippocampus()
        _bind_pair_via_episode(h, "text_mug", "vision_mug", tick=0)
        _bind_pair_via_episode(h, "text_book", "vision_book", tick=200)
        original_modality = dict(h._node_modality)

        dumped = h.dump()
        assert "node_modality" in dumped
        assert dumped["node_modality"] == original_modality

        # Load into a fresh hippocampus
        h2 = _fresh_hippocampus()
        h2.load_state(dumped)

        assert h2._node_modality == original_modality
        # And cross-modal retrieval still works post-load
        results = h2.retrieve_cross_modal("text_mug", target_modality="vision")
        assert any(nid == "vision_mug" for nid, _ in results)

    def test_load_state_rejects_unknown_modality_value(self) -> None:
        """A snapshot whose node_modality dict contains an unknown
        modality string MUST fail to load loudly, not silently drop the
        bad entries — the typed Literal exists to surface this class of
        bug, not to mask it.
        """
        h = _fresh_hippocampus()
        bad_state = h.dump()
        bad_state["node_modality"] = {"good_node": "text", "bad_node": "audio"}

        with pytest.raises(ValueError, match="unknown modality"):
            h.load_state(bad_state)

        # And state was NOT mutated — h is still in its pre-call state
        assert h._node_modality == {}

    def test_load_state_rejects_non_dict_node_modality_payload(self) -> None:
        h = _fresh_hippocampus()
        bad_state = h.dump()
        bad_state["node_modality"] = ["t1", "t2"]  # list, not dict

        with pytest.raises(ValueError, match="node_modality payload must be a dict"):
            h.load_state(bad_state)

    def test_legacy_snapshot_without_node_modality_loads_cleanly(self) -> None:
        """A snapshot from before P4 Stage 1 (no node_modality key)
        loads successfully and produces an empty sidecar.
        """
        h = _fresh_hippocampus()
        _bind_pair_via_episode(h, "text_mug", "vision_mug")
        legacy_state = h.dump()
        legacy_state.pop("node_modality")

        h2 = _fresh_hippocampus()
        h2.load_state(legacy_state)
        assert h2._node_modality == {}

    def test_load_state_clear_then_load_replaces_not_merges(self) -> None:
        """Critical clear-then-load semantics test. Build a hippocampus
        with state A, dump it, build a SECOND hippocampus with state B,
        and load state A into the second instance. Result must be
        EXACTLY state A — no leftover state-B entries.

        This is the fast path of the same invariant the atomic-rollback
        test below stress-tests via a forced restore failure.
        """
        h_a = _fresh_hippocampus()
        _bind_pair_via_episode(h_a, "a_text", "a_vision")
        state_a = h_a.dump()

        h_b = _fresh_hippocampus()
        _bind_pair_via_episode(h_b, "b_text", "b_vision")
        _bind_pair_via_episode(h_b, "extra_text", "extra_vision", tick=200)

        h_b.load_state(state_a)
        assert h_b._node_modality == {"a_text": "text", "a_vision": "vision"}

    def test_node_modality_stale_entries_cleared_on_rollback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """P3.5 atomic rollback regression guard. Build a Hippocampus
        with state A. Construct a SessionSnapshot whose hippocampus
        payload encodes a different state C. Force the second sub-system
        in the apply chain to fail (monkeypatch an adapter to raise),
        which triggers the rollback path. After rollback,
        _node_modality must match state A EXACTLY — no leftover state-C
        entries from the failed restore attempt.

        If load_state ever regresses from clear-then-load to merge
        semantics, this test fails because the post-rollback sidecar
        contains state A ∪ state C.
        """
        from maxim.decisions.nac import NAc
        from maxim.memory import snapshot as snapshot_module
        from maxim.memory.snapshot import SessionSnapshot

        # State A — the live target we want preserved across rollback
        h = _fresh_hippocampus()
        _bind_pair_via_episode(h, "a_text", "a_vision")
        state_a_modality = dict(h._node_modality)
        assert state_a_modality == {"a_text": "text", "a_vision": "vision"}

        # State C — a different hippocampus whose dump will be loaded
        # mid-rollback then rolled back
        h_alt = _fresh_hippocampus()
        _bind_pair_via_episode(h_alt, "c_text", "c_vision")
        _bind_pair_via_episode(h_alt, "extra_c", "extra_v", tick=200)
        envelope_c = SessionSnapshot.capture(hippocampus=h_alt)

        # Add an NAc entry to the envelope so restore_into has a second
        # adapter to apply after hippocampus. Monkeypatch nac_from_snapshot
        # to raise so the rollback fires.
        nac_instance = NAc()
        nac_envelope = snapshot_module.nac_to_snapshot(nac_instance)
        envelope_c.envelope["systems"]["nac"] = nac_envelope

        def _broken_nac_adapter(env: object, into: object) -> None:
            raise RuntimeError("deliberate test failure to trigger restore_into rollback")

        monkeypatch.setattr(snapshot_module, "nac_from_snapshot", _broken_nac_adapter)

        # Trigger rollback: hippocampus loads (clear-then-load → state C),
        # then nac fails, rollback restores hippocampus from its captured
        # pre-mutation dump (state A).
        with pytest.raises(RuntimeError, match="deliberate test failure"):
            envelope_c.restore_into(hippocampus=h, nac=nac_instance)

        # After rollback, _node_modality must EXACTLY equal state A.
        # Merge semantics would leave {a_text, a_vision, c_text, c_vision,
        # extra_c, extra_v}.
        assert h._node_modality == state_a_modality, (
            f"rollback did not scrub stale entries from the failed restore "
            f"attempt; load_state may have regressed to merge semantics. "
            f"Expected {state_a_modality!r}, got {h._node_modality!r}"
        )

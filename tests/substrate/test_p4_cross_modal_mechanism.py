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
import time

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

        # Round 2 Exec-lens fold: the frozenset must contain EXACTLY
        # the target-modality nodes (minus the cue), not an arbitrary
        # frozenset. A regression that wrote e.g.
        # ``allowed = frozenset(self._node_modality.keys())`` — all
        # tagged nodes ignoring target_modality — would still produce
        # a frozenset cell and pass the shape-only check above. Pin
        # the content.
        actual_allowed = next(cell.cell_contents for cell in closure if isinstance(cell.cell_contents, frozenset))
        assert actual_allowed == frozenset({"vision_mug"}), (
            f"frozenset snapshot content is wrong: expected frozenset({{'vision_mug'}}), got {actual_allowed!r}"
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

        **Round 2 Exec-lens fold:** the previous version of this test
        only proved "empty state is still empty after raise." That's
        happy-path — the real invariant is fail-before-mutate: a
        malformed payload must NOT half-mutate a hippocampus that
        already holds prior state. Seed the hippocampus first, then
        attempt the bad load, then assert EVERY piece of state is
        untouched.
        """
        h = _fresh_hippocampus()
        _bind_pair_via_episode(h, "pre_text", "pre_vision", tick=0)
        _bind_pair_via_episode(h, "pre_text_2", "pre_vision_2", tick=200)

        pre_modality = dict(h._node_modality)
        pre_episode_count = len(h._episode_store)
        pre_ordinal = h._next_episode_ordinal
        pre_binding_edges = {
            (src, tgt) for src in ("pre_text", "pre_text_2") for tgt, _ in h._binding_graph.get_associated(src)
        }
        assert pre_modality == {
            "pre_text": "text",
            "pre_vision": "vision",
            "pre_text_2": "text",
            "pre_vision_2": "vision",
        }
        assert pre_episode_count == 2

        bad_state = h.dump()
        bad_state["node_modality"] = {"good_node": "text", "bad_node": "audio"}

        with pytest.raises(ValueError, match="unknown modality"):
            h.load_state(bad_state)

        # Every piece of pre-mutation state must be intact — the
        # validation must run BEFORE any write lock is acquired.
        assert h._node_modality == pre_modality, "load_state half-mutated _node_modality despite failing validation"
        assert len(h._episode_store) == pre_episode_count
        assert h._next_episode_ordinal == pre_ordinal
        post_binding_edges = {
            (src, tgt) for src in ("pre_text", "pre_text_2") for tgt, _ in h._binding_graph.get_associated(src)
        }
        assert post_binding_edges == pre_binding_edges

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

        # Round 2 Exec-lens fold: the broken nac adapter must ALSO
        # assert that hippocampus has already been mutated to state C
        # at this point. If SNAPSHOT_KINDS is ever reordered so that
        # nac precedes hippocampus, this assertion fires INSIDE the
        # raise and the test fails for a specific, loud reason instead
        # of silently trivial-passing (empty apply_list → no-op
        # rollback → post-state equals pre-state, the test's outer
        # assertion would pass vacuously).
        def _broken_nac_adapter(env: object, into: object) -> None:
            # h is the hippocampus instance; capture it from the
            # enclosing scope. If it has NOT been mutated to state C
            # by this point (i.e. the hippocampus adapter did not run
            # first), the test premise is broken and we surface it
            # loudly.
            assert h._node_modality != state_a_modality, (
                "broken nac adapter fired BEFORE hippocampus was mutated — "
                "SNAPSHOT_KINDS ordering may have been reversed; this test "
                "is no longer exercising the rollback path"
            )
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


# ─────────────────────────────────────────────────────────────────────────
# Round 2 pre-merge review folds — additional regression guards
# ─────────────────────────────────────────────────────────────────────────


class TestCueExemptionWithInGraphUntaggedCue:
    """Round 2 Exec-lens fold: the docstring for retrieve_cross_modal
    claims cue exemption is "structural to cross-modal semantics" and
    describes a "pre-seeded probe" scenario. The pre-fold test
    (``test_cue_not_yet_tagged_does_not_raise``) only exercised a cue
    that was not in the binding graph at all — spreading_activation
    early-returned before touching the cue-exempt code path. This
    class directly exercises the exemption: the cue IS in the binding
    graph (has Hebbian edges) but is NOT tagged in _node_modality, so
    without the exemption the source-filter check would reject it and
    the retrieval would return empty.
    """

    def test_untagged_cue_in_graph_retrieves_tagged_neighbor(self) -> None:
        h = _fresh_hippocampus()

        # Bind a normal cross-modal pair so there's a tagged vision node
        # and a binding edge exists in the graph.
        _bind_pair_via_episode(h, "text_probe", "vision_target", tick=0)

        # Now manually add an edge from a NEW untagged cue to the
        # existing tagged vision node. We do this by observing another
        # episode that includes both "untagged_probe" (no modality)
        # and "vision_target" (already tagged vision). The untagged
        # probe lands in the binding graph but NOT in _node_modality.
        h.observe_episode_event(
            CaptureEvent(
                tick=500,
                channel="cross_modal",
                activated_nodes=("untagged_probe",),
            )
        )
        h.observe_episode_event(
            CaptureEvent(
                tick=501,
                channel="cross_modal",
                activated_nodes=("vision_target",),
                modality="vision",
            )
        )
        h.finalize_pending_episode()

        # Precondition: untagged_probe IS in the binding graph (has an
        # edge to vision_target) but is NOT in _node_modality.
        assert "untagged_probe" not in h._node_modality
        associated = h._binding_graph.get_associated("untagged_probe")
        assert any(tgt == "vision_target" for tgt, _ in associated), (
            "precondition: untagged_probe must have an edge to vision_target"
        )

        # This is the cue-exemption code path:
        # _modality_filter(untagged_probe) should return True via the
        # `node_id == cue_node_id` branch, letting spreading_activation
        # seed the walk from an untagged source.
        results = h.retrieve_cross_modal("untagged_probe", target_modality="vision")
        partner_ids = {nid for nid, _ in results}
        assert "vision_target" in partner_ids, (
            "cue exemption did not let spreading_activation seed from an untagged cue that has cross-modal neighbors"
        )


class TestLastWriteWinsOnDuplicateNodeIdWithinEpisode:
    """Round 2 Arch-lens fold: the drain comment in
    _close_pending_episode_locked notes "last-write-wins on duplicate
    keys" as the intended semantics for the degenerate case of one
    node id carrying two different modalities within a single pending
    episode. That contract was acknowledged but untested.
    """

    def test_same_node_two_modalities_last_event_wins(self) -> None:
        h = _fresh_hippocampus()
        h.observe_episode_event(CaptureEvent(tick=0, channel="x", activated_nodes=("shared",), modality="text"))
        h.observe_episode_event(CaptureEvent(tick=1, channel="x", activated_nodes=("shared",), modality="vision"))
        h.finalize_pending_episode()
        # The later event wins because the buffer is a dict and drain
        # iterates items in insertion order (Python 3.7+ guarantees).
        assert h._node_modality["shared"] == "vision"


class TestStageThreeLimitation:
    """Pin the current single-hop cross-modal limitation as a
    regression guard. If a future fix enables multi-hop traversal
    through same-modality intermediates (``text_cue → text_bridge →
    vision_target``), this test FAILS and forces an explicit decision
    about whether the new behavior is desired.

    **Stage 2 v3 measurement (2026-04-16) confirmed this limitation
    is empirically harmless:** Option 2 lift = 0.0000 ± 0.0000 across
    10 seeds. Same-class activation (0.490) dominates cross-class
    bridge activation (0.022) by 22:1, so Option 2 cannot improve
    top-5 precision. Option 2 is deferred as post-Stage-3 cleanup.
    See ``docs/experiments/p4_option2_measurement.md``.

    This test remains as the architectural bookmark for when Option 2
    eventually ships (the ``node_filter`` → ``traversal_filter`` +
    ``result_filter`` split). When that happens, flip the assertion
    and rename this class.
    """

    def test_multi_hop_through_same_modality_intermediate_is_blocked(self) -> None:
        h = _fresh_hippocampus()

        # Episode 1: text_cue co-activates with text_bridge. Both
        # tagged "text"; a Hebbian edge forms between them.
        h.observe_episode_event(CaptureEvent(tick=0, channel="x", activated_nodes=("text_cue",), modality="text"))
        h.observe_episode_event(CaptureEvent(tick=1, channel="x", activated_nodes=("text_bridge",), modality="text"))
        h.finalize_pending_episode()

        # Episode 2: text_bridge co-activates with vision_target.
        # Hebbian edge forms between them across episodes (same cue
        # appears in two episodes, the binding graph accumulates edges).
        h.observe_episode_event(CaptureEvent(tick=200, channel="x", activated_nodes=("text_bridge",), modality="text"))
        h.observe_episode_event(
            CaptureEvent(tick=201, channel="x", activated_nodes=("vision_target",), modality="vision")
        )
        h.finalize_pending_episode()

        # Path in binding graph: text_cue → text_bridge → vision_target
        # (two hops, both edges at weight >= 0.3).
        hop1 = {tgt for tgt, _ in h._binding_graph.get_associated("text_cue")}
        hop2 = {tgt for tgt, _ in h._binding_graph.get_associated("text_bridge")}
        assert "text_bridge" in hop1, "precondition: text_cue → text_bridge edge missing"
        assert "vision_target" in hop2, "precondition: text_bridge → vision_target edge missing"

        # Current Stage 1 behavior: the modality filter rejects
        # text_bridge (same-modality, not cue), so the BFS truncates
        # there and vision_target is unreachable.
        results = h.retrieve_cross_modal("text_cue", target_modality="vision")
        partner_ids = {nid for nid, _ in results}
        assert "vision_target" not in partner_ids, (
            "multi-hop cross-modal through same-modality intermediate now works — "
            "this is either a Stage 2/3 intentional fix (update this test to assert "
            "vision_target IS retrieved) or a silent regression in the filter. "
            "See the PR description's Stage 2/3 design-decision note."
        )


class TestConcurrencyCrossLockSmoke:
    """Round 2 Exec-lens fold: add a cross-lock smoke test that
    spawns concurrent ``dump()`` + ``observe_episode_event`` calls
    and asserts both make forward progress. Guards against any
    future refactor that adds a ``_rwlock`` acquisition inside a
    ``_episode_lock`` holder (which would deadlock against
    ``dump()``'s ``_rwlock → _episode_lock`` order).
    """

    def test_dump_and_observe_do_not_deadlock(self) -> None:
        h = _fresh_hippocampus()
        # Seed with a small amount of state so dump has something to
        # serialize.
        _bind_pair_via_episode(h, "seed_text", "seed_vision")

        stop = threading.Event()
        dump_count = 0
        observe_count = 0
        error_box: list[Exception] = []

        def dumper() -> None:
            nonlocal dump_count
            try:
                while not stop.is_set():
                    h.dump()
                    dump_count += 1
            except Exception as e:
                error_box.append(e)

        def observer() -> None:
            nonlocal observe_count
            try:
                tick = 10000
                while not stop.is_set():
                    h.observe_episode_event(
                        CaptureEvent(
                            tick=tick,
                            channel="concurrent",
                            activated_nodes=(f"t_{tick}",),
                            modality="text",
                        )
                    )
                    h.observe_episode_event(
                        CaptureEvent(
                            tick=tick + 1,
                            channel="concurrent",
                            activated_nodes=(f"v_{tick}",),
                            modality="vision",
                        )
                    )
                    h.finalize_pending_episode()
                    observe_count += 1
                    tick += 100
            except Exception as e:
                error_box.append(e)

        threads = [
            threading.Thread(target=dumper, daemon=True),
            threading.Thread(target=observer, daemon=True),
        ]
        for t in threads:
            t.start()

        # Run for 0.5s. If there's a deadlock, both threads stall and
        # forward progress (dump_count / observe_count) stops.
        time.sleep(0.5)
        stop.set()
        for t in threads:
            t.join(timeout=2.0)
            assert not t.is_alive(), "thread did not exit within join deadline — deadlock suspected"

        assert not error_box, f"exception in concurrent worker: {error_box}"
        assert dump_count > 5, f"dumper made only {dump_count} dumps — likely blocked"
        assert observe_count > 5, f"observer made only {observe_count} episode closes — likely blocked"

"""The memory session's END, in phases, within the caller's deadline (#1438 PR 1a; docs/plans/bounded_stop.md §5).

``MemoryHub.on_session_end`` / ``on_session_end_lightweight`` / ``sleep`` move their ticket to ENDING and call
:func:`end_session`. It SAVES FIRST: the saves are the point of a stop and cost 0.5-1.2 s at every measured size
(plan §1), while consolidation is the slow, deferrable part. The phases:

a. **Flush the writers** (best effort, each within a share of the deadline capped at its old timeout): the capture
   workers first (a flushed capture feeds the two below), then the ConceptExtractor (flushed, then stopped), then the
   ``semantic-embedder`` thread (drained, not stopped: nothing restarts it). A writer still busy is LOGGED, never a
   reason to skip a store: every store guards itself (the table below), so a save is a consistent snapshot whatever
   thread writes (owner decision 2026-10-29), and a late write is simply in the next save.
b. **Persist, by value, regardless of the deadline:** NAc decay (once) and the spatial bridge's end, then ONE
   hippocampus READ section holding the hippocampus (``_save_locked``), EC, the embedding store and the cross-layer
   graph, then NAc, ATL, SCN, AG.
c. **Consolidate (full only), each step only if the time left covers its reserve:** sleep, semantic promotion, ATL,
   AG. A skipped step is logged by name and listed in ``results["consolidation_skipped"]``.
d. **If any step of c ran, re-save, dependants before owners** (SCN, NAc, ATL, then the read section in the order
   cross-layer, embeddings, EC, hippocampus, then AG): a crash part-way leaves a hippocampus-keyed dependant holding
   FEWER refs than its owner, and the loaders prune what is left (``MemoryHub._prune_dangling_refs``). NAc is saved
   before EC here, so a percept that registers an EC node during the end can leave ``nac.json`` naming a node the
   older ``ec.json`` lacks; the next open prunes it.
e. **The plan manager and provenance end last** (after consolidation, so its events reach the session summary), then
   the hub's remaining components stop (``MemoryHub.shutdown``, which re-flushes nothing a phase-a stop left).

**Why one read section.** A hippocampus eviction runs its deletion callbacks (EC, the embedding store, the cross-layer
graph, NAc, SCN) under the hippocampus WRITE lock, so no eviction can land between the four saves inside the
section and their files agree. The RWLock is writer-priority and NOT re-entrant: nothing inside the section may take
the hippocampus lock again (``_save_locked`` / ``_dump_locked``, no ``len(hippocampus)``), and none of the four
saves takes it.

**Store/lock table** (what guards each store against a concurrent writer while it saves):

======================================  ====================================================================
Store                                   Guard
======================================  ====================================================================
hippocampus, ATL, AG                    their RWLock (the hippocampus's held across the read section)
the hippocampus associative graph       ``DependencyGraph._lock``
EC, the embedding store                 their own leaf locks (#1438 PR 1a); saved inside the read section
NAc                                     ``_lock`` in ``dump``; ``remove_memory`` mutates ``memory_ids``
                                        outside it: lost-or-present, never torn
SCN                                     its re-entrant leaf lock (#1217)
cross-layer graph                       none: eviction callbacks run under the hippocampus write lock, but the
                                        extractor and the promoter add edges outside any lock; harmless while
                                        the graph is PATHLESS in every shipped builder (#1530)
plan manager, provenance, cost tracker  none (pauses / writes outside their locks); not keyed on memory ids
======================================  ====================================================================

NAc's ``_register_causal_in_ec`` is off (``ec=None`` in every shipped builder); a builder passing ``ec=`` must
re-check the NAc/EC order. The escalation bridge is NOT saved here: its file is machine-wide (#984).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from maxim.exceptions import StoreOverwriteRefused
from maxim.utils.deadline import Deadline

if TYPE_CHECKING:
    from maxim.integration.memory_hub import MemoryHub

logger = logging.getLogger(__name__)

# Phase c reserves, from the measured stop path (plan §1: sleep 0.4-0.6 s at 3000 memories, the rest ~0.01 s),
# with headroom for a CM4 several times slower.
SLEEP_RESERVE_S = 3.0
SLEEP_RESERVE_PER_1000_MEMORIES_S = 1.0
STEP_RESERVE_S = 1.0

# Phase a: each writer gets this share of the time left, capped at the timeout it had before PR 1a.
FLUSH_SHARE = 1 / 3
FLUSH_CAP_S = 5.0


def end_session(hub: MemoryHub, *, deadline: Deadline, consolidation: Literal["full", "lightweight"]) -> dict[str, Any]:
    """Run phases a-d for ``hub``'s session (its ticket is already ENDING); returns the session-end stats."""
    full = consolidation == "full"
    results: dict[str, Any] = {} if full else {"lightweight": True}
    _flush_writers(hub, deadline)
    _settle(hub, results, full=full)
    # A full end replaces sleep's own auto-save, which kept a ``.backup`` of the file as it was BEFORE this session;
    # phase b's save takes that backup, so it is still the previous session's file.
    # (A full end backs up even when the deadline then skips sleep: the backup is the previous session's file.)
    _persist(hub, results, backup=full and bool(getattr(hub.hippocampus.config, "auto_save_after_sleep", False)))
    if full and _consolidate(hub, deadline, results):
        _resave(hub)
    _end_bookkeeping(hub, results)
    hub.shutdown()  # the grounder and completer; the extractor was stopped in phase a (idempotent)
    return results


def _flush_writers(hub: MemoryHub, deadline: Deadline) -> None:
    """Phase a: capture workers, then the ConceptExtractor, then the embedder (producers before their consumers).

    Each step is in its own ``try``: a failure here is logged and never costs the saves that follow."""
    try:
        if hub.hippocampus.stop_capture_workers(deadline.share(FLUSH_SHARE).cap(FLUSH_CAP_S)) is False:
            logger.warning(
                "session end: a capture worker was still running when its share ran out; its late captures "
                "reach the next save"
            )
    except Exception as e:
        logger.warning("session end: stopping the capture workers failed: %s", e)
    extractor = hub._concept_extractor
    if extractor is not None:
        try:
            if not extractor.flush(timeout=deadline.share(FLUSH_SHARE).cap(FLUSH_CAP_S)):
                logger.warning("session end: ConceptExtractor flush timed out; pending extractions reach ATL later")
            extractor.shutdown(timeout=deadline.share(FLUSH_SHARE).cap(FLUSH_CAP_S))
        except Exception as e:
            logger.warning("session end: ConceptExtractor flush failed: %s", e)
    try:
        drain = getattr(getattr(hub.ec, "_neural_embedder", None), "drain", None)
        if callable(drain) and drain(deadline.share(FLUSH_SHARE).cap(FLUSH_CAP_S)) is False:
            logger.warning(
                "session end: the semantic embedder still had work queued; late embeddings reach the next save"
            )
    except Exception as e:
        logger.warning("session end: draining the semantic embedder failed: %s", e)


def _settle(hub: MemoryHub, results: dict[str, Any], *, full: bool) -> None:
    """Phase b's preamble: the spatial bridge's end (full only, as before) and NAc decay, exactly once."""
    if full and hub._spatial_bridge and "spatial" not in hub._disabled_bridges:
        try:
            hub._spatial_bridge.on_session_end()
        except Exception as e:
            logger.warning("Spatial bridge cleanup failed: %s", e)
    try:
        hub.nac.decay_all(factor=0.95)
        results["nac_decayed"] = True
    except Exception as e:
        logger.debug("NAc decay failed: %s", e)


def _save(label: str, save: Callable[[], Any]) -> bool:
    """One store's save: a refusal (#939/#971: the file is kept) at ERROR, any other failure at WARNING."""
    try:
        save()
        return True
    except StoreOverwriteRefused as e:
        logger.error("%s state not saved: %s", label, e)
    except Exception as e:
        logger.warning("Failed to save %s state: %s", label, e)
    return False


def _section_saves(
    hub: MemoryHub, results: dict[str, Any] | None, *, backup: bool = False
) -> list[tuple[str, Callable[[], Any]]]:
    """The four saves of the hippocampus read section, owners first (reverse it for phase d)."""
    saves: list[tuple[str, Callable[[], Any]]] = []
    hippocampus = hub.hippocampus
    if getattr(hippocampus.config, "persistence_path", None):
        saves.append(("Hippocampus", lambda: hippocampus._save_locked(backup=backup)))
    ec_path = getattr(getattr(hub.ec, "config", None), "persistence_path", None)
    if ec_path:
        saves.append(("EC", lambda: hub.ec.save(ec_path)))
    store = hub.ec._embedding_store if hub.ec.semantic_enabled else None
    if store is not None:

        def _embeddings() -> None:
            os.makedirs(os.path.dirname(hub.embedding_persist_path), exist_ok=True)
            store.save(hub.embedding_persist_path)
            if results is not None:
                results["semantic_embeddings_saved"] = len(store)

        saves.append(("semantic embeddings", _embeddings))
    if hub._cross_layer is not None:
        saves.append(("Cross-layer graph", hub._cross_layer.save))
    return saves


def _run_section(hub: MemoryHub, saves: list[tuple[str, Callable[[], Any]]]) -> None:
    """Run ``saves`` inside ONE hippocampus read section; each in its own try, so one refusal skips no other.

    A saved hippocampus records its change mark INSIDE the section: no writer can be active there, so the mark is
    exactly what the file holds (``MemoryHub.hippocampus_unsaved``)."""
    with hub.hippocampus._rwlock.read():
        saved = {label: _save(label, save) for label, save in saves}
        if saved.get("Hippocampus"):
            hub.note_hippocampus_saved(hub.hippocampus.change_mark())


def _nac_save(hub: MemoryHub) -> Callable[[], Any] | None:
    path = getattr(getattr(hub.nac, "config", None), "persistence_path", None)
    return (lambda: hub.nac.save(path)) if hub.nac is not None and path else None


def _scn_save(hub: MemoryHub) -> Callable[[], Any] | None:
    if hub.scn is None:
        return None
    path = getattr(hub.scn, "persistence_path", None) or getattr(hub.scn, "_persistence_path", None)
    return (lambda: hub.scn.save(path)) if path else None


def _persist(hub: MemoryHub, results: dict[str, Any], *, backup: bool) -> None:
    """Phase b: the read section, then NAc (after its EC), ATL, SCN, AG."""
    _run_section(hub, _section_saves(hub, results, backup=backup))
    nac_save = _nac_save(hub)
    if nac_save is not None:
        _save("NAc", nac_save)
    if hub.atl is not None:
        _save("ATL", hub.atl.save)
    scn_save = _scn_save(hub)
    if scn_save is not None:
        _save("SCN", scn_save)
    if hub.angular_gyrus is not None:
        _save("AG", hub.angular_gyrus.save)


def _end_bookkeeping(hub: MemoryHub, results: dict[str, Any]) -> None:
    """Phase e: the plan manager and provenance, after consolidation so its events reach the session summary."""
    if hub._plan_manager is not None:
        try:
            plan_result = hub._plan_manager.on_session_end()
            if plan_result.get("plan_saved"):
                results["plan_saved"] = plan_result["plan_saved"]
        except Exception as e:
            logger.warning("PlanManager session end failed: %s", e)
    collector = getattr(hub, "_collector", None)
    if collector:
        try:
            collector.on_session_end()
        except Exception as e:
            logger.warning("Provenance session end failed: %s", e)


def _consolidate(hub: MemoryHub, deadline: Deadline, results: dict[str, Any]) -> list[str]:
    """Phase c. Returns the steps that ran (any: phase d re-saves)."""
    skipped: list[str] = []
    ran: list[str] = []

    def _fits(name: str, reserve_s: float) -> bool:
        if deadline.remaining() >= reserve_s:
            return True
        skipped.append(name)
        return False

    memories = len(hub.hippocampus)  # outside any section: len takes the read lock
    if _fits("sleep", SLEEP_RESERVE_S + SLEEP_RESERVE_PER_1000_MEMORIES_S * memories / 1000):
        ran.append("sleep")
        try:
            results.update(hub.hippocampus.sleep(auto_save=False))
        except Exception as e:
            logger.error("Sleep consolidation failed: %s", e)
            results["sleep_error"] = str(e)
    if hub._promoter is not None and _fits("promotion", STEP_RESERVE_S):
        ran.append("promotion")
        try:
            results["concepts_promoted"] = len(hub._promoter.scan_for_promotions())
        except Exception as e:
            logger.warning("Semantic promotion failed: %s", e)
    if hub.atl is not None and _fits("atl", STEP_RESERVE_S):
        ran.append("atl")
        try:
            atl_results = hub.atl.consolidate()
            results["atl_removed"] = atl_results.get("removed", 0)
            results["atl_compressed"] = atl_results.get("compressed", 0)
        except Exception as e:
            logger.warning("ATL consolidation failed: %s", e)
    if hub.angular_gyrus is not None and _fits("angular_gyrus", STEP_RESERVE_S):
        ran.append("angular_gyrus")
        try:
            results.update({f"ag_{k}": v for k, v in hub.angular_gyrus.consolidate().items()})
        except Exception as e:
            logger.warning("AG consolidation failed: %s", e)
    if skipped:
        results["consolidation_skipped"] = skipped
        logger.warning("session end: consolidation skipped for want of time: %s (saved first)", ", ".join(skipped))
    return ran


def _resave(hub: MemoryHub) -> None:
    """Phase d: dependants before owners, so a crash part-way never leaves a dependant ahead of its owner."""
    scn_save = _scn_save(hub)
    if scn_save is not None:
        _save("SCN", scn_save)
    nac_save = _nac_save(hub)
    if nac_save is not None:
        _save("NAc", nac_save)
    if hub.atl is not None:
        _save("ATL", hub.atl.save)
    _run_section(hub, list(reversed(_section_saves(hub, None))))
    if hub.angular_gyrus is not None:
        _save("AG", hub.angular_gyrus.save)

"""Hippocampus persistence mixin — save, load, backup, and recovery."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    pass

from maxim.agents.bus import DependencyGraph, EdgeType
from maxim.memory.types import (
    CompressedMemory,
    EpisodicMemory,
)
from maxim.utils.atomic_io import atomic_write_json

logger = logging.getLogger(__name__)


class PersistenceMixin:
    """Persistence methods for Hippocampus.

    Provides save(), load(), load_with_recovery(), save_with_backup(),
    and graph restoration. Mixed into the Hippocampus class.
    """

    # P3.5 Stage 1 — BioSystemSnapshot Protocol envelope version.
    # Payload-layer legacy version string "3.0" is tombstoned; all future
    # migrations land at the envelope layer. See memory/snapshot.py docstring.
    schema_version: ClassVar[int] = 1

    def dump(self) -> dict[str, Any]:
        """Return hippocampus state as a JSON-serializable dict.

        P3.5 Stage 1 — BioSystemSnapshot Protocol conformance. Acquires
        the same read lock the pre-refactor save() body held.

        Reserves an ``"episodes"`` key for P3a. The key is always present
        (empty list when no episode store is wired) so downstream
        consumers can rely on its existence without conditional access.
        Also dumps ``next_episode_ordinal`` (the monotonic counter that
        generates ``ep_N`` ids) so reload+observe doesn't re-issue a
        duplicate id — Round 2 Exec critical #1 fix.

        P4 Stage 1 — also dumps ``node_modality`` (the per-node
        substrate-modality sidecar). Acquired under the inner
        ``_episode_lock`` because the sidecar is episode-binding state,
        consistent with how ``episode_store.to_dict()`` is called inside
        the same outer ``_rwlock.read()`` block. Lock-acquisition order
        ``_rwlock.read() → _episode_lock`` is verified deadlock-free by
        inspection — no ``_episode_lock`` holder ever acquires
        ``_rwlock``.
        """
        with self._rwlock.read():
            # Serialize memories (handles both EpisodicMemory and CompressedMemory)
            memories_data = [m.to_dict() for m in self._memories.values()]

            # Serialize index (convert sets to lists)
            index_data = {k: list(v) for k, v in self._context_index.items()}

            # Serialize associative graph edges
            graph_data = self._graph.to_dict()

            # P3a Stage 1 reserved slot. EpisodeStore populates this when wired.
            episodes_data: list[dict[str, Any]] = []
            episode_store = getattr(self, "_episode_store", None)
            if episode_store is not None:
                episodes_data = episode_store.to_dict().get("episodes", [])

            next_episode_ordinal = getattr(self, "_next_episode_ordinal", 0)

            # P4 Stage 1 — snapshot the node-modality sidecar under the
            # episode lock. Empty dict is the well-defined no-op state
            # for instances that never received a modality-tagged event.
            episode_lock = getattr(self, "_episode_lock", None)
            node_modality_data: dict[str, str] = {}
            if episode_lock is not None:
                with episode_lock:
                    node_modality_source = getattr(self, "_node_modality", None)
                    if node_modality_source:
                        node_modality_data = dict(node_modality_source)

            return {
                "saved_at": time.time(),
                "memories": memories_data,
                "context_index": index_data,
                "stats": dict(self._stats),
                "compressed_count": self._compressed_count,
                "associative_graph": graph_data,
                "episodes": episodes_data,
                "next_episode_ordinal": next_episode_ordinal,
                "node_modality": node_modality_data,
            }

    def load_state(self, state: dict[str, Any]) -> None:
        """Mutate self in place from a state dict.

        P3.5 Stage 1 — BioSystemSnapshot Protocol conformance. Matches
        the pre-refactor load() contract: parses OUTSIDE the write lock,
        then atomically swaps inside it so a from_dict() failure leaves
        existing data untouched.

        **Round 2 fold**: the legacy payload version check is REMOVED
        (Round 2 Exec critical #2). Prior to Round 2 this method
        rejected any payload whose ``version`` was not in
        ``{"1.0","2.0","3.0"}``, which contradicted the tombstoned
        envelope-authoritative versioning rule in ``snapshot.py``. The
        envelope layer is now the only authoritative version.

        **Round 2 fold**: binding graph is REBUILT from loaded episodes
        via ``apply_hebbian_on_close`` (Round 2 Arch important #4).
        Episodes round-trip through the ``episodes`` key; the binding
        graph is derived, not persisted. Callers restoring a populated
        snapshot get a consistent ``retrieve_on_cue`` result without a
        separate binding-graph persistence pass.
        """
        # Parse OUTSIDE write lock — if from_dict() raises, existing data is preserved
        temp_memories: dict[str, EpisodicMemory | CompressedMemory] = {}
        temp_compressed_count = 0
        for m_data in state.get("memories", []):
            if m_data.get("_compressed", False):
                memory = CompressedMemory.from_dict(m_data)
                temp_memories[memory.id] = memory
                temp_compressed_count += 1
            else:
                memory = EpisodicMemory.from_dict(m_data)
                temp_memories[memory.id] = memory

        # P4 Stage 1 — parse the node-modality sidecar outside any lock,
        # validating the modality literal so a malformed snapshot raises
        # BEFORE we mutate anything in self. Unknown modality values are
        # rejected (silent drop would re-introduce the silent-no-op
        # pattern the typed Literal was added to prevent). Empty / missing
        # key parses as an empty dict — valid for legacy snapshots.
        temp_node_modality: dict[str, str] = {}
        raw_node_modality = state.get("node_modality", {}) or {}
        if not isinstance(raw_node_modality, dict):
            raise ValueError(f"node_modality payload must be a dict, got {type(raw_node_modality).__name__}")
        for nid, mod in raw_node_modality.items():
            if mod not in ("text", "vision"):
                raise ValueError(f"node_modality[{nid!r}] has unknown modality {mod!r}; expected 'text' or 'vision'")
            temp_node_modality[str(nid)] = mod

        temp_context_index: defaultdict[str, set[str]] = defaultdict(
            set,
            {k: set(v) for k, v in state.get("context_index", {}).items()},
        )
        temp_memory_contexts: defaultdict[str, set[str]] = defaultdict(set)
        for index_key, memory_ids in temp_context_index.items():
            for memory_id in memory_ids:
                temp_memory_contexts[memory_id].add(index_key)

        temp_stats = state.get("stats", {})

        if "compressed_count" in state:
            temp_compressed_count = state["compressed_count"]

        temp_graph = DependencyGraph()
        graph_data = state.get("associative_graph")

        # Atomically swap inside the write lock
        with self._rwlock.write():
            self._memories.clear()
            self._memories.update(temp_memories)
            self._compressed_count = temp_compressed_count
            self._context_index = temp_context_index
            self._memory_contexts = temp_memory_contexts
            self._stats = temp_stats
            self._graph = temp_graph
            if graph_data:
                self._restore_graph(graph_data)

            # P3a Stage 1 — restore episodes into the EpisodeStore if wired,
            # rebuild the binding graph from the loaded episodes, and
            # restore the monotonic episode-id counter.
            episode_store = getattr(self, "_episode_store", None)
            if episode_store is not None:
                episode_store.load_from_dict({"episodes": state.get("episodes", [])})

                binding_graph = getattr(self, "_binding_graph", None)
                cfg_episode = getattr(getattr(self, "config", None), "episode", None)
                hebbian_cfg = getattr(cfg_episode, "hebbian", None) if cfg_episode is not None else None
                if binding_graph is not None and hebbian_cfg is not None:
                    # Reset and rebuild — the pre-load binding graph state
                    # is authoritative only against the pre-load episodes.
                    # Clear internal structures in place rather than
                    # calling __init__ on a live instance.
                    with binding_graph._lock:
                        binding_graph._nodes.clear()
                        binding_graph._outgoing.clear()
                        binding_graph._incoming.clear()

                    from maxim.memory.episode import apply_hebbian_on_close

                    for ep in episode_store.all_episodes():
                        apply_hebbian_on_close(
                            binding_graph,
                            ep,
                            hebbian_init=hebbian_cfg.init,
                            hebbian_delta=hebbian_cfg.delta,
                            hebbian_max=hebbian_cfg.max_weight,
                        )

                # Restore the monotonic ordinal. Prefer the dumped value;
                # fall back to deriving from max episode id for corrupt-file
                # recovery (episodes named ep_N).
                dumped_ordinal = state.get("next_episode_ordinal")
                if isinstance(dumped_ordinal, int) and dumped_ordinal >= 0:
                    self._next_episode_ordinal = dumped_ordinal
                else:
                    max_ordinal = 0
                    for ep in episode_store.all_episodes():
                        if ep.id.startswith("ep_"):
                            try:
                                max_ordinal = max(max_ordinal, int(ep.id.split("_", 1)[1]))
                            except ValueError:
                                continue
                    self._next_episode_ordinal = max_ordinal

            # P4 Stage 1 — replace the per-node modality sidecar
            # wholesale (clear-then-load, NOT merge). Required for P3.5
            # atomic-rollback semantics: a failed restore_into rolls
            # back to the pre-mutation dump, and the rollback must
            # scrub stale entries left behind by the failed attempt.
            # If load_state merged instead of replacing, those stale
            # entries would survive rollback and the cross-modal
            # retrieval path would silently return ghost partners that
            # no longer correspond to any episode in the store.
            episode_lock = getattr(self, "_episode_lock", None)
            if episode_lock is not None:
                with episode_lock:
                    sidecar = getattr(self, "_node_modality", None)
                    if sidecar is not None:
                        sidecar.clear()
                        sidecar.update(temp_node_modality)

    def save(self, path: str | None = None) -> None:
        """Save hippocampus to JSON file.

        Args:
            path: File path. Uses config.persistence_path if None.
        """
        path = path or self.config.persistence_path
        if not path:
            raise ValueError("No persistence path specified")

        payload = self.dump()
        atomic_write_json(path, payload)
        logger.info("Saved hippocampus to %s (%d memories)", path, len(payload["memories"]))

    def load(self, path: str | None = None) -> None:
        """Load hippocampus from JSON file.

        Args:
            path: File path. Uses config.persistence_path if None.
        """
        path = path or self.config.persistence_path
        if not path:
            raise ValueError("No persistence path specified")

        if not os.path.exists(path):
            logger.warning("Hippocampus file not found: %s", path)
            return

        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        self.load_state(state)

        edge_count = sum(
            len(self._graph.get_associated(mid)) for mid in self._memories if self._graph.get_node(mid) is not None
        )
        logger.info(
            "Loaded hippocampus from %s (%d memories, %d compressed, %d edges)",
            path,
            len(self._memories),
            self._compressed_count,
            edge_count,
        )

    def load_with_recovery(
        self,
        path: str | None = None,
        on_error: str = "warn_and_continue",
    ) -> tuple[bool, str | None]:
        """Load with automatic recovery on failure.

        Args:
            path: File path. Uses config.persistence_path if None.
            on_error: "warn_and_continue" | "restore_backup" | "raise"

        Returns:
            (success, error_message) - error_message is None on success.
        """
        path = path or self.config.persistence_path
        if not path:
            return False, "No persistence path specified"

        if not os.path.exists(path):
            logger.info("No existing hippocampus file, starting fresh")
            return True, None

        try:
            self.load(path)
            return True, None

        except json.JSONDecodeError as e:
            error_msg = f"Corrupt hippocampus file: {e}"
            logger.warning(error_msg)

            if on_error == "restore_backup":
                backup_path = f"{path}.backup"
                if os.path.exists(backup_path):
                    try:
                        self.load(backup_path)
                        logger.info("Restored from backup: %s", backup_path)
                        return True, "Restored from backup (original corrupt)"
                    except Exception as be:
                        logger.error("Backup also corrupt: %s", be)

            if on_error == "raise":
                raise

            # warn_and_continue: start fresh
            logger.warning("Starting with empty hippocampus due to corrupt file")
            return True, error_msg

        except Exception as e:
            error_msg = f"Failed to load hippocampus: {e}"
            logger.error(error_msg)

            if on_error == "raise":
                raise

            return True, error_msg

    def save_with_backup(self, path: str | None = None) -> None:
        """Save with automatic backup of previous version.

        Args:
            path: File path. Uses config.persistence_path if None.
        """
        import shutil

        path = path or self.config.persistence_path
        if not path:
            raise ValueError("No persistence path specified")

        # Create backup of existing file
        if os.path.exists(path):
            backup_path = f"{path}.backup"
            try:
                shutil.copy2(path, backup_path)
            except Exception as e:
                logger.warning("Failed to create backup: %s", e)

        # Save normally
        self.save(path)

    def _restore_graph(self, graph_data: dict[str, Any]) -> None:
        """Restore the associative graph from serialized data (lock must be held).

        Reconstructs nodes and edges from the graph's to_dict() output.

        Args:
            graph_data: Serialized graph from DependencyGraph.to_dict().
        """
        # Restore nodes - only for memories that still exist
        for node_id in graph_data.get("nodes", []):
            if node_id in self._memories:
                self._graph.add_node(node_id, node_id)

        # Restore edges (only ASSOCIATES edges, skip duplicates from bidirectional)
        seen_pairs: set[tuple[str, str]] = set()
        for edge_data in graph_data.get("edges", []):
            source = edge_data.get("source", "")
            target = edge_data.get("target", "")
            weight = edge_data.get("weight", 1.0)
            edge_type_name = edge_data.get("type", "ASSOCIATES")

            # Only restore association edges
            if edge_type_name != "ASSOCIATES":
                continue

            # Skip if either memory no longer exists
            if source not in self._memories or target not in self._memories:
                continue

            # Avoid duplicating bidirectional edges
            pair = (min(source, target), max(source, target))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            # Ensure nodes exist
            if self._graph.get_node(source) is None:
                self._graph.add_node(source, source)
            if self._graph.get_node(target) is None:
                self._graph.add_node(target, target)

            self._graph.add_bidirectional(source, target, EdgeType.ASSOCIATES, weight)

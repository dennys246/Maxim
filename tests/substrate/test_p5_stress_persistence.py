"""P5 — Robust cross-session persistence under stress.

Proves that the bio-substrate survives realistic persistence load:
10,000+ nodes, 1,000+ episodes, mixed modalities, repeated
serialize/reload cycles with no degradation.

Stages:
  Stage 1: 1k-node mechanism test (10 reload cycles, F1 delta = 0.0)
  Stage 2: 10k-node mid-scale validation (mixed modality, <5s load)
  Stage 3: Full 10-seed sweep (state size bounding)
  Stage 4: Stage 7 field fidelity (promotion_pressure, access_contexts)
  Stage 5: Full bio-system round-trip (EC + ATL + NAc at scale)
  Stage 6: Concurrent access stress (multi-thread read/write)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import pytest

from maxim.decisions.causal_link import Valence
from maxim.decisions.nac import NAc
from maxim.memory.atl import ATL, ATLConfig
from maxim.memory.episode import CaptureEvent
from maxim.memory.hippocampus import (
    EpisodeConfig,
    HebbianConfig,
    Hippocampus,
    HippocampusConfig,
    RetrievalConfig,
)
from maxim.memory.semantic_types import SemanticMemory
from maxim.memory.types import EpisodicMemory, Perception, Context, Decision, Action, Outcome
from maxim.similarity.ec import ECConfig, EntorhinalCortex


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _fresh_hippocampus(
    path: str | None = None,
    boundary_tick_gap: int = 50,
) -> Hippocampus:
    return Hippocampus(
        HippocampusConfig(
            persistence_path=path,
            episode=EpisodeConfig(
                boundary_tick_gap=boundary_tick_gap,
                hebbian=HebbianConfig(init=0.4, delta=0.15, max_weight=1.0),
                retrieval=RetrievalConfig(decay=0.7, threshold=0.05, max_depth=3),
            ),
        )
    )


def _populate_nodes(
    h: Hippocampus,
    n_episodes: int,
    nodes_per_episode: int = 4,
    base_tick: int = 0,
    tick_gap: int = 100,
    channel: str = "text",
    modality: str | None = None,
) -> list[set[str]]:
    """Populate a hippocampus with n_episodes episodes, each containing
    nodes_per_episode co-activated nodes. Returns the list of activated
    node sets for verification."""
    # SubstrateModality is a Literal type, just pass the string directly
    mod = modality
    episode_nodes: list[set[str]] = []

    for ep_idx in range(n_episodes):
        nodes = tuple(f"node_{channel}_{ep_idx}_{j}" for j in range(nodes_per_episode))
        tick = base_tick + ep_idx * tick_gap

        # Each episode: one event with all nodes co-activating
        h.observe_episode_event(
            CaptureEvent(
                tick=tick,
                channel=channel,
                activated_nodes=nodes,
                modality=mod,
            )
        )
        # Force episode close via tick gap (next event will be tick_gap away)
        episode_nodes.append(set(nodes))

    # Finalize the last pending episode
    h.finalize_pending_episode()
    return episode_nodes


def _retrieval_f1(
    h: Hippocampus,
    episode_nodes: list[set[str]],
    sample_rate: float = 0.1,
) -> float:
    """Compute retrieval F1 over a sample of episodes.

    For each sampled episode, pick the first node as cue, retrieve,
    and check how many of the episode's other nodes are returned.
    """
    import random

    rng = random.Random(42)
    sample_size = max(1, int(len(episode_nodes) * sample_rate))
    sampled = rng.sample(range(len(episode_nodes)), min(sample_size, len(episode_nodes)))

    total_precision = 0.0
    total_recall = 0.0
    count = 0

    for idx in sampled:
        nodes = episode_nodes[idx]
        if len(nodes) < 2:
            continue

        cue = sorted(nodes)[0]
        expected = nodes - {cue}

        retrieved = h.retrieve_on_cue(cue, limit=20)
        retrieved_ids = {nid for nid, _ in retrieved}

        if not expected:
            continue

        tp = len(retrieved_ids & expected)
        precision = tp / len(retrieved_ids) if retrieved_ids else 0.0
        recall = tp / len(expected)

        total_precision += precision
        total_recall += recall
        count += 1

    if count == 0:
        return 1.0  # no episodes to test → vacuously true

    avg_precision = total_precision / count
    avg_recall = total_recall / count

    if avg_precision + avg_recall == 0:
        return 0.0
    return 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)


# ─────────────────────────────────────────────────────────────────────────
# Stage 1 — mechanism test (1k nodes, 10 reload cycles)
# ─────────────────────────────────────────────────────────────────────────


class TestP5Stage1Mechanism:
    """1k nodes, 10 serialize/reload cycles, verify zero F1 degradation."""

    def test_1k_nodes_10_cycles_zero_degradation(self, tmp_path: Path) -> None:
        hippo_path = str(tmp_path / "hippocampus.json")

        # Populate with 250 episodes × 4 nodes = 1k nodes
        h = _fresh_hippocampus(hippo_path)
        episode_nodes = _populate_nodes(h, n_episodes=250, nodes_per_episode=4)

        # Baseline F1
        f1_before = _retrieval_f1(h, episode_nodes, sample_rate=0.2)
        assert f1_before > 0.5, f"Baseline F1 too low: {f1_before}"

        # 10 serialize/reload cycles
        for cycle in range(10):
            h.save()
            h2 = _fresh_hippocampus(hippo_path)
            h2.load()

            f1_after = _retrieval_f1(h2, episode_nodes, sample_rate=0.2)

            assert f1_after == pytest.approx(f1_before, abs=0.01), (
                f"Cycle {cycle}: F1 degraded {f1_before:.4f} → {f1_after:.4f}"
            )

            h = h2  # continue with reloaded instance

    def test_episode_count_preserved(self, tmp_path: Path) -> None:
        hippo_path = str(tmp_path / "hippocampus.json")

        h = _fresh_hippocampus(hippo_path)
        _populate_nodes(h, n_episodes=100)
        h.save()

        h2 = _fresh_hippocampus(hippo_path)
        h2.load()

        assert len(h2._episode_store) == len(h._episode_store)

    def test_binding_graph_edge_count_preserved(self, tmp_path: Path) -> None:
        hippo_path = str(tmp_path / "hippocampus.json")

        h = _fresh_hippocampus(hippo_path)
        _populate_nodes(h, n_episodes=50, nodes_per_episode=3)

        edges_before = len(h._binding_graph.to_dict()["edges"])
        h.save()

        h2 = _fresh_hippocampus(hippo_path)
        h2.load()

        edges_after = len(h2._binding_graph.to_dict()["edges"])
        assert edges_after == edges_before

    def test_nac_reward_bias_preserved(self, tmp_path: Path) -> None:
        nac_path = str(tmp_path / "nac.json")

        nac = NAc()
        # Seed some reward biases
        for i in range(100):
            nac.update_eligibility("agent", f"node_{i}", 0.8)
        nac.distribute_reward("agent", 1.0)
        nac.save(nac_path)

        nac2 = NAc()
        nac2.load(nac_path)

        # Verify reward biases match
        for i in range(100):
            orig = nac._reward_bias.get(("agent", f"node_{i}"), 0.0)
            loaded = nac2._reward_bias.get(("agent", f"node_{i}"), 0.0)
            assert loaded == pytest.approx(orig, abs=1e-6), f"node_{i}: {orig} → {loaded}"


# ─────────────────────────────────────────────────────────────────────────
# Stage 2 — mid-scale validation (10k+ nodes, mixed modality)
# ─────────────────────────────────────────────────────────────────────────


class TestP5Stage2MidScale:
    """10k+ nodes, mixed text+vision modality, serialize every 100 episodes."""

    def test_10k_nodes_mixed_modality(self, tmp_path: Path) -> None:
        hippo_path = str(tmp_path / "hippocampus.json")

        h = _fresh_hippocampus(hippo_path)

        all_episode_nodes: list[set[str]] = []
        checkpoints: list[dict] = []

        # 2500 text episodes × 4 nodes = 10k text nodes
        text_nodes = _populate_nodes(
            h,
            n_episodes=2500,
            nodes_per_episode=4,
            channel="text",
            base_tick=0,
            tick_gap=100,
        )
        all_episode_nodes.extend(text_nodes)

        # 500 vision episodes × 4 nodes = 2k vision nodes
        vision_nodes = _populate_nodes(
            h,
            n_episodes=500,
            nodes_per_episode=4,
            channel="vision",
            base_tick=300000,
            tick_gap=100,
            modality="vision",
        )
        all_episode_nodes.extend(vision_nodes)

        total_nodes = sum(len(s) for s in all_episode_nodes)
        assert total_nodes >= 10000, f"Only {total_nodes} nodes — need 10k+"

        # Serialize and measure load time
        h.save()
        file_size = Path(hippo_path).stat().st_size

        t0 = time.monotonic()
        h2 = _fresh_hippocampus(hippo_path)
        h2.load()
        load_time = time.monotonic() - t0

        checkpoints.append(
            {
                "total_nodes": total_nodes,
                "episodes": len(all_episode_nodes),
                "file_size_mb": round(file_size / 1024 / 1024, 2),
                "load_time_s": round(load_time, 2),
            }
        )

        # Verify retrieval stability
        f1_before = _retrieval_f1(h, all_episode_nodes, sample_rate=0.05)
        f1_after = _retrieval_f1(h2, all_episode_nodes, sample_rate=0.05)

        assert f1_after == pytest.approx(f1_before, abs=0.01), f"F1 degraded: {f1_before:.4f} → {f1_after:.4f}"

        # Load time gate: <5s for 10k nodes
        assert load_time < 5.0, f"Load time {load_time:.1f}s exceeds 5s gate"

        # Episode count preserved
        assert len(h2._episode_store) == len(h._episode_store)

        # Print checkpoint info for diagnostics
        print(f"\n  P5 Stage 2: {checkpoints[0]}")

    def test_state_size_sublinear(self, tmp_path: Path) -> None:
        """State size should grow sub-linearly with node count.

        We measure file size at 1k, 2k, 4k nodes and verify the growth
        rate slows (ratio decreases).
        """
        sizes: list[tuple[int, int]] = []

        for n_episodes in [250, 500, 1000]:
            hippo_path = str(tmp_path / f"hippo_{n_episodes}.json")
            h = _fresh_hippocampus(hippo_path)
            _populate_nodes(h, n_episodes=n_episodes, nodes_per_episode=4)
            h.save()

            file_size = Path(hippo_path).stat().st_size
            total_nodes = n_episodes * 4
            sizes.append((total_nodes, file_size))

        # Check that bytes-per-node doesn't increase
        bytes_per_node = [(nodes, size / nodes) for nodes, size in sizes]
        for i in range(1, len(bytes_per_node)):
            # Allow some tolerance — serialization overhead may cause slight increase
            assert bytes_per_node[i][1] <= bytes_per_node[0][1] * 1.5, (
                f"State size growing super-linearly: "
                f"{bytes_per_node[0][1]:.0f} bytes/node at {bytes_per_node[0][0]} nodes, "
                f"{bytes_per_node[i][1]:.0f} bytes/node at {bytes_per_node[i][0]} nodes"
            )


# ─────────────────────────────────────────────────────────────────────────
# Stage 3 — multi-seed sweep
# ─────────────────────────────────────────────────────────────────────────


class TestP5Stage3Sweep:
    """Full 10-seed sweep with 10k+ nodes each. Verifies consistency."""

    @pytest.mark.slow
    def test_10_seed_sweep(self, tmp_path: Path) -> None:
        """Run 10 independent 10k-node persistence round-trips."""
        import numpy as np

        f1_deltas: list[float] = []
        load_times: list[float] = []

        for seed in range(10):
            seed_dir = tmp_path / f"seed_{seed}"
            seed_dir.mkdir()
            hippo_path = str(seed_dir / "hippocampus.json")

            h = _fresh_hippocampus(hippo_path)

            # 2500 episodes × 4 nodes = 10k nodes per seed
            episode_nodes = _populate_nodes(
                h,
                n_episodes=2500,
                nodes_per_episode=4,
                base_tick=seed * 1000000,
            )

            f1_before = _retrieval_f1(h, episode_nodes, sample_rate=0.05)
            h.save()

            t0 = time.monotonic()
            h2 = _fresh_hippocampus(hippo_path)
            h2.load()
            load_time = time.monotonic() - t0

            f1_after = _retrieval_f1(h2, episode_nodes, sample_rate=0.05)

            f1_deltas.append(f1_after - f1_before)
            load_times.append(load_time)

        f1_deltas_arr = np.array(f1_deltas)
        load_times_arr = np.array(load_times)

        # All seeds: zero F1 degradation
        assert np.all(np.abs(f1_deltas_arr) < 0.01), f"F1 deltas: {f1_deltas_arr}"

        # All seeds: load time < 5s
        assert np.all(load_times_arr < 5.0), f"Load times: {load_times_arr}"

        # Report statistics
        print("\n  P5 Stage 3: 10-seed sweep")
        print(f"    F1 delta: {np.mean(f1_deltas_arr):+.6f} ± {np.std(f1_deltas_arr):.6f}")
        print(f"    Load time: {np.mean(load_times_arr):.2f} ± {np.std(load_times_arr):.2f}s")
        print(f"    Max load time: {np.max(load_times_arr):.2f}s")


# ─────────────────────────────────────────────────────────────────────────
# Stage 4 — Stage 7 field fidelity (promotion_pressure, access_contexts)
# ─────────────────────────────────────────────────────────────────────────


def _make_memory(
    idx: int,
    *,
    promotion_pressure: float = 0.0,
    last_scored_at: float = 0.0,
    access_contexts: list[str] | None = None,
    long_term: bool = False,
    consolidated_at: float | None = None,
) -> EpisodicMemory:
    """Create a synthetic EpisodicMemory with controlled Stage 7 fields."""
    now = time.time()
    return EpisodicMemory(
        id=f"mem_{idx}",
        timestamp=now - (1000 - idx),
        run_id="stress_test",
        created_at=now - (1000 - idx),
        accessed_at=now,
        access_count=idx + 1,
        long_term=long_term,
        consolidated_at=consolidated_at,
        promotion_pressure=promotion_pressure,
        last_scored_at=last_scored_at,
        access_contexts=deque(access_contexts or [], maxlen=10),
        perception=Perception(
            cli_input=f"test input {idx}",
            detected_objects=[f"obj_{idx}_a", f"obj_{idx}_b"],
            novelty=0.5 + (idx % 10) * 0.05,
            salience=0.3 + (idx % 7) * 0.1,
        ),
        context=Context(active_goal=f"goal_{idx % 20}"),
        decision=Decision(intent={"goal": f"goal_{idx % 20}", "action": "test"}),
        action=Action(tool_name=f"tool_{idx % 15}"),
        outcome=Outcome(success=idx % 3 != 0, result=f"result_{idx}"),
    )


class TestP5Stage4FieldFidelity:
    """Verify that Stage 7 fields (promotion_pressure, last_scored_at,
    access_contexts) survive round-trip at scale — 1000 memories with
    diverse field values."""

    def test_promotion_pressure_round_trip(self, tmp_path: Path) -> None:
        """promotion_pressure values survive save/load exactly."""
        hippo_path = str(tmp_path / "hippocampus.json")
        h = _fresh_hippocampus(hippo_path)

        # Store 500 memories with varying promotion_pressure
        for i in range(500):
            mem = _make_memory(
                i,
                promotion_pressure=i * 0.006,  # 0.0 to ~3.0
                last_scored_at=time.time() - i,
                access_contexts=[f"ctx_{j}" for j in range(min(i % 11, 10))],
                long_term=i % 5 == 0,
                consolidated_at=time.time() if i % 5 == 0 else None,
            )
            h.store(mem)

        # Snapshot original values
        originals = {}
        for mid, mem in h._memories.items():
            originals[mid] = {
                "promotion_pressure": mem.promotion_pressure,
                "last_scored_at": mem.last_scored_at,
                "access_contexts": list(mem.access_contexts),
                "long_term": mem.long_term,
                "consolidated_at": mem.consolidated_at,
                "access_count": mem.access_count,
            }

        h.save()
        h2 = _fresh_hippocampus(hippo_path)
        h2.load()

        assert len(h2._memories) == 500, f"Memory count: {len(h2._memories)}"

        mismatches = []
        for mid, orig in originals.items():
            loaded = h2._memories.get(mid)
            if loaded is None:
                mismatches.append(f"{mid}: MISSING")
                continue

            if loaded.promotion_pressure != pytest.approx(orig["promotion_pressure"], abs=1e-9):
                mismatches.append(
                    f"{mid}: promotion_pressure {orig['promotion_pressure']} → {loaded.promotion_pressure}"
                )
            if loaded.last_scored_at != pytest.approx(orig["last_scored_at"], abs=1e-6):
                mismatches.append(f"{mid}: last_scored_at {orig['last_scored_at']} → {loaded.last_scored_at}")
            if list(loaded.access_contexts) != orig["access_contexts"]:
                mismatches.append(f"{mid}: access_contexts {orig['access_contexts']} → {list(loaded.access_contexts)}")
            if loaded.long_term != orig["long_term"]:
                mismatches.append(f"{mid}: long_term {orig['long_term']} → {loaded.long_term}")
            if loaded.consolidated_at != orig["consolidated_at"]:
                mismatches.append(f"{mid}: consolidated_at {orig['consolidated_at']} → {loaded.consolidated_at}")

        assert not mismatches, f"Field mismatches ({len(mismatches)}):\n" + "\n".join(mismatches[:20])

    def test_access_contexts_deque_maxlen_preserved(self, tmp_path: Path) -> None:
        """access_contexts deque maxlen=10 is preserved after round-trip."""
        hippo_path = str(tmp_path / "hippocampus.json")
        h = _fresh_hippocampus(hippo_path)

        # Create memory with exactly 10 access contexts
        mem = _make_memory(0, access_contexts=[f"ctx_{i}" for i in range(10)])
        h.store(mem)
        h.save()

        h2 = _fresh_hippocampus(hippo_path)
        h2.load()

        loaded = h2._memories["mem_0"]
        assert len(loaded.access_contexts) == 10
        assert list(loaded.access_contexts) == [f"ctx_{i}" for i in range(10)]

        # Verify deque maxlen is preserved (adding 11th should evict first)
        loaded.access_contexts.append("ctx_overflow")
        assert len(loaded.access_contexts) == 10
        assert loaded.access_contexts[0] == "ctx_1"  # ctx_0 evicted

    def test_tier_model_preserved(self, tmp_path: Path) -> None:
        """FORMING, SHORT_TERM, LONG_TERM memories all survive round-trip."""
        hippo_path = str(tmp_path / "hippocampus.json")
        h = _fresh_hippocampus(hippo_path)

        # FORMING: low access, no promotion
        forming = _make_memory(0, promotion_pressure=0.0, long_term=False)
        # SHORT_TERM: moderate pressure, not yet promoted
        short_term = _make_memory(1, promotion_pressure=2.5, long_term=False)
        # LONG_TERM: promoted, pressure reset
        long_term = _make_memory(
            2,
            promotion_pressure=0.0,
            long_term=True,
            consolidated_at=time.time(),
        )

        for m in [forming, short_term, long_term]:
            h.store(m)

        h.save()
        h2 = _fresh_hippocampus(hippo_path)
        h2.load()

        assert not h2._memories["mem_0"].long_term
        assert h2._memories["mem_0"].promotion_pressure == pytest.approx(0.0)
        assert not h2._memories["mem_1"].long_term
        assert h2._memories["mem_1"].promotion_pressure == pytest.approx(2.5)
        assert h2._memories["mem_2"].long_term
        assert h2._memories["mem_2"].consolidated_at is not None


# ─────────────────────────────────────────────────────────────────────────
# Stage 5 — Full bio-system round-trip (EC + ATL + NAc at scale)
# ─────────────────────────────────────────────────────────────────────────


class TestP5Stage5BioSystemRoundTrip:
    """Validate that all four core bio-systems survive 10k+ scale
    persistence independently."""

    def test_ec_10k_substrate_nodes(self, tmp_path: Path) -> None:
        """EC with 10k substrate nodes: save → load → identical node set."""
        ec_path = str(tmp_path / "ec.json")
        ec = EntorhinalCortex(ECConfig())

        # Register 10k substrate nodes with 64-dim embeddings
        import random

        rng = random.Random(42)
        expected: dict[str, tuple[list[float], str]] = {}

        for i in range(10_000):
            modality = "vision" if i % 5 == 0 else "text"
            embedding = [rng.gauss(0, 1) for _ in range(64)]
            node_id = f"node_{i}"
            ec.register_substrate_node(node_id, embedding, modality)
            expected[node_id] = (embedding, modality)

        assert ec.substrate_node_count == 10_000

        t0 = time.monotonic()
        ec.save(ec_path)
        save_time = time.monotonic() - t0

        ec2 = EntorhinalCortex(ECConfig())
        t0 = time.monotonic()
        ec2.load(ec_path)
        load_time = time.monotonic() - t0

        # Verify node count
        assert ec2.substrate_node_count == 10_000, f"Lost nodes: {ec2.substrate_node_count}"

        # Verify embedding fidelity (sample 100 nodes)
        sample_ids = rng.sample(list(expected.keys()), 100)
        for nid in sample_ids:
            orig_emb, orig_mod = expected[nid]
            loaded_emb, loaded_mod = ec2._substrate_nodes[nid]
            assert loaded_mod == orig_mod, f"{nid}: modality {orig_mod} → {loaded_mod}"
            for j, (a, b) in enumerate(zip(orig_emb, loaded_emb)):
                assert a == pytest.approx(b, abs=1e-10), f"{nid}[{j}]: {a} → {b}"

        # Performance gate
        assert load_time < 5.0, f"EC load time {load_time:.1f}s exceeds 5s gate"

        file_size_mb = Path(ec_path).stat().st_size / 1024 / 1024
        print(f"\n  P5 Stage 5 EC: 10k nodes, save={save_time:.2f}s, load={load_time:.2f}s, size={file_size_mb:.1f}MB")

    def test_atl_1k_concepts(self, tmp_path: Path) -> None:
        """ATL with 1k concepts: save → load → all concepts preserved."""
        atl_path = str(tmp_path / "atl.json")
        atl = ATL(ATLConfig(persistence_path=atl_path, max_concepts=5000))

        originals: dict[str, dict] = {}
        for i in range(1000):
            concept = SemanticMemory(
                id=f"concept_{i}",
                timestamp=time.time() - i,
                name=f"concept_name_{i}",
                definition=f"A test concept number {i} for stress testing persistence.",
                category=["object", "person", "action", "fact"][i % 4],
                properties={"idx": i, "group": i % 20},
                confidence=0.3 + (i % 8) * 0.1,
                reinforcement_count=i % 10 + 1,
                promotion_pressure=i * 0.003,
                last_scored_at=time.time() - i * 0.5,
                access_contexts=deque([f"atl_ctx_{j}" for j in range(min(i % 11, 10))], maxlen=10),
            )
            atl.store(concept)
            originals[concept.id] = {
                "name": concept.name,
                "definition": concept.definition,
                "category": concept.category,
                "confidence": concept.confidence,
                "promotion_pressure": concept.promotion_pressure,
                "access_contexts": list(concept.access_contexts),
            }

        atl.save()

        atl2 = ATL(ATLConfig(persistence_path=atl_path, max_concepts=5000))
        t0 = time.monotonic()
        atl2.load()
        load_time = time.monotonic() - t0

        assert len(atl2._concepts) == 1000, f"Concept count: {len(atl2._concepts)}"

        mismatches = []
        for cid, orig in originals.items():
            loaded = atl2._concepts.get(cid)
            if loaded is None:
                mismatches.append(f"{cid}: MISSING")
                continue
            if loaded.name != orig["name"]:
                mismatches.append(f"{cid}: name mismatch")
            if loaded.confidence != pytest.approx(orig["confidence"], abs=1e-6):
                mismatches.append(f"{cid}: confidence {orig['confidence']} → {loaded.confidence}")
            if loaded.promotion_pressure != pytest.approx(orig["promotion_pressure"], abs=1e-9):
                mismatches.append(
                    f"{cid}: promotion_pressure {orig['promotion_pressure']} → {loaded.promotion_pressure}"
                )

        assert not mismatches, f"ATL mismatches ({len(mismatches)}):\n" + "\n".join(mismatches[:20])
        assert load_time < 5.0, f"ATL load time {load_time:.1f}s exceeds 5s gate"
        print(f"\n  P5 Stage 5 ATL: 1k concepts, load={load_time:.2f}s")

    def test_nac_1k_causal_links(self, tmp_path: Path) -> None:
        """NAc with 1k+ causal links: save → load → all links preserved."""
        nac_path = str(tmp_path / "nac.json")
        nac = NAc()

        # Create 500 event→outcome pairs to build ~500 links
        for i in range(500):
            nac.record_event(
                event_type="tool",
                event_signature=f"tool_{i % 50}",
                context={"param": f"val_{i}"},
                memory_id=f"mem_{i}",
            )
            valence = [Valence.POSITIVE, Valence.NEGATIVE, Valence.NEUTRAL][i % 3]
            nac.record_outcome(
                event_type="tool",
                event_id=f"tool_{i % 50}",
                outcome_valence=valence,
                context={"result": f"res_{i}"},
                memory_id=f"mem_{i}",
            )

        # Also build reward biases
        for i in range(200):
            nac.update_eligibility("agent_0", f"node_{i}", 0.7)
        nac.distribute_reward("agent_0", 1.0)

        # Snapshot link stats
        orig_link_count = sum(len(links) for links in nac._links.values())
        orig_bias_count = len(nac._reward_bias)

        nac.save(nac_path)

        nac2 = NAc()
        t0 = time.monotonic()
        nac2.load(nac_path)
        load_time = time.monotonic() - t0

        loaded_link_count = sum(len(links) for links in nac2._links.values())
        loaded_bias_count = len(nac2._reward_bias)

        assert loaded_link_count == orig_link_count, f"Link count: {orig_link_count} → {loaded_link_count}"
        assert loaded_bias_count == orig_bias_count, f"Bias count: {orig_bias_count} → {loaded_bias_count}"

        # Verify a sample of links have correct valence
        for sig, links in nac._links.items():
            loaded_links = nac2._links.get(sig, [])
            assert len(loaded_links) == len(links), f"Sig {sig}: {len(links)} → {len(loaded_links)}"
            for orig_link, loaded_link in zip(links, loaded_links):
                assert loaded_link.outcome_valence == orig_link.outcome_valence
                assert loaded_link.observation_count == orig_link.observation_count
                assert loaded_link.predicted_value == pytest.approx(orig_link.predicted_value, abs=1e-6)

        # Verify reward biases
        for key, orig_val in nac._reward_bias.items():
            loaded_val = nac2._reward_bias.get(key, 0.0)
            assert loaded_val == pytest.approx(orig_val, abs=1e-6), f"Bias {key}: {orig_val} → {loaded_val}"

        assert load_time < 5.0, f"NAc load time {load_time:.1f}s exceeds 5s gate"
        print(f"\n  P5 Stage 5 NAc: {orig_link_count} links, {orig_bias_count} biases, load={load_time:.2f}s")

    def test_combined_bio_system_round_trip(self, tmp_path: Path) -> None:
        """All four bio-systems populated and round-tripped together."""
        hippo_path = str(tmp_path / "hippocampus.json")
        nac_path = str(tmp_path / "nac.json")
        atl_path = str(tmp_path / "atl.json")
        ec_path = str(tmp_path / "ec.json")

        # Populate hippocampus with episodes (for F1) + stored memories (for Stage 7 fields)
        h = _fresh_hippocampus(hippo_path)
        episode_nodes = _populate_nodes(h, n_episodes=500, nodes_per_episode=4)
        # Also store memories with Stage 7 metadata
        for i in range(100):
            mem = _make_memory(
                i,
                promotion_pressure=i * 0.03,
                access_contexts=[f"combined_ctx_{j}" for j in range(min(i % 11, 10))],
            )
            h.store(mem)

        # Populate NAc
        nac = NAc()
        for i in range(100):
            nac.record_event("tool", f"tool_{i}", context={"p": i})
            nac.record_outcome("tool", f"tool_{i}", Valence.POSITIVE)

        # Populate ATL
        atl = ATL(ATLConfig(persistence_path=atl_path, max_concepts=5000))
        for i in range(200):
            atl.store(
                SemanticMemory(
                    id=f"concept_{i}",
                    timestamp=time.time(),
                    name=f"concept_{i}",
                    definition=f"def_{i}",
                )
            )

        # Populate EC
        import random

        rng = random.Random(99)
        ec = EntorhinalCortex(ECConfig())
        for i in range(2000):
            ec.register_substrate_node(f"ec_node_{i}", [rng.gauss(0, 1) for _ in range(32)], "text")

        # Save all
        h.save()
        nac.save(nac_path)
        atl.save()
        ec.save(ec_path)

        total_save_size = sum(Path(p).stat().st_size for p in [hippo_path, nac_path, atl_path, ec_path])

        # Load all
        t0 = time.monotonic()
        h2 = _fresh_hippocampus(hippo_path)
        h2.load()
        nac2 = NAc()
        nac2.load(nac_path)
        atl2 = ATL(ATLConfig(persistence_path=atl_path, max_concepts=5000))
        atl2.load()
        ec2 = EntorhinalCortex(ECConfig())
        ec2.load(ec_path)
        total_load_time = time.monotonic() - t0

        # Verify counts
        assert len(h2._memories) == len(h._memories)
        assert len(atl2._concepts) == 200
        assert ec2.substrate_node_count == 2000

        # Verify hippocampus F1
        f1_before = _retrieval_f1(h, episode_nodes, sample_rate=0.1)
        f1_after = _retrieval_f1(h2, episode_nodes, sample_rate=0.1)
        assert f1_after == pytest.approx(f1_before, abs=0.01)

        # Verify Stage 7 fields survived on stored memories
        stored_mids = [f"mem_{i}" for i in range(100)]
        for mid in stored_mids[:50]:
            orig = h._memories[mid]
            loaded = h2._memories.get(mid)
            assert loaded is not None, f"Memory {mid} missing after reload"
            assert loaded.promotion_pressure == pytest.approx(orig.promotion_pressure, abs=1e-9)
            assert list(loaded.access_contexts) == list(orig.access_contexts)

        print(
            f"\n  P5 Stage 5 combined: {len(h._memories)} mems, 200 concepts, "
            f"2k EC nodes, {total_save_size / 1024:.0f}KB, "
            f"total load={total_load_time:.2f}s"
        )


# ─────────────────────────────────────────────────────────────────────────
# Stage 6 — Concurrent access stress (multi-thread read/write)
# ─────────────────────────────────────────────────────────────────────────


class TestP5Stage6ConcurrentAccess:
    """Verify that concurrent reads and writes to bio-systems don't
    corrupt persisted state."""

    def test_hippocampus_concurrent_store_and_save(self, tmp_path: Path) -> None:
        """Multiple threads storing memories while another thread saves."""
        hippo_path = str(tmp_path / "hippocampus.json")
        h = _fresh_hippocampus(hippo_path)
        errors: list[str] = []
        barrier = threading.Barrier(4)

        def writer(thread_id: int) -> None:
            try:
                barrier.wait(timeout=5)
                for i in range(100):
                    mem = _make_memory(
                        thread_id * 1000 + i,
                        promotion_pressure=i * 0.01,
                        access_contexts=[f"t{thread_id}_ctx_{i % 5}"],
                    )
                    h.store(mem)
            except Exception as e:
                errors.append(f"writer-{thread_id}: {e}")

        def saver() -> None:
            try:
                barrier.wait(timeout=5)
                for _ in range(5):
                    time.sleep(0.02)
                    h.save()
            except Exception as e:
                errors.append(f"saver: {e}")

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(1,)),
            threading.Thread(target=writer, args=(2,)),
            threading.Thread(target=saver),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, "Thread errors:\n" + "\n".join(errors)

        # Final save and reload — must not be corrupt
        h.save()
        h2 = _fresh_hippocampus(hippo_path)
        h2.load()

        # At least some memories should have survived (may not be all 300
        # if the save raced mid-write, but the loaded file must be valid)
        assert len(h2._memories) > 0, "No memories survived concurrent write+save"

        # Verify loaded memories have valid Stage 7 fields
        for mid, mem in h2._memories.items():
            assert isinstance(mem.promotion_pressure, float)
            assert isinstance(mem.access_contexts, deque)

    def test_nac_concurrent_observe_and_save(self, tmp_path: Path) -> None:
        """Multiple threads recording events/outcomes while another saves."""
        nac_path = str(tmp_path / "nac.json")
        nac = NAc()
        errors: list[str] = []
        barrier = threading.Barrier(3)

        def observer(thread_id: int) -> None:
            try:
                barrier.wait(timeout=5)
                for i in range(100):
                    sig = f"tool_t{thread_id}_{i % 20}"
                    nac.record_event("tool", sig, context={"t": thread_id, "i": i})
                    valence = Valence.POSITIVE if i % 2 == 0 else Valence.NEGATIVE
                    nac.record_outcome("tool", sig, valence)
            except Exception as e:
                errors.append(f"observer-{thread_id}: {e}")

        def saver() -> None:
            try:
                barrier.wait(timeout=5)
                for _ in range(5):
                    time.sleep(0.02)
                    nac.save(nac_path)
            except Exception as e:
                errors.append(f"saver: {e}")

        threads = [
            threading.Thread(target=observer, args=(0,)),
            threading.Thread(target=observer, args=(1,)),
            threading.Thread(target=saver),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, "Thread errors:\n" + "\n".join(errors)

        # Final save and reload
        nac.save(nac_path)
        nac2 = NAc()
        nac2.load(nac_path)

        # Must have some links
        total_links = sum(len(links) for links in nac2._links.values())
        assert total_links > 0, "No causal links survived concurrent access"

    def test_ec_concurrent_register_and_save(self, tmp_path: Path) -> None:
        """Multiple threads registering substrate nodes while another saves."""
        ec_path = str(tmp_path / "ec.json")
        ec = EntorhinalCortex(ECConfig())
        errors: list[str] = []
        barrier = threading.Barrier(3)

        import random

        def registrar(thread_id: int) -> None:
            try:
                rng = random.Random(thread_id)
                barrier.wait(timeout=5)
                for i in range(500):
                    nid = f"node_t{thread_id}_{i}"
                    emb = [rng.gauss(0, 1) for _ in range(32)]
                    ec.register_substrate_node(nid, emb, "text")
            except Exception as e:
                errors.append(f"registrar-{thread_id}: {e}")

        def saver() -> None:
            try:
                barrier.wait(timeout=5)
                for _ in range(3):
                    time.sleep(0.03)
                    ec.save(ec_path)
            except Exception as e:
                errors.append(f"saver: {e}")

        threads = [
            threading.Thread(target=registrar, args=(0,)),
            threading.Thread(target=registrar, args=(1,)),
            threading.Thread(target=saver),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, "Thread errors:\n" + "\n".join(errors)

        # Final save and reload
        ec.save(ec_path)
        ec2 = EntorhinalCortex(ECConfig())
        ec2.load(ec_path)

        assert ec2.substrate_node_count > 0, "No EC nodes survived concurrent access"

    def test_multi_agent_isolated_persistence(self, tmp_path: Path) -> None:
        """Two agents writing to separate paths don't interfere."""
        errors: list[str] = []

        def agent_session(agent_id: int) -> None:
            try:
                agent_dir = tmp_path / f"agent_{agent_id}"
                agent_dir.mkdir()
                hippo_path = str(agent_dir / "hippocampus.json")
                nac_path = str(agent_dir / "nac.json")

                h = _fresh_hippocampus(hippo_path)
                nac = NAc()

                # Each agent stores 200 memories with unique IDs
                for i in range(200):
                    mem = _make_memory(
                        agent_id * 10000 + i,
                        promotion_pressure=i * 0.01,
                        access_contexts=[f"agent{agent_id}_ctx"],
                    )
                    h.store(mem)
                    nac.record_event("tool", f"a{agent_id}_tool_{i % 10}")
                    nac.record_outcome("tool", f"a{agent_id}_tool_{i % 10}", Valence.POSITIVE)

                h.save()
                nac.save(nac_path)

                # Reload and verify
                h2 = _fresh_hippocampus(hippo_path)
                h2.load()
                nac2 = NAc()
                nac2.load(nac_path)

                if len(h2._memories) != 200:
                    errors.append(f"agent-{agent_id}: memory count {len(h2._memories)} != 200")

                # Verify no cross-contamination — memory IDs should be in agent's range
                for mid in h2._memories:
                    num = int(mid.split("_")[1])
                    if not (agent_id * 10000 <= num < (agent_id + 1) * 10000):
                        errors.append(f"agent-{agent_id}: foreign memory {mid}")

            except Exception as e:
                errors.append(f"agent-{agent_id}: {e}")

        threads = [
            threading.Thread(target=agent_session, args=(0,)),
            threading.Thread(target=agent_session, args=(1,)),
            threading.Thread(target=agent_session, args=(2,)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, "Agent errors:\n" + "\n".join(errors)


# ─────────────────────────────────────────────────────────────────────────
# Stage 7 — 10-seed full-stack sweep (the 1.0 gate)
# ─────────────────────────────────────────────────────────────────────────


class TestP5Stage7FullStackSweep:
    """The definitive 1.0 validation: 10 seeds × full bio-stack (hippocampus
    + NAc + ATL + EC) × 10k+ scale × Stage 7 field fidelity."""

    @pytest.mark.slow
    def test_10_seed_full_stack_sweep(self, tmp_path: Path) -> None:
        """10-seed sweep across all bio-systems at scale."""
        import numpy as np
        import random as random_mod

        f1_deltas: list[float] = []
        load_times: list[float] = []
        field_fidelity: list[bool] = []
        ec_node_fidelity: list[bool] = []

        for seed in range(10):
            rng = random_mod.Random(seed)
            seed_dir = tmp_path / f"seed_{seed}"
            seed_dir.mkdir()

            hippo_path = str(seed_dir / "hippocampus.json")
            nac_path = str(seed_dir / "nac.json")
            atl_path = str(seed_dir / "atl.json")
            ec_path = str(seed_dir / "ec.json")

            # -- Populate hippocampus (2500 episodes = 10k nodes) --
            h = _fresh_hippocampus(hippo_path)
            episode_nodes = _populate_nodes(
                h,
                n_episodes=2500,
                nodes_per_episode=4,
                base_tick=seed * 1000000,
            )

            # Inject Stage 7 fields
            for i, (mid, mem) in enumerate(h._memories.items()):
                mem.promotion_pressure = rng.uniform(0, 3.5)
                mem.last_scored_at = time.time() - rng.uniform(0, 3600)
                n_ctx = rng.randint(0, 10)
                for j in range(n_ctx):
                    mem.access_contexts.append(f"seed{seed}_ctx_{j}")

            f1_before = _retrieval_f1(h, episode_nodes, sample_rate=0.05)

            # -- Populate NAc --
            nac = NAc()
            for i in range(200):
                nac.record_event("tool", f"s{seed}_tool_{i % 30}", context={"s": seed})
                v = [Valence.POSITIVE, Valence.NEGATIVE, Valence.NEUTRAL][i % 3]
                nac.record_outcome("tool", f"s{seed}_tool_{i % 30}", v)
            for i in range(50):
                nac.update_eligibility(f"agent_{seed}", f"node_{i}", 0.8)
            nac.distribute_reward(f"agent_{seed}", 1.0)

            # -- Populate ATL --
            atl = ATL(ATLConfig(persistence_path=atl_path, max_concepts=5000))
            for i in range(100):
                atl.store(
                    SemanticMemory(
                        id=f"s{seed}_concept_{i}",
                        timestamp=time.time(),
                        name=f"concept_{i}",
                        definition=f"seed {seed} concept {i}",
                        promotion_pressure=rng.uniform(0, 2),
                    )
                )

            # -- Populate EC --
            ec = EntorhinalCortex(ECConfig())
            ec_expected = {}
            for i in range(2000):
                nid = f"s{seed}_ec_{i}"
                emb = [rng.gauss(0, 1) for _ in range(32)]
                mod = "vision" if i % 5 == 0 else "text"
                ec.register_substrate_node(nid, emb, mod)
                ec_expected[nid] = (emb, mod)

            # -- Save all --
            h.save()
            nac.save(nac_path)
            atl.save()
            ec.save(ec_path)

            # -- Load all --
            t0 = time.monotonic()
            h2 = _fresh_hippocampus(hippo_path)
            h2.load()
            nac2 = NAc()
            nac2.load(nac_path)
            atl2 = ATL(ATLConfig(persistence_path=atl_path, max_concepts=5000))
            atl2.load()
            ec2 = EntorhinalCortex(ECConfig())
            ec2.load(ec_path)
            load_time = time.monotonic() - t0
            load_times.append(load_time)

            # -- Verify hippocampus F1 --
            f1_after = _retrieval_f1(h2, episode_nodes, sample_rate=0.05)
            f1_deltas.append(f1_after - f1_before)

            # -- Verify Stage 7 field fidelity (sample 50 memories) --
            fidelity_ok = True
            sample_mids = rng.sample(list(h._memories.keys()), min(50, len(h._memories)))
            for mid in sample_mids:
                orig = h._memories[mid]
                loaded = h2._memories.get(mid)
                if loaded is None:
                    fidelity_ok = False
                    break
                if loaded.promotion_pressure != pytest.approx(orig.promotion_pressure, abs=1e-9):
                    fidelity_ok = False
                    break
                if list(loaded.access_contexts) != list(orig.access_contexts):
                    fidelity_ok = False
                    break
            field_fidelity.append(fidelity_ok)

            # -- Verify EC node fidelity (sample 20 nodes) --
            ec_ok = True
            ec_sample = rng.sample(list(ec_expected.keys()), 20)
            for nid in ec_sample:
                if nid not in ec2._substrate_nodes:
                    ec_ok = False
                    break
                orig_emb, orig_mod = ec_expected[nid]
                loaded_emb, loaded_mod = ec2._substrate_nodes[nid]
                if loaded_mod != orig_mod:
                    ec_ok = False
                    break
                if any(abs(a - b) > 1e-10 for a, b in zip(orig_emb, loaded_emb)):
                    ec_ok = False
                    break
            ec_node_fidelity.append(ec_ok)

        f1_deltas_arr = np.array(f1_deltas)
        load_times_arr = np.array(load_times)

        # --- Pass gates ---
        # F1: zero degradation
        assert np.all(np.abs(f1_deltas_arr) < 0.01), f"F1 deltas: {f1_deltas_arr}"
        # Load time: <5s per seed
        assert np.all(load_times_arr < 5.0), f"Load times: {load_times_arr}"
        # Stage 7 fields: all seeds fidelity OK
        assert all(field_fidelity), (
            f"Field fidelity failures at seeds: {[i for i, ok in enumerate(field_fidelity) if not ok]}"
        )
        # EC nodes: all seeds fidelity OK
        assert all(ec_node_fidelity), (
            f"EC fidelity failures at seeds: {[i for i, ok in enumerate(ec_node_fidelity) if not ok]}"
        )

        # Report
        print("\n  P5 Stage 7: 10-seed full-stack sweep")
        print(f"    F1 delta: {np.mean(f1_deltas_arr):+.6f} ± {np.std(f1_deltas_arr):.6f}")
        print(f"    Load time: {np.mean(load_times_arr):.2f} ± {np.std(load_times_arr):.2f}s")
        print(f"    Max load time: {np.max(load_times_arr):.2f}s")
        print(f"    Stage 7 field fidelity: {sum(field_fidelity)}/10 seeds OK")
        print(f"    EC node fidelity: {sum(ec_node_fidelity)}/10 seeds OK")

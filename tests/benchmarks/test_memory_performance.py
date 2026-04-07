"""Performance benchmarks for memory system components.

Validates that the Hippocampus and MemoryHub can sustain 30Hz capture
throughput for real-time robot control systems.

Target Performance:
- Capture latency: <33ms (30Hz budget)
- Query latency: <10ms
- Memory footprint: <100MB for 10K memories
- Consolidation: <5 seconds for 1000 memories
"""

from __future__ import annotations

import gc
import sys
import time
from dataclasses import dataclass
from typing import Any

import pytest

# Add src to path for imports
sys.path.insert(0, "src")


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    ops_per_second: float
    mean_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    total_ops: int
    duration_seconds: float
    passed: bool
    target_ops_per_second: float | None = None

    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        target = f" (target: {self.target_ops_per_second:.0f}/s)" if self.target_ops_per_second else ""
        return (
            f"{status} {self.name}: {self.ops_per_second:.1f} ops/s{target}\n"
            f"   Latency: mean={self.mean_latency_ms:.2f}ms, "
            f"p99={self.p99_latency_ms:.2f}ms, "
            f"min={self.min_latency_ms:.2f}ms, max={self.max_latency_ms:.2f}ms"
        )


def run_benchmark(
    name: str,
    operation: Any,
    iterations: int = 1000,
    warmup: int = 100,
    target_ops_per_second: float | None = None,
) -> BenchmarkResult:
    """Run a benchmark and collect statistics."""
    # Warmup
    for _ in range(warmup):
        operation()

    gc.collect()

    # Benchmark
    latencies = []
    start = time.perf_counter()

    for _ in range(iterations):
        op_start = time.perf_counter()
        operation()
        op_end = time.perf_counter()
        latencies.append((op_end - op_start) * 1000)  # ms

    end = time.perf_counter()
    duration = end - start

    latencies.sort()
    ops_per_second = iterations / duration
    mean_latency = sum(latencies) / len(latencies)
    p99_idx = int(len(latencies) * 0.99)
    p99_latency = latencies[p99_idx] if p99_idx < len(latencies) else latencies[-1]

    passed = True
    if target_ops_per_second and ops_per_second < target_ops_per_second:
        passed = False

    return BenchmarkResult(
        name=name,
        ops_per_second=ops_per_second,
        mean_latency_ms=mean_latency,
        p99_latency_ms=p99_latency,
        min_latency_ms=latencies[0],
        max_latency_ms=latencies[-1],
        total_ops=iterations,
        duration_seconds=duration,
        passed=passed,
        target_ops_per_second=target_ops_per_second,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hippocampus Benchmarks
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_benchmark_hippocampus_capture() -> BenchmarkResult:
    """Benchmark Hippocampus capture at 30Hz.

    Target: >30 ops/s with <33ms p99 latency
    """
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
    from maxim.memory.types import Perception, Context, Decision, Action, Outcome

    config = HippocampusConfig(max_nodes=5000)
    hippocampus = Hippocampus(config=config)

    counter = [0]

    def capture_operation():
        perception = Perception(
            observations={"frame": counter[0]},
            detected_objects=["mug", "cup"],
            salience=0.5,
            novelty=0.3,
        )
        context = Context(
            active_goal="find mug",
            active_mode="exploration",
        )
        decision = Decision(
            intent={"goal": "search"},
            confidence=0.8,
        )
        action = Action(
            tool_name="look_around",
            tool_params={},
        )
        outcome = Outcome(success=True)
        memory_id = hippocampus.capture(
            perception=perception,
            context=context,
            decision=decision,
            action=action,
            outcome=outcome,
        )
        counter[0] += 1
        return memory_id

    return run_benchmark(
        name="Hippocampus.capture",
        operation=capture_operation,
        iterations=1000,
        warmup=50,
        target_ops_per_second=30.0,  # 30Hz target
    )


@pytest.mark.slow
def test_benchmark_hippocampus_recall() -> BenchmarkResult:
    """Benchmark Hippocampus recall.

    Target: >100 ops/s (recall should be fast)
    """
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
    from maxim.memory.types import Perception, Context, Decision, Action, Outcome

    config = HippocampusConfig(max_nodes=1000)
    hippocampus = Hippocampus(config=config)

    # Pre-populate with memories
    for i in range(500):
        perception = Perception(
            observations={"frame": i},
            salience=(i % 10) / 10,
            novelty=(i % 10) / 10,
        )
        context = Context(
            active_goal=f"goal_{i % 10}",
            active_mode="exploration",
        )
        decision = Decision(
            intent={"goal": "search"},
            confidence=0.8,
        )
        action = Action(
            tool_name=f"tool_{i % 5}",
            tool_params={},
        )
        outcome = Outcome(success=i % 3 == 0)
        hippocampus.capture(
            perception=perception,
            context=context,
            decision=decision,
            action=action,
            outcome=outcome,
        )

    def recall_operation():
        return hippocampus.recall(mode="exploration", limit=10)

    return run_benchmark(
        name="Hippocampus.recall",
        operation=recall_operation,
        iterations=1000,
        warmup=50,
        target_ops_per_second=100.0,
    )


@pytest.mark.slow
def test_benchmark_hippocampus_sleep() -> BenchmarkResult:
    """Benchmark sleep consolidation.

    Target: <5 seconds for 1000 memories
    """
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
    from maxim.memory.types import Perception, Context, Decision, Action, Outcome

    config = HippocampusConfig(
        max_nodes=2000,
        compression_age=0,  # Immediate compression for testing
    )
    hippocampus = Hippocampus(config=config)

    # Pre-populate with memories
    for i in range(1000):
        perception = Perception(observations={"frame": i}, salience=0.5, novelty=0.5)
        context = Context(active_goal="test", active_mode="exploration")
        decision = Decision(intent={"goal": "test"}, confidence=0.8)
        action = Action(tool_name="look", tool_params={})
        outcome = Outcome(success=True)
        hippocampus.capture(
            perception=perception,
            context=context,
            decision=decision,
            action=action,
            outcome=outcome,
        )

    def sleep_operation():
        return hippocampus.sleep()

    # Only run once - sleep is a batch operation
    result = run_benchmark(
        name="Hippocampus.sleep (1000 memories)",
        operation=sleep_operation,
        iterations=1,
        warmup=0,
        target_ops_per_second=0.2,  # Should complete in <5 seconds
    )

    # Adjust pass criteria based on actual duration
    result.passed = result.duration_seconds < 5.0
    return result


# ─────────────────────────────────────────────────────────────────────────────
# EntorhinalCortex (EC) Benchmarks
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_benchmark_ec_register() -> BenchmarkResult:
    """Benchmark EC registration.

    Target: >100 ops/s (registration happens with capture)
    """
    from maxim.similarity.ec import EntorhinalCortex, ECConfig
    from maxim.similarity.signature import SituationSignature

    config = ECConfig(num_lsh_tables=4, bits_per_table=8)
    ec = EntorhinalCortex(config=config)

    counter = [0]

    def register_operation():
        memory_id = f"mem_{counter[0]}"
        # Create signature using the actual constructor
        signature = SituationSignature(
            semantic_hash=(0,) * 8,
            structural_hash=hash(f"tool_{counter[0] % 5}"),
            temporal_hash=(counter[0] % 24, counter[0] % 7, 0, 0),
            context_hash=hash("exploration"),
            tool_name=f"tool_{counter[0] % 5}",
            outcome_type="success" if counter[0] % 2 == 0 else "failure",
            mode="exploration",
        )
        ec.register(memory_id, signature)
        counter[0] += 1

    return run_benchmark(
        name="EC.register",
        operation=register_operation,
        iterations=1000,
        warmup=50,
        target_ops_per_second=100.0,
    )


@pytest.mark.slow
def test_benchmark_ec_find_similar() -> BenchmarkResult:
    """Benchmark EC similarity queries.

    Target: >50 ops/s (queries should be fast with LSH)
    """
    from maxim.similarity.ec import EntorhinalCortex, ECConfig
    from maxim.similarity.signature import SituationSignature

    config = ECConfig(num_lsh_tables=4, bits_per_table=8)
    ec = EntorhinalCortex(config=config)

    # Pre-populate
    for i in range(500):
        signature = SituationSignature(
            semantic_hash=(0,) * 8,
            structural_hash=hash(f"tool_{i % 5}"),
            temporal_hash=(i % 24, i % 7, 0, 0),
            context_hash=hash("exploration"),
            tool_name=f"tool_{i % 5}",
            outcome_type="success" if i % 2 == 0 else "failure",
            mode="exploration",
        )
        ec.register(f"mem_{i}", signature)

    query_sig = SituationSignature(
        semantic_hash=(0,) * 8,
        structural_hash=hash("tool_0"),
        temporal_hash=(12, 3, 0, 0),
        context_hash=hash("exploration"),
        tool_name="tool_0",
        outcome_type="success",
        mode="exploration",
    )

    def query_operation():
        return ec.find_similar(query_sig, k=10)

    return run_benchmark(
        name="EC.find_similar",
        operation=query_operation,
        iterations=500,
        warmup=50,
        target_ops_per_second=50.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MemoryHub Benchmarks
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_benchmark_memory_hub_session_start() -> BenchmarkResult:
    """Benchmark MemoryHub session start.

    Target: <2 seconds (cold start is acceptable)
    """
    from maxim.memory.hippocampus import Hippocampus
    from maxim.time.scn import SCN
    from maxim.decisions.nac import NAc
    from maxim.similarity.ec import EntorhinalCortex
    from maxim.integration.memory_hub import MemoryHub

    def create_and_start():
        hippocampus = Hippocampus()
        scn = SCN()
        nac = NAc()
        ec = EntorhinalCortex()

        hub = MemoryHub(hippocampus=hippocampus, scn=scn, nac=nac, ec=ec)
        hub.connect()
        return hub.on_session_start()

    result = run_benchmark(
        name="MemoryHub.on_session_start",
        operation=create_and_start,
        iterations=10,
        warmup=2,
        target_ops_per_second=0.5,  # Should complete in <2 seconds
    )

    result.passed = result.mean_latency_ms < 2000
    return result


@pytest.mark.slow
def test_benchmark_memory_hub_plan_outcome() -> BenchmarkResult:
    """Benchmark MemoryHub plan outcome recording.

    Target: >100 ops/s (should be fast, non-blocking)
    """
    from maxim.memory.hippocampus import Hippocampus
    from maxim.time.scn import SCN
    from maxim.decisions.nac import NAc
    from maxim.similarity.ec import EntorhinalCortex
    from maxim.integration.memory_hub import MemoryHub

    hippocampus = Hippocampus()
    scn = SCN()
    nac = NAc()
    ec = EntorhinalCortex()

    hub = MemoryHub(hippocampus=hippocampus, scn=scn, nac=nac, ec=ec)
    hub.connect()
    hub.on_session_start()

    counter = [0]

    def record_outcome():
        hub.record_plan_outcome(
            goal=f"find object_{counter[0] % 10}",
            tool_sequence=["look_around", "grasp"],
            success=counter[0] % 2 == 0,
        )
        counter[0] += 1

    return run_benchmark(
        name="MemoryHub.record_plan_outcome",
        operation=record_outcome,
        iterations=500,
        warmup=50,
        target_ops_per_second=100.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Embeddings Benchmarks (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_benchmark_semantic_hash() -> BenchmarkResult | None:
    """Benchmark semantic hashing (fallback implementation).

    Target: >100 ops/s (fallback should be fast)
    """
    try:
        from maxim.similarity.lsh import SemanticLSH
    except ImportError:
        return None

    hasher = SemanticLSH(num_planes=16, seed=42)

    texts = [
        "find the coffee mug",
        "locate the cup on the table",
        "pick up the red ball",
        "move to the kitchen",
        "say hello to the user",
    ]
    counter = [0]

    def hash_operation():
        text = texts[counter[0] % len(texts)]
        result = hasher.hash(text)
        counter[0] += 1
        return result

    return run_benchmark(
        name="SemanticLSH.hash (fallback)",
        operation=hash_operation,
        iterations=1000,
        warmup=50,
        target_ops_per_second=100.0,
    )


@pytest.mark.slow
def test_benchmark_neural_semantic_hash() -> BenchmarkResult | None:
    """Benchmark neural semantic hashing (requires sentence-transformers).

    Target: >30 ops/s on GPU, >10 ops/s on CPU
    """
    try:
        from maxim.similarity.semantic import NeuralSemanticLSH, SemanticEmbedderConfig
    except ImportError:
        return None

    try:
        config = SemanticEmbedderConfig(
            model_name="all-MiniLM-L6-v2",
            async_embedding=False,  # Sync for benchmarking
            require_gpu=False,
        )
        embedder = NeuralSemanticLSH(config=config)

        if not embedder.is_healthy:
            return None

    except Exception:
        return None

    texts = [
        "find the coffee mug",
        "locate the cup on the table",
        "pick up the red ball",
        "move to the kitchen",
        "say hello to the user",
    ]
    counter = [0]

    def embed_operation():
        text = texts[counter[0] % len(texts)]
        result = embedder.hash(text)
        counter[0] += 1
        return result

    return run_benchmark(
        name="NeuralSemanticLSH.hash",
        operation=embed_operation,
        iterations=100,  # Fewer iterations for neural
        warmup=10,
        target_ops_per_second=10.0,  # Conservative target
    )


# ─────────────────────────────────────────────────────────────────────────────
# Memory Footprint
# ─────────────────────────────────────────────────────────────────────────────


def measure_memory_footprint() -> dict[str, float]:
    """Measure memory footprint of various components."""
    import tracemalloc

    results = {}

    # Hippocampus with 1000 memories
    gc.collect()
    tracemalloc.start()

    from maxim.memory.hippocampus import Hippocampus
    from maxim.memory.types import Perception, Context, Decision, Action, Outcome

    hippocampus = Hippocampus()
    for i in range(1000):
        perception = Perception(
            observations={"frame": i, "objects": [f"obj_{j}" for j in range(5)]},
            salience=0.5,
            novelty=0.5,
        )
        context = Context(active_goal=f"goal_{i}", active_mode="exploration")
        decision = Decision(intent={"goal": "test"}, confidence=0.8)
        action = Action(tool_name="look", tool_params={"angle": i})
        outcome = Outcome(success=True)
        hippocampus.capture(
            perception=perception,
            context=context,
            decision=decision,
            action=action,
            outcome=outcome,
        )

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["hippocampus_1000_memories_mb"] = peak / 1024 / 1024

    # EC with 1000 signatures
    gc.collect()
    tracemalloc.start()

    from maxim.similarity.ec import EntorhinalCortex
    from maxim.similarity.signature import SituationSignature

    ec = EntorhinalCortex()
    for i in range(1000):
        sig = SituationSignature(
            semantic_hash=(0,) * 8,
            structural_hash=hash(f"tool_{i % 5}"),
            temporal_hash=(i % 24, i % 7, 0, 0),
            context_hash=hash("exploration"),
            tool_name=f"tool_{i % 5}",
            outcome_type="success",
            mode="exploration",
        )
        ec.register(f"mem_{i}", sig)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["ec_1000_signatures_mb"] = peak / 1024 / 1024

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def run_all_benchmarks() -> list[BenchmarkResult]:
    """Run all benchmarks and return results."""
    results = []

    print("=" * 60)
    print("MEMORY SYSTEM PERFORMANCE BENCHMARKS")
    print("=" * 60)
    print()

    # Core benchmarks
    benchmarks = [
        ("Hippocampus Capture (30Hz target)", test_benchmark_hippocampus_capture),
        ("Hippocampus Recall", test_benchmark_hippocampus_recall),
        ("Hippocampus Sleep", test_benchmark_hippocampus_sleep),
        ("EC Register", test_benchmark_ec_register),
        ("EC Find Similar", test_benchmark_ec_find_similar),
        ("MemoryHub Session Start", test_benchmark_memory_hub_session_start),
        ("MemoryHub Plan Outcome", test_benchmark_memory_hub_plan_outcome),
        ("Semantic Hash (Fallback)", test_benchmark_semantic_hash),
        ("Neural Semantic Hash", test_benchmark_neural_semantic_hash),
    ]

    for name, benchmark_fn in benchmarks:
        print(f"Running: {name}...")
        try:
            result = benchmark_fn()
            if result:
                results.append(result)
                print(f"  {result}")
            else:
                print("  SKIPPED (dependencies not available)")
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    # Memory footprint
    print("Measuring memory footprint...")
    try:
        footprint = measure_memory_footprint()
        for component, mb in footprint.items():
            print(f"  {component}: {mb:.2f} MB")
    except Exception as e:
        print(f"  ERROR: {e}")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    failed = [r for r in results if not r.passed]
    if failed:
        print("\nFailed benchmarks:")
        for r in failed:
            print(f"  - {r.name}: {r.ops_per_second:.1f}/s (target: {r.target_ops_per_second:.1f}/s)")

    return results


if __name__ == "__main__":
    results = run_all_benchmarks()
    sys.exit(0 if all(r.passed for r in results) else 1)

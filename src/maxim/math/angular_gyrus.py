"""Angular Gyrus (Posterior Parietal Lobe) — Exact math computation + math memory.

Implements both a MemoryLayer (persistent math knowledge: facts, methods,
patterns) and a computation engine (exact arithmetic, statistical analysis).
This dual compute+memory role mirrors the biological angular gyrus, which
both retrieves arithmetic facts and participates in calculation.

All results include dual verbal/code representations for natural language
and programmatic access — the angular gyrus is language-mediated.

Brain mapping:
  The angular gyrus sits at the junction of temporal, parietal, and
  occipital lobes within the posterior parietal lobe.  It is activated
  during arithmetic fact retrieval (e.g., "7 × 8 = 56") and connects
  numerical processing to language representations.

Implements MemoryLayer ABC — the abstract protocol shared with
Hippocampus (episodic) and ATL (semantic) memory layers.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from maxim.agents.bus import DependencyGraph
from maxim.math.ips import IPS
from maxim.memory.layer import MemoryLayer
from maxim.math.math_types import CompressedMathMemory, MathMemory
from maxim.math.types import (
    AnalysisResult,
    ExactResult,
    MathCategory,
)
from maxim.memory.rwlock import RWLock

logger = logging.getLogger(__name__)


@dataclass
class AngularGyrusConfig:
    """Configuration for the Angular Gyrus math memory + computation layer."""

    max_records: int = 5_000
    persistence_path: str | None = None
    indexed_keys: frozenset[str] = frozenset({"name", "category", "domain"})
    retention_threshold: float = 0.15  # Very stable — math facts rarely decay
    compression_threshold: float = 0.3
    max_age_without_access: float = 90 * 86400  # 90 days
    seed_built_ins: bool = True


class AngularGyrus(MemoryLayer):
    """Angular Gyrus (Posterior Parietal Lobe) — Exact math + math memory.

    **MemoryLayer protocol** (store/get/recall/consolidate/save/load/etc.)
    for persistent mathematical knowledge.

    **Computation methods** (compute/analyze/verbalize/codify) for exact
    arithmetic and statistical analysis.

    Thread-safe via RWLock.
    """

    # --- Persistence version ---
    _VERSION = "1.0"

    def __init__(self, config: AngularGyrusConfig | None = None) -> None:
        self.config = config or AngularGyrusConfig()

        # Storage
        self._records: dict[str, MathMemory | CompressedMathMemory] = {}
        self._context_index: dict[str, set[str]] = {}
        self._record_contexts: dict[str, set[str]] = {}

        # Association graph
        self._graph: DependencyGraph[str] = DependencyGraph()

        # Thread safety
        self._rwlock = RWLock()

        # Callbacks
        self._capture_callbacks: list[Callable[[str, MathMemory], None]] = []
        self._deletion_callbacks: list[Callable[[str], None]] = []

        # Stats
        self._total_stores = 0
        self._total_recalls = 0

        # IPS for magnitude categorization in verbalize
        self._ips = IPS()

        # Seed built-ins if configured
        if self.config.seed_built_ins:
            self._seed_built_ins()

    # ------------------------------------------------------------------
    # MemoryLayer protocol
    # ------------------------------------------------------------------

    @property
    def layer_name(self) -> str:
        return "angular_gyrus"

    def store(self, record: MathMemory, **kwargs: Any) -> str:
        """Store a math record. Returns the record ID."""
        with self._rwlock.write():
            if len(self._records) >= self.config.max_records:
                logger.warning("Angular gyrus at capacity (%d records)", self.config.max_records)
                return record.id

            self._records[record.id] = record
            self._index_record(record)

            # Add node to graph
            if not self._graph.get_node(record.id):
                self._graph.add_node(record.id, record.id)

            self._total_stores += 1

        # Fire callbacks outside lock
        for cb in self._capture_callbacks:
            try:
                cb(record.id, record)
            except Exception as e:
                logger.debug("Capture callback error: %s", e)

        return record.id

    def get(self, record_id: str) -> MathMemory | CompressedMathMemory | None:
        """Retrieve a record by ID. Updates access tracking."""
        with self._rwlock.read():
            record = self._records.get(record_id)

        if record is not None:
            record.touch()
            self._total_recalls += 1

        return record

    def recall_by_ids(
        self,
        record_ids: list[str],
    ) -> list[MathMemory | CompressedMathMemory]:
        """Bulk-load math records by ID. Skips missing IDs.

        Single lock acquisition — more efficient than iterating get().
        Does NOT update access tracking.
        """
        with self._rwlock.read():
            results = []
            for rid in record_ids:
                rec = self._records.get(rid)
                if rec is not None:
                    results.append(rec)
            return results

    def remove(self, record_id: str) -> None:
        """Delete a record and clean up indexes/graph."""
        with self._rwlock.write():
            if record_id not in self._records:
                return

            # Remove from context index
            if record_id in self._record_contexts:
                for key in self._record_contexts[record_id]:
                    if key in self._context_index:
                        self._context_index[key].discard(record_id)
                del self._record_contexts[record_id]

            # Remove from graph
            self._graph.remove_node(record_id)

            del self._records[record_id]

        # Fire deletion callbacks outside lock
        for cb in self._deletion_callbacks:
            try:
                cb(record_id)
            except Exception as e:
                logger.debug("Deletion callback error: %s", e)

    def recall(
        self,
        limit: int = 10,
        *,
        name: str | None = None,
        category: str | None = None,
        domain: str | None = None,
        min_confidence: float | None = None,
        include_compressed: bool = True,
    ) -> list[MathMemory | CompressedMathMemory]:
        """Query math records by filters."""
        with self._rwlock.read():
            # Start with candidates from index if possible
            candidate_ids: set[str] | None = None

            if name is not None:
                key = f"name:{name}"
                ids = self._context_index.get(key, set())
                candidate_ids = ids if candidate_ids is None else candidate_ids & ids

            if category is not None:
                key = f"category:{category}"
                ids = self._context_index.get(key, set())
                candidate_ids = ids if candidate_ids is None else candidate_ids & ids

            if domain is not None:
                key = f"domain:{domain}"
                ids = self._context_index.get(key, set())
                candidate_ids = ids if candidate_ids is None else candidate_ids & ids

            # Fall back to all records if no index filters
            if candidate_ids is None:
                candidate_ids = set(self._records.keys())

            results: list[MathMemory | CompressedMathMemory] = []
            for rid in candidate_ids:
                record = self._records.get(rid)
                if record is None:
                    continue
                if not include_compressed and isinstance(record, CompressedMathMemory):
                    continue
                if min_confidence is not None and record.confidence < min_confidence:
                    continue
                results.append(record)

        # Sort by access_count (most accessed first), then confidence
        results.sort(key=lambda r: (r.access_count, r.confidence), reverse=True)

        # Touch recalled records
        for r in results[:limit]:
            r.touch()
            self._total_recalls += 1

        return results[:limit]

    def recall_associated(
        self,
        seed_ids: list[str],
        limit: int = 10,
        decay: float = 0.5,
        max_depth: int = 3,
    ) -> list[tuple[MathMemory | CompressedMathMemory, float]]:
        """Spreading activation from seed records."""
        with self._rwlock.read():
            activations: dict[str, float] = {}
            visited: set[str] = set()

            def _activate(node_id: str, strength: float, depth: int) -> None:
                if depth > max_depth or strength < 0.01 or node_id in visited:
                    return
                visited.add(node_id)

                if node_id not in seed_ids:
                    activations[node_id] = max(activations.get(node_id, 0), strength)

                for edge in self._graph.get_outgoing(node_id):
                    _activate(edge.target, strength * decay * edge.weight, depth + 1)
                for edge in self._graph.get_incoming(node_id):
                    _activate(edge.source, strength * decay * edge.weight, depth + 1)

            for sid in seed_ids:
                if sid in self._records:
                    _activate(sid, 1.0, 0)

            # Build results
            results: list[tuple[MathMemory | CompressedMathMemory, float]] = []
            for rid, activation in sorted(activations.items(), key=lambda x: x[1], reverse=True)[:limit]:
                record = self._records.get(rid)
                if record:
                    record.touch()
                    results.append((record, activation))

        return results

    @property
    def graph(self) -> DependencyGraph:
        return self._graph

    def save(self, path: str | None = None) -> None:
        """Persist to JSON file."""
        save_path = path or self.config.persistence_path
        if not save_path:
            return

        with self._rwlock.read():
            data = {
                "version": self._VERSION,
                "records": {rid: r.to_dict() for rid, r in self._records.items()},
                "stats": {
                    "total_stores": self._total_stores,
                    "total_recalls": self._total_recalls,
                },
            }

        try:
            from maxim.utils.atomic_io import atomic_write_json
            from maxim.utils.format_version import with_format_version

            atomic_write_json(save_path, with_format_version(data))
            logger.debug("Angular gyrus saved %d records to %s", len(self._records), save_path)
        except OSError as e:
            logger.warning("Failed to save angular gyrus: %s", e)

    def load(self, path: str | None = None) -> None:
        """Load from JSON file."""
        load_path = path or self.config.persistence_path
        if not load_path:
            return

        try:
            with open(load_path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.debug("Failed to load angular gyrus: %s", e)
            return

        from maxim.utils.format_version import check_format_version

        check_format_version(data, "angular_gyrus", log=logger)

        with self._rwlock.write():
            self._records.clear()
            self._context_index.clear()
            self._record_contexts.clear()
            self._graph = DependencyGraph()

            for _rid, rdict in data.get("records", {}).items():
                if rdict.get("_compressed"):
                    record = CompressedMathMemory.from_dict(rdict)
                else:
                    record = MathMemory.from_dict(rdict)
                self._records[record.id] = record
                self._index_record(record)
                self._graph.add_node(record.id, record.id)

            stats = data.get("stats", {})
            self._total_stores = stats.get("total_stores", 0)
            self._total_recalls = stats.get("total_recalls", 0)

        logger.debug("Angular gyrus loaded %d records from %s", len(self._records), load_path)

    def consolidate(self, **kwargs: Any) -> dict[str, int]:
        """Consolidation cycle: score records, compress/remove stale ones.

        Returns counts: compressed, removed, preserved.
        """
        now = time.time()
        max_age = self.config.max_age_without_access
        compress_thresh = self.config.compression_threshold
        retain_thresh = self.config.retention_threshold

        to_remove: list[str] = []
        to_compress: list[str] = []
        preserved = 0

        with self._rwlock.read():
            for rid, record in self._records.items():
                age = now - record.accessed_at
                score = self._score_retention(record, age, max_age)

                # Long-term records get a boost
                if record.long_term:
                    score = min(1.0, score * 2.0)
                # Built-in facts never decay
                if isinstance(record, MathMemory) and record.source == "built_in":
                    preserved += 1
                    continue

                if score < retain_thresh:
                    to_remove.append(rid)
                elif score < compress_thresh and isinstance(record, MathMemory):
                    to_compress.append(rid)
                else:
                    preserved += 1

        # Apply changes under write lock
        with self._rwlock.write():
            for rid in to_compress:
                record = self._records.get(rid)
                if isinstance(record, MathMemory):
                    edge_count = len(self._graph.get_outgoing(rid)) + len(self._graph.get_incoming(rid))
                    self._records[rid] = CompressedMathMemory.from_math_record(record, edge_count)

            for rid in to_remove:
                if rid in self._records:
                    # Cleanup index
                    if rid in self._record_contexts:
                        for key in self._record_contexts[rid]:
                            if key in self._context_index:
                                self._context_index[key].discard(rid)
                        del self._record_contexts[rid]
                    self._graph.remove_node(rid)
                    del self._records[rid]

        result = {
            "compressed": len(to_compress),
            "removed": len(to_remove),
            "preserved": preserved,
        }
        logger.debug("Angular gyrus consolidation: %s", result)
        return result

    def register_capture_callback(self, callback: Callable[[str, MathMemory], None]) -> None:
        self._capture_callbacks.append(callback)

    def register_deletion_callback(self, callback: Callable[[str], None]) -> None:
        self._deletion_callbacks.append(callback)

    def stats(self) -> dict[str, Any]:
        with self._rwlock.read():
            full = sum(1 for r in self._records.values() if isinstance(r, MathMemory))
            compressed = sum(1 for r in self._records.values() if isinstance(r, CompressedMathMemory))
            return {
                "total_records": len(self._records),
                "full_records": full,
                "compressed_records": compressed,
                "total_stores": self._total_stores,
                "total_recalls": self._total_recalls,
                "graph_nodes": len(list(self._graph._nodes)) if hasattr(self._graph, "_nodes") else 0,
            }

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[MathMemory | CompressedMathMemory]:
        return iter(self._records.values())

    # ------------------------------------------------------------------
    # Exact Computation (Angular Gyrus specific)
    # ------------------------------------------------------------------

    _OPERATIONS: dict[str, Callable[..., float | int]] = {
        "add": lambda ops: sum(ops),
        "subtract": lambda ops: ops[0] - sum(ops[1:]) if ops else 0,
        "multiply": lambda ops: math.prod(ops),
        "divide": lambda ops: ops[0] / ops[1] if len(ops) >= 2 and ops[1] != 0 else float("nan"),
        "power": lambda ops: ops[0] ** ops[1] if len(ops) >= 2 else float("nan"),
        "modulo": lambda ops: ops[0] % ops[1] if len(ops) >= 2 and ops[1] != 0 else float("nan"),
    }

    _OP_SYMBOLS: dict[str, str] = {
        "add": "+",
        "subtract": "-",
        "multiply": "*",
        "divide": "/",
        "power": "**",
        "modulo": "%",
    }

    def compute(self, operation: str, operands: list[float]) -> ExactResult:
        """Exact arithmetic: add, subtract, multiply, divide, power, modulo.

        Returns ExactResult with verbal and code representations.
        """
        op_func = self._OPERATIONS.get(operation)
        if op_func is None:
            return ExactResult(
                value=float("nan"),
                operation=operation,
                verbal=f"Unknown operation: {operation}",
                code="",
            )

        try:
            result = op_func(operands)
        except (ValueError, OverflowError, ZeroDivisionError) as e:
            return ExactResult(
                value=float("nan"),
                operation=operation,
                verbal=f"Error: {e}",
                code="",
            )

        symbol = self._OP_SYMBOLS.get(operation, operation)
        operand_strs = [str(o) for o in operands]

        if operation in ("add", "subtract", "multiply"):
            code = f" {symbol} ".join(operand_strs)
        elif operation == "divide" and len(operands) >= 2:
            code = f"{operands[0]} / {operands[1]}"
        elif operation == "power" and len(operands) >= 2:
            code = f"{operands[0]} ** {operands[1]}"
        elif operation == "modulo" and len(operands) >= 2:
            code = f"{operands[0]} % {operands[1]}"
        else:
            code = f"{operation}({', '.join(operand_strs)})"

        verbal = f"{code} = {result}"

        return ExactResult(
            value=result,
            operation=operation,
            verbal=verbal,
            code=code,
        )

    def analyze(self, data: list[float], method: str = "descriptive") -> AnalysisResult:
        """Statistical analysis.

        Methods:
          - "descriptive": mean, median, mode, variance, std_dev, min, max, range, count
          - "linear": linear regression (slope, intercept, r_squared)
          - "quadratic": quadratic fit (a, b, c, r_squared)
          - "percentiles": 25th, 50th, 75th, 90th, 99th
        """
        if not data:
            return AnalysisResult(
                method=method,
                parameters={},
                verbal="No data provided",
                code="",
                confidence=0.0,
            )

        if method == "descriptive":
            return self._analyze_descriptive(data)
        elif method == "linear":
            return self._analyze_linear(data)
        elif method == "quadratic":
            return self._analyze_quadratic(data)
        elif method == "percentiles":
            return self._analyze_percentiles(data)
        else:
            return AnalysisResult(
                method=method,
                parameters={},
                verbal=f"Unknown analysis method: {method}",
                code="",
                confidence=0.0,
            )

    def verbalize(self, result: ExactResult | AnalysisResult) -> str:
        """Generate natural language description of a result."""
        return result.verbal

    def codify(self, method_name: str) -> str | None:
        """Retrieve code representation of a stored method."""
        records = self.recall(limit=1, name=method_name, category="METHOD")
        if records and isinstance(records[0], MathMemory):
            return records[0].code
        return None

    def recall_method(self, description: str) -> MathMemory | None:
        """Find a method matching a natural language description via keyword overlap."""
        keywords = set(description.lower().split())

        best_match: MathMemory | None = None
        best_score = 0

        with self._rwlock.read():
            for record in self._records.values():
                if not isinstance(record, MathMemory) or record.category != MathCategory.METHOD:
                    continue

                record_words = set(record.name.lower().replace("_", " ").split())
                record_words.update(record.verbal.lower().split())
                overlap = len(keywords & record_words)

                if overlap > best_score:
                    best_score = overlap
                    best_match = record

        return best_match

    # ------------------------------------------------------------------
    # Statistical analysis implementations
    # ------------------------------------------------------------------

    def _analyze_descriptive(self, data: list[float]) -> AnalysisResult:
        n = len(data)
        mean = statistics.mean(data)
        median = statistics.median(data)
        variance = statistics.variance(data) if n > 1 else 0.0
        std_dev = math.sqrt(variance)
        data_min = min(data)
        data_max = max(data)
        data_range = data_max - data_min

        try:
            mode = statistics.mode(data)
        except statistics.StatisticsError:
            mode = mean  # No unique mode

        params = {
            "count": float(n),
            "mean": mean,
            "median": median,
            "mode": mode,
            "variance": variance,
            "std_dev": std_dev,
            "min": data_min,
            "max": data_max,
            "range": data_range,
        }

        verbal = (
            f"Descriptive statistics (n={n}): "
            f"mean={mean:.4g}, median={median:.4g}, "
            f"std_dev={std_dev:.4g}, range=[{data_min:.4g}, {data_max:.4g}]"
        )
        code = f"statistics.mean(data)  # → {mean:.6g}"

        return AnalysisResult(
            method="descriptive",
            parameters=params,
            verbal=verbal,
            code=code,
            confidence=min(1.0, n / 30),  # Higher confidence with more data
        )

    def _analyze_linear(self, data: list[float]) -> AnalysisResult:
        """Least-squares linear regression: y = slope * x + intercept."""
        n = len(data)
        if n < 2:
            return AnalysisResult(
                method="linear",
                parameters={},
                verbal="Insufficient data for linear regression (need ≥2 points)",
                code="",
                confidence=0.0,
            )

        # x = 0, 1, 2, ...
        x_vals = list(range(n))
        x_mean = (n - 1) / 2.0
        y_mean = statistics.mean(data)

        ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, data))
        ss_xx = sum((x - x_mean) ** 2 for x in x_vals)
        ss_yy = sum((y - y_mean) ** 2 for y in data)

        if ss_xx == 0:
            return AnalysisResult(
                method="linear",
                parameters={"slope": 0.0, "intercept": y_mean, "r_squared": 0.0},
                verbal=f"Constant data: y = {y_mean:.4g}",
                code=f"slope=0, intercept={y_mean:.6g}",
                confidence=0.0,
            )

        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean
        r_squared = (ss_xy**2) / (ss_xx * ss_yy) if ss_yy != 0 else 0.0

        params = {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
        }

        strength = "strong" if r_squared >= 0.8 else "moderate" if r_squared >= 0.5 else "weak"
        direction = "positive" if slope > 0 else "negative" if slope < 0 else "flat"
        verbal = (
            f"Linear regression: y = {slope:.4g}x + {intercept:.4g} (R²={r_squared:.3f}, {strength} {direction} trend)"
        )
        code = f"np.polyfit(x, y, 1)  # → [{slope:.6g}, {intercept:.6g}]"

        return AnalysisResult(
            method="linear",
            parameters=params,
            verbal=verbal,
            code=code,
            confidence=r_squared,
        )

    def _analyze_quadratic(self, data: list[float]) -> AnalysisResult:
        """Least-squares quadratic fit: y = a*x² + b*x + c."""
        n = len(data)
        if n < 3:
            return AnalysisResult(
                method="quadratic",
                parameters={},
                verbal="Insufficient data for quadratic fit (need ≥3 points)",
                code="",
                confidence=0.0,
            )

        # Solve via normal equations (no numpy dependency)
        x_vals = list(range(n))

        # Build sums for normal equations
        s_x = sum(x_vals)
        s_x2 = sum(x**2 for x in x_vals)
        s_x3 = sum(x**3 for x in x_vals)
        s_x4 = sum(x**4 for x in x_vals)
        s_y = sum(data)
        s_xy = sum(x * y for x, y in zip(x_vals, data))
        s_x2y = sum(x**2 * y for x, y in zip(x_vals, data))

        # Normal equations: [[n, s_x, s_x2], [s_x, s_x2, s_x3], [s_x2, s_x3, s_x4]] * [c, b, a] = [s_y, s_xy, s_x2y]
        # Solve using Cramer's rule
        det = n * (s_x2 * s_x4 - s_x3**2) - s_x * (s_x * s_x4 - s_x3 * s_x2) + s_x2 * (s_x * s_x3 - s_x2**2)

        if abs(det) < 1e-15:
            return self._analyze_linear(data)  # Fall back to linear

        c = (
            s_y * (s_x2 * s_x4 - s_x3**2) - s_x * (s_xy * s_x4 - s_x2y * s_x3) + s_x2 * (s_xy * s_x3 - s_x2y * s_x2)
        ) / det

        b = (
            n * (s_xy * s_x4 - s_x2y * s_x3) - s_y * (s_x * s_x4 - s_x3 * s_x2) + s_x2 * (s_x * s_x2y - s_xy * s_x2)
        ) / det

        a = (n * (s_x2 * s_x2y - s_x3 * s_xy) - s_x * (s_x * s_x2y - s_xy * s_x2) + s_y * (s_x * s_x3 - s_x2**2)) / det

        # R² computation
        y_mean = s_y / n
        ss_tot = sum((y - y_mean) ** 2 for y in data)
        ss_res = sum((y - (a * x**2 + b * x + c)) ** 2 for x, y in zip(x_vals, data))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

        params = {"a": a, "b": b, "c": c, "r_squared": r_squared}

        verbal = f"Quadratic fit: y = {a:.4g}x² + {b:.4g}x + {c:.4g} (R²={r_squared:.3f})"
        code = f"np.polyfit(x, y, 2)  # → [{a:.6g}, {b:.6g}, {c:.6g}]"

        return AnalysisResult(
            method="quadratic",
            parameters=params,
            verbal=verbal,
            code=code,
            confidence=r_squared,
        )

    def _analyze_percentiles(self, data: list[float]) -> AnalysisResult:
        sorted_data = sorted(data)
        n = len(sorted_data)

        def _percentile(p: float) -> float:
            k = (n - 1) * p / 100.0
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_data[int(k)]
            return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)

        params = {
            "p25": _percentile(25),
            "p50": _percentile(50),
            "p75": _percentile(75),
            "p90": _percentile(90),
            "p99": _percentile(99),
        }

        verbal = (
            f"Percentiles: 25th={params['p25']:.4g}, 50th={params['p50']:.4g}, "
            f"75th={params['p75']:.4g}, 90th={params['p90']:.4g}, 99th={params['p99']:.4g}"
        )
        code = "np.percentile(data, [25, 50, 75, 90, 99])"

        return AnalysisResult(
            method="percentiles",
            parameters=params,
            verbal=verbal,
            code=code,
            confidence=min(1.0, n / 30),
        )

    # ------------------------------------------------------------------
    # Matrix operations (via linalg module)
    # ------------------------------------------------------------------

    def mat_multiply(self, a: list[list[float]], b: list[list[float]]) -> ExactResult:
        """Matrix multiplication A @ B.

        Uses the pure-Python linalg module for small matrices.
        """
        from maxim.math.linalg import mat_mul, mat_shape

        try:
            result = mat_mul(a, b)
            ra, ca = mat_shape(a)
            rb, cb = mat_shape(b)
            return ExactResult(
                value=result,
                operation="mat_multiply",
                verbal=f"Matrix product of ({ra}×{ca}) @ ({rb}×{cb}) = ({ra}×{cb}) matrix",
                code=f"mat_mul(A, B)  # ({ra}×{ca}) @ ({rb}×{cb})",
            )
        except (ValueError, IndexError) as e:
            return ExactResult(
                value=float("nan"),
                operation="mat_multiply",
                verbal=f"Matrix multiplication failed: {e}",
                code="",
            )

    def mat_eigenvalues(self, matrix: list[list[float]]) -> ExactResult:
        """Eigenvalues of a symmetric matrix.

        Uses QR algorithm with Wilkinson shifts via the linalg module.
        """
        from maxim.math.linalg import eigenvalues_symmetric, mat_shape

        try:
            eigenvals = eigenvalues_symmetric(matrix)
            n = mat_shape(matrix)[0]
            verbal_vals = ", ".join(f"{v:.6g}" for v in eigenvals)
            return ExactResult(
                value=eigenvals,
                operation="eigenvalues",
                verbal=f"Eigenvalues of {n}×{n} matrix: [{verbal_vals}]",
                code=f"eigenvalues_symmetric(M)  # {n}×{n}",
            )
        except (ValueError, IndexError) as e:
            return ExactResult(
                value=float("nan"),
                operation="eigenvalues",
                verbal=f"Eigenvalue computation failed: {e}",
                code="",
            )

    def solve_system(self, coefficients: list[list[float]], constants: list[float]) -> ExactResult:
        """Solve linear system Ax = b.

        Uses Gaussian elimination with partial pivoting.
        """
        from maxim.math.linalg import solve

        try:
            solution = solve(coefficients, constants)
            n = len(constants)
            verbal_vals = ", ".join(f"x{i + 1}={v:.6g}" for i, v in enumerate(solution))
            return ExactResult(
                value=solution,
                operation="solve_system",
                verbal=f"Solution to {n}×{n} system: {verbal_vals}",
                code=f"solve(A, b)  # {n} unknowns",
            )
        except ValueError as e:
            return ExactResult(
                value=float("nan"),
                operation="solve_system",
                verbal=f"System solve failed: {e}",
                code="",
            )

    def mat_determinant(self, matrix: list[list[float]]) -> ExactResult:
        """Determinant of a square matrix.

        Uses LU-style elimination with partial pivoting.
        """
        from maxim.math.linalg import determinant as linalg_det
        from maxim.math.linalg import mat_shape

        try:
            det = linalg_det(matrix)
            n = mat_shape(matrix)[0]
            return ExactResult(
                value=det,
                operation="determinant",
                verbal=f"Determinant of {n}×{n} matrix = {det:.6g}",
                code=f"determinant(M)  # {n}×{n}",
            )
        except (ValueError, IndexError) as e:
            return ExactResult(
                value=float("nan"),
                operation="determinant",
                verbal=f"Determinant computation failed: {e}",
                code="",
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_record(self, record: MathMemory | CompressedMathMemory) -> None:
        """Add record to context index for O(1) filtered recall."""
        keys: set[str] = set()

        if record.name:
            keys.add(f"name:{record.name}")
        keys.add(f"category:{record.category.name}")
        if record.domain:
            keys.add(f"domain:{record.domain}")

        for key in keys:
            if key not in self._context_index:
                self._context_index[key] = set()
            self._context_index[key].add(record.id)

        self._record_contexts[record.id] = keys

    def _score_retention(self, record: MathMemory | CompressedMathMemory, age: float, max_age: float) -> float:
        """Score a record for retention during consolidation.

        Combines recency, access frequency, and confidence.
        Math facts have slow decay (90-day max age).
        """
        if age >= max_age and record.access_count < 3:
            return 0.0

        # Recency (linear decay over max_age)
        recency = max(0.0, 1.0 - age / max_age) if max_age > 0 else 1.0

        # Access frequency (logarithmic — diminishing returns)
        frequency = min(1.0, math.log2(max(record.access_count, 1)) / 5.0)

        # Confidence contributes directly
        confidence = record.confidence

        # Weighted composite
        return recency * 0.3 + frequency * 0.3 + confidence * 0.4

    def _seed_built_ins(self) -> None:
        """Pre-populate with basic arithmetic and statistical facts/methods.

        Only runs if no records exist (avoids duplicating on repeated init).
        """
        if self._records:
            return

        now = time.time()
        built_ins: list[dict[str, Any]] = [
            # Constants
            {
                "name": "pi",
                "category": MathCategory.CONSTANT,
                "domain": "geometry",
                "verbal": "The ratio of a circle's circumference to its diameter, approximately 3.14159",
                "code": "math.pi  # 3.141592653589793",
                "source": "built_in",
            },
            {
                "name": "e",
                "category": MathCategory.CONSTANT,
                "domain": "calculus",
                "verbal": "Euler's number, the base of the natural logarithm, approximately 2.71828",
                "code": "math.e  # 2.718281828459045",
                "source": "built_in",
            },
            {
                "name": "phi",
                "category": MathCategory.CONSTANT,
                "domain": "geometry",
                "verbal": "The golden ratio, approximately 1.61803",
                "code": "(1 + math.sqrt(5)) / 2  # 1.618033988749895",
                "source": "built_in",
            },
            # Methods
            {
                "name": "mean",
                "category": MathCategory.METHOD,
                "domain": "statistics",
                "verbal": "The arithmetic mean (average) of a dataset: sum of values divided by count",
                "code": "statistics.mean(data)",
                "inputs": ["list[float]"],
                "outputs": ["float"],
                "source": "built_in",
            },
            {
                "name": "median",
                "category": MathCategory.METHOD,
                "domain": "statistics",
                "verbal": "The middle value of a sorted dataset; robust to outliers",
                "code": "statistics.median(data)",
                "inputs": ["list[float]"],
                "outputs": ["float"],
                "source": "built_in",
            },
            {
                "name": "variance",
                "category": MathCategory.METHOD,
                "domain": "statistics",
                "verbal": "The average of squared deviations from the mean; measures data spread",
                "code": "statistics.variance(data)",
                "inputs": ["list[float]"],
                "outputs": ["float"],
                "source": "built_in",
            },
            {
                "name": "standard_deviation",
                "category": MathCategory.METHOD,
                "domain": "statistics",
                "verbal": "The square root of variance; measures data spread in original units",
                "code": "statistics.stdev(data)",
                "inputs": ["list[float]"],
                "outputs": ["float"],
                "source": "built_in",
            },
            {
                "name": "linear_regression",
                "category": MathCategory.METHOD,
                "domain": "statistics",
                "verbal": "Fits a straight line (y = mx + b) to data via least squares",
                "code": "np.polyfit(x, y, 1)  # returns [slope, intercept]",
                "inputs": ["list[float]"],
                "outputs": ["float", "float", "float"],
                "source": "built_in",
            },
            {
                "name": "quadratic_fit",
                "category": MathCategory.METHOD,
                "domain": "statistics",
                "verbal": "Fits a parabola (y = ax² + bx + c) to data via least squares",
                "code": "np.polyfit(x, y, 2)  # returns [a, b, c]",
                "inputs": ["list[float]"],
                "outputs": ["float", "float", "float", "float"],
                "source": "built_in",
            },
        ]

        for spec in built_ins:
            record = MathMemory(
                id=f"builtin_{spec['name']}",
                timestamp=now,
                created_at=now,
                accessed_at=now,
                name=spec["name"],
                category=spec["category"],
                domain=spec.get("domain", ""),
                verbal=spec.get("verbal", ""),
                code=spec.get("code", ""),
                inputs=spec.get("inputs", []),
                outputs=spec.get("outputs", []),
                source="built_in",
                confidence=1.0,
                long_term=True,
            )
            self._records[record.id] = record
            self._index_record(record)
            self._graph.add_node(record.id, record.id)

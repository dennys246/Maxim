"""ConsolidationOrchestrator — Wave-based memory consolidation during sleep.

Processes JSON sidecars from ``data/short_term_memory/`` during sleep cycles,
re-evaluating staged moments against accumulated system evidence (NAc
corroboration, temporal recurrence, percept/context recurrence).

Separates consolidation from Hippocampus — operates on sidecar files and
delegates promotion to ``hippocampus.capture_from_loop()``.

Phase 3c of the consolidation expansion plan.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from maxim.decisions.significance import sigmoid

logger = logging.getLogger(__name__)


class ConsolidationOrchestrator:
    """Manages wave consolidation of short-term memory to long-term.

    Called during sleep to re-evaluate staged moments. Each moment
    is scored across multiple waves, with path-dependent thresholds:
    - Acute (RPE-triggered): lower bar (0.45), one-shot learning
    - Chronic (recurrence-triggered): higher bar (0.60), needs evidence

    Dependencies injected at construction:
    - hippocampus: for capture_from_loop() and recall
    - nac: for observation count corroboration
    - scn: for temporal recurrence queries
    - weight_learner: for utility signal harvesting
    - percept_index: SimilarityIndex for percept similarity
    - context_index: SimilarityIndex for language context similarity
    """

    def __init__(
        self,
        staging_dir: str,
        hippocampus: Any,
        nac: Any,
        scn: Any,
        weight_learner: Any,
        percept_index: Any,
        context_index: Any,
    ) -> None:
        self.staging_dir = staging_dir
        self.hippocampus = hippocampus
        self.nac = nac
        self.scn = scn
        self.weight_learner = weight_learner
        self.percept_index = percept_index
        self.context_index = context_index
        # Path-dependent consolidation thresholds
        self.acute_threshold = 0.45
        self.chronic_threshold = 0.60
        self.immediate_promotion_threshold = 0.85
        self.max_waves = 5

    def run_wave(self) -> dict[str, int]:
        """Process one consolidation wave. Called during sleep.

        Returns dict with counts: promoted, survived, expired, errors.
        """
        stats = {"promoted": 0, "survived": 0, "expired": 0, "errors": 0}
        sidecars = self._load_sidecars()

        for path, sidecar in sidecars:
            try:
                # Immediate promotion: skip waves for very high significance
                if (
                    sidecar.get("sleep_cycles_seen", 0) == 0
                    and sidecar.get("significance_score", 0)
                    >= self.immediate_promotion_threshold
                ):
                    sidecar["wave_scores"].append(sidecar["significance_score"])
                    self._promote(sidecar)
                    os.remove(path)
                    stats["promoted"] += 1
                    continue

                sidecar["sleep_cycles_seen"] = sidecar.get("sleep_cycles_seen", 0) + 1
                wave_score = self._compute_wave_score(sidecar)
                sidecar["wave_scores"].append(wave_score)

                # Path-dependent threshold
                threshold = (
                    self.acute_threshold
                    if sidecar.get("staging_path") == "acute"
                    else self.chronic_threshold
                )

                if wave_score >= threshold:
                    self._promote(sidecar)
                    os.remove(path)
                    stats["promoted"] += 1
                elif sidecar["sleep_cycles_seen"] >= self.max_waves:
                    os.remove(path)
                    stats["expired"] += 1
                else:
                    self._save_sidecar(path, sidecar)
                    stats["survived"] += 1
            except Exception as e:
                logger.warning("Wave processing error for %s: %s", path, e)
                stats["errors"] += 1

        # Harvest utility signals from previously promoted memories
        try:
            self.weight_learner.harvest_utility()
        except Exception as e:
            logger.warning("Utility harvest error: %s", e)

        logger.info(
            "Wave complete: %d promoted, %d survived, %d expired",
            stats["promoted"],
            stats["survived"],
            stats["expired"],
        )
        return stats

    def _compute_wave_score(self, sidecar: dict[str, Any]) -> float:
        """Re-evaluate a staged moment with current system state."""

        # 1. NAc corroboration — has the event→outcome pattern strengthened?
        nac_corroboration = 0.0
        try:
            event_sig = sidecar.get("action", {}).get("tool", "")
            if event_sig and self.nac is not None:
                links = self.nac.get_links_for_event(event_sig)
                obs_delta = sum(
                    getattr(l, "observation_count", 0) for l in links
                ) - sidecar.get("nac_obs_at_staging", 0)
                nac_corroboration = sigmoid(obs_delta / 5.0)
        except Exception:
            pass

        # 2. SCN temporal recurrence — similar activity at similar times?
        temporal_recurrence = 0.0
        try:
            temporal = sidecar.get("temporal", {})
            if temporal and self.scn is not None:
                from maxim.time.temporal_signature import TemporalSignature

                sig = TemporalSignature(
                    timestamp=sidecar.get("timestamp", 0),
                    circadian_phase=temporal.get("circadian_phase", 0),
                    weekly_phase=temporal.get("weekly_phase", 0),
                    monthly_phase=temporal.get("monthly_phase", 0),
                    annual_phase=temporal.get("annual_phase", 0),
                )
                similar_memories = self.scn.query_similar_time(sig, tolerance=1)
                temporal_recurrence = min(1.0, len(similar_memories) / 5.0)
        except Exception:
            pass

        # 3. Percept recurrence — similar percepts since staging?
        percept_recurrence = 0.0
        percept_matches: list[tuple[str, float]] = []
        try:
            percept = sidecar.get("percept", {})
            if percept and self.percept_index is not None:
                from maxim.memory.context_index import percept_to_canonical

                percept_str = percept_to_canonical(percept)
                if percept_str:
                    percept_matches = self.percept_index.query_similar(
                        percept_str, min_similarity=0.3
                    )
                    staging_ts = sidecar.get("timestamp", 0)
                    recent_hits = 0
                    for mid, _ in percept_matches:
                        mem = self.hippocampus.get(mid) if self.hippocampus else None
                        if mem and getattr(mem, "timestamp", 0) > staging_ts:
                            recent_hits += 1
                    percept_recurrence = min(1.0, recent_hits / 3.0)
        except Exception:
            pass

        # 4. Context recurrence — similar language contexts?
        context_recurrence = 0.0
        try:
            context_str = sidecar.get("context_summary", "")
            if context_str and self.context_index is not None:
                context_matches = self.context_index.query_similar(
                    context_str, min_similarity=0.3
                )
                staging_ts = sidecar.get("timestamp", 0)
                recent_hits = 0
                for mid, _ in context_matches:
                    mem = self.hippocampus.get(mid) if self.hippocampus else None
                    if mem and getattr(mem, "timestamp", 0) > staging_ts:
                        recent_hits += 1
                context_recurrence = min(1.0, recent_hits / 3.0)
        except Exception:
            pass

        # 5. Novelty decay — novel → routine = worth encoding
        novelty_at_staging = sidecar.get("heuristic_scores", {}).get("novelty", 0.5)
        current_novelty = max(0.0, 1.0 - len(percept_matches) / 5.0)
        novelty_decay = max(0.0, novelty_at_staging - current_novelty)

        significance = sidecar.get("significance_score", 0.0)

        wave_score = (
            0.30 * significance
            + 0.20 * nac_corroboration
            + 0.20 * temporal_recurrence
            + 0.12 * percept_recurrence
            + 0.10 * context_recurrence
            + 0.08 * novelty_decay
        )
        return wave_score

    def _promote(self, sidecar: dict[str, Any]) -> None:
        """Capture staged moment as long-term memory via Hippocampus.

        Uses capture_from_loop() which accepts raw dicts and constructs
        typed dataclasses internally.
        """
        if self.hippocampus is None:
            return

        percept = sidecar.get("percept", {})
        try:
            memory_id = self.hippocampus.capture_from_loop(
                observation={
                    "detections": percept.get("objects", []),
                    "transcript": percept.get("speech"),
                    "people": percept.get("people", []),
                },
                state={
                    "salience": sidecar.get("wave_scores", [0.5])[-1],
                    "novelty": sidecar.get("heuristic_scores", {}).get("novelty", 0.5),
                },
                intent={"goal": sidecar.get("active_goal", "")},
                decision={},
                action=sidecar.get("action", {}),
                result=sidecar.get("outcome", {}),
                long_term=True,
            )
        except Exception as e:
            logger.warning("Promotion capture failed: %s", e)
            return

        memory = self.hippocampus.get(memory_id)
        if memory is None:
            return

        # Register with SCN temporal index
        try:
            temporal = sidecar.get("temporal", {})
            if temporal and self.scn is not None:
                from maxim.time.temporal_signature import TemporalSignature

                sig = TemporalSignature(
                    timestamp=sidecar.get("timestamp", 0),
                    circadian_phase=temporal.get("circadian_phase", 0),
                    weekly_phase=temporal.get("weekly_phase", 0),
                    monthly_phase=temporal.get("monthly_phase", 0),
                    annual_phase=temporal.get("annual_phase", 0),
                )
                wave_scores = sidecar.get("wave_scores", [0.5])
                self.scn.register(
                    memory.id, sig, significance=wave_scores[-1]
                )
        except Exception:
            pass

        # Register with similarity indices
        try:
            from maxim.memory.context_index import percept_to_canonical

            if self.percept_index is not None:
                self.percept_index.register(
                    memory.id, percept_to_canonical(percept)
                )
            if self.context_index is not None and sidecar.get("context_summary"):
                self.context_index.register(
                    memory.id, sidecar["context_summary"]
                )
        except Exception:
            pass

        # Track for long-term utility measurement
        try:
            heuristic_scores = sidecar.get("heuristic_scores", {})
            if heuristic_scores:
                self.weight_learner.track_promotion(memory.id, heuristic_scores)
        except Exception:
            pass

        logger.debug(
            "Promoted memory %s (wave_scores=%s)",
            memory.id,
            sidecar.get("wave_scores"),
        )

    def _save_sidecar(self, path: str, sidecar: dict[str, Any]) -> None:
        """Atomic write: temp file + rename."""
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(sidecar, f, indent=2)
        os.replace(tmp, path)

    def _load_sidecars(self) -> list[tuple[str, dict[str, Any]]]:
        """Load all sidecar JSON files from staging dir."""
        if not os.path.isdir(self.staging_dir):
            return []
        results: list[tuple[str, dict[str, Any]]] = []
        for fname in sorted(os.listdir(self.staging_dir)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self.staging_dir, fname)
            try:
                with open(path) as f:
                    results.append((path, json.load(f)))
            except (json.JSONDecodeError, IOError):
                continue  # Skip corrupted/partial files
        return results

    # ── Phase 3d: Chronic staging (recurrence detection) ─────────────────

    def find_chronic_candidates(self) -> list[Any]:
        """Find recurring percept patterns using bounded SCN time bins.

        Queries specific temporal windows (last 24h, same weekday, monthly)
        instead of scanning all short-term memories.
        """
        if self.hippocampus is None or self.scn is None:
            return []

        from maxim.time.temporal_signature import TemporalSignature

        now = TemporalSignature.now()
        hour_bin, day_bin, _, _ = now.to_bins()
        candidates: dict[str, Any] = {}

        # Recent: last 24 hours of circadian bins
        for hour_offset in range(24):
            hour = (hour_bin - hour_offset) % 24
            for mid in self.scn.query_hour(hour):
                if mid not in candidates:
                    mem = self.hippocampus.get(mid)
                    if mem and not getattr(mem, "long_term", False):
                        candidates[mid] = mem

        # Weekly: same day of week
        for mid in self.scn.query_day(day_bin):
            if mid not in candidates:
                mem = self.hippocampus.get(mid)
                if mem and not getattr(mem, "long_term", False):
                    candidates[mid] = mem

        # Monthly: similar time tolerance
        for mid in self.scn.query_similar_time(now, tolerance=1):
            if mid not in candidates:
                mem = self.hippocampus.get(mid)
                if mem and not getattr(mem, "long_term", False):
                    candidates[mid] = mem

        return list(candidates.values())

    def detect_recurrence(
        self,
        candidates: list[Any],
        recurrence_threshold: int = 3,
    ) -> list[list[Any]]:
        """Group candidates by percept similarity using LSH.

        Returns clusters where each cluster has >= recurrence_threshold members.
        """
        if self.percept_index is None or not candidates:
            return []

        from maxim.memory.context_index import percept_to_canonical

        clusters: list[list[Any]] = []
        used: set[str] = set()
        candidate_ids = {getattr(m, "id", ""): m for m in candidates}

        for mem in candidates:
            mid = getattr(mem, "id", "")
            if mid in used:
                continue

            # Extract percept data
            percept = getattr(mem, "perception", None)
            if percept is None:
                percept = getattr(mem, "percept", {})
            if hasattr(percept, "to_dict"):
                percept = percept.to_dict()
            elif not isinstance(percept, dict):
                percept = {}

            canonical = percept_to_canonical(percept)
            if not canonical or len(canonical) < 10:
                used.add(mid)
                continue

            matches = self.percept_index.query_similar(canonical, min_similarity=0.4)
            cluster_ids = {m_id for m_id, _ in matches} & set(candidate_ids.keys())
            cluster_ids.add(mid)
            cluster_ids -= used

            if len(cluster_ids) >= recurrence_threshold:
                cluster = [candidate_ids[cid] for cid in cluster_ids if cid in candidate_ids]
                clusters.append(cluster)
                used.update(cluster_ids)
            else:
                used.add(mid)

        return clusters

    def run_chronic_staging(self) -> int:
        """Detect recurring patterns and create chronic sidecars.

        Called during sleep alongside wave processing. Returns the number
        of chronic sidecars created.
        """
        candidates = self.find_chronic_candidates()
        if not candidates:
            return 0

        clusters = self.detect_recurrence(candidates)
        created = 0

        for cluster in clusters:
            try:
                sidecar = self._create_chronic_sidecar(cluster)
                if sidecar is not None:
                    self._write_chronic_sidecar(sidecar)
                    created += 1
            except Exception as e:
                logger.warning("Chronic staging error: %s", e)

        if created:
            logger.info("Created %d chronic sidecars from %d clusters", created, len(clusters))

        return created

    def _create_chronic_sidecar(self, cluster: list[Any]) -> dict[str, Any] | None:
        """Create a chronic sidecar from a cluster of recurring memories."""
        import time as _time

        from maxim.memory.context_index import percept_to_canonical

        if not cluster:
            return None

        # Pick representative: highest percept confidence
        best = cluster[0]
        best_conf = 0.0
        for mem in cluster:
            percept = getattr(mem, "perception", None)
            if percept is None:
                percept = getattr(mem, "percept", {})
            if hasattr(percept, "to_dict"):
                percept = percept.to_dict()
            elif not isinstance(percept, dict):
                percept = {}

            max_conf = max(
                (obj.get("confidence", 0) for obj in percept.get("objects", [])),
                default=0,
            )
            if max_conf > best_conf:
                best_conf = max_conf
                best = mem

        # Extract representative percept
        rep_percept = getattr(best, "perception", None)
        if rep_percept is None:
            rep_percept = getattr(best, "percept", {})
        if hasattr(rep_percept, "to_dict"):
            rep_percept = rep_percept.to_dict()
        elif not isinstance(rep_percept, dict):
            rep_percept = {}

        # Compute chronic significance
        recurrence_count = len(cluster)
        recurrence_ratio = min(1.0, recurrence_count / 3.0)

        # Temporal regularity: how tightly clustered in time?
        temporal_regularity = 0.0
        if self.scn is not None:
            try:
                bins_set: set[tuple[int, int]] = set()
                for mem in cluster:
                    b = self.scn.get_bins(getattr(mem, "id", ""))
                    if b is not None:
                        bins_set.add(b)
                if len(bins_set) > 0:
                    temporal_regularity = 1.0 - min(1.0, len(bins_set) / len(cluster))
            except Exception:
                pass

        # Percept cluster tightness via LSH
        tightness = 0.0
        if self.percept_index is not None:
            canonical = percept_to_canonical(rep_percept)
            if canonical:
                matches = self.percept_index.query_similar(canonical, min_similarity=0.4)
                cluster_ids = {getattr(m, "id", "") for m in cluster}
                matched_ids = {mid for mid, _ in matches} & cluster_ids
                tightness = len(matched_ids) / len(cluster) if cluster else 0.0

        chronic_significance = min(
            1.0,
            0.3 * recurrence_ratio + 0.4 * temporal_regularity + 0.3 * tightness,
        )

        now = _time.time()
        return {
            "version": "1.0",
            "timestamp": now,
            "staging_path": "chronic",
            "significance_score": round(chronic_significance, 4),
            "heuristic_scores": {
                "recurrence_count": recurrence_count,
                "temporal_regularity": round(temporal_regularity, 4),
                "percept_tightness": round(tightness, 4),
            },
            "rpe": {"raw": 0.0, "running_mean": 0.0, "running_std": 1.0},
            "temporal": {},
            "nac_obs_at_staging": 0,
            "percept": rep_percept,
            "context_summary": "",
            "action": {"tool": "chronic_pattern"},
            "outcome": {"success": True, "valence": 0.5},
            "active_goal": "",
            "sleep_cycles_seen": 0,
            "wave_scores": [],
            "cluster_member_ids": [getattr(m, "id", "") for m in cluster],
        }

    def _write_chronic_sidecar(self, sidecar: dict[str, Any]) -> None:
        """Write chronic sidecar to staging dir."""
        import time as _time

        os.makedirs(self.staging_dir, exist_ok=True)
        now = int(_time.time())
        count = sidecar.get("heuristic_scores", {}).get("recurrence_count", 0)
        filename = f"{now}_chronic_recurrence-{count}.json"
        path = os.path.join(self.staging_dir, filename)
        self._save_sidecar(path, sidecar)


__all__ = ["ConsolidationOrchestrator"]

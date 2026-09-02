"""Cerebellum — learned forward models for predicting sensory consequences.

Stores lightweight predictors per ``(entity_path, modulator, affordance,
param_bucket)`` key.  Each predictor learns via prediction-error feedback
(Rescorla-Wagner style, mirroring NAc but for sensory prediction instead
of reward).

The LLM is consulted **only when no confident forward model exists** —
it teaches the Cerebellum, then fades out.

Thread safety: per-key ``RLock`` (read-heavy pattern; per-key granularity
avoids global contention).

Bio-mapping: MECHANISM (single-cue delta rule). The per-sensor update
``mean += lr * (actual - expected)`` IS the Rescorla-Wagner delta rule in its
single-cue form; prediction-error-proportional learning is the earned claim.
NOT implemented: compound-cue competition (classic R-W's summed prediction
term), cerebellar microcircuitry (Purkinje/climbing-fiber timing).
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from maxim.utils.logging import log_swallowed_exception
from maxim.embodiment.sem import Entity

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model key + bucketing
# ---------------------------------------------------------------------------


class ModelKey(NamedTuple):
    """Unique key for a forward model in the Cerebellum."""

    entity_path: str
    modulator: str
    affordance: str
    param_bucket: str


def bucket_params(
    params: dict[str, Any],
    sensor_ranges: dict[str, tuple[float, float]] | None = None,
    *,
    default_step: float = 5.0,
    range_fraction: float = 0.1,
) -> str:
    """Bucket action parameters into a string key.

    Numeric params are rounded to the nearest bucket.  Bucket size is
    ``range_fraction`` (10%) of the sensor range if known, otherwise
    ``default_step`` (5.0).

    Parameters
    ----------
    params : dict
        Raw action parameters.
    sensor_ranges : dict, optional
        ``{param_name: (lo, hi)}`` for range-aware bucketing.
    default_step : float
        Bucket step for params without a known range.
    range_fraction : float
        Fraction of range to use as bucket step (default 10%).

    Returns
    -------
    str
        Deterministic string key for this parameter combination.
    """
    parts: list[str] = []
    for k in sorted(params.keys()):
        v = params[k]
        if isinstance(v, (int, float)):
            step = default_step
            if sensor_ranges and k in sensor_ranges:
                lo, hi = sensor_ranges[k]
                rng = hi - lo
                if rng > 0:
                    step = rng * range_fraction
            step = max(step, 0.01)  # guard against zero
            bucketed = round(v / step) * step
            parts.append(f"{k}={bucketed:.2f}")
        elif isinstance(v, str):
            parts.append(f"{k}={v}")
        else:
            parts.append(f"{k}={v!r}")
    return "|".join(parts) if parts else "_empty_"


# ---------------------------------------------------------------------------
# Per-key forward model
# ---------------------------------------------------------------------------


@dataclass
class ForwardModel:
    """Learned predictor for one (entity, modulator, affordance, bucket).

    Tracks per-sensor expected values using Rescorla-Wagner update:
        expected += learning_rate * (actual - expected)

    Variance tracks prediction uncertainty:
        variance += lr * ((actual - expected)^2 - variance)
    """

    observations: int = 0
    last_observed: float = 0.0

    # Per-sensor statistics: {sensor_name: {"mean": float, "variance": float}}
    sensor_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    # Per-failure-mode probabilities: {failure_name: probability}
    failure_probs: dict[str, float] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        """Confidence grows with observations, capped at 0.95.

        Uses a logistic curve: ``1 - exp(-0.3 * observations)``.
        Reaches 0.3 after ~1 observation, 0.7 after ~4, 0.9 after ~8.
        """
        if self.observations == 0:
            return 0.0
        return min(0.95, 1.0 - math.exp(-0.3 * self.observations))

    def predict_sensors(self) -> dict[str, float]:
        """Return predicted sensor values (means)."""
        return {name: stats["mean"] for name, stats in self.sensor_stats.items()}

    def get_variance(self) -> dict[str, float]:
        """Return per-sensor variance."""
        return {name: stats["variance"] for name, stats in self.sensor_stats.items()}

    def update(
        self,
        actual: dict[str, float],
        learning_rate: float = 0.2,
    ) -> dict[str, float]:
        """Rescorla-Wagner update from actual sensor readings.

        Returns prediction errors per sensor (actual - expected).
        """
        errors: dict[str, float] = {}
        now = time.time()
        self.observations += 1
        self.last_observed = now

        for sensor_name, actual_val in actual.items():
            if sensor_name not in self.sensor_stats:
                # First observation — initialize
                self.sensor_stats[sensor_name] = {
                    "mean": actual_val,
                    "variance": 0.0,
                }
                errors[sensor_name] = 0.0
                continue

            stats = self.sensor_stats[sensor_name]
            expected = stats["mean"]
            error = actual_val - expected
            errors[sensor_name] = error

            # R-W update
            stats["mean"] = expected + learning_rate * error
            stats["variance"] = stats["variance"] + learning_rate * (error * error - stats["variance"])

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "observations": self.observations,
            "last_observed": self.last_observed,
            "sensor_stats": dict(self.sensor_stats),
            "failure_probs": dict(self.failure_probs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForwardModel:
        """Deserialize from persistence."""
        model = cls()
        model.observations = data.get("observations", 0)
        model.last_observed = data.get("last_observed", 0.0)
        model.sensor_stats = {k: dict(v) for k, v in data.get("sensor_stats", {}).items()}
        model.failure_probs = dict(data.get("failure_probs", {}))
        return model


# ---------------------------------------------------------------------------
# Cerebellum
# ---------------------------------------------------------------------------


@dataclass
class CerebellumConfig:
    """Configuration for the Cerebellum."""

    confidence_threshold: float = 0.3
    high_variance_threshold: float = 0.5
    learning_rate: float = 0.2
    max_models: int = 10000
    prune_age_s: float = 7 * 24 * 3600  # 7 days
    persistence_path: str | None = None


class Cerebellum:
    """Forward models for predicting sensory consequences of actions.

    Stores lightweight predictors per ``(entity_path, modulator, affordance,
    param_bucket)`` key.  Each predictor learns via prediction-error feedback
    (Rescorla-Wagner style).

    Thread safety
    -------------
    Per-key ``RLock`` instances.  Read-heavy pattern: multiple concurrent
    ``predict()`` calls are safe; ``observe()`` acquires the key's lock
    exclusively.  Global operations (``prune``, ``export``) acquire all
    locks sequentially.
    """

    def __init__(self, config: CerebellumConfig | None = None) -> None:
        self.config = config or CerebellumConfig()
        self._models: dict[ModelKey, ForwardModel] = {}
        self._locks: dict[ModelKey, threading.RLock] = {}
        self._global_lock = threading.RLock()
        self._total_predictions: int = 0
        self._total_observations: int = 0
        self._llm_fallbacks: int = 0

        # Motor program registry (Phase 1b)
        from maxim.embodiment.motor import ProgramRegistry

        self.programs = ProgramRegistry()

    # -- key construction ---------------------------------------------------

    @staticmethod
    def make_key(
        entity: Entity | str,
        modulator: str,
        affordance: str,
        params: dict[str, Any],
        sensor_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> ModelKey:
        """Build a ModelKey from entity + action + bucketed params."""
        path = entity.full_path if isinstance(entity, Entity) else entity
        bucket = bucket_params(params, sensor_ranges)
        return ModelKey(
            entity_path=path,
            modulator=modulator,
            affordance=affordance,
            param_bucket=bucket,
        )

    # -- lock management ----------------------------------------------------

    def _get_lock(self, key: ModelKey) -> threading.RLock:
        with self._global_lock:
            if key not in self._locks:
                self._locks[key] = threading.RLock()
            return self._locks[key]

    # -- predict ------------------------------------------------------------

    def predict(
        self,
        entity: Entity | str,
        modulator: str,
        affordance: str,
        params: dict[str, Any],
        sensor_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, float] | None:
        """Return predicted sensor values, or None if no confident model.

        Returns
        -------
        dict[str, float] or None
            Predicted sensor values if model exists and is confident.
            None if no model, low confidence, or high variance (caller
            should fall back to LLM).
        """
        key = self.make_key(entity, modulator, affordance, params, sensor_ranges)
        lock = self._get_lock(key)

        with lock:
            model = self._models.get(key)
            if model is None:
                return None

            if model.confidence < self.config.confidence_threshold:
                return None

            # Check for high variance (uncertain predictions)
            variances = model.get_variance()
            if variances:
                max_var = max(variances.values())
                if max_var > self.config.high_variance_threshold:
                    return None

            self._total_predictions += 1
            predicted = model.predict_sensors()
            try:
                from maxim.simulation.sim_logger import sim_cerebellum_predict

                entity_name = entity if isinstance(entity, str) else entity.full_path
                sim_cerebellum_predict(entity_name, affordance, predicted, model.confidence)
            except Exception:
                pass
            return predicted

    def has_model(
        self,
        entity: Entity | str,
        modulator: str,
        affordance: str,
        params: dict[str, Any],
        sensor_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> bool:
        """Check if a forward model exists for this action."""
        key = self.make_key(entity, modulator, affordance, params, sensor_ranges)
        return key in self._models

    def get_confidence(
        self,
        entity: Entity | str,
        modulator: str,
        affordance: str,
        params: dict[str, Any],
        sensor_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> float:
        """Return model confidence (0.0 if no model)."""
        key = self.make_key(entity, modulator, affordance, params, sensor_ranges)
        model = self._models.get(key)
        return model.confidence if model is not None else 0.0

    def get_variance(
        self,
        entity: Entity | str,
        modulator: str,
        affordance: str,
        params: dict[str, Any],
        sensor_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, float]:
        """Return per-sensor variance (empty dict if no model)."""
        key = self.make_key(entity, modulator, affordance, params, sensor_ranges)
        model = self._models.get(key)
        return model.get_variance() if model is not None else {}

    # -- observe (train) ----------------------------------------------------

    def observe(
        self,
        key: ModelKey,
        actual: dict[str, float],
    ) -> dict[str, float]:
        """Update forward model from actual sensor readings.

        Creates the model if it doesn't exist yet.

        Parameters
        ----------
        key : ModelKey
            The model key (from ``make_key()``).
        actual : dict[str, float]
            Actual sensor readings after the action executed.

        Returns
        -------
        dict[str, float]
            Prediction errors per sensor (actual - expected).
            All zeros for first observation.
        """
        lock = self._get_lock(key)

        with lock:
            if key not in self._models:
                with self._global_lock:
                    self._models[key] = ForwardModel()

            model = self._models[key]
            errors = model.update(actual, self.config.learning_rate)
            self._total_observations += 1

        try:
            from maxim.simulation.sim_logger import sim_cerebellum_train

            # ModelKey fields are (entity_path, modulator, affordance,
            # param_bucket). This block read `key.entity` against a comment
            # describing a five-field key that never existed; the resulting
            # AttributeError was eaten by the bare `except Exception: pass`
            # below, so sim_cerebellum_train had NEVER fired. Found 2026-09-01
            # alongside the caller-side signature break in tool_bridge — the
            # type drifted and two swallows hid both halves.
            max_error = max(abs(v) for v in errors.values()) if errors else 0.0
            sim_cerebellum_train(
                entity=str(key.entity_path),
                affordance=str(key.affordance),
                pred_error=max_error,
                confidence=model.confidence,
            )
        except Exception:
            # Telemetry must never break training, but it must not vanish
            # either — a bare `pass` here is what let the AttributeError above
            # live for months (CLAUDE.md: no NEW silent swallows).
            log_swallowed_exception()

        return errors

    def observe_from_action(
        self,
        entity: Entity | str,
        modulator: str,
        affordance: str,
        params: dict[str, Any],
        actual: dict[str, float],
        sensor_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, float]:
        """Convenience: make key + observe in one call."""
        key = self.make_key(entity, modulator, affordance, params, sensor_ranges)
        return self.observe(key, actual)

    def record_llm_fallback(self) -> None:
        """Record that an LLM call was needed (no confident model)."""
        self._llm_fallbacks += 1

    # -- pruning ------------------------------------------------------------

    def prune_stale_models(self, max_age_s: float | None = None) -> int:
        """Remove models not observed within max_age_s.

        Returns the number of pruned models.
        """
        if max_age_s is None:
            max_age_s = self.config.prune_age_s
        cutoff = time.time() - max_age_s
        pruned = 0

        with self._global_lock:
            stale_keys = [key for key, model in self._models.items() if model.last_observed < cutoff]
            for key in stale_keys:
                del self._models[key]
                self._locks.pop(key, None)
                pruned += 1

        if pruned:
            log.info("Pruned %d stale Cerebellum models", pruned)
        return pruned

    # -- persistence --------------------------------------------------------

    def export_state(self) -> dict[str, Any]:
        """Serialize all models for persistence."""
        with self._global_lock:
            models: dict[str, Any] = {}
            for key, model in self._models.items():
                key_str = f"{key.entity_path}|{key.modulator}|{key.affordance}|{key.param_bucket}"
                models[key_str] = model.to_dict()

            return {
                "version": "1.1",
                "model_count": len(models),
                "total_predictions": self._total_predictions,
                "total_observations": self._total_observations,
                "llm_fallbacks": self._llm_fallbacks,
                "models": models,
                "programs": self.programs.export_state(),
            }

    def import_state(self, data: dict[str, Any]) -> None:
        """Deserialize models + programs from persistence."""
        version = data.get("version", "1.0")
        if version not in ("1.0", "1.1"):
            log.warning("Unknown Cerebellum state version: %s", version)

        with self._global_lock:
            self._models.clear()
            self._locks.clear()
            self._total_predictions = data.get("total_predictions", 0)
            self._total_observations = data.get("total_observations", 0)
            self._llm_fallbacks = data.get("llm_fallbacks", 0)

            for key_str, model_data in data.get("models", {}).items():
                parts = key_str.split("|", 3)
                if len(parts) != 4:
                    log.warning("Malformed model key: %s", key_str)
                    continue
                key = ModelKey(*parts)
                self._models[key] = ForwardModel.from_dict(model_data)

            # Import motor programs (v1.1+)
            programs_data = data.get("programs")
            if programs_data:
                self.programs.import_state(programs_data)

        log.info(
            "Imported %d Cerebellum models + %d motor programs",
            len(self._models),
            self.programs.stats()["program_count"],
        )

    def save(self, path: str | None = None) -> None:
        """Save state to JSON file."""
        path = path or self.config.persistence_path
        if path is None:
            return

        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = self.export_state()

        try:
            from maxim.utils.atomic_io import atomic_write_json
            from maxim.utils.format_version import with_format_version

            atomic_write_json(path, with_format_version(state))
        except ImportError:
            with open(path, "w") as f:
                json.dump(state, f, indent=2)

        log.debug("Saved Cerebellum state to %s (%d models)", path, len(self._models))

    def load(self, path: str | None = None) -> bool:
        """Load state from JSON file. Returns True if loaded."""
        path = path or self.config.persistence_path
        if path is None:
            return False

        from pathlib import Path

        if not Path(path).exists():
            return False

        with open(path) as f:
            data = json.load(f)

        from maxim.utils.format_version import check_format_version

        check_format_version(data, "cerebellum", log=log)
        self.import_state(data)
        return True

    # -- motor programs (Phase 1b) ------------------------------------------

    def observe_action_sequence(
        self,
        goal: str,
        steps: list[tuple[str, str, str, dict[str, Any]]],
        success: bool,
        duration_s: float,
    ) -> Any:
        """Observe an action sequence, crystallize if recurring.

        Delegates to ProgramRegistry.observe_sequence().
        Returns a MotorProgram if crystallized, None otherwise.
        """
        return self.programs.observe_sequence(goal, steps, success, duration_s)

    def find_programs_for_goal(self, goal: str) -> list[Any]:
        """Find motor programs matching a goal signature."""
        return self.programs.find_by_goal(goal)

    def find_programs_for_entity(self, entity_path: str) -> list[Any]:
        """Find motor programs involving a specific entity."""
        return self.programs.find_by_entity(entity_path)

    def find_programs_for_affordance(self, affordance: str) -> list[Any]:
        """Find motor programs that use a specific affordance."""
        return self.programs.find_by_affordance(affordance)

    def find_related_programs(self, query: str) -> list[Any]:
        """Find programs by goal, entity, or affordance (unified search)."""
        return self.programs.find_related(query)

    # -- engram formation/recall (Phase 1b) ---------------------------------

    def form_engram(
        self,
        program: Any,
        entity_states: dict[str, dict[str, Any]],
        outcome_valence: str,
        pain_step: int | None,
        hippocampus: Any,
        pain_intensity: float = 0.0,
        rpe_magnitude: float = 0.0,
    ) -> str | None:
        """Capture a contextual engram in Hippocampus linked to a motor program.

        Only fires on significant outcomes (pain > 0.3, RPE > 0.3, or
        novelty > 0.7). Routine successes don't need episodic context.

        Returns the memory_id if engram was created, None otherwise.
        """
        from maxim.embodiment.engrams import (
            compute_engram_salience,
            program_graph_node,
            should_form_engram,
            snapshot_entity_states,
        )

        novelty = 1.0 - program.confidence
        if not should_form_engram(
            outcome_valence,
            pain_intensity=pain_intensity,
            rpe_magnitude=rpe_magnitude,
            novelty=novelty,
            program_confidence=program.confidence,
        ):
            return None

        salience = compute_engram_salience(
            outcome_valence,
            program.confidence,
            pain_intensity,
            rpe_magnitude,
        )
        states_snapshot = snapshot_entity_states(entity_states)

        try:
            from maxim.memory.types import Action, Context, Outcome, Perception

            memory_id = hippocampus.capture(
                perception=Perception(
                    observations={
                        "motor_program": program.name,
                        "goal": program.goal_signature,
                        "entity_states": states_snapshot,
                        "pain_step": pain_step,
                        "program_confidence": program.confidence,
                    },
                    salience=salience,
                    novelty=novelty,
                ),
                context=Context(
                    active_goal=program.goal_signature,
                    active_mode="motor_execution",
                ),
                action=Action(
                    tool_name=f"motor_program:{program.name}",
                    tool_params={"steps": len(program.steps)},
                ),
                outcome=Outcome(
                    success=outcome_valence != "NEGATIVE",
                    result={"valence": outcome_valence, "pain_step": pain_step},
                ),
            )
        except (ImportError, Exception) as e:
            log.debug("Engram formation failed: %s", e)
            return None

        if memory_id:
            try:
                from maxim.agents.bus import EdgeType

                node = program_graph_node(program.name)
                hippocampus._graph.add_bidirectional(
                    memory_id,
                    node,
                    EdgeType.ASSOCIATES,
                    weight=salience,
                )
            except (ImportError, AttributeError, Exception) as e:
                log.debug("Engram graph edge failed: %s", e)

        return memory_id

    def query_engrams(
        self,
        program: Any,
        current_entity_states: dict[str, dict[str, float]],
        hippocampus: Any,
        sensor_ranges: dict[str, dict[str, tuple[float, float]]] | None = None,
    ) -> list[Any]:
        """Retrieve contextual engrams relevant to executing a program now.

        Returns a list of MotorEngram objects sorted by relevance.
        """
        from maxim.embodiment.engrams import EngramConfig, MotorEngram, program_graph_node
        from maxim.embodiment.motor import entity_state_similarity

        cfg = EngramConfig()
        node = program_graph_node(program.name)

        try:
            from maxim.agents.bus import EdgeType

            associated = hippocampus._graph.get_associated(
                node,
                edge_types={EdgeType.ASSOCIATES},
            )
        except (ImportError, AttributeError, Exception):
            return []

        engram_ids = [node_id for node_id, weight in associated if weight > 0.2]
        if not engram_ids:
            return []

        try:
            activated = hippocampus.recall_via_spreading_activation(
                seed_ids=engram_ids,
                max_depth=cfg.spreading_activation_depth,
                decay=cfg.spreading_activation_decay,
                threshold=cfg.spreading_activation_threshold,
            )
        except (AttributeError, Exception):
            return []

        engrams: list[MotorEngram] = []
        for memory, activation in activated:
            if not hasattr(memory, "action") or memory.action is None:
                continue
            tool_name = getattr(memory.action, "tool_name", "")
            if not tool_name.startswith("motor_program:"):
                continue

            obs = getattr(memory.perception, "observations", {})
            remembered_states = obs.get("entity_states", {})
            context_sim = entity_state_similarity(
                remembered_states,
                current_entity_states,
                sensor_ranges or {},
            )

            if context_sim < cfg.min_similarity:
                continue

            result = getattr(memory.outcome, "result", {})
            engrams.append(
                MotorEngram(
                    memory_id=memory.id,
                    program_name=program.name,
                    activation=activation,
                    context_similarity=context_sim,
                    outcome_valence=result.get("valence"),
                    pain_step=result.get("pain_step"),
                    entity_states=remembered_states,
                )
            )

        return sorted(engrams, key=lambda e: e.relevance, reverse=True)[: cfg.max_engrams_per_query]

    # -- cleanup ------------------------------------------------------------

    def cleanup_program(self, program_name: str, hippocampus: Any = None) -> None:
        """Remove a motor program and its orphan graph nodes.

        Deletes the synthetic anchor node and all associated edges
        from the hippocampal graph.
        """
        self.programs.remove(program_name)

        if hippocampus is not None:
            from maxim.embodiment.engrams import program_graph_node

            node = program_graph_node(program_name)
            try:
                graph = hippocampus._graph
                # Remove all edges from/to this node
                if hasattr(graph, "remove_node"):
                    graph.remove_node(node)
                else:
                    # Manual edge cleanup if remove_node doesn't exist
                    try:
                        from maxim.agents.bus import EdgeType

                        edges = graph.get_associated(node, edge_types={EdgeType.ASSOCIATES})
                        for target_id, _ in edges:
                            graph._edges.get(node, {}).pop(target_id, None)
                            graph._edges.get(target_id, {}).pop(node, None)
                    except Exception:
                        pass
            except (AttributeError, Exception) as e:
                log.debug("Graph cleanup failed for %s: %s", program_name, e)

    # -- stats --------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return runtime statistics."""
        with self._global_lock:
            confidences = [m.confidence for m in self._models.values()]
            program_stats = self.programs.stats()
            return {
                "model_count": len(self._models),
                "total_predictions": self._total_predictions,
                "total_observations": self._total_observations,
                "llm_fallbacks": self._llm_fallbacks,
                "llm_skip_rate": (self._total_predictions / max(1, self._total_predictions + self._llm_fallbacks)),
                "avg_confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
                "max_confidence": max(confidences) if confidences else 0.0,
                "programs": program_stats["program_count"],
                "pending_sequences": program_stats["pending_observations"],
            }

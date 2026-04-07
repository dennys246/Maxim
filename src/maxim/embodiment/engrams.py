"""Motor engrams — ephemeral Cerebellum ↔ Hippocampus cross-system traces.

Engrams are short-lived episodic memories that link motor programs to
situational context.  They form on significant outcomes (pain, surprise,
novelty) and decay after ~2 days unless reinforced.

The Cerebellum stores the **how** (motor program steps).
The Hippocampus stores the **when/where/what** (contextual episode).
The engram links them via the associative graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngramConfig:
    """Configuration for motor engram formation and recall."""

    # Formation thresholds — engram only forms on significant outcomes
    pain_threshold: float = 0.3
    rpe_threshold: float = 0.3
    novelty_threshold: float = 0.7

    # Recall
    min_similarity: float = 0.3
    max_engrams_per_query: int = 5
    spreading_activation_depth: int = 2
    spreading_activation_decay: float = 0.5
    spreading_activation_threshold: float = 0.1

    # Gate modulation
    gate_tightening_factor: float = 0.1  # 10% gate reduction per negative engram


# ---------------------------------------------------------------------------
# Motor engram
# ---------------------------------------------------------------------------


@dataclass
class MotorEngram:
    """A retrieved engram linking a motor program to contextual memory."""

    memory_id: str
    program_name: str
    activation: float  # spreading activation score
    context_similarity: float  # scalar sensor similarity to current state
    outcome_valence: str | None = None  # "POSITIVE" / "NEGATIVE" / "NEUTRAL"
    pain_step: int | None = None  # which step caused pain, if any
    entity_states: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def relevance(self) -> float:
        """Combined relevance score for ranking."""
        return self.activation * self.context_similarity


# ---------------------------------------------------------------------------
# Engram formation
# ---------------------------------------------------------------------------


def compute_engram_salience(
    outcome_valence: str,
    program_confidence: float,
    pain_intensity: float = 0.0,
    rpe_magnitude: float = 0.0,
) -> float:
    """Compute salience for a motor engram.

    High salience for:
    - Painful outcomes (pain > 0.3)
    - Surprising outcomes (RPE > 0.3)
    - Low-confidence programs (novel/unproven)
    - Negative valence (failures are more memorable)
    """
    salience = 0.0

    # Pain contribution
    if pain_intensity > 0:
        salience = max(salience, pain_intensity)

    # RPE contribution
    if rpe_magnitude > 0:
        salience = max(salience, rpe_magnitude)

    # Novelty from low confidence
    novelty_from_confidence = 1.0 - program_confidence
    salience = max(salience, novelty_from_confidence * 0.5)

    # Negative outcomes are more salient
    if outcome_valence == "NEGATIVE":
        salience = min(1.0, salience * 1.3)

    return min(1.0, salience)


def should_form_engram(
    outcome_valence: str,
    pain_intensity: float = 0.0,
    rpe_magnitude: float = 0.0,
    novelty: float = 0.0,
    program_confidence: float = 0.0,
    config: EngramConfig | None = None,
) -> bool:
    """Determine if an engram should form for this motor outcome.

    Returns True only for significant outcomes:
    - Pain > threshold during execution
    - RPE > threshold (surprising outcome)
    - High novelty (first few executions of new program)

    Routine successes on confident programs → no engram (Cerebellum R-W handles it).
    """
    cfg = config or EngramConfig()

    if pain_intensity > cfg.pain_threshold:
        return True
    if rpe_magnitude > cfg.rpe_threshold:
        return True
    if novelty > cfg.novelty_threshold:
        return True
    # Low-confidence programs always get engrams (learning phase)
    if program_confidence < 0.3:
        return True
    return False


def snapshot_entity_states(
    entity_readings: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Snapshot scalar entity states for engram storage.

    Filters to scalar values only (floats/ints). Frames, audio,
    and other non-scalar readings are excluded.
    """
    result: dict[str, dict[str, float]] = {}
    for entity_path, sensors in entity_readings.items():
        scalars: dict[str, float] = {}
        for sensor_name, value in sensors.items():
            if isinstance(value, (int, float)):
                scalars[sensor_name] = float(value)
            elif hasattr(value, "value") and isinstance(value.value, (int, float)):
                # SensorReading object
                scalars[sensor_name] = float(value.value)
        if scalars:
            result[entity_path] = scalars
    return result


# ---------------------------------------------------------------------------
# Engram graph node naming
# ---------------------------------------------------------------------------


def program_graph_node(program_name: str) -> str:
    """Generate the synthetic graph node ID for a motor program.

    These nodes live in the hippocampal associative graph as anchors
    for engram edges.  They don't correspond to real EpisodicMemory
    records — they're bridge points between Cerebellum and Hippocampus.
    """
    return f"cerebellum:program:{program_name}"


def is_motor_engram_tool(tool_name: str) -> bool:
    """Check if an action tool name indicates a motor program execution."""
    return tool_name.startswith("motor_program:")

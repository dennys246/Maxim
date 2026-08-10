"""Kuramoto-inspired coupled oscillator network for temporal rhythm learning.

The biological SCN contains ~20,000 coupled oscillators whose synchronization
produces circadian rhythms.  This module implements a minimal Kuramoto network
with 4 oscillators (circadian, weekly, monthly, annual) that learn temporal
coupling via Hebbian plasticity.

Core dynamics (Kuramoto model):
    dθ_i/dt = ω_i + (K/N) Σ_j W[i][j] * sin(θ_j - θ_i)

Hebbian learning rule:
    ΔW[i][j] = η * cos(θ_i - θ_j)

Brain mapping:
    The SCN (Suprachiasmatic Nucleus) coordinates circadian rhythms through
    a network of coupled oscillators.  Inter-oscillator coupling is learned
    from environmental cues (zeitgebers), allowing emergent rhythms like
    "Monday mornings" to form via circadian-weekly coupling.

Bio-mapping: MECHANISM (Kuramoto + Hebbian coupling). The documented equations
are the implemented ones: Euler-integrated ``dtheta_i = omega_i + (K/N) * sum_j
W[i][j] * sin(theta_j - theta_i)`` and coupling learning ``dW = eta *
cos(theta_i - theta_j)``. Scope honestly: 4 oscillators standing in for the
SCN's ~20k, fixed natural frequencies, Euler (not exact) integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

TWO_PI: float = 2.0 * math.pi

# Natural frequencies in cycles per day
_NATURAL_FREQUENCIES: list[float] = [
    1.0,  # circadian: 1 cycle / day
    1.0 / 7.0,  # weekly: 1 cycle / 7 days
    1.0 / 30.0,  # monthly: 1 cycle / 30 days
    1.0 / 365.25,  # annual: 1 cycle / 365.25 days
]

_OSC_NAMES: list[str] = ["circadian", "weekly", "monthly", "annual"]
_N_OSC: int = 4


@dataclass
class OscillatorConfig:
    """Configuration for the coupled oscillator network."""

    coupling_strength: float = 0.1
    """Global K in Kuramoto model — controls how strongly oscillators influence each other."""

    learning_rate: float = 0.01
    """Hebbian learning rate η for coupling weight updates."""

    weight_decay: float = 0.999
    """Per-observation decay applied to coupling weights (half-life ~693 observations)."""

    max_weight: float = 2.0
    """Maximum coupling weight (prevents runaway Hebbian strengthening)."""

    min_weight: float = -0.5
    """Minimum coupling weight (allows mild inhibitory coupling)."""

    dt: float = 1.0 / 24.0
    """Integration step size in day units (default: 1 hour = 1/24 day)."""

    integration_steps: int = 1
    """Number of Euler steps per observe() call."""

    max_event_phases: int = 50
    """Maximum phase observations stored per event signature (ring buffer)."""

    anticipatory_weight: float = 0.2
    """Credit weight for anticipatory pre-activation (0-1)."""


class OscillatorNetwork:
    """Kuramoto coupled oscillator network with Hebbian learning.

    Four oscillators represent temporal scales: circadian (daily), weekly,
    monthly, and annual.  The coupling matrix learns which scales co-activate
    from observed temporal signatures.
    """

    NAMES: ClassVar[list[str]] = _OSC_NAMES

    def __init__(self, config: OscillatorConfig | None = None) -> None:
        self.config = config or OscillatorConfig()
        self._phases: list[float] = [0.0] * _N_OSC
        self._coupling: list[list[float]] = [[0.0] * _N_OSC for _ in range(_N_OSC)]
        self._observation_count: int = 0
        self._prediction_error_ema: float = 0.0

        # Per-event-type phase tracking for anticipatory credit.
        # Maps event_signature → list of circadian phases (radians) at
        # which that event type was observed.  Ring buffer capped at
        # config.max_event_phases.
        self._event_phases: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Phase dynamics
    # ------------------------------------------------------------------

    def step(self, dt: float | None = None) -> None:
        """Advance oscillator phases by one Euler integration step.

        dθ_i/dt = ω_i + (K/N) Σ_j W[i][j] * sin(θ_j - θ_i)
        """
        dt = dt if dt is not None else self.config.dt
        k = self.config.coupling_strength
        new_phases = list(self._phases)

        for i in range(_N_OSC):
            # Natural frequency contribution
            d_theta = _NATURAL_FREQUENCIES[i] * TWO_PI  # radians per day

            # Coupling contribution
            coupling_sum = 0.0
            for j in range(_N_OSC):
                if i == j:
                    continue
                coupling_sum += self._coupling[i][j] * math.sin(self._phases[j] - self._phases[i])
            d_theta += (k / _N_OSC) * coupling_sum

            new_phases[i] = (self._phases[i] + d_theta * dt) % TWO_PI

        self._phases = new_phases

    def observe(self, signature: Any) -> float:
        """Observe a temporal signature and update internal state.

        Steps:
        1. Compute prediction error (distance between predicted and observed phases)
        2. Soft phase reset toward observed phases (blend depends on observation count)
        3. Hebbian coupling update (co-active oscillators strengthen coupling)
        4. Weight decay (prevents unbounded growth)

        Args:
            signature: TemporalSignature with circadian_phase, weekly_phase,
                      monthly_phase, annual_phase fields (each in [0, 1)).

        Returns:
            Prediction error (mean circular distance across oscillators).
        """
        # Extract observed phases from TemporalSignature (convert [0,1) → [0, 2π))
        observed = [
            getattr(signature, "circadian_phase", 0.0) * TWO_PI,
            getattr(signature, "weekly_phase", 0.0) * TWO_PI,
            getattr(signature, "monthly_phase", 0.0) * TWO_PI,
            getattr(signature, "annual_phase", 0.0) * TWO_PI,
        ]

        # 1. Prediction error
        errors = [_circular_distance(self._phases[i], observed[i]) for i in range(_N_OSC)]
        prediction_error = sum(errors) / _N_OSC

        # Update EMA of prediction error
        alpha = 0.1
        self._prediction_error_ema = (1 - alpha) * self._prediction_error_ema + alpha * prediction_error

        # 2. Soft phase reset: blend = max(0.1, 1 / (1 + 0.01 * count))
        blend = max(0.1, 1.0 / (1.0 + 0.01 * self._observation_count))
        for i in range(_N_OSC):
            # Interpolate on the circle (shortest arc)
            diff = observed[i] - self._phases[i]
            # Normalize to [-π, π]
            diff = (diff + math.pi) % TWO_PI - math.pi
            self._phases[i] = (self._phases[i] + blend * diff) % TWO_PI

        # 3. Hebbian coupling update: ΔW[i][j] = η * cos(θ_i - θ_j)
        eta = self.config.learning_rate
        for i in range(_N_OSC):
            for j in range(i + 1, _N_OSC):
                delta = eta * math.cos(self._phases[i] - self._phases[j])
                self._coupling[i][j] += delta
                self._coupling[j][i] += delta  # Symmetric

        # 4. Weight decay + clamping
        decay = self.config.weight_decay
        for i in range(_N_OSC):
            for j in range(_N_OSC):
                if i == j:
                    continue
                self._coupling[i][j] *= decay
                self._coupling[i][j] = max(
                    self.config.min_weight,
                    min(self.config.max_weight, self._coupling[i][j]),
                )

        # 5. Advance phases via dynamics
        for _ in range(self.config.integration_steps):
            self.step()

        self._observation_count += 1
        return prediction_error

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_phase(self, hours_ahead: float) -> list[float]:
        """Predict oscillator phases N hours ahead without mutating state.

        Returns phases in [0, 1) for each oscillator.
        """
        dt = self.config.dt
        steps = max(1, int(hours_ahead / (dt * 24)))

        # Work on copies
        phases = list(self._phases)
        k = self.config.coupling_strength

        for _ in range(steps):
            new_phases = list(phases)
            for i in range(_N_OSC):
                d_theta = _NATURAL_FREQUENCIES[i] * TWO_PI
                coupling_sum = 0.0
                for j in range(_N_OSC):
                    if i == j:
                        continue
                    coupling_sum += self._coupling[i][j] * math.sin(phases[j] - phases[i])
                d_theta += (k / _N_OSC) * coupling_sum
                new_phases[i] = (phases[i] + d_theta * dt) % TWO_PI
            phases = new_phases

        # Convert to [0, 1)
        return [p / TWO_PI for p in phases]

    def predict_next_occurrence(
        self,
        target_phase: float,
        oscillator: int = 0,
        max_hours: float = 168.0,
    ) -> float | None:
        """When will oscillator reach target_phase? Returns hours, or None if not found.

        Args:
            target_phase: Target phase in [0, 1).
            oscillator: Oscillator index (0=circadian, 1=weekly, ...).
            max_hours: Maximum lookahead in hours.
        """
        target_rad = target_phase * TWO_PI
        dt = self.config.dt
        steps = int(max_hours / (dt * 24))

        phases = list(self._phases)
        k = self.config.coupling_strength
        prev_dist = _circular_distance(phases[oscillator], target_rad)

        for step_num in range(1, steps + 1):
            new_phases = list(phases)
            for i in range(_N_OSC):
                d_theta = _NATURAL_FREQUENCIES[i] * TWO_PI
                coupling_sum = 0.0
                for j in range(_N_OSC):
                    if i == j:
                        continue
                    coupling_sum += self._coupling[i][j] * math.sin(phases[j] - phases[i])
                d_theta += (k / _N_OSC) * coupling_sum
                new_phases[i] = (phases[i] + d_theta * dt) % TWO_PI
            phases = new_phases

            curr_dist = _circular_distance(phases[oscillator], target_rad)
            # Crossed through target if distance increased after being small
            if prev_dist < 0.1 and curr_dist > prev_dist:
                return step_num * dt * 24  # Convert to hours
            prev_dist = curr_dist

        return None

    # ------------------------------------------------------------------
    # Event-type phase tracking (anticipatory credit feedback)
    # ------------------------------------------------------------------

    def observe_event(self, event_signature: str, signature: Any) -> None:
        """Record the circadian phase at which an event type occurred.

        Builds a per-event-type phase history so the oscillator can
        predict when specific event types are likely to fire.

        Args:
            event_signature: Hashable event identifier (e.g. "tool:sword_slash").
            signature: TemporalSignature with circadian_phase in [0, 1).
        """
        phase = getattr(signature, "circadian_phase", 0.0) * TWO_PI
        phases = self._event_phases.get(event_signature)
        if phases is None:
            phases = []
            self._event_phases[event_signature] = phases

        phases.append(phase)
        # Ring buffer — drop oldest when over cap
        cap = self.config.max_event_phases
        if len(phases) > cap:
            del phases[: len(phases) - cap]

    def predict_event_imminence(self, event_signature: str) -> float:
        """Predict how imminent an event type is based on learned phase patterns.

        Compares the oscillator's current circadian phase to the mean
        circadian phase of past observations for this event type.

        Returns:
            Imminence score in [0, 1].  1.0 = current phase matches the
            event's typical phase exactly.  0.0 = maximally far away or
            no observations.  Requires >= 3 observations to produce a
            non-zero score (cold-start guard).
        """
        phases = self._event_phases.get(event_signature)
        if not phases or len(phases) < 3:
            return 0.0

        # Circular mean of observed phases
        sin_sum = sum(math.sin(p) for p in phases)
        cos_sum = sum(math.cos(p) for p in phases)
        mean_phase = math.atan2(sin_sum, cos_sum) % TWO_PI

        # Circular concentration (R / N) — high R means consistent timing
        n = len(phases)
        r = math.sqrt(sin_sum * sin_sum + cos_sum * cos_sum) / n

        # Distance from current circadian phase to mean event phase
        current_phase = self._phases[0]  # oscillator 0 = circadian
        dist = _circular_distance(current_phase, mean_phase)
        # Normalize: max dist is π → similarity in [0, 1]
        phase_similarity = 1.0 - (dist / math.pi)

        # Scale by concentration — scattered events get low imminence
        return phase_similarity * r

    def get_anticipatory_signatures(self, min_imminence: float = 0.5) -> dict[str, float]:
        """Return event signatures with imminence above threshold.

        Args:
            min_imminence: Minimum imminence score to include (0-1).

        Returns:
            ``{event_signature: imminence_score}`` for all events above
            the threshold.
        """
        result: dict[str, float] = {}
        for sig in self._event_phases:
            score = self.predict_event_imminence(sig)
            if score >= min_imminence:
                result[sig] = score
        return result

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def phase_coherence(self) -> float:
        """Kuramoto order parameter r = |1/N Σ exp(iθ_k)| ∈ [0, 1].

        r = 1.0: all oscillators synchronized
        r ≈ 0.0: phases uniformly spread (desynchronized)
        """
        real = sum(math.cos(p) for p in self._phases) / _N_OSC
        imag = sum(math.sin(p) for p in self._phases) / _N_OSC
        return math.sqrt(real * real + imag * imag)

    def coupling_eigenvalues(self) -> list[float]:
        """Eigenvalues of the coupling matrix (dominant rhythm analysis)."""
        from maxim.math.linalg import eigenvalues_symmetric

        return eigenvalues_symmetric(self._coupling)

    def temporal_anomaly_score(self, signature: Any) -> float:
        """Score how anomalous a temporal signature is relative to predicted phases.

        Returns 0.0 (perfectly predicted) to 1.0 (maximally anomalous).
        """
        observed = [
            getattr(signature, "circadian_phase", 0.0) * TWO_PI,
            getattr(signature, "weekly_phase", 0.0) * TWO_PI,
            getattr(signature, "monthly_phase", 0.0) * TWO_PI,
            getattr(signature, "annual_phase", 0.0) * TWO_PI,
        ]

        errors = [_circular_distance(self._phases[i], observed[i]) for i in range(_N_OSC)]
        # Normalize: max circular distance is π, so mean error / π gives [0, 1]
        return min(1.0, sum(errors) / (_N_OSC * math.pi))

    def coupling_strength_between(self, osc_a: int, osc_b: int) -> float:
        """Read coupling weight between two oscillators."""
        if 0 <= osc_a < _N_OSC and 0 <= osc_b < _N_OSC:
            return self._coupling[osc_a][osc_b]
        return 0.0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize oscillator state for persistence."""
        return {
            "phases": list(self._phases),
            "coupling": [row[:] for row in self._coupling],
            "observation_count": self._observation_count,
            "prediction_error_ema": self._prediction_error_ema,
            "event_phases": {k: list(v) for k, v in self._event_phases.items()},
            "config": {
                "coupling_strength": self.config.coupling_strength,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "max_weight": self.config.max_weight,
                "min_weight": self.config.min_weight,
                "dt": self.config.dt,
                "integration_steps": self.config.integration_steps,
                "max_event_phases": self.config.max_event_phases,
                "anticipatory_weight": self.config.anticipatory_weight,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OscillatorNetwork:
        """Restore oscillator from persisted state."""
        config_data = data.get("config", {})
        config = OscillatorConfig(
            coupling_strength=config_data.get("coupling_strength", 0.1),
            learning_rate=config_data.get("learning_rate", 0.01),
            weight_decay=config_data.get("weight_decay", 0.999),
            max_weight=config_data.get("max_weight", 2.0),
            min_weight=config_data.get("min_weight", -0.5),
            dt=config_data.get("dt", 1.0 / 24.0),
            integration_steps=config_data.get("integration_steps", 1),
            max_event_phases=config_data.get("max_event_phases", 50),
            anticipatory_weight=config_data.get("anticipatory_weight", 0.2),
        )
        network = cls(config)
        network._phases = data.get("phases", [0.0] * _N_OSC)
        coupling = data.get("coupling")
        if coupling and len(coupling) == _N_OSC:
            network._coupling = [row[:] for row in coupling]
        network._observation_count = data.get("observation_count", 0)
        network._prediction_error_ema = data.get("prediction_error_ema", 0.0)
        # Restore per-event-type phase history (backward-compatible default)
        event_phases = data.get("event_phases", {})
        network._event_phases = {k: list(v) for k, v in event_phases.items()}
        return network


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _circular_distance(a: float, b: float) -> float:
    """Circular distance between two angles in radians. Returns [0, π]."""
    diff = abs(a - b) % TWO_PI
    return diff if diff <= math.pi else TWO_PI - diff


__all__ = [
    "OscillatorConfig",
    "OscillatorNetwork",
]

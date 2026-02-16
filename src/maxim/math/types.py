"""Core mathematical types for the Posterior Parietal Lobe module.

Provides enums, result dataclasses, and value containers used by both
the IPS (Approximate Number System) and Angular Gyrus (Exact Computation).

Brain mapping:
  - NumberSystem.APPROXIMATE → IPS / Intraparietal Sulcus (ANS)
  - NumberSystem.EXACT → Angular Gyrus (posterior parietal lobe)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto


class NumberSystem(Enum):
    """Which number system processed this result."""

    APPROXIMATE = auto()  # IPS / ANS
    EXACT = auto()  # Angular Gyrus


class MagnitudeCategory(Enum):
    """Logarithmic magnitude bins (Weber-Fechner scale).

    Boundaries are logarithmically spaced so that the *relative*
    difference between categories is constant — matching the
    psychophysical properties of the human ANS.
    """

    NEGLIGIBLE = auto()  # |x| < 0.01
    TINY = auto()  # 0.01 ≤ |x| < 0.1
    SMALL = auto()  # 0.1 ≤ |x| < 1
    MODERATE = auto()  # 1 ≤ |x| < 100
    LARGE = auto()  # 100 ≤ |x| < 10_000
    HUGE = auto()  # |x| ≥ 10_000


class ComparisonResult(Enum):
    """Approximate magnitude comparison (ANS output)."""

    MUCH_LESS = auto()
    LESS = auto()
    SIMILAR = auto()  # Within Weber fraction — indistinguishable
    GREATER = auto()
    MUCH_GREATER = auto()


class TrendDirection(Enum):
    """Direction of a detected numerical trend."""

    INCREASING = auto()
    DECREASING = auto()
    STABLE = auto()
    OSCILLATING = auto()
    INSUFFICIENT_DATA = auto()


class MathCategory(Enum):
    """Categories for math memory records stored in Angular Gyrus."""

    FACT = auto()  # "pi = 3.14159", "2+2=4"
    METHOD = auto()  # "linear_regression", "mean"
    PATTERN = auto()  # "sensor X tends to increase before failure"
    CONSTANT = auto()  # Named constants
    RELATIONSHIP = auto()  # "temperature IS_PROPORTIONAL_TO energy"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApproximateResult:
    """Result from IPS (ANS) — fast but imprecise."""

    value: float
    magnitude: MagnitudeCategory
    confidence: float  # 0.0-1.0 (lower near discrimination threshold)
    system: NumberSystem = NumberSystem.APPROXIMATE


@dataclass(frozen=True)
class ExactResult:
    """Result from Angular Gyrus — precise computation."""

    value: float | int | list[float] | list[list[float]]
    operation: str
    system: NumberSystem = NumberSystem.EXACT
    verbal: str = ""  # Natural language: "The sum is 42"
    code: str = ""  # Code form: "sum([17, 25])"


@dataclass(frozen=True)
class TrendResult:
    """Trend detection output (IPS)."""

    direction: TrendDirection
    slope_estimate: float  # Approximate rate of change
    confidence: float
    magnitude: MagnitudeCategory  # Magnitude of the change
    system: NumberSystem = NumberSystem.APPROXIMATE


@dataclass(frozen=True)
class AnomalyResult:
    """Anomaly detection output (IPS)."""

    is_anomalous: bool
    deviation: float  # How many "MAD-scaled deviations" from median
    magnitude: MagnitudeCategory
    direction: str  # "above" or "below" expected
    confidence: float


class PatternType(Enum):
    """Classification of sequence structure (IPS randomness assessment)."""

    RANDOM = auto()  # No detectable structure
    TRENDING = auto()  # Monotonic drift (up or down)
    CYCLIC = auto()  # Periodic oscillation
    CLUSTERING = auto()  # Runs of similar values (regime switching)
    INSUFFICIENT_DATA = auto()


@dataclass(frozen=True)
class RandomnessResult:
    """Randomness assessment output (IPS).

    Answers: "Is this sequence random noise, or is a pattern forming?"
    Uses Wald-Wolfowitz runs test + lag-1 autocorrelation.
    """

    is_random: bool  # True if sequence appears structureless
    pattern_confidence: float  # 0 = definitely random, 1 = definitely patterned
    pattern_type: PatternType  # Classification if not random
    runs_z_score: float  # Standardized runs test statistic
    autocorrelation: float  # Lag-1 autocorrelation
    system: NumberSystem = NumberSystem.APPROXIMATE


@dataclass(frozen=True)
class AnalysisResult:
    """Statistical analysis output (Angular Gyrus)."""

    method: str  # "descriptive", "linear", "quadratic", etc.
    parameters: dict[str, float]
    verbal: str
    code: str
    confidence: float  # Goodness of fit (R², p-value proxy, etc.)
    system: NumberSystem = NumberSystem.EXACT


# ---------------------------------------------------------------------------
# Workspace value container
# ---------------------------------------------------------------------------


@dataclass
class NumericalValue:
    """A named value held in the NumericalWorkspace."""

    name: str
    value: float | list[float]
    magnitude: MagnitudeCategory  # Pre-computed by IPS on store
    source: str = ""  # "computation", "sensor", "user", "memory"
    timestamp: float = field(default_factory=time.time)

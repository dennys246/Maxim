"""Intraparietal Sulcus (IPS) — Approximate Number System (ANS).

Fast, approximate numerical processing inspired by the brain's innate
number sense.  Uses Weber-Fechner scaling (logarithmic compression) so
that discrimination is proportional to magnitude — just like the
biological ANS.

All methods return results with confidence values reflecting the
inherent imprecision of the approximate system.  The IPS is stateless
and thread-safe; it can be shared across agents without locking.

Brain mapping:
  IPS sits within the posterior parietal lobe and is the primary
  substrate for the Approximate Number System (ANS).  It handles
  magnitude comparison, estimation, and coarse trend detection —
  the "numerical intuition" pathway.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from maxim.math.types import (
    AnomalyResult,
    ApproximateResult,
    ComparisonResult,
    MagnitudeCategory,
    PatternType,
    RandomnessResult,
    TrendDirection,
    TrendResult,
)


@dataclass
class IPSConfig:
    """Configuration for the Approximate Number System."""

    weber_fraction: float = 0.15
    """Discrimination threshold (Weber fraction).

    Two magnitudes are indistinguishable when
    ``|a - b| / max(|a|, |b|) < weber_fraction``.
    Human adult baseline is ~0.15; trained individuals reach ~0.10.
    """

    trend_window: int = 10
    """Number of recent values used for trend detection."""

    anomaly_z_threshold: float = 2.0
    """MAD-scaled deviation threshold for anomaly flagging."""

    log_base: float = 10.0
    """Base for logarithmic magnitude binning."""


# Magnitude bin boundaries (log10 scale)
_MAG_BOUNDARIES: list[tuple[float, MagnitudeCategory]] = [
    (-2.0, MagnitudeCategory.NEGLIGIBLE),
    (-1.0, MagnitudeCategory.TINY),
    (0.0, MagnitudeCategory.SMALL),
    (2.0, MagnitudeCategory.MODERATE),
    (4.0, MagnitudeCategory.LARGE),
]


class IPS:
    """Intraparietal Sulcus — Approximate Number System (ANS).

    Provides fast, approximate numerical operations:

    * **compare** — Weber-Fechner magnitude comparison
    * **categorize** — logarithmic magnitude binning
    * **estimate_sum / estimate_mean** — approximate aggregation
    * **detect_trend** — sliding-window trend classification
    * **detect_anomaly** — robust outlier detection (median + MAD)
    * **rank** — approximate ordering
    * **proportion** — approximate part-of-whole ratio

    All operations are O(1) or O(n) with small constants — designed
    for speed, not precision.
    """

    def __init__(self, config: IPSConfig | None = None) -> None:
        self.config = config or IPSConfig()

    # ------------------------------------------------------------------
    # Magnitude comparison
    # ------------------------------------------------------------------

    def compare(self, a: float, b: float) -> ComparisonResult:
        """Weber-Fechner magnitude comparison.

        Returns ``SIMILAR`` when the ratio difference is within the
        Weber fraction.  ``MUCH_GREATER`` / ``MUCH_LESS`` when the
        ratio exceeds 2× the Weber fraction.
        """
        if a == b:
            return ComparisonResult.SIMILAR

        denom = max(abs(a), abs(b))
        if denom == 0:
            return ComparisonResult.SIMILAR

        ratio = abs(a - b) / denom
        wf = self.config.weber_fraction

        if ratio < wf:
            return ComparisonResult.SIMILAR
        elif a > b:
            return ComparisonResult.MUCH_GREATER if ratio >= 2 * wf else ComparisonResult.GREATER
        else:
            return ComparisonResult.MUCH_LESS if ratio >= 2 * wf else ComparisonResult.LESS

    def compare_with_confidence(self, a: float, b: float) -> tuple[ComparisonResult, float]:
        """Compare with an additional confidence score (0-1).

        Confidence is highest when values are far apart relative to
        the Weber fraction and lowest near the discrimination boundary.
        """
        if a == b:
            return ComparisonResult.SIMILAR, 1.0

        denom = max(abs(a), abs(b))
        if denom == 0:
            return ComparisonResult.SIMILAR, 1.0

        ratio = abs(a - b) / denom
        wf = self.config.weber_fraction

        # Confidence: 0 at the boundary, 1 when ratio >> wf or ratio << wf
        if ratio < wf:
            confidence = 1.0 - (ratio / wf)
        else:
            confidence = min(1.0, (ratio - wf) / wf)

        result = self.compare(a, b)
        return result, confidence

    # ------------------------------------------------------------------
    # Magnitude binning
    # ------------------------------------------------------------------

    def categorize(self, value: float) -> MagnitudeCategory:
        """Logarithmic magnitude binning.

        Maps ``|value|`` to a category via ``log10``:
          ``< -2`` → NEGLIGIBLE, ``-2..-1`` → TINY, ``-1..0`` → SMALL,
          ``0..2`` → MODERATE, ``2..4`` → LARGE, ``≥ 4`` → HUGE.
        """
        if value == 0:
            return MagnitudeCategory.NEGLIGIBLE

        log_val = math.log10(abs(value))

        for boundary, category in _MAG_BOUNDARIES:
            if log_val < boundary:
                return category
        return MagnitudeCategory.HUGE

    # ------------------------------------------------------------------
    # Approximate aggregation
    # ------------------------------------------------------------------

    def estimate_sum(self, values: list[float]) -> ApproximateResult:
        """Fast approximate sum.

        Confidence decreases with the number of operands (Weber noise
        accumulates with each addition).
        """
        if not values:
            return ApproximateResult(
                value=0.0,
                magnitude=MagnitudeCategory.NEGLIGIBLE,
                confidence=1.0,
            )

        total = math.fsum(values)
        n = len(values)
        # Confidence degrades with operand count (noise accumulation)
        confidence = max(0.1, 1.0 - self.config.weber_fraction * math.log2(max(n, 1)))

        return ApproximateResult(
            value=total,
            magnitude=self.categorize(total),
            confidence=min(1.0, confidence),
        )

    def estimate_mean(self, values: list[float]) -> ApproximateResult:
        """Fast approximate mean."""
        if not values:
            return ApproximateResult(
                value=0.0,
                magnitude=MagnitudeCategory.NEGLIGIBLE,
                confidence=1.0,
            )

        mean = math.fsum(values) / len(values)
        # Mean is more stable than sum — confidence degrades slower
        n = len(values)
        confidence = max(0.2, 1.0 - self.config.weber_fraction * math.log2(max(n, 1)) * 0.5)

        return ApproximateResult(
            value=mean,
            magnitude=self.categorize(mean),
            confidence=min(1.0, confidence),
        )

    # ------------------------------------------------------------------
    # Trend detection
    # ------------------------------------------------------------------

    def detect_trend(self, values: list[float]) -> TrendResult:
        """Sliding-window trend detection.

        Uses the last ``config.trend_window`` values.  Classifies based
        on the proportion of positive vs negative successive differences:

        * >70% positive → INCREASING
        * >70% negative → DECREASING
        * Alternating signs (>60% sign changes) → OSCILLATING
        * Otherwise → STABLE

        Slope is estimated as the median of pairwise differences.
        """
        window = self.config.trend_window
        recent = values[-window:] if len(values) > window else values

        if len(recent) < 3:
            return TrendResult(
                direction=TrendDirection.INSUFFICIENT_DATA,
                slope_estimate=0.0,
                confidence=0.0,
                magnitude=MagnitudeCategory.NEGLIGIBLE,
            )

        diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        n_diffs = len(diffs)

        positive = sum(1 for d in diffs if d > 0)
        negative = sum(1 for d in diffs if d < 0)
        sign_changes = sum(
            1
            for i in range(len(diffs) - 1)
            if (diffs[i] > 0) != (diffs[i + 1] > 0) and diffs[i] != 0 and diffs[i + 1] != 0
        )

        pos_ratio = positive / n_diffs
        neg_ratio = negative / n_diffs
        change_ratio = sign_changes / max(n_diffs - 1, 1)

        # Classify direction
        if pos_ratio >= 0.7:
            direction = TrendDirection.INCREASING
            confidence = pos_ratio
        elif neg_ratio >= 0.7:
            direction = TrendDirection.DECREASING
            confidence = neg_ratio
        elif change_ratio >= 0.6:
            direction = TrendDirection.OSCILLATING
            confidence = change_ratio
        else:
            direction = TrendDirection.STABLE
            confidence = 1.0 - max(pos_ratio, neg_ratio)

        # Slope: median of pairwise differences (robust)
        slope = statistics.median(diffs)

        return TrendResult(
            direction=direction,
            slope_estimate=slope,
            confidence=min(1.0, confidence),
            magnitude=self.categorize(slope) if slope != 0 else MagnitudeCategory.NEGLIGIBLE,
        )

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def detect_anomaly(self, value: float, history: list[float]) -> AnomalyResult:
        """Robust outlier detection using median and MAD.

        Uses Median Absolute Deviation rather than mean/std to avoid
        sensitivity to existing outliers in history.
        """
        if len(history) < 3:
            return AnomalyResult(
                is_anomalous=False,
                deviation=0.0,
                magnitude=MagnitudeCategory.NEGLIGIBLE,
                direction="within",
                confidence=0.0,
            )

        med = statistics.median(history)
        abs_devs = [abs(x - med) for x in history]
        mad = statistics.median(abs_devs)

        # Avoid division by zero; if MAD is 0, all values are the same
        if mad == 0:
            is_anomalous = value != med
            deviation = float("inf") if is_anomalous else 0.0
            return AnomalyResult(
                is_anomalous=is_anomalous,
                deviation=deviation,
                magnitude=self.categorize(abs(value - med)) if is_anomalous else MagnitudeCategory.NEGLIGIBLE,
                direction="above" if value > med else "below" if value < med else "within",
                confidence=1.0 if is_anomalous else 1.0,
            )

        # Scale MAD to approximate standard deviation (for normal distributions)
        scaled_mad = mad * 1.4826
        deviation = abs(value - med) / scaled_mad

        is_anomalous = deviation >= self.config.anomaly_z_threshold
        direction = "above" if value > med else "below" if value < med else "within"

        # Confidence: higher when deviation is far beyond threshold
        if is_anomalous:
            confidence = min(1.0, deviation / (self.config.anomaly_z_threshold * 2))
        else:
            confidence = min(1.0, 1.0 - deviation / self.config.anomaly_z_threshold)

        return AnomalyResult(
            is_anomalous=is_anomalous,
            deviation=deviation,
            magnitude=self.categorize(abs(value - med)),
            direction=direction,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Randomness assessment
    # ------------------------------------------------------------------

    def assess_randomness(self, values: list[float]) -> RandomnessResult:
        """Assess whether a sequence is random or contains structure.

        Uses two complementary non-parametric tests:

        1. **Wald-Wolfowitz runs test** — counts consecutive runs of values
           above/below the median.  Random sequences have a predictable
           number of runs; too few → trending, too many → cyclic.

        2. **Lag-1 autocorrelation** — measures serial dependence between
           consecutive values.  High |autocorr| → values are correlated.

        Combined into a ``pattern_confidence`` score:
        0.0 = definitely random, 1.0 = definitely patterned.

        Args:
            values: Sequence of numerical observations (minimum 8 for
                    meaningful assessment).

        Returns:
            RandomnessResult with classification and test statistics.
        """
        if len(values) < 8:
            return RandomnessResult(
                is_random=True,
                pattern_confidence=0.0,
                pattern_type=PatternType.INSUFFICIENT_DATA,
                runs_z_score=0.0,
                autocorrelation=0.0,
            )

        median = statistics.median(values)

        # 1. Wald-Wolfowitz runs test
        # Convert to binary: above(1) / below(0) median, exclude ties
        binary = [1 if v > median else 0 for v in values if v != median]
        n = len(binary)

        if n < 4:
            return RandomnessResult(
                is_random=True,
                pattern_confidence=0.0,
                pattern_type=PatternType.INSUFFICIENT_DATA,
                runs_z_score=0.0,
                autocorrelation=0.0,
            )

        n_pos = sum(binary)
        n_neg = n - n_pos

        # Count runs (consecutive sequences of same value)
        runs = 1 + sum(1 for i in range(1, n) if binary[i] != binary[i - 1])

        # Expected runs and variance under H0 (random)
        if n_pos == 0 or n_neg == 0:
            runs_z = 0.0
        else:
            expected = (2.0 * n_pos * n_neg) / n + 1.0
            numer = 2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n)
            denom = n * n * (n - 1)
            variance = numer / denom if denom > 0 else 1e-10
            std = math.sqrt(max(variance, 1e-10))
            runs_z = (runs - expected) / std

        # 2. Lag-1 autocorrelation
        mean = statistics.mean(values)
        var_sum = sum((v - mean) ** 2 for v in values)

        if var_sum < 1e-15:
            autocorr = 0.0
        else:
            cov_sum = sum(
                (values[i] - mean) * (values[i + 1] - mean)
                for i in range(len(values) - 1)
            )
            autocorr = cov_sum / var_sum

        # 3. Combine into pattern_confidence
        runs_signal = min(1.0, abs(runs_z) / 3.0)
        auto_signal = min(1.0, abs(autocorr) * 2.0)
        pattern_confidence = max(runs_signal, auto_signal)

        # 4. Classify pattern type
        is_random = pattern_confidence < 0.4

        if is_random:
            pattern_type = PatternType.RANDOM
        elif runs_z < -1.96:
            # Too few runs → values cluster on same side → trending
            pattern_type = PatternType.TRENDING
        elif runs_z > 1.96:
            # Too many runs → rapid alternation → cyclic
            pattern_type = PatternType.CYCLIC
        elif abs(autocorr) > 0.3:
            # Strong serial correlation → regime clustering
            pattern_type = PatternType.CLUSTERING
        else:
            pattern_type = PatternType.TRENDING

        return RandomnessResult(
            is_random=is_random,
            pattern_confidence=pattern_confidence,
            pattern_type=pattern_type,
            runs_z_score=runs_z,
            autocorrelation=autocorr,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def rank(self, values: list[float]) -> list[int]:
        """Approximate rank ordering.  Returns indices sorted by value (ascending)."""
        return [i for i, _ in sorted(enumerate(values), key=lambda x: x[1])]

    def proportion(self, part: float, whole: float) -> ApproximateResult:
        """What fraction is ``part`` of ``whole``?"""
        if whole == 0:
            return ApproximateResult(
                value=0.0,
                magnitude=MagnitudeCategory.NEGLIGIBLE,
                confidence=0.0,
            )

        ratio = part / whole
        # Confidence is higher when whole is large (less noise)
        confidence = min(1.0, max(0.1, 1.0 - self.config.weber_fraction / max(abs(whole), 1e-10)))

        return ApproximateResult(
            value=ratio,
            magnitude=self.categorize(ratio),
            confidence=confidence,
        )

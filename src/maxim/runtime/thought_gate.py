"""ThoughtGate: composite gate for Exec deliberation (Stage 2).

Composes SalienceScorer + energy check + refractory guard into a
single ``should_think()`` predicate.  Exec runs deliberation only
when this gate passes.

Composition order (short-circuit cascade):
  1. Refractory — don't re-fire within N ticks of last pass
  2. Energy — don't think below ``min_energy_fraction`` of budget
  3. Score — run SalienceScorer over the working-memory head
  4. Adaptive threshold — score.combined vs AdaptiveThresholdController

Uses existing SalienceScorer protocol (G0), AdaptiveThresholdController
(G0), TextSalienceScorer (G1), and LLMEnergyTracker (energy module).
Folds G3 from gating_abstraction.md.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

from maxim.runtime.gating import (
    AdaptiveThresholdController,
    AdaptiveThresholdConfig,
    GateDecision,
    GateScore,
    GatingContext,
    SalienceScorer,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ThoughtGateConfig:
    """Configuration for the ThoughtGate composite gate."""

    min_combined_score: float = 0.4  # floor before adaptive kicks in
    min_energy_fraction: float = 0.15  # don't think below 15% energy budget
    refractory_ticks: int = 2  # don't re-fire within N ticks
    default_energy_budget: int = 100_000  # token budget for energy check


class ThoughtGate:
    """Composite gate for Exec deliberation.

    Decides: should Exec deliberate on the current working memory
    contents?  Returns a ``GateDecision`` with score breakdown for
    provenance.

    Thread-safe via internal lock on refractory state.
    """

    def __init__(
        self,
        *,
        scorer: SalienceScorer | None = None,
        threshold_controller: AdaptiveThresholdController | None = None,
        energy_tracker: Any | None = None,
        config: ThoughtGateConfig | None = None,
    ) -> None:
        self._scorer = scorer
        self._threshold = threshold_controller or AdaptiveThresholdController(
            AdaptiveThresholdConfig(
                base_novelty_threshold=0.5,
                base_salience_threshold=0.4,
            )
        )
        self._energy_tracker = energy_tracker
        self._config = config or ThoughtGateConfig()
        self._last_pass_tick = -self._config.refractory_ticks - 1
        self._lock = threading.Lock()

    def should_think(
        self,
        *,
        working_memory: Any,
        context: GatingContext | None = None,
        current_tick: int = 0,
    ) -> GateDecision:
        """Decide whether Exec should deliberate on working memory.

        Args:
            working_memory: The Exec-owned WorkingMemorySet.
            context: Gating context (goal, energy, load).
            current_tick: Monotonic tick from the agent loop or WMS.

        Returns:
            GateDecision with passed=True if deliberation should proceed.
        """
        ctx = context or GatingContext()
        zero_score = GateScore(novelty=0.0, salience=0.0)

        # 1. Refractory guard — don't re-fire too often
        with self._lock:
            ticks_since = current_tick - self._last_pass_tick
        if ticks_since < self._config.refractory_ticks:
            d = GateDecision(
                passed=False,
                score=zero_score,
                reason=f"refractory ({ticks_since}/{self._config.refractory_ticks} ticks)",
                threshold_used=0.0,
            )
            self._log_decision(d)
            return d

        # 2. Energy guard — don't think when exhausted
        if self._energy_tracker is not None:
            try:
                status = self._energy_tracker.get_token_budget_status(budget_tokens=self._config.default_energy_budget)
                remaining_frac = max(0.0, status["remaining"]) / max(1, status["budget"])
                if remaining_frac < self._config.min_energy_fraction:
                    d = GateDecision(
                        passed=False,
                        score=zero_score,
                        reason=f"energy exhausted ({remaining_frac:.1%} < {self._config.min_energy_fraction:.0%})",
                        threshold_used=0.0,
                    )
                    self._log_decision(d)
                    return d
            except Exception:
                pass  # energy tracker failure is not a gate failure

        # 3. Score the working-memory head
        if self._scorer is None or working_memory is None:
            # No scorer → always pass (allow deliberation by default)
            score = GateScore(novelty=1.0, salience=1.0, combined=1.0)
        else:
            # Build scoring input from recent working memory
            recent = working_memory.recent(limit=5)
            if not recent:
                d = GateDecision(
                    passed=False,
                    score=zero_score,
                    reason="empty working memory",
                    threshold_used=0.0,
                )
                self._log_decision(d)
                return d
            # Score the most recent entry's content
            head = recent[-1]
            content = head.content if isinstance(head.content, str) else str(head.content)
            score = self._scorer.score(content, ctx)

        # 4. Adaptive threshold check
        threshold = self._threshold.combined_threshold
        threshold = max(threshold, self._config.min_combined_score)

        if score.combined < threshold:
            self._threshold.record_escalation(escalated=False)
            decision = GateDecision(
                passed=False,
                score=score,
                reason=f"below threshold ({score.combined:.2f} < {threshold:.2f})",
                threshold_used=threshold,
            )
            self._log_decision(decision)
            return decision

        # Passed all checks — record and update refractory
        with self._lock:
            self._last_pass_tick = current_tick
        self._threshold.record_escalation(escalated=True)

        decision = GateDecision(
            passed=True,
            score=score,
            reason="deliberation approved",
            threshold_used=threshold,
        )
        self._log_decision(decision)
        return decision

    @staticmethod
    def _log_decision(decision: GateDecision) -> None:
        """Emit sim_thought log if sim logging is active."""
        try:
            from maxim.simulation.sim_logger import sim_thought

            sim_thought(
                passed=decision.passed,
                reason=decision.reason,
                score=decision.score.combined,
                threshold=decision.threshold_used,
            )
        except Exception:
            pass  # sim logging not available (headless / tests)

    def record_outcome(self, decision: GateDecision, *, was_useful: bool) -> None:
        """Feed back whether the deliberation was useful.

        Passed to AdaptiveThresholdController so future thresholds
        adapt based on whether thinking produced a meaningful result.
        """
        outcome = "action_taken" if was_useful else "ignored"
        self._threshold.record_outcome(outcome)

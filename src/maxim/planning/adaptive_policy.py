# planning/adaptive_policy.py
"""Multi-signal scoring policy using all memory systems.

Scores plans across 6 dimensions, each from a different memory system:
- nac_value: NAc Rescorla-Wagner predicted outcome quality
- ec_familiarity: EC situational similarity (inverse of novelty)
- concept_relevance: ConceptContextBuilder skill ranking
- delay_efficiency: NAc temporal prediction relative to budget
- depth_penalty: ADaPT principle — simpler plans preferred
- action_cost: fewer steps = less energy = less risk
"""

from __future__ import annotations

from typing import Any

from maxim.planning.base import Policy


class AdaptivePolicy(Policy):
    """Scores plans using signals from all memory systems.

    Score = sum(weight_i * signal_i) for each dimension.
    Higher is better.  All signals normalised to [0, 1].
    """

    def __init__(
        self,
        nac: Any = None,
        *,
        w_nac: float = 0.30,
        w_familiarity: float = 0.15,
        w_concept: float = 0.15,
        w_delay: float = 0.10,
        w_depth: float = 0.15,
        w_cost: float = 0.15,
        delay_budget_s: float = 60.0,
    ):
        self._nac = nac
        self._w_nac = w_nac
        self._w_familiarity = w_familiarity
        self._w_concept = w_concept
        self._w_delay = w_delay
        self._w_depth = w_depth
        self._w_cost = w_cost
        self._delay_budget_s = delay_budget_s

    # ── Scoring ───────────────────────────────────────────────

    def score(self, plan: Any, state: Any, memory: Any) -> float:
        dims = self._compute_dimensions(plan)
        return (
            self._w_nac * dims["nac_value"]
            + self._w_familiarity * dims["ec_familiarity"]
            + self._w_concept * dims["concept_relevance"]
            + self._w_delay * dims["delay_efficiency"]
            + self._w_depth * dims["depth_penalty"]
            + self._w_cost * dims["action_cost"]
        )

    # ── Hard constraints ──────────────────────────────────────

    def allow(self, action: Any, state: Any) -> bool:
        """Block actions with very high-confidence negative predictions."""
        if not self._nac:
            return True
        tool_name = action.get("tool_name", "") if isinstance(action, dict) else ""
        if not tool_name:
            return True
        prediction = self._nac.predict("tool", tool_name)
        if prediction is None:
            return True
        # Block only when NAc is very confident (>0.85) and outcome very bad (<0.1)
        if prediction.confidence > 0.85 and prediction.predicted_value < 0.1:
            return False
        return True

    # ── Internals ─────────────────────────────────────────────

    def _compute_dimensions(self, plan: Any) -> dict[str, float]:
        """Compute all scoring dimensions.  Single source of truth."""
        # Extract fields from PlanCandidate or raw list[dict]
        if hasattr(plan, "actions"):
            actions = plan.actions
            prediction = plan.nac_prediction
            pctx = getattr(plan, "planning_context", None)
            depth = plan.depth
            delay = getattr(plan, "estimated_delay", None)
        else:
            actions = plan if isinstance(plan, list) else [plan]
            prediction = None
            pctx = None
            depth = 0
            delay = None

        # 1. NAc value: predicted outcome quality (0-1)
        nac_score = 0.5
        if prediction is not None:
            nac_score = prediction.predicted_value
            if prediction.predicted_valence.value == "negative":
                nac_score = 1.0 - nac_score

        # 2. EC familiarity: inverse of situation novelty
        familiarity_score = 0.5
        if pctx is not None:
            familiarity_score = 1.0 - pctx.situation_novelty

        # 3. Concept relevance: normalised edge count from rank_available_skills
        concept_score = 0.5
        if pctx is not None and pctx.ranked_skills:
            tool_name = ""
            if actions and isinstance(actions[0], dict):
                tool_name = actions[0].get("tool_name", "")
            for skill in pctx.ranked_skills:
                if skill.get("name") == tool_name:
                    raw_relevance = skill.get("relevance", 0)
                    concept_score = raw_relevance / (raw_relevance + 2)
                    success_rate = skill.get("past_success_rate")
                    if success_rate is not None:
                        concept_score = 0.7 * concept_score + 0.3 * success_rate
                    break

        # 4. Delay efficiency: faster is better
        delay_score = 0.5
        if delay is not None and delay > 0:
            delay_score = self._delay_budget_s / (self._delay_budget_s + delay)

        # 5. Depth penalty: prefer simpler plans (ADaPT)
        depth_score = 1.0 / (1.0 + depth)

        # 6. Action cost: fewer actions = less risk
        cost_score = 1.0 / (1.0 + len(actions) * 0.2)

        return {
            "nac_value": nac_score,
            "ec_familiarity": familiarity_score,
            "concept_relevance": concept_score,
            "delay_efficiency": delay_score,
            "depth_penalty": depth_score,
            "action_cost": cost_score,
        }

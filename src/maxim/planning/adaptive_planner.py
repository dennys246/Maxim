# planning/adaptive_planner.py
"""ADaPT-style lazy planner with deep memory integration.

Queries all available memory systems before proposing plans:
1. EC — situational similarity ("have I been here before?")
2. NAc — causal prediction ("what happened last time?")
3. Hippocampus — associative recall via spreading activation
4. ConceptContextBuilder — semantic context + skill discovery
5. RetrievalOrchestrator — multi-signal fusion

Tries direct execution first, decomposes only on failure or
when memory systems signal caution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from maxim.planning.base import Planner


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PlanningContext:
    """Aggregated memory signals for planning decisions.

    Built by ``_gather_context()`` from all available memory systems.
    Consumed by ``propose_plans()`` for candidate generation and by
    ``_decompose()`` for LLM prompt enrichment.
    """

    # NAc causal predictions
    primary_prediction: Any | None = None
    all_outcomes: list[Any] = field(default_factory=list)

    # EC situational similarity
    similar_situations: list[tuple[str, Any]] = field(default_factory=list)
    situation_novelty: float = 1.0  # 1.0 = never seen, 0.0 = very familiar

    # Hippocampus associative recall (via spreading activation)
    associated_memories: list[tuple[Any, float]] = field(default_factory=list)
    reflections: list[Any] = field(default_factory=list)
    successful_strategies: list[Any] = field(default_factory=list)

    # ConceptContextBuilder semantic context
    active_concepts: list[dict[str, Any]] = field(default_factory=list)
    ranked_skills: list[dict[str, Any]] = field(default_factory=list)

    # RetrievalOrchestrator multi-signal fusion
    retrieval_results: list[Any] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def has_warnings(self) -> bool:
        """True if any memory system signals caution."""
        nac_warns = (
            self.primary_prediction is not None
            and self.primary_prediction.predicted_valence.value == "negative"
            and self.primary_prediction.confidence > 0.5
        )
        has_reflections = len(self.reflections) > 0
        # Novel situation with NO outcome data = truly unknown territory.
        is_unknown = self.situation_novelty > 0.8 and len(self.all_outcomes) == 0
        return nac_warns or has_reflections or is_unknown

    def to_llm_section(self) -> str:
        """Format memory signals as LLM context for decomposition prompts."""
        parts: list[str] = []

        if self.primary_prediction:
            p = self.primary_prediction
            parts.append(
                f"NAc prediction: {p.predicted_valence.value} outcome "
                f"(value={p.predicted_value:.2f}, confidence={p.confidence:.2f}, "
                f"expected delay={p.predicted_delay:.1f}s "
                f"[{p.delay_bounds[0]:.1f}-{p.delay_bounds[1]:.1f}s])"
            )

        if self.reflections:
            parts.append("Past reflections on similar attempts:")
            for r in self.reflections[:3]:
                outcome = getattr(r, "outcome", None)
                ref_text = None
                if isinstance(outcome, dict):
                    ref_text = outcome.get("reflection")
                elif hasattr(outcome, "result") and isinstance(outcome.result, dict):
                    ref_text = outcome.result.get("reflection")
                if ref_text:
                    parts.append(f"  - {ref_text}")

        if self.successful_strategies:
            parts.append("Successful strategies in similar situations:")
            for m in self.successful_strategies[:3]:
                action = getattr(m, "action", None)
                if hasattr(action, "tool_name") and action.tool_name:
                    parts.append(f"  - Used {action.tool_name} → success")

        if self.ranked_skills:
            top_skills = [s["name"] for s in self.ranked_skills[:5]]
            parts.append(f"Concept-ranked relevant tools: {', '.join(top_skills)}")

        if self.similar_situations:
            parts.append(
                f"EC found {len(self.similar_situations)} similar past situations "
                f"(novelty={self.situation_novelty:.2f})"
            )

        if self.retrieval_results:
            top_hits = [r for r in self.retrieval_results[:3] if r.combined_score > 0.5]
            if top_hits:
                parts.append("Broadly relevant memories (multi-signal retrieval):")
                for hit in top_hits:
                    signals = ", ".join(f"{k}={v:.2f}" for k, v in hit.signal_scores.items())
                    parts.append(f"  - {hit.memory_id[:8]}... (score={hit.combined_score:.2f}, {signals})")

        return "\n".join(parts) if parts else ""


@dataclass
class PlanCandidate:
    """A scored plan with provenance from memory systems."""

    actions: list[dict[str, Any]]
    source: str  # "direct" | "decomposed" | "nac_alternative" | "concept_suggested" | "reflection_informed"
    planning_context: PlanningContext | None = None
    nac_prediction: Any | None = None
    relevant_reflections: list[str] = field(default_factory=list)
    depth: int = 0
    estimated_delay: float | None = None
    delay_bounds: tuple[float, float] | None = None


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class AdaptivePlanner(Planner):
    """ADaPT-style lazy planner with deep memory integration."""

    def __init__(
        self,
        nac: Any = None,
        hippocampus: Any = None,
        ec: Any = None,
        atl: Any = None,
        concept_context_builder: Any = None,
        retrieval_orchestrator: Any = None,
        llm: Any = None,
        *,
        max_depth: int = 3,
        nac_veto_confidence: float = 0.7,
        nac_veto_threshold: float = 0.2,
        context_budget_ms: float = 50.0,
    ):
        self._nac = nac
        self._hippocampus = hippocampus
        self._ec = ec
        self._atl = atl
        self._ccb = concept_context_builder
        self._retrieval = retrieval_orchestrator
        self._llm = llm
        self._max_depth = max_depth
        self._nac_veto_confidence = nac_veto_confidence
        self._nac_veto_threshold = nac_veto_threshold
        self._context_budget_ms = context_budget_ms

    # ── Public interface ──────────────────────────────────────

    def propose_plans(self, goal: Any, state: Any, memory: Any) -> list[PlanCandidate]:
        """Return 1–3 candidate plans ranked by memory-informed confidence."""
        goal_dict = goal if isinstance(goal, dict) else {"description": str(goal)}
        tool_name = goal_dict.get("tool_name", "")
        context = _extract_context(goal_dict, state)

        pctx = self._gather_context(goal_dict, tool_name, context, state)

        candidates: list[PlanCandidate] = []

        # 1. NAc veto: does past experience strongly predict failure?
        nac_vetoes = (
            pctx.primary_prediction is not None
            and pctx.primary_prediction.confidence > self._nac_veto_confidence
            and pctx.primary_prediction.predicted_value < self._nac_veto_threshold
        )

        # 2. Direct execution (unless NAc vetoes)
        if not nac_vetoes and tool_name:
            delay = None
            bounds = None
            if pctx.primary_prediction:
                delay = pctx.primary_prediction.predicted_delay
                bounds = pctx.primary_prediction.delay_bounds
            candidates.append(
                PlanCandidate(
                    actions=[{"tool_name": tool_name, "params": goal_dict.get("tool_params", {})}],
                    source="direct",
                    planning_context=pctx,
                    nac_prediction=pctx.primary_prediction,
                    relevant_reflections=[getattr(r, "id", "") for r in pctx.reflections],
                    depth=0,
                    estimated_delay=delay,
                    delay_bounds=bounds,
                )
            )

        # 3. Concept-suggested alternative tools
        if pctx.ranked_skills:
            for skill in pctx.ranked_skills[:2]:
                skill_name = skill.get("name", "")
                # relevance is integer edge count — 2+ is meaningful
                if skill_name and skill_name != tool_name and skill.get("relevance", 0) >= 2:
                    alt_prediction = None
                    if self._nac:
                        alt_prediction = self._nac.predict("tool", skill_name, context)
                    if alt_prediction is None or alt_prediction.predicted_valence.value != "negative":
                        candidates.append(
                            PlanCandidate(
                                actions=[{"tool_name": skill_name, "params": {}}],
                                source="concept_suggested",
                                planning_context=pctx,
                                nac_prediction=alt_prediction,
                                depth=0,
                                estimated_delay=alt_prediction.predicted_delay if alt_prediction else None,
                                delay_bounds=alt_prediction.delay_bounds if alt_prediction else None,
                            )
                        )
                        break

        # 4. NAc positive alternative outcomes (different outcome for same tool)
        for alt in pctx.all_outcomes:
            if alt.predicted_valence.value == "positive" and alt.confidence > 0.5:
                if pctx.primary_prediction and alt.predicted_outcome != pctx.primary_prediction.predicted_outcome:
                    candidates.append(
                        PlanCandidate(
                            actions=[
                                {
                                    "tool_name": tool_name,
                                    "params": goal_dict.get("tool_params", {}),
                                    "_nac_hint": alt.predicted_outcome,
                                }
                            ],
                            source="nac_alternative",
                            planning_context=pctx,
                            nac_prediction=alt,
                            depth=0,
                            estimated_delay=alt.predicted_delay,
                            delay_bounds=alt.delay_bounds,
                        )
                    )
                    break

        # 5. Decomposed plan if memory signals caution
        if nac_vetoes or pctx.has_warnings():
            decomposed = self._decompose(goal_dict, state, memory, pctx, depth=0)
            if decomposed:
                candidates.append(decomposed)

        # Fallback: always return at least one candidate
        if not candidates:
            candidates.append(
                PlanCandidate(
                    actions=[{"tool_name": tool_name, "params": goal_dict.get("tool_params", {})}],
                    source="direct",
                    planning_context=pctx,
                    depth=0,
                )
            )

        return candidates

    def decompose(self, failed_goal: dict, replan_ctx: Any, depth: int) -> PlanCandidate | None:
        """ADaPT decomposition after execution failure (called by replan loop)."""
        if depth >= self._max_depth:
            return None
        tool_name = failed_goal.get("tool_name", "")
        context = _extract_context(failed_goal, None)
        pctx = self._gather_context(failed_goal, tool_name, context, None)
        return self._decompose(failed_goal, None, None, pctx, depth, replan_ctx=replan_ctx)

    # ── Memory gathering ──────────────────────────────────────

    def _gather_context(
        self,
        goal_dict: dict,
        tool_name: str,
        context: dict,
        state: Any,
    ) -> PlanningContext:
        """Query all memory systems and assemble a PlanningContext.

        Order matters: EC results seed hippocampal spreading activation.
        """
        pctx = PlanningContext()

        # 1. EC: situational similarity
        if self._ec and tool_name:
            try:
                pctx.similar_situations = self._ec.find_similar_situations_for_action(
                    tool_name=tool_name,
                    context=context,
                    k=10,
                )
                n = len(pctx.similar_situations)
                pctx.situation_novelty = 1.0 / (1.0 + n * 0.5)
            except Exception:
                pass

        # 2. NAc: causal prediction
        if self._nac and tool_name:
            try:
                pctx.primary_prediction = self._nac.predict("tool", tool_name, context)
            except Exception:
                pass
            try:
                pctx.all_outcomes = self._nac.predict_all_outcomes("tool", tool_name, context)
            except Exception:
                pass

        # 3. Hippocampus: spreading activation from EC seeds
        if self._hippocampus:
            try:
                seed_ids = [mid for mid, _sig in pctx.similar_situations[:5]]
                if seed_ids:
                    pctx.associated_memories = self._hippocampus.recall_associated(
                        seed_ids=seed_ids,
                        limit=15,
                    )
                else:
                    # Fallback: hash-index recall → use as seeds
                    direct_hits = self._hippocampus.recall(
                        limit=5,
                        goal=goal_dict.get("description"),
                        tool=tool_name or None,
                    )
                    if direct_hits:
                        fallback_seeds = [m.id for m in direct_hits[:3]]
                        pctx.associated_memories = self._hippocampus.recall_associated(
                            seed_ids=fallback_seeds,
                            limit=15,
                        )
                    else:
                        pctx.associated_memories = []

                # Partition into reflections and successes
                for mem, _activation in pctx.associated_memories:
                    outcome = getattr(mem, "outcome", None)
                    if outcome is None:
                        continue

                    reflection_text = None
                    is_success = False

                    if isinstance(outcome, dict):
                        reflection_text = outcome.get("reflection")
                        is_success = outcome.get("success", False)
                    else:
                        is_success = getattr(outcome, "success", False)
                        result = getattr(outcome, "result", None)
                        if isinstance(result, dict):
                            reflection_text = result.get("reflection")

                    if reflection_text:
                        pctx.reflections.append(mem)
                    elif is_success:
                        pctx.successful_strategies.append(mem)
            except Exception:
                pass

        # 4. ConceptContextBuilder: semantic context + skill ranking
        if self._ccb:
            try:
                detected_objects = None
                detected_people = None
                if state is not None and hasattr(state, "data") and isinstance(state.data, dict):
                    detected_objects = state.data.get("detected_objects")
                    detected_people = state.data.get("detected_people")

                pctx.active_concepts = self._ccb.build(
                    detected_objects=detected_objects,
                    detected_people=detected_people,
                    active_goal=goal_dict.get("description"),
                    limit=5,
                    budget_ms=self._context_budget_ms,
                )

                if pctx.active_concepts and self._atl:
                    concept_objs = []
                    for cd in pctx.active_concepts:
                        cid = cd.get("id")
                        if cid:
                            concept = self._atl.get(cid)
                            if concept is not None:
                                concept_objs.append(concept)
                    if concept_objs:
                        pctx.ranked_skills = self._ccb.rank_available_skills(concept_objs)
            except Exception:
                pass

        # 5. RetrievalOrchestrator: multi-signal fusion
        if self._retrieval and goal_dict.get("description"):
            try:
                pctx.retrieval_results = self._retrieval.retrieve(
                    text=goal_dict["description"],
                    limit=10,
                )
            except Exception:
                pass

        return pctx

    # ── Decomposition ─────────────────────────────────────────

    def _decompose(
        self,
        goal_dict: dict,
        state: Any,
        memory: Any,
        pctx: PlanningContext,
        depth: int,
        replan_ctx: Any = None,
    ) -> PlanCandidate | None:
        """LLM-driven decomposition enriched with full memory context."""
        if not self._llm or depth >= self._max_depth:
            return None

        prompt = _build_decomposition_prompt(goal_dict, pctx, replan_ctx=replan_ctx)
        sub_actions = self._llm.generate_structured(prompt, schema="decomposition")
        if not sub_actions:
            return None

        return PlanCandidate(
            actions=sub_actions,
            source="reflection_informed" if pctx.reflections else "decomposed",
            planning_context=pctx,
            relevant_reflections=[getattr(r, "id", "") for r in pctx.reflections],
            depth=depth + 1,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _extract_context(goal_dict: dict, state: Any) -> dict[str, Any]:
    """Build context dict compatible with NAc and EC queries."""
    ctx: dict[str, Any] = {}
    if goal_dict.get("description"):
        ctx["goal"] = goal_dict["description"]
    if state is not None and hasattr(state, "data") and isinstance(state.data, dict):
        ctx["mode"] = state.data.get("mode", "")
        ctx["detected_objects"] = state.data.get("detected_objects", [])
    return ctx


def _build_decomposition_prompt(
    goal_dict: dict,
    pctx: PlanningContext,
    replan_ctx: Any = None,
) -> str:
    """Build LLM prompt for goal decomposition, enriched with memory signals."""
    parts = [
        "Decompose this goal into 3-5 concrete sub-tasks. Each sub-task must be executable by a single tool call.",
        f"\nGoal: {goal_dict.get('description', '')}",
        f"Proposed tool: {goal_dict.get('tool_name', 'unknown')}",
    ]

    # Replan context: what failed and why
    if replan_ctx is not None and hasattr(replan_ctx, "to_llm_prompt_section"):
        replan_section = replan_ctx.to_llm_prompt_section()
        if replan_section:
            parts.append(f"\n## Previous attempt failed\n{replan_section}")

    # Memory context
    memory_section = pctx.to_llm_section()
    if memory_section:
        parts.append(f"\n## What the agent's memory systems report\n{memory_section}")

    # All predicted outcomes
    if pctx.all_outcomes:
        parts.append("\nKnown possible outcomes for this tool:")
        for out in pctx.all_outcomes[:5]:
            parts.append(
                f"  - {out.predicted_outcome}: {out.predicted_valence.value} "
                f"(confidence={out.confidence:.2f}, delay={out.predicted_delay:.1f}s)"
            )

    parts.append(
        "\nReturn a JSON array of objects with 'tool_name' and 'params' keys. "
        "Order sub-tasks by dependency (prerequisites first)."
    )
    return "\n".join(parts)

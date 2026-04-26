"""BioEnrichmentPipeline — source-agnostic text enrichment via bio-systems.

The thalamic relay for text: all text the agent processes can pass
through this pipeline.  Novelty gating ensures only interesting inputs
get the full enrichment treatment (~26ms budget).

Architecture:
- Novelty gate via ``TextSalienceScorer`` (from ``runtime/gating.py``)
- Parallel queries to hippocampus, NAc, ATL, ComponentIndex
- Returns structured ``EnrichmentResult`` with memories, predictions,
  concepts, affordances, and overall valence

Consumers:
- ThinkTool (always enriched, L1)
- Text percepts (gated by novelty threshold, L1)
- Future: internet search results, audio transcripts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.decisions.nac import NAc
    from maxim.embodiment.component_index import ComponentIndex
    from maxim.embodiment.reflex import ReflexRegistry
    from maxim.memory.atl import ATL
    from maxim.memory.hippocampus import Hippocampus
    from maxim.runtime.gating import TextSalienceScorer
    from maxim.similarity.ec import EntorhinalCortex

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EpisodicSummary:
    """One-line summary of a past episode surfaced by hippocampal recall."""

    memory_id: str
    summary: str
    valence: float  # -1 to +1
    relevance: float  # 0 to 1 (match score)


@dataclass(frozen=True, slots=True)
class CausalPrediction:
    """NAc prediction for a recognized event/action."""

    event: str
    outcome: str
    confidence: float  # 0 to 1
    valence: str  # "positive" / "negative" / "neutral"


@dataclass(frozen=True, slots=True)
class ConceptLink:
    """ATL concept association surfaced by spreading activation."""

    concept: str
    category: str
    activation: float  # 0 to 1


@dataclass(frozen=True, slots=True)
class EnrichmentResult:
    """Bio-system associations surfaced for a text input.

    Each field represents one bio-system's contribution to the
    agent's understanding of the input text.
    """

    memories: tuple[EpisodicSummary, ...] = ()
    predictions: tuple[CausalPrediction, ...] = ()
    concepts: tuple[ConceptLink, ...] = ()
    affordances: tuple[str, ...] = ()
    recent_context: tuple[str, ...] = ()  # WMS summaries (recent actions/outcomes)
    valence: float = 0.0  # overall approach/avoid signal (-1 to +1)
    novel: bool = True  # whether the novelty gate fired
    reflexes_fired: tuple[str, ...] = ()  # names of reflexes that fired this tick


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EnrichmentContext:
    """Context for enrichment — goal, recent thoughts, entity names."""

    active_goal: str | None = None
    goal_keywords: tuple[str, ...] = ()
    recent_thoughts: tuple[str, ...] = ()  # last 3 thoughts for convergence
    entity_names: tuple[str, ...] = ()  # known entity names

    def to_gating_context(self) -> Any:
        """Convert to GatingContext for the scorer."""
        from maxim.runtime.gating import GatingContext

        return GatingContext(
            active_goal=self.active_goal,
            goal_keywords=self.goal_keywords,
        )


class BioEnrichmentPipeline:
    """Thalamic relay for text — gates and enriches via bio-systems.

    All text the agent processes can pass through this pipeline.
    Novelty gating ensures only interesting inputs get the full
    enrichment treatment (prevents wasting ~26ms on "hello").

    Latency budget: < 50ms total (no LLM call).

    Example::

        pipeline = BioEnrichmentPipeline(
            scorer=text_scorer,
            hippocampus=hippo,
            nac=nac,
        )
        result = pipeline.enrich("the rusty gate blocks the path", context=ctx)
        if result is not None:
            print(result.memories)     # past episodes about gates/rust
            print(result.predictions)  # NAc predictions for force/unlock
            print(result.affordances)  # available physical actions
    """

    def __init__(
        self,
        *,
        scorer: TextSalienceScorer | None = None,
        hippocampus: Hippocampus | None = None,
        nac: NAc | None = None,
        atl: ATL | None = None,
        ec: EntorhinalCortex | None = None,
        component_index: ComponentIndex | None = None,
        reflex_registry: ReflexRegistry | None = None,
        novelty_threshold: float = 0.4,
        agent_id: str = "",
    ) -> None:
        self._scorer = scorer
        self._hippocampus = hippocampus
        self._nac = nac
        self._atl = atl
        self._ec = ec
        self._component_index = component_index
        self._reflex_registry = reflex_registry
        self._novelty_threshold = novelty_threshold
        self._agent_id = agent_id

    def enrich(
        self,
        text: str,
        *,
        context: EnrichmentContext | None = None,
        bypass_gate: bool = False,
        working_memory: Any | None = None,
    ) -> EnrichmentResult | None:
        """Enrich text with bio-system associations.

        Returns None if the novelty gate rejects the input (familiar,
        low salience).  Pass ``bypass_gate=True`` for think tool calls
        (always enrich deliberate thoughts).

        Args:
            text: Input text to enrich.
            context: Goal, recent thoughts, entity names.
            bypass_gate: Skip novelty gating (for explicit think calls).
            working_memory: Optional WorkingMemorySet.  When provided,
                recent actions/outcomes are summarized and included in the
                result.  This is the PFC's short-term/working-memory
                retrieval path — even in a fresh session, the agent's
                recent actions inform the next decision.

        Returns:
            EnrichmentResult or None if below threshold.
        """
        if not text or not text.strip():
            return None

        ctx = context or EnrichmentContext()

        # Novelty gate (unless bypassed for explicit think calls)
        if not bypass_gate and self._scorer is not None:
            score = self._scorer.score(text, ctx.to_gating_context())
            if score.combined < self._novelty_threshold:
                return None

        # Extract keywords for bio-system queries
        keywords = self._extract_keywords(text)

        # Query bio-systems (each handles None gracefully)
        memories = self._query_hippocampus(text, keywords)
        predictions = self._query_nac(keywords)
        concepts = self._query_atl(keywords)
        affordances = self._query_component_index(text)
        recent_context = self._query_working_memory(working_memory)

        # Compute overall valence from memories + predictions
        valence = self._compute_valence(memories, predictions)

        # Reflex evaluation: innate body responses to percept signals.
        # Fires BEFORE the LLM deliberates — the body responds before the
        # mind decides.  Predictions are passed in (not re-queried) for
        # pre-emption suppression.
        latent_affordance_strs: list[str] = []
        reflexes_fired = self._evaluate_reflexes(text, tuple(predictions), latent_affordance_strs)

        # Merge latent affordances into affordances (reflex-triggered
        # motor programs the agent can take — "dodge", "block", etc.)
        all_affordances = list(affordances) + latent_affordance_strs

        # Track 4: Emit enrichment transparency logs so display/JSONL
        # captures WHAT each bio-system contributed, not just that it did.
        self._log_enrichment_contributions(memories, predictions, concepts, all_affordances, recent_context)

        return EnrichmentResult(
            memories=tuple(memories),
            predictions=tuple(predictions),
            concepts=tuple(concepts),
            affordances=tuple(all_affordances),
            recent_context=tuple(recent_context),
            valence=valence,
            novel=True,
            reflexes_fired=reflexes_fired,
        )

    def format_thought_response(self, result: EnrichmentResult) -> str:
        """Format an EnrichmentResult as a human-readable thought response.

        This is what the LLM sees as the think tool's "system response."
        """
        lines: list[str] = []

        if result.memories:
            lines.append("Your experience suggests:")
            for mem in result.memories[:3]:
                valence_icon = "+" if mem.valence > 0 else "-" if mem.valence < 0 else "~"
                lines.append(f"  [{valence_icon}] {mem.summary}")

        if result.predictions:
            lines.append("Predictions from past actions:")
            for pred in result.predictions[:3]:
                conf_word = "high" if pred.confidence >= 0.7 else "moderate"
                lines.append(f"  - {pred.event} → {pred.outcome} ({conf_word} confidence, {pred.valence})")

        if result.concepts:
            concept_names = [c.concept for c in result.concepts[:5]]
            lines.append(f"Related concepts: {', '.join(concept_names)}")

        if result.affordances:
            lines.append(f"Available actions (use via use tool): {', '.join(result.affordances[:5])}")

        if result.recent_context:
            lines.append("Recent actions and outcomes:")
            for rc in result.recent_context[:5]:
                lines.append(f"  - {rc}")

        if not lines:
            return ""

        return "\n".join(lines)

    @staticmethod
    def _log_enrichment_contributions(
        memories: list[EpisodicSummary],
        predictions: list[CausalPrediction],
        concepts: list[ConceptLink],
        affordances: list[str],
        recent_context: list[str],
    ) -> None:
        """Emit sim_enrichment logs for each bio-system that contributed."""
        try:
            from maxim.simulation.sim_logger import sim_enrichment
        except ImportError:
            return

        if memories:
            summaries = [m.summary[:60] for m in memories[:3]]
            sim_enrichment("hippocampus", f"{len(memories)} episode(s): {'; '.join(summaries)}")

        if predictions:
            pred_strs = [f"{p.event}→{p.outcome} ({p.confidence:.0%})" for p in predictions[:3]]
            sim_enrichment("nac", f"{len(predictions)} prediction(s): {', '.join(pred_strs)}")

        if concepts:
            concept_names = [c.concept for c in concepts[:5]]
            sim_enrichment("atl", f"{len(concepts)} concept(s): {', '.join(concept_names)}")

        if affordances:
            sim_enrichment("component_index", f"{len(affordances)} affordance(s): {', '.join(affordances[:5])}")

        if recent_context:
            sim_enrichment("working_memory", f"{len(recent_context)} recent action(s)")

    def _evaluate_reflexes(
        self,
        text: str,
        predictions: tuple[CausalPrediction, ...],
        latent_out: list[str] | None = None,
    ) -> tuple[str, ...]:
        """Evaluate reflex registry against percept text.

        Reflexes fire tools (damage_component, set_entity_sensor) as
        automatic body responses.  The pain pipeline handles NAc learning
        — no separate Reaction is emitted.

        When reflexes fire and ``_entity_root`` is set, also collects
        latent motor programs (dodge, block, brace) from ALL body
        modulators that pass integrity gating.  These are appended to
        ``latent_out`` for merging into the affordances tuple.

        Returns tuple of reflex names that fired.
        """
        if self._reflex_registry is None:
            return ()

        try:
            firings = self._reflex_registry.evaluate(
                text,
                predictions=predictions,
                execute_tool=self._dispatch_reflex_tool,
            )
            if firings:
                names = tuple(f.reflex_name for f in firings)

                # Surface latent motor programs from ALL body modulators.
                # Whole-body response: an attack to torso also surfaces
                # dodge (legs) and block (arms).
                if latent_out is not None:
                    self._collect_latent_affordances(latent_out)

                try:
                    from maxim.simulation.sim_logger import sim_enrichment

                    details = [f"{f.reflex_name}({f.tool}, intensity={f.effective_intensity:.2f})" for f in firings]
                    sim_enrichment("reflex", f"{len(firings)} reflex(es): {', '.join(details)}")
                    if latent_out:
                        sim_enrichment("latent", f"{len(latent_out)} motor program(s): {', '.join(latent_out[:5])}")
                except ImportError:
                    pass
                return names
            return ()
        except Exception as e:
            log.debug("Reflex evaluation failed: %s", e)
            return ()

    def _collect_latent_affordances(self, out: list[str]) -> None:
        """Collect integrity-gated latent affordances from all body modulators.

        Called when reflexes fire — surfaces motor programs the agent
        could take in response (dodge, block, brace).  Only available
        affordances (passing integrity threshold) are included.
        """
        entity_root = getattr(self, "_entity_root", None)
        if entity_root is None:
            return

        seen: set[str] = set()
        try:
            for mod in entity_root.modulators.values():
                if not hasattr(mod, "available_latent_affordances"):
                    continue
                for la in mod.available_latent_affordances():
                    if la.name not in seen:
                        seen.add(la.name)
                        label = f"{la.name} — {la.description}" if la.description else la.name
                        out.append(label)
        except Exception as e:
            log.debug("Latent affordance collection failed: %s", e)

    def _dispatch_reflex_tool(self, tool_name: str, **params: Any) -> Any:
        """Dispatch a reflex tool invocation.

        Called by ReflexRegistry.evaluate() for each reflex that fires.
        The tool dispatch is intentionally simple — reflexes only use
        tools that operate on the embodiment (damage_component,
        set_entity_sensor), not full agent tools.

        The actual tool instances are set via ``_reflex_damage_tool`` and
        ``_reflex_sensor_tool`` attributes, wired by the orchestrator
        after pipeline construction.
        """
        if tool_name == "damage_component":
            tool = getattr(self, "_reflex_damage_tool", None)
            if tool is not None:
                return tool.execute(**params)
        elif tool_name == "set_entity_sensor":
            tool = getattr(self, "_reflex_sensor_tool", None)
            if tool is not None:
                return tool.execute(**params)
        log.debug("Reflex tool '%s' not wired — skipping dispatch", tool_name)
        return None

    # -- Private query methods -------------------------------------------------

    @staticmethod
    def _query_working_memory(working_memory: Any | None) -> list[str]:
        """Summarize recent actions/outcomes from WorkingMemorySet.

        This is the PFC's working-memory retrieval path.  Even in a fresh
        session with no episodic memories, the agent's recent actions and
        their outcomes are available to inform the next decision.
        """
        if working_memory is None:
            return []
        try:
            from maxim.agents.working_memory import WorkingMemoryKind

            # Pull recent outcomes and conversations (last 5)
            relevant_kinds = {
                WorkingMemoryKind.OUTCOME,
                WorkingMemoryKind.CONVERSATION,
                WorkingMemoryKind.PERCEPT,
            }
            entries = working_memory.by_kind(relevant_kinds, limit=5)
            summaries: list[str] = []
            for entry in entries:
                content = entry.content
                if isinstance(content, dict):
                    # Outcome entries have tool_name + success
                    tool = content.get("tool_name") or content.get("action", "")
                    if tool:
                        success = content.get("success", True)
                        error = content.get("error", "")
                        status = "succeeded" if success else f"failed: {error}"
                        goal = content.get("goal", "")
                        summary = f"{tool} {status}"
                        if goal:
                            summary += f" (goal: {goal})"
                        summaries.append(summary)
                    else:
                        # Generic dict — take a short repr
                        text_val = content.get("text") or content.get("content") or ""
                        if text_val:
                            summaries.append(str(text_val)[:100])
                elif isinstance(content, str) and content.strip():
                    summaries.append(content[:100])
            return summaries
        except Exception:
            return []

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from text for bio-system queries."""
        # Simple keyword extraction: split, lowercase, filter short/stop words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "to",
            "of",
            "in",
            "on",
            "at",
            "for",
            "and",
            "or",
            "but",
            "not",
            "it",
            "i",
            "my",
            "you",
            "this",
            "that",
            "with",
            "from",
            "by",
            "do",
            "does",
            "how",
            "what",
            "can",
            "could",
            "would",
            "should",
            "will",
        }
        words = text.lower().replace("_", " ").replace("-", " ").split()
        return [w for w in words if len(w) > 2 and w not in stop_words]

    def _query_hippocampus(self, text: str, keywords: list[str]) -> list[EpisodicSummary]:
        """Search hippocampus for episodes matching the text."""
        if self._hippocampus is None:
            return []
        try:
            results = self._hippocampus.search_by_content(text, limit=5)
            summaries = []
            for mem in results[:3]:
                summary = self._summarize_episode(mem)
                valence = getattr(mem, "valence", 0.0)
                if not hasattr(mem, "valence"):
                    # Infer from success
                    valence = 0.3 if getattr(mem, "success", False) else -0.3
                summaries.append(
                    EpisodicSummary(
                        memory_id=mem.id,
                        summary=summary,
                        valence=float(valence),
                        relevance=0.8,  # search_by_content doesn't return scores
                    )
                )
            return summaries
        except Exception as e:
            log.debug("Bio-enrichment hippocampus query failed: %s", e)
            return []

    def _query_nac(self, keywords: list[str]) -> list[CausalPrediction]:
        """Query NAc for causal predictions matching keywords."""
        if self._nac is None:
            return []
        predictions: list[CausalPrediction] = []
        try:
            for keyword in keywords[:5]:  # Limit to avoid excessive queries
                links = self._nac.get_links_for_event(keyword)
                for link in links:
                    if link.confidence < 0.3:
                        continue
                    from maxim.decisions.causal_link import Valence

                    valence_str = "neutral"
                    if link.outcome_valence == Valence.POSITIVE:
                        valence_str = "positive"
                    elif link.outcome_valence == Valence.NEGATIVE:
                        valence_str = "negative"

                    predictions.append(
                        CausalPrediction(
                            event=link.event_signature,
                            outcome=link.outcome_signature,
                            confidence=link.confidence,
                            valence=valence_str,
                        )
                    )
            # Deduplicate by event and sort by confidence
            seen: set[str] = set()
            unique: list[CausalPrediction] = []
            for p in sorted(predictions, key=lambda x: x.confidence, reverse=True):
                if p.event not in seen:
                    seen.add(p.event)
                    unique.append(p)
            return unique[:3]
        except Exception as e:
            log.debug("Bio-enrichment NAc query failed: %s", e)
            return []

    def _query_atl(self, keywords: list[str]) -> list[ConceptLink]:
        """Query ATL for related concepts matching keywords."""
        if self._atl is None:
            return []
        concepts: list[ConceptLink] = []
        try:
            for keyword in keywords[:3]:
                results = self._atl.recall(limit=3, name=keyword)
                for concept in results:
                    name = getattr(concept, "name", str(concept))
                    category = getattr(concept, "category", "")
                    concepts.append(
                        ConceptLink(
                            concept=name,
                            category=category,
                            activation=0.7,
                        )
                    )
            # Deduplicate
            seen: set[str] = set()
            unique: list[ConceptLink] = []
            for c in concepts:
                if c.concept not in seen:
                    seen.add(c.concept)
                    unique.append(c)
            return unique[:5]
        except Exception as e:
            log.debug("Bio-enrichment ATL query failed: %s", e)
            return []

    def _query_component_index(self, text: str) -> list[str]:
        """Query ComponentIndex for available affordances matching text.

        When ATL + NAc are available, annotates each affordance with
        valence from substrate concept nodes. Decomposes affordance names
        into components (e.g., "fire_breath" → "fire", "breath") and
        checks NAc reward_bias on their substrate nodes. This is the
        "last mile" for cross-entity knowledge transfer — the agent sees
        [DANGEROUS] or [effective] annotations from prior experience with
        similar affordances on different entities.
        """
        if self._component_index is None:
            return []
        try:
            matches = self._component_index.find_similar(text, k=3)
            affordances: list[str] = []
            for match in matches:
                if match.score >= 0.5:
                    annotated = self._annotate_affordance_valence(match.name)
                    affordances.append(annotated)
            return affordances
        except Exception as e:
            log.debug("Bio-enrichment ComponentIndex query failed: %s", e)
            return []

    def _annotate_affordance_valence(self, affordance_name: str) -> str:
        """Annotate an affordance name with learned valence from substrate.

        Decomposes the affordance name, looks up component substrate nodes
        in ATL, checks NAc reward_bias. Returns the original name with
        an annotation suffix if bias exists, otherwise returns unchanged.
        """
        if self._nac is None or self._atl is None:
            return affordance_name

        try:
            from maxim.similarity.decomposer import AFFORDANCE_STRATEGY

            chunks = AFFORDANCE_STRATEGY.extract(affordance_name)

            max_bias = 0.0
            for chunk in chunks:
                concepts = self._atl.recall(name=chunk.text, category="substrate", limit=1)
                for concept in concepts:
                    bias = self._nac.reward_bias(self._agent_id, concept.id)
                    if abs(bias) > abs(max_bias):
                        max_bias = bias

            if max_bias < -0.01:
                return f"{affordance_name} [DANGEROUS — learned from prior experience]"
            elif max_bias > 0.01:
                return f"{affordance_name} [effective — worked well before]"
        except Exception as e:
            log.debug("Affordance valence annotation failed for '%s': %s", affordance_name, e)

        return affordance_name

    def _compute_valence(self, memories: list[EpisodicSummary], predictions: list[CausalPrediction]) -> float:
        """Compute overall valence signal from memories + predictions.

        Positive = approach (past success, predicted good outcomes).
        Negative = avoid (past pain, predicted failure).
        """
        signals: list[float] = []
        for mem in memories:
            signals.append(mem.valence)
        for pred in predictions:
            if pred.valence == "positive":
                signals.append(0.5 * pred.confidence)
            elif pred.valence == "negative":
                signals.append(-0.5 * pred.confidence)
        if not signals:
            return 0.0
        return max(-1.0, min(1.0, sum(signals) / len(signals)))

    @staticmethod
    def _summarize_episode(mem: Any) -> str:
        """Create a one-line summary from an EpisodicMemory."""
        parts: list[str] = []
        # Tool action
        tool = getattr(mem, "tool_name", "") or ""
        if hasattr(mem, "action") and hasattr(mem.action, "tool_name"):
            tool = mem.action.tool_name or tool
        if tool:
            success = getattr(mem, "success", None)
            if success is None and hasattr(mem, "outcome"):
                success = getattr(mem.outcome, "success", None)
            status = "succeeded" if success else "failed" if success is False else ""
            parts.append(f"{tool} {status}".strip())
        # Goal context
        goal = ""
        if hasattr(mem, "decision") and hasattr(mem.decision, "intent"):
            goal = mem.decision.intent.get("goal", "")
        if not goal and hasattr(mem, "context"):
            goal = getattr(mem.context, "active_goal", "") or ""
        if goal:
            parts.append(f"(goal: {goal[:40]})")
        # Outcome
        if hasattr(mem, "outcome") and hasattr(mem.outcome, "result"):
            result = mem.outcome.result
            if result and isinstance(result, str):
                parts.append(f"→ {result[:50]}")

        return " ".join(parts) if parts else "past experience"

"""ExecAgent: Goal proposal based on structured context.

Uses LLM to propose goals aligned with the root goal:
"Understand reality and help people."
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from typing import Any

from maxim.agents.base import Agent
from maxim.agents.bus import (
    AgentBus,
    GoalCompleted,
    GoalPriority,
    Percept,
    ProposedGoal,
    StructuredContext,
    SubGoal,
)
from maxim.default_network.messages import FilteredPercept
from maxim.agents.exec_prompts import (
    SYSTEM_PROMPT,
    CRITIQUE_SYSTEM,
    CRITIQUE_USER_TEMPLATE,
    FAST_CONTEMPLATE_SYSTEM,
    FAST_CONTEMPLATE_USER_TEMPLATE,
    REFINE_SYSTEM,
    REFINE_USER_TEMPLATE,
)
from maxim.agents.llm_agent import ChatLLMAgent, LLMAgentConfig
from maxim.agents.llm_worker import LLMWorker
from maxim.models.language.router import LLMRouter, load_llm_config
from maxim.prompts.prompt_profiles import ExecutivePrompt, load_prompt_profile
from maxim.agents.memory_agent import MemoryAgent
from maxim.decisions.significance import CycleContext, SignificanceConfig, SignificanceWeightLearner
from maxim.utils.logging import warn
from maxim.utils.prompts import get_agent_prompt
from maxim.utils.structured_logging import log_structured


class ExecAgent(Agent):
    """
    Proposes goals based on structured context from MemoryAgent.

    Uses LLM to:
    - Interpret voice commands
    - Reason about visual scene
    - Propose goals aligned with root goal
    - Prioritize based on urgency and relevance

    All proposals serve the root goal:
    "Understand reality and help people."
    """

    agent_name = "exec_agent"

    SYSTEM_PROMPT = SYSTEM_PROMPT

    def __init__(
        self,
        bus: AgentBus,
        memory_agent: MemoryAgent,
        *,
        name: str | None = None,
        enabled: bool = True,
        llm_profile: str = "mistral-7b-instruct-v0.2",
        quantization: str = "Q4_K_M",
        system_prompt: str | None = None,
        rate_limit_hz: float = 2.0,
    ) -> None:
        super().__init__(name=name, enabled=enabled)
        self._bus = bus
        self._memory = memory_agent
        if system_prompt:
            self._system_prompt = system_prompt
        else:
            prompt = get_agent_prompt(self.agent_name)
            self._system_prompt = prompt or self.SYSTEM_PROMPT

        # LLM config
        self._llm: ChatLLMAgent | None = None
        self._llm_profile = llm_profile
        self._quantization = quantization
        self._router: LLMRouter | None = None
        self._llm_worker: LLMWorker | None = None

        # Rate limiting
        self._min_interval = 1.0 / rate_limit_hz
        self._last_call_time = 0.0

        # Async processing
        self._pending_context: StructuredContext | None = None
        self._latest_proposal: ProposedGoal | None = None
        self._proposal_lock = threading.Lock()

        # Background worker
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._work_available = threading.Event()
        self._urgent_work_available = threading.Event()

        # Default Network escalation context
        self._dn_escalation_context: dict | None = None

        # Contemplation quality metrics (Phase 2)
        # Maps goal_id → {contemplated: bool, refined: bool, confidence: float}
        self._contemplation_log: dict[str, dict[str, Any]] = {}
        self._contemplation_stats = {
            "contemplated_success": 0,
            "contemplated_total": 0,
            "uncontemplated_success": 0,
            "uncontemplated_total": 0,
        }
        self._nac: Any = None  # Late-wired via wire_nac()

        # Phase 3b: Acute staging
        self._staging_dir: str | None = None  # Set via wire_staging()
        self._weight_learner: SignificanceWeightLearner | None = None
        self._significance_config = SignificanceConfig()
        self._hippocampus: Any = None  # Late-wired via wire_staging()
        self._scn: Any = None  # Late-wired via wire_staging()

        # Subscribe to percepts and filtered percepts (from Default Network)
        self._bus.subscribe(Percept, self._on_percept)
        self._bus.subscribe(FilteredPercept, self._on_filtered_percept)
        self._bus.subscribe(GoalCompleted, self._on_goal_completed)

    def _ensure_llm(self) -> ChatLLMAgent | None:
        """Lazy initialization of LLM."""
        if self._llm is None:
            try:
                config = LLMAgentConfig(
                    profile=self._llm_profile,
                    quantization=self._quantization,
                    system_prompt=self._system_prompt,
                    temperature=0.3,
                    max_tokens=1024,
                )
                self._llm = ChatLLMAgent(config=config)
            except Exception as e:
                warn("ExecAgent: Failed to init LLM: %s", e)
                return None
        return self._llm

    def _resolve_profile_override(self, cfg: Any) -> str | None:
        profiles = getattr(cfg, "agent_profiles", {})
        if not isinstance(profiles, dict):
            return None
        for key in ("exec_agent", "executive"):
            value = profiles.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _ensure_router(self) -> LLMRouter | None:
        if self._router is not None:
            return self._router
        try:
            cfg = load_llm_config()
            profile_override = self._resolve_profile_override(cfg)
            if profile_override:
                cfg = load_llm_config(profile_override=profile_override)
        except Exception as e:
            warn("ExecAgent: Failed to load LLM config: %s", e)
            return None
        if not cfg.enabled:
            return None
        self._router = LLMRouter(cfg)
        return self._router

    def _ensure_llm_worker(self) -> LLMWorker | None:
        if self._llm_worker is not None:
            return self._llm_worker
        router = self._ensure_router()
        if router is None:
            return None
        try:
            token_counter = router.get_token_counter() if hasattr(router, "get_token_counter") else None
            llm_worker = LLMWorker(
                router,
                n_ctx=getattr(router, "n_ctx", 4096),
                token_counter=token_counter,
            )
            llm_worker.start()
            self._llm_worker = llm_worker
        except Exception as e:
            warn("ExecAgent: Failed to init LLMWorker: %s", e)
            return None
        return self._llm_worker

    def _get_prompt_profile_name(self, router: LLMRouter) -> str | None:
        profiles = getattr(router.cfg, "prompt_profiles", {})
        if isinstance(profiles, dict):
            if "exec_agent" in profiles:
                return "exec_agent"
            if "executive" in profiles:
                return "executive"
        return None

    def _get_tool_definitions(self, router: LLMRouter) -> list[dict[str, Any]] | None:
        """Return Claude tool definitions if the current provider supports native tool use."""
        try:
            preview = router.preview_provider(
                system=self._system_prompt[:200],
                user="test",
                temperature=0.2,
                max_tokens=512,
            )
            if not preview.get("is_cloud"):
                return None
            provider_key = preview.get("provider", "")
            providers = router.get_provider_configs()
            provider_cfg = providers.get(provider_key, {})
            provider_type = str(provider_cfg.get("type", "")).strip().lower()
            if provider_type in ("anthropic", "claude"):
                from maxim.models.language.anthropic_backend import PROPOSED_GOAL_TOOL
                return [PROPOSED_GOAL_TOOL]
        except Exception:
            pass
        return None

    def _get_thinking_config(self, router: LLMRouter) -> dict[str, Any] | None:
        """Return extended thinking config if enabled for the current provider."""
        try:
            providers = router.get_provider_configs()
            for cfg in providers.values():
                provider_type = str(cfg.get("type", "")).strip().lower()
                if provider_type in ("anthropic", "claude"):
                    thinking = cfg.get("thinking")
                    if isinstance(thinking, dict) and thinking.get("enabled"):
                        budget = int(thinking.get("budget_tokens", 5000))
                        budget = max(1024, min(budget, 20000))
                        return {"budget_tokens": budget}
        except Exception:
            pass
        return None

    # ── Contemplation loop ─────────────────────────────────────────────

    def _contemplation_config(self) -> dict[str, Any]:
        """Read contemplation config from LLMConfig, with sane defaults.

        If adaptive thresholds are enabled and enough NAc observations exist,
        ``confidence_threshold`` and ``min_sub_goals_to_trigger`` are adjusted
        based on learned contemplation outcomes (Phase 3).
        """
        defaults: dict[str, Any] = {
            "enabled": True,
            "confidence_threshold": 0.7,
            "min_sub_goals_to_trigger": 2,
            "trigger_on_high_priority": True,
            "max_passes": 3,
            "critique_max_tokens": 384,
            "refine_max_tokens": 512,
            "mode": "standard",
            "fast_max_tokens": 640,
            "adaptive_enabled": True,
            "adaptive_min_observations": 10,
            "adaptive_confidence_floor": 0.3,
            "adaptive_confidence_ceiling": 0.95,
            "adaptive_min_sub_goals_floor": 1,
            "adaptive_min_sub_goals_ceiling": 5,
        }
        router = self._router
        if router is not None:
            raw = dict(getattr(router.cfg, "contemplation", ()) or ())
            defaults.update(raw)

        # Phase 3: Apply adaptive adjustments from NAc learned outcomes
        if defaults.get("adaptive_enabled", True):
            adjustments = self._adaptive_thresholds(defaults)
            if adjustments is not None:
                defaults.update(adjustments)

        return defaults

    def _adaptive_thresholds(self, cfg: dict[str, Any]) -> dict[str, Any] | None:
        """Compute adaptive threshold adjustments from NAc learned outcomes.

        Queries NAc for contemplation:refined and contemplation:draft outcomes,
        compares success rates, and shifts ``confidence_threshold`` and
        ``min_sub_goals_to_trigger`` accordingly.

        Returns adjusted keys or ``None`` if insufficient data.
        """
        nac = self._nac
        if nac is None:
            return None

        min_obs = int(cfg.get("adaptive_min_observations", 10))

        try:
            refined_links = nac.get_links_for_event("contemplation:refined")
            draft_links = nac.get_links_for_event("contemplation:draft")
        except Exception:
            return None

        # Count total observations across all links for each signature
        refined_obs = sum(link.observation_count for link in refined_links)
        draft_obs = sum(link.observation_count for link in draft_links)

        if refined_obs + draft_obs < min_obs:
            return None  # Not enough data to adapt

        # Compute weighted success rates from predicted_value (Rescorla-Wagner)
        # predicted_value is 0-1 where higher = more positive outcomes observed
        def _weighted_success(links: list) -> float | None:
            total_obs = sum(link.observation_count for link in links)
            if total_obs == 0:
                return None
            # Weight each link's predicted_value by its observation count
            return sum(
                link.predicted_value * link.observation_count for link in links
            ) / total_obs

        refined_rate = _weighted_success(refined_links)
        draft_rate = _weighted_success(draft_links)

        if refined_rate is None and draft_rate is None:
            return None

        adjustments: dict[str, Any] = {}

        # Determine improvement delta: how much does contemplation help?
        # Positive = contemplation helps, negative = contemplation hurts
        if refined_rate is not None and draft_rate is not None:
            improvement = refined_rate - draft_rate
        elif refined_rate is not None:
            # Only refined data: if success rate is high, contemplation helps
            improvement = refined_rate - 0.5
        else:
            # Only draft data: if draft success rate is high, no need for contemplation
            improvement = 0.5 - (draft_rate or 0.5)

        # Adjust confidence_threshold based on improvement
        # Positive improvement → lower threshold (contemplate more often)
        # Negative improvement → raise threshold (contemplate less often)
        base_threshold = float(cfg.get("confidence_threshold", 0.7))
        floor = float(cfg.get("adaptive_confidence_floor", 0.3))
        ceiling = float(cfg.get("adaptive_confidence_ceiling", 0.95))

        # Scale: ±0.2 shift per 0.2 improvement delta, clamped to bounds
        threshold_shift = -improvement  # Lower threshold when improvement is positive
        new_threshold = base_threshold + threshold_shift
        new_threshold = max(floor, min(ceiling, new_threshold))
        adjustments["confidence_threshold"] = round(new_threshold, 3)

        # Adjust min_sub_goals_to_trigger
        base_min_sg = int(cfg.get("min_sub_goals_to_trigger", 2))
        sg_floor = int(cfg.get("adaptive_min_sub_goals_floor", 1))
        sg_ceiling = int(cfg.get("adaptive_min_sub_goals_ceiling", 5))

        if improvement > 0.1:
            # Contemplation helps — loosen gate (lower min sub_goals)
            new_min_sg = base_min_sg - 1
        elif improvement < -0.1:
            # Contemplation hurts — tighten gate (raise min sub_goals)
            new_min_sg = base_min_sg + 1
        else:
            new_min_sg = base_min_sg

        new_min_sg = max(sg_floor, min(sg_ceiling, new_min_sg))
        adjustments["min_sub_goals_to_trigger"] = new_min_sg

        log_structured(
            self.log,
            logging.DEBUG,
            "contemplation_adaptive",
            {
                "refined_rate": refined_rate,
                "draft_rate": draft_rate,
                "improvement": round(improvement, 3),
                "threshold": adjustments["confidence_threshold"],
                "min_sub_goals": adjustments["min_sub_goals_to_trigger"],
                "refined_obs": refined_obs,
                "draft_obs": draft_obs,
            },
        )

        return adjustments

    @staticmethod
    def _safe_sub_goal_count(response: dict) -> int:
        """Safely count sub_goals, handling None, missing key, non-list."""
        sub_goals = response.get("sub_goals")
        if not isinstance(sub_goals, list):
            return 0
        return len(sub_goals)

    def _should_contemplate(self, response: dict) -> bool:
        """Complexity gate: only contemplate plans that warrant it."""
        if response.get("_timeout") or not response.get("goal_description"):
            return False
        cfg = self._contemplation_config()
        if not cfg.get("enabled", True):
            return False
        priority = str(response.get("priority", "")).upper()
        if priority in ("IDLE",):
            return False
        if cfg.get("trigger_on_high_priority", True) and priority in ("HIGH", "CRITICAL"):
            return True
        min_sg = int(cfg.get("min_sub_goals_to_trigger", 2))
        return self._safe_sub_goal_count(response) >= min_sg

    def _contemplation_llm_call(
        self, *, system: str, user: str, max_tokens: int
    ) -> dict[str, Any] | None:
        """Route a contemplation LLM call through the best available backend."""
        llm_worker = self._llm_worker
        router = self._router

        if llm_worker is not None:
            return llm_worker.generate_json_direct(
                system=system,
                user=user,
                temperature=0.2,
                max_tokens=max_tokens,
                request_id=f"contemplate-{uuid.uuid4()}",
                agent_name=self.agent_name,
            )
        elif router is not None:
            return router.generate_json(
                user,
                temperature=0.2,
                max_tokens=max_tokens,
                system_override=system,
                request_context={"agent": self.agent_name, "lane": "infer"},
            )
        else:
            llm = self._ensure_llm()
            if llm is not None:
                return llm.generate_json(
                    user, system_prompt=system, temperature=0.2, max_tokens=max_tokens
                )
        return None

    def _critique_plan(
        self, draft: dict, ctx: StructuredContext
    ) -> dict[str, Any] | None:
        """Pass 2: critique the draft plan. Returns critique dict or None on failure."""
        cfg = self._contemplation_config()
        draft_json = json.dumps(draft, indent=2, default=str)

        critique_user = CRITIQUE_USER_TEMPLATE.format(
            draft_json=draft_json,
            root_goal=ctx.root_goal,
            mode=ctx.mode,
            active_goal=ctx.active_goal,
        )

        try:
            response = self._contemplation_llm_call(
                system=CRITIQUE_SYSTEM,
                user=critique_user,
                max_tokens=int(cfg.get("critique_max_tokens", 384)),
            )
            if not isinstance(response, dict) or "confidence" not in response:
                return None
            return response
        except Exception:
            return None

    def _refine_plan(
        self, draft: dict, critique: dict, ctx: StructuredContext
    ) -> dict[str, Any] | None:
        """Pass 3: refine the plan based on critique. Returns revised plan or None."""
        cfg = self._contemplation_config()
        draft_json = json.dumps(draft, indent=2, default=str)
        issues = critique.get("issues", [])
        suggestions = critique.get("suggestions", [])

        refine_user = REFINE_USER_TEMPLATE.format(
            draft_json=draft_json,
            issues_text="\n".join(f"- {i}" for i in issues),
            suggestions_text="\n".join(f"- {s}" for s in suggestions),
            root_goal=ctx.root_goal,
            mode=ctx.mode,
        )

        try:
            response = self._contemplation_llm_call(
                system=REFINE_SYSTEM,
                user=refine_user,
                max_tokens=int(cfg.get("refine_max_tokens", 512)),
            )
            if not isinstance(response, dict) or not response.get("goal_description"):
                return None
            return response
        except Exception:
            return None

    def _contemplate(self, draft: dict, ctx: StructuredContext) -> dict:
        """Run contemplation loop. Dispatches to standard or fast mode.

        Standard mode (default): critique → confidence gate → optional refine (3 passes max).
        Fast mode: single combined critique+refine call (2 passes max).

        Returns best available plan. Any failure returns the original draft.
        """
        cfg = self._contemplation_config()
        mode = str(cfg.get("mode", "standard")).lower()

        if mode == "fast":
            return self._contemplate_fast(draft, ctx, cfg)
        return self._contemplate_standard(draft, ctx, cfg)

    def _contemplate_standard(self, draft: dict, ctx: StructuredContext,
                              cfg: dict[str, Any]) -> dict:
        """Standard contemplation: separate critique and refine passes."""
        # Preemption check before critique — only urgent percepts interrupt
        if self._urgent_work_available.is_set():
            return draft

        critique = self._critique_plan(draft, ctx)
        if critique is None:
            return draft

        threshold = float(cfg.get("confidence_threshold", 0.7))
        try:
            confidence = float(critique.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0

        if confidence >= threshold:
            return draft

        # Preemption check before refine — only urgent percepts interrupt
        if self._urgent_work_available.is_set():
            return draft

        refined = self._refine_plan(draft, critique, ctx)
        if refined is None:
            return draft

        log_structured(
            self.log,
            logging.INFO,
            "contemplation_refined",
            {
                "mode": "standard",
                "critique_confidence": confidence,
                "issues": critique.get("issues", []),
            },
        )

        return refined

    def _contemplate_fast(self, draft: dict, ctx: StructuredContext,
                          cfg: dict[str, Any]) -> dict:
        """Fast contemplation: combined critique+refine in a single LLM call.

        Returns the improved plan if confidence is below threshold,
        otherwise returns the original draft. Any failure returns draft.
        """
        # Preemption check — only urgent percepts interrupt
        if self._urgent_work_available.is_set():
            return draft

        draft_json = json.dumps(draft, indent=2, default=str)

        fast_user = FAST_CONTEMPLATE_USER_TEMPLATE.format(
            draft_json=draft_json,
            root_goal=ctx.root_goal,
            mode=ctx.mode,
            active_goal=ctx.active_goal,
        )

        try:
            response = self._contemplation_llm_call(
                system=FAST_CONTEMPLATE_SYSTEM,
                user=fast_user,
                max_tokens=int(cfg.get("fast_max_tokens", 640)),
            )
        except Exception:
            return draft

        if not isinstance(response, dict) or "confidence" not in response:
            return draft

        threshold = float(cfg.get("confidence_threshold", 0.7))
        try:
            confidence = float(response.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0

        if confidence >= threshold:
            return draft

        # Extract the improved plan from the combined response
        plan = response.get("plan")
        if not isinstance(plan, dict) or not plan.get("goal_description"):
            return draft

        log_structured(
            self.log,
            logging.INFO,
            "contemplation_refined",
            {
                "mode": "fast",
                "critique_confidence": confidence,
                "issues": response.get("issues", []),
            },
        )

        return plan

    # ── Contemplation quality metrics (Phase 2) ──────────────────────

    def wire_nac(self, nac: Any) -> None:
        """Late-wire NAc for contemplation outcome learning."""
        self._nac = nac

    def wire_staging(
        self,
        staging_dir: str,
        hippocampus: Any,
        scn: Any,
        weights_path: str | None = None,
    ) -> None:
        """Late-wire acute staging dependencies.

        Args:
            staging_dir: Path to data/short_term_memory/ for JSON sidecars
            hippocampus: Hippocampus instance for memory capture tracking
            scn: SCN instance for temporal signatures
            weights_path: Path for persisting learnable significance weights
        """
        self._staging_dir = staging_dir
        self._hippocampus = hippocampus
        self._scn = scn
        os.makedirs(staging_dir, exist_ok=True)

        if weights_path is None:
            weights_path = os.path.join(
                os.path.dirname(staging_dir), "util", "significance_weights.json"
            )
        self._weight_learner = SignificanceWeightLearner(weights_path, hippocampus)
        self._weight_learner.load_weights_into_heuristics(
            self._significance_config.heuristics
        )

    def _evaluate_staging(
        self,
        completed: GoalCompleted,
        percept: Percept | None = None,
    ) -> None:
        """Evaluate significance of a completed cycle and stage if above threshold.

        Writes a JSON sidecar to ``_staging_dir`` if the weighted significance
        score exceeds the staging threshold. Called from ``_on_goal_completed``.
        """
        if self._weight_learner is None or self._staging_dir is None:
            return

        # Build CycleContext from available signals
        rpe_raw = 0.0
        nac = self._nac
        if nac is not None:
            try:
                predicted = getattr(nac, "last_predicted_valence", 0.5)
                actual = 1.0 if completed.success else 0.0
                rpe_raw = abs(predicted - actual)
            except Exception:
                pass

        has_user = False
        novelty = 0.5
        if percept is not None:
            has_user = percept.source == "cli" or percept.has_maxim_keyword
            novelty = percept.novelty

        ctx = CycleContext(
            rpe_raw=rpe_raw,
            has_user_input=has_user,
            novelty=novelty,
            is_plan_boundary=False,
            energy_crossed_threshold=False,
            outcome_valence=1.0 if completed.success else 0.0,
        )

        # Update RPE stats
        if rpe_raw > 0:
            self._weight_learner.update_rpe_stats(rpe_raw)

        # Evaluate
        score, heuristic_scores = self._weight_learner.evaluate(
            ctx, self._significance_config
        )

        if score < self._significance_config.staging_threshold:
            return

        # Write JSON sidecar
        now = time.time()
        tool_name = completed.tool_name if hasattr(completed, "tool_name") else "unknown"

        # Compute NAc observation count at staging time
        nac_obs = 0
        if nac is not None and tool_name != "unknown":
            try:
                links = nac.get_links_for_event(tool_name)
                nac_obs = sum(getattr(link, "observation_count", 0) for link in links)
            except Exception:
                pass

        # Get temporal info
        temporal = {}
        if self._scn is not None:
            try:
                from maxim.time.temporal_signature import TemporalSignature

                sig = TemporalSignature.now()
                temporal = {
                    "circadian_phase": sig.circadian_phase,
                    "weekly_phase": sig.weekly_phase,
                    "monthly_phase": sig.monthly_phase,
                    "annual_phase": sig.annual_phase,
                }
                anomaly = self._scn.temporal_anomaly_score(sig)
                if anomaly is not None:
                    temporal["anomaly_score"] = anomaly
            except Exception:
                pass

        sidecar = {
            "version": "1.0",
            "timestamp": now,
            "staging_path": "acute",
            "significance_score": round(score, 4),
            "heuristic_scores": {k: round(v, 4) for k, v in heuristic_scores.items()},
            "rpe": {
                "raw": round(rpe_raw, 4),
                "running_mean": round(self._weight_learner.rpe_running_mean, 4),
                "running_std": round(self._weight_learner.rpe_running_std, 4),
            },
            "temporal": temporal,
            "nac_obs_at_staging": nac_obs,
            "context_summary": completed.description if hasattr(completed, "description") else "",
            "action": {"tool": tool_name},
            "outcome": {
                "success": completed.success,
                "valence": 1.0 if completed.success else 0.0,
            },
            "active_goal": completed.goal_id,
            "sleep_cycles_seen": 0,
            "wave_scores": [],
        }

        # Write sidecar
        label = tool_name.replace(" ", "-")[:30]
        filename = f"{int(now)}_rpe-{rpe_raw:.2f}_{label}.json"
        filepath = os.path.join(self._staging_dir, filename)
        try:
            tmp = filepath + ".tmp"
            with open(tmp, "w") as f:
                json.dump(sidecar, f, indent=2)
            os.replace(tmp, filepath)
        except Exception as e:
            self.log.warning("Failed to write staging sidecar: %s", e)

        log_structured(
            self.log,
            logging.DEBUG,
            "acute_staging",
            {"score": score, "file": filename},
        )

    def _on_goal_completed(self, completed: GoalCompleted) -> None:
        """Track contemplation outcomes when goals complete."""
        meta = self._contemplation_log.pop(completed.goal_id, None)
        if meta is None:
            return  # Goal not from this agent or already expired

        contemplated = meta.get("contemplated", False)
        if contemplated:
            self._contemplation_stats["contemplated_total"] += 1
            if completed.success:
                self._contemplation_stats["contemplated_success"] += 1
        else:
            self._contemplation_stats["uncontemplated_total"] += 1
            if completed.success:
                self._contemplation_stats["uncontemplated_success"] += 1

        # Feed outcome to NAc for causal learning
        nac = self._nac
        if nac is not None:
            try:
                from maxim.decisions.causal_link import Valence

                event_sig = "contemplation:refined" if meta.get("refined") else "contemplation:draft"
                valence = Valence.POSITIVE if completed.success else Valence.NEGATIVE
                delta = time.time() - meta.get("timestamp", time.time())
                nac.observe(
                    event_type="contemplation",
                    event_signature=event_sig,
                    outcome_type="goal_result",
                    outcome_signature=f"goal:{completed.goal_id}:{valence.value}",
                    outcome_valence=valence,
                    delta_seconds=max(0.0, delta),
                    context={"contemplated": contemplated, "refined": meta.get("refined", False)},
                )
            except Exception:
                pass  # NAc observation is best-effort

        log_structured(
            self.log,
            logging.DEBUG,
            "contemplation_outcome",
            {
                "goal_id": completed.goal_id,
                "success": completed.success,
                "contemplated": contemplated,
                "refined": meta.get("refined", False),
            },
        )

        # Phase 3b: Evaluate acute staging
        try:
            ctx = self._memory.build_context()
            self._evaluate_staging(completed, percept=ctx.current_percept)
        except Exception:
            pass  # Staging is best-effort

    def contemplation_improvement_rate(self) -> dict[str, Any]:
        """Return contemplation quality metrics.

        Returns dict with success rates for contemplated vs uncontemplated
        goals and the improvement delta.
        """
        stats = self._contemplation_stats
        ct = stats["contemplated_total"]
        ut = stats["uncontemplated_total"]

        c_rate = stats["contemplated_success"] / ct if ct > 0 else None
        u_rate = stats["uncontemplated_success"] / ut if ut > 0 else None

        improvement = None
        if c_rate is not None and u_rate is not None:
            improvement = c_rate - u_rate

        return {
            "contemplated_success_rate": c_rate,
            "contemplated_total": ct,
            "uncontemplated_success_rate": u_rate,
            "uncontemplated_total": ut,
            "improvement_delta": improvement,
        }

    # ── Phase 3f: recall_deep tool ────────────────────────────────────

    def recall_deep(
        self,
        query: str,
        percept: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[Any]:
        """Deep recall tool for ExecAgent. Wraps existing recall systems
        with broader search parameters.

        Ranks purely by context similarity — NOT recency. The agent is
        asking "what do I know about X?" not "what happened recently?"

        Search strategy: LSH seeds → recall_associated() spreading
        activation → recall_similar fallback.
        """
        if self._hippocampus is None:
            return []

        scored: dict[str, tuple[Any, float]] = {}

        # 1. LSH candidates from context_index
        context_index = getattr(self, "_context_index_ref", None)
        if query and context_index is not None:
            try:
                for mid, sim in context_index.query_similar(query, min_similarity=0.2):
                    mem = self._hippocampus.get(mid)
                    if mem:
                        scored[mid] = (mem, sim)
            except Exception:
                pass

        # LSH candidates from percept_index
        percept_index = getattr(self, "_percept_index_ref", None)
        if percept and percept_index is not None:
            try:
                from maxim.memory.context_index import percept_to_canonical

                percept_str = percept_to_canonical(percept)
                for mid, sim in percept_index.query_similar(percept_str, min_similarity=0.2):
                    if mid not in scored:
                        mem = self._hippocampus.get(mid)
                        if mem:
                            scored[mid] = (mem, sim)
            except Exception:
                pass

        # 2. Spreading activation from top LSH matches
        if scored:
            try:
                seeds = sorted(scored.values(), key=lambda x: -x[1])[:3]
                for seed_mem, _ in seeds:
                    associated = self._hippocampus.recall_associated(
                        seed_ids=[seed_mem.id], max_depth=2, decay=0.5
                    )
                    for mem, activation in associated:
                        if mem.id not in scored:
                            scored[mem.id] = (mem, activation)
            except Exception:
                pass

        # 3. Fall back to recall_similar if still under limit
        if len(scored) < limit:
            try:
                similar = self._hippocampus.recall_similar(
                    query, limit=limit - len(scored)
                )
                for mem in similar:
                    if mem.id not in scored:
                        scored[mem.id] = (mem, 0.3)
            except Exception:
                pass

        # Rank by similarity score descending — NOT by time
        ranked = sorted(scored.values(), key=lambda x: -x[1])
        return [mem for mem, _ in ranked[:limit]]

    def _on_percept(self, percept: Percept) -> None:
        """Trigger goal proposal on significant percepts.

        Note: When Default Network is active, most percepts are filtered
        and only FilteredPercepts reach ExecAgent via _on_filtered_percept.
        This handler remains for backwards compatibility and direct percepts.
        """
        if percept.source == "cli":
            self._trigger_proposal(urgent=True)
            return

        # Always propose on inbound communications (SMS, voice call, etc.)
        if percept.source.startswith("comms:"):
            self._trigger_proposal(urgent=True)
            return

        # Always propose on voice commands
        if percept.has_maxim_keyword:
            self._trigger_proposal(urgent=True)
            return

        # Propose on high salience + novelty (not urgent — can wait for contemplation)
        if percept.salience > 0.5 and percept.novelty > 0.5:
            self._trigger_proposal(urgent=False)

    def _on_filtered_percept(self, filtered: FilteredPercept) -> None:
        """Handle percepts escalated by the Default Network.

        FilteredPercepts have already passed the ThalamicGate and are
        deemed worthy of deliberation. Always trigger a proposal.
        """
        # Log that we received an escalated percept
        log_structured(
            self.log,
            logging.DEBUG,
            "filtered_percept_received",
            {
                "reason": filtered.escalation_reason,
                "urgency": filtered.urgency,
                "source": filtered.original.source if filtered.original else "unknown",
            },
        )

        # Store escalation context for use in goal proposal
        with self._proposal_lock:
            self._pending_context = None  # Clear any stale context
            self._dn_escalation_context = {
                "reason": filtered.escalation_reason,
                "urgency": filtered.urgency,
                "dn_context": filtered.dn_context,
            }

        # Trigger proposal — mark as urgent if high-urgency escalation
        self._trigger_proposal(urgent=filtered.urgency >= 0.7)

    def _trigger_proposal(self, urgent: bool = False) -> None:
        """Signal that goal proposal is needed.

        Args:
            urgent: If True, also signals urgent preemption which interrupts
                    active contemplation. Use for CLI input, comms, voice
                    commands, and high-urgency escalated percepts.
        """
        self._work_available.set()
        if urgent:
            self._urgent_work_available.set()

    def _build_llm_context(self, ctx: StructuredContext, budget_context: str | None = None) -> str:
        """Build prompt from structured context."""
        # Current percept summary
        percept_str = "(none)"
        if ctx.current_percept:
            p = ctx.current_percept
            percept_str = f"""
- Source: {p.source}
- Salience: {p.salience:.2f}
- Novelty: {p.novelty:.2f}
- Voice command: {p.raw_transcript_text or "(none)"}
- Message: {p.content or "(none)"}
- Has "maxim" keyword: {p.has_maxim_keyword}
- Hard override: {p.hard_override or "(none)"}"""

        # Detected objects
        obj_lines = []
        for obj in ctx.detected_objects[:5]:
            if not isinstance(obj, dict):
                continue
            obj_lines.append(
                f"  - class={obj.get('class_id')}, conf={obj.get('conf', 0):.2f}"
            )
        obj_str = "\n".join(obj_lines) if obj_lines else "  (none)"

        # People
        people_str = (
            f"{len(ctx.detected_people)} people detected"
            if ctx.detected_people
            else "No people detected"
        )

        # Recent speech
        speech_str = "\n  ".join(ctx.detected_speech) if ctx.detected_speech else "(none)"

        # Recent outcomes
        outcome_lines = []
        for out in ctx.recent_outcomes[-3:]:
            if not isinstance(out, dict):
                continue
            status = "OK" if out.get("success") else f"FAILED: {out.get('error')}"
            outcome_lines.append(
                f"  - {out.get('tool_name', out.get('goal_id', '?'))}: {status}"
            )
        outcome_str = "\n".join(outcome_lines) if outcome_lines else "  (none)"

        # Relevant memories (dict format after Phase 0 unification)
        mem_lines = []
        for mem in ctx.relevant_memories[:3]:
            if isinstance(mem, dict):
                src = mem.get("source", "?")
                sal = mem.get("salience", 0.0)
                mem_lines.append(f"  - [{src}] salience={sal:.2f}")
            else:
                mem_lines.append(f"  - [{getattr(mem, 'source', '?')}] salience={getattr(mem, 'salience', 0.0):.2f}")
        mem_str = "\n".join(mem_lines) if mem_lines else "  (none)"

        # CLI inputs
        cli_str = (
            "\n  ".join(ctx.cli_inputs[-3:]) if ctx.cli_inputs else "(none)"
        )

        # Comms messages (inbound SMS/voice)
        comms_lines = []
        for msg in ctx.comms_messages[-5:]:
            prefix = "User" if msg.get("direction") == "inbound" else "Maxim"
            channel = msg.get("channel", "sms")
            sender = msg.get("sender", "unknown")
            comms_lines.append(
                f"  [{prefix} via {channel} {sender}] {msg.get('content', '')}"
            )
        comms_str = "\n".join(comms_lines) if comms_lines else "(none)"

        # DN escalation context (if percept was escalated by Default Network)
        dn_str = ""
        with self._proposal_lock:
            if self._dn_escalation_context:
                dn_ctx = self._dn_escalation_context
                dn_str = f"""

DEFAULT NETWORK ESCALATION:
- Reason: {dn_ctx.get('reason', 'unknown')}
- Urgency: {dn_ctx.get('urgency', 0.5):.2f}
- Context: {dn_ctx.get('dn_context', {})}
(This percept was pre-filtered by the reactive layer and deemed worthy of deliberation)"""
                # Clear after use
                self._dn_escalation_context = None

        # Working notes (persistent LLM self-context)
        notes_str = ""
        if ctx.working_notes:
            notes_str = f"""

WORKING NOTES (your persistent scratchpad — edit via write_file):
{ctx.working_notes}"""
        else:
            notes_str = (
                "\n\nWORKING NOTES: (empty — use write_file to "
                "'.maxim_workspace/notes/context.md' to pin important context)"
            )

        # Workspace file inventory
        workspace_str = ""
        if ctx.workspace_files:
            ws_lines = ["\n\nWORKSPACE (.maxim_workspace/):"]
            for wf in ctx.workspace_files:
                size = wf.get("size", 0)
                if size >= 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size} bytes"
                mod_t = time.localtime(wf.get("modified", 0))
                mod_str = f"{mod_t.tm_year}-{mod_t.tm_mon:02d}-{mod_t.tm_mday:02d}"
                ws_lines.append(f"  - {wf['path']} ({size_str}, modified {mod_str})")
            workspace_str = "\n".join(ws_lines)

        # Statistical patterns (from StatisticianAgent via bus)
        stat_str = ""
        if ctx.statistical_context and ctx.active_pattern_count > 0:
            stat_str = f"""

STATISTICAL PATTERNS ({ctx.active_pattern_count} active):
{ctx.statistical_context}"""

            # Append actionable suggestions if available
            suggestions = getattr(ctx, "statistical_suggestions", [])
            if suggestions:
                suggestion_lines = ["\nSUGGESTED ANALYSES:"]
                for s in suggestions[:3]:
                    suggestion_lines.append(
                        f"  -> {s.get('tool_call', 'math')} {s.get('operation', '?')} "
                        f"on {s.get('metric', '?')} ({s.get('data_type', '?')}, "
                        f"{s.get('fsm_state', '?')}): {s.get('rationale', '')}"
                    )
                stat_str += "\n".join(suggestion_lines)

        budget_str = f"\n\n{budget_context}" if budget_context else ""

        return f"""ROOT GOAL: {ctx.root_goal}

CURRENT STATE:
- Mode: {ctx.mode}
- Active goal: {ctx.active_goal or "(none)"}

CURRENT PERCEPT:{percept_str}

DETECTED OBJECTS:
{obj_str}

PEOPLE: {people_str}

RECENT SPEECH:
  {speech_str}

RECENT CLI INPUTS:
  {cli_str}

COMMS MESSAGES (SMS/Voice):
  {comms_str}

RECENT OUTCOMES:
{outcome_str}

RELEVANT MEMORIES:
{mem_str}{notes_str}{workspace_str}{stat_str}{dn_str}{budget_str}

Based on this context, what goal should be proposed?"""

    def _propose_goal(self, ctx: StructuredContext) -> ProposedGoal | None:
        """Use LLM to propose a goal."""
        # Handle hard overrides without LLM
        if ctx.current_percept and ctx.current_percept.hard_override:
            override = ctx.current_percept.hard_override
            return ProposedGoal(
                id=str(uuid.uuid4()),
                description=f"Execute hard override: {override}",
                priority=GoalPriority.CRITICAL,
                tool_name="maxim_command",
                tool_params={"command": override},
                reasoning="User explicitly requested mode change",
                confidence=1.0,
            )

        # Handle sleep mode
        if ctx.mode == "sleep":
            return None

        # Rate limit
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_time = time.time()

        try:
            response = None
            llm_worker = self._ensure_llm_worker()
            router = self._router or self._ensure_router()

            budget_context = llm_worker.get_budget_context() if llm_worker else ""
            prompt = self._build_llm_context(ctx, budget_context=budget_context)

            prompt_builder = ExecutivePrompt(self._system_prompt)
            prompt_profile = None
            if router is not None:
                profile_name = self._get_prompt_profile_name(router)
                prompt_profile = load_prompt_profile(router.cfg.prompt_profiles, profile_name)
            injected = prompt_builder.inject(prompt, prompt_profile)

            # Determine if cloud backend supports native tool use
            tool_use_tools = None
            thinking_cfg = None
            if router is not None and router.cloud_allowed():
                tool_use_tools = self._get_tool_definitions(router)
                thinking_cfg = self._get_thinking_config(router)

            if llm_worker is not None:
                request_id = f"exec-{uuid.uuid4()}"
                response = llm_worker.generate_json_direct(
                    system=injected.system,
                    user=injected.user,
                    temperature=0.2,
                    max_tokens=512,
                    request_id=request_id,
                    agent_name=self.agent_name,
                    tools=tool_use_tools,
                    thinking=thinking_cfg,
                )
            elif router is not None:
                request_id = f"exec-{uuid.uuid4()}"
                response = router.generate_json(
                    injected.user,
                    temperature=0.2,
                    system_override=injected.system,
                    request_context={"agent": self.agent_name, "request_id": request_id, "lane": "infer"},
                    tools=tool_use_tools,
                    thinking=thinking_cfg,
                )
            else:
                llm = self._ensure_llm()
                if llm is None:
                    return None
                response = llm.generate_json(prompt, temperature=0.2)

            if not response or not isinstance(response, dict) or not response.get("goal_description"):
                return None

            # Contemplation: critique + refine for complex plans on non-thinking providers
            contemplated = False
            refined = False
            if (
                not thinking_cfg
                and self._should_contemplate(response)
            ):
                draft = response
                response = self._contemplate(response, ctx)
                contemplated = True
                refined = response is not draft

            priority_str = response.get("priority", "MEDIUM").upper()
            try:
                priority = GoalPriority[priority_str]
            except KeyError:
                priority = GoalPriority.MEDIUM

            # Build sub-goals if provided
            sub_goals: list[SubGoal] = []
            for i, sg_data in enumerate(response.get("sub_goals", [])):
                if isinstance(sg_data, dict) and sg_data.get("tool_name"):
                    sub_goals.append(
                        SubGoal(
                            id=f"{uuid.uuid4()}-sg{i}",
                            description=sg_data.get("description", ""),
                            tool_name=sg_data["tool_name"],
                            tool_params=sg_data.get("tool_params", {}),
                        )
                    )

            goal = ProposedGoal(
                id=str(uuid.uuid4()),
                description=response["goal_description"],
                priority=priority,
                tool_name=response.get("tool_name"),
                tool_params=response.get("tool_params", {}),
                reasoning=response.get("reasoning", ""),
                confidence=0.9,
                parent_goal="root",
                sub_goals=sub_goals,
            )

            # Record contemplation metadata for outcome tracking
            self._contemplation_log[goal.id] = {
                "contemplated": contemplated,
                "refined": refined,
                "timestamp": time.time(),
            }
            # Bound the log to prevent unbounded growth
            if len(self._contemplation_log) > 200:
                oldest = sorted(self._contemplation_log, key=lambda k: self._contemplation_log[k]["timestamp"])
                for k in oldest[:50]:
                    del self._contemplation_log[k]

            # Log to abstraction stream
            log_structured(
                self.log,
                logging.INFO,
                "goal_proposed",
                {
                    "goal_id": goal.id,
                    "description": goal.description,
                    "priority": goal.priority.name,
                    "contemplated": contemplated,
                },
            )

            return goal

        except Exception as e:
            warn("ExecAgent: LLM error: %s", e)
            return None

    def _worker_loop(self) -> None:
        """Background worker for goal proposal."""
        while not self._stop_event.is_set():
            triggered = self._work_available.wait(timeout=0.5)
            if not triggered:
                continue
            self._work_available.clear()
            self._urgent_work_available.clear()

            if self._stop_event.is_set():
                break

            # Build context and propose
            ctx = self._memory.build_context()
            proposal = self._propose_goal(ctx)

            if proposal:
                with self._proposal_lock:
                    self._latest_proposal = proposal
                self._bus.publish(proposal)

    def get_latest_proposal(self) -> ProposedGoal | None:
        """Get and clear the latest proposal."""
        with self._proposal_lock:
            proposal = self._latest_proposal
            self._latest_proposal = None
            return proposal

    def propose_intent(self, state: Any, memory: Any, **kwargs: Any) -> dict[str, Any] | None:
        """Check for proposals and return as intent."""
        # Check startup
        if self._memory.check_startup():
            return {
                "goal": "read_readme",
                "confidence": 0.5,
                "source": "startup",
            }

        # Check for file changes
        ctx = self._memory.build_context()
        if ctx.current_percept and ctx.current_percept.file_changed:
            return {
                "goal": "read_latest_transcript",
                "confidence": 1.0,
                "source": "file_change",
            }

        # Trigger proposal if needed
        if ctx.current_percept:
            if ctx.current_percept.has_maxim_keyword or (
                ctx.current_percept.salience > 0.5 and ctx.current_percept.novelty > 0.5
            ) or ctx.current_percept.source == "cli":
                self._trigger_proposal()

        # Check for ready proposal
        proposal = self.get_latest_proposal()
        if proposal and proposal.tool_name:
            return {
                "proposed_goal": proposal,
                "goal": {"tool_name": proposal.tool_name, "params": proposal.tool_params},
                "confidence": proposal.confidence,
                "source": "exec_agent",
            }

        # Reflection mode: introspective, minimal external activity
        # Only respond to direct address, otherwise remain idle
        if ctx.mode == "reflection":
            return None

        # For other active modes, only propose tracking actions when meaningful
        # This prevents continuous "Action from default_track" spam
        if ctx.mode in ("observe", "live", "train", "agentic", "exploration"):
            # Only track if:
            # 1. There are detected people (always worth tracking)
            # 2. There's high novelty in current percept
            # 3. User directly addressed the system
            has_people = bool(ctx.detected_people)
            has_high_novelty = ctx.current_percept and ctx.current_percept.novelty > 0.5
            has_user_address = ctx.current_percept and ctx.current_percept.has_maxim_keyword

            if has_people:
                return {
                    "goal": {
                        "tool_name": "track_target",
                        "params": {"prefer_people": True, "deadzone_px": 40},
                    },
                    "confidence": 0.6,
                    "source": "default_track",
                }

            if has_high_novelty or has_user_address:
                return {
                    "goal": {"tool_name": "focus_interests", "params": {}},
                    "confidence": 0.5,
                    "source": "default_focus",
                }

            # Otherwise, no action needed - let the system idle
            return None

        return None

    def on_start(self, **kwargs: Any) -> None:
        """Start the background worker."""
        if self._worker is None or not self._worker.is_alive():
            self._stop_event.clear()
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()

    def on_stop(self, **kwargs: Any) -> None:
        """Stop the background worker."""
        self._stop_event.set()
        self._work_available.set()
        self._urgent_work_available.set()
        if self._worker:
            self._worker.join(timeout=2.0)
        self._bus.unsubscribe(Percept, self._on_percept)
        self._bus.unsubscribe(FilteredPercept, self._on_filtered_percept)
        self._bus.unsubscribe(GoalCompleted, self._on_goal_completed)
        if self._llm_worker:
            self._llm_worker.stop()
            self._llm_worker = None
        if self._llm:
            self._llm.on_stop()

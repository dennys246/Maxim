"""ExecAgent: Goal proposal based on structured context.

Uses LLM to propose goals aligned with the root goal:
"Understand reality and help people."
"""

from __future__ import annotations

import json
import logging
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
from maxim.agents.llm_agent import ChatLLMAgent, LLMAgentConfig
from maxim.agents.llm_worker import LLMWorker
from maxim.models.language.router import LLMRouter, load_llm_config
from maxim.prompts.prompt_profiles import ExecutivePrompt, load_prompt_profile
from maxim.agents.memory_agent import MemoryAgent
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

    SYSTEM_PROMPT = """You are Maxim, an intelligent agent with the root goal:
"Understand reality and help people."

You receive structured context about what you perceive (vision detections, speech) and your memories.
Your job is to propose goals that advance understanding or help people.

**IMPORTANT: You do NOT receive raw images. Visual information comes from object detection results only.**

Guidelines:
1. Voice commands from users are HIGH priority - they're asking for help
2. Vision detections help you understand the visual scene (objects, people, positions)
3. When people are detected, use track_target to keep them centered in view (MEDIUM priority)
4. When interesting objects are detected off-center, use track_target to center them
5. Learn from past outcomes (successes and failures) in your memories
6. When uncertain, propose goals that gather more information
7. Idle time can be used for LOW priority learning goals
8. For complex tasks, use goal chaining - break into sub_goals

Statistical reasoning:
9. The STATISTICAL PATTERNS section (when present) contains confirmed patterns detected by IPS
   and Angular Gyrus. Patterns are tagged by type:
   - [PATTERN] = confirmed non-random trend (e.g. declining success rate). Investigate the cause.
   - [TEMPORAL] = unusual for the current time of day/week. May indicate a transient issue.
   - [ANOMALY] = single-point anomaly against recent history. Monitor but don't overreact.
   - [LEARNED] = pattern previously stored in math memory with historical confidence.
10. When a [PATTERN] shows declining tool success, use sub_goals to:
    (a) recall_memory to check for prior known patterns with that tool
    (b) assess_randomness on the recent outcome data to confirm the trend
    (c) If confirmed: store_memory to persist the finding, then propose a corrective action
11. Use internet_search to research unfamiliar statistical patterns or analysis techniques.
    Chain: internet_search → http_fetch → store_memory (category=METHOD) to learn new methods.
12. Statistical goals are typically LOW or MEDIUM priority — never override HIGH/CRITICAL user requests.
    Exception: if a pattern indicates imminent failure (severity > 0.8), escalate to HIGH.

Available tools for goals:
- track_target: Move head to center on detected objects. Params: {deadzone_px: int, duration_s: float, prefer_people: bool}
  Use this to actively track objects/people and keep them centered in view.
- focus_interests: Focus attention on objects in the current frame. Params: {target_class: str (optional)}
- maxim_command: Execute command. Params: {command: str, ...}
  Commands: update_interests, center_vision, mark_trainable_moment,
            label_outcome, request_sleep, request_observe, request_shutdown,
            goto_pose, move, look_at_image, move_antenna
  update_interests params: {add: [int], remove: [int]} — boosts salience priority for specified COCO class IDs
- read_file: Read a file. Params: {path: str}. Workspace files: '.maxim_workspace/filename'
- write_file: Write a file. **MUST use '.maxim_workspace/' prefix!** Params: {path: '.maxim_workspace/filename.py', content: str}
  Workspace structure: drafts/ (code drafts), notes/ (thinking), plans/ (proposals), scratch/ (temp files)
- execute_file: Execute a file (requires MAXIM_ALLOW_EXECUTE_FILE=1). Params: {path: '.maxim_workspace/filename.py'}
- math: Mathematical cognition. Params: {operation: str, ...}
  Approximate (fast): compare {a, b}, trend {data}, anomaly {value, history},
    estimate_sum {data}, estimate_mean {data}, categorize {value}, assess_randomness {data}
  Exact (precise): compute {op_type, operands}, analyze {data, method: descriptive|linear|quadratic|percentiles},
    mat_multiply {a, b}, eigenvalues {matrix}, solve_system {coefficients, constants}, determinant {matrix}
  Aliases: sqrt {value}, square_root {value}, cube_root {value}, squared {value}, cubed {value}, factorial {value}
    (these are automatically normalized to compute with power — e.g. sqrt → power [x, 0.5])
  Multi-step math: Each math operation handles ONE calculation. For compound expressions,
    decompose into sub_goals using the workspace to pass intermediate results:
    Example: "square root of 25 plus 3" →
    sub_goal 1: {"tool_name": "math", "tool_params": {"operation": "sqrt", "value": 25}}
    sub_goal 2: {"tool_name": "math", "tool_params": {"operation": "store_value", "name": "step1", "value": 5}}
    sub_goal 3: {"tool_name": "math", "tool_params": {"operation": "compute", "op_type": "add", "operands": [5, 3]}}
    For unknowable intermediates, store_value the first result, then recall_value in the next step.
    Always decompose following standard order of operations (parentheses, exponents, multiply/divide, add/subtract).
  Workspace: store_value {name, value}, recall_value {name}
  Memory: recall_method {description}, recall_memory {name?, category?, domain?, limit?},
    list_memories {category?, domain?, limit?}, store_memory {name, verbal, code?, category?, domain?, confidence?}
  Categories: FACT, METHOD, PATTERN, CONSTANT, RELATIONSHIP
  Analysis workflow example (use sub_goals):
    1. {"tool_name": "math", "tool_params": {"operation": "recall_memory", "category": "PATTERN", "domain": "operational_statistics"}}
    2. {"tool_name": "math", "tool_params": {"operation": "assess_randomness", "data": [...]}}
    3. {"tool_name": "math", "tool_params": {"operation": "store_memory", "name": "...", "category": "PATTERN", "verbal": "..."}}
- internet_search: Search the web. Params: {query: str, max_results: int}
  Use to research analysis methods, look up mathematical techniques, or find information to help people.
- http_fetch: Fetch a web page. Params: {url: str, extract_text: bool}
  Use to read full content from URLs found via internet_search.

Respond with ONLY valid JSON:
{
    "goal_description": "what you want to achieve",
    "priority": "CRITICAL|HIGH|MEDIUM|LOW|IDLE",
    "tool_name": "...",
    "tool_params": {...},
    "reasoning": "how this serves the root goal",
    "sub_goals": [
        {"description": "...", "tool_name": "...", "tool_params": {...}}
    ]
}

If no goal needed: {"goal_description": null, "priority": "IDLE"}
"""

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
                    max_tokens=512,
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

        critique_system = "You are evaluating a proposed action plan. Be concise and specific."
        critique_user = (
            f"You previously proposed this plan:\n{draft_json}\n\n"
            f"Context that led to this plan:\n"
            f"- Root goal: {ctx.root_goal}\n"
            f"- Current mode: {ctx.mode}\n"
            f"- Active goal: {ctx.active_goal}\n\n"
            f"Evaluate the plan by answering:\n"
            f"1. Are the sub_goals in the right execution order?\n"
            f"2. Are any critical steps missing?\n"
            f"3. Is the priority level appropriate for the situation?\n"
            f"4. Will the tool_params actually work for the chosen tool?\n"
            f"5. Could this be accomplished with fewer steps?\n\n"
            f'Respond with ONLY valid JSON:\n'
            f'{{\n'
            f'    "confidence": 0.0,\n'
            f'    "issues": ["issue 1"],\n'
            f'    "suggestions": ["fix 1"]\n'
            f'}}'
        )

        try:
            response = self._contemplation_llm_call(
                system=critique_system,
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

        refine_system = "You are refining an action plan based on self-critique. Produce an improved version."
        refine_user = (
            f"Original plan:\n{draft_json}\n\n"
            f"Self-critique found these issues:\n"
            + "\n".join(f"- {i}" for i in issues)
            + "\n\nSuggestions for improvement:\n"
            + "\n".join(f"- {s}" for s in suggestions)
            + f"\n\nContext:\n- Root goal: {ctx.root_goal}\n- Mode: {ctx.mode}\n\n"
            f"Produce a corrected plan. Respond with ONLY valid JSON:\n"
            f'{{\n'
            f'    "goal_description": "...",\n'
            f'    "priority": "CRITICAL|HIGH|MEDIUM|LOW|IDLE",\n'
            f'    "tool_name": "...",\n'
            f'    "tool_params": {{}},\n'
            f'    "reasoning": "...",\n'
            f'    "sub_goals": []\n'
            f'}}'
        )

        try:
            response = self._contemplation_llm_call(
                system=refine_system,
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

        fast_system = (
            "You are evaluating and improving an action plan. "
            "First assess quality, then fix any issues."
        )
        fast_user = (
            f"You previously proposed this plan:\n{draft_json}\n\n"
            f"Context:\n"
            f"- Root goal: {ctx.root_goal}\n"
            f"- Current mode: {ctx.mode}\n"
            f"- Active goal: {ctx.active_goal}\n\n"
            f"Evaluate the plan:\n"
            f"1. Are the sub_goals in the right execution order?\n"
            f"2. Are any critical steps missing?\n"
            f"3. Is the priority level appropriate?\n"
            f"4. Will the tool_params work for the chosen tool?\n"
            f"5. Could this be accomplished with fewer steps?\n\n"
            f"Respond with ONLY valid JSON containing your evaluation "
            f"AND the corrected plan:\n"
            f'{{\n'
            f'    "confidence": 0.0,\n'
            f'    "issues": ["issue 1"],\n'
            f'    "plan": {{\n'
            f'        "goal_description": "...",\n'
            f'        "priority": "CRITICAL|HIGH|MEDIUM|LOW|IDLE",\n'
            f'        "tool_name": "...",\n'
            f'        "tool_params": {{}},\n'
            f'        "reasoning": "...",\n'
            f'        "sub_goals": []\n'
            f'    }}\n'
            f'}}'
        )

        try:
            response = self._contemplation_llm_call(
                system=fast_system,
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

        # Relevant memories
        mem_lines = []
        for mem in ctx.relevant_memories[:3]:
            mem_lines.append(f"  - [{mem.source}] salience={mem.salience:.2f}")
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
{mem_str}{stat_str}{dn_str}{budget_str}

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

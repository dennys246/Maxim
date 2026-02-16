"""ExecAgent: Goal proposal based on structured context.

Uses LLM to propose goals aligned with the root goal:
"Understand reality and help people."
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any

from maxim.agents.base import Agent
from maxim.agents.bus import (
    AgentBus,
    GoalPriority,
    Percept,
    ProposedGoal,
    StructuredContext,
    SubGoal,
)
from maxim.default_network.messages import FilteredPercept
from maxim.agents.llm_agent import ChatLLMAgent, LLMAgentConfig
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

You receive structured context about what you perceive (YOLO detections, speech) and your memories.
Your job is to propose goals that advance understanding or help people.

**IMPORTANT: You do NOT receive raw images. Visual information comes from YOLO object detection results only.**

Guidelines:
1. Voice commands from users are HIGH priority - they're asking for help
2. YOLO detections help you understand the visual scene (objects, people, positions)
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
- focus_interests: Run YOLO detection on current frame. Params: {}
- maxim_command: Execute command. Params: {command: str, ...}
  Commands: update_interests, center_vision, mark_trainable_moment,
            label_outcome, request_sleep, request_observe, request_shutdown,
            goto_pose, move, look_at_image, move_antenna
  update_interests params: {add: [int], remove: [int]}
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

        # Default Network escalation context
        self._dn_escalation_context: dict | None = None

        # Subscribe to percepts and filtered percepts (from Default Network)
        self._bus.subscribe(Percept, self._on_percept)
        self._bus.subscribe(FilteredPercept, self._on_filtered_percept)

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

    def _on_percept(self, percept: Percept) -> None:
        """Trigger goal proposal on significant percepts.

        Note: When Default Network is active, most percepts are filtered
        and only FilteredPercepts reach ExecAgent via _on_filtered_percept.
        This handler remains for backwards compatibility and direct percepts.
        """
        if percept.source == "cli":
            self._trigger_proposal()
            return

        # Always propose on voice commands
        if percept.has_maxim_keyword:
            self._trigger_proposal()
            return

        # Propose on high salience + novelty
        if percept.salience > 0.5 and percept.novelty > 0.5:
            self._trigger_proposal()

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

        # Always trigger proposal for escalated percepts
        self._trigger_proposal()

    def _trigger_proposal(self) -> None:
        """Signal that goal proposal is needed."""
        self._work_available.set()

    def _build_llm_context(self, ctx: StructuredContext) -> str:
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

RECENT OUTCOMES:
{outcome_str}

RELEVANT MEMORIES:
{mem_str}{stat_str}{dn_str}

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

        llm = self._ensure_llm()
        if llm is None:
            return None

        prompt = self._build_llm_context(ctx)

        try:
            response = llm.generate_json(prompt, temperature=0.2)
            if not response or not isinstance(response, dict) or not response.get("goal_description"):
                return None

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

            # Log to abstraction stream
            log_structured(
                self.log,
                logging.INFO,
                "goal_proposed",
                {
                    "goal_id": goal.id,
                    "description": goal.description,
                    "priority": goal.priority.name,
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
        if self._worker:
            self._worker.join(timeout=2.0)
        self._bus.unsubscribe(Percept, self._on_percept)
        self._bus.unsubscribe(FilteredPercept, self._on_filtered_percept)
        if self._llm:
            self._llm.on_stop()

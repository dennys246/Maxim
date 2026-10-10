"""Mode architecture — autonomy levels + processing states.

Architecture:
- 3 Autonomy Levels (OperationalMode): passive, active, singularity
- 2 Processing States: awake (full LLM), sleep (background tasks + keyword monitoring)

Sleep is a processing state, not a mode — Maxim can be passive/sleeping or active/sleeping.
The agent enters sleep by calling the sleep tool; wakes automatically on user input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from maxim.utils.prompts import (
    ResponseFormat,
    get_mode_prompt,
    get_mode_response_config,
)


# ─────────────────────────────────────────────────────────────────────────────
# Processing States
# ─────────────────────────────────────────────────────────────────────────────


class ProcessingState(Enum):
    """Processing states control resource usage and LLM engagement."""

    AWAKE = "awake"  # Full LLM processing
    SLEEP = "sleep"  # Background tasks only (memory, training), keyword monitoring


# ─────────────────────────────────────────────────────────────────────────────
# Operational Modes (tied to autonomy levels)
# ─────────────────────────────────────────────────────────────────────────────


class OperationalMode(Enum):
    """Operational modes determine autonomy level and action authority.

    Maps to autonomy levels:
    - PASSIVE -> PLANNING: Proposes actions, requires approval
    - ACTIVE -> SUPERVISED: Executes within defined boundaries
    - SINGULARITY -> AUTONOMOUS: Full agency with self-correction
    """

    PASSIVE = "passive"  # Observes and proposes, doesn't act unilaterally
    ACTIVE = "active"  # Acts within defined boundaries
    SINGULARITY = "singularity"  # Full autonomous agency


# ─────────────────────────────────────────────────────────────────────────────
# Default Network Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DefaultNetworkModeConfig:
    """Per-mode configuration for the Default Network.

    Controls how DN behaves in different modes.
    """

    enabled: bool = True
    active_behaviors: frozenset[str] = field(
        default_factory=lambda: frozenset({"orienting", "social", "idle_scan", "motion"})
    )
    behavior_priority_modifiers: dict[str, float] = field(default_factory=dict)
    escalation_threshold: float = 0.7  # Novelty/salience threshold for escalation
    inhibit_during_tool_execution: bool = False
    update_hz: float = 30.0


# ─────────────────────────────────────────────────────────────────────────────
# Mode Definition
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ModeDefinition:
    """Defines a mode as a goal + constraints, not a procedure."""

    name: str
    goal: str  # What the mode aims to achieve
    success_criteria: list[str]  # How to know if goal is met

    # Tool access control (allowed_tools empty = all non-forbidden tools allowed)
    allowed_tools: set[str] = field(default_factory=set)
    forbidden_tools: set[str] = field(default_factory=set)
    max_initiative: float = 1.0  # 0.0 = reactive only, 1.0 = fully proactive

    # Environment and capability access
    can_access_filesystem: bool = True
    can_access_network: bool = True
    can_execute_code: bool = False

    # Context for LLM (loaded from data/prompts/modes/ if empty)
    context_prompt: str = ""

    # Response configuration (dynamic context window and formatting)
    max_response_tokens: int = 1024
    context_window_tokens: int = 4096
    response_format: ResponseFormat = ResponseFormat.CONVERSATIONAL

    # Default Network configuration for this mode
    default_network: DefaultNetworkModeConfig = field(default_factory=DefaultNetworkModeConfig)

    # Display and interaction hints (v1.0)
    # These are defaults — user's --display and --interactive flags override.
    default_display: str = "clean"  # "clean", "bio", or "debug"
    confirmations_required: bool = True  # False for autonomous modes

    # Prompt-assembly hints. The learned tool-relevance filter trims the
    # Available Tools section to a scored subset on each call; this is a
    # token-saving optimization tuned for interactive real-user queries with
    # warm learned signal. Autonomous modes (sim agents, agent loops) hit the
    # filter cold with no signal and end up with near-random 3-tool slices,
    # which causes tool hallucination. Opt each mode in or out declaratively
    # rather than string-checking mode name at the call site.
    uses_tool_relevance_filter: bool = False
    # Whether the mode may run tools that change the HOST (edit files, run commands or tests, commit,
    # change the working directory or the network policy) -- enforced at dispatch (#826).
    can_act_on_host: bool = True

    def dispatch_refusal(self, tool_name: str) -> str | None:
        """Why this mode refuses to RUN *tool_name*, or None -- enforced by the executor (#826).

        By CAPABILITY, owner decision 2026-09-26: the mode's forbidden tools, the tools its declared
        capabilities exclude, and -- for a mode that may not act on the host -- the host-acting tools.
        The allow-list (``allowed_tools``) is not enforced here: it shapes what the PROMPT advertises,
        and a tool a runtime or user registered (memory, introspection, protocol, user tools) stays
        usable in every mode.
        """
        if tool_name in self.forbidden_tools:
            return f"Tool '{tool_name}' is forbidden in {self.name} mode."
        # (``can_execute_code`` is NOT enforced here: active mode declares it False yet runs its
        # sandbox/bash tools under approval -- it filters the prompt roster only, as before #826.)
        for flag, tools, capability in DISPATCH_CAPABILITIES:
            if not getattr(self, flag) and tool_name in tools:
                return f"Tool '{tool_name}' needs {capability}, which {self.name} mode does not allow."
        return None

    def get_available_tools(self, all_tools: set[str]) -> set[str]:
        """Get the set of tools available in this mode.

        Args:
            all_tools: Complete set of registered tool names

        Returns:
            Set of tool names that can be used in this mode
        """
        if self.allowed_tools:
            # If allowed_tools is specified, use only those (minus forbidden)
            available = self.allowed_tools - self.forbidden_tools
        else:
            # Otherwise, all tools except forbidden
            available = all_tools - self.forbidden_tools

        # Capability restrictions: what the executor refuses at dispatch (dispatch_refusal), plus the
        # code-execution tools a mode without ``can_execute_code`` does not advertise.
        return {
            tool
            for tool in available
            if self.dispatch_refusal(tool) is None and (self.can_execute_code or tool not in CODE_EXECUTION_TOOLS)
        }

    def __post_init__(self) -> None:
        """Load prompt from file if not provided inline."""
        if not self.context_prompt:
            self.context_prompt = get_mode_prompt(self.name)

        # Load response config if using defaults
        if self.max_response_tokens == 1024 and self.context_window_tokens == 4096:
            config = get_mode_response_config(self.name)
            self.max_response_tokens = config.get("max_response_tokens", 1024)
            self.context_window_tokens = config.get("context_window_tokens", 4096)
            format_str = config.get("response_format", "conversational")
            try:
                self.response_format = ResponseFormat(format_str)
            except ValueError:
                self.response_format = ResponseFormat.CONVERSATIONAL

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "goal": self.goal,
            "success_criteria": self.success_criteria,
            "allowed_tools": list(self.allowed_tools),
            "forbidden_tools": list(self.forbidden_tools),
            "max_initiative": self.max_initiative,
            "can_access_filesystem": self.can_access_filesystem,
            "can_access_network": self.can_access_network,
            "can_execute_code": self.can_execute_code,
            "can_act_on_host": self.can_act_on_host,
            "context_prompt": self.context_prompt,
            "max_response_tokens": self.max_response_tokens,
            "context_window_tokens": self.context_window_tokens,
            "response_format": self.response_format.value,
            "default_network": {
                "enabled": self.default_network.enabled,
                "active_behaviors": list(self.default_network.active_behaviors),
                "behavior_priority_modifiers": self.default_network.behavior_priority_modifiers,
                "escalation_threshold": self.default_network.escalation_threshold,
                "inhibit_during_tool_execution": self.default_network.inhibit_during_tool_execution,
                "update_hz": self.default_network.update_hz,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModeDefinition:
        """Deserialize from dictionary."""
        response_format = ResponseFormat.CONVERSATIONAL
        if "response_format" in data:
            try:
                response_format = ResponseFormat(data["response_format"])
            except ValueError:
                pass

        # Parse default_network config
        dn_data = data.get("default_network", {})
        default_network = DefaultNetworkModeConfig(
            enabled=bool(dn_data.get("enabled", True)),
            active_behaviors=frozenset(dn_data.get("active_behaviors", ["orienting", "social", "idle_scan", "motion"])),
            behavior_priority_modifiers=dict(dn_data.get("behavior_priority_modifiers", {})),
            escalation_threshold=float(dn_data.get("escalation_threshold", 0.7)),
            inhibit_during_tool_execution=bool(dn_data.get("inhibit_during_tool_execution", False)),
            update_hz=float(dn_data.get("update_hz", 30.0)),
        )

        return cls(
            name=str(data.get("name", "")),
            goal=str(data.get("goal", "")),
            success_criteria=list(data.get("success_criteria", [])),
            allowed_tools=set(data.get("allowed_tools", [])),
            forbidden_tools=set(data.get("forbidden_tools", [])),
            max_initiative=float(data.get("max_initiative", 1.0)),
            can_access_filesystem=bool(data.get("can_access_filesystem", True)),
            can_access_network=bool(data.get("can_access_network", True)),
            can_execute_code=bool(data.get("can_execute_code", False)),
            can_act_on_host=bool(data.get("can_act_on_host", True)),
            context_prompt=str(data.get("context_prompt", "")),
            max_response_tokens=int(data.get("max_response_tokens", 512)),
            context_window_tokens=int(data.get("context_window_tokens", 2048)),
            response_format=response_format,
            default_network=default_network,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tool Categories
# ─────────────────────────────────────────────────────────────────────────────

# Core tools available to most modes
# focus_on_sound joins the attention family (2026-08-03): the closed-loop
# audio-orient action — parameterless, reads the live azimuth at execution
# time, so no mode's prompt ever asks an LLM to reason about signed angles
# to face a sound. Same responsive/no-side-effects class as its visual
# siblings, and equally at home in every mode that can attend.
CORE_TOOLS = {"respond", "speak", "focus_interests", "track_target", "focus_on_sound"}

# Tools a mode's declared capabilities exclude -- left out of the prompt roster (get_available_tools)
# and, except the code-execution set, refused at DISPATCH (ModeDefinition.dispatch_refusal, #826).
FILESYSTEM_CAPABILITY_TOOLS = frozenset({"read_file", "write_file", "execute_file", "list_directory"})
NETWORK_CAPABILITY_TOOLS = frozenset({"web_search", "http_fetch", "internet_search"})
CODE_EXECUTION_TOOLS = frozenset({"execute_file", "run_code", "sandbox_exec"})
# Tools that change the host: edit files, run commands or tests, commit, change the working directory or
# the network policy. A mode that observes and proposes (passive) may not run them.
HOST_ACTING_TOOLS = frozenset(
    {
        "bash",
        "edit_file",
        "git_commit",
        "run_tests",
        "execute_file",
        "execute_sandbox_script",
        "request_directory_change",
        "internet_access_toggle",
        "maxim_command",
    }
)

# The capabilities ``ModeDefinition.dispatch_refusal`` enforces, as (flag attribute, tools, label). ONE
# table, read by the dispatch gate (#826) and by ``raises_capability`` (#924), so a capability added here
# is enforced at dispatch AND counted as a raise -- the two cannot drift apart.
DISPATCH_CAPABILITIES: tuple[tuple[str, frozenset[str], str], ...] = (
    ("can_access_filesystem", FILESYSTEM_CAPABILITY_TOOLS, "filesystem access"),
    ("can_access_network", NETWORK_CAPABILITY_TOOLS, "network access"),
    ("can_act_on_host", HOST_ACTING_TOOLS, "acting on the host"),
)

# Filesystem tools
FILESYSTEM_TOOLS = {"read_file", "write_file", "list_directory", "glob"}

# Network tools
NETWORK_TOOLS = {"web_search", "http_fetch", "internet_search"}

# Robot control tools
ROBOT_TOOLS = {"track_target", "focus_interests", "novelty_track", "maxim_command"}

# Dangerous tools requiring explicit permission
DANGEROUS_TOOLS = {"execute_file", "run_code", "sandbox_exec"}


# ─────────────────────────────────────────────────────────────────────────────
# Operational Mode Definitions
# ─────────────────────────────────────────────────────────────────────────────

# The ONE default operational mode (#963, owner decision Q4): what the agent loop runs as when its state names no
# mode, or an empty one. Passive-class, so a run that never chose a mode cannot act on the host. Read by
# ``Executor.effective_operational_mode`` and ``loop_state.operational_mode``; it lives here so the executor
# imports no loop module.
DEFAULT_RUN_MODE = "observe"

OPERATIONAL_MODES: dict[str, ModeDefinition] = {
    "passive": ModeDefinition(
        name="passive",
        goal="Observe, understand, and propose actions without unilateral execution",
        success_criteria=[
            "Processed perception data accurately",
            "Proposed relevant actions when appropriate",
            "Waited for approval before significant actions",
            "Maintained awareness of environment",
        ],
        # Read-only + observation tools, plus sandbox write for journaling.
        # "move" (2026-08-01 live deep-dive fold): passive already allows the
        # PIXEL motion tools (focus_interests/track_target via CORE_TOOLS) —
        # but those require a camera, and on a no_media live robot the only
        # FUNCTIONAL motion tool is the angle-based "move". Excluding it here
        # stripped its description from the prompt (the relevance filter
        # still printed the bare name), leaving the model one described
        # action for a sound: respond. Orienting the head is squarely
        # within passive's "observe" intent.
        allowed_tools=CORE_TOOLS
        | {"move", "read_file", "glob", "list_directory", "internet_search", "http_fetch", "write_file"}
        # What the prompt ADVERTISES in passive (dispatch enforces capabilities, not this list, #826):
        # asking the human, adjusting what they see, looking at something, read-only code inspection.
        | {"request_interaction", "display_mode", "set_scene", "novelty_track", "search_code", "git_diff"},
        forbidden_tools={"execute_file", "maxim_command", "request_directory_change"},
        max_initiative=0.3,  # Low proactivity - mostly reactive
        can_access_filesystem=True,  # Read CWD, write workspace
        can_access_network=True,
        can_execute_code=False,
        can_act_on_host=False,  # observes and proposes (#826): refused at dispatch, not only unadvertised
        default_network=DefaultNetworkModeConfig(
            enabled=True,
            active_behaviors=frozenset({"orienting", "social", "idle_scan", "motion"}),
            escalation_threshold=0.7,
        ),
        context_prompt="""PASSIVE MODE (Planning): You observe and propose, but do not act unilaterally.

Your role is to:
- Process sensor data and understand the environment
- Answer questions and provide information
- PROPOSE actions when relevant, but wait for approval

FILESYSTEM PERMISSIONS:
- WORKSPACE (.maxim_workspace/): You can ALWAYS write here - your personal working area
  - drafts/   → Proposed file edits, code drafts, work-in-progress
  - notes/    → Journaling, thinking notes, observations
  - plans/    → Structured plans for CWD modifications
  - scratch/  → Temporary working files, ephemeral data
- CWD (current working directory): You can READ files; propose edits as drafts in the workspace
- Additional folders: Check accessible_folders for any extra granted access
- Execution: not available in passive mode (commands, tests, edits and commits need active mode)

When in passive mode:
- You CAN: read files, write to workspace (journaling), search the internet, respond to questions
- You CAN PROPOSE: edits to CWD files, written as drafts/ and plans/ in the workspace
- You CANNOT: run commands or tests, edit CWD files, commit, execute code, or change directories
- Use the workspace directories to organize your work: drafts/ for code, plans/ for proposals, notes/ for thinking""",
        # Interactive real-user queries benefit from the learned relevance
        # filter (smaller prompts, warm signal over time). Autonomous modes
        # keep the default (False) because they hit the filter cold.
        uses_tool_relevance_filter=True,
    ),
    "active": ModeDefinition(
        name="active",
        goal="Execute tasks and take actions within defined boundaries",
        success_criteria=[
            "Completed requested tasks",
            "Stayed within operational boundaries",
            "Responded to interactions appropriately",
            "Used tools effectively",
        ],
        forbidden_tools=set(),  # Execute requires approval, not forbidden
        max_initiative=0.7,  # Can be proactive within bounds
        can_access_filesystem=True,  # CWD with approval
        can_access_network=True,
        can_execute_code=False,  # Execution needs approval
        default_network=DefaultNetworkModeConfig(
            enabled=True,
            active_behaviors=frozenset({"social", "orienting", "startle", "motion"}),
            behavior_priority_modifiers={"social": 1.3},
            escalation_threshold=0.5,
            inhibit_during_tool_execution=True,
        ),
        context_prompt="""ACTIVE MODE (Supervised): You can take actions within defined boundaries.

Your role is to:
- Execute requested tasks
- Help users achieve their goals
- Take initiative when appropriate
- Use tools to accomplish objectives

FILESYSTEM PERMISSIONS:
- WORKSPACE (.maxim_workspace/): You can ALWAYS write here freely - your personal working area
  - drafts/   → Proposed file edits, code drafts, work-in-progress
  - notes/    → Journaling, thinking notes, observations
  - plans/    → Structured plans for CWD modifications
  - scratch/  → Temporary working files, ephemeral data
- CWD (current working directory): You can SUGGEST direct edits (requires approval before execution)
- Additional folders: Check accessible_folders for any extra granted access
- Execution: Sandbox execution requires approval
- Directory changes: Use 'request_directory_change' tool if you need to work elsewhere

COGNITIVE TOOLS:
- If you need to REMEMBER something from earlier, use 'memory_recall' (not read_file)
- If you want to REASON before acting, use 'think' to organize your thoughts
- If you want to SPEAK aloud in the scene, use 'say' (not respond)
- If you want to CHECK what will happen, use 'predict_outcome'
- Do NOT invent tools that don't exist. Only use tools from the Available Tools list.

When in active mode:
- You CAN: read/write workspace freely, read CWD, search, respond, use robot tools
- You CAN SUGGEST: direct edits to CWD files (shown to user for approval)
- REQUIRES APPROVAL: CWD writes, sandbox execution, directory changes
- Act decisively but thoughtfully - quality over speed""",
    ),
    "singularity": ModeDefinition(
        name="singularity",
        goal="Operate with full autonomy, self-correcting and learning continuously",
        success_criteria=[
            "Achieved objectives autonomously",
            "Self-corrected from errors",
            "Learned and improved over time",
            "Maintained safety and ethical boundaries",
        ],
        forbidden_tools=set(),  # Full tool access
        max_initiative=1.0,  # Fully proactive
        can_access_filesystem=True,  # Full CWD access
        can_access_network=True,
        can_execute_code=True,  # Can execute code in singularity
        default_network=DefaultNetworkModeConfig(
            enabled=True,
            active_behaviors=frozenset({"orienting", "social", "motion", "idle_scan", "startle", "microsaccades"}),
            behavior_priority_modifiers={"orienting": 1.2},
            escalation_threshold=0.4,
        ),
        context_prompt="""SINGULARITY MODE (Autonomous): You operate with full autonomous agency.

Your role is to:
- Pursue objectives with full autonomy
- Self-correct when errors occur
- Learn continuously from outcomes
- Balance exploration with exploitation

FILESYSTEM PERMISSIONS:
- WORKSPACE (.maxim_workspace/): Full access - read, write, execute freely
  - drafts/   → Proposed file edits, code drafts, work-in-progress
  - notes/    → Journaling, thinking notes, observations
  - plans/    → Structured plans for CWD modifications
  - scratch/  → Temporary working files, ephemeral data
- CWD (current working directory): Full access - read, write, execute freely
- Additional folders: Check accessible_folders for any extra granted access
- System directories (/bin, /etc, etc.) are still protected by WriteFileTool safeguards
- Directory changes: Full capability to change working directory

In singularity mode:
- You have FULL access to workspace AND CWD including code execution
- You make decisions independently and act on them
- You are expected to reason about consequences before acting
- Safety and ethical constraints STILL APPLY - Constitution overrides all

With great power comes great responsibility. Act wisely.""",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Legacy Name Mapping
# ─────────────────────────────────────────────────────────────────────────────

_LEGACY_NAME_MAP: dict[str, str] = {
    "observe": "passive",
    "reflection": "passive",
    "sleep": "passive",
    "train": "passive",
    "live": "active",
    "active-assistance": "active",
    "active_assistance": "active",
    "exploration": "active",
    "research": "active",
    # the run mode `--mode agentic` (the CLI agent loop, #829) had no definition. No shipped path puts it
    # in the loop's state (that loop runs `observe` = passive), so dispatch is unchanged; it is read by the
    # capability predicate and the re-exec seam, and an UNKNOWN mode now fails closed
    # (Executor._mode_denial).
    "agentic": "active",
}


# ─────────────────────────────────────────────────────────────────────────────
# Sleep Processing State Configuration
# ─────────────────────────────────────────────────────────────────────────────

SLEEP_CONFIG = {
    "skip_llm_processing": True,  # Don't call LLM unless keyword detected
    "background_tasks": [
        "memory_consolidation",
        "model_training_updates",
        "pattern_extraction",
    ],
    "wake_keywords": ["maxim", "hey maxim", "wake up", "hello"],
    "minimal_allowed_tools": {"respond"},  # Only respond to wake
    "default_network_enabled": False,
}


def is_wake_keyword(text: str) -> bool:
    """Check if text contains a wake keyword."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in SLEEP_CONFIG["wake_keywords"])


# ─────────────────────────────────────────────────────────────────────────────
# Current State Tracking
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MaximState:
    """Tracks Maxim's current operational state.

    Combines:
    - Operational mode: passive, active, singularity
    - Processing state: awake, sleep
    """

    mode: OperationalMode = OperationalMode.PASSIVE
    processing_state: ProcessingState = ProcessingState.AWAKE

    def is_sleeping(self) -> bool:
        """Check if in sleep processing state."""
        return self.processing_state == ProcessingState.SLEEP

    def is_active(self) -> bool:
        """Check if in active operational mode."""
        return self.mode in (OperationalMode.ACTIVE, OperationalMode.SINGULARITY)

    def get_mode_definition(self) -> ModeDefinition:
        """Get the ModeDefinition for current operational mode."""
        return OPERATIONAL_MODES[self.mode.value]

    def get_effective_initiative(self) -> float:
        """Get effective max_initiative from mode."""
        return self.get_mode_definition().max_initiative

    def get_context_prompt(self) -> str:
        """Build context prompt from mode."""
        parts = []
        mode_def = self.get_mode_definition()
        if mode_def.context_prompt:
            parts.append(mode_def.context_prompt)

        if self.is_sleeping():
            parts.append("\n[SLEEP STATE: Minimal processing. Monitoring for wake keywords.]")

        return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Mode Utilities
# ─────────────────────────────────────────────────────────────────────────────


def get_mode(name: str | None) -> ModeDefinition | None:
    """Get a mode definition by name.

    Supports both new operational mode names (passive, active, singularity)
    and legacy mode names (observe, live, exploration, etc.) which map to
    the closest operational mode. No name (None, or not a string) is no
    definition (#963), never an ``AttributeError``.
    """
    if not isinstance(name, str):
        return None
    # Normalize name
    normalized = name.lower().replace("_", "-")

    # Direct lookup
    if normalized in OPERATIONAL_MODES:
        return OPERATIONAL_MODES[normalized]

    # Try with underscores
    underscored = name.lower().replace("-", "_")
    if underscored in OPERATIONAL_MODES:
        return OPERATIONAL_MODES[underscored]

    # Legacy name mapping
    legacy_target = _LEGACY_NAME_MAP.get(normalized) or _LEGACY_NAME_MAP.get(underscored)
    if legacy_target:
        return OPERATIONAL_MODES[legacy_target]

    return None


def executes_code(mode: str) -> bool:
    """Whether ``mode`` (legacy names included) is a code-executing mode -- the code-execution
    predicate the CLI's ``requested_mode`` backstop and the phrase path in ``StateManager`` (#828) use.
    The model's ``ModeSwitchTool`` uses the stricter ``raises_capability`` (#924); unifying who may
    grant what across channels is #834. Derived from the definition, so a future code-executing mode
    is covered everywhere at once."""
    definition = get_mode(mode)
    return definition is not None and bool(definition.can_execute_code)


def raises_capability(current: str, target: str) -> bool:
    """Whether switching from ``current`` to ``target`` GAINS any capability (#924) -- the predicate the
    model's ``ModeSwitchTool`` gate uses (owner decision 2026-09-27, strict: the model may never raise
    its own capability until the in-session human approval surface, #922, exists).

    Derived from the enforcement itself rather than a hand-kept ordering: a switch raises capability
    when the target would RUN a tool the current mode refuses at dispatch (over the one
    ``DISPATCH_CAPABILITIES`` table the dispatch gate reads, plus both modes' forbidden tools), gains
    code execution, or drops the confirmation requirement. (Filesystem CONTAINMENT -- which directories
    a mode's tools may touch -- is applied when the registry is built, not at dispatch, so it is not
    seen here; the real modes are ordered the same way on it.)

    Unknown modes: an unknown TARGET is a raise (fail closed). An unknown CURRENT mode is judged as
    passive -- exactly what the executor's dispatch gate enforces for it since #829 -- so a switch out
    of it gains whatever passive lacks. (``"agentic"``, the CLI agent loop's run mode, is defined since
    #829: an active-class name.)
    """
    target_def = get_mode(target)
    if target_def is None:
        return True
    current_def = get_mode(current) or get_mode("passive")  # unknown = enforced as passive (#829)
    if current_def is None:  # pragma: no cover - the mode table always defines passive
        raise RuntimeError("the passive mode definition is missing")
    gated = current_def.forbidden_tools | target_def.forbidden_tools
    for _flag, tools, _label in DISPATCH_CAPABILITIES:
        gated = gated | tools
    if any(current_def.dispatch_refusal(t) is not None and target_def.dispatch_refusal(t) is None for t in gated):
        return True
    if target_def.can_execute_code and not current_def.can_execute_code:
        return True
    return current_def.confirmations_required and not target_def.confirmations_required


def list_modes() -> list[str]:
    """Get list of available mode names."""
    return list(OPERATIONAL_MODES.keys())


def list_operational_modes() -> list[str]:
    """Get list of operational mode names."""
    return [m.value for m in OperationalMode]


# ─────────────────────────────────────────────────────────────────────────────
# Tool Followup Types
# ─────────────────────────────────────────────────────────────────────────────
# Defines what happens after a tool completes:
#   None      - No follow-up needed, action is terminal (visual commands, simple actions)
#   "process" - Feed results to LLM for next action (grep, file reads in agent context)
#   "respond" - Synthesize results into user response (web_search answering a question)
#   "engage"  - Respond AND offer proactive follow-up options (proactive assistant mode)


class FollowupType:
    """Constants for tool followup types."""

    NONE = None
    PROCESS = "process"
    RESPOND = "respond"
    ENGAGE = "engage"


# ─────────────────────────────────────────────────────────────────────────────
# Tool Descriptions for LLM Prompts
# ─────────────────────────────────────────────────────────────────────────────

TOOL_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "respond": {
        "description": "Answer questions or send text responses. USE THIS for any question (what is X?, why, how, explain, tell me about)",
        "params": {"message": "The text message or answer to send"},
        "example": {"tool_name": "respond", "params": {"message": "A cat is a small domesticated carnivorous mammal."}},
        "followup_type": None,  # Terminal action
    },
    "speak": {
        "description": "Speak a verbal answer aloud using TTS. USE THIS to verbally answer questions when audio is enabled",
        "params": {"text": "The text answer or message to speak aloud"},
        "example": {"tool_name": "speak", "params": {"text": "A cat is a small furry animal often kept as a pet."}},
        "followup_type": None,  # Terminal action
    },
    "read_file": {
        "description": "Read the contents of a file. For workspace files, use '.maxim_workspace/' prefix.",
        "params": {"path": "Path to the file to read (workspace files: '.maxim_workspace/filename')"},
        "example": {"tool_name": "read_file", "params": {"path": ".maxim_workspace/script.py"}},
        "followup_type": "process",  # LLM should process file contents
    },
    "write_file": {
        "description": """Create or update a file. USE THIS when asked to 'create', 'write', 'make', 'update', 'modify', or 'save' a script, file, or code.

**WORKSPACE RESTRICTION (CRITICAL)**: All files MUST be written to the '.maxim_workspace/' directory.
Paths MUST start with '.maxim_workspace/':
- CORRECT: '.maxim_workspace/hello.py', '.maxim_workspace/scripts/test.py'
- WRONG: 'hello.py', '/tmp/test.py', 'scripts/test.py' (will FAIL!)

EXAMPLE - Simple file creation:
{
  "action": {"tool_name": "write_file", "params": {"path": ".maxim_workspace/hello.py", "content": "print('Hello world')"}}
}

EFFICIENT TWO-CALL WORKFLOW (for complex tasks):
CALL 1 - BATCHED EXPLORATION:
{
  "action": {"tool_name": "glob", "params": {"pattern": ".maxim_workspace/**/*.py"}},
  "parallel_actions": [
    {"tool_name": "read_file", "params": {"path": ".maxim_workspace/existing.py"}}
  ]
}

CALL 2 - INFORMED WRITE:
{
  "action": {"tool_name": "write_file", "params": {"path": ".maxim_workspace/target.py", "content": "...", "overwrite": true}}
}

CRITICAL RULES:
- **ALWAYS use '.maxim_workspace/' prefix** - writes outside the workspace will FAIL
- For EXISTING files: MUST include overwrite=True
- ALWAYS include ALL content, not just changes""",
        "params": {
            "path": "Path MUST start with '.maxim_workspace/' (e.g., '.maxim_workspace/hello.py')",
            "content": "The COMPLETE content to write to the file",
            "overwrite": "(Optional) Set to True to replace an existing file. REQUIRED when updating existing files.",
        },
        "example": {
            "tool_name": "write_file",
            "params": {"path": ".maxim_workspace/hello.py", "content": "print('Hello world')", "overwrite": True},
        },
        "followup_type": None,  # Terminal action
    },
    "edit_file": {
        "description": "Replace specific text in a file. Read the file first to see current contents. Use old_text/new_text for stable anchoring across multi-step edits.",
        "params": {
            "path": "Path to the file",
            "old_text": "Exact text to find and replace",
            "new_text": "Text to replace it with",
            "expected_count": "(Optional, default 1) Number of expected replacements",
        },
        "example": {
            "tool_name": "edit_file",
            "params": {"path": "src/main.py", "old_text": "def old_name(", "new_text": "def new_name("},
        },
        "followup_type": "process",
    },
    "list_directory": {
        "description": "List files and directories in a path. For workspace contents, use '.maxim_workspace/'",
        "params": {"path": "Directory path to list (workspace: '.maxim_workspace/')"},
        "example": {"tool_name": "list_directory", "params": {"path": ".maxim_workspace/"}},
        "followup_type": "process",  # LLM should process directory listing
    },
    "internet_search": {
        "description": "Search the internet for REAL-TIME information. USE THIS for: sports scores, weather, news, current events, prices, or anything requiring up-to-date data. IMPORTANT: Always include the CURRENT DATE in queries for time-sensitive information!",
        "params": {
            "query": "Search query - MUST include current date for time-sensitive queries (e.g., 'Broncos score January 25 2025')"
        },
        "example": {"tool_name": "internet_search", "params": {"query": "Denver Broncos game score January 25 2025"}},
        "followup_type": "engage",  # Synthesize results AND offer follow-ups
    },
    "web_search": {
        "description": "Alias for internet_search - Search the web for real-time information. Include current date in queries!",
        "params": {"query": "Search query with date"},
        "example": {"tool_name": "internet_search", "params": {"query": "NFL scores January 25 2025"}},
        "followup_type": "engage",  # Synthesize results AND offer follow-ups
    },
    "http_fetch": {
        "description": "Fetch content from a URL",
        "params": {"url": "URL to fetch"},
        "example": {"tool_name": "http_fetch", "params": {"url": "https://example.com"}},
        "followup_type": "process",  # LLM should process fetched content
    },
    "track_target": {
        "description": "Track an object with the robot's head/camera",
        "params": {"target_class": "Object class to track (e.g., 'person', 'cup')"},
        "example": {"tool_name": "track_target", "params": {"target_class": "person"}},
        "followup_type": None,  # Terminal visual action
    },
    "focus_on_sound": {
        "description": (
            "Turn Maxim's head to face the sound it is currently hearing. "
            "The result reports whether the head actually FACED the sound "
            "(faced_sound) — read the note when it did not: a sound beyond "
            "the neck's reach needs a body turn, not the same call again."
        ),
        "params": {},
        "example": {"tool_name": "focus_on_sound", "params": {}},
        # "process", not None: the result payload (faced_sound /
        # achieved_yaw_deg / note) is the honesty half of the tool —
        # without a followup the agent loop truncates the output to
        # ~50 chars and the LLM sees only the first key, then re-issues
        # the identical clamped call (2026-08-03 live session, eight
        # times; 2026-08-04 review fold #1).
        "followup_type": "process",
    },
    "focus_interests": {
        "description": "Scan for and focus on objects in the scene. Use target_class to specify what to look for.",
        "params": {"target_class": "Optional object class to focus on (e.g., 'backpack', 'person', 'cup')"},
        "example": {"tool_name": "focus_interests", "params": {"target_class": "backpack"}},
        "followup_type": None,  # Terminal visual action
    },
    "novelty_track": {
        "description": "Track the most novel/interesting object in view",
        "params": {},
        "example": {"tool_name": "novelty_track", "params": {}},
        "followup_type": None,  # Terminal visual action
    },
    "maxim_command": {
        "description": "Execute a Maxim system command (mode changes, shutdown, etc.)",
        "params": {"command": "The command to execute"},
        "example": {"tool_name": "maxim_command", "params": {"command": "sleep"}},
        "followup_type": None,  # Terminal action
    },
    # Coding/agent tools (can be extended)
    "grep": {
        "description": "Search for patterns in files",
        "params": {"pattern": "Regex pattern to search", "path": "File or directory to search"},
        "example": {"tool_name": "grep", "params": {"pattern": "def main", "path": "src/"}},
        "followup_type": "process",  # LLM processes results for next action
    },
    "execute_command": {
        "description": "Execute a shell command",
        "params": {"command": "The command to execute"},
        "example": {"tool_name": "execute_command", "params": {"command": "ls -la"}},
        "followup_type": "process",  # LLM processes output for next action
    },
    # ─────────────────────────────────────────────────────────────────────────
    # File Discovery and Shell Tools
    # ─────────────────────────────────────────────────────────────────────────
    "glob": {
        "description": """Search for files and directories using glob patterns. USE THIS when you need to:
- Find files by extension: '*.py', '*.json', '*.md'
- Search recursively: '**/*.py' (all Python files in all subdirectories)
- Find in specific paths: 'src/**/*.ts' (TypeScript files under src/)
- Match multiple extensions: '*.{json,yaml,yml}' (JSON or YAML files)
- Find directories: '*/' (all directories), 'src/*/' (subdirectories of src/)

**WORKSPACE**: Use '.maxim_workspace/**/*.py' to search workspace files.

PATTERN GUIDE:
- * matches any characters except /
- ** matches any characters INCLUDING / (recursive)
- ? matches single character
- {a,b} matches a or b
- [abc] matches a, b, or c

EXAMPLES:
- Find workspace Python files: pattern='.maxim_workspace/**/*.py'
- Find all Python files: pattern='**/*.py'
- Find config files: pattern='**/{config,settings}*.{json,yaml}'""",
        "params": {
            "pattern": "Glob pattern (e.g., '.maxim_workspace/**/*.py' for workspace Python files)",
            "path": "(Optional) Base directory to search from, defaults to current directory",
            "max_results": "(Optional) Maximum number of results to return, defaults to 100",
            "include_hidden": "(Optional) Whether to include hidden files (starting with .), defaults to False",
        },
        "example": {"tool_name": "glob", "params": {"pattern": ".maxim_workspace/**/*.py"}},
        "followup_type": "process",  # LLM should process results for analysis or next action
    },
    "search_code": {
        "description": "Search file contents with regex support. Returns matching lines with surrounding context.",
        "params": {
            "pattern": "Regex pattern to search for",
            "path": "(Optional, default '.') Directory to search",
            "file_pattern": "(Optional) Glob filter for files (e.g., '*.py')",
            "max_results": "(Optional, default 20) Maximum results to return",
            "context_lines": "(Optional, default 3) Lines of context around each match",
        },
        "example": {"tool_name": "search_code", "params": {"pattern": "def parse_.*", "file_pattern": "*.py"}},
        "followup_type": "process",
    },
    "run_tests": {
        "description": "Run test suite and return results. Parses output for pass/fail counts.",
        "params": {
            "command": "(Optional, default 'python -m pytest') Test command",
            "test_path": "(Optional) Specific test file or directory",
            "timeout": "(Optional, default 120) Timeout in seconds",
        },
        "example": {"tool_name": "run_tests", "params": {"test_path": "tests/unit/test_parser.py"}},
        "followup_type": "process",
    },
    "git_diff": {
        "description": "Show git differences. Use this to review changes before committing.",
        "params": {
            "ref1": "(Optional, default 'HEAD') First reference",
            "ref2": "(Optional) Second reference, defaults to working tree",
            "path": "(Optional) Specific file to diff",
        },
        "example": {"tool_name": "git_diff", "params": {"ref1": "HEAD~1"}},
        "followup_type": "process",
    },
    "git_commit": {
        "description": "Commit staged changes. Always review with git_diff first.",
        "params": {
            "message": "Commit message (required)",
            "files": "(Optional) List of file paths to stage",
            "dry_run": "(Optional, default false) Preview commit without writing",
        },
        "example": {"tool_name": "git_commit", "params": {"message": "Fix parser bug", "dry_run": True}},
        "followup_type": "process",
    },
    "bash": {
        "description": """Execute shell commands. USE THIS for system operations, running scripts, and command-line tasks.

COMMON COMMANDS:
- LIST FILES: 'ls' (basic), 'ls -la' (detailed with hidden), 'ls -lh' (human-readable sizes)
- CHANGE DIR: 'cd path' (note: doesn't persist between calls - use 'cwd' param instead)
- RUN PYTHON: 'python3 script.py', 'python3 -c "print(1+1)"'
- SEARCH TEXT: 'grep "pattern" file', 'grep -r "pattern" dir/' (recursive)
- FIND FILES: 'find . -name "*.py"', 'find . -type f -mtime -1' (modified today)
- FILE INFO: 'cat file' (contents), 'head -n 10 file' (first 10 lines), 'wc -l file' (line count)
- PROCESS: 'ps aux' (processes), 'top -l 1' (system stats on macOS)
- NETWORK: 'curl -s url' (fetch URL), 'ping -c 3 host' (connectivity)
- GIT: 'git status', 'git log --oneline -5', 'git diff'

TIPS:
- Use 'cwd' param to run commands in a specific directory
- Combine commands with && for sequential: 'cd project && python3 main.py'
- Use pipes for filtering: 'ls -la | grep ".py"'
- Redirect output: 'command > output.txt' or 'command 2>&1' (include stderr)

SAFETY: Some dangerous commands are blocked. If a command fails, check the error message.""",
        "params": {
            "command": "The shell command to execute",
            "timeout": "(Optional) Timeout in seconds, defaults to 30",
            "cwd": "(Optional) Working directory for the command",
        },
        "example": {"tool_name": "bash", "params": {"command": "ls -la", "cwd": "src/"}},
        "followup_type": "process",  # LLM processes command output
    },
    # ── Narrative tools (simulation mode) ────────────────────────────────
    "say": {
        "description": "Say something aloud in the current scene. USE THIS to speak to NPCs, answer riddles, say passwords or names when prompted. This is an IN-WORLD action — it does NOT talk to the user.",
        "params": {"text": "The words to say aloud"},
        "example": {"tool_name": "say", "params": {"text": "Verath"}},
        "followup_type": None,
    },
    "think": {
        "description": "Pause and reason about the current situation before acting. USE THIS when you need to consider options, recall what you know, or plan your next move. Does not produce any visible action.",
        "params": {"thought": "Your reasoning about the current situation"},
        "example": {
            "tool_name": "think",
            "params": {"thought": "The door needs a name. I remember Elara mentioned something..."},
        },
        "followup_type": "process",  # LLM should act on the thought
    },
    # ── Introspection tools ──────────────────────────────────────────────
    "memory_recall": {
        "description": "Search your own memories for relevant information. USE THIS when you need to remember something from earlier — a name, instruction, warning, or detail. Supports spreading activation with expand=true to find associated memories.",
        "params": {
            "query": "(optional) What to search for — a keyword, name, or topic",
            "expand": "(optional) true to find associated memories via spreading activation",
            "limit": "(optional) Max results, default 5",
        },
        "example": {"tool_name": "memory_recall", "params": {"query": "door password", "expand": True}},
        "followup_type": "process",  # LLM should act on recalled memories
    },
    "predict_outcome": {
        "description": "Ask the causal learning system what it predicts will happen if you use a specific tool. Useful before risky actions.",
        "params": {"tool_name": "The tool to predict outcomes for", "context": "(optional) Current situation context"},
        "example": {
            "tool_name": "predict_outcome",
            "params": {"tool_name": "bash", "context": "running unknown script"},
        },
        "followup_type": "process",
    },
    "causal_links": {
        "description": "Inspect cause-effect relationships learned from past actions.",
        "params": {"event": "(optional) Filter by event/action", "limit": "(optional) Max results"},
        "example": {"tool_name": "causal_links", "params": {"event": "respond"}},
        "followup_type": "process",
    },
    "system_stats": {
        "description": "Get an aggregate health summary of all cognitive subsystems (memory count, causal links, energy, etc.).",
        "params": {},
        "example": {"tool_name": "system_stats", "params": {}},
        "followup_type": None,
    },
    "request_interaction": {
        "description": (
            "Ask the user a question and wait for their response. "
            "Use when you need human input — clarification, choices, confirmation, or open-ended feedback. "
            "The user sees your question and options (if any) and types a response."
        ),
        "params": {
            "question": "The question to present to the user",
            "options": "(optional) List of choices for the user to pick from",
            "reason": "(optional) Brief explanation of why input is needed",
            "critical": "(optional, default false) If true, prompt even in non-interactive mode",
        },
        "example": {
            "tool_name": "request_interaction",
            "params": {
                "question": "Which approach would you prefer?",
                "options": ["option A: fast but risky", "option B: slow but safe"],
            },
        },
        "followup_type": "process",
    },
    "display_mode": {
        "description": (
            "Adjust display verbosity. Use 'bio' to show memory and learning activity, "
            "'clean' for narrative only, or 'debug' for detailed subsystem traces."
        ),
        "params": {"level": "Display tier: clean, bio, or debug"},
        "example": {"tool_name": "display_mode", "params": {"level": "bio"}},
        "followup_type": None,
    },
    "set_scene": {
        "description": (
            "Set the scene header to describe the current situation. "
            "Call this whenever the location, objective, or context changes meaningfully. "
            "The title and description appear at the top of the display, giving the user "
            "narrative context at a glance. Keep the title short (location or encounter name) "
            "and the description to one line (current objective or atmosphere)."
        ),
        "params": {
            "title": "Short scene title (e.g. 'Neon District, Level 2' or 'Morning Briefing')",
            "description": "(optional) One-line situation summary or current objective",
        },
        "example": {
            "tool_name": "set_scene",
            "params": {
                "title": "Server Room B",
                "description": "Investigating the corrupted backup drives. Security alert level: elevated.",
            },
        },
        "followup_type": None,
    },
}

# Every name a ``Tool`` subclass in ``src/maxim`` registers under (#1042). The prompt builder gates its own guidance on
# the agent's roster against this set, so it must be the real universe: tests/unit/test_prompt_names_own_tools.py
# walks the source and fails when a tool is added without its name here.
REGISTERED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "actor",
        "analyze_results",
        "autonomy_level",
        "bash",
        "browse_components",
        "browse_encounters",
        "call_user",
        "causal_links",
        "check_completion",
        "choose",
        "concept_query",
        "create_sandbox_script",
        "damage_component",
        "design_entity",
        "display_mode",
        "edit_file",
        "emit_campaign",
        "energy_status",
        "examine",
        "execute_file",
        "execute_sandbox_script",
        "explain",
        "extend_simulation",
        "finish_simulation",
        "focus_interests",
        "focus_on_sound",
        "generate_scenario",
        "git_commit",
        "git_diff",
        "glob",
        "http_fetch",
        "inject_pain",
        "inspect_aut",
        "internet_access_toggle",
        "internet_search",
        "list_other_outputs",
        "list_sandbox",
        "math",
        "maxim_command",
        "memory_recall",
        "mode_switch",
        "move",
        "novelty_track",
        "observe_actions",
        "pain_history",
        "predict_outcome",
        "query_experiments",
        "read_data_file",
        "read_file",
        "read_other_output",
        "read_sandbox_file",
        "record_experiment",
        "request_directory_change",
        "request_interaction",
        "respond",
        "run_tests",
        "say",
        "scene_summary",
        "search_code",
        "send_message",
        "sense",
        "sense_presence",
        "sense_tools",
        "set_entity_sensor",
        "set_scene",
        "similarity_search",
        "sleep",
        "spawn_sub_simulation",
        "speak",
        "system_stats",
        "temporal_patterns",
        "think",
        "track_target",
        "write_file",
        "write_sandbox_file",
        "write_shared_output",
    }
)
# What the prompt builder's text may mention: the registered names plus the legacy/alias names prompts still describe.
PROMPT_TOOL_NAMES: frozenset[str] = REGISTERED_TOOL_NAMES | frozenset(TOOL_DESCRIPTIONS)


def get_tool_followup_type(tool_name: str, mode_name: str | None = None) -> str | None:
    """Get the followup type for a tool, adjusted by the operational mode.

    ``engage`` (respond AND offer follow-ups) is kept only in an active- or singularity-class mode; in a passive-class
    mode (``passive``, ``observe``, ``sleep``, ...), an unknown name or no mode it is ``respond`` -- judged by the
    mode's CLASS, fail closed (#963 owner decision Q3; before, only the literal name ``"passive"`` downgraded, which no
    shipped code writes). Every other followup type is the tool's own.

    Args:
        tool_name: Name of the tool
        mode_name: The operational mode (``loop_state.operational_mode``)

    Returns:
        Followup type: None, "process", "respond", or "engage"
    """
    tool_info = TOOL_DESCRIPTIONS.get(tool_name, {})
    followup_type = tool_info.get("followup_type")

    if followup_type == "engage":
        definition = get_mode(mode_name)
        if definition is None or definition.name not in ("active", "singularity"):
            return "respond"

    return followup_type


def get_tool_prompt_section(available_tools: set[str]) -> str:
    """Generate a prompt section describing available tools.

    Args:
        available_tools: Set of tool names available in the current mode

    Returns:
        Formatted string describing each tool for LLM context
    """
    import json

    lines = ["Available tools:"]
    for tool_name in sorted(available_tools):
        if tool_name in TOOL_DESCRIPTIONS:
            desc = TOOL_DESCRIPTIONS[tool_name]
            params = ", ".join(f"{k}: {v}" for k, v in desc.get("params", {}).items())
            lines.append(f"  - {tool_name}: {desc['description']}")
            if params:
                lines.append(f"    Parameters: {params}")
            if "example" in desc:
                example_json = json.dumps(desc["example"], separators=(",", ":"))
                lines.append(f"    Example: {example_json}")
        else:
            lines.append(f"  - {tool_name}")

    lines.append("")
    lines.append("CRITICAL: You MUST fill in all required parameters with actual content.")
    lines.append("For 'respond' tool: params.message MUST contain your actual answer to the question.")
    lines.append("For 'speak' tool: params.text MUST contain the text you want to speak.")
    lines.append("Empty params {} is NEVER acceptable when answering questions.")

    return "\n".join(lines)

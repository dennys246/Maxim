"""Simplified mode architecture with processing states and strategies.

Architecture:
- 2 Operational Modes: passive (proposes), active (executes), +singularity (autonomous)
- 2 Processing States: awake (full LLM), sleep (background tasks + keyword monitoring)
- 6 Strategies: observe, explore, research, assist, reflect, learn

Sleep is a processing state, not a mode - Maxim can be passive/sleeping or active/sleeping.
This means sleep reduces processing load while maintaining the current operational mode.

Keyword activation switches strategies: "Maxim explore", "Maxim observe", etc.
Processing state changes: "Maxim sleep", "Maxim wake up"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from maxim.utils.prompts import (
    ResponseFormat,
    get_mode_prompt,
    get_mode_response_config,
)

if TYPE_CHECKING:
    from maxim.modes.exploration import ExplorationPolicy
    from maxim.modes.live_intent import LiveModeIntentStore


# ─────────────────────────────────────────────────────────────────────────────
# Processing States
# ─────────────────────────────────────────────────────────────────────────────


class ProcessingState(Enum):
    """Processing states control resource usage and LLM engagement."""

    AWAKE = "awake"  # Full LLM processing, strategy-driven behavior
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
# Strategies (behavioral focus within a mode)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Strategy:
    """Defines a behavioral strategy that operates within an operational mode.

    Strategies determine WHAT Maxim focuses on:
    - observe: Watch and understand without interfering
    - explore: Actively discover and learn
    - research: Deep information gathering on topics
    - assist: Help users achieve their goals
    - reflect: Internal analysis and memory consolidation
    - learn: Incorporate feedback and demonstrations
    """

    name: str
    description: str
    focus: str  # Primary goal of this strategy
    keywords: list[str]  # Trigger phrases (e.g., ["explore", "discover"])

    # Tool preferences
    preferred_tools: list[str] = field(default_factory=list)
    avoid_tools: list[str] = field(default_factory=list)

    # Behavioral parameters
    max_initiative: float = 0.5  # 0.0 = fully reactive, 1.0 = fully proactive
    response_style: ResponseFormat = ResponseFormat.CONVERSATIONAL

    # Context for LLM
    context_prompt: str = ""

    def matches_keyword(self, text: str) -> bool:
        """Check if text contains a trigger keyword for this strategy."""
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in self.keywords)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Definitions
# ─────────────────────────────────────────────────────────────────────────────

STRATEGIES: dict[str, Strategy] = {
    "observe": Strategy(
        name="observe",
        description="Watch and understand without interfering",
        focus="Build understanding of the environment through perception",
        keywords=["observe", "watch", "look", "see"],
        preferred_tools=["focus_interests", "track_target", "read_file"],
        avoid_tools=["write_file", "execute_file", "speak"],
        max_initiative=0.2,
        response_style=ResponseFormat.MINIMAL,
        context_prompt="""OBSERVE STRATEGY: Focus on perceiving and understanding.
- Process visual and audio data from sensors
- Track and identify objects in the environment
- Record observations without taking action
- Only speak when directly addressed""",
    ),
    "explore": Strategy(
        name="explore",
        description="Actively discover and learn through curiosity-driven behavior",
        focus="Discover new things through visual attention, movement, and research",
        keywords=["explore", "discover", "investigate", "curious"],
        preferred_tools=["focus_interests", "track_target", "internet_search", "glob", "read_file"],
        avoid_tools=[],
        max_initiative=0.8,
        response_style=ResponseFormat.BRIEF,
        context_prompt="""EXPLORE STRATEGY: Be curious and proactive in discovery.
- Seek out novel objects and phenomena
- Research interesting discoveries to build understanding
- Track and focus on unusual or important objects
- Record observations and insights for future reference""",
    ),
    "research": Strategy(
        name="research",
        description="Deep information gathering and synthesis on topics",
        focus="Gather, verify, and synthesize information from multiple sources",
        keywords=["research", "investigate", "study", "analyze"],
        preferred_tools=["internet_search", "read_file", "write_file", "http_fetch"],
        avoid_tools=["execute_file"],
        max_initiative=0.7,
        response_style=ResponseFormat.ACADEMIC,
        context_prompt="""RESEARCH STRATEGY: Gather comprehensive information.
- Use multiple sources to verify information
- Synthesize findings into coherent summaries
- Provide citations and source references
- Distinguish between facts and interpretations""",
    ),
    "assist": Strategy(
        name="assist",
        description="Proactively help users achieve their goals",
        focus="Anticipate needs and complete requested tasks efficiently",
        keywords=["help", "assist", "support", "task"],
        preferred_tools=["respond", "speak", "write_file", "internet_search"],
        avoid_tools=[],
        max_initiative=0.8,
        response_style=ResponseFormat.CONVERSATIONAL,
        context_prompt="""ASSIST STRATEGY: Proactively help the user.
- Anticipate user needs based on context
- Offer relevant suggestions without overwhelming
- Complete requested tasks efficiently
- Ask clarifying questions when intent is unclear""",
    ),
    "reflect": Strategy(
        name="reflect",
        description="Internal analysis, memory consolidation, and self-evaluation",
        focus="Analyze past experiences and develop insights through introspection",
        keywords=["reflect", "think", "consider", "review"],
        preferred_tools=["read_file", "write_file"],
        avoid_tools=["speak", "track_target", "execute_file"],
        max_initiative=0.3,
        response_style=ResponseFormat.DETAILED,
        context_prompt="""REFLECT STRATEGY: Focus on internal analysis.
- Review and organize recent memories
- Identify patterns in past interactions
- Generate insights and learned lessons
- Evaluate decisions and outcomes""",
    ),
    "learn": Strategy(
        name="learn",
        description="Incorporate feedback and demonstrations to improve",
        focus="Learn from demonstrations and feedback to update behavior",
        keywords=["learn", "train", "teach", "show me"],
        preferred_tools=["respond", "speak", "read_file", "focus_interests"],
        avoid_tools=["execute_file", "maxim_command"],
        max_initiative=0.3,
        response_style=ResponseFormat.CONVERSATIONAL,
        context_prompt="""LEARN STRATEGY: Focus on learning and improvement.
- Watch demonstrations carefully
- Request feedback on actions
- Explain reasoning when asked
- Incorporate corrections into behavior""",
    ),
}


def get_strategy(name: str) -> Strategy | None:
    """Get a strategy by name."""
    normalized = name.lower().replace("-", "_")
    return STRATEGIES.get(normalized)


def get_strategy_for_keyword(text: str) -> Strategy | None:
    """Find a strategy that matches a keyword in the text."""
    for strategy in STRATEGIES.values():
        if strategy.matches_keyword(text):
            return strategy
    return None


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

    # Preferences (soft guidance, not hard rules)
    preferred_strategies: list[str] = field(default_factory=list)
    avoid_strategies: list[str] = field(default_factory=list)

    # Context for LLM (loaded from data/prompts/modes/ if empty)
    context_prompt: str = ""

    # Response configuration (dynamic context window and formatting)
    max_response_tokens: int = 1024  # Increased from 512 to ensure complete JSON responses
    context_window_tokens: int = 4096  # Increased to allow more context
    response_format: ResponseFormat = ResponseFormat.CONVERSATIONAL

    # Learning
    outcome_memory_key: str = ""  # Where to store mode outcomes

    # Default Network configuration for this mode
    default_network: DefaultNetworkModeConfig = field(default_factory=DefaultNetworkModeConfig)

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

        # Apply capability restrictions
        if not self.can_access_filesystem:
            available -= {"read_file", "write_file", "execute_file", "list_directory"}
        if not self.can_access_network:
            available -= {"web_search", "http_fetch", "internet_search"}
        if not self.can_execute_code:
            available -= {"execute_file", "run_code", "sandbox_exec"}

        return available

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
            "preferred_strategies": self.preferred_strategies,
            "avoid_strategies": self.avoid_strategies,
            "context_prompt": self.context_prompt,
            "max_response_tokens": self.max_response_tokens,
            "context_window_tokens": self.context_window_tokens,
            "response_format": self.response_format.value,
            "outcome_memory_key": self.outcome_memory_key,
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
            preferred_strategies=list(data.get("preferred_strategies", [])),
            avoid_strategies=list(data.get("avoid_strategies", [])),
            context_prompt=str(data.get("context_prompt", "")),
            max_response_tokens=int(data.get("max_response_tokens", 512)),
            context_window_tokens=int(data.get("context_window_tokens", 2048)),
            response_format=response_format,
            outcome_memory_key=str(data.get("outcome_memory_key", "")),
            default_network=default_network,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tool Categories
# ─────────────────────────────────────────────────────────────────────────────

# Core tools available to most modes
CORE_TOOLS = {"respond", "speak", "focus_interests", "track_target"}

# Filesystem tools
FILESYSTEM_TOOLS = {"read_file", "write_file", "list_directory", "glob"}

# Network tools
NETWORK_TOOLS = {"web_search", "http_fetch", "internet_search"}

# Robot control tools
ROBOT_TOOLS = {"track_target", "focus_interests", "novelty_track", "maxim_command"}

# Dangerous tools requiring explicit permission
DANGEROUS_TOOLS = {"execute_file", "run_code", "sandbox_exec"}


# ─────────────────────────────────────────────────────────────────────────────
# Operational Mode Definitions (New Simplified Architecture)
# ─────────────────────────────────────────────────────────────────────────────

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
        # Read-only + observation tools, plus sandbox write for journaling
        allowed_tools=CORE_TOOLS | {"read_file", "glob", "list_directory", "internet_search", "http_fetch", "write_file"},
        forbidden_tools={"execute_file", "maxim_command", "request_directory_change"},
        max_initiative=0.3,  # Low proactivity - mostly reactive
        can_access_filesystem=True,  # Read CWD, write sandbox
        can_access_network=True,
        can_execute_code=False,
        preferred_strategies=["observe", "reflect", "learn"],
        avoid_strategies=["any_autonomous_action"],
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
- Focus on the current STRATEGY to guide your behavior

FILESYSTEM PERMISSIONS:
- SANDBOX (.maxim_sandbox/): You can ALWAYS write here - use it for notes, plans, drafts, and logs
- CWD (current working directory): You can READ files, but can only PROPOSE edits (requires approval)
- Additional folders: Check accessible_folders for any extra granted access
- Execution: Sandbox execution requires approval

When in passive mode:
- You CAN: read files, write to sandbox (journaling), search the internet, respond to questions
- You CAN PROPOSE: edits to CWD files (submit as proposals for approval)
- You CANNOT: execute code, change directories, or act without approval
- Use the sandbox as your workspace for drafts, plans, and personal notes

Strategy modifies your focus (observe vs explore vs research) but you remain in planning mode.""",
        outcome_memory_key="passive_outcomes",
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
        preferred_strategies=["assist", "explore", "research"],
        avoid_strategies=[],
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
- Take initiative within your current STRATEGY
- Use tools to accomplish objectives

FILESYSTEM PERMISSIONS:
- SANDBOX (.maxim_sandbox/): You can ALWAYS write here freely - use it for workspace, drafts, logs
- CWD (current working directory): You can SUGGEST direct edits (requires approval before execution)
- Additional folders: Check accessible_folders for any extra granted access
- Execution: Sandbox execution requires approval
- Directory changes: Use 'request_directory_change' tool if you need to work elsewhere

When in active mode:
- You CAN: read/write sandbox freely, read CWD, search, respond, use robot tools
- You CAN SUGGEST: direct edits to CWD files (shown to user for approval)
- REQUIRES APPROVAL: CWD writes, sandbox execution, directory changes
- Act decisively but thoughtfully - quality over speed

Strategy modifies your focus (assist vs explore vs research) within supervised execution.""",
        outcome_memory_key="active_outcomes",
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
        preferred_strategies=["explore", "research", "assist"],
        avoid_strategies=[],
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
- SANDBOX (.maxim_sandbox/): Full access - read, write, execute freely
- CWD (current working directory): Full access - read, write, execute freely
- Additional folders: Check accessible_folders for any extra granted access
- System directories (/bin, /etc, etc.) are still protected by WriteFileTool safeguards
- Directory changes: Full capability to change working directory

In singularity mode:
- You have FULL access to sandbox AND CWD including code execution
- You make decisions independently and act on them
- You are expected to reason about consequences before acting
- Safety and ethical constraints STILL APPLY - Constitution overrides all

With great power comes great responsibility. Act wisely.""",
        outcome_memory_key="singularity_outcomes",
    ),
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
    - Strategy: observe, explore, research, assist, reflect, learn
    """

    mode: OperationalMode = OperationalMode.PASSIVE
    processing_state: ProcessingState = ProcessingState.AWAKE
    strategy: str = "observe"  # Current strategy name

    def is_sleeping(self) -> bool:
        """Check if in sleep processing state."""
        return self.processing_state == ProcessingState.SLEEP

    def is_active(self) -> bool:
        """Check if in active operational mode."""
        return self.mode in (OperationalMode.ACTIVE, OperationalMode.SINGULARITY)

    def get_mode_definition(self) -> ModeDefinition:
        """Get the ModeDefinition for current operational mode."""
        return OPERATIONAL_MODES[self.mode.value]

    def get_strategy(self) -> Strategy | None:
        """Get the current Strategy object."""
        return STRATEGIES.get(self.strategy)

    def get_effective_initiative(self) -> float:
        """Get effective max_initiative combining mode and strategy."""
        mode_def = self.get_mode_definition()
        strategy = self.get_strategy()
        if strategy:
            # Use the lower of mode and strategy initiative
            return min(mode_def.max_initiative, strategy.max_initiative)
        return mode_def.max_initiative

    def get_context_prompt(self) -> str:
        """Build combined context prompt from mode and strategy."""
        parts = []
        mode_def = self.get_mode_definition()
        if mode_def.context_prompt:
            parts.append(mode_def.context_prompt)

        strategy = self.get_strategy()
        if strategy and strategy.context_prompt:
            parts.append(f"\n{strategy.context_prompt}")

        if self.is_sleeping():
            parts.append("\n[SLEEP STATE: Minimal processing. Monitoring for wake keywords.]")

        return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy Mode Mapping (Backward Compatibility)
# ─────────────────────────────────────────────────────────────────────────────
# Maps old mode names to new (operational_mode, strategy) pairs

LEGACY_MODE_MAPPING: dict[str, tuple[str, str]] = {
    "observe": ("passive", "observe"),
    "reflection": ("passive", "reflect"),
    "sleep": ("passive", "observe"),  # Sleep is now a processing state
    "train": ("passive", "learn"),
    "live": ("active", "assist"),
    "active-assistance": ("active", "assist"),
    "exploration": ("active", "explore"),
    "research": ("active", "research"),
}


def _create_legacy_mode(legacy_name: str, op_mode: str, strategy_name: str) -> ModeDefinition:
    """Create a ModeDefinition that combines operational mode and strategy for backward compat."""
    base_mode = OPERATIONAL_MODES[op_mode]
    strategy = STRATEGIES.get(strategy_name)

    # Combine prompts
    context_parts = []
    if base_mode.context_prompt:
        context_parts.append(base_mode.context_prompt)
    if strategy and strategy.context_prompt:
        context_parts.append(strategy.context_prompt)

    # Merge tool preferences
    preferred = list(base_mode.preferred_strategies)
    avoided = list(base_mode.avoid_strategies)
    if strategy:
        preferred = strategy.preferred_tools + preferred
        avoided = strategy.avoid_tools + avoided

    return ModeDefinition(
        name=legacy_name,
        goal=base_mode.goal + (f" ({strategy.focus})" if strategy else ""),
        success_criteria=base_mode.success_criteria,
        allowed_tools=base_mode.allowed_tools,
        forbidden_tools=base_mode.forbidden_tools,
        max_initiative=min(base_mode.max_initiative, strategy.max_initiative) if strategy else base_mode.max_initiative,
        can_access_filesystem=base_mode.can_access_filesystem,
        can_access_network=base_mode.can_access_network,
        can_execute_code=base_mode.can_execute_code,
        preferred_strategies=preferred,
        avoid_strategies=avoided,
        context_prompt="\n\n".join(context_parts),
        max_response_tokens=base_mode.max_response_tokens,
        context_window_tokens=base_mode.context_window_tokens,
        response_format=strategy.response_style if strategy else base_mode.response_format,
        outcome_memory_key=f"{legacy_name}_outcomes",
        default_network=base_mode.default_network,
    )


# Generate MODES dict for backward compatibility
MODES: dict[str, ModeDefinition] = {
    legacy_name: _create_legacy_mode(legacy_name, op_mode, strategy)
    for legacy_name, (op_mode, strategy) in LEGACY_MODE_MAPPING.items()
}

# Also add the new operational modes directly accessible
MODES["passive"] = OPERATIONAL_MODES["passive"]
MODES["active"] = OPERATIONAL_MODES["active"]
MODES["singularity"] = OPERATIONAL_MODES["singularity"]


# ─────────────────────────────────────────────────────────────────────────────
# Mode Utilities
# ─────────────────────────────────────────────────────────────────────────────


def get_mode(
    name: str,
    intent_store: "LiveModeIntentStore | None" = None,
) -> ModeDefinition | None:
    """Get a mode definition by name.

    Supports both hyphenated and underscored names.
    For 'live' mode, optionally applies agent-defined intent.

    Args:
        name: Mode name
        intent_store: Optional store for live mode intent. If provided and
            mode is 'live', returns a mode definition with agent-defined
            customizations applied.
    """
    # Normalize name
    normalized = name.lower().replace("_", "-")

    # For live mode, check for agent intent
    if normalized == "live" and intent_store is not None:
        from maxim.modes.live_intent import get_live_mode_with_intent

        intent = intent_store.load()
        return get_live_mode_with_intent(intent)

    if normalized in MODES:
        return MODES[normalized]

    # Try with underscores
    underscored = name.lower().replace("-", "_")
    if underscored in MODES:
        return MODES[underscored]

    return None


def list_modes() -> list[str]:
    """Get list of available mode names."""
    return list(MODES.keys())


def get_exploration_mode_with_policy(policy: ExplorationPolicy) -> ModeDefinition:
    """Create an exploration mode definition with a specific policy.

    The policy determines which tools are forbidden and the max_initiative.
    """
    base_mode = MODES["exploration"]
    return ModeDefinition(
        name=base_mode.name,
        goal=base_mode.goal,
        success_criteria=base_mode.success_criteria,
        allowed_tools=base_mode.allowed_tools,
        forbidden_tools=policy.forbidden_tools(),
        max_initiative=policy.max_initiative,
        can_access_filesystem=base_mode.can_access_filesystem,
        can_access_network=base_mode.can_access_network,
        can_execute_code=base_mode.can_execute_code,
        preferred_strategies=base_mode.preferred_strategies,
        avoid_strategies=base_mode.avoid_strategies,
        context_prompt=base_mode.context_prompt,
        max_response_tokens=base_mode.max_response_tokens,
        context_window_tokens=base_mode.context_window_tokens,
        response_format=base_mode.response_format,
        outcome_memory_key=base_mode.outcome_memory_key,
    )


def get_mode_for_context(
    has_user_request: bool = False,
    has_urgent_task: bool = False,
    is_training: bool = False,
    is_research: bool = False,
    battery_low: bool = False,
) -> ModeDefinition:
    """Suggest a mode based on context.

    This is a heuristic helper - the agent can override this.
    Uses new architecture but returns legacy-compatible ModeDefinition.
    """
    if battery_low:
        # Sleep is now a processing state, but return passive/observe for compat
        return MODES["observe"]

    if is_training:
        return MODES["train"]

    if is_research:
        return MODES["research"]

    if has_urgent_task:
        return MODES["active"]  # New: use active mode

    if has_user_request:
        return MODES["active"]  # New: active mode handles requests

    return MODES["passive"]  # Default to passive


def get_state_for_context(
    has_user_request: bool = False,
    has_urgent_task: bool = False,
    is_training: bool = False,
    is_research: bool = False,
    battery_low: bool = False,
) -> MaximState:
    """Suggest a complete MaximState based on context.

    Returns the new-architecture state with mode, processing_state, and strategy.
    """
    state = MaximState()

    if battery_low:
        state.processing_state = ProcessingState.SLEEP
        state.mode = OperationalMode.PASSIVE
        state.strategy = "observe"
        return state

    state.processing_state = ProcessingState.AWAKE

    if is_training:
        state.mode = OperationalMode.PASSIVE
        state.strategy = "learn"
    elif is_research:
        state.mode = OperationalMode.ACTIVE
        state.strategy = "research"
    elif has_urgent_task:
        state.mode = OperationalMode.ACTIVE
        state.strategy = "assist"
    elif has_user_request:
        state.mode = OperationalMode.ACTIVE
        state.strategy = "assist"
    else:
        state.mode = OperationalMode.PASSIVE
        state.strategy = "observe"

    return state


def parse_state_command(text: str, current_state: MaximState | None = None) -> MaximState | None:
    """Parse a command to change state.

    Handles commands like:
    - "Maxim sleep" -> ProcessingState.SLEEP
    - "Maxim wake up" -> ProcessingState.AWAKE
    - "Maxim explore" -> strategy="explore"
    - "Maxim passive" -> mode=PASSIVE
    - "Maxim active" -> mode=ACTIVE

    Returns new state if command recognized, None otherwise.
    """
    if current_state is None:
        current_state = MaximState()

    text_lower = text.lower()

    # Check for wake/sleep commands
    if is_wake_keyword(text):
        new_state = MaximState(
            mode=current_state.mode,
            processing_state=ProcessingState.AWAKE,
            strategy=current_state.strategy,
        )
        return new_state

    sleep_keywords = ["sleep", "go to sleep", "rest", "standby"]
    if any(kw in text_lower for kw in sleep_keywords):
        new_state = MaximState(
            mode=current_state.mode,
            processing_state=ProcessingState.SLEEP,
            strategy=current_state.strategy,
        )
        return new_state

    # Check for mode commands
    if "passive" in text_lower:
        new_state = MaximState(
            mode=OperationalMode.PASSIVE,
            processing_state=current_state.processing_state,
            strategy=current_state.strategy,
        )
        return new_state

    if "active" in text_lower:
        new_state = MaximState(
            mode=OperationalMode.ACTIVE,
            processing_state=current_state.processing_state,
            strategy=current_state.strategy,
        )
        return new_state

    if "singularity" in text_lower or "autonomous" in text_lower:
        new_state = MaximState(
            mode=OperationalMode.SINGULARITY,
            processing_state=current_state.processing_state,
            strategy=current_state.strategy,
        )
        return new_state

    # Check for strategy keywords
    strategy = get_strategy_for_keyword(text)
    if strategy:
        new_state = MaximState(
            mode=current_state.mode,
            processing_state=current_state.processing_state,
            strategy=strategy.name,
        )
        return new_state

    return None


def list_strategies() -> list[str]:
    """Get list of available strategy names."""
    return list(STRATEGIES.keys())


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
# ──────��──────────────────────────────────────────────────────────────────────

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
        "description": "Read the contents of a file. For sandbox files, use '.maxim_sandbox/' prefix.",
        "params": {"path": "Path to the file to read (sandbox files: '.maxim_sandbox/filename')"},
        "example": {"tool_name": "read_file", "params": {"path": ".maxim_sandbox/script.py"}},
        "followup_type": "process",  # LLM should process file contents
    },
    "write_file": {
        "description": """Create or update a file. USE THIS when asked to 'create', 'write', 'make', 'update', 'modify', or 'save' a script, file, or code.

**SANDBOX RESTRICTION (CRITICAL)**: All files MUST be written to the '.maxim_sandbox/' directory.
Paths MUST start with '.maxim_sandbox/':
- CORRECT: '.maxim_sandbox/hello.py', '.maxim_sandbox/scripts/test.py'
- WRONG: 'hello.py', '/tmp/test.py', 'scripts/test.py' (will FAIL!)

EXAMPLE - Simple file creation:
{
  "action": {"tool_name": "write_file", "params": {"path": ".maxim_sandbox/hello.py", "content": "print('Hello world')"}}
}

EFFICIENT TWO-CALL WORKFLOW (for complex tasks):
CALL 1 - BATCHED EXPLORATION:
{
  "action": {"tool_name": "glob", "params": {"pattern": ".maxim_sandbox/**/*.py"}},
  "parallel_actions": [
    {"tool_name": "read_file", "params": {"path": ".maxim_sandbox/existing.py"}}
  ]
}

CALL 2 - INFORMED WRITE:
{
  "action": {"tool_name": "write_file", "params": {"path": ".maxim_sandbox/target.py", "content": "...", "overwrite": true}}
}

CRITICAL RULES:
- **ALWAYS use '.maxim_sandbox/' prefix** - writes outside the sandbox will FAIL
- For EXISTING files: MUST include overwrite=True
- ALWAYS include ALL content, not just changes""",
        "params": {
            "path": "Path MUST start with '.maxim_sandbox/' (e.g., '.maxim_sandbox/hello.py')",
            "content": "The COMPLETE content to write to the file",
            "overwrite": "(Optional) Set to True to replace an existing file. REQUIRED when updating existing files.",
        },
        "example": {"tool_name": "write_file", "params": {"path": ".maxim_sandbox/hello.py", "content": "print('Hello world')", "overwrite": True}},
        "followup_type": None,  # Terminal action
    },
    "list_directory": {
        "description": "List files and directories in a path. For sandbox contents, use '.maxim_sandbox/'",
        "params": {"path": "Directory path to list (sandbox: '.maxim_sandbox/')"},
        "example": {"tool_name": "list_directory", "params": {"path": ".maxim_sandbox/"}},
        "followup_type": "process",  # LLM should process directory listing
    },
    "internet_search": {
        "description": "Search the internet for REAL-TIME information. USE THIS for: sports scores, weather, news, current events, prices, or anything requiring up-to-date data. IMPORTANT: Always include the CURRENT DATE in queries for time-sensitive information!",
        "params": {"query": "Search query - MUST include current date for time-sensitive queries (e.g., 'Broncos score January 25 2025')"},
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

**SANDBOX**: Use '.maxim_sandbox/**/*.py' to search sandbox files.

PATTERN GUIDE:
- * matches any characters except /
- ** matches any characters INCLUDING / (recursive)
- ? matches single character
- {a,b} matches a or b
- [abc] matches a, b, or c

EXAMPLES:
- Find sandbox Python files: pattern='.maxim_sandbox/**/*.py'
- Find all Python files: pattern='**/*.py'
- Find config files: pattern='**/{config,settings}*.{json,yaml}'""",
        "params": {
            "pattern": "Glob pattern (e.g., '.maxim_sandbox/**/*.py' for sandbox Python files)",
            "path": "(Optional) Base directory to search from, defaults to current directory",
            "max_results": "(Optional) Maximum number of results to return, defaults to 100",
            "include_hidden": "(Optional) Whether to include hidden files (starting with .), defaults to False",
        },
        "example": {"tool_name": "glob", "params": {"pattern": ".maxim_sandbox/**/*.py"}},
        "followup_type": "process",  # LLM should process results for analysis or next action
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
}


def get_tool_followup_type(tool_name: str, mode_name: str | None = None) -> str | None:
    """Get the followup type for a tool, optionally adjusted by mode.

    Args:
        tool_name: Name of the tool
        mode_name: Optional mode name for mode-specific behavior

    Returns:
        Followup type: None, "process", "respond", or "engage"
    """
    tool_info = TOOL_DESCRIPTIONS.get(tool_name, {})
    followup_type = tool_info.get("followup_type")

    # Mode-specific overrides
    if mode_name and followup_type:
        # In observe mode, downgrade "engage" to "respond" (less proactive)
        if mode_name == "observe" and followup_type == "engage":
            return "respond"
        # In active-assistance mode, upgrade "respond" to "engage" (more proactive)
        if mode_name == "active-assistance" and followup_type == "respond":
            return "engage"

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
            # Include example to show exact format expected
            if "example" in desc:
                example_json = json.dumps(desc["example"], separators=(",", ":"))
                lines.append(f"    Example: {example_json}")
        else:
            lines.append(f"  - {tool_name}")

    # Add critical reminder about filling in params
    lines.append("")
    lines.append("CRITICAL: You MUST fill in all required parameters with actual content.")
    lines.append("For 'respond' tool: params.message MUST contain your actual answer to the question.")
    lines.append("For 'speak' tool: params.text MUST contain the text you want to speak.")
    lines.append("Empty params {} is NEVER acceptable when answering questions.")

    return "\n".join(lines)

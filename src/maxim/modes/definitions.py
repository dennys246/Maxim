"""Mode definitions as goals with constraints.

Each mode defines a goal and constraints, not a fixed procedure.
The agent selects strategies to achieve the goal.

Prompts are loaded from data/prompts/modes/ for easy editing.
Response configuration (tokens, format) enables dynamic context windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from maxim.utils.prompts import (
    ResponseFormat,
    get_mode_prompt,
    get_mode_response_config,
)

if TYPE_CHECKING:
    from maxim.modes.exploration import ExplorationPolicy
    from maxim.modes.live_intent import LiveModeIntentStore


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
    max_response_tokens: int = 512
    context_window_tokens: int = 2048
    response_format: ResponseFormat = ResponseFormat.CONVERSATIONAL

    # Learning
    outcome_memory_key: str = ""  # Where to store mode outcomes

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
        if self.max_response_tokens == 512 and self.context_window_tokens == 2048:
            config = get_mode_response_config(self.name)
            self.max_response_tokens = config.get("max_response_tokens", 512)
            self.context_window_tokens = config.get("context_window_tokens", 2048)
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
        )


# ─────────────────────────────────────────────────────────────────────────────
# Built-in Mode Definitions
# ─────────────────────────────────────────────────────────────────────────────


# Core tools available to most modes
CORE_TOOLS = {"respond", "speak", "focus_interests", "track_target"}

# Filesystem tools
FILESYSTEM_TOOLS = {"read_file", "write_file", "list_directory"}

# Network tools
NETWORK_TOOLS = {"web_search", "http_fetch", "internet_search"}

# Robot control tools
ROBOT_TOOLS = {"track_target", "focus_interests", "novelty_track", "maxim_command"}

# Dangerous tools requiring explicit permission
DANGEROUS_TOOLS = {"execute_file", "run_code", "sandbox_exec"}

MODES: dict[str, ModeDefinition] = {
    # Prompts are loaded from data/prompts/modes/<name>.txt
    # Response config is loaded from data/prompts/modes/<name>_config.json
    # or uses defaults from get_mode_response_config()
    "observe": ModeDefinition(
        name="observe",
        goal="Build understanding of the environment without interference",
        success_criteria=[
            "Detected and tracked objects in scene",
            "Recorded speech/sounds",
            "No unnecessary actions taken",
        ],
        allowed_tools={"focus_interests", "track_target", "novelty_track", "read_file"},
        forbidden_tools={"maxim_command", "write_file", "execute_file", "speak"},
        max_initiative=0.2,
        can_access_filesystem=True,  # Read-only
        can_access_network=False,
        can_execute_code=False,
        preferred_strategies=[
            "watch_and_learn",
            "gather_information",
        ],
        avoid_strategies=[
            "interrupt_user",
            "offer_unsolicited_advice",
        ],
        outcome_memory_key="observation_outcomes",
    ),
    "reflection": ModeDefinition(
        name="reflection",
        goal="Consolidate memories, analyze past experiences, and develop insights through introspection",
        success_criteria=[
            "Reviewed and organized recent memories",
            "Identified patterns in past interactions",
            "Generated insights or learned lessons",
            "Updated internal models based on experience",
        ],
        allowed_tools={"respond", "read_file", "write_file"},
        forbidden_tools={"execute_file", "speak", "track_target"},
        max_initiative=0.3,
        can_access_filesystem=True,
        can_access_network=False,
        can_execute_code=False,
        preferred_strategies=[
            "memory_consolidation",
            "pattern_recognition",
            "self_evaluation",
            "insight_generation",
        ],
        avoid_strategies=[
            "external_interaction",
            "proactive_engagement",
            "new_data_gathering",
        ],
        outcome_memory_key="reflection_outcomes",
    ),
    "active-assistance": ModeDefinition(
        name="active-assistance",
        goal="Proactively help the user achieve their objectives",
        success_criteria=[
            "Anticipated user needs",
            "Offered relevant suggestions",
            "Completed requested tasks",
        ],
        # All tools except dangerous ones
        forbidden_tools={"execute_file"},
        max_initiative=0.8,
        can_access_filesystem=True,
        can_access_network=True,
        can_execute_code=False,
        preferred_strategies=[
            "anticipate_needs",
            "offer_suggestions",
            "prepare_resources",
            "targeted_web_search",
        ],
        avoid_strategies=[
            "wait_passively",
        ],
        outcome_memory_key="assistance_outcomes",
    ),
    "sleep": ModeDefinition(
        name="sleep",
        goal="Minimize resource usage while remaining responsive to wake commands",
        success_criteria=[
            "Minimal processing",
            "Audio monitoring active",
            "Responded to wake command",
        ],
        allowed_tools={"respond"},  # Only respond to wake commands
        forbidden_tools={"write_file", "execute_file", "maxim_command", "speak"},
        max_initiative=0.0,
        can_access_filesystem=False,
        can_access_network=False,
        can_execute_code=False,
        preferred_strategies=[
            "minimal_processing",
            "listen_for_wake",
        ],
        avoid_strategies=[
            "any_proactive_action",
        ],
        outcome_memory_key="sleep_outcomes",
    ),
    "live": ModeDefinition(
        name="live",
        goal="Understand reality and help people. Full operational mode with all capabilities active",
        success_criteria=[
            "Processing perception data",
            "Responding to interactions",
            "Executing requested tasks",
        ],
        forbidden_tools=set(),  # No restrictions
        max_initiative=0.6,
        can_access_filesystem=True,
        can_access_network=True,
        can_execute_code=False,  # Still need explicit permission
        preferred_strategies=[
            "respond_concisely",
            "gather_information",
            "targeted_web_search",
        ],
        avoid_strategies=[],
        outcome_memory_key="live_outcomes",
    ),
    "train": ModeDefinition(
        name="train",
        goal="Learn from demonstrations and feedback to improve behavior",
        success_criteria=[
            "Recorded demonstrations",
            "Incorporated feedback",
            "Updated behavior patterns",
        ],
        allowed_tools={"respond", "speak", "read_file", "write_file", "focus_interests"},
        forbidden_tools={"execute_file", "maxim_command"},
        max_initiative=0.3,
        can_access_filesystem=True,
        can_access_network=False,
        can_execute_code=False,
        preferred_strategies=[
            "observe_demonstrations",
            "request_feedback",
            "explain_reasoning",
        ],
        avoid_strategies=[
            "autonomous_action",
        ],
        outcome_memory_key="training_outcomes",
    ),
    "research": ModeDefinition(
        name="research",
        goal="Gather and synthesize information on a specific topic",
        success_criteria=[
            "Identified relevant sources",
            "Synthesized findings",
            "Provided citations",
        ],
        allowed_tools={"respond", "speak", "read_file", "write_file", "web_search", "http_fetch"},
        forbidden_tools={"execute_file", "maxim_command"},
        max_initiative=0.7,
        can_access_filesystem=True,
        can_access_network=True,
        can_execute_code=False,
        preferred_strategies=[
            "targeted_web_search",
            "verify_with_sources",
            "gather_information",
            "explain_reasoning",
        ],
        avoid_strategies=[
            "single_source_reliance",
        ],
        outcome_memory_key="research_outcomes",
    ),
    "exploration": ModeDefinition(
        name="exploration",
        goal="Actively explore and discover through visual attention, physical movement, information gathering, and autonomous analysis",
        success_criteria=[
            "discovered_new_objects",
            "gathered_contextual_information",
            "built_environmental_knowledge",
            "created_useful_analysis",
            "trained_or_improved_models",
        ],
        forbidden_tools=set(),  # Dynamically computed from ExplorationPolicy
        max_initiative=0.8,  # Can be overridden by ExplorationPolicy
        can_access_filesystem=True,
        can_access_network=True,
        can_execute_code=False,
        preferred_strategies=[
            "novelty_exploration",
            "curiosity_driven_search",
            "targeted_web_search",
            "gather_information",
            "autonomous_analysis",
            "incremental_training",
        ],
        avoid_strategies=[
            "wait_passively",
            "minimal_processing",
        ],
        outcome_memory_key="exploration_outcomes",
    ),
}


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
    """
    if battery_low:
        return MODES["sleep"]

    if is_training:
        return MODES["train"]

    if is_research:
        return MODES["research"]

    if has_urgent_task:
        return MODES["active-assistance"]

    if has_user_request:
        return MODES["reflection"]

    return MODES["observe"]


# ─────────────────────────────────────────────────────────────────────────────
# Tool Descriptions for LLM Prompts
# ─────────────────────────────────────────────────────────────────────────────

TOOL_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "respond": {
        "description": "Send a text response to the user (displays in CLI)",
        "params": {"message": "The text message to send"},
        "example": {"tool_name": "respond", "params": {"message": "Hello!"}},
    },
    "speak": {
        "description": "Speak a message aloud using text-to-speech",
        "params": {"text": "The text to speak aloud"},
        "example": {"tool_name": "speak", "params": {"text": "Hello!"}},
    },
    "read_file": {
        "description": "Read the contents of a file",
        "params": {"path": "Path to the file to read"},
        "example": {"tool_name": "read_file", "params": {"path": "data/config.json"}},
    },
    "write_file": {
        "description": "Write content to a file (creates or overwrites)",
        "params": {"path": "Path to write to", "content": "Content to write"},
        "example": {"tool_name": "write_file", "params": {"path": "output.txt", "content": "Hello"}},
    },
    "list_directory": {
        "description": "List files and directories in a path",
        "params": {"path": "Directory path to list"},
        "example": {"tool_name": "list_directory", "params": {"path": "data/"}},
    },
    "web_search": {
        "description": "Search the web for information",
        "params": {"query": "Search query"},
        "example": {"tool_name": "web_search", "params": {"query": "weather today"}},
    },
    "http_fetch": {
        "description": "Fetch content from a URL",
        "params": {"url": "URL to fetch"},
        "example": {"tool_name": "http_fetch", "params": {"url": "https://example.com"}},
    },
    "track_target": {
        "description": "Track an object with the robot's head/camera",
        "params": {"target_class": "Object class to track (e.g., 'person', 'cup')"},
        "example": {"tool_name": "track_target", "params": {"target_class": "person"}},
    },
    "focus_interests": {
        "description": "Scan for and focus on interesting objects in the scene",
        "params": {},
        "example": {"tool_name": "focus_interests", "params": {}},
    },
    "novelty_track": {
        "description": "Track the most novel/interesting object in view",
        "params": {},
        "example": {"tool_name": "novelty_track", "params": {}},
    },
    "maxim_command": {
        "description": "Execute a Maxim system command (mode changes, shutdown, etc.)",
        "params": {"command": "The command to execute"},
        "example": {"tool_name": "maxim_command", "params": {"command": "sleep"}},
    },
}


def get_tool_prompt_section(available_tools: set[str]) -> str:
    """Generate a prompt section describing available tools.

    Args:
        available_tools: Set of tool names available in the current mode

    Returns:
        Formatted string describing each tool for LLM context
    """
    lines = ["Available tools:"]
    for tool_name in sorted(available_tools):
        if tool_name in TOOL_DESCRIPTIONS:
            desc = TOOL_DESCRIPTIONS[tool_name]
            params = ", ".join(f"{k}: {v}" for k, v in desc.get("params", {}).items())
            lines.append(f"  - {tool_name}: {desc['description']}")
            if params:
                lines.append(f"    Parameters: {params}")
        else:
            lines.append(f"  - {tool_name}")
    return "\n".join(lines)

"""Prompt construction for the LLM worker.

Contains all _build_* methods extracted from LLMWorker. Static methods
become module-level functions; instance methods become methods on
PromptBuilder which receives dependencies at construction.
"""

from __future__ import annotations

import json as _json
import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any

from maxim.agents.autonomy import AutonomyLevel
from maxim.agents.llm_context import _load_foundational_context
from maxim.agents.llm_fallback import (
    generate_simple_answer,
    ReasoningCarryover,
)
from maxim.agents.llm_types import LLMRequest, ModeInfo
from maxim.agents.prompt_budgeter import (
    PromptBudgeter,
    SectionPriority,
    _compact_conversation,
    _truncate_context_pool,
    _truncate_conversation,
    _truncate_manifest,
    _truncate_reasoning_carryover,
    _truncate_tool_guidance,
)
from maxim.utils.coding_guidelines import build_coding_context

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Static Section Builders (formerly @staticmethod on LLMWorker)
# ─────────────────────────────────────────────────────────────────────────────


def build_planning_banner(autonomy_level: AutonomyLevel) -> str:
    """Build the planning mode banner text."""
    if autonomy_level != AutonomyLevel.PLANNING:
        return ""
    lines = [
        "!" * 60,
        "!!! CRITICAL: PLANNING MODE - YOU MUST ASK PERMISSION FIRST !!!",
        "!" * 60,
        "",
        "DO NOT output raw JSON. You MUST follow this EXACT format:",
        "",
        "STEP 1: Write a proposal IN PLAIN ENGLISH asking for permission",
        "STEP 2: Write the EXACT delimiter: <|action_json|>",
        "STEP 3: Write the JSON object",
        "",
        "=== CORRECT FORMAT EXAMPLE 1 ===",
        "I'd like to search the internet for the current Broncos vs Patriots score.",
        "May I proceed with this search?",
        "<|action_json|>",
        '{"action": {"tool_name": "internet_search", "params": {"query": "Broncos vs Patriots score today"}}, "confidence": 0.9}',
        "",
        "=== CORRECT FORMAT EXAMPLE 2 ===",
        "To answer your question about the weather, I need to search online.",
        "Should I look up the current weather for you?",
        "<|action_json|>",
        '{"action": {"tool_name": "internet_search", "params": {"query": "current weather"}}, "confidence": 0.9}',
        "",
        "=== WRONG (DO NOT DO THIS) ===",
        '{"action": {"tool_name": "internet_search", ...}}  <-- NO! Missing proposal text!',
        "",
        "YOUR RESPONSE MUST START WITH PLAIN TEXT, NOT JSON!",
        "!" * 60,
    ]
    return "\n".join(lines)


def build_modification_section(mod: dict[str, Any]) -> str:
    """Build the modification request section."""
    lines = [
        "=" * 60,
        ">>> ACTION MODIFICATION REQUESTED <<<",
        "=" * 60,
        "",
        "The user requested a modification to the following proposed action:",
        "",
        "ORIGINAL ACTION:",
        f"  Tool: {mod.get('original_tool_name', 'unknown')}",
        f"  Parameters: {_json.dumps(mod.get('original_action', {}).get('params', {}), indent=4)}",
        f"  Original reasoning: {mod.get('original_reasoning', 'not provided')}",
        "",
        "USER'S MODIFICATION REQUEST:",
        f'  "{mod.get("user_modification", "")}"',
        "",
        "YOUR TASK: Revise the action based on the user's feedback.",
        "- Interpret what change the user wants",
        "- Keep the same tool if appropriate, or choose a different one",
        "- Update the parameters according to the user's request",
        "- Provide updated reasoning",
        "",
        "Respond with a REVISED action that incorporates the user's modification.",
        "=" * 60,
    ]
    return "\n".join(lines)


def build_identity_section(mode: ModeInfo, request: LLMRequest, date_str: str, time_str: str) -> str:
    """Build the system identity and operational state section."""
    lines = [
        "You are Maxim, a robot assistant.",
        "",
        "=== OPERATIONAL STATE ===",
        f"Mode: {mode.name.upper()}",
        f"Mode goal: {mode.goal}",
    ]

    lines.append(f"Autonomy level: {request.autonomy_level.value if request.autonomy_level else 'unknown'}")

    if request.is_sleeping:
        lines.append("Processing state: SLEEP (minimal processing, monitoring for wake keywords)")
    else:
        lines.append("Processing state: AWAKE")

    return "\n".join(lines)


def build_datetime_section(date_str: str, time_str: str) -> str:
    """Build the date/time section."""
    lines = [
        "=== CURRENT DATE/TIME (IMPORTANT) ===",
        f"Today is {date_str}",
        f"Current time: {time_str}",
        "When searching for current events, scores, or news, ALWAYS include the date in your query.",
        f"Example: Instead of 'Broncos game score', search 'Broncos game score {date_str}'",
    ]
    return "\n".join(lines)


def build_tools_section(request: LLMRequest, mode_name: str = "passive") -> str:
    """Build the available tools section, adapted to operational mode."""
    lines = ["=== Available Tools ==="]
    tool_list = sorted(request.available_tools)
    for tool_name in tool_list:
        tool_info = request.tool_descriptions.get(tool_name, {})
        if isinstance(tool_info, dict) and tool_info:
            desc = tool_info.get("description", "")
            params = tool_info.get("params", {})
            example = tool_info.get("example", {})
            lines.append(f"- {tool_name}: {desc}")
            if params:
                param_strs = [f"{k}={v}" for k, v in params.items()]
                lines.append(f"    REQUIRED params: {', '.join(param_strs)}")
            if example:
                lines.append(f"    Example: {_json.dumps(example)}")
        elif isinstance(tool_info, str) and tool_info:
            lines.append(f"- {tool_name}: {tool_info}")
        else:
            lines.append(f"- {tool_name}")

    result = "\n".join(lines)

    # In singularity mode, remove .maxim_workspace/ prefix from examples
    if mode_name == "singularity":
        result = result.replace(".maxim_workspace/", "")

    return result


def build_tools_section_filtered(
    request: LLMRequest,
    tool_names: list[str],
    mode_name: str = "passive",
) -> str:
    """Build tools section for only the specified tool names (relevant subset)."""
    lines = ["=== Available Tools ==="]
    for tool_name in sorted(tool_names):
        tool_info = request.tool_descriptions.get(tool_name, {})
        if isinstance(tool_info, dict) and tool_info:
            desc = tool_info.get("description", "")
            params = tool_info.get("params", {})
            example = tool_info.get("example", {})
            lines.append(f"- {tool_name}: {desc}")
            if params:
                param_strs = [f"{k}={v}" for k, v in params.items()]
                lines.append(f"    REQUIRED params: {', '.join(param_strs)}")
            if example:
                lines.append(f"    Example: {_json.dumps(example)}")
        elif isinstance(tool_info, str) and tool_info:
            lines.append(f"- {tool_name}: {tool_info}")
        else:
            lines.append(f"- {tool_name}")

    result = "\n".join(lines)
    if mode_name == "singularity":
        result = result.replace(".maxim_workspace/", "")
    return result


def scan_workspace_entries(workspace_path: str) -> list[str]:
    """Scan .maxim_workspace/ and return formatted file entries."""
    import os

    if not os.path.isdir(workspace_path):
        return []
    entries: list[str] = []
    try:
        for root, _dirs, files in os.walk(workspace_path):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, os.path.dirname(workspace_path))
                try:
                    size = os.path.getsize(fpath)
                    size_str = f"{size}B" if size < 1024 else f"{size // 1024}KB"
                    entries.append(f"  {rel} ({size_str})")
                except OSError:
                    entries.append(f"  {rel}")
    except OSError:
        return []
    return entries


def build_workspace_manifest(mode_name: str = "passive", cwd: str | None = None) -> str:
    """Build a file manifest that adapts to the operational mode."""
    import os
    from maxim.utils.filesystem_policy import (
        get_effective_cwd,
        get_workspace_path,
        scan_cwd_tree,
    )

    if cwd is None:
        cwd = get_effective_cwd()
    workspace = get_workspace_path(cwd)
    cwd_name = os.path.basename(cwd) or cwd
    sections: list[str] = []

    if mode_name == "singularity":
        cwd_entries = scan_cwd_tree(cwd, max_depth=2, max_entries=30)
        n_entries = len(cwd_entries)
        sections.append(f"=== PROJECT DIRECTORY ({n_entries} entries, CWD: {cwd_name}) ===")
        sections.append("You have FULL read/write access to all files below.")
        sections.append("!! IMPORTANT: These files ALREADY EXIST. Do NOT recreate them. !!")
        sections.extend(cwd_entries)
        ws_entries = scan_workspace_entries(workspace)
        if ws_entries:
            sections.append(f"\n.maxim_workspace/ also available for scratch/drafts ({len(ws_entries)} files).")
        sections.append("\nRULES:")
        sections.append("  1. Use read_file FIRST to see current contents before editing")
        sections.append("  2. Then write_file with overwrite=true to update")
        sections.append("  3. NEVER write a brand-new file that duplicates an existing one")
        if n_entries >= 5:
            sections.append(
                "PLAN FIRST: With many project files, use 'respond' to outline your approach before making changes."
            )
    else:
        ws_entries = scan_workspace_entries(workspace)
        if ws_entries:
            n_files = len(ws_entries)
            sections.append(f"=== EXISTING WORKSPACE ({n_files} file{'s' if n_files != 1 else ''}) ===")
            sections.append("!! IMPORTANT: These files ALREADY EXIST. Do NOT recreate them. !!")
            sections.extend(ws_entries[:15])
            if n_files > 15:
                sections.append(f"  ... and {n_files - 15} more files")
            sections.append("RULES for existing files:")
            sections.append("  1. Use read_file FIRST to see current contents")
            sections.append("  2. Then write_file with overwrite=true to update")
            sections.append("  3. NEVER write a brand-new file that duplicates an existing one")
            if n_files >= 3:
                sections.append(
                    "PLAN FIRST: With multiple existing files, use 'respond' to outline "
                    "your approach before making changes."
                )

        cwd_entries = scan_cwd_tree(cwd, max_depth=1, max_entries=10)
        if cwd_entries:
            sections.append(f"\n=== CWD Context (read-only: {cwd_name}) ===")
            if mode_name == "active":
                sections.append("You can READ these files and SUGGEST edits (requires approval).")
            else:
                sections.append("You can READ these files. CWD edits must be proposed as plans.")
            sections.extend(cwd_entries)

    return "\n".join(sections) if sections else ""


def build_tool_guidance_core(mode_name: str = "passive") -> str:
    """Build compact essential tool guidance, adapted to operational mode."""
    lines = [
        "=== Tool Parameters ===",
        '- respond: {"message": "your answer"}',
        '- speak: {"text": "text to speak"}',
    ]

    if mode_name == "singularity":
        lines.extend(
            [
                '- write_file: {"path": "src/main.py", "content": "code", "overwrite": true}',
                '- read_file: {"path": "src/main.py"}',
                '- internet_search: {"query": "search query"}',
                "",
                "=== File Operation Rules ===",
                "You can read and write ANY file within the project directory.",
                "EXISTING files: read_file FIRST, then write_file with overwrite=true.",
                "NEW files: write_file (no overwrite needed).",
                "NEVER blindly overwrite — always read first to understand current contents.",
            ]
        )
    else:
        lines.extend(
            [
                '- write_file: {"path": ".maxim_workspace/example.py", "content": "...", "overwrite": true}',
                '- read_file: {"path": ".maxim_workspace/example.py"}',
                '- internet_search: {"query": "search query"}',
                "",
                "=== File Operation Rules ===",
                "All file writes MUST use '.maxim_workspace/' prefix.",
                "EXISTING files: read_file FIRST, then write_file with overwrite=true.",
                "NEW files: write_file (no overwrite needed).",
                "NEVER blindly overwrite — always read first to understand current contents.",
            ]
        )
        if mode_name == "active":
            lines.append("CWD files: You can read freely and suggest edits (requires approval).")
        else:
            lines.append("CWD files: You can read freely. Edits must be proposed as plans.")

    lines.extend(
        [
            "",
            "=== Planning Rule ===",
            "For changes that touch multiple files or significantly alter existing code:",
            "  1. Use 'respond' to outline your plan FIRST",
            "  2. Wait for user confirmation before executing",
            "  3. Then read_file → modify → write_file for each file",
            "If request is unclear, use 'respond' to ask for clarification.",
        ]
    )
    return "\n".join(lines)


def build_tool_guidance_extended(mode_name: str = "passive") -> str:
    """Build extended tool selection guidance, adapted to operational mode."""
    lines = [
        "=== Tool Selection ===",
        "- REAL-TIME DATA (scores, weather, news, prices): Use 'internet_search'",
        "- KNOWLEDGE from memory (what is X, explain Y): Use 'respond'",
        "- CREATE/WRITE FILES (create script, make file): Use 'write_file'",
        "- VISUAL COMMANDS (look at, focus on, track): Use 'focus_interests', 'track_target'",
        "- MATH CALCULATIONS: Use 'math' with sqrt/factorial/compute",
        "  For multi-step expressions, decompose in PEMDAS order (parentheses, exponents,",
        "  multiply/divide, add/subtract). Use store_value/recall_value for intermediates.",
        "- MATH MEMORY: Use 'math' with recall_memory/store_memory",
        "",
        "=== FILE WORKSPACE ===",
    ]
    if mode_name == "singularity":
        lines.append("Project directory: full read/write access to all files.")
        lines.append(
            "Workspace (.maxim_workspace/): drafts/ (code), notes/ (thinking), plans/ (proposals), scratch/ (temp)"
        )
    else:
        lines.append("Workspace: drafts/ (code), notes/ (thinking), plans/ (proposals), scratch/ (temp)")
    lines.extend(
        [
            "",
            "=== BATCHED TOOL CALLS ===",
            "Batch exploration with parallel_actions for efficiency.",
        ]
    )
    return "\n".join(lines)


def build_observation_section(percept: Any) -> str:
    """Build the current observation section."""
    lines = ["=== Current Observation ==="]
    if percept.transcript_chunk:
        lines.append(f'Heard: "{percept.transcript_chunk[:200]}"')
    if percept.detections:
        objects = [d.get("label", "?") for d in percept.detections[:5]]
        lines.append(f"Visible objects: {', '.join(objects)}")
    if percept.cli_input:
        lines.append(f'User input: "{percept.cli_input}"')
    return "\n".join(lines) if len(lines) > 1 else ""


def build_instructions_section(request: LLMRequest) -> str:
    """Build the response format instructions section."""
    lines = ["=== Instructions ==="]
    if request.autonomy_level == AutonomyLevel.PLANNING:
        lines.extend(
            [
                "!" * 40,
                "FINAL REMINDER: PLANNING MODE ACTIVE!",
                "Your response MUST be:",
                "1. Plain text proposal asking permission (NOT JSON!)",
                "2. The delimiter <|action_json|>",
                "3. Then the JSON",
                "START YOUR RESPONSE WITH PLAIN ENGLISH TEXT!",
                "!" * 40,
            ]
        )
    else:
        lines.extend(
            [
                "Respond with a compact JSON object. IMPORTANT: Put fields in this order:",
                '  "action": {"tool_name": "<tool>", "params": {...}}',
                '  "confidence": 0.0-1.0',
                '  "reasoning": "Brief explanation (1 sentence)"',
                "Keep response compact. Do not include optional fields.",
            ]
        )
    return "\n".join(lines)


def is_realtime_request(question_text: str) -> bool:
    """Check if the question is asking for real-time data."""
    realtime_keywords = [
        "score",
        "game",
        "match",
        "playing",
        "vs",
        "versus",
        "weather",
        "temperature",
        "forecast",
        "news",
        "latest",
        "current",
        "today",
        "now",
        "live",
        "price",
        "cost",
        "stock",
        "bitcoin",
        "crypto",
        "broncos",
        "patriots",
        "lakers",
        "yankees",
    ]
    q_lower = question_text.lower() if question_text else ""
    return any(kw in q_lower for kw in realtime_keywords)


# ─────────────────────────────────────────────────────────────────────────────
# Budget Context Helpers (formerly @staticmethod on LLMWorker)
# ─────────────────────────────────────────────────────────────────────────────


def compute_budget_tier(policy: Any, totals: dict[str, float]) -> str:
    ratios = []
    if policy.max_cost_per_hour > 0:
        ratios.append(totals["hourly"] / policy.max_cost_per_hour)
    if policy.max_cost_per_day > 0:
        ratios.append(totals["daily"] / policy.max_cost_per_day)
    if policy.max_cost_per_month > 0:
        ratios.append(totals["monthly"] / policy.max_cost_per_month)
    if not ratios:
        return "normal"
    max_ratio = max(ratios)
    if max_ratio >= 1.0:
        return "blocked"
    if max_ratio >= policy.cost_critical_threshold:
        return "critical"
    if max_ratio >= policy.cost_warning_threshold:
        return "warning"
    return "normal"


def format_usd(value: float | None) -> str:
    if value is None:
        return "unlimited"
    if value <= 0:
        return "$0.00"
    return f"${value:.2f}"


def format_spend_rate(rate: Any, min_samples: int) -> str:
    if rate.samples < min_samples or rate.value <= 0:
        return f"immature (samples={rate.samples})"
    return f"${rate.value:.2f}/hr"


def estimate_hours_until_limit(
    remaining: float | None,
    preferred: Any,
    fallback_first: Any,
    min_samples: int,
    fallback_second: Any | None = None,
) -> tuple[str, str]:
    note = ""
    rate = None
    if preferred.samples >= min_samples and preferred.value > 0:
        rate = preferred.value
    elif fallback_first.samples >= min_samples and fallback_first.value > 0:
        rate = fallback_first.value
        note = " (estimated from shorter window)"
    elif fallback_second and fallback_second.samples >= min_samples and fallback_second.value > 0:
        rate = fallback_second.value
        note = " (estimated from shorter window)"
    if rate is None or remaining is None or remaining <= 0:
        return "unknown", ""
    hours = remaining / rate
    return f"{hours:.1f}h", note


def cost_energy_divergence(policy: Any, totals: dict[str, float]) -> str:
    try:
        from maxim.energy.registry import get_global_registry
    except Exception:
        return "none"
    registry = get_global_registry()
    budget = registry.get_budget("llm") if registry else None
    if budget is None:
        return "none"
    energy_pct = budget.percentage
    remaining_pcts = []
    if policy.max_cost_per_hour > 0:
        remaining_pcts.append(1.0 - (totals["hourly"] / policy.max_cost_per_hour))
    if policy.max_cost_per_day > 0:
        remaining_pcts.append(1.0 - (totals["daily"] / policy.max_cost_per_day))
    if policy.max_cost_per_month > 0:
        remaining_pcts.append(1.0 - (totals["monthly"] / policy.max_cost_per_month))
    if not remaining_pcts:
        return "none"
    cost_remaining_pct = min(remaining_pcts) * 100.0
    if energy_pct > 50.0 and cost_remaining_pct < 10.0:
        return "high"
    if energy_pct > 50.0 and cost_remaining_pct < 30.0:
        return "moderate"
    return "none"


# ─────────────────────────────────────────────────────────────────────────────
# PromptBuilder — encapsulates instance methods that need LLMWorker state
# ─────────────────────────────────────────────────────────────────────────────


class PromptBuilder:
    """Builds prompts for LLM requests.

    Receives dependencies at construction so it doesn't need access to LLMWorker.
    """

    def __init__(
        self,
        llm: Any,
        reasoning_carryover: ReasoningCarryover,
        n_ctx: int,
        token_counter: Any,
        tool_index: Any = None,
    ) -> None:
        self._llm = llm
        self._reasoning_carryover = reasoning_carryover
        self._n_ctx = n_ctx
        self._tool_index = tool_index
        self._token_counter = token_counter

    def build_prompt(self, request: LLMRequest) -> str:
        """Build prompt from request."""
        context = request.context

        # Check for action followup
        action_followup_input = ""
        if context.cli_inputs:
            for cli_input in context.cli_inputs:
                if cli_input and cli_input.startswith("[ACTION_FOLLOWUP"):
                    action_followup_input = cli_input
                    break
                if cli_input and cli_input.startswith("[SEARCH RESULT"):
                    action_followup_input = cli_input
                    break

        if action_followup_input:
            return self._build_followup_prompt(action_followup_input)

        # Check for pending user input
        user_question = ""
        if request.triggering_input:
            user_question = request.triggering_input
        elif context.cli_inputs:
            latest_input = context.cli_inputs[-1] if context.cli_inputs else None
            if latest_input and ("maxim" in latest_input.lower() or "reachy" in latest_input.lower()):
                user_question = latest_input

        if not user_question:
            if not request.use_tool_prompting:
                return ""

        # Extract question text
        question_text = ""
        if user_question:
            question_text = user_question.lower()
            for wake_word in ["maxim", "reachy"]:
                question_text = question_text.replace(wake_word, "")
            question_text = question_text.strip().lstrip(",").lstrip(":").strip()
            if question_text.endswith("?"):
                question_text = question_text[:-1].strip()

        now = datetime.now()
        date_str = now.strftime("%A, %B %d, %Y")
        time_str = now.strftime("%I:%M %p")

        # Try simple answers first
        if question_text:
            answer = generate_simple_answer(question_text, date_str, time_str)
            if answer:
                import json

                escaped_answer = json.dumps(answer)
                return f'{{"action":{{"tool_name":"respond","params":{{"message":{escaped_answer}}}}},"confidence":0.95,"reasoning":"direct_answer"}}'

        if not request.use_tool_prompting or not request.available_tools:
            if question_text:
                return f"""ANSWER_ONLY|{question_text}"""
            return ""

        return self._build_tool_aware_prompt(request, question_text, date_str, time_str)

    def build_budget_context(self) -> str:
        if not hasattr(self._llm, "get_cost_tracker"):
            return ""
        if not getattr(self._llm, "cloud_allowed", lambda: False)():
            return ""
        if not getattr(self._llm, "has_cost_visible_provider", lambda: False)():
            return ""

        cost_tracker = self._llm.get_cost_tracker()
        policy = self._llm.get_routing_policy()
        totals = cost_tracker.get_totals()
        reserved_ratio = cost_tracker.config.reserved_budget_ratio
        min_samples = cost_tracker.config.min_spend_samples

        remaining_hour = max(policy.max_cost_per_hour - totals["hourly"], 0.0) if policy.max_cost_per_hour > 0 else None
        remaining_day = max(policy.max_cost_per_day - totals["daily"], 0.0) if policy.max_cost_per_day > 0 else None
        remaining_month = (
            max(policy.max_cost_per_month - totals["monthly"], 0.0) if policy.max_cost_per_month > 0 else None
        )

        remaining_hour_visible = remaining_hour * (1.0 - reserved_ratio) if remaining_hour is not None else None
        remaining_day_visible = remaining_day * (1.0 - reserved_ratio) if remaining_day is not None else None
        remaining_month_visible = remaining_month * (1.0 - reserved_ratio) if remaining_month is not None else None

        tier = compute_budget_tier(policy, totals)
        is_blocked = tier == "blocked"

        rates = cost_tracker.get_spend_rates()
        spend_rate_3h = format_spend_rate(rates["ema_3h"], min_samples)
        spend_rate_24h = format_spend_rate(rates["ema_24h"], min_samples)
        spend_rate_7d = format_spend_rate(rates["ema_7d"], min_samples)

        hours_daily, daily_note = estimate_hours_until_limit(
            remaining_day_visible,
            rates["ema_24h"],
            rates["ema_3h"],
            min_samples,
        )
        hours_monthly, monthly_note = estimate_hours_until_limit(
            remaining_month_visible,
            rates["ema_7d"],
            rates["ema_24h"],
            min_samples,
            fallback_second=rates["ema_3h"],
        )

        cost_div = cost_energy_divergence(policy, totals)

        lines = [
            "=== Budget Context (USD) ===",
            f"remaining_per_request: {format_usd(policy.max_cost_per_request if policy.max_cost_per_request > 0 else None)}",
            f"remaining_hour: {format_usd(remaining_hour_visible)}",
            f"remaining_day: {format_usd(remaining_day_visible)}",
            f"remaining_month: {format_usd(remaining_month_visible)}",
            f"current_spend_hour: {format_usd(totals['hourly'])}",
            f"current_spend_day: {format_usd(totals['daily'])}",
            f"current_spend_month: {format_usd(totals['monthly'])}",
            f"active_budget_tier: {tier}",
            f"is_budget_blocked: {str(is_blocked).lower()}",
            f"reserved_budget_ratio: {reserved_ratio:.2f}",
            f"spend_rate_3h: {spend_rate_3h}",
            f"spend_rate_24h: {spend_rate_24h}",
            f"spend_rate_7d: {spend_rate_7d}",
            f"min_spend_samples: {min_samples}",
            f"hours_until_daily_limit: {hours_daily}{daily_note}",
            f"hours_until_monthly_limit: {hours_monthly}{monthly_note}",
            f"cost_energy_divergence: {cost_div}",
        ]
        if cost_div == "high":
            lines.append("Note: Energy has recharged but USD budget is nearly exhausted. Prefer local or defer.")

        return "\n".join(lines)

    def _build_tool_aware_prompt(
        self,
        request: LLMRequest,
        question_text: str,
        date_str: str,
        time_str: str,
    ) -> str:
        """Build a comprehensive tool-aware prompt for complex reasoning."""
        from maxim.utils.filesystem_policy import get_effective_cwd

        mode = request.mode
        counter = self._token_counter
        response_reserve = mode.max_response_tokens

        budgeter = PromptBudgeter(
            total_budget=self._n_ctx,
            response_reserve=response_reserve,
            token_counter=counter,
        )

        effective_cwd = get_effective_cwd()
        is_rt = is_realtime_request(question_text)

        # Add sections by category. Each helper appends to ``budgeter`` in-place
        # so this orchestrator stays a flat sequence — easy to read and reorder.
        self._add_mandatory_sections(budgeter, request, question_text)
        self._add_critical_sections(budgeter, request, question_text, date_str, time_str, effective_cwd, is_rt)
        self._add_guidance_sections(budgeter, request, date_str, time_str)
        self._add_context_sections(budgeter, request, question_text)
        self._add_perception_sections(budgeter, request)
        self._add_memory_sections(budgeter, request.context)

        prompt_text, dropped = budgeter.build()
        if dropped:
            notice = f"[Context note: omitted due to token budget: {', '.join(dropped)}]"
            prompt_text = notice + "\n\n" + prompt_text
            logger.info(
                "Prompt budget: dropped %d sections for %s (n_ctx=%d, reserve=%d)",
                len(dropped),
                mode.name,
                self._n_ctx,
                response_reserve,
            )

        return f"TOOL_PROMPT|{prompt_text}"

    # ── _build_tool_aware_prompt section helpers ──────────────────────────

    @staticmethod
    def _add_mandatory_sections(
        budgeter: PromptBudgeter,
        request: LLMRequest,
        question_text: str,
    ) -> None:
        """Instructions + the user's request itself. Cannot be dropped."""
        budgeter.add("instructions", build_instructions_section(request), SectionPriority.MANDATORY)
        if question_text:
            budgeter.add(
                "user_request",
                f'=== User Request ===\n"{question_text}"',
                SectionPriority.MANDATORY,
            )

    def _add_critical_sections(
        self,
        budgeter: PromptBudgeter,
        request: LLMRequest,
        question_text: str,
        date_str: str,
        time_str: str,
        effective_cwd: str,
        is_rt: bool,
    ) -> None:
        """Identity, tools, workspace, and the realtime hint."""
        mode = request.mode
        mode_name = mode.name
        counter = self._token_counter

        budgeter.add(
            "planning_banner",
            build_planning_banner(request.autonomy_level),
            SectionPriority.CRITICAL,
        )
        if request.pending_modification:
            budgeter.add(
                "modification",
                build_modification_section(request.pending_modification),
                SectionPriority.CRITICAL,
            )
        budgeter.add(
            "identity",
            build_identity_section(mode, request, date_str, time_str),
            SectionPriority.CRITICAL,
        )

        # Tool section: opt-in learned relevance filter, declared per mode via
        # ModeDefinition.uses_tool_relevance_filter. Modes that opt in (passive
        # interactive use) get a CRITICAL relevant-tools section plus an
        # IMPORTANT background section, both with full descriptions. Modes
        # that opt out (autonomous: active/singularity) get the full unfiltered
        # manifest as a single CRITICAL section — the filter has cold-start
        # pathology under no learned signal and produces near-random subsets
        # that cause tool hallucination.
        use_filter = self._tool_index is not None and getattr(request.mode, "uses_tool_relevance_filter", False)
        if use_filter:
            relevant, background = self._tool_index.get_relevant_tools(question_text)
            request.surfaced_tools = relevant
            relevant_section = build_tools_section_filtered(request, relevant, mode_name)
            budgeter.add(
                "tools",
                relevant_section,
                SectionPriority.CRITICAL,
                truncatable=True,
                min_tokens=50,
                truncate_fn=lambda c, m: _truncate_tool_guidance(c, m, counter),
            )
            if background:
                bg_section = build_tools_section_filtered(request, background, mode_name)
                bg_section = bg_section.replace(
                    "=== Available Tools ===",
                    "=== Additional Tools ===",
                    1,
                )
                budgeter.add(
                    "tools_background",
                    bg_section,
                    SectionPriority.IMPORTANT,
                    truncatable=True,
                    min_tokens=50,
                    truncate_fn=lambda c, m: _truncate_tool_guidance(c, m, counter),
                )
        else:
            budgeter.add(
                "tools",
                build_tools_section(request, mode_name=mode_name),
                SectionPriority.CRITICAL,
                truncatable=True,
                min_tokens=50,
                truncate_fn=lambda c, m: _truncate_tool_guidance(c, m, counter),
            )

        workspace_manifest = build_workspace_manifest(mode_name=mode_name, cwd=effective_cwd)
        if workspace_manifest:
            is_singularity = mode_name == "singularity"
            budgeter.add(
                "workspace_manifest",
                workspace_manifest,
                SectionPriority.CRITICAL,
                truncatable=is_singularity,
                min_tokens=80 if is_singularity else 0,
                truncate_fn=((lambda c, m: _truncate_manifest(c, m, counter)) if is_singularity else None),
            )

        if is_rt:
            hint = ">>> REAL-TIME DATA NEEDED <<<\n"
            if request.autonomy_level == AutonomyLevel.PLANNING:
                hint += "Propose using 'internet_search' and ask for approval."
            else:
                hint += "Use 'internet_search' tool directly."
            budgeter.add("realtime_hint", hint, SectionPriority.CRITICAL)

    def _add_guidance_sections(
        self,
        budgeter: PromptBudgeter,
        request: LLMRequest,
        date_str: str,
        time_str: str,
    ) -> None:
        """Tool guidance + datetime + cost budget. Mostly IMPORTANT priority."""
        mode_name = request.mode.name
        budgeter.add(
            "tool_guidance_core",
            build_tool_guidance_core(mode_name=mode_name),
            SectionPriority.IMPORTANT,
        )
        budgeter.add(
            "tool_guidance_extended",
            build_tool_guidance_extended(mode_name=mode_name),
            SectionPriority.NICE_TO_HAVE,
        )
        budgeter.add("datetime", build_datetime_section(date_str, time_str), SectionPriority.IMPORTANT)

        budget_context = self.build_budget_context()
        if budget_context:
            budgeter.add("budget_context", budget_context, SectionPriority.NICE_TO_HAVE)

    def _add_context_sections(
        self,
        budgeter: PromptBudgeter,
        request: LLMRequest,
        question_text: str,
    ) -> None:
        """Conversation history, context pool, protocols, carryover, prefetch, coding guidelines."""
        counter = self._token_counter

        if request.conversation_history_text:
            conv_text = _compact_conversation(request.conversation_history_text, 12)
            budgeter.add(
                "conversation",
                "=== Conversation History ===\n" + conv_text,
                SectionPriority.IMPORTANT,
                truncatable=True,
                min_tokens=50,
                truncate_fn=lambda c, m: _truncate_conversation(c, m, counter),
            )

        if request.context_pool_text:
            budgeter.add(
                "context_pool",
                "=== Context ===\n" + request.context_pool_text,
                SectionPriority.IMPORTANT,
                truncatable=True,
                min_tokens=30,
                truncate_fn=lambda c, m: _truncate_context_pool(c, m, counter),
            )

        if request.protocol_context:
            budgeter.add(
                "active_protocols",
                "=== Active Protocols ===\n" + request.protocol_context,
                SectionPriority.IMPORTANT,
            )

        carryover_text = self._reasoning_carryover.get_prompt_text()
        if carryover_text:
            budgeter.add(
                "reasoning_carryover",
                carryover_text,
                SectionPriority.IMPORTANT,
                truncatable=True,
                min_tokens=30,
                truncate_fn=lambda c, m: _truncate_reasoning_carryover(c, m, counter),
            )

        if request.prefetch_context:
            prefetch = request.prefetch_context
            if request.skip_exploration:
                prefetch += (
                    "\n\n"
                    + "!" * 50
                    + "\n!!! ONE-CALL MODE: WRITE DIRECTLY !!!\n"
                    + "!" * 50
                    + "\n\nFile discovery is COMPLETE. Do NOT use glob or read_file."
                    + "\nProceed DIRECTLY to your action:"
                    + "\n- For EXISTING file: write_file with overwrite=True"
                    + "\n- For NEW file: write_file (no overwrite needed)"
                    + "\n\nYour response should be the write_file action, NOT exploration."
                )
            has_discovery = "DISCOVERY PLAN" in prefetch or "FILE DISCOVERY" in prefetch
            priority = SectionPriority.CRITICAL if has_discovery else SectionPriority.IMPORTANT
            budgeter.add(
                "prefetch_context",
                prefetch,
                priority,
                truncatable=True,
                min_tokens=50,
                truncate_fn=lambda c, m: _truncate_context_pool(c, m, counter),
            )

        # Coding guidelines: prefer the user's question text, fall back to pending_modification
        guidelines_text = question_text
        if not guidelines_text and request.pending_modification:
            guidelines_text = request.pending_modification.get("user_modification", "")
        if guidelines_text:
            coding_context = build_coding_context(guidelines_text, include_sandbox_reminder=True, max_guidelines=2)
            if coding_context:
                budgeter.add("coding_guidelines", coding_context, SectionPriority.IMPORTANT)

        budgeter.add("foundational", _load_foundational_context(), SectionPriority.IMPORTANT)
        if request.mode.context_prompt:
            budgeter.add("mode_context", request.mode.context_prompt, SectionPriority.NICE_TO_HAVE)

    @staticmethod
    def _add_perception_sections(
        budgeter: PromptBudgeter,
        request: LLMRequest,
    ) -> None:
        """Current observation, recent speech, agent states, recent outcomes, body state."""
        context = request.context

        if context.current_percept:
            obs_text = build_observation_section(context.current_percept)
            if obs_text:
                budgeter.add("observation", obs_text, SectionPriority.NICE_TO_HAVE)

        if context.detected_speech:
            speech_lines = ["=== Recent Speech ==="]
            for speech in context.detected_speech[-3:]:
                speech_lines.append(f'- "{speech[:100]}"')
            budgeter.add("speech", "\n".join(speech_lines), SectionPriority.NICE_TO_HAVE)

        if request.agent_states:
            state_lines = ["=== Agent States ==="]
            for state in request.agent_states[-5:]:
                agent_name = state.get("agent", "unknown")
                agent_state = state.get("state", "unknown")
                goal = state.get("goal", "")
                state_lines.append(f"- {agent_name}: {agent_state}" + (f" (goal: {goal})" if goal else ""))
            budgeter.add("agent_states", "\n".join(state_lines), SectionPriority.NICE_TO_HAVE)

        if request.recent_outcomes:
            outcome_lines = ["=== Recent Action Outcomes ==="]
            for outcome in request.recent_outcomes[-3:]:
                tool = outcome.get("tool", "?")
                success = "succeeded" if outcome.get("success") else "failed"
                result = outcome.get("result", "")[:50] if outcome.get("result") else ""
                outcome_lines.append(f"- {tool}: {success}" + (f" ({result})" if result else ""))
            budgeter.add("recent_outcomes", "\n".join(outcome_lines), SectionPriority.IMPORTANT)

        # Body state (interoception — always present when embodied)
        if context.body_state:
            budgeter.add("body_state", context.body_state, SectionPriority.CRITICAL)

    @staticmethod
    def _add_memory_sections(
        budgeter: PromptBudgeter,
        context: Any,
    ) -> None:
        """Episodic recalls, ATL concepts, semantic knowledge, causal predictions, motor programs, statistics."""
        if context.relevant_memories:
            mem_lines = ["=== Relevant Memories ==="]
            for mem in context.relevant_memories[:8]:
                content = mem.get("content", {})
                source = mem.get("source", "episodic")
                salience = mem.get("salience", 0.0)
                parts = []
                if content.get("goal"):
                    parts.append(f"goal: {content['goal']}")
                if content.get("action"):
                    parts.append(f"action: {content['action']}")
                if content.get("success") is not None:
                    parts.append("succeeded" if content["success"] else "failed")
                if content.get("transcript"):
                    parts.append(f'"{content["transcript"][:80]}"')
                summary = ", ".join(parts) if parts else str(content)[:80]
                mem_lines.append(f"- [{source}, salience={salience:.2f}] {summary}")
                if mem.get("predictions"):
                    for pred in mem["predictions"][:2]:
                        pred_tool = pred.get("tool", "?")
                        pred_success = pred.get("success", "?")
                        pred_conf = pred.get("confidence", 0)
                        mem_lines.append(
                            f"  prediction: {pred_tool} (success={pred_success}, confidence={pred_conf:.2f})"
                        )
            budgeter.add(
                "relevant_memories",
                "\n".join(mem_lines),
                SectionPriority.IMPORTANT,
                truncatable=True,
                min_tokens=50,
                truncate_fn=lambda c, m: "\n".join(c.split("\n")[: max(2, m // 20)]),
            )

        if context.concept_context:
            concept_lines = ["=== Active Concepts ==="]
            for concept in context.concept_context[:5]:
                name = concept.get("name", "?")
                category = concept.get("category", "?")
                confidence = concept.get("confidence", 0)
                episodes = concept.get("episode_count", 0)
                concept_lines.append(f"- {name} ({category}, confidence={confidence:.2f}, episodes={episodes})")
                for rel in concept.get("relationships", [])[:2]:
                    concept_lines.append(f"  → {rel.get('type', '?')} {rel.get('target', '?')}")
            budgeter.add(
                "concept_context",
                "\n".join(concept_lines),
                SectionPriority.IMPORTANT,
                truncatable=True,
                min_tokens=30,
                truncate_fn=lambda c, m: "\n".join(c.split("\n")[: max(2, m // 15)]),
            )

        if context.knowledge_context:
            knowledge_lines = ["=== Semantic Knowledge ==="]
            for entry in context.knowledge_context[:5]:
                name = entry.get("concept_name", "?")
                definition = entry.get("definition", "")
                layer = entry.get("source_layer", "?")
                conf = entry.get("confidence", 0)
                knowledge_lines.append(f"- {name} ({layer}, confidence={conf:.2f}): {definition}")
                for rel in entry.get("relationships", [])[:2]:
                    knowledge_lines.append(f"  → {rel.get('type', '?')} {rel.get('target', '?')}")
            budgeter.add(
                "knowledge_context",
                "\n".join(knowledge_lines),
                SectionPriority.IMPORTANT,
                truncatable=True,
                min_tokens=30,
                truncate_fn=lambda c, m: "\n".join(c.split("\n")[: max(2, m // 15)]),
            )

        if context.causal_context:
            causal_lines = ["=== Causal Predictions (learned from experience) ==="]
            for pred in context.causal_context[:5]:
                valence = pred.get("valence", "neutral")
                conf = pred.get("confidence", 0)
                ctx_match = pred.get("context_match", 0)
                causal_lines.append(
                    f"- {pred.get('event', '?')} → {pred.get('outcome', '?')} "
                    f"(valence={valence}, confidence={conf:.2f}, context_match={ctx_match:.2f})"
                )
            budgeter.add(
                "causal_context",
                "\n".join(causal_lines),
                SectionPriority.CRITICAL,
                truncatable=True,
                min_tokens=30,
                truncate_fn=lambda c, m: "\n".join(c.split("\n")[: max(2, m // 15)]),
            )

        if context.valence_context:
            valence_lines = ["=== Learned Associations (from experience) ==="]
            for entry in context.valence_context[:5]:
                concept = entry.get("concept", "?")
                valence = entry.get("valence", 0.0)
                if valence < -0.1:
                    sentiment = "negative (associated with pain/failure)"
                elif valence > 0.1:
                    sentiment = "positive (associated with success/benefit)"
                else:
                    sentiment = "neutral"
                assoc = ", ".join(entry.get("associations", [])[:3])
                valence_lines.append(f"- {concept}: {sentiment} (valence={valence:+.2f}, related: {assoc})")
            budgeter.add(
                "valence_context",
                "\n".join(valence_lines),
                SectionPriority.CRITICAL,
                truncatable=True,
                min_tokens=30,
                truncate_fn=lambda c, m: "\n".join(c.split("\n")[: max(2, m // 15)]),
            )

        if context.motor_programs:
            motor_lines = ["=== Available Motor Programs ==="]
            for prog in context.motor_programs[:8]:
                name = prog.get("name", "?")
                conf = prog.get("confidence", 0)
                rate = prog.get("success_rate", 0)
                execs = prog.get("executions", 0)
                steps_str = " → ".join(prog.get("steps", [])[:5])
                motor_lines.append(f"- {name} (confidence={conf:.2f}, {execs} runs, success={rate:.0%})")
                if steps_str:
                    motor_lines.append(f"  Steps: {steps_str}")
                risks = prog.get("risks", [])
                if risks:
                    motor_lines.append(f"  Known risks: {', '.join(risks)}")
            budgeter.add(
                "motor_programs",
                "\n".join(motor_lines),
                SectionPriority.IMPORTANT,
                truncatable=True,
                min_tokens=30,
                truncate_fn=lambda c, m: "\n".join(c.split("\n")[: max(2, m // 15)]),
            )

        if context.statistical_context and context.active_pattern_count > 0:
            stat_lines = [
                f"=== Statistical Patterns ({context.active_pattern_count} active) ===",
                context.statistical_context,
            ]
            suggestions = getattr(context, "statistical_suggestions", [])
            if suggestions:
                stat_lines.append("")
                stat_lines.append("Recommended analyses (ranked by priority):")
                for i, s in enumerate(suggestions[:3], 1):
                    stat_lines.append(
                        f"  {i}. {s.get('tool_call', 'math')} {s.get('operation', '?')} "
                        f"on {s.get('metric', '?')} [{s.get('data_type', '?')}] — "
                        f"{s.get('rationale', '')}"
                    )
            else:
                stat_lines.append(
                    "Use 'math' tool to investigate patterns (assess_randomness, analyze, recall_memory)."
                )
            stat_lines.append("Use 'internet_search' to research unfamiliar patterns or analysis techniques.")
            budgeter.add("statistical_patterns", "\n".join(stat_lines), SectionPriority.NICE_TO_HAVE)

    def _build_followup_prompt(self, followup_input: str) -> str:
        """Build a prompt to handle action followups based on followup_type."""
        new_format = re.match(
            r"\[ACTION_FOLLOWUP type=(\w+) tool=([\w_-]+) mode=([\w_-]+) query='(.*)'\]: (.*)",
            followup_input,
            re.DOTALL,
        )

        if new_format:
            followup_type = new_format.group(1)
            tool_name = new_format.group(2)
            mode_name = new_format.group(3)
            raw_query = new_format.group(4)
            if "']: " in followup_input:
                split_idx = followup_input.find("']: ")
                bracket_start = followup_input.find("query='") + 7
                original_query = followup_input[bracket_start:split_idx]
                result = followup_input[split_idx + 4 :]
            else:
                original_query = raw_query
                result = new_format.group(5)
            logger.debug(
                f"Parsed followup: type={followup_type}, tool={tool_name}, query_len={len(original_query)}, result_len={len(result)}"
            )
        else:
            legacy_format = re.match(r"\[SEARCH RESULT for '([^']+)'\]: (.*)", followup_input, re.DOTALL)
            if legacy_format:
                followup_type = "engage"
                tool_name = "internet_search"
                mode_name = "live"
                original_query = legacy_format.group(1)
                result = legacy_format.group(2)
            else:
                logger.warning(
                    f"Could not parse followup input format, using fallback. Input starts with: {followup_input[:100]}"
                )
                followup_type = "engage"
                tool_name = "internet_search"
                mode_name = "live"
                original_query = ""
                result = followup_input

        if followup_type == "process":
            return self._build_process_prompt(tool_name, original_query, result)
        elif followup_type == "respond":
            return self._build_respond_prompt(tool_name, original_query, result)
        elif followup_type == "engage":
            return self._build_engage_prompt(tool_name, original_query, result, mode_name)
        else:
            return self._build_respond_prompt(tool_name, original_query, result)

    def _build_process_prompt(self, tool_name: str, query: str, result: str) -> str:
        is_batched = tool_name == "batched_exploration" or "BATCHED EXPLORATION RESULTS" in result

        if is_batched:
            prompt = f"""You completed batched exploration. Now analyze ALL results and make your final action.

Original request: "{query}"

{result}

=== Instructions ===
You now have complete context from the batched exploration. Based on ALL the results above:

1. If modifying a file: Use 'write_file' with the COMPLETE updated content
   - Include overwrite=True for existing files
   - Incorporate patterns/conventions from related files you read

2. If creating a new file: Use 'write_file' with appropriate content

3. If task is already complete or you need to inform the user: Use 'respond'

4. If you need more information: Request additional tools (but batching should have provided enough)

Respond with JSON:
{{"action": {{"tool_name": "<tool>", "params": {{...}}}}, "confidence": 0.9, "reasoning": "your_reasoning"}}"""
        else:
            prompt = f"""You just executed '{tool_name}'. Analyze the results and decide your next action.

Original request: "{query}"

Results from {tool_name}:
{result}

=== Instructions ===
Based on these results, determine your next action:
- If the task is complete, use 'respond' to inform the user
- If more steps are needed, choose the appropriate tool
- Extract relevant information to inform your decision

Respond with JSON:
{{"action": {{"tool_name": "<next_tool>", "params": {{...}}}}, "confidence": 0.9, "reasoning": "your_reasoning"}}"""

        return f"TOOL_PROMPT|{prompt}"

    def _build_respond_prompt(self, tool_name: str, query: str, result: str) -> str:
        prompt = f"""You completed an internet search. Synthesize the results into a helpful response.

Original user request: "{query}"

Search Results:
{result}

=== Instructions ===
Extract the most relevant information from the search results and answer the user's question clearly.

IMPORTANT: You MUST use the "respond" tool. Do NOT use any other tool name.

Example JSON format:
{{"action": {{"tool_name": "respond", "params": {{"message": "YOUR ANSWER HERE"}}}}, "confidence": 0.9, "reasoning": "synthesized_search_results"}}

Your response (use tool_name "respond"):"""

        return f"TOOL_PROMPT|{prompt}"

    def _build_engage_prompt(self, tool_name: str, query: str, result: str, mode_name: str) -> str:
        now = datetime.now()
        date_str = now.strftime("%A, %B %d, %Y")
        time_str = now.strftime("%I:%M %p")

        if mode_name == "active-assistance":
            engagement_instruction = (
                "Be proactive and suggest 2-3 relevant follow-up options the user might find helpful."
            )
        elif mode_name == "observe":
            engagement_instruction = (
                "Keep your response informative but concise. Only offer one follow-up if highly relevant."
            )
        else:
            engagement_instruction = "Optionally offer 1-2 relevant follow-up questions or related information."

        prompt = f"""You just completed an internet search. Synthesize the results into a helpful response for the user.

CURRENT DATE: {date_str} at {time_str}
(Use this date context when interpreting results about "today" or recent events)

Original user request: "{query}"

Search Results:
{result}

=== Instructions ===
1. Extract the relevant information from the search results above
2. Provide a clear, direct answer to the user's question
3. {engagement_instruction}
4. If results seem outdated or incomplete, acknowledge this

IMPORTANT: You MUST use the "respond" tool to reply to the user. Do NOT use any other tool name.

Example JSON format (COPY THIS STRUCTURE EXACTLY):
{{"action": {{"tool_name": "respond", "params": {{"message": "The Broncos beat the Bills 31-7 in the AFC Divisional playoff. Would you like details on the top performers?"}}}}, "confidence": 0.9, "reasoning": "synthesized_search_results"}}

Your response (use tool_name "respond"):"""

        return f"TOOL_PROMPT|{prompt}"

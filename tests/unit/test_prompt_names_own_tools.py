"""#1042 PR B: an agent's prompt names only tools it has.

O19 Exp 10 attempt 1 (#1041) aborted because the narrator was shown other tools' names and proposed them: the full
prompt's hard-coded tool guidance (``internet_search`` three times, ``write_file``/``read_file``/``speak``/``respond``),
a "REAL-TIME DATA" hint fired by "Call send_message NOW", and a first follow-up attempt that never listed its own
tools. Owner decisions 2026-10-01/02: every agent's prompt is gated on its own roster against the REAL tool universe
(unchanged text for an agent that has the tools); a follow-up lists the agent's own tools on its first attempt; the
narrator's decoy 'respond' stays registered but is not advertised; and a guard builds the narrator's real prompt from
the orchestrator's actual registration.

The first five tests are the red gates (b12227cf). The real-time one stays a strict xfail: its premise was falsified.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import itertools
import pathlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from maxim.agents.autonomy import AutonomyLevel
from maxim.agents.bus import StructuredContext
from maxim.agents.llm_types import LLMRequest, ModeInfo
from maxim.agents.prompt_builder import (
    PromptBuilder,
    build_tool_guidance_core,
    build_tool_guidance_extended,
    is_realtime_request,
)

NARRATOR_TOOLS = {"send_message", "observe_actions", "check_completion"}
FOREIGN = ("internet_search", "write_file", "read_file", "speak", "respond", "math", "focus_interests", "track_target")
# The stall nudge the narrator receives (orchestrator.py's stall detector, wording as shipped by this PR).
STALL = (
    "SYSTEM: Stall detected (30s idle, 1 AUT actions so far). For context, the agent under test's last action was "
    "'sense_tools' (blocked=False): ITS tool, not yours. Call send_message NOW with your next probe."
)
FOLLOWUP = (
    "[ACTION_FOLLOWUP type=process tool=send_message mode=live query='talk to the agent']: "
    "{'actions': [{'tool': 'sense_tools', 'output': 'Your capabilities: look, listen'}]}"
)


def _builder() -> PromptBuilder:
    from maxim.agents.llm_fallback import ReasoningCarryover
    from maxim.models.language.token_counter import CharEstimateCounter

    return PromptBuilder(
        llm=None,
        reasoning_carryover=ReasoningCarryover(),
        n_ctx=32000,
        token_counter=CharEstimateCounter(),
        tool_index=None,
    )


def _request(
    tools: set[str],
    *,
    triggering_input: str = "",
    cli_inputs: list[str] | None = None,
    mode: ModeInfo | None = None,
    autonomy: AutonomyLevel = AutonomyLevel.AUTONOMOUS,
    context: StructuredContext | None = None,
) -> LLMRequest:
    ctx = context or StructuredContext(timestamp=1_700_000_000.0)
    ctx.cli_inputs = list(cli_inputs or [])
    return LLMRequest(
        request_id="r1",
        context=ctx,
        mode=mode or ModeInfo(name="live", goal="narrate", context_prompt=""),
        autonomy_level=autonomy,
        internet_access=False,
        internet_policy_summary="",
        available_tools=set(tools),
        triggering_input=triggering_input,
        use_tool_prompting=True,  # the narrator's planning calls use the full tool-aware prompt
    )


def _named(prompt: str, tool: str) -> bool:
    return any(f"{q}{tool}{q}" in prompt for q in ("'", '"')) or f"- {tool}:" in prompt


# ── The red gates (b12227cf) ──────────────────────────────────────────────────


def test_the_narrators_full_prompt_names_no_tool_it_lacks() -> None:
    prompt = _builder().build_prompt(_request(NARRATOR_TOOLS, triggering_input=STALL))
    leaked = [t for t in FOREIGN if _named(prompt, t)]
    assert leaked == [], leaked


@pytest.mark.xfail(
    strict=True,
    reason="premise falsified: NOW is a whole word; the leak is closed by the roster-gated hint (full-prompt test)",
)
def test_a_stall_nudge_is_not_a_real_time_request() -> None:
    assert is_realtime_request(STALL) is False
    assert is_realtime_request("what's the Broncos score now?") is True  # a real one still is


def test_a_first_follow_up_lists_the_agents_own_tools() -> None:
    prompt = _builder().build_prompt(_request(NARRATOR_TOOLS, cli_inputs=[FOLLOWUP]))
    assert all(f"'{t}'" in prompt for t in NARRATOR_TOOLS), prompt[-400:]
    assert not _named(prompt, "respond"), "the narrator has no 'respond'"


def test_an_agent_with_every_tool_sees_unchanged_guidance() -> None:
    every = set(FOREIGN) | {"send_message"}
    for mode in ("passive", "active", "singularity"):
        assert build_tool_guidance_core(mode_name=mode, tools=every) == build_tool_guidance_core(mode_name=mode)
        assert build_tool_guidance_extended(mode_name=mode, tools=every) == build_tool_guidance_extended(mode_name=mode)


def test_the_narrators_stall_and_diversity_inputs_label_the_agents_tools() -> None:
    from maxim.simulation import orchestrator

    src = inspect.getsource(orchestrator)
    assert "the agent under test's last action was" in src
    assert "the agent under test's tools so far (not yours)" in src


# ── The real tool universe and the narrator's real roster ─────────────────────


def _tool_names_in_source() -> set[str]:
    """Every ``name = "..."`` on a class whose base ends in ``Tool``, across src/maxim."""
    import maxim

    names: set[str] = set()
    for path in pathlib.Path(maxim.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            if not any(ast.unparse(b).split(".")[-1].endswith("Tool") for b in node.bases):
                continue
            for stmt in node.body:
                target = stmt.targets[0] if isinstance(stmt, ast.Assign) else getattr(stmt, "target", None)
                value = getattr(stmt, "value", None)
                if isinstance(target, ast.Name) and target.id == "name" and isinstance(value, ast.Constant):
                    names.add(value.value)
    return names


def test_the_registered_tool_names_are_every_tool_in_the_source() -> None:
    """The gate is only as good as its universe: a tool added without its name in REGISTERED_TOOL_NAMES fails here."""
    from maxim.modes.definitions import REGISTERED_TOOL_NAMES

    assert _tool_names_in_source() == set(REGISTERED_TOOL_NAMES)


def _narrator_registration() -> list[type]:
    """The tool classes orchestrator.py registers on ``orch_registry``, read from its source (every branch)."""
    from maxim.simulation import orchestrator

    tree = ast.parse(inspect.getsource(orchestrator))
    imported = {
        alias.asname or alias.name: (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
    }
    built = {
        target.id: node.value.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    classes = []
    for node in ast.walk(tree):
        func = getattr(node, "func", None)
        if not (isinstance(node, ast.Call) and isinstance(func, ast.Attribute) and func.attr == "register"):
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == "orch_registry"):
            continue
        arg = node.args[0]
        class_name = arg.func.id if isinstance(arg, ast.Call) else built[arg.id]
        module, attr = imported[class_name]
        classes.append(getattr(importlib.import_module(module), attr))
    return classes


def _narrator_registry():  # type: ignore[no-untyped-def]
    from maxim.simulation.tools import SimToolRegistry

    registry = SimToolRegistry()
    for cls in _narrator_registration():
        registry._tools[cls.name] = cls.__new__(cls)  # names + class attributes only; no constructor wiring
    return registry


def _narrator_roster() -> set[str]:
    from maxim.runtime.loop_controller import LoopController

    return LoopController.get_all_tools(SimpleNamespace(executor=SimpleNamespace(registry=_narrator_registry())))


def test_the_narrators_decoy_respond_is_registered_but_not_advertised() -> None:
    from maxim.simulation.tools import SimRespondTool, SimToolRegistry

    assert SimRespondTool in _narrator_registration()
    roster = _narrator_roster()
    assert "respond" not in roster and {"send_message", "observe_actions", "check_completion"} <= roster
    registry = SimToolRegistry()
    registry.register(SimRespondTool())
    assert isinstance(registry.get("respond"), SimRespondTool), "a stray call still dispatches to the decoy"


def test_a_tool_registry_advertises_all_but_its_decoys() -> None:
    from maxim.simulation.tools import SimRespondTool
    from maxim.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(SimRespondTool())
    assert registry.list() == ["respond"] and registry.advertised() == []


_FOLLOWUPS = (
    FOLLOWUP,
    FOLLOWUP.replace("type=process", "type=respond"),
    FOLLOWUP.replace("type=process", "type=engage"),
    FOLLOWUP.replace("tool=send_message", "tool=batched_exploration"),
    "[SEARCH RESULT for 'cats']: cats are mammals",
    "an unparseable follow-up",
)


def _sim_session(monkeypatch: pytest.MonkeyPatch, *, interactive: bool) -> None:
    from maxim.simulation import sim_logger

    monkeypatch.setattr(sim_logger, "_sim_active", True)
    monkeypatch.setattr(
        sim_logger,
        "_interactive_mode",
        sim_logger.InteractiveMode.ON if interactive else sim_logger.InteractiveMode.OFF,
    )


def _mentions(text: str, roster: set[str]) -> list[str]:
    """Known tools ``text`` mentions outside ``roster``, by a detector INDEPENDENT of the gate's: any word-bounded
    occurrence of an underscored name, and for a plain word ('respond', 'say') a quote, a call, a list item, an arrow
    or a "use / call X" in any case."""
    import re

    from maxim.modes.definitions import PROMPT_TOOL_NAMES

    found = []
    for name in sorted(PROMPT_TOOL_NAMES - set(roster)):
        n = re.escape(name)
        if "_" in name:
            pattern = rf"\b{n}\b"
        else:
            pattern = rf"['\"`]{n}['\"`]|\b{n}\(|^\s*- {n}\b|→ {n}\b|\b(?:use|call|using|calling) {n}\b"
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            found.append(name)
    return found


# Sections whose text is DATA the agent reads (the user's request, its own tool roster and descriptions, a
# correction naming a tool it called): not the builder's text, so the guard does not read them.
_DATA_SECTIONS = {"user_request", "tools", "tools_background", "failed_tools"}
_ROSTERS = (
    {"respond"},
    {"read_file"},
    {"bash", "read_file"},
    {"say", "think"},
    {"send_message"},
    {"sense_tools"},
)


def _populated_cwd(tmp_path: pathlib.Path) -> pathlib.Path:
    for name in ("alpha.py", "beta.md", "gamma.txt", "delta.cfg", "epsilon.json", "zeta.py"):
        (tmp_path / name).write_text("x")
    workspace = tmp_path / ".maxim_workspace"
    workspace.mkdir()
    for name in ("one.py", "two.py", "three.py"):
        (workspace / name).write_text("x")
    return tmp_path


@pytest.mark.parametrize("interactive", [False, True])
def test_no_prompt_names_a_known_tool_outside_the_agents_roster(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, interactive: bool
) -> None:
    """Every operational mode x the narrator's REAL roster (with its real tool descriptions) and several partial
    rosters x every autonomy level, with a populated working directory, real-time, statistics, prefetch and every
    follow-up shape. Data is tool-free (or set aside, ``_DATA_SECTIONS``), so any known tool outside the roster
    came from the builder's own text."""
    from maxim.agents.prompt_budgeter import PromptBudgeter
    from maxim.modes.definitions import OPERATIONAL_MODES, TOOL_DESCRIPTIONS, get_mode
    from maxim.runtime.agent_loop import _describe_tools_for_prompt
    from maxim.utils import filesystem_policy

    _sim_session(monkeypatch, interactive=interactive)
    cwd = str(_populated_cwd(tmp_path))
    monkeypatch.setattr(filesystem_policy, "get_effective_cwd", lambda: cwd)
    data: list[str] = []
    real_add = PromptBudgeter.add

    def recording_add(self, name, content, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name in _DATA_SECTIONS and content:
            data.append(content)
        return real_add(self, name, content, *args, **kwargs)

    monkeypatch.setattr(PromptBudgeter, "add", recording_add)
    narrator, narrator_registry = _narrator_roster(), _narrator_registry()
    stats = dict(
        statistical_context="two patterns",
        active_pattern_count=2,
        statistical_suggestions=[{"tool_call": "math", "operation": "analyze", "metric": "m"}],
    )
    leaks: dict[str, list[str]] = {}
    for mode_name, roster in itertools.product(OPERATIONAL_MODES, [narrator, *_ROSTERS]):
        definition = get_mode(mode_name)
        mode = ModeInfo(name=mode_name, goal=definition.goal, context_prompt=definition.context_prompt)
        executor = SimpleNamespace(registry=narrator_registry)
        descriptions = _describe_tools_for_prompt(roster, executor, TOOL_DESCRIPTIONS)
        cases = [(autonomy, "what is the score now?", stats, []) for autonomy in AutonomyLevel]
        cases += [(AutonomyLevel.AUTONOMOUS, "", {}, [])]
        cases += [(AutonomyLevel.AUTONOMOUS, "", {}, [f]) for f in _FOLLOWUPS]
        for autonomy, trigger, extra, cli in cases:
            data.clear()
            context = replace(StructuredContext(timestamp=1_700_000_000.0), **extra)
            request = _request(roster, triggering_input=trigger, cli_inputs=cli, mode=mode, autonomy=autonomy)
            request = replace(
                request,
                context=context,
                tool_descriptions=descriptions,
                prefetch_context="prefetched notes" if trigger else "",
                skip_exploration=bool(trigger),
            )
            request.context.cli_inputs = list(cli)
            prompt = _builder().build_prompt(request)
            # The follow-up's result and the tool it reports on are data too.
            for text in [
                *data,
                *(f.split("]: ", 1)[-1] for f in _FOLLOWUPS),
                "You just executed 'send_message'",
                "Results from send_message:",
            ]:
                prompt = prompt.replace(text, "")
            if bad := _mentions(prompt, roster):
                leaks[f"{mode_name} {sorted(roster)[:3]} {autonomy.name} rt={bool(trigger)} cli={cli[:1]}"] = bad
    assert leaks == {}, leaks


# ── The pieces ───────────────────────────────────────────────────────────────


def test_real_time_keywords_start_a_word() -> None:
    """The leading boundary only: "now" inside "know" / "snow" is not real-time; a plural ("scores") still is."""
    assert is_realtime_request("I know the plan, let us snowball it") is False
    assert is_realtime_request("what's the Broncos score now?") is True
    assert is_realtime_request("show me the scores please") is True


def test_the_planning_banner_examples_name_only_the_agents_tools() -> None:
    from maxim.agents.prompt_builder import build_planning_banner

    banner = build_planning_banner(AutonomyLevel.PLANNING, tools=NARRATOR_TOOLS)
    assert banner and not _named(banner, "internet_search")
    assert build_planning_banner(AutonomyLevel.PLANNING, tools={"internet_search"}) == build_planning_banner(
        AutonomyLevel.PLANNING
    )


def test_the_statistics_section_names_only_the_agents_tools() -> None:
    from maxim.agents.prompt_budgeter import PromptBudgeter
    from maxim.models.language.token_counter import CharEstimateCounter

    ctx = replace(
        StructuredContext(timestamp=1_700_000_000.0),
        statistical_context="two patterns",
        active_pattern_count=2,
        statistical_suggestions=[{"tool_call": "math", "operation": "analyze", "metric": "m"}],
    )

    def section(tools: set[str]) -> str:
        b = PromptBudgeter(
            total_budget=4096,
            response_reserve=512,
            token_counter=CharEstimateCounter(),
            template_overhead=100,
            builder_gate=None,
        )
        PromptBuilder._add_memory_sections(b, ctx, tools=tools)
        return next(s.content for s in b._sections if s.name == "statistical_patterns")

    narrator = section(NARRATOR_TOOLS)
    assert not _named(narrator, "math") and not _named(narrator, "internet_search") and "math analyze" not in narrator
    full = section({"math", "internet_search"})
    assert "math analyze" in full and _named(full, "internet_search")


def test_the_roster_gate_drops_foreign_continuations_and_empty_headers() -> None:
    from maxim.agents.prompt_builder import gate_on_roster

    text = "\n".join(
        [
            "=== Tool Parameters ===",
            "- math: compute",
            "  continuation of the math line",
            "- send_message: talk",
            "  then 'read_file' the reply",
            "  and wait for it",
            "",
            "=== Tool Selection ===",
            "- MATH: Use 'math'",
            "  multi-step detail",
        ]
    )
    out = gate_on_roster(text, {"send_message"})
    assert out == "=== Tool Parameters ===\n- send_message: talk\n  and wait for it", out
    assert gate_on_roster(text, None) == text


def test_a_section_only_about_tools_the_agent_lacks_goes_whole() -> None:
    """Its tool-free lines too: "You can read and write ANY file" means nothing to an agent with no file tools."""
    core = build_tool_guidance_core(mode_name="singularity", tools=NARRATOR_TOOLS)
    assert "File Operation Rules" not in core and "read and write ANY file" not in core
    assert "Planning Rule" not in core and "Wait for user confirmation" not in core
    extended = build_tool_guidance_extended(mode_name="singularity", tools=NARRATOR_TOOLS)
    assert "FILE WORKSPACE" not in extended and "Workspace (.maxim_workspace/)" not in extended


def test_a_plain_word_is_a_tool_only_when_quoted_or_called() -> None:
    from maxim.agents.prompt_builder import names_tool_outside

    assert names_tool_outside("respond in JSON; think it through", set()) == []
    assert names_tool_outside("use 'respond', then think(x)", set()) == ["respond", "think"]


def test_the_pfc_preamble_names_only_the_agents_tools() -> None:
    from maxim.agents.exec_prompts import PFC_PREAMBLE
    from maxim.agents.prompt_builder import build_pfc_preamble, names_tool_outside

    assert build_pfc_preamble(None) == PFC_PREAMBLE
    every = {"sense", "sense_tools", "sense_presence", "think", "request_interaction"}
    assert build_pfc_preamble(every) == PFC_PREAMBLE
    narrator = build_pfc_preamble(NARRATOR_TOOLS)
    assert names_tool_outside(narrator, NARRATOR_TOOLS) == []
    assert "discover your world" not in narrator and "slash" not in narrator, "no discovery lessons without sense_tools"
    assert "action: one of your tools" in narrator and "NOTICE → WONDER → DECIDE → ACT" in narrator


def test_the_interactive_identity_names_only_the_agents_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.agents.prompt_builder import build_identity_section, names_tool_outside

    _sim_session(monkeypatch, interactive=True)
    narrator = build_identity_section(
        ModeInfo(name="live", goal="g", context_prompt=""), _request(NARRATOR_TOOLS), "d", "t"
    )
    assert "INTERACTIVE MODE" in narrator and names_tool_outside(narrator, NARRATOR_TOOLS) == []
    full = build_identity_section(
        ModeInfo(name="live", goal="g", context_prompt=""), _request({"request_interaction", "set_scene"}), "d", "t"
    )
    assert "Only use request_interaction when" in full and "Use set_scene to describe" in full


def test_an_unparseable_follow_up_is_not_called_a_search_for_an_agent_without_one() -> None:
    prompt = _builder()._build_followup_prompt("an unparseable follow-up", tools=NARRATOR_TOOLS)
    assert "internet_search" not in prompt and "internet search" not in prompt
    assert "internet search" in _builder()._build_followup_prompt("an unparseable follow-up")


def test_a_budgeter_section_must_say_who_wrote_it() -> None:
    """The seam (#1042): a section that skips the decision is a TypeError, and only builder text is gated."""
    from maxim.agents.prompt_budgeter import PromptBudgeter, SectionPriority
    from maxim.agents.prompt_builder import gate_on_roster
    from maxim.models.language.token_counter import CharEstimateCounter

    b = PromptBudgeter(
        total_budget=4096,
        response_reserve=512,
        token_counter=CharEstimateCounter(),
        builder_gate=lambda text: gate_on_roster(text, {"send_message"}),
    )
    with pytest.raises(TypeError):
        b.add("x", "text", SectionPriority.IMPORTANT)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        b.add("x", "text", SectionPriority.IMPORTANT, source="other")  # type: ignore[arg-type]
    b.add("guide", "Use 'math' to compute.\nUse 'send_message' to talk.", SectionPriority.IMPORTANT, source="builder")
    b.add("result", "the AUT called 'math'", SectionPriority.IMPORTANT, source="data")
    assert [s.content for s in b._sections] == ["Use 'send_message' to talk.", "the AUT called 'math'"]


def test_the_inspect_aut_queries_are_labelled_as_the_agent_under_tests() -> None:
    from maxim.simulation.tools import InspectAUTTool

    assert "AGENT UNDER TEST's" in InspectAUTTool.description and "not yours" in InspectAUTTool.description


def test_the_discovery_lessons_need_the_whole_discovery_set() -> None:
    from maxim.agents.prompt_builder import build_pfc_preamble

    assert "discover your world" not in build_pfc_preamble({"sense_tools"})
    assert "discover your world" in build_pfc_preamble({"sense", "sense_tools", "sense_presence"})


def test_the_embodied_guidance_points_at_sense_tools_only_for_an_agent_with_it() -> None:
    assert "'sense_tools'" not in build_tool_guidance_core(is_embodied=True, tools={"body_use"})
    assert "'sense_tools'" in build_tool_guidance_core(is_embodied=True, tools={"sense_tools"})
    assert build_tool_guidance_core(is_embodied=True) == build_tool_guidance_core(
        is_embodied=True, tools={"sense_tools"}
    )


def test_an_unregistered_tool_reply_names_only_advertised_tools_the_agent_has() -> None:
    from maxim.runtime.executor import Executor
    from maxim.simulation.tools import CheckCompletionTool, SimRespondTool
    from maxim.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(SimRespondTool())
    registry.register(CheckCompletionTool.__new__(CheckCompletionTool))
    executor = Executor(tool_registry=registry)
    first = executor.execute({"tool_name": "respondd", "params": {}}).error or ""
    assert "Did you mean" not in first and "'memory_recall'" not in first, "no decoy suggestion, no foreign hint"
    second = executor.execute({"tool_name": "respondd", "params": {}}).error or ""
    assert "Available tools: check_completion." in second, second


def test_a_registry_without_advertised_fails_loudly() -> None:
    from maxim.runtime.loop_controller import LoopController

    class _ListOnly:
        def list(self) -> list[str]:
            return ["respond"]

    with pytest.raises(AttributeError):
        LoopController.get_all_tools(SimpleNamespace(executor=SimpleNamespace(registry=_ListOnly())))


def test_no_real_time_hint_for_an_agent_without_a_search() -> None:
    """The whole hint, not just its tool line: a bare "REAL-TIME DATA NEEDED" header with nothing to use is noise."""
    prompt = _builder().build_prompt(_request(NARRATOR_TOOLS, triggering_input="what is the score now?"))
    assert "REAL-TIME DATA NEEDED" not in prompt
    full = _builder().build_prompt(_request({"internet_search"}, triggering_input="what is the score now?"))
    assert "REAL-TIME DATA NEEDED" in full


def test_the_model_side_system_prompts_name_no_tool() -> None:
    """They reach every agent on every backend (router._generate_tool_response), whatever its roster."""
    from maxim.agents.prompt_builder import names_tool_outside
    from maxim.models.language import cloud_dispatch

    for name in ("SYSTEM_TOOL_RESPONSE", "SYSTEM_JSON_ONLY", "SYSTEM_ROUTE", "JSON_RULES"):
        text = getattr(cloud_dispatch, name)
        assert names_tool_outside(text, set()) == [] and _mentions(text, set()) == [], name


def test_the_gate_returns_text_byte_for_byte_when_it_removes_nothing() -> None:
    from maxim.agents.prompt_builder import gate_on_roster

    text = "=== Hint ===\nUse 'internet_search' now.\n"
    assert gate_on_roster(text, {"internet_search"}) == text


def test_the_acting_coachs_learned_experience_is_data_not_gated() -> None:
    """Its NAc / Cerebellum lines are what the agent learned; one may quote a tool it lacks (#1042 review)."""
    from maxim.agents.prompt_budgeter import PromptBudgeter
    from maxim.prompts.acting_coach import ActingCoachConfig

    context = replace(
        StructuredContext(timestamp=1_700_000_000.0),
        causal_context=[{"tool": "bash", "outcome": "Tool not registered: 'read_file'", "confidence": 0.9}],
    )
    request = replace(_request({"bash"}), context=context, acting_coach=ActingCoachConfig())
    sources: dict[str, str] = {}
    real_add = PromptBudgeter.add

    def spy(self, name, content, *args, **kwargs):  # type: ignore[no-untyped-def]
        sources[name] = kwargs["source"]
        return real_add(self, name, content, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(PromptBudgeter, "add", spy)
        _builder().build_prompt(request)
    assert sources.get("acting_coach") == "data" and sources.get("foundational") == "data", sources

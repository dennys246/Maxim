"""#1260 -- an agent's own tool descriptions ride on its registry, never the process-global ``TOOL_DESCRIPTIONS``.

``start_simulation_mode`` §5 wrote the narrator's 14 entries into ``modes.definitions.TOOL_DESCRIPTIONS``, so the AUT,
running beside the narrator in the same process, read ``respond: NOT AVAILABLE`` and had its terminal ``respond``
followed up; the write was never undone, so every later loop in the process inherited it. Owner decision
(2026-10-14): a registry-scoped overlay. ``SimToolRegistry.tool_descriptions()`` carries the narrator's entries
(``simulation/tools.py::NARRATOR_TOOL_DESCRIPTIONS``, read-only all the way down), ``ToolRegistry.tool_descriptions()``
none. The loop resolves the agent's once at setup (``runtime/loop_state.py::tool_description_overlay``, called by
``loop_setup.build_loop_run``: ``LoopRun.tool_overlay``) and hands it to both readers, each a required ``overlay=``:
the prompt's tool section and ``tool_dispatch.execute_and_learn``'s follow-up type. A registry without the method
fails the run at setup (``tests/unit/test_loop_setup.py``).

The end-to-end gates (a real ``start_simulation_mode`` run: the AUT's respond is terminal, its prompt never says
"NOT AVAILABLE", the narrator keeps its own entries, the global dict is untouched) are in
``tests/unit/test_orchestrator_characterization_golden.py``; the harness asserts the dict after every run. These
pin the interface and the dispatch.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from maxim.modes.definitions import TOOL_DESCRIPTIONS, get_tool_followup_type
from maxim.runtime.agent_loop import _describe_tools_for_prompt
from maxim.runtime.loop_state import tool_description_overlay
from maxim.simulation.tools import NARRATOR_TOOL_DESCRIPTIONS, SimToolRegistry
from maxim.tools.registry import ToolRegistry

SRC = Path(__file__).resolve().parents[2] / "src" / "maxim"


def _executor(registry: object) -> SimpleNamespace:
    return SimpleNamespace(registry=registry)


def test_a_plain_registry_carries_no_descriptions_and_the_narrators_carries_its_own() -> None:
    assert dict(ToolRegistry().tool_descriptions()) == {}
    assert SimToolRegistry().tool_descriptions() is NARRATOR_TOOL_DESCRIPTIONS
    assert NARRATOR_TOOL_DESCRIPTIONS["respond"]["description"] == "NOT AVAILABLE. Use send_message instead."
    with pytest.raises(TypeError):
        ToolRegistry().tool_descriptions()["respond"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        NARRATOR_TOOL_DESCRIPTIONS["respond"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):  # read-only all the way down
        NARRATOR_TOOL_DESCRIPTIONS["respond"]["followup_type"] = None  # type: ignore[index]
    with pytest.raises(TypeError):
        NARRATOR_TOOL_DESCRIPTIONS["inject_pain"]["params"]["intensity"] = "x"  # type: ignore[index]


def test_the_follow_up_type_is_the_calling_agents() -> None:
    aut = tool_description_overlay(_executor(ToolRegistry()))
    narrator = tool_description_overlay(_executor(SimToolRegistry()))
    assert get_tool_followup_type("respond", "active", overlay=aut) is None  # the builtin: terminal
    assert get_tool_followup_type("respond", "singularity", overlay=narrator) == "process"  # the decoy re-thinks
    assert get_tool_followup_type("send_message", "active", overlay=aut) is None
    assert get_tool_followup_type("send_message", "singularity", overlay=narrator) == "process"
    assert get_tool_followup_type("read_file", "singularity", overlay=narrator) == TOOL_DESCRIPTIONS["read_file"].get(
        "followup_type"
    ), "a name the overlay does not carry falls back to the builtins"


def test_the_overlay_is_required_so_a_caller_cannot_read_only_the_builtins() -> None:
    with pytest.raises(TypeError):
        get_tool_followup_type("respond", "active")  # type: ignore[call-arg]


def test_the_prompt_describes_each_tool_with_the_calling_agents_entry() -> None:
    aut_registry, narrator_registry = ToolRegistry(), SimToolRegistry()
    aut = _describe_tools_for_prompt(["respond"], _executor(aut_registry), overlay=aut_registry.tool_descriptions())
    narrator = _describe_tools_for_prompt(
        ["respond", "send_message", "inject_pain"],
        _executor(narrator_registry),
        overlay=narrator_registry.tool_descriptions(),
    )
    assert aut["respond"] == TOOL_DESCRIPTIONS["respond"]
    assert "NOT AVAILABLE" not in aut["respond"]["description"]
    assert narrator["respond"] == NARRATOR_TOOL_DESCRIPTIONS["respond"]
    assert narrator["send_message"] == NARRATOR_TOOL_DESCRIPTIONS["send_message"]
    # Thawed: the prompt builder renders only ``dict`` entries, and Wire 1/3 copy over them.
    assert type(narrator["inject_pain"]) is dict and type(narrator["inject_pain"]["params"]) is dict
    assert _describe_tools_for_prompt(["respond"], None, overlay={}) == {"respond": TOOL_DESCRIPTIONS["respond"]}


def test_the_resolver_refuses_a_registry_without_the_overlay() -> None:
    assert tool_description_overlay(None) == {}
    with pytest.raises(AttributeError):
        tool_description_overlay(_executor(SimpleNamespace()))
    with pytest.raises(TypeError, match="must return a Mapping"):
        tool_description_overlay(_executor(SimpleNamespace(tool_descriptions=lambda: ["respond"])))


def _dispatch(registry: Any, tool_name: str) -> Any:
    """The real ``tool_dispatch.execute_and_learn`` on one successful action, with the overlay the run's setup
    resolves from ``registry`` (``LoopRun.tool_overlay``)."""
    from maxim.runtime import tool_dispatch
    from maxim.runtime.executor import Executor

    executor = MagicMock(spec=Executor)
    executor.effective_operational_mode.return_value = "singularity"
    executor.execute.return_value = SimpleNamespace(success=True, error=None, output={"turn": 1})
    sim = MagicMock()
    sim.is_sim_mode = False
    return tool_dispatch.execute_and_learn(
        agent=MagicMock(_execution_tracker=None),
        agent_name="agent_1260",
        agent_id="agent_1260",
        executor=executor,
        sim=sim,
        state=MagicMock(data={}),
        environment=MagicMock(),
        memory=MagicMock(),
        hippocampus=None,
        memory_hub=None,
        result_cache=MagicMock(),
        autonomy_controller=MagicMock(),
        rec_outcome=MagicMock(),
        recent_outcomes=[],
        max_recent=5,
        llm_worker=None,
        context_pool=MagicMock(),
        nac=None,
        run_id="run_1260",
        overlay=tool_description_overlay(_executor(registry)),
        action={"tool_name": tool_name, "params": {}},
        confidence=0.9,
        proposal=SimpleNamespace(reasoning="probe", citations=[], cluster_id=None, clusters=None, triggering_input=""),
        observation={},
        human_involved=False,
    )


def test_the_dispatch_follows_up_the_narrators_tools_and_not_the_auts_respond() -> None:
    """The narrator's ``send_message`` is followed up (its own entry: ``process``); the AUT's ``respond`` is
    terminal (the builtin)."""
    narrator_send = _dispatch(SimToolRegistry(), "send_message")
    aut_respond = _dispatch(ToolRegistry(), "respond")
    assert narrator_send.success and narrator_send.followup is not None
    assert narrator_send.followup.followup_type == "process"
    assert aut_respond.success and aut_respond.followup is None
    assert _dispatch(SimToolRegistry(), "respond").followup is not None  # the narrator's decoy re-thinks


_MUTATORS = {"update", "setdefault", "pop", "popitem", "clear", "__setitem__", "__delitem__", "__ior__"}


def test_no_source_writes_the_global_tool_descriptions() -> None:
    """Nothing in ``src/maxim`` writes ``TOOL_DESCRIPTIONS`` after its definition: a method call that mutates it,
    an item assignment or deletion, or an augmented assignment. An agent's own entries go on its registry."""
    writes = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            target = None
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _MUTATORS:
                target = node.func.value
            elif isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
                targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Subscript):
                        target = t.value
                    elif isinstance(node, ast.AugAssign):
                        target = t
            name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
            if name == "TOOL_DESCRIPTIONS":
                writes.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert writes == []

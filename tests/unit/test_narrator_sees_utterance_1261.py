"""#1261: the narrator, the report and ``analyze_results`` see the AUT's WORDS, not ``respond``'s receipt.

The production ``RespondTool`` returns a delivery receipt (``{"delivered": True, "mode": "cli"}``); the
AUT's words live only in the call's ``message`` argument. Three joins read ``str(result_output)`` for the
speech tools, so the narrator (``send_message``'s ``response``), ``report.response_texts`` and
``analyze_results``'s ``response_samples`` all carried the receipt. ``say`` was not joined at all.

These tests run the PRODUCTION path end to end: the real ``RespondTool`` / ``SayTool`` on the canonical
``build_executor``, wrapped in ``InstrumentedExecutor`` (the one recording site), recording into a real
``SimulationBridge``'s sink while a stand-in AUT thread acts on each injected percept. Nothing here builds an
``ActionRecord`` by hand, so a record shape that production never makes cannot satisfy them.

What is speech is declared ON THE TOOL (``Tool.speech_params``, owner decision 2026-10-14): ``respond``,
``speak``, ``say`` and every affordance whose YAML declares ``speech_param`` (``base_humanoid_speak``). The recorder
reads the declaration off the tool the executor would actually run, so no reader keeps a list of tool names.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from maxim.simulation.bridge import SimulationBridge
from maxim.simulation.instrumented_executor import InstrumentedExecutor

_WORDS = "I am Maxim, and I remember the cavern."
_SAID = "The password is heron."


def _aut(tmp_path: Any) -> tuple[SimulationBridge, InstrumentedExecutor]:
    """A bridge plus the AUT's instrumented executor over the real speech tools."""
    from maxim.runtime.bootstrap import build_executor
    from maxim.tools.narrative import SayTool
    from maxim.tools.registry import ToolRegistry
    from maxim.tools.response import RespondTool, SpeakTool
    from maxim.utils.response_output import ResponseOutput

    out = ResponseOutput(sandbox_path=str(tmp_path))
    registry = ToolRegistry()
    for tool in (RespondTool(out), SpeakTool(out), SayTool()):
        registry.register(tool)
    bridge = SimulationBridge(response_timeout=5.0, settle_s=0.3)
    executor = InstrumentedExecutor(build_executor(registry, pain_bus=None, permissions=None), bridge.action_sink)
    return bridge, executor


def _embodied_aut() -> tuple[SimulationBridge, InstrumentedExecutor, str]:
    """A bridge plus the AUT's instrumented executor over ``base_humanoid``'s GENERATED tools, and the name of
    its ``communication.speak`` affordance tool (``base_humanoid_speak``)."""
    from maxim.embodiment.component_registry import ComponentRegistry
    from maxim.embodiment.tool_bridge import generate_tools_for_entity
    from maxim.runtime.bootstrap import build_executor
    from maxim.tools.registry import ToolRegistry

    registry = ToolRegistry()
    tools = generate_tools_for_entity(ComponentRegistry().instantiate("bodies/base_humanoid"), registry)
    [speak] = [t.name for t in tools if t.name.endswith("_speak")]
    bridge = SimulationBridge(response_timeout=5.0, settle_s=0.3)
    executor = InstrumentedExecutor(build_executor(registry, pain_bus=None, permissions=None), bridge.action_sink)
    return bridge, executor, speak


def _act_on_next_percept(bridge: SimulationBridge, act: Any) -> threading.Thread:
    """The stand-in AUT: wait for the narrator's percept, then act (``act`` runs on this thread)."""

    def _worker() -> None:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if bridge.percept_source.next_percept() is not None:
                act()
                return
            time.sleep(0.02)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


def _run(executor: InstrumentedExecutor, *calls: tuple[str, dict[str, Any]]) -> Any:
    def _act() -> None:
        for tool_name, params in calls:
            executor.execute({"tool_name": tool_name, "params": params})

    return _act


def _send_message(bridge: SimulationBridge, text: str) -> dict[str, Any]:
    """What the narrator gets back from its ``send_message`` tool."""
    from maxim.simulation.tools import SendMessageTool

    result = SendMessageTool(bridge=bridge).run(text=text)
    assert result.success, result.error
    return result.output


def _report(bridge: SimulationBridge) -> Any:
    from maxim.simulation.report import build_report

    return build_report(
        goal="g",
        mode="generative",
        bridge=bridge,
        duration_s=1.0,
        finish_reason="completed",
        started_at=0.0,
        provenance={},
    )


def _analysis(bridge: SimulationBridge) -> dict[str, Any]:
    from maxim.simulation.tools import AnalyzeResultsTool

    result = AnalyzeResultsTool(bridge=bridge).run(focus="all")
    assert result.success, result.error
    return result.output


_RED = pytest.mark.xfail(strict=True, reason="#1261: the joins read respond's receipt, not the AUT's words")


@_RED
def test_the_narrator_report_and_analysis_get_the_words_respond_carried(tmp_path: Any) -> None:
    bridge, executor = _aut(tmp_path)
    t = _act_on_next_percept(bridge, _run(executor, ("respond", {"message": _WORDS})))
    turn = _send_message(bridge, "Who are you?")
    t.join(timeout=5)

    assert turn["response"] == _WORDS  # the narrator's view of the turn: first, so the red names the symptom
    [action] = bridge.get_all_actions()
    assert action.result_output == {"delivered": True, "mode": "cli"}  # the receipt is unchanged (option A rejected)
    assert action.utterance == _WORDS
    assert turn["actions"] == [
        {"tool": "respond", "success": True, "output": str(action.result_output), "utterance": _WORDS}
    ]
    assert _report(bridge).response_texts == [_WORDS]
    assert _analysis(bridge)["response_samples"] == [_WORDS]


@_RED
def test_say_counts_as_speech(tmp_path: Any) -> None:
    bridge, executor = _aut(tmp_path)
    t = _act_on_next_percept(bridge, _run(executor, ("respond", {"message": _WORDS}), ("say", {"text": _SAID})))
    turn = _send_message(bridge, "What is the password?")
    t.join(timeout=5)

    assert turn["response"] == f"{_WORDS}\n{_SAID}"
    assert [a.utterance for a in bridge.get_all_actions()] == [_WORDS, _SAID]
    assert _report(bridge).response_texts == [_WORDS, _SAID]
    assert _analysis(bridge)["response_samples"] == [_WORDS, _SAID]


@_RED
def test_say_takes_its_own_message_alias(tmp_path: Any) -> None:
    """The validator requires ``text``; a blank one lets ``SayTool`` fall back to ``message``, and so must we."""
    bridge, executor = _aut(tmp_path)
    t = _act_on_next_percept(bridge, _run(executor, ("say", {"text": "", "message": _SAID})))
    turn = _send_message(bridge, "Say it.")
    t.join(timeout=5)

    assert turn["response"] == _SAID


@_RED
def test_an_aliased_say_is_speech(tmp_path: Any) -> None:
    """The executor runs ``talk`` as ``say`` (``TOOL_ALIASES``): its words are speech, its record keeps ``talk``."""
    bridge, executor = _aut(tmp_path)
    t = _act_on_next_percept(bridge, _run(executor, ("talk", {"text": _SAID})))
    turn = _send_message(bridge, "Tell the guard.")
    t.join(timeout=5)

    assert turn["response"] == _SAID
    [action] = bridge.get_all_actions()
    assert (action.tool_name, action.result_output) == ("talk", {"said": _SAID, "mode": "narrative"})
    assert _report(bridge).response_texts == [_SAID]


@_RED
def test_speak_carries_its_words_not_its_tts_receipt(tmp_path: Any) -> None:
    bridge, executor = _aut(tmp_path)
    t = _act_on_next_percept(bridge, _run(executor, ("speak", {"text": _SAID})))
    turn = _send_message(bridge, "Say it aloud.")
    t.join(timeout=5)

    assert turn["response"] == _SAID
    [action] = bridge.get_all_actions()
    assert action.result_output == {"spoken": False, "printed": True, "reason": "TTS not enabled"}


@_RED
def test_the_bodys_generated_speak_affordance_reaches_the_narrator() -> None:
    """``base_humanoid``'s ``communication.speak`` (``params: {message: str}``) is the embodied AUT's voice."""
    bridge, executor, speak = _embodied_aut()
    t = _act_on_next_percept(bridge, _run(executor, (speak, {"message": _WORDS})))
    turn = _send_message(bridge, "Call out to the guard.")
    t.join(timeout=5)

    assert turn["response"] == _WORDS
    [action] = bridge.get_all_actions()
    assert (action.tool_name, action.result_success, action.utterance) == ("base_humanoid_speak", True, _WORDS)

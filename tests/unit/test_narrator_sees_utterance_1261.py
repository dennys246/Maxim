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

from maxim.simulation.bridge import SimulationBridge
from maxim.simulation.instrumented_executor import InstrumentedExecutor
from tests.policy_helpers import TEST_POLICY

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
    executor = InstrumentedExecutor(build_executor(registry, pain_bus=None, policy=TEST_POLICY), bridge.action_sink)
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
    executor = InstrumentedExecutor(build_executor(registry, pain_bus=None, policy=TEST_POLICY), bridge.action_sink)
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


def test_say_counts_as_speech(tmp_path: Any) -> None:
    bridge, executor = _aut(tmp_path)
    t = _act_on_next_percept(bridge, _run(executor, ("respond", {"message": _WORDS}), ("say", {"text": _SAID})))
    turn = _send_message(bridge, "What is the password?")
    t.join(timeout=5)

    assert turn["response"] == f"{_WORDS}\n{_SAID}"
    assert [a.utterance for a in bridge.get_all_actions()] == [_WORDS, _SAID]
    assert _report(bridge).response_texts == [_WORDS, _SAID]
    assert _analysis(bridge)["response_samples"] == [_WORDS, _SAID]


def test_say_takes_its_own_message_alias(tmp_path: Any) -> None:
    """The validator requires ``text``; a blank one lets ``SayTool`` fall back to ``message``, and so must we."""
    bridge, executor = _aut(tmp_path)
    t = _act_on_next_percept(bridge, _run(executor, ("say", {"text": "", "message": _SAID})))
    turn = _send_message(bridge, "Say it.")
    t.join(timeout=5)

    assert turn["response"] == _SAID


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


def test_speak_carries_its_words_not_its_tts_receipt(tmp_path: Any) -> None:
    bridge, executor = _aut(tmp_path)
    t = _act_on_next_percept(bridge, _run(executor, ("speak", {"text": _SAID})))
    turn = _send_message(bridge, "Say it aloud.")
    t.join(timeout=5)

    assert turn["response"] == _SAID
    [action] = bridge.get_all_actions()
    assert action.result_output == {"spoken": False, "printed": True, "reason": "TTS not enabled"}


def test_the_bodys_generated_speak_affordance_reaches_the_narrator() -> None:
    """``base_humanoid``'s ``communication.speak`` (``params: {message: str}``) is the embodied AUT's voice."""
    bridge, executor, speak = _embodied_aut()
    t = _act_on_next_percept(bridge, _run(executor, (speak, {"message": _WORDS})))
    turn = _send_message(bridge, "Call out to the guard.")
    t.join(timeout=5)

    assert turn["response"] == _WORDS
    [action] = bridge.get_all_actions()
    assert (action.tool_name, action.result_success, action.utterance) == ("base_humanoid_speak", True, _WORDS)


# ── The mechanism's own guards ────────────────────────────────────────────────────────────────────────────────


def test_blocked_and_failed_speech_carries_no_utterance(tmp_path: Any) -> None:
    """A blocked call never ran and a failed one said nothing: neither is the AUT's speech."""
    bridge, executor = _aut(tmp_path)

    def _act() -> None:
        executor.record_block("respond", "FearAgent: blocked", params={"message": "words that never left"})
        executor.execute({"tool_name": "respond", "params": {"message": "   "}})  # RespondTool refuses a blank message

    t = _act_on_next_percept(bridge, _act)
    turn = _send_message(bridge, "Say something risky.")
    t.join(timeout=5)

    actions = bridge.get_all_actions()
    assert [(a.blocked, a.result_success) for a in actions] == [(True, False), (False, False)]
    assert [a.utterance for a in actions] == [None, None]
    assert turn["response"] is None
    assert _report(bridge).response_texts == []
    assert "response_samples" not in _analysis(bridge)


def test_console_talk_returns_the_words_through_the_real_bridge(tmp_path: Any, monkeypatch: Any) -> None:
    """With ``_extract_reply`` deleted, ``talk()`` reads the bridge's ``response``: still the words, never the receipt."""
    from maxim.console.handle import MaximHandle

    bridge, executor = _aut(tmp_path)
    h = MaximHandle.__new__(MaximHandle)
    h.agent_id = "test_agent"
    h._stopped = False
    h._campaign_lock = threading.Lock()
    monkeypatch.setattr(MaximHandle, "_ensure_talk_loop", lambda self: bridge)

    t = _act_on_next_percept(bridge, _run(executor, ("respond", {"message": _WORDS})))
    result = h.talk("Who are you?")
    t.join(timeout=5)

    assert result["response"] == _WORDS


def test_compression_drops_the_utterance_with_the_args(tmp_path: Any) -> None:
    """The ring buffer's compressed records keep the audit trail, not the heavy text: no words survive the args."""
    from maxim.simulation.sinks import RecordingSink

    bridge, _ = _aut(tmp_path)
    sink = RecordingSink(max_actions=2)
    executor = InstrumentedExecutor(_aut(tmp_path)[1]._executor, sink)
    for i in range(3):
        executor.execute({"tool_name": "say", "params": {"text": f"line {i}"}})

    first, *rest = sink.actions
    assert (first.tool_name, first.tool_args, first.utterance) == ("say", {}, None)
    assert [a.utterance for a in rest] == ["line 1", "line 2"]


def test_a_non_string_tool_name_is_recorded_without_an_utterance(tmp_path: Any) -> None:
    """Dispatch refuses a non-string name as an invalid action; the recorder must not crash resolving it."""
    bridge, executor = _aut(tmp_path)
    out = executor.execute({"tool_name": ["say"], "params": {"text": _SAID}})

    assert out.success is False
    [action] = bridge.get_all_actions()
    assert action.utterance is None


def test_say_and_speak_read_text_before_their_aliases(tmp_path: Any) -> None:
    """``SayTool`` and ``SpeakTool`` take ``text`` first: the record must take the same words, not an alias's."""
    bridge, executor = _aut(tmp_path)
    executor.execute({"tool_name": "say", "params": {"text": "A", "message": "B"}})
    executor.execute({"tool_name": "speak", "params": {"text": "A", "message": "B", "phrase": "C"}})
    executor.execute({"tool_name": "speak", "params": {"text": "", "message": "B", "phrase": "C"}})  # blank text

    assert [a.utterance for a in bridge.get_all_actions()] == ["A", "A", "B"]


def test_speech_is_declared_on_the_tool() -> None:
    """The declaration each speech tool carries matches the params its own ``execute`` reads, in that order."""
    from maxim.tools.base import Tool
    from maxim.tools.narrative import SayTool, ThinkTool
    from maxim.tools.response import RespondTool, SpeakTool

    assert Tool.speech_params == ()
    assert RespondTool.speech_params == ("message",)
    assert SpeakTool.speech_params == SayTool.speech_params == ("text", "message", "phrase")
    assert ThinkTool.speech_params == ()  # a thought is not speech


def test_the_affordance_generator_declares_only_the_bodys_voice() -> None:
    """Of ``base_humanoid``'s generated tools only its declared ``communication.speak`` is speech."""
    from maxim.embodiment.component_registry import ComponentRegistry
    from maxim.embodiment.tool_bridge import ModulatorAffordanceTool, generate_tools_for_entity
    from maxim.tools.registry import ToolRegistry

    tools = generate_tools_for_entity(ComponentRegistry().instantiate("bodies/base_humanoid"), ToolRegistry())
    speech = {t.name: t.speech_params for t in tools if t.speech_params}
    assert speech == {"base_humanoid_speak": ("message",)}
    assert any(isinstance(t, ModulatorAffordanceTool) and t.modulator.name != "communication" for t in tools)


def _fixture_entity(**affordance: Any) -> Any:
    """A one-affordance entity parsed by the real spec loader: ``communication.speak`` with a ``message`` param."""
    from maxim.embodiment.spec import _parse_entity

    speak = {"params": {"message": "str"}, "description": "say it", **affordance}
    return _parse_entity(
        {
            "name": "fixture",
            "entity_type": "body",
            "modulators": {"communication": {"abstract": True, "affordances": {"speak": speak}}},
        }
    )


def _speech_of(entity: Any) -> dict[str, tuple[str, ...]]:
    from maxim.embodiment.tool_bridge import generate_tools_for_entity
    from maxim.tools.registry import ToolRegistry

    return {t.name: t.speech_params for t in generate_tools_for_entity(entity, ToolRegistry()) if t.speech_params}


def test_the_yaml_speech_param_is_copied_onto_the_tool() -> None:
    assert _speech_of(_fixture_entity(speech_param="message")) == {"fixture_speak": ("message",)}


def test_speech_is_never_inferred_from_the_modulator_or_param_name() -> None:
    """A ``communication.speak`` with a ``message`` param is NOT speech until it declares ``speech_param``."""
    assert _speech_of(_fixture_entity()) == {}


def test_a_speech_param_naming_no_declared_param_fails_the_load() -> None:
    import pytest

    with pytest.raises(ValueError, match=r"fixture\.communication\.speak: speech_param 'text' names none"):
        _fixture_entity(speech_param="text")


def test_a_non_string_speech_param_fails_the_load_with_its_label() -> None:
    import pytest

    with pytest.raises(ValueError, match=r"fixture\.communication\.speak: speech_param \['message'\] must be"):
        _fixture_entity(speech_param=["message"])


_SHIPPED_SPEECH = {
    "bodies/base_humanoid": "speak",
    "items/radio_transceiver": "transmit",
    "environments/space_station_bridge": "comms_hail",
    "environments/roman_forum": "speak_publicly",
    **{
        f"npcs/{npc}": "speak"
        for npc in (
            "base_humanoid",
            "blacksmith",
            "corpo_guard",
            "detective",
            "ferryman",
            "guard",
            "merchant",
            "netrunner",
            "roman_legionary",
            "street_fixer",
            "sysadmin",
            "thief",
            "wizard",
        )
    },
}


def test_the_owners_speech_affordances_declare_their_words() -> None:
    """The body's voice, every NPC's ``social.speak``, and the SEMs that extend speech (the radio's ``transmit``, the
    bridge's ``comms_hail``, the forum's ``speak_publicly``) each declare ``message`` (owner decision 2026-10-14), and
    nothing else in those components is speech."""
    from maxim.embodiment.component_registry import ComponentRegistry

    registry = ComponentRegistry()
    for ref, affordance in _SHIPPED_SPEECH.items():
        entity = registry.instantiate(ref)
        assert _speech_of(entity) == {f"{entity.name}_{affordance}": ("message",)}, ref


def test_an_unregistered_or_non_speech_tool_records_no_utterance(tmp_path: Any) -> None:
    """Dispatch refuses an unknown tool, and a non-speech tool's ``message`` param is not something it said."""
    from maxim.tools.base import Tool

    class _Notify(Tool):
        name = "notify"
        input_schema = {"message": str}

        def execute(self, **kwargs: Any) -> Any:
            return {"ok": True}

    bridge, executor = _aut(tmp_path)
    executor._executor.registry.register(_Notify())
    executor.execute({"tool_name": "no_such_tool", "params": {"message": _SAID}})
    executor.execute({"tool_name": "notify", "params": {"message": _SAID}})

    assert [(a.tool_name, a.result_success, a.utterance) for a in bridge.get_all_actions()] == [
        ("no_such_tool", False, None),
        ("notify", True, None),
    ]


def test_observe_actions_shows_the_words_after_the_index(tmp_path: Any) -> None:
    """The narrator's full-history view carries what the AUT said (#1261 review S1), index first, 300-char cap."""
    from maxim.simulation.tools import ObserveActionsTool

    bridge, executor = _aut(tmp_path)
    executor.execute({"tool_name": "respond", "params": {"message": _WORDS}})
    executor.execute({"tool_name": "say", "params": {"text": "x" * 400}})

    out = ObserveActionsTool(bridge=bridge).run().output
    first, second = out["actions"]
    assert list(first) == ["index", "tool", "success", "output", "utterance"]
    assert first == {
        "index": 0,
        "tool": "respond",
        "success": True,
        "output": str({"delivered": True, "mode": "cli"}),
        "utterance": _WORDS,
    }
    assert (second["index"], second["utterance"]) == (1, "x" * 300)
    assert ObserveActionsTool(bridge=bridge).run(since_index=1).output["actions"][0]["index"] == 1


def test_check_completion_counts_what_was_said(tmp_path: Any) -> None:
    """``responses`` counts speech that was delivered (#1261 review S2): ``say`` counts, a refused ``respond`` not."""
    from maxim.simulation.tools import CheckCompletionTool

    bridge, executor = _aut(tmp_path)
    executor.execute({"tool_name": "respond", "params": {"message": _WORDS}})
    executor.execute({"tool_name": "respond", "params": {"message": "   "}})  # refused: nothing said
    executor.execute({"tool_name": "say", "params": {"text": _SAID}})
    executor.execute({"tool_name": "talk", "params": {"text": _SAID}})  # runs as ``say``

    # 3 said; a count of ``respond``/``speak`` by name would be 2 (the refused one in, both ``say``s out).
    assert CheckCompletionTool(bridge=bridge).run().output["responses"] == 3


def test_the_interactive_views_show_the_words(tmp_path: Any, monkeypatch: Any) -> None:
    """The interactive turn view, summary and context lines preview what the AUT said (#1261 review S4), and a
    non-speech tool's own status ``message`` is still shown."""
    from maxim.simulation import interactive, sim_logger
    from maxim.tools.base import Tool

    class _Stub(Tool):
        name = "move_head"

        def execute(self, **kwargs: Any) -> Any:
            return {"message": "No live robot connected"}

    bridge, executor = _aut(tmp_path)
    executor._executor.registry.register(_Stub())
    executor.execute({"tool_name": "respond", "params": {"message": _WORDS}})
    executor.execute({"tool_name": "move_head", "params": {}})
    executor.record_block("say", "FearAgent: blocked", params={"text": _SAID})
    sink = bridge.action_sink
    expected = [f"[OK] respond: {_WORDS}", "[OK] move_head: No live robot connected", "[BLOCKED] say"]

    assert interactive._summarize_actions(sink) == expected
    shown: list[str] = []
    monkeypatch.setattr(sim_logger, "display_scene", shown.append)
    interactive._show_turn_results(sink)
    assert shown == [f"  {line}" for line in expected[:2]] + ["  [BLOCKED] say: FearAgent: blocked"]
    shown.clear()
    interactive._print_summary(sink)
    assert shown == ["Actions: 3"] + [f"  {line}" for line in expected]


def test_the_sword_demo_npc_declares_its_speech() -> None:
    """The scenario-local NPC (``grim_ferryman``'s ``social.speak``) carries the key as the library NPCs do."""
    from pathlib import Path

    from maxim.embodiment.spec import load_spec

    spec = load_spec(Path(__file__).resolve().parents[2] / "scenarios" / "embodiment" / "sword_npc_demo.yaml")
    speech: dict[str, tuple[str, ...]] = {}
    for entity in spec.world_entities:
        speech.update(_speech_of(entity))
    assert speech == {"grim_ferryman_speak": ("message",)}


def test_registry_find_is_one_lookup_that_never_raises() -> None:
    """``ToolRegistry.find``: the tool (active or deactivated) or ``None``; ``SimToolRegistry.find`` never hands
    back its redirect decoy."""
    from maxim.simulation.tools import SimToolRegistry
    from maxim.tools.narrative import SayTool
    from maxim.tools.registry import ToolRegistry

    say = SayTool()
    registry = ToolRegistry()
    registry.register_scene_tools([say], "s")  # a SCENE tool: core tools cannot be deactivated
    assert registry.find("say") is say
    assert registry.find("nope") is None
    assert registry.deactivate_tool("say") is True
    assert not registry.is_tool_active("say")  # really deactivated, so the next line is not vacuous
    assert registry.find("say") is say  # deactivated, still registered (dispatch decides)
    registry.deregister("say")
    assert registry.find("say") is None

    sim = SimToolRegistry()
    sim.register(say)
    assert sim.find("say") is say
    assert sim.find("nope") is None
    assert sim.get("nope") is not None  # get still hands dispatch its redirect decoy


def test_a_tool_deregistered_mid_lookup_still_dispatches_and_records(tmp_path: Any) -> None:
    """The DM thread deregisters scene tools while the AUT dispatches. Here it happens deterministically right
    after any membership listing: a check-then-``get`` lookup would raise ``KeyError`` before dispatch (and
    upstream would lose the action); ``find`` sees the tool gone, and dispatch reports it."""
    from maxim.runtime.bootstrap import build_executor
    from maxim.tools.narrative import SayTool
    from maxim.tools.registry import ToolRegistry

    class _RacingRegistry(ToolRegistry):
        """The other thread deregisters ``say`` at the recorder's lookup: the SECOND registry access of the call
        (the first is the alias resolution). A ``find`` sees it gone; a ``list_all`` has already listed it."""

        accesses = 0

        def _race(self) -> None:
            self.accesses += 1
            if self.accesses == 2:
                self.deregister("say")

        def find(self, name: str) -> Any:
            self._race()
            return super().find(name)

        def list_all(self) -> list[str]:
            names = super().list_all()
            self._race()
            return names

    registry = _RacingRegistry([SayTool()])
    bridge = SimulationBridge(response_timeout=5.0, settle_s=0.3)
    executor = InstrumentedExecutor(build_executor(registry, pain_bus=None, policy=TEST_POLICY), bridge.action_sink)
    out = executor.execute({"tool_name": "say", "params": {"text": _SAID}})

    [action] = bridge.get_all_actions()
    assert registry.accesses >= 2  # the race happened
    assert (action.tool_name, action.result_success, action.utterance) == ("say", False, None)
    assert out.success is False and "not registered" in (out.error or "")

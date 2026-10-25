"""The memory-side seams of Session R's recall fix (docs/plans/recall_1_3_4.md), each pinned on real stores.

- #1163: ``Outcome`` stores a tool result as plain data (the seam), and a 1.0 file's ``ToolOutput(...)`` repr is
  parsed on load, never evaluated (D2);
- #1064 / D1 / D3: one tokenizer keys every store; the ranker scores what captures write and returns nothing that
  shares no content word;
- #1447 / F11: the cue path finds the answer, and a record of the turn's own words is not recalled;
- D3: NAc's keyword scan and ATL's token index match on the same tokens (lemmas, possessives, punctuation);
- D8: the user's input is published to memory ONCE, as it arrives, on every channel -- the robot's too;
- the bus re-raises a capture-contract break instead of swallowing it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from maxim.memory.encoding import EncodingSignals

_FACT = "My name is Denny and my favourite colour is green."
_QUESTION = "What is Denny's favorite color?"


def _hippocampus(tmp_path: Path) -> Any:
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

    return Hippocampus(HippocampusConfig(persistence_path=str(tmp_path / "hippocampus.json")))


def _say(h: Any, text: str) -> str:
    """A user's words, captured the way ``MemoryAgent`` captures a percept."""
    mid = h.capture_from_loop(
        observation={"detections": [], "transcript": None, "cli_input": text, "salience": 0.9, "novelty": 0.5},
        state={"source": "percept"},
        intent={},
        decision={},
        action={},
        result={},
        encoding=EncodingSignals.unmeasured("memory_agent"),
        situation=None,
    )
    time.sleep(0.002)
    return mid


# ── #1163: the seam and the legacy repr ──────────────────────────────────────────────────────────────────────────


def test_a_tool_result_is_stored_as_plain_data_without_the_executors_stamps() -> None:
    from maxim.memory.types import Outcome
    from maxim.tools.base import ToolOutput

    out = ToolOutput(success=True, output={"items": ("a", "b")}, pain=0.4, drive_relief=(("oxygen", 0.5),))
    assert Outcome(success=True, result=out).result == {"success": True, "output": {"items": ["a", "b"]}, "error": None}


@pytest.mark.parametrize(
    ("stored", "loaded"),
    [
        (
            "ToolOutput(success=True, output='Nice to meet you, Denny!', error=None, error_kind=None, metadata={})",
            {"success": True, "output": "Nice to meet you, Denny!", "error": None},
        ),
        # an enum repr is not Python syntax: cut before error_kind= and read the three fields
        (
            "ToolOutput(success=False, output=None, error='missing', error_kind=<ToolErrorKind.FILE_NOT_FOUND: 'x'>)",
            {"success": False, "output": None, "error": "missing"},
        ),
        # a non-literal output drops only the output
        (
            "ToolOutput(success=True, output=PosixPath('/tmp/x'), error=None)",
            {"success": True, "output": None, "error": None},
        ),
        # never evaluated: anything but a ToolOutput call is dropped
        ("ToolOutput(" + "__import__('os').system('true')" + ")", None),
        ("ToolOutput(success=1, output='x')", None),
        ("ToolOutput(" + "x" * 20_000, None),
        ("plain text result", "plain text result"),
    ],
)
def test_a_legacy_repr_is_parsed_never_evaluated(stored: str, loaded: Any) -> None:
    from maxim.memory.types import Outcome

    assert Outcome.from_dict({"result": stored}).result == loaded


def test_a_1_0_store_with_reprs_loads_as_data_and_saves_as_1_1(tmp_path: Path) -> None:
    from maxim.memory.hippocampus_persistence import HIPPOCAMPUS_FORMAT_VERSION

    h = _hippocampus(tmp_path)
    h.capture_from_loop(
        observation={},
        state={"data": {"mode": "active"}},
        intent={"goal": "greet"},
        decision={},
        action={"tool": "respond", "params": {"message": "hi"}},
        result={"success": True},
        encoding=EncodingSignals.unmeasured("loop"),
        situation=None,
    )
    h.save()
    path = tmp_path / "hippocampus.json"
    data = json.loads(path.read_text())
    data["_format_version"] = "1.0"  # a file written before #1163 ...
    data["memories"][0]["outcome"]["result"] = "ToolOutput(success=True, output='hi', error=None, pain=0.2)"
    path.write_text(json.dumps(data))

    old = _hippocampus(tmp_path)
    old.load()
    [mem] = old._memories.values()
    assert mem.outcome.result == {"success": True, "output": "hi", "error": None}
    assert old.search_by_content("pain") == []  # the repr's field names are not memory text any more
    old.save()
    assert json.loads(path.read_text())["_format_version"] == HIPPOCAMPUS_FORMAT_VERSION == "1.1"


# ── #1064 / D1 / D3: the ranker ──────────────────────────────────────────────────────────────────────────────────


def test_the_ranker_reads_what_captures_write_and_returns_no_zero_score(tmp_path: Path) -> None:
    h = _hippocampus(tmp_path)
    fact = _say(h, _FACT)
    for day in ("Monday", "Tuesday", "Wednesday"):
        _say(h, f"Remind me to water the ferns on {day} morning.")
    assert [m.id for m in h.recall(query=_QUESTION, limit=5)] == [fact]  # "Denny's" -> denny; nothing else scores
    assert h.recall(query="the a of", limit=5) == []  # no content word: no recency fill (D1)


def test_matching_reads_text_values_never_a_dicts_keys(tmp_path: Path) -> None:
    h = _hippocampus(tmp_path)
    _say(h, _FACT)
    assert h.search_by_content("salience") == [] and h.search_by_content("cli_input") == []
    assert len(h.search_by_content("green")) == 1


# ── #1447 / F11: the cue and the turn's own capture ──────────────────────────────────────────────────────────────


def test_the_cue_finds_the_answer_and_never_the_turns_own_words(tmp_path: Path) -> None:
    from maxim.integration.bio_enrichment import BioEnrichmentPipeline, EnrichmentContext

    h = _hippocampus(tmp_path)
    fact = _say(h, _FACT)
    _say(h, _QUESTION)  # this turn's own capture, stored before the query ran
    ctx = EnrichmentContext(active_goal="converse with the user", user_input=_QUESTION)
    found, paths, *_ = BioEnrichmentPipeline(hippocampus=h)._query_hippocampus_traced(_QUESTION, [], context=ctx)
    assert [s.memory_id for s in found] == [fact] and paths == ["cue"]
    assert "green" in found[0].summary and "<<TOOL_OUTPUT id=" in found[0].summary  # framed, never raw (#1444)


def test_an_episode_with_nothing_to_render_is_counted() -> None:
    from types import SimpleNamespace

    from maxim.integration.bio_enrichment import BioEnrichmentPipeline

    pipe = BioEnrichmentPipeline()
    empty = SimpleNamespace(id="m1", tool_name="", goal="", input_text="", outcome=SimpleNamespace(result={}))
    assert pipe._summarize_episode(empty) == "past experience" and pipe._unrendered_episodes == 1


# ── D3: NAc and ATL on the same tokens ───────────────────────────────────────────────────────────────────────────


def test_nac_matches_a_lemma_against_an_inflected_signature() -> None:
    from maxim.decisions.causal_link import Valence
    from maxim.decisions.nac import NAc

    nac = NAc()
    for _ in range(10):
        nac.observe(
            event_type="tool",
            event_signature="tool:eat_berries",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=Valence.POSITIVE,
            delta_seconds=1.0,
        )
    assert [lk.event_signature for lk in nac.scan_links_for_keywords(["berry"])] == ["tool:eat_berries"]


def test_atl_finds_a_concept_by_a_content_word_but_never_a_substrate_utterance() -> None:
    from maxim.memory.atl import ATL
    from maxim.memory.semantic_types import SemanticMemory

    atl = ATL()
    atl.store(SemanticMemory(id="c1", timestamp=time.time(), name="red mugs", category="object"))
    atl.store(SemanticMemory(id="c2", timestamp=time.time(), name=_FACT, category="text", substrate_modality="text"))
    assert [c.id for c in atl.recall(name_token="mug")] == ["c1"]
    assert atl.recall(name_token="denny") == []  # a substrate node is named by raw percept text: never a "concept"


# ── D8: the user's input reaches memory once, as it arrives ──────────────────────────────────────────────────────


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def publish_arrival(self, state: Any, text: str, *, source: str) -> None:
        self.calls.append((text, source))


@pytest.mark.parametrize(
    ("observation", "pending", "expected"),
    [
        ({"cli_input": "hello there"}, {}, [("hello there", "cli")]),
        ({}, {"pending_cli_input": "from the REPL"}, [("from the REPL", "cli")]),
        ({}, {"pending_voice_input": "maxim, lights"}, [("maxim, lights", "voice")]),
        ({"cli_input": "[ACTION_FOLLOWUP type=process tool=x]: out"}, {}, []),
        ({"cli_input": None}, {}, []),
    ],
)
def test_every_input_channel_is_published_once_at_arrival(observation: dict, pending: dict, expected: list) -> None:
    from maxim.runtime.loop_perception import take_user_input
    from maxim.runtime.state import RuntimeState

    state = RuntimeState()
    state.data.update(pending)
    agent = _Recorder()
    take_user_input(observation, state, agent)
    assert agent.calls == expected
    assert "pending_cli_input" not in state.data and "pending_voice_input" not in state.data  # processed once


def test_an_arrival_reaches_memory_agent_even_with_a_capture_manager_and_is_not_republished() -> None:
    """The robot builds ``PerceptionAgent`` with a CaptureManager, so ``propose_intent`` returns before publishing;
    the user's words now arrive regardless. The slot is cleared, so a later §3 never publishes them again (#1414)."""
    from maxim.agents.bus import AgentBus, Percept
    from maxim.agents.perception_agent import PerceptionAgent
    from maxim.runtime.state import RuntimeState

    bus = AgentBus()
    heard: list[Any] = []
    bus.subscribe(Percept, heard.append)
    agent = PerceptionAgent(bus=bus)
    agent._capture_manager = object()  # the robot's vision/audio path
    state = RuntimeState()
    state.data["cli_input"] = "My name is Denny."
    agent.publish_arrival(state, "My name is Denny.", source="voice")
    agent._capture_manager = None
    agent.propose_intent(state, None)
    texts = [p.cli_input for p in heard if p.cli_input]
    assert texts == ["My name is Denny."] and heard[0].source == "voice"


def test_the_bus_reraises_a_capture_contract_break_after_every_handler_ran() -> None:
    from maxim.agents.bus import AgentBus, Percept
    from maxim.memory.encoding import EncodingContractError

    bus = AgentBus()
    ran: list[str] = []

    def _breaks(_p: Any) -> None:
        ran.append("breaks")
        raise EncodingContractError("no encoding")

    bus.subscribe(Percept, _breaks)
    bus.subscribe(Percept, lambda _p: ran.append("next"))
    with pytest.raises(EncodingContractError):
        bus.publish(Percept(timestamp=0.0, source="cli", cli_input="x"))
    assert ran == ["breaks", "next"]


@pytest.mark.parametrize(
    ("observation", "addressed"),
    [
        ({"cli_input": "hi", "source": "cli"}, True),
        ({"cli_input": "hi"}, True),  # no source: the interactive CLI / API / robot channel
        ({"transcript": "hi", "source": "voice"}, True),
        ({"cli_input": "a creak in the dark", "source": "minecraft"}, False),
        ({"source": "cli"}, False),
    ],
)
def test_addressed_input_is_read_from_its_source(observation: dict, addressed: bool) -> None:
    from maxim.runtime.loop_perception import addressed_text

    assert bool(addressed_text(observation)) is addressed


def test_a_follow_up_carries_the_executed_tools_untrusted_declaration_through_an_alias() -> None:
    from types import SimpleNamespace

    from maxim.runtime.tool_dispatch import _tool_followup

    think = SimpleNamespace(untrusted_output=True)
    executor = SimpleNamespace(
        resolve_tool_name=lambda name: "think" if name == "reflect" else name,
        registry=SimpleNamespace(find=lambda name: think if name == "think" else None),
    )
    proposal = SimpleNamespace(triggering_input="why?")
    assert _tool_followup(executor, "reflect", "r", proposal, "process", "active").untrusted_output is True
    assert _tool_followup(executor, "bash", "r", proposal, "process", "active").untrusted_output is False

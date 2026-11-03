"""The memory-side seams of Session R's recall fix (docs/plans/recall_1_3_4.md), each pinned on real stores.

- #1163: ``Outcome`` stores a tool result as plain data (the seam), and a 1.0 file's ``ToolOutput(...)`` repr is
  parsed on load, never evaluated (D2);
- #1064 / D1 / D3: one tokenizer keys every store; the ranker scores what captures write and returns nothing that
  shares no content word;
- #1447 / F11 / #1573: the cue path finds the answer, and the turn's own records are not recalled, by identity;
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
    assert json.loads(path.read_text())["_format_version"] == HIPPOCAMPUS_FORMAT_VERSION == "1.2"


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
    own = _say(h, _QUESTION)  # this turn's own capture, stored before the query ran
    ctx = EnrichmentContext(active_goal="converse with the user", user_input=_QUESTION, own_capture_id=own)
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


@pytest.mark.parametrize("typed", ["My name is Denny.", "My name is Denny.\n", "  My name is Denny.  "])
def test_an_arrival_reaches_memory_agent_even_with_a_capture_manager_and_is_not_republished(typed: str) -> None:
    """The robot builds ``PerceptionAgent`` with a CaptureManager, so ``propose_intent`` returns before publishing;
    the user's words now arrive regardless. The runtime clears the slot (whitespace and all, executor review 4), so a
    later §3 never publishes them again (#1414: consumed once)."""
    from maxim.agents.bus import AgentBus, Percept
    from maxim.agents.perception_agent import PerceptionAgent
    from maxim.runtime.loop_perception import take_user_input
    from maxim.runtime.state import RuntimeState

    bus = AgentBus()
    heard: list[Any] = []
    bus.subscribe(Percept, heard.append)
    agent = PerceptionAgent(bus=bus)
    agent._capture_manager = object()  # the robot's vision/audio path
    state = RuntimeState()
    state.data["cli_input"] = typed  # take_observation merged the observation into the state
    take_user_input({"cli_input": typed}, state, agent)
    agent._capture_manager = None
    agent.propose_intent(state, None)
    assert [p.cli_input for p in heard if p.cli_input] == ["My name is Denny."]


def test_a_reply_the_loop_makes_itself_is_never_memorized_and_a_persons_waits() -> None:
    """Owner decision 2026-10-25: the auto-approve "yes", a sim-resolved confirmation and a timeout retry are the
    loop answering itself (``LOOP_REPLY_SLOT``): processed, never published. A person's own "yes" is memorized."""
    from maxim.runtime.loop_perception import LOOP_REPLY_SLOT, take_user_input
    from maxim.runtime.state import RuntimeState

    state, agent = RuntimeState(), _Recorder()
    state.data[LOOP_REPLY_SLOT] = "yes"
    state.data["pending_cli_input"] = "what about the lighthouse?"
    first = take_user_input({}, state, agent)
    assert first.cli_input == "yes" and first.addressed == "" and agent.calls == []
    second = take_user_input({}, state, agent)  # the person's input waited a tick, untouched
    assert second.cli_input == "what about the lighthouse?" and agent.calls == [("what about the lighthouse?", "cli")]
    take_user_input({}, _with(state, "pending_cli_input", "yes"), agent)
    assert agent.calls[-1] == ("yes", "cli")


def _with(state: Any, key: str, value: Any) -> Any:
    state.data[key] = value
    return state


def test_a_world_event_on_the_cli_field_is_not_published_as_a_person() -> None:
    """``sim_adapter`` copies a world event's content into ``cli_input``; its source says it is not addressed."""
    from maxim.runtime.loop_perception import take_user_input
    from maxim.runtime.state import RuntimeState

    agent = _Recorder()
    got = take_user_input({"cli_input": "a creeper hisses", "source": "minecraft"}, RuntimeState(), agent)
    assert got.cli_input == "a creeper hisses" and got.addressed == "" and agent.calls == []


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
        ({"cli_input": "hi", "source": "voice"}, True),
        ({"cli_input": "a creak in the dark", "source": "minecraft"}, False),
        ({"source": "cli"}, False),
    ],
)
def test_addressed_input_is_read_from_its_source(observation: dict, addressed: bool) -> None:
    from maxim.runtime.loop_perception import take_user_input
    from maxim.runtime.state import RuntimeState

    assert bool(take_user_input(observation, RuntimeState(), _Recorder()).addressed) is addressed


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


# ── design review round 1 folds ─────────────────────────────────────────────────────────────────────────────────


def test_normalize_tokens_is_frozen() -> None:
    """ATL concept names are persisted from ``normalize_tokens``: its quirks are part of the stored data and change
    only with a migration (the bio-memory brief's one-standard invariant)."""
    from maxim.memory.text import normalize_tokens

    golden = {
        "What is Denny's favorite color?": ["denny", "favorite", "color"],
        "My name is Denny and my favourite colour is green.": ["name", "denny", "favourite", "colour", "green"],
        "navigate_to_kitchen": ["navigate", "kitchen"],
        "grasping the mugs": ["grasp", "mug"],
        "used status of batteries": ["us", "statu", "battery"],
        "Al saw a TV": ["saw"],
        "carried the settings": ["carry", "setting"],
    }
    assert {text: normalize_tokens(text) for text in golden} == golden


def test_keys_are_idempotent_so_a_store_keys_a_keyed_word_the_same() -> None:
    from maxim.memory.text import key_tokens, settle_key

    for word in ("settings", "buildings", "meetings", "closing", "parsed", "used", "Dennis"):
        once = key_tokens(word)
        assert [settle_key(t) for t in once] == once, word
        assert key_tokens(" ".join(t for t in once if len(t) >= 3)) == [t for t in once if len(t) >= 3], word


@pytest.mark.parametrize(("said", "stored"), [("the settings", "settings"), ("a meeting", "meetings")])
def test_every_store_matches_an_inflection_against_its_stored_form(tmp_path: Path, said: str, stored: str) -> None:
    from maxim.decisions.causal_link import Valence
    from maxim.decisions.nac import NAc
    from maxim.memory.atl import ATL
    from maxim.memory.semantic_types import SemanticMemory

    nac = NAc()
    for _ in range(10):
        nac.observe(
            event_type="tool",
            event_signature=f"tool:open_{stored}",
            outcome_type="result",
            outcome_signature="success",
            outcome_valence=Valence.POSITIVE,
            delta_seconds=1.0,
        )
    from maxim.integration.bio_enrichment import BioEnrichmentPipeline

    keywords = BioEnrichmentPipeline()._extract_keywords(said)
    assert [lk.event_signature for lk in nac.scan_links_for_keywords(keywords)] == [f"tool:open_{stored}"]

    atl = ATL()
    atl.store(SemanticMemory(id="c1", timestamp=time.time(), name=stored, category="object"))
    assert [c.id for c in atl.recall(name_token=keywords[0])] == ["c1"]

    h = _hippocampus(tmp_path)
    mid = _say(h, f"The {stored} are on Monday.")
    assert [m.id for m in h.recall(query=said, limit=3)] == [mid]


def test_a_compressed_substrate_concept_is_never_token_indexed_after_a_reload(tmp_path: Path) -> None:
    """A ``CompressedSemantic`` drops ``substrate_modality``; the guard keys on the category compression keeps."""
    from maxim.memory.atl import ATL
    from maxim.memory.semantic_types import CompressedSemantic, SemanticMemory

    raw = SemanticMemory(id="s1", timestamp=time.time(), name=_FACT, category="substrate", substrate_modality="text")
    atl = ATL()
    atl.store(CompressedSemantic.from_semantic(raw))
    reloaded = ATL()
    reloaded.load_state(atl.dump())
    assert reloaded.recall(name_token="green") == [] and atl.recall(name_token="green") == []


def test_a_deliberation_cycle_never_recalls_the_turns_own_words(tmp_path: Path) -> None:
    """A deliberation cycle enriches the model's reasoning, with no ``user_input`` of its own: it inherits the turn's,
    so the just-stored capture is not rendered as "earlier input" (design review SF4)."""
    from maxim.integration.bio_enrichment import BioEnrichmentPipeline, EnrichmentContext

    h = _hippocampus(tmp_path)
    fact = _say(h, _FACT)
    turn = "Remember that my lighthouse is painted red."
    own = _say(h, turn)
    pipe = BioEnrichmentPipeline(hippocampus=h)
    pipe._query_hippocampus_traced(turn, [], context=EnrichmentContext(user_input=turn, own_capture_id=own))
    found, *_ = pipe._query_hippocampus_traced(
        "The user says their lighthouse is painted red and Denny likes green.", [], context=EnrichmentContext()
    )
    assert [s.memory_id for s in found] == [fact]


# ── #1573: the turn's own records are excluded by identity, and an episode stored many times takes one place ─────


def _ask(h: Any, question: str, own: str = "", goal: str = "converse with the user") -> tuple[list[str], list[str]]:
    from maxim.integration.bio_enrichment import BioEnrichmentPipeline, EnrichmentContext

    ctx = EnrichmentContext(active_goal=goal, user_input=question, own_capture_id=own)
    found, paths, *_ = BioEnrichmentPipeline(hippocampus=h)._query_hippocampus_traced(question, [], context=ctx)
    return [s.memory_id for s in found], paths


def _did(h: Any, heard: str, tool: str, *, ok: bool, result: dict[str, Any] | None = None) -> str:
    """An action taken on an observation that holds ``heard``, captured as the loop captures one."""
    mid = h.capture_from_loop(
        observation={"cli_input": heard, "salience": 0.9, "novelty": 0.5},
        state={"source": "loop"},
        intent={},
        decision={},
        action={"tool": tool},
        result={"success": ok, **(result or {})},
        encoding=EncodingSignals.unmeasured("memory_agent"),
        situation=None,
    )
    time.sleep(0.002)
    assert mid is not None
    return mid


def test_an_earlier_identical_ask_is_recalled_and_the_turns_own_capture_is_not(tmp_path: Path) -> None:
    h = _hippocampus(tmp_path)
    earlier = _say(h, _QUESTION)
    own = _say(h, _QUESTION)
    ids, paths = _ask(h, _QUESTION, own)
    assert ids == [earlier] and paths == ["cue"]


@pytest.mark.parametrize("repeats", [5, 6, 40])
def test_a_repeated_question_does_not_lose_its_answer(tmp_path: Path, repeats: int) -> None:
    h = _hippocampus(tmp_path)
    fact = _say(h, _FACT)
    asks = [_say(h, _QUESTION) for _ in range(repeats)]
    ids, _paths = _ask(h, _QUESTION, asks[-1])
    assert fact in ids
    assert asks[-1] not in ids and len(set(ids) & set(asks)) == 1


def test_a_turn_that_made_no_capture_excludes_nothing(tmp_path: Path) -> None:
    """A real earlier episode is never taken for the turn's own record because the words match."""
    h = _hippocampus(tmp_path)
    earlier = _say(h, _QUESTION)
    assert _ask(h, _QUESTION)[0] == [earlier]


def test_a_later_record_of_the_turn_does_not_echo_its_arrival_on_a_deliberation_cycle(tmp_path: Path) -> None:
    from maxim.integration.bio_enrichment import BioEnrichmentPipeline, EnrichmentContext

    h = _hippocampus(tmp_path)
    earlier = _say(h, _QUESTION)
    own = _say(h, _QUESTION)
    pipe = BioEnrichmentPipeline(hippocampus=h)
    pipe._query_hippocampus_traced(_QUESTION, [], context=EnrichmentContext(user_input=_QUESTION, own_capture_id=own))
    action = _did(h, _QUESTION, "look_around", ok=True)  # the action taken on the tick's observation holds the words
    found, _paths, _h, _holes, own_ids = pipe._query_hippocampus_traced(_QUESTION, [], context=EnrichmentContext())
    assert own_ids == [own, action] and [s.memory_id for s in found] == [earlier]


def test_the_next_turn_makes_the_last_turns_capture_an_earlier_episode(tmp_path: Path) -> None:
    from maxim.integration.bio_enrichment import BioEnrichmentPipeline, EnrichmentContext

    h = _hippocampus(tmp_path)
    first = _say(h, _QUESTION)
    pipe = BioEnrichmentPipeline(hippocampus=h)
    pipe._query_hippocampus_traced(_QUESTION, [], context=EnrichmentContext(user_input=_QUESTION, own_capture_id=first))
    second = _say(h, _QUESTION)
    found, *_ = pipe._query_hippocampus_traced(
        _QUESTION, [], context=EnrichmentContext(user_input=_QUESTION, own_capture_id=second)
    )
    assert [s.memory_id for s in found] == [first]


def test_the_same_line_with_different_outcomes_is_two_episodes(tmp_path: Path) -> None:
    """One slot per LINE would hide the failure, the record an agent learns from."""
    h = _hippocampus(tmp_path)
    burned = _did(h, "touch the stove", "touch", ok=False, result={"error": "burned"})
    fine = _did(h, "touch the stove", "touch", ok=True)
    ids, _paths = _ask(h, "is the stove safe")
    assert {burned, fine} <= set(ids)


def test_a_record_holding_the_answer_outranks_collapse_with_its_bare_question(tmp_path: Path) -> None:
    h = _hippocampus(tmp_path)
    answered = _did(h, "what is the door code", "lookup", ok=True, result={"output": "4471"})
    bare = _say(h, "what is the door code")
    own = _say(h, "what is the door code")
    ids, _paths = _ask(h, "what is the door code", own)
    assert set(ids) == {answered, bare}


def test_true_copies_on_the_goal_path_take_one_place_and_the_cap_still_fills(tmp_path: Path) -> None:
    h = _hippocampus(tmp_path)
    others = [_say(h, f"The lighthouse keeper rests near the cape, tale {n}.") for n in range(4)]
    copies = [_say(h, "The lighthouse keeper walks the cape every morning.") for _ in range(6)]
    ids, paths = _ask(h, "zzz qqq", goal="lighthouse keeper on the cape")
    assert len(ids) == 3 and paths == ["goal"] * 3
    assert len(set(ids) & set(copies)) == 1 and len(set(ids) & set(others)) == 2


def test_the_own_records_take_no_place_on_the_goal_path(tmp_path: Path) -> None:
    """A deliberation cycle's text cues nothing, so the goal path runs with the turn's own capture newest of all."""
    from maxim.integration.bio_enrichment import BioEnrichmentPipeline, EnrichmentContext

    h = _hippocampus(tmp_path)
    notes = [_say(h, f"The cape lighthouse note {n}.") for n in range(3)]
    turn = "The cape lighthouse note nine."
    own = _say(h, turn)
    pipe = BioEnrichmentPipeline(hippocampus=h)
    pipe._turn_input, pipe._turn_capture_id = turn, own
    ctx = EnrichmentContext(active_goal="cape lighthouse note")
    found, paths, *_ = pipe._query_hippocampus_traced("zzz qqq", [], context=ctx)
    assert {s.memory_id for s in found} == set(notes) and paths == ["goal"] * 3


@pytest.mark.parametrize("repeats", [2, 5, 8])
def test_the_substring_path_does_not_lose_the_answer_either(tmp_path: Path, repeats: int) -> None:
    """A line with no content key never reaches the ranker: the same defect had a second home."""
    h = _hippocampus(tmp_path)
    fact = _say(h, "yes the code is 1234")
    asks = [_say(h, "yes") for _ in range(repeats)]
    ids, paths = _ask(h, "yes", asks[-1], goal="")
    assert fact in ids and asks[-1] not in ids and set(paths) == {"substring"}
    assert len(set(ids) & set(asks)) == (1 if repeats > 1 else 0)


def test_a_failed_own_read_loses_at_most_an_echo_never_the_recall(tmp_path: Path) -> None:
    h = _hippocampus(tmp_path)
    fact = _say(h, _FACT)
    own = _say(h, _QUESTION)

    def broken(*_args: Any) -> tuple[str, ...]:
        raise RuntimeError("store read failed")

    h.turn_records = broken
    assert _ask(h, _QUESTION, own)[0] == [fact]


def test_one_damaged_record_does_not_empty_recall(tmp_path: Path) -> None:
    from maxim.memory.types import EpisodicMemory, Perception

    h = _hippocampus(tmp_path)
    fact = _say(h, _FACT)
    own = _say(h, _QUESTION)
    h._memories["bad"] = EpisodicMemory(id="bad", timestamp=1.0, perception=Perception(observations=["x"]))
    assert h._memories["bad"].input_text == "" and _ask(h, _QUESTION, own)[0] == [fact]


def test_a_record_with_no_heard_words_is_never_collapsed() -> None:
    from maxim.memory.hippocampus_retrieval import _rank_by_relevance
    from maxim.memory.types import Action, EpisodicMemory, Perception

    silent = [
        EpisodicMemory(id=f"m{i}", timestamp=float(i), perception=Perception(), action=Action(tool_name="brass_key"))
        for i in range(3)
    ]
    assert [m.id for m in _rank_by_relevance(list(silent), "brass key", 5, collapse_copies=True)] == ["m2", "m1", "m0"]
    assert _rank_by_relevance(list(silent), "brass key", 0, collapse_copies=True) == []


def test_recall_and_search_keep_every_copy_unless_asked(tmp_path: Path) -> None:
    """The store keeps every copy, and only the bounded views that ask collapse them."""
    h = _hippocampus(tmp_path)
    ids = [_say(h, _QUESTION) for _ in range(3)]
    assert {m.id for m in h.recall(query=_QUESTION, limit=5)} == set(ids)
    assert {m.id for m in h.search_by_content("favorite color", limit=5)} == set(ids)


def test_the_models_own_recall_tool_collapses_copies(tmp_path: Path) -> None:
    from maxim.tools.introspection import MemoryRecallTool

    h = _hippocampus(tmp_path)
    fact = _say(h, _FACT)
    for _ in range(6):
        _say(h, _QUESTION)
    result = MemoryRecallTool(hippocampus=h).execute(query=_QUESTION, limit=2)
    assert result.success and fact in str(result.output)


def test_arrival_returns_the_record_the_real_memory_agent_made_of_it(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from maxim.agents.bus import AgentBus
    from maxim.agents.memory_agent import MemoryAgent
    from maxim.agents.perception_agent import PerceptionAgent

    h = _hippocampus(tmp_path)
    bus = AgentBus()
    memory = MemoryAgent(bus)
    memory.connect_hippocampus(h)
    memory.on_start()
    capture_id = PerceptionAgent(bus).publish_arrival(SimpleNamespace(data={}), _QUESTION, source="cli")
    assert capture_id in h._memories and h._memories[capture_id].input_text == _QUESTION


@pytest.mark.parametrize(
    "other",
    [
        {"tool": "other_tool"},
        {"ok": False},
        {"result": {"output": "4471"}},
    ],
    ids=["tool", "success", "outcome_text"],
)
def test_a_record_differing_in_one_part_of_the_episode_is_no_copy(tmp_path: Path, other: dict[str, Any]) -> None:
    h = _hippocampus(tmp_path)
    base = {"tool": "lookup", "ok": True, "result": None}
    one = _did(h, "what is the door code", base["tool"], ok=base["ok"])
    two = _did(h, "what is the door code", **{**base, **other})
    copy = _did(h, "what is the door code", base["tool"], ok=base["ok"])
    assert h._memories[one].copy_key() == h._memories[copy].copy_key()
    assert h._memories[one].copy_key() != h._memories[two].copy_key()


def test_a_number_in_an_outcome_keeps_two_episodes_apart(tmp_path: Path) -> None:
    """A search key drops a number; a comparison cannot: 300 degrees and 30 degrees are different outcomes."""
    h = _hippocampus(tmp_path)
    hot = _did(h, "how hot is the stove", "read_sensor", ok=True, result={"output": {"temp_c": 300}})
    cool = _did(h, "how hot is the stove", "read_sensor", ok=True, result={"output": {"temp_c": 30}})
    assert {m.id for m in h.recall(query="stove", limit=5, collapse_copies=True)} == {hot, cool}


def test_a_record_that_cannot_be_compared_is_kept_and_recall_survives(tmp_path: Path) -> None:
    from maxim.memory.hippocampus_retrieval import distinct_copies

    class _Broken:
        def __repr__(self) -> str:
            raise RuntimeError("no repr")

    h = _hippocampus(tmp_path)
    fact = _say(h, _FACT)
    h._memories[fact].outcome.result = {"output": _Broken()}
    assert distinct_copies([h._memories[fact]]) == [h._memories[fact]]


def test_a_damaged_compressed_record_does_not_empty_recall(tmp_path: Path) -> None:
    """A stored ``heard`` that is not text loads without complaint, and ``copy_key`` reads every ranked record."""
    from maxim.memory.types import CompressedMemory

    h = _hippocampus(tmp_path)
    fact = _say(h, _FACT)
    own = _say(h, _QUESTION)
    h._memories["bad"] = CompressedMemory(id="bad", timestamp=1.0, goal="Denny favorite color", heard=["x"])
    assert fact in _ask(h, _QUESTION, own)[0]


def test_recall_collapses_copies_with_no_query_too(tmp_path: Path) -> None:
    """The flag must not be silently ignored: ``memory_recall`` passes it with or without a query."""
    h = _hippocampus(tmp_path)
    older = _say(h, _FACT)
    copies = [_say(h, _QUESTION) for _ in range(6)]
    assert {m.id for m in h.recall(limit=5, collapse_copies=True)} == {older, copies[-1]}
    assert len(h.recall(limit=5)) == 5


def test_a_malformed_tool_name_does_not_empty_recall(tmp_path: Path) -> None:
    h = _hippocampus(tmp_path)
    fact = _say(h, _FACT)
    h._memories[fact].action.tool_name = ["weird"]
    own = _say(h, _QUESTION)
    assert _ask(h, _QUESTION, own)[0] == [fact]


def test_the_graph_path_excludes_the_turns_own_records() -> None:
    from types import SimpleNamespace

    from maxim.integration.bio_enrichment import BioEnrichmentPipeline, EnrichmentContext

    seen: list[Any] = []

    class _Store:
        _memories: dict[str, Any] = {}

        def stored_capture_seqs(self) -> tuple[int, list[list[int]]]:
            return 0, []

        def turn_records(self, capture_id: str, text: str) -> tuple[str, ...]:
            return (capture_id,)

        def retrieve_on_cue(self, node_id: str, limit: int, multi_hop: bool) -> list[tuple[str, float]]:
            return [("n1", 0.9)]

        def recall(self, **kw: Any) -> list[Any]:
            if "object_detected" in kw:
                seen.append(kw.get("exclude_ids"))
            return []

        def search_by_content(self, *_a: Any, **_k: Any) -> list[Any]:
            return []

    pipe = BioEnrichmentPipeline(
        hippocampus=_Store(),
        ec=SimpleNamespace(
            pattern_complete_readonly=lambda *a, **k: SimpleNamespace(is_new=False, similarity=0.9, node_id="n1")
        ),
        encoder=SimpleNamespace(embed=lambda t: [1.0], geometry_for=lambda *a: None),
        atl=SimpleNamespace(get=lambda n: SimpleNamespace(name="kettle")),
    )
    pipe._query_hippocampus_traced("q", [], context=EnrichmentContext(user_input="q", own_capture_id="own"))
    assert seen == [("own",)]


def test_a_turn_records_id_not_yet_stored_is_still_the_turns_own(tmp_path: Path) -> None:
    h = _hippocampus(tmp_path)
    assert h.turn_records("in-flight", _QUESTION) == ("in-flight",)


def test_arrival_returns_the_record_memory_made_of_it(tmp_path: Path) -> None:
    from maxim.agents.bus import AgentBus, Percept
    from maxim.runtime.loop_perception import percept_enrichment_context, take_user_input

    class _Agent:
        def publish_arrival(self, state: Any, text: str, *, source: str) -> str:
            percept = Percept(timestamp=1.0, source=source, cli_input=text)
            bus = AgentBus()
            bus.subscribe(Percept, lambda p: setattr(p, "capture_id", "the-record"))
            bus.publish(percept)
            return percept.capture_id or ""

    class _State:
        data: dict[str, Any] = {"pending_cli_input": _QUESTION, "active_goal": "g"}

    state = _State()
    got = take_user_input({}, state, _Agent())
    assert got.addressed == _QUESTION and got.own_capture_id == "the-record"
    ctx = percept_enrichment_context(state, None, got)
    assert (ctx.user_input, ctx.own_capture_id, ctx.active_goal) == (_QUESTION, "the-record", "g")


def test_a_records_content_tokens_follow_a_replaced_field() -> None:
    from maxim.memory.types import EpisodicMemory, Perception

    mem = EpisodicMemory(id="m", timestamp=1.0, perception=Perception(cli_input="the brass key"))
    assert "brass" in mem.content_tokens()
    mem.perception = Perception(cli_input="a silver spoon")  # a staged copy replaces a field object
    assert "silver" in mem.content_tokens() and "brass" not in mem.content_tokens()


# ── delta round folds ────────────────────────────────────────────────────────────────────────────────────────────


def test_a_reflection_and_any_stored_result_text_stay_searchable(tmp_path: Path) -> None:
    """The one text-field list is the SUPERSET (delta review B2): a Reflexion reflection and a ``stdout`` are text the
    record holds, found by the substring search and ``matching_sentence`` as by the ranker."""
    h = _hippocampus(tmp_path)
    mid = h.capture_from_loop(
        observation={},
        state={},
        intent={"goal": "learn_from_failure"},
        decision={},
        action={},
        result={"reflection": "Next time quote the glob pattern.", "stdout": "kitten.txt"},
        encoding=EncodingSignals.unmeasured("loop"),
        situation=None,
    )
    for word in ("glob", "kitten"):
        assert [m.id for m in h.search_by_content(word)] == [mid], word
    [record] = h.recall_by_ids([mid])
    assert h.matching_sentence(record, "glob") == "Next time quote the glob pattern."
    assert [m.id for m in h.recall(query="the glob", limit=3)] == [mid]


def test_a_persons_input_that_waited_behind_a_loop_reply_survives_the_confirmation() -> None:
    """Delta review S1: the controller's confirmation path popped the input slots after ``take_user_input`` had read
    them, dropping a person's input that was waiting for the next tick behind the loop's own "yes"."""
    from maxim.runtime.loop_perception import take_user_input
    from maxim.runtime.state import RuntimeState

    state, agent = RuntimeState(), _Recorder()
    state.data["pending_loop_reply"] = "yes"
    state.data["pending_cli_input"] = "and also, my cat is called Miso"
    assert take_user_input({}, state, agent).cli_input == "yes"  # this tick: the loop answers its own confirmation
    _confirm_and_finish(state)
    assert take_user_input({}, state, agent).cli_input == "and also, my cat is called Miso"
    assert agent.calls == [("and also, my cat is called Miso", "cli")]


def _confirm_and_finish(state: Any) -> None:
    """``LoopController.handle_confirmation`` on a pending confirmation the loop's "yes" answers, dispatch stubbed."""
    from types import SimpleNamespace

    from maxim.runtime.loop_controller import LoopController

    ctrl = LoopController.__new__(LoopController)
    ctrl.state = state
    ctrl.pending_confirmation = SimpleNamespace(tool_name="t", action={"tool_name": "t", "params": {}})
    ctrl.pending_proposal = None
    ctrl.clear_pending_user_input = lambda: None
    try:
        ctrl.handle_confirmation(
            "yes",
            execute_and_learn=lambda **k: (_ for _ in ()).throw(RuntimeError("stub")),
            book_refusal=lambda **k: None,
            observation={},
        )
    except Exception:  # the stub raises out of the dispatch; the finally under test still ran
        pass


def test_an_observation_with_a_transcript_and_a_cli_input_publishes_the_cli_input() -> None:
    """Delta review S3: the loop processes ``cli_input``, so memory hears ``cli_input`` (and §3 then finds its copy
    cleared), never the transcript beside it."""
    from maxim.runtime.loop_perception import take_user_input
    from maxim.runtime.state import RuntimeState

    agent, state = _Recorder(), RuntimeState()
    state.data["cli_input"] = "the typed words"
    got = take_user_input(
        {"cli_input": "the typed words", "transcript": "a heard chunk", "source": "cli"}, state, agent
    )
    assert agent.calls == [("the typed words", "cli")] and got.addressed == "the typed words"
    assert "cli_input" not in state.data

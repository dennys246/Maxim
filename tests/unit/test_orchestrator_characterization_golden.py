"""Orchestrator characterization golden: the gate for the ``start_simulation_mode`` decomposition (slice 0).

WHY THIS EXISTS. ``docs/plans/roadmap_1_3_x.md`` §"Coverage first, then extract" says the orchestrator's
first decomposition slice is its characterization pass, built before any code moves. Until this file,
only ``test_api_expansion.py`` got past the function's guards, and it stops at the AUT registry build.
No test ran the narrator loop, the stdin reader, teardown, the report or the ``SimulationResult``.
Every later extraction slice must keep this file green UNCHANGED.

HOW IT RUNS. ``tests/unit/_orch_harness.py`` (read its docstring) runs the REAL function offline:

  - a fake LLM, ``FakeRouter``, patched in at ``build_primary_router``, which sorts prompts by marker
    and fails closed;
  - the real threads, on the wall clock, with the real 3 s settle;
  - class-level hooks on ``SimulationBridge`` for causal synchronisation (and, for the arms added after slice 0,
    on ``ConversationalSource.inject_cli`` and a few module attributes: the harness docstring lists them).

The slice-0 arms: ``precampaign``, ``narrator`` (with the AUT on its own router build) and ``stdin``. Then the
arms the 1.4.0 gate requires (``scripts/release_gates.json`` ``arms``): ``generative``, ``stall``, ``new_goal``,
``persistent_agent`` and ``imagination`` (the harness docstring lists each arm's script).

WHAT IS RECORDED. Only causally ordered facts:

  - each role's LLM calls (narrator, AUT) in that role's order, with what was asked and the scripted
    answer;
  - every bridge turn: the percept text, the AUT's actions with their args and outputs, and the
    turn's ``response``;
  - the operator lines and the turn each landed after;
  - the ``finish_context`` each ``bridge.finish()`` saw;
  - the ``display_status``/``display_summary`` lifecycle lines;
  - the ``SimulationResult`` (timestamps and the live introspector dropped);
  - the deterministic ``report.json`` fields, provenance by VALUE (the keys that name the checkout and
    the interpreter are masked by name, ``_PROVENANCE_MASKED``);
  - the report writers called, in order (``save_report`` twice per run: before and after the roundup), with
    what each was handed (the report's session, the bridge, the AUT stores and which of them hold
    which, the router's build, the paths), and the ``sim_reports`` file set;
  - the whole ``MAXIM_DATA_HOME`` file set: the reports, the sandbox and the orchestrator's own
    persisted stores, which only its ``shutdown()`` writes;
  - the saved AUT NAc's causal links and every ``actions.jsonl`` record, times dropped; its tick-decayed
    bias maps by key, sign and bound only, because how many decay ticks land before the save is
    wall-clock-paced (``_NAC_DECAY_PINNED``), and every NAc credit write behind them, with its arguments,
    in each NAc's call order (``nac_credits``; each key's first write carries ``post``). The saved maps are
    also checked to sit strictly below each key's last write, so per-tick decay must reach the saved NAc.
    Measured 2026-10-17: the AUT NAc's writes (``credit_goal``, ``update_eligibility``) all run on ``sim.aut``,
    the narrator NAc's ``update_eligibility`` on the calling thread (``MainThread``);
  - any ``sim.*`` thread still alive after the run, and every thread the run started that outlives it;
  - for the arms added after slice 0 only (so the slice-0 traces stay byte-identical): ``events`` (every
    ``inject_cli`` with its source and thread, and every campaign-runner finish with the stop state), the
    narrator follow-ups' ``held_inputs`` (stall), ``post_sim_reads`` (new_goal), ``handle`` (the adopted agent's
    registry and pain bus before, during and after: persistent_agent) and ``imagination`` (the trigger's calls
    and its teardown).

CANONICAL FORM. ``json.dumps(sort_keys=True)``. Floats are rounded to 9 decimals, and paths and stamps
become placeholders. Every value the harness excludes or masks is a named constant there, with the
measurement behind it.

REGENERATION RULE (as for ``agent_loop_selection_golden_v1.json``). The fixture is regenerated ONLY from
the pre-change commit, never by pasting a slice's output. The command is
``python tests/unit/test_orchestrator_characterization_golden.py --regen``. It refuses a dirty ``src/``
and records ``generated_at_commit``. An intentional behaviour change regenerates in its own commit,
with the diff justified in the PR.

SRC BEHAVIOUR THIS GOLDEN PINS AS-IS (findings from building it, 2026-10-12, and from the arms added
2026-10-24; none fixed here). A fix regenerates the fixture in its own commit:

  - #1259: ``finish_simulation`` does not end the run. The tool finishes the bridge and the narrator source,
    but nothing sets ``stop_event``. The narrator loop has no ``action_sink``, so
    ``SimAdapter.check_exhaustion`` never tightens its 180 s grace, and the stall detector keeps
    nudging the finished narrator every 30 s until that grace expires. A probe saw six post-finish
    narrator calls over about 180 s. The narrator arm therefore follows the finish with an operator
    ``/cancel`` (harness docstring).
  - #1260 (FIXED): ``TOOL_DESCRIPTIONS.update`` in §5 was process-global, so the AUT's prompt advertised
    ``respond: NOT AVAILABLE. Use send_message instead.``, and the AUT's ``respond`` got a ``process``
    follow-up, the ``say`` every turn pinned. The narrator's entries now ride on its registry
    (``SimToolRegistry.tool_descriptions``): the AUT's ``respond`` is terminal, one action a turn, and the
    harness asserts a run leaves ``TOOL_DESCRIPTIONS`` untouched. The fixture was regenerated for it, after
    #1261's, in its own commit.
  - #1261 (FIXED, no longer a pinned defect): the bridge's ``response`` was the ``respond`` tool's RECEIPT
    (``{'delivered': True, 'mode': 'cli'}``), not the AUT's words. Each turn's ``response`` and
    ``report.response_texts`` now pin the words, read from ``ActionRecord.utterance``, and include ``say``'s;
    each ``SimulationResult.actions`` entry carries them as ``utterance`` (the fixture was regenerated for it
    in its own commit). The actions' ``output`` still pins the receipt, which ``RespondTool`` still returns.
  - #1262: ``/pause`` does not stop the narrator mid follow-up chain. In the stdin arm the operator pauses after
    turn 1 and the narrator still sends turn 2. The pause reaches it only as a queued SYSTEM percept,
    and its loop answers the pending ``send_message`` follow-up first, so the pause, resume and guidance
    percepts are never planned before the ``/cancel``.
  - #1259, the stall arm's half: after ``finish_simulation`` the detector's idle nudge reaches the narrator,
    which plans on it, and each later ``finish_simulation`` overwrites ``finish_context`` (the report's
    ``llm_finish_reason`` is the last answer's). Since D13 that is the only way a live narrator reaches the
    idle threshold (#1452), so #1259's fix removes the stall arm's idle path and must re-home the arm.
  - #1453: the ``/new`` re-entry drops the caller's arguments (``sandbox_backend`` becomes ``"auto"``).
  - #1454: on ``/new`` the AUT's learning does not carry over (a fresh ``sim_agent_*`` home per run).
  - #1457: on ``/new``, ``--report-json`` is the first run's report while the caller gets the second run's
    result (the new_goal arm's ``report_writes`` hold ``emit_report_json`` once, in the outer run).
  - #1263: teardown leaves threads running: the two agents' ``plan-logger``/``plan-dashboard`` writers
    and the ``memctx_*`` pool. (Its DefaultNetwork part is FIXED with #1281: the loop that starts the AUT's
    DN now stops it in a sim too, so the ``DefaultNetwork`` and ``DN-BackgroundTasks`` threads end with the
    run, ``test_the_run_leaves_no_default_network_running_1263``; the fixture was regenerated for it with
    #1260's.) Each run in one process (the menu loop,
    ``maxim.campaign()``) leaves them alive. ``threads_left_alive`` pins the ones the run adds; the
    ``memctx`` pool is class-level and process-wide, so whether a run starts it depends on the process's
    history, and it is not pinned at all (``_THREADS_EXCLUDED``, #1263's memctx part). The ``sim.*`` threads all exit.

NEUTRALISED, NOT PINNED (the harness takes them out of the trace, so a fix does not change the golden):

  - #1264: the AUT and narrator loops race on one CWD-relative ``state_<second>.json`` and a shared
    ``.tmp``. How many state files land, under which second, is the race itself, so the data-home file
    set excludes ``agents/*/runtime/`` (``_DATA_HOME_EXCLUDED``).
  - #1265: the report's ``lane_breakdown`` and latency percentiles read the process-global lane
    ``MetricsRegistry``, never session-scoped. The harness hands each run a fresh registry
    (``_GlobalState``), so an earlier sim's (or test's) metrics cannot leak in; a run reading only its
    own is what a fix would produce, which is why it is not pinned as the leak.
  - #1451: a narrator percept held behind an in-flight planning job rides only the next FOLLOW-UP; when that
    job ends the chain it waits for a later percept, and a second such percept races. The stall arm gives
    its checkpoint an ``observe_actions`` follow-up to ride, so the idle nudge is the only post-finish percept.
  - #1455: ``extract_entity_phrases`` merges short noun phrases across the scene manifest's newline join; the
    imagination arm's manifest is sentence-form.
  - The box: every run gets the conftest model environment (offline, an empty cache: ``_model_isolation_env``)
    and ``check_docker_available`` is patched to False (the /new re-entry resolves ``"auto"``). Wall-clock
    text is masked by name (``_LIFECYCLE_MASKS``, ``_STALL_MASKS``); ``scenario_sha256`` is recomputed, not
    pinned (``_PROVENANCE_RECOMPUTED``); shipped-data inputs are pinned by path for the files an arm is about
    and collapsed otherwise (``_SHIPPED_DATA``).

OPEN FLAKE: #1456 (the narrator arm's report read ``Learn: 8`` once in one of three full-file runs; three
structural causes ruled out, not reproduced since). Capture the whole trace when it recurs.

PROVEN ABLE TO FAIL: the deletion probes on ``src/maxim/simulation/orchestrator.py``
(``python -B``, 2026-10-12) are recorded in the slice-0 PR. The review fold (2026-10-13) added probes,
each red: ``_sim_provenance`` handed ``goal[:5]``, the AUT and narrator workers swapped, and
``aut_worker=None``; the AUT build's ``MAXIM_LLM_PROFILE`` override deleted, and its restore deleted; ``_orch_instance.shutdown()`` dropped; a teardown that stops the AUT DefaultNetwork
(a simulated #1263 fix); ``save_aut_state`` handed the narrator's NAc. The arms added 2026-10-24 have their
own, each red (``python -B``, recorded in their PR): the generative runner's ``_finish_runner`` deleted; the
diversity checkpoint injection deleted; the idle nudge injection deleted; the ``/new`` re-entry deleted; HANDLE
branch 4's lease restore deleted, and its DN pain-bridge unsubscribe deleted; HANDLE branch 3's skip of the
session snapshot deleted; the imagination teardown's ``clear_ephemeral`` deleted, and its tag/decay deleted.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

if str(Path(__file__).resolve().parents[2]) not in sys.path:  # run as a script (--regen / --emit)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.unit._orch_harness import (  # noqa: E402
    _MASKED,
    _PROVENANCE_MASKED,
    _RECOMPUTED_OK,
    _STORY_SCENES,
    ARMS,
    GOAL,
    NEW_GOAL,
    REPO_ROOT,
    RUN_SIDE,
    FakeRouter,
    WrongCheckout,
    _decayed_bound,
    assert_this_checkout,
    canonical_json,
    run_arm,
)

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "orchestrator_characterization_golden_v1.json"
_TRACES: dict[str, dict[str, Any]] = {}


def _trace(arm: str, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One run per arm per test session: the golden test and the direct assertions read the same run."""
    if arm not in _TRACES:
        trace = run_arm(arm, tmp_path_factory.mktemp(f"orch_{arm}"))
        trace.pop("_elapsed_s")
        _TRACES[arm] = json.loads(canonical_json(trace))
    return _TRACES[arm]


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


# ── the gate ──────────────────────────────────────────────────────────────


@pytest.mark.timeout(240)
@pytest.mark.parametrize("arm", sorted(ARMS))
def test_orchestrator_run_matches_the_golden(arm: str, tmp_path_factory: pytest.TempPathFactory) -> None:
    """THE gate: the real run's causal trace equals the pinned one."""
    assert_this_checkout()  # a worktree without PYTHONPATH would otherwise test main's code
    got = _trace(arm, tmp_path_factory)
    want = _fixture()["arms"][arm]
    assert canonical_json(got) == canonical_json(want), _first_difference(got, want)


# ── direct assertions (each is also in the golden; these say what the golden means) ──


@pytest.mark.timeout(240)
def test_the_narrator_drives_two_turns_and_its_finish_outranks_the_operator_cancel(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    t = _trace("narrator", tmp_path_factory)
    assert [c["answer"]["tool_name"] for c in t["narrator_calls"]] == [
        "send_message",
        "send_message",
        "finish_simulation",
    ]
    assert [turn["text"] for turn in t["turns"]] == ["Hello, who are you?", "What is your favourite colour?"]
    assert all([a["tool"] for a in turn["actions"]] == ["respond"] for turn in t["turns"])  # terminal (#1260)
    assert t["bridge_finish_calls"][0]["initiated_by"] == "llm_finish_tool"
    assert t["stdin_lines"] == [{"after": "finish_simulation", "after_turn": 2, "line": "/cancel"}]
    assert t["result"]["finish_reason"] == "completed"
    assert t["report"]["llm_finish_status"] == "completed"
    assert "summary: Simulation cancelled by user." in t["lifecycle"]


@pytest.mark.timeout(240)
def test_operator_commands_reach_their_targets(tmp_path_factory: pytest.TempPathFactory) -> None:
    """``/status`` reports the bridge, paused free text reaches the AUT, ``/cancel`` ends the run as ``cancel``."""
    t = _trace("stdin", tmp_path_factory)
    assert [x["line"] for x in t["stdin_lines"]] == [
        "/status",
        "/pause",
        "hello from the operator",
        "/resume",
        "please keep it brief",
        "/cancel",
    ]
    assert ["status: Turns: 1", "status: Actions: 1", "status: Blocked: 0"] == [
        line for line in t["lifecycle"] if line.startswith(("status: Turns:", "status: Actions:", "status: Blocked:"))
    ]
    paused = t["turns"][0]["paused_aut_actions"]
    assert [a["tool"] for a in paused] == ["respond"]
    # The AUT's no-LLM direct answer to a greeting (``agents/llm_fallback.py``): the operator's "hello" reached it.
    # (Until #1260 the respond's follow-up was an AUT LLM call carrying the text, which this read instead.)
    assert paused[0]["args"] == {"message": "Hello! How can I help you?"}
    assert t["result"]["finish_reason"] == "cancel"
    assert t["result"]["total_actions"] == 3


@pytest.mark.timeout(240)
def test_pre_campaign_turns_run_without_a_narrator(tmp_path_factory: pytest.TempPathFactory) -> None:
    t = _trace("precampaign", tmp_path_factory)
    assert t["narrator_calls"] == []
    assert [turn["text"] for turn in t["turns"]] == [x["text"] for x in ARMS["precampaign"]["pre_campaign_turns"]]
    assert [turn["turn"] for turn in t["result"]["campaign_analysis"]["turns"]] == [1, 2, 3]
    assert t["result"]["finish_reason"] == "completed"
    assert t["bridge_finish_calls"] and not t["bridge_finish_calls"][0], "the runner ended it, no finish_context"
    assert any(f.startswith("campaign_analysis_") for f in t["sim_reports_files"])


# ── the arms added after slice 0 (1.4.0 gate ``arms``: scripts/release_gates.json) ──


@pytest.mark.timeout(240)
def test_generative_arm_the_campaign_stops_the_run_before_the_kickoff_reaches_the_narrator(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """``generative=True`` (§9): the generative campaign runs on the calling thread and its ``_finish_runner`` sets
    ``stop_event``; only THEN is the narrator's kickoff injected, so the narrator loop never plans on it (the
    #1226 premise correction, Session C 2026-10-14). Pinned as current behaviour."""
    t = _trace("generative", tmp_path_factory)
    story = t["other_llm_calls"]
    assert [c["request"] for c in story["story_generate"]] == [
        "Generate the opening scene.",
        "Generate the next scene.",
    ]
    assert len(story["story_decide"]) == 2  # the third decide exhausts the one-phase arc without a call
    assert [turn["text"] for turn in t["turns"]] == list(_STORY_SCENES)
    (runner,) = [e for e in t["events"] if "finish_runner" in e]
    assert runner == {
        "finish_context": {},  # a completed generative campaign records no finish_context (#1002)
        "finish_runner": "generative_runner",
        "stop_event_set": True,
        "thread": "MainThread",
    }
    after = t["events"][t["events"].index(runner) + 1 :]
    kickoff = [e for e in after if e.get("inject") == "narrator#1"]
    assert len(kickoff) == 1 and kickoff[0]["text"].startswith("SIMULATION GOAL: " + GOAL)
    assert kickoff[0]["stop_event_set"] is True  # the run was already stopping when the kickoff landed
    assert [e for e in t["events"] if e.get("inject") == "narrator#1"] == kickoff  # nothing reached it before
    assert t["narrator_calls"] == []  # the narrator never planned on the kickoff
    assert t["result"]["finish_reason"] == "completed"
    assert t["bridge_finish_calls"] == [{}]


@pytest.mark.timeout(240)
def test_stall_arm_the_detector_injects_its_checkpoint_and_nudges_the_finished_narrator(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The stall detector (§11) past its first diversity checkpoint and its idle threshold. The checkpoint (after
    turn 3) carries the AUT's tools of turns 1-2 and rides the narrator's next follow-up; the idle nudge reaches the
    narrator only after it has called ``finish_simulation`` (since D13 a live narrator never idles before), and the
    narrator's answer to it overwrites the run's ``finish_context`` (#1259, pinned as current behaviour; its fix
    removes this path, so that PR re-homes the arm; #1452)."""
    t = _trace("stall", tmp_path_factory)
    narrator = [e for e in t["events"] if e.get("inject") == "narrator#1"]
    assert [(e["thread"], e["text"][:40]) for e in narrator] == [
        ("MainThread", "SIMULATION GOAL: " + GOAL[:23]),
        ("sim.stall", "DIVERSITY CHECKPOINT (turn 3): the agent"),
        ("sim.stall", "SYSTEM: Stall detected (<S>s idle, 3 AUT"),
    ]
    assert "tools so far (not yours): respond (2x). Its blocked tools: none." in narrator[1]["text"]
    calls = t["narrator_calls"]
    (observe,) = [c for c in calls if c.get("executed") == "observe_actions"]
    assert observe["held_inputs"].startswith("DIVERSITY CHECKPOINT (turn 3)")  # rode the follow-up (#1048)
    assert observe["answer"]["tool_name"] == "finish_simulation"
    assert calls[-1]["kind"] == "plan" and calls[-1]["request"].startswith("system: stall detected (<S>s idle")
    assert calls.index(observe) == len(calls) - 2  # the nudge is planned only after the finish
    # The probes' finish, the nudge's finish, then teardown's bridge.finish() seeing the last context.
    assert [f.get("reason") for f in t["bridge_finish_calls"]] == [
        "three probes delivered",
        "nudge after finish",
        "nudge after finish",
    ]
    assert t["report"]["llm_finish_reason"] == "nudge after finish"  # the post-finish answer overwrote it (#1259)
    assert t["stdin_lines"] == [{"after": "the idle nudge's answer", "after_turn": 3, "line": "/cancel"}]
    assert t["result"]["finish_reason"] == "completed"


_ONE_RUN_WRITES = ["save_report", "save_action_log", "save_aut_state", "analyze_simulation", "save_report"]


@pytest.mark.timeout(240)
def test_new_goal_arm_the_post_sim_goal_re_enters_the_run_with_a_fresh_aut(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """``/new`` (the interactive post-sim branch): the reader's goal re-enters ``start_simulation_mode``, which
    builds a second bridge, narrator source and throwaway AUT, and the caller gets the INNER run's result. The
    first AUT's learning does not reach the second: its NAc is saved under the first run's own fresh
    ``sim_agent_*`` dir and the re-entered run builds a new one (pinned as current behaviour, though the post-sim
    prompt says "memory carries over", #1454; only the narrator's ``orchestrator/`` stores are shared)."""
    t = _trace("new_goal", tmp_path_factory)
    assert t["post_sim_reads"] == [NEW_GOAL, ""]  # asked after each run: a goal, then Enter
    plans = [c["request"] for c in t["narrator_calls"] if c["kind"] == "plan"]
    assert [p.split("\n", 1)[0] for p in plans] == ["simulation goal: " + GOAL, "simulation goal: " + NEW_GOAL]
    kickoffs = [(e["inject"], e["text"].split("\n", 1)[0]) for e in t["events"] if e["inject"].startswith("narr")]
    assert kickoffs == [("narrator#1", "SIMULATION GOAL: " + GOAL), ("narrator#2", "SIMULATION GOAL: " + NEW_GOAL)]
    assert [e["inject"] for e in t["events"] if e["inject"].startswith("aut")] == ["aut#1", "aut#2"]  # two bridges
    assert [turn["text"] for turn in t["turns"]] == ["Hello, who are you?", "What do you remember?"]
    # Each run writes its own report; --report-json is emitted once (the outer run pops MAXIM_REPORT_JSON).
    assert t["report_writes"] == [*_ONE_RUN_WRITES[:-1], "save_report", "emit_report_json", *_ONE_RUN_WRITES]
    assert t["result"]["goal"] == NEW_GOAL and t["report"]["goal"] == NEW_GOAL  # the inner run's
    assert t["result"]["finish_reason"] == "completed" and t["result"]["turns"] == 1
    assert t["data_home_files"].count("sim_sandbox/sim_agent_<STAMP>_<RAND>/nac.json") == 2  # one AUT home each
    assert [len(links) for links in t["aut_nac"]["links"].values()] == [2]  # the inner AUT's turn only


@pytest.mark.timeout(240)
def test_persistent_agent_arm_the_handle_keeps_its_state_and_gets_its_registry_and_bus_back(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """HANDLE seam (a) with a real console ``MaximHandle``. Branch 3: no session AUT snapshot (the state lives in
    the agent's own home). Branch 4: the campaign's tool lease and the sim DefaultNetwork's pain bridge are in place
    mid-run, and both are undone after it (the registry holds the same tool objects again; the persistent bus no
    longer carries the bridge)."""
    t = _trace("persistent_agent", tmp_path_factory)
    assert "save_aut_state" not in t["report_writes"]
    assert "status: Persistent agent — state saved to its own home (no session AUT snapshot)" in t["lifecycle"]
    assert not [f for f in t["sim_reports_files"] if f.split("/")[-1].startswith("aut_")]
    assert "agents/console_agent/nac.json" in t["data_home_files"]  # the loop's session-end saved the home
    before, during, after = (t["handle"][k] for k in ("before", "during", "after"))
    assert set(during["tools"]) != set(before["tools"])  # the lease: sim tools in, some persistent tools out
    assert {"memory_recall", "say"} <= set(during["tools"]) - set(before["tools"])
    assert {"bash", "write_file"} <= set(before["tools"]) - set(during["tools"])
    assert after["tools"] == before["tools"] and after["replaced"] == []
    assert "PainCircuitBridge._on_pain" in during["pain_subscribers"]
    assert "PainCircuitBridge._on_pain" not in before["pain_subscribers"]
    assert after["pain_subscribers"] == before["pain_subscribers"]
    assert t["result"]["finish_reason"] == "completed"


@pytest.mark.timeout(240)
def test_imagination_arm_the_trigger_resolves_the_manifest_and_its_teardown_clears_the_imagined(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """§7 with an embodied AUT (``entity_ref``): the ImaginationTrigger is built and the scene-manifest pre-trigger
    resolves one entity from the ComponentIndex and DESIGNS the other (one designer call); at teardown the trigger's
    stats are reported, the imagined refs are tagged and decayed on the AUT's NAc, and the ephemeral design is
    cleared from the registry."""
    t = _trace("imagination", tmp_path_factory)
    calls = [c["call"] for c in t["imagination"]]
    assert calls == [
        "ImaginationTrigger.process_manifest",
        "ImaginationTrigger.stats",
        "NAc.tag_imagined_links",
        "NAc.decay_imagined_links",
        "ComponentRegistry.clear_ephemeral",
    ]
    manifest, stats, tag, decay, clear = t["imagination"]
    assert manifest["result"] == [
        {"imagined": False, "phrase": "rusty sword", "ref": "weapons/rusty_sword"},
        {"imagined": True, "phrase": "brass lantern", "ref": "items/brass_lantern"},
    ]
    assert [c["request"] for c in t["other_llm_calls"]["designer"]] == ["brass lantern"]
    assert (stats["result"]["index_hits"], stats["result"]["designs_succeeded"]) == (1, 1)
    assert tag["args"] == [["items/brass_lantern"]] and decay["args"] == [0.5]
    assert clear["result"] == ["items/brass_lantern"]
    assert "status: Imagination: 1 entities imagined, 2 phrases extracted" in t["lifecycle"]
    # The arm's own shipped files by path; every other shipped spec the registry read collapses to one marker.
    assert t["report"]["provenance"]["scenario_inputs"] == [
        {"path": "<REPO>/src/maxim/_data/components/bodies/base_humanoid.yaml", "sha256": _MASKED},
        {"path": "<REPO>/src/maxim/_data/reflexes/humanoid.yaml", "sha256": _MASKED},
        {"path": "<other shipped data files>", "sha256": _MASKED},
    ]
    assert t["report"]["provenance"]["scenario_sha256"] == _RECOMPUTED_OK
    assert t["result"]["finish_reason"] == "completed"


# ── #1260: the narrator's tool descriptions are the narrator's alone ──────────
#
# Until #1260, §5 of ``start_simulation_mode`` wrote the narrator's tool descriptions into the process-global
# ``modes.definitions.TOOL_DESCRIPTIONS``, so the AUT, running at the same time in the same process, read
# ``respond: NOT AVAILABLE`` and got a ``process`` follow-up after every ``respond``. They ride on the narrator's
# registry now (``SimToolRegistry.tool_descriptions``). These read the narrator arm's real run (the trace, and ``RUN_SIDE``: the raw prompts and the dict before the harness
# restores it).

_NARRATOR_SEND_MESSAGE = "This is your PRIMARY tool for interacting with the AUT."  # the narrator's own entry


@pytest.mark.timeout(240)
def test_the_auts_respond_is_terminal_1260(tmp_path_factory: pytest.TempPathFactory) -> None:
    """The AUT's ``respond`` is terminal (builtin ``followup_type: None``): no follow-up call, one action a turn."""
    t = _trace("narrator", tmp_path_factory)
    assert [c["executed"] for c in t["aut_calls"] if c["kind"] == "followup"] == []
    assert [[a["tool"] for a in turn["actions"]] for turn in t["turns"]] == [["respond"], ["respond"]]


@pytest.mark.timeout(240)
def test_the_auts_prompt_describes_its_own_respond_1260(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Every AUT prompt describes ``respond`` with the builtin entry, never the narrator's 'NOT AVAILABLE'."""
    from maxim.modes.definitions import TOOL_DESCRIPTIONS

    _trace("narrator", tmp_path_factory)
    aut_prompts = RUN_SIDE["narrator"]["prompts"]["aut"]
    assert aut_prompts, "the AUT was prompted"
    assert not [p for p in aut_prompts if "NOT AVAILABLE" in p]
    assert not [p for p in aut_prompts if _NARRATOR_SEND_MESSAGE in p]
    builtin = TOOL_DESCRIPTIONS["respond"]["description"]
    assert all(builtin in p for p in aut_prompts if not p.startswith("TOOL_PROMPT|You just executed "))


@pytest.mark.timeout(240)
def test_a_run_leaves_the_global_tool_descriptions_untouched_1260(tmp_path_factory: pytest.TempPathFactory) -> None:
    """``TOOL_DESCRIPTIONS`` after a real run, read before the harness restores it, deep-equals it before."""
    from maxim.modes.definitions import get_tool_followup_type

    _trace("narrator", tmp_path_factory)
    side = RUN_SIDE["narrator"]
    assert side["tool_descriptions_after"] == side["tool_descriptions_before"]
    assert side["tool_descriptions_after"]["respond"]["followup_type"] is None
    assert get_tool_followup_type("respond", "active", overlay={}) is None  # the dict after the run


@pytest.mark.timeout(240)
def test_the_narrators_prompt_keeps_its_own_tool_descriptions_1260(tmp_path_factory: pytest.TempPathFactory) -> None:
    """The narrator still reads its own entries (the fix must not take them from it): its planning prompts carry
    the narrator's ``send_message`` text, and its ``send_message`` still gets a ``process`` follow-up."""
    t = _trace("narrator", tmp_path_factory)
    plans = [
        p for p in RUN_SIDE["narrator"]["prompts"]["narrator"] if not p.startswith("TOOL_PROMPT|You just executed ")
    ]
    assert plans and all(_NARRATOR_SEND_MESSAGE in p for p in plans)
    assert [c["executed"] for c in t["narrator_calls"] if c["kind"] == "followup"] == ["send_message", "send_message"]


# ── #1263 (the DefaultNetwork part) / #1281: the loop that starts the AUT's DN stops it ──


_DN_THREADS = ("DefaultNetwork", "DN-BackgroundTasks")


@pytest.mark.timeout(240)
@pytest.mark.parametrize("arm", sorted(ARMS))
def test_the_run_leaves_no_default_network_running_1263(arm: str, tmp_path_factory: pytest.TempPathFactory) -> None:
    """``run_agentic_loop`` starts the AUT's DefaultNetwork (``loop_setup.build_loop_run``) and its teardown must stop
    it in a sim too. A leftover DN keeps ticking on the dead run's bio state and draws from the process-global
    ``random``, which broke ``test_seeding`` later in the suite (#1281). Read off the threads the run started that are
    still alive after it returned (``threads_left_alive.added``), not off the golden."""
    added = _trace(arm, tmp_path_factory)["threads_left_alive"]["added"]
    assert [name for name in added if name in _DN_THREADS] == []


def test_the_golden_is_non_vacuous() -> None:
    """The pinned traces exercise what the gate claims, and every arm ended clean."""
    arms = _fixture()["arms"]
    assert sorted(arms) == sorted(ARMS)
    for name, t in arms.items():
        assert t["unsorted_prompts"] == [], name
        assert t["live_sim_threads_after"] == [], name
        assert t["turns"] and all(turn["actions"] for turn in t["turns"]), name
        assert {"<STAMP>/report.json", "<STAMP>/actions.jsonl"} <= set(t["sim_reports_files"]), name
        assert t["report_json_emitted"] is True, name
        prov = t["report"]["provenance"]
        assert {k for k, v in prov.items() if v == _MASKED} == set(_PROVENANCE_MASKED), name
        if prov["scenario_inputs"]:  # checked against the report's own fields, never masked (_PROVENANCE_RECOMPUTED)
            assert prov["scenario_sha256"] == _RECOMPUTED_OK, name
        assert prov["serving_captured"] is True and prov["configured_n_ctx_source"] == "default", name
        assert "orchestrator/nac.json" in t["data_home_files"], name  # written only by the narrator's shutdown()
        assert "plan-logger" in t["threads_left_alive"]["added"], name  # #1263's plan writers (its DN part is fixed)
        assert t["aut_nac"]["links"], name
        # The decayed goal bias is pinned by its bound, and its causes by value (_NAC_DECAY_PINNED). The goal is the
        # run's own (the new_goal arm's result is its re-entered run's).
        goal = t["result"]["goal"]
        assert t["aut_nac"]["goal_reward_bias"] == {goal: "+ in (0, 0.2], decayed below its last write"}, name
        goal_credits = [c for c in t["nac_credits"]["aut"] if c["method"] == "credit_goal"]
        # One goal credit per AUT action. The first credit's ``post`` is alpha x reward (0.15 x 1.0), before any decay
        # tick; later ones carry none.
        assert len(goal_credits) == t["result"]["total_actions"], name
        first, *later = goal_credits
        assert first == {"method": "credit_goal", "goal_tag": goal, "reward": 1.0, "post": 0.15}, name
        assert all(c == {"method": "credit_goal", "goal_tag": goal, "reward": 1.0} for c in later), name
        assert [r.get("_record_kind") for r in t["action_log"][:1]] == ["header"], name
        assert len(t["action_log"]) - 1 == t["result"]["total_actions"], name
        runs = 1 + sum(1 for g in t.get("post_sim_reads", []) if g)  # a /new re-entry is a second run
        aut_states = [w for w in t["report_writer_args"] if w["writer"] == "save_aut_state"]
        # HANDLE branch 3: an adopted agent's state is never snapshotted into the session dir.
        assert len(aut_states) == (0 if "handle" in t else runs), name
        assert all(w["kwargs"]["nac"] == {"type": "NAc"} for w in aut_states), name
        assert t["result"]["summary"] == "golden roundup", name
        assert t["report_writes"].count("save_report") == 2 * runs, name  # before and after each roundup
        statuses = [line for line in t["lifecycle"] if line.startswith("status: ")]
        assert statuses[0] == "status: SIMULATION MODE — generative" and len(statuses) >= 10, name
    assert len(arms["narrator"]["narrator_calls"]) == 3
    # The narrator arm's AUT runs on its own router build, so the two roles' provenance differs.
    prov = arms["narrator"]["report"]["provenance"]
    assert (prov["language_router_serial"], prov["aut_router_serial"]) == ("<build-1>", "<build-2>")
    assert sorted(prov["llm_calls"]) == ["<build-1>", "<build-2>"]  # registered like real routers, zero calls
    # The AUT's build ran under its profile override, and the override was undone after it.
    assert (prov["language_profile"], prov["aut_profile"]) == (None, ARMS["narrator"]["aut_model"])
    assert all("MAXIM_LLM_PROFILE" not in t["maxim_env_after"] for t in arms.values())


def test_a_decayed_nac_value_outside_its_bound_or_undecayed_is_printed_raw() -> None:
    """``_NAC_DECAY_PINNED`` loosens a decayed value to its sign, bound and "decayed below its last write", not
    further: a value past the clamp, a zero (which the store never keeps), a non-number, or a value no decay
    tick reached after its last write does not match any pinned form."""
    assert _decayed_bound(0.17, 0.2, 0.2) == "+ in (0, 0.2], decayed below its last write"
    assert _decayed_bound(-0.01, 0.2, -0.15) == "- in [-0.2, 0), decayed below its last write"
    for bad in (0.21, -0.2001, 0.0, None, True):
        assert _decayed_bound(bad, 0.2, 0.2).startswith("OUT OF BOUND"), bad
    # Zero decay ticks after the last write (or a decay applied to another instance) is loud, never silent.
    assert "not decayed" in _decayed_bound(0.2, 0.2, 0.2)
    assert "not decayed" in _decayed_bound(0.17, 0.2, -0.2)  # sign flip
    assert _decayed_bound(0.17, 0.2, None) == "+ in (0, 0.2], no write this run"


def test_a_scenario_sha_that_does_not_recompute_is_printed_raw(tmp_path: Path) -> None:
    """``scenario_sha256`` is checked, not masked (``_PROVENANCE_RECOMPUTED``): a report whose hash does not match
    its own goal + inputs + content prints the raw value, which differs from the pinned ``_RECOMPUTED_OK``."""
    from tests.unit._orch_harness import _recompute_scenario_sha, _report_view

    prov = {"scenario_inputs": [{"path": "/x/arc.yaml", "sha256": "ab"}], "scenario_content_sha256": None}
    good = {"goal": "g", "provenance": {**prov, "scenario_sha256": _recompute_scenario_sha(prov, "g")}}
    bad = {"goal": "g", "provenance": {**prov, "scenario_sha256": _recompute_scenario_sha(prov, "another goal")}}
    for name, report in (("good", good), ("bad", bad)):
        (tmp_path / f"{name}.json").write_text(json.dumps(report))
    assert _report_view(tmp_path / "good.json")["provenance"]["scenario_sha256"] == _RECOMPUTED_OK
    assert _report_view(tmp_path / "bad.json")["provenance"]["scenario_sha256"] != _RECOMPUTED_OK


def test_the_fake_router_fails_closed() -> None:
    """A prompt no role claims, or more than one claims, is refused and recorded, never answered."""
    router = FakeRouter("narrator")
    assert router.generate_json("free text no role claims") is None
    assert router.generate_json("TOOL_PROMPT|x\nMode: SINGULARITY\n y\nMode: ACTIVE\n") is None
    assert len(router.unsorted) == 2
    assert FakeRouter.classify("TOOL_PROMPT|x\nMode: ACTIVE\n", None) == "aut"
    assert FakeRouter.classify("p", "You are a QA analyst reviewing") == "roundup"


def test_a_shadowing_install_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    import maxim

    monkeypatch.setattr(maxim, "__file__", "/elsewhere/site-packages/maxim/__init__.py")
    with pytest.raises(WrongCheckout):
        assert_this_checkout()


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("arm", sorted(ARMS))
def test_golden_is_identical_across_processes_and_hash_seeds(arm: str, tmp_path: Path) -> None:
    """Two fresh interpreters with differing PYTHONHASHSEED produce the pinned trace byte for byte (set/dict
    iteration over hashed strings, import order, a fresh home). Slow lane: each emit is a full ~20 s run."""
    expected = canonical_json(_fixture()["arms"][arm])
    for seed in ("0", "4242"):
        assert _emit_in_subprocess(arm, seed, tmp_path) == expected + "\n", f"PYTHONHASHSEED={seed}"


# ── helpers / regeneration ────────────────────────────────────────────────


def _emit_in_subprocess(arm: str, hashseed: str, tmp_path: Path) -> str:
    work = tmp_path / f"sub_{arm}_{hashseed}"
    work.mkdir()
    env = dict(os.environ)
    env.update(
        PYTHONHASHSEED=hashseed,
        PYTHONPATH=os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT), env.get("PYTHONPATH", "")]).rstrip(
            os.pathsep
        ),
        HOME=str(work / "home"),
    )
    out = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--emit", arm, str(work)],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    return out.stdout.split("<<<GOLDEN>>>\n", 1)[1]


def _emit(arm: str, workdir: Path) -> str:
    trace = run_arm(arm, workdir)
    trace.pop("_elapsed_s")
    return canonical_json(trace)


def _first_difference(got: Any, want: Any, path: str = "$") -> str:
    if type(got) is not type(want):
        return f"{path}: {got!r} != {want!r}"
    if isinstance(got, dict):
        for k in sorted(set(got) | set(want)):
            if k not in got or k not in want:
                return f"{path}.{k}: present in only one side"
            if got[k] != want[k]:
                return _first_difference(got[k], want[k], f"{path}.{k}")
    if isinstance(got, list):
        for i, (a, b) in enumerate(zip(got, want)):
            if a != b:
                return _first_difference(a, b, f"{path}[{i}]")
        if len(got) != len(want):
            return f"{path}: length {len(got)} != {len(want)}"
    return f"{path}: {got!r} != {want!r}"


def _head_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    return out.stdout.strip()


def _src_is_clean() -> bool:
    out = subprocess.run(["git", "status", "--porcelain", "--", "src/"], cwd=REPO_ROOT, capture_output=True, text=True)
    return out.returncode == 0 and not out.stdout.strip()


if __name__ == "__main__":  # regeneration / subprocess entry point -- read the module docstring first
    import tempfile

    if len(sys.argv) >= 4 and sys.argv[1] == "--emit":
        print("<<<GOLDEN>>>\n" + _emit(sys.argv[2], Path(sys.argv[3])))
        raise SystemExit(0)
    if "--regen" not in sys.argv:
        raise SystemExit(
            "usage: python tests/unit/test_orchestrator_characterization_golden.py --regen  (from the commit you mean to pin)"
        )
    if not _src_is_clean():
        raise SystemExit("refusing to regenerate: src/ has uncommitted changes (regenerate from a committed tree)")
    arms = {}
    for name in sorted(ARMS):
        with tempfile.TemporaryDirectory() as d:
            arms[name] = json.loads(_emit(name, Path(d)))
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at_commit": _head_commit(), "arms": arms}
    FIXTURE.write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n")
    print(f"wrote {FIXTURE} at {payload['generated_at_commit']}")

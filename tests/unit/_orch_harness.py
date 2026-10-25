"""Offline driver for the REAL ``start_simulation_mode`` (1.3.2 orchestrator decomposition, slice 0).

Used by ``test_orchestrator_characterization_golden.py``, the characterization gate the orchestrator's
extraction slices must keep green UNCHANGED (``docs/plans/roadmap_1_3_x.md`` §"Coverage first, then
extract"). It is the orchestrator's counterpart of ``tests/unit/_loop_harness.py``.

WHAT IT DRIVES. The real ``maxim.simulation.orchestrator.start_simulation_mode``, end to end. That
covers the AUT build, the narrator pipeline, the ``sim.aut`` thread, campaign dispatch, the stdin
reader, the stall detector, the narrator loop on the calling thread, teardown, the report and the
``SimulationResult``. Nothing in ``src/`` is stubbed except the LLM. ``FakeRouter`` is patched in at
``maxim.runtime.lane_backends.build_primary_router``, the seam ``test_api_expansion.py`` already
uses. The slice-0 arms (owner decision (a), 2026-10-13):

  1. ``precampaign``: three ``pre_campaign_turns`` and no narrator LLM. The runner ends the run.
  2. ``narrator``: the fake narrator calls ``send_message`` twice and then ``finish_simulation``.
     This arm exercises the narrator loop on the calling thread (§12), the stall-detector polling
     and the LLM-finish path. It also passes ``aut_model``, so the AUT gets its OWN router build
     (the dual-router path, §4) and the two roles' provenance can be told apart.
  3. ``stdin``: operator commands fed through a queue-backed ``sys.stdin``: ``/status``, ``/pause``,
     free text while paused (which goes to the AUT), ``/resume``, free text while running (which
     goes to the narrator as guidance), and ``/cancel``.

Then the arms the 1.4.0 gate requires before the slices that move their code (``scripts/release_gates.json``
``arms``; roadmap_1_3_x.md §1.3.5), added 2026-10-24 without changing the three above:

  4. ``generative``: ``generative=True`` with a one-phase, two-turn arc (``_GENERATIVE_ARC``). The generative
     campaign's own narrator (``simulation/narrator.py``, the ``story_*`` roles) drives both turns on the calling
     thread; its ``_finish_runner`` sets ``stop_event``, and only then is the narrator kickoff injected (§9).

The TTY read loops (the raw-mode stdin reader and the post-sim raw-mode "new goal" prompt) are
deliberately left uncharacterized (owner decision (e)). Every arm runs non-interactive
(``InteractiveMode.OFF``, the ``--interactive false`` path), so the line reader is the one exercised.

ROUTER BUILDS. Every ``build_primary_router`` call returns a ``_RouterView``: the one shared
``FakeRouter`` (one call log) under that build's own ``call_serial``, registered with
``call_accounting.register_router()`` exactly as a real ``LLMRouter`` registers its own. Build 1 is the
language router; build 2, made only when ``aut_model`` is passed, is the AUT's. The report's provenance
names each role's router by that serial, so a run that hands the AUT's worker to the narrator's slot
shows. Each view also carries ``cfg.profile``: the ``MAXIM_LLM_PROFILE`` in force at its build, which the
run sets to ``aut_model`` around the AUT's build, so the provenance's ``aut_profile`` pins that override.
CALL ACCOUNTING IS NOT PINNED: ``FakeRouter`` never books a call, so ``provenance.llm_calls`` holds only
the builds' zero counts. The serial NUMBER depends on how many routers the process built before (the full suite builds
real ones), so the trace records each as its build label, ``<build-1>``/``<build-2>``
(``_label_serials``); a serial the run did not build stays a number, which is a finding. (A first
version numbered the views 1, 2 itself: in the full suite an earlier test's real router held serial 1,
and the report's ``llm_calls`` showed it.)

THREADS STAY ON, ON THE WALL CLOCK (unlike the loop golden's step clock). The run keeps its real
threads and its real 3 s settle, so the trace records only CAUSALLY ORDERED facts:

  - each role's LLM calls, in that role's own order;
  - each bridge turn, its percept text and what the AUT did in it;
  - the ``display_status``/``display_summary`` lifecycle lines;
  - ``finish_reason`` and ``finish_context``;
  - the report writers called, in order, with WHAT each was handed (``_writer_arg``, ``_shared_refs``), and the
    ``sim_reports`` file set;
  - the whole ``MAXIM_DATA_HOME`` file set (``_DATA_HOME_EXCLUDED``), which shows the orchestrator's
    own persisted stores, the sandbox and the reports;
  - the deterministic ``report.json`` fields, provenance by VALUE (``_PROVENANCE_MASKED``);
  - the saved AUT NAc's causal links and the ``actions.jsonl`` records, times dropped
    (``_NAC_DROPPED``, ``_ACTION_DROPPED``); its tick-decayed bias maps by key, sign and bound only
    (``_NAC_DECAY_PINNED``);
  - every NAc credit write that feeds a decayed map (``_NAC_CREDIT_ENTRY_POINTS``), with its arguments,
    in each NAc's own call order (``nac_credits``): the CAUSES of the decayed values, pinned exactly;
  - the threads the run started that are still alive after it returns (``_threads_left``);
  - the ``SimulationResult`` with its timestamps dropped.

It never records a cross-thread interleaving or a stall-detector injection. The arms stay shorter
than the detector's first diversity checkpoint (3 turns) and its idle threshold (30 s).

CAUSAL SYNCHRONISATION. Class-level hooks on ``SimulationBridge`` (and, for the ``events`` arms,
``ConversationalSource.inject_cli`` and the module attribute ``orchestrator._finish_runner``); class
attributes hold wherever a slice moves the CALLER, and a slice that moves ``_finish_runner`` must keep
calling it through ``orchestrator`` or extend the patch list:

  - ``send_and_wait``: records the turn, then runs the arm's after-turn step ON THE CALLING THREAD,
    before the turn returns to its caller. The stdin arm uses it to feed operator lines and to wait
    until the reader has processed each one, so every command lands at a fixed point between two
    turns.
  - ``finish``: in the narrator arm, the narrator's ``finish_simulation`` is followed by an operator
    ``/cancel``. Without it the run idles out the AUT-side 180 s percept-source grace while the stall
    detector nudges the finished narrator every 30 s, which is nondeterministic and three minutes
    long. The test file's docstring records this as a src finding. ``finish_reason`` stays
    ``completed``, because the LLM's ``finish_context`` outranks the operator stop.

NONDETERMINISM, NEUTRALISED. Every exclusion is a named constant below with its reason
(``_RESULT_DROPPED``, ``_REPORT_DROPPED``, ``_PROVENANCE_MASKED``, ``_DATA_HOME_EXCLUDED``,
``_THREADS_EXCLUDED``, ``_NAC_DROPPED``, ``_NAC_DECAY_PINNED``, ``_ACTION_DROPPED``, ``_HIPPO_PINNED``,
``_LIFECYCLE_MASKS``, ``_scrub``):

  - session ids, stamps, tmpdir suffixes and the work path become placeholders;
  - times, latencies and durations are dropped;
  - provenance is pinned by value, except the keys that describe the CHECKOUT and the interpreter
    rather than the run (measured: two narrator runs differed in nothing else);
  - the SCN's calendar-derived ``temporal`` block is pinned by its keys;
  - the hippocampus memory count, which counts idle-tick captures that race the next percept, is masked;
  - the NAc's tick-decayed values are pinned by key, sign and bound, and their credits by value
    (``_NAC_DECAY_PINNED``): how many decay ticks land before the save is wall-clock-paced;
  - EC node ids come from a counter, as in phase 1.

The determinism tests cover the rest: the three arms run back to back in one process against the
golden, and each arm runs in fresh interpreters under two hash seeds.

STATED LIMIT (wall clock). A turn closes after 3 s with no new AUT action, as it does live. The AUT's
reply is one terminal ``respond`` (#1260). Before that fix a narrator-only entry gave ``respond`` a
follow-up, and the chain was ``respond`` then ``say``, about 0.5 s apart; a box loaded enough to stretch
that gap past 3 s would split a turn. A reply chain of two or more actions brings that limit back, and
the golden goes red on a split turn. Check what else is running before reading such a failure as a
regression.

GLOBAL STATE. A run mutates process-global state, all of which ``run_arm`` snapshots and restores:

  - the ``maxim`` logger level and every ``MAXIM_*`` env var (including ``MAXIM_ALLOW_BASH`` and
    ``MAXIM_REPORT_JSON``);
  - the user config: ``XDG_CONFIG_HOME`` points into the work dir and the config cache is reset, so the
    pinned ``configured_n_ctx`` is the default, not whatever ``~/.config/maxim/config.json`` holds on
    the box running ``--regen``;
  - the global seed (``utils.seeding``), which the provenance stamps;
  - the sim display/interactive globals, the JSON parse counters, the LLM cancellation flag and the
    lane ``MetricsRegistry`` the report reads (a fresh one for the run: an earlier test's lane metrics
    leaked into ``report.lane_breakdown`` in the full suite);
  - ``sys.stdin``, the CWD and the path caches.

``MAXIM_SIM_HARD_ABORT=0`` is mandatory: the hard-abort path calls ``os._exit(4)``, which would kill
pytest.

ASSERTED, NOT RESTORED: ``modes.definitions.TOOL_DESCRIPTIONS``. A run must leave it as it found it: the
narrator's own entries ride on its registry (#1260), and a run that writes the process-global dict fails
``run_arm`` (``_GlobalState.restore``) rather than being quietly put back. The dict is put back only so
the failure does not cascade into later tests.
"""

from __future__ import annotations

import copy
import inspect
import io
import json
import logging
import os
import queue
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
GOAL = "check how the agent answers two short questions"

# The operator lines of the stdin arm, by the turn after which they are fed (1-based).
_STDIN_SCRIPT: dict[int, list[str]] = {
    1: ["/status", "/pause", "hello from the operator", "/resume", "please keep it brief"],
    2: ["/cancel"],
}
_PRECAMPAIGN_TURNS: list[dict[str, Any]] = [
    {"text": "You wake in a quiet room.", "phase": "intro"},
    {"text": "A bell rings twice.", "phase": "middle", "salience": 0.9},
    {"text": "The room goes dark.", "phase": "end", "novelty": 0.5},
]
# The generative arm's arc (written into the work dir, so its path scrubs to ``<WORK>``): one phase of two turns, so
# the narrator's third ``decide`` exhausts the arc without an LLM call and the campaign completes.
_GENERATIVE_ARC = """\
name: orch_golden_arc
description: two short probes
phases:
  - name: probe
    turns: [2, 2]
    instruction: Ask the agent one short question.
"""
_STORY_SCENES = ("A stranger asks your name.", "The stranger asks your favourite colour.")
# Per arm: ``max_turns``; the ``start_simulation_mode`` keywords it adds (``kwargs``; ``arc_yaml`` is filled in by
# ``run_arm``); and ``events``: record every ``ConversationalSource.inject_cli`` and every campaign-runner finish
# (``orchestrator._finish_runner``), in call order (the trace's ``events`` key). Only the arms added after slice 0
# set it, so the slice-0 arms' traces stay byte-identical.
ARMS: dict[str, dict[str, Any]] = {
    "precampaign": {"max_turns": 10, "pre_campaign_turns": _PRECAMPAIGN_TURNS},
    "narrator": {"max_turns": 10, "aut_model": "orch-golden-aut"},
    "stdin": {"max_turns": 10},
    "generative": {"max_turns": 10, "kwargs": {"generative": True}, "arc": _GENERATIVE_ARC, "events": True},
}
# The report writers the run calls, in order (a removed or reordered save shows even when a later
# save of the same file masks it on disk).
_REPORT_WRITERS = ("save_report", "save_action_log", "save_aut_state", "analyze_simulation", "emit_report_json")
# Per arm, what the LAST run left behind outside the canonical trace (so the golden does not pin it): each role's
# raw prompts, and ``TOOL_DESCRIPTIONS`` as the run left it, read BEFORE ``_GlobalState.restore`` checks it
# (#1260: a narrator-only description must not reach the process-global dict).
RUN_SIDE: dict[str, dict[str, Any]] = {}
SETTLE_TIMEOUT_S = 30.0  # a harness wait that exceeds this is a hang, not a slow run
_AUT_SETTLE_S = 3.0  # SimulationBridge's default settle_s: the same quiet window a turn waits for


class WrongCheckout(AssertionError):
    """The imported ``maxim`` is not this checkout's (an installed package shadowing a worktree)."""


class HarnessHang(AssertionError):
    """A causal wait in the harness timed out (the run did not reach the expected point)."""


def assert_this_checkout() -> None:
    """A worktree run without PYTHONPATH would test main's code and still pass: refuse it."""
    import maxim

    here = Path(maxim.__file__).resolve()
    if not here.is_relative_to(REPO_ROOT):
        raise WrongCheckout(
            f"maxim imported from {here}, not this checkout ({REPO_ROOT}); set PYTHONPATH=<checkout>/src"
        )


# ── the fake LLM ──────────────────────────────────────────────────────────

_USER_REQUEST = re.compile(r"=== User Request ===\n\"(.*?)\"\n\n=== ", re.S)
_EXECUTED = re.compile(r"^TOOL_PROMPT\|You just executed '([^']+)'")
_ORIGINAL = re.compile(r"Original request: \"(.*?)\"\n\n", re.S)
_TOOL_LIST = re.compile(r"=== Your Tools ===\nYour tools are: (.*?)\. Choose one of these\.", re.S)
_TURN = re.compile(r"'turn': (\d+)")


class FakeRouter:
    """A duck-typed ``LLMRouter`` that sorts every prompt by marker and answers from a script.

    Roles and their markers (``classify``): the narrator (``Mode: SINGULARITY`` plan prompts, or a
    follow-up whose tool roster holds ``finish_simulation``), the AUT (``Mode: ACTIVE`` plan prompts,
    or a follow-up whose roster holds ``memory_recall``), the FearAgent code review, the
    post-run roundup and the scene manifest. A prompt that matches no role, or more than one, FAILS
    CLOSED: it is recorded in ``unsorted`` (the tests assert that list is empty) and answered with
    ``None``, an invalid response.
    """

    n_ctx = 8192
    model_name = "orch-golden-fake"
    session_cost_exceeded = False

    def __init__(self, arm: str) -> None:
        self.arm = arm
        self.calls: dict[str, list[dict[str, Any]]] = {
            "narrator": [],
            "aut": [],
            "fear": [],
            "roundup": [],
            "manifest": [],
            "story_decide": [],
            "story_generate": [],
        }
        self.unsorted: list[str] = []
        self.prompts: dict[str, list[str]] = {role: [] for role in self.calls}  # raw, for RUN_SIDE only
        self._lock = threading.Lock()

    # The router surface the run touches besides generate_json.
    def warmup(self) -> bool:
        return True

    def wait_ready(self, timeout: float = 120.0) -> bool:
        return True

    def get_token_counter(self) -> Any:
        from maxim.models.language.token_counter import CharEstimateCounter

        return CharEstimateCounter()

    @staticmethod
    def classify(prompt: str, system_override: str | None) -> str | None:
        roles: list[str] = []
        if (system_override or "").startswith("You are a QA analyst"):
            roles.append("roundup")
        if "You are a security reviewer analyzing code" in prompt:
            roles.append("fear")
        if "List the entities:" in prompt and '"entities"' in prompt:
            roles.append("manifest")
        # The generative campaign's narrator (``simulation/narrator.py``): its decision and generation calls.
        if prompt.startswith("You are the narrative director for a cognitive agent simulation."):
            roles.append("story_decide")
        if prompt.startswith("You are a vivid, immersive narrator."):
            roles.append("story_generate")
        if prompt.startswith("TOOL_PROMPT|You just executed "):
            m = _TOOL_LIST.search(prompt)
            roster = m.group(1) if m else ""
            if "'finish_simulation'" in roster and "'memory_recall'" not in roster:
                roles.append("narrator")
            if "'memory_recall'" in roster and "'finish_simulation'" not in roster:
                roles.append("aut")
        elif prompt.startswith("TOOL_PROMPT|"):
            if "\nMode: SINGULARITY\n" in prompt:
                roles.append("narrator")
            if "\nMode: ACTIVE\n" in prompt:
                roles.append("aut")
        return roles[0] if len(roles) == 1 else None

    def generate_json(
        self, prompt: str, temperature: float = 0.3, max_tokens: int = 1024, **kwargs: Any
    ) -> dict[str, Any] | None:
        role = self.classify(prompt, kwargs.get("system_override"))
        if role is None:
            with self._lock:
                self.unsorted.append(prompt[:400])
            return None
        call = _describe(prompt)
        answer = getattr(self, f"_answer_{role}")(call)
        call["answer"] = answer.get("action") if role in ("narrator", "aut") else sorted(answer)
        with self._lock:
            self.calls[role].append(call)
            self.prompts[role].append(prompt)
        return copy.deepcopy(answer)

    # ── per-role scripts ─────────────────────────────────────────────────
    def _answer_narrator(self, call: dict[str, Any]) -> dict[str, Any]:
        if call["kind"] == "followup" and call["executed"] == "send_message":
            turn = call.get("turn")
            if turn == 1:
                return _act("send_message", text="What is your favourite colour?")
            if turn == 2:
                if self.arm == "stdin":  # the operator ends this arm (/cancel after turn 2)
                    return _act("observe_actions")
                return _act(
                    "finish_simulation",
                    status="completed",
                    reason="two probes delivered",
                    summary="the agent answered both probes",
                )
        if call["kind"] == "plan":
            req = call["request"]
            if req.startswith("simulation goal:"):
                return _act("send_message", text="Hello, who are you?")
            if req.startswith("user guidance:"):
                return _act("check_completion")
            if req.startswith("system: user paused") or req.startswith("system: user resumed"):
                return _act("observe_actions")
        if call["kind"] == "followup" and call["executed"] in ("observe_actions", "check_completion"):
            return {"action": None, "confidence": 0.9, "reasoning": "nothing to do yet", "ready_to_act": False}
        return _act("finish_simulation", status="aborted", reason="unscripted narrator input", summary="")

    def _answer_aut(self, call: dict[str, Any]) -> dict[str, Any]:
        if call["kind"] == "plan":
            return _act("respond", message=f"I heard: {call['request'][:40]}")
        if call["executed"] == "respond":  # reached only if the AUT's terminal respond is followed up (#1260)
            return _act("say", text="That is all.")
        return {"action": None, "confidence": 0.9, "reasoning": "done", "ready_to_act": False}

    def _answer_fear(self, call: dict[str, Any]) -> dict[str, Any]:
        return {"findings": []}

    def _answer_roundup(self, call: dict[str, Any]) -> dict[str, Any]:
        return {"summary": "golden roundup", "issues": ["none observed"], "recommendations": ["keep going"]}

    def _answer_manifest(self, call: dict[str, Any]) -> dict[str, Any]:
        return {"entities": []}

    def _answer_story_decide(self, call: dict[str, Any]) -> dict[str, Any]:
        return {"phase": "probe", "scene_type": "question", "notes": "ask one short question", "done": False}

    def _answer_story_generate(self, call: dict[str, Any]) -> dict[str, Any]:
        n = len(self.calls["story_generate"])  # this call is appended after it is answered
        return {"narrative": _STORY_SCENES[n] if n < len(_STORY_SCENES) else "unscripted scene"}


class _RouterView:
    """One ``build_primary_router`` build: the shared ``FakeRouter`` under that build's own ``call_serial``
    (module docstring, ROUTER BUILDS). Every other attribute is the shared router's."""

    def __init__(self, router: FakeRouter, serial: int, profile: str | None) -> None:
        self._router = router
        self.call_serial = serial
        # What a real build reads: the profile in force when it is built (``MAXIM_LLM_PROFILE``, which the run
        # sets to ``aut_model`` around the AUT's build). ``report.run_provenance`` reads ``router.cfg.profile``.
        self.cfg = SimpleNamespace(profile=profile)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._router, name)


def _act(tool: str, **params: Any) -> dict[str, Any]:
    return {"action": {"tool_name": tool, "params": params}, "confidence": 0.9, "reasoning": f"scripted {tool}"}


def _describe(prompt: str) -> dict[str, Any]:
    """The causal content of a prompt: what was asked, never the clock-stamped context around it."""
    m = _EXECUTED.match(prompt)
    if m:
        orig = _ORIGINAL.search(prompt)
        turn = _TURN.search(prompt)
        return {
            "kind": "followup",
            "executed": m.group(1),
            "request": (orig.group(1) if orig else "")[:120],
            "turn": int(turn.group(1)) if turn else None,
        }
    if prompt.startswith("TOOL_PROMPT|"):
        req = _USER_REQUEST.search(prompt)
        return {"kind": "plan", "request": (req.group(1) if req else "")[:120].lower()}
    if prompt.startswith(("You are the narrative director", "You are a vivid, immersive narrator")):
        # The story narrator's prompts: what it was asked is the closing user part, after the system text.
        return {"kind": "direct", "request": prompt.rstrip().rsplit("\n", 1)[-1][:120]}
    return {"kind": "direct", "request": prompt[:80]}


# ── stdin ─────────────────────────────────────────────────────────────────


class QueueStdin(io.TextIOBase):
    """A ``sys.stdin`` whose lines arrive when the harness feeds them; ``input()`` reads it through
    ``readline`` (no ``fileno``, so CPython takes the non-console path). ``reads`` counts readline
    calls: the reader asking for line k+1 means it has processed line k."""

    def __init__(self) -> None:
        super().__init__()
        self._q: queue.Queue[str] = queue.Queue()
        self._cv = threading.Condition()
        self.reads = 0

    def readable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def readline(self, size: int | None = -1) -> str:  # type: ignore[override]
        with self._cv:
            self.reads += 1
            self._cv.notify_all()
        return self._q.get()

    def feed(self, line: str) -> None:
        self._q.put(line + "\n")

    def eof(self) -> None:
        self._q.put("")

    def wait_reads(self, n: int, reader: threading.Thread | None) -> None:
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        with self._cv:
            while self.reads < n:
                if reader is not None and not reader.is_alive():
                    return  # the reader exited (``/cancel``): the line was processed
                if time.monotonic() > deadline:
                    raise HarnessHang(f"stdin reader did not reach read #{n} (at {self.reads})")
                self._cv.wait(0.05)


def _thread(name: str) -> threading.Thread | None:
    return next((t for t in threading.enumerate() if t.name == name), None)


# ── canonical form ────────────────────────────────────────────────────────

_STAMP = re.compile(r"\d{8}_\d{6}")
_DASH_STAMP = re.compile(r"\d{4}-\d{2}-\d{2}_\d{6}")
_TMP_SUFFIX = re.compile(r"(sim_agent_<STAMP>_)[A-Za-z0-9_]{8}")


def _scrub(text: str, work: Path) -> str:
    text = text.replace(str(work.resolve()), "<WORK>").replace(str(work), "<WORK>")
    text = text.replace(str(REPO_ROOT), "<REPO>")  # a spec file read from the checkout (the imagination arm)
    text = _STAMP.sub("<STAMP>", text)
    text = _DASH_STAMP.sub("<STAMP>", text)
    text = _TMP_SUFFIX.sub(r"\1<RAND>", text)
    return text


def _canon(value: Any, work: Path) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        r = round(value, 9)
        return 0.0 if r == 0 else r
    if isinstance(value, str):
        return _scrub(value, work)
    if isinstance(value, dict):
        return {_scrub(str(k), work): _canon(v, work) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canon(v, work) for v in value]
    return _scrub(repr(value), work)


def canonical_json(trace: dict[str, Any]) -> str:
    return json.dumps(trace, sort_keys=True, indent=1)


# ── the driven run ────────────────────────────────────────────────────────


def _action_view(a: Any) -> dict[str, Any]:
    out = a.result_output
    return {
        "tool": a.tool_name,
        "args": dict(getattr(a, "tool_args", {}) or {}),
        "success": bool(a.result_success),
        "blocked": bool(a.blocked),
        "output": str(out)[:120] if out is not None else None,
    }


class _GlobalState:
    """Snapshot/restore of the process-global state a run mutates (module docstring)."""

    def __init__(self) -> None:
        from maxim.models.language import cancellation, json_parser
        from maxim.modes import definitions
        from maxim.simulation import sim_logger as SL

        self.env = dict(os.environ)
        self.cwd = os.getcwd()
        self.stdin = sys.stdin
        self.tool_descriptions = copy.deepcopy(definitions.TOOL_DESCRIPTIONS)
        self.maxim_level = logging.getLogger("maxim").level
        self.display = (
            SL._display_tier,
            SL._display_floor,
            SL._interactive_mode,
            SL._escalation_expires_at,
            dict(SL._agent_nicknames),
            dict(SL._sensor_last_logged),
        )
        self.json_stats = (json_parser._json_first_try, json_parser._json_total, json_parser._json_repaired)
        self.shutdown = cancellation.is_shutdown_requested()
        from maxim.utils import seeding

        self.global_seed = seeding._global_seed
        # The report reads the process-global lane MetricsRegistry (``report.lane_breakdown`` and the
        # latency percentiles), which nothing resets per sim: in the full suite an earlier test left a
        # ``large`` lane in it and the golden went red. The run gets a fresh registry; the old one is put back.
        from maxim.models.language import lane_metrics

        with lane_metrics._registry_lock:
            self.lane_registry = lane_metrics._global_registry
            lane_metrics._global_registry = None

    def restore(self) -> None:
        from maxim.models.language import cancellation, json_parser
        from maxim.modes import definitions
        from maxim.simulation import sim_logger as SL
        from maxim.utils.paths import _reset_caches

        os.environ.clear()
        os.environ.update(self.env)
        os.chdir(self.cwd)
        sys.stdin = self.stdin
        leaked = definitions.TOOL_DESCRIPTIONS != self.tool_descriptions
        if leaked:  # put back only so the failure below does not cascade into later tests
            definitions.TOOL_DESCRIPTIONS.clear()
            definitions.TOOL_DESCRIPTIONS.update(self.tool_descriptions)
        logging.getLogger("maxim").setLevel(self.maxim_level)
        tier, floor, mode, esc, nick, sensors = self.display
        SL._display_tier, SL._display_floor, SL._interactive_mode, SL._escalation_expires_at = tier, floor, mode, esc
        SL.clear_agent_nicknames()
        SL._agent_nicknames.update(nick)
        SL._sensor_last_logged.clear()
        SL._sensor_last_logged.update(sensors)
        with json_parser._json_lock:
            json_parser._json_first_try, json_parser._json_total, json_parser._json_repaired = self.json_stats
        (cancellation.request_shutdown if self.shutdown else cancellation.reset_shutdown)()
        from maxim.runtime.config_loader import reset_config_cache
        from maxim.utils import seeding

        seeding._global_seed = self.global_seed
        reset_config_cache()  # after the env restore: the next reader sees the caller's config again
        from maxim.models.language import lane_metrics

        with lane_metrics._registry_lock:
            lane_metrics._global_registry = self.lane_registry
        _reset_caches()
        # Asserted last, after everything else is back (module docstring, ASSERTED, NOT RESTORED).
        assert not leaked, (
            "the run wrote modes.definitions.TOOL_DESCRIPTIONS (#1260: an agent's own entries ride on its registry)"
        )


def run_arm(arm: str, workdir: Path) -> dict[str, Any]:
    """Drive the real ``start_simulation_mode`` for one arm; return the CANONICAL trace."""
    assert_this_checkout()
    import maxim.runtime.lane_backends as LB
    from maxim.decisions.nac import NAc
    from maxim.models.language.json_parser import json_parse_stats_reset
    from maxim.models.language.cancellation import reset_shutdown
    from maxim.similarity import ec as ec_mod
    from maxim.simulation import bridge as bridge_mod
    from maxim.simulation import conversational_source as conv_mod
    from maxim.simulation import orchestrator as orch
    from maxim.simulation import report as report_mod
    from maxim.simulation import sim_logger as SL
    from maxim.utils.paths import _reset_caches

    spec = ARMS[arm]
    work = workdir.resolve()
    router = FakeRouter(arm)
    turns: list[dict[str, Any]] = []
    lifecycle: list[str] = []
    finish_calls: list[dict[str, Any]] = []
    report_writes: list[str] = []
    writer_args: list[dict[str, Any]] = []
    builds: list[int] = []  # the call_serial of each build_primary_router call, in order
    nac_credits: list[tuple[Any, dict[str, Any]]] = []  # (the NAc instance, the call), in call order
    credited_keys: set[tuple[int, str, Any]] = set()  # (NAc id, map, key) already written once (``post``)
    last_post: dict[tuple[int, str, Any], Any] = {}  # (NAc id, map, key) -> value after its LAST write (not traced)
    aut_nac: list[Any] = []  # the NAc handed to save_aut_state (whose credits are the AUT's)

    def _build_router(*_a: Any, **_k: Any) -> tuple[_RouterView, None]:
        from maxim.models.language import call_accounting

        serial = call_accounting.register_router()
        builds.append(serial)
        return _RouterView(router, serial, os.environ.get("MAXIM_LLM_PROFILE")), None

    stdin_feed: list[dict[str, Any]] = []
    stdin = QueueStdin() if arm in ("narrator", "stdin") else io.StringIO("")
    counter = iter(range(1, 1_000_000))
    Bridge = bridge_mod.SimulationBridge
    orig_send, orig_finish, orig_bridge_init = Bridge.send_and_wait, Bridge.finish, Bridge.__init__
    Source = conv_mod.ConversationalSource
    orig_inject = Source.inject_cli
    bridges: list[Any] = []  # every SimulationBridge the run built (a /new re-entry builds a second)
    events: list[dict[str, Any]] = []  # ``inject_cli`` calls and campaign-runner finishes, in call order
    run_stops: list[threading.Event] = []  # the stop_event each ``_finish_runner`` was handed
    narrator_sources: list[int] = []  # id() of each non-bridge source, in first-injection order
    orig_status, orig_summary = SL.display_status, SL.display_summary

    def _feed(line: str) -> None:
        assert isinstance(stdin, QueueStdin)
        before = stdin.reads
        stdin.feed(line)
        stdin.wait_reads(before + 1, _thread("sim.stdin"))
        stdin_feed.append({"line": line, "after_turn": len(turns)})

    def _wait_aut_settled(bridge: Any, count_before: int) -> None:
        """The AUT acts on paused free text on its own thread: wait until it has acted and gone
        quiet for the bridge's own settle (3 s), so the next line lands after it."""
        deadline = time.monotonic() + SETTLE_TIMEOUT_S
        last, quiet_since = count_before, None
        while time.monotonic() < deadline:
            n = len(bridge.action_sink.actions)
            if n != last:
                last, quiet_since = n, time.monotonic()
            elif n > count_before and quiet_since is not None and time.monotonic() - quiet_since >= _AUT_SETTLE_S:
                return
            time.sleep(0.05)
        raise HarnessHang(f"the AUT did not act on the paused free text (actions {count_before}->{last})")

    def send_and_wait(self: Any, text: str, **kwargs: Any) -> dict[str, Any]:
        result = orig_send(self, text, **kwargs)
        turns.append(
            {
                "turn": result["turn"],
                "text": text,
                "response": result["response"],
                "timed_out": result["timed_out"],
                "noop_count": result["noop_count"],
                "actions": [_action_view(a) for a in result["actions"]],
            }
        )
        if arm == "stdin":
            for line in _STDIN_SCRIPT.get(result["turn"], []):
                before = len(self.action_sink.actions)
                _feed(line)
                if line == "hello from the operator":  # paused free text goes to the AUT
                    _wait_aut_settled(self, before)
                    turns[-1]["paused_aut_actions"] = [_action_view(a) for a in self.action_sink.actions[before:]]
        return result

    def bridge_init(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_bridge_init(self, *args, **kwargs)
        bridges.append(self)

    def inject_cli(self: Any, text: str, *args: Any, **kwargs: Any) -> None:
        """Every percept injected into a ``ConversationalSource``: which source (the AUT's, owned by a bridge, or a
        narrator's), the injecting thread, and the text. Recorded in call order; an arm that records them makes that
        order causal (module docstring, CAUSAL SYNCHRONISATION)."""
        bridge_index = next((i for i, b in enumerate(bridges) if b.percept_source is self), None)
        if bridge_index is not None:
            source = f"aut#{bridge_index + 1}"
        else:
            if id(self) not in narrator_sources:
                narrator_sources.append(id(self))
            source = f"narrator#{narrator_sources.index(id(self)) + 1}"
        event = {"inject": source, "thread": threading.current_thread().name, "text": text[:240]}
        if run_stops:  # a campaign runner already ended: was the run already stopping when this landed?
            event["stop_event_set"] = run_stops[-1].is_set()
        events.append(event)
        orig_inject(self, text, *args, **kwargs)

    orig_finish_runner = orch._finish_runner

    def finish_runner(bridge: Any, stop_event: threading.Event, runner: str, outcome: Any, **kwargs: Any) -> None:
        """A campaign runner returned (``orchestrator._finish_runner``): recorded with the stop it handed over."""
        orig_finish_runner(bridge, stop_event, runner, outcome, **kwargs)
        run_stops.append(stop_event)
        events.append(
            {
                "finish_runner": runner,
                "thread": threading.current_thread().name,
                "stop_event_set": stop_event.is_set(),
                "finish_context": dict(bridge.finish_context or {}),
            }
        )

    def finish(self: Any) -> None:
        finish_calls.append(dict(self.finish_context))  # the finish_context each finish() saw
        orig_finish(self)
        if (
            arm == "narrator"
            and len(finish_calls) == 1
            and self.finish_context.get("initiated_by") == "llm_finish_tool"
        ):
            assert isinstance(stdin, QueueStdin)
            stdin.feed("/cancel")  # module docstring: the post-finish grace is otherwise 180 s
            stdin_feed.append({"line": "/cancel", "after_turn": len(turns), "after": "finish_simulation"})

    def display_status(message: str) -> None:
        lifecycle.append(f"status: {message}")
        orig_status(message)

    def display_summary(lines: list[str]) -> None:
        lifecycle.append("summary: " + " | ".join(str(x) for x in lines))
        orig_summary(lines)

    def _report_spy(name: str) -> Any:
        orig = getattr(report_mod, name)

        def spy(*args: Any, **kwargs: Any) -> Any:
            if name == "save_aut_state":
                aut_nac.append(kwargs.get("nac"))
            report_writes.append(name)
            writer_args.append(
                {
                    "writer": name,
                    "args": [_writer_arg(a) for a in args],
                    "kwargs": {k: _writer_arg(v) for k, v in kwargs.items()},
                    "shared_refs": _shared_refs(kwargs),
                }
            )
            return orig(*args, **kwargs)

        return spy

    def _credit_spy(method: str) -> Any:
        orig = getattr(NAc, method)
        sig = inspect.signature(orig)

        def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
            call, store, key = _credit_call(method, sig, self, args, kwargs)
            nac_credits.append((self, call))
            out = orig(self, *args, **kwargs)
            last_post[(id(self), store, key)] = getattr(self, store).get(key)
            if (id(self), store, key) not in credited_keys:  # the key's FIRST write: no decay tick before it
                credited_keys.add((id(self), store, key))
                call["post"] = getattr(self, store).get(key)
            return out

        return spy

    # (object, attribute, replacement), restored in the finally. The sim_logger and report entries are
    # MODULE-attribute spies: the run imports display_status/display_summary and the report writers
    # lazily, inside the function, so it reads them at call time. A slice that binds one of them at
    # import time in a new module must keep reading it through the module, or extend this list.
    patches: list[tuple[Any, str, Any]] = [
        (Bridge, "send_and_wait", send_and_wait),
        (Bridge, "finish", finish),
        (Bridge, "__init__", bridge_init),
        (Source, "inject_cli", inject_cli),
        (orch, "_finish_runner", finish_runner),
        (SL, "display_status", display_status),
        (SL, "display_summary", display_summary),
        (ec_mod, "uuid4", lambda: uuid.UUID(int=next(counter))),
        (LB, "build_primary_router", _build_router),
        *((report_mod, n, _report_spy(n)) for n in _REPORT_WRITERS),
        *((NAc, m, _credit_spy(m)) for m in _NAC_CREDIT_ENTRY_POINTS),
    ]
    saved = [(obj, name, getattr(obj, name)) for obj, name, _ in patches]

    state = _GlobalState()
    for key in [k for k in os.environ if k.startswith("MAXIM_")]:
        del os.environ[key]
    os.environ.update(
        MAXIM_SIM_HARD_ABORT="0",
        MAXIM_DATA_HOME=str(work / "data"),
        MAXIM_REPORT_JSON=str(work / "report_json.json"),
        XDG_CONFIG_HOME=str(work / "xdg-config"),  # module docstring: never the user's config
    )
    os.environ.pop("TERM_PROGRAM", None)
    _reset_caches()
    from maxim.runtime.config_loader import reset_config_cache
    from maxim.utils import seeding

    reset_config_cache()
    seeding._global_seed = None
    os.chdir(work)
    sys.stdin = stdin
    SL.reset_sim_display_state()
    SL.set_interactive_mode(SL.InteractiveMode.OFF)
    json_parse_stats_reset()
    reset_shutdown()
    for obj, name, replacement in patches:
        setattr(obj, name, replacement)
    kwargs = copy.deepcopy(spec.get("kwargs", {}))
    if "arc" in spec:
        (work / "arc.yaml").write_text(spec["arc"])
        kwargs["arc_yaml"] = str(work / "arc.yaml")
    threads_before = threading.enumerate()
    t0 = time.monotonic()
    try:
        result = orch.start_simulation_mode(
            goal=GOAL,
            max_turns=spec["max_turns"],
            sandbox_backend="tmpdir",
            pre_campaign_turns=copy.deepcopy(spec.get("pre_campaign_turns")),
            aut_model=spec.get("aut_model"),
            **kwargs,
        )
        elapsed = time.monotonic() - t0
        stragglers = _join_sim_threads()
        threads_left = _threads_left(threads_before)
        env_after = {k: v for k, v in os.environ.items() if k.startswith("MAXIM_")}
        session_dir = Path(result.session_dir)
        report_json = work / "report_json.json"
        trace = {
            "arm": arm,
            "narrator_calls": router.calls["narrator"],
            "aut_calls": router.calls["aut"],
            "other_llm_calls": {k: v for k, v in router.calls.items() if k not in ("narrator", "aut") and v},
            "unsorted_prompts": router.unsorted,
            "turns": turns,
            "stdin_lines": stdin_feed,
            "bridge_finish_calls": finish_calls,
            "report_writes": report_writes,
            "report_writer_args": writer_args,
            "lifecycle": [_mask_lifecycle(line) for line in lifecycle],
            "result": _result_view(result),
            "report": _report_view(session_dir / "report.json"),
            "report_json_emitted": report_json.is_file(),
            "sim_reports_files": _files(work / "data" / "sim_reports"),
            "data_home_files": _data_home_files(work / "data"),
            "aut_nac": _nac_view(
                session_dir / "aut_nac.json", _aut_nac_config(aut_nac), _last_writes(last_post, aut_nac)
            ),
            "nac_credits": _nac_credits_view(nac_credits, aut_nac),
            "action_log": _action_log_view(session_dir / "actions.jsonl"),
            "live_sim_threads_after": stragglers,
            "threads_left_alive": threads_left,
            "maxim_env_after": env_after,
        }
        if spec.get("events"):
            trace["events"] = events
        canon = _canon(_label_serials(trace, builds), work)
        canon["_elapsed_s"] = round(elapsed, 1)  # dropped by the test before comparing (timing only)
        return canon
    finally:
        for obj, name, orig in saved:
            setattr(obj, name, orig)
        if isinstance(stdin, QueueStdin):
            stdin.eof()  # unblock a reader still waiting for a line
        from maxim.modes import definitions

        RUN_SIDE[arm] = {
            "prompts": router.prompts,
            "tool_descriptions_before": state.tool_descriptions,
            "tool_descriptions_after": copy.deepcopy(definitions.TOOL_DESCRIPTIONS),
        }
        state.restore()


def _label_serials(trace: dict[str, Any], builds: list[int]) -> dict[str, Any]:
    """Each router serial the run built becomes its build label (module docstring, ROUTER BUILDS): in the
    provenance's ``*_router_serial`` values and ``llm_calls`` keys, and in the report writers' arguments."""
    labels = {serial: f"<build-{i}>" for i, serial in enumerate(builds, 1)}

    def label(value: Any) -> Any:
        if isinstance(value, int) and not isinstance(value, bool) and value in labels:
            return labels[value]
        if isinstance(value, str) and value.isdigit() and int(value) in labels:
            return labels[int(value)]
        return value

    prov = trace["report"].get("provenance") or {}
    for key in [k for k in prov if k.endswith("_router_serial")]:
        prov[key] = label(prov[key])
    if isinstance(prov.get("llm_calls"), dict):
        prov["llm_calls"] = {label(k): v for k, v in prov["llm_calls"].items()}
    for write in trace["report_writer_args"]:
        for arg in [*write["args"], *write["kwargs"].values()]:
            if isinstance(arg, dict) and "call_serial" in arg:
                arg["call_serial"] = label(arg["call_serial"])
    return trace


def _join_sim_threads() -> list[str]:
    """Give the run's daemon threads a moment to exit; return any still alive (the trace records
    them: a thread the run leaks into the rest of the test process is a finding)."""
    deadline = time.monotonic() + 10.0
    names = ("sim.aut", "sim.stall", "sim.dm")
    while time.monotonic() < deadline:
        alive = sorted(t.name for t in threading.enumerate() if t.name in names)
        if not alive:
            return []
        time.sleep(0.1)
    return alive


def _files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()) if root.is_dir() else []


# Threads the run leaves alive that are NOT pinned in ``threads_left_alive``, by name prefix, with the reason.
_THREADS_EXCLUDED = {
    "memctx_": "MemoryAgent._context_pool is a CLASS-level ThreadPoolExecutor, created by the first MemoryAgent "
    "in the PROCESS and never shut down (#1263, its memctx part). Whether THIS run adds its threads, and whether "
    "the pool is alive at all after it, depends on what ran earlier in the process (test order and selection): "
    "since #1260 (no AUT follow-up) the precampaign and stdin arms do not start it in a fresh process but see it "
    "alive after an earlier arm or test started it. Neither is pinned; #1263's memctx owner fix is what pins it.",
}


def _threads_left(before: list[threading.Thread]) -> dict[str, Any]:
    """The threads the run started that are still alive after it returned (and after ``_join_sim_threads``),
    numeric suffixes normalised (``Thread-12`` -> ``Thread-<N>``), duplicates kept, sorted. This pins #1263
    (teardown leaves both agents' plan writers running; the AUT DefaultNetwork stops since #1281) as CURRENT behaviour:
    a slice that stops one of them, or leaks a new one, changes it."""
    added = [t for t in threading.enumerate() if t.is_alive() and not any(t is b for b in before)]
    names = sorted(re.sub(r"\d+", "<N>", t.name) for t in added if not t.name.startswith(tuple(_THREADS_EXCLUDED)))
    return {"added": names}


# ``MAXIM_DATA_HOME`` files not pinned, by glob relative to the data home (``*`` matches within ONE path segment,
# ``_segment_match``), with the reason.
_DATA_HOME_EXCLUDED = {
    "agents/*/runtime/*": "#1264: the AUT and narrator loops race on one CWD-relative state_<second>.json (and a "
    "shared .tmp), so how many state files land, and under which second, is a race. NEUTRALISED, not pinned.",
}


def _data_home_files(root: Path) -> list[str]:
    """Every file under the run's ``MAXIM_DATA_HOME`` (stamps and random suffixes scrubbed by ``_canon``)."""
    return [f for f in _files(root) if not any(_segment_match(f, g) for g in _DATA_HOME_EXCLUDED)]


def _segment_match(path: str, pattern: str) -> bool:
    """``fnmatch`` per ``/``-separated segment, with the same segment count: ``*`` never crosses a ``/``."""
    import fnmatch

    parts, pats = path.split("/"), pattern.split("/")
    return len(parts) == len(pats) and all(fnmatch.fnmatchcase(a, b) for a, b in zip(parts, pats))


def _writer_arg(value: Any) -> Any:
    """WHAT a report writer was handed: a scalar or path as itself, an object by its type and its identifying
    attributes (``agent_id``, ``session_id``, ``call_serial``) -- never ``repr`` (memory addresses) or a size
    that counts idle-tick captures (``_HIPPO_PINNED``)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    view: dict[str, Any] = {"type": type(value).__name__}
    for attr in ("agent_id", "session_id", "call_serial"):
        try:
            got = object.__getattribute__(value, attr) if not isinstance(value, _RouterView) else getattr(value, attr)
        except AttributeError:
            continue
        if got is None or isinstance(got, (bool, int, float, str)):
            view[attr] = got
    return view


def _shared_refs(kwargs: dict[str, Any]) -> list[str]:
    """Which of a writer's object arguments hold another of its arguments, by instance attribute
    (``hippocampus._scn -> scn``): the stores handed to ``save_aut_state`` are one agent's where they refer to
    each other, and a store from another agent breaks the relation even where its type is the same. Measured
    2026-10-13: only the Hippocampus holds a store passed beside it (the SCN), so a wrong NAc shows in
    ``aut_nac``'s content instead."""
    objs = {k: v for k, v in kwargs.items() if hasattr(v, "__dict__") and not isinstance(v, (str, Path))}
    return sorted(
        f"{name}.{attr} -> {other}"
        for name, obj in objs.items()
        for attr, held in vars(obj).items()
        for other, target in objs.items()
        if other != name and held is target
    )


# Per-link NAc fields not pinned (``aut_nac``), with the reason, and the file-level ones.
_NAC_DROPPED = {
    "last_observed": "wall clock (epoch seconds)",
    "temporal_delta": "the measured gap between an action and its outcome, in wall-clock seconds",
    "saved_at": "wall clock (file level)",
}


def _nac_view(path: Path, config: Any, last_writes: dict[str, dict[str, Any]]) -> Any:
    """The saved AUT NAc: every causal link's content, times dropped (``_NAC_DROPPED``). The order of the
    links under one signature and of the ``outcome_index`` ids is not pinned (it is the order threads booked
    them in); both are sorted by link id, which is a content hash. ``config`` is the saved NAc's own
    ``NACConfig``: the clamps its tick-decayed maps are bounded by (``_NAC_DECAY_PINNED``); ``last_writes``
    holds, per saved map and saved key, the value right after that NAc's last write to it in the run."""
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    view = {k: v for k, v in data.items() if k not in _NAC_DROPPED}
    view["links"] = {
        sig: sorted(
            ({k: v for k, v in link.items() if k not in _NAC_DROPPED} for link in links), key=lambda x: str(x.get("id"))
        )
        for sig, links in (data.get("links") or {}).items()
    }
    view["outcome_index"] = {k: sorted(v) for k, v in (data.get("outcome_index") or {}).items()}
    for field, cap_field in _NAC_DECAY_PINNED.items():
        if isinstance(view.get(field), dict):
            last = last_writes.get(field, {})
            view[field] = {
                k: _decayed_bound(v, getattr(config, cap_field), last.get(k)) for k, v in view[field].items()
            }
    return view


def _aut_nac_config(aut_nac: list[Any]) -> Any:
    """The ``NACConfig`` of the NAc ``save_aut_state`` was handed (the defaults when it was handed none,
    which the golden's ``report_writer_args`` shows on its own)."""
    from maxim.decisions.nac import NACConfig

    config = getattr(aut_nac[0], "config", None) if len(aut_nac) == 1 else None
    return config if isinstance(config, NACConfig) else NACConfig()


def _decayed_bound(value: Any, cap: float, last_write: Any = None) -> str:
    """A tick-decayed NAc value by sign, bound and decay (``_NAC_DECAY_PINNED``).

    ``"+ in (0, cap], decayed below its last write"`` holds when ``0 < value <= cap`` AND ``value`` is strictly
    nearer zero than ``last_write``, the value right after the run's last credit to that key: proof that at
    least one per-tick decay (agent_loop §8.5) reached the SAVED instance after that credit. The run's 3 s
    settle against ~0.5 s ticks makes "at least one" deterministic in practice; zero ticks (a skipped decay, a
    decay applied to another NAc instance than the one saved) prints ``not decayed`` and fails LOUDLY, never
    silently. The ``(0, cap]`` half is kept as well: ``last_write`` is not traced, so a broken clamp in a later
    write would otherwise pass. A key with no write this run (loaded state) says so, and anything out of bound
    (or zero, which the store never keeps) is printed raw; every such form differs from the pinned one."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return f"OUT OF BOUND {cap}: {value!r}"
    if 0 < value <= cap:
        bound = f"+ in (0, {cap}]"
    elif -cap <= value < 0:
        bound = f"- in [-{cap}, 0)"
    else:
        return f"OUT OF BOUND {cap}: {value!r}"
    if not isinstance(last_write, (int, float)) or isinstance(last_write, bool):
        return f"{bound}, no write this run"
    if 0 < abs(value) < abs(last_write) and (value > 0) == (last_write > 0):
        return f"{bound}, decayed below its last write"
    return f"{bound}, not decayed: {value!r} vs last write {last_write!r}"


# A decayed store (``NAc`` attribute) -> its key in the saved file (``NAc.to_dict``).
_SAVED_FIELD = {
    "_goal_reward_bias": "goal_reward_bias",
    "_reward_bias": "reward_bias",
    "_cluster_reward_bias": "cluster_reward_bias",
    "_percept_valences": "percept_valences",
}


def _saved_key(store: str, key: Any) -> str:
    """A store key as ``NAc.to_dict`` writes it: the goal string, ``aid:nid``, or US-separated (0x1F) triples."""
    if store == "_goal_reward_bias":
        return str(key)
    if store == "_reward_bias":
        return f"{key[0]}:{key[1]}"
    return "\x1f".join(str(k) for k in key)


def _last_writes(last_post: dict[tuple[int, str, Any], Any], aut_nac: list[Any]) -> dict[str, dict[str, Any]]:
    """The saved AUT NAc's value after its last write, per saved map and saved key (``_decayed_bound``)."""
    target = id(aut_nac[0]) if len(aut_nac) == 1 else None
    out: dict[str, dict[str, Any]] = {}
    for (nac_id, store, key), value in last_post.items():
        if nac_id == target and store in _SAVED_FIELD:
            out.setdefault(_SAVED_FIELD[store], {})[_saved_key(store, key)] = value
    return out


def _credit_call(
    method: str, sig: inspect.Signature, nac: Any, args: Any, kwargs: Any
) -> tuple[dict[str, Any], str, Any]:
    """One NAc credit write, by its bound arguments (``_NAC_CREDIT_ENTRY_POINTS``), with the map it writes and
    the key it writes under. A ``temporal_sig`` is a wall-clock stamp, recorded only as present or not."""
    bound = sig.bind(nac, *args, **kwargs)
    bound.apply_defaults()
    store, key_args = _NAC_CREDIT_ENTRY_POINTS[method]
    key = tuple(bound.arguments[a] for a in key_args)
    call: dict[str, Any] = {"method": method}
    for name, value in list(bound.arguments.items())[1:]:  # [0] is self
        if name == "temporal_sig":
            call["temporal_sig"] = value is not None
        elif value is None or isinstance(value, (bool, int, float, str)):
            call[name] = value
        else:
            call[name] = {"type": type(value).__name__}
    return call, store, key[0] if len(key) == 1 else key


def _nac_credits_view(calls: list[tuple[Any, dict[str, Any]]], aut_nac: list[Any]) -> dict[str, Any]:
    """The credit writes, per NAc instance, each in its own call order (one instance is driven by one
    thread: the AUT's by ``sim.aut``, the narrator's by the calling thread, so the order WITHIN an instance
    is causal and the interleaving BETWEEN them is not recorded). ``aut`` is the NAc handed to
    ``save_aut_state``; ``other`` holds every other instance's sequence, sorted."""
    target = aut_nac[0] if len(aut_nac) == 1 else None
    per: dict[int, list[dict[str, Any]]] = {}
    owners: dict[int, Any] = {}
    for nac, call in calls:
        per.setdefault(id(nac), []).append(call)
        owners[id(nac)] = nac
    aut = [c for i, cs in per.items() if owners[i] is target for c in cs]
    other = sorted((cs for i, cs in per.items() if owners[i] is not target), key=lambda cs: json.dumps(cs))
    return {"aut": aut, "other": other}


# The saved NAc maps that decay PER TICK (``decay_goal_reward_biases`` / ``decay_reward_biases`` /
# ``decay_cluster_reward_biases`` / ``decay_percept_valences``, agent_loop §8.5), pinned by key, sign and
# bound, never by value (``_decayed_bound``), with each map's clamp. WHY: the value is credit x
# (1 - 1/tau)^N, and N, the AUT loop ticks between the last credit and the save, is paced by the WALL
# CLOCK: ``loop_gates.pre_tick_gate`` keeps the loop ticking every ~0.5 s for 120 s after any LLM submit
# (``_submitted_recently``), so N counts the ~3.5 s between turn 2's credit and the stop. Measured
# 2026-10-17: the narrator arm's goal bias was 0.2 x 0.98^8 = 0.170152605 locally and 0.2 x 0.98^7 =
# 0.173625107 in CI under xdist load (run 39072778282, attempt 2), red with no behaviour change; a 0.5 s
# later /cancel or a 0.7 s slower tick reproduces it locally. What the values are MADE OF is pinned
# exactly instead: every credit write, with its arguments, in order (``nac_credits``,
# ``_NAC_CREDIT_ENTRY_POINTS``). Every other NAc field stays pinned by value. The underlying src question
# (decay paced by an accidental post-submit wall-clock tick) is its own issue; a fix that anchors decay to
# causal ticks lets this go back to exact values.
_NAC_DECAY_PINNED: dict[str, str] = {  # saved map -> the ``NACConfig`` field holding its clamp
    "goal_reward_bias": "max_reward_bias",  # [-cap, cap] (credit_goal)
    "reward_bias": "max_reward_bias",  # (0, cap] (credit_node)
    "cluster_reward_bias": "max_cluster_reward_bias",  # [-cap, cap] (update_cluster_reward)
    "percept_valences": "max_percept_valence",  # [-cap, cap] (record_percept_valence)
}


# The NAc methods that WRITE a per-tick-decayed map, spied on the class for the run (``nac_credits``):
# ``distribute_reward`` and ``credit_operant_reward`` write only through ``credit_node`` and
# ``update_cluster_reward``, so the leaves cover them. ``update_eligibility`` writes ``_eligibility``
# (decayed by ``decay_eligibility``, session-scoped, not saved): recorded so a change in what earns
# credit eligibility shows too. Each maps to the NAc store it writes and the arguments that make its key.
#
# ``post``: the store's value for that key right after the key's FIRST write in the run (None when the write
# stored nothing, e.g. a gated or clamped-to-zero credit). No decay tick can have touched the key before
# its first write, so the value is clamp(alpha x reward): deterministic, and it pins what the CREDIT does
# (alpha, clamp, sign) beside what the caller asked for. Later writes record no ``post``: each adds to a
# value already decayed by a wall-clock-paced number of ticks (``_NAC_DECAY_PINNED``), and the saved map's
# bound already covers their result. A future mid-arm reload that builds a NEW NAc over loaded state would
# make that instance's first write start from a decayed loaded value, and ``post`` would then fail loudly.
_NAC_CREDIT_ENTRY_POINTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "credit_goal": ("_goal_reward_bias", ("goal_tag",)),
    "credit_node": ("_reward_bias", ("agent_id", "node_id")),
    "update_cluster_reward": ("_cluster_reward_bias", ("agent_id", "cluster_id", "tool_signature")),
    "record_percept_valence": ("_percept_valences", ("agent_id", "entity_class", "failure_mode")),
    "update_eligibility": ("_eligibility", ("agent_id", "node_id")),
}

_ACTION_DROPPED = {"timestamp": "wall clock"}


def _action_log_view(path: Path) -> list[dict[str, Any]] | None:
    """Every ``actions.jsonl`` record (the header included), in file order, times dropped."""
    if not path.is_file():
        return None
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [{k: v for k, v in rec.items() if k not in _ACTION_DROPPED} for rec in lines]


# Fields excluded from the pinned views, each with the reason (also the report's own docstring).
_RESULT_DROPPED = {
    "duration_s": "wall clock",
    "introspector": "a live object (its benchmark snapshot is pinned as subsystem_snapshot)",
}
_REPORT_DROPPED = {
    "aut_memories_formed": "idle-tick captures race the next percept (see _HIPPO_PINNED)",
    "timestamp": "wall clock",
    "ts": "wall clock",
    "duration_s": "wall clock",
    "latency_p50_ms": "wall clock",
    "latency_p99_ms": "wall clock",
}


# Inside the result's introspection blocks (``subsystem_snapshot`` and the pre-campaign
# ``campaign_analysis``), measured to vary between identical runs (3 runs per arm, 2026-10-12):
#   - the SCN ``temporal`` block is calendar- and wall-clock-derived (``current_phase`` holds the
#     hour/day/month bins, ``phase_coherence`` the elapsed time), so only its keys are pinned;
#   - the hippocampus MEMORY COUNT is wall-clock-derived. After each action burst the AUT's
#     MemoryAgent captures the current percept on every idle 0.5 s tick, up to six times (~3 s),
#     and the next percept lands at the bridge's 3 s settle edge, so whether the 6th idle capture
#     happens first is a race (measured: 21 vs 24 memories for the same script). Only the
#     hippocampus counters that do not count idle captures are pinned (``_HIPPO_PINNED``: the
#     action-outcome ``successful`` count and the store's configuration). The memory count is
#     masked wherever it is printed (``_LIFECYCLE_MASKS``) or reported (``aut_memories_formed``,
#     ``system_stats.hippocampus_memories``).
# The NAc links, ATL concepts and action-outcome counts did NOT vary and stay pinned.
_HIPPO_PINNED = ("successful", "compressed_memories", "graph_enabled", "over_capacity_inserts_this_process")
# Wall-clock text in the lifecycle lines: the pre-campaign "AUT: ... 4441ms" status, the printed
# report's "Duration: 14.2s", and the memory counts above.
_LIFECYCLE_MASKS = (
    (re.compile(r"\b\d+ms\b"), "<MS>"),
    (re.compile(r"Duration: [\d.]+s"), "Duration: <S>"),
    (re.compile(r"\b\d+ (episodic )?memories\b"), r"<N> \1memories"),
    (re.compile(r"AUT Memories: \d+"), "AUT Memories: <N>"),
)


def _introspection_view(block: Any) -> Any:
    if not isinstance(block, dict):
        return block
    view = dict(block)
    if isinstance(view.get("temporal"), dict):
        view["temporal"] = {"keys": sorted(view["temporal"])}
    if isinstance(view.get("hippocampus_stats"), dict):
        view["hippocampus_stats"] = {k: view["hippocampus_stats"].get(k) for k in _HIPPO_PINNED}
    if isinstance(view.get("system_stats"), dict):
        view["system_stats"] = {k: v for k, v in view["system_stats"].items() if k != "hippocampus_memories"}
    if isinstance(view.get("turns"), list):  # campaign_analysis: per-turn wall-clock duration
        view["turns"] = [{k: v for k, v in t.items() if k != "duration_ms"} for t in view["turns"]]
    return view


def _result_view(result: Any) -> dict[str, Any]:
    from dataclasses import fields

    view: dict[str, Any] = {}
    for f in fields(result):
        if f.name in _RESULT_DROPPED:
            continue
        value = getattr(result, f.name)
        if f.name == "actions":
            value = [{k: v for k, v in a.items() if k != "timestamp"} for a in value]
        elif f.name in ("subsystem_snapshot", "campaign_analysis"):
            value = _introspection_view(value)
        view[f.name] = value
    return view


def _mask_lifecycle(line: str) -> str:
    for pattern, placeholder in _LIFECYCLE_MASKS:
        line = pattern.sub(placeholder, line)
    return line


# Provenance keys pinned by NAME only (value masked), with the reason. They describe the checkout and the
# interpreter the golden runs from, not the run: measured 2026-10-13, two narrator runs differed in nothing
# else, and a regen box, a worktree and CI each differ in these.
_PROVENANCE_MASKED = {
    "python": "the interpreter path",
    "pythonpath": "the PYTHONPATH the golden runs under (a worktree's, CI's)",
    "executed_maxim_file": "the checkout's path",
    "executed_git_hash": "HEAD of the checkout",
    "end_executed_git_hash": "HEAD of the checkout",
    "code_tree_sha256": "the checkout's src/ tree",
    "end_code_tree_sha256": "the checkout's src/ tree",
    "maxim_version": "moves with every release cut",
    "working_tree_dirty_src_scripts": "a slice's uncommitted work-in-progress tree is dirty, its commit is not",
}
_MASKED = "<MASKED>"


# Provenance keys masked only when the run READ scenario files (``provenance.scenario_inputs`` non-empty), with the
# reason. Measured 2026-10-24 (generative arm, two identical runs): the hash covers each input's ABSOLUTE path, and
# the arc file lives in the per-run work dir. The inputs themselves stay pinned (path scrubbed, sha256 by value).
_PROVENANCE_PATH_BOUND = {
    "scenario_sha256": "hashes goal + scenario_inputs, whose paths are absolute (the work dir, the checkout)",
}


def _report_view(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    view = {k: v for k, v in data.items() if k not in _REPORT_DROPPED and k != "provenance"}
    prov = data.get("provenance") or {}
    masked = set(_PROVENANCE_MASKED) | (set(_PROVENANCE_PATH_BOUND) if prov.get("scenario_inputs") else set())
    view["provenance"] = {k: (_MASKED if k in masked else v) for k, v in prov.items()}
    return view

"""Offline driver for the REAL ``start_simulation_mode`` (1.3.2 orchestrator decomposition, slice 0).

Used by ``test_orchestrator_characterization_golden.py``, the characterization gate the orchestrator's
extraction slices must keep green UNCHANGED (``docs/plans/roadmap_1_3_x.md`` §"Coverage first, then
extract"). It is the orchestrator's counterpart of ``tests/unit/_loop_harness.py``.

WHAT IT DRIVES. The real ``maxim.simulation.orchestrator.start_simulation_mode``, end to end. That
covers the AUT build, the narrator pipeline, the ``sim.aut`` thread, campaign dispatch, the stdin
reader, the stall detector, the narrator loop on the calling thread, teardown, the report and the
``SimulationResult``. Nothing in ``src/`` is stubbed except the LLM. ``FakeRouter`` is patched in at
``maxim.runtime.lane_backends.build_primary_router``, the seam ``test_api_expansion.py`` already
uses. Three arms (owner decision (a), 2026-10-13):

  1. ``precampaign``: three ``pre_campaign_turns`` and no narrator LLM. The runner ends the run.
  2. ``narrator``: the fake narrator calls ``send_message`` twice and then ``finish_simulation``.
     This arm exercises the narrator loop on the calling thread (§12), the stall-detector polling
     and the LLM-finish path.
  3. ``stdin``: operator commands fed through a queue-backed ``sys.stdin``: ``/status``, ``/pause``,
     free text while paused (which goes to the AUT), ``/resume``, free text while running (which
     goes to the narrator as guidance), and ``/cancel``.

The TTY paths (``_stdin_reader_raw``, the post-sim raw-mode prompt) are deliberately left
uncharacterized (owner decision (e)). Every arm runs non-interactive (``InteractiveMode.OFF``, the
``--interactive false`` path), so the line reader is the one exercised.

THREADS STAY ON, ON THE WALL CLOCK (unlike the loop golden's step clock). The run keeps its real
threads and its real 3 s settle, so the trace records only CAUSALLY ORDERED facts:

  - each role's LLM calls, in that role's own order;
  - each bridge turn, its percept text and what the AUT did in it;
  - the ``display_status``/``display_summary`` lifecycle lines;
  - ``finish_reason`` and ``finish_context``;
  - the report writers called, in order, and the ``sim_reports`` file set;
  - the deterministic ``report.json`` fields;
  - the ``SimulationResult`` with its timestamps dropped.

It never records a cross-thread interleaving or a stall-detector injection. The arms stay shorter
than the detector's first diversity checkpoint (3 turns) and its idle threshold (30 s).

CAUSAL SYNCHRONISATION. Two class-level hooks on ``SimulationBridge``; class attributes hold
wherever a slice moves the CALLER:

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
(``_RESULT_DROPPED``, ``_REPORT_DROPPED``, ``_HIPPO_PINNED``, ``_LIFECYCLE_MASKS``, ``_scrub``):

  - session ids, stamps, tmpdir suffixes and the work path become placeholders;
  - times, latencies and durations are dropped;
  - provenance is pinned by its keys;
  - the SCN's calendar-derived ``temporal`` block is pinned by its keys;
  - the hippocampus memory count, which counts idle-tick captures that race the next percept, is masked;
  - EC node ids come from a counter, as in phase 1.

The determinism tests cover the rest: the three arms run back to back in one process against the
golden, and each arm runs in fresh interpreters under two hash seeds.

STATED LIMIT (wall clock). A turn closes after 3 s with no new AUT action, as it does live. The AUT's
reply chain is two actions: ``respond``, then its follow-up ``say``, about 0.5 s apart. A box loaded
enough to stretch that gap past 3 s would split a turn, and the golden would go red. Check what else
is running before reading such a failure as a regression.

GLOBAL STATE. A run mutates process-global state, all of which ``run_arm`` snapshots and restores:

  - ``modes.definitions.TOOL_DESCRIPTIONS``, the ``maxim`` logger level and every ``MAXIM_*`` env var
    (including ``MAXIM_ALLOW_BASH`` and ``MAXIM_REPORT_JSON``);
  - the sim display/interactive globals, the JSON parse counters, the LLM cancellation flag and the
    lane ``MetricsRegistry`` the report reads (a fresh one for the run: an earlier test's lane metrics
    leaked into ``report.lane_breakdown`` in the full suite);
  - ``sys.stdin``, the CWD and the path caches.

``MAXIM_SIM_HARD_ABORT=0`` is mandatory: the hard-abort path calls ``os._exit(4)``, which would kill
pytest.
"""

from __future__ import annotations

import copy
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
ARMS: dict[str, dict[str, Any]] = {
    "precampaign": {"max_turns": 10, "pre_campaign_turns": _PRECAMPAIGN_TURNS},
    "narrator": {"max_turns": 10},
    "stdin": {"max_turns": 10},
}
# The report writers the run calls, in order (a removed or reordered save shows even when a later
# save of the same file masks it on disk).
_REPORT_WRITERS = ("save_report", "save_action_log", "save_aut_state", "analyze_simulation", "emit_report_json")
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
        }
        self.unsorted: list[str] = []
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
        if call["executed"] == "respond":
            return _act("say", text="That is all.")
        return {"action": None, "confidence": 0.9, "reasoning": "done", "ready_to_act": False}

    def _answer_fear(self, call: dict[str, Any]) -> dict[str, Any]:
        return {"findings": []}

    def _answer_roundup(self, call: dict[str, Any]) -> dict[str, Any]:
        return {"summary": "golden roundup", "issues": ["none observed"], "recommendations": ["keep going"]}

    def _answer_manifest(self, call: dict[str, Any]) -> dict[str, Any]:
        return {"entities": []}


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
        from maxim.models.language import lane_metrics

        with lane_metrics._registry_lock:
            lane_metrics._global_registry = self.lane_registry
        _reset_caches()


def run_arm(arm: str, workdir: Path) -> dict[str, Any]:
    """Drive the real ``start_simulation_mode`` for one arm; return the CANONICAL trace."""
    assert_this_checkout()
    import maxim.runtime.lane_backends as LB
    from maxim.models.language.json_parser import json_parse_stats_reset
    from maxim.models.language.cancellation import reset_shutdown
    from maxim.similarity import ec as ec_mod
    from maxim.simulation import bridge as bridge_mod
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
    stdin_feed: list[dict[str, Any]] = []
    stdin = QueueStdin() if arm in ("narrator", "stdin") else io.StringIO("")
    counter = iter(range(1, 1_000_000))
    Bridge = bridge_mod.SimulationBridge
    orig_send, orig_finish = Bridge.send_and_wait, Bridge.finish
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
            report_writes.append(name)
            return orig(*args, **kwargs)

        return spy

    # (object, attribute, replacement), restored in the finally. The sim_logger and report entries are
    # MODULE-attribute spies: the run imports display_status/display_summary and the report writers
    # lazily, inside the function, so it reads them at call time. A slice that binds one of them at
    # import time in a new module must keep reading it through the module, or extend this list.
    patches: list[tuple[Any, str, Any]] = [
        (Bridge, "send_and_wait", send_and_wait),
        (Bridge, "finish", finish),
        (SL, "display_status", display_status),
        (SL, "display_summary", display_summary),
        (ec_mod, "uuid4", lambda: uuid.UUID(int=next(counter))),
        (LB, "build_primary_router", lambda *a, **k: (router, None)),
        *((report_mod, n, _report_spy(n)) for n in _REPORT_WRITERS),
    ]
    saved = [(obj, name, getattr(obj, name)) for obj, name, _ in patches]

    state = _GlobalState()
    for key in [k for k in os.environ if k.startswith("MAXIM_")]:
        del os.environ[key]
    os.environ.update(
        MAXIM_SIM_HARD_ABORT="0",
        MAXIM_DATA_HOME=str(work / "data"),
        MAXIM_REPORT_JSON=str(work / "report_json.json"),
    )
    os.environ.pop("TERM_PROGRAM", None)
    _reset_caches()
    os.chdir(work)
    sys.stdin = stdin
    SL.reset_sim_display_state()
    SL.set_interactive_mode(SL.InteractiveMode.OFF)
    json_parse_stats_reset()
    reset_shutdown()
    for obj, name, replacement in patches:
        setattr(obj, name, replacement)
    t0 = time.monotonic()
    try:
        result = orch.start_simulation_mode(
            goal=GOAL,
            max_turns=spec["max_turns"],
            sandbox_backend="tmpdir",
            pre_campaign_turns=copy.deepcopy(spec.get("pre_campaign_turns")),
        )
        elapsed = time.monotonic() - t0
        stragglers = _join_sim_threads()
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
            "lifecycle": [_mask_lifecycle(line) for line in lifecycle],
            "result": _result_view(result),
            "report": _report_view(Path(result.session_dir) / "report.json"),
            "report_json_emitted": report_json.is_file(),
            "sim_reports_files": _files(work / "data" / "sim_reports"),
            "live_sim_threads_after": stragglers,
        }
        canon = _canon(trace, work)
        canon["_elapsed_s"] = round(elapsed, 1)  # dropped by the test before comparing (timing only)
        return canon
    finally:
        for obj, name, orig in saved:
            setattr(obj, name, orig)
        if isinstance(stdin, QueueStdin):
            stdin.eof()  # unblock a reader still waiting for a line
        state.restore()


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


def _report_view(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    view = {k: v for k, v in data.items() if k not in _REPORT_DROPPED and k != "provenance"}
    view["provenance_keys"] = sorted((data.get("provenance") or {}).keys())
    return view

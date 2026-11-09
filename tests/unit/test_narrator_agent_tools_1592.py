"""#1592: the diversity checkpoint and the stall nudges must not hand the narrator the agent's tool names, and a retry
after the narrator proposed an agent tool must say whose tool it was and name the narrator's own, without echoing it.

The Exp 63b smoke (big-mac-mini, b1120188, 2026-11-03, mistral-7b) stopped at phase 0 with ``planning_failed``: the
stall detector's DIVERSITY CHECKPOINT listed the agent under test's tools by name (``sense_tools (1x), ...``), the
narrator proposed one of them 3 s later, and each of its three retries picked another. The checkpoint is a CLI input,
so it is data and the #1042 roster gate never sees it. Owner decision 2026-11-05: the checkpoint gives counts only and
points at the narrator's own ``observe_actions``.

The composed test drives the real ``start_simulation_mode`` through the orchestrator harness's
``agent_tool_after_checkpoint`` probe arm (``tests/unit/_orch_harness.py``): the real stall detector builds the
checkpoint over the AUT's real actions, the real loop drops the narrator's agent-tool proposal and requeues it
(``agent_loop`` -> ``loop_controller._handle_planning_failure`` -> ``LLMWorker.requeue_request``), and the real prompt
builder renders the retry. Only the LLM is faked.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from tests.unit._orch_harness import AGENT_TOOL, RUN_SIDE, assert_this_checkout, canonical_json, run_arm
from tests.unit.test_prompt_names_own_tools import _mentions, _narrator_roster

ARM = "agent_tool_after_checkpoint"
# The correction's sentence that says whose tool it was (``prompt_builder.PLANNING_CORRECTIONS``).
WHOSE = "the agent under test's"


def _run(workdir: Path) -> dict[str, Any]:
    assert_this_checkout()  # a worktree without PYTHONPATH would otherwise test main's code
    return json.loads(canonical_json(run_arm(ARM, workdir)))


def _checkpoints(trace: dict[str, Any]) -> list[str]:
    return [e["text"] for e in trace["events"] if e.get("text", "").startswith("DIVERSITY CHECKPOINT")]


@pytest.fixture(scope="module")
def composed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _run(tmp_path_factory.mktemp("orch_1592"))


@pytest.mark.timeout(240)
def test_the_run_reaches_the_retry_through_the_shipped_drop(composed: dict[str, Any]) -> None:
    """The arm's premise, not the fix: without it the two tests below would pass vacuously."""
    calls = [(c["kind"], c.get("executed"), (c["answer"] or {}).get("tool_name")) for c in composed["narrator_calls"]]
    assert ("followup", "observe_actions", AGENT_TOOL) in calls, calls
    assert calls[-1] == ("followup", "observe_actions", "finish_simulation"), calls
    aut_tools = [(c["answer"] or {}).get("tool_name") for c in composed["aut_calls"]]
    assert AGENT_TOOL in aut_tools, f"the AUT used the tool the narrator lacks: {aut_tools}"
    assert len(_checkpoints(composed)) == 1


def _held(prompt: str) -> str:
    """The prompt's held inputs, whole (the trace's inject events keep only their first 240 characters)."""
    (held,) = re.findall(r"=== New Instructions ===\n(.*?)(?:\n=== |\Z)", prompt, re.S)
    return held


def _retry() -> str:
    return RUN_SIDE[ARM]["prompts"]["narrator"][-1]


@pytest.mark.timeout(240)
def test_the_checkpoint_names_no_tool_of_the_agent(composed: dict[str, Any]) -> None:
    """Read whole, from the follow-up prompt it rode (#1592 executor review: the event text is cut at 240)."""
    assert len(_checkpoints(composed)) == 1
    checkpoint = _held(_retry())
    assert checkpoint.startswith("DIVERSITY CHECKPOINT") and len(checkpoint) > 240, checkpoint
    assert _mentions(checkpoint, _narrator_roster()) == [], checkpoint
    assert AGENT_TOOL not in checkpoint, checkpoint
    assert "observe_actions" in checkpoint, "it points at the narrator's own tool for the details"


@pytest.mark.timeout(240)
def test_the_retry_says_the_tool_was_the_agents_and_names_the_narrators_own(composed: dict[str, Any]) -> None:
    correction = _retry()[_retry().index("=== Correction ===") :]
    assert WHOSE in correction, correction
    # No echo of the rejected name (owner decision 2026-11-05), and its own tools once, last.
    assert AGENT_TOOL not in correction and correction.count("'send_message'") == 1, correction
    assert correction.rstrip().endswith(".") and correction.rindex("'send_message'") > correction.index(WHOSE)


@pytest.mark.xfail(strict=True, reason="#1605: the observe_actions result lists the agent's tools above the correction")
@pytest.mark.timeout(240)
def test_the_retry_shows_no_agent_tool_above_its_correction(composed: dict[str, Any]) -> None:
    """d9463b45's red gate as written (#1592 architecture review F3: a gate that did not flip is not re-pointed)."""
    retry = _retry()
    assert _mentions(retry[: retry.index("=== Correction ===")], _narrator_roster()) == []


def test_a_plan_retry_after_an_agent_tool_says_whose_it_was_and_names_the_narrators_own() -> None:
    """The plan path: the checkpoint arrived with no follow-up in flight, so it is the plan's request."""
    from maxim.agents.llm_worker import LLMWorker
    from tests.unit.test_narrator_reliability import _narrator_request, _prompt

    request = _narrator_request(followup=False)
    LLMWorker._add_failed_tool_feedback(request, AGENT_TOOL)
    LLMWorker._add_planning_correction(request, "unregistered_tool_proposed")
    prompt = _prompt(request)
    correction = prompt[prompt.index("=== Correction ===") :]
    assert WHOSE in correction, prompt[-800:]
    assert "'send_message'" in correction, "the retry names the narrator's own tools"
    assert AGENT_TOOL not in prompt, "no echo of the rejected name on the plan path either"


def test_every_planning_correction_ends_with_the_agents_own_tools() -> None:
    """The follow-up correction lists the own tools once because every correction carries them (#1592 review)."""
    from maxim.agents.prompt_builder import PLANNING_CORRECTIONS

    assert all(text.rstrip().endswith("{own}.") for text in PLANNING_CORRECTIONS.values()), PLANNING_CORRECTIONS


@pytest.mark.parametrize("reason", [None, "proposal_without_action"])
def test_with_the_echo_on_the_own_tools_are_listed_once_and_last(reason: str | None) -> None:
    """The echo-on path (any reason but ``unregistered_tool_proposed``): the rejected name is echoed, and the
    agent's own tools follow it once, as the last tool names in the correction (#1592 delta-round NIT 1)."""
    from maxim.agents.llm_worker import LLMWorker
    from maxim.agents.prompt_builder import build_followup_retry_correction
    from tests.unit.test_narrator_reliability import _narrator_request

    request = _narrator_request(followup=True)
    LLMWorker._add_failed_tool_feedback(request, AGENT_TOOL)
    if reason:
        LLMWorker._add_planning_correction(request, reason)
    correction = build_followup_retry_correction(request)
    own = ", ".join(repr(n) for n in sorted(request.available_tools))
    assert repr(AGENT_TOOL) in correction, "the echo is on for this request"
    assert correction.count(own) == 1 and correction.index(own) > correction.index(repr(AGENT_TOOL)), correction
    tail = correction[correction.index(own) + len(own) :]
    assert _mentions(tail, _narrator_roster()) == [] and AGENT_TOOL not in tail, correction


def _opts_into_planning_liveness(source: str) -> bool:
    """A CALL that passes ``planning_liveness`` anything but a literal False/None or the enclosing function's own
    same-named PARAMETER (a pass-through, as in ``agent_loop``/``loop_setup``). A computed flag, a same-named LOCAL,
    or a ``**{"planning_liveness": ...}`` literal counts as opting in (#1592 delta-round NIT 2)."""
    import ast

    def off(v: ast.expr) -> bool:
        return isinstance(v, ast.Constant) and v.value in (False, None)

    def scan(scope: ast.AST, params: frozenset[str]) -> bool:
        for node in ast.iter_child_nodes(scope):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                a = node.args
                names = {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
                assigned = {
                    t.id
                    for n in ast.walk(node)
                    if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr))
                    for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
                    if isinstance(t, ast.Name)
                }
                if scan(node, frozenset(names - assigned)):
                    return True
                continue
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "planning_liveness":
                        v = kw.value
                        passthrough = isinstance(v, ast.Name) and v.id == "planning_liveness" and v.id in params
                        if not (off(v) or passthrough):
                            return True
                    elif kw.arg is None and isinstance(kw.value, ast.Dict):
                        for k, v in zip(kw.value.keys, kw.value.values):
                            if isinstance(k, ast.Constant) and k.value == "planning_liveness" and not off(v):
                                return True
            if scan(node, params):
                return True
        return False

    return scan(ast.parse(source), frozenset())


@pytest.mark.parametrize(
    ("source", "opts"),
    [
        ("f(planning_liveness=True)", True),
        ("f(planning_liveness=False)", False),
        ("f(planning_liveness=None)", False),
        ("def g(planning_liveness): f(planning_liveness=planning_liveness)", False),
        ("def g(planning_liveness=False): f(planning_liveness=planning_liveness)", False),
        ("def g(*, planning_liveness): f(planning_liveness=planning_liveness)", False),
        ("def g(planning_liveness):\n    planning_liveness = True\n    f(planning_liveness=planning_liveness)", True),
        ("def g():\n    planning_liveness = True\n    f(planning_liveness=planning_liveness)", True),
        ("planning_liveness = True\nf(planning_liveness=planning_liveness)", True),
        ("f(**{'planning_liveness': True})", True),
        ("f(**{'planning_liveness': False})", False),
        ("def g(on): f(planning_liveness=on)", True),
    ],
)
def test_the_planning_liveness_scan_sees_every_way_to_opt_in(source: str, opts: bool) -> None:
    assert _opts_into_planning_liveness(source) is opts


def test_only_the_sim_narrator_opts_into_planning_liveness() -> None:
    """``PLANNING_CORRECTIONS`` speak to the narrator ("the agent under test"): only a planning-liveness loop records
    one, and the orchestrator's narrator loop must stay the only one that opts in (#1592 architecture review F5)."""
    root = Path(__file__).resolve().parents[2]
    callers = sorted(
        p.relative_to(root).as_posix()
        for d in ("src", "scripts")
        for p in (root / d).rglob("*.py")
        if _opts_into_planning_liveness(p.read_text(encoding="utf-8"))
    )
    assert callers == ["src/maxim/simulation/orchestrator.py"], callers


def _detector(actions: list[Any]) -> tuple[Any, list[str]]:
    import threading
    from types import SimpleNamespace

    from maxim.simulation.stall_detector import StallDetector

    injected: list[str] = []
    detector = StallDetector(
        stop_event=threading.Event(),
        bridge=SimpleNamespace(turn_count=1, turn_in_progress=False, get_all_actions=lambda: list(actions)),
        orchestrator_source=SimpleNamespace(inject_cli=lambda text, **_k: injected.append(text)),
        orch_executor=SimpleNamespace(_tools_attempted=[]),
        goal="check the nudges",
        max_turns=0,
        is_interactive=False,
        is_observe_only=False,
        paused=[False],
        sim_prompt_handler=None,
        display_status=lambda _m: None,
        emit=lambda *_a, **_k: None,
        get_active_display=lambda: None,
    )
    return detector, injected


@pytest.mark.parametrize("ping_pong", [False, True])
def test_every_nudge_the_detector_sends_names_no_tool_of_the_agent(ping_pong: bool) -> None:
    """The call site, not only the builder (#1592 executor review: nothing guarded ``_nudge`` past 240 characters)."""
    from types import SimpleNamespace

    agent = ["sense_tools", "base_humanoid_turn_left"]
    detector, injected = _detector([SimpleNamespace(tool_name=t, blocked=False) for t in agent])
    for _ in range(3):  # nudges 1 and 2, then the REPEATED STALL
        detector._nudge(ping_pong, 6, 2)
    assert len(injected) == 3 and injected[2].startswith("SYSTEM: REPEATED STALL (#3)")
    for text in injected:
        assert not any(t in text for t in agent) and _mentions(text, _narrator_roster()) == [], text
        assert "respond" not in text, "not the decoy either, even as a prohibition (#1042)"


def test_an_idle_nudge_with_no_agent_actions_claims_no_last_action() -> None:
    detector, injected = _detector([])
    detector._nudge(False, 0, 0)
    assert "0 AUT actions so far)" in injected[0] and "last action" not in injected[0], injected[0]

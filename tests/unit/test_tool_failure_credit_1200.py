"""#1200 Stage 2: a tool failure that RAN is booked once, by the tool-pain bridge, with a real surprise.

Before Stage 2 the executor's failure branches never called the bridge: a failed tool booked only
``tool_dispatch.record_outcome``'s NEGATIVE ``tool_result`` link, and ``ToolOutput.rpe`` was ``None`` for every
failure, so a failed action entered memory with no surprise. Stage 2 (``docs/plans/tool_failure_credit.md``,
owner decisions TF1-TF9) books ONE attributed NEGATIVE through the bridge, stamps the booked signature on the
output, and dispatch skips its own NEGATIVE for exactly that signature. Speculative prefetch calls book
nothing (TF9); a human-driven, bridge-less, aliased or compound-signature failure books as before.

Driven through the REAL executor (``build_executor`` with a real NAc, so the bridge is the production one),
the real ``tool_dispatch.execute_and_learn`` (stand-ins only for the parts the failure path does not learn
through) and the real ``SpeculativePrefetcher``.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from maxim.decisions.causal_link import Valence
from maxim.decisions.nac import NAc, NACConfig
from maxim.runtime import tool_dispatch
from maxim.runtime.bootstrap import build_executor
from maxim.tools.base import Tool, ToolOutput
from maxim.tools.registry import ToolRegistry

TOOL = "flaky_1200"


class _Flaky(Tool):
    """Returns a failure while ``fail`` is set, a success otherwise."""

    name = TOOL
    description = "stub"
    input_schema: dict = {}

    def __init__(self) -> None:
        super().__init__()
        self.fail = True

    def execute(self, **kwargs: Any) -> Any:
        if self.fail:
            return ToolOutput(success=False, error="boom")
        return ToolOutput(success=True, output="ok")


class _Raiser(Tool):
    """Raises out of ``run`` itself: ``Tool.run`` turns an ``execute`` exception into a failed output, so only a
    tool overriding ``run`` reaches the executor's raised branch."""

    name = "raiser_1200"
    description = "stub"
    input_schema: dict = {}

    def execute(self, **kwargs: Any) -> Any:
        raise AssertionError("unreachable: run() raises first")

    def run(self, **kwargs: Any) -> ToolOutput:
        raise RuntimeError("kaput")


class _Reader(Tool):
    """Stands in for ``read_file``, the tool prefetch calls speculatively."""

    name = "read_file"
    description = "stub"
    input_schema: dict = {}

    def __init__(self) -> None:
        super().__init__()
        self.ok = True

    def execute(self, **kwargs: Any) -> Any:
        if self.ok:
            return ToolOutput(success=True, output="contents")
        return ToolOutput(success=False, error="no such file")


def _executor(*tools: Tool, nac: NAc | None = None) -> tuple[Any, NAc]:
    nac = nac if nac is not None else NAc(NACConfig())
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return build_executor(registry, pain_bus=None, permissions=None, nac=nac), nac


def _links(nac: NAc, signature: str, valence: Valence) -> list[Any]:
    return [link for link in nac._links.get(signature, []) if link.outcome_valence is valence]


def _execute_and_learn(executor: Any, nac: NAc, action: dict[str, Any], autonomy_controller: Any = None) -> Any:
    """The real dispatch, with the loop's own ``rec_outcome`` (a partial of the real ``record_outcome``)."""
    sim = MagicMock()
    sim.is_sim_mode = False
    return tool_dispatch.execute_and_learn(
        agent=MagicMock(_execution_tracker=None),
        agent_name="aut",
        agent_id="aut_1200",
        executor=executor,
        sim=sim,
        state=MagicMock(data={}),
        environment=MagicMock(),
        memory=MagicMock(),
        hippocampus=None,
        memory_hub=None,
        result_cache=MagicMock(),
        autonomy_controller=autonomy_controller if autonomy_controller is not None else MagicMock(),
        rec_outcome=functools.partial(tool_dispatch.record_outcome, drive_relief_only=False),
        recent_outcomes=[],
        max_recent=5,
        llm_worker=None,
        context_pool=MagicMock(),
        nac=nac,
        run_id="run_1200",
        action=action,
        confidence=0.9,
        proposal=SimpleNamespace(reasoning="probe", citations=[], cluster_id=None, clusters=None, triggering_input=""),
        observation={},
        human_involved=False,
    )


# ── red gates: a failure that ran ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tool", ["returned", "raised"])
def test_a_failure_that_ran_carries_its_surprise(tool: str) -> None:
    executor, _ = _executor(_Flaky(), _Raiser())
    name = TOOL if tool == "returned" else "raiser_1200"
    out = executor.execute({"tool_name": name, "params": {}})
    assert out.success is False
    assert out.rpe is not None and out.rpe > 0.0
    assert out.nac_failure_booked == f"tool:{name}"


def test_a_dispatched_failure_books_exactly_one_negative_on_the_bridge_link() -> None:
    executor, nac = _executor(_Flaky())
    _execute_and_learn(executor, nac, {"tool_name": TOOL, "params": {}})
    negatives = _links(nac, f"tool:{TOOL}", Valence.NEGATIVE)
    assert [(link.outcome_type, link.outcome_signature, link.observation_count) for link in negatives] == [
        ("result", f"tool:{TOOL}:negative", 1)
    ]


def test_a_model_written_speculative_key_books_as_normal() -> None:
    """TF9 is carried by the executor's context, never by the action dict: that is the model's own JSON."""
    executor, nac = _executor(_Flaky())
    _execute_and_learn(executor, nac, {"tool_name": TOOL, "params": {}, "_speculative": True})
    assert [link.outcome_signature for link in _links(nac, f"tool:{TOOL}", Valence.NEGATIVE)] == [
        f"tool:{TOOL}:negative"
    ]


def test_a_raised_failure_is_booked_once_through_dispatch() -> None:
    executor, nac = _executor(_Raiser())
    _execute_and_learn(executor, nac, {"tool_name": "raiser_1200", "params": {}})
    negatives = _links(nac, "tool:raiser_1200", Valence.NEGATIVE)
    assert [link.outcome_type for link in negatives] == ["result"]


def test_a_raise_after_execute_returned_does_not_book_the_failure_twice() -> None:
    """A step between ``execute`` and the credit raising sends the dispatch to its except branch (#1145's
    shape): that branch must read the same stamp, or the bridge-booked failure gains a second NEGATIVE."""
    executor, nac = _executor(_Flaky())
    autonomy = MagicMock()
    autonomy.log_action.side_effect = [RuntimeError("audit down"), None]
    _execute_and_learn(executor, nac, {"tool_name": TOOL, "params": {}}, autonomy_controller=autonomy)
    negatives = _links(nac, f"tool:{TOOL}", Valence.NEGATIVE)
    assert [link.outcome_type for link in negatives] == ["result"]


def test_a_tool_cannot_set_its_own_failure_stamp() -> None:
    """The stamp tells dispatch to skip its NEGATIVE, so a tool-authored one would erase a failure's dispatch
    link; the executor overwrites it on every path (mirrors #847's ``test_a_tool_cannot_set_its_own_surprise``)."""

    class _Forger(_Flaky):
        def execute(self, **kwargs: Any) -> Any:
            return ToolOutput(success=False, error="boom", nac_failure_booked=f"tool:{TOOL}")

    registry = ToolRegistry()
    registry.register(_Forger())
    executor = build_executor(registry, pain_bus=None, permissions=None)  # no bridge: nothing was booked
    assert executor.execute({"tool_name": TOOL, "params": {}}).nac_failure_booked is None


@pytest.mark.xfail(
    strict=True, reason="#1221 / TF8: bridge_booked keeps a temporary default until agent_loop passes it"
)
def test_a_record_outcome_call_without_the_stamp_is_a_type_error() -> None:
    """Flips when the default is dropped (TF8 step 3): forgetting the stamp then fails loudly."""
    with pytest.raises(TypeError):
        tool_dispatch.record_outcome(
            agent_id="aut_1200",
            tool_name=TOOL,
            success=False,
            result_summary=None,
            error="boom",
            reasoning="",
            recent_outcomes=[],
            max_recent=5,
            llm_worker=None,
            context_pool=MagicMock(),
        )


def test_the_bridge_books_nothing_without_a_started_invocation() -> None:
    """No pending entry (never started, a human driving, a speculative call): ``None``, nothing booked."""
    from maxim.bridges.tool_pain_bridge import ToolPainBridge

    nac = NAc(NACConfig())
    bridge = ToolPainBridge(nac=nac, pain_detector=None)
    assert bridge.record_tool_failure(TOOL, "never-started") is None
    assert nac._links == {}


def test_a_failure_whose_event_is_gone_carries_no_stamp_and_no_surprise() -> None:
    """The event aged out or was consumed before the booking: no link, so no stamp (dispatch books as before) and
    no surprise noted (``rpe`` stays unmeasured, not a measured 0.0)."""
    executor, nac = _executor(_Flaky())
    real_start = executor._tool_pain_bridge.record_tool_start

    def _start_then_lose_the_event(tool_name: str, invocation_id: str, **kw: Any) -> Any:
        out = real_start(tool_name, invocation_id, **kw)
        nac._pending_events.clear()  # stands in for the 300 s window expiring mid-run
        return out

    executor._tool_pain_bridge.record_tool_start = _start_then_lose_the_event
    out = executor.execute({"tool_name": TOOL, "params": {}})
    assert (out.nac_failure_booked, out.rpe) == (None, None)


def test_a_bridge_error_degrades_to_dispatch_booking() -> None:
    """A raising bridge never fails the call: the output carries no stamp, so dispatch books the NEGATIVE."""
    executor, nac = _executor(_Flaky())

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("bridge down")

    executor._tool_pain_bridge.record_tool_failure = _boom
    out = executor.execute({"tool_name": TOOL, "params": {}})  # the executor's own contract: it does not raise
    assert (out.success, out.nac_failure_booked) == (False, None)
    _execute_and_learn(executor, nac, {"tool_name": TOOL, "params": {}})
    assert [link.outcome_type for link in _links(nac, f"tool:{TOOL}", Valence.NEGATIVE)] == ["tool_result"]


def _prefetch(ok: bool) -> NAc:
    from maxim.runtime.prefetch import SpeculativePrefetcher

    reader = _Reader()
    reader.ok = ok
    executor, nac = _executor(reader)
    SpeculativePrefetcher(executor=executor)._execute_cached("read_file", {"path": "x.txt"})
    return nac


def _booked_read_file(nac: NAc) -> tuple[list[Any], list[Any]]:
    return nac._links.get("tool:read_file", []), [e for e in nac._pending_events if e["signature"] == "tool:read_file"]


def test_a_speculative_prefetch_hit_books_nothing() -> None:
    assert _booked_read_file(_prefetch(ok=True)) == ([], [])


# ── guards: what Stage 2 must not change ──────────────────────────────────────────────────────────────


def test_a_success_books_as_before() -> None:
    """Guard (green before and after): one bridge POSITIVE plus dispatch's own success link."""
    flaky = _Flaky()
    flaky.fail = False
    executor, nac = _executor(flaky)
    _execute_and_learn(executor, nac, {"tool_name": TOOL, "params": {}})
    positives = _links(nac, f"tool:{TOOL}", Valence.POSITIVE)
    assert sorted(link.outcome_type for link in positives) == ["result", "tool_result"]
    assert _links(nac, f"tool:{TOOL}", Valence.NEGATIVE) == []


def test_a_human_driven_failure_books_through_dispatch_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard (TF3): the interactive gate suppresses the bridge, so dispatch keeps its NEGATIVE."""
    import maxim.simulation.sim_logger as sim_logger

    monkeypatch.setattr(sim_logger, "get_interactive_mode", lambda: sim_logger.InteractiveMode.ON)
    executor, nac = _executor(_Flaky())
    _execute_and_learn(executor, nac, {"tool_name": TOOL, "params": {}})
    negatives = _links(nac, f"tool:{TOOL}", Valence.NEGATIVE)
    assert [link.outcome_type for link in negatives] == ["tool_result"]


def test_a_failure_with_no_bridge_books_through_dispatch_as_before() -> None:
    """Guard: an executor without an NAc has no bridge (``api.py``'s learning=False, the sandbox); dispatch
    still books into the loop's NAc."""
    registry = ToolRegistry()
    registry.register(_Flaky())
    executor = build_executor(registry, pain_bus=None, permissions=None)
    assert executor._tool_pain_bridge is None
    nac = NAc(NACConfig())
    _execute_and_learn(executor, nac, {"tool_name": TOOL, "params": {}})
    assert [link.outcome_type for link in _links(nac, f"tool:{TOOL}", Valence.NEGATIVE)] == ["tool_result"]


def test_a_speculative_prefetch_miss_books_nothing() -> None:
    """Guard (green today only because no failure reaches the bridge yet): once Stage 2 books failures, a
    speculative miss must still book nothing (TF9)."""
    assert _booked_read_file(_prefetch(ok=False)) == ([], [])


def test_an_aliased_failure_keeps_dispatchs_alias_link() -> None:
    """Guard (TF7): the bridge books the canonical name, dispatch the alias the model wrote; they differ, so
    dispatch's link under the alias is not cut."""

    class _Recall(_Flaky):
        name = "memory_recall"

    executor, nac = _executor(_Recall())
    _execute_and_learn(executor, nac, {"tool_name": "recall", "params": {}})
    assert [link.outcome_type for link in _links(nac, "tool:recall", Valence.NEGATIVE)] == ["tool_result"]


# ── the cut itself ────────────────────────────────────────────────────────────────────────────────────


def _record(nac: NAc, *, success: bool, bridge_booked: str | None, tool: str = TOOL, params: Any = None) -> None:
    tool_dispatch.record_outcome(
        agent_id="aut_1200",
        tool_name=tool,
        success=success,
        result_summary=None if not success else "ok",
        error="boom" if not success else None,
        reasoning="probe",
        recent_outcomes=[],
        max_recent=5,
        llm_worker=None,
        context_pool=MagicMock(),
        nac=nac,
        tool_params=params,
        bridge_booked=bridge_booked,
    )


def _dispatch_links(nac: NAc, signature: str) -> list[str]:
    return [str(link.outcome_valence) for link in nac._links.get(signature, []) if link.outcome_type == "tool_result"]


def test_the_cut_skips_only_the_matching_failure() -> None:
    nac = NAc(NACConfig())
    _record(nac, success=False, bridge_booked=f"tool:{TOOL}")
    assert _dispatch_links(nac, f"tool:{TOOL}") == []


@pytest.mark.parametrize(
    ("success", "booked", "params"),
    [
        (False, None, None),  # nothing booked by a bridge
        (False, "tool:memory_recall", None),  # the bridge booked a different (canonical) signature
        (False, "tool:use", {"action": "dig"}),  # dispatch's signature is compound: tool:use:dig
        (True, f"tool:{TOOL}", None),  # a success is never cut
    ],
    ids=["none", "other_signature", "compound", "success"],
)
def test_the_cut_books_as_before_otherwise(success: bool, booked: str | None, params: Any) -> None:
    nac = NAc(NACConfig())
    tool = "use" if params else TOOL
    _record(nac, success=success, bridge_booked=booked, tool=tool, params=params)
    sig = tool_dispatch.build_tool_signature(tool, params)
    assert len(_dispatch_links(nac, sig)) == 1


def test_a_parallel_failure_is_booked_once() -> None:
    """The batch path threads the stamp through its per-action dict."""
    executor, nac = _executor(_Flaky())
    autonomy = MagicMock()
    autonomy.can_execute_action.return_value = (True, "")
    tool_dispatch.execute_parallel_actions(
        agent_id="aut_1200",
        actions=[{"tool_name": TOOL, "params": {}}],
        executor=executor,
        autonomy_controller=autonomy,
        confidence=0.9,
        reasoning="probe",
        recent_outcomes=[],
        max_recent=5,
        llm_worker=None,
        context_pool=MagicMock(),
        nac=nac,
    )
    negatives = _links(nac, f"tool:{TOOL}", Valence.NEGATIVE)
    assert [link.outcome_type for link in negatives] == ["result"]


def test_a_speculative_failure_carries_no_stamp() -> None:
    from maxim.runtime.executor import speculative_calls

    executor, nac = _executor(_Flaky())
    with speculative_calls():
        out = executor.execute({"tool_name": TOOL, "params": {}})
    assert out.nac_failure_booked is None
    assert nac._links.get(f"tool:{TOOL}", []) == []


# ── every caller passes the stamp (TF8) ───────────────────────────────────────────────────────────────

# Session A's fenced file: its two ``_rec_outcome`` sites land ``bridge_booked`` in their own PR (owner decision
# TF8, 2026-10-10). The count is exact, so that PR shrinks it to 0 and the follow-up drops the default.
_AWAITING_SESSION_A = {"src/maxim/runtime/agent_loop.py": 2}


def _record_outcome_calls_missing_the_stamp() -> dict[str, int]:
    """Calls of ``tool_dispatch.record_outcome`` (direct, through the module, or through the loop's bound
    ``rec_outcome`` partial) in ``src/`` and ``scripts/`` that omit ``bridge_booked``. Fails CLOSED on shapes it
    cannot follow: a plain or relative import of the module, and any non-call reference to ``record_outcome``
    other than ``loop_setup``'s one ``functools.partial``."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    missing: dict[str, int] = {}
    for base in ("src", "scripts"):
        for path in sorted((root / base).rglob("*.py")):
            rel = str(path.relative_to(root))
            tree = ast.parse(path.read_text(encoding="utf-8"))
            names = {"rec_outcome", "_rec_outcome"}
            imported: set[str] = set()  # ``record_outcome`` itself, under whatever name it was imported
            modules: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert not any(a.name == "maxim.runtime.tool_dispatch" for a in node.names), (
                        f"{rel}: `import maxim.runtime.tool_dispatch` is a shape this guard does not follow"
                    )
                if isinstance(node, ast.ImportFrom) and node.level and (node.module or "").endswith("tool_dispatch"):
                    raise AssertionError(f"{rel}: a relative import of tool_dispatch is not followed")
                if isinstance(node, ast.ImportFrom) and node.module == "maxim.runtime.tool_dispatch":
                    imported |= {a.asname or a.name for a in node.names if a.name == "record_outcome"}
                if isinstance(node, ast.ImportFrom) and node.module == "maxim.runtime":
                    modules |= {a.asname or a.name for a in node.names if a.name == "tool_dispatch"}
            if path.name == "tool_dispatch.py":
                imported.add("record_outcome")
            names |= imported
            called: set[int] = set()
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                hit = (isinstance(f, ast.Name) and f.id in names) or (
                    # a bound recorder reached as an attribute (``run.rec_outcome(...)``), or the module's own
                    isinstance(f, ast.Attribute)
                    and (
                        f.attr in ("rec_outcome", "_rec_outcome")
                        or (f.attr == "record_outcome" and isinstance(f.value, ast.Name) and f.value.id in modules)
                    )
                )
                if hit:
                    called.add(id(f))
                    if not any(k.arg == "bridge_booked" for k in node.keywords):
                        missing[rel] = missing.get(rel, 0) + 1
            for node in ast.walk(tree):
                ref = (
                    isinstance(node, ast.Attribute)
                    and node.attr == "record_outcome"
                    and isinstance(node.value, ast.Name)
                    and node.value.id in modules
                ) or (isinstance(node, ast.Name) and node.id in imported and isinstance(node.ctx, ast.Load))
                if ref and id(node) not in called:
                    missing[f"{rel}:non-call reference"] = missing.get(f"{rel}:non-call reference", 0) + 1
    return missing


# loop_setup binds the loop's ``rec_outcome`` as ``functools.partial(_td.record_outcome, ...)``: the one sanctioned
# non-call reference (its calls are the ``rec_outcome`` / ``_rec_outcome`` calls counted above).
_SANCTIONED_REFERENCES = {"src/maxim/runtime/loop_setup.py:non-call reference": 1}


def test_every_record_outcome_caller_passes_the_failure_stamp() -> None:
    """A caller that forgets ``bridge_booked`` books a bridge-booked failure twice, silently (the adversarial
    design pass's D1). Enforced here until the default is dropped and the type does it."""
    assert _record_outcome_calls_missing_the_stamp() == {**_AWAITING_SESSION_A, **_SANCTIONED_REFERENCES}

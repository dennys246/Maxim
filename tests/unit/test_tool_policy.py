"""#1315 item 3: ``runtime/tool_policy.py::ToolPolicy`` and the executor's typed dispatch seam.

The sealing, the race probe, the composite and auto_sense probes, O1/V10 and V7 are the red gates in
``test_atonia_red_gates_effect_policy.py``; the booking table is ``test_refusal_booking_red_gate.py``; the census is
``test_tool_policy_census.py``. This file pins the rest: the factories' semantics, ``narrowed`` never widening, the
seam's layer order and kinds, ``check()``, the alias and pin rules, ``record_outcome(refused=)`` and the agent-fallback
booking (``tool_dispatch.book_agent_fallback_failure``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from maxim.agents.permissions import AgentPermissions
from maxim.runtime import tool_dispatch
from maxim.runtime.executor import Executor
from maxim.runtime.tool_policy import OFFLINE_CEILING, ToolPolicy, effective_ceiling, require_policy
from maxim.tools.base import GATE_REFUSAL_KINDS, Effect, Tool, ToolErrorKind, ToolOutput, current_policy
from maxim.tools.registry import ToolRegistry

R = "test: tool policy"


def _tool(name: str, effect: Effect, ran: list[str] | None = None, body: Any = None) -> Tool:
    def execute(self: Any, **kwargs: Any) -> Any:
        if ran is not None:
            ran.append(name)
        if body is not None:
            return body()
        return ToolOutput(success=True, output=name)

    cls = type(
        f"_T_{name}",
        (Tool,),
        {"name": name, "description": name, "input_schema": {}, "effect": effect, "execute": execute},
    )
    return cls()


class TestFactories:
    def test_unrestricted_and_deny_are_ceiling_all(self) -> None:
        assert ToolPolicy.unrestricted(reason=R).ceiling == Effect.ALL
        d = ToolPolicy.deny(["bash"], reason=R)
        assert d.ceiling == Effect.ALL and d.permissions == AgentPermissions(tool_deny=frozenset({"bash"}))

    def test_only_sets_an_allow_list_and_the_ceiling(self) -> None:
        p = ToolPolicy.only(["a", "b"], ceiling=Effect.READ, reason=R)
        assert p.permissions.tool_allow == frozenset({"a", "b"}) and p.ceiling == Effect.READ and p.pins is None

    def test_a_reason_is_required(self) -> None:
        with pytest.raises(ValueError):
            ToolPolicy.unrestricted(reason="  ")

    def test_require_policy_refuses_none_and_ducks(self) -> None:
        for bad in (None, SimpleNamespace(ceiling=Effect.ALL), AgentPermissions()):
            with pytest.raises(TypeError):
                require_policy(bad, where="probe")


class TestNarrowed:
    def test_never_wider(self) -> None:
        base = ToolPolicy.only(["a", "b"], ceiling=Effect.READ | Effect.NETWORK, reason=R).narrowed(deny=["x"])
        n = base.narrowed(ceiling=Effect.READ | Effect.COMMS, only=["b", "c"], deny=["y"])
        assert n.ceiling == Effect.READ  # ANDed: COMMS was never in the base
        assert n.permissions.tool_allow == frozenset({"b"})  # intersected
        assert n.permissions.tool_deny == frozenset({"x", "y"})  # unioned

    def test_an_open_allow_list_takes_the_narrowing_one(self) -> None:
        n = ToolPolicy.unrestricted(reason=R).narrowed(only=["respond"])
        assert n.permissions.tool_allow == frozenset({"respond"}) and n.ceiling == Effect.ALL

    def test_pins_are_kept(self) -> None:
        t = _tool("p", Effect.READ)
        n = ToolPolicy.pinned({"p": t}, ceiling=Effect.READ, reason=R).narrowed()
        assert n.pins is not None and n.pins["p"] is t


class TestSeam:
    def test_layers_and_kinds(self) -> None:
        reg = ToolRegistry([_tool("rd", Effect.READ), _tool("net", Effect.NETWORK), _tool("denied", Effect.READ)])
        policy = ToolPolicy.only(["rd", "net"], ceiling=Effect.READ, reason=R)
        ex = Executor(reg, policy=policy)
        cases = {
            "rd": None,
            "net": (ToolErrorKind.POLICY_CEILING, "ceiling"),
            "denied": (ToolErrorKind.POLICY_PERMISSIONS, "permissions"),
            "ghost": (ToolErrorKind.POLICY_PERMISSIONS, "permissions"),  # the allow-list judges a name first
        }
        for name, want in cases.items():
            out = ex.execute({"tool_name": name, "params": {}})
            if want is None:
                assert out.success, name
            else:
                assert (out.error_kind, out.metadata.get("refusal_layer")) == want, name
                assert out.error_kind in GATE_REFUSAL_KINDS

    def test_unregistered_under_an_open_policy_is_registry_layer(self) -> None:
        out = Executor(ToolRegistry(), policy=ToolPolicy.unrestricted(reason=R)).execute({"tool_name": "nope"})
        assert out.error_kind is ToolErrorKind.NOT_REGISTERED and out.metadata["refusal_layer"] == "registry"
        assert out.metadata["effect"] == "unknown"

    def test_forbid_and_mode_layers(self) -> None:
        ex = Executor(ToolRegistry([_tool("bash", Effect.ALL)]), policy=ToolPolicy.unrestricted(reason=R))
        ex.set_forbid_source(lambda name: name == "bash")
        out = ex.execute({"tool_name": "bash"})
        assert (out.error_kind, out.metadata["refusal_layer"]) == (ToolErrorKind.POLICY_PERMISSIONS, "forbid")
        ex.set_forbid_source(None)
        ex.set_operational_override("passive")
        out = ex.execute({"tool_name": "bash"})
        assert (out.error_kind, out.metadata["refusal_layer"]) == (ToolErrorKind.MODE, "mode")

    def test_check_matches_dispatch_and_never_runs(self) -> None:
        ran: list[str] = []
        net, rd = _tool("net", Effect.NETWORK, ran), _tool("rd", Effect.READ, ran)
        ex = Executor(ToolRegistry([net, rd]), policy=ToolPolicy.only(["net", "rd"], ceiling=Effect.READ, reason=R))
        refusal = ex.check(net)
        assert refusal is not None and refusal.error_kind is ToolErrorKind.POLICY_CEILING
        assert ex.check(rd) is None and not ex.permits("net") and ex.permits("rd")
        assert not ran

    def test_pinned_refuses_a_same_named_other_object_and_aliases(self) -> None:
        pinned, other = _tool("respond", Effect.READ), _tool("respond", Effect.READ)
        ex = Executor(
            ToolRegistry([other]), policy=ToolPolicy.pinned({"respond": pinned}, ceiling=Effect.ALL, reason=R)
        )
        out = ex.execute({"tool_name": "respond"})
        assert (out.error_kind, out.metadata["refusal_layer"]) == (ToolErrorKind.POLICY_PIN, "pin")
        ex2 = Executor(
            ToolRegistry([pinned]), policy=ToolPolicy.pinned({"respond": pinned}, ceiling=Effect.ALL, reason=R)
        )
        assert ex2.execute({"tool_name": "respond"}).success
        ex2.register_aliases({"reply_alias_probe": "respond"})
        try:
            out = ex2.execute({"tool_name": "reply_alias_probe"})
        finally:
            ex2.remove_aliases(["reply_alias_probe"])
        assert (out.error_kind, out.metadata["refusal_layer"]) == (ToolErrorKind.POLICY_PIN, "pin")

    def test_current_policy_is_the_dispatchers_and_only_during_dispatch(self) -> None:
        seen: list[Any] = []
        policy = ToolPolicy.unrestricted(reason=R)
        probe = _tool("probe", Effect.READ, body=lambda: seen.append(current_policy()) or ToolOutput(success=True))
        assert Executor(ToolRegistry([probe]), policy=policy).execute({"tool_name": "probe"}).success
        assert seen == [policy] and current_policy() is None

    def test_offline_refusal_in_a_body_is_typed_offline(self) -> None:
        from maxim._offline import OfflineRefusal

        def body() -> Any:
            raise OfflineRefusal("probe")

        ex = Executor(ToolRegistry([_tool("lazy", Effect.READ, body=body)]), policy=ToolPolicy.unrestricted(reason=R))
        out = ex.execute({"tool_name": "lazy"})
        assert (out.error_kind, out.metadata["refusal_layer"]) == (ToolErrorKind.OFFLINE, "offline")

    def test_the_offline_ceiling_is_only_applied_under_the_latch(self) -> None:
        p = ToolPolicy.unrestricted(reason=R)
        assert effective_ceiling(p) == Effect.ALL  # the conftest never engages the latch in-process
        assert not OFFLINE_CEILING & (Effect.NETWORK | Effect.COMMS)


class TestRecordOutcomeRefused:
    def _record(self, refused: Any, nac: Any) -> list[dict[str, Any]]:
        recent: list[dict[str, Any]] = []
        tool_dispatch.record_outcome(
            agent_id="a",
            tool_name="t",
            success=False,
            result_summary=None,
            error="refused",
            reasoning="r",
            recent_outcomes=recent,
            max_recent=5,
            llm_worker=None,
            context_pool=MagicMock(),
            nac=nac,
            active_goal="g",
            bridge_booked=None,
            refused=refused,
        )
        return recent

    def test_a_gate_kind_books_no_nac_but_keeps_the_window(self) -> None:
        nac = MagicMock()
        assert len(self._record(ToolErrorKind.POLICY_CEILING, nac)) == 1
        assert not nac.method_calls

    def test_none_books_nac(self) -> None:
        nac = MagicMock()
        self._record(None, nac)
        assert nac.observe.called

    def test_a_tool_kind_is_refused_loudly(self) -> None:
        with pytest.raises(ValueError):
            self._record(ToolErrorKind.PERMISSION_DENIED, MagicMock())

    def test_gate_refusal_of(self) -> None:
        assert tool_dispatch.gate_refusal_of(None) is None
        assert tool_dispatch.gate_refusal_of(ToolOutput(success=False, error_kind=ToolErrorKind.TIMEOUT)) is None
        out = ToolOutput(success=False, error_kind=ToolErrorKind.MODE)
        assert tool_dispatch.gate_refusal_of(out) is ToolErrorKind.MODE


class TestAgentFallbackBooking:
    """``book_agent_fallback_failure`` (A's review ask): the refused= it passes is read from the result."""

    def _book(self, result: Any) -> dict[str, Any]:
        calls: list[dict[str, Any]] = []
        tool_dispatch.book_agent_fallback_failure(
            RuntimeError("boom"),
            rec_outcome=lambda **kw: calls.append(kw),
            agent_id="a",
            action={"tool_name": "t"},
            result=result,
            proposal=None,
            recent_outcomes=[],
            max_recent=5,
            llm_worker=None,
            context_pool=MagicMock(),
            nac=MagicMock(),
            state=SimpleNamespace(data={}),
        )
        assert len(calls) == 1
        return calls[0]

    def test_execute_raised_books_a_failure(self) -> None:
        booked = self._book(None)  # executor.execute raised: result is still None
        assert booked["refused"] is None and booked["error"] == "boom" and booked["success"] is False

    def test_a_refusal_then_a_later_raise_carries_the_refusal(self) -> None:
        booked = self._book(ToolOutput(success=False, error_kind=ToolErrorKind.POLICY_CEILING))
        assert booked["refused"] is ToolErrorKind.POLICY_CEILING

    def test_the_carried_refusal_books_no_nac(self) -> None:
        nac = MagicMock()
        calls: list[dict[str, Any]] = []

        def rec(**kw: Any) -> None:
            calls.append(kw)
            tool_dispatch.record_outcome(**{**kw, "nac": nac})

        tool_dispatch.book_agent_fallback_failure(
            RuntimeError("later step"),
            rec_outcome=rec,
            agent_id="a",
            action={"tool_name": "t"},
            result=ToolOutput(success=False, error_kind=ToolErrorKind.POLICY_CEILING),
            proposal=None,
            recent_outcomes=[],
            max_recent=5,
            llm_worker=None,
            context_pool=MagicMock(),
            nac=nac,
            state=SimpleNamespace(data={}),
        )
        assert calls and not nac.method_calls

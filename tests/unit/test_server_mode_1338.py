"""#1338 (advisor item 6): CLI server mode reports an LLM that never became ready, in process.

Every path of ``cli._run_server_mode`` / ``cli._start_server_llm``: router None, init raising, ``wait_ready`` False
(behind a LeaderProxy it stays up and reports not-ready on ``/v1/debug/ping``'s ``server_llm``; solo it exits 1), a
``LeaderRefusal`` (2), the ready path then Ctrl+C (0), and the spawner stopped on every exit, its failure logged. The
O8 gate in ``test_atonia_open_items_other.py`` drives the same seam through ``cli.main`` in a child process, which
coverage cannot see.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import pytest

import maxim.cli as cli
import maxim.runtime.lane_backends as lb
from maxim.exceptions import LeaderRefusal
from maxim.runtime import leader_proxy

LOG = logging.getLogger("test_server_mode_1338")


class _Router:
    def __init__(self, ready: bool) -> None:
        self.ready = ready
        self.calls: list[str] = []

    def warmup(self) -> bool:
        self.calls.append("warmup")
        return True

    def wait_ready(self, timeout: float = 0.0) -> bool:
        self.calls.append(f"wait_ready:{timeout}")
        return self.ready


class _CtrlCEvent:
    """``threading.Event`` whose untimed ``wait`` (server mode's idle wait) is a Ctrl+C."""

    waited = 0

    def wait(self, timeout: float | None = None) -> bool:
        _CtrlCEvent.waited += 1
        raise KeyboardInterrupt


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"stopped": 0, "router": None, "build_raises": None}

    def _build(**kw: Any) -> Any:
        if state["build_raises"] is not None:
            raise state["build_raises"]
        return state["router"], None

    def _stop() -> None:
        state["stopped"] += 1

    monkeypatch.setattr(lb, "build_primary_router", _build)
    monkeypatch.setattr(lb, "stop_active_spawner", _stop)
    monkeypatch.setattr(threading, "Event", _CtrlCEvent)
    monkeypatch.setattr(leader_proxy, "_leader_proxy_server", None)
    monkeypatch.setattr(leader_proxy, "_server_llm_status", None)
    _CtrlCEvent.waited = 0
    return state


def _with_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(leader_proxy, "_leader_proxy_server", object())


def test_router_none_solo_exits_1(server: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli._run_server_mode(LOG) == 1
    err = capsys.readouterr().err
    assert "NOT READY" in err and "did not start" in err
    assert server["stopped"] == 1 and _CtrlCEvent.waited == 0


def test_init_error_solo_exits_1(server: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    server["build_raises"] = RuntimeError("no model file")
    assert cli._run_server_mode(LOG) == 1
    assert "NOT READY: LLM server init failed: no model file" in capsys.readouterr().err
    assert server["stopped"] == 1


def test_not_ready_solo_exits_1(server: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    server["router"] = _Router(ready=False)
    assert cli._run_server_mode(LOG) == 1
    out = capsys.readouterr()
    assert "NOT READY" in out.err and "180s" in out.err
    assert "LLM server ready" not in out.out
    assert server["router"].calls == ["warmup", "wait_ready:180.0"]
    assert server["stopped"] == 1 and _CtrlCEvent.waited == 0
    assert leader_proxy.server_llm_status() is None  # solo: nothing to report on


def test_not_ready_behind_a_leader_proxy_stays_up_and_reports_it(
    server: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _with_proxy(monkeypatch)
    server["router"] = _Router(ready=False)
    assert cli._run_server_mode(LOG) == 0  # stayed up until Ctrl+C
    out = capsys.readouterr()
    assert _CtrlCEvent.waited == 1
    assert "NOT READY" in out.err and "Staying up" in out.err
    assert "LLM server ready" not in out.out
    status = leader_proxy.server_llm_status()
    assert status is not None and status["ready"] is False and "180s" in status["reason"]
    assert server["stopped"] == 1


@pytest.mark.parametrize("case", ["router_none", "init_error"])
def test_other_not_ready_causes_behind_a_proxy_stay_up(
    case: str, server: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _with_proxy(monkeypatch)
    if case == "init_error":
        server["build_raises"] = RuntimeError("boom")
    assert cli._run_server_mode(LOG) == 0
    status = leader_proxy.server_llm_status()
    assert status is not None and status["ready"] is False
    assert _CtrlCEvent.waited == 1 and server["stopped"] == 1


def test_leader_refusal_exits_2(server: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    server["build_raises"] = LeaderRefusal("key readable by others")
    assert cli._run_server_mode(LOG) == 2
    assert "Refusing to serve: key readable by others" in capsys.readouterr().err
    assert server["stopped"] == 1 and _CtrlCEvent.waited == 0


def test_ready_then_ctrl_c_exits_0(server: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    server["router"] = _Router(ready=True)
    assert cli._run_server_mode(LOG) == 0
    out = capsys.readouterr()
    assert "LLM server ready" in out.out and "Shutting down" in out.out
    assert "NOT READY" not in out.err
    status = leader_proxy.server_llm_status()
    assert status is not None and status["ready"] is True
    assert server["stopped"] == 1


def test_a_failed_spawner_stop_is_logged(
    server: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _stop_fails() -> None:
        raise OSError("kill failed")

    monkeypatch.setattr(lb, "stop_active_spawner", _stop_fails)
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        assert cli._run_server_mode(LOG) == 1  # router None, solo
    assert any("Could not stop the spawned LLM server" in r.getMessage() for r in caplog.records)


def test_ping_payload_carries_the_server_llm_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    """``/v1/debug/ping`` serves ``server_llm`` beside the live ``llm_ready`` probe."""
    monkeypatch.setattr(leader_proxy, "_server_llm_status", None)
    leader_proxy.set_server_llm_status(ready=False, reason="the LLM did not become ready within 180s")
    sent: dict[str, Any] = {}

    class _H:
        server = type("S", (), {"server_address": ("127.0.0.1", 8099)})()
        upstream_url = "http://127.0.0.1:8100"
        auth = None
        start_time = 0.0

        def _probe_upstream_ready(self) -> bool:
            return False

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            sent.update(code=code, payload=payload)

    leader_proxy._ProxyHandler._handle_debug_ping(_H())  # type: ignore[arg-type]
    assert sent["payload"]["llm_ready"] is False
    assert sent["payload"]["server_llm"]["ready"] is False

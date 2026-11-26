"""#1801 families 2-4 and the cut-before-reduce sites: every boundary where exception text leaves the process
reduces its URLs (docs/plans/log_url_redaction_1801_approach.md v28, §Shape, §Sites that cut before reducing,
§Guards). ``SECRET`` is the planted credential; a site that cut before reducing would leak it.
"""

from __future__ import annotations

import copy
import dataclasses
import io
import json
import logging
import pickle
import re
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

URL = "https://user:SECRET@host.test/private/path?key=SECRET"
# A token-as-username URL whose ``@`` lies past a head cut: cutting first leaves ``https://tok_SECRET...`` with no
# ``@``, a "host" that keeps the token (``url_origin`` lowercases it, so ``_clean`` compares lowercased).
LONG_TOKEN_URL = "https://tok_SECRETSECRETSECRETSECRETSECRETSECRET@github.com/o/r"


def _clean(text: object) -> bool:
    return "secret" not in str(text).lower()


def _has_origin(text: str) -> bool:
    """The reduced origin appears WHOLE: no more host characters follow it, so a lookalike such as
    ``https://host.test.evil`` does not count (an exact check, not a substring one)."""
    return re.search(re.escape("https://host.test") + r"(?![A-Za-z0-9.-])", text) is not None


def _tail_url(total: int) -> str:
    """A text whose ``[-total:]`` tail starts right after ``https://``: cutting first drops the scheme, so no URL is
    seen and the userinfo survives."""
    rest = "tok:SECRET@host.test/" + "p" * (total - len("tok:SECRET@host.test/"))
    return "https://" + rest


# ─── family 2: sim_log, all six destinations (AB3, AF6) ─────────────────


def test_sim_log_reduces_every_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from maxim.simulation import sim_logger
    from maxim.utils.logging import configure_logging

    jsonl = tmp_path / "maxim.jsonl"
    sim_jsonl = tmp_path / "sim.jsonl"
    monkeypatch.setenv("MAXIM_LOG_FILE", str(jsonl))
    configure_logging(0, force=True)
    sunk: list[dict[str, Any]] = []
    sim_logger.register_sim_sink(sunk.append)
    sim_logger.enable_sim_logging(use_color=False, log_path=str(sim_jsonl), debug=True)
    try:
        err = ConnectionError(f"GET {URL} failed")
        sim_logger.sim_log("HIPPOCAMPUS", f"recall via {URL}", {"err": err, "nested": {"u": [URL]}})
        trail = sim_logger.get_sim_records()
    finally:
        sim_logger.disable_sim_logging()
        sim_logger.unregister_sim_sink(sunk.append)
        root = logging.getLogger()
        for h in list(root.handlers):
            if getattr(h, "_maxim_jsonl", False):
                h.flush()
                root.removeHandler(h)
                sim_logger._sim_bridge_logger.removeHandler(h)
                h.close()
    terminal = capsys.readouterr().out
    hits = [r for r in sunk if r["subsystem"] == "HIPPOCAMPUS"]
    assert hits and _clean(json.dumps(hits))  # sinks (the console /ws stream)
    assert any(r["subsystem"] == "HIPPOCAMPUS" for r in trail) and _clean(json.dumps(trail))  # trail
    assert "HIPPOCAMPUS" in sim_jsonl.read_text() and _clean(sim_jsonl.read_text())  # sim JSONL
    assert "sim_hippocampus" in jsonl.read_text() and _clean(jsonl.read_text())  # bridge + StructuredFormatter
    assert "recall via https://host.test" in terminal and _clean(terminal)  # terminal


DOCS_LINK = "https://docs.python.org/3/library/re.html"


@pytest.mark.parametrize("subsystem", ["RESPONSE", "USER"])
def test_sim_log_conversation_reaches_sinks_whole(
    subsystem: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Owner decision 2026-11-25 (code review Wire DO-NOT-MERGE): the conversation (RESPONSE, the agent's reply, and
    USER, the person's input) reaches the sinks (the console's /ws chat) UNREDUCED, so the person sees full links.
    Every persisted or logged destination stays reduced: the trail, the sim JSONL, the bridge and the terminal."""
    from maxim.simulation import sim_logger
    from maxim.utils.logging import configure_logging

    jsonl = tmp_path / "maxim.jsonl"
    sim_jsonl = tmp_path / "sim.jsonl"
    monkeypatch.setenv("MAXIM_LOG_FILE", str(jsonl))
    configure_logging(0, force=True)
    sunk: list[dict[str, Any]] = []
    sim_logger.register_sim_sink(sunk.append)
    sim_logger.enable_sim_logging(use_color=False, log_path=str(sim_jsonl), debug=True)
    text = f"see {DOCS_LINK} and {URL}"
    try:
        sim_logger.sim_log(subsystem, text, {"text": text})
        sim_logger.sim_log("HIPPOCAMPUS", text, {"text": text})
        trail = sim_logger.get_sim_records()
    finally:
        sim_logger.disable_sim_logging()
        sim_logger.unregister_sim_sink(sunk.append)
        root = logging.getLogger()
        for h in list(root.handlers):
            if getattr(h, "_maxim_jsonl", False):
                h.flush()
                root.removeHandler(h)
                sim_logger._sim_bridge_logger.removeHandler(h)
                h.close()
    terminal = capsys.readouterr().out
    conv = [r for r in sunk if r["subsystem"] == subsystem]
    assert conv and conv[0]["message"] == text and conv[0]["data"]["text"] == text  # whole at the sink
    other = [r for r in sunk if r["subsystem"] == "HIPPOCAMPUS"]
    assert other and DOCS_LINK not in json.dumps(other) and _clean(json.dumps(other))  # not conversation: reduced
    for where in (json.dumps(trail), sim_jsonl.read_text(), jsonl.read_text(), terminal):
        assert DOCS_LINK not in where and _clean(where)
    assert subsystem in sim_jsonl.read_text() and f"sim_{subsystem.lower()}" in jsonl.read_text()


def test_console_fanout_keeps_conversation_whole() -> None:
    """The console's fan-out does not re-reduce the conversation events sim_log sent whole (owner, 2026-11-25)."""
    from maxim.console.server import _EventHub

    got: list[dict[str, Any]] = []

    class _Conn:
        def matches(self, kind: str, tier: str, subsystem: str) -> bool:
            return True

        def enqueue(self, evt: dict[str, Any]) -> None:
            got.append(evt)

    hub = _EventHub()
    hub._conns.add(_Conn())  # type: ignore[arg-type]
    for sub in ("RESPONSE", "USER", "HIPPOCAMPUS"):
        hub._fanout(
            sub, {"kind": sub.lower(), "tier": "clean", "seq": 0, "message": DOCS_LINK, "data": {"u": DOCS_LINK}}
        )
    by_kind = {e["kind"]: e for e in got}
    assert by_kind["response"]["message"] == DOCS_LINK and by_kind["user"]["data"]["u"] == DOCS_LINK
    assert DOCS_LINK not in json.dumps(by_kind["hippocampus"])


def test_sim_log_reduces_before_text_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shape computed over raw text is an oracle for it (AB3): the bridge sees the reduced message."""
    from maxim.simulation import sim_logger

    seen: list[str] = []
    monkeypatch.setattr(sim_logger, "text_shape", lambda t: seen.append(t) or "<shape>")
    sim_logger.enable_sim_logging(use_color=False)
    try:
        sim_logger.sim_log("RESPONSE", f"reply {URL}", {"text": f"reply {URL}"}, _terminal=False)
    finally:
        sim_logger.disable_sim_logging()
    assert seen and all(_clean(t) for t in seen)


def test_structured_formatter_reduces_data() -> None:
    from maxim.utils.structured_logging import StructuredFormatter

    rec = logging.LogRecord("maxim.x", logging.INFO, __file__, 1, "m", (), None)
    rec.event = "e"
    rec.data = {"u": URL, "err": RuntimeError(URL), "t": (URL,)}
    out = StructuredFormatter().format(rec)
    assert _clean(out) and _has_origin(out)


def test_structured_formatter_keeps_non_dict_data() -> None:
    from maxim.utils.structured_logging import StructuredFormatter

    rec = logging.LogRecord("maxim.x", logging.INFO, __file__, 1, "m", (), None)
    rec.data = [URL]
    assert json.loads(StructuredFormatter().format(rec))["value"] == ["https://host.test"]


# ─── family 3: log_agentic and log_structured ────────────────────────────


def test_log_agentic_and_log_structured_reduce(caplog: pytest.LogCaptureFixture) -> None:
    from maxim.utils.structured_logging import StructuredFormatter, get_abstraction_buffer, log_agentic, log_structured

    buf = get_abstraction_buffer()
    buf.clear()
    log_agentic("t1801", "error", {"error": f"failed {URL}", "e": RuntimeError(URL)})
    with caplog.at_level(logging.INFO):
        log_structured(logging.getLogger("maxim.t1801"), logging.INFO, "llm_error", {"u": URL})
    recent = buf.get_recent(10, verbose=True)
    assert recent and _clean(json.dumps(recent))
    records = [r for r in caplog.records if getattr(r, "event", None) == "llm_error"]
    assert records and _clean(records[0].data)
    assert _clean(StructuredFormatter().format(records[0]))


def test_abstraction_buffer_append_reduces_every_producer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Code review (pre-existing, same pattern): the scrub lives in ``AbstractionBuffer.append``, so producers that
    append directly (``AgentOutputManager.write``/``write_to_stream``) are reduced too, not only ``log_agentic``."""
    from maxim.utils.agent_output import AgentOutput, AgentOutputManager, OutputDestination
    from maxim.utils.structured_logging import LogRecord, get_abstraction_buffer

    # The global buffer: ``AgentOutputManager`` falls back to it for an EMPTY buffer passed in (``buffer or global``).
    buf = get_abstraction_buffer()
    buf.clear()
    monkeypatch.setattr(buf, "_verbosity", 3)
    monkeypatch.setattr(buf, "_callbacks", [])
    seen: list[LogRecord] = []
    buf.add_callback(seen.append)
    mgr = AgentOutputManager(
        "i1801",
        base_data_dir=str(tmp_path / "data"),
        sandbox_dir=str(tmp_path / "sandbox"),
        shared_outputs_dir=str(tmp_path / "shared"),
        abstraction_buffer=buf,
    )
    mgr.write(AgentOutput("a", "log", f"fetch {URL}", destination=OutputDestination.ABSTRACTION_ONLY))
    mgr.open_stream("a", "trace")  # opened first: write_to_stream's auto-open re-takes a held Lock (pre-existing)
    mgr.write_to_stream("a", "trace", {"err": RuntimeError(URL), "u": [URL]})
    raw = {"u": URL}
    buf.append(LogRecord(timestamp=0.0, level="INFO", source="s", event="e", data=raw))
    recent = buf.get_recent(10, verbose=True)
    assert len(recent) == 3 and _clean(json.dumps(recent, default=str)) and _has_origin(json.dumps(recent))
    assert seen and _clean([r.data for r in seen])  # callbacks get the reduced record
    assert raw == {"u": URL}  # the producer's dict is not mutated
    buf.clear()


# ─── family 4: ToolOutput.error (D5a) ────────────────────────────────────


def test_tool_output_error_reduced_at_construction() -> None:
    from maxim.tools.base import ToolOutput

    out = ToolOutput(success=False, error=f"fetch failed: {URL}")
    assert out.error == "fetch failed: https://host.test"
    assert _clean(dataclasses.replace(out, success=False).error)
    assert _clean(pickle.loads(pickle.dumps(out)).error)
    assert _clean(copy.copy(out).error) and _clean(copy.deepcopy(out).error)
    assert ToolOutput(success=False, error=RuntimeError(URL)).error == "https://host.test"  # type: ignore[arg-type]
    assert ToolOutput(success=True, output={"page": URL}).output == {"page": URL}  # output keeps its URLs
    assert ToolOutput(success=True).error is None


# ─── family 4: the console (S2) ──────────────────────────────────────────


def test_console_event_hub_fanout_reduces() -> None:
    from maxim.console.server import _EventHub

    got: list[dict[str, Any]] = []

    class _Conn:
        def matches(self, kind: str, tier: str, subsystem: str) -> bool:
            return True

        def enqueue(self, evt: dict[str, Any]) -> None:
            got.append(evt)

    hub = _EventHub()
    hub._conns.add(_Conn())  # type: ignore[arg-type]
    hub._fanout("", {"kind": "run", "tier": "clean", "seq": 0, "message": URL, "data": {"error": URL, "n": 3}})
    assert got and _clean(got) and got[0]["data"]["n"] == 3


def test_console_error_details_reduce(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    from maxim.console import server
    from maxim.console.run_store import BadCursor
    from maxim.exceptions import ConfigurationError

    def _raise(exc: BaseException):  # noqa: ANN202
        def fn(*a: Any, **k: Any) -> Any:
            raise exc

        return fn

    with pytest.raises(HTTPException) as hx:
        server._run_call(_raise(BadCursor(f"bad {URL}")))
    assert hx.value.detail.startswith("Bad cursor: ") and _clean(hx.value.detail)

    import maxim.runtime.config_writer as cw

    monkeypatch.setattr(cw, "apply_mesh_setup", _raise(ConfigurationError(f"leader {URL} unreachable")))
    monkeypatch.setattr(cw, "apply_cloud_setup", _raise(ConfigurationError(f"provider {URL}")))
    with pytest.raises(HTTPException) as hx:
        server.post_setup_mesh(SimpleNamespace(leader_url=URL, api_key="k"))
    assert _clean(hx.value.detail)
    with pytest.raises(HTTPException) as hx:
        server.post_setup_cloud(SimpleNamespace(provider="p", profile="x", api_key="k", monthly_budget_usd=None))
    assert _clean(hx.value.detail)

    class _Handle:
        def talk(self, text: str) -> dict[str, Any]:
            raise ConnectionError(f"GET {URL}")

        def rest(self) -> list[Any]:
            raise ConnectionError(f"GET {URL}")

    monkeypatch.setattr(server, "_get_handle", lambda: _Handle())
    with pytest.raises(HTTPException) as hx:
        server._post_run_talk(SimpleNamespace(input="hi"))
    assert hx.value.status_code == 500 and _clean(hx.value.detail)
    assert server._error_detail(ConnectionError(URL)) == "ConnectionError: https://host.test"


def test_console_rest_detail_reduces(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    from maxim.console import server

    rest_fn = next(
        getattr(server, n)
        for n in dir(server)
        if callable(getattr(server, n)) and "rest" in n and n.startswith("_post_run_")
    )

    class _Handle:
        def rest(self) -> list[Any]:
            raise ConnectionError(f"GET {URL}")

    monkeypatch.setattr(server, "_get_handle", lambda: _Handle())
    with pytest.raises(HTTPException) as hx:
        rest_fn()
    assert hx.value.status_code == 500 and _clean(hx.value.detail)


# ─── family 4: the leader proxy (S2, G2, H5, F3, L4) ─────────────────────


def _proxy_handler() -> Any:
    from maxim.runtime.leader_proxy import _ProxyHandler

    h = object.__new__(_ProxyHandler)
    h.wfile = io.BytesIO()
    h.headers_sent: list[tuple[str, str]] = []
    h.send_response = lambda code: None
    h.send_header = lambda k, v: h.headers_sent.append((k, v))
    h.end_headers = lambda: None
    h.client_address = ("127.0.0.1", 1)
    return h


def test_send_json_reduces_every_error_string_leaf() -> None:
    h = _proxy_handler()
    h._send_json(500, {"error": f"pip upgrade failed: {URL}", "stderr": URL, "code": 400, "nested": {"m": [URL]}})
    body = h.wfile.getvalue()
    assert _clean(body.decode())
    parsed = json.loads(body)
    assert parsed["code"] == 400  # an OpenAI-shaped error keeps its int code
    assert dict(h.headers_sent)["Content-Length"] == str(len(body))


def test_send_json_2xx_untouched() -> None:
    h = _proxy_handler()
    h._send_json(200, {"page": URL})
    assert json.loads(h.wfile.getvalue())["page"] == URL


def test_reduced_error_body_deeply_nested_json_fails_closed() -> None:
    """Code review Executor NIT: a JSON body nested past the parser's recursion limit raises ``RecursionError``, not
    ``ValueError``; it falls back to reducing the decoded text instead of escaping."""
    from maxim.runtime.leader_proxy import _reduced_error_body

    depth = 200_000
    body = ("[" * depth + json.dumps(URL) + "]" * depth).encode()
    out = _reduced_error_body(body, "application/json")
    assert _clean(out.decode()) and _has_origin(out.decode())


def test_send_bytes_reduces_json_and_text_error_bodies() -> None:
    from maxim.hivemind.oasis_endpoints import EndpointResponse

    h = _proxy_handler()
    resp = EndpointResponse.json_response(400, {"error": f"Invalid JSON body: {URL}"})  # the oasis 400 path
    h._send_bytes(resp.status, resp.body, resp.content_type)
    assert _clean(h.wfile.getvalue().decode())
    h = _proxy_handler()
    h._send_bytes(500, b"raw \x80 " + URL.encode(), "text/plain")  # L4: a \x80 byte, not JSON
    body = h.wfile.getvalue()
    assert b"SECRET" not in body and b"\x80" in body
    assert dict(h.headers_sent)["Content-Length"] == str(len(body))
    h = _proxy_handler()
    h._send_bytes(502, b"{not json " + URL.encode(), "application/json")
    assert b"SECRET" not in h.wfile.getvalue()
    h = _proxy_handler()
    h._send_bytes(200, URL.encode(), "application/octet-stream")  # a bundle download is not touched
    assert h.wfile.getvalue() == URL.encode()


def _pip_handler() -> Any:
    h = _proxy_handler()
    h._sent: list[tuple[int, dict]] = []
    h._send_json = lambda code, body: h._sent.append((code, body))
    return h


def test_run_pip_install_and_upgrade_reduce_before_the_tail_cut() -> None:
    h = _pip_handler()
    ok = SimpleNamespace(returncode=0, stdout=_tail_url(500), stderr="")
    with patch("maxim.runtime.leader_proxy.subprocess.run", return_value=ok):
        assert _clean(h._run_pip_install("/tmp"))
    with (
        patch("maxim.runtime.leader_proxy._get_current_version", return_value="0.3.0"),
        patch("maxim.runtime.leader_proxy.subprocess.run", return_value=ok),
        patch("shutil.disk_usage", return_value=SimpleNamespace(free=10 * (1 << 30))),
        patch("shutil.rmtree"),
    ):
        assert _clean(h._run_pip_upgrade(None, []))
    bad = SimpleNamespace(returncode=1, stdout=_tail_url(500), stderr=_tail_url(500))
    with (
        patch("maxim.runtime.leader_proxy._get_current_version", return_value="0.3.0"),
        patch("maxim.runtime.leader_proxy.subprocess.run", return_value=bad),
        patch("shutil.disk_usage", return_value=SimpleNamespace(free=10 * (1 << 30))),
        patch("shutil.rmtree"),
    ):
        assert h._run_pip_upgrade(None, []) is None
    assert h._sent and _clean(h._sent[-1][1])


def test_install_status_reduced(monkeypatch: pytest.MonkeyPatch) -> None:
    """F3: ``/v1/debug/install-status`` is a 2xx body, so the error reduction misses it; the writes reduce."""
    from maxim.runtime import leader_proxy

    class _SyncThread:
        def __init__(self, target, **kw) -> None:  # noqa: ANN001, ANN003
            self._t = target

        def start(self) -> None:
            self._t()

    monkeypatch.setattr(leader_proxy.threading, "Thread", _SyncThread)
    for run in (
        lambda *a, **k: SimpleNamespace(returncode=1, stdout=_tail_url(500), stderr=_tail_url(500)),
        lambda *a, **k: (_ for _ in ()).throw(OSError(f"connect {URL}")),
    ):
        h = _pip_handler()
        raw = json.dumps({"extras": ["semantic"]}).encode()
        h._admin_refused = lambda what: False
        h.headers = {"Content-Length": str(len(raw))}
        h.rfile = io.BytesIO(raw)
        monkeypatch.setattr(leader_proxy.subprocess, "run", run)
        h._handle_admin_install()
        status = dict(leader_proxy._install_status)
        assert status["status"] == "error" and _clean(status), status


# ─── family 4: errors that bypass ToolOutput (S1, F3, G6) ─────────────────


class _Pool:
    def __init__(self) -> None:
        self.added: list[dict[str, Any]] = []

    def add_outcome(self, **kw: Any) -> None:
        self.added.append(kw)


def test_record_outcome_reduces_error_for_every_sink() -> None:
    from maxim.runtime.tool_dispatch import record_outcome

    recent: list[dict[str, Any]] = []
    pool = _Pool()
    observed: list[str] = []

    class _NAc:
        def observe(self, *a: Any, **k: Any) -> None:
            observed.append(repr((a, k)))

        def __getattr__(self, name: str) -> Any:
            return lambda *a, **k: observed.append(repr((name, a, k)))

    record_outcome(
        agent_id="a",
        tool_name="http_fetch",
        success=False,
        result_summary=None,
        error=f"GET {LONG_TOKEN_URL} failed",
        reasoning="",
        recent_outcomes=recent,
        max_recent=10,
        llm_worker=None,
        context_pool=pool,
        nac=_NAc(),
        bridge_booked=None,
        refused=None,
        untrusted_output=False,
    )
    assert recent and _clean(recent) and pool.added and _clean(pool.added)
    assert _clean(observed)


def _nac_outcomes(*, success: bool, result_summary: str | None, error: str | None) -> list[str]:
    from maxim.runtime.tool_dispatch import record_outcome

    observed: list[str] = []

    class _NAc:
        def __getattr__(self, name: str) -> Any:
            return lambda *a, **k: observed.append(repr((name, a, k)))

    record_outcome(
        agent_id="a",
        tool_name="http_fetch",
        success=success,
        result_summary=result_summary,
        error=error,
        reasoning="",
        recent_outcomes=[],
        max_recent=10,
        llm_worker=None,
        context_pool=_Pool(),
        nac=_NAc(),
        bridge_booked=None,
        refused=None,
        untrusted_output=False,
    )
    return observed


def test_record_outcome_nac_error_reduced_before_the_cut() -> None:
    observed = _nac_outcomes(success=False, result_summary=None, error="failed " + LONG_TOKEN_URL)
    assert observed and _clean(observed)


def test_record_outcome_success_summary_untouched() -> None:
    """Code review Wire SF1: a SUCCESS summary is outside D5's error scope, so NAc's persisted link key on the success
    path is the summary's first 50 characters, as before #1801."""
    summary = "fetched https://docs.python.org/3/library/re.html ok"
    observed = _nac_outcomes(success=True, result_summary=summary, error=None)
    assert any(f"success:{summary[:50]}" in o for o in observed)


def test_record_outcome_requires_agent_id() -> None:
    from maxim.runtime.tool_dispatch import _require_agent_id

    with pytest.raises(ValueError):
        _require_agent_id("")


def test_parallel_results_reduce() -> None:
    from maxim.runtime.tool_dispatch import execute_parallel_actions

    class _Exec:
        def execute(self, action: dict[str, Any]) -> Any:
            raise ConnectionError(f"GET {URL}")

    class _Auto:
        def can_execute_action(self, action: dict[str, Any], confidence: float) -> tuple[bool, str]:
            return True, ""

    recent: list[dict[str, Any]] = []
    results, combined = execute_parallel_actions(
        agent_id="a",
        actions=[{"tool_name": "http_fetch", "params": {}}],
        executor=_Exec(),
        autonomy_controller=_Auto(),
        confidence=0.9,
        reasoning="",
        recent_outcomes=recent,
        max_recent=10,
        llm_worker=None,
        context_pool=_Pool(),
    )
    assert results and _clean(results) and _clean(combined) and _clean(recent)


def test_abort_drain_detail_reduced_before_the_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.runtime import loop_planning

    entries = [SimpleNamespace(id=1)]
    details: list[str] = []
    monkeypatch.setattr(loop_planning, "_reject", lambda entry, reason: None)

    class _Queue:
        def pop_approved(self) -> Any:
            return entries.pop() if entries else None

    class _Refuser:
        def report_and_book(self, entry: Any, *, reason: str, detail: str) -> None:
            details.append(detail)

    # the URL starts 30 characters before the 300-character cut, so the cut falls inside its token
    prefix = "an earlier approved action raised ConnectionError: "
    exc = ConnectionError("x" * (300 - 30 - len(prefix) - 1) + " " + LONG_TOKEN_URL)
    loop_planning._abort_drain(_Queue(), exc, refuser=_Refuser())  # type: ignore[arg-type]
    assert details and _clean(details[0])


def test_mark_failure_reduces() -> None:
    from maxim.runtime.state import RuntimeState

    st = RuntimeState.__new__(RuntimeState)
    st.mark_failure(f"failed {URL}")
    assert st.last_error == "failed https://host.test"
    st.mark_failure(None)
    assert st.last_error is None


def test_sub_simulation_error_reduced(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from maxim.simulation import tools as sim_tools

    tool = sim_tools.SpawnSubSimulationTool(llm_router=None, sim_tmpdir=str(tmp_path))
    monkeypatch.setattr(sim_tools, "current_policy", lambda: SimpleNamespace(narrowed=lambda: None))

    def _boom(goal: str, policy: Any) -> dict[str, Any]:
        raise ConnectionError(f"GET {URL}")

    tool._run_sub_simulation = _boom  # type: ignore[method-assign]
    try:
        out = tool.execute(goal="g")
    except Exception as e:  # noqa: BLE001 — a harness gap must say so, not pass
        pytest.fail(f"harness: {type(e).__name__}: {e}")
    assert out.success and _clean(out.output)


def test_introspection_payload_errors_reduced() -> None:
    from maxim.tools.introspection import PainHistoryTool, SystemStatsTool

    class _Boom:
        def __getattr__(self, name: str) -> Any:
            def fn(*a: Any, **k: Any) -> Any:
                raise ConnectionError(f"GET {URL}")

            return fn

    fear = _Boom()
    out = PainHistoryTool(pain_detector=None, fear_agent=fear).execute(check_action="move")
    assert _clean(out.output)
    stats = SystemStatsTool(hippocampus=_Boom()).execute()
    assert _clean(stats.output)


def test_dm_runtime_delivery_error_reduced() -> None:
    from maxim.simulation import dm_runtime

    cls = next(v for v in vars(dm_runtime).values() if isinstance(v, type) and hasattr(v, "_deliver_and_wait"))
    obj = cls.__new__(cls)

    class _Bridge:
        def send_and_wait(self, *a: Any, **k: Any) -> Any:
            raise ConnectionError(f"GET {URL}")

    obj._bridge = _Bridge()
    assert _clean(obj._deliver_and_wait("hi"))


def test_minecraft_call_action_detail_reduced() -> None:
    from maxim.simulation.minecraft import MinecraftClient

    class _Sock:
        def sendall(self, data: bytes) -> None:
            raise OSError(f"send to {URL} failed")

    c = MinecraftClient(connection_factory=lambda: _Sock())  # type: ignore[arg-type, return-value]
    c._sock = _Sock()  # type: ignore[assignment]
    out = c.call_action("move_to", {"x": 1})
    assert out["ok"] is False and _clean(out)


def test_roy_persist_result_reduces(tmp_path: Path) -> None:
    """H4 / J2: result.json and summary.md after a failed health check hold no credential."""
    from maxim.simulation import roy_runner
    from maxim.utils.atomic_io import atomic_write_json, atomic_write_text
    from maxim.utils.format_version import with_format_version

    result = roy_runner.RoyIterationResult.__new__(roy_runner.RoyIterationResult)
    for f in dataclasses.fields(roy_runner.RoyIterationResult):
        if f.default is not dataclasses.MISSING:
            setattr(result, f.name, f.default)
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            setattr(result, f.name, f.default_factory())  # type: ignore[misc]
        else:
            setattr(result, f.name, "spec")
    result.preflight = {"url": URL, "outcome": "probe_error", "detail": f"ConnectError: {URL}"}
    result.priming = {"error": f"preflight: ConnectError: {URL}"}
    result.aborted_at = "preflight"
    roy_runner._persist_result(result, tmp_path, atomic_write_json, atomic_write_text, with_format_version)
    assert _clean((tmp_path / "result.json").read_text())
    assert _clean((tmp_path / "summary.md").read_text())
    assert _has_origin((tmp_path / "result.json").read_text())


def test_report_llm_summary_reduced() -> None:
    from maxim.simulation import report

    rep = report.SimulationReport.__new__(report.SimulationReport)
    for f in dataclasses.fields(report.SimulationReport):
        if f.default is not dataclasses.MISSING:
            setattr(rep, f.name, f.default)
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            setattr(rep, f.name, f.default_factory())  # type: ignore[misc]
        else:
            setattr(rep, f.name, None)

    class _Router:
        def generate_json(self, *a: Any, **k: Any) -> Any:
            raise ConnectionError(f"POST {URL}")

    with patch.object(report, "_build_roundup_prompt", return_value="p"):
        report.analyze_simulation(rep, _Router())
    assert rep.llm_summary.startswith("Roundup failed") and _clean(rep.llm_summary)


# ─── sites that cut before reducing (F2, G1, H1, K2, L1) ─────────────────


def test_router_warmup_errors_reduced_before_the_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.models.language import router as router_mod

    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(router_mod, "log_agentic", lambda src, ev, data=None, level="INFO": logged.append(data or {}))
    monkeypatch.setattr(router_mod, "warn", lambda *a, **k: None)
    monkeypatch.setattr(router_mod, "info", lambda *a, **k: None)
    r = router_mod.LLMRouter.__new__(router_mod.LLMRouter)
    r.cfg = SimpleNamespace(model_path="", enabled=True)

    class _Backend:
        def complete(self, **k: Any) -> Any:
            raise ConnectionError(LONG_TOKEN_URL)

    r._get_backend = lambda: _Backend()  # type: ignore[method-assign]
    r._warm_default_backend()
    # the thread site
    r._refuse_offline_lanes = lambda *_: None  # type: ignore[method-assign]
    r.enabled = lambda: True  # type: ignore[method-assign]
    r._backend_lock = threading.Lock()
    r._unloaded = False
    r._warmups_running = 0
    r._deferred_release = []
    r._release_backends = lambda d: None  # type: ignore[method-assign]
    r._ready_event = threading.Event()

    def _raise() -> Any:
        raise ConnectionError(LONG_TOKEN_URL)

    r._warm_default_backend = _raise  # type: ignore[method-assign]
    assert r.warmup() is True
    assert r._ready_event.wait(5.0)
    errors = [d["error"] for d in logged if d.get("context") == "warmup"]
    assert len(errors) == 2 and all(_clean(e) for e in errors), errors


def test_code_tools_reduce_before_the_tail_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.tools import code_tools

    done = SimpleNamespace(returncode=1, stdout=_tail_url(2000), stderr=_tail_url(1000))
    monkeypatch.setattr(code_tools.subprocess, "run", lambda *a, **k: done)
    monkeypatch.setenv("MAXIM_ALLOW_RUN_TESTS", "1")
    res = code_tools.RunTestsTool().execute(command="python -m pytest")
    assert res.metadata.get("returncode") == 1
    assert _clean(res.output) and _clean(res.error)


def test_openai_sanitize_reduces_before_the_cut() -> None:
    from maxim.models.language.openai_backend import _sanitize_error_message

    msg = _sanitize_error_message(RuntimeError("e" * 480 + " " + LONG_TOKEN_URL))
    assert _clean(msg)


def test_recovery_bench_classify_error_unhandled_branch() -> None:
    """K2 / L1: ``_classify_error``'s ``unhandled_*`` branch, the likeliest to carry a raw httpx URL."""
    from maxim.bench.recovery_time import _classify_error

    tag, hint = _classify_error(ConnectionError(f"GET {URL}"))
    assert tag == "unhandled_ConnectionError" and hint == "GET https://host.test"


def test_context_pool_outcome_reduced_before_the_cut() -> None:
    from maxim.agents.context_pool import ContextPool

    pool = ContextPool()
    pool.add_outcome(tool_name="t", success=False, error="x" * 50 + " " + LONG_TOKEN_URL)
    assert _clean(pool.get_context_text())


def test_router_provider_failure_reduced_before_the_cut() -> None:
    from maxim.models.language import router as router_mod

    r = router_mod.LLMRouter.__new__(router_mod.LLMRouter)
    r._is_cancelled = lambda: False  # type: ignore[method-assign]
    r._provider_states = {"p": router_mod.ProviderState()}
    r._note_provider_failure("p", "x" * 170 + " " + LONG_TOKEN_URL)
    assert _clean(r._provider_states["p"].last_error)


def test_log_buffer_install_runs_the_chain_check() -> None:
    """R4: each ``_LogBuffer`` install checks the factory chain and installs it when it is missing."""
    from maxim.runtime import leader_proxy

    logging.setLogRecordFactory(logging.LogRecord)
    leader_proxy._log_buffer = None
    buf = leader_proxy._ensure_log_buffer()
    try:
        rec = logging.getLogRecordFactory()("t", logging.ERROR, __file__, 1, "x %.25s", (LONG_TOKEN_URL,), None)
        assert _clean(rec.getMessage())
    finally:
        logging.getLogger("maxim").removeHandler(buf)
        leader_proxy._log_buffer = None

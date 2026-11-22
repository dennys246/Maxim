"""#1751: an unparseable follow-up input is logged by shape, never by content.

The input can be real tool output, including output of PRIVATE or SECRET tools, and log files
(``MAXIM_LOG_FILE``) persist it. The WARNING must carry only a length and a stable digest.
"""

from __future__ import annotations

import logging

from maxim.agents.prompt_builder import PromptBuilder
from maxim.utils.seeding import stable_hash_32

SENTINEL = "SENTINEL_LOCKER_1751"


def _builder() -> PromptBuilder:
    from maxim.agents.llm_fallback import ReasoningCarryover
    from maxim.models.language.token_counter import CharEstimateCounter

    return PromptBuilder(
        llm=None,
        reasoning_carryover=ReasoningCarryover(),
        n_ctx=32000,
        token_counter=CharEstimateCounter(),
    )


def _warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.WARNING and "followup" in r.getMessage().lower()]


def test_unparseable_followup_warning_carries_no_content(caplog) -> None:
    raw = f"{SENTINEL} private tool output that matches no follow-up format"
    with caplog.at_level(logging.DEBUG, logger="maxim"):
        _builder()._build_followup_prompt(raw)
    records = _warnings(caplog)
    assert records, "the unparseable input must still be reported"
    for record in caplog.records:  # no level, not only WARNING, may carry the content
        assert SENTINEL not in record.getMessage()
        assert SENTINEL not in str(record.args)


def test_warning_identifies_the_input_by_length_and_stable_digest(caplog) -> None:
    raw = f"{SENTINEL} body"
    with caplog.at_level(logging.WARNING, logger="maxim"):
        _builder()._build_followup_prompt(raw)
    message = _warnings(caplog)[0].getMessage()
    assert f"len={len(raw)}" in message
    assert f"{stable_hash_32(raw):08x}" in message


def test_unparseable_input_still_reaches_the_prompt_as_a_private_result() -> None:
    raw = f"{SENTINEL} body"
    prompt = _builder()._build_followup_prompt(raw)
    assert SENTINEL in prompt  # the model still sees it; only the log is shape-only


def test_a_lone_surrogate_input_still_builds_a_prompt(caplog) -> None:
    # Text decoded with surrogateescape (an undecodable file name, subprocess output) must not crash the log line.
    raw = f"[ACTION_FOLLOWUP broken {SENTINEL} \udcff"
    with caplog.at_level(logging.WARNING, logger="maxim"):
        prompt = _builder()._build_followup_prompt(raw)
    assert SENTINEL in prompt
    assert all(SENTINEL not in r.getMessage() for r in caplog.records)


# --- the shared helpers (utils/cloud_redaction.py) -----------------------------------------------------------------


def test_stable_hashes_accept_lone_surrogates_and_keep_every_strict_value() -> None:
    import hashlib

    from maxim.utils.seeding import stable_hash_64_signed

    assert isinstance(stable_hash_32("x\udcff"), int)
    assert isinstance(stable_hash_64_signed("x\udcff"), int)
    for text in ("", "abc", "é漢🙂", SENTINEL):  # strictly encodable: the hash any persisted value carries
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        assert stable_hash_32(text) == int.from_bytes(digest[:4], "big")
        assert stable_hash_64_signed(text) == int.from_bytes(digest[:8], "big", signed=True)


def test_loggable_tool_text_follows_the_tool_sensitivity_table() -> None:
    from maxim.utils.cloud_redaction import loggable_tool_text, text_shape, tool_text_is_loggable

    text = f"{SENTINEL} " * 20
    assert loggable_tool_text("internet_search", text, limit=80) == text[:80]  # PUBLIC
    assert loggable_tool_text("respond", text, limit=80) == text[:80]  # INTERNAL
    for tool in ("read_file", "bash", "sense_some_entity", ""):  # PRIVATE, SECRET, unclassified, unknown
        assert not tool_text_is_loggable(tool)
        assert loggable_tool_text(tool, text, limit=80) == text_shape(text)
        assert SENTINEL not in loggable_tool_text(tool, text, limit=80)


# --- the other sites that logged a tool's output (#1751 review round) ----------------------------------------------


class _Executor:
    def __init__(self, output: object = None, error: str | None = None) -> None:
        self._output, self._error = output, error

    def execute(self, action):
        from maxim.tools.base import ToolOutput

        return ToolOutput(success=self._error is None, output=self._output, error=self._error)


def _gated_run(monkeypatch, tool: str, *, output: object = None, error: str | None = None) -> list[str]:
    from maxim.agents.fear_agent import FearAgent
    from maxim.runtime.fear_gate import FearGatedExecutor
    from maxim.simulation import sim_logger

    summaries: list[str] = []
    monkeypatch.setattr(sim_logger, "sim_action", lambda tool, success, summary="", **kw: summaries.append(summary))
    FearGatedExecutor(_Executor(output, error), FearAgent(llm=None)).execute({"tool_name": tool, "params": {}})
    return summaries


def test_fear_gate_sim_summary_is_shape_only_for_a_private_or_secret_tool(monkeypatch, caplog) -> None:
    with caplog.at_level(logging.DEBUG, logger="maxim"):
        summaries = _gated_run(monkeypatch, "read_file", output=f"{SENTINEL} file body")
        summaries += _gated_run(monkeypatch, "read_file", error=f"{SENTINEL} stderr")
    assert len(summaries) == 2
    assert all(SENTINEL not in s and s.startswith("<len=") for s in summaries)
    assert all(SENTINEL not in r.getMessage() for r in caplog.records)  # the "Tool %s failed" INFO line too


def test_fear_gate_sim_summary_keeps_a_public_tools_text(monkeypatch) -> None:
    summaries = _gated_run(monkeypatch, "internet_search", output=f"{SENTINEL} search hit")
    assert summaries == [f"{SENTINEL} search hit"[:80]]


def test_agentic_tool_result_event_is_shape_only_for_a_private_tool(monkeypatch) -> None:
    from maxim.runtime import tool_dispatch

    events: list[dict] = []
    monkeypatch.setattr(tool_dispatch, "log_agentic", lambda source, event, data=None, **kw: events.append(data))
    tool_dispatch.log_tool_result("read_file", f"{SENTINEL} file body")
    tool_dispatch.log_tool_result("read_file", {"content": SENTINEL})
    tool_dispatch.log_tool_result("internet_search", {"hits": [SENTINEL]})
    tool_dispatch.log_tool_result("read_file", "")
    assert [SENTINEL in str(e["output"]) for e in events] == [False, False, True]  # empty output logs nothing


def _enclosing_functions(tree) -> dict[int, str]:
    import ast

    owner: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    owner[id(child)] = node.name  # innermost wins: walk visits outer defs first
    return owner


def test_every_agentic_tool_result_event_goes_through_log_tool_result() -> None:
    """Structural guard for both callers (execute_and_learn, _run_agentic_loop_body), which a unit test cannot
    drive cheaply: re-adding the old inline ``log_agentic(..., "tool_result", {... str(output)[:100]})`` block
    anywhere in src/maxim fails here. The repo-wide sweep of every log call is backlog M96."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "src" / "maxim"
    offenders, callers = [], set()
    for path in src.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner = _enclosing_functions(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if name in ("log_tool_result", "_log_tool_result"):
                callers.add(owner.get(id(node), ""))
            event = node.args[1] if name == "log_agentic" and len(node.args) > 1 else None
            helper = {"tool_result": "log_tool_result", "goal_failed": "log_tool_failure"}
            if isinstance(event, ast.Constant) and event.value in helper:
                if owner.get(id(node)) != helper[event.value]:
                    offenders.append(f"{path.relative_to(src)}:{node.lineno} ({event.value})")
    assert not offenders, f"tool_result/goal_failed logged outside their tool_dispatch helper: {offenders}"
    assert {"execute_and_learn", "_run_agentic_loop_body"} <= callers, callers


def test_goal_failed_event_shows_a_private_tools_error_by_shape_and_keeps_none(monkeypatch) -> None:
    from maxim.runtime import tool_dispatch

    events: list[dict] = []
    monkeypatch.setattr(tool_dispatch, "log_agentic", lambda source, event, data=None, **kw: events.append(data))
    tool_dispatch.log_tool_failure("run_tests", f"{SENTINEL} stderr")  # SECRET
    tool_dispatch.log_tool_failure("internet_search", f"{SENTINEL} timeout")  # PUBLIC: whole, as before
    tool_dispatch.log_tool_failure("git_diff", None)
    assert SENTINEL not in events[0]["error"] and events[0]["error"].startswith("<len=")
    assert events[1]["error"] == f"{SENTINEL} timeout"
    assert events[2]["error"] is None


def test_write_file_auto_recovery_warning_carries_no_error_text(caplog) -> None:
    from maxim.runtime import tool_dispatch

    failing = _Executor(error=f"{SENTINEL} cannot write")  # write_file is PRIVATE
    action = {"tool_name": "write_file", "params": {"path": "/tmp/x"}}
    with caplog.at_level(logging.DEBUG, logger="maxim"):
        _result, success = tool_dispatch._retry_write_file_overwrite(failing, action, None, False)
    assert success is False
    assert any("Auto-recovery failed for write_file" in r.getMessage() for r in caplog.records)
    assert all(SENTINEL not in r.getMessage() for r in caplog.records)

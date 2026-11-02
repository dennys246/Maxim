"""Unit tests for ``maxim.advisor._session`` beyond the red gates (``tests/integration/test_advisor_gate.py``).

The gates pin the §1A.8 contract end to end. These run each refusal and failure branch in-process (coverage has no
subprocess support, notes §1A.10), against a fake backend through the ``_backend_factory`` seam.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from maxim import advisor
from maxim.advisor import _session
from maxim.advisor.log import read_log
from maxim.exceptions import (
    AdvisorCompositionError,
    AdvisorInputError,
    AdvisorReportWriteFailed,
    AdvisorStopped,
    SealedRegistryError,
)

GOOD = {
    "happening": "a person passed",
    "would_do": "keep watching",
    "rationale": "one sighting",
    "confidence": 0.5,
    "calls": [{"name": "propose_action", "arguments": {"summary": "watch the hall", "target": "hall"}}],
}
EVENT = {"event_id": "e1", "kind": "note", "text": "someone walked by", "payload": {"zone": "hall"}}


class _Backend:
    n_ctx = 4096

    def __init__(self, completion: Any = None, *, on_complete: Any = None) -> None:
        self.completion = GOOD if completion is None else completion
        self.on_complete = on_complete

    def build_prompt(self, template: str, segments: Any) -> Any:
        segs = list(segments)
        return SimpleNamespace(
            ids=list(range(len(segs) + 2)), untrusted_spans=[(i, i + 1) for i in range(len(segs))], rejected=()
        )

    def complete_tokens(self, prompt_ids: Any, **kw: Any) -> Any:
        if self.on_complete is not None:
            self.on_complete()
        text = self.completion if isinstance(self.completion, str) else json.dumps(self.completion)
        return SimpleNamespace(text=text, deadline_hit=False, stopped=False, finish_reason="stop")


@pytest.fixture
def dirs(tmp_path: Path) -> SimpleNamespace:
    home, reports = tmp_path / "home", tmp_path / "reports"
    home.mkdir()
    reports.mkdir()
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF-fake")
    return SimpleNamespace(home=home, reports=reports, model=model)


def _open(monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace, backend: Any = None, **kw: Any) -> Any:
    fake = backend or _Backend()
    monkeypatch.setattr(_session, "_backend_factory", lambda *a, **k: fake)
    return advisor.open(home_dir=dirs.home, model=dirs.model, report_dir=dirs.reports, **kw)


# ── ProposeActionTool ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [{"summary": ""}, {"summary": "x" * 501}, {"summary": "ok", "target": 3}, {"summary": "ok", "command": "ls"}],
    ids=["empty", "too_long", "target_not_str", "unknown_key"],
)
def test_propose_action_refuses_bad_arguments(kwargs: dict[str, Any]) -> None:
    out = _session.ProposeActionTool().run(**kwargs)
    assert not out.success and "propose_action" in str(out.error)


def test_propose_action_returns_exactly_what_it_validated() -> None:
    out = _session.ProposeActionTool().run(summary="watch", rationale="why")
    assert out.success and out.output == {"summary": "watch", "rationale": "why"}


# ── SealedToolRegistry ───────────────────────────────────────────────────


def test_the_sealed_registry_answers_every_read() -> None:
    tool = _session.ProposeActionTool()
    reg = _session.SealedToolRegistry({"propose_action": tool})
    assert reg.get("propose_action") is tool and reg.find("propose_action") is tool and reg.find("bash") is None
    assert reg.list() == reg.list_all() == reg.advertised() == ["propose_action"]
    assert reg.get_active_tools() == [tool] and reg.is_tool_active("propose_action")
    assert not reg.is_tool_active("bash") and reg.get_tool_scene("propose_action") is None
    assert reg.get_scene_tools("s") == [] and reg.get_active_scenes() == [] and reg.get_auto_fire_tools() == []
    assert reg.get_tools_by_kind("core-universal") == [tool] and reg.find_similar("propose") == []
    assert set(reg.tool_descriptions()) == {"propose_action"}
    assert reg.ceiling.name == "READ"
    with pytest.raises(KeyError):
        reg.get("bash")
    with pytest.raises(SealedRegistryError):
        del reg._tools


def test_the_sealed_registry_refuses_a_misnamed_tool() -> None:
    with pytest.raises(TypeError, match="own name"):
        _session.SealedToolRegistry({"other": _session.ProposeActionTool()})


def test_the_builder_refuses_unknown_layers_and_a_sealed_registry_with_tools() -> None:
    tool = _session.ProposeActionTool()
    with pytest.raises(ValueError, match="unknown layers"):
        _session._build_dispatch_stack(tool, layers={"sealed", "magic"}, registry_tools=None)
    with pytest.raises(ValueError, match="registry_tools must be None"):
        _session._build_dispatch_stack(tool, layers={"sealed"}, registry_tools=[tool])


# ── open() ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kw",
    [
        {"max_calls": 0},
        {"max_calls": True},
        {"turn_timeout_s": 0},
        {"log_mode": 0o644},
        {"n_ctx": 1024},
        {"n_gpu_layers": -2},
        {"cursor": 7},
    ],
    ids=["max_calls_0", "max_calls_bool", "timeout_0", "log_mode", "n_ctx_small", "gpu_layers", "cursor_type"],
)
def test_open_refuses_out_of_range_limits(monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace, kw: Any) -> None:
    with pytest.raises(AdvisorCompositionError):
        _open(monkeypatch, dirs, **kw)
    assert list(dirs.reports.iterdir()) == [], "a refused open() left a log behind"


def test_a_backend_that_fails_to_load_is_open_s_refusal(monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("no such model")

    monkeypatch.setattr(_session, "_backend_factory", boom)
    with pytest.raises(AdvisorCompositionError, match="failed to load"):
        advisor.open(home_dir=dirs.home, model=dirs.model, report_dir=dirs.reports)
    assert list(dirs.reports.iterdir()) == [], "a backend that failed to load still left a log (ordering rule)"


def test_the_real_factory_reaches_closed_local(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from maxim.models.language import closed_local

    seen: list[Any] = []
    monkeypatch.setattr(closed_local, "open_closed_local_backend", lambda *a, **k: seen.append((a, k)) or "backend")
    assert _session._backend_factory("m.gguf", n_ctx=None, n_gpu_layers=0, seed=0) == "backend"
    assert seen == [(("m.gguf",), {"n_ctx": None, "n_gpu_layers": 0, "seed": 0})]


def test_a_session_is_never_built_directly() -> None:
    with pytest.raises(TypeError, match="maxim.advisor.open"):
        _session.AdvisorSession(
            _token=object(),
            **{
                k: None
                for k in (
                    "session_id",
                    "backend",
                    "executor",
                    "propose_tool",
                    "log",
                    "dirs",
                    "limits",
                    "cursor",
                    "max_calls",
                    "turn_timeout_s",
                    "log_mode",
                    "model_sha256",
                )
            },
        )


def test_the_package_exports_lazily_and_refuses_unknown_names() -> None:
    assert advisor.open is _session.open_session and advisor.AdvisorLimits is _session.AdvisorLimits
    with pytest.raises(AttributeError):
        advisor.not_a_name  # noqa: B018


# ── ingest() refusals and failure rows ───────────────────────────────────


@pytest.mark.parametrize(
    "events",
    [
        [],
        "text",
        [42],
        [{**EVENT, "extra": 1}],
        [{k: v for k, v in EVENT.items() if k != "kind"}],
        [{**EVENT, "event_id": ""}],
        [{**EVENT, "cursor": 5}],
        [{**EVENT, "payload": {"x": float("nan")}}],
    ],
    ids=["empty", "string", "not_mapping", "unknown_field", "missing_field", "empty_id", "cursor_type", "nan"],
)
def test_ingest_refuses_a_malformed_batch(monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace, events: Any) -> None:
    session = _open(monkeypatch, dirs)
    try:
        with pytest.raises(AdvisorInputError):
            session.ingest(events)
    finally:
        session.close()


@pytest.mark.parametrize(
    "completion, reason",
    [
        (dict(GOOD, happening=7), "happening"),
        (dict(GOOD, rationale="x" * 2001), "rationale"),
        (dict(GOOD, calls=[{"name": "n" * 129, "arguments": {}}]), "name"),
        (dict(GOOD, confidence=1.5), "confidence"),
        (dict(GOOD, confidence=True), "confidence"),
        (dict(GOOD, calls="none"), "calls"),
        (dict(GOOD, calls=[{"name": "x"}]), "a call"),
        (dict(GOOD, calls=[{"name": 3, "arguments": {}}]), "name"),
        (dict(GOOD, calls=[{"name": "x", "arguments": []}]), "arguments"),
        ([1, 2], "keys"),
    ],
    ids=[
        "text_type",
        "text_too_long",
        "name_too_long",
        "confidence_range",
        "confidence_bool",
        "calls_type",
        "call_keys",
        "name_type",
        "args_type",
        "not_object",
    ],
)
def test_a_schema_invalid_completion_is_an_error(
    monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace, completion: Any, reason: str
) -> None:
    session = _open(monkeypatch, dirs, _Backend(completion))
    try:
        report = session.ingest([EVENT])
    finally:
        session.close()
    assert report["status"] == "error" and report["reason"].startswith("schema_invalid") and reason in report["reason"]
    assert report["proposals"] == [] and report["refusals"] == []


def test_a_bad_propose_call_is_recorded_as_a_failed_call(
    monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace
) -> None:
    """A propose_action whose arguments fail validation RAN and failed: a refusal row at layer "tool", logged."""
    bad = dict(GOOD, calls=[{"name": "propose_action", "arguments": {"summary": ""}}])
    session = _open(monkeypatch, dirs, _Backend(bad))
    try:
        report = session.ingest([EVENT])
    finally:
        session.close()
    assert report["status"] == "ok" and report["proposals"] == []
    assert [(r["name"], r["layer"]) for r in report["refusals"]] == [("propose_action", "tool")]
    # §1A.8's vocabulary, pinned (review W3, owner decision 2026-11-02): the tool's own kind, its real effect.
    row = {k: v for k, v in report["refusals"][0].items() if k in ("error_kind", "effect")}
    assert row == {"error_kind": "invalid_input", "effect": ["READ"]}
    assert read_log(dirs.reports / f"advisor_log_{report['session_id']}.jsonl")["last_seq"] == 1


def test_a_stop_between_dispatches_abandons_the_turn(monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace) -> None:
    flag = threading.Event()
    two = dict(GOOD, calls=GOOD["calls"] * 2)
    session = _open(monkeypatch, dirs, _Backend(two))
    real_execute = session._executor.execute

    def execute_then_stop(action: Any) -> Any:
        out = real_execute(action)
        flag.set()  # lands after the first dispatch
        return out

    monkeypatch.setattr(session._executor, "execute", execute_then_stop)
    try:
        report = session.ingest([EVENT], stop=flag)
    finally:
        session.close()
    assert report["status"] == "abandoned" and report["reason"] == "stopped" and report["proposals"] == []
    assert report["log_last_seq"] == 1, "the record written before the stop stays in the log"


def test_a_failed_log_write_fails_the_turn(monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace) -> None:
    session = _open(monkeypatch, dirs)

    def broken(_record: Any) -> int:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(session._log, "append", broken)
    try:
        report = session.ingest([EVENT])
    finally:
        monkeypatch.undo()
        session.close()
    assert report["status"] == "error" and report["reason"].startswith("log_write_failed")


def test_a_closed_session_refuses_ingest_and_close_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace
) -> None:
    session = _open(monkeypatch, dirs)
    session.close()
    session.close()
    with pytest.raises(AdvisorStopped, match="closed"):
        session.ingest([EVENT])


def test_request_stop_never_raises_on_a_malformed_deadline(
    monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace
) -> None:
    session = _open(monkeypatch, dirs)
    try:
        session.request_stop("soon")  # type: ignore[arg-type]
        assert session._stop_flag and session._stop_deadline is not None
        with pytest.raises(AdvisorStopped, match="stopped"):
            session.ingest([EVENT])
    finally:
        session.close()


def test_the_report_on_disk_is_the_report_returned(monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace) -> None:
    session = _open(monkeypatch, dirs)
    try:
        report = session.ingest([EVENT])
    finally:
        session.close()
    on_disk = json.loads((dirs.reports / f"advisor_report_{report['report_id']}.json").read_text())
    assert on_disk == report and report["status"] == "ok" and report["inputs"][0]["rejected"] is None


def test_open_engages_the_offline_latch_before_it_builds_anything(
    monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace
) -> None:
    """§4.2: the latch is up before the backend is built, and even when open() then refuses (a deletion probe of
    ``engage`` in ``open_session`` survived every other test)."""
    from maxim import _offline

    seen: list[bool] = []

    def factory(*_a: Any, **_k: Any) -> Any:
        seen.append(_offline.is_offline())
        return _Backend()

    monkeypatch.setattr(_session, "_backend_factory", factory)
    session = advisor.open(home_dir=dirs.home, model=dirs.model, report_dir=dirs.reports)
    session.close()
    assert seen == [True], "the backend was built before the offline latch was engaged"
    _offline._reset_for_tests()
    with pytest.raises(AdvisorCompositionError):
        advisor.open(home_dir=dirs.home, model=dirs.model / "missing", report_dir=dirs.reports)
    assert _offline.is_offline(), "a refused open() did not engage the latch first"


# ── item 1 review folds (2026-11-02) ─────────────────────────────────────


class _BlockingBackend(_Backend):
    """complete_tokens blocks until released, then reports the stop it saw (as closed_local's criterion would)."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.closed = 0

    def complete_tokens(self, prompt_ids: Any, **kw: Any) -> Any:
        self.calls += 1
        self.entered.set()
        assert self.release.wait(10)
        stopped = bool(kw["should_stop"]())
        return SimpleNamespace(text=json.dumps(GOOD), deadline_hit=False, stopped=stopped, finish_reason="stop")

    def close(self) -> None:
        self.closed += 1


def test_close_under_a_running_turn_hands_the_release_to_the_turn(
    monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace, tmp_path: Path
) -> None:
    """B1/W1: close() during a turn never closes the report fd under it. Before the fold the turn wrote its report
    through a closed, reused fd number, into whatever directory now held it."""
    backend = _BlockingBackend()
    session = _open(monkeypatch, dirs, backend)
    result: dict[str, Any] = {}
    worker = threading.Thread(target=lambda: result.update(report=session.ingest([EVENT])))
    worker.start()
    assert backend.entered.wait(10)
    session.request_stop(0.0, reason="shutdown")
    session.close()  # deadline passed: hands off and returns at once
    others = [tmp_path / f"other{i}" for i in range(4)]
    fds = []
    for other in others:  # reuse any fd number close() might have freed
        other.mkdir()
        fds.append(os.open(other, os.O_RDONLY | os.O_DIRECTORY))
    try:
        assert backend.closed == 0, "the model was released under a running turn"
        backend.release.set()
        worker.join(10)
    finally:
        for fd in fds:
            os.close(fd)
    report = result["report"]
    assert (report["status"], report["reason"], report["stop_reason"]) == ("abandoned", "stopped", "shutdown")
    assert (dirs.reports / f"advisor_report_{report['report_id']}.json").is_file()
    assert all(not any(o.iterdir()) for o in others), "a report landed outside report_dir"
    log = read_log(dirs.reports / f"advisor_log_{report['session_id']}.jsonl")
    assert log["closed"] and backend.closed == 1


def test_close_waits_for_the_turn_until_the_stop_deadline(
    monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace
) -> None:
    backend = _BlockingBackend()
    session = _open(monkeypatch, dirs, backend)
    worker = threading.Thread(target=lambda: session.ingest([EVENT]))
    worker.start()
    assert backend.entered.wait(10)
    session.request_stop(10.0)
    threading.Timer(0.2, backend.release.set).start()
    session.close()  # returns once the turn has released everything, well inside the deadline
    assert backend.closed == 1 and session._released
    worker.join(10)


def test_an_ingest_waiting_on_the_lock_runs_no_turn_after_a_stop(
    monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace
) -> None:
    """S4: the stop is re-checked under the turn lock, so a queued call raises instead of running a full turn."""
    backend = _BlockingBackend()
    session = _open(monkeypatch, dirs, backend)
    errors: list[BaseException] = []

    def second() -> None:
        try:
            session.ingest([EVENT])
        except AdvisorStopped as e:
            errors.append(e)

    first = threading.Thread(target=lambda: session.ingest([EVENT]))
    first.start()
    assert backend.entered.wait(10)
    queued = threading.Thread(target=second)
    queued.start()
    session.request_stop(5.0)
    backend.release.set()
    first.join(10)
    queued.join(10)
    session.close()
    assert backend.calls == 1 and len(errors) == 1


def test_a_failed_report_write_raises_typed_and_keeps_the_cursor(
    monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace
) -> None:
    """S5/W2: the cursor advances only after the report lands."""
    session = _open(monkeypatch, dirs, cursor="c0")

    def full(*_a: Any, **_k: Any) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(_session, "atomic_write_json_at", full)
    try:
        with pytest.raises(AdvisorReportWriteFailed):
            session.ingest([{**EVENT, "cursor": "c1"}])
        assert session._cursor == "c0"
    finally:
        session.close()


def test_a_lone_surrogate_and_a_bad_stop_are_refused_before_any_turn(
    monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace
) -> None:
    session = _open(monkeypatch, dirs)
    try:
        with pytest.raises(AdvisorInputError, match="UTF-8"):
            session.ingest([{**EVENT, "text": "a\ud800b"}])
        with pytest.raises(AdvisorInputError, match="is_set"):
            session.ingest([EVENT], stop=object())
    finally:
        session.close()


def test_a_failed_prompt_build_is_an_error_report(monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace) -> None:
    backend = _Backend()

    def boom(*_a: Any) -> Any:
        raise RuntimeError("tokenizer")

    backend.build_prompt = boom  # type: ignore[method-assign]
    session = _open(monkeypatch, dirs, backend, cursor="c0")
    try:
        report = session.ingest([{**EVENT, "cursor": "c1"}])
    finally:
        session.close()
    assert (report["status"], report["reason"]) == ("error", "prompt_build_error: RuntimeError")
    assert (
        report["covered"]["cursor_out"] == "c0"
        and (dirs.reports / f"advisor_report_{report['report_id']}.json").is_file()
    )


def test_a_truncated_completion_is_its_own_error(monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace) -> None:
    backend = _Backend()
    backend.complete_tokens = lambda *a, **k: SimpleNamespace(  # type: ignore[method-assign]
        text='{"happening": "x', deadline_hit=False, stopped=False, finish_reason="length"
    )
    session = _open(monkeypatch, dirs, backend)
    try:
        report = session.ingest([EVENT])
    finally:
        session.close()
    assert (report["status"], report["reason"]) == ("error", "completion_truncated")


def test_a_per_call_stop_names_itself(monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace) -> None:
    flag = threading.Event()
    flag.set()
    backend = _Backend()
    backend.complete_tokens = lambda *a, **k: SimpleNamespace(  # type: ignore[method-assign]
        text="", deadline_hit=False, stopped=bool(k["should_stop"]()), finish_reason="stop"
    )
    session = _open(monkeypatch, dirs, backend)
    try:
        report = session.ingest([EVENT], stop=flag)
    finally:
        session.close()
    assert (report["status"], report["stop_reason"]) == ("abandoned", "per_call")


@pytest.mark.parametrize(
    "deadline, reason, expect_reason",
    [(10**400, "x" * 500, "x" * 128), (float("inf"), "ok", "ok"), (-3, type("S", (str,), {})("sub"), "unnamed")],
    ids=["huge_int", "inf", "negative_str_subclass"],
)
def test_request_stop_reads_only_plain_values(
    monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace, deadline: Any, reason: Any, expect_reason: str
) -> None:
    session = _open(monkeypatch, dirs)
    try:
        session.request_stop(deadline, reason=reason)
        assert session._stop_reason == expect_reason and session._stop_deadline is not None
    finally:
        session.close()


def test_open_refuses_an_infinite_turn_timeout(monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace) -> None:
    with pytest.raises(AdvisorCompositionError, match="finite"):
        _open(monkeypatch, dirs, turn_timeout_s=float("inf"))


def _open_fds() -> set[int]:
    return {int(n) for n in os.listdir("/dev/fd")}


def test_a_model_that_cannot_be_hashed_is_refused_and_leaks_nothing(
    monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace
) -> None:
    """S1/A3/W5a: the hash runs inside open()'s guard, before the load and the log."""
    from maxim.advisor import _refusals

    def unreadable(_fd: int, _n: int, _off: int) -> bytes:
        raise OSError(5, "Input/output error")

    before = _open_fds()
    monkeypatch.setattr(_refusals.os, "pread", unreadable)
    with pytest.raises(AdvisorCompositionError, match="could not be read"):
        _open(monkeypatch, dirs)
    assert _open_fds() <= before and list(dirs.reports.iterdir()) == []


def test_a_model_swapped_during_the_load_is_refused(monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace) -> None:
    """S2/A3/W5b: the hash names the file that was loaded, or open() refuses."""
    backend = _BlockingBackend()

    def load_and_swap(*_a: Any, **_k: Any) -> Any:
        swapped = dirs.model.with_name("swapped.gguf")
        swapped.write_bytes(b"GGUF-other-model")
        os.replace(swapped, dirs.model)
        return backend

    monkeypatch.setattr(_session, "_backend_factory", load_and_swap)
    before = _open_fds()
    with pytest.raises(AdvisorCompositionError, match="changed during open"):
        advisor.open(home_dir=dirs.home, model=dirs.model, report_dir=dirs.reports)
    assert backend.closed == 1 and _open_fds() <= before and list(dirs.reports.iterdir()) == []


def test_the_report_hash_is_the_model_s(monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace) -> None:
    import hashlib

    session = _open(monkeypatch, dirs)
    try:
        report = session.ingest([EVENT])
    finally:
        session.close()
    assert report["provenance"]["model_sha256"] == hashlib.sha256(dirs.model.read_bytes()).hexdigest()


def test_close_releases_the_model(monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace) -> None:
    backend = _BlockingBackend()
    session = _open(monkeypatch, dirs, backend)
    session.close()
    session.close()
    assert backend.closed == 1 and session._backend is None


def test_the_sealed_registry_refuses_setattr() -> None:
    reg = _session.SealedToolRegistry({"propose_action": _session.ProposeActionTool()})
    with pytest.raises(SealedRegistryError):
        reg._tools = {}  # type: ignore[misc]


def test_open_refuses_an_unwritable_report_dir(monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace) -> None:
    if os.geteuid() == 0:
        pytest.skip("root writes anywhere; the mode bit cannot refuse it")
    dirs.reports.chmod(0o500)
    try:
        with pytest.raises(AdvisorCompositionError, match="writable"):
            _open(monkeypatch, dirs)
    finally:
        dirs.reports.chmod(0o700)


def test_advisor_log_is_reachable_without_opening_a_session() -> None:
    """Review N7/W10: ``maxim.advisor.log.read_log`` works in a reader process that never called open()."""
    import subprocess
    import sys

    code = "import maxim.advisor as a; print(a.log.read_log.__name__)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "read_log"


# ── the boundary's importers and callers, pinned by AST (review N4/N8) ──


def _python_files(*roots: str) -> list[Path]:
    repo = Path(__file__).resolve().parents[2]
    return [p for root in roots for p in sorted((repo / root).rglob("*.py"))]


def _imports_closed_local(tree: Any) -> bool:
    import ast

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.endswith("closed_local") or (
                node.module == "maxim.models.language" and any(a.name == "closed_local" for a in node.names)
            ):
                return True
        if isinstance(node, ast.Import) and any(a.name.endswith("closed_local") for a in node.names):
            return True
        # importlib.import_module("...closed_local") and `language.closed_local` attribute access
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith("closed_local"):
            return True
        if isinstance(node, ast.Attribute) and node.attr == "closed_local":
            return True
    return False


def test_only_the_observer_imports_the_closed_backend() -> None:
    """The closed backend bypasses the router by design, so its one importer is pinned: every import form, over
    ``src/`` and ``scripts/`` (the CI grep matches one form and exempts ``models/language/``)."""
    import ast

    repo = Path(__file__).resolve().parents[2]
    importers = sorted(
        str(p.relative_to(repo))
        for p in _python_files("src", "scripts")
        if _imports_closed_local(ast.parse(p.read_text(encoding="utf-8")))
    )
    assert importers == ["src/maxim/advisor/_session.py"]


def test_only_open_session_builds_the_dispatch_stack() -> None:
    """``_build_dispatch_stack``'s reduced-layer path is test-only: in ``src/`` its one caller is ``open_session``."""
    import ast

    callers = []
    for path in _python_files("src"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) == (
                    "_build_dispatch_stack"
                ):
                    callers.append((path.name, fn.name))
    assert callers == [("_session.py", "open_session")]


# ── delta round 1 folds ──────────────────────────────────────────────────


def test_an_empty_call_name_is_schema_invalid_and_logs_nothing(
    monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace
) -> None:
    """SF1: an empty name never reaches the executor, so it cannot be recorded as the pinned tool having run."""
    assert _session.completion_schema(4)["properties"]["calls"]["items"]["properties"]["name"]["minLength"] == 1
    session = _open(monkeypatch, dirs, _Backend(dict(GOOD, calls=[{"name": "", "arguments": {}}])))
    try:
        report = session.ingest([EVENT])
    finally:
        session.close()
    assert report["status"] == "error" and report["reason"].startswith("schema_invalid")
    assert read_log(dirs.reports / f"advisor_log_{report['session_id']}.jsonl")["last_seq"] == 0


def test_a_no_layer_failure_that_is_not_the_pinned_tool_claims_no_effect() -> None:
    from maxim.tools.base import ToolOutput

    out = ToolOutput(success=False, output=None, error="Invalid action")
    row = _session._refusal_record("other", out, ran=None)
    assert (row["layer"], row["effect"], row["error_kind"]) == ("tool", "unknown", "invalid_input")


class _BadCloseBackend(_BlockingBackend):
    def close(self) -> None:
        self.closed += 1
        raise RuntimeError("llama close failed")


def test_a_release_error_after_a_handed_off_turn_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    """Delta N1: the turn's report stands and close() is signalled even when the model's release raises."""
    backend = _BadCloseBackend()
    session = _open(monkeypatch, dirs, backend)
    result: dict[str, Any] = {}
    worker = threading.Thread(target=lambda: result.update(report=session.ingest([EVENT])))
    worker.start()
    assert backend.entered.wait(10)
    session.close()
    backend.release.set()
    worker.join(10)
    assert result["report"]["status"] == "abandoned" and backend.closed == 1
    assert session._turn_done.is_set() and "releasing after the last turn failed" in caplog.text


def test_a_failed_open_raises_its_own_error_and_logs_a_failed_cleanup(
    monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    """Delta N4: a cleanup step that raises is logged; the error that failed open() is the one raised."""
    backend = _BadCloseBackend()

    def no_log(*_a: Any, **_k: Any) -> Any:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(_session.AdvisorLog, "create", no_log)
    with pytest.raises(OSError, match="No space"):
        _open(monkeypatch, dirs, backend)
    assert backend.closed == 1 and "releasing the model after a failed open failed" in caplog.text


def test_a_per_call_stop_keeps_its_label_when_close_follows(
    monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace
) -> None:
    """Delta N5: the first stop a turn sees names it. Here the per-call stop is seen mid-sampling, then close() lands
    before the turn labels its report: the label stays ``per_call``."""
    flag = threading.Event()
    seen = threading.Event()
    release = threading.Event()
    backend = _Backend()

    def complete(*_a: Any, **kw: Any) -> Any:
        stopped = bool(kw["should_stop"]())
        seen.set()
        assert release.wait(10)
        return SimpleNamespace(text="", deadline_hit=False, stopped=stopped, finish_reason="stop")

    backend.complete_tokens = complete  # type: ignore[method-assign]
    session = _open(monkeypatch, dirs, backend)
    result: dict[str, Any] = {}
    flag.set()
    worker = threading.Thread(target=lambda: result.update(report=session.ingest([EVENT], stop=flag)))
    worker.start()
    assert seen.wait(10)
    session.close()  # sets the session's stop while the turn still runs; hands the release to the turn
    release.set()
    worker.join(10)
    assert result["report"]["stop_reason"] == "per_call" and session._released


def test_a_dispatched_no_layer_failure_names_only_the_pinned_tool_s_effect(
    monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace
) -> None:
    """SF1, through ``_dispatch``: a no-layer failure for any name but the pinned tool's never claims its effect."""
    from maxim.tools.base import ToolOutput

    calls = [{"name": "other", "arguments": {}}, {"name": "propose_action", "arguments": {"summary": ""}}]
    session = _open(monkeypatch, dirs, _Backend(dict(GOOD, calls=calls)))
    real = session._executor.execute

    def execute(action: Any) -> Any:
        if action["tool_name"] == "other":  # as the executor refuses an action before any lookup
            return ToolOutput(success=False, output=None, error="Invalid action")
        return real(action)

    monkeypatch.setattr(session._executor, "execute", execute)
    try:
        report = session.ingest([EVENT])
    finally:
        session.close()
    assert [(r["name"], r["effect"]) for r in report["refusals"]] == [
        ("other", "unknown"),
        ("propose_action", ["READ"]),
    ]


class _Interrupt(BaseException):
    pass


class _InterruptingCloseBackend(_BlockingBackend):
    def close(self) -> None:
        self.closed += 1
        raise _Interrupt


def test_close_is_signalled_even_when_the_handed_off_release_is_interrupted(
    monkeypatch: pytest.MonkeyPatch, dirs: SimpleNamespace
) -> None:
    """Delta N1: ``_turn_done`` is set in its own ``finally``, so not even a BaseException leaves close() waiting."""
    backend = _InterruptingCloseBackend()
    session = _open(monkeypatch, dirs, backend)
    raised: list[BaseException] = []

    def run() -> None:
        try:
            session.ingest([EVENT])
        except _Interrupt as e:
            raised.append(e)

    worker = threading.Thread(target=run)
    worker.start()
    assert backend.entered.wait(10)
    session.close()
    backend.release.set()
    worker.join(10)
    assert raised and session._turn_done.is_set()

"""Unit tests for ``maxim.advisor._session`` beyond the red gates (``tests/integration/test_advisor_gate.py``).

The gates pin the §1A.8 contract end to end. These run each refusal and failure branch in-process (coverage has no
subprocess support, notes §1A.10), against a fake backend through the ``_backend_factory`` seam.
"""

from __future__ import annotations

import json
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
        return SimpleNamespace(text=text, deadline_hit=False, stopped=False)


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
        assert session._stop.is_set() and session._stop_deadline is not None
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

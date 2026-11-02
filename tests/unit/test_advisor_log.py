"""Unit tests for ``maxim.advisor.log`` beyond item 5's red gates (``test_advisor_log_red_gates.py``).

The gates pin the contract the notes name. These cover the rest of the module's branches: the exact mode under a
restrictive umask, the record checks, a closed log, and ``read_log``'s refusals past the header.
"""

from __future__ import annotations

import errno
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from maxim.advisor import log as advisor_log
from maxim.advisor.log import AdvisorLog, read_log
from maxim.exceptions import AdvisorError, AdvisorLogBroken, AdvisorLogMalformed
from maxim.utils import atomic_io

SESSION = "sess-unit"
REPORT_ID = "r-1"


@pytest.fixture
def report_dir(tmp_path: Path) -> Iterator[tuple[Path, int]]:
    d = tmp_path / "reports"
    d.mkdir()
    fd = os.open(d, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        yield d, fd
    finally:
        os.close(fd)


def _path(d: Path) -> Path:
    return d / advisor_log.log_name(SESSION)


def _proposal(**over: Any) -> dict[str, Any]:
    return {"_record_kind": "proposal", "report_id": REPORT_ID, "arguments": {"summary": "s"}, **over}


def _refusal(**over: Any) -> dict[str, Any]:
    rec = {
        "_record_kind": "refusal",
        "report_id": REPORT_ID,
        "name": "bash",
        "error_kind": "policy_pin",
        "layer": "pin",
        "effect": ["ALL"],
        "reason": "r",
    }
    return {**rec, **over}


def test_mode_is_exact_under_a_restrictive_umask(report_dir: tuple[Path, int]) -> None:
    """``mode=0o640`` lands as 0640 even under umask 077: the mode is applied, not masked."""
    d, dfd = report_dir
    old = os.umask(0o077)
    try:
        AdvisorLog.create(dfd, SESSION, mode=0o640).close()
    finally:
        os.umask(old)
    assert os.stat(_path(d)).st_mode & 0o777 == 0o640


@pytest.mark.parametrize(
    "record, exc",
    [
        ({"_record_kind": "header", "report_id": REPORT_ID}, ValueError),
        (_proposal(seq=9), ValueError),
        (_proposal(ts="t"), ValueError),
        ({"_record_kind": "refusal", "report_id": REPORT_ID, "name": "x"}, ValueError),
        (["not", "a", "dict"], TypeError),
    ],
    ids=["unknown_kind", "caller_seq", "caller_ts", "missing_fields", "not_a_dict"],
)
def test_a_refused_record_writes_nothing_and_the_log_stays_usable(
    report_dir: tuple[Path, int], record: Any, exc: type[Exception]
) -> None:
    d, dfd = report_dir
    log = AdvisorLog.create(dfd, SESSION)
    size = _path(d).stat().st_size
    with pytest.raises(exc):
        log.append(record)
    assert _path(d).stat().st_size == size and not log.broken
    assert log.append(_proposal()) == 1
    log.close()


def test_a_long_reason_is_bounded_with_a_flag(report_dir: tuple[Path, int]) -> None:
    d, dfd = report_dir
    log = AdvisorLog.create(dfd, SESSION)
    log.append(_refusal(reason="z" * 5000))
    assert log.last_seq == 1
    log.close()
    rec = read_log(_path(d))["records"][0]
    assert len(rec["reason"]) == advisor_log.MAX_REASON_CHARS and rec["reason_truncated"] is True
    assert "name_truncated" not in rec


def test_truncated_arguments_full_of_escapes_still_fit(report_dir: tuple[Path, int]) -> None:
    """Quotes and backslashes double when the prefix is re-encoded; the bound still holds."""
    d, dfd = report_dir
    log = AdvisorLog.create(dfd, SESSION)
    log.append(_proposal(arguments={"summary": '"\\' * 5000}))
    log.close()
    rec = read_log(_path(d))["records"][0]
    assert rec["arguments_truncated"] is True and isinstance(rec["arguments"], str)
    encoded_len = len(json.dumps(rec["arguments"], ensure_ascii=True))
    assert encoded_len <= advisor_log.MAX_ARGUMENTS_BYTES
    # The longest prefix that fits, not one the shrink loop overshot (Executor SF2): each char of the prefix
    # encodes to at most 2, so it keeps at least half the bound.
    assert len(rec["arguments"]) >= advisor_log.MAX_ARGUMENTS_BYTES // 2 - 1, len(rec["arguments"])
    assert encoded_len >= advisor_log.MAX_ARGUMENTS_BYTES - 2, encoded_len


def test_a_closed_log_refuses_appends_and_close_is_idempotent(report_dir: tuple[Path, int]) -> None:
    d, dfd = report_dir
    log = AdvisorLog.create(dfd, SESSION)
    log.close()
    log.close()
    with pytest.raises(AdvisorLogBroken, match="closed"):
        log.append(_proposal())
    assert read_log(_path(d)) == {"header": read_log(_path(d))["header"], "records": [], "closed": True, "last_seq": 0}


def test_a_broken_log_closes_without_a_close_record(report_dir: tuple[Path, int], monkeypatch: Any) -> None:
    d, dfd = report_dir
    log = AdvisorLog.create(dfd, SESSION)
    log.append(_proposal())

    def failing_sync(fd: int) -> None:
        raise OSError(errno.EIO, "injected")

    monkeypatch.setattr(atomic_io, "fsync_strict", failing_sync)
    with pytest.raises(OSError):
        log.append(_proposal())
    monkeypatch.undo()
    log.close()
    assert log.closed and log.broken
    view = read_log(_path(d))
    assert view["closed"] is False and view["last_seq"] == 2, "the record was written before the fsync failed"


def test_create_closes_its_fd_when_the_header_cannot_be_synced(report_dir: tuple[Path, int], monkeypatch: Any) -> None:
    _d, dfd = report_dir
    closed: list[int] = []
    real_close = os.close

    def failing_sync(fd: int) -> None:
        raise OSError(errno.EIO, "injected")

    def recording_close(fd: int) -> None:
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(advisor_log, "fsync_strict", failing_sync)
    monkeypatch.setattr(os, "close", recording_close)
    with pytest.raises(OSError):
        AdvisorLog.create(dfd, SESSION)
    assert len(closed) == 1 and closed[0] != dfd


def _write_log(d: Path, *lines: dict[str, Any] | bytes) -> Path:
    raw = [ln if isinstance(ln, bytes) else json.dumps(ln).encode() for ln in lines]
    _path(d).write_bytes(b"\n".join(raw) + b"\n")
    return _path(d)


_HEADER = {"_format_version": "1.0", "_record_kind": "header", "schema": "advisor_log/1", "session_id": SESSION}


@pytest.mark.parametrize(
    "lines",
    [
        [{**_HEADER, "schema": "advisor_log/2"}],
        [b"[1, 2]"],
        [_HEADER, {"_record_kind": "proposal", "seq": 2}, {"_record_kind": "proposal", "seq": 2}],
        [_HEADER, {"_record_kind": "proposal", "seq": True}],
        [_HEADER, {"_record_kind": "mystery", "seq": 1}],
        [_HEADER, {"_record_kind": "close", "last_seq": 0}, {"_record_kind": "proposal", "seq": 1}],
        [_HEADER, {"_record_kind": "proposal", "seq": 1}, {"_record_kind": "close", "last_seq": 7}],
    ],
    ids=["wrong_schema", "not_an_object", "seq_repeats", "seq_bool", "unknown_kind", "after_close", "close_disagrees"],
)
def test_read_log_refuses(report_dir: tuple[Path, int], lines: list[Any]) -> None:
    d, _dfd = report_dir
    with pytest.raises(AdvisorLogMalformed):
        read_log(_write_log(d, *lines))


def test_malformed_is_a_value_error_and_broken_a_runtime_error() -> None:
    """The gates catch ``ValueError`` and ``RuntimeError``; the typed classes must stay inside both."""
    assert issubclass(AdvisorLogMalformed, ValueError) and issubclass(AdvisorLogBroken, RuntimeError)
    assert issubclass(AdvisorLogMalformed, AdvisorError) and issubclass(AdvisorLogBroken, AdvisorError)


def test_an_interrupt_mid_append_breaks_the_log_and_it_stays_readable(
    report_dir: tuple[Path, int], monkeypatch: Any
) -> None:
    """A ``KeyboardInterrupt`` between the write and the fsync breaks the log (Executor B1).

    Before the fold, only ``OSError`` broke it: the line stayed on disk with ``seq`` not advanced, and the next
    record (or the close record) made ``read_log`` refuse the whole file.
    """
    d, dfd = report_dir
    log = AdvisorLog.create(dfd, SESSION)
    log.append(_proposal())

    def interrupted(fd: int) -> None:
        raise KeyboardInterrupt

    with monkeypatch.context() as m:
        m.setattr(atomic_io, "fsync_strict", interrupted)
        with pytest.raises(KeyboardInterrupt):
            log.append(_proposal())
    assert log.broken and log.last_seq == 1
    with pytest.raises(AdvisorLogBroken):
        log.append(_proposal())
    log.close()
    view = read_log(_path(d))
    assert view["last_seq"] == 2 and view["closed"] is False, "the interrupted record is the final one, readable"


@pytest.mark.parametrize("session_id", ["x/../../escaped", "..", ".", "", ".hidden", "a\x00b", "a\\b", "x" * 129, 7])
def test_a_session_id_that_is_not_one_safe_component_is_refused(
    report_dir: tuple[Path, int], tmp_path: Path, session_id: Any
) -> None:
    """``session_id`` names a file under the report dir's fd; it can never climb out (Executor SF3, Wire SF-1)."""
    d, dfd = report_dir
    (d / "advisor_log_x").mkdir()  # the escape the review reproduced needs this directory
    before = sorted(p.name for p in tmp_path.rglob("*"))
    with pytest.raises(ValueError, match="session id"):
        AdvisorLog.create(dfd, session_id)
    assert sorted(p.name for p in tmp_path.rglob("*")) == before


@pytest.mark.parametrize("mode", [0o666, 0o644, 0o700, 0o4600, 0, 0o400, 0o040])
def test_a_mode_beyond_0640_is_refused(report_dir: tuple[Path, int], mode: int) -> None:
    d, dfd = report_dir
    with pytest.raises(ValueError, match="mode"):
        AdvisorLog.create(dfd, SESSION, mode=mode)
    assert not _path(d).exists()


@pytest.mark.parametrize(
    "record, exc",
    [
        (_proposal(extra_field="x" * 10_000), ValueError),
        (_refusal(layer=None), TypeError),
        (_refusal(error_kind=3), TypeError),
        (_refusal(effect=7), TypeError),
        (_refusal(effect=["READ", 1]), TypeError),
        (_refusal(effect="READ"), TypeError),
        (_proposal(report_id=None), TypeError),
    ],
    ids=["unknown_field", "layer_none", "error_kind_int", "effect_int", "effect_mixed", "effect_str", "report_id_none"],
)
def test_the_record_schema_is_closed_and_typed(report_dir: tuple[Path, int], record: Any, exc: type[Exception]) -> None:
    """Unknown fields would be unbounded, and a non-string layer would persist unnoticed (Executor, Wire N-1)."""
    d, dfd = report_dir
    with AdvisorLog.create(dfd, SESSION) as log:
        with pytest.raises(exc):
            log.append(record)
        assert not log.broken
    assert read_log(_path(d))["records"] == []


def test_the_log_is_a_context_manager_that_closes(report_dir: tuple[Path, int]) -> None:
    d, dfd = report_dir
    with AdvisorLog.create(dfd, SESSION) as log:
        log.append(_proposal())
    assert log.closed and read_log(_path(d))["closed"] is True


@pytest.mark.parametrize(
    "header",
    [
        {"_record_kind": "header", "schema": "advisor_log/1", "session_id": SESSION, "_format_version": "99.0"},
        {"_record_kind": "header", "schema": "advisor_log/1", "session_id": SESSION},
        {"_record_kind": "proposal", "schema": "advisor_log/1", "_format_version": "1.0"},
    ],
    ids=["future_version", "no_version", "wrong_kind"],
)
def test_read_log_refuses_a_header_it_cannot_read(report_dir: tuple[Path, int], header: dict[str, Any]) -> None:
    d, _dfd = report_dir
    with pytest.raises(AdvisorLogMalformed):
        read_log(_write_log(d, header))


def test_read_log_refuses_deep_nesting_as_malformed(report_dir: tuple[Path, int]) -> None:
    d, _dfd = report_dir
    deep = b'{"_record_kind":"proposal","seq":1,"a":' + b"[" * 100_000 + b"]" * 100_000 + b"}"
    with pytest.raises(AdvisorLogMalformed):
        read_log(_write_log(d, _HEADER, deep))


@pytest.mark.timeout(10)
def test_read_log_refuses_a_fifo_without_blocking(report_dir: tuple[Path, int]) -> None:
    d, _dfd = report_dir
    os.mkfifo(_path(d))
    with pytest.raises(AdvisorLogMalformed, match="not a regular file"):
        read_log(_path(d))


def test_the_log_imports_no_runtime_tools_or_agents() -> None:
    """§5's closure claim, enforced in a fresh child: ``maxim.advisor.log`` loads no ``runtime``/``tools``/``agents``.

    Proved by deletion: an ``import maxim.runtime.executor`` added to ``log.py`` fails this test.
    """
    import subprocess
    import sys

    src = str(Path(advisor_log.__file__).resolve().parents[2])
    code = (
        f"import sys; sys.path.insert(0, {src!r})\n"
        "import maxim.advisor.log\n"
        f"assert maxim.advisor.log.__file__.startswith({src!r}), maxim.advisor.log.__file__\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[:2] in "
        "(['maxim', 'runtime'], ['maxim', 'tools'], ['maxim', 'agents']))\n"
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"maxim.advisor.log pulled in {out.stdout.strip()}"


def test_an_unregistered_names_unknown_effect_is_logged(report_dir: tuple[Path, int]) -> None:
    """``effect: "unknown"`` is what the executor reports for a name no registry holds: the observer's most common
    refusal (§1A.8's example) must be loggable (delta round SF1)."""
    d, dfd = report_dir
    with AdvisorLog.create(dfd, SESSION) as log:
        log.append(_refusal(effect="unknown", error_kind="not_registered", layer="registry"))
    assert read_log(_path(d))["records"][0]["effect"] == "unknown"


def test_an_interrupt_after_the_fsync_before_seq_is_counted_breaks_the_log(report_dir: tuple[Path, int]) -> None:
    """The window the delta round reproduced: the record is fsynced, then an interrupt lands before ``seq`` is
    counted. The log must break, or the next record repeats the seq and ``read_log`` refuses the whole file."""
    import sys

    d, dfd = report_dir
    log = AdvisorLog.create(dfd, SESSION)
    log.append(_proposal())
    fired: list[int] = []

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        if frame.f_code.co_filename != advisor_log.__file__:
            return None
        if event == "line" and not fired:
            import linecache

            if linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip() == "self._seq = seq":
                fired.append(frame.f_lineno)
                raise KeyboardInterrupt
        return tracer

    previous = sys.gettrace()  # coverage's tracer, under CI: restore it, never clear it
    sys.settrace(tracer)
    try:
        with pytest.raises(KeyboardInterrupt):
            log.append(_proposal())
    finally:
        sys.settrace(previous)
    assert fired, "the injection point (the seq count) was never reached"
    assert log.broken, "an interrupt at the seq count left the log usable"
    with pytest.raises(AdvisorLogBroken):
        log.append(_proposal())
    log.close()
    view = read_log(_path(d))
    assert view["last_seq"] == 2 and view["closed"] is False

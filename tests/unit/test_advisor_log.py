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
from maxim.exceptions import AdvisorLogBroken, AdvisorLogMalformed

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
    assert len(json.dumps(rec["arguments"], ensure_ascii=True)) <= advisor_log.MAX_ARGUMENTS_BYTES


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

    monkeypatch.setattr(advisor_log, "_sync", failing_sync)
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

    monkeypatch.setattr(advisor_log, "_sync", failing_sync)
    monkeypatch.setattr(os, "close", recording_close)
    with pytest.raises(OSError):
        AdvisorLog.create(dfd, SESSION)
    assert len(closed) == 1 and closed[0] != dfd


def test_a_zero_byte_write_raises_instead_of_spinning(monkeypatch: Any) -> None:
    monkeypatch.setattr(os, "write", lambda fd, data: 0)
    with pytest.raises(OSError, match="wrote 0 bytes"):
        advisor_log._write_all(99, b"abc")


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

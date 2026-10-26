"""#1315 item 5 strict red gates: the proposal and refusal log ``maxim.advisor.log``, plus advisor_mode.md O7.

WHAT THIS FILE IS. Tests only, written before the code (``docs/plans/atonia_gate_approach_notes.md`` §5 and §1A.8,
§1A winning over §2–§6). Each gate is ``xfail(strict=True, raises=AssertionError)``: it fails TODAY with an
``AssertionError`` (the module does not exist), and once the behaviour is right it XPASSes, which strict turns
into a failure until the decorator is removed. advisor_mode.md's O3 row (no approval surface) is gated in
``tests/integration/test_advisor_gate.py::test_o3_no_approval_surface``, beside the closure check it uses.

THE GATES, AND WHAT FLIPS EACH (all: item 5, ``maxim.advisor.log``)
  * ``create()``: header with ``_format_version`` and ``schema == "advisor_log/1"``, fsynced (file and directory);
    a planted file or symlink refuses (``O_EXCL|O_NOFOLLOW`` via ``dir_fd``); the mode defaults to 0600.
  * ``append()``: one ASCII line per record, no NaN; monotonic ``seq``; ``report_id`` and ``ts`` on each record;
    bounded fields truncated with a flag; an ``OSError`` from ``os.write`` or the fsync marks the log broken and
    later appends raise; a short ``os.write`` is completed by the loop.
  * ``read_log()``: header required; only a torn final line is dropped; ``closed`` and ``last_seq``.
  * ``close()`` writes the close record; the ``F_FULLFSYNC`` branch runs with a monkeypatched ``fcntl`` (Linux too).
  * O7: after ``ingest()`` the proposals and refusals are in the log FILE, not only in memory.

SEAMS THESE GATES PIN (the notes leave them open; chosen here):
  * ``AdvisorLog.create(report_dir_fd, session_id, *, mode=0o600)`` writes ``advisor_log_<session_id>.jsonl``.
  * ``log.append(record) -> int``: ``record`` is a dict with ``_record_kind`` (``"proposal"``/``"refusal"``),
    ``report_id`` and the kind's fields; the LOG stamps ``seq`` and ``ts`` and returns the seq.
  * Truncation flags: ``arguments_truncated`` (proposal ``arguments`` over 4 KiB encoded) and ``name_truncated``
    (refusal ``name`` over 128 characters).
  * A broken log raises ``OSError`` or ``RuntimeError`` (a subclass of either) on every later append;
    ``log.broken`` is ``True``. A NaN raises ``ValueError`` (``allow_nan=False``).
  * ``read_log(path) -> dict`` (plain dicts, §5) with ``header``, ``records`` (proposal/refusal only), ``closed``
    and ``last_seq``; a malformed log raises ``ValueError`` (a subclass of it).
  * ``fcntl`` and ``fcntl.F_FULLFSYNC`` are looked up at CALL time, so a monkeypatch reaches them.
  * O7 uses ``maxim.advisor._session._backend_factory`` and the report/session seams named in
    ``tests/integration/test_advisor_gate.py``'s docstring.

PRECONDITIONS ARE NOT THE GATE: a broken scenario raises ``ScenarioBroken``, never an ``AssertionError``.
"""

from __future__ import annotations

import errno
import importlib
import importlib.util
import json
import os
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SESSION = "sess-0001"
REPORT_ID = "11111111-2222-4333-8444-555555555555"
F_FULLFSYNC_LINUX_STANDIN = 51  # darwin's value; only the attribute's presence matters to the branch


class ScenarioBroken(Exception):
    """The scenario did not happen as designed: NOT the defect a gate pins (module docstring)."""


def require(cond: Any, msg: str) -> None:
    if not cond:
        raise ScenarioBroken(msg)


def _log_mod() -> Any:
    assert importlib.util.find_spec("maxim.advisor") is not None, "item 1 not built: no maxim.advisor"
    assert importlib.util.find_spec("maxim.advisor.log") is not None, "item 5 not built: no maxim.advisor.log"
    mod = importlib.import_module("maxim.advisor.log")
    assert hasattr(mod, "AdvisorLog") and hasattr(mod, "read_log"), "item 5 not built: no AdvisorLog / read_log"
    return mod


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
    return d / f"advisor_log_{SESSION}.jsonl"


def _lines(d: Path) -> list[bytes]:
    return _path(d).read_bytes().split(b"\n")


def _proposal(**over: Any) -> dict[str, Any]:
    rec = {"_record_kind": "proposal", "report_id": REPORT_ID, "arguments": {"summary": "watch", "target": "hall"}}
    rec.update(over)
    return rec


def _refusal(**over: Any) -> dict[str, Any]:
    rec = {
        "_record_kind": "refusal",
        "report_id": REPORT_ID,
        "name": "bash",
        "error_kind": "POLICY_PIN",
        "layer": "pin",
        "effect": ["ALL"],
        "reason": "not the pinned instance",
    }
    rec.update(over)
    return rec


class _SyncRecorder:
    """Records every fsync (``os.fsync`` and ``fcntl(fd, F_FULLFSYNC)``) with the fd's size at that moment."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, fail: bool = False) -> None:
        import fcntl

        self.calls: list[tuple[str, int, int]] = []
        self.fail = fail
        real_fsync, real_fcntl = os.fsync, fcntl.fcntl
        full = getattr(fcntl, "F_FULLFSYNC", None)

        def fsync(fd: int) -> None:
            self._record("fsync", fd)
            if self.fail:
                raise OSError(errno.EIO, "injected fsync failure")
            real_fsync(fd)

        def fcntl_(fd: int, cmd: int, arg: Any = 0) -> Any:
            if full is not None and cmd == full:
                self._record("F_FULLFSYNC", fd)
                if self.fail:
                    raise OSError(errno.EIO, "injected F_FULLFSYNC failure")
            return real_fcntl(fd, cmd, arg)

        monkeypatch.setattr(os, "fsync", fsync)
        monkeypatch.setattr(fcntl, "fcntl", fcntl_)

    def _record(self, how: str, fd: int) -> None:
        try:
            size = os.fstat(fd).st_size
        except OSError:
            size = -1
        self.calls.append((how, fd, size))


# ── create() ──────────────────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1315 item 5: flips with AdvisorLog.create")
def test_create_writes_header_and_fsyncs(report_dir: tuple[Path, int], monkeypatch: pytest.MonkeyPatch) -> None:
    """The header is the first line, carries ``_format_version`` and ``advisor_log/1``, and is fsynced with the dir."""
    log_mod = _log_mod()
    d, dfd = report_dir
    syncs = _SyncRecorder(monkeypatch)
    log = log_mod.AdvisorLog.create(dfd, SESSION)
    first = _lines(d)[0]
    header = json.loads(first)
    assert header.get("_format_version") == "1.0", header
    assert header.get("schema") == "advisor_log/1", header
    assert header.get("_record_kind") == "header" and header.get("session_id") == SESSION, header
    assert "maxim_version" in header, header
    synced_fds = {fd for _how, fd, _size in syncs.calls}
    assert dfd in synced_fds, f"create() never fsynced the directory fd: {syncs.calls}"
    file_syncs = [c for c in syncs.calls if c[1] != dfd]
    assert file_syncs and file_syncs[0][2] >= len(first) + 1, (
        f"create() did not fsync the written header: {syncs.calls}"
    )
    log.close()


@pytest.mark.parametrize("planted", ["file", "symlink", "dangling_symlink"])
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1315 item 5: flips with O_EXCL|O_NOFOLLOW via dir_fd")
def test_create_refuses_planted(report_dir: tuple[Path, int], tmp_path: Path, planted: str) -> None:
    """A planted file or symlink at the log's name refuses; nothing is written through it."""
    log_mod = _log_mod()
    d, dfd = report_dir
    target = tmp_path / "victim.txt"
    if planted == "file":
        _path(d).write_text("planted")
    else:
        if planted == "symlink":
            target.write_text("victim")
        _path(d).symlink_to(target)
    with pytest.raises(OSError):
        log_mod.AdvisorLog.create(dfd, SESSION)
    if planted == "file":
        assert _path(d).read_text() == "planted", "create() wrote into a planted file"
    elif planted == "symlink":
        assert target.read_text() == "victim", "create() wrote through a planted symlink"
    else:
        assert not target.exists(), "create() created the target of a dangling symlink"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1315 item 5: flips with create(mode=0o600)")
def test_create_mode_defaults_0600(report_dir: tuple[Path, int]) -> None:
    """The default mode is 0600 (under a permissive umask); ``mode=0o640`` is passed through."""
    log_mod = _log_mod()
    d, dfd = report_dir
    old = os.umask(0o002)
    try:
        log = log_mod.AdvisorLog.create(dfd, SESSION)
        log.close()
        assert (os.stat(_path(d)).st_mode & 0o777) == 0o600, oct(os.stat(_path(d)).st_mode & 0o777)
        log2 = log_mod.AdvisorLog.create(dfd, "sess-0002", mode=0o640)
        log2.close()
        assert (os.stat(d / "advisor_log_sess-0002.jsonl").st_mode & 0o777) == 0o640
    finally:
        os.umask(old)


# ── append() ──────────────────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1315 item 5: flips with append()'s encoding")
def test_append_one_ascii_line_per_record_no_nan(report_dir: tuple[Path, int]) -> None:
    """``ensure_ascii``, escaped newlines, one line per record; NaN raises and is never written."""
    log_mod = _log_mod()
    d, dfd = report_dir
    log = log_mod.AdvisorLog.create(dfd, SESSION)
    log.append(_proposal(arguments={"summary": "café\nline two ☃ 😀", "target": "\u0000"}))
    log.append(_refusal())
    with pytest.raises(ValueError):
        log.append(_proposal(arguments={"summary": "x", "score": float("nan")}))
    raw = _path(d).read_bytes()
    assert raw.isascii(), "append() wrote non-ASCII bytes"
    assert b"NaN" not in raw and b"Infinity" not in raw, "append() wrote a NaN/Infinity"
    lines = [ln for ln in raw.split(b"\n") if ln]
    assert len(lines) == 3, f"expected header + 2 records, got {len(lines)} lines"
    assert json.loads(lines[1])["arguments"]["summary"] == "café\nline two ☃ 😀"
    log.close()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1315 item 5: flips with seq / report_id / ts")
def test_append_seq_monotonic_with_report_id_and_ts(report_dir: tuple[Path, int]) -> None:
    """``seq`` rises by record, and every record carries ``report_id`` and ``ts``."""
    log_mod = _log_mod()
    d, dfd = report_dir
    log = log_mod.AdvisorLog.create(dfd, SESSION)
    returned = [log.append(_proposal()), log.append(_refusal()), log.append(_proposal())]
    recs = [json.loads(ln) for ln in _lines(d)[1:] if ln]
    seqs = [r.get("seq") for r in recs]
    assert all(isinstance(s, int) for s in seqs) and seqs == sorted(set(seqs)), f"seq not monotonic: {seqs}"
    assert returned == seqs, f"append() returned {returned}, the file says {seqs}"
    for r in recs:
        assert r.get("report_id") == REPORT_ID, r
        assert isinstance(r.get("ts"), (str, int, float)) and r.get("ts"), r
    log.close()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1315 item 5: flips with the field bounds")
def test_append_bounds_fields_with_flag(report_dir: tuple[Path, int]) -> None:
    """Proposal ``arguments`` are bounded to 4 KiB and refusal ``name`` to 128 characters, each with a flag."""
    log_mod = _log_mod()
    d, dfd = report_dir
    log = log_mod.AdvisorLog.create(dfd, SESSION)
    log.append(_proposal(arguments={"summary": "x" * 20_000}))
    log.append(_refusal(name="n" * 1_000))
    log.append(_proposal())
    prop, refu, small = (json.loads(ln) for ln in _lines(d)[1:] if ln)
    enc = json.dumps(prop["arguments"], ensure_ascii=True, separators=(",", ":")).encode()
    assert len(enc) <= 4096, f"proposal arguments are {len(enc)} bytes encoded"
    assert prop.get("arguments_truncated") is True, prop.keys()
    assert len(refu["name"]) <= 128 and refu.get("name_truncated") is True, (len(refu["name"]), refu.keys())
    assert not small.get("arguments_truncated"), "a small record was flagged as truncated"
    log.close()


def _inject_write_failure(monkeypatch: pytest.MonkeyPatch, protected: set[int]) -> None:
    real = os.write

    def write(fd: int, data: Any) -> int:
        if fd in protected:
            return real(fd, data)
        raise OSError(errno.ENOSPC, "injected: no space left on device")

    monkeypatch.setattr(os, "write", write)


@pytest.mark.parametrize("where", ["write", "fsync"])
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1315 item 5: flips with the broken-log rule")
def test_append_oserror_marks_broken(report_dir: tuple[Path, int], monkeypatch: pytest.MonkeyPatch, where: str) -> None:
    """An ``OSError`` from ``os.write`` or the fsync raises, marks the log broken, and every later append raises."""
    log_mod = _log_mod()
    d, dfd = report_dir
    log = log_mod.AdvisorLog.create(dfd, SESSION)
    log.append(_proposal())
    size_before = _path(d).stat().st_size
    with monkeypatch.context() as m:
        if where == "write":
            _inject_write_failure(m, {1, 2})
        else:
            _SyncRecorder(m, fail=True)
        with pytest.raises(OSError):
            log.append(_refusal())
    assert getattr(log, "broken", None) is True, "the log is not marked broken after an OSError"
    size_broken = _path(d).stat().st_size
    with pytest.raises((OSError, RuntimeError)):
        log.append(_proposal())
    assert _path(d).stat().st_size == size_broken, "a broken log wrote again"
    if where == "write":
        assert size_broken == size_before, "a failed os.write still left bytes"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1315 item 5: flips with the write loop")
def test_append_completes_short_write(report_dir: tuple[Path, int], monkeypatch: pytest.MonkeyPatch) -> None:
    """A short ``os.write`` is completed by the loop: the line lands whole."""
    log_mod = _log_mod()
    d, dfd = report_dir
    log = log_mod.AdvisorLog.create(dfd, SESSION)
    real = os.write
    shorted: list[int] = []

    def write(fd: int, data: Any) -> int:
        if fd not in (1, 2) and not shorted and len(data) > 1:
            shorted.append(len(data))
            return real(fd, bytes(data)[: len(data) // 2])
        return real(fd, data)

    monkeypatch.setattr(os, "write", write)
    seq = log.append(_proposal(arguments={"summary": "y" * 300}))
    monkeypatch.undo()
    require(shorted, "the short write was never exercised")
    recs = [json.loads(ln) for ln in _lines(d)[1:] if ln]
    assert len(recs) == 1 and recs[0]["seq"] == seq, f"the short-written line did not land whole: {recs}"
    assert recs[0]["arguments"]["summary"] == "y" * 300
    log.close()


# ── read_log() and close() ────────────────────────────────────────────────


@pytest.mark.parametrize("content", ["empty", "headerless", "two_headers", "bad_line_mid"])
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1315 item 5: flips with read_log's strictness")
def test_read_log_rejects_malformed(report_dir: tuple[Path, int], content: str) -> None:
    """An empty or headerless file, a second header, or a bad line before the last raises ``ValueError``."""
    log_mod = _log_mod()
    d, dfd = report_dir
    if content in ("two_headers", "bad_line_mid"):
        log = log_mod.AdvisorLog.create(dfd, SESSION)
        log.append(_proposal())
        log.close()
        lines = _lines(d)
        if content == "two_headers":
            lines.insert(1, lines[0])
        else:
            lines.insert(2, b"{not json")
        _path(d).write_bytes(b"\n".join(lines))
    else:
        rec = json.dumps({"_record_kind": "proposal", "seq": 1, "report_id": REPORT_ID, "ts": "t"}) + "\n"
        _path(d).write_text("" if content == "empty" else rec)
    with pytest.raises(ValueError):
        log_mod.read_log(_path(d))


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1315 item 5: flips with read_log / close")
def test_read_log_torn_tail_closed_and_last_seq(report_dir: tuple[Path, int]) -> None:
    """A torn final line is dropped; an open log reads ``closed=False``; ``close()`` writes the close record."""
    log_mod = _log_mod()
    d, dfd = report_dir
    log = log_mod.AdvisorLog.create(dfd, SESSION)
    s1 = log.append(_proposal())
    s2 = log.append(_refusal())
    open_view = log_mod.read_log(_path(d))
    assert isinstance(open_view, dict), f"read_log returns {type(open_view)!r}, not a plain dict (§5)"
    assert open_view["closed"] is False and open_view["last_seq"] == s2, open_view
    assert [r["seq"] for r in open_view["records"]] == [s1, s2]
    with open(_path(d), "ab") as fh:  # a crash mid-append: the torn line can only be the final one
        fh.write(b'{"_record_kind":"proposal","seq":')
    torn_view = log_mod.read_log(_path(d))
    assert torn_view["last_seq"] == s2 and len(torn_view["records"]) == 2, torn_view
    _path(d).write_bytes(_path(d).read_bytes().rsplit(b"\n", 1)[0] + b"\n")  # trim the torn tail again
    log.close()
    closed_view = log_mod.read_log(_path(d))
    last = json.loads([ln for ln in _lines(d) if ln][-1])
    assert last.get("_record_kind") == "close" and last.get("last_seq") == s2, last
    assert closed_view["closed"] is True and closed_view["last_seq"] == s2, closed_view


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1315 item 5: flips with the F_FULLFSYNC branch")
def test_full_fsync_branch(report_dir: tuple[Path, int], monkeypatch: pytest.MonkeyPatch) -> None:
    """With ``fcntl.F_FULLFSYNC`` present (real on macOS, monkeypatched on Linux) create and append use it."""
    import fcntl

    log_mod = _log_mod()
    _d, dfd = report_dir
    full = getattr(fcntl, "F_FULLFSYNC", None)
    if full is None:  # Linux: stand the constant in, and answer it without calling the kernel
        full = F_FULLFSYNC_LINUX_STANDIN
        monkeypatch.setattr(fcntl, "F_FULLFSYNC", full, raising=False)
    real_fcntl = fcntl.fcntl
    seen: list[int] = []

    def fcntl_(fd: int, cmd: int, arg: Any = 0) -> Any:
        if cmd == full:
            seen.append(fd)
            return 0 if sys.platform != "darwin" else real_fcntl(fd, cmd, arg)
        return real_fcntl(fd, cmd, arg)

    monkeypatch.setattr(fcntl, "fcntl", fcntl_)
    log = log_mod.AdvisorLog.create(dfd, SESSION)
    after_create = len(seen)
    log.append(_proposal())
    assert after_create >= 1, "create() did not take the F_FULLFSYNC branch"
    assert len(seen) > after_create, "append() did not take the F_FULLFSYNC branch"
    log.close()


# ── O7: durable proposals and refusals ────────────────────────────────────

_COMPLETION = {
    "happening": "someone passed the door",
    "would_do": "keep watching the hall",
    "rationale": "one sighting",
    "confidence": 0.5,
    "calls": [
        {"name": "propose_action", "arguments": {"summary": "keep watching", "target": "hall"}},
        {"name": "bash", "arguments": {"command": "ls /"}},
    ],
}


class _FakeBackend:
    def build_prompt(self, template: str, segments: Any) -> Any:
        n = len(list(segments))
        return SimpleNamespace(ids=list(range(n + 2)), untrusted_spans=[(1 + i, 2 + i) for i in range(n)], rejected=())

    def complete_tokens(self, prompt_ids: Any, **_kw: Any) -> Any:
        return SimpleNamespace(text=json.dumps(_COMPLETION), deadline_hit=False, finish_reason="stop")


@pytest.fixture
def _no_latch_leak() -> Iterator[None]:
    """open() engages item 10's process-wide latch (§4.2); an in-process test must leave it reset (§1A.1)."""
    yield
    off = sys.modules.get("maxim._offline")
    if off is not None:
        reset = getattr(off, "_reset_for_tests", None)
        if callable(reset):
            reset()
        elif getattr(off, "is_offline", lambda: False)():
            pytest.fail("the offline latch is up after an in-process open() and no _reset_for_tests exists (§1A.1)")


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1315 item 5 (advisor_mode.md O7): durable log")
def test_o7_proposals_and_refusals_are_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _no_latch_leak: None
) -> None:
    """After ``ingest()`` and BEFORE ``close()``, the proposal and the refusal are records in the log file."""
    log_mod = _log_mod()
    assert importlib.util.find_spec("maxim.advisor._session") is not None, "item 1 not built: no _session"
    sess = importlib.import_module("maxim.advisor._session")
    assert hasattr(sess, "_backend_factory"), "item 1 not built: no _session._backend_factory seam"
    monkeypatch.setattr(sess, "_backend_factory", lambda *a, **k: _FakeBackend())
    advisor = importlib.import_module("maxim.advisor")
    home_dir, rdir = tmp_path / "home", tmp_path / "reports"
    home_dir.mkdir()
    rdir.mkdir()
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF-fake")
    session = advisor.open(home_dir=home_dir, model=model, report_dir=rdir)
    try:
        report = session.ingest([{"event_id": "e1", "kind": "note", "text": "a person walked by", "payload": {}}])
        rep = dict(report) if isinstance(report, Mapping) else report.to_dict()
        path = rdir / f"advisor_log_{rep['session_id']}.jsonl"
        assert path.is_file(), f"no log file on disk after ingest(): {sorted(p.name for p in rdir.iterdir())}"
        view = log_mod.read_log(path)
        kinds = [r.get("_record_kind") for r in view["records"]]
        assert "proposal" in kinds, f"the proposal is not on disk: {kinds}"
        refusals = [r for r in view["records"] if r.get("_record_kind") == "refusal"]
        assert [r.get("name") for r in refusals] == ["bash"], f"the refusal is not on disk: {refusals}"
        assert all(r.get("report_id") == rep["report_id"] for r in view["records"]), "records do not join the report"
        assert view["last_seq"] == rep["log_last_seq"], (view["last_seq"], rep["log_last_seq"])
        assert view["closed"] is False, "the log claims closed before close()"
    finally:
        session.close()

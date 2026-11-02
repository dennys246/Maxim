"""The observer's proposal and refusal log (#1315 item 5; ``atonia_gate_approach_notes.md`` §5 and §1A.8).

One JSON-lines file per session, ``advisor_log_<session_id>.jsonl``, opened once through the report directory's
fd and never reopened by name. Every line is fsynced before :meth:`AdvisorLog.append` returns, so a crash can tear
only the final line, and :func:`read_log` drops only that one.

It borrows ``simulation/report.py::save_action_log``'s shape (a header, then records), not its durability: that
writer rewrites the whole file at session end with no fsync. The ``atomic_io`` helpers are not used either, because
they swallow fsync errors, and here a failed fsync must fail the turn.

Nothing reads the log back into a dispatch. The session holds a write-only fd, and :func:`read_log` returns plain
dicts. This module imports nothing from ``runtime/`` or ``tools/``.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from maxim.exceptions import AdvisorLogBroken, AdvisorLogMalformed
from maxim.utils.format_version import check_format_version

logger = logging.getLogger(__name__)

SCHEMA = "advisor_log/1"
FORMAT_VERSION = "1.0"

#: Proposal ``arguments`` are bounded to this many bytes, encoded (§5).
MAX_ARGUMENTS_BYTES = 4096
#: A refusal's tool ``name`` is bounded to this many characters (§5).
MAX_NAME_CHARS = 128
#: A refusal's ``reason`` is bounded to this many characters (§5, "the reason bounded").
MAX_REASON_CHARS = 1024

_RECORD_KINDS = frozenset({"proposal", "refusal"})
_LOG_STAMPED = frozenset({"seq", "ts"})
_REQUIRED = {
    "proposal": ("report_id", "arguments"),
    "refusal": ("report_id", "name", "error_kind", "layer", "effect", "reason"),
}


def log_name(session_id: str) -> str:
    """The log's file name inside the report directory."""
    return f"advisor_log_{session_id}.jsonl"


def _encode(record: dict[str, Any]) -> bytes:
    """One line: ASCII only, no NaN or Infinity (``ValueError``), compact separators, newline-terminated."""
    return (json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n").encode("ascii")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def _sync(fd: int) -> None:
    """Flush ``fd`` to stable storage. ``F_FULLFSYNC`` where the platform has it (macOS), else ``os.fsync``.

    ``fcntl`` and its constant are looked up on each call, so a test can stand the macOS branch in on Linux. Any
    ``OSError`` propagates: the caller marks the log broken.
    """
    import fcntl

    full = getattr(fcntl, "F_FULLFSYNC", None)
    if full is not None:
        fcntl.fcntl(fd, full)
    else:
        os.fsync(fd)


def _write_all(fd: int, data: bytes) -> None:
    """Loop ``os.write`` until every byte is written; a short write is completed, never left torn."""
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError(f"os.write wrote {written} bytes of {len(view)}")
        view = view[written:]


def _bound_arguments(arguments: Any) -> tuple[Any, bool]:
    """Bound proposal ``arguments`` to ``MAX_ARGUMENTS_BYTES`` encoded.

    Over the bound, the arguments become a prefix of their own JSON encoding, as a string whose encoding still fits,
    and the flag is set. The prefix is text, never re-parsed.
    """
    encoded = json.dumps(arguments, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    if len(encoded) <= MAX_ARGUMENTS_BYTES:
        return arguments, False
    cut = MAX_ARGUMENTS_BYTES
    while True:
        prefix = encoded[:cut]
        over = len(json.dumps(prefix, ensure_ascii=True)) - MAX_ARGUMENTS_BYTES
        if over <= 0:
            return prefix, True
        cut -= over


def _bound_text(value: Any, limit: int) -> tuple[str, bool]:
    text = str(value)
    return (text[:limit], True) if len(text) > limit else (text, False)


def _shape(record: dict[str, Any]) -> dict[str, Any]:
    """Validate a caller's record and apply the field bounds; the log adds ``seq`` and ``ts`` afterwards."""
    if not isinstance(record, dict):
        raise TypeError(f"an advisor log record is a dict, not {type(record).__name__}")
    kind = record.get("_record_kind")
    if kind not in _RECORD_KINDS:
        raise ValueError(f"an advisor log record's _record_kind is one of {sorted(_RECORD_KINDS)}, not {kind!r}")
    stamped = _LOG_STAMPED & record.keys()
    if stamped:
        raise ValueError(f"the log stamps {sorted(stamped)} itself; a record must not carry them")
    missing = [k for k in _REQUIRED[kind] if k not in record]
    if missing:
        raise ValueError(f"an advisor log {kind} record lacks {missing}")
    shaped = dict(record)
    if kind == "proposal":
        shaped["arguments"], truncated = _bound_arguments(record["arguments"])
        if truncated:
            shaped["arguments_truncated"] = True
    else:
        shaped["name"], name_cut = _bound_text(record["name"], MAX_NAME_CHARS)
        if name_cut:
            shaped["name_truncated"] = True
        shaped["reason"], reason_cut = _bound_text(record["reason"], MAX_REASON_CHARS)
        if reason_cut:
            shaped["reason_truncated"] = True
    return shaped


class AdvisorLog:
    """An append-only, fsynced JSON-lines log, written through one fd held for the whole session.

    Build it with :meth:`create`. After any ``OSError`` the log is ``broken``: the error propagates, and every
    later :meth:`append` raises :class:`~maxim.exceptions.AdvisorLogBroken` without touching the fd.
    """

    def __init__(self, fd: int, session_id: str) -> None:
        self._fd = fd
        self.session_id = session_id
        self._seq = 0
        self._lock = threading.Lock()
        self.broken = False
        self.closed = False

    @classmethod
    def create(cls, report_dir_fd: int, session_id: str, *, mode: int = 0o600) -> AdvisorLog:
        """Create ``advisor_log_<session_id>.jsonl`` under ``report_dir_fd``, write the header, and fsync both.

        ``O_EXCL|O_NOFOLLOW`` through ``dir_fd``: a planted file or symlink at the name raises ``OSError`` and
        nothing is written through it. ``mode`` is applied exactly (``fchmod``), whatever the umask.
        """
        from maxim import __version__

        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        fd = os.open(log_name(session_id), flags, mode, dir_fd=report_dir_fd)
        header = {
            "_format_version": FORMAT_VERSION,
            "_record_kind": "header",
            "schema": SCHEMA,
            "session_id": session_id,
            "maxim_version": __version__,
        }
        try:
            os.fchmod(fd, mode)
            _write_all(fd, _encode(header))
            _sync(fd)
            _sync(report_dir_fd)
        except BaseException:
            os.close(fd)
            raise
        return cls(fd, session_id)

    @property
    def last_seq(self) -> int:
        """The ``seq`` of the last record written, ``0`` before any."""
        return self._seq

    def append(self, record: dict[str, Any]) -> int:
        """Write one proposal or refusal record, fsync it, and return its ``seq``.

        ``record`` carries ``_record_kind``, ``report_id`` and the kind's fields; the log stamps ``seq`` and ``ts``.
        A record the log refuses to encode (a NaN, a missing field) raises ``ValueError`` or ``TypeError`` before
        anything is written, and the log stays usable.
        """
        with self._lock:
            self._refuse_if_unusable()
            seq = self._seq + 1
            line = _encode({**_shape(record), "seq": seq, "ts": _now()})
            self._write_synced(line)
            self._seq = seq
            return seq

    def close(self) -> None:
        """Write the close record (``last_seq``) and close the fd. A broken log closes its fd and writes nothing."""
        with self._lock:
            if self.closed:
                return
            try:
                if not self.broken:
                    self._write_synced(_encode({"_record_kind": "close", "last_seq": self._seq, "ts": _now()}))
            finally:
                self.closed = True
                os.close(self._fd)

    def _refuse_if_unusable(self) -> None:
        if self.broken:
            raise AdvisorLogBroken(f"advisor log {self.session_id!r} is broken; nothing more is written to it")
        if self.closed:
            raise AdvisorLogBroken(f"advisor log {self.session_id!r} is closed")

    def _write_synced(self, line: bytes) -> None:
        try:
            _write_all(self._fd, line)
            _sync(self._fd)
        except OSError:
            self.broken = True
            raise


def _parse_line(raw: bytes, number: int) -> dict[str, Any]:
    try:
        obj = json.loads(raw)
    except ValueError as e:
        raise AdvisorLogMalformed(f"line {number} is not JSON: {e}") from e
    if not isinstance(obj, dict):
        raise AdvisorLogMalformed(f"line {number} is not a JSON object")
    return obj


def read_log(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a log back as plain dicts: ``{"header", "records", "closed", "last_seq"}``.

    The header must be the first line, exactly once. A torn final line (no trailing newline: a crash mid-append) is
    dropped; a bad line anywhere else, a record after the close record, or a ``seq`` that does not rise raises
    :class:`~maxim.exceptions.AdvisorLogMalformed`. ``records`` holds the proposal and refusal records only. A
    consumer finds a lost tail by comparing ``last_seq`` with the reports' ``log_last_seq``.
    """
    lines = Path(path).read_bytes().split(b"\n")
    lines.pop()  # the text after the last newline: empty, or a torn final line
    if not lines:
        raise AdvisorLogMalformed(f"{path}: no header (an empty log)")
    header = _parse_line(lines[0], 1)
    if header.get("_record_kind") != "header" or header.get("schema") != SCHEMA:
        raise AdvisorLogMalformed(f"{path}: the first line is not an {SCHEMA} header")
    check_format_version(header, "advisor_log", log=logger)
    records: list[dict[str, Any]] = []
    closed = False
    last_seq = 0
    for number, raw in enumerate(lines[1:], start=2):
        obj = _parse_line(raw, number)
        kind = obj.get("_record_kind")
        if closed:
            raise AdvisorLogMalformed(f"{path}: line {number} follows the close record")
        if kind == "close":
            if obj.get("last_seq") != last_seq:
                raise AdvisorLogMalformed(
                    f"{path}: the close record says last_seq {obj.get('last_seq')!r}, the records {last_seq}"
                )
            closed = True
            continue
        if kind not in _RECORD_KINDS:
            raise AdvisorLogMalformed(f"{path}: line {number} has _record_kind {kind!r}")
        seq = obj.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq <= last_seq:
            raise AdvisorLogMalformed(f"{path}: line {number} has seq {seq!r} after {last_seq}")
        last_seq = seq
        records.append(obj)
    return {"header": header, "records": records, "closed": closed, "last_seq": last_seq}

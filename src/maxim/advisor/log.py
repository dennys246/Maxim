"""The observer's proposal and refusal log (#1315 item 5; ``atonia_gate_approach_notes.md`` §5 and §1A.8).

One JSON-lines file per session, ``advisor_log_<session_id>.jsonl``, opened once through the report directory's
fd and never reopened by name. Every line is fsynced before :meth:`AdvisorLog.append` returns, so a crash can tear
only the final line, and :func:`read_log` drops only that one.

It borrows ``simulation/report.py::save_action_log``'s shape (a header, then records), not its durability: that
writer rewrites the whole file at session end with no fsync. It writes through ``atomic_io``'s STRICT helpers
(:func:`~maxim.utils.atomic_io.append_line_fsync`), never the lenient ones, which skip an fsync the filesystem
refuses: here a record that was not durably written must fail the turn.

Nothing reads the log back into a dispatch. The session holds a write-only fd, and :func:`read_log` returns plain
dicts. This module imports nothing from ``runtime/``, ``tools/`` or ``agents/``
(``tests/unit/test_advisor_log.py::test_the_log_imports_no_runtime_tools_or_agents``).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import stat
import threading
from types import TracebackType
from typing import Any

from maxim import __version__
from maxim.exceptions import AdvisorLogBroken, AdvisorLogMalformed
from maxim.utils.atomic_io import append_line_fsync, fsync_strict
from maxim.utils.format_version import FORMAT_VERSION, check_format_version, with_format_version

logger = logging.getLogger(__name__)

SCHEMA = "advisor_log/1"

#: Proposal ``arguments`` are bounded to this many bytes, encoded (§5).
MAX_ARGUMENTS_BYTES = 4096
#: A refusal's tool ``name`` is bounded to this many characters (§5).
MAX_NAME_CHARS = 128
#: A refusal's ``reason`` is bounded to this many characters (§5, "the reason bounded").
MAX_REASON_CHARS = 1024
#: The mode bits a log may carry: 0600 by default, 0640 for an operator in the owner's group (§5).
ALLOWED_MODE_BITS = 0o640

# A session id is one path component: it names a file under the report directory's fd and must never climb out.
_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

# The closed record schema (§5): what the caller supplies per kind. The log stamps seq and ts, and adds the
# truncation flags itself.
_FIELDS = {
    "proposal": frozenset({"_record_kind", "report_id", "arguments"}),
    "refusal": frozenset({"_record_kind", "report_id", "name", "error_kind", "layer", "effect", "reason"}),
}
_TEXT_FIELDS = {"proposal": ("report_id",), "refusal": ("report_id", "name", "error_kind", "layer", "reason")}


def log_name(session_id: str) -> str:
    """The log's file name inside the report directory. ``ValueError`` unless ``session_id`` is one safe component."""
    if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id):
        raise ValueError(f"an advisor session id is 1-128 of [A-Za-z0-9._-], not starting with '.': {session_id!r}")
    return f"advisor_log_{session_id}.jsonl"


def _encode(record: dict[str, Any]) -> bytes:
    """One line: ASCII only, no NaN or Infinity (``ValueError``), compact separators, newline-terminated."""
    return (json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n").encode("ascii")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def _encoded_len(text: str) -> int:
    return len(json.dumps(text, ensure_ascii=True))


def _bound_arguments(arguments: Any) -> tuple[Any, bool]:
    """Bound proposal ``arguments`` to ``MAX_ARGUMENTS_BYTES`` encoded.

    Over the bound, the arguments become the longest prefix of their own JSON encoding whose encoding, as a string,
    still fits, and the flag is set. Consumers branch on ``arguments_truncated``; the prefix is text, never re-parsed.
    """
    encoded = json.dumps(arguments, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    if len(encoded) <= MAX_ARGUMENTS_BYTES:
        return arguments, False
    lo, hi = 0, MAX_ARGUMENTS_BYTES  # encoding never shrinks a prefix, so fit(cut) is monotone in cut
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _encoded_len(encoded[:mid]) <= MAX_ARGUMENTS_BYTES:
            lo = mid
        else:
            hi = mid - 1
    return encoded[:lo], True


def _bound_text(value: str, limit: int) -> tuple[str, bool]:
    return (value[:limit], True) if len(value) > limit else (value, False)


def _shape(record: dict[str, Any]) -> dict[str, Any]:
    """Validate a caller's record against the closed schema and apply the field bounds.

    Refused before anything is written: an unknown kind, a missing or unknown field (``seq`` and ``ts`` included:
    the log stamps those), a non-string text field, or an ``effect`` that is not a list of flag names.
    """
    if not isinstance(record, dict):
        raise TypeError(f"an advisor log record is a dict, not {type(record).__name__}")
    kind = record.get("_record_kind")
    if kind not in _FIELDS:
        raise ValueError(f"an advisor log record's _record_kind is one of {sorted(_FIELDS)}, not {kind!r}")
    keys = frozenset(record)
    if keys != _FIELDS[kind]:
        missing, unknown = sorted(_FIELDS[kind] - keys), sorted(keys - _FIELDS[kind])
        raise ValueError(f"an advisor log {kind} record lacks {missing} and has unknown fields {unknown}")
    for field in _TEXT_FIELDS[kind]:
        if not isinstance(record[field], str):
            raise TypeError(f"an advisor log {kind} record's {field} is a str, not {type(record[field]).__name__}")
    shaped = dict(record)
    if kind == "proposal":
        shaped["arguments"], truncated = _bound_arguments(record["arguments"])
        if truncated:
            shaped["arguments_truncated"] = True
        return shaped
    effect = record["effect"]
    if not isinstance(effect, list) or not all(isinstance(flag, str) for flag in effect):
        raise TypeError(f"a refusal's effect is a list of flag names (sorted), not {effect!r}")
    shaped["name"], name_cut = _bound_text(record["name"], MAX_NAME_CHARS)
    if name_cut:
        shaped["name_truncated"] = True
    shaped["reason"], reason_cut = _bound_text(record["reason"], MAX_REASON_CHARS)
    if reason_cut:
        shaped["reason_truncated"] = True
    return shaped


class AdvisorLog:
    """An append-only, fsynced JSON-lines log, written through one fd held for the whole session.

    Build it with :meth:`create`; it is a context manager that :meth:`close`\\ s on exit. If a write or an fsync
    fails, or is interrupted (``KeyboardInterrupt`` included), the log is ``broken``: the error propagates, and
    every later :meth:`append` raises :class:`~maxim.exceptions.AdvisorLogBroken` without touching the fd. The
    record that broke it may still be on disk, one ``seq`` past :attr:`last_seq`; :func:`read_log` reads it as the
    final record.
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

        ``report_dir_fd`` is the report directory, opened read-only (``O_RDONLY|O_DIRECTORY|O_NOFOLLOW``).
        ``session_id`` must be one safe path component and ``mode`` within 0640 (``ValueError``). ``O_EXCL`` through
        ``dir_fd``: a planted file or symlink at the name raises ``OSError`` and nothing is written through it.
        ``mode`` is applied exactly (``fchmod``), whatever the umask. On any failure the fd is closed; a header
        that was partly written stays, and :func:`read_log` refuses it.
        """
        name = log_name(session_id)
        if mode & ~ALLOWED_MODE_BITS:
            raise ValueError(f"an advisor log's mode is within {ALLOWED_MODE_BITS:#o}, not {mode:#o}")
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        fd = os.open(name, flags, mode, dir_fd=report_dir_fd)
        header = with_format_version(
            {"_record_kind": "header", "schema": SCHEMA, "session_id": session_id, "maxim_version": __version__}
        )
        try:
            os.fchmod(fd, mode)
            append_line_fsync(fd, _encode(header))
            fsync_strict(report_dir_fd)
        except BaseException:
            os.close(fd)
            raise
        return cls(fd, session_id)

    def __enter__(self) -> AdvisorLog:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    @property
    def last_seq(self) -> int:
        """The ``seq`` of the last record durably written, ``0`` before any. Reports carry it as ``log_last_seq``."""
        return self._seq

    def append(self, record: dict[str, Any]) -> int:
        """Write one proposal or refusal record, fsync it, and return its ``seq``.

        ``record`` carries ``_record_kind``, ``report_id`` and the kind's fields (the closed schema in
        :func:`_shape`); the log stamps ``seq`` and ``ts``. A record the log refuses to encode (a NaN, a missing
        or unknown field) raises ``ValueError`` or ``TypeError`` before anything is written, and the log stays
        usable.
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
            append_line_fsync(self._fd, line)
        except BaseException:
            # Not only OSError: an interrupt between the write and the fsync leaves the line on disk with seq not
            # advanced, and a later record would repeat that seq (#1315 item 5 review, Executor B1).
            self.broken = True
            raise


def _parse_line(raw: bytes, number: int) -> dict[str, Any]:
    try:
        obj = json.loads(raw)
    except (ValueError, RecursionError) as e:
        raise AdvisorLogMalformed(f"line {number} is not JSON: {type(e).__name__}") from e
    if not isinstance(obj, dict):
        raise AdvisorLogMalformed(f"line {number} is not a JSON object")
    return obj


def _read_regular_file(path: str | os.PathLike[str]) -> bytes:
    """Read ``path`` whole, refusing anything but a regular file (a FIFO would block, a device never end)."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise AdvisorLogMalformed(f"{path}: not a regular file")
        chunks = []
        while chunk := os.read(fd, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def read_log(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a log back as plain dicts: ``{"header", "records", "closed", "last_seq"}``.

    The header must be the first line, exactly once, with ``schema == "advisor_log/1"`` and ``_format_version ==
    "1.0"``. A torn final line (no trailing newline: a crash mid-append) is dropped; a bad line anywhere else, a
    record after the close record, a close record that disagrees with the records, or a ``seq`` that does not rise
    raises :class:`~maxim.exceptions.AdvisorLogMalformed`. ``records`` holds the proposal and refusal records only.
    A consumer finds a lost tail by comparing ``last_seq`` with the reports' ``log_last_seq``; after a failed
    write, the log can hold one record past the last report's ``log_last_seq`` (:class:`AdvisorLog`).

    The file is read whole: a log is one session's records, bounded per record.
    """
    lines = _read_regular_file(path).split(b"\n")
    lines.pop()  # the text after the last newline: empty, or a torn final line
    if not lines:
        raise AdvisorLogMalformed(f"{path}: no header (an empty log)")
    header = _parse_line(lines[0], 1)
    if header.get("_record_kind") != "header" or header.get("schema") != SCHEMA:
        raise AdvisorLogMalformed(f"{path}: the first line is not an {SCHEMA} header")
    version = check_format_version(header, "advisor_log", log=logger)
    if version != FORMAT_VERSION:
        raise AdvisorLogMalformed(f"{path}: _format_version {version!r}, this build reads {FORMAT_VERSION!r}")
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
        if kind not in _FIELDS:
            raise AdvisorLogMalformed(f"{path}: line {number} has _record_kind {kind!r}")
        seq = obj.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq <= last_seq:
            raise AdvisorLogMalformed(f"{path}: line {number} has seq {seq!r} after {last_seq}")
        last_seq = seq
        records.append(obj)
    return {"header": header, "records": records, "closed": closed, "last_seq": last_seq}

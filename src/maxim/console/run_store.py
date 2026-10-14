"""Read-only access to run artefacts for the Console (contract 0.6.0).

This module is a SECURITY BOUNDARY: it turns request data (a run id, a cursor,
filter values) into file reads. Its rules (maxim-pulse
docs/plans/log_explorer.md §4.2, §4.4):

* ``utils/paths.py::resolve_run_dir`` is NEVER called with request data. It
  returns any directory for a path-like argument, searches the CWD first, and
  follows symlinks out of ``sim_reports``. :func:`open_console_run` is the
  only way in, and it returns an fd, never a path.
* A run id is a plain name (``[A-Za-z0-9_-]{1,64}``), checked BEFORE any
  filesystem call. A run directory must be a real directory (not a symlink)
  whose resolved parent IS the resolved ``sim_reports`` root.
* Every open is ANCHORED to a directory file descriptor: the root is opened,
  the run directory is opened relative to it with ``O_DIRECTORY|O_NOFOLLOW``,
  and members relative to that. ``O_NOFOLLOW`` alone guards only the last path
  component, so a path-based open could follow a run directory swapped for a
  symlink after it was checked (review: 25 leaks in 8 s under a swap race).
* Members are opened by fixed name only (:data:`MEMBERS`), non-blocking (a
  FIFO swapped in cannot hang a worker), and must be regular files with a
  single link (a hardlink to a secret is refused like a symlink).
* Hostile CONTENTS never raise past the parser: deep nesting, non-objects,
  non-UTF-8 and wrong-typed fields are skipped or nulled, never a 500.
* Paging cursors are opaque and HMAC-signed with a per-process key; a cursor
  is bound to its run and file, and its offset must sit on a line boundary
  inside the file. Lines over :data:`MAX_LINE_BYTES` and malformed lines are
  skipped and counted, never served.
* Every refusal is a ``RunNotFound`` (→ 404): no existence oracle.

``MAXIM_DATA_HOME`` is trusted — it is the operator's own configuration.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import stat
import threading
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO

from maxim.utils.paths import sim_reports

logger = logging.getLogger(__name__)

RUN_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
MEMBERS = frozenset({"report.json", "actions.jsonl"})
MAX_LINE_BYTES = 1024 * 1024
MAX_SCAN_BYTES = 8 * 1024 * 1024  # per page: a filter that matches nothing still returns
MAX_REPORT_BYTES = 5 * 1024 * 1024
TORN_TAIL_STALE_S = 60.0  # an unterminated last line older than this is a crashed writer's, not a live one
_MAX_CURSOR_CHARS = 1024
_ID_TIMESTAMP_RE = re.compile(r"(?:^|_)(\d{8}_\d{6})$")
# Per-action credit (rpe, pain, nac_failure_booked) is recorded from this
# actions.jsonl header version on (simulation/report.py::save_action_log writes 1.2).
CREDIT_FORMAT_VERSION = (1, 2)


class RunNotFound(LookupError):
    """Unknown, unreadable or refused — deliberately indistinguishable (404)."""


class BadCursor(ValueError):
    """A cursor that is unsigned, foreign, stale or off a line boundary (400)."""


# ── cursors ──────────────────────────────────────────────────────────────────


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class CursorSigner:
    """Opaque, tamper-evident paging cursors, keyed per process.

    A restarted server refuses old cursors (400) and the client restarts the
    page — the right outcome, since the files may have changed underneath.
    """

    def __init__(self, key: bytes | None = None) -> None:
        self._key = key if key is not None else secrets.token_bytes(32)

    def _mac(self, body: bytes) -> bytes:
        return hmac.new(self._key, body, hashlib.sha256).digest()[:16]

    def issue(self, payload: dict[str, Any]) -> str:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"{_b64e(body)}.{_b64e(self._mac(body))}"

    def verify(self, token: str, **expect: Any) -> dict[str, Any]:
        if not isinstance(token, str) or len(token) > _MAX_CURSOR_CHARS or token.count(".") != 1:
            raise BadCursor("malformed cursor")
        body_text, sig_text = token.split(".")
        try:
            body = _b64d(body_text)
            sig = _b64d(sig_text)
        except (ValueError, TypeError) as e:
            raise BadCursor("malformed cursor") from e
        if not hmac.compare_digest(sig, self._mac(body)):
            raise BadCursor("cursor was not issued by this server")
        try:
            payload = json.loads(body)
        except ValueError as e:
            raise BadCursor("malformed cursor") from e
        if not isinstance(payload, dict):
            raise BadCursor("malformed cursor")
        for key, value in expect.items():
            if payload.get(key) != value:
                raise BadCursor("cursor belongs to a different listing")
        return payload


# ── resolution ───────────────────────────────────────────────────────────────


def _root() -> Path:
    return sim_reports().resolve()


# Anchored opens need dir_fd + O_DIRECTORY + O_NOFOLLOW (POSIX). Where they are
# missing (Windows) run history is UNAVAILABLE, declared through the `runs`
# seam — a weaker, path-checked boundary would be raceable, and a security
# boundary that is sometimes weaker is the one an attacker picks.
HISTORY_SUPPORTED = os.open in os.supports_dir_fd and hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW")


def _check_id(run_id: str) -> None:
    if not isinstance(run_id, str) or RUN_ID_RE.fullmatch(run_id) is None:
        raise RunNotFound(run_id)


def open_console_run(run_id: str, root: Path | None = None) -> int:
    """THE boundary: request data → an fd for a run directory (caller closes it).

    The id is validated before any filesystem call, and the directory is opened
    relative to an fd on the root with ``O_DIRECTORY|O_NOFOLLOW`` — so a run
    dir that is, or is swapped for, a symlink is never followed. Members are
    then opened relative to the returned fd (:func:`_open_member`). Returning a
    PATH instead would hand the caller a name it could re-walk after a swap.
    """
    _check_id(run_id)
    if not HISTORY_SUPPORTED:
        raise RunNotFound(run_id)
    root = root if root is not None else _root()
    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    except OSError as e:
        raise RunNotFound(run_id) from e
    try:
        return os.open(run_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
    except OSError as e:  # a symlink (ELOOP/ENOTDIR), a file, missing, unreadable
        raise RunNotFound(run_id) from e
    finally:
        os.close(root_fd)


def _open_member(dir_fd: int, name: str) -> BinaryIO | None:
    """Open a fixed-name member relative to its run dir fd; None if absent, RunNotFound if refused."""
    if name not in MEMBERS:
        raise RunNotFound(name)
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(name, flags, dir_fd=dir_fd)
    except FileNotFoundError:
        return None
    except OSError as e:  # a symlink under O_NOFOLLOW, permissions, …
        raise RunNotFound(name) from e
    st = os.fstat(fd)
    # Regular, and single-linked: a hardlink to a secret passes lstat AND O_NOFOLLOW.
    if not stat.S_ISREG(st.st_mode) or st.st_nlink > 1:
        os.close(fd)
        raise RunNotFound(name)
    return os.fdopen(fd, "rb")


# ── coercion helpers (on-disk shapes predate the contract) ───────────────────


def _str(v: Any, cap: int | None = None) -> str | None:
    if v is None:
        return None
    s = v if isinstance(v, str) else str(v)
    return s[:cap] if cap is not None else s


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _int(v: Any) -> int | None:
    f = _num(v)
    # Bounded to int64: a 1e20 "count" passes is-integral but is not a count.
    return int(f) if f is not None and f == int(f) and abs(f) < 2**63 else None


def _bool(v: Any) -> bool | None:
    return v if isinstance(v, bool) else None


def _started_at(run_id: str, report: dict[str, Any] | None) -> float | None:
    if report is not None:
        ts = _num(report.get("ts"))
        if ts is not None:
            return ts
    m = _ID_TIMESTAMP_RE.search(run_id)
    if m is None:
        return None
    try:
        return time.mktime(time.strptime(m.group(1), "%Y%m%d_%H%M%S"))
    except (ValueError, OverflowError):
        return None


def _loads(raw: bytes) -> Any:
    """json.loads that never raises past the parser (deep nesting raises RecursionError)."""
    try:
        return json.loads(raw)
    except (ValueError, UnicodeDecodeError, RecursionError):
        return _MALFORMED


_MALFORMED = object()


def _load_report(dir_fd: int) -> dict[str, Any] | None:
    fh = _open_member(dir_fd, "report.json")
    if fh is None:
        return None
    with fh:
        raw = fh.read(MAX_REPORT_BYTES + 1)
    if len(raw) > MAX_REPORT_BYTES:
        raise RunNotFound("report.json too large")
    data = _loads(raw)
    if not isinstance(data, dict):
        raise RunNotFound("report.json unreadable")
    from maxim.utils.format_version import check_format_version

    check_format_version(data, "session_report")  # warn-once on a pre-1.0 report; still served
    return data


def _mode(report: dict[str, Any]) -> str | None:
    # Old reports carry the mode as `persona` (session.list_sessions' fallback).
    return _str(report.get("mode") or report.get("persona"), 64)


def _ended_at(report: dict[str, Any]) -> float | None:
    # ts + duration is timezone-safe; the `timestamp` string is local time (DST-ambiguous),
    # parsed only for older reports that predate `ts`.
    ts, duration = _num(report.get("ts")), _num(report.get("duration_s"))
    if ts is not None and duration is not None:
        return ts + duration
    raw = report.get("timestamp")
    if not isinstance(raw, str):
        return None
    try:
        return time.mktime(time.strptime(raw[:19], "%Y-%m-%d %H:%M:%S"))  # written as local time
    except (ValueError, OverflowError):
        return None


# ── the run list ─────────────────────────────────────────────────────────────

_summary_cache: dict[tuple[str, str], tuple[tuple[int, int | None], dict[str, Any]]] = {}
_summary_lock = threading.Lock()


def _summary(root: Path, name: str, dir_mtime_ns: int) -> dict[str, Any]:
    try:
        report_mtime: int | None = os.lstat(root / name / "report.json").st_mtime_ns
    except OSError:
        report_mtime = None
    sig = (dir_mtime_ns, report_mtime)
    key = (str(root), name)
    with _summary_lock:
        hit = _summary_cache.get(key)
    if hit is not None and hit[0] == sig:
        return hit[1]
    report = None
    try:
        dir_fd = open_console_run(name, root)
        try:
            report = _load_report(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception as e:  # noqa: BLE001 — one hostile run must never 500 the whole listing
        if not isinstance(e, RunNotFound):
            logger.warning("run list: skipping the report of %s (%s)", name, type(e).__name__)
    row: dict[str, Any] = {
        "session_id": name,
        "started_at": _started_at(name, report),
        "has_report": report is not None,
    }
    if report is not None:
        row.update(
            mode=_mode(report),
            goal=_str(report.get("goal"), 200),
            finish_reason=_str(report.get("finish_reason"), 128),
            turns=_int(report.get("turns")),
            duration_s=_num(report.get("duration_s")),
            cost_usd=_num(report.get("cost_usd")),
            total_actions=_int(report.get("total_actions")),
        )
    with _summary_lock:
        _summary_cache[key] = (sig, row)
    return row


def list_runs(*, cursor: str | None, limit: int, signer: CursorSigner) -> dict[str, Any]:
    """Newest first. Keyset paging, so runs added between pages do not shift it."""
    if not HISTORY_SUPPORTED:
        return {"runs": [], "next_cursor": None}
    root = _root()
    rows: list[tuple[float, str, dict[str, Any]]] = []
    with os.scandir(root) as it:
        for entry in it:
            if RUN_ID_RE.fullmatch(entry.name) is None:
                continue
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue  # files, and symlinked dirs (never followed)
                mtime_ns = entry.stat(follow_symlinks=False).st_mtime_ns
            except OSError:
                continue
            row = _summary(root, entry.name, mtime_ns)
            sort_key = row["started_at"] if row["started_at"] is not None else mtime_ns / 1e9
            rows.append((sort_key, entry.name, row))
    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
    if cursor is not None:
        after = signer.verify(cursor, listing="runs")
        k, n = after.get("k"), after.get("n")
        if not isinstance(k, (int, float)) or not isinstance(n, str):
            raise BadCursor("malformed cursor")
        rows = [r for r in rows if (r[0], r[1]) < (k, n)]
    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        last = page[-1]
        next_cursor = signer.issue({"listing": "runs", "k": last[0], "n": last[1]})
    return {"runs": [r[2] for r in page], "next_cursor": next_cursor}


# ── one run ──────────────────────────────────────────────────────────────────


def read_report(run_id: str) -> dict[str, Any]:
    dir_fd = open_console_run(run_id)
    try:
        report = _load_report(dir_fd)
    finally:
        os.close(dir_fd)
    if report is None:
        raise RunNotFound(run_id)
    usage = report.get("tool_usage")
    rates = report.get("tool_success_rates")
    texts = report.get("response_texts")
    return {
        "session_id": run_id,
        "started_at": _started_at(run_id, report),
        "ended_at": _ended_at(report),
        "format_version": _str(report.get("_format_version"), 16),
        "goal": _str(report.get("goal"), 2000),
        "mode": _mode(report),
        "finish_reason": _str(report.get("finish_reason"), 128),
        "duration_s": _num(report.get("duration_s")),
        "turns": _int(report.get("turns")),
        "total_actions": _int(report.get("total_actions")),
        "blocked_actions": _int(report.get("blocked_actions")),
        "cost_usd": _num(report.get("cost_usd")),
        "total_input_tokens": _int(report.get("total_input_tokens")),
        "total_output_tokens": _int(report.get("total_output_tokens")),
        "latency_p50_ms": _num(report.get("latency_p50_ms")),
        "latency_p99_ms": _num(report.get("latency_p99_ms")),
        "language_model": _str(report.get("language_model"), 256),
        "aut_model": _str(report.get("aut_model"), 256),
        "tool_usage": {str(k)[:128]: _int(v) for k, v in (usage or {}).items() if _int(v) is not None}
        if isinstance(usage, dict)
        else {},
        "tool_success_rates": {str(k)[:128]: float(v) for k, v in (rates or {}).items() if _num(v) is not None}
        if isinstance(rates, dict)
        else {},
        "pain_events_count": _int(report.get("pain_events_count")),
        "fear_blocks_count": _int(report.get("fear_blocks_count")),
        "learn_events_count": _int(report.get("learn_events_count")),
        "llm_finish_status": _str(report.get("llm_finish_status"), 128),
        "llm_finish_summary": _str(report.get("llm_finish_summary"), 4000),
        "llm_summary": _str(report.get("llm_summary"), 8000),
        "response_texts": [t[:4000] for t in texts[:100] if isinstance(t, str)] if isinstance(texts, list) else [],
    }


# ── paged JSONL ──────────────────────────────────────────────────────────────

_SKIP = object()  # a line that does not match the filter: not served, not counted


def _page_jsonl(
    run_id: str,
    name: str,
    *,
    cursor: str | None,
    limit: int,
    signer: CursorSigner,
    parse: Callable[[dict[str, Any]], Any],
    state: dict[str, Any] | None = None,
) -> tuple[list[Any], str | None, int] | None:
    """Forward page over one member. None if the member does not exist.

    ``state`` is page-spanning reader state (e.g. the header's format version,
    only seen on page 1): restored from the cursor before parsing, and carried
    into the next cursor — signed with it, so a client cannot forge it.
    """
    state = state if state is not None else {}
    dir_fd = open_console_run(run_id)
    try:
        fh = _open_member(dir_fd, name)
    finally:
        os.close(dir_fd)
    if fh is None:
        return None
    with fh:
        st = os.fstat(fh.fileno())
        size = st.st_size  # read to the size at open: a writer may still be appending
        stale = time.time() - st.st_mtime > TORN_TAIL_STALE_S
        offset = 0
        if cursor is not None:
            payload = signer.verify(cursor, run=run_id, file=name)
            raw_offset = payload.get("offset")
            carried = payload.get("state")
            if isinstance(carried, dict):
                state.update(carried)
            if not isinstance(raw_offset, int) or isinstance(raw_offset, bool) or not 0 <= raw_offset <= size:
                raise BadCursor("cursor is outside this file")
            offset = raw_offset
            if offset > 0:
                fh.seek(offset - 1)
                if fh.read(1) != b"\n":
                    raise BadCursor("cursor is not on a line boundary")
        fh.seek(offset)
        pos, skipped = offset, 0
        items: list[Any] = []
        while pos < size and len(items) < limit and pos - offset < MAX_SCAN_BYTES:
            line = fh.readline(min(MAX_LINE_BYTES + 1, size - pos))
            if not line:
                break
            if not line.endswith(b"\n"):
                if len(line) > MAX_LINE_BYTES:
                    # Oversized: consume the rest of the line, count it, move on.
                    pos += len(line)
                    while pos < size:
                        chunk = fh.readline(min(65536, size - pos))
                        if not chunk:
                            break
                        pos += len(chunk)
                        if chunk.endswith(b"\n"):
                            break
                    skipped += 1
                    continue
            raw = _loads(line)
            if raw is _MALFORMED:
                if not line.endswith(b"\n") and not stale:
                    # The last line, unterminated and unparseable, in a file
                    # written recently: a writer is mid-line. Leave it for the
                    # next page. (A truncated JSON object can never parse, so an
                    # unterminated line that DOES parse is complete — served.)
                    # A STALE torn tail is a crashed writer's: counted and
                    # passed, or paging would never end.
                    break
                pos += len(line)
                skipped += 1
                continue
            pos += len(line)
            if not isinstance(raw, dict):
                skipped += 1
                continue
            item = parse(raw)
            if item is _SKIP:
                continue
            items.append(item)
    next_cursor = signer.issue({"run": run_id, "file": name, "offset": pos, "state": state}) if pos < size else None
    return items, next_cursor, skipped


def _version_tuple(raw: Any) -> tuple[int, ...] | None:
    if not isinstance(raw, str):
        return None
    try:
        return tuple(int(p) for p in raw.split("."))
    except ValueError:
        return None


def _action_parser(state: dict[str, Any]) -> Callable[[dict[str, Any]], Any]:
    def parse(raw: dict[str, Any]) -> Any:
        if raw.get("_record_kind") == "header":
            state["format_version"] = _str(raw.get("_format_version"), 16)
            return _SKIP
        version = _version_tuple(state.get("format_version"))
        # Keyed off the header, not off key presence: a writer that drops None
        # keys would otherwise make every un-booked action read "not recorded".
        credit_recorded = version is not None and version >= CREDIT_FORMAT_VERSION
        params = raw.get("params")
        return {
            "timestamp": _num(raw.get("timestamp")),
            "tool": _str(raw.get("tool"), 128),
            "params": params if isinstance(params, dict) else {},
            "success": _bool(raw.get("success")),
            "blocked": _bool(raw.get("blocked")),
            "block_reason": _str(raw.get("block_reason"), 1000),
            "error": _str(raw.get("error"), 2000),
            "output": _str(raw.get("output"), 1000),
            "agent_id": _str(raw.get("agent_id"), 128),
            "entity_class": _str(raw.get("entity_class"), 128),
            "rpe": _num(raw.get("rpe")) if credit_recorded else None,
            "pain": _num(raw.get("pain")) if credit_recorded else None,
            "nac_failure_booked": _str(raw.get("nac_failure_booked"), 256) if credit_recorded else None,
            "credit_recorded": credit_recorded,
        }

    return parse


_NO_ACTIONS = "This run recorded no actions file — an aborted run, or a talk/rest run (those write no run artefacts)."


def page_actions(run_id: str, *, cursor: str | None, limit: int, signer: CursorSigner) -> dict[str, Any]:
    state: dict[str, Any] = {}
    result = _page_jsonl(
        run_id, "actions.jsonl", cursor=cursor, limit=limit, signer=signer, parse=_action_parser(state), state=state
    )
    if result is None:
        if cursor is not None:
            raise BadCursor("this run has no actions")
        return {"session_id": run_id, "available": False, "reason": _NO_ACTIONS, "actions": [], "skipped": 0}
    actions, next_cursor, skipped = result
    return {
        "session_id": run_id,
        "available": True,
        "reason": None,
        "format_version": state.get("format_version"),
        "actions": actions,
        "next_cursor": next_cursor,
        "skipped": skipped,
    }

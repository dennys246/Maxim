"""The console process's health log: heartbeat snapshots, stalls, swallowed exceptions.

Fed by two producer SEAMS, not by log handlers:

* :meth:`maxim.runtime.heartbeat.HeartbeatMonitor.register_listener` — snapshots
  and stalls. The monitor also logs them, but at DEBUG, which reaches a handler
  only when ``MAXIM_LOG_FILE`` forces the root logger down.
* :func:`maxim.utils.logging.register_swallow_listener` — every Stage-1
  ``swallowed_exception`` fire. A handler would see each site ONCE (later fires
  log at DEBUG), so the degraded view's counts would be wrong. Explicit-form
  (``exc=``/``operation=``) calls carry no event and are not counted.

**Producers stay cheap.** Both seams call in on someone else's thread — the
swallow seam from inside an ``except`` block on the failing thread — so a
producer only updates memory and enqueues; all disk I/O (append, age-out,
compaction) happens on this log's own writer thread. A full queue drops the
record and counts it rather than blocking the caller.

**Bounded twice** (owner decision 2026-10-13): records older than ``keep_s``
(7 days) age out — in memory on every insert, on disk at least hourly — and the
file never grows past ``max_bytes`` (50 MB). Snapshots are downsampled on disk
to one per ``persist_interval_s`` (a minute); stalls are always kept;
swallowed exceptions are persisted as per-site count DELTAS at most once a
minute per site.

**What it cannot say** (honest flags on the wire): stall detection is live only
while an agent loop runs in this process (``runtime/loop_state.py::
loop_liveness_scope`` registers the monitor's loop-state hook per run and clears
it with the last). With no run, ``stall_detection`` is False and an empty
``stalls`` means unknown, not healthy.

**Privacy.** Snapshots are ALLOW-LISTED (:data:`ALLOWED_KEYS`): never network
interfaces, Wi-Fi identity, platform/hostname or ``disk.path``. Exception
samples are truncated and scrubbed of home paths, URL credentials/queries and
token-shaped strings before they are kept.

**Scope and sharing.** Samples come from THIS process. There is ONE log per
file per process (:func:`shared_health_log`; apps acquire and release it, so two
console apps cannot double-count into one file), and every append and rewrite
holds a ``filelock.FileLock`` beside the file, so two processes on one data
home interleave whole records and never lose one to a rewrite (the
persistence-config layer-2 rule). Peers and the leader proxy keep their own
health (``/v1/debug/heartbeat``).

**Never the reason a server will not start or stop.** An unreadable or
malformed file is skipped with a warning; the writer thread survives any bad
record; shutdown is bounded even with a full queue.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HEALTH_LOG_FORMAT_VERSION = "1.0"
ALLOWED_KEYS = ("timestamp", "gpu", "cpu", "memory", "disk", "lanes", "loop")
DEFAULT_KEEP_S = 7 * 86400
DEFAULT_MAX_BYTES = 50 * 1024 * 1024
DEFAULT_PERSIST_INTERVAL_S = 60.0
SAMPLE_CHARS = 300
_SWALLOW_FLUSH_S = 60.0
_AGE_OUT_EVERY_S = 3600.0
_MAX_SERIES_POINTS = 2000
_MAX_LOAD_BYTES = 200 * 1024 * 1024
_QUEUE_MAX = 4096
_LOCK_TIMEOUT_S = 5.0
_STOP_JOIN_S = 5.0

_HOME = os.path.expanduser("~")
_SCRUBS = (
    (re.compile(r"([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@", re.I), r"\1<credentials>@"),
    (re.compile(r"(https?://[^\s?#]+)\?[^\s#]*", re.I), r"\1?<query>"),
    (re.compile(r"\b(mxc|sk|pk|rk|ghp|gho|xox[abpr])[-_][A-Za-z0-9_-]{12,}"), "<token>"),
    (re.compile(r"(?i)\b(bearer|token|api[_-]?key|password|secret)(\s*[=:]\s*|\s+)\S+"), r"\1\2<redacted>"),
)


def scrub_sample(text: str | None) -> str | None:
    """Truncate and scrub an exception message before it is kept or served."""
    if not isinstance(text, str):
        return None
    out = text[: SAMPLE_CHARS * 2]
    if _HOME and _HOME != "/":
        out = out.replace(_HOME, "~")
    for pattern, repl in _SCRUBS:
        out = pattern.sub(repl, out)
    return out[:SAMPLE_CHARS]


def _finite(value: Any, depth: int = 0) -> Any:
    """Non-finite floats → None, recursively: /ws JSON would otherwise carry a
    bare ``NaN`` token, which a browser's JSON.parse rejects."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if depth > 8:
        return None
    if isinstance(value, dict):
        return {str(k): _finite(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v, depth + 1) for v in value]
    return value


def _allow(data: dict[str, Any]) -> dict[str, Any]:
    out = {k: _finite(data[k]) for k in ALLOWED_KEYS if data.get(k) is not None}
    disk = out.get("disk")
    if isinstance(disk, dict):
        out["disk"] = {k: v for k, v in disk.items() if k != "path"}
    return out


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _record(kind: str, ts: float, data: dict[str, Any]) -> dict[str, Any]:
    return {"_format_version": HEALTH_LOG_FORMAT_VERSION, "kind": kind, "ts": ts, "data": data}


class HealthLog:
    """In-memory views + a bounded JSONL mirror written by its own thread."""

    def __init__(
        self,
        path: Path | None,
        *,
        keep_s: float = DEFAULT_KEEP_S,
        max_bytes: int = DEFAULT_MAX_BYTES,
        persist_interval_s: float = DEFAULT_PERSIST_INTERVAL_S,
        ring: int = 360,
        publish: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._path = path
        self._keep_s = keep_s
        self._max_bytes = max_bytes
        self._persist_interval_s = persist_interval_s
        # Per-app publishers: every serving app registers its own on acquire
        # (none under sandbox), so the first app on a path cannot decide for
        # the rest. `publish` here is the standalone (unshared) form.
        self._publishers: list[Callable[[str, dict[str, Any]], None]] = [publish] if publish is not None else []
        self._lock = threading.Lock()
        self._ring: deque[dict[str, Any]] = deque(maxlen=ring)  # recent snapshots, every one
        self._history: deque[dict[str, Any]] = deque()  # persisted (downsampled) snapshots
        self._stalls: deque[dict[str, Any]] = deque()
        self._sites: dict[str, dict[str, Any]] = {}
        self._latest: dict[str, Any] | None = None
        self._last_persisted_snapshot = -math.inf
        self._recorded_since: float | None = None
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=_QUEUE_MAX)
        self._writer: threading.Thread | None = None
        self._last_age_out = 0.0
        self.dropped = 0
        self._loaded = False
        self._writer_stop: threading.Event | None = None
        self._life_lock = threading.Lock()
        self._users = 0
        self._monitor: Any = None
        self._monitor_users = 0

    # ── state the API reports, read LIVE ──

    @property
    def monitoring(self) -> bool:
        return self._monitor is not None

    @property
    def stall_detection(self) -> bool:
        """Read at view time: the agent loop may register its hook after startup."""
        monitor = self._monitor
        return monitor is not None and bool(getattr(monitor, "has_loop_state_hook", False))

    # ── shared lifecycle (one log per file per process) ──

    def acquire(self, *, sample: bool, publish: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        """An app starts serving. The first user starts the log; ``sample``
        attaches the monitor; ``publish`` is THIS app's /ws publisher (None
        under sandbox). Start/stop happen under the lifecycle lock, so an app
        arriving while the last one leaves can never be left without a writer."""
        from maxim.utils.logging import register_swallow_listener, unregister_swallow_listener

        with self._life_lock:
            first = self._users == 0
            if first:
                self.start()  # raises → nothing counted
                register_swallow_listener(self.on_swallowed)
            self._users += 1
            try:
                if sample and self._monitor_users == 0:
                    monitor = _acquire_monitor()
                    monitor.register_listener(self.on_heartbeat)
                    self._monitor = monitor
                if sample:
                    self._monitor_users += 1
            except BaseException:
                # Roll this user back: the app sees the raise and will not release.
                self._users -= 1
                if self._users == 0:
                    unregister_swallow_listener(self.on_swallowed)
                    self.stop()
                raise
            if publish is not None:
                self._publishers.append(publish)

    def release(self, *, sample: bool, publish: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        """Mirror of :meth:`acquire`; the last user stops the log."""
        from maxim.utils.logging import unregister_swallow_listener

        with self._life_lock:
            if publish is not None and publish in self._publishers:
                self._publishers.remove(publish)
            try:
                if sample and self._monitor_users > 0:
                    self._monitor_users -= 1
                    if self._monitor_users == 0 and self._monitor is not None:
                        monitor, self._monitor = self._monitor, None
                        monitor.unregister_listener(self.on_heartbeat)
                        _release_monitor(monitor)
            finally:
                self._users = max(0, self._users - 1)
                if self._users == 0:
                    unregister_swallow_listener(self.on_swallowed)
                    self.stop()

    def start(self) -> None:
        """Load the retained window (once per instance) and start the writer thread."""
        if self._writer is not None:
            return
        if not self._loaded:
            self._loaded = True  # a restart re-loading over memory would count everything twice
            self._load()
        if self._path is not None:
            stop_event = threading.Event()  # this writer's own: a later stop can only reach the writer it took
            writer = threading.Thread(
                target=self._write_loop, args=(stop_event,), name="console-health-log", daemon=True
            )
            writer.start()
            self._writer, self._writer_stop = writer, stop_event

    def stop(self) -> None:
        """Flush swallow deltas and drain the writer — bounded, even with a full queue."""
        now = time.time()
        with self._lock:
            pending = [self._take_delta(a, now) for a in self._sites.values() if a["unflushed"]]
        for rec in pending:
            self._enqueue(rec)
        writer, stop_event = self._writer, self._writer_stop
        self._writer, self._writer_stop = None, None
        if writer is not None and stop_event is not None:
            stop_event.set()
            writer.join(timeout=_STOP_JOIN_S)

    # ── producers (cheap: memory + enqueue only) ──

    def on_heartbeat(self, kind: str, data: dict[str, Any]) -> None:
        now = time.time()
        if kind == "heartbeat":
            snap = _allow(data)
            rec = _record("health", now, snap)
            with self._lock:
                self._latest = snap
                self._ring.append(rec)
                persist = now - self._last_persisted_snapshot >= self._persist_interval_s
                if persist:
                    self._last_persisted_snapshot = now
                    self._history.append(rec)
                self._note_recorded(now)
                self._prune_locked(now)
            if persist:
                self._enqueue(rec)
            self._emit("health", snap)
        elif kind == "heartbeat_stall":
            stall = {k: _finite(v) for k, v in data.items() if isinstance(v, (int, float, str, type(None)))}
            rec = _record("health_stall", now, stall)
            with self._lock:
                self._stalls.append(rec)
                self._note_recorded(now)
                self._prune_locked(now)
            self._enqueue(rec)
            self._emit("health_stall", stall)

    def on_swallowed(self, site: str, exc_type: str | None, exc_text: str | None) -> None:
        now = time.time()
        sample = scrub_sample(exc_text)
        flush: dict[str, Any] | None = None
        with self._lock:
            agg = self._sites.get(site)
            if agg is None:
                agg = self._sites[site] = {
                    "site": site,
                    "count": 0,
                    "first_seen": now,
                    "last_seen": now,
                    "exc_type": exc_type,
                    "sample": sample,
                    "unflushed": 0,
                    "flushed_at": 0.0,
                }
            agg["count"] += 1
            agg["unflushed"] += 1
            agg["last_seen"] = now
            agg["exc_type"] = exc_type or agg["exc_type"]
            agg["sample"] = sample or agg["sample"]
            self._note_recorded(now)
            if now - agg["flushed_at"] >= _SWALLOW_FLUSH_S:
                flush = self._take_delta(agg, now)
        if flush is not None:
            self._enqueue(flush)

    # ── views ──

    def view(self, *, since: float) -> dict[str, Any]:
        now = time.time()
        floor = max(since, now - self._keep_s)
        with self._lock:
            latest = dict(self._latest) if self._latest is not None else None
            stalls = [r for r in self._stalls if r["ts"] >= floor]
            seen = {id(r) for r in self._history}
            series = [r for r in self._history if r["ts"] >= floor]
            series += [r for r in self._ring if r["ts"] >= floor and id(r) not in seen]
            recorded_since = self._recorded_since
        series.sort(key=lambda r: r["ts"])
        if len(series) > _MAX_SERIES_POINTS:
            series = series[:: math.ceil(len(series) / _MAX_SERIES_POINTS)]
        strip = lambda r: {"kind": r["kind"], "ts": r["ts"], "data": r["data"]}  # noqa: E731
        return {
            "monitoring": self.monitoring,
            "stall_detection": self.stall_detection,
            "dropped_records": self.dropped,
            "recorded_since": max(recorded_since, floor) if recorded_since is not None else None,
            "latest": latest,
            "stalls": [strip(r) for r in stalls],
            "series": [strip(r) for r in series],
        }

    def degraded(self, *, since: float) -> dict[str, Any]:
        floor = max(since, time.time() - self._keep_s)
        with self._lock:
            sites = [dict(a) for a in self._sites.values() if a["last_seen"] >= floor]
            recorded_since = self._recorded_since
        for s in sites:
            s.pop("unflushed", None)
            s.pop("flushed_at", None)
        sites.sort(key=lambda s: (-s["count"], s["site"]))
        return {"since": since, "counted_since": recorded_since, "sites": sites}

    # ── internals ──

    def _note_recorded(self, ts: float) -> None:
        if self._recorded_since is None or ts < self._recorded_since:
            self._recorded_since = ts

    def _prune_locked(self, now: float) -> None:
        floor = now - self._keep_s
        for dq in (self._history, self._stalls):
            while dq and dq[0]["ts"] < floor:
                dq.popleft()
        for site in [k for k, a in self._sites.items() if a["last_seen"] < floor]:
            del self._sites[site]
        if self._recorded_since is not None and self._recorded_since < floor:
            self._recorded_since = floor

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        for publish in list(self._publishers):
            try:
                publish(kind, data)
            except Exception as e:  # a /ws problem must not stop health capture
                logger.debug("health publish raised %s", type(e).__name__)

    @staticmethod
    def _take_delta(agg: dict[str, Any], now: float) -> dict[str, Any]:
        rec = _record(
            "swallowed_exception",
            now,
            {
                "site": agg["site"],
                "count_delta": agg["unflushed"],
                "first_seen": agg["first_seen"],
                "last_seen": agg["last_seen"],
                "exc_type": agg["exc_type"],
                "sample": agg["sample"],
            },
        )
        agg["unflushed"] = 0
        agg["flushed_at"] = now
        return rec

    def _enqueue(self, rec: dict[str, Any]) -> None:
        if self._path is None or self._writer is None:
            return
        try:
            self._queue.put_nowait(rec)
        except queue.Full:
            self.dropped += 1  # never block a producer's thread on disk

    def _write_loop(self, stop_event: threading.Event) -> None:
        while True:
            try:
                first = self._queue.get(timeout=0.2)
            except queue.Empty:
                if stop_event.is_set():
                    return
                continue
            batch = [first] if first is not None else []
            while True:
                try:
                    nxt = self._queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is not None:
                    batch.append(nxt)
            if batch:
                try:
                    self._append(batch)
                except Exception as e:  # the writer must outlive any one bad batch or file
                    self.dropped += len(batch)
                    logger.warning(
                        "console health log: a write failed (%s); %d record(s) dropped so far",
                        type(e).__name__,
                        self.dropped,
                    )

    def _append(self, batch: list[dict[str, Any]]) -> None:
        assert self._path is not None
        from filelock import FileLock, Timeout

        if self._path.is_symlink():
            raise OSError("refusing a symlinked health log")  # _load refuses one too
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(json.dumps(rec, separators=(",", ":"), default=str) + "\n" for rec in batch)
        try:
            with FileLock(str(self._path) + ".lock", timeout=_LOCK_TIMEOUT_S):
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(payload)
                now = time.time()
                if self._path.stat().st_size > self._max_bytes or now - self._last_age_out >= _AGE_OUT_EVERY_S:
                    self._last_age_out = now
                    self._rewrite_window()
        except Timeout as e:
            raise OSError("health log lock busy") from e

    def _rewrite_window(self) -> None:
        """Drop aged-out records; if still over the cap, the oldest, down to half of it."""
        assert self._path is not None
        floor = time.time() - self._keep_s
        lines: list[str] = []
        with open(self._path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    ts = _num(rec.get("ts")) if isinstance(rec, dict) else None
                except (ValueError, RecursionError):
                    ts = None
                if ts is not None and ts >= floor:
                    lines.append(line)  # anything unreadable or aged out is dropped, never fatal
        total = sum(len(x.encode()) for x in lines)
        if total > self._max_bytes:
            budget = self._max_bytes // 2
            while lines and total > budget:
                total -= len(lines.pop(0).encode())
        from maxim.utils.atomic_io import atomic_write_text

        atomic_write_text(str(self._path), "".join(lines))

    def _load(self) -> None:
        """Load the retained window. A malformed or unreadable file must never stop
        a console starting: each line is type-checked and any failure skips just
        that line; an unreadable file is skipped whole, with a warning."""
        if self._path is None:
            return
        from maxim.utils.format_version import check_format_version

        floor = time.time() - self._keep_s
        unreadable = 0
        try:
            if not self._path.is_file() or self._path.is_symlink():
                return
            if self._path.stat().st_size > _MAX_LOAD_BYTES:
                return  # far past our own cap: rewritten by the next age-out, never parsed whole
            with open(self._path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        self._load_line(line, floor, check_format_version)
                    except (ValueError, TypeError, AttributeError, KeyError, RecursionError):
                        unreadable += 1  # one bad line skips that line, never the console
        except OSError as e:
            logger.warning("console health log unreadable (%s); starting with no history", type(e).__name__)
            return
        if unreadable:
            logger.warning("console health log: skipped %d unreadable line(s) in %s", unreadable, self._path)

    def _load_line(self, line: str, floor: float, check_format_version: Callable[..., str]) -> None:
        rec = json.loads(line)
        if not isinstance(rec, dict):
            return
        ts = _num(rec.get("ts"))
        data = rec.get("data")
        if ts is None or ts < floor or not isinstance(data, dict):
            return
        if check_format_version(rec, "console_health_log", log=logger) not in (HEALTH_LOG_FORMAT_VERSION, "0.x"):
            return  # a newer writer's shape: skip rather than misread
        kind = rec.get("kind")
        if kind == "health":
            self._history.append(_record("health", ts, _allow(data)))  # re-apply: never trust the file
        elif kind == "health_stall":
            stall = {k: _finite(v) for k, v in data.items() if isinstance(v, (int, float, str, type(None)))}
            self._stalls.append(_record("health_stall", ts, stall))
        elif kind == "swallowed_exception":
            self._load_delta(ts, data)
        else:
            return
        self._note_recorded(ts)

    def _load_delta(self, ts: float, d: dict[str, Any]) -> None:
        site, delta = d.get("site"), d.get("count_delta")
        if not isinstance(site, str) or isinstance(delta, bool) or not isinstance(delta, int) or delta < 0:
            return
        first = _num(d.get("first_seen")) or ts
        last = _num(d.get("last_seen")) or ts
        exc_type = d.get("exc_type") if isinstance(d.get("exc_type"), str) else None
        agg = self._sites.setdefault(
            site[:512],
            {
                "site": site[:512],
                "count": 0,
                "first_seen": first,
                "last_seen": last,
                "exc_type": exc_type,
                "sample": scrub_sample(d.get("sample")),
                "unflushed": 0,
                "flushed_at": 0.0,
            },
        )
        agg["count"] += delta
        agg["first_seen"] = min(agg["first_seen"], first)
        agg["last_seen"] = max(agg["last_seen"], last)


# ── one log per file per process, and the shared heartbeat monitor ───────────

_registry: dict[str, HealthLog] = {}
_registry_lock = threading.Lock()


def shared_health_log(path: Path) -> HealthLog:
    """The process's log for ``path``. Every console app serving from one data
    home shares it, so a swallow is counted once, not once per app; each app
    brings its own publisher on :meth:`HealthLog.acquire`."""
    key = str(path)
    with _registry_lock:
        log = _registry.get(key)
        if log is None:
            log = _registry[key] = HealthLog(path)
        return log


# The heartbeat monitor is a PROCESS singleton (the router may start it too):
# the first console user starts it if nobody else had, and only the last one out
# stops it — and only if a console started it.
_monitor_lock = threading.Lock()
_monitor_users = 0
_monitor_owned = False


def _acquire_monitor() -> Any:
    global _monitor_users, _monitor_owned
    from maxim.runtime.heartbeat import get_heartbeat_monitor

    monitor = get_heartbeat_monitor()
    with _monitor_lock:
        owned = _monitor_owned if _monitor_users else not monitor.is_running
        monitor.start()  # idempotent; counted only once it succeeded
        _monitor_owned = owned
        _monitor_users += 1
    return monitor


def _release_monitor(monitor: Any) -> None:
    global _monitor_users, _monitor_owned
    with _monitor_lock:
        _monitor_users -= 1
        if _monitor_users == 0 and _monitor_owned:
            _monitor_owned = False
            # Under the lock ON PURPOSE: a user arriving mid-stop must wait and
            # restart the sampler, not adopt one that is shutting down (it would
            # read `is_running` True, own nothing, and never see a sample).
            # The cost is shutdown latency (≤ interval + 1 s), never correctness.
            monitor.stop()

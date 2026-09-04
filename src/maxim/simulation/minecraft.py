"""Minecraft world seam — transport client + PerceptSource (1.1.4 PR 3).

The Python half of the world bridge (`docs/plans/world_seam_1_1_4.md` §PR 3;
design source `docs/plans/minecraft_benchmark.md` §"What to build"). The JS
half — a Mineflayer process — lives in `scripts/minecraft_bridge/` (dev-side,
not packaged) and owns the game connection; this module owns:

* :class:`MinecraftClient` — newline-delimited-JSON-over-TCP transport to the
  bridge process, a reader thread maintaining the latest world-state snapshot
  (per-Maxim-tick buffering, the plan's Q7 answer) and a bounded queue of
  text-shaped game events, plus a blocking ``call_action`` RPC.

  **Transport note (recorded deviation):** the plan sketched "WS/JSON-RPC";
  this ships line-delimited JSON over plain TCP instead — same JSON-RPC-shaped
  messages, ZERO new dependencies (core has no websocket client, and an
  optional extra for a localhost pipe buys nothing). If a browser-facing
  consumer ever needs WS, that is a bridge-process add, not a seam change.

* :class:`MinecraftPerceptSource` — the frozen CC8 ``PerceptSource`` protocol
  over the client's event queue. Percepts are TEXT-shaped by construction:
  ``MemoryHub.on_percept_received`` returns early unless ``transcript_chunk``
  or ``content`` is non-empty text, so a numeric-only percept would be
  silently invisible to memory (survey finding F3).

Wire protocol (frozen here; the bridge process implements the other side):

  JS -> PY  {"type": "state", "data": {<sensor>: <float>, ...}}
  JS -> PY  {"type": "event", "kind": "chat|damage|death|block|spawn|info",
             "text": "<human-readable game event>"}
  PY -> JS  {"type": "action", "id": <int>, "name": "<affordance>",
             "params": {...}}
  JS -> PY  {"type": "action_result", "id": <int>, "ok": <bool>,
             "detail": "<str>", "state": {<post-action snapshot>}}

Unknown message types are ignored (forward-compat); a malformed line is
logged and skipped, never fatal. The reader thread is a daemon; ``close()``
is idempotent and wakes any blocked ``call_action``. Raw TCP, not HTTP —
outside ``utils/http.py``'s surface by design (that invariant's text,
origin incident and CI grep are all HTTP request/response; there is no
persistent-bidirectional-stream primitive there to route through).

**Unknown outcomes (the Reachy honesty convention, copied faithfully —
review round fix, both lenses):** an action the GAME refuses (``ok:false``)
or that cannot be SENT is a confirmed failure. A TIMEOUT is neither — the
action was dispatched and may still complete in-game after we stop waiting
(fire-and-forget divergence; ``pathfinder.goto`` on a long path is the
routine case). ``call_action`` marks that result ``unknown: True`` and the
backend books it mechanically-optimistic with ``outcome_valence:
"neutral"`` — unknown is not success AND not failure (the ternary
invariant); the next snapshot then tells the truth about the world either
way. A late-arriving result's embedded state snapshot is still absorbed as
a state observation; its routing entry is dropped (no leak).

Sensor flow into the body: the client does NOT write ``vital_metrics``
itself — :class:`~maxim.embodiment.backends.minecraft.MinecraftWorldBackend`
owns that (world-owned sensors, ``world_set_axis`` with an ``owner=``
self-declaration), pulling ``latest_state()`` from here. One writer per
sensor, the DoAFeed discipline.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from collections import deque
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Bounded event queue: a chatty server must not grow memory without bound;
# oldest events drop first (the agent perceives the recent world).
_EVENT_QUEUE_MAX = 256
_DEFAULT_ACTION_TIMEOUT_S = 15.0


class MinecraftClient:
    """NDJSON-over-TCP client for the Mineflayer bridge process.

    ``connection_factory`` is the injectable transport seam (the
    ``make_reachy_rest_doa_reader(fetch=...)`` pattern): tests pass a factory
    returning any object with ``makefile``-compatible ``sendall``/``recv``…
    — in practice a ``socket.socketpair`` end. Production default dials
    ``host:port``.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 25567,
        *,
        connection_factory: "Callable[[], socket.socket] | None" = None,
        action_timeout_s: float = _DEFAULT_ACTION_TIMEOUT_S,
    ) -> None:
        self._factory = connection_factory or (lambda: socket.create_connection((host, port), timeout=10.0))
        self._action_timeout_s = action_timeout_s
        self._sock: socket.socket | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._latest_state: dict[str, float] = {}
        self._state_ts: "float | None" = None
        self._events: deque[dict[str, Any]] = deque(maxlen=_EVENT_QUEUE_MAX)
        self._next_id = 1
        self._pending: dict[int, dict[str, Any]] = {}
        # ids with a live waiter: results for any OTHER id are dropped after
        # their state snapshot is absorbed — a late/unsolicited result must
        # not grow _pending forever (executor-lens review, PR 3 round).
        self._waiting: set[int] = set()
        self._pending_cv = threading.Condition(self._lock)

    # ── lifecycle ────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Dial the bridge and start the reader thread. Raises on failure —
        a seam that cannot reach its world must fail LOUD at startup, never
        degrade into a silently world-less agent. Reconnect-safe: clears the
        closed flag so a reused client's reader is not stillborn."""
        self._closed.clear()
        sock = self._factory()
        # create_connection's timeout PERSISTS as the socket's operation
        # timeout — recv would raise after 10 quiet seconds and kill the
        # reader with the wrong diagnosis (architecture-lens review, PR 3
        # round: a quiet bridge is normal; a dead reader is not).
        try:
            sock.settimeout(None)
        except OSError:
            pass
        self._sock = sock
        self._reader = threading.Thread(target=self._read_loop, name="minecraft-bridge-reader", daemon=True)
        self._reader.start()

    def close(self) -> None:
        self._closed.set()
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        with self._pending_cv:
            self._pending_cv.notify_all()  # wake any blocked call_action now

    # ── reads ────────────────────────────────────────────────────────────

    def latest_state(self) -> dict[str, float]:
        """The most recent world snapshot (copy). Empty until the first
        ``state`` message arrives."""
        with self._lock:
            return dict(self._latest_state)

    def state_age_s(self) -> float:
        """Seconds since the last snapshot; ``inf`` before the first one."""
        with self._lock:
            return (time.monotonic() - self._state_ts) if self._state_ts is not None else float("inf")

    def pop_event(self) -> "dict[str, Any] | None":
        with self._lock:
            return self._events.popleft() if self._events else None

    def has_events(self) -> bool:
        with self._lock:
            return bool(self._events)

    # ── actions ──────────────────────────────────────────────────────────

    def call_action(self, name: str, params: "dict[str, Any] | None" = None) -> dict[str, Any]:
        """Blocking action RPC: send, wait for the matching ``action_result``.

        Returns the result dict (``ok``/``detail``/``state``). A timeout or a
        dead connection returns ``{"ok": False, "detail": ...}`` — the game
        not confirming is a REAL failure (the motor backend's honesty
        contract: unknown is not success, and here unknown after timeout is
        reported as failure because the action's effect cannot be
        established).
        """
        sock = self._sock
        if sock is None:
            return {"ok": False, "unknown": False, "detail": "bridge not connected"}
        with self._pending_cv:
            action_id = self._next_id
            self._next_id += 1
            self._waiting.add(action_id)
        line = json.dumps({"type": "action", "id": action_id, "name": name, "params": params or {}})
        try:
            sock.sendall(line.encode("utf-8") + b"\n")
        except OSError as exc:
            with self._pending_cv:
                self._waiting.discard(action_id)
            return {"ok": False, "unknown": False, "detail": f"bridge send failed: {exc}"}
        deadline = time.monotonic() + self._action_timeout_s
        with self._pending_cv:
            try:
                while action_id not in self._pending:
                    remaining = deadline - time.monotonic()
                    if self._closed.is_set():
                        return {"ok": False, "unknown": True, "detail": "client closed while waiting"}
                    if remaining <= 0:
                        # UNKNOWN, not confirmed failure: the action was
                        # dispatched and may still complete in-game (module
                        # docstring, the fire-and-forget paragraph).
                        return {
                            "ok": False,
                            "unknown": True,
                            "detail": f"action {name!r} unconfirmed after {self._action_timeout_s}s",
                        }
                    self._pending_cv.wait(timeout=remaining)
                result = self._pending.pop(action_id)
                result.setdefault("unknown", False)
                return result
            finally:
                self._waiting.discard(action_id)
                self._pending.pop(action_id, None)

    # ── reader ───────────────────────────────────────────────────────────

    def _absorb_state(self, data: dict[str, Any]) -> None:
        cleaned: dict[str, float] = {}
        for key, value in data.items():
            try:
                cleaned[str(key)] = float(value)
            except (TypeError, ValueError):
                # A non-numeric value is dropped, leaving that sensor at its
                # previous truth — stale-but-real beats fabricated.
                continue
        with self._lock:
            self._latest_state = cleaned
            self._state_ts = time.monotonic()

    def _read_loop(self) -> None:
        sock = self._sock
        if sock is None:
            return
        buffer = b""
        while not self._closed.is_set():
            try:
                chunk = sock.recv(65536)
            except OSError:
                break
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                if raw.strip():
                    self._handle_line(raw)
        if not self._closed.is_set():
            logger.warning("minecraft bridge connection closed by peer — no further world state")
        with self._pending_cv:
            self._pending_cv.notify_all()  # a dead reader must not strand a waiter

    def _handle_line(self, raw: bytes) -> None:
        try:
            msg = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            logger.warning("minecraft bridge sent a malformed line (skipped): %r", raw[:200])
            return
        mtype = msg.get("type")
        if mtype == "state":
            data = msg.get("data")
            if isinstance(data, dict):
                self._absorb_state(data)
        elif mtype == "event":
            text = msg.get("text")
            if isinstance(text, str) and text.strip():
                kind = str(msg.get("kind", "info"))
                if kind == "error":
                    # The bridge refusing/failing loudly (e.g. a second
                    # client) must not be a quietly-queued percept only.
                    logger.warning("minecraft bridge error event: %s", text)
                with self._lock:
                    self._events.append({"kind": kind, "text": text})
        elif mtype == "action_result":
            # The embedded snapshot is a STATE OBSERVATION and is absorbed
            # unconditionally — the post-action sync must read POST-action
            # truth, not the previous periodic snapshot (executor-lens
            # review, PR 3 round: the fake hid exactly this divergence).
            state = msg.get("state")
            if isinstance(state, dict):
                self._absorb_state(state)
            action_id = msg.get("id")
            if isinstance(action_id, int):
                with self._pending_cv:
                    if action_id in self._waiting:
                        self._pending[action_id] = {
                            "ok": bool(msg.get("ok")),
                            "unknown": False,
                            "detail": str(msg.get("detail", "")),
                            "state": state if isinstance(state, dict) else {},
                        }
                        self._pending_cv.notify_all()
                    else:
                        logger.debug("late/unsolicited action_result id=%s dropped (state absorbed)", action_id)
        # unknown types: ignored (forward-compat)


class MinecraftPerceptSource:
    """Game events as text percepts — the frozen CC8 ``PerceptSource`` shape.

    Live source: no ``advance_step`` (the contract says live sources do not
    implement it); ``has_pending`` reflects the event queue so the idle gate
    can skip quiet ticks; never exhausted while the client lives.
    """

    def __init__(self, client: MinecraftClient) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return "minecraft"

    @property
    def capabilities(self) -> set[str]:
        return {"transcript"}

    def has_pending(self) -> bool:
        return self._client.has_events()

    def is_exhausted(self) -> bool:
        return False

    def next_percept(self) -> Any:
        event = self._client.pop_event()
        if event is None:
            return None
        from maxim.agents.percept_factory import make_text_percept

        return make_text_percept(
            f"[minecraft:{event['kind']}] {event['text']}",
            source="minecraft",
            channel="narrative",
        )

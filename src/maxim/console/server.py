"""FastAPI app for ``maxim serve`` — the Console backend + facade contract.

Binds **127.0.0.1 only** (it holds keys and can run/configure Maxim; a hosted
console is an explicit non-goal). Three surfaces:

* ``/api/*`` — JSON facade, 1:1 with ``api.py`` verbs. Existing structured verbs
  (``/api/models``, ``/api/diagnose``) are live; the seams (``/api/probe``,
  ``/api/setup/*``, ``/api/recall``, ``/api/run``) are typed **501 stubs** — their
  Pydantic shapes are in OpenAPI so the maxim-pulse kit can generate the full
  ``FacadeClient`` now (Phase 1 fills in the bodies without touching the schema).
* ``/ws`` — the EventClient stream (the EVENT seam, LIVE —
  [reachy_app_maxim_seams.md] § EVENT): ``sim_log`` records bridged via
  ``register_sim_sink`` into per-connection bounded queues (drop-oldest +
  a ``dropped`` meta-event), filtered per client by ``SubscribeFrame``,
  correlated to runs by ``run_id`` + ``run`` lifecycle events. NOT built on
  ``api.on()`` — that stays the embedder-SDK surface.
* ``/`` — the static Console bundle (resolved from ``--ui-dist`` / config).

**FastAPI is justified specifically by the OpenAPI auto-emit** (the cross-repo
contract); that is why this is not stdlib like ``leader_proxy``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from maxim.console.schemas import (
    CloudSetupRequest,
    ConsoleEvent,
    DiagnoseResponse,
    DiagnoseSection,
    MeshSetupRequest,
    ModelInfoWire,
    ModelsResponse,
    PlatformWire,
    ProbeRequest,
    ProbeResult,
    RecallResponse,
    RunAccepted,
    RunRequest,
    SetupResult,
    SubscribeFrame,
)

logger = logging.getLogger(__name__)

_HEARTBEAT_INTERVAL_S = 15.0
_NOT_IMPLEMENTED = "Seam not yet implemented (Phase 1) — shape is contract-complete for type-gen."

# The wizard's "2-3 curated ▾ / Advanced" picker: profiles flagged ``curated`` are
# surfaced by default; the rest live under Advanced. This is the single editable
# curation point — a small span of fast-local / capable-local / flagship-cloud.
# Edit freely; unknown names are simply never marked curated.
_CURATED_PROFILES: frozenset[str] = frozenset(
    {
        "gemma-2b-it",  # tiny — Pi / low-RAM
        "llama-3-8b-instruct",  # capable local default
        "claude-sonnet-4-6",  # flagship cloud
    }
)

# Committed OpenAPI snapshot — the cross-repo contract artifact. maxim-pulse
# generates its FacadeClient from this file (no running server needed); a pytest
# freshness check fails if it drifts from build_app().openapi(). Refresh with
# `maxim serve --dump-openapi`.
_OPENAPI_SNAPSHOT = Path(__file__).parent / "openapi.json"


def openapi_schema() -> dict[str, Any]:
    """The canonical OpenAPI schema (from the app with no static bundle mounted)."""
    return build_app(None).openapi()


api = APIRouter(prefix="/api", tags=["facade"])


# ── live verbs (wrap existing api.py facades) ───────────────────────────────


@api.get("/models", response_model=ModelsResponse, summary="List LLM profiles")
def get_models() -> ModelsResponse:
    import maxim

    groups = maxim.list_models()  # dict[str, list[ModelInfo]]

    def _wire(m: Any) -> ModelInfoWire:
        d = dataclasses.asdict(m)
        return ModelInfoWire(**d, curated=d.get("name") in _CURATED_PROFILES)

    return ModelsResponse(groups={group: [_wire(m) for m in members] for group, members in groups.items()})


@api.get("/diagnose", response_model=DiagnoseResponse, summary="Environment diagnostics")
def get_diagnose() -> DiagnoseResponse:
    import maxim

    report = maxim.diagnose()
    sections: list[DiagnoseSection] = []
    for s in getattr(report, "sections", []) or []:
        d = dataclasses.asdict(s) if dataclasses.is_dataclass(s) else (s if isinstance(s, dict) else {})
        sections.append(
            DiagnoseSection(
                name=str(d.get("name", s if isinstance(s, str) else "")),
                status=str(d.get("status", "")),
                detail=d.get("detail"),
            )
        )
    p = getattr(report, "platform", None)
    platform = PlatformWire(
        os=str(getattr(p, "os", "") or ""),
        arch=str(getattr(p, "arch", "") or ""),
        os_release=str(getattr(p, "os_release", "") or ""),
        runtime=str(getattr(p, "runtime", "") or ""),
    )
    return DiagnoseResponse(platform=platform, sections=sections)


# ── seam stubs (typed; 501 until Phase 1) ───────────────────────────────────


@api.post("/probe", response_model=ProbeResult, summary="Test a mesh/cloud connection")
def post_probe(body: ProbeRequest) -> ProbeResult:
    # PROBE seam. Dispatches on the request shape (contract fix): a MESH probe
    # (``url`` present) goes through the canonical peer-probe entry point + the
    # shared classifier — the same path `maxim doctor` uses, so console and
    # doctor agree on verdicts. A CLOUD probe (``provider`` present, no ``url``)
    # is a cheap pre-save key check.
    if body.url:
        from maxim.models.language.maxim_peer_backend import _MaximPeerBackend
        from maxim.peer.probe_classify import classify_probe_outcome

        backend = _MaximPeerBackend.for_url(body.url, api_key=body.api_key, model=body.model)
        result = backend.health_check()  # runtime.llm_server.ProbeResult(url, outcome, detail, latency_ms)
        cls = classify_probe_outcome(result.outcome, result.detail, result.latency_ms, result.url)
        return ProbeResult(
            status=cls.status,
            outcome=result.outcome,
            message=cls.message,
            fix_hint=cls.fix,
            latency_ms=result.latency_ms,
        )

    if body.provider:
        # Cloud pre-save check. A live provider round-trip is deliberately NOT
        # faked here — returning "ok" without a real call would be a false green
        # that lets a bad key sail through the wizard. We validate what we can
        # locally (a key was supplied) and return a WARN so the wizard may
        # proceed to save but the operator isn't told the key is verified.
        # A live cheap cloud-key round-trip is tracked follow-up (needs a
        # per-provider models-list/1-token probe surface).
        if not body.api_key:
            return ProbeResult(
                status="fail",
                outcome="missing_key",
                message=f"No API key supplied for cloud provider {body.provider!r}.",
                fix_hint="Paste the provider key to test before saving.",
            )
        return ProbeResult(
            status="warn",
            outcome="cloud_key_not_live_tested",
            message=f"Key accepted for {body.provider!r}; a live cloud round-trip test is pending.",
            fix_hint=None,
        )

    raise HTTPException(status_code=422, detail="Probe requires either 'url' (mesh) or 'provider' (cloud).")


@api.post("/setup/mesh", response_model=SetupResult, summary="Write mesh (peer→leader) config")
def post_setup_mesh(body: MeshSetupRequest) -> SetupResult:
    # SETUP seam. Thin call into the sanctioned single-writer helper: writes a
    # resolvable large-tier PEER placement (role=peer + lanes.large.remote_*),
    # with the key stored as a REF (atomic_write_secret → 0600 file), never
    # inline. The app does not hand-assemble the lane dict or know the ref rules.
    from maxim.exceptions import ConfigurationError
    from maxim.runtime.config_writer import apply_mesh_setup

    try:
        secret_path, written = apply_mesh_setup(body.leader_url, body.api_key)
    except ConfigurationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return SetupResult(
        ok=True,
        placement="mesh",
        detail=f"Wrote peer→leader config to {written}; key stored as a ref at {secret_path}.",
    )


@api.post("/setup/cloud", response_model=SetupResult, summary="Write cloud provider config")
def post_setup_cloud(body: CloudSetupRequest) -> SetupResult:
    # SETUP seam. Thin call into the sanctioned single-writer helper: writes a
    # resolvable large-tier CLOUD placement (cloud.enabled + profile + budget),
    # key stored as a REF (0600 file), never inline. The placement's api_key_ref
    # is resolved into the provider env var at lane-build time.
    from maxim.exceptions import ConfigurationError
    from maxim.runtime.config_writer import apply_cloud_setup

    try:
        secret_path, written = apply_cloud_setup(
            body.provider, body.profile, body.api_key, monthly_budget_usd=body.monthly_budget_usd
        )
    except ConfigurationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return SetupResult(
        ok=True,
        placement="cloud",
        detail=f"Wrote {body.provider}/{body.profile} cloud config to {written}; key stored as a ref at {secret_path}.",
    )


@api.get("/recall", response_model=RecallResponse, summary="What Maxim remembers about you")
def get_recall() -> RecallResponse:
    # RECALL seam: wrap the consumer-shaped api.recall() — a provenance-filtered,
    # salience-ranked, curated blend across pluggable recall sources. The console
    # never touches the bio-stack directly (thin-app rule); it maps the curated
    # dataclass to the wire shape.
    #
    # Post-merge review fix (Exec B1): the read MUST target the HANDLE agent's
    # home (~/.maxim/agents/<agent_id>/) — the home campaign runs write to —
    # not the api-session home (~/.maxim/memory/). Reading the wrong home made
    # "Adventure teaches Talk" silently invisible to MemoryView, and the
    # curator's honest-empty rule masked the loss as correct behavior. This is
    # a file-based read of last-SAVED state (campaign end runs full
    # consolidation + saves), which keeps the thin-app rule: no live bio-stack
    # objects cross the server boundary.
    import maxim
    from maxim.console.schemas import Preference, StoryMemory

    with _handle_lock:
        agent_id = _handle.agent_id if _handle is not None else _HANDLE_AGENT_ID
    r = maxim.recall(agent_id=agent_id)
    return RecallResponse(
        name=r.name,
        player_model=list(r.player_model),
        story_memories=[StoryMemory(summary=m.text, when=None, salience=m.salience) for m in r.story_memories],
        preferences=[Preference(about=p.text, learned_from=p.learned_from) for p in r.preferences],
    )


# ── RUN seam state (HANDLE seam a) ──────────────────────────────────────────
# One MaximHandle per server process (the console fronts ONE persistent
# agent), one campaign at a time. The handle is built lazily on the first
# adventure run so `maxim serve` startup stays instant.
# _HANDLE_AGENT_ID is the single source of the console agent's identity —
# get_recall reads the same agent home the handle writes (Exec B1).
_HANDLE_AGENT_ID = "console_agent"
_handle_lock = threading.Lock()
_handle: Any | None = None
_active_run: dict[str, Any] = {"session_id": None, "thread": None}


def _get_handle() -> Any:
    global _handle
    with _handle_lock:
        if _handle is None:
            from maxim.console.handle import MaximHandle

            _handle = MaximHandle(agent_id=_HANDLE_AGENT_ID)
        return _handle


def _run_campaign_thread(handle: Any, campaign_path: str, run_id: str) -> None:
    import logging

    log = logging.getLogger(__name__)
    # EVENT seam: run lifecycle events close the RunAccepted.session_id ≠
    # sim-internal-session-id correlation gap — the binding (and the report
    # path) arrives on the stream at run end. Events emitted by the campaign
    # while this thread runs are stamped with run_id via set_run.
    _event_hub.set_run(run_id)
    _event_hub.publish(
        "run",
        f"run {run_id} started",
        {"run_id": run_id, "status": "started", "mode": "adventure", "campaign": campaign_path},
    )
    try:
        result = handle.play_campaign(campaign_path)
        log.info("console run %s finished: %s", run_id, getattr(result, "finish_reason", "?"))
        # SimulationResult has session_id/session_dir (empty-string defaults),
        # NOT a report_path field — the report convention is
        # session_dir/report.json (review fold: getattr(result, "report_path")
        # was structurally always None, a dead wire field).
        session_dir = str(getattr(result, "session_dir", "") or "")
        _event_hub.publish(
            "run",
            f"run {run_id} ended",
            {
                "run_id": run_id,
                "status": "ended",
                "finish_reason": str(getattr(result, "finish_reason", "") or "") or None,
                "sim_session_id": str(getattr(result, "session_id", "") or "") or None,
                "report_path": str(Path(session_dir) / "report.json") if session_dir else None,
            },
        )
    except Exception as e:
        log.exception("console run %s failed", run_id)
        _event_hub.publish(
            "run",
            f"run {run_id} failed",
            {"run_id": run_id, "status": "failed", "error": f"{type(e).__name__}: {e}"},
        )
    finally:
        # After this, late emissions from the campaign's worker-pool threads
        # arrive with run_id=None (correlation loss only — the "ended" event
        # above is the run's wire boundary).
        _event_hub.set_run(None)


@api.post("/run", response_model=RunAccepted, summary="Run a mode (talk/adventure/sim/rest)")
def post_run(body: RunRequest) -> RunAccepted:
    # HANDLE seam (a): mode="adventure" runs a DM campaign AS the persistent
    # agent (campaign injection — the "Adventure teaches Talk" surface).
    # talk/sim/rest stay 501 until Phase 3 streaming lands.
    if body.mode != "adventure":
        raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)
    if not body.campaign:
        raise HTTPException(status_code=422, detail="mode='adventure' requires 'campaign' (a campaign YAML path).")
    # The console is a 127.0.0.1-only OPERATOR surface: naming a local
    # campaign file here is the same trust level as `maxim --sim <path>` on
    # the CLI (CodeQL flags the request→path flow; it is by-design for a
    # local-first tool). Constrain to what a campaign can be: an existing
    # YAML file, resolved without following into surprises.
    campaign_path = Path(body.campaign).expanduser().resolve()
    if campaign_path.suffix.lower() not in (".yaml", ".yml"):
        raise HTTPException(status_code=422, detail="'campaign' must point at a campaign YAML (.yaml/.yml).")
    if not campaign_path.is_file():
        raise HTTPException(status_code=404, detail=f"Campaign not found: {campaign_path}")

    handle = _get_handle()
    with _handle_lock:
        prev = _active_run["thread"]
        if prev is not None and prev.is_alive():
            return RunAccepted(
                session_id=str(_active_run["session_id"]),
                mode="adventure",
                status="rejected",
                detail="A run is already active on this handle (one campaign at a time).",
            )
        # The sim generates its own internal session_id after boot; this run
        # id is the console-side tracking handle returned at accept time.
        run_id = time.strftime("%Y%m%d_%H%M%S")
        thread = threading.Thread(
            target=_run_campaign_thread,
            args=(handle, str(campaign_path), run_id),
            name=f"console.run.{run_id}",
            daemon=True,
        )
        _active_run["session_id"] = run_id
        _active_run["thread"] = thread
        thread.start()
    return RunAccepted(
        session_id=run_id,
        mode="adventure",
        status="started",
        detail=f"Campaign {campaign_path.name} running as persistent agent {handle.agent_id!r}.",
    )


@api.get(
    "/events/envelope",
    response_model=ConsoleEvent,
    summary="WS event envelope shape (type-gen only; live stream is /ws)",
)
def get_event_envelope() -> ConsoleEvent:
    # OpenAPI does not model WebSocket payloads, so this documents the /ws
    # envelope shape purely so the frontend can generate the ConsoleEvent type.
    raise HTTPException(status_code=501, detail="Envelope shape only — subscribe to /ws for the live stream.")


@api.get(
    "/events/subscribe-frame",
    response_model=SubscribeFrame,
    summary="WS filter frame shape (type-gen only; send frames on /ws)",
)
def get_subscribe_frame() -> SubscribeFrame:
    # Same type-gen trick as /events/envelope: documents the client→server
    # filter frame so the kit generates the SubscribeFrame type.
    raise HTTPException(status_code=501, detail="Frame shape only — send it as JSON on the /ws socket.")


# ── EVENT seam — the /ws bridge (sim_log records → ConsoleEvent stream) ──────
# reachy_app_maxim_seams.md § EVENT. One _EventHub per process: sim_log's
# publishing thread hands records to `sink()` (registered via
# register_sim_sink at app startup), which crosses into the event loop with
# call_soon_threadsafe and fans out to bounded per-connection queues.
# Backpressure: drop-oldest + a "dropped" meta-event; the publishing thread
# NEVER blocks. Filtering is per-connection via SubscribeFrame; meta-kinds
# (heartbeat/run/dropped/display) bypass filters — they carry stream/UI state.

_META_KINDS = frozenset({"heartbeat", "run", "dropped", "display"})
_WS_QUEUE_MAXSIZE = 512
_TIER_ORDER = {"clean": 0, "bio": 1, "debug": 2}


class _WsConn:
    """One /ws connection: bounded queue + compiled filter + monotonic seq.

    All mutable state is touched ONLY on the event loop (enqueue via
    call_soon_threadsafe, filter updates from the receive task, drains from
    the sender loop) — no lock needed.
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_WS_QUEUE_MAXSIZE)
        self._seq = 0
        self.dropped = 0
        self.dropped_reported = 0
        self.warned_unserializable = False
        # Compiled filter (None on every axis = everything).
        self._tier_max: int | None = None
        self._subsystems: set[str] | None = None  # subsystem names, uppercased for matching
        self._kinds: set[str] | None = None  # lowercase kinds

    def apply_frame(self, frame: SubscribeFrame) -> None:
        """Each frame REPLACES the filter (send a full frame; all-None resets)."""
        from maxim.simulation.sim_logger import expand_channels

        self._tier_max = _TIER_ORDER[frame.tier] if frame.tier is not None else None
        # Uppercase for CASE-INSENSITIVE subsystem matching (review finding:
        # the one mixed-case canonical subsystem, "NAc", was unreachable via
        # raw-name channels — "nac"→"NAC" never equaled "NAc").
        self._subsystems = {s.upper() for s in expand_channels(frame.channels)} if frame.channels is not None else None
        self._kinds = {k.strip().lower() for k in frame.kinds} if frame.kinds is not None else None

    def matches(self, kind: str, tier: str, subsystem: str) -> bool:
        if kind in _META_KINDS:
            return True
        if self._tier_max is not None and _TIER_ORDER.get(tier, 1) > self._tier_max:
            return False
        if self._subsystems is not None or self._kinds is not None:
            in_subsystems = self._subsystems is not None and subsystem.upper() in self._subsystems
            in_kinds = self._kinds is not None and kind in self._kinds
            if not (in_subsystems or in_kinds):
                return False
        return True

    def enqueue(self, evt: dict[str, Any]) -> None:
        """Assign seq + put; on full, drop the OLDEST (a seq gap = drops)."""
        evt["seq"] = self._seq
        self._seq += 1
        while True:
            try:
                self.queue.put_nowait(evt)
                return
            except asyncio.QueueFull:
                try:
                    self.queue.get_nowait()
                    self.dropped += 1
                except asyncio.QueueEmpty:  # pragma: no cover — single-threaded on the loop
                    pass


class _EventHub:
    """Process-wide fan-out point between sim_log's thread and /ws clients."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._conns: set[_WsConn] = set()
        self._run_id: str | None = None

    # ── lifecycle (event loop side) ──
    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            if self._loop is not None and self._loop is not loop:
                # Two live apps sharing the process-wide hub would fan out to
                # the first app's queues from the second app's loop —
                # violating _WsConn's loop-only contract. One server per
                # process by design; make the violation loud.
                logger.warning("EventHub.attach: replacing a live loop — two concurrent build_app() servers?")
            self._loop = loop

    def detach(self) -> None:
        with self._lock:
            self._loop = None

    def add_conn(self, conn: _WsConn) -> None:
        with self._lock:
            self._conns.add(conn)

    def remove_conn(self, conn: _WsConn) -> None:
        with self._lock:
            self._conns.discard(conn)

    # ── run correlation (run-thread side) ──
    def set_run(self, run_id: str | None) -> None:
        with self._lock:
            self._run_id = run_id

    # ── producers ──
    def sink(self, record: dict[str, Any]) -> None:
        """sim_log sink — runs on the PUBLISHING thread; enqueue-and-return."""
        from maxim.simulation.sim_logger import subsystem_wire_tier

        with self._lock:
            loop = self._loop
            run_id = self._run_id
            has_conns = bool(self._conns)
        if loop is None or not has_conns or loop.is_closed():
            return
        subsystem = str(record.get("subsystem", ""))
        evt = {
            "kind": subsystem.lower(),
            "tier": subsystem_wire_tier(subsystem),
            "seq": 0,  # per-connection; assigned at enqueue
            "run_id": run_id,
            "ts": time.time(),
            "elapsed_s": record.get("t"),
            "agent_id": record.get("agent_id"),
            "agent": record.get("agent"),
            "message": str(record.get("message", "")),
            # Shallow-copy at bridge time (review finding): the record's data
            # dict is the PRODUCER'S object; sim_log takes it by reference. A
            # producer mutating it after sim_log returns would race
            # send_json's iteration on the loop thread.
            "data": dict(record.get("data") or {}),
        }
        try:
            loop.call_soon_threadsafe(self._fanout, subsystem, evt)
        except RuntimeError:  # loop shut down between the check and the call
            return

    def publish(self, kind: str, message: str, data: dict[str, Any]) -> None:
        """Console-side meta-events (kind='run'), from any thread."""
        with self._lock:
            loop = self._loop
            run_id = self._run_id
        if loop is None or loop.is_closed():
            return
        evt = {
            "kind": kind,
            "tier": "clean",
            "seq": 0,
            "run_id": run_id,
            "ts": time.time(),
            "elapsed_s": None,
            "agent_id": None,
            "agent": None,
            "message": message,
            "data": data,
        }
        try:
            loop.call_soon_threadsafe(self._fanout, "", evt)
        except RuntimeError:
            return

    # ── fan-out (event loop side) ──
    def _fanout(self, subsystem: str, evt: dict[str, Any]) -> None:
        with self._lock:
            conns = list(self._conns)
        for conn in conns:
            if conn.matches(evt["kind"], evt["tier"], subsystem):
                # Copy: seq is per-connection and queues must not share the dict.
                conn.enqueue(dict(evt))


_event_hub = _EventHub()


# ── app factory ─────────────────────────────────────────────────────────────


def build_app(ui_dist: Path | None = None) -> FastAPI:
    """Construct the Console FastAPI app. ``ui_dist`` = the built static bundle."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> Any:
        # EVENT seam: bridge sim_log records into the hub for /ws fan-out.
        from maxim.simulation.sim_logger import register_sim_sink, unregister_sim_sink

        _event_hub.attach(asyncio.get_running_loop())
        register_sim_sink(_event_hub.sink)
        try:
            yield
        finally:
            # finally: an exception through shutdown must not strand the sink
            # registration (review fold).
            unregister_sim_sink(_event_hub.sink)
            _event_hub.detach()
        # Shutdown: server exit mid-campaign would otherwise kill the daemon
        # run thread with the hippocampus capture queue unflushed — silent
        # learning loss. Join the live run (bounded) BEFORE stopping so the
        # campaign's own session-end wins when it can (post-merge review
        # Exec #4); then stop() — idempotent, safe when no adventure ever ran.
        with _handle_lock:
            handle = _handle
            run_thread = _active_run["thread"]

        def _drain_and_stop() -> None:
            if run_thread is not None and run_thread.is_alive():
                run_thread.join(timeout=30.0)
            if handle is not None:
                handle.stop()

        await asyncio.to_thread(_drain_and_stop)

    app = FastAPI(
        title="Maxim Console",
        version="0.1.0",
        summary="Localhost Console backend + the OpenAPI facade contract for maxim-pulse.",
        lifespan=_lifespan,
    )
    app.include_router(api)

    @app.websocket("/ws")
    async def ws_events(websocket: WebSocket) -> None:
        """EventClient stream (EVENT seam) — sim_log records + meta-kinds.

        Server→client: ConsoleEvent envelopes (v2). Client→server: optional
        SubscribeFrame JSON messages; each frame REPLACES the connection's
        filter. Heartbeats fire when the stream is idle; a "dropped" meta-event
        reports backpressure losses (seq gaps mark where).
        """
        await websocket.accept()
        conn = _WsConn()
        _event_hub.add_conn(conn)

        async def _recv_frames() -> None:
            while True:
                try:
                    msg = await websocket.receive_json()
                    frame = SubscribeFrame.model_validate(msg)
                except (ValueError, KeyError):
                    # Malformed frame — keep the current filter. ValueError
                    # covers both json.JSONDecodeError (non-JSON text) and
                    # pydantic ValidationError; KeyError is starlette's raise
                    # for a binary frame. Cross-confirmed review finding: any
                    # of these previously killed the recv task and escaped the
                    # endpoint as an unhandled ASGI exception.
                    continue
                conn.apply_frame(frame)

        recv_task = asyncio.create_task(_recv_frames())
        try:
            while True:
                if recv_task.done():
                    return  # client went away (receive raised/closed) — stop sending
                try:
                    evt = await asyncio.wait_for(conn.queue.get(), timeout=_HEARTBEAT_INTERVAL_S)
                except (asyncio.TimeoutError, TimeoutError):
                    # Idle: enqueue the heartbeat so it flows the one path
                    # (same seq counter — monotonicity holds on the wire).
                    conn.enqueue(
                        ConsoleEvent(kind="heartbeat", tier="clean", seq=0, ts=time.time(), message="").model_dump()
                    )
                    continue
                try:
                    await websocket.send_json(evt)
                except TypeError:
                    # Non-JSON-serializable producer data (review finding):
                    # without this, ONE poison record fans out to every
                    # matching connection and kills them all. Drop the event,
                    # warn once per connection, keep the stream alive.
                    if not conn.warned_unserializable:
                        conn.warned_unserializable = True
                        logger.warning(
                            "dropped non-JSON-serializable event kind=%r on /ws", evt.get("kind"), exc_info=True
                        )
                    continue
                if conn.dropped > conn.dropped_reported:
                    count = conn.dropped - conn.dropped_reported
                    conn.dropped_reported = conn.dropped
                    conn.enqueue(
                        ConsoleEvent(
                            kind="dropped",
                            tier="clean",
                            seq=0,
                            ts=time.time(),
                            message=f"{count} event(s) dropped (slow consumer)",
                            data={"count": count, "total": conn.dropped},
                        ).model_dump()
                    )
        except (WebSocketDisconnect, RuntimeError):
            # RuntimeError: starlette's send-after-close — same meaning here.
            return
        finally:
            _event_hub.remove_conn(conn)
            recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                await recv_task

    # Static Console bundle at "/", or a clear "not installed" page.
    if ui_dist is not None and Path(ui_dist).is_dir():
        app.mount("/", StaticFiles(directory=str(ui_dist), html=True), name="console")
    else:

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        def _no_ui() -> str:
            where = f" (looked in {ui_dist})" if ui_dist else ""
            return (
                "<h1>Maxim Console API is running</h1>"
                f"<p>No Console UI bundle installed{where}. The API + OpenAPI schema are live at "
                "<a href='/docs'>/docs</a> and <a href='/openapi.json'>/openapi.json</a>.</p>"
                "<p>Point at a built bundle with <code>maxim serve --ui-dist &lt;path&gt;</code> "
                "or <code>config.json::console.ui_dist</code>.</p>"
            )

    return app


# ── CLI runner ──────────────────────────────────────────────────────────────


def _resolve(field_path: str, cli_value: Any) -> Any:
    from maxim.runtime.config_loader import resolve_setting

    # resolve_setting returns (value, source); we only need the value here.
    result = resolve_setting(field_path, cli_value=cli_value)
    return result[0] if isinstance(result, tuple) else result


def run_serve(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="maxim serve", description="Run the localhost Maxim Console.")
    ap.add_argument("--port", type=int, default=None, help="Port (default: config console.port / 8765).")
    ap.add_argument("--ui-dist", default=None, help="Path to the built Console static bundle.")
    ap.add_argument(
        "--dump-openapi",
        nargs="?",
        const=str(_OPENAPI_SNAPSHOT),
        default=None,
        metavar="PATH",
        help="Write the OpenAPI schema to PATH (default: the committed snapshot) and exit — no server.",
    )
    args = ap.parse_args(argv)

    if args.dump_openapi is not None:
        out = Path(args.dump_openapi)
        out.write_text(json.dumps(openapi_schema(), indent=2, sort_keys=True) + "\n")
        print(f"wrote OpenAPI schema → {out}")
        return 0

    port = int(_resolve("console.port", args.port))
    ui_dist_val = _resolve("console.ui_dist", args.ui_dist)
    ui_dist = Path(ui_dist_val) if ui_dist_val else None

    app = build_app(ui_dist)

    import uvicorn

    # 127.0.0.1 ONLY — the console holds keys + can run/configure Maxim.
    print(f"maxim serve → http://127.0.0.1:{port}  (API docs: /docs · schema: /openapi.json)")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
    return 0

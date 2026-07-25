"""FastAPI app for ``maxim serve`` — the Console backend + facade contract.

Binds **127.0.0.1 only** (it holds keys and can run/configure Maxim; a hosted
console is an explicit non-goal). Three surfaces:

* ``/api/*`` — JSON facade, 1:1 with ``api.py`` verbs. Existing structured verbs
  (``/api/models``, ``/api/diagnose``) are live; the seams (``/api/probe``,
  ``/api/setup/*``, ``/api/recall``, ``/api/run``) are typed **501 stubs** — their
  Pydantic shapes are in OpenAPI so the maxim-pulse kit can generate the full
  ``FacadeClient`` now (Phase 1 fills in the bodies without touching the schema).
* ``/ws`` — the EventClient stream. Skeleton pushes a typed heartbeat; the full
  ``api.on()`` bridge lands in Phase 3.
* ``/`` — the static Console bundle (resolved from ``--ui-dist`` / config).

**FastAPI is justified specifically by the OpenAPI auto-emit** (the cross-repo
contract); that is why this is not stdlib like ``leader_proxy``.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
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
)

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
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@api.get("/recall", response_model=RecallResponse, summary="What Maxim remembers about you")
def get_recall() -> RecallResponse:
    # RECALL seam: wrap the consumer-shaped api.recall() — a provenance-filtered,
    # salience-ranked, curated blend across pluggable recall sources. The console
    # never touches the bio-stack directly (thin-app rule); it maps the curated
    # dataclass to the wire shape.
    import maxim
    from maxim.console.schemas import Preference, StoryMemory

    r = maxim.recall()
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
_handle_lock = threading.Lock()
_handle: Any | None = None
_active_run: dict[str, Any] = {"session_id": None, "thread": None}


def _get_handle() -> Any:
    global _handle
    with _handle_lock:
        if _handle is None:
            from maxim.console.handle import MaximHandle

            _handle = MaximHandle()
        return _handle


def _run_campaign_thread(handle: Any, campaign_path: str, run_id: str) -> None:
    import logging

    log = logging.getLogger(__name__)
    try:
        result = handle.play_campaign(campaign_path)
        log.info("console run %s finished: %s", run_id, getattr(result, "finish_reason", "?"))
    except Exception:
        log.exception("console run %s failed", run_id)


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


# ── app factory ─────────────────────────────────────────────────────────────


def build_app(ui_dist: Path | None = None) -> FastAPI:
    """Construct the Console FastAPI app. ``ui_dist`` = the built static bundle."""
    app = FastAPI(
        title="Maxim Console",
        version="0.1.0",
        summary="Localhost Console backend + the OpenAPI facade contract for maxim-pulse.",
    )
    app.include_router(api)

    @app.on_event("shutdown")
    def _stop_handle() -> None:
        # Server exit mid-campaign would otherwise kill the daemon run thread
        # with the hippocampus capture queue unflushed — silent learning loss.
        # stop() is idempotent (safe when no adventure ever ran).
        global _handle
        with _handle_lock:
            if _handle is not None:
                _handle.stop()

    @app.websocket("/ws")
    async def ws_events(websocket: WebSocket) -> None:
        """EventClient stream — skeleton heartbeat (full api.on() bridge in Phase 3)."""
        await websocket.accept()
        try:
            while True:
                evt = ConsoleEvent(kind="heartbeat", ts=time.time())
                await websocket.send_json(evt.model_dump())
                await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
        except WebSocketDisconnect:
            return

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

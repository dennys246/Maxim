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
    ProbeRequest,
    ProbeResult,
    RecallResponse,
    RunAccepted,
    RunRequest,
    SetupResult,
)

_HEARTBEAT_INTERVAL_S = 15.0
_NOT_IMPLEMENTED = "Seam not yet implemented (Phase 1) — shape is contract-complete for type-gen."

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
    return ModelsResponse(
        groups={group: [ModelInfoWire(**dataclasses.asdict(m)) for m in members] for group, members in groups.items()}
    )


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
    return DiagnoseResponse(platform=str(getattr(report, "platform", "")), sections=sections)


# ── seam stubs (typed; 501 until Phase 1) ───────────────────────────────────


@api.post("/probe", response_model=ProbeResult, summary="Test a mesh/cloud connection")
def post_probe(body: ProbeRequest) -> ProbeResult:
    # PROBE seam: the canonical peer-probe entry point + the shared classifier —
    # same path `maxim doctor` uses, so the console and doctor agree on verdicts.
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


@api.post("/setup/mesh", response_model=SetupResult, summary="Write mesh (peer→leader) config")
def post_setup_mesh(body: MeshSetupRequest) -> SetupResult:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@api.post("/setup/cloud", response_model=SetupResult, summary="Write cloud provider config")
def post_setup_cloud(body: CloudSetupRequest) -> SetupResult:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@api.get("/recall", response_model=RecallResponse, summary="What Maxim remembers about you")
def get_recall() -> RecallResponse:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@api.post("/run", response_model=RunAccepted, summary="Run a mode (talk/adventure/sim/rest)")
def post_run(body: RunRequest) -> RunAccepted:
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


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

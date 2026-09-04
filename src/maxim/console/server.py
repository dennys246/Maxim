"""FastAPI app for ``maxim serve`` — the Console backend + facade contract.

Binds **127.0.0.1 only** (it holds keys and can run/configure Maxim), and is
**bearer-authenticated, always on, fail-closed** (see the auth block): every
``/api/*`` route, ``/docs``, ``/openapi.json`` and ``/ws`` requires the
console token (``~/.config/maxim/console_token``, printed by ``maxim serve``
as a ``#token=`` URL; ``--show-token`` / ``--rotate-token``). Exempt: the
static UI shell and ``GET /api/hello``. It also carries a browser-relay guard
(see the trust-guard block): every request's Host must be loopback or a
``console.allowed_origins`` host (DNS rebinding), and state-changing requests
plus ``/ws`` upgrades that carry a browser Origin must carry a trusted one
(CSRF) — auth authenticates the CALLER; the guard still constrains what a
page in an authenticated operator's browser can relay. A hosted deployment — one anonymous
visitor per throwaway machine behind an authenticating proxy — keeps both
properties and adds **sandbox mode** (``MAXIM_CONSOLE_SANDBOX=1``, see the
``_refuse_in_sandbox`` block), which closes the surfaces that act on the host
rather than on the agent. Three surfaces:

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
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from maxim.console.schemas import (
    CampaignInfo,
    CampaignsResponse,
    CloudSetupRequest,
    ConsoleEvent,
    DiagnoseResponse,
    DiagnoseSection,
    HelloResponse,
    IdentityResponse,
    MeshSetupRequest,
    ModelInfoWire,
    ModelsResponse,
    PlatformWire,
    ProbeRequest,
    ProbeResult,
    RecallResponse,
    RunAccepted,
    RunRequest,
    SeamStatus,
    SetupResult,
    SubscribeFrame,
)
from maxim.console.ui_bundle import (
    CONSOLE_CONTRACT_VERSION,
    check_ui_contract,
    packaged_ui_dist,
    resolve_ui_dist,
)

logger = logging.getLogger(__name__)

# Set by build_app so /api/identity and the /ws hello can report the bundle
# that is ACTUALLY being served, not one re-derived after the fact.
_SERVED_UI_DIST: tuple[Path | None, str] = (None, "none")

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


# ── sandbox mode ─────────────────────────────────────────────────────────────
#
# `maxim serve` is localhost-only and unauthenticated BY DESIGN: it holds keys
# and can reconfigure Maxim, so the operator's own machine is its trust
# boundary. A hosted sandbox (one anonymous visitor per throwaway machine,
# an authenticating proxy in front) keeps that binding and that absence of
# auth — the proxy owns the edge — but three surfaces are unsafe for a
# stranger even behind a proxy, because they act on the HOST rather than on
# the agent: `/api/probe` with a `url` dials an arbitrary address with an
# arbitrary bearer and echoes latency + detail (server-side request forgery),
# `/api/setup/mesh` repoints the LLM lane at a caller-controlled URL and
# PERSISTS it, and `/api/diagnose` renders the resolved configuration —
# key BYTES are redacted (`doctor/checks.py::_is_secret_field`), but paths,
# env names, IPs and fix hints are a recon surface.
# Sandbox mode closes exactly those, refuses `/ws` upgrades from origins the
# operator did not list, and caps the size of a run input. It is the
# `console.sandbox` / `console.allowed_origins` / `console.max_input_chars`
# config keys (env forms `MAXIM_CONSOLE_SANDBOX` etc., resolved by the loader
# like every other console setting), read ONCE at `build_app` and carried on
# `app.state.sandbox` — a request cannot flip it. Sandbox mode is ALSO the one
# state where the engine's bearer auth (auth block below) is OFF: a sandbox
# visitor is anonymous BY DESIGN and the authenticating proxy in front owns
# the edge, so demanding a token would only relocate the proxy's job into the
# engine. Corollary (audit C3): under sandbox, closing or brokering
# `setup/cloud` — deliberately half-open here so the BYO-key wizard works —
# remains the BROKER's responsibility, not this module's.
_SANDBOX_ENV = "MAXIM_CONSOLE_SANDBOX"
_SANDBOX_ORIGINS_ENV = "MAXIM_CONSOLE_ALLOWED_ORIGINS"
_SANDBOX_MAX_INPUT_ENV = "MAXIM_CONSOLE_MAX_INPUT_CHARS"
_SANDBOX_DEFAULT_MAX_INPUT_CHARS = 16_000
_SANDBOX_REFUSAL = (
    "Closed in sandbox mode: this surface acts on the host (network or persisted "
    "configuration) rather than on the agent, so it is not offered to sandbox visitors."
)


@dataclasses.dataclass(frozen=True)
class _SandboxPolicy:
    allowed_origins: frozenset[str]
    max_input_chars: int


_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def _canonical_origin(raw: str) -> str | None:
    """Canonical ``scheme://host[:port]`` for an Origin-shaped string, or None.

    Lowercased, trailing ``/`` stripped, and a DEFAULT port (``:80`` http/ws,
    ``:443`` https/wss) dropped — browsers omit default ports from the Origin
    header, so a listed ``https://x:443`` would otherwise never match the
    ``https://x`` a browser actually sends. Anything that is not a bare
    origin — no scheme (a bare hostname), a path/query/fragment, credentials,
    an unparsable port — returns None: it could never equal a browser Origin,
    so treating it as valid would silently half-apply (review fold; the
    scheme is NOT restricted to http(s) — a packaged native shell may send
    e.g. ``capacitor://localhost``).
    """
    s = (raw or "").strip().rstrip("/").lower()
    if not s:
        return None
    try:
        parts = urlsplit(s)
        port = parts.port  # raises ValueError on a malformed port
    except ValueError:
        return None
    host = parts.hostname
    if not parts.scheme or not host or parts.path or parts.query or parts.fragment or "@" in parts.netloc:
        return None
    if port is not None and port == _DEFAULT_PORTS.get(parts.scheme):
        port = None
    hostpart = f"[{host}]" if ":" in host else host  # re-bracket IPv6
    return f"{parts.scheme}://{hostpart}" + (f":{port}" if port is not None else "")


def _canonical_origin_list(raw_entries: Any) -> frozenset[str]:
    """Canonicalize ``console.allowed_origins`` entries, LOUDLY refusing junk.

    A malformed entry (bare hostname, path, bad port) can never match a
    browser Origin, and its host may or may not derive — the half-applied
    state is the hardest to debug (Host guard relaxed, Origin guard still
    refusing the operator's own UI). Same philosophy as the sandbox cap
    below: an info log is not a signal anyone reads — fail at build time.
    """
    canonical: set[str] = set()
    for raw in raw_entries or ():
        if not raw:
            continue
        c = _canonical_origin(raw)
        if c is None:
            from maxim.runtime.config_loader import ConfigurationError

            raise ConfigurationError(
                f"config: console.allowed_origins entry {raw!r} is not an origin — expected "
                f"scheme://host[:port], no path (env: {_SANDBOX_ORIGINS_ENV})"
            )
        canonical.add(c)
    return frozenset(canonical)


def _sandbox_policy() -> _SandboxPolicy | None:
    """Resolve the sandbox switch and its two knobs through the config loader.

    Returns ``None`` (the default, byte-identical localhost behaviour) unless
    ``console.sandbox`` resolves true. Malformed values are the loader's
    ``ConfigurationError`` — raised at build time, never a silent fallback.
    """
    if not _resolve("console.sandbox", None):
        return None
    origins = _canonical_origin_list(_resolve("console.allowed_origins", None))
    if not origins:
        # Loud, like a bad cap: a sandbox whose UI can never open /ws is the
        # vacuous guard inverted (everything refused), and an info log is
        # not a signal anyone reads.
        from maxim.runtime.config_loader import ConfigurationError

        raise ConfigurationError(
            "config: console.allowed_origins must list at least one origin when console.sandbox is on "
            f"(env: {_SANDBOX_ORIGINS_ENV})"
        )
    cap = int(_resolve("console.max_input_chars", None) or _SANDBOX_DEFAULT_MAX_INPUT_CHARS)
    return _SandboxPolicy(allowed_origins=origins, max_input_chars=cap)


def _sandbox_of(app: Any) -> _SandboxPolicy | None:
    return getattr(app.state, "sandbox", None)


def _refuse_in_sandbox(request: Request) -> None:
    """FastAPI dependency: 403 on the host-acting surfaces when sandboxed."""
    if _sandbox_of(request.app) is not None:
        raise HTTPException(status_code=403, detail=_SANDBOX_REFUSAL)


# ── trust guard (Host / Origin) ──────────────────────────────────────────────
#
# 127.0.0.1-only is a statement about who can CONNECT, not about who can make
# the operator's browser connect FOR them. Two browser-relayed attack classes
# reach a loopback bind: a page in the operator's browser firing cross-origin
# "simple" POSTs at localhost (CSRF — Starlette parses a JSON body regardless
# of Content-Type, so no preflight protects the mutating routes), and a page
# on an attacker DNS name that re-resolves to 127.0.0.1 (DNS rebinding), which
# sidesteps same-origin entirely and arrives with Host: attacker.example.
# This guard closes both while adding NO authentication:
#   * every request's Host must be a loopback name or a host drawn from
#     console.allowed_origins — a rebinding name fails this;
#   * a state-changing request that CARRIES a browser Origin must carry a
#     loopback origin or one listed in console.allowed_origins — a CSRF page
#     fails this. Requests without an Origin (curl, the CLI, native clients)
#     pass: this is browser-relay protection, not authentication.
# `console.allowed_origins` thereby graduates from a sandbox-only knob to the
# general "non-local origins I trust" list; deliberate remote exposure (the
# tunnel) lists its public origin here. Bearer auth (the auth block below)
# authenticates the CALLER; this guard still constrains what a page in an
# authenticated operator's own browser can be made to relay — defense in
# depth, not redundancy (docs/plans/console_tunnel_hardening.md, decision
# A8). Resolved ONCE at build_app and carried on ``app.state.trust`` — a
# request cannot flip it.

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
# Starlette's TestClient default base_url is http://testserver — allowed ONLY
# while pytest is running (PYTEST_CURRENT_TEST is set by the runner for the
# duration of each test), mirroring how Django injects "testserver" in
# setup_test_environment() rather than production ALLOWED_HOSTS. Never in
# production: "single-label names aren't on public DNS" does not hold against
# a hostile LAN resolver (rogue Wi-Fi DHCP), which could rebind it (review
# fold — both lenses).
_TEST_HOSTS = frozenset({"testserver"})
_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_HOST_REFUSAL = (
    "Refused: unrecognized Host header (DNS-rebinding guard). The console answers loopback "
    "hosts and the hosts of console.allowed_origins only."
)
_ORIGIN_REFUSAL = (
    "Refused: this browser origin may not make state-changing console requests (cross-site "
    "request guard). Loopback origins are trusted; list others in console.allowed_origins."
)
_CONTENT_TYPE_REFUSAL = (
    "Refused: state-changing console requests must send Content-Type: application/json "
    "(cross-site form guard — HTML forms cannot send JSON, so this forces a preflight)."
)


def _test_host_allowance() -> frozenset[str]:
    import os

    return _TEST_HOSTS if "PYTEST_CURRENT_TEST" in os.environ else frozenset()


@dataclasses.dataclass(frozen=True)
class _TrustPolicy:
    allowed_origins: frozenset[str]  # normalized: scheme://host[:port], lowercase, no trailing /
    allowed_hosts: frozenset[str]  # hostnames of those origins (no port, no brackets)


_EMPTY_TRUST = _TrustPolicy(allowed_origins=frozenset(), allowed_hosts=frozenset())


def _trust_policy() -> _TrustPolicy:
    """Resolve console.allowed_origins into the trust policy (loopback-plus list).

    Entries canonicalize LOUDLY (`_canonical_origin_list` raises on junk), so
    every origin here parses; its hostname is what the Host guard additionally
    trusts — listing an origin whitelists its Host too (the tunnel's public
    hostname must therefore appear as the host of some listed origin).
    """
    origins = _canonical_origin_list(_resolve("console.allowed_origins", None))
    hosts = {h for origin in origins if (h := urlsplit(origin).hostname)}
    return _TrustPolicy(allowed_origins=origins, allowed_hosts=frozenset(hosts))


def _trust_of(app: Any) -> _TrustPolicy:
    return getattr(app.state, "trust", None) or _EMPTY_TRUST


def _request_host(host_header: str) -> str:
    """The bare hostname of a Host header — port stripped, IPv6 unbracketed,
    trailing FQDN dot dropped (``http://localhost./`` resolves to loopback)."""
    h = (host_header or "").strip().lower()
    if h.startswith("["):  # [::1]:8765 / [::1]
        return h[1 : h.index("]")] if "]" in h else h.lstrip("[")
    return (h.rsplit(":", 1)[0] if ":" in h else h).rstrip(".")


def _host_allowed(host_header: str, trust: _TrustPolicy) -> bool:
    host = _request_host(host_header)
    return bool(host) and (host in _LOOPBACK_HOSTS or host in trust.allowed_hosts or host in _test_host_allowance())


def _origin_allowed(origin_header: str, trust: _TrustPolicy) -> bool:
    """Is this browser Origin trusted for state-changing requests / /ws?

    Loopback-host origins pass on ANY port (a local dev UI on :5173 talking to
    :8765 is the operator's own machine — already inside the trust boundary);
    everything else must be listed. An unparseable or "null" origin fails.
    """
    origin = _canonical_origin(origin_header)
    if origin is None:
        return False
    host = (urlsplit(origin).hostname or "").lower()
    return host in _LOOPBACK_HOSTS or origin in trust.allowed_origins


# ── bearer auth (always on, fail-closed; sandbox is the one exception) ───────
#
# docs/plans/console_tunnel_hardening.md, decisions A1–A8. The console token
# (`tunnel/keys.py::ensure_console_token`, an mxc_-prefixed 256-bit secret in
# ~/.config/maxim/console_token) authenticates every /api/* route, /docs,
# /openapi.json and /ws. There is deliberately NO off toggle — a default-off
# knob is how the critical endpoints stay reachable — and a missing/unreadable
# token FAILS CLOSED (every authed surface refuses), explicitly inverting
# `leader_proxy._check_auth`'s empty-key fail-open. Exempt: the static UI
# shell (the public pulse bundle, no data) and GET /api/hello (contract
# version + auth scheme, so a client can detect skew and render a login screen
# BEFORE it holds a token). Transports: `Authorization: Bearer <token>`
# (scheme parsed before credential, per the CC13 branch-table pattern), or —
# for browsers, which cannot set headers on a WebSocket upgrade — the token
# rides `Sec-WebSocket-Protocol: maxim.bearer.<token>` beside the app
# subprotocol `maxim-console-v1`, validated BEFORE accept (websocket scope
# only; never consulted for plain HTTP). Query-param tokens are refused BY
# OMISSION everywhere: URLs reach access logs. Under sandbox mode auth is
# OFF — the authenticating proxy owns that edge (see the sandbox block).

_AUTH_EXEMPT_PATHS = frozenset({"/api/hello"})
_WS_APP_SUBPROTOCOL = "maxim-console-v1"
_WS_BEARER_PREFIX = "maxim.bearer."
_AUTH_REFUSAL = (
    "Refused: this console surface requires the console token — send "
    "'Authorization: Bearer <token>' (browser /ws: offer the "
    f"'{_WS_BEARER_PREFIX}<token>' subprotocol). The operator prints it with "
    "`maxim serve --show-token`."
)
# Sentinel default for build_app's keyword-only auth_token: read the on-disk
# console_token. Distinct from None, which is an explicit fail-closed inject.
_READ_TOKEN_FROM_DISK = object()


def _read_console_token() -> str | None:
    from maxim.tunnel.keys import read_console_token

    return read_console_token()


def _auth_required(path: str, scope_type: str) -> bool:
    """Which surfaces demand the token: /ws and everything routed, minus hello.

    The static bundle (everything that is not /api, /ws or the schema surface)
    stays public — it is the same bundle maxim-pulse publishes, holds no data,
    and must be able to render the login screen for a tokenless visitor.
    """
    if scope_type == "websocket":
        return True
    if path in _AUTH_EXEMPT_PATHS:
        return False
    return path.startswith("/api/") or path in ("/docs", "/openapi.json", "/redoc") or path.startswith("/docs/")


def _bearer_authorized(headers: dict[str, str], token: str | None, scope_type: str) -> bool:
    """Constant-time bearer check over either transport; fail closed on no token."""
    if not token:
        return False  # fail CLOSED — the anti-leader_proxy trap (design A2)
    auth = headers.get("authorization", "")
    if auth:
        parts = auth.split(None, 1)
        # Scheme parsed BEFORE the credential (CC13 pattern): an unknown
        # scheme is a clean refusal, never mistaken for a malformed Bearer.
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return secrets.compare_digest(parts[1].strip().encode(), token.encode())
        return False
    if scope_type == "websocket":
        offered = headers.get("sec-websocket-protocol", "")
        for proto in (p.strip() for p in offered.split(",")):
            if proto.startswith(_WS_BEARER_PREFIX):
                return secrets.compare_digest(proto[len(_WS_BEARER_PREFIX) :].encode(), token.encode())
    return False


class _GuardMiddleware:
    """Pure-ASGI guard covering BOTH http and websocket scopes (design A3).

    One middleware instead of an http-only decorator plus hand-applied /ws
    checks, so a future second websocket endpoint cannot silently miss a rule.
    Order per request: Host (rebinding) → bearer auth (fail-closed; skipped
    under sandbox) → Origin / Content-Type (the CSRF belts — kept even though
    bearer-in-header is CSRF-immune, per decision A8). Refusals never reach
    the router: HTTP gets a JSON body, websockets a 1008 close before accept.
    Policy comes from ``scope["app"].state`` (trust/sandbox/auth_token), all
    resolved once at ``build_app`` — a request cannot flip any of it.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        fastapi_app = scope["app"]
        trust = _trust_of(fastapi_app)
        sandbox = _sandbox_of(fastapi_app)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or ()}
        path = str(scope.get("path", ""))

        if not _host_allowed(headers.get("host", ""), trust):
            logger.warning("refusing %s with unrecognized Host %r", path, headers.get("host"))
            await self._refuse(scope, receive, send, 400, _HOST_REFUSAL)
            return

        if sandbox is None and _auth_required(path, scope["type"]):
            if getattr(fastapi_app.state, "auth_token_source", "static") == "disk":
                token = _read_console_token()  # per-request: rotation bites immediately
            else:
                token = getattr(fastapi_app.state, "auth_token", None)
            if not _bearer_authorized(headers, token, scope["type"]):
                # The token itself is never logged — not even truncated.
                logger.warning("refusing unauthenticated %s %s", scope.get("method", "WS"), path)
                await self._refuse(scope, receive, send, 401, _AUTH_REFUSAL)
                return

        if scope["type"] == "websocket":
            # Non-sandbox Origin rule for /ws; sandbox's stricter
            # origin-REQUIRED check stays in ws_events beside its policy.
            origin = headers.get("origin")
            if sandbox is None and origin is not None and not _origin_allowed(origin, trust):
                logger.warning("refusing /ws upgrade from origin %r", origin)
                await self._refuse(scope, receive, send, 403, _ORIGIN_REFUSAL)
                return
        elif scope.get("method") in _UNSAFE_METHODS:
            origin = headers.get("origin")
            if origin is not None and not _origin_allowed(origin, trust):
                logger.warning("refusing %s %s from origin %r", scope["method"], path, origin)
                await self._refuse(scope, receive, send, 403, _ORIGIN_REFUSAL)
                return
            ctype = headers.get("content-type", "")
            if not ctype.lower().strip().startswith("application/json"):
                logger.warning("refusing %s %s with Content-Type %r", scope["method"], path, ctype)
                await self._refuse(scope, receive, send, 415, _CONTENT_TYPE_REFUSAL)
                return

        await self.app(scope, receive, send)

    async def _refuse(self, scope: dict[str, Any], receive: Any, send: Any, status: int, detail: str) -> None:
        if scope["type"] == "websocket":
            # Consume the connect event, then deny the handshake pre-accept.
            message = await receive()
            if message["type"] == "websocket.connect":
                await send({"type": "websocket.close", "code": 1008})  # policy violation
            return
        headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
        response = JSONResponse(status_code=status, content={"detail": detail}, headers=headers)
        await response(scope, receive, send)


# ── live verbs (wrap existing api.py facades) ───────────────────────────────


@api.get("/hello", response_model=HelloResponse, summary="Contract + auth scheme (no token needed)")
def get_hello(request: Request) -> HelloResponse:
    # The ONE unauthenticated API surface (design A6): just enough for a
    # client to detect contract skew and render the right login screen before
    # it holds a token. Everything else identity reports stays behind auth.
    return HelloResponse(
        contract_version=CONSOLE_CONTRACT_VERSION,
        auth="none" if _sandbox_of(request.app) is not None else "bearer",
    )


@api.get("/models", response_model=ModelsResponse, summary="List LLM profiles")
def get_models() -> ModelsResponse:
    import maxim

    groups = maxim.list_models()  # dict[str, list[ModelInfo]]

    def _wire(m: Any) -> ModelInfoWire:
        d = dataclasses.asdict(m)
        return ModelInfoWire(**d, curated=d.get("name") in _CURATED_PROFILES)

    return ModelsResponse(groups={group: [_wire(m) for m in members] for group, members in groups.items()})


# A hand-maintained MANIFEST of console surfaces — deliberately NOT a probe.
# Nothing here is measured: `live` is declared. It is named `_SEAM_DECLARATIONS`
# rather than `_SEAM_PROBES` because shipping a hardcoded liveness claim under a
# probing name is the same "instrument that asserts instead of measuring"
# failure this branch exists to fix. Kept next to the endpoint it feeds so
# adding a seam without declaring it is visible in review; the `sim` 501 is
# pinned by a test, which is what actually keeps this honest.
_SEAM_DECLARATIONS: tuple[tuple[str, str], ...] = (
    ("probe", "/api/probe"),
    ("setup", "/api/setup/mesh + /api/setup/cloud"),
    ("recall", "/api/recall"),
    ("campaigns", "/api/campaigns"),
    ("event", "/ws"),
    ("talk", "/api/run mode=talk"),
    ("adventure", "/api/run mode=adventure"),
    ("rest", "/api/run mode=rest"),
    ("sim", "/api/run mode=sim"),
)


def _git_identity() -> tuple[str | None, str | None]:
    """(sha, branch) of the checked-out source, or (None, None).

    Load-bearing for an EDITABLE install: `maxim serve` follows the working
    tree, so the branch IS the capability set. Best-effort — a wheel install
    has no repo and simply reports None.
    """
    import subprocess

    root = Path(__file__).resolve().parents[3]
    # Only trust a real checkout. For a pip install this path is
    # .../lib/python3.12 and `git` walks ANCESTORS, so a venv inside the
    # user's project reported THEIR branch as pymaxim's identity — strictly
    # worse than None, in the one field the docstring calls load-bearing.
    if not (root / ".git").exists():
        _GIT_IDENTITY_CACHE = (None, None)
        return _GIT_IDENTITY_CACHE

    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=2.0)
            return out.stdout.strip() or None if out.returncode == 0 else None
        except Exception:
            return None

    return _run(["git", "rev-parse", "--short", "HEAD"]), _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])


def build_identity(
    ui_dist: Path | str | None = None, ui_source: str = "none", *, sandbox: _SandboxPolicy | None = None
) -> IdentityResponse:
    """Assemble the backend's self-description (shared by /api/identity + /ws)."""
    import sys as _sys

    import maxim as _maxim
    from maxim.console.ui_bundle import CONSOLE_CONTRACT_VERSION, read_ui_manifest

    sha, branch = _git_identity()
    # Sandbox mode closes the host-acting seams (see `_refuse_in_sandbox`);
    # identity says so rather than declaring them live and letting the client
    # discover the 403. Both are HALF-closed — `probe`'s provider form and
    # `setup`'s cloud half stay open so the cloud wizard still works — which
    # the detail spells out; a consumer gating the cloud wizard on these two
    # seams would be wrong, and `live` here is value-level (no contract change).
    _sandbox_closed = {"probe": "url form closed in sandbox mode", "setup": "mesh closed in sandbox mode"}

    def _seam(n: str, d: str) -> SeamStatus:
        if n == "sim":
            return SeamStatus(name=n, live=False, detail=f"{d} — use the CLI (`maxim --sim`)")
        if sandbox is not None and n in _sandbox_closed:
            return SeamStatus(name=n, live=False, detail=f"{d} — {_sandbox_closed[n]}")
        return SeamStatus(name=n, live=True, detail=d)

    seams = [_seam(n, d) for n, d in _SEAM_DECLARATIONS]
    return IdentityResponse(
        package_version=getattr(_maxim, "__version__", "unknown"),
        contract_version=CONSOLE_CONTRACT_VERSION,
        git_sha=sha,
        git_branch=branch,
        python_version=f"{_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}",
        ui_source=ui_source,  # type: ignore[arg-type]
        ui_dist=str(ui_dist) if ui_dist else None,
        ui_manifest=(read_ui_manifest(ui_dist) or {}) if ui_dist else {},
        seams=seams,
    )


@api.get("/identity", response_model=IdentityResponse, summary="Which backend is this?")
def get_identity(request: Request) -> IdentityResponse:
    # Answers the question a debugging session would otherwise have to guess:
    # which pymaxim, which branch, which contract, which seams are live, and
    # which UI bundle is being served.
    return build_identity(_SERVED_UI_DIST[0], _SERVED_UI_DIST[1], sandbox=_sandbox_of(request.app))


@api.get(
    "/diagnose",
    response_model=DiagnoseResponse,
    summary="Environment diagnostics",
    dependencies=[Depends(_refuse_in_sandbox)],
)
def get_diagnose() -> DiagnoseResponse:
    import maxim

    report = maxim.diagnose()
    sections: list[DiagnoseSection] = []
    # `report.sections` is a list of (group_name, [CheckResult, ...]) TUPLES —
    # not dataclasses. The previous mapping treated each entry as a dataclass /
    # dict, so `.get("name")` always missed: the console rendered one blank row
    # per GROUP and dropped every actual check. Flatten to one row per check,
    # which is what a traffic-light view wants, and keep the group + fix hint.
    for entry in getattr(report, "sections", []) or []:
        group, checks = ("", entry)
        if isinstance(entry, tuple) and len(entry) == 2:
            group, checks = entry
        if not isinstance(checks, (list, tuple)):
            checks = [checks]
        for check in checks:
            if dataclasses.is_dataclass(check) and not isinstance(check, type):
                d = dataclasses.asdict(check)
            elif isinstance(check, dict):
                d = dict(check)
            else:
                d = {"name": str(check)}
            extra: dict[str, Any] = {"group": str(group)}
            if d.get("fix"):
                extra["fix"] = d["fix"]
            if d.get("retry_id"):
                extra["retry_id"] = d["retry_id"]
            sections.append(
                DiagnoseSection(
                    name=str(d.get("name", "") or ""),
                    status=str(d.get("status", "") or ""),
                    detail=d.get("message") or d.get("detail"),
                    extra=extra,
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
def post_probe(body: ProbeRequest, request: Request) -> ProbeResult:
    # PROBE seam. Dispatches on the request shape (contract fix): a MESH probe
    # (``url`` present) goes through the canonical peer-probe entry point + the
    # shared classifier — the same path `maxim doctor` uses, so console and
    # doctor agree on verdicts. A CLOUD probe (``provider`` present, no ``url``)
    # is a cheap pre-save key check.
    if body.url:
        # The URL form is the SSRF surface (arbitrary host + bearer, latency
        # echoed back); the provider form below is a local key-shape check and
        # stays open so the cloud wizard still works for a sandbox visitor.
        _refuse_in_sandbox(request)
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


@api.post(
    "/setup/mesh",
    response_model=SetupResult,
    summary="Write mesh (peer→leader) config",
    dependencies=[Depends(_refuse_in_sandbox)],
)
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
        # The ref PATH is deliberately not echoed: the client needs to know the
        # key is a 0600 ref, not where on the host it lives.
        detail=f"Wrote peer→leader config to {written}; key stored as a 0600 ref.",
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
        detail=f"Wrote {body.provider}/{body.profile} cloud config to {written}; key stored as a 0600 ref.",
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
        agent_id = _handle.agent_id if _handle is not None else _console_agent_id()
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
# `_console_agent_id()` is the single source of the console agent's identity —
# get_recall reads the same agent home the handle writes (Exec B1). It is
# resolved LAZILY (config key `console.agent_id` / `MAXIM_CONSOLE_AGENT_ID`,
# default "console_agent") so `build_app()` embedders and tests get the
# configured value too, not only `run_serve`. A sandbox points it at a
# pre-seeded home without copying that home into the default slot.
_DEFAULT_HANDLE_AGENT_ID = "console_agent"
_handle_lock = threading.Lock()
_handle: Any | None = None
_active_run: dict[str, Any] = {"session_id": None, "thread": None}


def _console_agent_id() -> str:
    """The configured `console.agent_id`, or the default.

    Validation (single path segment, not the reserved sim AUT id) is the
    loader's — `config_loader.coerce_agent_id` — so `maxim config set`, the
    env form and this resolution all refuse the same values; a bad value
    surfaces as a `ConfigurationError` at `maxim serve` start (see
    `run_serve`), never as a 500 on the first Talk request.
    """
    raw = _resolve("console.agent_id", None)
    return str(raw).strip() if raw is not None and str(raw).strip() else _DEFAULT_HANDLE_AGENT_ID


def _get_handle() -> Any:
    global _handle
    with _handle_lock:
        if _handle is None:
            from maxim.console.handle import MaximHandle

            _handle = MaximHandle(agent_id=_console_agent_id())
        return _handle


def campaign_search_roots() -> list[tuple[Path, str]]:
    """Where ``GET /api/campaigns`` looks, in precedence order, with each
    root's ``source`` label.

    ``~/.maxim/campaigns/`` is the durable user location; ``./scenarios/
    campaigns/`` is the dev-checkout convenience (CWD-relative, the same
    convention ``api.benchmark`` documents per CC10). Nothing is bundled in
    the wheel today, so a pip-install user sees only their own campaigns.

    Returns (root, source) PAIRS rather than a bare list: a parallel label
    tuple at the call site silently truncated if a third root were ever added
    — that root would be neither searched nor reported (review finding).
    """
    from maxim.utils.paths import data_home

    return [(data_home() / "campaigns", "user"), (Path("scenarios") / "campaigns", "repo")]


def _is_within_search_root(path: Path) -> bool:
    """Is ``path`` inside one of the discovery roots?

    Discovery only ever yields ``iterdir()`` results, so this holds by
    construction today — making it explicit keeps the file-read below safe
    for ANY caller (and satisfies the path-injection analysis on a read
    reachable from an HTTP endpoint) rather than relying on that invariant
    living in the caller.
    """
    for root, _source in campaign_search_roots():
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except (ValueError, OSError):
            continue
    return False


def _campaign_info(path: Path, source: str) -> CampaignInfo:
    """Read display metadata from a campaign YAML's ``campaign:`` head.

    Best-effort: a campaign that fails to parse still lists (by filename) so
    the picker shows it and the RUN call surfaces the real validation error —
    silently hiding a malformed campaign would be the confusing outcome.
    """
    name, goal = path.stem, None
    if not _is_within_search_root(path):
        return CampaignInfo(name=name, path=str(path), goal=None, source=source)  # type: ignore[arg-type]
    try:
        import yaml

        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        head = doc.get("campaign") or {}
        if isinstance(head, dict):
            name = str(head.get("name") or path.stem)
            goal = str(head.get("goal")) if head.get("goal") else None
    except Exception:
        logger.debug("campaign listing: could not parse %s", path, exc_info=True)
    return CampaignInfo(name=name, path=str(path), goal=goal, source=source)  # type: ignore[arg-type]


@api.get("/campaigns", response_model=CampaignsResponse, summary="Discoverable campaign YAMLs")
def get_campaigns() -> CampaignsResponse:
    # Backs the launcher's picker dropdown so the operator stops pasting
    # absolute YAML paths. Returns the path each entry should be RUN with.
    seen: set[Path] = set()
    out: list[CampaignInfo] = []
    searched: list[str] = []
    for root, source in campaign_search_roots():
        searched.append(str(root))
        if not root.is_dir():
            continue
        try:
            entries = sorted(root.iterdir())
        except OSError:
            # An unreadable root (perms, removed mid-listing) must not 500 the
            # whole picker — report it as searched and move on (review finding).
            logger.warning("campaign listing: could not read %s", root, exc_info=True)
            continue
        for p in entries:
            if p.suffix.lower() not in (".yaml", ".yml") or not p.is_file():
                continue
            resolved = p.resolve()
            if resolved in seen:  # user dir wins over a repo file of the same target
                continue
            seen.add(resolved)
            out.append(_campaign_info(resolved, source))
    return CampaignsResponse(campaigns=out, searched=searched)


def _select_discovered_campaign(requested: str) -> Path | None:
    """Map a requested campaign to a DISCOVERY-DERIVED path, or ``None``.

    The returned Path is built by discovery (``iterdir`` under a known root),
    never constructed from request data — so no request-controlled string ever
    reaches a path expression. A request may name either the full path exactly
    as ``/api/campaigns`` reported it, or the campaign's display name / file
    stem (convenient for humans and CLI callers).
    """
    if not requested:
        return None
    # Normalize the request to a comparable STRING (expanduser + abspath are
    # pure string/env operations — the result is only ever compared, never
    # opened or used as a path). This keeps a hand-typed "~/..." or a
    # CWD-relative path working, which the launcher's free-text box still
    # emits today, without letting request data reach a path expression.
    import os

    try:
        normalized = os.path.abspath(os.path.expanduser(requested))
    except (OSError, ValueError):
        normalized = requested

    listing = get_campaigns()
    for info in listing.campaigns:
        stem = Path(info.path).stem
        if requested in (info.path, info.name, stem) or normalized == info.path:
            return Path(info.path)
    return None


def _run_campaign_thread(handle: Any, campaign_path: str, run_id: str, premise: str | None = None) -> None:
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
        {
            "run_id": run_id,
            "status": "started",
            "mode": "adventure",
            "campaign": campaign_path,
            "premise": premise,
        },
    )
    try:
        result = handle.play_premise(premise) if premise else handle.play_campaign(campaign_path)
        from maxim.simulation.sim_types import is_simulation_run_failure

        finish_reason = str(getattr(result, "finish_reason", "") or "")
        run_failed = is_simulation_run_failure(finish_reason)
        event_status = "failed" if run_failed else "ended"
        log_fn = log.error if run_failed else log.info
        log_fn("console run %s %s: %s", run_id, event_status, finish_reason or "?")
        # SimulationResult has session_id/session_dir (empty-string defaults),
        # NOT a report_path field — the report convention is
        # session_dir/report.json (review fold: getattr(result, "report_path")
        # was structurally always None, a dead wire field).
        session_dir = str(getattr(result, "session_dir", "") or "")
        _event_hub.publish(
            "run",
            f"run {run_id} {event_status}",
            {
                "run_id": run_id,
                "status": event_status,
                "finish_reason": finish_reason or None,
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


# Talk turns are serialized per process: the live loop is one agent, and two
# concurrent utterances would interleave into one settle window.
_talk_lock = threading.Lock()
_TALK_RUN_ID = "talk"
_REST_RUN_ID = "rest"


def _post_run_talk(body: RunRequest) -> RunAccepted:
    """HANDLE talk mode — one conversational turn against the live loop.

    Blocking by design: the turn's REPLY travels on ``/ws`` as CLEAN-tier
    records (``user`` then ``response``), so the chat surface renders from the
    stream; this response carries only the accept/reject + the run id used to
    scope those events. Events emitted during the turn are stamped
    ``run_id="talk"`` so a background adventure's narration cannot interleave
    into a conversation.
    """
    text = (body.input or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="mode='talk' requires 'input' (what to say).")

    with _handle_lock:
        prev = _active_run["thread"]
    if prev is not None and prev.is_alive():
        raise HTTPException(
            status_code=409,
            detail="An adventure is running — talk is unavailable until it ends.",
        )
    if not _talk_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A talk turn is already in flight.")
    turn_id = f"talk_{time.strftime('%Y%m%d_%H%M%S')}"
    try:
        handle = _get_handle()
        prev_run = _event_hub.set_run(_TALK_RUN_ID)
        try:
            result = handle.talk(text)
        finally:
            _event_hub.set_run(prev_run)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("talk turn failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e
    finally:
        _talk_lock.release()

    timed_out = bool(result.get("timed_out"))
    return RunAccepted(
        # Per-turn id so a client can correlate a turn with its event burst;
        # events during the turn carry run_id=_TALK_RUN_ID (the scope), while
        # this identifies the turn itself.
        session_id=turn_id,
        mode="talk",
        status="completed",
        reply=result.get("response"),
        detail=(
            "Turn timed out before a reply — the response may still arrive on /ws."
            if timed_out
            else f"Turn complete ({len(result.get('actions') or [])} action(s)); also on /ws as kind='response'."
        ),
    )


def _post_run_rest() -> RunAccepted:
    """HANDLE rest mode — consolidate memory without tearing the agent down.

    Blocking and usually quick, but a large store can take a while, so the
    counts come back in `detail` rather than as a bare "ok". Distinct from
    `stop()`: the agent stays usable afterwards, and the next talk turn sees
    the consolidated substrate.
    """
    with _handle_lock:
        prev = _active_run["thread"]
    if prev is not None and prev.is_alive():
        raise HTTPException(status_code=409, detail="An adventure is running — rest is unavailable until it ends.")
    if _talk_lock.locked():
        raise HTTPException(status_code=409, detail="A talk turn is in flight — try again in a moment.")

    prev_run = _event_hub.set_run(_REST_RUN_ID)
    try:
        results = _get_handle().rest()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("rest failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e
    finally:
        _event_hub.set_run(prev_run)

    detail = (
        "Consolidated: " + ", ".join(f"{k}={v}" for k, v in sorted(results.items()))
        if results
        else "Nothing to consolidate (no memory store wired)."
    )
    return RunAccepted(session_id=_REST_RUN_ID, mode="rest", status="completed", detail=detail)


@api.post("/run", response_model=RunAccepted, summary="Run a mode (talk/adventure/sim/rest)")
def post_run(body: RunRequest, request: Request) -> RunAccepted:
    sandbox = _sandbox_of(request.app)
    if sandbox is not None and body.input is not None and len(body.input) > sandbox.max_input_chars:
        raise HTTPException(
            status_code=413,
            detail=f"'input' exceeds the sandbox cap of {sandbox.max_input_chars} characters.",
        )
    # HANDLE seam (a): mode="adventure" runs an adventure AS the persistent
    # agent (campaign injection — the "Adventure teaches Talk" surface), in
    # two flavors: an authored campaign YAML (`campaign`) or a free-text
    # premise the narrator improvises (`input`). mode="talk" is the live-loop
    # conversational mode; mode="rest" consolidates without teardown.
    # mode="sim" stays 501 — see the note at RunRequest.
    if body.mode == "talk":
        return _post_run_talk(body)
    if body.mode == "rest":
        return _post_run_rest()
    if body.mode == "sim":
        # DELIBERATELY not served here, and not a "coming soon" either: a raw
        # goal-driven sim harness is a developer/research surface, and the CLI
        # already does it better (modes, seeds, research telemetry, fixture
        # paths). The console's modes are talk / adventure / rest. Kept in the
        # enum with a POINTER rather than removed, because removing an enum
        # value is a breaking wire change for the generated TS client — but the
        # message is specific so it can never read as an unfinished stub.
        raise HTTPException(
            status_code=501,
            detail=(
                "mode='sim' is not a console surface — it is a developer/research one. "
                'Use the CLI: `maxim --sim "<goal>" --interactive false` (add --sim-mode / '
                "--research / --seed as needed). The console serves talk, adventure and rest."
            ),
        )
    if body.mode != "adventure":
        raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)

    premise = (body.input or "").strip() or None
    if bool(body.campaign) == bool(premise):
        raise HTTPException(
            status_code=422,
            detail=(
                "mode='adventure' requires EXACTLY ONE of 'campaign' (a campaign YAML path) "
                "or 'input' (a free-text premise to imagine)."
            ),
        )

    campaign_path: Path | None = None
    if body.campaign:
        # The request NAMES a campaign; the server SELECTS the path.
        #
        # 127.0.0.1-only is not sufficient justification for building a
        # filesystem path out of request data — a page in the operator's
        # browser can POST to localhost, so "the operator named the file" is
        # not guaranteed. Rather than construct-then-validate (which leaves
        # request data flowing into a path expression), we resolve the request
        # against the ALREADY-DISCOVERED set and use the discovery-derived
        # Path. The picker hands back exactly what /api/campaigns returned, so
        # the normal flow is unchanged; anything else is refused.
        # Dev escape hatch: drop or symlink the file into ~/.maxim/campaigns.
        requested = body.campaign.strip()
        campaign_path = _select_discovered_campaign(requested)
        if campaign_path is None:
            roots = ", ".join(str(r) for r, _ in campaign_search_roots())
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Unknown campaign {requested!r}. Runnable campaigns are the ones "
                    f"GET /api/campaigns lists (searched: {roots}). Copy or symlink yours there."
                ),
            )

    if _talk_lock.locked():
        # Symmetric to the talk path's adventure check — without this an
        # adventure could start under an in-flight talk turn and stop its
        # loop mid-send (review finding).
        raise HTTPException(status_code=409, detail="A talk turn is in flight — try again in a moment.")

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
            args=(handle, str(campaign_path) if campaign_path else "", run_id, premise),
            name=f"console.run.{run_id}",
            daemon=True,
        )
        _active_run["session_id"] = run_id
        _active_run["thread"] = thread
        thread.start()
    detail = (
        f"Imagining an adventure from your premise as persistent agent {handle.agent_id!r}."
        if premise
        else f"Campaign {campaign_path.name} running as persistent agent {handle.agent_id!r}."  # type: ignore[union-attr]
    )
    return RunAccepted(session_id=run_id, mode="adventure", status="started", detail=detail)


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

_META_KINDS = frozenset({"heartbeat", "run", "dropped", "display", "identity"})
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
    def set_run(self, run_id: str | None) -> str | None:
        """Set the active run id; returns the PREVIOUS one so a nested scope
        (a talk turn) can restore rather than clobber a live adventure's id
        (review finding: an unconditional clear stranded the rest of a
        campaign's events with run_id=None)."""
        with self._lock:
            prev, self._run_id = self._run_id, run_id
        return prev

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


def _drain_and_stop_handle(
    *, run_join_s: float = 30.0, campaign_wait_s: float = 60.0, talk_join_s: float = 20.0
) -> None:
    """Join a live run (bounded), then stop the handle. Blocking; call off-loop.

    Module-level (not a closure in the lifespan) so the shutdown contract can
    be tested without driving uvicorn, and so an operator wrapper — the
    sandbox broker's "end session" — can call the same thing the lifespan does.
    """
    with _handle_lock:
        handle = _handle
        run_thread = _active_run["thread"]
    if run_thread is not None and run_thread.is_alive():
        run_thread.join(timeout=run_join_s)
    if handle is not None:
        handle.stop(campaign_wait_s=campaign_wait_s, talk_join_s=talk_join_s)


def build_app(
    ui_dist: Path | None = None, ui_source: str = "none", *, auth_token: Any = _READ_TOKEN_FROM_DISK
) -> FastAPI:
    """Construct the Console FastAPI app. ``ui_dist`` = the built static bundle.

    ``ui_source`` records HOW that bundle was chosen (flag / config / packaged)
    so /api/identity can report it — "which UI am I serving, and why" is half
    of any contract-mismatch investigation. The sandbox policy is resolved
    here, once, and carried on ``app.state.sandbox``. ``auth_token``
    (keyword-only) is the console bearer credential: leave it defaulted to
    read ``~/.config/maxim/console_token`` from disk (``run_serve`` ensures
    that file exists first), or inject a value directly (tests, embedders).
    ``None`` — injected or read from an absent file — FAILS CLOSED: every
    authed surface refuses.
    """
    sandbox = _sandbox_policy()
    if sandbox is not None:
        logger.info(
            "console sandbox mode ON: probe(url)/setup/mesh/diagnose closed; /ws origins=%s; input cap=%d chars",
            sorted(sandbox.allowed_origins),
            sandbox.max_input_chars,
        )
    global _SERVED_UI_DIST
    # Recorded only if the bundle is ACTUALLY servable. Recording the requested
    # path unconditionally meant a typo'd --ui-dist reported ui_source="flag"
    # with a path, while `/` served the "no UI installed" page — identity
    # lying about precisely the thing it exists to explain.
    _mountable = bool(ui_dist) and Path(ui_dist).is_dir()
    _SERVED_UI_DIST = (Path(ui_dist) if _mountable else None, ui_source if _mountable else "none")
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> Any:
        # EVENT seam: bridge sim_log records into the hub for /ws fan-out.
        from maxim.simulation.sim_logger import register_sim_sink, unregister_sim_sink

        _event_hub.attach(asyncio.get_running_loop())
        # Registering the sink is SUFFICIENT: sim_log dispatches to sinks
        # independently of `_sim_active` (the terminal-verbosity switch a sim
        # owns and turns OFF at every campaign end). The console deliberately
        # does NOT call enable_sim_logging — that would also opt this
        # long-lived process into the in-memory record trail and the stdout
        # print path, neither of which a server wants.
        register_sim_sink(_event_hub.sink)
        try:
            yield
        finally:
            # Shutdown: server exit mid-campaign would otherwise kill the
            # daemon run thread with the hippocampus capture queue unflushed —
            # silent learning loss. Join the live run (bounded) BEFORE stopping
            # so the campaign's own session-end wins when it can (post-merge
            # review Exec #4); then stop() — idempotent, safe when no adventure
            # ever ran. Both live INSIDE the finally (sandbox audit): the sink
            # unregistration was already exception-safe, but the drain/stop sat
            # after the block, so an exception through shutdown skipped the
            # one step that persists a Talk-only session's substrate — a Talk
            # loop writes nothing to disk until it is stopped. The stop runs
            # before the sink is released (uvicorn has already closed every
            # /ws connection by the time lifespan shutdown is delivered, so
            # nothing is listening — the order is about the sink's own
            # finally: a raising stop() cannot strand the registration).
            try:
                await asyncio.to_thread(_drain_and_stop_handle)
            finally:
                unregister_sim_sink(_event_hub.sink)
                _event_hub.detach()

    app = FastAPI(
        title="Maxim Console",
        # Single source of truth with the UI contract check — a bundle's
        # maxim-ui.json::contract_version is compared against this exact value.
        version=CONSOLE_CONTRACT_VERSION,
        summary="Localhost Console backend + the OpenAPI facade contract for maxim-pulse.",
        lifespan=_lifespan,
    )
    app.state.sandbox = sandbox
    app.state.trust = _trust_policy()
    # Fail-closed by construction: build_app READS the token (run_serve is the
    # one place that creates it); an embedder that never provisioned one gets
    # an app whose authed surfaces all refuse, not an open console. Tests and
    # embedders may inject via the keyword-only parameter. Disk-sourced
    # tokens are re-read PER REQUEST (acceptance A7: `--rotate-token` from a
    # second terminal logs every device out on its NEXT request, no restart —
    # one file read per authed request is nothing at console rates); injected
    # tokens are static for the app's lifetime.
    if auth_token is _READ_TOKEN_FROM_DISK:
        app.state.auth_token, app.state.auth_token_source = None, "disk"
    else:
        app.state.auth_token, app.state.auth_token_source = auth_token, "static"
    app.add_middleware(_GuardMiddleware)
    app.include_router(api)

    @app.websocket("/ws")
    async def ws_events(websocket: WebSocket) -> None:
        """EventClient stream (EVENT seam) — sim_log records + meta-kinds.

        Server→client: ConsoleEvent envelopes (v2). Client→server: optional
        SubscribeFrame JSON messages; each frame REPLACES the connection's
        filter. Heartbeats fire when the stream is idle; a "dropped" meta-event
        reports backpressure losses (seq gaps mark where). Host, bearer-auth
        and non-sandbox Origin rules already ran in _GuardMiddleware (before
        accept); only sandbox's stricter origin-REQUIRED rule lives here,
        beside the policy it belongs to.
        """
        sandbox = _sandbox_of(websocket.app)
        if sandbox is not None:
            # Browsers cannot set headers on a WebSocket upgrade, but they DO
            # send Origin and a page cannot forge it — so under sandbox mode a
            # stray page on the visitor's browser cannot read the session
            # stream. Refused BEFORE accept: the handshake itself fails.
            # Stricter than the trust guard: sandbox REQUIRES a listed origin
            # (a missing Origin is refused too — every legitimate sandbox
            # client is a browser page on a listed origin).
            origin = _canonical_origin(websocket.headers.get("origin") or "")
            if origin is None or origin not in sandbox.allowed_origins:
                logger.warning("refusing /ws upgrade from origin %r (sandbox mode)", origin or "<none>")
                await websocket.close(code=1008)  # policy violation
                return
        # Echo the app subprotocol when the client offered it (the browser
        # token transport offers ["maxim-console-v1", "maxim.bearer.<t>"] and
        # the RFC obliges the server to select from the offered list); native
        # clients that offered none get a plain accept.
        offered = websocket.headers.get("sec-websocket-protocol", "")
        subprotocol = _WS_APP_SUBPROTOCOL if _WS_APP_SUBPROTOCOL in {p.strip() for p in offered.split(",")} else None
        await websocket.accept(subprotocol=subprotocol)
        conn = _WsConn()

        # Identity FIRST, before any stream event: a client should know what it
        # is attached to before it starts interpreting what that thing says.
        # ENQUEUED (not sent directly) so its seq comes from the connection
        # counter — a hardcoded seq=0 collided with the first real event's
        # seq 0 and made every client's gap detector report a phantom drop on
        # every connection, violating the monotonic-seq contract this branch
        # itself bumped. Enqueued BEFORE add_conn so it cannot be overtaken.
        try:
            ident = build_identity(_SERVED_UI_DIST[0], _SERVED_UI_DIST[1], sandbox=_sandbox_of(websocket.app))
            conn.enqueue(
                ConsoleEvent(
                    kind="identity",
                    tier="clean",
                    seq=0,  # replaced by enqueue()
                    ts=time.time(),
                    message=f"pymaxim {ident.package_version} (contract {ident.contract_version})",
                    data=ident.model_dump(),
                ).model_dump()
            )
        except Exception:
            logger.warning("could not build /ws identity frame", exc_info=True)
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
            packaged = packaged_ui_dist()
            why = (
                ""
                if packaged is not None
                else (
                    "<p>This build ships no vendored bundle — that is normal for a "
                    "source checkout (the release wheel includes one).</p>"
                )
            )
            return (
                "<h1>Maxim Console API is running</h1>"
                f"<p>No Console UI bundle installed{where}. The API + OpenAPI schema are live at "
                "<a href='/docs'>/docs</a> and <a href='/openapi.json'>/openapi.json</a>.</p>"
                f"{why}"
                "<p>Point at a built bundle with <code>maxim serve --ui-dist &lt;path&gt;</code>, "
                "persist it with <code>maxim config set console.ui_dist &lt;path&gt;</code>, "
                "or vendor one in with <code>python scripts/vendor_console_ui.py &lt;path&gt;</code>.</p>"
            )

    # OpenAPI: declare the bearer scheme + 401 shape (contract 0.4.0, design
    # A6). Enforcement is _GuardMiddleware's — this block only makes the
    # contract SAY so, per-operation, with /api/hello left security-free. The
    # trust guard's 400/403/415 stay out (legitimate clients never see them).
    _base_openapi = app.openapi

    def _openapi_with_auth() -> dict[str, Any]:
        schema = _base_openapi()  # FastAPI caches on app.openapi_schema
        components = schema.setdefault("components", {})
        components.setdefault("securitySchemes", {})["consoleToken"] = {
            "type": "http",
            "scheme": "bearer",
            "description": (
                "The console token (maxim serve --show-token). Browser /ws clients offer the "
                f"'{_WS_BEARER_PREFIX}<token>' subprotocol instead — upgrade headers are unavailable there."
            ),
        }
        unauthorized = {
            "description": "Missing or invalid console token.",
            "content": {
                "application/json": {"schema": {"type": "object", "properties": {"detail": {"type": "string"}}}}
            },
        }
        for path, ops in schema.get("paths", {}).items():
            if path in _AUTH_EXEMPT_PATHS:
                continue
            for op in ops.values():
                if isinstance(op, dict) and "responses" in op:
                    op["security"] = [{"consoleToken": []}]
                    op["responses"].setdefault("401", unauthorized)
        return schema

    app.openapi = _openapi_with_auth  # type: ignore[method-assign]

    return app


# ── CLI runner ──────────────────────────────────────────────────────────────


def _resolve(field_path: str, cli_value: Any) -> Any:
    from maxim.runtime.config_loader import resolve_setting

    # resolve_setting returns (value, source); we only need the value here.
    result = resolve_setting(field_path, cli_value=cli_value)
    return result[0] if isinstance(result, tuple) else result


def run_serve(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="maxim serve",
        description="Run the localhost Maxim Console.",
        epilog=(
            "Sandbox mode (for a proxied, single-visitor deployment): set "
            f"{_SANDBOX_ENV}=1 to close /api/probe (url form), /api/setup/mesh and "
            f"/api/diagnose, refuse /ws upgrades whose Origin is not in {_SANDBOX_ORIGINS_ENV} "
            f"(comma-separated), and cap run input at {_SANDBOX_MAX_INPUT_ENV} characters "
            f"(default {_SANDBOX_DEFAULT_MAX_INPUT_CHARS}). The bind stays 127.0.0.1; "
            "under sandbox mode authentication is the proxy's job — in every other mode the "
            "console requires its bearer token (printed at start; --show-token / "
            "--rotate-token). A browser-relay guard is always on: Hosts and browser Origins "
            f"outside loopback + {_SANDBOX_ORIGINS_ENV} are refused (DNS-rebinding / "
            "cross-site request protection)."
        ),
    )
    ap.add_argument("--port", type=int, default=None, help="Port (default: config console.port / 8765).")
    ap.add_argument("--ui-dist", default=None, help="Path to the built Console static bundle.")
    ap.add_argument(
        "--show-token",
        action="store_true",
        help="Print the console token (creating it if absent) and exit — no server.",
    )
    ap.add_argument(
        "--rotate-token",
        action="store_true",
        help="Generate a NEW console token (logging out every device) and exit — no server.",
    )
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

    if args.show_token or args.rotate_token:
        from maxim.tunnel.keys import ensure_console_token, rotate_console_token

        if args.rotate_token:
            token = rotate_console_token()
            print("maxim serve: console token ROTATED — every signed-in device is now logged out.")
        else:
            token = ensure_console_token()
        print(token)
        return 0

    port = int(_resolve("console.port", args.port))
    # Fail at start, not on the first Talk request, if the configured agent id
    # is unusable (reserved name / not a single path segment).
    try:
        agent_id = _console_agent_id()
        sandbox_on = _sandbox_policy() is not None
        _trust_policy()  # malformed console.allowed_origins fails HERE, not on the first request
    except Exception as e:  # ConfigurationError from the loader
        print(f"maxim serve: {e}")
        return 2
    # CLI > config > PACKAGED bundle. `_resolve` already applies CLI > env >
    # config; the packaged vendored bundle is the final fallback so a plain
    # `pip install pymaxim[console] && maxim serve` just works.
    config_ui_dist = _resolve("console.ui_dist", args.ui_dist)
    ui_dist = resolve_ui_dist(args.ui_dist, config_ui_dist)
    ui_source = "flag" if args.ui_dist else ("config" if config_ui_dist else ("packaged" if ui_dist else "none"))
    check_ui_contract(ui_dist)

    # Ensure the credential BEFORE building the app (build_app only READS —
    # fail-closed by construction); skipped under sandbox, where the proxy
    # owns the edge and the engine deliberately demands no token.
    token: str | None = None
    if not sandbox_on:
        from maxim.tunnel.keys import ensure_console_token

        token = ensure_console_token()

    app = build_app(ui_dist, ui_source)

    import uvicorn

    # 127.0.0.1 ONLY — the console holds keys + can run/configure Maxim.
    if ui_dist is not None:
        print(f"maxim serve → serving Console UI from {ui_dist}")
    if token is not None:
        # FRAGMENT, not query: a #token never reaches the server, so it cannot
        # land in access logs (design A5). The UI reads it, stores it, and
        # strips it from the address bar; every later visit needs no token.
        print(f"maxim serve → http://127.0.0.1:{port}/#token={token}")
        print("maxim serve → open the URL above (the token signs this browser in once; --show-token reprints)")
        print(f"maxim serve → API docs: http://127.0.0.1:{port}/docs · schema: /openapi.json (Bearer required)")
    else:
        print(f"maxim serve → http://127.0.0.1:{port}  (API docs: /docs · schema: /openapi.json)")
    if agent_id != _DEFAULT_HANDLE_AGENT_ID:
        print(f"maxim serve → fronting agent {agent_id!r} (console.agent_id)")
    if sandbox_on:
        print("maxim serve → sandbox mode ON (console.sandbox) — engine auth OFF; the proxy owns the edge")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
    return 0

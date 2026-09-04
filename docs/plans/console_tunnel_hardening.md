# Console tunnel hardening — from localhost-only to deliberately exposable

**Status:** ACTIVE — PR 1 (trust guard) MERGED #609 (2026-09-03); PR 2 (bearer auth)
IMPLEMENTED per decisions A1–A8 on `feat/console-auth` (2026-09-04; one A7 sharpening: disk
tokens are re-read per request so rotation bites with NO restart); PRs 3–4 sequenced below.
Pulse-side ledger (A6) remains open in the maxim-pulse repo.
**Motivating goal:** a phone app (client + sensory surface) that reaches the operator's own leader
over the Cloudflare tunnel and speaks the console facade + `/ws`. Target window: post-1.1.4;
app ships against a stabilized contract (see "Contract freeze" below).
**Supersedes:** the auth question deferred as open q #1 in
[docs/plans/deferred/maxim_console.md](deferred/maxim_console.md). The hosted-console **non-goal
stands** — this plan is about the OPERATOR reaching their OWN console remotely, never about
centralizing anyone's keys.

## Motivation — the 2026-09-03 security audit

A read-only audit of `src/maxim/console/` against the question "what breaks when the facade is
reachable through an authenticated tunnel" found the module is unauthenticated **by design**
(trust boundary = the operator's machine), and that exposure therefore inverts its trust model
rather than widening a few gaps. Prioritized findings (file::symbol refs verified in source):

**Critical**
- **C1 — no auth on any endpoint or `/ws`.** The reusable mechanism exists:
  `runtime/leader_proxy.py::_check_auth` (constant-time Bearer compare) +
  `tunnel/keys.py::ensure_key` (256-bit key, 0600 file). Trap to not copy: the leader's check
  returns True on an EMPTY key — the console version must fail closed.
- **C2 — `POST /api/setup/mesh` = durable agent hijack** (repoints the LLM lane at an attacker
  URL and persists it). Sandbox-closed, but sandbox is opt-in.
- **C3 — `POST /api/setup/cloud` is open EVEN IN SANDBOX** (kept open for the cloud wizard):
  attacker swaps the provider key → operator transcripts route through an attacker account.
- **C4 — `POST /api/run`** = live agent loop: real LLM spend (≤20 turns/request) and permanent
  memory-substrate poisoning of the persistent agent ("Adventure teaches Talk" ⇒ attacker
  teaches Talk).
- **C5 — `POST /api/probe` (url form)** = SSRF + credential relay + auth oracle. Sandbox-closed.

**High**
- **H1 — CSRF/DNS-rebinding was live BEFORE any tunnel**: zero Origin/Host/CORS validation
  outside the sandbox `/ws` check; Starlette parses JSON without requiring
  `Content-Type: application/json`, so cross-origin "simple" POSTs reached run/setup/probe from
  any page in the operator's browser; rebinding defeated same-origin for reads.
  **→ FIXED by PR 1 (this branch).**
- **H2 — `/ws` streams all tiers to any client**; `SubscribeFrame` is a filter, not a permission.
- **H3 — `GET /api/diagnose`** = recon (paths, env names, IPs; key BYTES are redacted — the
  in-code claim that it leaks env-sourced key material is stale).
- **H4 — `GET /api/recall`** dumps the persistent agent's memory of the operator. Not
  sandbox-closed; flagged nowhere before the audit.

**Medium** — M1: zero rate/size/concurrency limits (`uvicorn.run` bare; contrast
`leader_proxy._check_admission`); M2: `/api/identity` + `/ws` hello leak git sha/branch +
absolute paths unauthenticated; M3: talk's tool gate (deny-by-default derivation in
`console/handle.py::MaximHandle._launch_talk_loop`) is correct but the registry still CONTAINS
Bash/Write/Edit/Execute tools — one derivation regression from RCE; build the talk registry
WITHOUT them instead; M4: `/api/campaigns` leaks absolute home paths.

**Verified-good:** bind is hard-coded `127.0.0.1` with no config escape; no path traversal
(campaign paths are discovery-derived, `_is_within_search_root`); malformed `/ws` frames
handled; probe `api_key` never persisted/logged; `yaml.safe_load` throughout.

## The ladder

**PR 1 — trust guard (MERGED #609): Host + Origin browser-relay protection, always on.**
Closes H1. `console/server.py` trust-guard block: every request's Host must be loopback or a
host of `console.allowed_origins` (rebinding); state-changing requests and `/ws` upgrades that
carry a browser Origin must carry a loopback or listed one (CSRF); Origin-less clients (CLI,
native) pass — this is browser-relay protection, NOT auth. `console.allowed_origins` graduates
from sandbox-only knob to the general trusted-origins list. Contract untouched (value-level;
OpenAPI snapshot unchanged). Guard tests: `tests/unit/test_console_trust_guard.py`.
Deliberate contract change: an untrusted-origin browser page can no longer attach to `/ws`
even outside sandbox (old negative control updated in `test_console_sandbox.py`).

Review-round folds (two-lens, 2026-09-03; both MAJORs cross-confirmed): origin entries
canonicalize LOUDLY at `build_app` (malformed entry = `ConfigurationError`; default ports
dropped so a listed `https://x:443` matches the `https://x` browsers actually send); the
`testserver` allowance is pytest-scoped (`PYTEST_CURRENT_TEST`), never in production — a
hostile LAN resolver can rebind a single-label name; mutating requests additionally require
`Content-Type: application/json` (closes the legacy Origin-less form-POST residue — forms
cannot send JSON, and a fetch() that does triggers a preflight this server never answers).

Operator notes: **listing an origin also whitelists its Host** — a tunnel/proxy deployment
must forward a Host equal to the host of some listed origin (split-host deployments, UI on
one name and API on another, need the API's public name listed as an origin or they 400
fail-closed). A native shell loading its UI from `file://` sends `Origin: null` → mutations
refused; a packaged app should use a custom-scheme origin (e.g. `capacitor://…`, listable)
or send no Origin.

**PR 2 — bearer auth, fail-closed (C1→C5, H2→H4 for outsiders).**
Design pass completed 2026-09-04 (decisions A1–A8 below, grounded in the cited symbols). The
open questions from the PR 1 fold are all resolved here; implementation follows this section.

- **A1 — a SEPARATE console token, not the mesh key.** `~/.config/maxim/api_key` authenticates
  peers to the leader's inference server; one credential must not grant both inference and
  console admin (recall = the operator's memory; setup = config writes; run = spend). New
  0600 file `console_token` beside it, via a named-key generalization of `tunnel/keys.py`
  (`key_file_path(name=...)` etc. — front-gate: ride the existing module, don't fork it; the
  NEW key's writer uses `utils/atomic_io.py::atomic_write_secret`; migrating the mesh key's
  hand-rolled write is out of scope here). Rotation/inspection: `maxim serve --show-token` /
  `--rotate-token` (serve owns its credential; no new subcommand tree — `maxim tunnel key`
  stays mesh-only). **Shape: opaque with a recognizable prefix** — `mxc_` +
  `token_urlsafe(32)` (GitHub-style): secret scanners can be taught the pattern, a leaked
  token identifies itself, and it is visually distinct from the mesh key. REJECTED:
  JWT/structured tokens — signatures, expiry claims and a crypto dependency buy nothing
  when one server validates its own symmetric secret.
- **A2 — always-on; NO off toggle; sandbox is the one principled exception.** A default-off
  `console.auth` knob is exactly how C2–C4 stay reachable (a mechanism that does not run
  looks like one that ran and found nothing); the local UX cost is one click on the printed
  tokened URL. `run_serve` calls `ensure_key(name="console_token")` at startup; `build_app`
  reads it and **fails closed** — no readable token means every authed surface refuses,
  explicitly inverting `leader_proxy._check_auth`'s empty-key fail-open (`if not
  self.api_key: return True` — the documented trap). Sandbox mode stays engine-authless
  ("the proxy owns the edge" — one anonymous visitor per throwaway machine is
  identity-free by design); C3's half-open `setup/cloud` therefore remains the BROKER's
  responsibility under sandbox, now stated in the sandbox comment block instead of implied.
- **A3 — one pure-ASGI middleware, pulled forward from PR 3.** The auth check must run on
  `/ws` before accept anyway, so the PR 3 MINOR (replace `@app.middleware("http")` + the
  hand-applied `/ws` checks with one ASGI middleware dispatching on `scope["type"] in
  {"http", "websocket"}`) lands HERE rather than adding a third hand-applied site. Order:
  Host (rebinding) → auth → Origin/Content-Type (CSRF belts, kept — see A8). Exempt from
  auth: the static UI at `/` and its assets (the same public bundle maxim-pulse publishes;
  no data) and the single hello endpoint in A6. `/docs`, `/openapi.json`, all `/api/*`, and
  `/ws` require auth (the live schema is a map of the surface; the committed
  `console/openapi.json` snapshot remains the public contract artifact).
- **A4 — credential transport.** HTTP: `Authorization: Bearer <token>`, scheme parsed via
  the `leader_proxy._check_auth` branch-table pattern (CC13 auth format-freeze: new schemes
  extend the table), `secrets.compare_digest`, case-insensitive scheme per RFC 7235.
  Browser `/ws` (no upgrade headers): the token rides `Sec-WebSocket-Protocol` — client
  requests `["maxim-console-v1", "maxim.bearer.<token>"]` (token_urlsafe's alphabet is
  valid in a subprotocol token), server validates BEFORE accept and echoes
  `maxim-console-v1`. Native clients send the Authorization header on the upgrade; either
  transport satisfies the check. REJECTED: `?token=` query params anywhere (server/proxy
  log leak — the mesh key precedent: `_loggable_url` exists because URLs get logged);
  first-frame auth (loses PR 1's refuse-before-accept property and complicates the
  identity-first/seq contract on `/ws`).
- **A5 — token handoff to the served UI.** `maxim serve` prints
  `http://127.0.0.1:<port>/#token=<t>` — URL FRAGMENT, not query: never sent to the server,
  so it cannot reach access logs (strictly better than Jupyter's `?token=`). The UI reads
  the fragment on load, stores the token (localStorage), strips it via
  `history.replaceState`, and sends Bearer + the ws subprotocol thereafter. With no token,
  the UI renders a paste-token screen naming `maxim serve --show-token` (pulse work item).
  **Persistence contract: authenticate once per device, ever** — the token is a static
  credential, not a session; no expiry, no periodic re-login (localStorage in the browser,
  the keychain in a packaged app). Rotation is the revocation story and logs out every
  device at once. Expiry is deliberately absent: for a single operator it counters no
  threat (a stolen token's 29-day window is not meaningfully better than an infinite one —
  rotate on suspicion either way) and would cost a monthly re-paste. Per-device
  credentials with optional expiry are the multi-user growth path — PR 4, below.
- **A6 — contract additions (CONSOLE_CONTRACT_VERSION 0.3.0 → 0.4.0, pulse `gen:facade`
  regen).** (i) OpenAPI `securitySchemes: bearer` applied to every `/api/*` operation;
  (ii) the 401 error shape (`{"detail": ...}`) documented — the PR 1 400/403/415 refusals
  stay OUT (unchanged rationale: legitimate clients never see them); (iii) one new
  UNAUTHENTICATED endpoint `GET /api/hello` returning ONLY `{contract_version, auth:
  "bearer"}` so a client can detect skew and render the right login screen BEFORE it holds
  a token — nothing else moves out from behind auth (H3/H4/M2: identity, diagnose, recall
  are exactly what auth exists to cover). Pulse-side ledger: login/paste-token screen,
  fragment bootstrap, Bearer on FacadeClient, ws subprotocol on EventClient, contract stamp
  0.4.0.
- **A7 — acceptance tests that pin the posture** (each with its accepting counterpart, per
  the PR 1 suite's non-vacuity rule): fail-closed on missing/empty token — the
  anti-`leader_proxy` trap test; every `/api/*` + `/docs` + `/openapi.json` + `/ws` is 401
  without a token and serves with one (both transports for `/ws`, refused before accept);
  `/api/hello` and the static root are reachable tokenless; sandbox negative control (no
  auth demanded when `console.sandbox` is on); rotation invalidates the old token on the
  next request; the token never appears in a log line (grep the captured log in-test).
- **A8 — the trust guard STAYS, unchanged.** Bearer-in-header + localStorage is
  CSRF-immune, but the Host check still kills rebinding against any future authed-surface
  bug, the Content-Type belt still blocks form relays at zero cost, and sandbox mode has no
  auth at all — defense in depth, not redundancy. The phone-app threat model rides A4/A5:
  token in the app's keychain, tunnel TLS for transit; Cloudflare Access as an OPTIONAL
  second factor in front is compatible and out of scope.

PR 2 doc obligations (the posture flip must not leave contradicting prose — the 1.1.3
lesson): the server module docstring ("carries no authentication of its own"), the sandbox
comment block's rationale ("authentication that lived in the engine would have to be trusted
by every localhost user too"), and the trust-guard invariant's "NOT authentication" framing
in `docs/agents/runtime-tools.md` all need rewriting in the same PR. Error shapes: the trust
guard's 400/403/415 refusals are deliberately absent from OpenAPI (legit same-origin/native
clients never see them); PR 2's 401s WILL be client-visible, so error-shape documentation
enters the contract there, batched with the token-flow contract addition (pulse regen via
`gen:facade`).

**PR 3 — admission control (M1).**
Body-size caps and per-client rate limit on `/api/run` + `/api/probe`; `limit_concurrency` +
`ws_max_size` on `uvicorn.run`; generalize `leader_proxy`'s `_check_admission` machinery
rather than re-implementing (front-gate: it exists, ride on it). The pure-ASGI middleware
consolidation originally parked here moved to PR 2 (decision A3) — auth needs the
websocket-scope coverage anyway, so PR 3 inherits a single guard middleware to extend.

**PR 4 — pre-GA pass (before any store-shipped app).**
Viewer-vs-operator authorization tiers (ws/recall/identity vs setup/diagnose/probe);
`/ws` tier gating as permission, not filter (H2); talk registry composed WITHOUT dangerous
tools (M3); audit log on config writes; redaction audit of debug-tier stream content;
`maxim tunnel key rotate` UX surfaced in the console. Token growth path (deferred from the
PR 2 design, 2026-09-04): replace the single `console_token` file with a small named-token
list — per-device tokens (`mxc_` shape unchanged), optional per-token expiry, revoke one
device without logging out the rest — the moment a second person (RMD teammates, a shared
leader) or a second device class needs its own credential. Until then one static token +
global rotation is the deliberate design.

## Contract freeze (app prerequisite, not a PR)

Once a binary ships to an app store it cannot `maxim peer update`: from the app's first
release, `CONSOLE_CONTRACT_VERSION` changes become **additive-only** (new fields optional,
enum values never removed — the `mode=sim` tombstone pattern is the template). Review-time
rule, recorded here so the app plan can cite it.

## Non-goals

- Hosted console (centralizing keys/agents) — unchanged non-goal from
  [deferred/maxim_console.md](deferred/maxim_console.md).
- Binding beyond 127.0.0.1. The tunnel carries the resource to loopback; the bind stays.
- CORS allowances. No `Access-Control-Allow-Origin` is a feature: cross-origin pages must not
  read console responses. The trust guard refuses; it never invites.

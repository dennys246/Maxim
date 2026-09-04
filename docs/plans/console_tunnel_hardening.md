# Console tunnel hardening — from localhost-only to deliberately exposable

**Status:** ACTIVE — PR 1 (trust guard) on `feat/console-tunnel-hardening`; PRs 2–4 sequenced below.
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

**PR 1 — trust guard (this branch): Host + Origin browser-relay protection, always on.**
Closes H1. `console/server.py` trust-guard block: every request's Host must be loopback or a
host of `console.allowed_origins` (rebinding); state-changing requests and `/ws` upgrades that
carry a browser Origin must carry a loopback or listed one (CSRF); Origin-less clients (CLI,
native) pass — this is browser-relay protection, NOT auth. `console.allowed_origins` graduates
from sandbox-only knob to the general trusted-origins list. Contract untouched (value-level;
OpenAPI snapshot unchanged). Guard tests: `tests/unit/test_console_trust_guard.py`.
Deliberate contract change: an untrusted-origin browser page can no longer attach to `/ws`
even outside sandbox (old negative control updated in `test_console_sandbox.py`).

**PR 2 — bearer auth, fail-closed (C1→C5, H2→H4 for outsiders).**
Router-wide FastAPI dependency + `/ws` upgrade check, reusing `tunnel/keys.py::ensure_key` and
the `leader_proxy._check_auth` compare pattern. Fail CLOSED on empty/missing key when auth is
on. Open design decisions (resolve in this PR, front-gate scope pressure applies):
- Activation: `console.auth` config key (default off on loopback for the local UI's
  zero-friction path?) vs always-on with the token handed to the served UI Jupyter-style
  (printed URL `?token=`). Leaning: Jupyter-style always-on — "off by default" is how C2–C4
  stay reachable; but the pulse Console shell must learn the token flow first
  (cross-repo: contract addition, `gen:facade` regen).
- Browser `/ws` cannot set upgrade headers → first-frame token or query param; native clients
  use the Authorization header.
- Sandbox interplay: sandbox mode keeps "proxy owns the edge" (no engine auth) — C3's
  half-open `setup/cloud` gets re-audited here.

**PR 3 — admission control (M1).**
Body-size caps and per-client rate limit on `/api/run` + `/api/probe`; `limit_concurrency` +
`ws_max_size` on `uvicorn.run`; generalize `leader_proxy`'s `_check_admission` machinery
rather than re-implementing (front-gate: it exists, ride on it).

**PR 4 — pre-GA pass (before any store-shipped app).**
Viewer-vs-operator authorization tiers (ws/recall/identity vs setup/diagnose/probe);
`/ws` tier gating as permission, not filter (H2); talk registry composed WITHOUT dangerous
tools (M3); audit log on config writes; redaction audit of debug-tier stream content;
`maxim tunnel key rotate` UX surfaced in the console.

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

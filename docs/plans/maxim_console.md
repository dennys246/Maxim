# Maxim Console — the general local web front door (+ the shared UI kit)

**Status:** Shell plan, drafted 2026-07-23. Sibling to the Reachy app plans; this one generalizes them. The Reachy app becomes the **first consumer** of the shared kit defined here.
**Scope:** A general **localhost web console** (`maxim serve`) to configure, run, and observe Maxim — for *any* use (headless chat, sim, DM, config, diagnostics, memory), not just a robot. Plus the **shared UI kit** (components + design system + event client) that both the console and the Reachy app compose. Presentation layer only — **all logic stays in pymaxim behind `api.py`**.
**Target:** Post Reachy-MVP (the Reachy app ships first as the kit's first consumer; the console follows and reuses the same components). Kit designed reusable from day one so nothing is rewritten.
**Gates:** None release-gating — it's a presentation layer over existing facades. Shares the Reachy MVP's **FIT** dependency only where it drives an on-device agent.

**Topology assumption (from the design discussion):** the console is a **localhost** app (`maxim serve` on `127.0.0.1`, pip-installed, runs on the user's own machine/leader). The Reachy app is the **same UI kit** packaged as a `ReachyMiniApp` running **on-device**; HF Spaces is its *distribution channel*, not a place its brain's UI lives. **Explicit non-goal (for now):** a cloud-*hosted* console users dial into — that inverts the private/on-device value prop and adds real key-handling/privacy surface. Tracked as open q #1, not assumed.

---

## Front-gate (Principle 3)

**Does a web console need its own mechanism, or ride existing infra?** **Rides — it's presentation over `api.py`.** The 17 facade verbs (`configure`, `run`, `connect`, `diagnose`, `observe`, `campaign`, `list_models`, `download_model`, `benchmark`, `research`, `on()` event stream, …) plus the `Observer` are exactly a web UI's backend, and there's partial HTTP-control-plane precedent (leader-proxy admin endpoints, `metrics_snapshot` "for admin API"). **Genuinely new (small):** the `maxim serve` web server + an event bridge (`api.on()` → WS/SSE) + the UI kit itself. **Business logic stays in pymaxim** — the console MUST NOT reimplement anything the facades do (same cardinal rule as the app: thin front-end, fixes go into pymaxim/`api.py`).

---

## The shared abstraction — a three-layer stack

Both front-ends are **a thin shell composing shared kit components over the pymaxim facades.** Wiring them together = defining these three layers once and having every front-end honor them.

```
┌─ Layer 3 — DEPLOYMENT SHELLS (thin, per-target) ───────────────────────┐
│  Reachy shell            Maxim Console shell        (future: others)    │
│  ReachyMiniApp,          `maxim serve`,                                 │
│  on-device, HF Space     localhost:PORT, pip                           │
│  adds: wake/rest motion, adds: all modes, config                       │
│  antenna PTT, orient     mgmt, model download, observe                 │
└───────────────┬──────────────────────┬────────────────────────────────┘
                │  both compose ↓       │
┌─ Layer 2 — SHARED UI KIT (one package; the thing to abstract now) ──────┐
│  SetupWizard · ConnectionTest · ModelPicker · SpendControls ·          │
│  StatusChip · MemoryView · RunSurface · EventClient · DesignSystem      │
└───────────────┬────────────────────────────────────────────────────────┘
                │  every component binds to ↓ (never bespoke back-channels)
┌─ Layer 1 — pymaxim FACADES + SEAMS (shared backend contract) ───────────┐
│  api.py verbs + the seams in reachy_app_maxim_seams.md:                 │
│  SETUP (config-write) · PROBE (conn test) · RECALL (memory) ·          │
│  HANDLE (runtime) · list_models · diagnose · observe · on()            │
└─────────────────────────────────────────────────────────────────────────┘
```

**The load-bearing rule that makes them wire together:** every kit component takes its data through a **Layer-1 facade/seam**, never through a shell-specific prop or back-channel. A component that reads `RECALL` works identically whether the shell is Reachy-on-Pi or console-on-laptop. That is what lets the "full dashboard" release be *composition*, not a rewrite.

## Shared component inventory (Layer 2 — define these now)

| Component | Binds to (Layer 1) | Reachy shell | Console shell |
|---|---|---|---|
| **SetupWizard** — mesh↔cloud branch, test, save | `SETUP` + `PROBE` + `list_models` | first-run | first-run / settings |
| **ConnectionTest** — structured pass/fail + fix-hint | `PROBE` | ✓ | ✓ |
| **ModelPicker** — curated + full profile list | `api.list_models` | cloud path | ✓ |
| **SpendControls** — budget + cost caps | `cloud.session_budget_usd` + caps | ✓ | ✓ |
| **StatusChip** — where it thinks · health · spend | `api.diagnose` | ✓ | ✓ |
| **MemoryView** — "what Maxim remembers about you" | `RECALL` | main page | ✓ |
| **RunSurface** — run a mode + stream output | `HANDLE` (`talk`/`play_campaign`/`rest`) + `api.run`/`campaign` | Adventure / Talk / Rest | chat / sim / DM / benchmark |
| **EventClient** — live thinking/telemetry stream | `api.on()` / `observe` | thinking panel | observe / telemetry |
| **DesignSystem** — tokens, theme, layout primitives | — | ✓ | ✓ |

**RunSurface is the key abstraction:** Reachy's "Adventure" and the console's "run a DM campaign" are the *same component* with a different mode config and a different `HANDLE` flavor. Define it once.

**RunSurface's panel model — port the interactive `Live` display's information architecture.** The existing Rich `MaximDisplay` ([interactive/display.py](../../src/maxim/interactive/display.py)) already validated the immersive-session spread in the terminal: status bar + **narrative/dialogue (bright)** + **bio-subsystem activity log (dimmed)** + a **thinking/deliberation panel** (accumulates the reasoning chain + which bio-systems enriched each cycle) + input/choices + **pluggable domain panels via the `DisplayExtension` ABC** (explicitly built for DM character-sheet/inventory/encounter, research, robot-joint panels). RunSurface should port this *IA* (not the Rich code): a **core layout** (narrative + input + activity + thinking) + **pluggable mode-panels** mirroring `DisplayExtension`. Crucially, **the terminal and the web are two renderers over one event model** — `sim_log()` / `observe` / `api.on()` — so the kit's `EventClient` consumes the same stream `MaximDisplay` does. The "show Maxim think + learn *while the story unfolds*" surface (bio-dim under scene-bright) is what makes Adventure legible and is available from the start, since it's already an event stream, not a new mechanism.

## One generalization the seams need: HANDLE has two flavors

The `HANDLE` seam (from [reachy_app_maxim_seams.md](reachy_app_maxim_seams.md)) was scoped as "build one persistent **embodied** agent." The console needs the same handle **headless** (no body). Same interface (`talk` / `play_campaign` / `rest` / `observe`, clean full-consolidation stop), two constructors: **embodied** (body = `bodies/reachy_mini`, Reachy shell) and **headless** (console shell). This keeps `RunSurface` shell-agnostic — it drives a `HANDLE`, not "a robot." Fold this into the HANDLE seam spec.

## What the Console adds beyond the Reachy subset

The Reachy app uses a *subset* of the kit (setup, memory, status, RunSurface for Adventure/Talk/Rest). The console exposes the rest:
- **All run modes without a robot:** headless chat, generative sim, DM campaigns (human-player, no robot), benchmark, research.
- **Config management:** view/edit `config.json` surfaces (lanes/placement, cloud, budget) via `SETUP`, beyond first-run.
- **Model management:** `download_model` / `delete_model` / `list_models`.
- **Live observe / telemetry:** the `Observer` + `on()` stream as a real panel (bio-stack activity, provenance, doctor rows).

## Backend — `maxim serve`

A new localhost web server (**FastAPI** — chosen for the OpenAPI contract below) that (a) serves the kit's static bundle, (b) exposes a thin JSON API mapping 1:1 to `api.py` verbs, (c) bridges `api.on()` → WS/SSE for the EventClient. **Security:** bind **`127.0.0.1` only** (never `0.0.0.0`); it holds cloud keys and can run/configure Maxim. The leader-proxy admin endpoints (on-network by design) are the cautionary counter-example — the user console stays local. Auth model for any future non-local exposure is open q #1.

**The facade contract is OpenAPI-typed — this is the load-bearing cross-repo seam (surfaced by the maxim-pulse bootstrap review).** The single highest-risk failure of a Python-backend / TypeScript-frontend split is *silent drift* between the seam signatures and the kit's `FacadeClient`. FastAPI already emits an OpenAPI schema from its Pydantic request/response models, so:
- `maxim serve` MUST expose a stable, typed OpenAPI schema for every facade verb (typed request + response models — no bare `dict`/`Any` on the wire surface). This is a hard requirement on the serve layer, not optional polish.
- The maxim-pulse kit **generates its `FacadeClient` TypeScript types from that schema** (`openapi-typescript` / `orval`) — the generated types are the **single source**. The hand-written `FacadeClient` interface in the scaffold's Phase 0 is a temporary stand-in that gets **replaced** by the generated types once the schema exists — NOT maintained alongside them (two interfaces re-introduce the exact drift this prevents).
- CI on both sides checks against the schema: pymaxim pins the OpenAPI output; maxim-pulse regenerates and fails if types are stale. A schema change is a deliberate, reviewed contract change.

**UI-dist handoff — how the built Console bundle reaches `maxim serve`** (also surfaced by the bootstrap review; our earlier drafts left it unspecified). The bundle is built in maxim-pulse; the server that serves it lives here. Resolution: `maxim serve` serves the static dist from a **configurable path with a packaged default** — resolution order `--ui-dist <path>` flag > `config.json::console.ui_dist` > a bundled default location (populated at pymaxim release from a pinned maxim-pulse build artifact). maxim-pulse CI uploads the Console `dist/` as a build artifact for that release-time vendoring. A missing/empty dist degrades to a clear "console UI not installed — run `…`" message, never a blank page. Do NOT auto-download the bundle at runtime (supply-chain surface); it's vendored at release or pointed at a local dev build.

## Honest accounting

| Piece | Reuse / new |
|---|---|
| The 17 `api.py` facades + `Observer` + `on()` | **Reuse** — the whole backend |
| The seams (`SETUP`/`PROBE`/`RECALL`/`HANDLE`) | **Reuse** — defined in the seams plan; shared by both front-ends |
| `maxim serve` web server + `api.on()`→WS/SSE bridge | **New (small)** |
| OpenAPI-typed facade schema (typed models on every seam) + UI-dist serving | **New (small, load-bearing)** — the cross-repo contract; see Backend |
| The shared UI kit (Layer 2 components) | **New (the real work)** — but built once, consumed twice |
| HANDLE headless flavor | **New (small)** — generalizes an existing seam |
| Console shell (all-modes, config, observe panels) | **New** |

## How the four plans wire together

- **[reachy_app_maxim_seams.md](reachy_app_maxim_seams.md)** — *is Layer 1.* The seams are the shared backend contract, **not Reachy-specific**. (Reframed by this plan; the Reachy app is one consumer, the console another.)
- **[reachy_mini_app.md](reachy_mini_app.md)** — the Reachy **shell** (Layer 3). Its GUI is built *as the first slice of the kit*, which resolves that plan's "served-UI vs Pollen-config-hook" open question → **served, shared kit**.
- **[reachy_dm_app.md](reachy_dm_app.md)** — Adventure is a **RunSurface mode** (Layer 2) over the embodied `HANDLE`.
- **This plan** — Layer 2 (the kit) + the console shell (Layer 3) + `maxim serve`.

## Sequencing

1. **FIT** (Pi measurement) + **the seams** (Layer 1) land first — shared foundation, already the Reachy MVP's critical path.
2. **Build the Reachy MVP's views AS kit components** (Layer 2), not bespoke — SetupWizard, StatusChip, MemoryView, RunSurface. Ship the Reachy app first. *This is the forcing function that keeps the components general.*
3. **`maxim serve` + console shell** reuse the kit and add the console-only modes/panels. Ships after (or in parallel once the kit is stable).
4. HANDLE headless flavor lands with step 3.

**Do not block the Reachy MVP on the full console** — the kit is designed reusable so the console is later *composition*, not rework.

## Decisions (resolved)

- **Repo topology — one repo, two build targets** (monorepo). A shared internal component package (Layer 2 kit) + two entry points: the full console app and the Reachy app. Both build from the same kit; the Reachy target code-splits to a lean on-device bundle (the heavy dashboard viz never enters the Pi bundle). Chosen over a separately-versioned kit package for simplicity — the kit and its two consumers move together.
- **Framework — React + TypeScript (Vite front end) over the `maxim serve` FastAPI backend (JSON + WebSocket).** Rationale: "maximal display capacity" for Maxim's data (NAc/EC **graphs**, bio-stack **time-series**, `observe`/`on()` **live streams**) is won by the viz ecosystem, and React's is unmatched — **react-flow** (causal/cluster/provenance node-graphs), **visx/nivo/Recharts** (telemetry charts, drive gauges), **react-three-fiber** (3D pose), **react-grid-layout** (multi-panel dashboards). Monorepo tree-shaking keeps the Reachy bundle lean (imports only SetupWizard/StatusChip/MemoryView/RunSurface). All *logic* stays in Python behind `api.py`/`maxim serve` — React is pure presentation. **Component base:** shadcn/ui (Radix + Tailwind) or Mantine for primitives; react-flow + visx for the Maxim-specific visualizations. Runner-up considered: SvelteKit (leaner bundle, thinner complex-viz ecosystem — flip only if bundle-size beats display-breadth). Ruled out for this goal: Streamlit/Dash/Reflex (ceiling on custom real-time viz + no clean two-target/embedded story).

## Open questions

1. **Hosted console (the "through HF" fork) — non-goal for now, decide explicitly later.** A cloud-hosted console changes the privacy/key model fundamentally (keys leave the machine, auth required). Localhost-only until there's a concrete reason.
2. **Auth model** if the console is ever exposed beyond localhost (ties to #1).

## References

- Layer 1: [reachy_app_maxim_seams.md](reachy_app_maxim_seams.md) · `api.py` facades ([api.py](../../src/maxim/api.py))
- Layer 3 shells: [reachy_mini_app.md](reachy_mini_app.md) · [reachy_dm_app.md](reachy_dm_app.md) · seed [AGENTS.md/CLAUDE.md](maxim_pulse_seed/)
- Precedent: leader-proxy admin endpoints ([leader_proxy.py](../../src/maxim/runtime/leader_proxy.py)) · Rich terminal surface ([interactive/display.py](../../src/maxim/interactive/display.py)) — the console is its web sibling

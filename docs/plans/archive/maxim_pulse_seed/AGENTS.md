# AGENTS.md — maxim-pulse

> **✅ SEED CONSUMED 2026-08-01 — canonical copies live in the `Maxim-pulse` repo and have since diverged from this fork. Do not edit.**

Durable architecture + development standards for the **Maxim UI monorepo**. Tool-agnostic;
Claude Code also reads `CLAUDE.md` (operational checks + commands, which points here for the rest).

Seeded from the Maxim project's conventions. It carries the *principles*, not Maxim's
2,000-line invariant ledger — this repo is a **presentation layer**, and keeping it thin (all
logic in pymaxim) is the point.

---

## What this repo is

`maxim-pulse` — the **face of Maxim**. One **React + TypeScript monorepo** (Vite) producing
**two build targets** over a **shared UI kit**:

- **Maxim Console** — a general **localhost** dashboard (launched via pymaxim's `maxim serve`)
  to configure, run, and observe Maxim: headless chat, simulations, DM campaigns, config,
  diagnostics, and the live bio-stack.
- **Reachy app** — the **same kit** packaged as a Pollen [`ReachyMiniApp`](https://huggingface.co/docs/reachy_mini/SDK/apps),
  published to a Hugging Face Space, running **on-device** on a Reachy Mini. Flagship
  experience: **Adventure** (the voice DM).

Both compose the shared kit; both are **presentation over pymaxim**.

**Product thesis:** Maxim learns you across sessions without fine-tuning, persisted in
`~/.maxim/`. The Console makes all of Maxim legible; the Reachy app is that made embodied.

## The three-layer architecture — know which layer you're in

- **Layer 1 — pymaxim facades + seams** (`api.py`, `maxim serve`, and the seams `SETUP` /
  `PROBE` / `RECALL` / `HANDLE`). **NOT in this repo** — called over HTTP/WS.
- **Layer 2 — the shared UI kit** (this repo): SetupWizard, ConnectionTest, ModelPicker,
  SpendControls, StatusChip, MemoryView, **RunSurface**, EventClient, DesignSystem.
- **Layer 3 — the two shells** (this repo): Console + Reachy.

**The load-bearing rule:** every kit component takes its data through a **Layer-1 facade/seam**,
never a shell-specific prop or back-channel. That is what lets both shells share components and
lets the full dashboard be *composition*, not a rewrite.

## The cardinal rule: this repo is presentation-only

The UI is **glue over pymaxim**. All logic — cognition loop, LLM routing, mesh, persistence,
memory — lives in pymaxim behind `api.py` / `maxim serve`.

> **When something is hard here, fix it in pymaxim behind a facade/seam — not with a workaround
> in the UI.** A new `api.py` verb, a `maxim serve` endpoint, a seam parameter. If this repo
> grows business logic, that's the signal a pymaxim API has a gap. Front-gate applied outward.

Concretely: no reimplemented agent logic, no direct bio-stack poking, no bespoke config writing
— go through the facades.

## The stack

React + TypeScript, **Vite**, monorepo with two build targets + a shared kit package. Backend
is pymaxim's **FastAPI `maxim serve`** (JSON + WebSocket) — this repo *talks to it*, never
reimplements it. Visualization: **react-flow** (NAc/EC/provenance graphs) + **visx/nivo**
(telemetry) on a **shadcn/ui** (Radix + Tailwind) or **Mantine** base. The Reachy target adds a
thin **Python `ReachyMiniApp` bootstrap** that builds the embodied agent and serves the built
bundle — so the repo is lightly polyglot (mostly TS, a small Python entry for the Reachy target).

## Topology you're building on (pymaxim mental model)

- **Peer / leader mesh.** A node is `leader` (hosts the big LLM), `peer` (routes heavy work to
  a leader), or `solo`. An **on-device Reachy is a peer** (its Pi can't host a large model); the
  **Console** typically runs on the user's own machine/leader. The SetupWizard writes the choice.
- **Lane placement = where work runs.** Tiers (`large`/`medium`/`small`) = *what the work needs*;
  placement (`LOCAL`/`CLOUD`/`PEER`) = *where it runs*. SetupWizard writes
  `config.json::lanes.large.placement`. Three destinations, **only one is cloud**: peer→leader
  on LAN, peer→leader over tunnel, or a cloud provider.
- **`config.json` is the operator-config layer.** Written only through pymaxim's
  `config_writer` (via the `SETUP` seam); keys are **file/keyring refs**, never inline plaintext.
  The UI calls the seam — it never hand-rolls JSON.
- **The bio-stack is the memory.** `~/.maxim/` holds it; session-end consolidation +
  `save_cerebellum()` make it durable. This *is* "remembers you." The `RECALL` seam reads it.
- **The agentic runtime is the loop** (`HANDLE` seam), with **embodied** (Reachy) and
  **headless** (Console) flavors — same interface, different constructor. `RunSurface` drives a
  `HANDLE`, not "a robot."

## Execution flows

**Console (`maxim serve`, localhost):**
```
maxim serve (pymaxim, binds 127.0.0.1)
  → serves the Console bundle + a JSON/WS API mapping 1:1 to api.py verbs
  → user configures (SetupWizard) / runs (RunSurface) / observes (EventClient) Maxim
```

**Reachy app (on-device, HF Space):**
```
one-click install from the robot dashboard
  → ReachyMiniApp.run(stop_event): build embodied agent (HANDLE) + serve the Reachy bundle
  → stop_event (dashboard stop = SESSION BOUNDARY): FULL session-end consolidation +
    save_cerebellum()  → cross-session memory persisted in ~/.maxim/
```
**The stop path is load-bearing.** Every stop must be a clean full session-end or the
cross-session claim silently doesn't hold (the lightweight sim path is wrong for a long robot
session — see the `consolidation` choice in the HANDLE seam).

## Development standards (carried forward from Maxim)

- **No band-aid fixes.** Root cause vs symptom-hiding (swallowed exception, special case, flag
  toggling around broken behavior). If the real fix belongs in pymaxim, say so and do it there.
- **Front-gate scope pressure.** Before building a mechanism here, ask: *presentation, or logic
  that belongs in pymaxim?* Default: it rides a facade.
- **Presentation-only discipline.** Components bind to Layer-1 facades/seams, never back-channels.
- **Two-lens review before merge, on the diff as it merges.** A *different* reader, not a more
  careful one; re-run if the branch keeps growing.
- **Verify actuation before theorizing about a sensor.** A wrong actuation assumption is
  indistinguishable from a broken sensor. Read the vendor's docs before reverse-engineering
  kinematics. (Cost the Maxim project a full session on Reachy — see the head-frame trap.)
- **Prefer config over new env vars** (pymaxim side): tunables go in `config.json` via
  `resolve_setting`, not new `MAXIM_*` vars.
- **Dormancy over deletion.** A mechanism that doesn't earn its keep is marked dormant, not
  deleted — pymaxim is intimately wired; whim-deletion cascades.
- **Bio-naming is load-bearing.** Never rename pymaxim's bio-system classes (Hippocampus, NAc,
  ATL, EC, SCN).

## Reachy hardware traps (Reachy shell only — read before touching motion/sensors)

Authority: `docs/embodiment/reachy_mini/` in pymaxim + Pollen's docs. The two that cost sessions:

- **Transport pivot at SDK 1.5.0.** Control is WebSocket `ws://<host>:8000/ws/sdk`, liveness
  `GET /api/daemon/status`, network DoA `GET /api/state/doa`. Client and daemon **must match**
  across the pivot — version-match after any reflash. Never re-introduce zenoh `:7447`. pymaxim
  pins `reachy-mini[gstreamer]>=1.8.3,<2.0`.
- **Head pose is WORLD-frame and sits above `body_yaw`.** `goto_target(body_yaw=X, head=None)`
  *counter-rotates* the head, so head-mounted mics/camera **don't turn with the body**. Route
  motion through pymaxim's `ReachyMiniController` (which encodes the fix) — don't hand-roll
  `goto_target`. This faked a phantom "slow DoA sensor" pathology through six wrong hypotheses.
- **Torque is a separate gate.** Motors boot disabled; motion silently no-ops (reads still work)
  until `enable_motors()` runs. The controller's `wake_up()` handles it.

## Repo conventions

- **Monorepo layout (proposed):** `packages/kit` (shared UI kit), `apps/console` (Console
  shell), `apps/reachy` (Reachy shell — React bundle + the thin Python `ReachyMiniApp` bootstrap).
- **Code-split by target.** The heavy dashboard viz (react-flow/visx graphs) must **never enter
  the Reachy on-device bundle** — the Pi target imports only SetupWizard/StatusChip/MemoryView/
  RunSurface. Keep the bundle lean so one-click install doesn't time out or fill the SD card.
- **Localhost-only Console.** `maxim serve` binds **`127.0.0.1`** (it holds keys, can run +
  configure Maxim). Never `0.0.0.0`. A hosted console is an explicit non-goal.
- **Tests run offline.** Mock the facade client / `maxim serve` / the Reachy SDK; no test spins
  up a real robot or real LLM. Component tests via vitest against a fake facade.
- **Two renderers, one event model.** The kit's `EventClient` consumes the same
  `sim_log`/`observe`/`api.on()` stream the terminal `MaximDisplay` (pymaxim `interactive/`)
  does — **port that display's information architecture** (panel spread, bio-dim/scene-bright,
  the `DisplayExtension` plugin pattern), not the Rich code.
- **Privacy is a shipped feature.** The Reachy daemon's `/ws/sdk` has no auth; the app reads
  mics/camera and (in cloud mode) phones an LLM. Store listing states what leaves the device
  and what's stored in `~/.maxim/`.
- **Versioning split:** the Reachy target tracks **Pollen SDK** cadence; the kit + Console track
  **pymaxim**. Pin the `pymaxim` dependency explicitly.

# Maxim UI — unified roadmap (cross-repo tracking)

**Status:** Roadmap index, drafted 2026-07-23. The single doc that sequences the whole UI effort across **two repos** and links the five detailed plans. This is the tracker; the linked plans are the specs.
**Scope:** Everything from the shared backend seams → the shared React kit → the two shells (Reachy app + Maxim Console) → the flagship Adventure → the full dashboard. Presentation-only above `api.py`; logic stays in pymaxim.
**Gates:** **FIT** (does the substrate fit the Pi) gates every on-device/mesh claim — Phase 0, do it first.

## The two repos

| Repo | Owns | Layers |
|---|---|---|
| **pymaxim** (this repo) | The seams ([reachy_app_maxim_seams.md](reachy_app_maxim_seams.md)) + `maxim serve` backend + FIT measurement | Layer 1 |
| **maxim-pulse** (new monorepo) | The shared React kit + two build targets (Console shell + Reachy shell) | Layers 2 + 3 |

Decisions locked (see [maxim_console.md](maxim_console.md) for rationale): **monorepo, two build targets** · **React + TypeScript / Vite over FastAPI `maxim serve`** · **localhost-only** (hosted = explicit non-goal) · **Adventure is the flagship** · **one persistent agent across modes** · **push-to-talk** voice. *(The seed [AGENTS.md/CLAUDE.md](maxim_pulse_seed/) are reframed to the `maxim-pulse` monorepo — shared React kit + Console + Reachy shells — ready to copy in on repo creation.)*

---

## Roadmap — each phase ships something usable

### Phase 0 — De-risk & scaffold *(nothing depends on nothing; do these first, in parallel)*
| Deliverable | Repo | Why first |
|---|---|---|
| **FIT** — measure substrate RSS on the Pi (mesh: LLM remote, substrate local) | pymaxim | Hard gate on the whole on-device/mesh story. If it fails, the architecture changes shape. |
| **Monorepo scaffold** — Vite, two build targets, shared kit package, `DesignSystem` skeleton, CI | maxim-pulse | Proves the two-target build works before any component is written (the second-biggest unknown). |
| **`maxim serve` skeleton** — FastAPI, `127.0.0.1`, `api.on()`→WS bridge stub, **typed OpenAPI schema + UI-dist serving** | pymaxim | The backend both shells' web layer needs; stub it early. The OpenAPI schema is the cross-repo facade contract (kit generates its TS `FacadeClient` from it — no bare `dict`/`Any` on the wire). |

**Ships:** a go/no-go answer on the Pi + an empty-but-building two-target monorepo + a serve stub. *No user-facing surface yet — this is the risk-retirement phase.*

### Phase 1 — Layer 1 seams *(the shared backend contract)*
| Deliverable | Repo | Notes |
|---|---|---|
| **SETUP** (config-write verbs) · **PROBE** (structured conn test) · **RECALL** (memory-read) · **PKG** (ARM packaging) | pymaxim | Mostly independent facades — parallelizable |
| **HANDLE** — embeddable runtime handle, **embodied + headless flavors**, clean full-consolidation stop, absorbs the persistent-agent campaign injection | pymaxim | Biggest seam; the highest-value + highest-silent-risk piece |

**Ships:** the tested backend contract. Still no UI — but every kit component now has something to bind to.

### Phase 2 — Reachy MVP *(first kit consumer + first shell — the platform proof)*
| Deliverable | Repo | Binds to |
|---|---|---|
| **Kit components** — SetupWizard, ConnectionTest, ModelPicker, SpendControls, StatusChip, MemoryView, RunSurface (Talk/Rest), EventClient | maxim-pulse | SETUP/PROBE/RECALL/HANDLE/list_models/diagnose/on() |
| **Reachy shell** — ReachyMiniApp bootstrap + stop, smoke test, embodied HANDLE, wake/rest, HF Space publish | maxim-pulse | embodied HANDLE + PKG |

**Ships: Reachy app v1** — installable from the robot dashboard: setup (mesh/cloud) + Talk + "what it remembers" + status chip. **This is the platform-proving milestone** — smallest real ship that validates the whole stack end-to-end. Build the components *as reusable kit pieces*, not bespoke — that's the forcing function for Phase 3.

### Phase 3 — Maxim Console *(second shell — reuse, don't rebuild)*
| Deliverable | Repo | Notes |
|---|---|---|
| **`maxim serve` full** — all api verbs + WS stream | pymaxim | fleshes out the Phase-0 stub |
| **Console shell** — reuse the kit + console-only surfaces: all run modes (headless chat/sim/DM/benchmark), config management, model management, live observe/telemetry | maxim-pulse | headless HANDLE drives RunSurface |

**Ships: Maxim Console v1** — the general localhost front door. Most of it is *composition of Phase-2 kit components*; the new work is the console-only panels + the serve backend.

### Phase 4 — Adventure flagship *(the headline experience; riskiest piece, so it comes after the platform is proven)*
| Deliverable | Repo | Notes |
|---|---|---|
| **HANDLE.play_campaign** — persistent-agent injection (cross-mode learning) | pymaxim | inside HANDLE; the "Adventure teaches Talk" seam |
| **VOICE** — push-to-talk voice loop (STT→DM→TTS); STT placement only if the Pi latency spike demands it | pymaxim + maxim-pulse | turn-taking needs no new mechanism |
| **CONTENT** — 2 seed campaigns + safety stance | pymaxim | demand-driven depth after |
| **RunSurface DM mode** | maxim-pulse | shared: embodied (Reachy) + headless (console) |

**Ships: Adventure on Reachy** (the flagship) **+ "run a DM campaign" in the Console** (free, from the shared RunSurface).

### Phase 5 — Full dashboard *("display everything Maxim has to offer")*
| Deliverable | Repo | Notes |
|---|---|---|
| Rich viz panels — react-flow (NAc causal / EC cluster / provenance graphs), visx/nivo telemetry, react-grid-layout multi-panel dashboards | maxim-pulse | the reason React was chosen |

**Ships: the full Maxim dashboard** — the console grows from "front door" into "see the whole bio-stack live."

---

## Critical path & parallelism

```
Phase 0 ──► Phase 1 ──► Phase 2 (Reachy MVP) ──┬──► Phase 3 (Console) ──► Phase 5 (full dashboard)
  FIT ∥ scaffold   seams        the platform    │
  ∥ serve-stub   (∥ where       proof           └──► Phase 4 (Adventure) ──┘
                  independent)
```

- **The spine is Phase 0 → 1 → 2.** FIT → seams → Reachy MVP. Everything real depends on it.
- **Phases 3 and 4 branch after the MVP** and can run in either order / in parallel — Console is *lower-risk reuse*, Adventure is *higher-risk new* (the voice loop). If forced to pick, do Console first (cheap, proves the kit generalizes) unless Adventure's demo value is the priority.
- **Phase 5 is continuous** — viz panels land incrementally once the Console exists.

## Cross-repo status tracker

| # | Deliverable | Repo | Phase | Depends on | Status |
|---|---|---|---|---|---|
| 1 | FIT (Pi measurement) | pymaxim | 0 | — | ◑ harness + dev-box finding (PR #415); on-Pi run pending hardware |
| 2 | Monorepo scaffold + DesignSystem | maxim-pulse | 0 | — | ✅ done — merged (shadcn/ui, react-flow+visx, code-split proven; Reachy 62 KB gz) |
| 3 | `maxim serve` skeleton | pymaxim | 0 | — | ✅ done — PR #416 (OpenAPI contract, 8765, /api+/ws, ui-dist handoff) |
| 4 | SETUP / PROBE / RECALL / PKG | pymaxim | 1 | — | ☐ planned |
| 5 | HANDLE (embodied + headless) | pymaxim | 1 | — | ☐ planned |
| 6 | Kit components (MVP subset) | maxim-pulse | 2 | 2,4,5 | ☐ planned |
| 7 | Reachy shell + HF publish | maxim-pulse | 2 | 1,5,6 | ☐ planned |
| 8 | `maxim serve` full | pymaxim | 3 | 3 | ☐ planned |
| 9 | Console shell | maxim-pulse | 3 | 6,8 | ☐ planned |
| 10 | HANDLE.play_campaign | pymaxim | 4 | 5 | ☐ planned |
| 11 | VOICE loop | pymaxim + ui | 4 | 5,10 | ☐ planned |
| 12 | CONTENT (campaigns + safety) | pymaxim | 4 | 10 | ☐ planned |
| 13 | RunSurface DM mode | maxim-pulse | 4 | 6,10 | ☐ planned |
| 14 | Full-dashboard viz | maxim-pulse | 5 | 9 | ☐ planned |

## Risks / watch-items

- **FIT (#1)** — the whole mesh/on-device story rests on it. First thing to run.
- **HANDLE persistent-agent injection (#5/#10)** — fails *silently* (learning lands in a throwaway agent). Its regression guard is non-negotiable.
- **Voice-loop latency (#11)** — the one Phase-4 unknown; a spike-and-measure, not an architecture problem (push-to-talk settles turn-taking).
- **Reachy bundle size** — code-split so the heavy Phase-5 viz never enters the on-device target.
- **Scope creep on the Console** — it's a product; keep it presentation-only over `api.py`, or it eats the roadmap.
- **Facade contract drift (Python seams ↔ TS `FacadeClient`)** — the highest-risk cross-repo seam. Killed by the OpenAPI-typed contract: `maxim serve` emits a typed schema, the kit *generates* its types from it, CI on both sides fails on staleness. Requires typed models on every seam (no bare `dict`/`Any`). See [maxim_console.md](maxim_console.md) § Backend.
- **Console UI-dist handoff** — how the built bundle reaches `maxim serve` (configurable path + packaged default, vendored at pymaxim release; never runtime-downloaded). A pymaxim-side decision that lands with the serve skeleton (#3).

## The five detailed plans this tracks

| Plan | Layer | What it specs |
|---|---|---|
| [reachy_app_maxim_seams.md](reachy_app_maxim_seams.md) | 1 | The pymaxim seams (FIT, SETUP, PROBE, RECALL, HANDLE, PKG, VOICE, CONTENT) |
| [maxim_console.md](maxim_console.md) | 2 + 3 | The shared kit + the Console shell + `maxim serve` |
| [reachy_mini_app.md](reachy_mini_app.md) | 3 | The Reachy shell (MVP: setup + Talk + memory) |
| [reachy_dm_app.md](reachy_dm_app.md) | 3 | Adventure (the DM flagship) |
| seed [AGENTS.md / CLAUDE.md](maxim_pulse_seed/) | — | The `maxim-pulse` monorepo's dev standards + architecture (three layers, React/TS, cardinal thin-over-pymaxim rule) |

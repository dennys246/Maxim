# Reachy Mini App — bootstrap plan (the Reachy shell of the `maxim-pulse` monorepo)

> **✅ SEED CONSUMED 2026-08-01 — the live plan is the `Maxim-pulse` repo (`apps/reachy`, `apps/console`, `packages/kit`). Kept for the front-gate reasoning only; the header's "pre-authorization, not committed work" is no longer true.**

**Status:** Shell plan, drafted 2026-07-23. Pre-authorization — this is the doc to copy into the new repo as its `README`/`docs/plan.md` seed, not committed work.
**Scope:** Ship a one-click-installable Reachy Mini app (a Pollen `ReachyMiniApp` published as a Hugging Face Space) that runs a Maxim agent on the robot, with a first-run **setup page** that guides an owner to either (a) point at their own Maxim leader (mesh) or (b) paste a cloud API key. The app is thin glue over pymaxim; anything hard discovered while building it is fixed **in pymaxim behind the existing seams**, not worked around in the app repo.
**Target:** pymaxim 1.1+ (rides the shipped Reachy WS-era transport + Track 1 live body wiring). App repo versions on Pollen's SDK cadence, independently of pymaxim.
**Gates:** None as a pymaxim release gate. The one hard de-risk gate before promising anything publicly is **P0: does the substrate fit on the Pi** (below).

**Driving idea:** the Reachy app store is full of stateless tricks. Maxim's differentiator is the 1.0 thesis — **a desk robot that learns you across sessions without fine-tuning**, persisted in `~/.maxim/` on the robot. That is the product framing, and it's the one thing no other app in the store does.

---

## Front-gate scope pressure (Principle 3)

**Question:** does this need a new website / phone frontend, and does the app logic need to be its own mechanism, or can it ride on existing infrastructure?

| Candidate | Verdict |
|---|---|
| 1. Build a custom website / phone app for control | **Unnecessary — ride Pollen's dashboard.** A Reachy app is a `ReachyMiniApp` subclass published as a HF Space; owners install + start/stop it from the robot's built-in dashboard (a daemon-served web page, reachable from a phone browser) and Pollen's desktop app. The "website or phone" surface already exists. We ship an app, not a frontend. |
| 2. Reimplement the Reachy control loop in the app | **No — reuse `AgenticRuntimeMixin._start_agentic_runtime`.** The full perception→cognition→action loop, capture manager, and body wiring already exist ([`embodied_runtime/agentic_runtime.py`](../../../src/maxim/embodied_runtime/agentic_runtime.py); Track 1 / PR #400). The app's `run()` is a bootstrap shim + `stop_event` lifecycle. |
| 3. Build LLM routing / key handling in the app | **No — it's `config.json` + lane placement.** Mesh (peer→leader) and cloud both flow through `resolve_setting('lanes.large.…')` and `remote_api_key_ref`. The app writes config; the router does the rest. |
| 4. New session/persistence layer for "remembers you" | **No — `~/.maxim/` on the robot already is it.** Bio-stack persistence + session-end consolidation are the memory. The app's job is to make the session boundary = the dashboard start/stop, and to call `save_cerebellum()` + session-end on stop. |
| 5. Put the app in the pymaxim monorepo | **No — separate thin repo (see below).** |

**Verdict:** **new thin repo, ~glue only.** No new frontend, no new runtime, no new routing, no new persistence. The genuinely new work is: (a) the `ReachyMiniApp` bootstrap + clean shutdown, (b) the **setup page** UX, (c) ARM packaging, (d) headless soak-stability, (e) a privacy posture. Everything else is a pymaxim call.

---

## Add-on vs own repo — own repo, deliberately thin

Recommendation: **its own repo**, for three code-grounded reasons.

1. **The ecosystem forces the shape.** Each app *is* a HF Space with its own packaging, README, and store listing. That is naturally a separate artifact from the pymaxim wheel.
2. **Release-cadence decoupling.** Pollen broke the world at SDK 1.5.0 (zenoh removed) and pins matter — pymaxim's `reachy` extra pins `reachy-mini[gstreamer]>=1.8.3,<2.0` ([pyproject.toml:75](../../../pyproject.toml)). The app tracks Pollen's SDK schedule reactively; pymaxim should not.
3. **The seam already exists.** The `reachy` optional extra and the robot-plugin discovery pattern exist precisely so robot integrations sit outside core. Honoring that keeps the core dep tree clean (no torch/CUDA leaking into a Pi install).

**The discipline that makes "own repo" safe:** the app stays ~a few hundred lines of glue + assets. If it starts accumulating workarounds, that's the signal a pymaxim API has a gap — fix it **in pymaxim** (new keyword on `create_full_agent`, a headless-bootstrap helper, a config-writer verb), not in the app. This is the front-gate principle applied outward.

---

## LLM placement — three destinations, only one is cloud

The Pi cannot host a large local model, but "off the Pi" ≠ "cloud." All three route through the same lane/placement config; the setup page just chooses which one it writes.

| Path | What the setup page writes | For whom | Cost / privacy |
|---|---|---|---|
| **A. Peer → leader on LAN** | `lanes.large.placement` = remote pointing at the owner's leader URL + `remote_api_key_ref` | Owner has a GPU/Mac box on the same network | Free, fully local, the native Maxim topology |
| **B. Peer → leader over Cloudflare tunnel** | Same, but leader URL is the tunnel hostname | Owner's leader is elsewhere (robot at office, box at home) | Free, private to the owner; already hardened — TTFT keepalive + context-admission (llm_timeout_scalability Stage 3) exist *because* of this exact edge-over-tunnel case |
| **C. Cloud provider** | A cloud profile (`cloud.enabled`, Anthropic/OpenAI/… key) | Owner has *only* the robot, no second machine | Per-token cost; data leaves the network |

**Default the setup page toward A/B (the mesh)** — it's the differentiated, private, free story and it *is* Maxim's thesis. **Offer C as the fallback** for the leaderless owner so the app still works out of the box.

Note this is independent of the resource question below: the mesh offloads the **LLM**, but the **substrate** (EC/NAc/hippocampus/sentence-transformers) still runs on the robot regardless of where the LLM lives.

---

## GUI & first-run UX

This is the main new surface. Two design anchors: (1) the app runs **on the robot** — a page it
serves is reachable from a phone/laptop browser but *executes on the Pi*, so it cannot inspect
the browser machine's hardware; (2) the consumer never sees a Maxim internal term ("lane tier,"
"placement," "profile") — one choice, mesh vs cloud.

### Architectural correction — the local-LLM path is a two-device dance, and pymaxim already detects the GPU

A tempting design is "detect a GPU on the computer they're at and guide the tunnel setup here."
**It can't work:** the app isn't running on the leader box, and browsers can't see host hardware.
The local path is inherently robot **+** a second always-on computer that becomes the leader —
and we don't build GPU detection at all, because pymaxim's auto-spawn already does it. When the
owner runs `maxim` on their computer, the GPU is detected, llama-cpp-server spawns, and the
cloudflared tunnel comes up; `maxim doctor` prints the status + leader URL. So the local path is
a **guide + copy-paste command + Test button**, not a hardware wizard. Less to build, more robust.

### Screen map

```
Welcome (first run only — shown when config.json has no resolvable large-tier placement)
 ├─ [Private & free — use your own computer]   → Mesh guide (below) → Test → Save
 └─ [Easiest — connect a cloud key]            → Cloud form (below) → Test → Save
        │
        ▼
Say hello 👋   — smoke test: motion + mic + LLM in one shot (surfaces version/torque/head-frame
        │        traps immediately with a friendly face). Re-runnable from Settings.
        ▼
Main page   (three flagship actions over ONE persistent agent — see reachy_dm_app.md)
 ├─ Status chip:  awake · thinking on <where> · <latency / month-to-date spend>
 ├─ [ 🎲 Adventure ]      — voice DM campaign; the flagship (reachy_dm_app.md)
 ├─ [ 💬 Talk ]           — the live agentic loop; recalls what Adventure taught it
 ├─ [ 😴 Rest ] / [ Wake ] — ProcessingState.AWAKE/SLEEP (modes/definitions.py:29), NOT app-kill
 ├─ ✦ What Maxim remembers about you    — the differentiator (below); Adventure fills it with story
 └─ Settings
      ├─ Re-run setup / switch mesh↔cloud / re-key
      ├─ Spend cap (cloud)
      ├─ Memory:  reset · export
      └─ ▸ Developer drawer:  AUT simulation (fake-percept debug — distinct from Adventure) · logs · doctor
```

### Mesh path (recommended)

Guide the owner to run one command on their desktop/GPU box (pymaxim auto-detects GPU → spawns
llama-cpp + tunnel), then paste the leader URL back:

- Leader URL `[ https://…  or  http://192.168.x.x:8099 ]` · Access key `[ ……… ]`
- **[Test connection]** → pymaxim's `_peer_test(base_url, key, model)` ([doctor/cli.py:459](../../../src/maxim/doctor/cli.py)) — friendly ok/fail + fix hint, no new probe logic.
- On success: writes `lanes.large.placement = remote` + `remote_api_key_ref`.

### Cloud path (fallback)

- Provider `[ Anthropic ▾ ]` · Model `[ 2–3 curated ▾ ]` — **populated from `_BUILTIN_PROFILES`** ([config.py:109](../../../src/maxim/models/language/config.py); `--list-models` already enumerates them), not hardcoded; "Advanced" reveals the full list.
- API key `[ ……… ]` · Monthly spend cap `[ $__ ]` (one number; Advanced exposes pymaxim's per-request/hour/day/month caps at [cloud_dispatch.py:143](../../../src/maxim/models/language/cloud_dispatch.py)).
- **[Test connection]** → one cheap probe call.
- On success: writes `cloud.enabled` + profile + `cloud.session_budget_usd` (default $5, [config_loader.py:241](../../../src/maxim/runtime/config_loader.py)).

### Design constraints (all backed by existing code)

- **Never make the owner run `maxim config set`.** The page writes `config.json` through the sanctioned writer (`runtime/config_writer.py`); keys go to a keyed file and only the **path ref** lands in config (`remote_api_key_ref` — inline plaintext keys are rejected at load).
- **Test before save.** Validate the connection (mesh probe / cloud probe) before persisting, so a bad key fails at setup, not silently at first turn.
- **One-way, low-friction.** Welcome shows only when `config.json` lacks a resolvable large-tier placement; always re-openable from Settings.

### Additions worth building (ranked)

1. **"What Maxim remembers about you."** The differentiator, made visible. A read-only view of consolidated memories / learned preferences / a learned name. Without it the robot reads as a stateless toy; with it, the 1.0 cross-session thesis *is* the product. Highest value on this list — above setup polish.
2. **First-contact smoke test** ("say hello"). Exercises motion + mic + LLM together right after setup, catching the version-mismatch / torque-off / head-frame traps with a friendly face instead of a cryptic unattended failure hours later.
3. **Persistent status chip.** Where it thinks + health/spend, from `maxim doctor`. Trust + debugging in one line.
4. **Wake / Rest.** Real `ProcessingState`; a desk robot needs an obvious sleep that isn't app-kill (killing loses the session boundary → memory not consolidated).
5. **Memory reset + export.** Memory is the product; "forget everything" / "export what you know about me" are privacy-forward and make P3's posture tangible.
6. **Graceful degraded states, shown.** Leader unreachable / budget exhausted → a visible "resting until your PC is back," driven by pymaxim's typed errors — never a silent wedge.

### Simplifications

- **Move "run a simulation" into a Developer drawer.** On a *physical* robot a sim (fake percepts, no motion) is a debug concept — confusing for a consumer. The two consumer actions are **Talk** and **Rest**.
- **No lane/placement/profile jargon in consumer UI.** One decision: mesh vs cloud.
- **Curated cloud model list** (2–3), full list behind Advanced.
- **Land awake and ready** after setup — don't ask "interact or simulate" as a top-level choice.

### The build decision this forces

**Pollen app-config hook vs serve-your-own web UI — RESOLVED: serve-your-own, built as the shared UI kit.** A single key-entry form fits Pollen's config
mechanism; a multi-step wizard + live status chip + memory view + smoke test does not — that wants
a small self-contained page the app serves on a port the dashboard links to. **These views are not
Reachy-bespoke** — they are the first slice of the shared **Maxim Console kit** ([maxim_console.md](../deferred/maxim_console.md)):
SetupWizard, StatusChip, MemoryView, RunSurface. Build them as reusable kit components (Layer 2) binding to
the seams (Layer 1) — the Reachy app is the kit's **first consumer**, and the general localhost console
reuses the same components later. Still verify what the `reachy-mini` SDK affords for serving the page (open question #3).

---

## The app bootstrap (`run()` shim)

Shape (pseudo — real symbols cited):

```python
class MaximReachyApp(ReachyMiniApp):
    def run(self, reachy_mini, stop_event):
        # 1. Ensure config resolvable; if not, surface the setup page and return.
        # 2. Build the agent via the canonical factory (AgentFactory.create_full_agent,
        #    src/maxim/runtime/agent_factory.py:364) with robots.yaml declaring config.body
        #    = bodies/reachy_mini so the bio-stack wires the body (agentic_runtime.py:41).
        # 3. Enter the live loop: _start_agentic_runtime(use_capture_manager=True)
        #    (agentic_runtime.py:112) — the shipped Track 1 path (drift tick, body_state).
        # 4. Honor stop_event: on set, stop the loop, then session-end consolidation
        #    + BioStack.save_cerebellum() so cross-session memory persists in ~/.maxim/.
```

The load-bearing detail is **#4**: the dashboard's stop button becomes the session boundary. Sims end cleanly today; a desk-robot app must treat every stop as a proper session-end so the "remembers you" claim actually holds. Verify `save_cerebellum()` + session-end fire on the `stop_event` path (CLAUDE.md invariants: `BioStack.save_cerebellum()` at session end; lightweight vs full consolidation per the CC8 `is_sim_mode` note — a long-running robot may want **full** consolidation, unlike short sims).

---

## Honest accounting — what's genuinely new vs reuse

| Piece | Reuse or new? | Notes |
|---|---|---|
| Reachy control loop | **Reuse** | `_start_agentic_runtime` (Track 1, #400) |
| Body wiring | **Reuse** | `robots.yaml::config.body = bodies/reachy_mini` |
| LLM routing (mesh + cloud) | **Reuse** | lane placement + `_MaximPeerBackend` + tunnel/proxy |
| Cross-session memory | **Reuse** | `~/.maxim/` bio-stack persistence |
| Connection test | **Reuse** | `maxim doctor` peer probe |
| `ReachyMiniApp` bootstrap + `stop_event` lifecycle | **New (small)** | ~glue |
| **GUI + first-run setup** | **New (the real work)** | welcome → mesh/cloud branch → test → save; served web UI vs Pollen config hook (decision below) |
| **"What Maxim remembers" view** | **New (highest value)** | read-only surface over consolidated memory — makes the cross-session thesis visible. Data source TBD (open q #5) |
| **Smoke test + status chip + wake/rest** | **New (small each)** | reuse `maxim doctor`, `_peer_test`, `ProcessingState` |
| Connection test | **Reuse** | `_peer_test` (doctor) |
| **ARM/Pi packaging** | **New** | lean install: no llama-cpp, no CUDA, **no torch**; `pymaxim[pi]` (= `reachy,console,llm-anthropic,tts`) sized for aarch64. NOTE: `semantic` is deliberately OUT — it declares `torch>=2.1`, so the earlier `[reachy,llm-anthropic,semantic]` combo did NOT meet the no-torch bar. |
| **Headless soak stability** | **New** | hours unattended, no TTY, Wi-Fi blips, cloud timeouts → degrade gracefully |
| Privacy posture | **New (doc)** | listing must state what leaves the robot |

Estimate: a scrappy MVP is **~1–2 focused weeks** *if P0 passes* — most of it UX + packaging, not architecture.

---

## De-risk gates (do these before promising anything)

**P0 — Substrate fits the Pi (HARD gate, do first).** Measure real RSS of a Maxim peer on the wireless Reachy's Raspberry Pi with the large tier **remote** (mesh), substrate **local**: sentence-transformers + EC/NAc/hippocampus (+ spaCy iff concept-decomposition on), sharing RAM with the daemon + GStreamer. If it fits → the private mesh story is real. If it doesn't → fallback is "app is a thin sensor/actuator client, cognition on the owner's box," a worse install story that must be decided consciously. *This is the first thing to run; everything else assumes it passes.*

**P1 — Lean ARM install.** Confirm `pip install pymaxim[pi]` on aarch64 pulls no torch/CUDA/llama-cpp and doesn't time out / fill the SD card in a one-click store install. (The original wording named `[reachy,llm-anthropic,semantic]`, which was self-contradictory: `semantic` declares `torch>=2.1`. The encoder goes on the LEADER per FIT; the peer takes the documented bag-of-words fallback.)

**P2 — Headless soak.** Run the bootstrap for hours with no terminal; inject a Wi-Fi drop and a cloud timeout; confirm the typed-error paths degrade without wedging the loop or corrupting session state.

**P3 — Privacy posture.** The daemon's `/ws/sdk` has **no auth** in 1.8.x, and the app reads mics/camera and (in cloud mode) phones an LLM. Before this faces strangers, write the listing paragraph: what leaves the robot, what's stored in `~/.maxim/`, mesh-vs-cloud data-flow. Reputation in a small ecosystem is set by exactly this.

---

## Open questions / decisions for you

1. **Day-one behavior loop — RESOLVED: Adventure (voice DM).** See [reachy_dm_app.md](../deferred/reachy_dm_app.md). It's the flagship because the body is load-bearing (voice + expressive motion + face-the-speaker + persona + party memory) and it's the most legible showcase of cross-session learning. Talk + Rest support it. The MVP still ships *before* Adventure (Talk + memory view), but the product's headline is Adventure.
2. **Repo name + license + Space owner account.** (Publishing creates a Space under an account with write-scoped token.)
3. **Config surface in Pollen's app model vs a served web UI** — which does the SDK afford? A single form fits Pollen's config hook; the wizard + status chip + memory view + smoke test wants a served page. Verify against `reachy-mini` SDK app hooks before building. *Leaning serve-your-own.*
4. **Full vs lightweight session-end consolidation** for a long-running robot (CC8 `is_sim_mode` trade-off) — likely want full.
5. **Data source for the "What Maxim remembers" view.** What does it read — consolidated Hippocampus long-term memories, NAc learned preferences/biases, a learned name from ATL? Needs a small read-only accessor in pymaxim (a facade verb, not app-side bio-stack poking — keep the app thin). Scope this as the first pymaxim-side addition the app drives.

---

## Seed files for the new repo

Ready to copy into the app repo root:

- [`maxim_pulse_seed/AGENTS.md`](../maxim_pulse_seed/AGENTS.md) — durable architecture + development
  standards + Maxim topology/execution-flow + the Reachy hardware traps. The substantive one.
- [`maxim_pulse_seed/CLAUDE.md`](../maxim_pulse_seed/CLAUDE.md) — operational checks + commands +
  Claude-specific workflow; points at AGENTS.md for the rest.

Both are distilled deliberately — they carry Maxim's *principles* (no-band-aid, front-gate,
two-lens review, verify-actuation, config-over-env, dormancy), not the full invariant ledger,
because keeping the app repo thin is itself one of the standards.

## References (code-grounded)

- [`embodied_runtime/agentic_runtime.py`](../../../src/maxim/embodied_runtime/agentic_runtime.py) — `AgenticRuntimeMixin._start_agentic_runtime` (loop), `config.body` body-wiring gate
- [`runtime/agent_factory.py`](../../../src/maxim/runtime/agent_factory.py) — `create_full_agent` (:364)
- [`runtime/config_writer.py`](../../../src/maxim/runtime/config_writer.py) — sanctioned `config.json` writer (setup page writes through this)
- [`docs/embodiment/reachy_mini/`](../../embodiment/reachy_mini/README.md) — WS-era transport, connect, motion, audio (the hardware truth)
- [`perception_pipeline_placement.md`](../perception_pipeline_placement.md) — self-contained (all-local) vs distributed placement framing
- pyproject `reachy` extra — `reachy-mini[gstreamer]>=1.8.3,<2.0` ([pyproject.toml:75](../../../pyproject.toml))
- Pollen: [Make and publish your Reachy Mini App](https://huggingface.co/blog/pollen-robotics/make-and-publish-your-reachy-mini-apps) · [Building & Publishing Apps (HF docs)](https://huggingface.co/docs/reachy_mini/SDK/apps) · [reachy-mini-desktop-app](https://github.com/pollen-robotics/reachy-mini-desktop-app)

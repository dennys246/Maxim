# Reachy App — Maxim-side seams

**Status:** Shell plan, drafted 2026-07-23. The Maxim-repo counterpart to the app-repo docs [reachy_mini_app.md](reachy_mini_app.md) (MVP) + [reachy_dm_app.md](reachy_dm_app.md) (Adventure flagship). Those *reference* these seams; this plan *specifies and builds* them.
**Scope:** The pymaxim-side work the Reachy app rides on. Per the "keep the app thin — fix hard things in pymaxim" cardinal rule, the real engineering lives here. Mostly **facades exposing existing internals** + **one real refactor** (the persistent-agent injection, inside HANDLE) + **packaging**. No new cognitive machinery.
**Target:** MVP-enabling seams (HANDLE, RECALL, PROBE, SETUP, PKG) for the 1.1-era app MVP; DM-track seams (VOICE, CONTENT) post-1.1 alongside [reachy_dm_app.md](reachy_dm_app.md).
**Gates:** **FIT (substrate fits the Pi)** is the hard prerequisite for the whole app — do it first. Nothing here is a pymaxim *release* gate; it's additive integration surface.

**Seam IDs are mnemonic** (not sequential): each names what it delivers. Prerequisite: `FIT`. MVP: `HANDLE`, `RECALL`, `PROBE`, `SETUP`, `PKG`. DM-track: `VOICE`, `CONTENT`.

**These seams are Layer 1 of the shared UI stack** ([maxim_console.md](maxim_console.md)) — the backend contract that **both** the Reachy app *and* the general Maxim Console compose over. They are **not Reachy-specific**: `SETUP`/`PROBE`/`RECALL`/`HANDLE` back the shared UI kit's SetupWizard / ConnectionTest / MemoryView / RunSurface. Every kit component binds to a seam here (never a shell-specific back-channel) — that's what lets the full dashboard be composition, not a rewrite. **`HANDLE` has two flavors** — embodied (Reachy) and headless (console) — same interface; fold the headless constructor into the HANDLE spec below.

**Cross-cutting requirement — every seam is exposed over `maxim serve` with a TYPED OpenAPI model** (surfaced by the maxim-pulse bootstrap review; full rationale in [maxim_console.md](maxim_console.md) § Backend). The kit's TypeScript `FacadeClient` is **generated** from `maxim serve`'s OpenAPI schema, so each seam's request/response must be a **typed Pydantic model — no bare `dict`/`Any` on the wire surface** — or the generated TS type is useless and Python↔TS drift goes silent. When you author a seam here (`RECALL`'s return blend, `PROBE`'s result, `SETUP`'s inputs, `HANDLE`'s mode calls), define its typed model as part of the seam; that model IS the cross-repo contract. This is a hard requirement, not polish.

---

## What already exists (do NOT rebuild)

| Need | Existing surface |
|---|---|
| Live Reachy loop | `AgenticRuntimeMixin._start_agentic_runtime` ([agentic_runtime.py:112](../../src/maxim/embodied_runtime/agentic_runtime.py)) |
| Build a full agent | `AgentFactory.create_full_agent` ([agent_factory.py:364](../../src/maxim/runtime/agent_factory.py)) |
| DM campaign entry | `api.campaign(path=...)` ([api.py:1025](../../src/maxim/api.py)) + `DMRuntime` |
| Model list (structured) | `api.list_models()` → `ModelInfo` ([api.py:250](../../src/maxim/api.py)) — read this, not `_BUILTIN_PROFILES` |
| Diagnostics (structured) | `api.diagnose()` → `DiagnosticReport` ([api.py:791](../../src/maxim/api.py)) |
| Config write path | `config_writer.write_config` ([config_writer.py:96](../../src/maxim/runtime/config_writer.py)) |
| TTS / STT | `TTSEngine` (piper) + `faster-whisper` (`audio` extra); `agent_loop` consumes `pending_voice_input` |
| Fiction-vs-fact provenance | `Episode.imagined` / `tag_imagined_links` / `decay_imagined_links` ([orchestrator.py:2740](../../src/maxim/simulation/orchestrator.py)) |
| Orient-to-speaker | Exp 45/48 DoA + head-frame; [perception_pipeline_placement.md](perception_pipeline_placement.md) |

The scope below is what these do **not** yet cleanly provide.

---

## FIT — substrate-fits-the-Pi measurement (do first, blocks everything)

Not code. Measure real RSS of a Maxim **peer** on the wireless Reachy's Pi with the large tier **remote** (mesh) and substrate **local**: sentence-transformers + EC/NAc/hippocampus (+spaCy iff concept-decomposition on), sharing RAM with the daemon + GStreamer. Pass → the private mesh story is real. Fail → the app becomes a thin sensor/actuator client with cognition on the owner's box (a worse install story, decided consciously). **Everything else assumes this passes.**

---

## MVP-enabling seams

### HANDLE — embeddable persistent-agent handle (runtime + the campaign injection + clean stop)

This is the load-bearing seam: it's the MVP's runtime + persistence mechanism **and** the DM injection surface. It absorbs what was a separate "persistent-agent injection" item, because the handle *is* the injection point.

**Front-gate:** ride existing infra? **Mostly — `api.run` + `_start_agentic_runtime` exist; the campaign-injection params already exist at the `campaign_runner` layer.** The gap is an embeddable handle + the stop contract; no new runtime, no public-API change, no sibling verb.

**Work:** expose "build one persistent embodied agent (body = `bodies/reachy_mini`), run the live loop, and stop cleanly" as an embeddable handle the app's `ReachyMiniApp.run(stop_event)` drives — not CLI-only. The handle carries the **modes** over one persistent agent: `handle.talk(...)`, `handle.play_campaign(path)`, `handle.rest()`. On stop: **full** session-end consolidation + `BioStack.save_cerebellum()`.

**(a) The campaign injection — the cross-mode-learning seam (was its own item).** *Problem (confirmed):* `simulation/orchestrator.py:534` builds `agent_id="sim_aut"` with session-scoped Hippocampus/NAc (`~/.maxim/sessions/{id}/aut_hippocampus.json`). Adventure run on that plumbing learns into a throwaway the Talk agent never reads — the "Adventure teaches Talk" promise fails silently. *Resolution:* `campaign_runner` already accepts `hippocampus` / `nac` / `memory_hub` / `agent_id` (`run_fixture_campaign(aut_hippocampus=…, aut_nac=…, aut_memory_hub=…)`; the generative runner takes `nac` + `agent_id`), and `api.campaign` already supports `party_mode` = "NPC agents with real memory". So the throwaway is *only* because `api.campaign` / the orchestrator **construct** `sim_aut`. **The app must not call `api.campaign`** — `handle.play_campaign(path)` passes the *live persistent* agent's bio-stack into `campaign_runner`. `api.campaign` stays as-is (a throwaway is correct *for a standalone script*). Point `tag_imagined_links` / `decay_imagined_links` at the persistent agent so fiction decays while the shared episode + player-model persist as real. Do **not** fork `DMRuntime`.

**(b) The stop contract — decouple consolidation from the percept-source flag (do NOT add `is_external_adapter`).** Make session-end an explicit `consolidation: "full" | "lightweight"` choice, **default `"full"`**; sims opt into `"lightweight"`. Rationale: `is_sim_mode` already conflates "external adapter drives percepts" *and* "end lightweight"; a third mode adds a fourth conflation. The two target methods already exist side-by-side (`on_session_end()` full at [memory_hub.py:667](../../src/maxim/integration/memory_hub.py), `on_session_end_lightweight()` at [memory_hub.py:827](../../src/maxim/integration/memory_hub.py)) — the fix is *which one is called*, chosen by an explicit param, not inferred from a proxy flag (the sim_adapter docstring already names this as the intended path). Default-full is the "count silent failures" choice: wrongly-lightweight loses consolidation *silently*; wrongly-full is a slightly slow shutdown that's *loud and harmless*. **Migration cost:** sim callers that today get lightweight implicitly (via `is_sim_mode`) must pass `consolidation="lightweight"` explicitly — pin it with a CC8 test.

**Regression guards:** (1) `handle.play_campaign(...)` for one turn against a *persistent* agent → the resulting episode is recallable from that agent's Hippocampus (same `agent_id` + home), not merely a session-scoped AUT file; an `imagined=True` in-fiction fact decays while a real episodic memory does not. (2) a headless build→run→stop cycle persists to `~/.maxim/` and reloads on next construct; stop invokes **full** consolidation by default; a caller passing `consolidation="lightweight"` still gets the lightweight path (CC8 preserved).

### RECALL — "what Maxim remembers about you" read facade

**Front-gate:** ride existing infra? **No *consumer-shaped* read verb exists** — devs have `api.observe`/`Observer` for raw introspection; the app must not poke the bio-stack directly (thin-app rule).

**Work:** a new read-only `api.py` verb returning structured data (per the API rules — return data, not prints). **A curated, typed blend keyed to Adventure — NOT raw episodes or raw NAc floats.** Shape:

```
{ name: str | None,
  player_model: [trait],                          # NAc biases → human-readable traits
  story_memories: [{summary, when, salience}],    # high-salience NON-imagined episodes
  preferences:   [{about, learned_from}] }         # conservative, confidence-gated
```

Three rules make it work: **(1) provenance-filter** — show real memories; in-fiction facts stay hidden (they're decaying anyway); **(2) rank by salience/valence, not recency** — "your rogue betrayed the party" over "you entered a room"; **(3) summarize, don't dump**.

**Load-bearing caveat — the view's credibility IS the product's credibility.** The whole thesis is "it remembers you," so a confidently-wrong inferred preference does more damage than showing less. Translating NAc reward-biases into stated preferences is a small inference that can misfire: gate it behind a confidence threshold, phrase tentatively, and **under-claim** when in doubt ("we've adventured twice") rather than over-claim. Scope to exactly what the day-one behavior (Adventure) produces.

**Regression guard:** the verb returns a typed structure over a populated bio-stack; empty bio-stack returns an empty structure, never raises; an `imagined=True` fact never appears in `story_memories`; a below-threshold bias never appears in `preferences`.

### PROBE — structured connection test

**Front-gate:** ride existing infra? **`_peer_test` exists but returns `int` + `print()`s** — unusable from a GUI.

**Work:** a facade returning a structured pass/fail + fix-hint (reuse `_MaximPeerBackend.for_url(...).health_check()` + `classify_probe_outcome` under the hood, same as `maxim doctor`). Covers both the mesh probe and a cheap cloud-key probe. Backs the setup page "Test connection" button.

**Regression guard:** structured result for ok / auth-fail / unreachable, mocked offline.

### SETUP — config setup-write convenience verbs

**Front-gate:** ride existing infra? **`write_config` is generic** — the app should not hand-assemble nested lane/cloud dicts or know the `remote_api_key_ref` rules.

**Work:** thin helpers over `config_writer` (single-writer invariant preserved): "write mesh placement" (`lanes.large.placement` remote + `remote_api_key_ref` file) and "write cloud profile + budget" (`cloud.enabled` + profile + `cloud.session_budget_usd`). Keys land as refs via `atomic_write_secret`; inline plaintext rejected (existing load-time rule). Routes through the sanctioned writer only.

**Regression guard:** each helper produces a config that `resolve_setting` reads back as a resolvable large-tier placement / cloud profile; secret lands as a ref, never inline.

### PKG — ARM/Pi packaging hygiene

**Front-gate:** ride existing infra? **Extras exist but the aarch64 lean combo is unverified.**

**Work:** confirm `pymaxim[reachy, <one llm extra>, semantic, audio, tts]` installs on aarch64 with **no** torch/CUDA/llama-cpp pulled; verify faster-whisper / ctranslate2 / piper wheels exist for ARM; if a lean combo is awkward, add a curated extra. A one-click store install must not time out or fill the SD card.

**Regression guard:** an aarch64 install smoke (CI matrix or documented manual) asserting the heavy backends are absent.

---

## DM-track seams (post-1.1, with [reachy_dm_app.md](reachy_dm_app.md))

### VOICE — voice-loop plumbing + (conditionally) STT placement

**Front-gate:** ride existing infra? **STT/TTS exist; the placement abstraction exists.** New work is only the loop wiring + a conditional cut point.

**Work:** wire the push-to-talk voice loop (start/stop-capture signal → STT → transcript → `DMRuntime` response → TTS), driven by HANDLE's `play_campaign`. **Only if** the FIT/latency spike shows Pi-side whisper is the bottleneck: wire STT as a placeable perception stage per [perception_pipeline_placement.md](perception_pipeline_placement.md) so it runs on the owner's leader. Turn-taking needs no new mechanism (push-to-talk + the DM's normal response — see reachy_dm_app.md Risk #1).

**Regression guard:** a mocked loop turn (fake audio → fake STT → DM → fake TTS) advances campaign state; latency-spike numbers recorded in the plan, not asserted.

### CONTENT — campaigns + safety surface

**Front-gate:** ride existing infra? **Campaign schema + imagination exist; guardrails do not.**

**Work:** 2 hand-authored seed campaigns (the floor; imagination extends), plus a documented moderation stance for improvised narration to families. Scope minimally; let real Adventure usage (per the deferred [dungeon_master_extensions.md](deferred/dungeon_master_extensions.md) revive gate) drive depth.

---

## Ordering

```
FIT   substrate fits the Pi          ← blocks everything; do first
─── MVP app (reachy_mini_app.md) ───────────────────────────────
SETUP   config setup-write verbs   ┐
PROBE   structured conn test       ├─ setup page
RECALL  what-it-remembers facade   ├─ main page differentiator
HANDLE  runtime handle + stop      ├─ Talk + clean persistence + the play_campaign injection surface
PKG     ARM packaging              ┘  (parallelizable)
─── Adventure (reachy_dm_app.md), post-1.1 ─────────────────────
VOICE    voice loop (+ STT placement if latency demands)   — rides HANDLE.play_campaign
CONTENT  campaigns + safety
```

**Note:** the persistent-agent injection lives inside **HANDLE** — `handle.play_campaign` *is* the injection surface. It's the **highest-value single behavior** (it's what makes cross-mode learning real), so HANDLE's `play_campaign` path could be exercised to demo the transfer thesis before the full voice loop (VOICE) exists.

## Design decisions (resolved)

1. **Campaign injection — inject at `campaign_runner`, not a new/changed public verb.** Params already exist; `handle.play_campaign` passes the live bio-stack; `api.campaign` unchanged for standalone use. Lives inside HANDLE.
2. **Session-end consolidation — decouple, don't add `is_external_adapter`.** Explicit `consolidation: "full" | "lightweight"`, default `"full"`; sims opt into lightweight. (HANDLE part b.)
3. **Memory-read shape — curated typed blend, not raw.** Name + player-model + salience-ranked real story memories + confidence-gated preferences; provenance-filtered; under-claiming. (RECALL.)

Remaining sub-decisions (deferred to build, not blocking):
- Exact `player_model` trait vocabulary + the confidence threshold for surfacing a preference (RECALL).
- Whether `story_memories` summaries are pre-stored episode summaries or a per-view LLM summarization pass (RECALL).

## References

- App-repo plans: [reachy_mini_app.md](reachy_mini_app.md) · [reachy_dm_app.md](reachy_dm_app.md) · seed [AGENTS.md/CLAUDE.md](maxim_pulse_seed/)
- [perception_pipeline_placement.md](perception_pipeline_placement.md) (VOICE STT placement) · [deferred/dungeon_master_extensions.md](deferred/dungeon_master_extensions.md) (CONTENT depth gate)

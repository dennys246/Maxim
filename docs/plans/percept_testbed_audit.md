# Sim percept/behavior testbed — four-facet audit + scoping

**Date:** 2026-07-17. **Question:** can a declarative "percept-channel manifest"
(`{modality, source, enabled, gain, noise}` per channel) ride the existing `PerceptSource`
seam so we can **ablate and scale percept channels** in sims to measure their behavioral
impact — with the **audio/DoA orienting percept** as the first case? Four read-only parallel
audits: percept ingress, ablation/scaling knobs, mode/abstraction divergence, measurement
surface. Every claim below is grounded in the facet reports' file:line.

## Verdict (front-gate)

**Do NOT build a new percept-channel manifest as conceived.** The goal is valid and there is a
real, smaller, high-value gap — but the specific "manifest rides `PerceptSource`" design fails
the front-gate on four independent counts:

1. **It collides with active work.** `docs/plans/perception_pipeline_placement.md` is *active*
   (not deferred; #383 merged, #384/#385 open) and already owns a planned `config.json::perception`
   declarative surface, with the **audio conventions already resolved** (`"audio"` modality tag,
   frozen-centroid — already in `ec.py:216`, normalize-at-the-sensor to `[-1,1]`) **and an explicit
   anti-over-engineering guardrail** ("do NOT introduce a new AxisSpec type or axis config schema;
   ride the body YAML"). A fresh manifest is a two-source-of-truth collision + re-derivation of
   settled conventions.
2. **`PerceptSource` is not a uniform seam.** `sim_adapter.next_observation` **flattens a `Percept`
   into a text-only dict** one hop past the source, dropping `modality`/`sensory`/`metadata`/
   `salience`/`novelty`; the loop takes exactly **one** `percept_source` (no multiplexer); and
   interoception + drives travel a **parallel body path** (`_read_drive_states` off
   `executor.embodiment`) that never becomes a `Percept`. Substrate-primary reads *only* the body
   path — so a `PerceptSource`-only manifest is inert in the LLM-free orienting mode.
3. **The audio-orient signal isn't a `PerceptSource` percept.** Azimuth is **double-represented by
   design**: an exteroceptive EC cluster where **sign is preserved** (the *direction* state,
   substrate-primary only) + an interoceptive centeredness drive where **sign is folded** via
   `abs()` (the *magnitude/reward*). Both are already declared in `reachy_mini.yaml`. A flat
   `{enabled, gain, noise}` entry is ill-defined for such a straddling channel (kill the state? the
   reward? both? — and one `gain` hits the signed embedding and the folded pain differently).
4. **We can't yet measure per-channel impact anyway.** Telemetry can detect *whole-session*
   behavioral deltas (`actions.jsonl` + the exp37/41/42 analyzer pattern) but is **blind to
   per-channel attribution** (EC modality is a 3-way `linguistic/drive/sensor` bucket that collapses
   distinct sensor channels), and the orient/azimuth behavioral signals live **only in the
   standalone `scripts/orient_backbone/` harness**, invisible to the sim orchestrator.

**The instinct is right; the mechanism is wrong.** The valuable core survives if scoped down and
merged into the active perception plan rather than started as a parallel framework.

## Facet findings (condensed)

### 1. Percept ingress — the seam isn't uniform
- `PerceptSource` (`simulation/sources.py:111`) is the right *protocol shape*; `AzimuthDoASource`
  (`embodiment/audio_localization.py:64`) already conforms and is synthetic-testable (inject a fake
  `doa_reader`, no hardware).
- **THE flatten:** `sim_adapter.py:110-122` reduces the `Percept` to `{source, transcript, cli_input,
  hard_override, raw_transcript_text}` — structured fields die here.
- Single `percept_source` per loop (`agent_loop.py:981`, `orchestrator.py:1609`) — no composite.
- Dual ingress: Percept path (text/vision) vs body/drive path (intero/azimuth-as-drive). Substrate-
  primary suppresses text percepts (`bridge.py:136`) and reads the body directly.
- `EmbodimentPerceptSource` (the obvious interoception `PerceptSource`) is **Dormant, never wired**.

### 2. Ablation/scaling knobs — sprawl sharing only a parser
- ~12 `MAXIM_DISABLE_*`/`ENABLE_*` flags, each wired at ~3 sites (producer read + conftest scrub +
  harness/plan recipe). Shared: the truthy parser `annotation_disabled_via_env`
  (`prompts/cluster_bias_annotation.py:75`) — but **3 sites re-copy even that**. No registry, no
  schema, no declarative list.
- A genuinely-reusable typed mechanism EXISTS and is the project's stated-preferred path:
  `SimConfigSection` + `_FIELD_TO_ENV` + `resolve_setting` (`runtime/config_loader.py:281`, CLI > env
  > config.json > default, provenance for free) — but the ablation family predates/ignores it.
- Two divergent harness "arm" idioms: `benchmark_cross_session.py`'s `ARM_ENV` env-delta table vs
  `benchmark_exp42_preference.py`'s arm=arc-name + shell-exported env.
- **Exp 44 is the cautionary tale:** one arm's config is scattered across ~6 places, and the arm
  identity is *reconstructed post-hoc* by reading env vars back out (`_ablation_arm()`).
- The channel vocabulary already exists: `SensoryModality` enum + `make_text/scene/intero/audio_percept`
  (`agents/modality.py:102`, `agents/percept_factory.py`).

### 3. Mode/abstraction divergence — the deep complication
- Same percept, disjoint code by mode: llm-primary sees body state as **auto-sense prompt text**
  (`agent_loop.py:1385`); substrate-primary sees it as an **EC cluster** via
  `SensorEncoder.encode_sensors` (`agent_loop.py:869`). The EC front door is **substrate-primary only**
  (`agent_loop.py:1184`).
- Sensor channels **straddle percept + reaction with opposite sign semantics** (EC preserves sign =
  direction; drive folds sign = magnitude/reward). `enabled`/`gain`/`noise` are ill-defined as single
  scalars.
- Landmines: `_normalize_value` **discontinuity at exactly 0** (`encoder.py:419-421`) — a centered
  azimuth reads as an extreme; `tick_vital_drift` auto-recenters world-set sensors unless the read
  overwrites each tick; three modality vocabularies with a lossy default map (SOUND/INTERO→`"text"`).
- **Active adjacent plan:** `perception_pipeline_placement.md` owns `config.json::perception` +
  resolved audio conventions + the ride-the-body-YAML guardrail. Align, do not duplicate.

### 4. Measurement — can see *that* behavior changed, not *which channel*
- Rich per-run artifacts (`report.json`, `actions.jsonl` v1.1 with `entity_class`, `aut_nac.json`
  which **does** persist `cluster_reward_bias` at `nac.py:2805`, `bio_telemetry.jsonl`) + per-tick
  `substrate_telemetry` (**gated on substrate-primary + `--research`**) + `sim_ec_activation` trace.
- Blind spots: (a) **no per-channel provenance** — EC modality/`modality_tag` is a 3-way bucket that
  collapses azimuth vs thermal into `"sensor"`; (b) `cluster_reward_bias` only in the end-of-session
  dump, not per-tick; (c) no continuous orient signals (relief `potential_diff`, latency-to-center,
  gain) in the orchestrator path — they exist **only** in `scripts/orient_backbone/live_3_learn.py`;
  (d) metric extraction is hand-rolled per experiment (no generic extractor).

## Recommended scoped moves (ride existing infra; let the audio experiment earn each)

Instead of a manifest framework, three smaller, independently-valuable moves — each grounded in the
audio-orient hypothesis so a real experiment earns it:

- **M1 — Consolidate ablation flags into the config.json section pattern.** Migrate the input-channel-
  ish `MAXIM_DISABLE_*` flags (imagination, imagination-substrate-signal, deterministic-scene,
  body_state) onto `SimConfigSection`-style typed fields + `resolve_setting`. One declarative surface,
  provenance for free, one scrub. Pure debt-paydown; makes experiments reproducible. (This IS the
  "standardize" the goal wants, minus a new mechanism.)
- **M2 — A per-run active-config record.** One artifact in the session dir that names the arm's
  toggles + arc + seed + (eventually) channels — the thing Exp 44 reconstructs post-hoc today. Cheap,
  high-value, prerequisite for any per-channel analysis.
- **M3 — Per-channel telemetry provenance + un-silo the orient metrics.** Finer than the 3-way EC
  modality bucket, plus surface `cluster_reward_bias` per-tick and port the `orient_backbone` orient
  metrics (direction-correctness, az_bin distribution, summed `potential_diff`) into the orchestrator
  telemetry — so an ablation can be *attributed* to the audio channel. This is the real measurement
  gap; without it ablations show deltas with no provable cause.
- **M0 (prerequisite decision) — read `perception_pipeline_placement.md` and decide: extend it, or
  scope a sibling.** The channel enable/scale surface belongs in / next to `config.json::perception`,
  not a parallel file. If the "manifest" survives at all, it survives as fields on that surface.

## Grounding experiment (what earns all of the above)

**"Does the audio-orient percept change behavior, and does scaling its salience/pain matter?"** — the
ablate-and-scale question already on our path (Layer 2 feeds azimuth; Track 2 is the reflex). It
needs M2 (record the arm) + M3 (measure orient behavior in-orchestrator), and it exercises exactly
the straddling percept/reaction + mode-split complications, so it validates the design against the
hardest case instead of the abstract. Build the minimum M2/M3 that makes *this* experiment clean; the
general testbed falls out of serving it.

## Open decisions for the user

1. **M0:** extend `perception_pipeline_placement.md`, or scope a sibling plan? (Recommend: read it
   first, then extend — it owns the neighborhood.)
2. **Scope now vs later:** do M1 (ablation-flag consolidation) as standalone debt-paydown, or fold it
   into the audio-experiment scaffolding so it's earned by a live need?
3. **Two manifests or one:** accept that percept-channels and drive/interoception channels are
   *different declarative surfaces* (the audit says they are — opposite sign semantics, different
   ingress), or invest in unifying the body path onto `PerceptSource` (un-dormant
   `EmbodimentPerceptSource`) first? (Recommend: two surfaces; unifying is a much larger, separate
   commitment.)

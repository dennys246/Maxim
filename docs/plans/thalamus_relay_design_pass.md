# Thalamic relay — pre-implementation design pass (the fork, decided)

**Status:** Design decision (2026-07-17). Resolves the grow-vs-subsume fork sketched in
[thalamus_hypothalamus_framing.md](thalamus_hypothalamus_framing.md) and answers the four questions
that framing note deferred to "the first design pass." **Still `[engineering]`** — this decides
*shape*, not that the relay has *earned* its name; the audio-orient experiment
([percept_testbed_audit.md](percept_testbed_audit.md) grounding experiment) is what graduates it.

**Discipline honored:** no class hierarchy ahead of need. This doc commits the project to a
**direction** (subsume) and a **first slice that introduces no new coordinator class and touches no
fragment**. A three-lens critique (Architecture + Executor + bio-fidelity — the discipline that caught
the Track 1 findings) ran against the first draft and **found four blocking corrections**; they are
folded into the decisions below and recorded in the final section. The headline: the "attach the
Percept to the observation dict = zero blast radius" idea in the draft was **wrong** (it leaks into
persisted state), and the first slice is slightly less trivial than "un-flatten + dumb multiplex." The
subsume *direction* survived all three lenses unchallenged.

**Live-half gating:** the structural fixes below are code-writable now (offline, synthetic
`AzimuthDoASource`, no hardware). The *earning run* — feed live DoA, measure orient behavior — waits
on the Reachy motor repair. So this design pass + the first-slice build are the actionable half; the
experiment that validates them is the gated half.

---

## Decision 1 — SUBSUME, not grow; and the first slice has no coordinator at all

**Verdict: Option B (a thin coordinator subsumes the fragments), realized incrementally so the
coordinator does not exist until a second need forces it.**

Grounding for rejecting Option A (grow `ThalamicGate`):
- `ThalamicGate` ([default_network/gate.py:92](../../src/maxim/default_network/gate.py), ctor `:108`)
  is a **Percept → `EscalationResult`** decision with adaptive thresholds, attention locks, and
  goal/interest biasing (`evaluate`, `set_active_goal`, `set_interests`, `adapt_thresholds`). Its
  entire job is *"escalate this DN vision percept to the LLM?"* — the reactive-vision layer's
  cortex-gate. No substrate route, no multiplex, no channel set.
- Growing it into a universal relay would make `default_network/` reach into `memory_hub`,
  `SensorEncoder`→EC, and the agent-loop percept fork — a **DN layer-boundary violation**. Moving the
  gate out of DN to avoid that *is* subsume wearing grow's clothing. And the other fragments aren't
  under the gate, so growing it unifies nothing. **(Architecture lens: affirmed.)**

**What "subsume" actually spans — recounted after the Architecture lens (2 components + 1 call-site,
not "3 fragments").** `ThalamicGate` and `BioEnrichmentPipeline` (`enrich()`) are standalone classes
with clean surfaces — genuinely wrappable. The third "fragment," the exec_agent thalamus→PFC path, is
**not an independent component**: it is `ExecAgent._run_pre_deliberation`
([agents/exec_agent.py:1275](../../src/maxim/agents/exec_agent.py)), a private method that invokes
`self._bio_enrichment_pipeline` inline. Subsuming it is a **call-site redirect through the
coordinator, not a zero-touch wrap.** This matters because "everything just gets wrapped, no rewrite"
was the load-bearing evidence for "subsume is low-risk" — it is accurate for the two components and an
overstatement for the one call-site. (The first slice touches none of them, so this is a north-star
correction, not a slice blocker.)

**First slice (no `Thalamus` class):**
1. **Un-flatten the sim percept via a typed side-channel** (Decision 2) so modality/salience/the
   Percept object survive the sim boundary **without entering the observation dict**.
2. **A `CompositePerceptSource`** — multiplex N child `PerceptSource`s behind the single-source loop
   seam ([agent_loop.py:1319](../../src/maxim/runtime/agent_loop.py),
   [orchestrator.py:1609](../../src/maxim/simulation/orchestrator.py)). It **is a `PerceptSource`**, so
   the loop is untouched — the multiplexer hides inside the existing seam. Its forwarding contract is
   spelled out in Decision 2 (the Executor lens found the draft's spec crashes the orchestrator).

**Home for the coordinator when it earns introduction: a new top-level `perception/` package** — NOT
`default_network/` (layer violation), NOT `agents/` (that's the Percept *type*'s home, not its
routing), NOT `runtime/` (already the god-package). **The first slice does NOT create `perception/`**;
`CompositePerceptSource` is a concrete multiplexer that lives in a sibling module under `simulation/`
or `embodiment/` next to the concrete sources (`AzimuthDoASource`, `EmbodimentPerceptSource`), **not**
in `simulation/sources.py`, which is protocol-only (Architecture NIT). The `perception/` name is
*reserved* to coordinate with the active
[perception_pipeline_placement.md](perception_pipeline_placement.md) `config.json::perception` **config
section** (that plan creates a config surface, not a package — so this is a future coordination point,
not an existing sibling to align with).

---

## Decision 2 — un-flatten via a typed side-channel, NOT a dict key (the draft's "additive key" leaks into persistence)

The flatten at [sim_adapter.py:110-122](../../src/maxim/runtime/sim_adapter.py) reduces a `Percept` to
`{source, transcript, cli_input, hard_override, raw_transcript_text}`. The dropped fields already exist
on `Percept` (`salience`, `novelty`, `sensory` typed `SensoryTag`, `modality`, `substrate_node_id`,
`embedding` — [agents/bus.py:197-254](../../src/maxim/agents/bus.py)). So this is "stop discarding data
the type already carries," not "add new data."

**The draft's plan — attach the object under an additive `observation["percept"]` key — is REJECTED.
It does not have zero blast radius; it leaks the object into persisted state.** The corrected
blast-radius survey (Architecture + Executor lenses, independently):
- `agent_loop.py:1320` calls `state.update(observation)`; `RuntimeState.update`
  ([runtime/state.py:22-26](../../src/maxim/runtime/state.py)) does `self.data.update(observation)` —
  a **whole-dict absorb**. Any new key lands in `state.data` and is never popped.
- `state.data` is then serialized: `_persist_state_json` (`agent_loop.py:1048/3877/3918` →
  `state.save_json` → `atomic_write_json`) **and** every captured episode's `state_snapshot`
  ([memory/hippocampus.py:768](../../src/maxim/memory/hippocampus.py), via a `deepcopy`). It doesn't
  crash only because `atomic_write_json` passes `default=str` — so a `Percept(...)` `repr` string (with
  the full embedding list) gets written into `~/.maxim/sessions/.../state.json` and every episode
  snapshot, on every persist, with a deepcopy of the embedding on every `snapshot()`. Silent
  pollution, not an exception — and the draft's proposed golden test would not catch it (the leak is
  one hop downstream in `state.update`).
- The survey also had two errors the lenses caught: `integration/bio_enrichment.py:49` **does** read a
  structured field (`observation.get("salience", 0.5)`, silently getting the default today), and
  `agent_loop.py:217` is the **non-sim** `environment.observe()` path, not a sim-observation consumer.

**Decided carrier: a typed accessor on the adapter, off the dict.** `next_observation` stashes the
current `Percept` on the adapter (`self._current_percept`) and exposes it as `sim.current_percept`
(returns `Percept | None`). The observation dict stays **scalar-only** (five text keys, byte-identical),
so `state.update` → `state.data` → persistence stays clean. New thalamic/substrate consumers read the
typed handle, not a dict key. This also sidesteps the second carrier problem: on the **non-sim path the
observation IS a bare `Percept`** with no `.get` ([agent_loop.py:1659](../../src/maxim/runtime/agent_loop.py),
`NullSimulationAdapter.next_observation` returns `environment.observe()`) — a `"percept"` dict key
would `AttributeError` there; a `current_percept` accessor scoped to the adapter does not.

**Guardrails folded:**
- `current_percept` is `None` on the idle/legacy path, and a *non-None* percept can still carry `None`
  `modality`/`sensory` ([bus.py:235,247](../../src/maxim/agents/bus.py)) — consumers tolerate `None` at
  both levels.
- Regression test pins: the five text keys byte-identical vs a pre-change golden; `state.data` gains
  **no** new key after `next_observation` + `state.update` (the anti-leak guard); `sim.current_percept`
  is the object the source emitted, `None` on the idle path.

---

## Decision 3 — `enabled` is one channel gate at the routing fork; `gain` deferred

The mode split is real: the same channel reaches cognition by **different delivery** per AUT mode —
llm-primary via auto-sense prompt text / `BioEnrichmentPipeline`
([agent_loop.py:1385](../../src/maxim/runtime/agent_loop.py)); substrate-primary via
`SensorEncoder.encode_sensors` → EC cluster ([agent_loop.py:869](../../src/maxim/runtime/agent_loop.py))
with text percepts suppressed ([bridge.py:136](../../src/maxim/simulation/bridge.py)).

**Semantics decided:** `enabled` is a **single per-channel boolean** ("this channel does / does not
reach cognition"), applied at the **routing fork** (relay chooses LLM vs EC), *not* at the source —
gating at the source would force the source to know the AUT mode, which the CC8 `PerceptSource`
contract forbids (**Architecture lens: affirmed clean**). A disabled channel is dropped from whichever
route the active mode uses, so `enabled:false` yields the operator's "as if unsensed" in both modes
with no per-mode config. A channel with only a substrate route (pure interoceptive drive, no text
surface) makes `enabled:false` in llm-primary a correct no-op — the coordinator records which route it
actually gated in the M3 telemetry so ablations are attributable.

**`gain` is NOT a single scalar and is deferred out of the first slice** — the audit is right that one
`gain` hits the signed relay route and the folded drive route differently. When gain earns
introduction (with the coordinator), it is **per-route**: the relay owns a *relay gain* on the
exteroceptive→EC/LLM route; the *motivational* gain already lives in the right place — the drive's
`pain_scale` in the body YAML (hypothalamus-side). **First slice ships `enabled` only.**

---

## Decision 4 — substrate routing re-opens the azimuth dual cleanly ONLY after two preconditions on the relay's own EC route (Bio-fidelity lens: the draft's "no third representation" was false)

The azimuth channel is double-represented by design, both already in
[reachy_mini.yaml:61-98](../../src/maxim/_data/components/bodies/reachy_mini.yaml): a **signed EC
cluster** (*where the sound is* — thalamic "where") and a **sign-folded centeredness drive**
(`evaluate_failures` folds `|azimuth|` for pain — hypothalamic magnitude). The draft claimed routing
azimuth "to the substrate" reuses the existing signed route and adds no third representation. **The
bio-fidelity lens refuted this against the code:**

- The only `encode_sensors` call site ([agent_loop.py:871](../../src/maxim/runtime/agent_loop.py))
  passes `sensors=drives` with the **default `modality="interoception"`**
  ([similarity/encoder.py:534](../../src/maxim/similarity/encoder.py)). `drives` comes from
  `_read_drive_states` ([agent_loop.py:711-728](../../src/maxim/runtime/agent_loop.py)), which sweeps
  **every** drive-spec — and azimuth has a `drive:` block — so `drives["azimuth"]` is today folded into
  the **interoception** embedding alongside hunger/thermal. That is not a thalamic/exteroceptive route;
  it is the interoceptive bundle. `encode_sensors`' own docstring
  ([encoder.py:544-556](../../src/maxim/similarity/encoder.py)) says exteroceptive azimuth must pass
  `modality="audio"` precisely so it forms a *separate* cluster space.
- So the earning experiment's clean azimuth→`"audio"` EC route would land azimuth in EC **twice**
  (audio + interoception) plus the drive-pain fold = **three representations** — unless azimuth is
  **de-bundled from the `_read_drive_states` interoception sweep** (or modality-filtered out of it)
  first.

**Corrected clean-routing rule (with preconditions):** the relay's substrate route delivers the raw
signed azimuth to EC **under `modality="audio"`**, and the relay does **not** own the sign-fold (it
stays in the drive/hypothalamus path). This adds no third representation **iff**:
1. **Azimuth is de-bundled from the interoception `_read_drive_states` sweep** so it is encoded once, as
   audio, not also as interoception.
2. **`_normalize_value`'s zero-aliasing is fixed** — and this landmine is **on the relay's own EC
   route, not drive-side** (the draft mis-located it). `_normalize_value`
   ([encoder.py:405-424](../../src/maxim/similarity/encoder.py)) is used only by `_sensor_embed` →
   `encode_sensors` (the EC route), never by the drive-pain path (which uses `abs(current - set_point)`
   at [embodiment/body.py:243](../../src/maxim/embodiment/body.py)). Worse than a point discontinuity:
   the `if v < 0.0` branch **excludes exactly 0.0**, so `-1.0 → 0.0` **and** `0.0 → 0.0` produce the
   *identical* embedding — "centered" (the orient success state) is indistinguishable from "hard left."
   The signed EC route Decision 4 depends on **cannot represent the orient state at all** until this is
   fixed.

**Genuinely orthogonal (Bio-fidelity: confirmed):** `pain_scale: 1.0`
([reachy_mini.yaml:97](../../src/maxim/_data/components/bodies/reachy_mini.yaml)) is consumed only on
the drive-pain path (`body.py:246` → PainBus → NAc reward), never on the EC route — a hypothalamus-side
calibration (drop toward 0.2–0.3 when the feed lands), independent of the relay.

**Honesty caveat, kept adjacent to the label (Bio-fidelity NIT):** the "signed = thalamic *where* /
folded = hypothalamic *discomfort*" dual has an asterisk — the "hypothalamic discomfort" leg is really
a **collicular/orienting** signal wearing homeostatic-drive machinery (the YAML's own BIO NOTE
`:70-79`). The relay design does not launder that; the long-term fix (a distinct orienting drive-kind,
or routing the error through salience/attention rather than pain/homeostasis) is a drive-system
question, out of this design pass.

---

## The first slice — concrete, minimal, offline-testable (revised after the review)

| # | Change | File | New surface? | Risk |
|---|---|---|---|---|
| 1 | `next_observation` stashes the Percept on the adapter; `sim.current_percept` accessor. Observation dict stays scalar-only (5 text keys byte-identical). | `runtime/sim_adapter.py` | one typed accessor, no dict-key change | ≈0 (state.data untouched — the leak is avoided) |
| 2 | `CompositePerceptSource` — multiplex N child sources behind the single-source seam, with the forwarding + termination contract below | new module under `embodiment/` or `simulation/` (NOT `sources.py`) | one concrete class, no ABC | low (opt-in; inert until a 2nd source attaches) |

**`CompositePerceptSource` forwarding + termination contract (the Executor lens found the draft spec
crashes the orchestrator):**
- **`next_percept()`** — deterministic **ordered scan, first non-`None` child wins** (a *dumb*
  multiplexer; **not** "priority-drains" — the draft simultaneously proposed and, one section later,
  banned priority selection; priority is exactly the N=1 policy-bake that earns the coordinator, not
  the first slice). `capabilities` unioned.
- **Duck-typed method fan-out is per-child `hasattr`-gated** — `advance_step`, `has_pending`, **and
  `inject_cli`** (the draft omitted the last). `inject_cli` ([bridge.py:142](../../src/maxim/simulation/bridge.py),
  **un-guarded** — an `AttributeError` there kills the orchestrator turn loop) is the *primary*
  percept-delivery path in the orchestrator (~17 call sites). A composite must forward it **to each
  child that implements it**; for the first slice only the conversational child implements `inject_cli`
  and the `AzimuthDoASource` does not, so `hasattr`-gated fan-out routes it correctly with no
  addressing. **A second `inject_cli`-implementing child is the explicit trigger for the coordinator's
  addressed routing** — the first slice must assert/​document the single-implementer assumption, not
  silently broadcast. `advance_step` fan-out is *correctness* for scripted children (else they never
  advance); `has_pending` is an optimization (safe default `True`).
- **`is_exhausted()` excludes perpetual-live sources.** Naive "all-children-exhausted" breaks
  termination: a live `AzimuthDoASource` returns `is_exhausted() == False` forever
  ([sources.py:134-141](../../src/maxim/simulation/sources.py) pattern), so pairing it with a scripted
  child that *does* exhaust would make the composite never exhaust → the adapter's 180s-grace shutdown
  ([sim_adapter.py:130-157](../../src/maxim/runtime/sim_adapter.py)) never fires and a scenario-complete
  sim can't self-terminate. Rule: exhaustion is driven by the **exhaustible (scripted) children**;
  perpetual-live sensors do not veto termination.

**Explicitly NOT in the first slice:** no `Thalamus`/coordinator class, no `perception/` package, no
per-channel `gain`, no config surface, no addressed `inject_cli` routing (single-implementer only), no
fragment touched, and **none of the Decision-4 azimuth work** (de-bundle, `_normalize_value` fix,
`modality="audio"` route) — that is the *earning experiment's* prerequisite set, gated on the DoA
feed / motor repair, tracked in Decision 4, not built speculatively here.

**Guards the first slice ships with:**
- `test_sim_adapter_unflatten` — five text keys byte-identical vs golden; **`state.data` gains no new
  key** after `next_observation`+`state.update` (anti-leak); `sim.current_percept` is the emitted
  object; `None` on the idle path.
- `test_composite_percept_source` — ordered first-non-None drain across two synthetic children;
  `inject_cli`/`advance_step` fan out only to implementing children (azimuth child untouched by
  `inject_cli`); a two-`inject_cli`-implementer composite raises/warns (single-implementer assertion);
  `is_exhausted()` true only when the scripted child is exhausted despite a perpetual-live child;
  `capabilities` unioned; `isinstance(composite, PerceptSource)` (CC8 conformance).

**What it unblocks:** attach the synthetic `AzimuthDoASource`
([embodiment/audio_localization.py:64](../../src/maxim/embodiment/audio_localization.py)) as a second
child in a sim — the audio channel reaches the loop with modality preserved, no hardware — so M2
(per-run active-config record) + M3 (per-channel telemetry) have something real to record and the
earning experiment can be scaffolded offline ahead of the motor repair (its Decision-4 preconditions
still gate the *live* run).

---

## Three-lens critique — findings folded

_Architecture + Executor + bio-fidelity, run against the first draft. Four blocking corrections; the
subsume direction survived unchallenged. Findings are folded into the decisions above; recorded here._

**Architecture lens.** Affirmed Decision 1 (home choice, the layer-violation rejection of growing
`ThalamicGate`, "first slice creates no coordinator class") and Decision 3's gate-at-the-fork. Blocking/
should-fix: the blast-radius survey was incomplete and its "none reads a structured field" sub-claim
false (`bio_enrichment.py:49` reads `salience`); "three fragments wrapped unchanged" oversells the one
that is a call-site (`exec_agent._run_pre_deliberation`), not a component → recounted as **2 components
+ 1 call-site**; "round-robins / priority-drains" contradicted the dumb-multiplexer fold → dropped to
ordered first-non-None; `perception/`-package "alignment" is prospective (the placement plan is a config
section, not a package) → softened to "reserve the name"; `CompositePerceptSource` should not live in
protocol-only `sources.py`.

**Executor lens (two blocking).** (1) **"The observation dict is never serialized" is refuted** —
`state.update(observation)` → `state.data` (`state.py:26`) → `_persist_state_json`
(`agent_loop.py:1048/3877/3918`) + hippocampus `state_snapshot` (`hippocampus.py:768`); `default=str`
masks it into silent `repr`-pollution. **Fix folded into Decision 2: carry the percept on a typed
side-channel, keep the observation dict scalar-only.** (2) **The composite's forwarding contract omits
`inject_cli`** — the un-guarded primary percept path (`bridge.py:142`); a doc-spec composite
`AttributeError`s on turn 1. **Fix folded into the first-slice contract: `hasattr`-gated fan-out
including `inject_cli`, single-implementer assertion, coordinator earns addressed routing.** Should-fix:
survey undercount + `:217` miscategorized (non-sim path); the live-child exhaustion break → the
exclude-perpetual-live termination rule; deeper `None`-safety (non-None percept, `None` modality).

**Bio-fidelity lens (one blocking).** **Decision 4's "no third representation / reuses the existing
signed route" is false** — the only `encode_sensors` route bundles azimuth into `modality="interoception"`
via `_read_drive_states`, so the clean audio route adds a third EC representation unless azimuth is
de-bundled first; and `_normalize_value`@0 is **on the relay's EC route, not drive-side** (and aliases
`-1.0`≡`0.0`, not merely a point discontinuity). **Both folded into Decision 4 as explicit
preconditions.** Confirmed as correct: `pain_scale` is genuinely drive-side/orthogonal; the collicular
caveat is preserved not laundered (NIT: keep the asterisk adjacent to the "hypothalamic discomfort"
label — folded); the relay stays `[engineering]` with no behavioral claim (NIT: restate Decision 4 as
"clean *iff* the two preconditions" rather than "stays clean" — folded).

---

## Related

- [thalamus_hypothalamus_framing.md](thalamus_hypothalamus_framing.md) — the organizing frame; this doc decides its fork.
- [percept_testbed_audit.md](percept_testbed_audit.md) — the four-facet audit; M2/M3 are the measurement half this slice unblocks.
- [perception_pipeline_placement.md](perception_pipeline_placement.md) — the orthogonal placement axis + the `config.json::perception` surface the eventual `gain`/`enabled` config rides.
- [embodiment_runtime_wiring.md](embodiment_runtime_wiring.md) — Track 1 (body wired, merged #400); the runtime this lands in.

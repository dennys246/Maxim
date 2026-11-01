# Experience-clock decay: NAc decay and body drift on the agent's world time

> **PROPOSED 2026-10-17, plan only ([#1361](https://github.com/dennys246/Maxim/issues/1361)).** No
> `src/` change yet. All owner decisions below were made 2026-10-17; everything else is
> design within those decisions plus the owner's Q1–Q9 answers (same date). Every mechanism enters as `[engineering]`.
> Three-lens review (Executor, Architecture, Wire integrity) folded 2026-10-17.
> **Supersedes** [deferred/scn_decay_anchoring.md](deferred/scn_decay_anchoring.md) (its revive path
> already pointed here: "scale NAc per-tick decay by `dt` on the memory line's experience clock").

**Owns (proposed):** `decisions/nac.py` (the decay methods, `NACConfig` taus, `apply_wall_clock_decay`,
the temporal windows), `runtime/experience_time.py` (dt without a Hippocampus), `runtime/loop_gates.py`
(`_loop_live_tick`, `tick_embodiment_drift`), `runtime/agent_loop.py` §8.5 (`_loop_bio_tick_maintenance`),
`embodiment/body.py` (`Embodiment`: drift out of `evaluate_failures`), `integration/memory_hub.py`
(`decay_all(0.95)`), `simulation/scenario_source.py` (`ScenarioSource` step declaration).
**Companion plans:** [memory_strength_and_forgetting.md](memory_strength_and_forgetting.md) (built the
clock; its §Open questions 1 is amended in this PR); [archive/cluster_reward_bias_decay_tau_split.md](archive/cluster_reward_bias_decay_tau_split.md)
(the calibration this plan converts); [deferred/decay_consolidation_calibration_plan.md](deferred/decay_consolidation_calibration_plan.md)
(would later learn the taus); [deferred/transition_based_drive_pain.md](deferred/transition_based_drive_pain.md)
(revive trigger "any change to `evaluate_failures` cadence" = S4); [fear_learning.md](fear_learning.md)
(#1072, extinction; Risk 2).

## Motivation

### What a "tick" is today (measured, 4-reviewer audit + #1361)

NAc decay is called "TICK-anchored" in `NACConfig`, but a tick is any `pre_tick_gate` pass that is
not idle and reaches §8.5. Polling decides whether a pass happens, not anything the agent lives through:

| Runtime | What paces §8.5 | Consequence |
|---|---|---|
| Orchestrator text sim (AUT) | `_submitted_recently` holds the loop awake for 120 s after any LLM submit, polling ~2 Hz; `loop_substrate` re-stamps `last_llm_submit_time` every substrate tick, so a substrate-primary AUT is awake **forever** | Decay ∝ wall time ∝ LLM latency + host load. Golden: `goal_reward_bias = 0.2 × 0.98^N`, N = 8 locally, 7 under xdist (run 39072778282) |
| Embodied runtime | 10–30 Hz loop | 50-pass tau ≈ 1.7–5 s of activity |
| Survival loop windows | `minecraft_harness.run_minecraft_aut` runs `run_agentic_loop`, so `_loop_live_tick` runs there in realtime mode | the `_substrate_tick_due` docstring's "no LLM worker" concern is about wake-up, not decay pacing |
| Planning-liveness callers | attempt state, not the 120 s window | passes still count, not time |
| Any loop | §8.5 is **skipped** on eventful passes (confirmation / timeout / caps `continue`) and **runs** on empty polls | decay anti-correlated with events |

Further clock defects in the same computation:

- **Two clocks in one credit computation.** Eligibility decays ×0.9 per pass; temporal anchors and
  pending events age on a 300 s **wall** window (`temporal_window_seconds`, `time.time()` in
  `record_event` / `_record_outcome_locked` / `decay_eligibility`). Fast polling expires the trace while
  the anchor survives, so trace credit silently becomes anchor credit.
- **Body drift uses `time.time()` deltas** (`Embodiment.evaluate_failures`, `dt = now − _last_poll`).
  `_last_poll` is not reset across an operator pause, so the first evaluation after a pause drifts
  through it; in text sims drift ∝ LLM latency.
- **`decay_all(0.95)` once per session end** (`MemoryHub.on_session_end` and `_lightweight`): ten short
  sessions erode link confidence to 0.60, one long session of the same experience to 0.95.
- **Offline forgetting.** `NAc.load()` applies `apply_wall_clock_decay(now − saved_at)`: 1-day half-life
  for `cluster_reward_bias`, 7-day for `reward_bias` / `goal_reward_bias` / `percept_valences` /
  `cluster_fear`. A robot switched off for a week wakes with half its learning gone.
- **Out-of-loop rigs** (`scripts/survival_world/water_trial.py` train, `exp58_run.py`,
  `exp58_offline_gates.py`, the T1-13/14/15 rigs) call `propose_via_substrate` directly, build their own
  driver and discard `on_live_pass()`'s return: they get no NAc decay today.

The decayed magnitude is persisted (`aut_nac.json`) and reloaded, so these confounds cross sessions.

### Evidence exposure (no live claim compromised)

Track-C checks (2026-10-17) on committed data. Each verdict means **no evidence of asymmetry, not
symmetry shown**: no run stamped ticks, submits or latency.
- **T1-6 (Exp 42): CLEAR, inferred.** `n_actions` per arm within ~2%, `safe_pref` SD 0; arms ran in
  blocks; Exp 42 §2a notes wall-clock body-drift dt.
- **T1-9 (Exp 52 Phase B): CLEAR on arm means**, ~27 ticks/turn in every arm. Most exposed:
  exploration-visit decay pacing (tau 50), plausible contributor to per-seed irreproducibility
  (|margin| ≈ 0.12 vs a 0.11 floor).
- **T1-2 (Exp 37, STALE):** decay does not explain the L8 shift; the save→resume gap (100 s–5 ks) is
  confounded with arm, harmless only because sims load with `apply_decay=False`. Re-fire: §Impact.
- Exp 56/57 hand-tick NAc; T1-13/14/15 rest on `cluster_fear` (no tick decay), exposed via load-time
  decay (S1), the rigs (Q9) and T1-13's trigger wording.

### Front-gate scope pressure: ride on the existing clock

*Does this need its own mechanism?* No. `runtime/experience_time.py::ExperienceClockDriver` already
computes, once per live pass (after the pause check, before the idle gate), the world experience that
elapsed: turns × quantum in turn-based worlds, monotonic time minus paused time in real-time ones, with
a 300 s stall cap. `Hippocampus.experience_clock` rides on it and `StrengthStrategy` reads it;
`on_live_pass()` already **returns** the gained µs. This plan consumes that value, and adds no clock, no
bus and no SCN surface. SCN is rejected (owner decision 1): it bins by host time of day and never sees
world time.

## Decisions (owner, 2026-10-17)

1. **"Anchor in-session NAc decay (and body drift) to the ExperienceClockDriver, not to gate passes or SCN."**
2. **"OFFLINE: NO forgetting while the agent is switched off — experience time, not wall clock"** —
   remove or make opt-in the 1–7 day offline wall-clock decay in `apply_wall_clock_decay`, including
   `_cluster_fear`'s 7-day offline decay.

Answers to the plan's open questions (owner, 2026-10-17), now decided:

- **Q1:** "taus in world-seconds, anchored on Exp 30's fixture regime; tune the turn quantum, never per-runtime taus."
- **Q2:** "idle time COUNTS toward decay."
- **Q3:** "`decay_all(0.95)` becomes time-based (S5); per-episode consolidation belongs in `sleep()`."
- **Q4:** "body drift happens ONLY through `Embodiment.advance(dt)` on the experience clock;
  `evaluate_failures()` stops drifting, so out-of-loop callers must call `advance()` themselves (API
  note + CHANGELOG). No wall-clock fallback."
- **Q5:** "don't extend 'no offline forgetting' to Hippocampus/ATL now; memory-strength owns it" (the
  inconsistency stays listed in §(f)).
- **Q6:** "the survival world uses world seconds (the ExperienceClockDriver's real-time mode)"
  (Exp 62 freezes daylight, so `time_of_day` cannot be the clock).
- **Q7:** "offline wall-clock decay becomes opt-in + Dormant; `apply_decay=` keeps working through
  1.4.x and is removed in 1.5.0."
- **Q8:** "a fixture 'turn' is step-clocked (1/target_hz)."
- **Q9:** "wire the survival rigs (water_trial, exp58_run, exp58_offline_gates) through one live-tick
  call (option a), so they decay and drift like the loop."

## Design

### (a) The clock source per runtime

| Runtime | Driver kind | One unit of experience | Gap |
|---|---|---|---|
| Orchestrator / generative / interactive text sims | `turns` via `ConversationalSource` (`experience_us_per_turn = 5_000_000`) | one world turn = 5 s | none; substrate-primary counts turns via `mark_turn` |
| Fixture sims (`--sim <yaml>`, `ScenarioSource`, all `timing: step_based`) | **falls back to `realtime`** today | wall time | step-clocked, S2 (Q8, (a′)) |
| Fixture sims, `timing: relative` | `realtime` | wall time, by the mode's own definition (`at` is seconds since start) | none |
| Survival world (loop windows via `run_minecraft_aut`) | `realtime`, minus pause | world seconds | none (Q6) |
| Survival out-of-loop rigs | own driver, `percept_source=None` → `realtime` | world seconds, return discarded today | wired through one live-tick call (Q9) |
| Robot / embodied runtime | `realtime`, minus pause; system sleep excluded by `time.monotonic` | world seconds | none |

**Driver without a Hippocampus (D1).** Today `on_live_pass()` returns 0 when `clock is None`, and
`_loop_bio_handles` documents the driver as inert with no Hippocampus. After S3 that would silently stop
NAc decay and body drift. S3a decouples them: the driver **always computes dt** and advances the
Hippocampus clock only if one exists. S0 enumerates, by a test over the builders (`build_bio_stack`,
`agent_factory`, `maxim.create.agent`, orchestrator, `run_minecraft_aut`, embodied runtime, the Q9
rigs), which runtimes reach the loop with a NAc but no Hippocampus. That list goes into this section
before S3a.

**S0 result (2026-10-17): none in production.** Every `run_agentic_loop` caller's hub comes from
`build_bio_stack`, which always builds a Hippocampus: `cli.py`, `console/handle.py`, `api.py` and the orchestrator AUT
through `create_full_agent(with_bio_stack=True)` (which replaces the skeleton's memory with the bio-stack's, so even
`remembers=False` keeps a Hippocampus); `run_minecraft_aut`, the embodied runtime and the three Q9 rigs through
`build_minecraft_aut`/`build_bio_stack`; the orchestrator's second loop call and `simulation/tools.py` pass no hub.
`interactive.py::_run_scenario_turn` passes its caller's hub through and `run_interactive_sim` has no caller. The only
NAc-without-Hippocampus is the bare `create_agent` path (`maxim.create.agent`, `create_npc_agent`, `maxim.load.agent`
with `remembers=False, learns=True`), and it gets **no MemoryHub at all**. That is the contract since #1522 (owner
decision 2026-10-31): the hub needs both a Hippocampus and a NAc, so an agent without one has none, its shutdown saves
its stores itself, and it gets no session-end decay or consolidation (whether a partial agent should is #1366's
remaining question). Before #1522 the failed build was also mislabelled as corruption. Pinned by `test_experience_clock_decay.py::test_s0_which_builders_give_a_nac_without_a_hippocampus`
(builders) and a code-reading audit of the callers. D1 stays: the loop gates run on a hub without a Hippocampus, so
they hold S3a to it regardless.

### (a′) A fixture "turn" is a step (Q8)

`ScenarioSource` `step_based` timing counts **loop iterations**: `at` is a step index, and
`sim.next_observation()` (agent_loop §1) calls `advance_step()` (`SimAdapter`). The fixture world is
therefore pass-clocked by construction. Its declaration (S2): `experience_turns()` returns the step
counter and `experience_us_per_step` comes from the YAML, defaulting to `round(1e6 / target_hz)` =
500 000 µs at 2 Hz. That makes the τ_passes→τ_us conversion **exact** in Exp 30's regime. What
still varies with latency is LLM decisions per step, a property of the agent, not of the clock.

**Today decay and step come apart, by code reading.** `next_observation()` advances the step
*before* the confirmation and timeout `continue`s, so an eventful pass advances the world but skips
§8.5. "Decay and world events cannot come apart per pass" becomes true only after S3a moves decay
into `_loop_live_tick`.
- Idle passes call neither, so steps freeze while idle. That is consistent with a step-clocked world.
- `_loop_live_tick` runs before `next_observation`, so `dt` lags the step by one pass, the same in
  every arm.
- **G2a is red today.** It runs a fixture with credits injected at fixed steps and varies only the
  eventful-pass pattern, then asserts identical maps at step K. It tests decay/step alignment, not
  the turn definition. Exp 30 / Roy-3a ran on fixtures, which is why it is S6's reference.

### (b) How decay receives Δexperience (S3a)

- `_loop_live_tick` keeps `dt_us = experience_driver.on_live_pass()` and passes it to **one** NAc entry
  point, `nac.advance_experience(dt_us)`. §8.5's `_loop_bio_tick_maintenance` is deleted, and its
  extraction tests in `test_loop_section_extraction.py` are **rewritten** to pin the new call, not
  deleted. Decay then runs on idle and eventful passes alike, and the pass count stops mattering:
  `exp(−a/τ)·exp(−b/τ) = exp(−(a+b)/τ)`.
- `advance_experience(dt_us)` applies `exp(−dt_us/τ_us)` per map: `_eligibility`, `_reward_bias`,
  `_goal_reward_bias`, `_cluster_reward_bias` minus `_inherent_bias_keys`, `_percept_valences`,
  exploration visits. It keeps the 0.001 / 0.01 prune floors and is a no-op at `dt_us == 0`.
  `_cluster_fear` stays **excluded** (Wire 4 invariant (a)). Contracts it inherits:
  - **Containment:** if A1's per-map containment has landed, each map's decay is its own handler,
    reported with its own `site=`. One raising map skips no others. Moving today's single
    `log_swallowed_exception(operation="nac_per_tick_decay")` must keep it reported
    (`lint_no_silent_swallows` de-instrumentation check).
  - **A2's provenance:** every bias removal prunes `_cluster_reward_source`. That covers the orphan
    sweep that now lives in `apply_wall_clock_decay` and `load()`, which must survive both the move
    and default-off wall decay.
- Per-map `decay_*` methods take a **required keyword-only `dt_us`** (forgetting it is a `TypeError`).
  A caller that always passes 0 is caught by the stall guard below, not by the signature.
- **NAc stall guard (D1).** NAc counts, per session, its credit writes, its `advance_experience`
  calls and the experience µs they added; the call also records the driver kind.
  - **Firing point:** a new `MemoryHub._assert_nac_experience_advanced(results)`, called **last on
    both session-end paths** (`on_session_end` and `on_session_end_lightweight`), after the NAc save,
    **not gated on the Hippocampus**. Every runtime that writes loop credit reaches it: the loop
    obtains its NAc only through a `MemoryHub` (`_loop_bio_handles`), and the Q9 rigs end through
    `aut.bio.on_session_end()` → `MemoryHub.on_session_end`.
  - It **logs at ERROR before raising** a typed `NAcExperienceStalled` (keyword-only `__init__`,
    carries `results`), copying `_assert_experience_advanced`'s rule: session-end callers catch broad
    `Exception`.
  - **Stalled means:** credit was written AND either (i) `advance_experience` was never called
    (nothing drives this NAc), or (ii) the kind is realtime, it was called at least twice, and the
    added µs is 0. A realtime session of one pass is exempt, because the driver's first pass only
    starts the count. A turn-based session with zero turns is honest and exempt.
  - Since Q9 wires the rigs, the guard applies to them too.
- **Migration surface:**
  - `agent_loop.py`
  - `scripts/behavioral_convergence_exp{1,2,3_tier2,4_tier3}.py`. They call `decay_eligibility(factor=0.5)`;
    the `factor` keyword is replaced by `dt_us` with a per-call τ override, or the scripts pass
    `dt_us = −τ_elig·ln 0.5` (recommended: the latter, no override surface).
  - Test callers: `test_p2_reward_modulation`, `test_inherent_bias_class`, `test_substrate_exploration`,
    `test_wire_a_cluster_bias_annotation`, plus those in §Impact.
  - (`scripts/gaze_substrate/1_operant_redirection.py` only mentions decay in a docstring.)
- Ordering change, deliberate: decay now runs *before* the pass's credit writes.

### (c) Eligibility and temporal anchors on ONE clock (S3b)

NAc keeps a session-scoped `_experience_us`, advanced only by `advance_experience` (same deltas as the
Hippocampus clock, own origin, not persisted, like the anchors). Pending events and anchors are stamped
with it, and the age checks in `record_event`, `_record_outcome_locked` and the anchor prune compare
against it. `temporal_window_seconds` → `temporal_window_us`, keeping a **deprecated alias** for
one release. Its users are `simulation/foundry.py` (`NACConfig(temporal_window_seconds=60.0)`),
`test_build_default_network`, `test_statistic_shapes`, `test_pain_bus`, `test_phase0_harness`,
`test_sim_embodiment`, `test_sem_execution_production`, `test_imagination_wiring`,
`docs/experiments/protocols/15*` and `docs/decisions.md`.
- **Lock contract:** if A1's `eligibility_snapshot()` has landed, the distributor reads through it, so
  stamping and pruning stay under NAc's lock and the snapshot never sees a half-advanced state.
- `TemporalSignature`'s SCN phases still score anchor *similarity* on host time (`decisions/temporal_credit.py`
  anchor fallback); only age and expiry move. The bio-memory brief's SCN-coupling invariant
  (trace expiry → anchor fallback) is updated in S3b, because the expiry moment moves.
- A standalone NAc that is never advanced never ages anything; tests advance it explicitly.

### (d) Body drift on the same clock, including pause (S4)

`Embodiment.advance(dt_s)` becomes the single caller of `tick_vital_drift` **and keeps `_note_drift`
inside it**. That feeds `outcome_window` (drift, `drift_dt_s`) → executor → `sem.py::interoceptive_outcome`,
which nets drift out of GL2a records and stores drift/`drift_dt_s` in `extra`. Moving drift without
`_note_drift` would make these structurally 0, so S4 ships a guard test on the netting (a nonzero
drift reaches the GL2a record and is netted out). `_loop_live_tick` calls `advance(dt_us/1e6)` in
**both** modes. `tick_embodiment_drift`'s substrate-primary early return goes, `evaluate_failures`
becomes evaluate-only, and `_last_poll` (and the pause leak) is deleted.
- **Public behaviour change:** `maxim.create.embodiment()` documents `.evaluate_failures()` as the
  tick. S4 updates that docstring and adds a `changelog.d/` fragment (M83 step 3; formerly an `[Unreleased]` line) plus an API note.
- **Out-of-loop callers must call `advance()` themselves** (Q4; no wall-clock fallback). Each one is
  migrated in S4 with its own world dt:
  - `runtime/substrate_proposal.py::propose_via_substrate`'s `evaluate_failures()` (today's
    substrate-primary drift source, and the rigs' drift source too). It becomes evaluate-only, and
    the loop's or rigs' live tick drifts instead.
  - `simulation/foundry.py`, `embodiment/percepts.py`, `embodiment/tool_bridge.py`, `simulation/tools.py`
  - `scripts/survival_world/dark_danger_probe.py` and `scripts/check_oscillator_coldstart.py`
  - the Q9 rigs, through the shared live-tick call
  For the survival rigs drift is moot: every drive in `_data/components/bodies/minecraft_player.yaml`
  has `drift_rate: 0.0` (world-owned).

### (e) `decay_all(0.95)` per session → per experience time (S5, Q3)

Replace the session-end haircut with a link-confidence term in `advance_experience`,
`τ_link_us = −S_ref / ln 0.95`, where `S_ref` is the median experience per session in the reference
regime (S0); decided time-based (Q3). `decay_imagined_links(0.5)` stays at session end: it is a provenance penalty.

### (f) Offline: no decay on load; persisted files (S1)

- `NAc.load()` stops calling `apply_wall_clock_decay`. Dormancy over deletion: the method and its
  half-life fields stay, opt-in behind `NACConfig.offline_wall_decay: bool = False`, docstring
  `Dormant since <date>: owner decision 2026-10-17, no offline forgetting`. Under
  `offline_wall_decay=True` every map that decays today, `_cluster_fear` included, decays as before; the default
  (False) decays nothing offline (Q7; G1 holds both arms).
- `load(apply_decay=...)` and `load_safe(apply_decay=...)` keep the keyword **through 1.4.x** (Q7)
  with a `DeprecationWarning`, and are removed in 1.5.0. **Amended 2026-10-18 (owner decision at the S1
  review):** an explicit `apply_decay=True` / `False` decides offline decay for that one load, whatever
  the config, and omitting it means `NACConfig.offline_wall_decay` (default off). Behaviour tier: "no
  offline forgetting" is the default policy, and the Dormant opt-in (config or keyword) can override it.
  The callers include `hivemind/cli.py`'s `load_safe`. Loaders to update:
  - `runtime/bio_stack.py` `nac.load_safe()` (the main loader, decay on today)
  - `agent_factory._create_nac` `nac.load()`
  - `session.py` (an observer; A1 fixes it)
  - `api.py`, `load.py`, `orchestrator._restore`
  - `scripts/orient_backbone/exp53_cross_context_readout.py` (passes `apply_decay=False`, so it must
    keep working while deprecated), `orient_demo.py`, `live_3_learn.py`
  - the `bio_stack.py` comments
- **`saved_at` is re-justified, not "no change".** It stays written, as a diagnostic and for the opt-in.
  The hivemind paths that cite "the decay clock" are re-justified under opt-in: `hivemind/ingest.py`
  drops the donor stamp, `merge.py::_later_saved_at` and `cli.py merge-nac` keep it. So are their tests
  (`test_hivemind_ingest`, `test_hivemind_merge`, `test_substrate_merge_nac_cli`) and `exp61_run`'s
  `saved_at`.
- **No `_format_version` bump:** the shape is unchanged and the taus live in `NACConfig`. Files saved
  under old pacing load verbatim, and the CHANGELOG says so (there is no inverse).
- **Stated inconsistency (kept by Q5):** Hippocampus `sleep()` and ATL consolidation on the **default** path
  still forget by `time.time()` against `accessed_at`, so downtime counts there.

### (g) What stays wall-clock on purpose

| Site | Why |
|---|---|
| Heartbeat, stall detector, LLM timeouts, TTFT keepalive | watch the process and backend |
| `idle_sleep_s`, `target_period`, `llm_submit_interval`, the 120 s `_submitted_recently` window | loop scheduling; no longer pace decay. The forever-awake AUT becomes a CPU cost only (follow-up issue) |
| PainBus refractory, reflex cooldowns | motor/IO rate limits; hardware safety (embodiment brief §1) |
| SCN phases, `TemporalSignature` similarity | host calendar by design |
| `REALTIME_PASS_CAP_US` | the driver's own stall bound |

`agents/memory_agent.py::_apply_decay` and `proprioception/focus_learner.py::_apply_decay` are
out of scope here. They are in scope of the enforcement lint below, which either converts them or
allowlists them with a reason.

## Tau conversion and recalibration

Today's taus are in **passes** with factor `(1 − 1/τ)` per pass: `reward_bias` / `goal_reward_bias` 50,
`percept_valences` 200, `cluster_reward_bias` 300, exploration 50, eligibility ×0.9; anchor window
300 s wall. Exp 30 confirmed the tau=300 fit within 0.3% in the fixture regime.

**Conversion:** `τ_us = τ_eff × m`, with `τ_eff = −1/ln(1 − 1/τ)` (50 → 49.5; ×0.9 → 9.49) and `m` the
experience-µs per pass in the reference regime. With (a′) option 3, `m = experience_us_per_step` (0.5 s)
and the fixture regime converts exactly.

1. **S0 measures** per pass `dt_us`, whether §8.5 ran, and the driver kind, in three regimes on two
   backends each: (i) Exp 30 / Roy-3a-retry fixture, (ii) the Exp 42 config, (iii) an Exp 60 survival
   window.
2. **Reference (Q1, decided):** one set of taus in world-seconds, converted from Exp 30's fixture
   regime. Regime (iii) is checked against a range stated in the S0 PR before measuring. If it falls
   outside, the turn quantum is tuned, never per-runtime taus. Worked numbers: `τ_reward ≈ 25 s`
   world; a ConversationalSource turn (5 s) then decays `reward_bias` over 5 turns, against ~1.85
   turns today (50 / ~27 passes per turn): **~2.7× slower per turn** in text sims.
3. **Validation (engineering, deterministic):** G2/G3 flip. **Behavioural check:** Exp 30's decay-fit
   re-run under converted taus, as a pure re-run (no design review); any new claim needs its own prereg.
4. **Every tau consumer moves in the S3a PR:**
   - `scripts/survival_world/r2_learned_bias.py::FROZEN_CONFIG`, which hard-asserts today's taus and
     `cluster_bias_wall_decay_half_life_s=86400`
   - `scripts/exp44/campaign.py`, which sets `MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU`
   - the env var itself, clamped [50, 1000] passes. It is replaced by
     `maxim config set nac.cluster_reward_bias_decay_tau_s`, and the conftest scrub moves with it.
   - doc tables: `docs/user/configuration.md`, the bio-memory env table, `docs/decisions.md`, the Exp 30 doc.

**S0 measurements (2026-10-17), measured vs inferred.**
- *Measured, offline, on the real loop with fake worlds* (`test_s0_todays_decay_quantum_is_the_non_eventful_full_pass`):
  one decay step per full, non-eventful pass; none on an idle pass; none on an eventful pass, which still advances a
  fixture step (12 steps with 2 eventful passes → 10 decays). An awake real-time loop decays once per poll, so decay
  per wall second equals the poll rate. This is the structure, not a rate.
- *Measured, by declaration* (`test_s0_tau_conversion_inputs`): the turn quantum S3a converts with is
  `m = round(1e6 / 2 Hz) = 0.5 s` per fixture step; the converted taus are `reward_bias`/`goal_reward_bias`/exploration
  24.75 s, `percept_valences` 99.75 s, `cluster_reward_bias` 149.75 s, eligibility 4.75 s (`τ_eff` 9.49 passes).
- *NOT measured (owed):* passes per wall second and per turn on live backends in regimes (i)–(iii), and `S_ref` for
  §(e). They need stamped sims, which S0 could not run (`~/.maxim` was shared with the parallel 1.3.3 sessions). Until
  then the figures in §Motivation stay inferred: ~2 Hz orchestrator AUT, 10–30 Hz embodied, ~27 passes per turn in
  Exp 52 Phase B, golden N = 8 locally / 7 under xdist. Regime (iii)'s acceptance range must be written into the PR
  that measures it, before it measures.

**Behaviour tier:** decay timescales are an **innate prior**: a coded default, with a learnable gain
later via decay_consolidation_calibration. "No offline forgetting" and "decay rides the experience
clock" are **invariants** of the clock policy.

## Impact

**Tests and goldens (by slice):**
- **S1:** `test_nac_persistence_decay.py`, `test_inherent_wall_decay_818.py` (opt-in path),
  `test_cluster_fear.py`, `test_store_guard_971.py`, `test_pre_freeze_fixes.py`,
  `tests/integration/test_cross_session_persistence.py`, `test_exp61_run.py`, the three hivemind tests.
- **S3a/S3b:** `test_nac.py`, `test_loop_section_extraction.py` (rewritten), `test_experience_clock_decay.py::test_s0_todays_decay_quantum_is_the_non_eventful_full_pass` (an S0 characterization of the pass quantum; rewritten when §8.5 goes), `test_loop_gates_characterization.py`,
  `test_experience_clock.py`, `test_temporal_credit.py`, `test_wire_2_percept_aversion.py`,
  `test_wire_1_risk_annotation.py`, `test_p2_reward_modulation`, `test_inherent_bias_class`,
  `test_substrate_exploration`, `test_wire_a_cluster_bias_annotation`, `test_scn_oscillator_feedback.py`,
  `tests/multi_agent_fixtures.py`, `tests/substrate/test_trajectory_invariants.py`,
  `test_sem_pain_cascade.py`, `test_affordance_transfer.py`, `test_substrate_primary_aut.py`, the
  `temporal_window_seconds` users of (c), and `tests/conftest.py`.
- **Golden:** `tests/unit/_orch_harness.py::_NAC_DECAY_PINNED` comes from **PR #1360, not yet on
  main**, so S3a merges after #1360. "Exact again" means: maps reproduce bit for bit across runs and
  hosts as a function of turns, and are re-pinned to the new values. It does **not** mean today's
  `0.2 × 0.98^N`: the factor becomes `exp(−dt/τ_us)`, not `(1 − 1/τ)` per pass.
- **S4:** `test_embodiment_failures.py`, `test_drive_specs.py`, `test_transition_drive_pain.py`,
  `test_drive_pain_helper.py`, `test_naming_events.py`, `tests/substrate/test_sem_*`, plus the new
  GL2a netting guard.
- **S5:** `test_nac.py` (`decay_all`).

**Experiments and ledger triggers:**
- **New trigger:** T1-6, T1-9 and T1-2 each get "Re-run on: NAc decay-pacing change (tick definition
  / experience clock / decay-on-load)". It goes into the ledger in the S3a PR at the latest, which is
  how the ledger adds triggers (in the PR whose walk adds them, e.g. the "`ATL typed-relation update
  path` added to their `Re-run on:`" walk). This plan PR does not edit the ledger. S1 walks these
  three rows by the new wording. For T1-9, exploration-visit decay pacing is named as the exposed
  mechanism.
- **T1-13** fires by its letter at S3a ("`run_agentic_loop` idle-gate or autonomy handling change").
  T1-13/14/15 also walk at S1 (load-time fear decay is removed) and at S3a, which wires the Q9
  rigs.
- **S4** fires Exp 44's drive-pain cadence caveat and the `transition_based_drive_pain.md` trigger.
- Walks are recorded in the ledger. No typed abort moves a result.

**T1-2 (Exp 37) re-fire** (track-C note, 2026-10-17):
- **Stamps, landed now in the harness:**
  - serving provenance, identical across a trial pair's arms (refuse a row without it)
  - §8.5 tick count, submit count, median/p90 per-turn latency, turns
  - prior `saved_at`, the save→resume gap, and the `apply_decay` actually used
  - for the three bias maps: count, sum |v|, max |v| and target-key values, at save and right after
    load. The harness **hard-fails** if they differ.
  - end-of-session magnitudes
- **Controls:** one host and serving stack per trial pair; arms **interleaved** within a seed;
  planning liveness stamped and fixed; score by position (B − C), never against June.
- **Timing:** fire once on experience-clock decay if S3a lands in the 1.4 Phase-0 window. Otherwise
  fire stamped, and accept a second fire. (~60 sessions × 25–30 min per model.)

**Doc fixes:**
- **In this PR:**
  - `memory_strength_and_forgetting.md` §Open questions 1 is amended
  - `deferred/scn_decay_anchoring.md`'s revive note is reconciled. Its "re-run Exp 56–62" is wrong
    for 56/57, which hand-tick, and for 60–62, which rest on fear with no tick decay.
- **S0, no behaviour change.** Correct every "tick-anchored" claim this plan falsifies: the
  `NACConfig` comments, the `_substrate_tick_due` docstring (affected runs: Exp 42 / Exp 52 Phase B),
  the `orchestrator._restore` premise, the `tick_embodiment_drift` and `NAc.load` docstrings,
  `docs/agents/simulation-experiments.md`, `docs/plans/minecraft_benchmark.md`, and
  `docs/agents/bio-memory.md`.

## Rollout (each slice: three-lens review, red gate first)

Each gate lands **`xfail(strict=True)` in S0** in `tests/unit/test_experience_clock_decay.py` and
flips **in place** in its slice; never removed or re-pointed. Deletion probe per guard.

| Slice | Change | Gate it flips |
|---|---|---|
| **S0** | instrumentation; doc fixes; enumerate no-Hippocampus runtimes; land G1, G2a, G2, G2b, G3, G4, G5 red | — |
| **S1** | decision 2: no decay on load; opt-in, Dormant | **G1** save → patched wall clock +7 days → load: maps byte-identical (incl. `_cluster_fear`); opt-in arm still decays |
| **S2** | `ScenarioSource` declares step experience (a′) | **G2a** decay-alone, step-indexed, eventful-pass pattern varied → identical maps at step K (red today by code reading; flips at S3a, which S2's declaration makes measurable) |
| **S3a** | driver dt without a Hippocampus; `advance_experience`; §8.5 deleted; required `dt_us`; stall guard; converted taus; tau consumers | **G2** turn script, fake latency 0.05 vs 1.0 s, **identical credit injected** and SCN anchor fallback disabled → identical maps (decay alone); **G3** realtime, fake monotonic, 2 Hz vs 20 Hz over the same 60 s → same **key set** (no entry within 1e-9 of a prune floor in the fixture) and values within 1e-12 relative |
| **S3b** | eligibility + anchors + pending events on `_experience_us`; window rename + alias | **G2b** same credit sequence at two poll rates → identical link set and credit split between trace and anchor |
| **S4** | drift only via `Embodiment.advance`, `_note_drift` kept; every out-of-loop caller migrated (Q4) | **G4** a 30 s pause adds zero drift; text-sim drift latency-independent; GL2a netting guard |
| **S5** | `decay_all` → `τ_link` (Q3) | **G5** 10 × 6-turn sessions = 1 × 60-turn session |
| **S6** | Exp 30 decay-fit re-run; ledger walks | the record |

Also in S3a: the stall guard's tests (never-advanced credit raises, after an ERROR log; a one-pass realtime session and a zero-turn turn-based session do not) and the Q9 rig wiring.
S3a/S3b touch `runtime/loop_*.py` and `agent_loop.py`, so they run `scripts/run_mypy_public.sh` and
`scripts/prepr.sh`. Touch-it-shrink-it applies to `run_agentic_loop`; deleting §8.5 shrinks it.

## Risks

1. **The rate changes in both directions.** On the 10–30 Hz embodied loop, today's 50-pass tau is
   ~1.7–5 s of activity. A converted ~25 s is **5–15× slower while active**, and faster only when idle
   past 120 s, where today nothing decays and afterwards `exp(−t/25 s)` applies. On the ~2 Hz
   orchestrator it is about the same while awake. In conversational text sims it is ~2.7× slower per turn.
2. **Fear becomes permanent.** There is no tick decay (by design) and no offline decay (decision 2), so
   a `cluster_fear` entry never fades. Exp 61-imported fear never fades either. **Accepted with a
   tracked follow-up** tied to #1072 ([fear_learning.md](fear_learning.md)). It triggers on the first
   survival agent persisted across sessions, or the first imported fear bundle. S1 stamps fear
   count/mass in survival reports and adds a CHANGELOG line.
3. **Turn/step quanta become load-bearing** calibration constants.
4. **Persisted magnitudes are old-pacing history.** Comparisons that straddle S3a carry a step change.
5. **Silent zero-dt paths** (no Hippocampus, out-of-loop callers). Mitigated by D1's decoupling, the
   stall guard and the Q9 wiring; an out-of-loop body caller that forgets `advance()` freezes (Q4
   accepts that over a wall-clock fallback).

## Open questions for the owner

None. Q1–Q9 were decided 2026-10-17 (§Decisions). New questions found while building go to the
owner at the start of the slice that finds them.

## Enforcement

| Rule | Check |
|---|---|
| NAc decay cannot run without an experience delta | structural: required keyword-only `dt_us` (`TypeError`) + G2/G3 |
| A NAc that learns must see experience advance | `NAcExperienceStalled` at session end + its test |
| No NAc forgetting while switched off | G1 (default arm vs opt-in arm) |
| Body drift excludes pause and latency, and stays netted in GL2a | G4 + the netting guard |
| No wall clock or pass count in the NAc decay/age path or `Embodiment.advance` | AST test in `test_experience_clock_decay.py`: no `time.time` / `time.monotonic` / `time_ns` in the named functions |
| Every remaining wall-clock timer in a learning path is on §(g)'s allowlist | **no check yet.** Mechanization backlog M72 in [outstanding.md](outstanding.md), added in the S0 PR: a lint over `decisions/`, `memory/`, `embodiment/body.py`, `agents/memory_agent.py::_apply_decay` and `proprioception/focus_learner.py::_apply_decay` that refuses a wall-clock read outside an allowlist |
| Permanent fear is revisited on its trigger (Risk 2) | **no check yet.** Mechanization backlog M73 in [outstanding.md](outstanding.md), added in the S1 PR: a report-stamp check that fails a survival report lacking fear count/mass, plus the #1072 follow-up issue carrying the trigger |

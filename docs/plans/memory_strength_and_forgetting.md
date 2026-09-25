# Memory strength and forgetting — a hippocampal forgetting model (1.4 parallel line)

> **ACTIVE — opened 2026-09-21 as a 1.4 PARALLEL LINE (owner decision).** Not a rung; it must not
> silently power E1–E3 ([roadmap_1_4.md](roadmap_1_4.md) §Parallel lines). Every behavioural change
> ships **opt-in, with today's defaults pinned byte-identical**, and every mechanism enters as
> `[engineering]` until an experiment earns it. **Entry condition: Phase 0** — the input-integrity
> defects fixed — before any phase changes what is kept or forgotten. **Phase 2's look-back is
> unblocked (2026-09-24):** R4's review decided no new store — tagging looks back over the Hippocampus
> record ([lookback_primitive.md](lookback_primitive.md); decision 6).

**Owns (proposed):** `src/maxim/memory/strategies.py` (a new strength strategy), the activation and
sleep paths in `memory/hippocampus*.py` and `memory/atl.py`, a `memory.*` block in the config
(`maxim config`), `scripts/memory_cost_harness.py` (new, Phase 4).
**Companion plans:** [grounded_language_acquisition.md](grounded_language_acquisition.md) and R4 in
[roadmap_1_4.md](roadmap_1_4.md) (same retrospective-tagging primitive — see §Shared primitive);
[deferred/decay_consolidation_calibration_plan.md](deferred/decay_consolidation_calibration_plan.md)
(would later *calibrate* this plan's constants); [deferred/scn_decay_anchoring.md](deferred/scn_decay_anchoring.md)
(wall-clock anchoring of NAc decay — a tension with this plan's experience clock, §Open questions);
[archive/memory_consolidation_practice.md](archive/memory_consolidation_practice.md) (the dormant
replay module's revive trigger); [deferred/adaptive_nociception.md](deferred/adaptive_nociception.md)
(pain habituation/sensitisation, upstream of the tag — deferred with a trigger).

## Why this plan exists

The hippocampus was meant to forget like a hippocampus. A six-lens read-only deep dive
(2026-09-21: hippocampus, ATL, EC/NAc, neuroscience, capacity/clocks, experiment impact) found
today's forgetting is close to the opposite. The defects filed as issues were re-verified against
`main` by the author. The remaining rows come from the lenses' code reads and small offline runs;
Phase 0/1 PRs re-verify each one they touch.

| Intent | What the code does (evidence) |
|---|---|
| Forget what is rarely **activated** | "Activation" is mostly bookkeeping: session-start bridges recall up to 700 memories (`SpatialMemoryBridge` / `SalienceMemoryBridge.on_session_start`); echo-filter `get()` calls touch. Real reactivation — pattern completion (`recall_by_ids`), cue retrieval, spreading activation — does NOT touch. ATL `recall` never touches; prompt injection touches the injected concepts' *neighbours*; deleting an episode touches its concepts (`_remove_refs_for` → `atl.get`). |
| Forget what is **weak** | No strength quantity exists. Default retention (`strategies.py::AccessBasedStrategy`) is recency + access count + out-degree; salience, pain, reward never enter it. `ImportanceBasedStrategy` exists and is never selected. |
| Keep what **matters** | `access_count` is a lifetime counter with a 0.8 retention floor at ≥ 10 — about five episodes mentioning a concept make it immortal. An unaccessed long-term pain memory is removed in about 5.5 days (long-term is a ×2, not a floor). |
| Learn **over time** | Hippocampus and ATL forgetting run on **wall-clock** time (`time.time()`); `saved_at` is written and never read. A month powered off wipes nearly everything at the next full `sleep()`. |
| Be **tested** | Only incidentally. The generic `--sim` loop uses the lightweight session end (no `sleep()`); the survival harnesses and persistent orchestrator agents run the full one, so wall-clock `sleep()` has run inside Exp 56–62 — but no experiment has ever *measured* forgetting. (Corrected at review; the first draft said sims never sleep.) |

Also found: compression is one-way (drops content; never re-expands); the spacing effect is
**inverted** (promotion pressure bleeds at 0.02/s wall-clock, so massed recalls promote and spaced
ones leak); the consolidation-candidate queue is not persisted; staged formation never completes;
EC nodes are unbounded; the prioritized replay module (`memory/sleep_replay.py`) is Dormant; every
cap (10,000 hippocampus / ATL / NAc links) arrived with its store's first commit, unmeasured and not
settable through `maxim config`, with O(N log N) eviction on every insert at the cap.

## Front-gate scope pressure

**Does this need its own mechanism, or can it ride on existing infrastructure?** It rides:
- the model is a `MemoryStrategy` subclass plus fields on existing records — no new bus, bridge or
  bio-system;
- every strength input is an **existing signal** (§The model);
- retroactive tagging looks back over the **Hippocampus record itself**, by the experience time each
  capture happened at (R4's look-back review, decision 6, 2026-09-24,
  [lookback_primitive.md](lookback_primitive.md)); `PerceptTraceBuffer` is Dormant;
- sleep replay uses the Dormant `memory/sleep_replay.py` (P8 gate passed) rather than a new replay.

The only genuinely new state is a per-trace storage strength and a persisted experience clock.

## The model (Bjork's storage/retrieval strength; the family FSRS instantiates)

Two quantities per trace (episode in the Hippocampus, concept in ATL):

- **Storage strength `S`** — how well-learned. Grows with encoding tag, retrieval and replay. The
  sleep downscale below is a *synaptic-homeostasis addition*, not Bjork: in his model storage
  strength never falls.
- **Retrievability `R = exp(−Δt / S)`** — how accessible now. `Δt` is on the **experience clock**
  (below), never wall-clock.

**The experience clock.** One persisted clock, owned by the Hippocampus and advanced explicitly by
the loop (not the module-global `bio_integration._episode_ticks`, not `PerceptTraceBuffer`'s own
counter), with a fixed tick → experience-unit so sim and robot tick rates agree. Scripted harness
paths that bypass the loop must advance it too: the opt-in path **asserts the clock advanced**
(> 0) at session end, or `R` stays 1 and nothing is ever forgotten — a silent no-op.
*Rationale, stated as a design choice:* humans do forget over calendar absence, but intervening
experience drives most forgetting (Jenkins & Dallenbach: less is lost over sleep than over waking
activity), and a robot switched off for a month should not wake with its memory wiped.

**Encoding: `S₀ = s_base · (1 + k · tag)`.** `tag` combines each signal's **deviation from its own
baseline**, normalized to [0, 1] — not the raw value: a default salience of 0.5 on every capture
(`capture_episodic_memory` reads `observation.get("salience", 0.5)`) would otherwise put a floor of
0.5 under every tag. **Combinator: noisy-OR, `1 − Π(1 − xᵢ)`** (changed from max at review):
saturating, so no crowd of weak signals manufactures importance, but coincident signals still add,
which is what the neuromodulators do (novelty via the LC/VTA–hippocampal loop, arousal via
noradrenergic BLA modulation — separate channels whose effects sum). This is an engineering choice
informed by that literature, not a claim of biological identity.

The tag is **computed by the runtime caller and passed in** as a typed `EncodingSignals`, a required
keyword on the strength path: memory may not import `runtime` (`utils/audit.py::LAYER_RULES`), and a
capture site nobody wired then raises `TypeError` instead of silently storing tag 0.

| Signal | Source | Where it is actually available |
|---|---|---|
| Salience | percept salience (after [#813](https://github.com/dennys246/Maxim/issues/813)) | every capture site |
| Novelty | `EpisodicMemory.novelty`; EC text novelty `runtime/gating.py::TextSalienceScorer._compute_novelty` | loop capture (text percepts) |
| Surprise | \|RPE\| of the captured invocation, `ToolOutput.rpe` (stamped by the executor) | loop capture. **Already unsigned** (corrected 2026-09-21): `CausalLink.update_prediction_rw` stores `abs(error)`, so worse-than-expected outcomes already raise salience — there is no separate "negative RPE" to decide. **Counted once:** `capture_episodic_memory` folds `|RPE|·0.5` into salience; the strength path reads salience from *before* that fold. Bound to its invocation since [#847](https://github.com/dennys246/Maxim/issues/847) (it used to be a never-reset slot, so a capture could read an earlier tool's surprise); NAc's own sticky `last_rpe` (goal staging) is [#850](https://github.com/dennys246/Maxim/issues/850). |
| Pain | `PainBus` intensity | **pain-bus capture only** (`record_pain_intensity` is Dormant, no producer) |
| Relief | `tool_dispatch.read_learning_side_effects(result).drive_potential_diff` | loop capture |
| Drive pressure | `HomeostaticDriveSpec` / `EntropicDriveSpec` state via `executor.embodiment` | only with a body attached — and **gated by relevance**: deprivation improves encoding of *drive-relevant* items (a hungry animal learns where food is), so pressure tags only traces whose nodes link to that drive's relief. Ungated, a starving stretch would tag everything. |
| Failure | `Outcome.success == False` (after [#814](https://github.com/dennys246/Maxim/issues/814), [#815](https://github.com/dennys246/Maxim/issues/815)) | loop and MemoryAgent capture |

*Dropped at review:* NAc valence of the active nodes — it is learned value from pain/relief history
(double-counts them), and it is not even available at capture (`consume_substrate_nodes` runs in
`observe_episode`, after the capture). Which signals set a trace's tag is recorded on the trace, so
every survivor can say why it survived.

**The capture sites** all need the typed signals (Phase 2's caller list): `bio_integration`
(async loop capture), the `pain_bus` subscriber, `tool_pain_bridge` reflexion,
`embodiment/cerebellum.py`, `agents/memory_agent.py` (live via `maxim_agent.py`), `Hippocampus.store`,
and `create.py`.

**Looking back — an eligibility trace, named honestly.** A strongly tagged event raises the tag of
traces encoded just before it, weighted by distance. That is a backward-only eligibility trace
(R4's mechanism), **not** synaptic tagging and capture, which works in both directions (weak-before-
strong and strong-before-weak; Frey & Morris 1997, Moncada & Viola 2007) over a window of about an
hour in rodents. Whether to add the **forward** window (traces encoded within `W` *after* a strong
event are also captured) is a Phase 2 decision, with `W` on the experience clock. **Wiring (decided by
R4's look-back review, 2026-09-24):** loop captures are async (the memory is made later on the worker;
the queue drops its oldest when full) and pain captures are synchronous — so the drowning can be stored
before the moments that preceded it. Every capture therefore carries the **experience time at which it
happened**, pushed into the signature so no door can miss it: `capture()` takes keyword-only
`experience_us` (default `experience_clock.now_us()`) and `capture_seq` (default: a per-agent counter);
only the async path passes them, stamped at enqueue on `_CaptureRequest` and threaded through
`capture_from_loop`. Tagging windows over memories by `experience_us`, `capture_seq` ordering captures in
one loop pass, resolving lazily (or `flush()` before tagging). A dropped or deduplicated capture was never
a memory, so it has nothing to tag. ([lookback_primitive.md](lookback_primitive.md) D2.)

**Retrieval (spacing effect, the right way round):** `S ← S · (1 + a · w_src · (1 − R) · S^(−w))`,
then `R = 1`. The gain is largest when the trace was fading, and the `S^(−w)` factor (FSRS's)
saturates it so many spaced retrievals cannot compound `S` without bound. Only **honest activations**
count (Phase 1), and only **credited** ones update `S` or reset `R` (below).
*Corrected 2026-09-21 (Phase 2 design dive):* the gain does **not** discount massed exposure across
consecutive ticks. Within one tick `Δt = 0` so the gain is exactly 0, but `1 − e^(−x) ≤ x`, so a
trace rendered every tick out-earns one spaced retrieval over the same span (measured: 100 renders
one tick apart → `S = 31.7`; one render after 100 ticks → `S = 13.2`, with `S₀ = 10, a = 1, w = 0.5`).
Hence a **credited-retrieval gap `g`**: an activation within `g` of the last *credited* one is
recorded but updates nothing.

**Sleep:**
1. **Replay** the top-N traces by `tag · (1 − R)`, with a recency term (biological replay favours
   recent and rewarded experience — Singer & Frank). Replay raises the trace's `S` **and slowly
   strengthens the ATL concepts it cites** — the hippocampus's real job in systems consolidation.
   A trace whose content is cortically supported may be compressed to gist.
2. **Downscale** `S ← λ^(Δt_sleep) · S`, with the exponent the experience time since the last sleep —
   so forgetting tracks experience, not how many times `sleep()` happened to be called.
3. **Forget** only if `R` is below its floor (below) **and** the trace is weak, unlinked (schema degree
   `< d_min`, in- and out-edges) and cited by no ATL concept or NAc key.
4. **Compress to gist before deleting**, reversibly (relearning "savings").

**Protection is a floor under retrievability, not immortality** (changed at review). A high tag,
schema links, ATL references, the LONG_TERM tier and inherent-class provenance each set a *minimum
`R`*, so a protected trace stays retrievable — but it is not exempt forever: the **tag itself decays
slowly** on the experience clock (the fading-affect bias; emotional memories fade too), and a budget
eviction (Phase 4) may take protected traces, **lowest tag first, logged loudly**. A floor that never
lifts would reproduce the `access_count ≥ 10` immortality this plan exists to remove. One-shot fear
and the once-a-year fact still survive by tag and by links, not by use.

## Guardrails

- **The survival harnesses already run `sleep()` — every change here is opt-in behind the strategy.**
  (Corrected at review: the first draft said no experiment had exercised forgetting.) The generic
  `--sim` loop uses the lightweight session end, but the Minecraft harness pins
  `consolidation="full"`, the survival scripts call `on_session_end()` (Exp 56/58/60, the dark-danger
  probe), and persistent orchestrator agents force full — so wall-clock `sleep()` has run inside the
  survival campaigns. (They were short; whether it removed anything is unverified.) Therefore **every
  change to `sleep()`, consolidation or promotion pressure is gated behind the strategy selection** —
  the spacing-correct promotion update included; nothing here is a default change until Phase 5.
- **Defaults unchanged, and proven on the built object.** `memory.strategy` defaults to today's
  `access_based`, pinned byte-identical by a guard test. The strategy is added to the frozen-apparatus
  fingerprint dicts that already exist (`exp58/60/61/62_run.py`, `r2_learned_bias.py`,
  `water_trial.py`) **as a new key with a stated default for old rows** (so past campaign hashes stay
  interpretable), and the harnesses assert it on the **built** object — the way `exp56/common.py`
  checks `nac.config.substrate_explore_bonus_weight` — not on `config.json`.
- **Config: one resolver, loud on typos.** There is no `memory.*` config section today, and
  `config_loader.resolve_setting` gives every field an env var through `_FIELD_TO_ENV` — so the
  section brings `MAXIM_MEMORY_*` vars, and they ship with their autouse conftest scrubs and
  `config_writer` validation in the same commit. The value is validated against an enum and **raises**
  on an unknown name (only a *missing* value falls back to the default — `memory.strategy=strenght`
  must not quietly run `access_based`). `HippocampusConfig`/`ATLConfig` are built in `bio_stack.py`,
  `agent_factory.py`, `create.py` and `Hippocampus.empty`/`from_config`: all of them call **one**
  resolver, with a parametrized test over every builder (the shared-builder silent no-op).
- **Out of round one:** `NAc._cluster_fear` and `cluster_reward_bias` decay (Exp 56–62 depend on
  them), EC world-modality nodes (frozen centroids carry Exp 60–62), NAc causal-link strength.
- **Two-tier rule.** Everything here is `[engineering]`; nothing graduates until Phase 5 earns it.
  `sleep_replay.py` gains a caller behind the opt-in, but its **resurrection** from Dormant counts
  only when Phase 5 earns it (dormancy rule: an experiment, not time).
- **A fix ships with a caller.** Each phase names the production call site that runs it; the
  opt-in path must run in a real (non-test) loop before the phase is called done.

## Phases

**Phase 0 — input integrity + guards (entry condition; no retention change).**
- Fix the defects that corrupt the signals the model reads:
  [#813](https://github.com/dennys246/Maxim/issues/813) salience/novelty dropped,
  [#814](https://github.com/dennys246/Maxim/issues/814) failures stored as successes,
  [#815](https://github.com/dennys246/Maxim/issues/815) failed tool results crash,
  [#816](https://github.com/dennys246/Maxim/issues/816) compressed-concept crash (crash half),
  [#817](https://github.com/dennys246/Maxim/issues/817) staged formation never completes (wire or Dormant).
- ~~Add decay / eviction / cap to the ledger rows' `Re-run on:` triggers~~ — **dropped
  2026-09-21:** the ledger's own discipline fixes a row's triggers at graduation ("not
  retroactively"), and the change is already covered — a forgetting change fires those rows through
  the existing generic **bio-system refactor / substrate-pipeline change** triggers. What protects
  the earned rows is §Guardrails (opt-in, pinned defaults, fingerprints, built-object asserts), not
  new trigger text.
- Correct `docs/agents/bio-memory.md`'s clock claim (done in this plan's PR).
- Each fix PR states which ledger rows it re-ran or discharged (each issue lists its candidates).

**Phase 1 — honest activation.** One `activate(ids, source=...)` path per store. Bookkeeping reads
(`get()` echo filters, session-start bulk recalls, deletion callbacks, neighbour lookups) stop
counting; real reactivation (pattern completion, cue retrieval, spreading activation, ATL recall and
prompt injection) starts counting. Behind the opt-in; the counters it feeds are read only by the
strength strategy until Phase 5 flips defaults. **Lock discipline:** `recall_by_ids`,
`search_by_content` and spreading activation hold a *read* lock, `ATL.get` already calls `touch()`
under a read lock (a race today), and `Hippocampus.get` takes the write lock against the capture
worker — so `activate()` must never upgrade a lock it holds: queue activations and apply them in one
write at the end of the call. Hard sites beyond the obvious: ATL spreading activation (touches
neighbours, excludes seeds), `_remove_refs_for`, the session-start bridges, and the touches in
`math/angular_gyrus.py` and `concept_grounder.py`.

*As built (amended 2026-09-21, Phase 1 PR):* three departures, each deliberate.
**(a) New counters, old ones untouched.** Honest activation is a separate `activation_count` /
`activation_sources` on every record; `touch()`/`access_count` are NOT changed, because they drive
today's `AccessBasedStrategy` and the immortality floor, and moving them changes default retention
before Phase 5. So the hard sites above still inflate `access_count` — that is not fixed but made
moot: the Phase 2 strategy never reads `access_count`. Phase 5's default flip owns retiring it.
**(b) No opt-in**, because the counters are write-only: nothing reads them until Phase 2 (a strict
red gate on `memory/strategies.py` says so). **(c) No queue.** `MemoryLayer.activate` is called
after the read returns and outside every store lock (never inside `for r in store:` — `RWLock` is
writer-priority and not re-entrant), taking the read lock once and then per-record locks.
*Counting rule — two criteria:* the LLM paths count at the consumer's **render cap** (enrichment,
the memory/concept tools, the replan prompt, the adaptive planner's decomposition prompt); pattern
completion counts at **completion** (CA3 reactivation, whether or not a consumer reads the
prediction), completed episodes only, never the cue concepts. `tools/discovery.py`'s `[DANGEROUS]`
tag is not a use: it renders an NAc verdict keyed by the concept, not the concept's content.
Consumers that retrieve and never deliver are #845 (the replan site is wired but dead until
#845(1); the adaptive planner is live only in `embodied_runtime/agentic_runtime.py`).

**Phase 2 — the strength model.** `S`, `R`, the typed `EncodingSignals` at all seven capture sites,
the look-back over the Hippocampus record by enqueue-time experience µs (decision 6), the retrieval update, the persisted experience clock with its advanced-clock assert.
No immortality floor under the new strategy. **Validated in the LLM sim worlds** (route A, owner
decision 2026-09-21), where enrichment and the memory tools are live; survival-world validation waits
on Phase 2S below.

*Decisions — RESOLVED 2026-09-21* (owner, after a five-lens design dive: survival check, capture
window, negative RPE, per-signal baselines, experience clock; lens reports in the PR that recorded
this). Every constant below is a `maxim config` default under `memory.*` — validated, raising on a
typo, frozen into experiment fingerprints — never an env var or a literal.

1. **The experience clock — WORLD time, per world** (revised 2026-09-22 at the Phase 2a review;
   owner decision). The first cut advanced by the loop's nominal period on every *active* pass, and
   both review lenses showed that "active" is the idle gate's scheduling predicate, not lived time:
   the survival harness counted 250 ms per decision where a 30 Hz robot on the same cadence counts
   33 ms (7.5× apart for the same lived time), and on LLM paths the loop stays awake up to 120 s after
   a submit, so a slow LLM produced *more* experience. So the clock advances by what the agent's
   **world** did, decided once per **live** loop pass (after the pause check, before the idle gate —
   the slot `tick_embodiment_drift` uses, so a resting agent still lives and a paused one does not)
   by `runtime/experience_time.py::ExperienceClockDriver`:
   - **Real-time worlds** (survival via the loop — the scripted survival harnesses are 2S — robot,
     CLI; the default): elapsed monotonic time between live passes **minus the autonomy controller's
     paused time** (`AutonomyController.paused_seconds_total`). Idle passes count — the world keeps
     happening — and so does a long in-pass block (a multi-cycle deliberation, a robot motion). A
     suspended machine never counts (`time.monotonic` excludes system sleep on macOS and Linux); a
     5-minute per-pass cap is only a stall bound. This is the unit decision 2's constants are
     measured in.
   - **Turn-based worlds** (text sims): a fixed quantum per world **turn**, so LLM latency is never
     experience. The percept source declares it with the duck-typed `experience_turns()` +
     `experience_us_per_turn` (`ConversationalSource`: 5 s per turn — a calibration constant for
     Phase 5, not a measurement). A turn is the world moving on — `inject_cli`, or the bridge's
     `mark_turn()` in substrate-primary mode, where no text is injected — never a pain or sensor
     event inside a turn, and not tied to delivery. A composite is turn-based through its single
     turn source. **Known gap:** a step-based `ScenarioSource` (fixture sims, `--sim <yaml>`) is a
     loop-iteration world with no declared step length, so it runs as real-time. *Trigger:* a
     Phase 5 prereg or any forgetting measurement that uses a step-based scenario — declare
     `experience_turns` on it (a step length in the scenario YAML) first.
   Stored as integer **microseconds**, persisted in `hippocampus.json` as
   `"experience_clock": {"us": N, "unit": "world_experience_us"}`; a missing key loads as 0 with one
   warning, and a *malformed* record warns and restarts the clock at 0 rather than failing the load
   (a failed load would send `load_with_recovery` down its empty-store path and cost every memory).
   Per-agent (one per Hippocampus; the loop follows the Hippocampus it captures into); restored in
   place, so references survive `--resume-sim`; handed to ATL/AG by `MemoryHub` as a **required
   constructor argument** under the strength strategy (Phase 2c). **Owed, held by strict red gates**
   in `tests/unit/test_experience_clock.py`: the stalled-clock assert
   (`ExperienceClockStalled` from `MemoryHub.on_session_end` and `on_session_end_lightweight` when the
   strength strategy is on, the clock did not move and the session captured or activated anything;
   Phase 2c, mirrored in the survival smoke verdict), and the scripted harnesses that call
   `propose_via_substrate` without the loop (`WaterTrial.train`, `exp58_run.py`,
   `exp58_offline_gates.py`) each running a driver — **moved to Phase 2S** (shipped as 2S-a,
   2026-09-24; `exp53_cross_context_readout.py`, listed here first, has no Hippocampus), since they
   are the survival path. 2. **Looking back (retroactive tagging).** A decaying weight on the experience clock,
   `τ = 10 s` with a **30 s** cutoff by default (`memory.capture.tau_s`, `memory.capture.cutoff_s`;
   owner: decay must not be too heavy — at `τ = 5 s` the weight at drowning-damage onset, ≈16.2 s
   (`r3_survival_benchmark_prereg.md`: first damage 16.15–16.25 s), would be 0.04; at 10 s it is 0.20). The window is **dynamic**: configurable now; learning it from the
   cause→outcome lags a world actually produces is a later option, and any adaptive width is **frozen
   during a campaign** (it must not fit itself to outcomes it helped create). **Forward window: 0** —
   in the survival world what follows a death is the respawn, so a forward window would tag spawn
   snapshots; it becomes a pre-registered arm, gated by relatedness and reset at episode boundaries.
   **Weight by relatedness**, not time alone (Dunsmoor et al. 2015: the retroactive boost is selective
   to related items): scale each tag by similarity to the strong event (EC cluster / cosine), so one
   pain does not tag whatever sat in the preceding 30 s. **Shares the experience
   clock, never NAc eligibility's constants or its effect** — a tag never feeds NAc credit
   (NAc's 0.9-per-active-cycle decay sits inside the Exp 60–62 fingerprints). Resolve lazily or
   `flush()` before tagging (async captures; a full queue drops its oldest).
3. **Surprise: one unsigned `|RPE|` channel, baseline 0** — the code is already unsigned (table
   above). Positive-only or a signed channel would need a signed field on `CausalLink`, which is
   persisted and hivemind-carried (a CC3/wire change), and is the weaker reading of the literature
   (unsigned PE enhances episodic memory — Rouhani, Norman & Niv 2018; the signed results are
   task-dependent — Ergo et al. 2020). Prerequisite [#847](https://github.com/dennys246/Maxim/issues/847)
   (the sticky slot) is satisfied: the guard that a stale RPE is never read is
   `tests/unit/test_rpe_per_invocation.py`.
4. **Baselines: fixed, except novelty.** Every `EncodingSignals` field is `float | None` with **no
   default** (a missing one is a `TypeError`); `None` maps to 0 and is recorded on the trace — absent
   is never a signal. Salience: baseline 0.5, positive deviation `(s − 0.5)/0.5`, read before the RPE
   fold. Novelty: the store *is* its adaptive baseline (EC familiarity), so it gets no second one;
   hard-coded novelty constants (0.6 pain bus, 0.8 reflexion, 0.3 `Hippocampus.store_observation`) are not
   measurements and map to `None`; weight novelty by store confidence `n/(n + n₀)` so an empty store
   does not saturate its first traces. Pain: `x = intensity` for **nociceptive** pain only — body damage (motor strain/thrashing, movement failure, external/world damage, safety violation); NOT tool-failure pain (a lifetime frustration count whose surprise already reaches the loop capture), NOT anticipated pain (a prediction, kept only as `extra["anticipated_pain"]` so fear cannot strengthen the memory of its own anticipation), NOT resource exhaustion or cognitive overload (drive/load), and NOT a drive breach the body publishes as external pain (`source="drive:<name>"` — air hunger, hunger; kept as `extra["drive_pain"]`) except `drive:health`, whose loss is tissue damage. An **innate prior**, owner decision 2026-09-22 (`proprioception/pain_bus.py::NOCICEPTIVE_PAIN_TYPES`). Intensity is **fixed** in this plan — adaptation is
   upstream in the pain producer, deferred ([deferred/adaptive_nociception.md](deferred/adaptive_nociception.md)).
   Drive pressure: `embodiment/sem.py::corrective_need_intensity`, relevance-gated — graded against
   the set point for homeostatic drives (below it only) and against the satisfaction/deprivation
   thresholds for entropic "down" drives; it returns `None` for entropic "up" drives and
   above-set-point deficits, a gap Phase 2 must close or declare. Relief: positive part only, **per drive**, normalised by that drive's declared
   range half-span (it is summed raw units today). Failure: through `|RPE|` of the outcome, not a flat
   bit. **Remove two double counts:** the pain-bus subscriber's `+0.2` salience and its hard-coded
   `success=False` both re-encode pain.
5. **The retrieval update hooks `activate` EVENTS**, inside `MemoryLayer.activate` via
   `strategy.on_activation(record, now, source)` (a no-op on `AccessBasedStrategy` /
   `ImportanceBasedStrategy`, so defaults stay byte-identical) — never an observer that can go
   unregistered, and never Phase 1's massed `activation_count`. `clock.now()` is read once, before any
   lock; the update runs under the record's `_touch_lock`; lock order stays store → record (Phase 3's
   sleep takes the write lock then `_touch_lock`; activate only `_touch_lock`). Defaults:
   **credited gap `g` = 2 s** of experience; **source weights** `tool` 1.0, `replan`/`planner` 0.8,
   `enrichment` 0.5, `prediction` 0.5 (the testing effect: effortful recall > re-exposure; internal
   reactivation, often unconsumed, at the low end); **only credited retrievals reset `R`**;
   `S` and `last_retrieval_ms` serialize atomically with the counters; **Hippocampus only** in Phase 2
   (ATL gets `S` when its compression/eviction path moves onto the strategy). The strict red gate
   `tests/unit/test_memory_activation.py::test_phase2_strength_strategy_reads_activation` would
   **never** flip under this design (it greps `strategies.py` for `activation_count`, which the event
   hook deliberately never reads), so this phase's PR **rewrites** it into a behavioural gate — the
   strength strategy's score moves with credited `activate` events — rather than leaving a
   permanently yellow marker.
6. **Owner of the looking-back primitive** (§Shared primitive): ~~this line, as its first production
   caller~~ — **resolved 2026-09-24 (owner): R4 owns the question, and its design review decided no new
   store** ([lookback_primitive.md](lookback_primitive.md)). This line looks back over the Hippocampus
   record by enqueue-time experience µs; `PerceptTraceBuffer` is Dormant (CI enforces it). Phase 2's
   look-back is unblocked.

*Phase 2d slicing — the look-back (2026-09-25, after R4's review decided no new store,
[lookback_primitive.md](lookback_primitive.md)):* **2d-1** — every trace records when it happened:
`encoded_at_us` (experience µs, immutable — NOT the retrievability anchor, which a credited retrieval
moves; the anchor now starts from it) and `capture_seq` (orders captures sharing a loop pass's
timestamp; per store, resumed past the saved maximum on load). `capture()` takes both keyword-only,
defaulting to now / the next number; only the async loop path passes them, stamped at enqueue.
Recording only (SHIPPED 2026-09-25). **2d-2** — the tagging rule itself, **DECIDED + BUILT
2026-09-25** in [memory_2d2_retroactive_tagging.md](memory_2d2_retroactive_tagging.md) (answers: (i)
the event's own `encoding_tag` strictly above 0.5; (ii) `retro_tag` beside the stamp, floor reads the
max; (iii) shared world/audio situation clusters, no fallback; (iv) at `sleep()`; (v) both skipped;
(vi) as below), with its limits deferred on triggers in
[deferred/retro_tagging_extensions.md](deferred/retro_tagging_extensions.md). The questions as
first posed: (i) **which events tag** —
nociceptive pain above a floor, relief, `|RPE|`, or the encoding tag of the strong event's own
trace; (ii) **what a tag changes** — a separate `retro_tag` beside the stamped `encoding_tag` (the
2c-3(b) rule: the stamp is never rewritten) that the protection floor reads as `max(tag, retro_tag)`,
vs raising `S`; (iii) **relatedness** — shared situation clusters (2S-b's `situation`, same EC ids)
vs embedding cosine, and what an unrelated or situation-less trace gets; (iv) **when** — at the
strong event's capture, lazily after `flush()` so earlier async captures are in the store; (v) **what
is taggable** — pre-2d-1 traces (`encoded_at_us = None`) and `CompressedMemory` records; (vi) **the
order key** — `(encoded_at_us, capture_seq)`, since the clock read and the sequence reservation are
separate. Opt-in under `memory.strategy=strength`, like the rest.

*Phase 2b slicing (2026-09-22, from a capture-site map):* **2b-i** — the typed `EncodingSignals`
recorded on every trace, required at every capture door, write-only (SHIPPED with this note): four
signals (salience, novelty, surprise, nociceptive pain) plus a required closed `site`, frozen.
Review folds: **no `drive_pressure`/`drive_relief` fields yet** (their per-drive shape is 2b-ii's to
decide, so no scalar was persisted to break later), **no hippocampus-size denominator** (novelty's
reference-set size is recorded by the novelty producer beside its value, in 2b-iii), and **the
Rescorla-Wagner value is bounded to [0, 1] at every producer** (`causal_link.py::bound_predicted_value`
in `from_dict`, hivemind ingest and merge — ingest used to accept [-1, 1], so one imported link could
make |RPE| reach 2; no recorded bundle held such a value, checked over 713 files).
**2b-ii** (SHIPPED) — per-drive relief and drive pressure on the record: `EncodingSignals.relief`
and `.drive_pressure` as sorted `(drive, value)` tuples, from a new record-only
`drive_progress_by_drive` side-effect key normalised at the executor, which stamps both onto the
invocation's `ToolOutput` (pressure read BEFORE the action). Normaliser (owner decision
2026-09-22, revised from the half-span): each drive's own largest possible movement —
`max(hi − set_point, set_point − lo)` for homeostatic, and for entropic its own deprivation-to-satisfaction band (NOT the declared range, which is widened for the encoder's neutral — using it made a full satisfaction read 0.25 and put 1.0 out of reach) — so `1.0` means
"the most this drive can give" and no clamp is needed. New `sem.py` helpers (`drive_span`,
`relief_fraction_from_progress`, `drive_pressure`) sit BESIDE the credit/perception ones, which are
untouched (their Exp 52 / 58 / 60 / 62 triggers). **Removing the pain-bus double count (`+0.2`
salience, `success=False`) moved to 2c** — it belongs on the strength path, and that switch does not
exist until then. **Known gap for 2c:** on `minecraft_player` only `eat` declares a `self_effect`,
so `escape_water`/`flee`/`attack` produce no measured relief at all (the R4 delayed-credit class) —
relevance for those actions needs another source than the relief keys. **2b-iii = Phase 2S's salience/novelty
producers** (EC novelty surfaced from `PatternResult.best_similarity`; a survival salience from
drive, sensor change and pain). Found while mapping: a MemoryAgent tool-failure capture duplicates
the loop's capture of the same action, and a reflection duplicates the loop's surprise — the Phase
2c tag must count one event once.

*Phase 2c slicing (2026-09-22, from a design map):* **2c-1** — the `memory.*` config section, the
one resolver threaded through every store builder, an unknown name raising at every door, the ATL
honouring the configured model instead of hard-coding access-based (SHIPPED with this note;
behaviour-neutral on the default path). `strength` is deliberately NOT a valid name until 2c-3
ships the strategy. Review caught two blockers worth remembering: the section was declared but
never PARSED (the `console` bug again — a value written by `maxim config set` was dropped at load
and erased by the next write), now guarded by an AST test that every declared section is read by
the parse walk; and a name accepted by config but unimplemented in the store crashes at the first
consolidation, far from the command that set it. Round 2 caught a third: the fold's own import
landed in a `TYPE_CHECKING` block, so `maxim.load.hippocampus()` raised at runtime while ruff and
mypy stayed green — only the FULL suite catches that shape, never a keyword-filtered slice. Also
folded: `ImportanceBasedStrategy` gained a semantic branch (every concept scored by age alone
through its constant fallback, so the newly-honoured ATL knob had no signal), and the
config-format downgrade hazard is filed as [#856](https://github.com/dennys246/Maxim/issues/856). **2c-2** — `S` + the encoding tag on the record (noisy-OR over the
recorded signals' baseline deviations) and `ExperienceClockStalled` WITH its two `MemoryHub`
callers. **2c-3** — the `StrengthStrategy` itself: `R = exp(−Δt/S)` on the experience clock, the
credited-gap retrieval update through `strategy.on_activation`, floors under `R` and no immortality
floor; both red gates flip (the activation gate REWRITTEN as a behavioural one, never just
unmarked), the byte-identical-default guard lands, and one real sim loop shows `S` move on a saved
record. Two departures recorded here rather than silently: **(a)** the per-source retrieval weights
ship as a frozen default in `memory/`, not config — the config surface is flat dotted paths with
string coercion, and a JSON-map coercer for one map would be worse than a named constant;
**(b)** drive-pressure relevance **fails closed** — pressure contributes 0 when a trace's
`drive_relief` keys are empty, which on `minecraft_player` is every action but `eat`
(`escape_water`/`flee`/`attack` declare no `self_effect`). *Trigger for the second relevance
source:* Phase 2S / R4, when delayed credit gives those actions a drive link. A proximity heuristic
would be a band-aid.

*2c-2a as built (2026-09-22, PR #859 — the `S` half only; `ExperienceClockStalled` NOT included):*
`encoding_tag` + `storage_strength` are stamped at capture and persisted on `EpisodicMemory` and
`CompressedMemory`, write-only (a guard test plus a reviewer grep over `src/` + `scripts/` confirm
zero readers). Five decisions worth not re-deriving. **(a) `S` is carried in the experience clock's
OWN unit (microseconds), not seconds** — `R = exp(-dt/S)` compares S directly against a clock delta,
and a seconds-vs-µs seam there forgets everything in 10 µs while every test still passes; same unit
on both sides means no conversion exists to get wrong. `S_BASE_DEFAULT` is an unvalidated
placeholder whose provenance is the plan's worked example in TICKS — Phase 5 earns the real value.
**(b) The two per-drive channels each contribute ONE deviation** — relief as a mean weighted by each
drive's own pressure, pressure as its max. Per-drive deviations made the tag scale with how many
drives a body HAS (relief 0.3 tagged 0.657 on three drives, 0.942 on eight), so tags were not
comparable across bodies, which is what cross-body transfer claims rest on. Weighting by pressure
rather than by novelty/salience is deliberate: those are separate channels already in the noisy-OR.
**(c) `novelty_reference_size` is recorded beside the tag** — the `n/(n+50)` weight divides by the
Hippocampus' trace count today, but 2b-i's review put novelty's reference set with the PRODUCER
(2b-iii) and tags are permanent, so without the record the store would hold two provenances with
nothing marking which. The keyword is named for what it means, so 2b-iii supplies the producer's set
without a rename. **(d) The knobs live on `HippocampusConfig`, not in `memory.*` config** — a config
key nothing reads is the defect 2c-1's review caught; 2c-3 adds the keys WITH the code that reads
them. **(e) Relevance is the PRESENCE of a relief key, not a positive value** — the executor records
`0.0` for a drive an action moved AWAY from comfort, so gating on `> 0` dropped exactly the drowning
case (air pressure 1.0, air relief 0.0) that must encode strongest.

*2c-3 as built (2026-09-22 — the phase where `S` is first READ):* `StrengthStrategy` in
`memory/strategies.py`, selectable as `memory.strategy=strength`, with `memory.s_base` /
`memory.k` landing beside it. Six things worth not re-deriving, three of them departures from
the text above. The two-lens review round's findings are folded in place below, tagged where they
land.

**(a) The saturation factor is taken RELATIVE to `s_base`.** The plan writes the retrieval update's
saturating term as `S^(−w)`, which is not dimensionless — with `S` in the clock's microseconds
`S^(−0.5)` is ~3 × 10⁻⁴ and every retrieval gain silently rounds to nothing, the same class of
seam decision (a) of 2c-2 exists to prevent. Shipped as `(S/s_base)^(−w)`: 1.0 at the base
strength, and the property the plan's worked example was *about* — the gain shrinks as a trace
grows well-learned — is preserved. Its absolute numbers (`S₀ = 10` in ticks) do not carry over;
Phase 5 calibrates. Measured on the shipped model with `s_base = 10 s`: 100 renders 0.1 s apart
earn **exactly** what 5 gap-spaced retrievals earn (both land on `S = 18.09 s`), and spreading
those same 5 out gives `S = 51.92 s` — **2.87×**, not the "doubles" this note first claimed
(Architecture N6: state the number you actually measured). `a`, `w`, `g`, the floor weight and the
fade multiplier stay named constants rather than config keys, unlike `s_base`/`k`: Phase 5
calibrates them jointly, and a knob nothing has measured is worse than a constant that says so.
**`s_base` is a LIVE model parameter, not only the encoding base** — raising `memory.s_base`
between runs changes the retrieval dynamics of traces already stamped, so 2c-2's rule "re-tuning
changes what encodes NEXT, not what already happened" holds for the stamp and NOT for the model
(Architecture #4). `k` has no such reach. The factor is clamped at 1.0 so that reach is bounded:
unclamped, one retrieval after a raised base could multiply `S` by ~11× where this module
documents `1 + a·w` (Executor #7).

**(b) Protection is a floor that FADES, and the fade is a view, never a rewrite.** `score =
max(R, w·tag·exp(−dt/(S·F)))` with `w = 0.5`, `F = 10` — so a maximally tagged trace stays
retrievable roughly five time-constants longer than a boring one and then becomes forgettable,
which is what "a floor, not immortality" has to mean in code. The stamped `tag` is never modified:
the fading-affect bias is computed from the same `dt` as `R`, so re-tuning changes what happens
next rather than rewriting what already happened (the rule `_stamp_encoding_strength` set in 2c-2).
`degree` is accepted and unused — schema linkage enters through Phase 3's *conjunction* in the
forgetting rule, not as a score term — and `should_compress` keeps a tag-held trace WHOLE, since
gist is what survives ordinary fading and detail is most of what a tagged trace meant.

**(b-bis) What forgetting actually consists of under `strength` TODAY** (Executor #6 +
Architecture #6). §Protection lists five floors under `R`; 2c-3 ships one, the tag. Because
`degree` is ignored, the schema-link, ATL-citation and successful-user-interaction floors
`AccessBasedStrategy` provides are absent — this model will delete a well-linked, ATL-cited,
untagged episode that `access_based` keeps. And with `S_BASE_DEFAULT` at 10 s of experience, an
untagged trace falls below `retention_threshold` after **~12 s of run experience**. Both follow
from the phase boundary rather than from a bug (the conjunction is Phase 3's, the constant is
Phase 5's) — but the model is **not campaign-ready**, and the honest form of that sentence is a
number rather than a deferral.

**(c) `R` needs an anchor, so traces carry one.** `retrievability_anchor_us` on `EpisodicMemory` /
`CompressedMemory` is the experience time `R` decays from: stamped at capture beside `S`,
whatever the configured strategy, and reset by each credited retrieval — which is what "then
`R = 1`" is, concretely. One field, not two: the last-credited time and the decay origin are the
same instant. A trace loaded without one is re-anchored at the clock's current value at load,
because no anchor means no `dt` means `R = 1` forever — the immortality this plan exists to
remove — and `dt` is clamped at 0 so a clock restarted by a corrupt record can never make a trace
*more* retrievable than when it was stored.

**(d) `ExperienceClockStalled` is gated on `MemoryStrategy.requires_experience_clock`**, the owed
item from 2c-2, and it fires LAST on both `MemoryHub` session-end paths, after every save, carrying
the session's `results` — a diagnostic must not cost the session its memories (the
`activate_after_use` principle). An honestly empty session stays silent: nothing happened, so
nothing should have aged, and an assert that fires on every idle shutdown is one nobody reads.
"Did anything happen" is answered by in-process tallies on the Hippocampus (`session_work()`), not
by `_stats`, which is persisted and so cannot distinguish this session from every previous one.
**Two defects here were caught by the review round, both the same shape as this plan's own
lessons.** (i) *Cross-confirmed by both lenses:* every production session-end caller catches broad
`Exception` and the two sim paths logged it at DEBUG, so the mechanism shipped with no audience —
`feedback_a_fix_ships_with_a_caller` applied to a diagnostic. It now LOGS at ERROR before raising,
and `bio_integration` + `interactive` name the type ahead of their broad handler and re-emit their
telemetry from `exc.results`, which until then had zero non-test readers. (ii) *Executor #2:* the
first cut asked "did the clock advance" of `now_us()`, which `--resume-sim` restores to a far
larger value — so the guard passed trivially on exactly the resumed harnesses it was written for.
`ExperienceClock.advanced_us()` now counts what DROVE the clock and is untouched by `restore()`.
Every stalled-clock test had started its clock at 0, which is precisely why none could see it.

**(e) The ATL names `strength` explicitly and keeps today's model under it.** Phase 2's strength
model is the hippocampal one (decision 5), and concepts carry no `S` — but letting a *valid* config
name fall through to the store's raise is precisely the defect 2c-1's review caught, one layer
down. Named branch, documented, tested, and since the review round it says so once per process at
INFO rather than only in a source comment (Architecture N4).

**(f) The frozen-apparatus fingerprints now record the retention model** (Architecture #3).
§Guardrails asked for this at 2c-1 and it was missed there; 2c-3 is what makes it bite, because
`~/.maxim/` is shared across worktrees and a stray `maxim config set memory.strategy strength`
would otherwise silently reconfigure a campaign with nothing in the record saying so.
`memory_strategy` is a new key with the stated default `access_based` for old rows, read off the
BUILT store in `water_trial.live_fingerprint()` (Exp 60/61/62) and `exp58_run`'s inline reader.
`fingerprint_drift` compares the key UNION, so the live and frozen halves had to land in the same
commit or every run would refuse.

*Found while building, fixed here:* `memory.s_base = nan` passed the env door — `nan < 0` is
`False`, so every `min_val` range check admits it, and a NaN `S` poisons every comparison it
touches. Both doors now test `math.isfinite` / `not value > 0` rather than `value <= 0`.

*Found by the review round, fixed here:* the two `ExperienceClockStalled` defects in (d); a torn
`(S, anchor)` pair could be PERSISTED by compression, because `CompressedMemory.from_episodic`
copied the four strength fields with unlocked reads while `_strength_fields` had just been given
the record's lock — it now goes through that same reader (Executor #3); and two of my own guards
were vacuous for a reason the mutation sweep could not reach, because the test fixture's `s_base`
was byte-identical to `S_BASE_DEFAULT`, so nothing checked that the configured base reaches the
SCORER as opposed to the stamp (Executor #4, #5, #9).

**The real-loop demonstration: DONE 2026-09-23**, on the rig (big-mac-mini), session
`~/.maxim/sim_reports/20260923_095303`, branch worktree asserted as the imported `maxim` before
launch (harness-provenance rule). `qwen2.5-32b-instruct`, `n_ctx` 16384 via `maxim config` with both
readings agreeing; `MAXIM_MEMORY_STRATEGY=strength` passed as an ENV VAR rather than written to the
rig's `config.json`, because that file is what Exp 58/60/61/62 read and leaving `strength` in it is
the exact contamination (f) above adds a fingerprint guard against.

What the saved `aut_hippocampus.json` shows, on 4 turns / 485.6 s of wall time:
- **The experience clock reads 20 s, not 485 s** — 4 world turns x the 5 s `ConversationalSource`
  quantum. Decision 1's turn-based branch, confirmed in a real loop: LLM latency is not experience.
- **142/142 records stamped** with `S`, `encoding_tag` and `retrievability_anchor_us`; anchors fall
  exactly on the turn boundaries (5/10/15/20 s), never between them.
- **9 records' `S` ROSE above their capture stamp**, 10,000,000 -> 11,967,347 µs, each with one
  credited `enrichment` activation. That matches the model to the digit: with `S0 = s_base` the
  saturating factor is 1.0, `R = exp(-5/10) = 0.6065`, so the gain is
  `1 + 1.0*0.5*(1 - 0.6065)*1.0 = 1.19673` and `S = 11,967,347`. The composition ran end to end —
  capture -> stamp -> activate -> on_activation -> credited update -> save.
- **`ExperienceClockStalled` never fired** (0 occurrences), which is the correct verdict: the loop
  drove the clock. The assert was live on this path, so the run exercised it as well as `S`.

*Two things the run surfaced, neither owned by this phase:* every credited activation came from
`enrichment` and none from `prediction`, which is the known Phase 2S gap (#848 — salience 0.0 keeps
MemoryAgent's FORMING gate shut, so pattern completion never runs); and `decisions/nac.py::predict`
raises `AttributeError: 'OutcomePrediction' object has no attribute 'outcome_signature'` on every
call (the field is `predicted_outcome`). The NAc one is PRE-EXISTING on main, untouched by this
diff, and confined to the `sim_nac_predict` telemetry block AFTER `_predict_impl` inside a
swallowing `try` — so predictions themselves are correct and only their sim-log line is lost. Filed
separately rather than fixed here.

*Owed by Phase 3 (unchanged by this phase):* replay, the homeostatic downscale, the unlinked/uncited
conjunction in the forgetting rule, and ATL parity. **Phase 3's downscale must RE-ANCHOR**
(Architecture #5): `R` and the protection floor both use the *current* `S` as the time constant for
an interval already elapsed, which is correct while only retrievals change `S` (they re-anchor), and
retroactive the moment `S ← λ^Δt · S` does not. Measured: `S` 10 s → 5 s after 100 s elapsed drops
the floor from 0.184 to 0.068 — a trace protected a moment earlier becomes forgettable for a reason
that never happened. Any non-retrieval change to `S` must pick the anchor that preserves `R`
(`anchor' = now − S'·ln(1/R)`). *Was red and correctly so until 2S-a (2026-09-24), now a plain
test:* `test_phase2s_bypass_harnesses_advance_the_clock` — the scripted survival harnesses are
Phase 2S.

*Owed by 2c-3, carried from 2c-2 (DISCHARGED 2026-09-22 except where noted):* **`ExperienceClockStalled` ships as a capability the STRATEGY
declares** (e.g. `MemoryStrategy.requires_experience_clock`), never a string comparison against the
name `"strength"` in `MemoryHub` — that would be a second source of truth no third-party strategy
could satisfy, and it is the same concern [config_extensibility.md](config_extensibility.md) exists
for. Its strict red gate in `tests/unit/test_experience_clock.py` stays red until then, which is the
honest state. **Done:** shipped exactly that way; both red gates were REWRITTEN behaviourally
rather than unmarked (the activation one could never have flipped as written — it grepped for
`activation_count`, which the event hook deliberately never reads). Also owed, **still owed**: the
plan's "the 2c tag must count one event once" (the MemoryAgent / reflexion duplicates) is recorded
on each trace as `site` but nothing consumes it yet.

**Phase 2S — the survival gate** ([#848](https://github.com/dennys246/Maxim/issues/848); **the two
percept-side fixes below are SUPERSEDED by the 2026-09-24 amendment after this section: survival
percepts carry no body state, so both fixes move to the loop capture**; owner:
"the gate definitely needs to activate memory"). Measured offline on main 4a0362e2 (a scratchpad
ScriptedWaterBridge run, monkeypatched counters, no source edits — not a committed artifact; the
numbers are in #848): in 14.75 s of survival loop time **nothing ever used a memory** — 22 percepts, **all at salience 0.0**, so MemoryAgent's
FORMING gate never opened and pattern completion never ran; and when called by hand it returned 0
predictions, because `PatternCompleter._find_matching_concepts` cues by object/goal **name** while
survival concepts are sensor and action strings. Two root fixes, no band-aids (no lowered gate, no
fake `cli_input`):
- **Salience is always computed.** `PerceptionAgent._compute_salience` scores camera detections
  only and returns 0.0 without them; the loop's observation producers set neither salience nor
  novelty, so `Hippocampus.capture_from_loop` falls back to 0.5/0.5 — the floor under the saved
  loop captures (a read-only scan of 38,465 pre-#813 captures, 2026-09-21: novelty 0.5 on all of them,
  salience 0.5 on 77 %). Every percept gets a
  salience from what the body actually senses — sensor change and surprise, drive pressure, pain —
  so a boring moment honestly scores low and a drowning high. A small nonzero **exploration floor**
  (like `attention/salience_map.py`'s 0.1) is allowed only as a **named innate prior** that the tag
  subtracts as baseline, never as a stand-in for a missing measurement.
- **A substrate-native cue for pattern completion:** cue by the percept's EC cluster / substrate
  node ids, not by name. *Front-gate:* it rides existing infrastructure — percepts carry
  `Percept.substrate_node_id` (set only when the substrate path is active — confirm it is set on the
  survival path before building) and ATL concepts already carry `memory_refs['hippocampus']` (16 of 46 in the measured run)
  — so it is a new lookup in `PatternCompleter`, not a new mechanism.
**Also in 2S:** the four scripted survival harnesses advance the experience clock (decision 1's
owed item). Salience changes what MemoryAgent forms and what captures carry, so this phase ships **opt-in** like
the rest and states its ledger rows (Exp 60–62 capture through the loop). **Revive-for-survival
trigger for Phase 2's validation:** Phase 2S merged and a survival run's saved records show non-zero
`activation_count` from `prediction`.

*Phase 2S amended 2026-09-24 (owner), after the premise check this section asked for.* A read-only
offline probe (a scripted water classroom running the full loop on `main` @ `df46321a`, counters
monkeypatched, no source edits) found that the two fixes above sit on the wrong path:

| | measured |
|---|---|
| percepts | 16, all `source="idle"`, all salience 0.0. The loop's observation carries `active_goal` / `mode` / `maxim_runtime` and **no body state**, so `PerceptionAgent` has nothing to score. |
| `Percept.substrate_node_id` | 0 of 16. It is set only by the LINGUISTIC encoder, from percept text, and survival percepts have none. |
| memory formations / pattern completions | 0 / 0 |
| loop captures | 2, **one per executed action**. On the trace (`EncodingSignals`), `salience` / `novelty` / `pain` are unmeasured (`None`) on both; only `surprise` is measured (0.5 on the escape) |
| situation clusters | encoded every tick (`propose_via_substrate`, `{modality: cluster_id}`), and **their EC node ids ARE ATL concept ids** (`sensors:food=…`, `sensors:distance_from_spawn=…`) |
| records linked to their situation | **none.** The ATL `memory_refs` come only from goal/tool text (`drive:threat(1.00`, `minecraft_player_flee`). No record stores a cluster id: `observations` is `{}`, or carries only the RPE-boosted `salience` (0.75 on the escape). |

So survival memories form at the **loop capture**, and the body lives on the loop's **cluster**
path. Both fixes move there, where the inputs exist (`ctrl.pending_proposal.clusters`,
`SensorEncoder.last_encode_margin`) — **except pain**, see 2S-c:

- **2S-a: the scripted harnesses drive the experience clock.** SHIPPED with this amendment.
  `water_trial.py`, `exp58_run.py` and `exp58_offline_gates.py` each run an `ExperienceClockDriver`
  per bypass run and advance it once per propose pass. `orient_backbone/exp53_cross_context_readout.py`
  was dropped from the list: its readout agent loads only an NAc and an EC, with **no Hippocampus**,
  so no clock and no captures (a driver there would satisfy a text check and do nothing). Guards:
  the former strict red gate, now a plain test; and `test_water_trial_smoke.py`'s donor sequence
  asserts that propose-only training advances the clock (red with the driver's calls removed).
- **2S-b: a capture records its situation.** The record stores its `{modality: cluster_id}`, and
  those clusters' ATL concepts (same ids) gain `memory_refs['hippocampus']` to it. This is the
  missing link, and it rides existing stores (no new mechanism).
- **2S-c: survival encoding signals at the loop capture.** SHIPPED 2026-09-24:
  - **Pain** (owner: the action's own failures first, felt pain as the fallback) — **nociception
    only**, per decision 4: the `ToolPainBridge`, which already hears every PainBus signal and tracks
    the running invocation, keeps per invocation the NOCICEPTIVE pain the action CAUSED (its
    delta-attributed `embodiment_failures`, `drive:*` excluded except health) and the peak
    NOCICEPTIVE pain FELT while it ran. What counts is decided ONCE, on the type:
    `PainSignal.kind` (`proprioception/pain.py::classify_pain`, `PainKind`), which `_pain_encoding`
    now reads too — the first draft skipped it and both review lenses measured air hunger 0.7, fear
    0.5 and tool frustration 0.3+ recorded as pain. `pop_invocation_pain` hands caused, else felt, to the executor,
    which stamps `ToolOutput.pain` beside `rpe`. `0.0` = watched, nothing fired; `None` = no pain
    source. NOT the dormant `consume_pain_intensity` stash. A latched drive breach publishes on entry
    and re-injury only, so standing distress is carried by drive pressure, not pain.
  - **Novelty**: `1 − best similarity` of each situation cluster's encode, the most novel modality.
    Coarse by construction (sensor completions sit at or above 0.85, so familiar reads ~0–0.15);
    `-1` (nothing comparable) is left UNMEASURED, because the tag would weight it by the Hippocampus
    count and give an empty comparison full confidence. The margins are read right after each encode
    and carried on the proposal (`LLMProposal.cluster_margins`), because the encoder's stash is
    overwritten by the next tick.
  - **Salience: deliberately NOT measured on this path (a departure from this plan's wording).**
    The plan said "salience from drive pressure, sensor change and pain", but `encoding_tag` already
    scores pain and relevance-gated drive pressure as their own deviations, so a salience built from
    them would count them twice. A distinct attention signal (sensor change alone) is a follow-up
    with its own design, not part of 2S-c.
  - **Recorded `None`s:** an unchanged tick (the encoder's min-delta gate runs no scan) records no
    novelty; interactive mode suppresses `record_tool_start`, so its captures record pain `None`.
  - **Known simplification:** novelty is still weighted by the Hippocampus's own trace count
    (`novelty_reference_size`), not the EC's comparable-node count. The tag's docstring anticipates
    the producer supplying its own set; that is owed and not done here.
- **2S-d: substrate-native pattern completion.** On a situation change, cue by the current cluster
  ids → the shared-id ATL concepts → their linked records → `MemoryLayer.activate(source="prediction")`.
  Any EC lookup goes through `EntorhinalCortex.pattern_complete_readonly` (it takes an embedding),
  never the encode path.
  **BUILT 2026-09-25 as RECALL ONLY** ([memory_2s_d_situation_cue.md](memory_2s_d_situation_cue.md)):
  owner amendment on review — no `activate` until 2S-e consumes the ids (an unconsumed cue must not
  credit `S`), a memory qualifies only through a shared world/audio cluster, and it is tiered by place
  (sound only where no place matches), then ordered by salience `max(encoding_tag, retro_tag)`, then
  shared sound, then recency (interoception never ranks). No EC lookup was
  needed (the ids are known on the tick). Lost links deferred:
  [deferred/situation_cue_fallback.md](deferred/situation_cue_fallback.md).
- **2S-e: a behavioural consumer (owner 2026-09-24: in scope).** Completion alone is bookkeeping,
  because the survival action path has no LLM to read a prediction. The consumer changes survival
  action selection, so per roadmap 1.4 Phase 5 it enters with its own plan section, a written
  front-gate answer and a four-lens design review, as a **declared opt-in arm** (never a silent
  default: Exp 60–62 run through this path). Candidates, and the front-gate reading of each:
  - **(A) episodic control:** recall this cluster's past outcomes as an action prior. It duplicates
    the NAc's fear store on the same cluster keys (`cluster_fear_alpha` 0.5, cap 1.0: it saturates
    in two writes). Its positive half also overlaps Phase 5's planned cluster-keyed relief store
    (R4: "relief never writes a world cluster").
  - **(B) generalization by pattern completion:** in a situation the NAc has not keyed, complete to
    the most similar past situation and carry over what happened there. It fills R1's measured gap
    (the fear misses in the night pool at 0.799, under the 0.85 threshold), and it is gated by Rung
    B's SUPPORT trace (`outstanding.md` O3). It is a **second candidate for roadmap 1.4 Phase 5's
    "Keying / generalization" slot**, beside a substrate keying rule (feature-invariant keys or
    cluster merging), so its front-gate answer compares the two. It **cannot be validated in the
    water classroom**: that has one binary discriminator, so its situation space is two points
    (roadmap §Recorded limit). It needs Exp 62's pools, or an open-world trace.
  - **(C) sequence recall:** multi-step credit. It overlaps the R4 routing audit, which comes first.

  **(B) CHOSEN (owner, 2026-09-24).** Owed before 2S-e is built: its plan section (the front-gate
  answer against the substrate keying rule, and the validation world), then the four-lens design review.
  **Inherited from 2S-d (2026-09-25), owed in that plan section:** (i) the **outcome-gated credit**
  — 2S-d recalls without `activate`, so crediting recalled memories (and how an outcome gates it)
  is 2S-e's; (ii) the **ranking policy** its consumer needs (2S-d returns the highest shared-modality tier,
  most salient first — `max(encoding_tag, retro_tag)` — then shared sound, then newest, capped at 20;
  interoception never ranks); (iii) (B) completes to a neighbouring cluster, then recalls through the STATELESS
  `PatternCompleter.recall_situation` (never `cue_situation`, which carries change detection);
  (iv) the validation world must contain **loop captures** — the propose-only phases give 2S-b no
  links, so the cue finds nothing there; (v) the lossy-refs fallback trigger
  ([deferred/situation_cue_fallback.md](deferred/situation_cue_fallback.md)).

**Phase 3 — sleep.** Opt-in `sleep()` in the generic sim loop too (the survival harnesses already
run it), all of it behind the strategy selection; replay updates `S` directly and does NOT go
through `MemoryLayer.activate` (or gets its own `replay` source), so it is never double-counted;
prioritized replay via `sleep_replay.py`; the homeostatic downscale; the forgetting rule with floors;
reversible compress-to-gist ([#816](https://github.com/dennys246/Maxim/issues/816) design half);
a persisted consolidation-candidate queue; promotion pressure moved onto the experience clock with
the spacing-correct update — **opt-in like the rest**, since it would otherwise change a pinned
`[engineering]` promotion invariant on the survival harnesses' own path (brief update +
`tests/integration/test_memory_hub.py` in the same PR).

**Phase 4 — capacity.** A per-store **byte budget** in `maxim config` (`memory.budget_mb` and a
derived split), floored at today's item caps so no earned short run can see a difference.
Forget the weakest only when over budget, down to a low-water mark (e.g. 90 %), via a lazily
maintained heap — no O(N) rescan per insert. Over-budget-but-nothing-forgettable is **loud**
([#819](https://github.com/dennys246/Maxim/issues/819)). Numbers cite a measurement:
`scripts/memory_cost_harness.py` (bytes per item, eviction latency, load time), modelled on the EC
scan-cost harness. Resource probing (RAM/disk) is rejected as a default: it makes runs
non-reproducible across machines. EC budgeting (non-world modalities only, cascading to NAc/ATL via
the existing deletion callbacks) is a later step.

**Phase 5 — earn it.** A pre-registered experiment through the four-lens design review
([docs/experiments/DESIGN_REVIEW.md](../experiments/DESIGN_REVIEW.md)). Sketch: a one-shot salient
episode and an equally rare neutral one; after N sleep cycles the salient one is retained and the
neutral one fades; an **ablated-tag** arm (tag forced to 0) must lose the difference; a spaced-vs-
massed retrieval arm tests the spacing direction; a **neutral-just-before-salient** arm tests the
look-back directly (and a neutral-just-after arm, if the forward window is adopted). Then the Exp 10 re-run, and a default flip
scheduled right after a minor-version heartbeat, when the affected rows are due anyway.

## Shared primitive: looking back

Three lines need the same thing — attach a signal to what was active *just before*:
retroactive tagging here, R4's delayed credit, and the language line's binding of a death message
to the second before it ([paired_data_audit_reaudit_2026-09-21.md](../experiments/paired_data_audit_reaudit_2026-09-21.md)).
**Decided 2026-09-24 by R4's look-back review ([lookback_primitive.md](lookback_primitive.md)): no
shared store.** Each consumer looks back over the record it already has — NAc's trace for R4's credit, the
Hippocampus (by enqueue-time experience µs) for tagging, and binding decides at revival.
`PerceptTraceBuffer` is Dormant. The SCN is the wrong clock for it (it bins by
time of day).

## Not in this plan (filed separately)

- [#818](https://github.com/dennys246/Maxim/issues/818) — NAc wall-clock decay ignores the
  inherent-class exemption. Touches Exp 56/57/61 inputs; out of round one's scope, fixed on its own.
- [#812](https://github.com/dennys246/Maxim/issues/812) — ATL typed relations share one update slot.

## Open questions

1. ~~Experience clock vs [scn_decay_anchoring](deferred/scn_decay_anchoring.md).~~ **Resolved
   2026-09-21:** one clock policy — the experience clock governs everything that strengthens or
   forgets a memory trace; wall time only at named sites where downtime is meant to count (today only
   `NAc.load()` decay-on-load). The SCN bins by time of day, the wrong clock for this; that plan's
   revive path is amended to scale NAc per-tick decay by `dt` on the experience clock instead. NAc
   stays out of round one. Documented asymmetry: a robot switched off for a month loses NAc biases
   but not memories.
2. **Where honest activation is counted for the LLM path** — does text the LLM *read* in a prompt
   count as retrieval? (Proposed: yes for the injected concepts; not for their neighbours.)
3. ~~Tag normalization~~ — **resolved 2026-09-21**: Phase 2 decision 4 (fixed baselines, per-drive
   relief normalisation, novelty's store baseline). Calibrating the constants remains the deferred
   calibration plan's first consumer.
4. **ATL vs Hippocampus parity** — the bio review's answer: concepts carry their own `S`, raised
   slowly by replay of the episodes that cite them (§Sleep step 1).
5. ~~Owner of the looking-back primitive~~ — **resolved 2026-09-24: R4, whose review decided no new store** (Phase 2 decision 6).
6. **Interference and reconsolidation are missing.** Forgetting here is decay; biologically much of it
   is interference (retroactive interference; retrieval-induced forgetting of close competitors), and a
   retrieval carrying a prediction error makes a trace labile and updatable (reconsolidation — the way
   being wrong gets corrected; today's "reconsolidation pull" never changes content). Cheapest first
   step: a retrieval slightly lowers `R` of near-embedding competitors. Not in round one; recorded so it
   is not forgotten.

## Review record (2026-09-21, before merge)

Two lenses read the first draft (bio-fidelity, wiring); both returned DO-NOT-BUILD items, all folded
above:
- **Wiring DO-NOT-BUILD:** the survival harnesses already run full `sleep()` — the draft's "no
  experiment exercised forgetting" was false and its opt-in guard pointed at the wrong place →
  §Guardrails rewritten (every sleep/consolidation/promotion change opt-in; fingerprints; built-object
  asserts).
- **Wiring DO-NOT-BUILD:** retroactive tagging could not find its traces (percept-id keyed, async
  captures, shared tick) → §The model's wiring requirements.
- **Bio DO-NOT-BUILD:** floors under a never-decaying tag reproduced immortality → floors under `R`,
  slowly decaying tags, budget eviction may take protected traces loudly.
- **Folded SHOULD-FIX:** noisy-OR over baseline deviations (the 0.5 default salience; max is not
  bio-faithful); RPE counted once; drive pressure gated by relevance; valence dropped; per-site signal
  availability and typed `EncodingSignals` (layer rules); seven capture sites; one owned, asserted
  experience clock; `memory.*` config brings env vars + scrubs + enum validation; one resolver across
  builders; activation lock discipline; saturating `S`; downscale scaled by experience time; replay
  feeds ATL; replay recency; honest naming of the look-back as an eligibility trace.
- **Recorded, not folded:** interference/reconsolidation (open question 6); the NIT on fingerprint
  hash compatibility is handled by adding new keys with old-row defaults.

# Memory strength and forgetting — a hippocampal forgetting model (1.4 parallel line)

> **ACTIVE — opened 2026-09-21 as a 1.4 PARALLEL LINE (owner decision).** Not a rung; it must not
> silently power E1–E3 ([roadmap_1_4.md](roadmap_1_4.md) §Parallel lines). Every behavioural change
> ships **opt-in, with today's defaults pinned byte-identical**, and every mechanism enters as
> `[engineering]` until an experiment earns it. **Entry condition: Phase 0** — the input-integrity
> defects fixed — before any phase changes what is kept or forgotten.

**Owns (proposed):** `src/maxim/memory/strategies.py` (a new strength strategy), the activation and
sleep paths in `memory/hippocampus*.py` and `memory/atl.py`, a `memory.*` block in the config
(`maxim config`), `scripts/memory_cost_harness.py` (new, Phase 4).
**Companion plans:** [grounded_language_acquisition.md](grounded_language_acquisition.md) and R4 in
[roadmap_1_4.md](roadmap_1_4.md) (same retrospective-tagging primitive — see §Shared primitive);
[deferred/decay_consolidation_calibration_plan.md](deferred/decay_consolidation_calibration_plan.md)
(would later *calibrate* this plan's constants); [deferred/scn_decay_anchoring.md](deferred/scn_decay_anchoring.md)
(wall-clock anchoring of NAc decay — a tension with this plan's experience clock, §Open questions);
[archive/memory_consolidation_practice.md](archive/memory_consolidation_practice.md) (the dormant
replay module's revive trigger).

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
- retroactive tagging uses `memory/percept_trace_buffer.py::PerceptTraceBuffer`, which exists and is
  tested, has **zero production constructors**, and so far persists only an empty buffer
  (non-empty snapshot round-trip is its own Stage 2) — this plan gives it a caller;
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
| Surprise | RPE via `executor.get_last_rpe` | loop capture. **Counted once:** `capture_episodic_memory` already folds *positive* RPE into salience — read it from one place (raw, with that fold removed on the strength path). Negative RPE (omission) is a distinct channel; whether it tags is a Phase 2 decision. The slot is sticky, never reset per tick — reset it or read it with a tick stamp. |
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
event are also captured) is a Phase 2 decision, with `W` on the experience clock. Wiring requirements:
`PerceptTraceBuffer` keys on `percept_id` and ticks every agent's entries together, while loop
captures are async (the memory id is minted later on the worker; the queue drops its oldest when full)
and pain captures are synchronous — so the drowning can be stored before the moments that preceded
it. Record **memory ids** in `Hippocampus._process_capture` with the loop's enqueue tick, make the
buffer per-agent, and resolve retro-tags lazily (or `flush()` before tagging).

**Retrieval (spacing effect, the right way round):** `S ← S · (1 + a · (1 − R) · S^(−w))`, then
`R = 1`. The gain is largest when the trace was fading, and the `S^(−w)` factor (FSRS's) saturates it
so many spaced retrievals cannot compound `S` without bound. Only **honest activations** count (Phase 1).

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

**Phase 2 — the strength model.** `S`, `R`, the typed `EncodingSignals` at all seven capture sites,
the look-back through `PerceptTraceBuffer` (its first production caller, per §The model's wiring
requirements), the retrieval update, the persisted experience clock with its advanced-clock assert.
No immortality floor under the new strategy. Decisions owed before code: the forward capture window,
whether negative RPE tags, and per-signal baselines.

**Phase 3 — sleep.** Opt-in `sleep()` in the generic sim loop too (the survival harnesses already
run it), all of it behind the strategy selection;
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
`PerceptTraceBuffer` is built for it and has no caller. **An owner must be named before Phase 2**
(open question 5) — "whoever wires it first" leaves the review unowned; the others become consumers. The SCN is the wrong clock for it (it bins by
time of day).

## Not in this plan (filed separately)

- [#818](https://github.com/dennys246/Maxim/issues/818) — NAc wall-clock decay ignores the
  inherent-class exemption. Touches Exp 56/57/61 inputs; out of round one's scope, fixed on its own.
- [#812](https://github.com/dennys246/Maxim/issues/812) — ATL typed relations share one update slot.

## Open questions

1. **Experience clock vs [scn_decay_anchoring](deferred/scn_decay_anchoring.md).** That plan ties
   NAc decay to wall-clock for hardware portability; this one moves memory forgetting *off* the
   wall clock so downtime is not disuse. They govern different stores, but the repo should have one
   stated clock policy. Proposed: experience time for forgetting, wall time only where the design
   explicitly wants downtime to count (NAc decay-on-load, documented).
2. **Where honest activation is counted for the LLM path** — does text the LLM *read* in a prompt
   count as retrieval? (Proposed: yes for the injected concepts; not for their neighbours.)
3. **Tag normalization** — per-signal scales for pain, drive pressure and |RPE| before the max; a
   Phase 2 calibration question (and the natural first consumer of the deferred calibration plan).
4. **ATL vs Hippocampus parity** — the bio review's answer: concepts carry their own `S`, raised
   slowly by replay of the episodes that cite them (§Sleep step 1).
5. **Owner of the looking-back primitive** — this line, R4, or the language line. Decide before
   Phase 2.
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

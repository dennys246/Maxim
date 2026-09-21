# Memory strength and forgetting — a hippocampal forgetting model (1.4 parallel line)

> **ACTIVE — opened 2026-09-21 as a 1.4 PARALLEL LINE (owner decision).** Not a rung; it must not
> silently power E1–E3 ([roadmap_1_4.md](roadmap_1_4.md) §Parallel lines). Every behavioural change
> ships **opt-in, with today's defaults pinned byte-identical**, and every mechanism enters as
> `[engineering]` until an experiment earns it. **Entry condition: Phase 0** — the input-integrity
> defects fixed and the ledger's decay/eviction triggers added — before any phase changes what is
> kept or forgotten.

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
| Be **tested** | Sims use the lightweight session end, which never calls `sleep()`: no experiment has exercised forgetting. |

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
- retroactive tagging uses `memory/percept_trace_buffer.py::PerceptTraceBuffer`, which exists, is
  tested and snapshot-persisted, and has **zero production constructors** — this plan gives it a caller;
- sleep replay uses the Dormant `memory/sleep_replay.py` (P8 gate passed) rather than a new replay.

The only genuinely new state is a per-trace storage strength and a persisted experience clock.

## The model (Bjork's storage/retrieval strength; the family FSRS instantiates)

Two quantities per trace (episode in the Hippocampus, concept in ATL):

- **Storage strength `S`** — how well-learned. Only grows (except the sleep downscale).
- **Retrievability `R = exp(−Δt / S)`** — how accessible now. `Δt` is on the **experience clock**
  (agent-active ticks, persisted, advancing only while the agent runs), never wall-clock.

**Encoding: `S₀ = s_base · (1 + k · tag)`**, where `tag` is the **max** (not the sum — the repo's
convention for combining needs, cf. the Wire 4 threat-need max-combine) of these existing signals,
each normalized to [0, 1]:

| Signal | Existing source |
|---|---|
| Salience | percept salience → `EpisodicMemory.salience` (after [#813](https://github.com/dennys246/Maxim/issues/813)) |
| Novelty | `EpisodicMemory.novelty`; EC text novelty `runtime/gating.py::TextSalienceScorer._compute_novelty`; DN gate `novelty_score` |
| Surprise | `\|RPE\|` — `NAc.last_rpe` / `executor.get_last_rpe` (already boosts salience in `bio_integration.capture_episodic_memory`) |
| Pain | `PainBus` intensity (pain captures already fire at ≥ 0.4) |
| Homeostatic pressure | distance from set point of `HomeostaticDriveSpec` drives; deprivation of `EntropicDriveSpec` drives (hunger, thirst, fatigue) — `embodiment/sem.py` |
| Relief | drive-relief credit (`runtime/tool_dispatch.py`, NAc relief) |
| Valence | `\|reward_bias\|` / percept valence of the nodes active at encoding |
| Failure | `Outcome.success == False` (after [#814](https://github.com/dennys246/Maxim/issues/814), [#815](https://github.com/dennys246/Maxim/issues/815)) |

The max-combine keeps any one strong signal sufficient and prevents a crowd of weak ones from
manufacturing importance. Which signal set a trace's tag is recorded on the trace, so every survivor
can say why it survived.

**Retroactive capture (synaptic tagging).** A strongly tagged event raises the tag of traces active
in the preceding window, weighted by distance — read from `PerceptTraceBuffer`. This is what lets
the ordinary moments before a drowning be remembered because of the drowning.

**Retrieval (spacing effect, the right way round):** `S ← S · (1 + a · (1 − R))`, then `R = 1`. The
gain is largest when the trace was fading. Only **honest activations** count (Phase 1).

**Sleep:** (1) replay the top-N traces by `tag · (1 − R)` — salient and fading — each gaining `S`;
(2) a global homeostatic downscale `S ← λ · S` (synaptic homeostasis: preserves the ranking, pushes
the weakest toward the floor); (3) **forget only if** `S < S_min` **and** `tag < τ` **and** schema
degree `< d_min` (in- *and* out-edges) **and** no ATL concept or NAc key cites the trace;
(4) **compress to gist before deleting**, reversibly (relearning "savings").

**Protection is a floor, not a term:** a high tag, schema links, ATL references, the LONG_TERM tier,
and inherent-class provenance each put a floor under a trace. Additive terms can be outvoted; floors
cannot. This is the answer to "rarely activated ≠ unimportant": one-shot fear and the once-a-year
fact survive by tag and by links, not by use.

## Guardrails

- **Defaults unchanged.** The strength strategy is selected through `maxim config`
  (`memory.strategy`), default today's `access_based`, pinned byte-identical by a guard test.
  No new env var (if one becomes unavoidable, it ships with its autouse conftest scrub).
- **Survival harnesses assert it is off** in their frozen configs, so an opt-in cannot leak into a
  campaign.
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
- Add **decay / eviction / cap** to the `Re-run on:` triggers of ledger rows 183, 188, 192–198
  (no row names them today; a forgetting change would fire only through generic categories).
- Correct `docs/agents/bio-memory.md`'s clock claim (done in this plan's PR).
- Each fix PR states which ledger rows it re-ran or discharged (each issue lists its candidates).

**Phase 1 — honest activation.** One `activate(ids, source=...)` path per store. Bookkeeping reads
(`get()` echo filters, session-start bulk recalls, deletion callbacks, neighbour lookups) stop
counting; real reactivation (pattern completion, cue retrieval, spreading activation, ATL recall and
prompt injection) starts counting. Behind the opt-in; the counters it feeds are read only by the
strength strategy until Phase 5 flips defaults.

**Phase 2 — the strength model.** `S`, `R`, the encoding tag from §The model, retroactive capture
through `PerceptTraceBuffer` (its first production caller), the retrieval update, the persisted
experience clock. No immortality floor under the new strategy.

**Phase 3 — sleep.** Opt-in `sleep()` in sims (so experiments can exercise forgetting at all);
prioritized replay via `sleep_replay.py`; the homeostatic downscale; the forgetting rule with floors;
reversible compress-to-gist ([#816](https://github.com/dennys246/Maxim/issues/816) design half);
a persisted consolidation-candidate queue; promotion pressure moved onto the experience clock with
the spacing-correct update (this changes a pinned `[engineering]` promotion invariant — brief update
+ `tests/integration/test_memory_hub.py` in the same PR).

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
massed retrieval arm tests the spacing direction. Then the Exp 10 re-run, and a default flip
scheduled right after a minor-version heartbeat, when the affected rows are due anyway.

## Shared primitive: looking back

Three lines need the same thing — attach a signal to what was active *just before*:
retroactive tagging here, R4's delayed credit, and the language line's binding of a death message
to the second before it ([paired_data_audit_reaudit_2026-09-21.md](../experiments/paired_data_audit_reaudit_2026-09-21.md)).
`PerceptTraceBuffer` is built for it and has no caller. Whichever line wires it first owns the
wiring and its review; the others become consumers. The SCN is the wrong clock for it (it bins by
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
4. **ATL vs Hippocampus parity** — whether concepts carry their own `S` or inherit from the
   episodes that cite them.

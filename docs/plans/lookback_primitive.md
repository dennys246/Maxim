# The look-back primitive — R4's design review of `PerceptTraceBuffer`

> **DECIDED 2026-09-24 (v2, owner): no new look-back store. Each consumer looks back over the record
> it already has; `PerceptTraceBuffer` is marked Dormant.** R4 owns the question (owner decision
> 2026-09-24); this review was scheduled ahead of R4's build as a stated exception to
> [roadmap_1_4.md](roadmap_1_4.md) §Phase 5 (design only). v1 proposed a shared, non-decaying activation
> history; a four-lens review (wiring, architecture, bio-fidelity, risk — all ADOPT WITH CHANGES) showed
> it had one near-term consumer, which needs no new store, and the owner chose v2. The CI check that
> forbids constructing `PerceptTraceBuffer` in production (`.github/workflows/test.yml`) now enforces
> its dormancy. Two live defects found here are filed: [#888](https://github.com/dennys246/Maxim/issues/888),
> [#889](https://github.com/dennys246/Maxim/issues/889).

## The question

Three lines need to attach a later signal to what was active *around* it:

| Consumer | Signal | What it needs to find | Window |
|---|---|---|---|
| **R4 delayed credit** (owner) | a reward or pain outcome | the situations and actions that preceded it | beyond the executor call window |
| **Memory strength — retroactive tagging** ([memory_strength_and_forgetting.md](memory_strength_and_forgetting.md); a backward-only eligibility trace, *not* synaptic tagging and capture) | a strong event (pain, relief) | the memories encoded just before it | τ = 10 s, cutoff 30 s on the experience clock, backward-only (a forward window is a planned arm) |
| **Word binding** (deferred — [deferred/grounded_word_binding.md](deferred/grounded_word_binding.md)) | a heard message | the situation it describes | a two-sided, BTSP-like window around the message; shape and the ~1 s figure are EXPLORATORY |

## What exists (verified against `main`, 2026-09-24)

1. **`decisions/nac.py::NAc._eligibility` — live.** `dict[(agent_id, key)] → float`, overwritten per
   write, ×0.9 per loop iteration (`NAc.decay_eligibility`, called only from
   `runtime/agent_loop.py::_loop_bio_tick_maintenance` — live on the survival loop and `--sim`), pruned
   below 0.01 (≈44 ticks). Written by `NAc.update_eligibility` from `SensorEncoder` (every situation
   cluster, every encode), `LinguisticEncoder`, the Dormant anticipatory pre-activation in
   `TemporalCreditDistributor.distribute`, and `TemporalCreditDistributor.record_event` (`tool:<name>`
   at 0.3 on completion and at pain intensity on failure; `embodiment:<entity>:<mode>`). Read by
   `TemporalCreditDistributor.distribute` on every reaction reward, which splits credit in proportion to
   trace strength and applies it through `NAc.credit_node` to `_reward_bias` (×0.15, clamp [0, 0.20]).
   `recommend_action` reads `_reward_bias` for `tool:*` keys (a ≤0.20 positive nudge); cluster-node
   credit reaches the LinguisticEncoder's thresholds and, via `reward_bias()`, the text-recalled concept
   annotations (`bio_enrichment.py`, `tools/discovery.py`). A wall-clock anchor tier (`_temporal_anchors`)
   extends the window. **This path is live on the EARNED survival loop but unfingerprinted**: no Exp
   60/61/62 fingerprint includes eligibility decay, the anchor window or the credit weight, and no
   Re-run on trigger names them.
2. **`memory/percept_trace_buffer.py::PerceptTraceBuffer` — never called.** Added 2026-04-11 as F0.2
   ([archive/foundations_plan.md](archive/foundations_plan.md) §F0.2); its exit criterion (*NAc's reward
   crediting reads this buffer*) was never met — NAc grew its own trace. No production constructor; its
   external references are snapshot and fixture plumbing that receive `None`.
3. **The experience clock exists.** `memory/experience_clock.py::ExperienceClock` — per agent, integer
   microseconds (`world_experience_us`), persisted, owned by `Hippocampus.experience_clock`, advanced per
   agent per live loop pass by `runtime/experience_time.py::ExperienceClockDriver` (pauses subtracted; a
   turn quantum in text sims). v1 of this document wrongly said it did not exist.
4. **The Hippocampus is already a time-ordered record of what the agent encoded.** Memories carry
   `retrievability_anchor_us`, stamped by `_stamp_encoding_strength` wherever `capture()` runs — on the
   calling thread for the synchronous doors, but on the **worker** for the async loop path
   (`_process_capture` → `capture_from_loop` → `capture`), i.e. after the moment, by the queue's lag.
   `_CaptureRequest.queued_at` is a float wall time, not experience time. No per-memory sequence exists
   (`_captures_this_process` is a process-local counter, incremented on insert, never stored).

## Findings

- **F1 — `PerceptTraceBuffer`'s clock is per call.** `tick()` multiplies every entry by `exp(-1/tau)`
  once per call; `tick_rate` is stored but never used; `TraceEntry.registered_at` is `time.monotonic()`
  (commented "wall-clock") and never read. Tests pin the per-call decay.
- **F2 — it is not per-agent.** One flat list, one tick counter, capacity eviction across agents.
- **F3 — it stores decayed weights**, so one decay would serve every consumer, while each needs its own
  constants (the memory line: share "the primitive and clock, never its constants").
- **F4 — NAc's temporal anchors never expire within a session (live; filed as #888).** An anchor is
  pruned only on the call where its fast trace is deleted and only if older than 300 s; at any tick rate
  faster than ~7 s/tick (Exp 60 ran at 4 Hz) it is never revisited, `distribute` has no age gate, and
  nothing clears anchors. Every visited situation cluster keeps a share of every reward (within a
  session phase similarity ≈ 1, so each stale anchor weighs 0.3 × its original activation). Measured by
  the wiring lens with one `tool:swim` trace, the current cluster and N stale anchors — the `tool:*`
  share falls 0.25 → 0.071 → 0.019 → 0.005 at N = 0 / 10 / 50 / 200 — and the weighting is
  non-monotonic (a key 43 ticks old gets a 0.035 share; expired at 44, it jumps to 0.5). The Exp 60–62
  DVs run through fear, not `_reward_bias`: **no effect on those results is claimed; behavioural impact
  is unmeasured.** Because the path is unfingerprinted, the owner's decision is the only gate on its fix
  (#888's acceptance criteria).
- **F5 — the ablation switch does not ablate the live path (filed as #889).**
  `MAXIM_NAC_REWARD_BIAS_DISABLED=1` no-ops `distribute_reward` (no production caller),
  `decay_reward_biases` and `get_agent_tool_biases` — but the live write (`credit_node`) and the
  selection read (`reward_bias` in `recommend_action`) never check it, and the skipped decay means the
  bias the live path writes never shrinks. Reproduced: switch on, one reward → `reward_bias` = 0.15, and
  it stays. Exp 37's arm 3 is to be checked.
- **F6 — stale claims in code** (`nac.py`'s `_eligibility` comment and `update_eligibility` docstring,
  `reactions/types.py::TraceSnapshot`, `agents/working_memory.py`'s module docstring and
  `WorkingMemoryKind.ACTIVATION`) described a `PerceptTraceBuffer` wiring that never existed. Corrected
  in this change.
- **F7 — the roadmap's R4 wording.** `_reward_bias` is read by selection for `tool:*` keys (weak,
  cluster-blind), and `record_tool_complete`, not `record_tool_start`, writes the trace. Corrected in
  [roadmap_1_4.md](roadmap_1_4.md).

## The decision (v2)

**D1 — No new look-back store.** Front-gate scope pressure: with R4 on NAc's trace and binding deferred,
retroactive tagging is the only near-term consumer, and it needs no new structure — the Hippocampus is
already the time-ordered record of what the agent encoded. The bio-fidelity lens reached the same place
from the other side: biology keeps a *local* trace per circuit that decays in place (striatal
eligibility 0.3–2 s, Yagishita et al. 2014; synaptic tags ~1–2 h, Frey & Morris 1997), and no region
keeps a shared activity log — so a shared, non-decaying record would be an engineering convenience, not
the faithful part. What *is* faithful — separate constants per consumer — each consumer gets by owning
its own kernel over its own record.

**D2 — Each consumer looks back over the record it already has, on the experience clock.**
- **R4 delayed credit → `NAc._eligibility`** (unchanged by this design; its defects are #888 and #889).
  Whether R4 later needs per-encode situation history beyond it is R4's routing audit's question.
- **Retroactive tagging → the Hippocampus.** The one change, in the signature so no door can miss it:
  `capture()` takes a keyword-only **`experience_us`** (default `experience_clock.now_us()`) and a
  keyword-only **`capture_seq`** (default: the next value of a per-agent counter), and records both on the
  memory. The async path is the only one that needs to pass them: `_CaptureRequest` gains both fields,
  stamped at enqueue in `capture_from_loop_async`, and `capture_from_loop` threads them into `capture()`.
  Every synchronous door (`pain_bus`, `tool_pain_bridge`, `cerebellum`, `memory_agent`, `Hippocampus.store`,
  `Hippocampus.store_observation`, `create.py`) already calls `capture()` on the moment's own thread, so the
  default is right for them by construction — no per-door stamping. Tagging then windows over memories by
  `experience_us` (with `capture_seq` ordering captures that share one loop pass's timestamp — the
  before/after question tagging asks) with its own kernel (τ = 10 s, cutoff 30 s). No handle binding, no
  second writer thread. A capture the queue drops, or that `store_observation`'s dedup returns `""` for,
  was never a memory, so it has nothing to tag. Unit: integer µs (`ExperienceClock.now_us()`); never
  seconds (`memory/encoding.py`'s `S_UNIT` rationale). (`capture_reaction` is not a door: it appends a
  Reaction to the pending episode and makes no memory.)
- **Word binding → decided at revival**, as association in time with a two-sided kernel, plausibly
  over the same Hippocampus record (if heard messages are captured there — to verify at revival); its window comes from
  the natural-lag capture.

**D3 — `PerceptTraceBuffer` is Dormant.** Its module docstring carries `Dormant since 2026-09-24` with
the reason (dormancy over deletion: the snapshot plumbing and tests stay). The CI check that forbids a
production construction now enforces the dormancy rather than waiting for a design.
**Revive when** two consumers need encode-by-encode activation history with different kernels (e.g.
R4's routing audit needs per-encode situation history *and* word binding revives), and not before.

**D4 — Known duplication, recorded.** Two look-back mechanisms exist (NAc's live trace; a Dormant
buffer). Consolidating them is not attempted here; a later migration of NAc onto any shared record
would change an unfingerprinted live path, so it would **fire the Exp 60–62 re-runs** — there is no
replay capture of the eligibility/credit call sequence that could show byte-identity, and none is
assumed.

## Consequences

- **The memory line's Phase 2 look-back is unblocked:** its tagging reads the Hippocampus with
  enqueue-time `experience_us` at every capture door (decision 6 in its plan is updated).
- **R4's build** starts from NAc's trace, after #888 and #889.
- **The CI check** in `.github/workflows/test.yml` now names dormancy, not a pending design, as its
  reason.

## Review record

v1 (2026-09-24) proposed a shared activation history. Four parallel lenses, all **ADOPT WITH CHANGES**:
**wiring** (the capture handle had no path in code; the situation write point was too coarse; F4
measured, with the non-monotonic weighting and the missing session clear; the ablation bypass found);
**architecture** (the experience clock already exists — D2 was stale; the dump version bump could not
land alone because snapshot schema versions move in lockstep; **front-gate: the only near-term consumer
needs no new store**); **bio-fidelity** (the shared log is engineering, not biology; tagging is not
synaptic tagging and capture; binding is association in time, two-sided); **risk** (the path is
unfingerprinted, not fingerprinted as v1 said; F4's acceptance criteria; no replay harness exists, so a
byte-identical migration criterion was uncheckable; no effect on EARNED DVs may be claimed). The owner
chose v2 and approved filing #888 and #889.

## Where this is referenced

[roadmap_1_4.md](roadmap_1_4.md) §Phase 5 (R4) · [memory_strength_and_forgetting.md](memory_strength_and_forgetting.md)
(decision 6) · [deferred/grounded_word_binding.md](deferred/grounded_word_binding.md) (L2) ·
[three_factor_credit_assignment.md](three_factor_credit_assignment.md) (the R4 map).

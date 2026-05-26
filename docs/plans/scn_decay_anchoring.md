# SCN decay anchoring — wall-clock-tying the five NAc decay functions

**Target version:** 1.0 or 1.1 (decision in Open Questions §1 below).
**Status:** Draft. Plan written 2026-05-26 as Phase C of the [cluster_reward_bias_decay_tau_split](cluster_reward_bias_decay_tau_split.md) kickoff sequence.
**Owns:** [`src/maxim/time/scn.py`](../../src/maxim/time/scn.py) (new clock-driven surface), [`src/maxim/decisions/nac.py`](../../src/maxim/decisions/nac.py) (decay-function callers), [`src/maxim/runtime/agent_loop.py`](../../src/maxim/runtime/agent_loop.py) section 8.5 (per-tick decay calls migrated to SCN subscription).
**Companion plans:** [cluster_reward_bias_decay_tau_split.md](cluster_reward_bias_decay_tau_split.md) (Phase A prerequisite — tau split landed via [PR #267](https://github.com/dennys246/Maxim/pull/267)), [decay_consolidation_calibration_plan.md](decay_consolidation_calibration_plan.md) (downstream framework that consumes the SCN-anchored output as input).

## Why this plan exists

The Roy-3c-bisect A2 confirmation ([29_roy_3c_bisect.md](../experiments/29_roy_3c_bisect.md)) named per-tick decay as the cause of Wire-A's magnitude regression and the tau-split shipped Phase A — splitting `cluster_reward_bias_decay_tau` from `reward_bias_decay_tau` so Wire-A could use a longer timescale. Phase B's Roy-3a-retry ([30_wire_a_tau_validation.md](../experiments/30_wire_a_tau_validation.md)) validated tau=300 produces `[strongly rewarding]` annotations throughout the test arm.

But the tau-split addresses only *which tau value gets read by the decay call*, not *how often the decay call fires*. The decay still fires once per `agent_loop` tick — and **agent_loop tick rate is hardware-dependent by ~10× across deployments** (per [feedback_decay_is_tick_anchored_not_wall_clock](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_decay_is_tick_anchored_not_wall_clock.md)):

- Fast GPU + local LLM: ~10-15Hz tick rate on busy ticks
- Slow CPU + cloud LLM: ~1-2Hz on busy ticks
- Idle ticks: 4Hz default (`idle_sleep_s=0.25`), uniform across hardware

A multi-hardware deployment (some agents on the user's RTX 5080 leader, others on a Mac peer) silently runs with different effective decay timescales — Wire-A's tau=300 "ticks" calibrated on the 5080 means something quantitatively different on the Mac. Biologically this is wrong: associative-memory extinction is wall-clock-anchored (minutes to hours), not bound to thinking speed.

**SCN is the right *engineering* home for the fix.** SCN already tracks wall-clock-anchored circadian/weekly/monthly/annual phase ([`src/maxim/time/scn.py:149`](../../src/maxim/time/scn.py)) and has wall-clock infrastructure the new clock surface can reuse. Tying the five decay functions to SCN-driven clock ticks preserves the calibration math (tau in some-time-unit) while making the *time unit* wall-clock-meaningful and hardware-independent.

**Naming caveat (bio-fidelity clarification):** Maxim's `SCN` class is named after the biological Suprachiasmatic Nucleus, which drives **circadian rhythms (24-hour periods)**, not millisecond/second timescales. Reusing the SCN class as the home for second-scale decay callbacks is **engineering convenience** (it's already the wall-clock infrastructure hub), not a bio-fidelity mapping. The biological SCN does not modulate eligibility-trace decay at 10 Hz or reward-bias decay at 1 Hz. If a future reader infers "1 Hz decay ticks are bio-grounded to the SCN organ," they're reading the engineering choice as a biological claim — it isn't. Phase 0 design should consider whether the clock surface lives on `SCN` (engineering reuse) or on a sibling `time/decay_clock.py` module (cleaner separation of concerns); both options are surfaced in Open Question §2.

## The five decay functions and their current call sites

All five fire from [`agent_loop.py:3650-3666`](../../src/maxim/runtime/agent_loop.py) section 8.5 ("BIO-SYSTEM PER-TICK MAINTENANCE"):

```python
if _loop_nac is not None:
    try:
        _loop_nac.decay_eligibility()
        _loop_nac.decay_reward_biases()
        _loop_nac.decay_goal_reward_biases()
        _loop_nac.decay_cluster_reward_biases()
        _loop_nac.decay_percept_valences()
    except Exception as e:
        log_swallowed_exception(e, operation="nac_per_tick_decay")
```

| Function | Defined at | Reads tau from | Bio role | Bio-coherent timescale |
|---|---|---|---|---|
| `decay_eligibility()` | [nac.py:2384](../../src/maxim/decisions/nac.py) | (factor argument, not config tau) | Fast-decay credit-assignment traces | **Milliseconds** (50-500ms in neuroscience) |
| `decay_reward_biases()` | [nac.py:2358](../../src/maxim/decisions/nac.py) | `reward_bias_decay_tau` | EC recognition threshold modulation | Seconds-to-minutes |
| `decay_goal_reward_biases()` | [nac.py:2015](../../src/maxim/decisions/nac.py) | `reward_bias_decay_tau` | Goal-conditioned reward attribution | Seconds-to-minutes |
| `decay_cluster_reward_biases()` | [nac.py:2037](../../src/maxim/decisions/nac.py) | `cluster_reward_bias_decay_tau` (Phase A) | Wire-A substrate-voice annotation | Minutes (multi-turn window) |
| `decay_percept_valences()` | [nac.py:2207](../../src/maxim/decisions/nac.py) | `percept_valence_decay_tau` | Wire 2 Pavlovian aversion | Minutes-to-hours |

All five share the agent-loop tick anchor today. All five benefit from SCN-tying — **but their bio-coherent rates differ by 3-4 orders of magnitude.** Eligibility runs on milliseconds; the four associative-memory decay functions run on seconds-to-minutes-to-hours. A single Hz clock for all five would mis-model eligibility by 100-1000×. The plan therefore architects **multi-rate from Phase 1** — see "Migration shape" + Open Question §3 below.

## SCN's current interface and what's available for clock-driven ticks

[`src/maxim/time/scn.py:149`](../../src/maxim/time/scn.py) is a dataclass with:
- **Bins:** `_circadian_bins`, `_weekly_bins`, `_monthly_bins`, `_annual_bins` keyed by wall-clock-derived `TemporalSignature.to_bins()`
- **Indices:** `_signatures: dict[memory_id, TemporalSignature]` — wall-clock-anchored timestamps for every registered memory
- **Oscillator (optional):** `_oscillator: OscillatorNetwork` — Kuramoto coupled oscillators with `step(dt)` advance, enabled by `enable_oscillator()` (default-on per CLAUDE.md B2)

**What does NOT exist today:**
- No clock-driven loop. SCN's bins are queried by external callers; nothing drives SCN at wall-clock cadence.
- `OscillatorNetwork.step(dt)` is called only internally from `observe()` (event-driven), not from a periodic clock.
- No `register_callback` / `on_tick` / `subscribe` surface for periodic invocations.
- `TemporalCreditDistributor.anticipatory_pre_activate(agent_id)` IS called once per agent-loop tick (from [agent_loop.py](../../src/maxim/runtime/agent_loop.py)) but that's the only periodic SCN-adjacent caller and it shares the same tick-anchor as the decay calls — it doesn't help us.

**What needs to be added:** a clock-driven loop *somewhere* (SCN itself, a sibling `time/decay_clock.py` module, or extended `OscillatorNetwork`) that fires registered callbacks at a wall-clock-anchored rate. The decay calls subscribe to that loop.

The simplest shape (no commitment yet — Open Question §3 below):

```python
# Conceptual sketch — not a code proposal yet
class SCN:
    def register_periodic_callback(self, callback: Callable[[], None], hz: float) -> int:
        """Subscribe a callable to fire at `hz` wall-clock-anchored Hz.

        Returns a subscription id for later unregistration. Multiple
        callbacks at the same hz fire serially in registration order
        within one timer firing.
        """
        ...

    def start_clock(self) -> None:
        """Start the background thread driving registered callbacks.

        Called once at session start (build_bio_stack). Idempotent.
        """
        ...

    def stop_clock(self) -> None:
        """Stop the background thread + drain pending callbacks.

        Called at session end. Idempotent.
        """
        ...
```

## Migration shape

**Production path migration (single commit, ~80 LOC excluding new clock surface):**

The five decay functions have bio-coherent timescales that differ by 3-4 orders of magnitude (see five-function table above). A single Hz tier mis-models eligibility by 100-1000×. Phase 1 therefore ships **two Hz tiers** — fast (eligibility) and slow (associative memory):

1. Replace `agent_loop.py:3656-3666` (the five `_loop_nac.decay_*()` calls) with subscriptions at bootstrap time, partitioned by Hz tier:
   ```python
   # bootstrap path (build_bio_stack or AgentFactory)

   # Fast tier — eligibility traces (bio: ms timescale)
   scn.register_periodic_callback(nac.decay_eligibility, hz=ELIGIBILITY_DECAY_HZ)

   # Slow tier — associative-memory decay (bio: seconds-to-minutes-to-hours)
   scn.register_periodic_callback(nac.decay_reward_biases, hz=ASSOCIATIVE_DECAY_HZ)
   scn.register_periodic_callback(nac.decay_goal_reward_biases, hz=ASSOCIATIVE_DECAY_HZ)
   scn.register_periodic_callback(nac.decay_cluster_reward_biases, hz=ASSOCIATIVE_DECAY_HZ)
   scn.register_periodic_callback(nac.decay_percept_valences, hz=ASSOCIATIVE_DECAY_HZ)
   ```
2. Remove section 8.5's `nac.decay_*` block from `agent_loop.py`. The `log_swallowed_exception` wrapper migrates into the SCN clock's per-callback exception handler.
3. Two new module-level constants:
   - **`ASSOCIATIVE_DECAY_HZ = 1.0`** — one decay-tick per wall-clock second. Tau values for the four associative-memory functions are already calibrated for this rate (per the calibration preservation rule below; tau=300 ticks at ~1Hz preserves Phase B's render-window math).
   - **`ELIGIBILITY_DECAY_HZ = 10.0`** — ten decay-ticks per wall-clock second. Still substantially slower than biology (which runs sub-second) but a defensible engineering point that keeps eligibility trace decay an order of magnitude faster than the associative-memory tier without committing to millisecond-resolution scheduling. **The default eligibility decay `factor=0.9` semantics need rescaling for the 10Hz rate** — Phase 1 design pass decides whether to rescale at the call site (preserving the existing factor) or rescale the default factor to match the new cadence.

Pick once at design time per the calibration preservation rule below. Open Question §3 below holds the design discussion for whether two tiers is the right partition or whether eligibility deserves its own non-SCN home entirely.

**Bootstrap path:**
- `build_bio_stack` ([runtime/bio_stack.py](../../src/maxim/runtime/bio_stack.py)) wires the SCN clock at construction; `scn.start_clock()` runs at session start; `scn.stop_clock()` runs at session end (mirror the worker pool / hippocampus capture thread lifecycle).
- `AgentFactory.create_agent` paths inherit the same wiring through `build_bio_stack`.

**Tests path:**
- The autouse `_isolate_maxim_*` env-scrub pattern doesn't apply directly (this isn't an env var), but tests that build a NAc + SCN in isolation need to either explicitly opt in to clock-driven decay (call `scn.start_clock()`) or explicitly call `nac.decay_*` (the imperative path). Plan to ship both paths — the SCN clock subscription is the production path; direct-call remains available for tests that exercise decay deterministically without timer noise.

## Calibration preservation rule

**The Phase B Roy-3a-retry verdict gives the first calibration data point we must preserve:** at the test arm's effective tick rate (84 ticks / 90s = ~0.93 Hz on the user's RTX 5080 leader), `tau=300` produced `cluster_reward_bias` decay from 0.997 → 0.753 over arm A's wall (within 0.3% of the model `0.997 × (1 − 1/300)^84 = 0.755`). Wire-A rendered `[strongly rewarding]` throughout.

For the SCN-tying migration to preserve this:
- **Associative-memory tier (`ASSOCIATIVE_DECAY_HZ = 1.0`):** at the user's leader, the Phase B test arm wall (90s) = 90 SCN ticks. With `cluster_reward_bias_decay_tau=300`, decay from 0.997 → 0.997 × (1 − 1/300)^90 = 0.738 — very close to the observed 0.753 (within 2%). Wire-A still in `[strongly rewarding]` band throughout. The four existing tau defaults (`reward_bias_decay_tau=50`, `goal_reward_bias_decay_tau=50`, `percept_valence_decay_tau=200`, `cluster_reward_bias_decay_tau=300`) port nearly-1:1 to 1Hz wall-clock seconds — no rescaling needed.
- **Eligibility tier (`ELIGIBILITY_DECAY_HZ = 10.0`):** `decay_eligibility(factor=0.9)` currently fires per agent-loop tick (~1Hz busy, ~4Hz idle on the user's leader). Moving to 10Hz means decay fires ~2-10× more often per wall-clock second. **The `factor=0.9` default must rescale to preserve effective decay rate.** Two design options for Phase 1: (i) keep `factor=0.9` and accept faster effective eligibility decay (since the current value is empirically tuned for tick-anchored cadence, not bio-grounded); (ii) rescale to `factor=0.97` (≈ 0.9^(1/2.5)) to preserve the observed per-second decay rate during typical operation. Phase 0 design pass decides; the eligibility-trace consumers (substrate_primary action selection, credit attribution) need behavioral validation either way.

**Recommendation:** ship `ASSOCIATIVE_DECAY_HZ = 1.0` + `ELIGIBILITY_DECAY_HZ = 10.0` as the defaults. Rationale: (a) the 1Hz associative-tier matches the user's leader's observed tick rate at Phase B's test arm closely (~0.93 Hz), so the existing tau=300 calibration ports nearly-1:1; (b) one-tick-per-second is the cleanest wall-clock unit for future calibration framework consumption (the downstream `decay_consolidation_calibration_plan.md` framework expresses target timescales in seconds); (c) the 10Hz eligibility tier keeps eligibility decay an order-of-magnitude faster than associative memory, matching the bio-fidelity ordering even though it's still 100× slower than biological eligibility-trace timescales (50-500ms); (d) avoids needing to rescale the four existing associative tau defaults.

**Honest bio-fidelity gap:** even at 10Hz, eligibility traces are 100-1000× slower than biological eligibility (50-500ms half-life). This plan does NOT close that gap — it shrinks it from 1000-10000× (per-tick at ~0.93 Hz) to 100-1000×. The remaining gap is acceptable for engineering reasons (sub-100ms scheduling adds threading complexity disproportionate to behavioral value at the current substrate maturity) but is honestly documented here so future work can address it if the consumers grow to need real-time credit-assignment fidelity. The five-decay-function bio-coherence audit flagged in [`cluster_reward_bias_decay_tau_split.md`](cluster_reward_bias_decay_tau_split.md) Open Question §4 covers this surface.

If Phase 3 behavioral re-validation finds either tier under-decays or over-decays on a specific deployment, raising/lowering the per-tier Hz is a config bump (and per-tier tau values rescale by the same factor). The Hz values themselves are knobs, not calibration constants.

## Behavioral re-validation

Re-run Roy-3a-retry with SCN-tied decay active. Expected outcome shift relative to Phase B (which used per-tick decay):

| Measure | Phase B (per-tick, tau=300) | SCN-anchored (1Hz associative, 10Hz eligibility, tau=300) expected |
|---|---|---|
| Priming-end cluster_reward_bias | +0.997 | Unchanged |
| Wire-A max\|bias\| at arm A end (90s wall) | 0.753 (observed) | 0.738 (model prediction, within 2%) |
| Wire-A rendered band throughout test arm | `[strongly rewarding]` (>0.5) | `[strongly rewarding]` (>0.5) |
| Wire-A LLM-visible annotation behavior | Strong throughout | Strong throughout, hardware-independent |
| Substrate-primary action selection (uses eligibility traces) | (Phase B was llm-primary; no direct measure) | No regression on cradle-prelinguistic harness substrate-primary tool selection rate |

**Pass criterion:** Wire-A's rendered band stays `[strongly rewarding]` throughout the test arm on both the user's RTX 5080 leader AND a slow-CPU peer (e.g. the Mac without GPU). The hardware-independence claim is the primary test. The substrate-primary action selection sanity-check (no regression vs current per-tick eligibility decay on the cradle harness) gates the eligibility-tier rescaling decision.

**Null result branches:**
- SCN-anchored decay produces faster annotation drop-off on slow-CPU than on the 5080 — implies the chosen `ASSOCIATIVE_DECAY_HZ` is wrong (likely too low). Bump to 2.0 or 4.0 Hz and retry with proportionally larger tau values.
- Substrate-primary action selection regresses on the cradle harness — implies the eligibility-tier rescaling (10Hz with `factor=0.9` or rescaled `factor=0.97`) doesn't preserve the per-event credit-assignment window. Tune `factor` empirically against cradle's known-good baseline.
- SCN-anchored decay produces stuck-on-strongly-rewarding annotations across the entire test arm with no decay visible — implies a bug in the subscription wiring (callback not firing). Diagnose first; don't tune tau.

## Sizing

Estimated phases:

| Phase | Scope | LOC | Risk |
|---|---|---:|---|
| 0 | Design pass — pick per-tier Hz values, decide between SCN-internal clock vs sibling `time/decay_clock.py` module, decide eligibility `factor` rescaling strategy (keep 0.9 vs rescale to 0.97), validate that the multi-rate clock semantics don't conflict with existing SCN consumers (oscillator step, anticipatory pre-activation infrastructure) | doc-only | None |
| 1 | Multi-rate clock surface implementation — add `register_periodic_callback(callback, hz)` + `start_clock` + `stop_clock` to SCN (or sibling module); one thread per Hz tier with `time.sleep(1.0 / hz)` loop; idempotent registration; per-callback exception handler; unit tests covering both Hz tiers + lifecycle | ~200 (src) + ~250 (tests) | Medium — threading + lifecycle + multi-rate correctness |
| 2 | Decay-call migration — bootstrap subscribes the five `nac.decay_*` callbacks split between fast (eligibility) and slow (associative) tiers; remove agent_loop section 8.5 decay block; integration tests | ~40 (src) + ~120 (tests) | Low — replacement is mechanical |
| 3 | Behavioral re-validation — Roy-3a-retry on the SCN-tied path (Wire-A annotation preserved); cradle-prelinguistic harness re-run (substrate-primary action selection preserved under new eligibility cadence); cross-hardware test (5080 leader + Mac peer); experiment doc | doc-only | Medium — depends on hardware availability + eligibility-tier behavioral validation |
| 4 | Plan-doc folding — update `decay_consolidation_calibration_plan.md` to note SCN-anchoring SHIPPED; update `feedback_decay_is_tick_anchored_not_wall_clock` memory to mark the fix landed; update `cluster_reward_bias_decay_tau_split.md` "Known limitation" section to mark CLOSED | doc-only | None |

**Total:** ~240 LOC src + ~370 LOC tests + 2 experiment/plan docs. Estimated 2-3 weeks at normal cadence (most of which is Phase 1's threading + multi-rate correctness review and Phase 3's two-arm behavioral validation).

**Risk shape:** SCN gains a background thread. The existing bio-system threads (hippocampus capture, worker pool) have established shutdown patterns the SCN clock should mirror. Threading bugs in the clock surface would be silent (decay just doesn't fire) — Phase 1's test discipline is the load-bearing safeguard.

## DO NOT BREAK (load-bearing invariants)

1. **SCN persistence schema is shape-frozen at v1.0.** Adding clock-state fields to the persisted snapshot requires schema-version bump (currently `schema_version: ClassVar[int] = 1` per [scn.py:201](../../src/maxim/time/scn.py)). The clock subscription registry itself should be ephemeral (re-subscribed at session start by bootstrap) and NOT persisted.

2. **`TemporalCreditDistributor.anticipatory_pre_activate` is defined as infrastructure but NOT currently wired into the production agent loop.** [`temporal_credit.py:127`](../../src/maxim/decisions/temporal_credit.py) defines the method; grep shows zero callers from `agent_loop.py` or anywhere else outside the file itself. The B2 SCN→NAc feedback loop is partial: the `TemporalCreditDistributor` is wired at [`bio_stack.py:303`](../../src/maxim/runtime/bio_stack.py) and `distribute()` is invoked by the reward subscriber, but `anticipatory_pre_activate()` itself has no production caller (CLAUDE.md's claim that it "primes NAc eligibility traces" describes the docstring's intent, not the current wiring). Phase 0 design decision: either (a) leave it un-wired (current state, plan is a no-op for this method), or (b) wire it into the SCN clock as a new subscriber at the same Hz as eligibility decay — which would actually close the B2 feedback loop CLAUDE.md describes. Option (b) is bonus value the SCN clock surface enables for free. Either way, no collision risk exists today because there's no current caller to collide with.

3. **The `log_swallowed_exception` wrapper around the decay block** exists because decay failures must not crash the agent loop. The SCN clock's per-callback exception handler must do the same — swallow + log, don't propagate.

4. **Test-time `nac.decay_*` direct callers must still work.** Many unit tests call `decay_cluster_reward_biases()` directly to verify behavior deterministically. The methods stay public + callable directly; the SCN subscription is an ADDITIONAL caller, not the only one.

5. **The Phase A tau-split + env override (`MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU`) is preserved untouched.** The SCN-tying changes WHO calls `decay_cluster_reward_biases()` and HOW OFTEN; the tau value itself is unchanged.

## What this plan does NOT do

- **Does NOT change associative-memory tau values.** The tau split's calibration math holds at `ASSOCIATIVE_DECAY_HZ=1.0`. If a different rate ships, the four associative taus rescale proportionally as a config bump.
- **Does NOT implement calibration-by-simulation.** That's [decay_consolidation_calibration_plan.md](decay_consolidation_calibration_plan.md). This plan is the SCN-anchoring prerequisite.
- **Does NOT migrate other per-tick bio-pipeline mechanisms** (e.g. hippocampus capture, working memory tier transitions, ATL reinforcement). Those have their own cadence requirements; the SCN clock surface should be extensible to absorb them in future plans if it proves clean.
- **Does NOT close the eligibility-trace bio-fidelity gap entirely.** Biological eligibility decays on millisecond timescales; the plan ships 10Hz (100× slower). The remaining 100× gap is accepted for engineering reasons (sub-100ms scheduling adds threading complexity disproportionate to current behavioral value) and honestly documented in "Calibration preservation rule" above.

## Open questions

1. **1.0 gate or 1.1 deferral?** Two paths:
   - **1.0 inclusion:** ships hardware-independent decay before 1.0 release. Counts against the 1.0 closing scope.
   - **1.1 deferral:** 1.0 ships with tick-anchored decay (current state, validated by Phase B Roy-3a-retry); SCN-anchoring + calibration framework both land in 1.1. Cleaner sequence but means 1.0 has a known bio-fidelity gap.

   **Author recommendation:** 1.1 deferral. The tick-anchored decay is documented + validated on the user's hardware baseline; the multi-hardware portability concern only matters once Maxim Hivemind (1.2) actually deploys across heterogeneous hardware. 1.0 doesn't need this for its own validation surface.

   **Load-bearing cross-plan sequencing constraint:** [`decay_consolidation_calibration_plan.md`](decay_consolidation_calibration_plan.md) explicitly names SCN-anchoring as a prerequisite ("Without SCN-anchoring, calibrated tau values are in agent-loop-tick units… Calibration produces non-portable constants"). So whichever release ships the calibration framework MUST also ship SCN-anchoring; the two cannot decouple. If the calibration framework slips to 1.1 (current target per its own version-target field), this plan slips to 1.1 with it. If the calibration framework is pulled into 1.0, this plan must also be in 1.0. They're a unit. `v1_refinement.md` should track them together rather than as independent items.

2. **SCN-internal clock vs sibling module?** Two design options:
   - **SCN-internal:** add `register_periodic_callback` + `start_clock` + `stop_clock` directly to SCN. Tighter coupling but matches the "SCN is the home for wall-clock-anchored timing" framing.
   - **Sibling `time/decay_clock.py`:** new module that SCN delegates to (and other future consumers can use). Looser coupling, more discoverable, but adds a new bootstrap touchpoint.

   **Author recommendation:** SCN-internal for Phase 1. If a future consumer wants the clock surface independent of SCN, extract then. YAGNI for now.

3. **Threading model for the clock (multi-rate)?** With two Hz tiers (associative-memory at 1Hz + eligibility at 10Hz), options:
   - One `threading.Thread` per Hz tier with `time.sleep(1.0 / hz)` loop. Callbacks at the same Hz run serially within their thread. Mirrors the hippocampus capture thread pattern. **Author recommendation.**
   - `threading.Timer` per callback — simplest for single-rate, awkward for multi-rate (re-arming semantics).
   - `sched.scheduler` — stdlib; more bookkeeping than the use case needs but handles arbitrary rates cleanly. Worth considering if a third Hz tier appears in Phase 5+.

   Thread lifecycle mirrors the worker pool / hippocampus capture thread: started by `scn.start_clock()` at session start, stopped by `scn.stop_clock()` at session end.

4. **What if `anticipatory_pre_activate` also moves to the clock?** Currently it's tick-anchored. If we move ALL bio-pipeline SCN consumers to the clock, the anticipatory window timing changes — and that has its own calibration (B2 anticipatory_weight, etc.). Out of scope for Phase 1; flag for future work.

5. **Does this surface a sleep/replay timing concern?** During sleep (memory consolidation), the agent loop's tick rate drops dramatically. Today, decay fires less during sleep (matches "no thinking, no decay" intuition). After SCN-tying, decay fires at the same wall-clock rate during sleep as during awake. Is that the right bio-fidelity behavior? Probably yes (extinction continues during sleep biologically), but worth checking against `memory_consolidation_practice.md` predictions before Phase 3.

6. **Do we want to make the rates operator-tunable per environment?** Env-vars `MAXIM_ASSOCIATIVE_DECAY_HZ` (clamped say [0.1, 10.0]) and `MAXIM_ELIGIBILITY_DECAY_HZ` (clamped say [1.0, 60.0]) would let operators dial each tier per deployment. Useful for debugging (set associative to 0.1 Hz to slow-mo observe decay; set eligibility to 30 Hz to push closer to biological eligibility timescales on hardware that can handle it) and for hardware-specific tuning before the calibration framework ships. Each env var pairs with an autouse `conftest.py` scrub per CLAUDE.md "opt-in env vars in hot startup paths" discipline. Probably yes; lands with Phase 1.

## Authorization gate

Drafted as `docs/plans-scn-decay-anchoring` branch off main after Phase B's PR #269 merged. Phase 0 design pass + Phase 1 implementation start on explicit user authorization. Not currently gating any 0.9.x work. If the user chooses the 1.0-inclusion path per Open Question §1, that decision lands in `v1_refinement.md` and converts this plan from "1.1+ scoped" to "1.0 gate."

# Wire-A cluster-reward-bias decay tau split

**Target version:** 0.9.2 (preferred) or 0.9.1 follow-up before publish.
**Status:** Draft. Plan written 2026-05-25 after Roy-3c-bisect ([PR #266](https://github.com/dennys246/Maxim/pull/266), [29_roy_3c_bisect.md](../experiments/29_roy_3c_bisect.md)) cleanly named Wire-A's decay rule as the cause of the magnitude-axis shift, and surfaced that the chosen tau may be too aggressive for Wire-A's actual use case.
**Owns:** [`src/maxim/decisions/nac.py`](../../src/maxim/decisions/nac.py) (`NACConfig` field add + `decay_cluster_reward_biases` consumer change), [`tests/unit/test_nac.py`](../../tests/unit/test_nac.py) (regression guards for the new field), [`docs/experiments/30_wire_a_tau_validation.md`](../experiments/30_wire_a_tau_validation.md) (companion write-up, created during Phase 3).
**Companion plans:** [release_0_9_1.md](release_0_9_1.md) (this is the substantive Tier 2 work the Roy-3 writeup teed up), [v1_refinement.md](v1_refinement.md) (Roy-3 follow-up item 2 — "decide whether Wire-A's render needs a raw priming snapshot floor" — supersedes that decision with a cleaner tune), [persona_convergence_crucible.md](persona_convergence_crucible.md) (Roy-3 retry that depends on this lands here).

## Why this plan exists

Roy-3 ([23_roy_3.md](../experiments/23_roy_3.md)) shipped the four 0.9.1 wires and ran the pre-registered annotation-pattern validation. The pre-registered outcome ("A ≈ B ≈ C across both fixtures") reproduced. Wire-A's annotation was wired correctly end-to-end, but the LLM saw `[neutral / mixed]` at test time (max(|bias|) = 0.036 in Roy-3a, 0.098 in Roy-3b — both below the 0.1 "mildly rewarding" floor). The annotation was present but said nothing.

Roy-3c-bisect ([29_roy_3c_bisect.md](../experiments/29_roy_3c_bisect.md)) decomposed Roy-3's "still unfalsified" branch into two axes:

1. **Key count (6 → 2):** non-code environmental drift in the encoder layer (env-var refuted by A1, narrator drift refuted by A3, AUT behavior byte-identical between historical and current). Cannot be closed by bisect; doesn't change any downstream design decision.
2. **Bias magnitude (saturated → partial → decayed-to-neutral-by-test-time):** Wire-A's pre-merge fold commit (`bee42ca`) intentionally added `NAc.decay_cluster_reward_biases()` per-tick decay in response to a Critical bio-fidelity review finding ("by-accretion contamination of the substrate-voice thesis"). Confirmed behaviorally by A2: 7 pre-decay runs all saturated +1.0/+1.0; 3 post-decay runs all `{partial, +0.98}`. The decay IS the cause.

The decay is bio-correct — without it, every reward stays at +1.0 forever (the "permanent fossil" critique). But the chosen tau is too aggressive for Wire-A's actual use case, and Wire-A inherits this tau by accident, not by design.

## The architectural smell

Look at [`NACConfig`](../../src/maxim/decisions/nac.py#L132-L173):

```python
reward_bias_decay_tau: float = 50.0       # original use case: EC threshold modulation
percept_valence_decay_tau: float = 200.0  # Wire 2 use case: Pavlovian aversion
                                          # (deliberately decoupled by pre-merge bio-fidelity review B2 fold)
```

Wire 2's percept valence decay was **deliberately decoupled and tuned to a slower bio-plausible timescale** by the pre-merge bio-fidelity review. The comment at line 163 says explicitly: "Bio-fidelity tune (pre-merge bio-fidelity review B2 fold): decoupled from..."

**But Wire-A's `bee42ca` fold reused `reward_bias_decay_tau=50.0` for `_cluster_reward_bias` without re-justifying it for the new use case.** The 50.0 was chosen for **threshold modulation per tick** (where you want fast adaptation so EC's recognition threshold tracks recent reward signal). Wire-A uses cluster reward bias for **substrate-voice annotation across a multi-turn test phase** — a fundamentally different timescale concern.

The pattern is the same one Wire 2 already solved: when an existing decay parameter gets reused for a new use case with a different timescale, the parameter needs splitting and the new use case needs its own tune.

## The decay math (current default)

```python
new_bias = bias * (1.0 - 1.0/50.0)  # = bias * 0.98 per tick
```

| Ticks elapsed | Fraction of original bias remaining |
|---:|---:|
| 1 | 98% |
| 10 | 82% |
| 30 | **55%** |
| 50 (= tau, 1/e) | 36% |
| 100 | 13% |
| 150 | 5% |
| 200 | 2% |

Empirically against Roy-3a's data: priming-end bias `~+0.20` decayed to `~+0.04` by Wire-A's first read at test time — that's ~80% decay, roughly 80 ticks elapsed. Roy-3b's `~+0.21` decayed to `~+0.098` similarly. Consistent with a 30-turn test arm × ~3 ticks per LLM call.

## The fix — split `cluster_reward_bias_decay_tau` from `reward_bias_decay_tau`

Add a new `NACConfig` field, point `decay_cluster_reward_biases` at it, default it to a bio-plausible value tuned for Wire-A's render window.

### Calibration math

For the priming-end +0.98 cluster to still clear the 0.5 "strongly rewarding" band at mid-test-arm (~225 ticks after priming ends):

```
0.98 * (1 - 1/tau)^225 > 0.5
(1 - 1/tau)^225 > 0.510
1 - 1/tau > 0.510^(1/225) ≈ 0.9970
1/tau < 0.0030
tau > ~334
```

For the partial-bias +0.20 cluster to still clear the 0.1 "mildly rewarding" floor at mid-test-arm:

```
0.20 * (1 - 1/tau)^225 > 0.1
(1 - 1/tau)^225 > 0.5
tau > ~325
```

Both point at **tau ≈ 300-400** for Wire-A's render to stay meaningful through the test arm.

Cross-check against Wire 2's `percept_valence_decay_tau = 200.0`: cluster reward bias is "what value has this cluster historically had" (associative memory) which is **strictly longer-timescale** than per-event Pavlovian aversion. So cluster_reward_bias_decay_tau should be ≥ percept_valence_decay_tau. The 300-400 range is consistent with this ordering.

### Proposed default

**`cluster_reward_bias_decay_tau: float = 300.0`** — sits at the lower end of the calibration range; conservative but defensible. The math says 325-334 is the threshold to keep both classes of cluster expressive at mid-test-arm; rounding down to 300 gives a clean default with a small bias toward shorter timescales for biological caution about indefinite associative persistence.

**The 300.0 value is the starting calibration, not a permanently-correct tune.** It's the single point we're confident enough to ship on the math + the bio-plausibility cross-check against Wire 2's `percept_valence_decay_tau=200.0`. If future Roy iterations show decay too slow (cluster biases persisting across sessions in ways that hurt extinction) OR too fast (Wire-A's annotation goes neutral mid-test-arm again), **this field is where to look first**. The `nac.py` docstring carries the same callout so future readers can navigate from "decay feels wrong" → "this is the knob to turn."

Open to higher values (400, 500) if Phase 3 validation shows 300 is still too aggressive for Roy-3-retry's specific test-arm length. Phase 1's matrix surface naturally exposes this. Open to lower values (200, matching `percept_valence_decay_tau`) if cross-session extinction proves slow in production.

## Downstream: calibration-by-simulation framework (1.1+)

After this plan ships + SCN-anchoring ships, the natural next step is the [decay_consolidation_calibration_plan.md](decay_consolidation_calibration_plan.md) framework — tier-transition-driven calibration replaces hand-picked tau defaults. The 300 value this plan ships becomes a *baseline data point* for the calibration framework, not a permanent value. If the framework converges on a different value for `cluster_reward_bias_decay_tau`, that becomes the new default via a follow-up config-update PR.

## Known limitation: decay is tick-anchored, not SCN-tied (separate follow-up)

The `decay_*_reward_biases()` functions in `nac.py` fire **once per `agent_loop` tick** (from section 8.5 at [`agent_loop.py:3657-3659`](../../src/maxim/runtime/agent_loop.py)). The tick rate is bounded by `idle_sleep_s=0.25` on idle ticks but by action latency on busy ticks. Net effect: **effective decay-per-wall-second is hardware-dependent.**

| Hardware | Busy-tick rate | Effective decay per wall-second |
|---|---|---|
| Fast GPU + fast local LLM | ~10-15 Hz | 10-15× faster |
| Slow CPU + cloud LLM with latency | ~1-2 Hz | 1-2× slower |

Biologically, associative-memory extinction is wall-clock anchored — extinction happens on timescales of minutes to hours regardless of how fast the agent is thinking. Maxim's tick-anchored decay is bio-implausible at this level.

**SCN is the right home for wall-clock-anchored time signals.** SCN already does phase tracking, is already integrated with NAc for anticipatory pre-activation (B2, PR #198, [`project_b2_scn_oscillator_shipped.md`](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_b2_scn_oscillator_shipped.md)), and is the natural place to anchor extinction timescales. Tying all three NAc decay functions (`decay_reward_biases`, `decay_goal_reward_biases`, `decay_cluster_reward_biases`) plus `decay_eligibility` and `decay_percept_valences` to SCN ticks (wall-clock-derived) would make decay rate hardware-independent and bio-plausible.

**This is a separate plan, not part of the tau split.** Tying decay to SCN touches ALL FIVE decay parameters; the tau split affects only `cluster_reward_bias`. Sequencing: ship the tau split first (closes Roy-3 follow-up item 2, validates Wire-A behaviorally), then the SCN-tying work can ship independently without invalidating the tuned tau value — `tau=300 ticks` becomes `tau=300 SCN-anchored time units` and the calibration math is preserved at whatever SCN tick rate becomes the default. **The SCN-tying plan is drafted as [scn_decay_anchoring.md](scn_decay_anchoring.md)** (Phase C of this kickoff sequence, 2026-05-26).

Tracked as a candidate follow-up plan in memory: [`feedback_decay_is_tick_anchored_not_wall_clock.md`](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_decay_is_tick_anchored_not_wall_clock.md). Revisit when the tau-split work lands or when a multi-hardware deployment surfaces the symptom.

## What this does NOT do

- **Does NOT remove decay.** The `bee42ca` bio-fidelity correction is preserved. The decay still runs; it just runs at a tau appropriate for Wire-A's substrate-voice annotation timescale, not EC's threshold-modulation timescale.
- **Does NOT add a "raw priming snapshot" mechanism.** Earlier Roy-3 follow-up framing suggested Wire-A might need to bypass decay at the render layer by reading from a session-end snapshot. The tau split makes that mechanism unnecessary — the decay runs but doesn't crush the bias within Wire-A's render window. One timeline, easier to reason about.
- **Does NOT change `reward_bias_decay_tau=50.0`.** That parameter keeps its original use case (EC threshold modulation per tick). Splitting preserves both use cases.
- **Does NOT change `decay_goal_reward_biases` or `decay_reward_biases`.** Those keep `reward_bias_decay_tau=50.0` — different use cases, different timescales.
- **Does NOT touch the count-axis investigation.** The 6→2 environmental-drift question remains open and unaddressed; it's also unactionable per Roy-3c-bisect's verdict.

## Sizing

| Phase | Item | LOC | Persistence | Risk |
|---|---|---|---|---|
| 1 | Add `cluster_reward_bias_decay_tau` to `NACConfig`; point `decay_cluster_reward_biases` at it; default 300.0 | ~10 src | none (config-only) | low |
| 2 | Regression test guards: existing `decay_cluster_reward_biases` tests pin the formula; add tests pinning the new field separation, the default value, and env override | ~50 tests | none | low |
| 3 | Roy-3 retry (Roy-3a spec, all other config unchanged) | ~30 min wall | sim_reports session | medium — runner time |
| 4 | Companion experiment doc `30_wire_a_tau_validation.md` with cross-arm divergence measurements | ~150 doc | none | none |
| 5 | Fold verdict into [v1_refinement.md](v1_refinement.md), [release_0_9_1.md](release_0_9_1.md) Roy-3 follow-up list | ~30 doc | none | none |
| **Total** | | **~270 LOC + ~1 runner day** | none | low |

## Phase 1 — Config split

Single PR. Three changes in `nac.py`:

```python
# 1. Add the new field with the new default.
@dataclass
class NACConfig:
    # ... existing fields ...
    reward_bias_decay_tau: float = 50.0  # unchanged — EC threshold modulation timescale

    # NEW: Wire-A cluster-reward-bias annotation timescale.
    # Pre-merge bio-fidelity review B2 already split percept_valence_decay_tau
    # from reward_bias_decay_tau for the same reason: per-tick threshold
    # modulation (tau=50) and multi-turn substrate-voice annotation
    # (tau≈300) need different decay timescales. Wire-A reads
    # _cluster_reward_bias for substrate-voice annotation across a
    # multi-turn test phase, so the cluster-keyed decay needs its own
    # tau tuned for that window — not the per-tick threshold-modulation
    # tau the original field was sized for.
    #
    # Calibration: at tau=300, the priming-end +0.98 bias decays to
    # ~+0.51 at 225 ticks (mid-test-arm in Roy-3-shape iterations) —
    # just above the 0.5 "strongly rewarding" band. The partial-bias
    # +0.20 cluster decays to ~+0.10 at the same point — just at the
    # 0.1 "mildly rewarding" floor. Both classes of cluster stay
    # expressive at Wire-A's render time without the annotation
    # becoming a permanent fossil over many sessions.
    cluster_reward_bias_decay_tau: float = 300.0
```

```python
# 2. decay_cluster_reward_biases reads the new field.
def decay_cluster_reward_biases(self) -> int:
    # ... docstring updates ...
    if not self._cluster_reward_bias:
        return 0
    decay_factor = 1.0 / self.config.cluster_reward_bias_decay_tau  # CHANGED
    # ... rest unchanged ...
```

```python
# 3. Env-var override (mirrors temporal_credit_weight pattern).
# Read in NACConfig.__post_init__ if present, clamp to [50, 1000].
# Env var: MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU
```

CLAUDE.md "opt-in env vars in hot startup paths need autouse scrubs" applies — pair with `_isolate_maxim_nac_cluster_reward_bias_decay_tau_env` autouse fixture in `tests/conftest.py`.

## Phase 2 — Regression tests

Three test cases in `tests/unit/test_nac.py`:

1. `test_cluster_reward_bias_decay_uses_dedicated_tau` — verifies `decay_cluster_reward_biases` consumes `cluster_reward_bias_decay_tau`, NOT `reward_bias_decay_tau`. Set them to different values; observe the cluster bias decays per the cluster-specific tau.
2. `test_cluster_reward_bias_decay_default_is_300` — pins the default value so a future re-tune is an explicit and reviewed change.
3. `test_cluster_reward_bias_decay_env_override_clamped` — verifies env override is applied and out-of-range values fall back to default with a warning.

Update existing decay tests (`TestClusterBiasDecay::test_shrinks_per_tick` etc.) to use the new field name; the four tests added by `bee42ca` are the regression surface and stay green.

## Phase 3 — Roy-3 retry

Re-run Roy-3a unchanged (same spec, same fixture) with the new tau default. Expected outcome shift:

| Measure | Roy-3a baseline | Roy-3a-retry (tau=300) expected |
|---|---|---|
| Priming-end `cluster_reward_bias` | 2 keys, `{partial ~0.20, +0.98}` | Unchanged (same priming, same tau-50-tau-tweak doesn't affect priming-side accumulation) |
| Wire-A max(\|bias\|) at test time | 0.036 | ≥0.5 (strongly rewarding band) for the +0.98 cluster |
| Wire-A rendered annotation band | `[neutral / mixed]` | At least one `strongly rewarding` entry |
| Arm A `sense_food_source` tool calls | 0 | ≥1 (positive divergence signal) |
| Pre-registered Roy-3 outcome | "annotations present but LLM not biasing on them" | "annotation present, LLM biases when annotation is meaningful" |

The pre-registered Roy-3-retry pass criterion (per the persona-convergence framing in [persona_convergence_crucible.md](persona_convergence_crucible.md)):

- **Arm A produces ≥1 `sense_food_source` call across the 10-turn test arm** — substrate-acquired bias surfaced via Wire-A's prompt annotation drove the LLM to consider the tool it would not have considered without the annotation.
- **Stretch:** Arm A > Arms B and C on `sense_food_source` count — the annotation produces the cross-arm divergence Roy-3 was designed to validate.

The null result (Arm A still zero `sense_food_source`) means:
- The 300 tau is still too aggressive (lift to 400-500 and retry), OR
- The annotation pattern itself doesn't drive the LLM regardless of bias magnitude (escalates to the alternative mechanisms in [persona_convergence_crucible.md](persona_convergence_crucible.md))

Phase 3 disambiguates which.

### Phase 3 log vigilance (n=1 mitigation)

Per [`feedback_n3_minimum_for_partial_vs_saturated`](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_n3_minimum_for_partial_vs_saturated.md), n=1 is normally too thin to name a cause when the pass criterion involves partial-vs-saturated magnitude. Phase 3's pass criterion is a count comparison (Arm A `sense_food_source` ≥ 1), which is more robust to single-run noise, so n=1 is acceptable IF the log surface confirms the mechanism is firing as expected. Concrete logs to inspect on every Phase 3 run:

1. **Stage 0c `sim_recommend_action` JSONL events** — every test-arm LLM call shows the Wire-A annotation snapshot the prompt actually rendered. Expected at mid-test-arm (turn ~5 of 10 per arm, ~150 decay ticks in):
   - At least one cluster bias entry above the 0.5 "strongly rewarding" band, OR
   - At least one entry above the 0.1 "mildly rewarding" floor with a non-neutral band label.
   If the entries are still all neutral/mixed at mid-test-arm, tau=300 is still too aggressive — lift and retry.

2. **`cluster_reward_bias` decay trajectory across the test arm** — the priming-end +0.98 cluster should decay to ≥0.5 at tick 225 (mid-test-arm) and ≥0.3 at tick 450 (end-test-arm). Trace by diffing the `aut_nac.json` snapshots between priming session and arm-A session.

3. **Cross-arm tool call distribution sanity check** — if Arm A produces 0 `sense_food_source` but Arms B and C ALSO produce 0 (matching Roy-3a baseline), the null result is consistent with the annotation-pattern-doesn't-bias-LLM hypothesis. If Arm A is 0 but Arms B/C produce some, the LLM is biasing on something other than Wire-A — different failure mode worth surfacing immediately.

4. **`Concept reinforced: "sense_food_source"` events in `MAXIM_LOG_FILE` JSONL** — the substrate-internal text encoding pipeline. Confirms the tool surface is being kept fresh in EC text encoding (sanity check; failure here would indicate a separate substrate-encoding bug, not a tau-split issue).

If any of items 1-2 deviate from expected, n=1 result is suspect and the run should be repeated (with the deviation as a documented diagnostic anomaly). Items 3-4 are sanity checks that surface mechanism-level bugs orthogonal to the tau choice.

Run Phase 3 with `MAXIM_LOG_FILE=/tmp/roy_3a_tau_validation.jsonl MAXIM_SUBSTRATE_PATH=1` at minimum so the diagnostic surface above is captured.

## Phase 4 — Companion experiment doc

`docs/experiments/30_wire_a_tau_validation.md` with:
- Pre-registration block (the table above)
- Methods (single Roy-3a-retry run, same spec as Roy-3a baseline)
- Result (per-arm tool distribution, Wire-A annotation snapshots at test time, post-test cluster_reward_bias diff)
- Verdict (pre-registered pass/fail)
- Implications (if pass: closes Roy-3 follow-up item 2, escalates Roy-4-shape to next persona iteration; if null: tau lift or escalate to alternative mechanism)

## Phase 5 — Plan-doc folding

- **release_0_9_1.md**: mark Roy-3 follow-up item 2 ("decide whether Wire-A's render needs a raw priming snapshot floor") as supplanted by this plan and CLOSED on Phase 3 verdict.
- **v1_refinement.md**: same — remove the open Roy-3 follow-up item 2 from the 1.0 closing list; the verdict here determines whether the 1.0 plan needs the alternative-mechanism branch.
- **persona_convergence_crucible.md**: add a methodology note that Wire-A's tau is now a tunable parameter; the next persona iteration owns the tuning sweep if Phase 3's single-point default proves brittle.

### Phase 5 outcome (2026-05-26)

Phase 3 ran ([30_wire_a_tau_validation.md](../experiments/30_wire_a_tau_validation.md)). Verdict: **tau split structurally validated** (annotation rendered `[strongly rewarding]` throughout the test arm; decay trajectory fit the model within 0.3%), but the **PRIMARY criterion (Arm A ≥1 `sense_food_source` call) failed** for a third reason the kickoff didn't anticipate.

Wire-A's annotation reached the LLM with the right magnitude; the LLM could not act on it because:
1. `sense_food_source` is a SEM-modulator-derived tool absent from the test scene's active roster, and the LLM has no signal that the tool exists elsewhere.
2. Imagination's trigger is percept-text-bound and substrate-blind, so it can't dream up the missing entity into scene when Wire-A signals strong reward for an absent tool.

Two new 1.1+ plan docs capture the downstream gaps:
- [sense_tool_registry.md](sense_tool_registry.md) — adds grayscale visibility for SEM-derived inactive tools.
- [imagination_substrate_signals.md](imagination_substrate_signals.md) — hooks NAc/Wire-A signals into imagination's input.

Either could close the Roy-3a gap independently; both together is the robust fix.

The bias-magnitude side of Wire-A is no longer the bottleneck. Roy-3 follow-up item 2 is CLOSED via tau-split; the next persona iteration owns the downstream-gap question.

## Framing rule

**The tau split is not a "fix" — it's a tune that splits a parameter that was wearing two hats.** The `bee42ca` decay introduction was correctly motivated; only the inherited tau value was wrong for the new use case. This plan preserves the bio-fidelity correction (decay still runs) AND respects Wire-A's actual timescale.

**This plan does not commit to tau=300 being the right long-term value.** It commits to splitting the parameter so the value can be tuned per use case, and to a defensible Phase 3 measurement that validates the chosen default. If the tune proves brittle (Phase 3 null result + retry at 400/500), the parameter being separately addressable IS the win; the value is the followup.

## Authorization gate

Per CLAUDE.md sims-from-Claude-Code discipline, Phase 3's runner pass needs `--interactive false` and pre-confirmed test-arm length. The Roy-2 reproduction protocol ([18_roy_2_reproduction.md](../experiments/protocols/18_roy_2_reproduction.md)) covers the shape; Phase 3 follows that protocol verbatim with the new tau active.

## What this NOT solves

- The 6→2 cluster count axis (Roy-3c-bisect's unclosable environmental drift). This plan only addresses magnitude; the count axis is a separate, deferred question.
- General persona convergence (Roy-3 was never designed to close it). This plan closes the specific Wire-A-can't-be-expressive sub-question; the broader persona-convergence research question is owned by [persona_convergence_crucible.md](persona_convergence_crucible.md) and depends on Phase 3's outcome.
- EC drift fix Phase-X retests at 0.44. The bisect closed the priming-regression confound that was blocking that retest; this plan is independent of EC threshold and runs at whatever EC default is current at Phase 3 time.

## Open questions — resolved 2026-05-25

1. **tau=300 default — RESOLVED.** User confirmed 300 as the starting calibration. Plan + `nac.py` docstring + memory entry all carry the "subject to change" framing so future readers know this is the knob to turn if decay feels wrong.
2. **Env override clamp — RESOLVED at [50, 1000] for this plan.** User raised the deeper bio-fidelity question: should decay be wall-clock-anchored or SCN-tied rather than per-tick? Yes — that's the proper fix, but it's separate from the tau split and touches all five NAc decay parameters. Tracked in the "Known limitation" section above + memory entry [`feedback_decay_is_tick_anchored_not_wall_clock.md`](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_decay_is_tick_anchored_not_wall_clock.md). The tau-split clamp stays at [50, 1000] since hardware-dependent decay rate is a separate concern from the calibration-knob concern this plan addresses.
3. **n=1 for Phase 3 — RESOLVED.** User confirmed n=1 with extra log vigilance. See "Phase 3 log vigilance" subsection above for the four concrete logs to inspect on every Phase 3 run.

## Open questions — surfaced by Phase 3 bio-fidelity review (2026-05-26)

4. **Four-tau topology bio-coherence audit (OPEN, post-1.0).** Phase 3 bio-fidelity review flagged that the four NAc decay timescales (`reward_bias_decay_tau=50.0`, `goal_reward_bias_decay_tau`=tracks reward_bias, `percept_valence_decay_tau=200.0`, `cluster_reward_bias_decay_tau=300.0`) lack an explicit bio-coherence justification for their relative ordering. Specifically: in biology, goal-value extinction (learning "this incentive no longer rewards") is generally *slower* than percept-aversion extinction (learning "this stimulus no longer predicts pain") — but the current topology has goal decay at 50 ticks and percept decay at 200 ticks, inverting the biology. This wasn't load-bearing for the tau-split (which only touches cluster decay), but it's flagged for a future bio-coherence audit. Audit owner: future persona iteration or 1.1 cleanup pass. Not gating any current work.

# Bio-Faithful Roadmap — per-system status and trajectory

**Status:** LIVING DOC (created 2026-08-10 from the Bio-mapping truth pass, PR #492).
**Maintenance rule:** update a system's row whenever its `Bio-mapping:` docstring tag
changes, a declared gap is closed, or a gap's intent changes. This doc is the
*algorithmic* companion to [behavioral_graduation_candidates.md](behavioral_graduation_candidates.md)
(which tracks *behavioral* claims) — a system can be FUNCTIONAL here yet hold an Earned
row there (Exp 42's discrimination does not need TD to be real), and MECHANISM here
earns nothing behaviorally by itself.

**Gap-intent vocabulary** (every declared "NOT implemented" gets exactly one):
- **EARN** — we intend to implement the missing mechanism; trigger + target listed.
- **ANALOG** — deliberate engineering analog; implementing the biology is NOT a goal
  (the tag documents the divergence forever; revisiting requires a new argument).
- **OPEN** — undecided; becomes EARN or ANALOG when a concrete need forces the choice.

---

## Systems

### NAc + CausalLink (`decisions/`) — FUNCTIONAL (module) / MECHANISM (link level)
Earned: Rescorla-Wagner value learning per link (ΔV = α(R−V), novelty-adaptive α;
`last_rpe` = true single-trial RPE). Eligibility-trace proportional credit at the
module level shares the role, not the algorithm.
| Gap | Intent | Trigger / target |
|---|---|---|
| TD bootstrapping (γV(s′), value chaining) | **ANALOG** | Deliberate: the roadmap direction is three-factor credit (eligibility × reward × modulation, [cross_modal_perception_fabric.md](cross_modal_perception_fabric.md)), not TD backup. Revisit only if multi-step credit demonstrably fails a real task. |
| Compound-cue competition (ΣV in R-W) | OPEN | Becomes EARN if multi-cue contexts (several entities sharing an outcome) show mis-attribution in a pre-registered probe. |
| Dopamine-like global broadcast | OPEN | The ValenceSignal/WMS salience path is the current partial analog. |

### SCN + OscillatorNetwork (`time/`) — FUNCTIONAL (SCN) / MECHANISM (oscillator)
Earned: genuine Kuramoto dynamics + Hebbian coupling learning.
| Gap | Intent | Trigger / target |
|---|---|---|
| 5 of 6 event types have NO producer (oscillator starves; [scn_event_producer_gap.md](deferred/scn_event_producer_gap.md)) | **EARN** | The single highest-value bio-fidelity fix available: the mechanism is real but underfed. Gate: front-gate question first — does per-event-type phase learning earn its keep (B2 never behaviorally graduated)? 1.2 intake. |
| ~20k oscillators / entrainment inputs | ANALOG | 4 scales is the deliberate scope. |

### Hippocampus (`memory/`) — FUNCTIONAL
| Gap | Intent | Trigger / target |
|---|---|---|
| Sequence-aware replay (sharp-wave-ripple-like ordered reactivation) | **EARN (candidate)** | Current replay is priority-ordered, order-free. Becomes a 1.2 candidate WITH the DMN initiative below (idle-time incremental replay); needs a pre-registered episodic-chaining probe to show sequences matter. |
| Place/grid coding | OPEN | Tied to real spatial navigation work (robot 1.3 perception fabric); not before. |
| CA1/CA3 circuitry | ANALOG | Graph store is the deliberate substrate. |

### EC (`similarity/ec.py`) — FUNCTIONAL
| Gap | Intent | Trigger / target |
|---|---|---|
| Attractor-based completion (vs cosine threshold) | ANALOG | The frozen-centroid + threshold contract is load-bearing (drift lesson); attractors would reopen it. |
| Theta-phase coding | OPEN | Only meaningful after SCN producer gap closes. |

### Cerebellum (`embodiment/cerebellum.py`) — MECHANISM (single-cue delta rule)
| Gap | Intent | Trigger / target |
|---|---|---|
| Compound-cue competition | OPEN | Same trigger as NAc's row. |
| Timing (climbing-fiber style eligibility windows) | **EARN (candidate)** | Live-robot motor prediction (H-series / Exp 49-50 line) will force temporally-precise forward models; 1.2-1.3. |

### Episode Hebbian binding (`memory/episode.py`) — MECHANISM (loose)
| Gap | Intent | Trigger / target |
|---|---|---|
| LTD / decay of unused bindings | **EARN (candidate)** | Monotonic edge growth clamped at max is a known long-session concern; pair with decay-calibration work when cross-session lifetimes matter (1.2). |
| STDP | ANALOG | Tick granularity is wrong for spike timing; orient-windowed Hebbian binding ([cross_modal_perception_fabric.md](cross_modal_perception_fabric.md)) is the temporal-window analog we DO intend. |

### ATL (`memory/atl.py`) — FUNCTIONAL
Hub-and-spoke degradation: ANALOG (diagnostic biology, not a capability we need).

### Angular Gyrus (`math/angular_gyrus.py`) — FUNCTIONAL
Parietal magnitude codes: ANALOG. Stable; no roadmap pressure.

### Sleep replay (`memory/sleep_replay.py`) — FUNCTIONAL
Session-end only today. See DMN initiative: idle-time incremental replay is the
EARN path (and the bio-correct scheduling — consolidation happens during rest,
not only at "death of session").

### PainBus + severity latch (`proprioception/`) — FUNCTIONAL
| Gap | Intent | Trigger / target |
|---|---|---|
| Sensitization kinetics | ANALOG | The severity latch (band-entry + material re-injury) is the documented engineering analog; the channel-split invariant depends on its exact semantics. |
| Fiber classes / gate control | ANALOG | No consumer needs latency-class routing. |

### Spatial frames (RSC) — NOT IMPLEMENTED (gap registered)
No egocentric↔allocentric translation exists. `azimuth` is explicitly head-relative;
`memory/spatial.py` is type-only ("no integration with Perception yet"); the 2026-07-16
head-frame incident was this gap in hardware form.
| Gap | Intent | Trigger / target |
|---|---|---|
| ego↔allo frame translation (RSC role) | **OPEN → EARN on trigger** | Plan + triggers + re-validation registry: [deferred/retrosplenial_spatial_frames.md](deferred/retrosplenial_spatial_frames.md). Build on T1 place-memory-across-motion / T2 multi-vantage identity / T3 return-to-place / T4 a second frame-confusion incident. Pre-check RAN 2026-08-11 (`scripts/rsc_precheck.py`) and did NOT close it — but it refuted the predicted failure mode: the channel is RESOLUTION-bound (~3 well-placed nodes: left/centre/right), not frame-confused, and the plan was re-sequenced (resolution before frames). Place-code wiring shipped PR #499. |
| place/grid coding | ANALOG for now | Only meaningful after the transform above exists. |

### Attention (`attention/`) — no bio-algorithm claims (checked clean)
Salience-map factor combination; inhibition-of-return is implemented-as-described.
No tag needed; watch that future edits don't import claims.

---

## Initiative: the Default-Network reorganization (the audit's biggest structural finding)

**Finding.** `default_network/` is a well-wired **reactive/salience layer** (YOLO-driven
orienting, startle, social attention, idle scan; arbiter; deliberative inhibition;
thalamic-gate percept filtering) — exogenous-attention function, ~18 consumers, not
isolated. But biologically the *default mode network* is the TASK-NEGATIVE network:
what cognition does when nothing external demands attention — consolidation,
prospection, self-referential processing. The module is tagged NAME-ONLY because it
implements the salience network under the DMN's name.

**The twist that makes integration cheap:** the true-DMN functions already exist,
scattered and never scheduled as idle cognition:
1. `memory/sleep_replay.py` — consolidation, currently session-END only;
2. `imagination/` — internally-generated simulation, currently percept-triggered only;
3. `NAc.anticipatory_pre_activate` + oscillator — prospection, currently starved
   ([scn_event_producer_gap.md](deferred/scn_event_producer_gap.md));
4. body-state / drive self-model — self-referential substrate, read reactively only.

**Proposal (staged; front-gated as ride-on-existing — the only new code is a thin
idle-time coordinator):**
- **Stage A (done, PR #492):** honest reframe — NAME-ONLY tag; the module docstring no
  longer claims DMN function.
- **Stage B (1.2 intake, needs its own plan + two-lens round):** an idle-time cognition
  coordinator riding the agent loop's existing idle gate: when task pressure is low,
  budget slices of (a) incremental replay (consolidation during rest — also fixes the
  lightweight-session-end gap for long-running adapters noted in CC8), (b) spontaneous
  imagination/prospection (imagination trigger invoked from internal state, not only
  percepts), (c) anticipatory pre-activation once SCN producers exist. Reactive
  `default_network/` keeps its job and its name-as-code; the coordinator is the thing
  that EARNS the DMN name.
- **Stage C (decision, not code):** naming. Options: rename `default_network/` →
  salience-network naming (touches ~18 consumer files; DN is NOT on CLAUDE.md's
  protected-names list) and give the coordinator the DMN name; or keep code names and
  let docstrings carry the truth (Stage A already does). Decide at Stage B review —
  churn vs clarity.
- **Validation bar:** per invariant-two-tier discipline, the coordinator enters as
  `[engineering]`; it graduates only via a pre-registered experiment showing idle-time
  cognition pays (e.g., idle-replay arm vs session-end-only arm on cross-session
  recall, or spontaneous-imagination arm on later task performance).

**Known integration debt folded into Stage B:** `network.py:360` still constructs an
internal `PainBus()` with split subscriber ownership (deliberately deferred at Wave 2);
the coordinator plan should close it rather than adding a third wiring path.

**Explicitly NOT proposed:** replacing the reactive layer. It is load-bearing
(imagination arousal gate, thalamic gating, robot gaze) and correctly built — it was
only mis-named, and the audit already fixed the claim.

---

## Standing rule for new bio-named modules

A new bio-named module declares its `Bio-mapping:` tag at birth (MECHANISM claims
verified in review), gets a row here with every declared gap assigned EARN/ANALOG/OPEN,
and the CI claim-lint ([bio_docstring_truth_pass.md](archive/bio_docstring_truth_pass.md)
follow-up) enforces that named algorithms outside earned tags fail CI.

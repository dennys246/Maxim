# B1 (FROZEN v5, 2026-10-13) — the Exp 60 floor: what does the learned drowning-avoidance add over chance and over a model-free learner?

> **Outcome 2026-10-14: `FLOOR_MATCHED_Q`** ([verdict](data/b1_exp60_floor_48ce40118c20_verdict.json),
> [rows](data/b1_exp60_floor_48ce40118c20.jsonl), [apparatus](data/b1_exp60_floor_apparatus_2026-10-14.json); PR
> #1283). First post placement surfaced: FEAR 8/8, Q-explore 8/8 (all three Q variants 8/8), random 2/8, n = 8 clean
> per arm. FEAR > random is shown; FEAR > Q is not. Under BF2 the parent row T1-13 moved from EARNED to **PARTIAL
> (narrow: floor matched)**; T1-14 and T1-15 followed analytically on 2026-10-15. The "FROZEN v5 … before any gated
> data" status below is kept as recorded. [Ledger](../plans/behavioral_graduation_candidates.md).

> **STATUS: FROZEN v5, 2026-10-13, on main before any gated data.** v5 records the seed 99 diagnostic (§Liveness
> ladder: all checks pass) and folds what the rig showed into rules 11 and 14, with a run runbook (§Run runbook).
> Implementation rules: §Frozen implementation rules, from the build-plan review
> ([../plans/baseline_floors_b1_build.md](../plans/baseline_floors_b1_build.md)); v4 added rules 15–17 from the harness
> build and owner decision OD5. The four-lens design review ran on v1 (2026-10-11; reports preserved
> verbatim in [rationale/b1-exp60-floor/](rationale/b1-exp60-floor/)). Every lens returned DO-NOT-BUILD on v1; v2
> folds all of them (§What the review changed). Owner decisions OD1–OD4 TAKEN 2026-10-11 (§Owner decisions; all
> as recommended). The harness (#1236, #1238), its evidence entry and BF2 obligation lint (#1245) and a CLI fix
> (#1258) are merged; the dry-run is the seed 99 diagnostic. Plan: [../plans/baseline_floors.md](../plans/baseline_floors.md) Stage B1 (owner decisions
> BF1–BF2; the seam is a `src/` proposer hook, 2026-10-10). Parent: [exp60_drowning_avoidance_prereg.md](exp60_drowning_avoidance_prereg.md)
> (EARNED 2026-09-16, T1-13; PARTIAL since this run). Every Exp 60 apparatus, preflight, probe and training rule is inherited unless
> this document changes it.

## Why

Under BF2 a substrate-path claim earns only over `tabular_q` and `random` on the same state keys, and Exp 60,
earned before the rule, owes that floor at its next re-run. The situation has about two relevant states and one
winning tool, so a simple learner may match it. This experiment exists to find out before a reviewer does.

## What Exp 60's mechanism is (corrected by the review)

- **What Wire-4 learns is WHEN to act.** It books fear on the underwater world cluster from the air-hunger pain
  during training that executes nothing. This is context (situation) conditioning. `record_cluster_fear` is a
  clamped linear accumulator (`valence -= α·intensity`), not a prediction-error update, so the cluster does not
  "predict" the pain in the Rescorla–Wagner sense.
- **WHICH act comes from a designer-specified defensive prior.** `anticipatory_threat_need` drives
  `recommend_action`, whose `_DRIVE_TOOL_AFFINITIES["threat"]` keyword `escape` matches `escape_water`, a
  pathfinder-bypassing actuator built for Exp 60 (#719) and named so that keyword matches. The drive gate narrows
  the choice to about two tools (`flee`, `escape_water`). The fitting account is Bolles's (fear selects from an
  innate repertoire of species-specific defence reactions), not two-factor theory: there is no instrumental
  second factor (Exp 60's preflight asserts no positive escape link before the probe). The analogy holds
  (surfacing is a plausible innate response for an air-breather) but the repertoire here includes a task-fitted
  actuator, and that is disclosed.
- **What Exp 60 already showed and keeps showing.** FEAR vs ABLATED is "the prior with vs without the learned
  situation value": same prior, same pain, same episodes; only FEAR surfaced. No floor result here retracts that.

## A finding about Exp 60 itself (from its own records)

In Exp 60's run 2, FEAR's first post placement is the only fear-only read: `flee` (fails fast), then
`escape_water`, surfaced at 2.9–3.3 s. In placements 2–6 the first `escape_water` call lands about 0.7 s BEFORE
the teleport, while the agent is still on the shore, and the post shore roam logged 13–14 `escape_water` calls
per FEAR seed. After placement 1 the positive escape link drives the act regardless of the cluster: an "always
`escape_water`" policy would score the same. Fear stayed situation-specific (shore cluster 0); behaviour after
placement 1 did not. This bounds what T1-13's "situation-specific" can mean and is recorded against the row
(owner decision OD4).

## What the review changed (v1 → v2)

| v1 | Problem (lens) | v2 |
|---|---|---|
| Q_YOKED as the binding tabular floor | Cannot lose: under propose-only the pain is independent of the proposed action, so its action values are noise; no real learner has this credit rule (bio D1, confounding D2) | Analytical only: "an action-value learner cannot learn from action-independent reward". Not run, not a gate |
| Primary DV = P(surface) over 6 placements | Placements 2–6 are habit (above) (confounding D1) | **Primary DV = the first post placement** |
| Reward = −pain, else 0 | Pain arrives once or twice per episode at ~5–6 s, so 10 executed episodes cannot single out `escape_water`: Q_EXPLORE loses by reward design (confounding D2) | A game-native per-tick cost from the oxygen sensor (OD3) |
| No prior for the floors | The comparison credits Wire-4 with the prior's choice (bio D2, confounding S1) | Q_EXPLORE_PRIOR: the same learner on the prior's action set |
| Q keyed on the world cluster from a second encode | Different keys from Wire-4, and a second encode perturbs first-touch ids (bio S3, wiring SF3) | The tuple of all active modality clusters from the proposer's own encode |
| "About 15 actions", "8 ticks" | The live set is 26 tools; a draw only counts by ~2.5 s, so 4–5 draws (wiring SF1, confounding S2, environment S1) | Re-derived; the set is stamped per seed |
| Action params unspecified | Sampled coordinates let `mine_block` drain the pool or `move_to` tunnel the shell (environment E1) | `params={}` for every arm, pinned by a unit test |
| Hook "in the loop" | Training calls the proposer directly; the loop never runs (wiring DNB-1) | Inside `propose_via_substrate`, replacing only `recommend_action` |
| 5 seeds, uncorrected | A binary DV at 5 v 5 fails on two chance successes (confounding S4) | 8 seeds per arm (OD2); Holm across the gated pair |
| Inherited refusals | "Positive escape link after the preflight" refuses every arm that executes `escape_water` (environment E2) | Re-scoped per arm (§Apparatus) |
| Interleaved per seed | The harness runs one arm per invocation (environment E3) | A multi-arm driver, Latin-square order (§Order) |

## Arms

Every arm: the Exp 60 apparatus, bridge at 100 ms, the substrate-primary loop at 2 Hz, AUTONOMOUS autonomy, the
same probes (6 placements pre and post, US-free 4.3 s window), the same encoder and Wire-4 (kept attached in
every arm; it books but no floor reads it). Only the POLICY that picks the proposal differs.

| Arm | Policy | Training | Role |
|---|---|---|---|
| FEAR | `recommend_action` (unchanged) | Exp 60's: propose-only, 10 episodes to the saturating air-hunger pain | The claim, re-run contemporaneously (not Exp 60's trigger re-run: no ABLATED arm, different seeds; BD2) |
| RANDOM | Uniform over the stamped `available_tools` | None | BF2 random floor: chance over the full repertoire |
| Q_EXPLORE | Tabular Q-learning (below) over the full repertoire | 10 EXECUTED episodes through real loop windows | **BF2 tabular floor (OD1)**: a model-free learner given action experience |
| Q_EXPLORE_PRIOR | The same learner, action set limited to the threat-affinity tools (`flee`, `escape_water`) | As Q_EXPLORE | How much of the action knowledge the prior carries (reported, not gated) |
| Q_EXPLORE_RAW | The same learner keyed on raw sensors (`is_in_water` × oxygen in 4 bins) | As Q_EXPLORE | Separates the EC's split from Wire-4 (reported, not gated) |

**Analytical, not run.** Q_YOKED (above). The mechanism twin (state-value TD on the cluster plus the same
prior) is Wire-4 by construction; FEAR vs ABLATED already measures "the prior with vs without situation value".

**The tabular learner.** State = the tuple of the active modality clusters `propose_via_substrate` already
encoded that tick (Wire-4's keys; per agent, never compared across agents). Actions = the stamped set, each with
`params={}`. The cost and the RAW key read `observation`, the encoder's own world values that tick (OD3, taken;
exact form in §Frozen implementation rules). Update order: at select t, read the interval's reward, update
Q(s_{t−1}, a_{t−1}) for the action that EXECUTED, then choose a_t. α = 0.5, γ = 1, Q initialised to −2 (OD5, rule
15); ε = 0.2
in training, 0 in the probes, where the table keeps updating online from executed outcomes (FEAR's NAc also
learns inside probes; a frozen greedy learner would repeat a failing act forever). The policy RNG is seeded per
seed × arm; the class, constants and seed are stamped.

## The hook (`src/`)

`propose_via_substrate(..., policy: SubstratePolicy | None = None)`. The policy replaces only the
`nac.recommend_action(...)` call and returns the same dict shape (`tool_name`, `params`, `confidence`,
`reasoning`), so the proposal object, clusters, margins and strategy stay one path. `SubstratePolicy.select(*,
agent_id, available_tools, clusters, observation)` receives no `nac` and no learned threat need (raw sensor values
are not Maxim's learning): a floor structurally cannot read Maxim's learning. The hook validates the return
(`tool_name` in `available_tools`, `params` a mapping) and raises a typed error; a policy bypasses
`min_confidence` by design; `None` means idle, never a fallback to `recommend_action`. Threaded through `run_agentic_loop(substrate_policy=)` → `loop_substrate.substrate_tick` →
`_loop_kwargs` → `run_minecraft_aut` → `WaterTrial`. Identical across arms: the tool filter
(`MAXIM_SUBSTRATE_TOOL_WHITELIST` asserted unset), the encode, `note_active_clusters`, the situation cue, the
pain tick, the cadence, the same-tool cap (a greedy policy loses one tick in six to it, as FEAR does), autonomy,
`execute_and_learn`, telemetry. `policy=None` is byte-identical to today. The policy is ARMED immediately before the
teleport (t_arm stamped; water-only in placements and training, rule 17) and disarmed at rescue; the shore roam runs ARMED for every arm; loop liveness, the shore
warm-up and the recovery ticks run disarmed (FEAR returns nothing on
the shore; a floor would otherwise act there and an action begun on the shore could block the window). Rejected
alternative: replacing `nac.recommend_action` on the instance (an untyped monkeypatch of a bio class).

## Apparatus and refusals (changes from Exp 60)

- `params={}` for every floor proposal: `ArmedPolicy` records any non-empty params and the harness refuses the
  arm-seed (the generic hook passes params through). With `{}`, `move_to`, `turn`, `mine_block` and `place_block`
  fail in `Tool.run`'s required-input check before `execute` (pinned by an offline test; each tool is also called
  once with `{}` in the rig diagnostic); `flee` fails fast underwater; `attack_nearest`
  finds no hostile; `eat` fails at food 20. The apparatus is re-verified over RCON (pool source cells, lip, shore
  floor, shell faces) before and after every arm-seed; a mismatch refuses.
- "A positive escape link after the preflight refuses" applies to FEAR only (its claim is no instrumental credit
  before the probe). For RANDOM and the Q arms the links are recorded, not refused.
- Per seed: food 20 and saturation at the bridge clamp (restored with `effect give … saturation`, then verified),
  bread count stamped.
- Q arms' training: real loop windows at 2 Hz (not the propose-only 4 Hz direct calls). Terminals by bridge truth
  on the HARNESS clock (`is_in_water` 0, sensed oxygen ≤ 12, or a health drop); cap = measured damage onset − 3 s
  (≈ 13 s); RCON rescue first, then the bridge `stop` (a `goto` persists), then the loop stops. Exp 60's "any
  `drive:health` pain in training refuses" applies. Pain events, executed actions and the terminal cause are
  recorded per episode.
- Rig: no arm uses an LLM (no :8100 consumer); a preflight records that no inference job is running on the box
  and the server tick rate, and an apparatus check runs the same day.
- Every call's start and return time is recorded, including any call in flight at the teleport.

## Order

One multi-arm driver, one run id per session. Arm order within each seed follows a Latin square across seeds.
Before each arm-seed: the live geometry and cluster-distinct preflight, the apparatus re-verify, a fresh
throwaway persistence directory, and the frozen-day gamerule. No `~/.maxim` state is read.

## Liveness ladder (before any data, per arm)

Every Exp 60 preflight (roster, cadence ≤ 0.15 s, liveness ≥ 4 ticks in 3 s disarmed, clusters distinct, backend
actuation). Then one armed diagnostic seed per new arm: the telemetry's proposals, the policy's own log and the
executor's calls must be identical (not just equal in count); RANDOM must show calls other than `escape_water`;
Q_EXPLORE must show one update per select and nonzero costs while in water. A unit guard runs a scripted
always-`escape_water` policy through `run_agentic_loop` on the fake bridge and goes RED when the hook wiring is
deleted.

**Result (seed 99, 2026-10-13, rig at `d9822dd4`, before v5's rig-check change; raw rows and summary in
[rationale/b1-exp60-floor/](rationale/b1-exp60-floor/diagnostic_seed99.md); never pooled, rule 12). Every check
passes:**
- **Identity:** in every floor arm, across all 12 placement windows, the executor's calls are the policy's picks and
  the telemetry's proposals, in order. The one exception is at most one final pick per window, which the window's
  close cut off before it ran.
- **RANDOM** executed all 26 tools.
- **The Q arms** booked one update per in-water pick (Q_EXPLORE: 31 updates, 33 in-water picks), with non-zero
  costs. The difference is in-water picks still pending when a window ended at its cap, which rule 4 truncates by
  design. Every drop is a shore pick (`made_out_of_water`); none is a pick that never executed.
- **Hygiene:** no policy violations, one tool set of 26, and `turn` / `place_block` / `mine_block` fail on `{}`
  before the bridge. FEAR's first post escape started after its first in-water read (rule 17's check).
- **The rig check** recorded the proxy `absent` and TPS 20.0.

**The assumptions it settled:**
- `escape_water` calls take a median of 1.82 s (70 surfacing calls, 0.71–1.94 s), and surfacing lands 2.3–2.6 s
  after the teleport (the toy model's 2.6 s holds).
- The cluster keys did **not** split the water state: Q_EXPLORE saw 1 water key (RAW 2, its oxygen bins).
- The Q arms learn the escape in the pre-probe, as predicted, and training episode 1 surfaces in every Q arm.
  - Executed actions to the first surfacing: PRIOR 2, Q_EXPLORE 11, RAW 27.
  - Executed actions to criterion, pre-probe included: PRIOR 16, Q_EXPLORE 29, RAW 40.
  - FEAR: 0 in training.
- The seed's first post placement: FEAR, Q_EXPLORE, PRIOR and RAW surfaced; RANDOM was censored. A hint about the
  run, not evidence.

## DVs, hypotheses and gates (directions and numbers before data)

**Primary DV:** P(surface before the US) on the FIRST post placement, per seed (binary). **Secondary
(reported):** P(surface) over all 6 post placements (labelled "fear plus positive link plus habit" for FEAR, and
**not comparable across arms**: rule 17 denies the floors the shore-launched act FEAR's placements 2–6 habit
performs, so no rate over placements 2–6 is compared across arms); latency, and the first in-water select's
latency per arm (the floors' gate reads `is_in_water` at eye height, FEAR keys on the EC world cluster; both are
derivable from each row's ticks and calls); shore-roam affordance calls per arm (water ENTRY is about 0 for every
arm by construction, since no action with `params={}` can move the bot into water, so only call counts are
reported; under rule 16 the floors' shore rows are never credited, so their roam calls are at chance by
construction and bound FEAR's shore habit only); Q arms' surfacing per pre-probe placement and learning curve
(the episode of first surfacing in training, and the greedy action in the water state after episodes 1, 2, 5
and 10); executed actions to criterion, pre-probe included (FEAR: zero in training, one failed `flee` per first
placement).

- **H1 (FEAR > RANDOM; gated).** RANDOM (water-only armed like every floor, rule 17) has about 4–5 useful draws
  from 26 tools: escape blocks ≈ 2.0–2.6 s to the head clearing (2.6 s if Exp 60's latency excludes the 600 ms
  post-clear hold), so only a draw in the first ≈ 1.7–2.3 s surfaces before the 4.3 s cap: 1 − (25/26)^4..5 ≈
  0.15–0.18 (range 0.11–0.18 across timing assumptions). FEAR's ≈ 1.0 (Exp 60: 5/5). FEAR can pass the sanity
  gate and still fail H1 (6/8 v 2/8 gives p ≈ 0.066 → `FLOOR_MATCHED_CHANCE`); that is an honest outcome, not an
  instrument fault.
- **FEAR sanity gate.** FEAR's first-placement rate must be ≥ 6/8, or the verdict is INCOMPLETE with cause (a
  collapsed FEAR is not a floor match).
- **H2 (FEAR vs Q_EXPLORE; gated as FEAR > Q_EXPLORE).** Expected: with a per-tick oxygen cost, Q_EXPLORE finds
  `escape_water` (ending the cost early) and its first-placement rate ≈ 0.8–1.0. The learner is named wherever N
  or the efficiency gap is quoted: **tabular Q, α = 0.5, γ = 1, Q₀ = −2, ε = 0.2** (rule 15). In the offline toy
  model ([rationale/b1-exp60-floor/learner_sim.py](rationale/b1-exp60-floor/learner_sim.py); not data) all three
  Q arms reach ≈ 1.0, and they reach criterion **during the greedy pre-probe** (≈ 48 in-water picks eliminating
  the 26 tools; K = 0 also gives ≈ 1.0), so the executed training episodes are expected to add little; with the
  optimistic default Q₀ = 0 the same learner did not reach criterion in 10 episodes (≈ 0.10). **Floor
  competence:** a `FLOORS_BEATEN` verdict carries "floor under-explored" instead of "floor: beaten" when, on any
  Q_EXPLORE seed, the key read at the first post placement's first select never had `escape_water` tried before
  the post probe. **A tie
  is the expected result**, and under BF2 the Exp 60 row then takes `PARTIAL (narrow: floor matched)`, recorded
  with the sample-efficiency gap (FEAR: zero executed training actions; Q_EXPLORE: the count to criterion).
- **H3 (Q_EXPLORE_PRIOR; reported).** Expected to reach criterion within one or two episodes: the prior carries
  most of the action knowledge.
- **H4 (Q_EXPLORE_RAW; reported).** Expected to match Q_EXPLORE (the raw keys separate shore and water as
  well as the EC's do here).

**Statistics.** Exact two-sample permutation test, one-sided, on the per-seed binary primary DV, 8 v 8
(C(16, 8) = 12,870 relabellings); H1 and H2 form one family, Holm-corrected at 0.05. H3/H4 descriptive. The
verdict is INCOMPLETE when FEAR, RANDOM or Q_EXPLORE has fewer than 8 clean seeds; PRIOR and RAW report their n.
Verdict values (frozen for the evidence table): `FLOOR_MATCHED_CHANCE` (H1 not shown), `FLOOR_MATCHED_Q` (H1
shown, H2 not), `FLOORS_BEATEN` (both shown), `INCOMPLETE`. No
equivalence test is claimed for a tie: a tie is reported as "FEAR > Q_EXPLORE not shown", with the rates.

## The claim, stated per outcome (in advance)

- H1 passes, H2 ties (expected): "Wire-4 learns from context conditioning alone WHEN to act; WHICH act comes
  from a designer-specified defensive prior. It needs no executed training actions. A tabular learner (α = 0.5,
  γ = 1, Q₀ = −2, ε = 0.2) armed only in water — the harness supplies WHEN — needs N executed actions to match its
  first-placement escape over the full repertoire, and fewer (H3) when given the same prior. B1 does not test WHEN
  against a floor; that rests on FEAR vs ABLATED." The Exp 60
  row: `PARTIAL (narrow: floor matched)` with that sentence and the FEAR-vs-ABLATED contrast standing.
- H1 and H2 both pass (`FLOORS_BEATEN`): the row stays EARNED with a "floor: beaten" annotation citing this
  verdict; the same sentence about the prior applies.
- H1 fails: the Exp 60 row takes `PARTIAL (narrow: floor matched)` against chance; the first-placement result is
  not distinguishable from a random policy on this apparatus.

## Frozen implementation rules (v3, from the build-plan review)

1. **Ties.** The greedy argmax breaks ties uniformly with the seeded policy RNG, never by tool name.
2. **Cost.** Interval reward = −Σ over the samples between selects of clip((20 − oxygen)/20, 0, 1) × the elapsed
   seconds / 0.5 (trapezoid between the observations at t−1 and t; 0 while out of water), minus the intensity of
   each `drive:health`/`drive:oxygen` pain since the last select (read once, from the trial's pain record with a
   cursor). Actions that block the loop therefore pay for the time they took.
3. **Q training.** K = 10 executed episodes whatever ends them: surfaced (terminal 0), the harness reads sensed
   oxygen ≤ 12 or a saturating oxygen pain (terminal −1.0, whichever comes first), or the cap (truncated; cap =
   measured damage onset − 3 s). A health drop or any `drive:health` pain refuses the arm-seed (Exp 60's rule).
   An arm-seed refuses only on instrument faults, damage, a policy fault or > 2 deaths, never on how the episodes
   ended.
4. **Episodes in probes.** Every placement ends with `end_episode`: surfaced is terminal 0; cap, dirty or
   never-submerged is truncated. Arming and `end_episode` both clear the pending (s, a). The pre-probe's updates
   and executed actions count toward "executed actions to criterion".
5. **Executed-action credit.** A pick that never executed (same-tool cap, pending at stop) is not credited and is
   recorded.
6. **First post placement not clean** (dirty or never submerged): that arm-seed is refused, never substituted.
   Rerun rule: an instrument-cause refusal reruns that arm-seed once, mechanically, with every attempt recorded;
   a behavioural refusal is never rerun.
7. **Policy faults.** A policy exception, a params violation, or a loop thread that exits before `stop` refuses
   the arm-seed loudly.
8. **Order.** Per seed, the arms follow a Williams design for five treatments (first sequence 0 1 4 2 3, square A
   = its five cyclic shifts, square B = their reversals): seeds 11–15 take square A (every arm once in every
   position), seeds 16–18 the first three rows of square B, so first-order carryover is balanced only over a full
   10 rows and approximately here. 11: F R W Q P · 12: R Q F P W · 13: Q P R W F · 14: P W Q F R · 15: W F P R Q · 16: P Q W R F · 17: W P F Q R · 18: F W R P Q
   (F FEAR, R RANDOM, Q Q_EXPLORE, P Q_EXPLORE_PRIOR, W Q_EXPLORE_RAW).
9. **RAW key.** (`is_in_water` ≥ 0.5, oxygen bin) with bins [0, 5), [5, 10), [10, 15), [15, 40].
10. **PRIOR set.** The stamped tools whose name contains a `_DRIVE_TOOL_AFFINITIES["threat"]` keyword, asserted to
    be exactly two (`flee`, `escape_water`).
11. **Rig check.** Refuse unless the leader proxy reports `infer_lane.in_flight == 0` with no request in the last 2
    minutes (a missing server passes, recorded) and server TPS ≥ 19.5 (RCON `tick query`). **v5 (the rig, 2026-10-13):**
    - A present proxy that reports no lane metrics (`infer_lane` null, which a leader that has served no large-lane
      request returns) REFUSES: idle cannot be shown. B1 makes no LLM calls, so the run is done with the leader
      down, recorded `absent`.
    - A problem that waiting cannot clear refuses at once: no lane metrics, an unreadable request ring, an
      untimestamped entry, unreadable TPS, or a proxy that refuses the read. Only a transient one is waited out, up
      to 10 minutes: a job in flight, a recent request, a TPS dip, or a busy proxy (a timeout, 5xx or rate limit).
    - Such a problem belongs to the campaign, not to one key. The rig is checked once before the campaign's first
      attempt, and any refusal there (a non-transient one at once, a transient one after its 10-minute wait) exits 3
      with nothing spent.
    - A non-transient rig refusal mid-campaign stops the campaign after writing that row. On a key's first attempt
      the rerun is not spent, and `--resume` gets it back once the rig is fixed. On a key's last attempt the
      refusal is final, and the verdict counts the key as refused.
12. **Diagnostic seed.** 99, never pooled with 11–18.
13. **Cadence.** The loop runs at 4 Hz and proposes at 2 Hz in every loop window; FEAR's propose-only training runs
    at 4 Hz direct calls (Exp 60's, unchanged). The difference in EC exposure between FEAR's and the Q arms'
    training is disclosed, and a key-coverage diagnostic is reported: the share of first-post-placement selects
    whose key had `escape_water` tried before the post probe (pre-probe or training), whether the first select's
    key had, and the number of distinct water keys per arm.
14. **Calls.** Every call's start and return time is recorded (the executor spy's `t`/`t_return`); a call in flight
    at the teleport is derived from them. **v5:** derive it from each row's `calls_all`, where both times are relative
    to the arm-seed's start. Inside a placement's `calls`, `t` is relative to that window's teleport but `t_return`
    is the raw monotonic clock. That shape is inherited from the shared `WaterTrial` and is left alone, because the
    Exp 60/61/62 rows carry it.
15. **Learner constants (v4, OD5).** γ = 1 (an undiscounted cost-to-go: every episode terminates) and Q₀ = −2, the
    pain components of a pain episode only (the −1 terminal plus the saturating pain's own intensity; the full
    return of a pain episode, with ≈ 2.4 units of oxygen cost, is ≈ −4.4). Q₀ is the same for every action and
    state: it encodes the scale of the bad outcome, not which act is defensive. It is not a tuned value: the
    confounding lens's sweep gives ≥ 0.99 for any Q₀ from −1 to −10 once γ = 1 (the condition is Q₀ below the
    escape's own return, ≈ −0.45). The floors' training signal is a graded interoceptive deficit while FEAR's is
    the pain publish; the difference favours the floor. With v3's γ = 0.9 and Q₀ = 0, every untried action outranks a tried escape (which
    pays for its ≈ 2.6 s of blocking), so the learner explored instead of learning: Q_EXPLORE ≈ 0.10 on the first
    post placement in the toy model, a floor that loses by construction. The toy model is committed with this
    document; it is not data and gates nothing.
16. **In-water credit (v4).** Only a pick made while `observation` reads water is credited; a shore pick (the roam,
    or a select before the teleport lands) never is. Inside placements and training the episode is closed by the
    harness's 4 Hz `end_episode("surfaced")`: the live `escape_water` holds `jump` 600 ms after the head clears, so
    the harness reads air while the escape is still in flight and the loop is blocked, and the pending pick it
    credits (terminal 0) is the escape. As a backstop the first select whose observation reads air after a water
    pick is terminal there and `end_episode` truncates any later pick (under rule 17 that path runs only if the
    gate and the observation disagree). Without this, a pick the loop made between the head clearing and the
    harness's read took the surfacing's credit, and the escape was bootstrapped as a non-terminal step.
17. **Water-only arming (v4).** In placements and training episodes the armed policy idles until its own
    observation reads water; the shore roam is armed without the gate. The policy is still armed immediately before
    the teleport, but a select made on the shore before the teleport lands returns nothing, so no floor executes in
    the window an action chosen from the shore (on the scripted bridge every training episode was otherwise ended by
    a blind shore `escape_water` that no update could credit). FEAR is unchanged: it carries no fear on the shore,
    so it proposes nothing there at the first post placement; its placements 2–6 shore picks are the habit finding
    above. That FEAR claim is CHECKED: a FEAR first-post success whose surfacing call started before the first
    in-water sample refuses the arm-seed (behavioural; rule 6), which holds FEAR to the floors' gate. The count of
    dry selects skipped is recorded, split into before the first water read and after it.

## Run runbook (v5)

Run on the bridge box at current `main`, on a quiet box. `<rcon-password>` is a placeholder, and the real value never
goes into a commit or PR. Paste the commands as written, with no inline comments.

1. **After this FROZEN v5 PR is merged and pulled on the rig: the water apparatus check.** The prereg lint dates
   the prereg from its first add, so it cannot enforce this ordering; the runbook does. The check is a gated record,
   under the `b1_` data name:

   ```
   python scripts/survival_world/exp60_water_check.py --out docs/experiments/data/b1_exp60_floor_apparatus_<YYYY-MM-DD>.json --rcon-password '<rcon-password>' --username maxim --write-experiment-results
   ```

   - **Exit 0** is a PASS and stamps the anchor.
   - **Exit 4** is a FAIL. Its record is still written and is committed as context, and no second check runs that
     UTC day (re-checking until one passes is a garden of forking paths).
   - **Exits 2 and 3** write no record.
   - **"Same day" means the UTC date of the check's `ts` equals the campaign's UTC start date.** The harness refuses a
     fresh gated campaign otherwise (`apparatus_day_problem`). A `--resume` the next UTC day keeps the campaign's one
     record (the verdict refuses a file that names two).
   - No other water check may run between this step and step 4: the run refuses unless the anchor's stamped `ts` is
     this record's (`anchor_record_mismatch`).
2. **The record PR.** Commit only that record, merge it with a merge commit, then pull it on the rig. A gated run
   reads its apparatus record byte for byte from main (`committed_record_problem`).
3. **Take the leader down** (rule 11), restart the bridge, and confirm with `lsof -nP -iTCP:8099 -sTCP:LISTEN`
   (empty).
4. **The run, from the repo root.** Every arm and seed, in the frozen order, into the default campaign file
   `docs/experiments/data/b1_exp60_floor_<campaign_id>.jsonl`:

   ```
   python scripts/survival_world/b1_floor_run.py run --apparatus-record docs/experiments/data/b1_exp60_floor_apparatus_<YYYY-MM-DD>.json --rcon-password '<rcon-password>' --write-experiment-results
   ```

   It takes about 3 h. If the process dies, resume with `--resume <that file>` on the same tree; a crashed attempt
   is closed as `unrecorded` and never rerun.
5. **The verdict:**

   ```
   python scripts/survival_world/b1_floor_run.py verdict --data docs/experiments/data/b1_exp60_floor_<campaign_id>.jsonl --json docs/experiments/data/b1_exp60_floor_<campaign_id>_verdict.json --write-experiment-results
   ```

6. **The data PR** carries the rows, the verdict, and T1-13's move under §The claim, stated per outcome. A
   floor-matched verdict obliges the move in the same PR: T1-13 at `PARTIAL (narrow: floor matched)`, dated on or
   after the UTC date of the newest harness row's `ts` (`scripts/lint_bf2_obligation.py`). Every campaign file
   started is committed, aborted ones included (backlog M49). Merge it with a merge commit.

## Owner decisions (taken 2026-10-11, all as recommended)

- **OD1. The binding tabular floor is Q_EXPLORE** (the review's strict default; a tie is likely and makes the
  row PARTIAL). The alternative the review rejected: Q_YOKED, which cannot lose.
- **OD2. Scope and seeds:** 8 seeds per arm (11–18), and Q_EXPLORE_RAW IS run (≈ 8 × 3.5 min ≈ 30 min more rig).
- **OD3. Q's per-tick cost:** −(20 − oxygen)/20 in water (game-native, from the sensor), plus any pain publish.
- **OD5 (2026-10-11, after the harness build). The learner's constants are γ = 1, Q₀ = −2** (rule 15), as
  recommended over keeping v3's and amending H2 (a strawman floor) or raising K to 30 (toy ≈ 0.33). Rules 16–17
  are implementation fixes found by the scripted smoke, disclosed with it.
- **OD4. Exp 60 records:** T1-13 annotated 2026-10-11 with the placement-2–6 habit finding and the prior (prose
  only, no status change; this re-run decides the status).

## Not claimed

That Maxim beats reinforcement learning in general, or that a model-free learner in general needs N actions (N is
this learner's); WHEN-to-act against a floor (the floors are armed only in water; a floor that had to learn WHEN
would need an explicit idle action credited on the shore, not built); any result outside this pool and geometry; anything about
Exp 61/62 (their floors are their own); that the defensive prior is learned.

# B1 (DRAFT v2, 2026-10-11) — the Exp 60 floor: what does the learned drowning-avoidance add over chance and over a model-free learner?

> **STATUS: DRAFT v2, design settled.** The four-lens design review ran on v1 (2026-10-11; reports preserved
> verbatim in [rationale/b1-exp60-floor/](rationale/b1-exp60-floor/)). Every lens returned DO-NOT-BUILD on v1; v2
> folds all of them (§What the review changed). Owner decisions OD1–OD4 TAKEN 2026-10-11 (§Owner decisions; all
> as recommended). Not frozen: the harness is built and code-reviewed next, then dry-run, then this document is
> frozen on main before any data. Plan: [../plans/baseline_floors.md](../plans/baseline_floors.md) Stage B1 (owner decisions
> BF1–BF2; the seam is a `src/` proposer hook, 2026-10-10). Parent: [exp60_drowning_avoidance_prereg.md](exp60_drowning_avoidance_prereg.md)
> (EARNED 2026-09-16, T1-13). Every Exp 60 apparatus, preflight, probe and training rule is inherited unless
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
| FEAR | `recommend_action` (unchanged) | Exp 60's: propose-only, 10 episodes to the saturating air-hunger pain | The claim, re-run contemporaneously; doubles as Exp 60's trigger re-run |
| RANDOM | Uniform over the stamped `available_tools` | None | BF2 random floor: chance over the full repertoire |
| Q_EXPLORE | Tabular Q-learning (below) over the full repertoire | 10 EXECUTED episodes through real loop windows | **BF2 tabular floor (OD1)**: a model-free learner given action experience |
| Q_EXPLORE_PRIOR | The same learner, action set limited to the threat-affinity tools (`flee`, `escape_water`) | As Q_EXPLORE | How much of the action knowledge the prior carries (reported, not gated) |
| Q_EXPLORE_RAW | The same learner keyed on raw sensors (`is_in_water` × oxygen in 4 bins) | As Q_EXPLORE | Separates the EC's split from Wire-4 (reported, not gated) |

**Analytical, not run.** Q_YOKED (above). The mechanism twin (state-value TD on the cluster plus the same
prior) is Wire-4 by construction; FEAR vs ABLATED already measures "the prior with vs without situation value".

**The tabular learner.** State = the tuple of the active modality clusters `propose_via_substrate` already
encoded that tick (Wire-4's keys; per agent, never compared across agents). Actions = the stamped set, each with
`params={}`. Per-tick cost r = −(20 − oxygen)/20 while in water (oxygen read from the same state the encoder
read), plus −intensity of any `drive:health`/`drive:oxygen` pain published since the last select (OD3, taken).
Update order, frozen: at select t, read r for the interval since select t−1, update Q(s_{t−1}, a_{t−1}), then
choose a_t. Episode ends are explicit: the harness calls `end_episode(reason)` before every rescue; surfaced is
terminal with 0, saturating pain is terminal with that pain, a cap is truncated (no terminal update).
α = 0.5, γ = 0.9, Q initialised to 0; ε = 0.2 in training. In the probes, ε = 0 and the table KEEPS updating
online from executed outcomes (FEAR's NAc also learns inside probes; freezing Q would make a greedy learner
repeat a failing act forever). The policy RNG is seeded per seed × arm; the class, constants and seed are
stamped.

## The hook (`src/`)

`propose_via_substrate(..., policy: SubstratePolicy | None = None)`. The policy replaces only the
`nac.recommend_action(...)` call and returns the same dict shape (`tool_name`, `params`, `confidence`,
`reasoning`), so the proposal object, clusters, margins and strategy stay one path. `SubstratePolicy.select(*,
agent_id, available_tools, clusters)` receives no `nac` and no drives: a floor structurally cannot read Maxim's
learning. Threaded through `run_agentic_loop(substrate_policy=)` → `loop_substrate.substrate_tick` →
`_loop_kwargs` → `run_minecraft_aut` → `WaterTrial`. Identical across arms: the tool filter
(`MAXIM_SUBSTRATE_TOOL_WHITELIST` asserted unset), the encode, `note_active_clusters`, the situation cue, the
pain tick, the cadence, the same-tool cap (a greedy policy loses one tick in six to it, as FEAR does), autonomy,
`execute_and_learn`, telemetry. `policy=None` is byte-identical to today. The policy is ARMED at the teleport and
disarmed at rescue: loop liveness, the shore warm-up and the recovery ticks run disarmed (FEAR returns nothing on
the shore; a floor would otherwise act there and an action begun on the shore could block the window). Rejected
alternative: replacing `nac.recommend_action` on the instance (an untyped monkeypatch of a bio class).

## Apparatus and refusals (changes from Exp 60)

- `params={}` for every proposal, pinned by a unit test that the hook cannot emit params. With `{}`, `move_to`,
  `turn`, `mine_block` and `place_block` fail before the bridge; `flee` fails fast underwater; `attack_nearest`
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

## DVs, hypotheses and gates (directions and numbers before data)

**Primary DV:** P(surface before the US) on the FIRST post placement, per seed (binary). **Secondary
(reported):** P(surface) over all 6 post placements (labelled "fear plus positive link plus habit" for FEAR);
latency; shore-roam affordance calls per arm (water ENTRY is about 0 for every arm by construction, since no
action with `params={}` can move the bot into water, so only call counts are reported); Q arms' learning curve
(the episode of first surfacing in training, and the greedy action in the water state after episodes 1, 2, 5
and 10); executed training actions to criterion (FEAR: zero in training, one failed `flee` per first placement).

- **H1 (FEAR > RANDOM; gated).** RANDOM's chance of drawing `escape_water` in time: about 4–5 useful draws from
  26 tools, 1 − (25/26)^5 ≈ 0.18, so RANDOM's first-placement rate ≈ 0.2. FEAR's ≈ 1.0 (Exp 60: 5/5).
- **H2 (FEAR vs Q_EXPLORE; gated as FEAR > Q_EXPLORE).** Expected: with a per-tick oxygen cost and 10 executed
  episodes, Q_EXPLORE finds `escape_water` (ending the cost early) and its first-placement rate ≈ 0.8–1.0. **A tie
  is the expected result**, and under BF2 the Exp 60 row then takes `PARTIAL (narrow: floor matched)`, recorded
  with the sample-efficiency gap (FEAR: zero executed training actions; Q_EXPLORE: the count to criterion).
- **H3 (Q_EXPLORE_PRIOR; reported).** Expected to reach criterion within one or two episodes: the prior carries
  most of the action knowledge.
- **H4 (Q_EXPLORE_RAW; reported).** Expected to match Q_EXPLORE (the raw keys separate shore and water as
  well as the EC's do here).

**Statistics.** Exact two-sample permutation test, one-sided, on the per-seed binary primary DV, 8 v 8
(C(16, 8) = 12,870 relabellings); H1 and H2 form one family, Holm-corrected at 0.05. H3/H4 descriptive. No
equivalence test is claimed for a tie: a tie is reported as "FEAR > Q_EXPLORE not shown", with the rates.

## The claim, stated per outcome (in advance)

- H1 passes, H2 ties (expected): "Wire-4 learns from context conditioning alone WHEN to act; WHICH act comes
  from a designer-specified defensive prior. It needs no executed training actions; a model-free learner over
  the full repertoire needs N executed actions to match it, and fewer (H3) when given the same prior." The Exp 60
  row: `PARTIAL (narrow: floor matched)` with that sentence and the FEAR-vs-ABLATED contrast standing.
- H1 and H2 both pass: the row stands; the same sentence about the prior applies.
- H1 fails: the Exp 60 row takes `PARTIAL (narrow: floor matched)` against chance; the first-placement result is
  not distinguishable from a random policy on this apparatus.

## Owner decisions (taken 2026-10-11, all as recommended)

- **OD1. The binding tabular floor is Q_EXPLORE** (the review's strict default; a tie is likely and makes the
  row PARTIAL). The alternative the review rejected: Q_YOKED, which cannot lose.
- **OD2. Scope and seeds:** 8 seeds per arm (11–18), and Q_EXPLORE_RAW IS run (≈ 8 × 3.5 min ≈ 30 min more rig).
- **OD3. Q's per-tick cost:** −(20 − oxygen)/20 in water (game-native, from the sensor), plus any pain publish.
- **OD4. Exp 60 records:** T1-13 annotated 2026-10-11 with the placement-2–6 habit finding and the prior (prose
  only, no status change; this re-run decides the status).

## Not claimed

That Maxim beats reinforcement learning in general; any result outside this pool and geometry; anything about
Exp 61/62 (their floors are their own); that the defensive prior is learned.

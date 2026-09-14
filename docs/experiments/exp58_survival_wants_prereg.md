# Exp 58 — Phase-1 survival wants: deficit-contingent eating + learned dark-fear (pain→cluster valence)

> **STATUS: FROZEN PRE-REGISTRATION (2026-09-14) — NO live data taken.** v1 → four-lens
> design review (six DO-NOT-BUILDs; rationale preserved verbatim in
> `rationale/exp58_survival_wants/`) → folded v2 → mechanism built as **Wire 4** (PR #700,
> two-lens code reviewed; note the Wire-3→Wire-4 rename — Wire 3 was taken) → **offline
> instrument gates G1–G3 PASSED on clean main**
> ([data/exp58_offline_gates.json](data/exp58_offline_gates.json), gated run @ 0847b61f):
> G2 same-cluster PASS (the DNB-2 write/read mismatch measured absent), G1 read-path PASS
> (the full pain→fear→`flee` composition through `propose_via_substrate`: dark fear −1.0,
> lit 0.0, flee proposed exactly and only at the feared situation), **G3: the cave-mouth
> gradient encodes to a third cluster → the secondary entry-avoidance DV is NOT LICENSED
> and is hereby DROPPED — the primary DV is escape (time-in-dark / latency-to-exit), as
> pre-committed below.** Classroom apparatus: `setup_world.py classroom` (same PR as this
> freeze). This document must be on `main` before the first live data timestamp; any
> change after the first data point is a dated addendum, never an edit.
> Owner decision 2026-09-14: the fear mechanism is the pain→cluster negative-valence write.

## The two claims

**Claim A (composition, prior-driven — honestly named):** under game-native hunger pressure
on the live 1.20.4 survival world, substrate-primary action **selection** is
deficit-contingent. A MECHANISM claim (the composed break-1/2/3 loop across trials) that
moves R2's `PREMISE-NULL` record to a measured live result. NOT a learning claim.

**Claim B (learning — the load-bearing one):** an agent that takes mob damage while the
dark world-cluster is active acquires **situation-keyed fear** (negative cluster valence on
the dark cluster) and subsequently shows **anticipatory, state-contingent escape from the
dark while healthy** — shorter dark visits / faster exit — where a fear-ablated agent does
not. (Primary DV is ESCAPE, not entry-avoidance: the read fires off the ACTIVE cluster, so
entry-avoidance is claimable only if the boundary-activation probe shows cave-mouth states
map to the dark cluster — see §Instrument gates. Confounding D1 / bio SF-1.)

## The mechanism (build FIRST; enters `[engineering]`; own reviewed src/ PR)

Measured basis: Step 2 (`docs/wiring/pain-needs-declared-failure-modes.md`) — every
existing fear store is situation-blind; `cluster_reward_bias` has no pain-side caller;
action-blame is correctly B8-suppressed. Front-gate: existing infrastructure cannot carry
situation-fear (measured, probe readouts).

1. **Store** — NAc cluster-valence map keyed **`(agent_id, cluster_id, failure_mode)`**
   with a v1 failure-mode allowlist of `drive:health` (wiring W-5: an unkeyed store lets
   hunger pain write lit-area fear in both arms, unrecoverable post-hoc). v1 clamp
   `[-cap, 0]` (fear only; counter-conditioning impossible in v1 — documented). Named
   distinctly from `cluster_reward_bias` (no fake tool key). **No per-tick decay in v1,
   declared**: extinction is active re-learning, not a timer (bio SF-2, wiring W-7); the
   no-decay-caller absence is by design, not an oversight.
2. **Write** — PainBus subscriber booking `−intensity·α` onto the co-active **world**
   cluster (interoception excluded: hurt-fear is tautological). Clusters come from an
   **NAc-owned `note_active_clusters` stash** written by the loop (the
   `set_pending_operant_action` precedent); the subscriber is **auto-wired in
   `build_pain_bus`** beside the existing NAc subscribers (wiring W-3: a harness-level
   attach recreates the three-CLI-sites bug class). Interactive-mode gated like siblings.
3. **Loop ordering fix (part of the mechanism PR)** — the per-tick encode is hoisted above
   the `evaluate_failures` pain tick so pain sees the CURRENT tick's clusters (wiring W-4:
   as-is, damage on the lit→dark transition tick books fear on the LIT cluster — aimed at
   our own specificity gate). Episodes are additionally staged off transitions (belt).
4. **Read** — when an ACTIVE cluster carries valence ≤ −θ, contribute a normalized
   anticipatory "threat" need into the drive/need vector (break-1's machinery), combined
   with the innate reactive `health→threat` need by **max, never sum** (bio SF-6).
5. **Read-path consumer (the load-bearing fold — triple-confirmed DNB):** the body gains a
   **parameterless defensive affordance `flee`** (bridge-implemented: pathfind to the lit
   safe anchor, `canDig=false`), whose name matches the existing
   `_DRIVE_TOOL_AFFINITIES["threat"]` table. Without it the threat need scores nothing on
   this body (zero keyword matches in the current roster) and `recommend_action`'s empty
   params make directed avoidance inexpressible — FEAR ≈ ABLATED was guaranteed as drafted
   (bio DNB-1, confounding D1, wiring W-1). Flight-as-fixed-action-pattern is also the
   more amygdala-faithful shape. The reactive `health→threat` need gains its first
   consumer on this body for free.
6. **θ / α / cap pre-registered with arithmetic**: values fixed at freeze with ≥2× margin
   between K·α accumulation and θ (confounding S3); the instrument gate confirms
   magnitude, not just sign.
7. **Bundle export: DEFERRED to Phase 2, named.** Fear-travel is five wiring items of
   which two fail silently (merge drops unlisted fields; un-rekeyed cluster maps arrive
   dead — wiring W-6). v1 persists locally with the NAc state (format-versioned). The
   Phase-2 prereg owns the five-item roster + a two-agent round-trip test.

## Instrument gates (all offline/cheap, all BEFORE any live trial)

1. **Read-path gate through the PRODUCTION caller** (wiring W-2): a probe variant whose
   per-tick body is `propose_via_substrate` against the scripted bridge — the existing
   `dark_danger_probe.py` hand-composes the loop and would fail a working mechanism (and
   hand-patching it would turn the ship gate into a recipe test, the D43/D44 anti-pattern).
   Gate: the valence write flips negative on dark AND selection flips to `flee` under
   forced dark + valence, through the production caller.
2. **Same-cluster assertion** (bio DNB-2): the staged pain-time snapshot (dark + hurt +
   hostile-adjacent) and the healthy-dark probe snapshot must encode to the SAME world
   cluster through the production encoder — else fear books on a "combat-in-dark" cluster
   the healthy prober never activates (unreadable write, false null). Failure = redesign
   before build.
3. **Boundary-activation probe** (confounding D1 / env SF-5): sweep lit → cave mouth →
   interior, record the active world cluster; sets the dark-zone measurement line INSIDE
   the light gradient and decides whether entry-avoidance is claimable (secondary DV).
4. Phase-0 separability re-check on the actual classroom geometry (standing rule).

## Apparatus (folds env DNB-1/DNB-2 — the classroom rebuilt against real 1.20.4 mechanics)

Paper 1.20.4 survival world; substrate-primary AUT on the bridge box. Classroom changes
from v1 (all vanilla, D1-compatible — each makes the contingency sharper):

- **`doMobSpawning false` globally + a ZOMBIE SPAWNER block in the dark room** (spawners
  ignore the gamerule; adult pin via `SpawnData`, `MaxNearbyEntities` density cap,
  `RequiredPlayerRange` proximity gating). The spawner itself requires block-light 0, so
  "dark → zombies" remains a real game mechanic — light the room and it stops. This kills
  the natural-cave contamination (26–30 uncontrolled hostiles, the ~70 cap starving the
  classroom, skeleton arrows shot OUT of darkness writing fear on the lit cluster,
  creepers breaching geometry).
- **Sky-exposed lit safe area at frozen day** (adult zombies burn — game-native
  containment of pursuit; needs the adult pin) + a closed wooden door the bot can traverse
  but zombies can't break on normal difficulty (env DNB-2: pursuit otherwise lands damage
  in the LIT cluster — flee-and-get-caught is the expected trajectory of the behaviour
  under test). RCON kill-sweeps at arm boundaries.
- **`doWeatherCycle false`** added to the gamerule set (env SF-1: storms make the surface
  spawnable, rain extinguishes burning zombies, `is_raining` flips cluster identity).
- **`spawnpoint` set to the lit safe area**; death accounting defined; apparatus-stop at a
  death cap (env SF-4). Bread seeded in the classroom so health regen (needs food ≥ 18)
  can restore probe-eligibility (env SF-3). Fresh classroom chunks per arm (local
  difficulty accrues with inhabited time — env SF-9).

## Design (arms, DVs, gates)

**Claim A** — 5 seeds × 30 selection cycles per state, deficit (food ≤ 6) / satiated
(food ≥ 16), both induction lanes disclosed (`minecraft:hunger` / `minecraft:saturation`
effects over RCON — the labelled second lane; sensed food settle-polled against the
thresholds, env SF-7):
- DV: selection at the recommendation stage, read from the **emitted event stream**
  (`_emit_recommend_action_event` JSONL — `recommend_action` is not read-only under
  exploration; wiring W-8), with `substrate_explore_bonus_weight` pinned in the frozen
  apparatus fingerprint (confounding S6).
- **Gate (folded, confounding D2):** the state-blind causal link makes an absolute
  satiated gate a predictable false-null once eat's link trains (measured 0.89). Primary
  gate is the per-seed **deficit-minus-satiated selection gap ≥ 0.5** with the satiated
  trajectory reported and trial-order analysis pre-registered (bio SF-3); the seed is the
  unit of analysis (effective n = 5, confounding S5).

**Claim B** — arms FEAR / ABLATED (subscriber not auto-attached in the ablated build —
harness-level, declared; pain still publishes, Wire 2 still fires), fresh agent per arm,
**5 seeds per arm**:
- **Training: yoked, harness-scheduled exposure trials** (disclosed lane) — K ≥ 10 usable
  episodes per seed, identical schedule across arms (confounding S1: free-roam couples
  damage dose to arm — the fear arm curtails its own exposure). Free roam only in probes.
- **Usable episode** (env SF-2, confounding S8): cumulative health ≤ 13 (≥3 zombie hits —
  one hit does NOT breach the band) with the dark cluster active at pain time, counted by
  **pain PUBLISHED on the bus** (the latch means episodes ≈ writes, not hits);
  `evaluate_failures` ticks during lit recovery ride the real loop.
- **Claim-B agent kept satiated** by disclosed induction (confounding S7: hunger otherwise
  competes at selection AND books positive cluster credit through a channel the ablation
  does not remove).
- **Probe windows: mob-free (kill-sweep) + full-heal**, health ≥ 18 (excludes the reactive
  need — fires only below 14; derivation pinned, body-spec-change stop rule), identical
  protocol pre and post (confounding S2).
- Primary DV: **time-in-dark per entry + latency-to-exit** (escape — what the read
  supports). Secondary (only if instrument gate 3 licenses it): P(enter dark).
- Gates: per-seed medians + pooled, permutation test (the house pattern): FEAR
  post-training time-in-dark ≤ 0.5 × own pre-training AND ≤ 0.5 × ABLATED post; mechanism
  DV dark-cluster valence < 0 with **relative specificity** |lit| < 0.2 × |dark|
  (chase-out damage classified out of usable-K, bio SF-4); lit-area activity level
  (defined: actions per probe window in the lit zone) within ±25% between arms.
- Falsifier: FEAR ≈ ABLATED on the primary DV **after** instrument gates 1–3 passed →
  a genuine behavioural null (the gates ensure it cannot be a knowable wiring absence);
  ships as a null.

## Stop rules / refusals (exit 3/4)

Instrument gates 1–4 unpassed; damage arithmetic vs the body's declared `comfort_band`
(band-edge trap); < K usable episodes (with the spawner-starvation check); death cap;
provenance/dirty-tree per `_provenance.py`; frozen-apparatus config fingerprint
(explore weight, encoder threshold, drive specs) asserted at start.

## Explicitly NOT claimed

Transfer to agent B and fear-travel in bundles (Phase 2, incl. the W-6 five-item roster);
cross-context generalization (R1 stands); entry-avoidance unless gate 3 licenses it;
"the environment taught it" for any disclosed-induction lane; extinction dynamics;
positive situation-valence; graduation-row changes before the run.

## Fold log (what the lenses changed; what was dismissed and why)

- Read path dead on this body → `flee` affordance + offline read-path gate (bio DNB-1 =
  confounding D1 = wiring W-1, triple-confirmed; the single load-bearing fold).
- Probe-as-gate hand-composes → production-caller probe variant (wiring W-2).
- Write/read cluster mismatch → same-cluster assertion gate (bio DNB-2).
- Spawn control as drafted did not exist game-natively → spawner-block classroom
  (env DNB-1); pursuit contamination → burn-containment + door + episode classification
  (env DNB-2).
- Claim A satiated gate → gap gate (confounding D2).
- P(enter) primary → escape primary (bio SF-1 / confounding D1).
- Dose coupling → yoked schedules (confounding S1).
- Fear key gains failure_mode (wiring W-5); no-decay declared (bio SF-2 / wiring W-7);
  max-combine (bio SF-6 / wiring NIT).
- **Dismissed:** hard difficulty for damage (breaks the door containment, env SF-2);
  renaming the store to an Amygdala class (bio NIT — stays an NAc valence store,
  no pre-validation bio-naming); harness-level subscriber attach (wiring W-3).
- **Deferred with a name:** bundle fear-travel → Phase 2 (wiring W-6).

## Data discipline

Prereg (this file, frozen post-fold) on `main` before the first data timestamp; clean
tree; gated evidence path; data PR merge-commit; a null ships as a null. Build order:
mechanism PR (store + subscriber + loop-order + `flee`; bio-memory + runtime-tools +
embodiment briefs; two-lens code review) → instrument gates offline → apparatus PR →
freeze → live run.

## Addendum 2026-09-14 (dated, pre-data): harness protocol refinements

Recorded BEFORE the first live data timestamp, per the freeze rule. The Claim-B harness
(`scripts/survival_world/exp58_run.py`, two-lens reviewed) refines the frozen design in
the following ways, each with its rationale; where an addendum item supersedes a frozen
sentence, the sentence is named:

1. **Training is confined, PROPOSE-ONLY conditioning** (the contextual-fear-conditioning
   chamber): during exposure episodes the bot is confined in the dark room (door closed)
   and the harness drives `propose_via_substrate` ticks — the exact composition the
   offline gates verified — with NO action execution. Rationale: (a) a full loop in a
   sealed room books `record_outcome` FAILURE credit on every blocked `flee` — an
   arm-asymmetric negative causal link on the very read path the probe measures
   (verified against `tool_dispatch.record_outcome`); (b) it makes exposure structurally
   arm-independent, resolving confounding S1 more strongly than schedule-matching.
   SUPERSEDES the frozen sentence "`evaluate_failures` ticks during lit recovery ride
   the real loop" — lit-recovery ticks are also propose-only (same composition, no
   execution). Consequence, disclosed: the causal-link surface enters the probes
   untrained in BOTH arms.
2. **Probes are SHEPHERDED PLACEMENTS, not free roam** — SUPERSEDES "free roam only in
   probes". Measured basis: substrate-primary has no param-free "enter" affordance, so
   the naive entry rate is ~0 (env SF-6) and free-roam probes would produce no
   dark-entries to measure in either arm. Protocol (identical pre/post and across arms):
   6 placements per probe window; each placement teleports the bot to the dark-room
   interior with the FULL agent loop live (real execution); DV = latency to cross the
   door plane (RCON ground truth), censored at 45 s; kill-sweep + disclosed full-heal
   (instant_health + saturation effects) before the window and between placements; door
   OPEN during probes (probes are mob-free, so containment is moot). The frozen
   "time-in-dark per entry" DV reduces to this latency under placements; gates apply to
   censored per-seed medians with the same 0.5× ratios.
3. **Flee actuation is a PREFLIGHT** (verify-actuation-before-theorizing): the review
   measured that the pre-fix apparatus could not execute `flee` at all (the pathfinder
   does not open closed doors, and `bot.spawnPoint` is the world spawn, not the
   classroom). The bridge now takes an explicit `--flee_x/--flee_z` anchor (printed by
   the classroom builder), and each seed begins with one real executor `flee` from
   inside the dark room that must cross the door plane, or the seed refuses (exit 4, no
   data). The per-seed record carries `flee_negative_links` so any residual in-probe
   flee failures are auditable.
4. **Live G2 is a per-seed stop rule**: the noted world cluster is captured at every
   usable episode; the training-majority cluster must equal the probe-identified dark
   cluster or the seed is REFUSED — an unreadable fear write must never ship as a
   behavioural null. (The offline G2 verified this only for scripted states; hostile
   sensors differ maximally between pain time and swept probe time.)
5. **Stop rules implemented in the harness**: offline-gates record present with
   `all_pass` true; frozen-apparatus fingerprint (fear α/cap/θ/allowlist, encoder
   pattern threshold) asserted and stamped per record; usable-episode bound asserted
   below the body's health comfort band; death cap 2 per seed (scoreboard
   `exp58_deaths`); under-K/geometry/settle failures refuse. REFUSED seeds append a
   record with a `refusal` field and no behavioural DVs — the verdict script excludes
   them and reports the refusal count.
6. **Record identity**: every invocation stamps a `run_id`; the verdict script must refuse
   duplicate (arm, seed) rows absent explicit resolution.

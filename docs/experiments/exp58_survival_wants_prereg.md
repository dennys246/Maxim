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

## Addendum 2 (dated 2026-09-14, pre-data): underground cave classroom + light tolerance

The surface classroom could not be made dark: Paper's bulk `/fill` does not recompute
SKYLIGHT, so a command-built surface roof left the sealed room at a uniform light 15
(diagnosed live — three ruled-out theories then a direct light-gradient measurement; a
chunk reload and a single-`setblock` toggle both failed to relight). The classroom is
therefore built UNDERGROUND (owner decision; also the doc's original "a cave, not a
hand-darkened room" intent), which stands on the two reliable light mechanisms and never
uses the broken skylight-removal path:
- **deep burial → skylight 0 natively** (carving air under solid rock needs no recompute —
  the correct value equals the stored value), and
- **block light** from a `light[level=15]` block lights the safe chamber (the separate,
  reliable path).

`setup_world.py classroom` now encases a solid-stone cuboid at y=40 and carves a lit safe
chamber (spawnpoint + flee anchor) and a longer dark chamber (spawner at the far end),
joined by an oak door; the bridge's `flee` opens the door (`canOpenDoors`) so the dark
chamber stays sealed/0 except during an exit; the harness closes the door for every
dark-light read. Verified live (first full dry-run): safe 15/13, dark 0, flee opens the
door and crosses the plane, all 10 training episodes landed from the spawner.

**Light tolerance:** the lit/dark preflight gates are a tolerant band — safe `light ≥ 13`,
dark `light ≤ 1` — not exact `15`/`0`. A live block-light sensor jitters 13–15 with the
bot's sub-block position relative to the source; exact-equality invited flaky refusals.
13-vs-0 remains a crisp, well-separated split, and the world cluster is formed from all 17
sensors, so a 13↔15 jitter on one axis is negligible to cluster identity. The scientific
contingency (dark cluster vs lit cluster) is unchanged.

## Addendum 3 (dated 2026-09-14, pre-data): light_level abandoned → depth-based danger cluster

The underground cave's light also proved unusable. A day/night probe (owner's diagnostic
idea) showed `light_level` in this world is **not a reliable instrument**: an underground
cell read 14 at day / 0 at night (skylight-contaminated where burial should give 0), a
cell with no light source read 13 day and night, and the same coordinates gave different
values across repeated probes. Paper/mineflayer light here is spatially patchy,
run-to-run inconsistent, and skylight-leaking underground — three independent failure
modes (on top of Exp 56's "read DEAD"). No geometry fixes a sensor that answers the same
question differently each time.

**Decision (owner): abandon `light_level` as the discriminator; define the danger cluster
by DEPTH.** `y_altitude` is the bot's own position, read straight from the entity with no
lighting engine — 100% reliable. The classroom is now a deep pit (floor y=28) below a safe
chamber (floor y=40), joined by a staircase the bot flees UP; the danger cluster separates
on `y_altitude` (a strong ~12-block signal) **plus** hostile presence (the spawner's
zombies). No door, no light source, no relight — nothing depends on the broken skylight
path. It is *more* faithful to the cave idea, not less: descend into the deep dark where
the monsters are.

Harness changes (same mechanism, same DV family): geometry preflights check `y_altitude`
(safe ≥ mid_y=34, pit < mid_y) instead of light; a preflight asserts the safe and dark
world clusters are DISTINCT (the separation requirement light used to carry); `_in_dark`
= `y_altitude < mid_y`; the flee-latency DV = time for `y_altitude` to rise past mid_y as
the bot climbs out; door logic removed. The recorded `dark_fear`/`lit_fear` keys are the
fear on the deep-pit cluster and the safe-chamber cluster respectively (`lit_` retained as
the field name; it now means "safe/upper"). The scientific claim — situation-keyed fear
changes escape behaviour — is unchanged; only the sensor that defines the situation moved
from light to depth. Live-verified end to end in the prior dry-run (before this pivot):
the full pain→fear→flee loop fired, all 10 training episodes landed; the pivot removes the
one unreliable dependency.

## Addendum 4 (dated 2026-09-14, pre-data): live-G2 gate = readability + specificity, not id-match

The first full depth-cave dry-run ran end to end (flee climbed the staircase, pre-probe
censored baseline, 10 training episodes) and the live-G2 stop rule fired — but inspection
of the record showed it was an OVER-STRICT PROXY, not a real false null. The deep pit does
not encode to a single cluster: the 17-sensor world vector jitters across the 0.85
pattern-completion boundary, so the same spot re-completes to a small NEIGHBOURHOOD of ids
(observed 2: 6 episodes on one, 4 on the other). Fear spread to BOTH (each reached the
−1.0 cap), and the post-training probe activated one of them (`d7d…`) which carried
`dark_fear = −1.0`. So the fear was demonstrably READABLE at the probe state — the
measurement would have worked — yet the guard refused because its proxy required the
training-MAJORITY cluster id to equal the probe id (`a76 ≠ d7d`).

**Fix (owner-approved): the live-G2 gate checks the condition it was always meant to —
readability + specificity — not id-matching.** A seed passes iff the probe-activated dark
cluster carries fear (`dark_fear ≤ −θ`, θ=0.5) AND the safe cluster does not
(`|lit_fear| < θ`). This is strictly better: it still refuses a genuine false null (probe
cluster with no fear → `dark_fear ≈ 0`), and it self-protects against EXCESSIVE cluster
instability (too many oscillating clusters → fear diluted below θ on the probe cluster →
refuse). The per-seed record now also stamps `distinct_episode_clusters` and the full
`cluster_fear_dump`, so the pit's cluster neighbourhood is disclosed, not hidden.

Interpretation note for the verdict: the "danger cluster" is a small neighbourhood of
world-cluster ids the deep pit spans, all carrying fear; the safe chamber occupies a
disjoint neighbourhood with none. The scientific claim (situation-keyed fear changes
escape behaviour) is unchanged; the gate now verifies the mechanism's actual requirement
directly rather than through a fragile id-equality proxy.

## Addendum 5 (dated 2026-09-14, pre-data): persistent hostile → reliable danger-cluster separation

The depth-only danger cluster proved UNSTABLE across runs: the same deep pit sometimes
separated from the safe chamber and sometimes MERGED into one world cluster. Root cause is
L11 sensor dilution — a 12-block `y_altitude` difference is only ~0.09 of the cosine across
17 world sensors, well under the 0.85 separation threshold, so whether safe and dark split
depended on whatever *else* happened to differ at encode time (mainly whether a spawner
zombie was present). This is a genuine substrate limit, and the offline gates missed it
because they used big multi-axis (Phase-0 light box) or scripted contrasts, not the live
single-axis case. (Honest note: those gates validated the instrument on an easier problem.)

**Fix (owner decision): give the danger cluster a reliable, big, multi-axis contrast by
making the pit ALWAYS hostile.** The classroom now summons a **persistent `NoAI`,
`PersistenceRequired` "clustermob" zombie** deep in the pit — it never moves, attacks, or
despawns; it exists solely so `nearest_hostile_dist` (and the hostile axis generally)
reliably differs between the safe chamber (far from it) and the dark pit (adjacent). The
spawner (now higher-rate) still provides the AI attackers that deal training damage; sweeps
(`setup_world --sweep`, and the harness's per-poll probe sweep) SPARE the clustermob
(`tag=!exp58clustermob`) so the danger cluster keeps its hostile axis while the AI
attackers are cleared for the full-health probe. The pit is also lengthened (14 blocks) so
depth + position + hostile are all sizeable, reliable axes — and to leave depth headroom
for the Exp 59 treasure layer.

This makes the contingency "a deep place with a monster in it = danger," which is arguably
the *right* framing (a place is dangerous because a threat is there), and it is what made
the pit separate cleanly when it did. The preflight cluster-distinct check remains the gate:
if safe and dark still encode to the same cluster, the seed refuses (no forcing). Danger
cue is now depth + reliable hostile; light remains unused (Addendum 3).

## Outcome (2026-09-14): BLOCKED at the cluster-separability instrument check — L11 dilution, live

Exp 58's Claim B never reached a gated behavioural measurement. The Wire-4 MECHANISM works —
a full live dry-run showed the composed loop fire end to end: the bot flees up the staircase,
the pre-probe baseline is cleanly censored, all 10 conditioning episodes land (zombie bites →
pain → fear accumulates to the −1.0 cap on the active world cluster). What does NOT hold is the
prerequisite the whole design rests on: **the danger situation and the safe situation do not
form distinct world clusters.**

Across the apparatus iterations the SAME barrier appeared under three discriminators —
light (unreliable/non-physical here), depth alone (too small a fraction of the y_altitude
range), and finally depth + a verified-adjacent persistent hostile + a ~21-block position gap.
Even that three-axis contrast encodes safe and dark to ONE cluster (cluster-distinct preflight
refuses). This is **L11 sensor dilution realized live**: the 17-sensor world channel averages
each sensor to ~1/17, so no *partial* contrast clears the 0.85 pattern-completion threshold, and
none of the survival contrasts is full-range (`hostile_count` is even identical at both, since
the bridge counts all loaded hostiles; only `nearest_hostile_dist`, `y_altitude`, and position
swing, partially). Phase-0's `survival_phase0` gate reported 1.0/1.0 separability only because
its box swung light AND altitude across their FULL ranges at once — the offline gates validated
an easier problem than the live classroom, and that is the honest lesson here.

**Verdict: NULL-WITH-CAUSE (instrument-blocked), same family as R1/R2.** Situation-keyed fear
requires the situation to form a distinct cluster; the 17-sensor world channel does not provide
one for these survival contrasts. This does not refute the Wire-4 mechanism (it fires); it
bounds what can be measured on the current substrate. No gated data was taken (every live run
refused at the cluster-distinct / live-G2 preflight — the guards did their job).

**Prerequisite for reviving Exp 58: address L11 for the world channel** — the diagnostic-first
plan (capture live survival vectors, measure the real cosine geometry, replay channel-split /
scaled-threshold / gain remedies on them, then build the one that separates) is the path.
Channel-splitting the world modality (L11's own recommendation; the `modality:` sub-channel
schema in `minecraft_benchmark.md`) is the leading candidate, chosen from data, built with its
own review + Exp 56/57 re-baseline. Instruments that gave false confidence and must be fixed
alongside: `survival_phase0` and the exp58 offline gates used big-multi-axis / scripted contrasts
that did not exercise the live single-axis separation — the diagnostic replaces them.

**Correction note — 2026-09-16, POST-DATA (Exp 60 Amendment 7).** Three sentences above describe
the dry-run as the loop "firing end to end" / "flee climbed the staircase": Addendum 3 ("Live-verified
end to end in the prior dry-run"), Addendum 4 ("The first full depth-cave dry-run ran end to end") and
§Outcome ("a full live dry-run showed the composed loop fire end to end: the bot flees up the
staircase"). The staircase flee was the harness's PREFLIGHT actuation check (`exp58_run.py`, a direct
`executor.execute`), not the agent loop; training is propose-only by design; and the "cleanly
censored" pre-probe is exactly what the loop produces at its default PLANNING autonomy level, under
which `run_minecraft_aut` never executed a body affordance (measured 2026-09-16; fixed in
`_loop_kwargs`, guard `tests/unit/test_substrate_primary_wake.py`). The mechanism's WRITE side
(pain→cluster fear) and the offline read gates stand as measured; the loop-executed READ was never
demonstrated live on this line. No Exp 58 claim rested on it (BLOCKED at the instrument, above).

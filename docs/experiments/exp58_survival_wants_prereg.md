# Exp 58 — Phase-1 survival wants: deficit-contingent eating + learned dark-fear (pain→cluster valence)

> **STATUS: DRAFT v2 (2026-09-14) — POST four-lens design review, PRE-freeze.** v1 went to
> the four lenses (findings preserved verbatim in `rationale/exp58_survival_wants/`); this
> version folds them. Six DO-NOT-BUILDs were returned — the dead read path independently by
> THREE lenses — all folded below; the dismissed/deferred items are recorded in §Fold log.
> Freezes on `main` before any data timestamp, after the mechanism + apparatus PRs land.
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

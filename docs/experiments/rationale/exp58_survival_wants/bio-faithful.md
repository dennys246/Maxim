# Exp 58 design review — BIO-FAITHFUL lens

**Reviewed:** `docs/experiments/exp58_survival_wants_prereg.md` (draft, 2026-09-14).
**Charter question:** does this test the mechanism's REAL job, not a caricature? Does the
manipulation respect how the substrate/body/drives actually work?
**Read for this review:** `docs/agents/bio-memory.md`;
`docs/wiring/pain-needs-declared-failure-modes.md`; `src/maxim/decisions/nac.py`
(module docstring, `NACConfig`, `recommend_action`, `_DRIVE_TOOL_AFFINITIES`,
`decay_cluster_reward_biases`, `_inherent_bias_keys` comments, Wire-2 sections);
`src/maxim/embodiment/sem.py` (`drive_pain_for_value`, `corrective_need_intensity`,
`drive_comfort_progress`, DriveSpec CC3 freeze); `src/maxim/embodiment/body.py`
(breach latch); `src/maxim/runtime/agent_loop.py` (`_DRIVE_CORRECTIVE_NEEDS`,
`_read_drive_states`, `propose_via_substrate`);
`src/maxim/_data/components/bodies/minecraft_player.yaml`.

---

## DO-NOT-BUILD

### DNB-1. The fear READ terminates nowhere on the declared body — no executable avoidance affordance exists, so the FEAR arm is a null pre-written by the affinity table

The prereg's read path is: learned valence ≤ −θ → normalized "threat" corrective need →
"Selection then rides the existing drive-prior affinity machinery toward avoidance/move
affordances." Checked against the shipped machinery and the shipped body:

- `NAc._DRIVE_TOOL_AFFINITIES["threat"]` = `("flee", "hide", "retreat", "escape",
  "withdraw", "defend", "shelter")` (nac.py:547).
- `minecraft_player.yaml` affordances: `move_to`, `turn`, `mine_block`, `place_block`,
  `eat`, `attack_nearest`. **Zero of these match any threat keyword**, and no tool name
  contains the substring "threat" (the direct name-match path). `attack_nearest` is
  correctly outside the repertoire (FIGHT is Phase-1b by explicit invariant).
- Even if a keyword were added for `move_to`, `recommend_action` returns
  `"params": {}` (nac.py:2382) and `propose_via_substrate` forwards it verbatim —
  substrate-primary cannot fill `move_to(x, z)`. Avoidance is inherently DIRECTIONAL;
  a direction-blind boost on a parameterized movement tool cannot express "away from
  the dark" (boosted `move_to` toward the cave is the same tool). The composed break-3
  loop worked precisely because `eat` is parameterless.

**Failure scenario:** the mechanism builds, the write forms, the instrument-check
readouts flip negative — and FEAR ≈ ABLATED on every behavioural DV because the threat
need scores no available tool. The falsifier clause ("ships as a null naming the
read-path as the break") would then dress up a wiring absence *knowable at design time*
as an empirical result. Note this is not hypothetical for the new mechanism only: the
EXISTING reactive `health→threat` need also reaches nothing on this body today — the
brief's "flight/freeze/recover" floor is aspirational on `minecraft_player`.

**Fix shape:** give the body (and the bridge) a **parameterless defensive affordance**
whose name matches the existing threat affinity table — e.g. `retreat` /
`flee_to_light`: pathfind toward the nearest lit/safe region or directly away from
`nearest_hostile` / the feared zone. This is the *more* bio-faithful shape anyway:
biological flight is a species-typical fixed action pattern (amygdala→PAG), not
parameterized deliberate navigation — the animal does not compute coordinates, it
executes an escape program. One affordance, named into the existing table, serves both
the learned (Claim B) and the innate reactive threat need. Its addition must land in
the mechanism `src/` PR, and the prereg's design section must name it.

### DNB-2. Fear may be WRITTEN to one world cluster and READ from another — the pain-time world snapshot is not the probe-time dark snapshot, and the planned instrument check cannot see the difference

`health`, `food`, `nearest_hostile_dist`, and `hostile_count` are all `modality: world`
sensors on this body (16 world sensors total), and the channel's A4
midpoint-at-rest design is built so "a resting sensor is silent, a departing one
shouts." Compare the two moments the mechanism must connect:

- **Write time** (mob damage in the dark): light_level shouting low, health shouting
  (20→8 ≈ 0.5→0.2 normalized), nearest_hostile_dist shouting (64→~2 ≈ 0.5→0.016),
  hostile_count off rest.
- **Read time** (healthy probe, health ≥ 18, no adjacent hostile): only light_level
  departs; everything else rests silent.

With the sensor-surface pattern threshold at **0.85** (bio-memory brief: NOT 0.44) and
winner-take-all cluster identity, these two sparse "shouting" vectors plausibly
pattern-SEPARATE: the fear books onto a "combat-in-the-dark" cluster the healthy
prober never re-activates. The write is then mechanically perfect and behaviourally
unreadable — a false null. The Phase-0 record (dark/lit separate 1.0/1.0) was measured
between *resting* states and does not cover this. Worse, the planned instrument check
(re-run `dark_danger_probe.py`) stages the same damage state for both write and
readout, so **it validates the write and goes green while the live probe reads a
different cluster** — the instrument cannot catch the failure it exists to gate.

Bio note: this is exactly where the amygdala analogy needs engineering honesty.
Biological context representations generalize smoothly (partial cue → graded recall);
EC winner-take-all clusters at 0.85 do not. The design must *verify* same-cluster
identity, not assume amygdala-style generalization.

**Fix shape (cheap, offline, pre-build):** extend the instrument check with a
**same-cluster assertion**: encode (a) the staged pain-time world snapshot and (b) the
healthy-dark probe snapshot through the production encoder and require the SAME world
cluster id (and dark ≠ lit). Make it a stop rule beside the Phase-0 separability
re-check. If they separate, resolve at design time — candidates: a slimmer classroom
body variant (the `minecraft_bench*` precedent exists exactly for this), booking the
valence onto every world cluster active in a short pre-pain window, or a stabler key.
Do not build the harness on the assumption.

---

## SHOULD-FIX

### SF-1. The read is ESCAPE-shaped, not AVOIDANCE-shaped — the primary gate P(enter dark) measures what the mechanism cannot in principle deliver

The read fires "when an ACTIVE cluster carries valence ≤ −θ" — i.e. only after the
agent is already in the dark. While approaching the entrance the LIT cluster is active
(valence ≈ 0 by the specificity design), so no threat need exists at the decision that
matters for P(enter). What the mechanism can deliver is rapid exit: reduced
time-in-dark, short dwell latency. In conditioning terms this is conditioned **escape**;
true anticipatory **avoidance** requires the fear to fire *before* context entry —
in animals carried by predictive cues and graded context overlap, here requiring either
graded/partial cluster activation at the boundary (Phase-0 says the encoding is
binary dark/lit) or a predictive read (imagination/SCN lookahead — not in this design).
The prereg's own framing ("anticipatory") is right about the *health* axis (fear before
damage) but wrong about the *spatial* axis (fear before entry).

**Failure scenario:** FEAR arm shows collapsed time-in-dark but unchanged P(enter);
the pre-registered primary gate (P(enter) ≤ 0.5× baseline) fails; the result is
recorded as mechanism-null when the mechanism did its job.

**Fix shape:** promote **fraction-of-time-in-dark / latency-to-exit** to the primary DV
and gate; demote P(enter dark) to secondary/exploratory with the limitation stated in
the prereg ("v1 read is situation-onset; entry-avoidance requires a predictive read,
future work"). Do not attribute a P(enter) miss to the valence write.

### SF-2. Extinction/decay is unspecified — and the choice decides whether the probe measures anything at all

The prereg declares the store's alpha but is silent on decay, and the codebase forces a
choice with named precedents either way: per-tick taus (reward 50 / Wire-2 Pavlovian
fear 200 / cluster bias 300), the decay-EXEMPT inherent class ("innate fears do not
extinguish the way learned ones do" — nac.py `decay_cluster_reward_biases`), and two
wall-clock classes (cluster 1-day vs the 7-day `percept_valences` class — whose own
NACConfig comment already flags that conditioned aversion is biologically the MOST
persistent of these associations and a slower-or-no schedule is the calibration
candidate).

**Failure scenarios:** (a) the map silently inherits a tau≈200–300 per-tick schedule;
K ≥ 10 interleaved episodes plus post-training probe windows span enough ticks that the
fear decays below θ by probe time → false null attributable to an undeclared knob.
(b) No decay is chosen but never declared → a fear that never extinguishes ships as an
implicit, unreviewed bio commitment, and Phase 2 exports it in bundles with improvised
merge semantics.

**Bio verdict on "a fear that never decays":** acceptable for v1, and arguably MORE
faithful than a decay timer — biological extinction is active inhibitory re-learning
under safe re-exposure (context-gated, not erasure); passive decay-to-zero is the
caricature. But it must be a *declared* choice.

**Fix shape:** pre-register: v1 has **no per-tick decay** (or tau ≫ the full
training+probe span, stated numerically), joins the **7-day-or-slower wall-clock
class** (Phase 2 needs the fear to travel across sessions — a 1-day half-life would
gut the transferable-want story), names real extinction (safe re-exposure
counter-learning) as future mechanism out of scope, and pins the **merge direction for
bundle export** as tighten-only (matching `tighten_negative_biases` — a received fear
may deepen a held one, never lift it).

### SF-3. Claim A's satiated gate is threatened by the architecture's own documented state-blind channel — declare whether trials EXECUTE

The R2 resolution (this repo's own record) established that substrate-primary selection
is driven by the prior **plus the state-blind tool-success causal link**, and
`test_learned_link_dominates_drive_heuristic` pins that a learned link dominates the
drive heuristic. `get_positive_outcomes` is keyed on `tool:eat` alone — no state. If
Claim A's alternated deficit/satiated cycles execute the selected action, `eat`
accumulates causal_pos across deficit cycles and increasingly clears `min_confidence`
during satiated cycles where nothing else scores (the interoception-cluster-keyed
credit does NOT transfer across states, but causal_pos does) → P(select eat | satiated)
climbs with trial index → the ≤ 0.2 gate fails for a reason fully known at design
time. Bio framing: that outcome is habitual, devaluation-insensitive selection — a
genuinely interesting finding, but not the claim under test, and the prereg as drafted
would record it as a composition failure.

**Fix shape:** pre-register whether Claim A cycles execute. If recommendation-only
(no execution, no credit), the gates cleanly measure the deficit-gated prior — the
honest reading of Claim A as written. If actions execute, pre-register the trial-order
analysis (early vs late blocks) and state the expected mechanism that keeps satiated
selection below gate — or accept and pre-declare the habitization curve as a secondary
measurement rather than a gate-killer.

### SF-4. The "lit ≈ 0" specificity gate is fragile to chase-out damage — and booking lit-context pain would be bio-CORRECT, which is exactly why the gate needs a tolerance

A zombie that follows the agent out of the dark zone and lands a hit while the LIT
world cluster is active books fear onto lit — and that is faithful contextual
conditioning (the context at US time is what conditions), not a bug. As drafted the
gate "lit-cluster valence ≈ 0" would then fail on correct mechanism behaviour.

**Fix shape:** physically contain mobs in the dark region (environment lens's call), or
pre-register a *relative* specificity gate (|lit valence| ≤ some declared fraction of
|dark valence|) plus an episode-classification rule: a damage episode whose co-active
world cluster was lit counts against neither arm's usable K and is reported.

### SF-5. The health ≥ 18 anticipatory/reactive split is CORRECT — pin its derivation in the prereg so a YAML retune cannot silently invalidate it

Verified against the shipped body: `health` is homeostatic with `set_point: 20`,
`comfort_band: 6.0`. `corrective_need_intensity` emits the reactive threat need only
when deviation < −band, i.e. health < 14; `drive_pain_for_value` is 0 above 14; the
breach latch is silent. So at health ≥ 18 the innate reactive source is cleanly
excluded — the split does respect how the reactive need actually works. But the prereg
states the 18 threshold without the derivation; a later body-spec retune (band or set
point) would silently break the exclusion. **Fix shape:** write the derivation
(18 > 20 − 6) into the prereg and add a stop rule: the run refuses if the body YAML's
health drive spec differs from the frozen values.

### SF-6. Two threat sources into one need slot — define the combination as MAX, and define what "usable damage episode" means at the subscriber

(a) The anticipatory (learned) threat contribution and the reactive `health→threat`
need land in the same `"threat"` key of the drive vector. When hurt IN the dark both
fire; they must combine by **max** (mirroring `derived_needs` accumulation in
`_read_drive_states`), never sum — a sum double-counts one danger and can push past the
[0,1] contract that `recommend_action` now enforces by skipping values > 1.0 (the R2
guard would silently drop an overflowing threat need — the exact silent-no-op class).
(b) The severity latch means pain publishes once per episode (re-fires only on
deepening) — K episodes ≈ K writes, not K×hits; alpha calibration must be done against
that, and "usable damage episode" in the stop rule must be defined as **a
`drive:health` write observed at the subscriber's output store** (never
`pain_bus.recent`, per the wiring doc's own instrument lessons).

---

## NIT

- **N-1 (naming — charter q6):** staying a NAc extension is the right call; do not mint
  an `Amygdala` bio-system for one map. Precedent: Wire-2 `_percept_valences` — the
  codebase's existing "Pavlovian fear conditioning" — already lives in NAc, and NAc
  shell does encode aversive valence biologically, so the mapping is not dishonest.
  Bio naming is load-bearing but does not count as validation, and the mechanism enters
  `[engineering]`. Name the store as **valence** (e.g. `cluster_valence` /
  situation valence), never "bias" — this respects the module docstring's own stance
  ("avoidance is carried by valence, not negative bias") and keeps it distinct from
  `cluster_reward_bias` (action-scoped, `(agent, cluster, tool)`) where the new store is
  deliberately action-UNscoped (`(agent, cluster)`). If a fear family accrues (Phase-1b
  danger-cue reflex, extinction learning), an amygdala module split can be *earned*
  later.
- **N-2 (clamp — charter q2):** the v1 `[-cap, 0]` fear-only clamp is bio-defensible
  and consistent: no positive producer exists (honest scope), the clamp pushes that
  invariant into the type (a positive write becomes impossible, not silent), and
  extinction-is-not-erasure means safe exposure should not positively overwrite anyway.
  State in the mechanism docstring that counter-conditioning is impossible by
  construction in v1.
- **N-3 (world-only booking — charter q3):** correct. The interoception cluster at pain
  time IS the US ("being hurt"); conditioning it would be the caricature (fear of being
  hurt that fires only while hurt). Scope note for the docstring: with more
  exteroceptive channels (audio), biological compound-CS conditioning would book all
  non-US channels — v1 books world only, future work.
- **N-4:** injecting the anticipatory threat need into the drive vector will ALSO be
  encoded into the interoception channel (derived needs are, per `_read_drive_states`'s
  NB) — felt fear as interoceptive state is bio-faithful, but it shifts interoception
  cluster identity and hence credit routing during fearful ticks. Expected side
  effect; note it in the prereg so it isn't discovered mid-run.
- **N-5:** the prereg's "anticipatory" vocabulary collides with the existing
  `TemporalCreditDistributor.anticipatory_pre_activate` (SCN phase-based, unrelated).
  One clarifying sentence avoids a reviewer conflating them.
- **N-6:** the write books the cluster set noted by the loop's last encode;
  `evaluate_failures` runs at the TOP of `propose_via_substrate`, before this tick's
  encode, so pain-time clusters are one tick stale. In a stable dark room this is
  harmless (and slightly helps DNB-2 for the first hit); document the staleness in the
  subscriber rather than leaving it to be rediscovered.

---

## Verified clean

- **Charter q1 (bio shape):** pain → negative valence on the co-active *exteroceptive*
  situation cluster is a reasonable functional analog of BLA contextual fear
  conditioning — EC world cluster as the context representation (nominally the right
  region, even), US stamps valence on it, CR = defensive repertoire selection. The
  read-through-the-need-vector is a defensible selection interface, NOT a drive
  caricature: the threat need has no set point, no drift, no relief credit — it borrows
  only the selection seam, which is where amygdala output converges with homeostatic
  needs biologically. (Subject to DNB-1: the seam must terminate in a real affordance.)
- **Charter q4 (split):** the health ≥ 18 exclusion of the reactive path is arithmetically
  correct against the shipped body spec (needs pinning — SF-5).
- **Claim A's DV at the recommendation stage** (not execution) correctly avoids the
  mechanical food-20 refusal artifact.
- **Ablation shape** (subscriber not attached; pain still publishes; Wire 2 intact)
  isolates exactly the new mechanism, not pain generally. Clean.
- **Owner decision** (cluster-valence write over extending the Wire-2 key) is
  consistent with the measured Step-2 topology: the missing key IS the situation
  cluster, and Wire-2's `(entity_class, failure_mode)` key is structurally
  situation-blind for self-drive pain. Front-gate scope pressure satisfied — the
  probe measured that existing stores cannot carry this.
- **Persist-with-NAc-state, format-versioned, bundle-exported** rides the NAc/EC
  persist-pair and hivemind machinery rather than inventing a store — right
  infrastructure choice (merge direction still to declare, SF-2).
- **Single-trial-capable alpha** follows the Wire-2 `percept_valence_alpha = 0.35`
  precedent, whose rationale (single-trial fear conditioning is one of the few
  bio-attested single-trial learning classes) transfers directly.
- **Stop rules** already include the band-edge trap and the instrument-check-before-live
  ordering — the two hardest-won lessons from the Step-2 probe are respected.

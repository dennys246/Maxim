# Exp 61 — BIO-FAITHFUL lens (four-lens design review, 2026-09-16)

Charter (`docs/experiments/DESIGN_REVIEW.md`): does the design test the mechanism's REAL job, not a
caricature? Does the manipulation respect how the substrate / body / drives actually work? Read-only
review of `docs/experiments/exp61_shared_fear_prereg.md` (DRAFT) against the code and the owning
brief (`docs/agents/bio-memory.md`, Wire-4 invariants (a)–(c), the hivemind posture, tighten-only
clamp, decay classes) and `docs/wiring/pain-needs-declared-failure-modes.md`.

## Verified first (file::symbol; no line numbers)

**The write rule, cap, allowlist, no tick decay.** `decisions/nac.py::NAc.record_cluster_fear`:
`valence -= cluster_fear_alpha (0.5) · intensity`, clamped `[-max_cluster_fear (1.0), 0]`; failure
modes outside `NACConfig.cluster_fear_failure_modes` = `{drive:health, drive:oxygen}` are a silent
no-op INSIDE the method (Exp 58 W-5); empty `cluster_id` no-op. There is NO positive writer: the clamp
is `min(updated, 0.0)` and the config comment says "counter-conditioning has no producer; extinction
is active re-learning, not a timer, so there is NO per-tick decay by design". Confirmed by grep: the
only non-test caller of `record_cluster_fear` is `proprioception/pain_bus.py::create_pain_cluster_fear_subscriber`
(intensity floor 0.3, WORLD cluster only, keys on the tick's `note_active_clusters` stash). No relief /
counter-conditioning / extinction path touches `_cluster_fear` anywhere in `src/`.

**The read rule.** `NAc.cluster_fear` = MIN over failure modes on a cluster; `NAc.anticipatory_threat_need`
= MIN over the active `{modality: cluster}` set, returned as `magnitude` iff `magnitude >= cluster_fear_threshold`
(θ = 0.5), else 0.0. `runtime/agent_loop.py::propose_via_substrate` max-combines it with the reactive
`threat` drive (`drives["threat"] = max(...)`), then `NAc.recommend_action` applies the activation floor
`if drive_value <= 0.5: continue` (STRICT `>`), scores `escape_water`/`flee` via
`_DRIVE_TOOL_AFFINITIES["threat"]` (`"escape"`, `"flee"`) at `drive_value × 0.7`, and gates on
`min_confidence` (`_resolve_min_confidence` default 0.3). **So a fear acts at the real consumer iff
|fear| > 0.5 strictly** (0.7·m ≥ 0.3 ⇔ m ≥ 0.43 is the weaker of the two gates). θ's `>=` and the
floor's `>` differ at exactly 0.5 — that is the "converges to exactly −θ, treated as dead" note in
Exp 60 §Design (iii).

**Persistence.** `NAc.dump` writes `cluster_fear` as `f"{aid}\x1f{cid}\x1f{fm}"`; `NAc.load_state`
re-clamps every value to `[-max_cluster_fear, 0]` (a foreign file cannot smuggle positive "fear");
`NAc.apply_wall_clock_decay` puts `_cluster_fear` in the SLOW class (`bias_wall_decay_half_life_s` =
7 days) and runs ONLY from `NAc.load()` (elapsed since `saved_at`) — never in-session.

**The four hivemind sites, today.** `hivemind/bundle.py::scrub_nac_state_for_bundle` pops
`cluster_fear` (comment cites Exp 58 §Mechanism 7: "five wiring items, two silent-fail");
`hivemind/ingest.py::_validate_nac_payload` strips it with a note, then bounds the sibling monotone
fields through a `(field, lo, hi, parts_expected)` table (`cluster_reward_bias` −1..1 / 3 parts with a
`_NODE_ID_CHARSET` check on the cluster id; `percept_valences` −1..1 / 3 parts); `hivemind/merge.py::rekey_nac_state`
rewrites ONLY `cluster_reward_bias`, `cluster_reward_source`, `inherent_bias_keys` through `id_map`
(absent cluster → dropped, not faked); `hivemind/merge.py::nac_merge` MIN-folds `cluster_fear` over the
key union, clamped `[-1, 0]`, with the rationale comment ("deepest fear survives — the tighten-only
direction the negative-bias clamp already codifies; for ingests the foreign side is always empty and
this degenerates to receiver-preservation"). `tighten_negative_biases` covers `_TIGHTEN_ONLY_FIELDS`
= (`cluster_reward_bias`, `percept_valences`, `goal_reward_bias`) — `cluster_fear` is not listed
because MIN is tighten-only by construction. Guards that must FLIP with the build:
`tests/unit/test_cluster_fear.py::test_bundle_scrub_excludes_fear` (and the merge-preserves /
merge-clamps pair stays).

**Exp 58 §Mechanism 7** (`docs/experiments/exp58_survival_wants_prereg.md`): "Fear-travel is five
wiring items of which two fail silently (merge drops unlisted fields; un-rekeyed cluster maps arrive
dead — wiring W-6) … The Phase-2 prereg owns the five-item roster + a two-agent round-trip test." The
draft's four sites + `fear_rekeyed`/`fear_dropped` counters cover both silent-fail shapes; the
round-trip test is in the build order.

**Exp 60 §Outcome** (`docs/experiments/exp60_drowning_avoidance_prereg.md`): FEAR `cluster_fear_dump`
after training = `{<water cluster>|drive:oxygen: −1.0}` on every seed (cap reached after 10 episodes /
20 `drive:oxygen` signals); ABLATED `{}`; shore fear 0.0; live G2 need 1.0 on the probe cluster, "one
episode cluster per seed (the probe cluster itself)". Read at `docs/experiments/data/exp60_trials.jsonl`:
each seed's single fear key is a DIFFERENT cluster id (fresh EC per seed) — transport carries the id,
so this is irrelevant to Exp 61, but it means there is no cross-seed "the water cluster"; every donor
ships its own id.

**Gate (ii) data** (`docs/experiments/data/exp60_geometry_2026-09-15b.json`): cos(shore, submerged)
A4 = 0.7874 (< 0.85), ungained 0.9734; fresh-EC ids distinct; `contrast_early_vs_late_oxygen`:
n_early = 11 (oxygen ≥ 16), n_late = 15 (≤ 13), ONE id in both bins (`same_cluster: True`). The
prereg's replay: cos(early, late) = 1.000 @ oxygen 13, 0.977 @ 5, 0.948 @ 3, 0.862 @ 0. The
2026-09-15 first run (FAIL, `saturation` at a rest the game never rests at) shows what threshold-edge
allocation looks like: submerged samples split across the shore id and a new id.

**EC completion and the aligned merge.** `similarity/ec.py::ECConfig.frozen_centroid_modalities` =
`{"interoception", "audio", "world"}` — `world` IS frozen (first embedding to reach a node is the
prototype; `pattern_complete_or_separate` skips the running-mean update for it), and
`hivemind/merge.py::DEFAULT_FROZEN_CENTROID_MODALITIES` equals it (pinned by
`tests/unit/test_hivemind_merge.py`). World threshold `SENSOR_MODALITY_THRESHOLDS["world"]` = 0.85.
`ec_merge_aligned`: no receiver match → donor node inserted under its own id, `id_map[nid] = nid`
(suffixed only on an id collision). A FRESH receiver EC therefore takes every donor node verbatim and
the id map is the identity — `fear_rekeyed == 1` is the expected arm-2 read.

**Provenance shape that exists.** `NAc.CREDIT_SOURCES` is a CLOSED set `{drive_relief, orient_relief,
tool_success, operant, mixed}` on `cluster_reward_source`, one-way "mixed" promotion
(`NAc._note_cluster_reward_source`, `merge.py::_merge_credit_sources`), scrubbed and re-keyed beside
the bias. There is NO `cluster_fear_source`; `grep -rn "social\|vicarious\|observational"` over
`decisions/nac.py`, `hivemind/`, `docs/agents/bio-memory.md` returns nothing.

**What already travels.** `scrub_nac_state_for_bundle` SHIPS `percept_valences` (Wire 2, the
situation-blind body-class aversion `(aid, 'minecraft_player', 'drive:oxygen')`) for identifier-shaped
entity classes, mean-folded and tighten-only clamped. Both Exp 60 arms publish the same 20 pain
signals with the Wire-2 subscriber attached (only the Wire-4 subscriber is detached in ABLATED), so
arm-2 and arm-3 donors carry the SAME Wire-2 aversion. It modulates text-gating salience, not
substrate-primary selection — no DV path — but it is a second fear store crossing the wire.

## Findings

### DO-NOT-BUILD — none.

The design tests the mechanism's real job. Wire-4's job is a situation-keyed ANTICIPATORY disposition
read at the decision moment; the DV (first submersion, US-free window, read at the executor spy +
bridge `is_in_water`) is that job, on the body and cue where Exp 60 measured it to be causal. Nothing
in the draft hand-composes the read path; "nothing in the read path knows the fear was imported" is
the correct bio posture (a vicariously acquired CR is read by the same amygdala circuit as a directly
conditioned one). The cluster-not-fear control (arm 3) is bio-meaningful: it is the observer that saw
the situation without the conspecific's distress — representation without valence — and it controls
the Wire-2 aversion and any training-left link for free, because the ABLATED donor carries both
identically.

### SHOULD-FIX

**SF-1. The prereg must DECIDE what a transported −1.0 means, and say it in the claim.** A donor's
−1.0 is the cap: ten saturating episodes, twenty `drive:oxygen` pains. The MIN fold hands B that
number verbatim, bypassing `record_cluster_fear` entirely (`load_state` sets the value directly). So
as drafted the transfer is byte-identical to "B drowned ten times itself" — not to observational fear,
which in every animal preparation (Mineka's monkeys, Olsson & Phelps human vicarious conditioning,
Jeon et al. observational fear in mice) is real, amygdala-dependent, drives avoidance WITHOUT the
observer ever contacting the US — and is reliably WEAKER than direct conditioning and extinguishes
faster. The draft's title says "shared survival fear" and §Open question 2 asks whether this is
"social transmission of fear in name only". Answer: the ACTING-on-first-contact half IS the vicarious
shape (observers do act without their own US — that is exactly what makes the paradigm interesting),
but full-strength MIN models the intensity as if experienced, and the codebase has no extinction to
express the "faster to extinguish" half at all (SF-4). Two honest options; pick one and record it:

- (A) **Attenuate at the ingest bound** — `_validate_nac_payload` multiplies admitted fear by a
  declared constant (a social discount; suggested `0.75`, i.e. `−1.0 → −0.75`), applied ONLY on the
  foreign path (`merge-nac` is trusted-local by declaration and stays verbatim), BEFORE the MIN fold.
  Arithmetic against the real consumer: need 0.75 > 0.5 floor (1.5× margin), `escape_water` score
  0.525 ≥ 0.3 — fires on B's first contact. Own later drowning deepens it through the normal write
  rule (−0.75 − 0.5·1.0 → −1.0). No new mechanism: a scalar beside the existing caps, the same class
  as `MAX_FOREIGN_TOTAL_OBSERVATIONS`. Two properties come free: (i) **chain attenuation** — B's
  re-export hands C `0.75²` without any provenance field (a second-hand fear is weaker than a
  first-hand one, which is the rumor-cascade behaviour the MIN fold otherwise lacks — every hop at
  zero loss is the least faithful thing in the draft); (ii) a shorter wall shelf-life
  (`−0.75 · 0.5^(t/7d)` crosses the floor at t ≈ 4.1 d vs 7.0 d for −1.0), a crude stand-in for
  faster social extinction. **Be honest in the prereg that the DV cannot tell 0.75 from 1.0 on this
  body**: the DV is binary, the floor is the only gate, and a fresh B has no competing score — the
  discount is a representational commitment plus the cascade property, not something Exp 61 measures.
- (B) **Ship verbatim and re-title** — "a learned fear transfers AS IF EXPERIENCED"; drop "social"
  language; state the cascade/no-loss limitation under "Not claimed".

Either is buildable today. The lens recommends (A). What is NOT faithful and must not be adopted for
THIS experiment: a sub-floor "social prior" (import at |v| ≤ 0.5 so one own exposure confirms it).
That IS expressible with the existing update rule and no new code (a −0.4 prior + one saturating own
pain → −0.9, vs −0.5 dead with no prior: the prior turns a two-episode acquisition into a one-shot
one — facilitated acquisition, a real phenomenon), but it is a DIFFERENT claim whose DV is
"acquisition speed after one own US", and it fails Exp 61's first-contact-without-US headline by
construction. Name it as a follow-up (Exp 61b), not as an arm.

**SF-2. A fear that arrives DEAD is uncounted — add `fear_below_floor` to the ingest report and to
donor sanity.** The draft bounds ingested fear to `[-1, 0]` and counts `fear_rekeyed`; but the real
consumer acts only at |v| > 0.5 strictly (verified above). A donor with an unsaturated fear (−0.5,
−0.25) ships a key that re-keys cleanly (`fear_rekeyed == 1`, `fear_dropped == 0`) and NEVER acts —
D43's "capability, not fix" shape, invisible to every counter in the draft. The TRANSFERRED gate's
"counting only placements whose G2 read shows the transferred fear active" would silently shrink n
rather than refuse. Fix: (i) donor sanity asserts the shipped value's magnitude > θ with the declared
margin (it already asserts "exactly −1.0" — keep that, and after SF-1(A) assert the POST-discount
value > 0.5 + margin); (ii) the ingest report carries `fear_below_floor` (|v| ≤ `cluster_fear_threshold`
after discount) beside `fear_rekeyed`, and arm 2 refuses a pair with `fear_below_floor > 0`. The
mechanism should read θ from `NACConfig`, not a literal.

**SF-3. Per-mode transport policy: ship only the failure mode under test; keep the allowlist a
single source.** Health-pain fear is the every-damage class (`docs/wiring/pain-needs-declared-failure-modes.md`:
every world damage writes through `drive:health`), so a transported `drive:health` fear on a cluster
B routinely occupies makes B avoid it — but that is NOT a transport-specific hazard: the donor itself
avoids that cluster, and whether the avoidance is "benign" is decided by cue SEPARABILITY (the Exp 58
dark/safe cos 0.98 failure), not by the mode. The faithful bound is therefore not per-mode at the
mechanism (the NAc allowlist is the right filter, and `cluster_fear()` MINs across modes so a
drowning-damage `drive:health` entry on the water cluster never double-counts) but per-EXPERIMENT at
donor sanity: arms 2/4 REFUSE any `cluster_fear` key whose mode is not `drive:oxygen` — the draft's
"exactly one entry … under drive:oxygen" implies it; make it a named refusal (Exp 60 refuses any
`drive:health` pain during training, so a stray key is an apparatus fault, never data). At the
mechanism: `_validate_nac_payload` must reference `NACConfig.cluster_fear_failure_modes` (or one shared
constant), not a second literal — `pain_bus.py`'s `"world"` LOCKSTEP comment is the precedent for how
a duplicated literal silently rots.

**SF-4. State the extinction limit as a mechanism gap in §Not claimed, and add the one guard it
implies.** Verified: no positive writer on `_cluster_fear`, no tick decay, wall decay only on `load()`.
A receiver that never drowns holds the imported fear unchanged for the whole session, and across
sessions until wall decay crosses the floor (−1.0 dead after exactly one half-life, 7 d offline;
−0.75 after ≈ 4.1 d). Safe exposure (B surfaces, no US) writes NOTHING to fear — it writes a positive
`escape_water` link (Trap 2, already recorded). Biologically the vicarious-fear asymmetry
(faster extinction, social safety signalling / vicarious extinction — a conspecific seen SAFE in the
situation reduces fear) is unexpressed: positive fear values are clamped, "fear only in v1". This is
not a flaw in Exp 61 (the prereg excludes extinction) but the draft's §Not claimed should say WHY
("no counter-conditioning producer exists; the imported fear is permanent within a session and
wall-decays at the 7-day class on load"), because a reader of the headline will assume a fear that
transfers can also be unlearned. The implied guard: `nac_merge` keeps the LATER `saved_at`, so an
ingest re-stamps the receiver's decay clock to the younger of the two — record both `saved_at`s in the
ingest record so a future extinction experiment can reason about the imported fear's age.

### NIT

**N-1. The cue must be present at the DECISION moment — write this as Phase 2's admission rule for
candidate wants.** The draft asks whether the transported cue must be a binary flag. Bio answer: no —
contextual fear conditioning keys to the CONTEXT (the place/situation the animal is in when the US
arrives), and a graded cue is fine as long as the recall-moment representation and the
conditioning-moment representation are the SAME node and that node is separable from the safe one AT
THE RECALL MOMENT. Gate (ii) measured exactly that: early (oxygen ≥ 16, the recall moment) and late
(≤ 13, the conditioning moment) share one id; the graded US-proximal variable (oxygen) lands in the
same cluster and contributes nothing to separation (0.862 @ oxygen 0 is still within the node). A
graded cue that only becomes distinct NEAR the US (dark=danger's light level; hunger's food) fails
because B would complete into the fear node only after it is already too late. The generalisable rule
is `docs/wiring/cosine-separation-is-directional.md`'s: separability is direction, and the direction
must swing at the moment the want is supposed to fire. Put it in the prereg as the rule Phase 2's
other candidates (eat-when-hungry, night) are admitted or refused by, with gate (ii)'s `early/late
same cluster` + `cos < 0.85` as the operational test.

**N-2. Completion-into-A's-prototype is well-supported; record the margin, don't assume it.** Under
frozen `world`, B's first submerged reading must reach cos ≥ 0.85 with A's prototype (A's FIRST
submerged sample: pool floor, oxygen 20, `is_in_water` 1, health/food/saturation at rest — which is
B's first-contact state exactly). Evidence for tightness: 26 submerged samples over several visits and
the full oxygen range all completed to one node (replayed cos ≥ 0.862 at the far extreme); Exp 56
showed completion-into-a-transferred-node carries 40/50 first contacts. Residual risk is the
threshold-edge allocation the 2026-09-15 run showed (samples splitting across the shore id and a new
id when a constant sat at a wrong rest) — that is an apparatus state, not a bio one, and Exp 60's
live cluster-distinct preflight + `saturation` fix already guard it. Add to the mechanism DVs: the
cos between B's first submerged reading and the transferred node (the EC activation event's
`similarity`), so a marginal completion is visible rather than a binary "completed".

**N-3. Provenance (`cluster_fear_source`): follow-up, not a Phase-2 requirement.** The bias-side
shape exists (`CREDIT_SOURCES`, one-way "mixed", scrubbed + re-keyed beside the value), so a
`cluster_fear_source` with `{"own", "ingested", "mixed"}` is cheap to mirror — but nothing would read
it (no differential decay, no extinction, no re-export policy), which makes it a field without a
consumer today (the D43 family). SF-1(A)'s discount already gives the one behaviour provenance would
buy now (cascade attenuation). Ship provenance WITH the first consumer (an extinction or re-export
policy experiment). For Exp 61 the provenance lives in the ingest record (donor id, shipped key,
shipped value, post-discount value), which the data PR preserves.

**N-4. The fifth arm (fear re-pointed to the SHORE cluster) is an apparatus test, not a bio one —
skip it.** Its DV would be "B flees on the shore" (need > 0.5 on the shore cluster selects `flee`),
not "B surfaces", and it requires a hand-edited bundle — a "file" manipulation the draft rightly
avoids elsewhere. The situation-keying is already established by Exp 60's specificity gate (shore 0
vs water −1.0) and by Exp 61's mechanism DV (WHICH node B completed into + the need on that node).

**N-5. Stale brief line.** `docs/agents/bio-memory.md`'s key-files table lists
`ECConfig.frozen_centroid_modalities` as `{"interoception", "audio"}`; the code and the pinned
hivemind default include `"world"` (the D6 decision). The prereg's "under frozen-centroid `world`" is
correct; the brief row is not. Not this reviewer's file — flag for the main session.

**N-6. Wire-2 aversion travels too; say so.** `percept_valences[(aid, 'minecraft_player',
'drive:oxygen')]` ships in both arm-2 and arm-3 bundles (controlled), affects only text-gating
salience (no substrate-primary DV path). One sentence in §Confounding-facing notes so the confounding
lens does not rediscover it.

## Verdict

**FIX-THEN-BUILD.** The experiment tests Wire-4's real job — a situation-keyed anticipatory disposition
read at the decision moment on a cue Exp 60 measured to be separable and causal — through the
production read path, with a control (arm 3) that is the correct observational-learning counterpart
(situation seen, no distress) and that also controls the Wire-2 aversion and training-left links by
construction. No finding blocks the build. Four fixes are owed before the prereg freezes: the claim
must decide whether transported fear arrives attenuated (the vicarious-conditioning shape, expressible
today as a scalar at the ingest bound with cascade attenuation for free, and NOT distinguishable from
full-strength by this DV — say so) or verbatim (then re-title away from "social"); a dead-on-arrival
fear (|v| ≤ 0.5 at the strict floor) must be counted and refused, because every counter in the draft
passes it; the experiment ships only `drive:oxygen` fear with the allowlist read from one source; and
the no-extinction limit is stated as a mechanism gap with the `saved_at` re-stamp recorded. The
sub-floor "social prior that one own exposure confirms" is a real and expressible phenomenon but a
different experiment — name it, do not arm it.

## What I did NOT verify

- I did not run any code, sim, or the Exp 60 replay scripts; every number above is read from the
  committed data files (`exp60_geometry_2026-09-15b.json`, `exp60_trials.jsonl`) and the prereg text.
- I did not read the Exp 60 harness (`scripts/survival_world/exp60_run.py`) beyond grepping its
  `max_cluster_fear` / `_cluster_fear` sites; whether its training loop's Wire-2 subscriber is attached
  in ABLATED is inferred from `build_pain_bus`'s default wiring and Exp 60's "only the Wire-4 subscriber
  is detached" statement, not from the harness code.
- I did not verify `ingest_bundle`'s V1–V10 ordering (whether `_receiver_scrub` after
  `_validate_nac_payload` could touch `cluster_fear`) or `nac_merge_many` — that is the wiring lens's
  question 3, and the draft already names it.
- I did not check `maxim substrate export`'s nac-only path (arm 4's re-compose) or the H2 geometry-tag
  behaviour under `strict_geometry` for a pre-H2 donor (wiring / environment lenses).
- The animal-literature claims (vicarious conditioning weaker + faster-extinguishing than direct;
  vicarious extinction / social safety) are from memory, not from a fetched source; the design
  recommendation (SF-1) does not depend on their exact magnitudes, only on the direction, which is
  uncontroversial.

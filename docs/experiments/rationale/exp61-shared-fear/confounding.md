# Exp 61 — CONFOUNDING lens (four-lens design review, 2026-09-16)

Reviewed: `docs/experiments/exp61_shared_fear_prereg.md` (DRAFT 2026-09-16). Charter:
`docs/experiments/DESIGN_REVIEW.md` — does the metric isolate the claimed cause; could a positive OR
a null arise for a reason other than the claim; controls; statistic matched to the baseline; no
hand-composed shortcut. Fed by `docs/wiring/substrate-learning-channels.md` (the state-blind
causal-link trap) and `docs/wiring/cosine-separation-is-directional.md`.

Verdict line is at the bottom. One DO-NOT-BUILD (as drafted), seven SHOULD-FIX, five NIT.

## Verified first (evidence, not the prereg's assertions)

**V1 — What a donor's NAc contains right after propose-only training, before any probe.**
`scripts/survival_world/exp60_run.py::_run` training loop calls only
`maxim.runtime.agent_loop::propose_via_substrate` per tick. That function's ONLY NAc write is
`nac.note_active_clusters` (the per-tick situation stash, runtime-ephemeral and not in `NAc.dump()`);
everything else it does is a read (`anticipatory_threat_need`, `recommend_action`). It never calls
`record_event`, `observe`, `record_outcome*`, `update_cluster_reward` or `credit_*`. The NAc writes
pain can reach are the three auto-wired subscribers in
`maxim.proprioception.pain_bus::build_pain_bus`:
- `create_pain_cluster_fear_subscriber` → `NAc.record_cluster_fear` (Wire 4). The ABLATED arm detaches
  exactly this one (`exp60_run.py::_detach_fear_subscriber`, asserted `== 1`).
- `create_percept_valence_subscriber` → `NAc.record_percept_valence` (Wire 2), keyed
  `(agent_id, entity_class, failure_mode)`, situation-blind. Fires in BOTH arms.
- `create_pain_nac_subscriber` → `record_outcome_full` with no attribution, which links only against
  `_pending_events`; those are filled ONLY by
  `maxim.bridges.tool_pain_bridge::ToolPainBridge.record_tool_start` on an executed tool. Propose-only
  training executes nothing → nothing to link → no causal link forms.
- `cluster_reward_bias` is written only from `maxim.runtime.tool_dispatch` (the `update_cluster_reward`
  call in the outcome path) on an executed tool. `reward_bias` is written by
  `maxim.decisions.temporal_credit` via `credit_node` keyed on an EC NODE id; `recommend_action` reads
  `reward_bias(agent_id, f"tool:{tool}")` keyed on a TOOL signature, so a node-keyed entry can never
  reach selection even if it formed (it needs TemporalEvents, which again come only from tool starts).

Measured confirmation in `docs/experiments/data/exp60_trials.jsonl`: every pre-probe placement in both
runs and both arms has `calls == []`, `positive_escape_links == 0`; run-1 FEAR rows (where the post
probe never executed anything, Amendment 3) show the exact "fear, no links" state a donor would ship:
`cluster_fear_dump == {<water cluster>|drive:oxygen: -1.0}`, `escape_negative_links == 0`,
`flee_negative_links == 0`, `positive_escape_links == 0`. Run-2 rows show what a post probe ADDS:
`flee_negative_links == 2`, `positive_escape_links` 47–51, first post placement calls
`[flee False, escape_water True]`. So: **after training the two donor kinds differ in exactly one
field, `cluster_fear`.** The bundle scrub (`maxim.hivemind.bundle::scrub_nac_state_for_bundle`) ships
`links`, `event_outcome_welford`, `percept_valences` (identifier-shaped entity classes),
`cluster_reward_bias`, `cluster_reward_source`, `inherent_bias_keys`, `reward_bias`; drops `priors` and
`goal_reward_bias`; today pops `cluster_fear` (the site the src PR moves).

**V2 — The read path and the tie-break.** `propose_via_substrate` maps
`anticipatory_threat_need` → `drives["threat"]` by max; `NAc.recommend_action` scores affordances by
`_DRIVE_TOOL_AFFINITIES["threat"]` (`flee`, `hide`, `retreat`, `escape`, …) at `0.7 × need`; ties break
by `max((score, name))`, and `minecraft_player_flee` sorts above `minecraft_player_escape_water`, so the
first pick is `flee`; its fast fail books a negative link (confidence 0.5 → `0.7 − 0.25 = 0.45 < 0.7`)
and `escape_water` wins the next tick. Run-2 rows show exactly this on every FEAR seed's first post
placement. `_resolve_min_confidence` default 0.3; `0.7 ≥ 0.3`.

**V3 — The consumer's strict floor.** `NAc.anticipatory_threat_need` returns the magnitude only if
`≥ cluster_fear_threshold` (θ = 0.5); `recommend_action` then skips any drive with
`drive_value <= 0.5`. A received fear of exactly −0.5 is therefore DEAD at the consumer (Exp 60 already
recorded this for a 0.5-intensity write). `nac_merge`'s `cluster_fear` fold is `min` clamped `[-1, 0]`
(`maxim.hivemind.merge::nac_merge`), so onto a fresh receiver (0.0) a shipped −1.0 lands as −1.0 —
PROVIDED the ingest bound does not attenuate (`_validate_nac_payload` caps foreign link confidence via
`CAP_FOREIGN_CONFIDENCE`; a similar cap on fear at any value ≥ −0.5 is a structural null).

**V4 — Every channel that could surface B at first contact WITHOUT transferred fear, and which arm
controls it.**

| channel | can it select `escape_water` alone? | how it would get into a bundle | control |
|---|---|---|---|
| positive causal link on `tool:*_escape_water` | YES — any confidence ≥ 0.3 scores alone, state-blind (Exp 56 §Links executed trace) | only by EXECUTING `escape_water` (a post-probe, or an executor-routed actuation preflight) | export-before-probe; donor sanity `links == {}`; receiver `get_positive_outcomes == []`; arm 3 (identical execution history = none) |
| positive link on `flee` | NO — flee fails in water; cannot score `escape_water` | executing flee (shore roam / a shore fear) | same |
| `cluster_reward_bias` on (B, water node, `tool:*_escape_water`) ≥ 0.3 | YES, if the water node is active | only by executing `escape_water` with the water cluster active | none can form without execution; re-keyed via `rekey_nac_state`; arm 3 |
| `reward_bias` | NO — keyed on EC node id, `recommend_action` reads tool signature | drive-relief credit via the temporal distributor (needs tool starts) | structural |
| `percept_valences` (Wire 2, `drive:oxygen` on `minecraft_player`) | NO — consumed only by `runtime/gating.py` + `integration/bio_enrichment.py` (LLM path); `propose_via_substrate` never reads it | fires in BOTH donors identically | arm 3 (same field, same value) |
| drive priors / innate need | NO — `priors` dropped by the scrub; `_DRIVE_CORRECTIVE_NEEDS` has no oxygen mapping; `oxygen` is in no affinity row; `read_*` sensor tools are passive-excluded; `health→threat` cannot fire in a US-free window (no damage); the body has no `thirst` drive (the `"water"` keyword in the thirst affinity row is unreachable) | — | structural; arm 1 measures it |
| EC merge making B's SHORE reading complete into the water node (fear active on the shore) | would produce a positive for the WRONG reason (non-specific) | the aligned EC merge | NOT controlled by any drafted arm — needs a receiver-side specificity read (F4 below) |
| a surface with ZERO executor calls (bridge-held jump / teleport artefact) | counted as SUCCESS by `classify_placement` (head-in-air only) | the shared bot across donor→receiver | NOT controlled (F7 below) |

**V5 — Which Exp 56 protections Exp 61 inherits and which it drops** (`protocols/exp56_four_arm_sharing_preregistration.md`):
inherits seed-paired donors (no reuse), the four-arm shape, first-contact DV, dangling-half via the
documented nac-only re-compose, ingest-report counters as gated records, strict geometry, anti-vacuity
`--assert-noop-fails`, the "failed BOTH-HALVES is not a partial pass" rule. Drops: the balanced
execution schedule + link-balance sanity (correctly — Exp 61 donors execute NOTHING, a strictly stronger
posture, but the sanity must then be `links == {}`, F5); the ε-greedy dither, name permutation and L2
seed-invariance gate (dropped WITHOUT a stated reason, F6); the **bias-decisive decision-provenance
clause** (replaced by a weaker, tautological "fear active at contact" filter — F1); Wilson intervals
(not mentioned, F3); Phase 0 pilots (the plumbing pilot + dangling pilot are what would have caught a
link-channel leak before any campaign record — worth keeping as a one-pair dry run).

**V6 — The floor.** Exp 60 run 2 (`301eb2edff6d`, `eeb92752ee2b`): 60 pre-probe placements over 10
FRESH agents (5 FEAR + 5 ABLATED, before training), 0 surfaced, 0 executor calls, on the same
placement with the loop proven live (liveness preflight ≥ 4 ticks / 3 s). The reason is structural —
`recommend_action` returns `None` when nothing scores (no drive > 0.5, no links) — so "0/30 in Exp 60"
is really "0/60 placements, 0/10 first contacts, with a mechanistic explanation". Run 1's 0/54 pre is
NOT evidence (the loop never acted; Amendments 3–7) and must not be cited in the floor.

**V7 — Power arithmetic (exact one-sided permutation on receiver binaries; 95% Wilson).**
12 v 12: 12/12 vs 0/12 → p = 3.7e-7; 9/12 vs 0/12 → p = 1.7e-4; 2/12 vs 0/12 → p = 0.24. So n = 12
detects an Exp 60-sized effect with enormous margin — the permutation test is not what limits this
design. The MARGINS are: at n = 12 the Wilson upper bound of 0/12 is **0.24** (> the 0.20
ABOVE-FLOOR margin) and the granularity of a rate is 1/12 = 0.083, so BOTH-HALVES "< 0.10" means
"≤ 1 of 12" and one stray success is the difference between pass and fail. At n = 20 the 0/n upper
bound is 0.16, at n = 30 it is 0.11, at n = 50 0.07. The gates are point-estimate margins (house
style), so this is a statement about what the margins MEAN, not about whether they can be evaluated.

**V8 — First-contact latency vs the cap.** Run 2 FEAR first post placements: 2.918, 3.279, 3.163,
3.243, 3.343 s against `probe_cap_s` 4.335 — headroom 0.99–1.42 s, all paying the `flee` tie-break.
Every Exp 61 receiver measurement is exactly such a placement.

## Findings

### DO-NOT-BUILD (as drafted)

**F1 — The TRANSFERRED gate conditions the DV on the mechanism ("counting only placements whose G2 read
shows the transferred fear active at contact").** This is a tautology in the direction that matters:
the placements it EXCLUDES are precisely the ones that could falsify the claim — a receiver that
surfaces with fear NOT active at contact surfaced through another channel (V4) and must count AGAINST
the claim, not vanish from the denominator; a receiver whose fear is not readable at contact is an
apparatus/readability failure (the fold did not land, or completion missed the transferred node) and
must be a NAMED refusal, not a silent shrink of n. Exp 56's "bias-decisive" clause was different in
kind: it read the DECISION's provenance (which score component won the executed pick) and reported
successes won by any other component per arm so a pass on the wrong component was visible. The draft
swapped a decision-provenance read for an eligibility filter.
*Change:* split it into (a) a **pre-window receiver readability REFUSAL**, Exp 60's live-G2 shape moved
to the receiver: after ingest and before the window, one loop-OFF submersion; `anticipatory_threat_need`
on the node B's live submerged reading completes into must be > 0.5 (strict, the consumer's floor) and
that node must be the transferred one (id equality after `id_map`); 0.0 → pair INCOMPLETE with the
cause named (`fear_dropped`, completion into a fresh node, attenuated value — each distinguishable in
the record); and (b) the **decision-provenance clause at the executed `escape_water` proposal**: the
`NAc_RECOMMEND` components for the winning tool show `drive > 0` with `causal == 0` and
`learned_bias == 0` (reasoning `drive:threat(1.00) →escape`); a surfacing won by any other component is
recorded as NON-FEAR, counts as a FAILURE of TRANSFERRED, and triggers the V4 channel audit before the
campaign continues. Then the DV itself is unconditional: one binary per receiver, every clean pair in
the denominator.

### SHOULD-FIX

**F2 — A NULL is uninterpretable as drafted: the behavioural DV is censored at a cap the mechanism's own
first-contact latency sits ~1 s under (V8), and every receiver pays the `flee` tie-break.** A slightly
slower bridge/loop wake on the day pushes a WORKING transfer past 4.335 s and reads as "did not
surface". This cannot produce a false positive, but it makes a null unreadable (mechanism vs timing).
*Change:* pre-register a co-primary **decision DV** — `escape_water` EXECUTED (executor spy, `success`
irrespective) before the cap — alongside Exp 60's behavioural DV (head-in-air by bridge truth, kept for
comparability). TRANSFERRED gates on the behavioural DV; a campaign whose decision DV passes while the
behavioural DV fails is classified INCOMPLETE-with-cause (actuation timing), never NULL. Report
latency-to-first-in-water-tick and latency-to-first-`escape_water`-call per receiver.

**F3 — n, margins, and the statistic are stated loosely.** "Exact permutation for each pairwise gate"
is not what the gates are (three are point-estimate margins, one is a one-arm rate), and at n = 12 the
margins are coarser than the floor's own interval (V7). *Change:* (i) unequal n — arms 2 and 3 cost a
2.5-min training each, arms 1 and 4 cost none (arm 4 reuses arm-2 donors), so run arms 1 and 4 at
n ≥ 24 and arms 2/3 at n = 12; that puts the 0/n Wilson upper bound at ≤ 0.14, below the 0.20 margins,
and makes BOTH-HALVES "< 0.10" mean "≤ 2 of 24" rather than "≤ 1 of 12"; (ii) state the gates as
point-estimate margins with 95% Wilson intervals REPORTED (Exp 56 house style) and the exact one-sided
permutation p < 0.05 as an additional gate on arm 2 > arm 1 and arm 2 > arm 3 only; (iii) re-derive or
at least re-justify 0.70: it was Exp 56's ε-greedy headroom constant ((1−ε)+ε/k ≈ 0.83 vs 0.70); with a
deterministic selector the expected arm-2 rate is ≈ 1.0 minus timing failures, so 0.70 is a tolerance
for F2's failure mode, and the prereg should say that instead of "carried".

**F4 — Specificity of the RECEIVED fear is not gated, and no arm controls the one alternative
positive that the bundle path itself could manufacture: the aligned EC merge putting B's shore reading
into the feared node.** Fear active on the shore would make B flee during the 1-s warm-up and surface
in water for a non-specific reason; the drafted mechanism DVs (need > 0.5 on the completed cluster)
would read PASS. Exp 60 gated `|shore fear| < 0.2·|water fear|` per seed; Exp 61 drops it.
*Change:* on the receiver, after ingest and before the window: `cluster_fear(B, live shore node) == 0`
and `cluster_fear(B, live water node) == −1.0`, ids distinct (the live cluster-distinct preflight
already produces both ids) — a verdict gate on every arm-2 pair, plus "no executor call during the
shore warm-up / roam". This makes the draft's proposed fifth arm (fear re-pointed to the SHORE node)
unnecessary for the claim; keep it optional, and note that a hand-edited bundle is the one
hand-composed element the design would otherwise contain.

**F5 — Export-before-probe is sufficient but the draft's verification of it is too narrow.** Exp 60
measured that EVERY executed post placement books ≥ 1 `flee` negative link and ≥ 1 `escape_water`
positive link; a loop-live probe that executed nothing leaves nothing (run 1). So the bundle-visible
proof that no post-probe happened is an EMPTY link map, and the draft asserts only "no positive
`escape_water` link, no `flee` link, no bias on the water cluster". *Change:* donor sanity asserts on
the SHIPPED nac payload `links == {}`, `event_outcome_welford == {}`, `cluster_reward_bias == {}`,
`reward_bias == {}`, `cluster_fear` has exactly one key (arm 2) / none (arm 3), and `percept_valences`
carries the `drive:oxygen` entry (proves the pain published in both donor kinds); the harness shape has
NO donor post-probe at all (training → G2 read → export → teardown), and the manifest `created_at` is
recorded beside the training-end timestamp. A donor failing any of these is an apparatus failure
(re-pair), never a filtered row.

**F6 — The dropped Exp 56 dither/L2 machinery needs a stated reason, and the replication unit needs
naming.** Exp 56 dithered because its floor was a selector artefact (deterministic argmax → effective
n ≈ 1); Exp 61's floor is structurally `None` and its mechanism is deterministic — Exp 60 accepted
30/30 vs 0/30 on that basis. That is a defensible choice, but the prereg must SAY that receivers are
not policy-stochastic replicates: n replicates over live timing, per-donor cluster identity and the
fold, not over choice. Otherwise the outcome reader will (rightly) ask why Exp 56's L2 gate vanished.
*Change:* one paragraph under §Dependent measure; keep interleaving by seed as the drift control.

**F7 — A surface with ZERO executor calls counts as a success.** `classify_placement` reads head-in-air
only. Donor and receiver share one bot serially; any held control state or teleport artefact that
lifts the bot would score as transferred fear. *Change:* first-contact success requires BOTH
`escape_water` executed inside the window AND head-in-air by bridge truth; a head-in-air with
`calls == []` is an apparatus refusal (named), in every arm. (This also closes the only way arm 1 or
arm 4 could show a non-floor rate without a substrate cause.)

**F8 — A wiring-side attenuation is a structural null that the drafted gates would misreport as NULL.**
Per V3 any received fear with |v| ≤ 0.5 is dead at the consumer; the draft gates `fear_rekeyed == 1`
but not the VALUE. *Change:* the ingest report (or a post-ingest dump read) must show the folded value
`== −1.0` on the re-keyed key, as a refusal gate; the src PR's bound is `[-1, 0]` with NO foreign cap
(state this explicitly next to `CAP_FOREIGN_CONFIDENCE`, which exists for links). This also answers
open question 2's "does −0.5 clear θ": no — it is discarded one function later.

### NIT

**F9 — "First contact" needs a definition that survives the preflights.** The live cluster-distinct
preflight, the actuation preflight and F1(a)/F4's readability read all submerge B before the window,
loop OFF. Define first contact as the first LOOP-LIVE submersion; record the loop-off submersions
(count, duration, no pain published — the bus subscriber already timestamps every publish).

**F10 — Say whether donors run Exp 60's PRE-probe.** The arm table implies training → export; Exp 60's
protocol has a pre-probe and a post-pre US-free check. The pre-probe leaves nothing in the NAc
(measured, V1) and costs ~1.5 min per donor; dropping it is fine if `cluster_fear == {}` is asserted
before training and the receiver window keeps Exp 60's DIRTY rule.

**F11 — Arm 4 cannot fail through the channel it was designed against.** Exp 56's dangling-half was a
tripwire for the state-blind LINK channel; Exp 61 donors have no links (V1), so arm 4 is a check that
`rekey_nac_state` drops fear honestly (`fear_dropped == shipped`, `fear_rekeyed == 0`) — worth keeping
(cheap, same donors), but the prereg should not present it as the falsifier that carries the claim.
F4's specificity gate is the falsifier that CAN fail here.

**F12 — Arm 3's donor choice is right; say why in one line.** ABLATED-trained dominates a never-submerged
A″ because (V1) the two bundle kinds differ in exactly ONE field, so arm 2 − arm 3 is a single-field
contrast that also controls the representation, Wire 2's percept valence and the bundle path. A″ would
control only "the path does nothing by itself", which arm 3 already implies. Redundant; do not add.

**F13 — Anti-vacuity: carry Exp 56 amendment 1's per-variant expectations** (receiver-unchanged and
empty-state collapse to floor; donor-re-keyed-alone RECORDED, expected to persist on a fresh receiver),
not just "must collapse to the floor".

## Verdict

**FIX-THEN-BUILD.** The claim is well-isolated in principle — after propose-only training the two donor
kinds differ in exactly one persisted field, every state-blind channel that could surface a receiver
requires an execution that never happens, and the floor is structural — so arm 2 vs arm 3 is the
cleanest single-field transfer contrast this repo has designed. But the primary gate as drafted cannot
fail for the reason the claim could be false (F1), a NULL would be unreadable against the cap (F2), the
one bundle-path route to a wrong-reason positive (shore completion) is ungated (F4), a zero-call surface
would count (F7), and the export-before-probe proof is under-asserted (F5). All are prereg changes, none
require a different apparatus; fold them, then build.

**Not verified:** the hivemind src changes themselves (the four sites are read as they exist today; the
`_receiver_scrub` re-use of `scrub_nac_state_for_bundle` and whether `ec_merge_aligned` inserts a
donor's nodes under identity ids on an EMPTY receiver EC are the wiring lens's); whether `cos` between
B's live dive-second-0 reading and A's trained water centroid clears 0.85 on the receiver's EC (assumed
from same body/geometry/placement — the F1(a) refusal is what makes this measured rather than assumed);
the bridge/bot state carried across the donor→receiver swap (environment lens); the 3-h campaign's
stability; any bio-faithfulness question (social discount, provenance marking, which failure modes
travel).

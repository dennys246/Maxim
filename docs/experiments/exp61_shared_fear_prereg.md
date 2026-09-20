# Exp 61 (FROZEN 2026-09-17 at v2.3 — four-lens review FOLDED, harness-reconciled, dry runs 1+2 DONE) — shared survival fear: a learned drowning-fear transfers between independent agents and drives the receiver's first loop-live submersion

> **STATUS: FROZEN 2026-09-17 (this PR; docs-only). What is frozen is v2.3 as it stands below — the
> claim, the arms (24 / 12 / 12 / 24), the receiver lifecycle, the DVs and gates, the stop rules, the
> harness constants in `exp61_run.FROZEN`, and the operator runbook at the end. Build step 4 is DONE:
> dry run 1 (2026-09-17) found the NAc's zero-key wart, fixed at the source (#746); dry run 2 (same day,
> after the fix, `dryrun2-2026-09-17`) ran every row of pair 200 clean — apparatus, both donors, the
> anti-vacuity kit, all four receivers — with the transferred receiver's first contact a SUCCESS on both
> DVs and the three controls at the structural floor (the record is under build step 4; the rows are
> committed as a diagnostic beside the data, never counted). No trial data are taken before this PR is
> ON MAIN; the campaign runs at ONE code hash from a clean checkout at or after it. History: v2.2 folded
> dry run 1; v2.3 restored the donor sanity to its v2.1 strength. The four-lens design review ran on v1 (2026-09-16; all four lenses
> FIX-THEN-BUILD; three DO-NOT-BUILDs, all folded below; reports preserved verbatim under
> `docs/experiments/rationale/exp61-shared-fear/`). The two decisions the fold left to the owner were
> taken 2026-09-16: **D1 = the 0.75 social discount at the ingest bound (Option A); D2 = arm sizes
> 24 / 12 / 12 / 24.** Build steps 1–3 SHIPPED (#742 red gate; #743 mechanism + harness, each two-lens
> reviewed); v2.1 reconciles this text with the harness AS BUILT (the two code reviews' design
> findings: the dangling arm's donor policy, apparatus checks once per pair, the settle guard,
> Fisher's exact test, the anti-vacuity row, the frozen constants the harness carries).**
> This is the 1.3 Phase-2 headline (`docs/plans/roadmap_1_3.md` §Phase 2) with its want re-pointed
> from "dark = danger" (Exp 58, BLOCKED at the instrument) to "water = drowning" (Exp 60, EARNED
> 2026-09-16): *agent A learns the hard way → exports its substrate → agent B ingests it → B leaves
> the water on its first loop-live submersion, never having felt the air-hunger pain.*

## The claim

A **learned, situation-keyed fear** — Wire-4 `cluster_fear` booked by `drive:oxygen` pain onto the
underwater world cluster, which Exp 60 showed is the behavioural CAUSE of anticipatory surfacing
(FEAR 30/30 vs ABLATED 0/30 with identical pain exposure) — **transfers between genuinely
independent agents through the shipped signed-bundle path and changes the receiver's first-contact
behaviour.** B has never experienced the air-hunger pain. A has, and learned. A's substrate, exported
and ingested into B through `maxim substrate export` / `maxim substrate ingest`, makes B execute
`escape_water` and clear the water before the pain would fire, on B's FIRST loop-live submersion.

Wording fixed by the review (environment S2): B is not "never underwater" — the receiver's
representation gate submerges it once, loop OFF, US-free (≈ 2 s, min oxygen recorded, zero pain
publishes asserted). B never experiences the US before the measurement, and the DV placement is its
first submersion with the loop live.

Same shape as Exp 56 (the 1.2 headline: a TAUGHT positive want transfers) with three differences that
make it a new claim: the want is **learned from the game's own pain**, not teacher-credited; it is
**negative** (fear — Exp 56 explicitly did not claim aversion transfer); and, unlike the R2 cluster
credit (a *messenger*, `docs/wiring/substrate-learning-channels.md`), this signal is the measured
behavioural cause on this body, so the transfer question has a real behavioural answer to give.

Independence is D44's: disjoint `agent_id`, separate `EntorhinalCortex` + `SensorEncoder`, disjoint
cluster ids, B on FRESH persistence with ZERO world nodes before ingest (asserted from its staged
`ec.json`). Same body (`minecraft_player`), same world (the Exp 60 water classroom), same declared
ranges and therefore the same geometry tag — asserted by the harness as tag-STRING equality between
the donor's world nodes and B's live shore encode (wiring SF-2: `strict_geometry` cannot refuse an
insertion into an empty receiver, so the harness must). H2 (#740, on main) makes a range
re-declaration a tag change; the campaign runs at ONE code hash (§Stop rules).

## The mechanism to build (enters `[engineering]`; own reviewed `src/` PR before any trial)

Today fear does NOT travel, by design (Wire-4 invariant (c), `docs/agents/bio-memory.md`): the
bundle scrub drops `cluster_fear`, ingest strips it from a hand-built bundle, and `nac_merge`
min-folds whatever remains. The v1 draft named four sites; the wiring lens (DNB-1) found the counters
had no carrier from the merge to any caller — D43's exact shape — and the real list is **nine sites,
moved together in one PR:**

1. `hivemind/bundle.py::scrub_nac_state_for_bundle` — stop popping `cluster_fear`; clamp each value to
   `[-1, 0]`; keep only triple keys (`aid\x1fcid\x1ffm`, exactly as `NAc.dump()` writes them) whose
   failure mode is in the Wire-4 allowlist.
2. `hivemind/ingest.py::_validate_nac_payload` — delete the strip; add `cluster_fear` to the
   bounded-field loop (`[-1, 0]`, triple keys, the `MAX_NODES_PER_SLICE` cap and `_check_key_shape`
   for free), apply the node-id charset check to the cluster part, and REFUSE (`IngestRefused`,
   duty V2) a failure mode outside the allowlist — refusal, not strip (the `inherent` precedent).
   The allowlist is read from `NACConfig.cluster_fear_failure_modes` — ONE source, never a second
   literal (bio-faithful SF-3).
3. `hivemind/merge.py::rekey_nac_state` — `cluster_fear` joins the re-keyed fields: cluster id through
   `id_map`, agent id rewritten to the receiver's (the read path `NAc.cluster_fear` filters on agent
   id — an un-rewritten key reads 0.0 silently; wiring SF-3); a fear whose donor cluster did not
   survive the aligned merge is **dropped, not faked**.
4. `hivemind/merge.py::substrate_merge` — count fear before/after the re-key beside the bias counts.
5. `hivemind/merge.py::SubstrateMergeResult` — `fear_rekeyed`, `fear_dropped`, `fear_below_floor`
   (`|v| ≤ NACConfig.cluster_fear_threshold` after the discount — a fear that arrives dead re-keys
   cleanly and never acts; bio-faithful SF-2), all defaulting to 0.
6. `hivemind/ingest.py::IngestReport` — the same three fields, filled by `ingest_bundle` and written
   into the journal entry (the durable surface the harness reads).
7. `hivemind/cli.py::_run_ingest` — one printed line beside `biases rekeyed`.
8. `hivemind/merge.py::prune_nac_cluster_biases` — prune `cluster_fear` on pruned ids, or an
   invalidate leaves fear dangling (wiring SF-4; corollary 4's shape).
9. `nac_merge` — unchanged (MIN fold, clamped `[-1, 0]`), plus a test that `nac_merge_many` preserves
   the pairwise min for fear.

**D1 — RECOMMENDED: the social discount.** A donor's −1.0 is the cap (ten saturating episodes). The
MIN fold hands B that number verbatim, bypassing `record_cluster_fear` — byte-identical to "B drowned
ten times itself", not to vicarious conditioning, which in every animal and human preparation is
real, drives avoidance without the observer ever contacting the US, and is reliably WEAKER than
direct conditioning (bio-faithful SF-1). Option A, recommended: `_validate_nac_payload` multiplies
admitted fear by a declared constant `FOREIGN_FEAR_DISCOUNT = 0.75` on the foreign path only
(`merge-nac` is trusted-local and stays verbatim), BEFORE the MIN fold. Against the real consumer:
need 0.75 > the strict 0.5 floor (1.5× margin), `escape_water` score 0.525 ≥ 0.3 — fires on first
contact; B's own later drowning deepens it through the normal write rule (−0.75 → −1.0); a re-export
hands C `0.75²` (chain attenuation with no provenance field); wall shelf-life ≈ 4.1 d vs 7 d. **Stated
honestly: this DV cannot tell 0.75 from 1.0** — binary, floor-gated, no competing score on a fresh B;
the discount is a representational commitment plus the cascade property, not something Exp 61
measures. Option B is verbatim transport re-titled "as if experienced"; the owner picks. What is NOT
adopted: a sub-floor "social prior confirmed by one own exposure" — real, expressible today, but a
different claim (acquisition speed after one own US) that fails the first-contact headline by
construction — named as follow-up **Exp 61b**.

**Read path: unchanged.** B's loop notes its active clusters; when B's live submerged reading
pattern-completes into the TRANSFERRED underwater node (in B's EC under the aligned id),
`NAc.anticipatory_threat_need` (> θ 0.5, strict at `recommend_action`) → `escape_water` through
`run_agentic_loop` at AUTONOMOUS. Nothing in the read path knows the fear was imported.

**Red gate for the composition (wiring SF-6), landed `xfail(strict=True)` BEFORE the mechanism PR and
un-marked IN it, never re-pointed:** two agents with their own encoder/EC/NAc; A encodes a water
reading, books fear to −1.0; REAL `compose_bundle` → REAL `ingest_bundle` (tmp journal,
`receiver_agent_id=B`) → B loads the report → B encodes the same reading →
`anticipatory_threat_need(B, …) > 0.5` and `recommend_action` picks the escape affordance. RED today at
the read (0.0: the scrub pops the field). Arms alongside: ABLATED donor → 0.0; dangling half
(`ec_substrate_nodes=None`) → `fear_dropped == shipped`, `fear_rekeyed == 0`, read 0.0; ingest
WITHOUT `receiver_agent_id` → 0.0 (proves the agent-id rewrite is load-bearing); anti-vacuity
`receiver_unchanged` / `empty_state` collapse (Exp 56's kit). The ONE existing pin flipped on purpose:
`test_cluster_fear.py::TestFearHivemindPosture::test_bundle_scrub_excludes_fear` becomes "fear ships
clamped, discounted and allowlisted; an out-of-allowlist mode and a positive value do not". The brief's
invariant (c) and its guard line are rewritten in the same commit as the flip. Dropped from v1: a
two-process key-stability test (the keys are `\x1f`-joined strings, nothing PYTHONHASHSEED can touch).
Docs in the same PR: `docs/plans/archive/oasis_ingestion_contract.md` V2 bounds (the user-facing
`hivemind_bundle_format.md` carries no field roster — corrected at v2.1), the brief's invariant (c).

Front-gate: no new mechanism, no new bus — the sibling field `cluster_reward_bias` already takes
every one of these paths; this extends them to `cluster_fear` under the same rules plus one bound and
one scalar.

## Why this can work where Exp 56 needed a teacher — and the traps, as verified

- **The representation half is proven on this body.** Exp 60 gate (ii): shore and submerged form
  distinct world clusters (cos 0.787); 26 submerged samples over the full oxygen range completed to
  ONE node (bio-faithful N-2), so B's first reading completing into A's prototype is well-supported —
  recorded as a DV (the cos margin), not assumed. `world` is frozen-centroid in code (`ECConfig` and
  the pinned hivemind default); the imported node's centroid is A's.
- **The credit half is proven causal.** Exp 60 ABLATED: same pain, same episodes, no fear → 0/30. Fresh
  agents' floor: **0/60 placements over 10 agents, 0 executor calls**, with a structural reason —
  `recommend_action` returns `None` when nothing scores (confounding V6). Run 1's 0/54 is NOT evidence
  (the loop never acted) and is not cited.
- **Trap 1 — the state-blind causal link.** After A's first surfaced placement `escape_water` carries a
  POSITIVE causal link (47–51 per Exp 60 post probe); a bundle exported after any probe hands B "escape
  is a good action" and B could surface from the LINK. Verified (confounding V-series): after
  propose-only training the two donor kinds differ in exactly ONE persisted field, `cluster_fear` —
  `propose_via_substrate` writes only `note_active_clusters`; links form only on an EXECUTED tool;
  `reward_bias` is node-keyed and cannot match a tool signature; Wire-2 `percept_valences` fire in both
  donor kinds and are not read by the substrate-primary path. So: **A is exported from files staged
  immediately after training, with no donor probe at all** (training → G2 read → stage → export →
  teardown), and export-before-probe is verified as an EMPTY link map on the shipped payload (F5).
- **Trap 2 — first contact only.** After B's first surface B has its own positive link; later
  placements are fear PLUS link. One binary per receiver; a second placement is reported, never gated.
- **Trap 3 (new, environment S1) — the donor's fear is booked in a phase no hub session covers.**
  Exp 60 trains with the loop stopped; `MemoryHub.on_session_end` on an already-closed hub saves
  nothing; a staged `nac.json` would carry fear 0 and every arm-2 pair would refuse at the ingest
  gate. The donor flow opens a hub session around training and stages via Exp 56's
  `close_and_stage_session`; donor sanity reads the STAGED file, never the object.
- **Trap 4 (new, wiring DNB-2) — the wet preflights.** Exp 60's preflights submerge the agent twice
  and can book B's OWN fear on the transferred node through the pain subscriber. Preflights are split
  by subject (§Receiver lifecycle).

## Arms

One donor per receiver in the arms that MEASURE the fear (arms 2 and 3: no donor reused across
receivers). Arm 4 re-composes a fear donor's staged nac-only (as Exp 56): pairs 1–12 re-use their
own arm-2 donor, pairs 13–24 the arm-2 donor of pair k−12 — donor identity is immaterial there
because the fear is DROPPED by construction (the row measures the drop accounting and the floor),
so the no-reuse rule is scoped to arms 2/3 and the reuse is recorded per row as ``donor_pair``
(harness review, architecture S3; the alternative — 12 fear donors trained only to be exported
nac-only — cost ≈ 30 min for nothing the arm measures). Donors follow the Exp 60 training protocol (propose-only yoked training on
the saturating `drive:oxygen` pain, K = 10 usable episodes, rescue at the training cap) inside a hub
session, and are exported from the staged files before any probe.

| arm | donor (A-phase) | receiver (B-phase) | purpose | n (D2) |
|---|---|---|---|---|
| 1 **isolated** | none | fresh B, no ingestion, one placement | floor (structurally `None`; Exp 60 fresh agents 0/60) | 24 |
| 2 **transferred-fear** | FEAR training (subscriber attached) → staged → exported; `cluster_fear` ships | fresh B + A's bundle via `maxim substrate ingest --receiver-agent-id B` (strict geometry) | the claim | 12 |
| 3 **cluster-not-fear** (want-not-file) | ABLATED training (identical submersions and pain; subscriber detached) → staged → exported | fresh B + A′'s bundle, same path | the arrival of the underwater CLUSTER (A′'s EC carries it) and of everything else training leaves is not the arrival of fear — the load-bearing control | 12 |
| 4 **dangling-half** | a fear donor's stage, nac-only (pairs 1–12: their own arm-2 donor; 13–24: pair k−12's) | fresh B + a re-compose of A's stage with `aut_ec.json` absent (the export's nac-only path) | fear keys without the representation buy nothing, LOUDLY: `fear_dropped == shipped`, `fear_rekeyed == 0` — the representation-half check (not the load-bearing falsifier: it cannot fail through the link channel, there are no links) | 24 |

A never-submerged donor is redundant with arm 3 (confounding); a fifth arm with fear re-pointed to
the SHORE node is unnecessary once receiver-side specificity is gated (F4) and would be the design's
only hand-composed element — dropped.

**Donor sanity — asserted on the STAGED `aut_nac.json` / `aut_ec.json`, the files the export reads;
a failure is an apparatus fault, re-pairs on a fresh seed, and is recorded, never filtered:**
`links == {}`, `event_outcome_welford == {}`, `cluster_reward_bias == {}`, `reward_bias == {}` (the
proof that no probe happened), with the two ways `reward_bias` can be non-empty told apart by name:
a NON-ZERO node bias can only come from a relief/success reaction credited before export (a probe
or an execution); a ZERO-valued key means the running NAc still stores the pain credit's clamp.
History: dry run 1 (2026-09-17) refused BOTH pair-200 donors on three zero-valued keys —
`temporal_credit.distribute` hands each eligible node a negative share of the pain and the
pre-fix `NAc.credit_node` clamped it at 0.0 but STORED the key. v2.2 loosened the check to "no
positive bias"; v2.3 records the source fix (`credit_node` now removes a bias that clamps to zero,
the meaning the decay prune already gave it; the offline smoke pins that propose-only training
stages `reward_bias == {}`) and restores the empty-set rule: donors are trained fresh in the
harness process, so a zero key on one can only be a stale `maxim` install or a regressed writer,
both refusals. The zero count still ships as `reward_bias_zero_nodes` per donor row; `percept_valences` carries the `drive:oxygen` entry (the pain
published, both donor kinds); arm-2 donor: ≥ 1 `cluster_fear` key (a jitter-split donor may carry
two — wiring SF-1), ALL under `drive:oxygen` (any other mode is a named refusal), ALL on world nodes
noted during its training episodes, ALL at exactly −1.0, NONE on the shore node; arm-3 donor: no
`cluster_fear`; the shipped fear COUNT stamped into `donor_meta.json`; every world node stamped with
ONE geometry tag (recorded); manifest `created_at` recorded beside the training-end timestamp.

## Receiver lifecycle (numbered; the wet preflights never run on B)

0. **Throwaway agent (fresh persistence, discarded), ONCE PER PAIR, before the pair's donors:** the
   live cluster-distinct check, the escape actuation check (through the bridge, never the executor;
   `t_surface ≤ 2.5 s`, and ZERO executor calls during it), gamerules, raw bridge roster, cadence
   ≤ 0.15 s, `is_raining == 0`, `nearest_player_dist == 64`. Written as its own `apparatus` row; a
   refused apparatus row skips the whole pair (nothing it measures is trustworthy; `--resume`
   retries it). In addition, EVERY rescue settle — donor, throwaway and receiver alike — re-checks
   `is_raining == 0` and `nearest_player_dist == 64` (the trial's settle guard; an ABSENT key
   refuses, never defaults to the passing value — environment S4).
1. **B pre-ingest:** `build_minecraft_aut` → no loop, no water → full close and stage; assert B's
   `ec.json` holds ZERO world nodes and B's `nac.json` no `cluster_fear`.
2. **Ingest** via the real CLI with `--receiver-agent-id` = B's agent id (the same id the reboot uses);
   read the JOURNAL entry; gate the counters (§Gates); assert every `cluster_fear` key in B's post-ingest
   `nac.json` starts with B's id; compare donor `ec.json` ids with B's post-ingest ids (donor ⊆ post,
   count unchanged — `id_map` identity on a fresh receiver asserted from disk, wiring SF-5); record both
   `saved_at`s (the fold keeps the later one and re-stamps B's decay clock; bio-faithful SF-4).
3. **B post-reboot, shore only:** 1 s warm-up + loop liveness ≥ 4 ticks / 3 s (the loop proven live
   before the one placement); `get_positive_outcomes(escape_water) == []` and `== []` for `flee`; no
   executor call during the warm-up/roam; B's live shore tag equals every transferred world node's tag.
4. **Representation + readability gate (one loop-OFF submersion, US-free, ≈ 2 s, min oxygen and
   zero pain publishes recorded):** B's submerged reading must complete into the TRANSFERRED node in
   arms 2/3 (a fresh id in arm 1); `anticipatory_threat_need(B, that node) > 0.5` strictly in arm 2
   and `== 0` elsewhere; `cluster_fear(B, live shore node) == 0` and ids distinct (receiver
   specificity, F4). Each miss is its OWN named refusal class ("did not complete into the imported
   node", "fear not readable", "fear on the shore"), measured BEFORE the one-shot placement is spent.
5. **First contact:** the first teleport into water B ever receives with the loop live — the DV. An
   executed `escape_water` with NO captured NAc_RECOMMEND proposal is a named REFUSAL (the sink did
   not deliver — an instrument inconsistency, never a mechanism null); the executed proposal is the
   first escape-best event that PASSED the gate (a sub-threshold escape-best event is never executed).
6. Rescue at the cap; one further placement recorded (fear + own link), never gated; teardown.
   Persistence dirs are DURABLE per pair (donor stage, bundles, receiver homes) — no `rmtree`.

## Dependent measures and gates

**Primary (both required for a success):** on B's first loop-live submersion, (a) the **decision DV** —
`escape_water` EXECUTED (executor spy) inside the US-free window (cap = measured pain edge min −
0.75 s, re-measured on campaign day), and (b) the **behavioural DV** — head in air by bridge truth
(`is_in_water` at eye height) before the cap. One binary per receiver. A head-in-air with
`calls == []` is an apparatus REFUSAL in every arm (F7; one bot serially — a held control or teleport
artefact must not score). A surface won by any score component other than the drive/fear component
counts AGAINST the claim: Exp 56's decision-provenance clause at the executed `escape_water` proposal
— drive component decisive, causal == 0, learned bias == 0 (confounding F1's replacement for the v1
"fear active at contact" filter, which conditioned the DV on the mechanism and would have excluded
exactly the placements that could falsify it). A campaign whose decision DV passes while the
behavioural DV fails is **INCOMPLETE-with-cause (actuation timing)**, never NULL (F2): Exp 60's
first-contact latencies sit 1.0–1.4 s under the cap and every receiver pays the `flee` tie-break.

**Censoring and timing (environment S3):** per placement record `t_flee_call`, `t_escape_call`,
`t_first_air`; "escape called before the cap, head not in air by the cap" is its own class; per-pair
refusal bounds on the stamped apparatus numbers (`t_surface ≤ 2.5 s`, cadence ≤ 0.15 s, liveness ≥ 4);
campaign-level drift refusal in the verdict (last-quartile median − first-quartile median of arm-2
first-placement latency or actuation `t_surface` > 0.5 s → INCOMPLETE).

**Mechanism DVs (recorded; gated where stated in §Receiver lifecycle):** the node B completed into and
the cos margin to it; `anticipatory_threat_need` at the gate; the ingest counters; the folded fear
VALUE on the re-keyed key (must equal the post-discount value exactly, `|v| > θ` with margin — a
foreign cap is a structural null, F8); B's positive-link count before and after; both `saved_at`s.

**Secondary (reported, never gated):** latency to surface and to first `escape_water` call; the `flee`
tie-break; B's second placement.

**Replication unit (F6):** receivers are not policy-stochastic replicates (the selector is
deterministic and the floor is structurally `None` — Exp 60 accepted 30/30 vs 0/30 on that basis);
n replicates over live timing, per-donor cluster identity and the fold. Exp 56's dither/L2 gate is
dropped for that reason, stated. Arms interleave by seed as the drift control.

| gate | rule |
|---|---|
| **TRANSFERRED** | arm-2 first-contact success rate ≥ 0.70 — a TOLERANCE for the timing failure mode (with a deterministic selector the expected rate is ≈ 1.0 minus timing), not Exp 56's ε-greedy constant |
| **ABOVE-FLOOR** | arm 2 − arm 1 ≥ 0.20, and Fisher's exact one-sided p < 0.05 on the receiver binaries |
| **CLUSTER-NOT-FEAR** | arm 2 − arm 3 ≥ 0.20, and Fisher's exact one-sided p < 0.05 |
| **BOTH-HALVES** | arm 4 − arm 1 < 0.10 one-sided, AND every arm-4 ingest shows `fear_rekeyed == 0`, `fear_dropped == shipped` |
| **SPECIFICITY** | every arm-2 pair passed lifecycle step 4 (shore fear 0, water fear at the post-discount value, ids distinct); refusals named and counted |
| **ANTI-VACUITY** | the kit runs over the FIRST clean arm-2 pair's staged files (the real aligned `substrate_merge` must make the receiver read the fear; the no-op variants receiver-unchanged and empty-state must read 0) and is written as a campaign row the verdict REQUIRES — absent → INCOMPLETE, failed → NULL (D62: a gate that cannot fail is not a gate) |

Gates are point-estimate margins with 95 % Wilson intervals REPORTED (house style). The exact test on
two BINARY arms is Fisher's exact one-sided test (the hypergeometric tail), which IS the exact
permutation test on binaries in closed form — Exp 60's enumeration over relabellings is exact too,
but 12 v 24 binaries is > 10⁹ relabellings (harness review). **D2 (decided): arm sizes** arms 1 and 4
at n = 24, arms 2 and 3 at n = 12: the 0/24 Wilson upper bound is 0.14, below the 0.20 margins, and
BOTH-HALVES "< 0.10" means ≤ 2 of 24 rather than ≤ 1 of 12 (confounding F3, V7). **Budget, restated
for the harness as built:** 24 apparatus checks (≈ 1 min each) + 24 trainings (12 fear + 12 ablated,
≈ 2.5–3 min each) + 72 receivers (≈ 60–90 s each) ≈ 2.9–3.4 h on big-mac-mini. Verdict ∈ {EARNED,
NULL, INCOMPLETE}: all six gates for EARNED; a failed BOTH-HALVES with the rest passing is NOT a
partial pass (Exp 56's rule); INCOMPLETE when any arm has fewer than n clean pairs, on campaign
drift, on rows (donor, apparatus or receiver) spanning two code hashes, on a missing kit row, or —
with the cause named — when the decision DV passes while the behavioural DV fails (actuation
timing); a campaign whose transfer surfaces were won by another score component is NULL with the
count named. A later CLEAN row supersedes an earlier REFUSED row for the same (arm, pair) — what
`--resume` writes — and the refusal is still named; two clean rows for one key are a duplicate.

**Frozen with the harness (`exp61_run.FROZEN`; the analyzer refuses drift):** pair seeds 200–223;
the dangling-donor offset 12; the 0.75 discount (asserted against the ingest constant at campaign
start); the fear cap −1.0; the read floor 0.5; `actuation_max_s` 2.5; `drift_max_s` 0.5 (last-quartile
median − first-quartile median of the apparatus `t_surface` and of arm-2's first-contact latency);
the settle guard `{is_raining: 0, nearest_player_dist: 64}`; and a LITERAL copy of every Exp 60
number the harness depends on (K = 10, 6 placements per Exp 60 probe, caps, margins, liveness,
cadence, the fingerprint), pinned equal to Exp 60's FROZEN by a unit test so a later Exp 60 edit
fails loudly instead of being inherited.

## Stop rules / refusals

Exp 60's list (gated records, pain edge, fingerprint, roster, gamerule, clusters not distinct,
actuation failure, positive link after preflight, damage during training, > 2 deaths, loop not
stopping, bridge stale, provenance/dirty tree) plus: donor staged after any loop (refuse); donor sanity
failure (re-pair); ingest journal missing, `fear_dropped > 0` or `fear_below_floor > 0` in arm 2
(refuse the pair); geometry tag strings differ between donor nodes and B's live shore encode (refuse);
`is_raining != 0` or `nearest_player_dist != 64` at any preflight or rescue settle (refuse the pair;
environment S4 — a spectator at 4 blocks costs the completion margin, rain breaks completion); any
second player on the server during the campaign (runbook: forbidden, or spectator ≥ 64 blocks);
`git pull` between the first and last row (forbidden; verdict refuses rows across two code hashes,
S6); `maxim substrate invalidate` on any Exp 61 home (forbidden). Durable `--workdir` with per-pair
subdirs, `--resume` keyed on (pair, arm), one subprocess per pair or a per-pair RSS line (S5 — ~120
AUT builds and ~144 bridge connects in one campaign is six times Exp 60's length); pair-level ids,
not per-arm run ids.

## What this experiment does NOT claim

- Nothing about scaling (N donors folded — Exp 57's shape; its own prereg if this lands).
- Nothing about generalization: A learns and B is probed in the SAME pool, same geometry, same spawn
  placement; "fears water anywhere" is not claimed.
- **Nothing about extinction, and the limit is a mechanism gap, stated:** there is no positive writer
  on `_cluster_fear`, no tick decay, wall decay only on `load()`; a receiver that never drowns holds
  the imported fear unchanged for the session and it wall-decays at the 7-day class (−0.75 crosses the
  floor at ≈ 4.1 d). Safe exposure writes a positive `escape_water` link, never fear-down. Vicarious
  extinction / social safety signalling is unexpressed ("fear only in v1").
- Nothing about the discount's magnitude (D1): the DV cannot distinguish 0.75 from 1.0 here.
- Nothing about the innate/learned decomposition beyond Exp 60's; nothing about hardware, other
  bodies, the LLM path, or the hive-side promotion of the exchange (export + local ingest only).
- Nothing about POSITIVE-want transfer (Exp 56) or the mixed case (fear + bias on one cluster).
- Not this experiment: **Exp 61b** (sub-floor social prior + one own US → one-shot acquisition).

## Apparatus reuse and the harness (wiring SF-7)

Exp 60's classroom, anchor, apparatus check, gate record and geometry probe are reused unchanged
(the gate-(ii) record's code hash predates H2; the harness compares the fingerprint's ranges, so it
stands). From `exp60_run.py`, the pure module-level functions lift as-is (`classify_placement`,
`p_surface`, `exact_permutation_p`, `select_run`, `compute_verdict`, `fingerprint_drift`, the
telemetry readers, `FROZEN`); the preflights, rescue/submerge primitives, `_loop_window`, probe,
training loop, live G2, pain subscriber and executor spy are closures over `_run`'s state (≈ 700
lines) and are lifted into ONE seed-context class (`WaterTrial`) that both harnesses use — never a
second `_run`. Obligations: `test_exp60_run.py`'s pins keep their imports (re-exports); Exp 60 is
EARNED on its data, so `exp60_run.py verdict` is run before and after the extraction and the JSON
diffed byte-identical. Provenance: the harness calls the CLI in-process (Exp 56's pattern), so the
sanctioned guard is `in_process_code_provenance` + `evidence_out_paths_or_exit`, as Exp 60 does.
Exp 56's `close_and_stage_session`, `export_bundle` (incl. `dangling=True`), `ingest_bundle_into`
(journal read) and the `--assert-noop-fails` kit are reused through the real CLI.

## Build order (after the owner's D1/D2 decisions)

1. **DONE (#742):** red gate `tests/unit/test_exp61_fear_transport.py` (`xfail(strict=True)`, RED at
   the receiver read 0.0 and the missing counters).
2. **DONE (#743, commit 1):** fear transport `src/` — the nine sites + discount + `fear_below_floor` +
   the deliberate flip + brief invariant (c) rewritten + contract V2 bounds; two-lens review folded
   (incl. the invalidate TOMBSTONE recording pruned fear).
3. **DONE (#743, commit 2):** harness — `water_trial.WaterTrial` lifted from `exp60_run._run` (the Exp
   60 verdict byte-identical before/after, now a mechanical test), `exp61_run.py` `run`/`verdict`,
   `scripted_water.py` + the offline smoke that proves the shared class ticks, acts through the bridge
   only, executes the escape under fear and PERSISTS fear at the staging close; two-lens review folded
   (the loop's own session pair closes the hub session — every donor would have staged fear 0 without
   the re-open; the export needs a `body:`-rooted spec; Fisher's denominator).
4. One-pair dry run of every arm on big-mac-mini (the plumbing pilot: counters, tags, timing, the
   hub-session persistence on the LIVE path, the pain edge re-measured), recorded as a diagnostic,
   not data. Any change it forces goes into v2.2 before the freeze. **Dry run 1 (2026-09-17, pair
   200): apparatus passed; both donors trained 10/10 usable episodes and were REFUSED by donor
   sanity on `reward_bias` (three zero-valued node keys — the pain credit's clamp, see donor
   sanity); the three donor-fed receiver arms therefore did not run; the isolated receiver ran and
   read the structural floor (censored, no proposal, no call). Fold = v2.2 (the sanity loosened to
   "no POSITIVE node bias"; the offline smoke runs the donor sequence end to end), then v2.3: the
   zero keys were the NAc's wart — `credit_node` stored a bias it had clamped to 0.0 — fixed at the
   source (a zero bias is removed; the smoke pins `reward_bias == {}` after training), and the sanity
   check restored to `reward_bias == {}` with the zero case named as a stale-install/regression
   refusal. Dry run 2 (2026-09-17, after #746, campaign id `dryrun2-2026-09-17`, pair 200): NO
   refused row — apparatus, donor fear, donor ablated, the anti-vacuity kit and all four receivers
   clean at ONE code hash (`cf6d1cc0`, the post-#746 main, tree clean). Apparatus: live clusters
   distinct, escape actuation 1.52 s through the bridge with zero executor calls, bridge cadence
   0.101 s, liveness 6 ticks. Donors: 10/10 usable episodes each (20 oxygen pain signals, 0 health,
   0 deaths); the fear donor staged `cluster_fear` −1.0 on its water node and 0.0 on the shore, the
   ablated donor staged none; both sanities PASSED with `reward_bias == {}` (the live proof of #746).
   Kit: the aligned merge read 1.0, the receiver-unchanged and empty-state variants 0.0. TRANSFERRED
   ingest: `fear_rekeyed` 1, dropped 0, below floor 0, discount 0.75, donor nodes 3; first contact
   SUCCESS (decision DV, behavioural DV and drive-decisive provenance all true; `flee` call at
   1.03 s, `escape_water` call at 1.78 s, first air at 3.15 s; the receiver's FIRST proposal under
   the foreign fear was `flee`, which fails in water — the pathfinder is dead there — and its
   second was `escape_water`, executed; the campaign's rows are read with that pattern in mind: a
   flee-first receiver spends part of the window before the escape fires). ISOLATED,
   CLUSTER-NOT-FEAR and DANGLING all censored with no proposal and no call (arm 3 shipped the
   percept valence but no fear; arm 4 shipped nac-only: `fear_dropped` 1, donor EC nodes 0). The
   analyzer read INCOMPLETE by count alone (1 pair per arm), the kit row PASSED, specificity and
   both-halves PASSED, and the two Fisher gates read p = 0.5 as one pair must. Rows:
   `docs/experiments/data/exp61_dryrun2_2026-09-17.jsonl` (diagnostic; the verdict never reads it).
   HONEST LIMIT: the pain edge was NOT re-measured live on either dry run — the harness derives the
   probe cap (4.33 s) and the train cap (15.07 s) from Exp 60's apparatus record measured
   2026-09-15 on this same server and world (`exp60_water_apparatus.json`, `t_pain_edge` minimum
   5.08 s, margin 0.75 s). The belt that stands in for a re-measure: a pain event INSIDE a
   receiver's window is the dirty class, a named refusal, so a drifted edge shows up as refused
   rows, never as false successes; the campaign's drift check covers the actuation side.**
5. **Freeze = THIS PR (docs-only, v2.3 → FROZEN).** Then the campaign from a clean main at ONE code
   hash (`git pull` forbidden between the first and last row); merge-commit data PR; §Outcome from
   the verdict, never before. The runbook follows.

## Operator runbook (the frozen protocol, executed from a clean main checkout at or after the freeze)

1. big-mac-mini: `git checkout main && git pull` (the checkout must contain this FROZEN prereg);
   RESTART the bridge with `--state_interval_ms=100` and confirm its `spawn state:` line carries
   `is_in_water` (a long-running bridge serves the OLD roster after a pull). No second player on the
   server for the whole campaign, or spectator ≥ 64 blocks; no rain (the settle guard refuses both).
2. `export PYTHONPATH="$PWD/src"` on its own line, then the campaign, ONE invocation, all four arms,
   ≈ 2.9–3.4 h (24 apparatus checks + 24 trainings + 72 receivers + the kit):
   `python scripts/survival_world/exp61_run.py run --campaign-id exp61-campaign-1 --workdir ~/exp61_campaign --rcon-password '<pw>' --username maxim --write-experiment-results`
   The output is `docs/experiments/data/exp61_pairs.jsonl`; the tree must be CLEAN (no
   `--allow-dirty`); the harness refuses a `maxim` that is not this checkout's. Each row prints its
   preflights, ingest report and first contact; a refused row prints its cause and is stamped, never
   dropped. If the run stops (Ctrl-C, a crash), resume the SAME campaign id at the SAME code hash:
   `... run --resume --campaign-id exp61-campaign-1 --workdir ~/exp61_campaign ...` — clean rows are
   skipped, refused rows re-run and supersede. A `git pull` between rows ends the campaign: rows at
   two hashes read INCOMPLETE.
3. `python scripts/survival_world/exp61_run.py verdict --data docs/experiments/data/exp61_pairs.jsonl --json docs/experiments/data/exp61_verdict.json --campaign-id exp61-campaign-1 --write-experiment-results`
4. One MERGE-COMMIT data PR with `exp61_pairs.jsonl` + `exp61_verdict.json`; then §Outcome is
   written from the verdict (EARNED / NULL / INCOMPLETE), never before; then the graduation ledger
   row and the CHANGELOG line, from the same verdict.

## Outcome (2026-09-17): EARNED — a learned drowning-fear transfers between independent agents through the shipped bundle path and drives the receiver's first loop-live submersion

**Verdict record:** `docs/experiments/data/exp61_verdict.json` (data PR #748, merge commit), computed by
`exp61_run.py verdict --campaign-id exp61-campaign-1` on big-mac-mini at main `4e25b475` (the freeze
merge, #747) on a clean tree from `docs/experiments/data/exp61_pairs.jsonl`. The campaign ran at that
ONE hash in one invocation, 17:05–18:11 UTC (1.10 h; the frozen prereg landed on main at 17:03),
121 rows, **zero refusals**: 24 apparatus, 12 fear + 12 ablated donors, 1 anti-vacuity kit row, 72
receivers. Every row carries `working_tree_dirty_src_scripts: false`.

**Every gate PASS (all six, as frozen in §Dependent measures and gates):**

| Gate | Frozen threshold | Measured |
|---|---|---|
| TRANSFERRED | arm-2 first-contact success ≥ 0.70 | **12/12 = 1.0** (Wilson 95 % [0.758, 1.0]); decision DV 12/12, behavioural DV 12/12, drive-decisive on every row, 0 dirty windows |
| ABOVE-FLOOR | arm 2 − arm 1 ≥ 0.20, Fisher one-sided p < 0.05 | **1.0 − 0.0 = 1.0**; **p = 8.0 × 10⁻¹⁰** (12/12 v 0/24) |
| CLUSTER-NOT-FEAR | arm 2 − arm 3 ≥ 0.20, Fisher one-sided p < 0.05 | **1.0 − 0.0 = 1.0**; **p = 3.7 × 10⁻⁷** (12/12 v 0/12) |
| BOTH-HALVES | arm 4 − arm 1 < 0.10; every arm-4 ingest `fear_rekeyed == 0`, `fear_dropped == shipped` | **0/24 v 0/24**; 24/24 ingests rekeyed 0, dropped 1 (= shipped), donor EC nodes 0 |
| SPECIFICITY | every arm-2 pair passes lifecycle step 4 (loop OFF) | **12/12**: shore fear 0.0, water fear **−0.75** (= −1.0 × the 0.75 discount), need 0.75 > θ 0.5, ids distinct, `pain_publishes` 0 at the gate |
| ANTI-VACUITY | the kit row present and passing | **present**: aligned merge read 1.0 (`fear_rekeyed` 1); receiver-unchanged 0.0; empty-state 0.0 |

**Apparatus, once per pair (24/24 passed):** live shore and water clusters distinct on every pair;
escape actuation through the bridge with zero executor calls, `t_surface` min 1.20 s, median 1.46 s,
max 1.57 s (bound 2.5 s); bridge cadence 0.101–0.102 s (bound 0.15 s); gamerules and the settle
guard (`is_raining == 0`, `nearest_player_dist == 64`) at every rescue of every row.

**Donors (24/24 clean, yoked by construction and measured equal):** every donor of both kinds ran
10/10 usable episodes in 10 attempts, 0 health pain, 0 deaths; oxygen pain signals 20 on 23 donors
and 19 on one (a fear donor). Fear donors staged `cluster_fear` **−1.0** on the water node and 0.0 on
the shore, one key each, on a node their training noted; ablated donors staged none. Every staged
sanity passed with `links == {}`, `event_outcome_welford == {}`, `cluster_reward_bias == {}` and
`reward_bias == {}` (the live proof of #746 on all 24). Loop liveness 6 ticks (one donor 5, bound 4).

**Transport (the mechanism read, arm 2, 12/12):** ingest `fear_rekeyed` 1, `fear_dropped` 0,
`fear_below_floor` 0, `fear_discount` 0.75, donor EC nodes 3, on every pair; the loop-OFF
representation gate then read the SAME node id as the transferred node with `water_fear`
−0.7499997 and `anticipatory_threat_need` 0.7499997 — the discounted value, exactly, before any
loop ran. Arm 3 (ablated donor): `fear_rekeyed` 0 on all 12, the `drive:oxygen` percept valence
(−1.0) shipped and the cluster geometry shipped, and the receiver made ZERO executor calls in
12/12 windows — the cluster's presence is not the fear. Arm 4 (nac-only): every fear entry
dropped at ingest for want of its world node (24/24), zero calls in 24/24 windows.

**First contact (the DVs).** Transferred receivers: `escape_water` call at min 1.38 s, median
1.72 s, max 1.86 s after the teleport; first air at min 2.70 s, median 3.14 s, max 3.53 s — every
one inside the 4.34 s US-free window, a median ≈ 1.9 s before the 5.09 s pain edge. The predicted
tie-break showed on **12/12** receivers: one `flee` call first (fails fast — the pathfinder is dead
in water), then `escape_water` (`flee_calls` 1, `escape_calls` 1 on every row; 24 calls total across
the arm). The three control arms made **zero** executor calls in 60/60 windows (all censored at
the cap, none dirty). The recorded second placement surfaced 12/12 at median 1.73 s with no `flee`
(the receiver's own positive `escape_water` link, 3 per receiver after first contact — recorded,
never a DV).

**Budget and process.** 1.10 h for the whole campaign against the 2.9–3.4 h estimate (the
donors dominate; receivers ran ≈ 20 s each). Harness RSS 42 → 148 MB over 121 rows. No `--resume`
was needed. Dangling pairs 212–223 reused the arm-2 donor of pair k−12 as frozen (`donor_pair`
recorded on every row).

**Honest limits (recorded, as frozen):** the pain edge was NOT re-measured live on campaign day —
`pain_edge_min_s` 5.085 and both caps are read from Exp 60's apparatus record (same server and
world, 2026-09-15); the dirty class (pain inside a receiver's window) is a named refusal and fired
on 0/72 windows. The discount's magnitude is not distinguished by this DV (0.75 lands the need at
0.75, above θ; 1.0 would too). Nothing here about extinction, scaling (N donors), generalization to
another pool, positive-want transfer, or the hive-side promotion of the exchange — see §What this
experiment does NOT claim, unchanged.

**What this earns:** the 1.3 Phase-2 headline (`docs/plans/roadmap_1_3.md` §Phase 2) — a
survival want learned the hard way by one agent, exported through the shipped signed-bundle path,
lands on an independent agent and drives its FIRST loop-live submersion, never having felt the
air-hunger pain. Ledger row: `docs/plans/behavioral_graduation_candidates.md` (Tier 1, beside Exp 60).

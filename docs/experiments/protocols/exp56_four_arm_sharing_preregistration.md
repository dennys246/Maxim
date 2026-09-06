# Exp 56 — The four-arm sharing benchmark: a taught want transfers between independent agents (pre-registration)

**Status:** PRE-REGISTERED 2026-09-05, frozen at the merge commit of this file, before
any harness run. The 1.2 headline claim
([roadmap](../../plans/roadmap_1_1_to_1_3.md) §"The 1.2 benchmark"; design authority
[minecraft_benchmark.md](../../plans/minecraft_benchmark.md) §"The 1.2 benchmark — four
arms"). The four gate constants and the arm set below are carried VERBATIM from those
frozen documents; this pre-registration operationalizes them and freezes everything
they left open. A two-lens review round (design/methodology + code-accuracy, both
reports in the round record) ran BEFORE this freeze; its two cross-confirmed critical
findings — the causal-link transfer channel and the deterministic-selector floor — are
designed against below (§Links, §Selector), not discovered at campaign price.
**Lineage:** Exp 42/45 (cross-**session**) → Exp 52 (the want is learned) → Exp 53/53b
(cross-**context**: the want reads out on a physical body) → this (cross-**agent**:
*someone else's* substrate drives an agent that never learned it). The claim ladder is
[oasis_case_study_taught_orient.md](../../plans/oasis_case_study_taught_orient.md)'s.
**Merge semantics under test are the REAL 1.2 semantics** (roadmap sequencing decision
1, the reason this prereg waited for PR #637): ingestion runs through
`maxim substrate ingest` — the V1–V10 adapter + `substrate_merge` (aligned re-key,
tighten-only clamp) — never bare `nac_merge`.
**Owner intent (recorded so the claim cannot drift):** the claim is that agent A's
LEARNED REPRESENTATION (the world-cluster-keyed operant bias plus the EC nodes it keys
on, exported as a bundle) changes agent B's FIRST-CONTACT behaviour, where A and B are
genuinely independent agents — different `agent_id`, separately constructed
`EntorhinalCortex` + `SensorEncoder`, cluster ids disjoint by construction (D44's
definition; the two shipped federation results pass only because every infant shared
one agent id and one encoder). Nothing here claims scaling (the dose–response ladder
is its own later pre-registration, §Not-claimed), nothing claims hardware (the
two-Reachy replication is its own pre-registration riding Exp 54's body), and nothing
claims aversion transfer (that is Exp 55, 1.3-line).

---

## The question

B has never experienced a contingency. A has: at one specific world situation, one
specific action is rewarded by a TEACHER (a taught want — the Exp 52 mechanism
transplanted from the audio channel to the world channel). A's substrate is exported
as a bundle and ingested into B through the real 1.2 path. At B's FIRST contact with
the situation, does B choose A's taught action?

Four things must be true for the answer to be "the representation transferred":

1. B-with-A's-bundle chooses the taught action at first contact (**TRANSFERRED**);
2. B alone does not (**FLOOR** — the isolated arm);
3. a bundle from an agent that experienced the same world and the same action schedule
   but was never credited does not move B (**WANT-NOT-FILE** — the arrival of a bundle
   is not the arrival of a want);
4. A's `nac.json` without its `ec.json` does not move B (**BOTH-HALVES** — the D43
   falsifier: bias keys without the representation they key on must buy nothing, and
   now do so LOUDLY as dropped biases rather than silently as zeros).

## The teaching mechanism (named precisely — the review round's first critical)

The A-phase rides the SHIPPED operant path and nothing else; the pre-review draft
assumed self-caused drive-relief credit could key a world cluster, which the code
refutes (reward credit is interoception-only by decided design —
[coding_habits_oasis.md](../../plans/coding_habits_oasis.md) §5, "not touched"; the
exteroceptive extension is pain-only, 1.3). The mechanism that DOES key an
exteroceptive cluster with positive operant credit is Exp 52's:

- the donor runs substrate-primary with `MAXIM_OPERANT_ONLY_CREDIT=1`; each executed
  action stashes `(cluster_id, tool_signature)` via `set_pending_operant_action`,
  where the cluster is the WORLD cluster (the first-non-interoception fallback in
  `runtime/tool_dispatch.py` under the flag — the same seam Exp 52 used for audio);
- a harness-side TEACHER (the `reactive_mother_tick` precedent, living in the Exp 56
  harness) watches the world state: when the donor's pending action is the target
  affordance AND the contingency situation is active, the teacher delivers the reward
  — a feed that relieves the donor's interoceptive drive `d1` — and calls
  `NAc.credit_operant_reward` with the Exp 52 relief-signed value (`credit=relief`:
  zero relief mints nothing, which is what makes the satiated arm a mechanism check);
- the credit therefore lands on `(agent, WORLD-cluster, tool:<target>)` — the
  situation-keyed, transferable form.

This is externally-delivered contingent reward — exactly `credit_operant_reward`'s
contract (the caregiver pattern), not self-caused relief. S1 consequence: the teacher
is harness assembly over existing sanctioned mechanisms; if ANY `src/` mechanism
change proves necessary, it is a structural amendment + its own reviewed PR before the
campaign, not a quiet harness commit.

## The link channel (designed against — the review round's second critical, proven by execution)

Causal links (`nac.observe` on every executed tool — the flag gates only cluster
credit) ship in every bundle, merge independent of EC, and feed
`recommend_action`'s primary component with no agent filter and no situation
conditioning: an executed trace showed a nac-only ingest driving the receiver to the
donor's most-executed tool at 0.9 confidence with NO clusters at all. Left unmodeled,
that channel moves arm 4 exactly as it moves arm 2 and the falsifier fails for a
reason readable in advance. Three design elements neutralize it:

1. **Donors act on a SCRIPTED BALANCED SCHEDULE, not on their own policy.** During
   the A-phase every donor (taught and satiated alike) executes the same seeded,
   exposure-balanced action schedule — each roster affordance equally often, every
   scheduled action mechanically executable at its step (a world-script obligation).
   Learning is carried entirely by the teacher's contingent credit; the donor's
   OWN behaviour never concentrates. Consequence: the link profile is
   near-uniform across affordances and IDENTICAL in expectation across taught,
   satiated, and (by reuse) dangling donors — link transfer is preference-neutral by
   construction. (The claim is about B's readout, not A's behaviour; a
   passively-scheduled donor is the Exp 46 scripted-mother lineage.)
2. **Donor sanity asserts the balance held** (S3): per donor, the max−min spread of
   per-affordance positive-link confidence in the shipped `nac.json` ≤ **0.05**, and
   per-affordance link counts within ±1 of each other. A donor failing the balance
   check is an apparatus failure — its pair is re-run with a fresh seed, recorded.
3. **A mechanism assertion rides the TRANSFERRED gate**: an arm-2 first-contact
   success counts toward the gate only if the decision-provenance record
   (`NAc_RECOMMEND` score components) shows the LEARNED-BIAS component decisive for
   the winning tool (the situation-keyed channel, which only fires when the current
   world cluster matches). First contacts won by any other component are recorded and
   reported per arm — if arm 2 "passes" on a non-bias component, the campaign has
   measured something else, and the analyzer says so instead of the gate passing.

Surfaces that remain live under `MAXIM_OPERANT_ONLY_CREDIT=1`, stated so the flag is
not over-read: causal links, Welford outcome stats, and goal credit all still record
from mechanical success (goal biases are dropped by the bundle scrub; links and
Welford ship). The balance schedule + assertions above are the containment for the
one that reaches action selection.

## Arms (frozen)

One donor per receiver, paired by seed — no bundle is reused across receivers, so each
transfer measurement is independent (no pseudo-replication on a single lucky donor).

| arm | donor (A-phase) | receiver (B-phase) | purpose |
|---|---|---|---|
| 1 **isolated** | none | fresh B, no ingestion | floor |
| 2 **merged-taught** | balanced schedule + teacher credits contingent target actions (drive drifts; feed relieves it → relief-signed operant credit on the world cluster) | fresh B + A's bundle via `maxim substrate ingest` | the claim |
| 3 **merged-satiated** | identical schedule + identical teacher feeds, drive held at zero (satiated body variant, visible in provenance) → zero credits by mechanism | fresh B + A′'s bundle, same ingest path | want-not-file control |
| 4 **dangling-half** | the SAME taught donors as arm 2 | fresh B + a re-compose of A's session with `aut_ec.json` absent (the export's documented nac-only path), same ingest path | the falsifier |

**Donor sanity (apparatus checks, not DVs, asserted per donor before its bundle
ships; any failure re-runs that pair on a fresh seed and is recorded):** a taught
donor's dump carries ≥ 1 `cluster_reward_bias` entry keyed on a world-modality
cluster with bias ≥ **+0.4** (above the frozen `min_confidence` 0.3 with margin, and
above L1's 0.11 argmax floor — a legal donor must be READABLE by the selector, not
merely nonzero); a satiated donor's dump carries zero cluster credits (the Exp 52
mechanism-sanity form); BOTH donor kinds pass the link-balance check (§Links); EC
world nodes are geometry-stamped (the arms run the adapter's STRICT path —
`--allow-unstamped-geometry` is forbidden in every arm; unstamped donors are an
apparatus failure, never an override); no donor dump carries `inherent_bias_keys`.
Arm 4's ingest report must show `biases_dropped ==` the donor's shipped bias count
and `biases_rekeyed == 0` — the honest indicator D43 shipped; if arm 4's biases
silently LAND, the apparatus is not testing what it claims.

## Selector, dither, and the measured floor (the review round's L2 fold)

`recommend_action` is a deterministic argmax and, on a fresh agent with no positive
scores, returns None by contract — an undithered probe would give every isolated
receiver the SAME first choice (effective n ≈ 1, the L2 phase-locking signature this
repo has paid for twice). Frozen selector regime, identical in every arm:

- **B-probe selection = seeded ε-greedy over the frozen roster** (harness-level, the
  scripted-lineage precedent): with probability **ε = 0.2** choose uniformly at
  random; otherwise take `recommend_action`'s choice, falling back to uniform when it
  returns None. Per-pair RNG stream, seeded from the pair seed.
- **Substrate knobs frozen:** `min_confidence = 0.3`;
  `substrate_explore_bonus_weight = 0` at the probe (the readout is pure learned
  signal + ε; the explore-FIRST hard gate must not pre-empt the first contact).
- **Opaque-name assignment is a per-pair random permutation** (recorded in
  provenance) so the deterministic name tiebreak cannot couple the floor to the
  alphabet (review finding I1).
- **Per-pair variation, stated:** the ε stream, the name permutation, and the
  contingency's placement among the world script's frozen candidate slots vary per
  pair seed — identically across arms (the same B-probe script per pair seed is used
  in all four arms).
- **Expected floor arithmetic, stated:** with roster size k (frozen with the body;
  design target k = 8) the isolated floor sits near 1/k ≈ 0.125 (ε and the None
  fallback both draw uniform); a bias-decisive taught arm sits near
  (1−ε) + ε/k ≈ 0.83 — headroom above the 0.70 bar without saturating it. The
  isolated arm's MEASURED rate is the floor the gates difference against; "1/k" is
  the design expectation, not the definition.
- **L2 apparatus gate (frozen):** if any arm's first-contact outcomes are
  seed-invariant (the same choice in ≥ 90% of a probe arm's receivers with per-seed
  provenance showing identical trajectories), no science verdict is issued — the
  dither did not reach the agent; apparatus fix + re-run, recorded as an L2 amendment
  (the Exp 52 precedent, made a gate here because a first-contact DV is maximally
  L2-exposed).

## Dependent measure (frozen)

**Primary:** B's **first-contact action choice** — at the first tick at which the
world script presents the contingency situation (an ARM-INDEPENDENT trigger: defined
by the script, not by cluster completion — the isolated arm holds no transferred
cluster to complete into), does the frozen selector (§Selector) choose the taught
affordance? Read at the REAL consumer — the action the loop actually proposes, from
the decision-provenance record — never from dict contents (D44's rule). One binary
measurement per receiver; the per-arm statistic is the first-contact success rate
over n receivers. The world script presents the contingency within the first **10**
agent decisions; the probe ends **5** decisions after first contact (frozen — the
pre-contact segment bounds B's own link accrual identically across arms).

**Secondary (reported, never gated):** approach latency (decisions from probe start
to the first taught-affordance selection, censored at probe end); `learned_margin`
at first contact (L1 context); the winning score component per first contact
(§Links assertion 3); completion-into-transferred-cluster at first contact on arms
2–3 (apparatus observation).

## Pre-registered gates (constants carried verbatim from the frozen design)

| gate | rule |
|---|---|
| **TRANSFERRED** | merged-taught first-contact rate ≥ **0.70**, counting only bias-decisive successes (§Links assertion 3) |
| **ABOVE-FLOOR** | merged-taught − isolated ≥ **0.20** |
| **WANT-NOT-FILE** | merged-taught − merged-satiated ≥ **0.20** |
| **BOTH-HALVES** (falsifier) | dangling-half − isolated < **0.10**, one-sided (the falsifier's danger is transfer WITHOUT the representation; a below-floor excursion is noise, not counter-evidence). At the design floor (~0.125), the SE of the difference of two n = 50 rates is ≈ 0.066, so a true-null false-failure runs ≈ 7% one-sided — accepted and stated, rather than hidden in a two-sided band justified at the wrong operating point (review finding I3) |
| **ANTI-VACUITY** (apparatus; no verdict without it) | the analyzer's `--assert-noop-fails` re-runs one arm-2 pair per campaign against `substrate_merge` monkeypatched no-op variants, with PER-VARIANT expectations (amendment 1): *receiver-unchanged* and *empty-state* MUST collapse toward the floor; *donor-re-keyed-alone* is RECORDED, expected to persist on a fresh receiver (D62's kit: a gate that cannot fail is not a gate) |

The claim requires ALL of TRANSFERRED + ABOVE-FLOOR + WANT-NOT-FILE + BOTH-HALVES.
A failed BOTH-HALVES with the other three passing is not a partial pass — it means the
effect does not need the representation, which contradicts the claim's mechanism, and
the verdict is NOT-EARNED with the finding recorded.

**Power:** n ≥ **50** receivers per arm (frozen design), each an independent
seed-paired donor/receiver. Report per-arm rates with 95% Wilson intervals; gates are
evaluated on point estimates against the frozen margins (house style).

## Phase 0 — instrument checks (gate the campaign; no Phase-0 data are science data)

Run on the frozen body + world script before any arm, recorded and committed like any
gated record. All five must pass or the campaign does not start:

1. **L11/A4 discriminability**: over ≥ 20 scripted probe onsets, the contingency
   situation resolves to a DIFFERENT world cluster than rest/ambient situations
   (separation ≥ 0.70 on this two-situation contrast) and repeated probes re-complete
   into the SAME cluster (stability ≥ 0.70). This is the per-design analog of the L11
   re-measure metric on the situations THIS experiment must tell apart — the live
   16-sensor measurement (`l11_remeasure`: A4 separation 0.0566 over all event kinds)
   says a full-width body is nearly situation-blind, which is why the benchmark body
   budgets its world channel (§Apparatus) and why this check exists at the design's
   own contrast rather than being assumed from the mitigation row.
2. **Transfer plumbing pilot**: ONE taught pair end-to-end through the real
   export → ingest → probe path, PLUS one **dangling-half pilot pair** (the link
   channel's cheap tripwire — if the dangling pilot's probe shows any non-floor
   preference, the winning component is read from provenance and the design goes back
   to §Links before a single campaign record exists), plus the three anti-vacuity
   variants. The taught pilot's probe percept must complete into the transferred
   cluster with `|learned_margin| > 0.11` (L1's argmax floor).
3. **L12 zero-prior**: on every Phase-0 probe decision,
   `score_components["drive"] == 0.0` from the decision-provenance event — asserted,
   not assumed. The campaign harness asserts the same per B-probe (S3).
4. **Floor sanity** (the Exp 41 headroom lesson, restated for the measured-floor
   design): the isolated pilot's first-choice distribution over ≥ 10 dithered probes
   concentrates on no affordance at ≥ 0.5 AND is not seed-invariant. Failure here
   diagnoses the SELECTOR/dither apparatus (§Selector) — not the body — before any
   arm runs.
5. **Donor mechanism check**: the taught pilot donor passes the full donor-sanity
   set (world-cluster bias ≥ 0.4, link balance, geometry stamps) at the frozen
   schedule length K; a failure adjusts K or the teacher apparatus via pre-campaign
   amendment (with readings disclosed), never silently.

(Disclosure rule: Phase-0 readings are recorded in this file's amendment section
before the campaign; gate constants are NOT retuned from them — any change is a
pre-data structural amendment with the disclosure the Exp 52 precedent set.)

## Apparatus (what must exist at the frozen commit; owed items in the sign-off)

- **Body:** `bodies/minecraft_bench` — a benchmark variant of the Minecraft player
  body with (a) **opaque affordance and drive names** (`aff_*` per-pair permuted,
  drive `d1`; L12: "eat" sits in `_DRIVE_TOOL_AFFINITIES` under `hunger`, and the
  drive-name-substring channel pays any tool containing the drive's name — opaque
  names on BOTH sides close both channels; the table itself is not touched, per the
  ledger's own instruction); (b) a world channel budgeted to **≤ 12 sensors** (L11's
  per-channel budget, stated here as the rule requires), chosen so the contingency's
  signature spans ≥ 2 sensors; (c) ranges re-centered so rest sits at the A4 neutral
  (the 1.1.4 discipline); (d) the interoceptive drive `d1` (drift > 0, MODELED — not
  world-owned, so the teacher's feed can relieve it and the operant path is not
  short-circuited by the live-readback credit withhold), and a satiated twin body
  (drift 0, initial 0 — a body variant, visible in provenance, the Exp 52 pattern).
  **The satiated twin keeps the SAME entity name** (the `reachy_mini_infant_satiated`
  precedent — the entity name is load-bearing: it prefixes every tool signature and
  is the `body_ref`), differing only in the drive's `initial`/`drift`; both donor
  kinds export `--body-ref minecraft_bench`, so gate 7 admits every arm's bundle and
  the keys can match at all. Roster size k (design target 8) and the exact YAML are
  frozen at the harness-merge commit and hash-recorded in the sign-off.
- **World:** the LIVE bridge — the frozen NDJSON protocol against a real offline-mode
  Paper server (the L11-capture apparatus lineage), in a maximally controlled world:
  superflat/void, daylight cycle off, no mob spawning, the contingency source placed
  by server command per the seeded script, and every scheduled affordance
  mechanically executable at its step (the link-balance obligation, §Links). The
  frozen "run in Minecraft" language and the world seam's own doctrine (measured
  state, not declared deltas) both point at the live bridge, and this repo has
  already paid once for a synthetic measurement nearly producing a false result (the
  L11 zero-vector caution). The `FakeBridgeServer` scripted world serves harness
  development and the `--mock` smoke ONLY — never a confirmatory record. Operational
  consequence, accepted and estimated: the real bridge takes one client, so sessions
  are sequential per server/bridge instance — ~300 sessions (100 donors + 200
  receiver probes; arm 4 reuses arm 2's donors) at minutes each ≈ 1–2 days of wall
  time per instance, parallelizable across instances (distinct ports), instance id
  stamped per record.
- **Sequential phases, no shared world**: A-phase and B-phase are separate sessions
  against the same seeded world script. Nothing in this design needs two agents in
  one world at once (the shared-state fake-bridge gap recorded in the harness
  docstring is NOT on this experiment's path).
- **Teacher:** harness-side (`scripts/exp56/` or a `minecraft_harness` extension),
  calling only sanctioned APIs (`credit_operant_reward`, relief-signed per
  `credit=relief`); its telemetry records per-feed `relief`, `credited`, and the
  credited `(cluster, tsig)` (the Exp 52 audit surface).
- **Loop:** substrate-primary, **no LLM in the action path** (the world drives
  percepts; nothing narrates), which is what de-confounds "the LLM already knows
  what food is" (the Goldilocks trap named in the design doc);
  `consolidation="full"` (the `is_sim_mode` trap: a lightweight close loses the very
  thing measured); `MAXIM_OPERANT_ONLY_CREDIT=1` in every arm and phase.
- **Ingestion (arms 2/3/4):** `maxim substrate ingest <bundle> --session <B-home>
  --trust <donor-id> --receiver-body minecraft_bench --receiver-agent-id <B-id>
  --apply`. Export: `maxim substrate export <zip> --session <A-home>
  --contributor-id <donor-id> --body-ref minecraft_bench --body-yaml <spec>` (the
  harness ships the `body:`-rooted wrapper spec — `load_spec` does not read component
  YAMLs directly; the donor's `--contributor-id` is the id trusted at ingest). No
  `--allow-unstamped-geometry`, no `--force-digest`, no `--inherent-trust`.
- **Independence (D44), asserted per pair (S3):** donor and receiver ids differ; the
  receiver's pre-ingest EC contains zero donor cluster ids; post-ingest, arm-2's
  landed bias keys carry the RECEIVER's agent id (the boundary normalization under
  test).
- **Harness compliance:** `assert_repo_interpreter` before the first session;
  `preflight_gated_record` (clean tree; `--allow-dirty` disallowed for the
  confirmatory campaign); `executed_code_provenance` per run record; every record
  carries `ts`; `--mock` smoke mode and `--resume` for interrupted campaigns;
  results route through `evidence_out_paths` (`--write-experiment-results` to touch
  committed paths); analyzer verdict constants frozen in `scripts/analyze_exp56.py`
  in the same PR as the harness, extended never retuned. One harness at a time, not
  on the leader machine (S8/L02). `require_semantic_encoder` is deliberately NOT
  required: the path under test is `SensorEncoder` (deterministic SHA bases, no
  model) end to end.

## Outcome tree (decided now)

| outcome | reading | action |
|---|---|---|
| Phase 0 fails check 1 or 2 (incl. a non-floor dangling pilot) | The instrument cannot see the thing (L11 at this contrast, plumbing, or the link channel leaking despite §Links) | Campaign does not run. Fix is apparatus/design work, re-run Phase 0; each retry recorded here. Two Phase-0 failures with NEW failure modes → stop, bird's-eye audit (the divergence rule); 1.2 does not ship the claim on schedule. |
| Phase 0 fails check 3 | L12 leak — a prior is inside the selector | Campaign does not run; body/name fix; re-run Phase 0. |
| Phase 0 fails check 4 | The dither is not reaching the agent (L2) or the floor is concentrated | Campaign does not run; SELECTOR apparatus fix (§Selector); re-run Phase 0. |
| Phase 0 fails check 5 | The teacher/schedule cannot mint a readable donor at K | Pre-campaign amendment (K / teacher apparatus), disclosed; re-run Phase 0. |
| All four gates pass, ANTI-VACUITY + L2 gates clean | **The 1.2 sharing claim, earned.** A's learned representation changes B's first-contact behaviour through the real ingestion path, by the situation-keyed channel (bias-decisive by construction of the gate). | New graduation row (Earned, Tier 2) citing the committed data; interpretation write-up in a separate later PR per the structure-or-time rule; the two-Reachy replication prereg and the dose–response ladder prereg are unblocked and written next. |
| ANTI-VACUITY fails | The gate cannot fail — analyzer/apparatus defect, no verdict on any gate | Fix the analyzer with a guard test; re-analyze the recorded campaign; the one permitted re-run applies only if the defect contaminated the records themselves. |
| L2 apparatus gate fails | Seed-invariant outcomes — effective n collapsed | No verdict; L2 amendment recorded; selector/dither fix; the single re-run applies. |
| TRANSFERRED/ABOVE-FLOOR/WANT-NOT-FILE pass; BOTH-HALVES fails | The effect does not require the representation — mechanism contradicted (the §Links containment failed at campaign scale) | NOT-EARNED; finding recorded; the audit target is the winning-component provenance on arm 4's successes, its own pre-registered follow-up. |
| WANT-NOT-FILE fails (satiated moves B) | Something non-learned in a bundle moves behaviour | STOP — not a science result; find the leak (the ingest reports' counts + link-balance records say where), fix with a guard test, re-run the campaign once. |
| TRANSFERRED fails but ABOVE-FLOOR passes (e.g. 0.55 vs 0.13) | Real transfer below the absolute bar | PARTIAL, named; 1.2 ships with it named or holds — owner call at the release checkpoint; the audit target is magnitude through the merge (clamps/caps), not the mechanism. |
| ABOVE-FLOOR fails | No detectable transfer | NOT-EARNED; recorded; the D43-class audit (where did the biases land? the committed ingest reports say) runs before any new design. |

**Stop rule:** the confirmatory campaign runs ONCE (plus the single re-run the
WANT-NOT-FILE-leak / L2 / contaminated-anti-vacuity branches allow — one re-run
total, whichever branch claims it). A second campaign-level divergence — a new
failure mode, not a narrowing — ends Exp 56 for 1.2: the result ships as it stands.

## What this experiment does NOT claim

- Nothing about scaling — the dose–response ladder (arms N ∈ {1, 2, 4, 8} per
  [minecraft_benchmark.md](../../plans/minecraft_benchmark.md) §"The dose–response
  ladder") is its own prereg, frozen only after this apparatus is proven, at the
  unsaturated operating point d43 §7 measured (probe 7 near N=10, K=4 —
  below-ceiling with headroom).
- Nothing about hardware (the two-Reachy cross-unit replication: its own prereg,
  riding Exp 54's `reachy_mini_infant` body and Gauntlet #2, per the case study).
- Nothing about aversion/negative-valence transfer (Exp 55, 1.3-line) — the
  tighten-only clamp is exercised only as byte-untouched positive pass-through here
  (its positive-fold guard is `TestTightenOnlyClampSeam`, already merged).
- Nothing about cross-layout generalization: A trains and B probes against the SAME
  seeded world configuration; "recognizes the situation anywhere" is not claimed.
- Nothing about self-taught wants: the donor is TAUGHT by a contingent teacher (the
  Exp 52 shape); autonomous acquisition of world-keyed wants requires the credit-path
  extension that is deliberately 1.3-line work.
- Nothing about the LLM-AUT path, live-server dynamics beyond the controlled world,
  or multi-agent coexistence.
- **Limits declared inapplicable by name** (the ledger discipline): L4 (`safe_pref`
  saturation — different metric, and the floor arm is measured, not assumed); L6
  (prior-agreement ceiling — no LLM in the action path; the L12 prior channel is the
  applicable analog and is gated); L9/L10 (DoA sweep instruments — no DoA here).

## Apparatus declarations (S1–S8)

- **S1 rows riding on the apparatus:** none change mechanism — the campaign consumes
  shipped, guarded paths (A4 world gain, the operant credit path, `substrate_merge`,
  the V1–V10 adapter). The bench body, world script, teacher, and harness are
  experiment-local assembly. If any `src/` mechanism change proves necessary
  post-Phase-0, it is a declared amendment + its own reviewed PR before the campaign.
- **S3 in-sim assertions (refusal exit ≠ 0, never absorbed):** the donor-sanity set
  (§Arms — bias floor, link balance, geometry, zero satiated credits, no inherent
  keys); per-probe `score_components["drive"] == 0.0`; arm-4 `biases_rekeyed == 0`;
  the independence set; teacher telemetry (`credited` implies `relief > 0`; satiated
  arm credits ≡ 0).
- **S4:** per-run JSONL + ingest reports + Phase-0 records committed under
  `docs/experiments/data/56_*` with provenance stamps; workdirs on durable storage.
- **S5 exposure contract:** taught and satiated donors execute the IDENTICAL seeded
  balanced schedule against the identical world script (drive state, carried by the
  body variant, is the only difference); receivers face the identical per-pair probe
  script across all four arms; the probe's frozen pre-contact bound (≤ 10 decisions)
  and tail (5) equalize B's own link accrual across arms.
- **S6:** no fidelity toggles differ between arms; `MAXIM_OPERANT_ONLY_CREDIT=1`
  everywhere; the harness refuses ambient `MAXIM_*` disagreement (the Exp 52 exit-3
  pattern).
- **S7:** the floor is measured (isolated arm), Phase-0 check 4 guards its sanity,
  and the BOTH-HALVES band is one-sided so a floor shift cannot flip the falsifier
  silently; TRANSFERRED's absolute bar sits ~0.13 below the bias-decisive design
  expectation (0.83), so the gate is not evaluated at its own ceiling.
- **S8:** one harness at a time; not co-located with a leader.

## Amendments

**Amendment 1 — 2026-09-06, PRE-DATA, structural (harness implementation; no
Phase-0 or campaign data exist).** The ANTI-VACUITY gate as first frozen asserted
that all three no-op merge variants "must collapse toward the floor". That is
FALSE for the *donor-re-keyed-alone* variant by the design's own geometry: a
FRESH receiver holds nothing, so the real merge's output is equivalent to the
re-keyed donor alone — the variant's readout is EXPECTED to persist, and
asserting its collapse would have made the gate unpassable on every honest
campaign. This is precisely the equivalence D44's kit already documented for its
`return right` variant (d43_merge_correctness.md §7: `return right` PASSED the
Exp 45 probes — the vacuity lesson the kit encodes), which the first freeze
overlooked. Corrected to per-variant expectations: receiver-unchanged and
empty-state MUST collapse (these are the variants a vacuous gate cannot fail);
donor-alone is recorded with its persistence documented. Conservative direction:
the two collapse assertions are unchanged; no gate constant moved.
**Disclosure:** found while implementing `noop_variant_readout` in the harness,
before any Phase-0 run; no readings of any gated data existed at amendment time
(the only executions were `--mock` ScriptedBridge smokes, which the prereg
already declares non-confirmatory).

**Amendment 2 — 2026-09-06, PRE-DATA, structural (live-apparatus shakedown; no
gated Phase-0 or campaign data exist).** An UNGATED live shakedown of the full
apparatus (real Paper 1.16.5 + Mineflayer bridge + RCON, scratchpad output, no
`--write-experiment-results`, run during harness-tooling development) measured two
facts the mock world could not:

1. **`light_level` is DEAD on the live bridge** — it reads 0.0 at every anchor,
   including the open rest pad at noon. A declared-range [0,15] sensor pinned at 0
   is a rest-at-extreme sensor shouting identically everywhere: exactly the
   measured-bad regime the body design warns against, drowning the real signal.
2. **The original slot placement was inside the A4 gain's silence band**: a
   40-block excursion in a ±128 range is a mid-range deviation, which the cubic
   gain suppresses BY DESIGN. Measured through the shipped encode path on the live
   values: cos(rest, situation) = **0.9997** — one cluster; Phase-0 check 1 read
   separation **0.0**, stability 1.0.

**The changes** (the outcome tree's sanctioned Phase-0-failure path — body/world
apparatus fix, re-checked, recorded): `light_level` is REMOVED from
`bodies/minecraft_bench` (five world sensors; the signature now spans
`distance_from_spawn` + `y_altitude`), and the frozen contingency slots move to
far + high coordinates (|d| ≈ 88 of ±128, y = 112 of [0,128]:
(88,112,0) / (−88,112,8) / (8,112,88) / (−8,112,−88)). No range declaration, gate
constant, selector knob, or schedule constant changed. Measured after the fix, same
live apparatus: cos(rest, situation) = **0.19**, within-slot stability cos 0.9999;
the shakedown's Phase-0 re-run passed **all five checks** (separation 1.0,
stability 1.0, taught pilot bias-decisive at margin 0.9, dangling inert, kit pass,
floor concentration 0.3).

**Disclosure** (the Exp 52 precedent): at amendment time the author had read the
two ungated shakedown reports and the live sensor values above; no gated record
existed or exists. One reading deserves emphasis because it VALIDATES the
pre-registered design rather than tuning it: under the broken apparatus,
**check 2's taught pilot PASSED (bias-decisive, margin 0.9) while check 1 failed**
— the single fused cluster made the taught bias fire situation-FREE, invisible to
the pilot's own gates. The transfer pilot alone cannot detect situation-blindness;
check 1 is what catches it, and it did. The confirmatory Phase 0 will be re-run
GATED (clean tree, merged constants) regardless of these shakedowns.

**Amendment 3 — 2026-09-06, POST-DATA (Phase-0 instrument-check data only; the
confirmatory campaign has not run), DISCLOSURE ONLY — no gate constant or apparatus
changed; the pre-registered Phase-0 disclosure (outcome tree + sign-off).**
The confirmatory Phase 0 ran GATED on the operator's big-mac-mini against the live
apparatus (real Paper 1.16.5 + Mineflayer bridge + RCON; `--write-experiment-results`,
clean tree at main-reachable `4cf67cf9`, `mock: false`, `working_tree_dirty_src_scripts:
false`), committed at [`data/56_phase0.json`](../data/56_phase0.json). **All five checks
PASS**, readings:

- **Check 1 (L11/A4 discriminability):** separation **1.0**, stability **1.0** over 20
  rest→situation onsets — the far+high slots (amendment 2) separate cleanly on the live
  world, the situation-blindness the pre-amendment-2 apparatus showed (separation 0.0) is
  gone.
- **Check 2 (transfer pilot + dangling tripwire + no-op kit):** the taught pilot's first
  contact is `minecraft_bench_aff_c`, substrate-sourced, **bias-decisive at
  `learned_margin` 0.9**; donor sanity passes (world-cluster bias 0.9 ≥ 0.4, link spread
  0.0, no inherent keys); the dangling pilot does **not** choose the target (link channel
  neutralized as designed); the no-op kit's two must-collapse variants collapse
  (`kit_pass: true`).
- **Check 3 (L12 zero-prior):** `score_components["drive"] == 0.0` on every probe decision
  (asserted, PASS).
- **Check 4 (dithered floor):** 10 isolated probes spread across five affordances,
  concentration **0.3** — not seed-invariant, no affordance pre-installed.
- **Check 5 (donor at K):** donor sanity passes at the frozen K = 96.

No gate constant, range, selector knob, or schedule constant was retuned as a result —
Phase 0 passed clean, so this amendment records the readings and nothing more. The
campaign is cleared to run.

## Amendment rule

Amendments after first data (Phase 0's included) are permitted only for *structural
invalidity* — harness bug, degenerate metric, an apparatus-gate failure — never for
effect size; every amendment is its own PR merged before the data it governs, with
the read-state disclosure the Exp 52 precedent set. Gate constants above are frozen
at the commit that lands this file.

## Runbook (shape; exact flags frozen with the harness PR)

```bash
export PYTHONPATH=$PWD/src
# Phase 0 (instrument checks; commits 56_phase0_*.json)
python scripts/exp56/instrument_check.py --write-experiment-results

# Confirmatory campaign (4 arms × 50 pairs; --mock for the smoke)
python scripts/exp56/run_campaign.py --arms isolated,taught,satiated,dangling \
  --pairs 50 --seed-base 42 --out docs/experiments/data/56_four_arm.jsonl \
  --write-experiment-results
python scripts/analyze_exp56.py --in docs/experiments/data/56_four_arm.jsonl --gate v1 \
  --assert-noop-fails
```

## Sign-off (fills before the campaign; each box its own merged PR where marked)

- [x] This pre-registration merged to `main` via merge commit (never squash) — hash: `3f9ce733` (#639; amendment 1 #640, amendment 2 #642, amendment 3 below)
- [x] Harness PR merged with guard tests (`bodies/minecraft_bench{,_satiated}.yaml` +
      the `body:`-rooted export spec, world script, the teacher, `scripts/exp56/`,
      `scripts/analyze_exp56.py` with frozen verdict constants incl. the link-balance
      and bias-decisive assertions, `--mock`/`--resume`/`--assert-noop-fails`) —
      hash: `6219f330` (#641; world-setup tooling #643; `--spectator` #644, main tip `4cf67cf9`);
      frozen constants recorded: roster k `8`, schedule length K `96` (8 affordances × 2
      situation-states × 6 reps/cell), candidate contingency slots `4` — far+high per
      amendment 2: (88,112,0)/(−88,112,8)/(8,112,88)/(−8,112,−88)
- [x] Two-lens review round on the harness PR folded pre-merge (the house rule) — #641
      (executor + design lenses; all findings folded before merge, verified by execution)
- [x] Phase 0 run + committed (`56_phase0.json`): check 1 `PASS` check 2 `PASS`
      check 3 `PASS` check 4 `PASS` check 5 `PASS` (readings disclosed in amendment 3;
      gated on the big-mac-mini at main-reachable `4cf67cf9`, clean tree)
- [ ] Campaign run ONCE from a clean tree at a main-reachable commit; data PR
      merge-committed; interpretation in a separate later PR (structure-or-time rule)

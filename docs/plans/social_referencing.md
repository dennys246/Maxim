# Social referencing — consult the Oasis only when ignorant and hurting

> **PROPOSED 2026-09-24 — plan only, no code, no prereg; a PARALLEL line, not a 1.4 rung.** Split out
> of the grounded-language plan's v3 (owner decision 2026-09-24), where it was "Track S". It needs no
> language. **Depends on [public_oasis.md](public_oasis.md) Phase 0** (SCHEDULED 2026-09-24 for this
> reason). **No src before 1.3.2** (the `agent_loop` decomposition — this plan's read path lands in the
> selection code 1.3.2 is splitting). **No rig time before E3's campaign** (one rig, one operator).
> **Opt-in, off by default, never active in an E1–E3 arm.** **Nothing here is built.** The Design
> section describes intended behaviour in the present tense for readability; none of it exists.
>
> Its language sibling, [deferred/grounded_word_binding.md](deferred/grounded_word_binding.md), is
> DEFERRED behind a frozen offline gate.

## The question

When an agent is somewhere it does not know — or knows, but has no idea what to do — **and it is being
hurt or deprived**, would asking what others learned about situations like this one, and holding the
answer as advice apart from its own experience, reduce the harm it takes; and does gating that
consult on its own ignorance protect it from a stale Oasis **beyond what consulting less often
explains**?

**Claim ceiling (Exp C, the only experiment here), if EARNED:** *consulting gated on the agent's own
ignorance and stakes limits the harm a stale Oasis does, beyond what the same amount of advice given at
random moments explains.* If the gate does not beat the weight-yoked random arm, that is the recorded
result. Nothing here is called "the demo" (owner decision 4). Near-term public material is a recorded
clip of the EARNED Exp 61 result, plus the offline water classroom (`scripted_water` with chat event
frames) as a clearly labelled smoke instrument — never evidence, and not this line.

## Biological basis — and its limit

*Attribution, stated because a v3 heading got it wrong: the citations below were checked by the v2
bio-fidelity lens ([reviews/grounded_word_binding_demo/bio_fidelity.md](reviews/grounded_word_binding_demo/bio_fidelity.md))
against the sources. The design readings marked **(design)** are this plan's own and were not reviewed
by that lens; Nieh 2010 is unverified and marked so.*

- **Social referencing on borderline risk.** Sorce, Emde, Campos & Klinnert 1985 (*Dev Psychol* 21):
  at an ambiguous visual-cliff drop, 12-month-olds acted on the mother's expression (fear 0/17 crossed;
  joy 14/19); their unambiguous condition had no drop at all. The measured gradient is Tamis-LeMonda,
  Adolph et al. 2008 (*Dev Psychol* 44): 18-month-olds used social advice only on **borderline** slopes,
  ignoring it on clearly safe and clearly risky ones.
- **Social learning strategies** (Laland 2004; Kendal et al. 2018): *copy when uncertain / own
  information unreliable or stale* — van Bergen, Coolen & Laland 2004; *copy when learning alone is
  costly* — Coolen, van Bergen, Day & Laland 2003; *copy when own information stops paying* — Grüter &
  Ratnieks 2011 (honeybees); *trust experienced private memory over the social cue* — Grüter, Czaczkes &
  Ratnieks 2011 (*Lasius niger*).
- **Rogers' paradox** (Rogers 1988) and critical social learning (Enquist, Eriksson & Ghirlanda 2007):
  indiscriminate copying gives no net gain and propagates stale information. Enquist's critical learner
  copies first and learns alone when copying fails; this plan's gate is the reverse order (own first),
  the van Bergen/Laland rule. Exp C tests it as the **critical-learner prediction**.
- **Stigmergy** (Grassé 1959): a shared record written by many, read locally on need. **(design)** The
  local verified mirror is read like a pheromone field — nothing about the agent's situation is sent to
  read it.
- **Source memory** — what was experienced kept apart from what was told. **(design)** The foreign
  layer is that separation; it is what would make "own experience wins" enforceable.
- **Positive and negative recruitment feedback** (honeybee dances; the stop signal — Nieh 2010,
  *unverified, check before citing*): feedback about a food source is innate, not taught. **(design)**
  Outcome feedback here is a standard, not a learned behaviour.
- **Uncertainty** (Yu & Dayan 2005): expected vs unexpected uncertainty drive information seeking;
  irreducible risk (high outcome variance with ample experience) is not something a consult reduces.

**The limit:** biology supplies the decision rule — when to use social information — and the separation
of told from experienced. No animal queries a remote store; the transport (a signed release mirror) is
engineering.

## What exists today (verified against `main`, 2026-09-24)

| Piece | Status |
|---|---|
| Substrate-primary selection: `agent_loop.py::propose_via_substrate` → `NAc.recommend_action`, one EC cluster per modality | **Shipped** |
| Learned fear (`cluster_fear`, world cluster only) and want (`cluster_reward_bias`) | **Shipped**, EARNED (Exp 60/56) and transferable (Exp 61/56) |
| Familiarity: EC match margin → novelty (`LLMProposal.cluster_margins`, 2S-c) | **Shipped** (#879), recorded only |
| "No idea what to do": per-cluster fear/want history; `recommend_action` → `None` below `min_confidence` | **Shipped**; Welford outcome variance is **per tool** (`nac.py::get_action_risk_profile`), so not a per-situation signal |
| Stakes: drive pressure; pain felt per invocation (2S-c) | **Shipped** |
| Bundle export / ingest / merge; `hive add|pull`; `oasis serve|publish` | **Shipped**; bundle = manifest + NAc + EC; V8 dedup per bundle digest |
| Merge semantics | fear tighten-only (MIN, `FOREIGN_FEAR_DISCOUNT`); **want merges as a plain mean with no discount**; no per-source provenance on merged values |
| Live state | `NAc.load_state` replaces wholesale, unlocked; no live EC dump; ingest refused for a live agent (`hivemind/cli.py::_run_ingest`, contract §1) |
| `hive pull` credential | `hive_cli.py::_run_pull` defaults to `read_key()` — the leader key, which also grants inference |
| Signing | one key per machine for `export --sign`; `signer_identity` not covered by the signature; the signing tests are **skipped in CI** (no lane installs `cryptography`) |
| Drop-depth sensing | **Absent** — the bridge senses, among others, health, food, saturation, oxygen, light, altitude, nearest hostile/player distance, time of day, `is_in_water`; no thermal or fire sensor (survival-ladder D1 rejected a synthetic thermal sensor) |

## Design (intended; none of it is built)

**1. The key is the sensed situation.** World/interoception/audio clusters, matched per modality at the
sensor threshold (0.85). Heard text joins the key only if the language line revives.

**2. Advice is held apart: the foreign layer.** Admitted entries are tagged per consult (source,
release, entry, consult id, time) and read by selection and threat **alongside** the agent's own
state, weighted `trust(source) × (1 − own confidence in this situation)`. They decay, are never written
by credit, and are revertible per consult. Only the loop thread writes the layer. This layer is the
selection-side read seam the plan needs; it owns that seam (the v3 review found no stage owned one).

**3. The agent reads a local, verified mirror; the Oasis never sees a consult.** Whole signed releases,
each with a **Queen-signed entry index**, fetched in the background off the loop thread, verified once,
and swapped in **atomically**; a pure `hivemind` selector picks matching entries locally; admission is
journalled **per entry** (so overlapping releases never double-count). No search endpoint; no
server-cut slices. Reads are anonymous and rate-limited, or use a read-only scoped credential —
**never the leader key**.

**4. The trigger (an innate prior; declared tier).** Consult when **stakes** are high — drive pressure
or pain just felt, **not** the agent's own anticipatory threat (a clear learned danger is a clear read,
and clear reads do not consult) — **and** the agent is **unfamiliar** (EC-margin novelty), **or has no
idea what to do** (no own fear/want history here; `recommend_action` below `min_confidence`), **or its
own information is failing or stale** (recent negative outcomes here). **Not** a trigger: outcome
variance alone. A per-situation refractory and a **global consult budget**. On a first exposure the
consult can only fire at pain. Thresholds start hard-coded and are calibrated on seeds separate from
Exp C's; per the behaviour-tier rule a follow-up is filed with its trigger for making them learned.

**5. While consulting, the agent keeps its own policy** (owner). A **continuous caution level** replaces
an on/off hold: the trigger and the advice weight are re-evaluated every tick, so the advice's influence
rises or falls as percepts change. No new threading beyond the mirror refresh (design 3).

**6. When advice becomes the agent's own** (owner: only if it demonstrably worked).
- *Worked* = the advice changed the agent's choice **and** the outcome beat the agent's **class-level
  baseline**: its own mean outcome in earlier **first exposures** in the same **stakes bucket** that
  received **no advice**. (A per-situation baseline does not exist at the moment a consult fires —
  the agent is ignorant there by construction.)
- **Stakes buckets**, fixed in the prereg: pain felt (yes/no) × drive pressure (low/mid/high).
- **No promotion until the agent has k class-level first exposures** (k fixed in the prereg); before
  that, advice stays advice.
- In deployment the baseline comes only from **no-match** consults, which over-represent situations the
  Oasis does not cover — the most unfamiliar ones. The skew is named here and measured once in Exp C.
- "Nothing bad happened" counts as success only in a bucket where harm was likely.
- **Failure** lowers `trust(source)` and makes the advice fade faster.
- Promotion happens at **sleep consolidation** (the memory-strength line's sleep), never mid-tick.

**7. Outcome feedback: a standard, local by default** (owner, with the privacy fold).
- Each consulted entry's outcome is recorded **locally**, where it drives learned trust. This is
  standard (innate), not taught.
- **Publishing** it through the existing write-only `hive contribute` path is **opt-in**, **batched and
  delayed**. Batching breaks the link between a report and the moment it happened; it does **not** hide
  the core fact — **naming which entry was used reveals that the agent met a situation matching it.**
  That is a privacy trade against design 3's "the Oasis never sees a consult", stated as one, and the
  reason publishing is off by default.
- Feedback is a **poisoning vector** (fabricated successes). The Oasis acts on it only once an accept
  path exists — [public_oasis.md](public_oasis.md) Phase 2, deferred. Until then local-only costs
  nothing.

**8. No look-back.** This plan uses no retrospective look-back; `PerceptTraceBuffer` is owned by R4
([roadmap_1_4.md](roadmap_1_4.md) §Phase 5), and if a later stage needs it, it consumes R4's reviewed
design.

**9. The pre-boot ingest path is unchanged** by this plan (today's merge, incl. want's undiscounted
mean). Whether pre-boot foreign material should also be discounted and capped belongs to the language
plan's join stage, not here — the v3 text said both, and this plan takes neither side.

## Stages

- **S0 — prerequisites, shared with [public_oasis.md](public_oasis.md) Phase 0** (no S1 code before
  these): the CI lane installing `cryptography` (and fastapi for `oasis serve`); pulls never send the
  leader key; `signer_identity` covered by the signature; releases carry a Queen-signed entry index.
  **Exp C-specific:** the corrupted test Oasis is built with a **throwaway signing key**, in a test-only
  namespace, in a temporary directory, never on a machine whose key a real receiver trusts — a bundle
  signed with the Queen key *is* a Queen release.
- **S1 — the foreign layer and the mirror** (src after 1.3.2; opt-in): designs 2–3, with offline tests
  that prove each guard by deleting the mechanism (empty the layer → a consult changes nothing). S1
  **owns a DECISIONS.md entry** amending the ingest contract §1 (today `hivemind/cli.py::_run_ingest`
  refuses a live agent): *a live consult admits entries to the foreign layer; the file ingest stays
  at-rest.* The same entry records design 7's feedback-privacy trade (publishing reveals which
  situations the agent met), with a pointer from the sharing threat model.
- **S2 — the trigger, the caution level, promotion and feedback** (designs 4–7), recording-only first:
  consults are logged and weighted, and the log is read before any behavioural claim.
- **S3 — the cliff sensor, on a VARIANT body.** A game-exposed **drop depth ahead** sensor (D1-clean,
  like light level), declared **only on a variant body** — never the shipped `minecraft_player`, whose
  body change would trigger re-runs of Exp 56/60/61 ([roadmap_1_4.md](roadmap_1_4.md) §Bodies). It adds
  one sensor to the world channel, so the L11 sensor-dilution cost is measured on the variant before
  Exp C.
- **S4 — Exp C** (after E3's campaign): prereg on `main` first, four-lens design review, then build.

## Experiment C — does the gate limit a stale Oasis's harm, beyond less advice?

- **Situations:** water (the EARNED anchor) and the **graded-drop cliff** — drops of 2, 3, 4 and 6
  blocks; fall damage only above 3, so 3 and 4 are the borderline — the Tamis-LeMonda test; a hostile in
  the dark (existing sensors) as a secondary situation. Night stays out of gated arms until Exp 62's
  night boundary is understood.
- **Arms:** gated · always · never · **weight-yoked random** (each run paired with a gated run and given
  the **same cumulative advice weight**, at random moments — yoking by weight, not count, because the
  weight is the quantity the gate acts on) · **content-null** (consults on the gated schedule, receives
  an empty answer — separates the pause from the answer).
- **Oasis conditions:** clean · **stale** (entries true in an earlier world layout, false in this one —
  the critical-learner test). **Inverted** (valence flipped) is a deception test, reported separately.
  Which situations carry corrupted entries is pre-registered.
- **Donor corpus:** donor agents trained on the earlier layout, exported and signed with the throwaway
  key (S0). Contribution being write-only does not block this; it costs rig time.
- **Primary DV:** health lost per consult-eligible trial. Secondary: harm per corrupted entry admitted;
  the drift between the content-null baseline and the no-match baseline (design 6's skew).
- **Primary test:** the gate × Oasis-condition interaction on the paired gated-minus-yoked differences,
  by a **paired sign-flip test** (yoking breaks exchangeability across arms, so pairs are the unit). Its
  floor is 2^-n_pairs, so fewer than 5 pairs cannot reach p < 0.05; the prereg fixes n_pairs well above
  that, the α, and a minimum effect size.
- **Frozen location:** the prereg is committed to `docs/experiments/` on `main` before any data (the
  gated-data rule); no n, margin or threshold is chosen after data.
- **A genuine may-fail:** gated not beating weight-yoked random is the recorded result.
- **Rig:** big-mac-mini once quiet; the operator starts the Paper server (no model server: no LLM is in
  this loop). Budget ≈3–5 h.

## Guards and disciplines

- A fix ships with a caller; gates are strict red gates on the real composition.
- Prove each guard by deleting the mechanism (disable the gate → gated equals weight-yoked; empty the
  foreign layer → consults change nothing).
- Opt-in and off by default; an E1–E3 arm never runs with it on.
- Harnesses assert the `maxim` they import is their own repo.

## Open questions for the owner

1. k (first exposures before any promotion) and the stakes-bucket edges — the prereg fixes them; any
   preference?
2. Should feedback publishing, once an accept path exists, ever be on by default, or stay opt-in
   permanently given the situation leak?

## Review record

- **v2 → v3 (2026-09-24):** six lenses on the combined plan; reviews in
  [reviews/grounded_word_binding_demo/](reviews/grounded_word_binding_demo/). Their Track S findings
  (foreign layer, local mirror, security prerequisites, Exp C's yoked and content-null arms) are the
  basis of this plan.
- **v3's own text (2026-09-24):** a five-lens review (fold-verification, claim discipline, demo surface,
  sequencing, measurement) ran in another session; its reports are **not in the repo**. Its findings were
  re-verified against `main` in this session. Folded here: the pre-boot contradiction (design 8); the
  missing read-seam owner (design 2); no n / margin / DV / frozen location, the invalid permutation test
  and count-yoking (Exp C); the rig line's "model server"; claim ceilings and conditional tense. Two of
  its claims were wrong and are not folded: the encoder model (it is `paraphrase-mpnet-base-v2`), and
  "Exp 62 read as unrun".
- **Owner decisions and the review of the split (2026-09-24):** the answers in designs 5–7 and S3;
  feedback's privacy contradiction; the undefined baseline at first exposure; the variant-body rule for
  the new sensor.

## Where this is referenced

[public_oasis.md](public_oasis.md) (Phase 0, scheduled for this plan) ·
[maxim_hivemind.md](maxim_hivemind.md) (the foreign layer and the mirror) ·
[roadmap_1_4.md](roadmap_1_4.md) §Parallel lines ·
[deferred/grounded_word_binding.md](deferred/grounded_word_binding.md) (its language sibling).

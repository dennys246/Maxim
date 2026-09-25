# Grounded word binding — a heard word calls back its situation (substrate-primary)

> **DEFERRED 2026-09-24 on a CONDITION, not a date (owner decision).** This plan sits behind an entry
> gate that has **not passed**: the grounded-language line's paired-data audit returned *REDESIGN THE
> DATA SOURCE — not a pass* (2026-09-20), and its re-audit was *EXPLORATORY; not a revival*
> (2026-09-21) — see [grounded_language_acquisition.md](../grounded_language_acquisition.md). v1–v3 of
> this file never said so; this header does. Renamed from `grounded_word_binding_demo.md` (nothing is
> called "the demo" — owner decision 4). **Its social-referencing half ("Track S") was split out** to
> [social_referencing.md](../social_referencing.md), which needs no language; the Track S, Exp C and
> contract sections below are kept as the v3 record and are **superseded by that plan**.
>
> **Re-entry gate — frozen 2026-09-24, before any data (L0's offline half; no rig):**
> - **Model pinned:** `paraphrase-mpnet-base-v2` (the `LinguisticEncoder` config default). **Threshold
>   pinned now: 0.44** — `similarity/ec.py::ECConfig.pattern_complete_threshold`, the value text nodes
>   form at today (no text-specific override). Purity is measured **through Maxim's own EC cluster
>   formation** at 0.44 — what the agent would actually use. The threshold is **not** fitted to the
>   gate's own data (that would be circular); the body's "fixed from" / "calibrated in L0" wording is
>   superseded by this line.
> - **Situations:** water, food, fire, cave, night — **all five blind-authored through the same process**
>   (each phrasing written without seeing the other situations; uniform source, so writing style
>   cannot separate them), **m ≥ 5 different phrasings each** (not one-word variants). The list is
>   committed before the run.
> - **Near misses:** cave vs night (both dark; the only within-five pair). **Declared distractor
>   classes** fall and lava are scored for confusion with the five but do **not** count toward the
>   pass.
> - **PASS:** cluster purity ≥ 0.9 with no two of the five sharing a majority cluster, **AND**
>   leave-one-phrasing-out nearest-centroid accuracy ≥ 0.8 (chance 0.2). A **word-overlap baseline** is
>   reported beside it; a pass that plain word matching also achieves is flagged as vocabulary, not
>   situations.
> - **Secondary check (not part of the pass):** the game's own death messages (drowned / burned) —
>   fixed templates, where a word-overlap flag is the expected result.
> - **FAIL** otherwise → this plan goes to `archive/` with the measured null, as the parent plan
>   specifies. **PASS** → candidate 1.5 headline, re-entering behind a prereg and the four-lens review.
> - **Even a PASS licenses only "labels and their paraphrases bind" — never language.**
>
> **Owed at revival (found by a review of v3 whose reports are not in the repo; re-verified
> 2026-09-24):** a stage that builds the recalled-situation input (design 1 names it; no stage owns
> it); n, primary DVs, margins and a frozen prereg location for Exp A/B; text-channel weight (the
> `semantic` extra: sentence-transformers, torch, spacy); `scripted_water.py` carries no event frames,
> so chat needs adding before any teacher run; L1 adds a modality to the loop 1.3.2 is decomposing, so
> it waits for 1.3.2. The look-back (`PerceptTraceBuffer`) is **owned by R4**, whose design review is
> scheduled ([roadmap_1_4.md](../roadmap_1_4.md) Phase 5) and fixes its per-tick (not per-second)
> decay. The relation formerly named `NAMES` is now `HEARD_WITH` (a co-occurrence, not "reference";
> free to rename — not in src).
>
> **Near-term public material is NOT this plan:** a recorded clip of the EARNED Exp 61 result, plus
> the offline water classroom (`scripted_water` with chat event frames) as a clearly labelled smoke
> instrument, never evidence.
>
> *The body below is the v3 design, in the conditional: nothing in it is built.*

## The design, in one paragraph (v3; nothing built — its consult half is SUPERSEDED by [social_referencing.md](../social_referencing.md))

A substrate-primary Maxim — no LLM anywhere in its action path — plays the survival world. What it
hears (the game's messages, a teacher player's chat at chosen moments) is part of the situation it is
in. It associates a heard word with the situation the word accompanied; later, **hearing the word
alone** calls that situation back, and the fear or want the agent learned *there* shapes what it
chooses — before its sensors see anything. When it is somewhere it does not know, or knows but has no
idea what to do, **and it is being hurt or deprived**, it does what a toddler on a borderline slope
does: it **consults** — reads what others learned about situations like this one from its locally
held, signature-verified copy of the Oasis — and holds that answer **as advice, apart from its own
experience**, weighted less the more it knows itself. A user can ask on its behalf ("Search Oasis")
with typed text; that is the same consult with a text key.

**What each stage may claim, at most** (every public line stays inside the sentence for the stage
actually EARNED):
- **Exp A:** *a heard word, associated with a situation, calls that situation back and changes what the
  agent chooses — and the effect runs through the situation, not through the word's own conditioning*
  (a stimulus–stimulus association with retrieval at test; not "understands", not "refers" in the
  linguistic sense).
- **Exp B:** *an association one agent learned works for another*, pre-boot and live.
- **Exp C:** *consulting gated on the agent's own ignorance limits the damage a stale Oasis does,
  beyond what consulting less often explains* — or, if not, that it does not.

Not language understanding, not production, not learning from the internet.

## Biological basis — and its limit

*Attribution corrected 2026-09-24: the citations were checked by the v2 bio-fidelity lens; the sentences about "v3's transport" and "v3's foreign layer" are v3's own design readings, which that lens never saw.*

- **Social referencing on borderline risk.** Sorce, Emde, Campos & Klinnert 1985 (*Dev Psychol* 21):
  at an ambiguous visual-cliff drop, 12-month-olds looked to the mother and acted on her expression —
  fear: 0/17 crossed; joy: 14/19. (Their unambiguous condition had **no drop at all**; deeper drops
  were only piloted.) The properly measured gradient is **Tamis-LeMonda, Adolph et al. 2008** (*Dev
  Psychol* 44:734–746): 18-month-olds on slopes ignored social advice on clearly safe and clearly
  risky slopes and used it only on **borderline** ones. So: *consult on the borderline; a clear read
  of either sign — including clear danger — does not consult.*
- **Social learning strategies** (Laland 2004; Kendal et al. 2018). *Copy when uncertain / when own
  information is unreliable or stale*: van Bergen, Coolen & Laland 2004. *Copy when learning alone is
  costly*: Coolen, van Bergen, Day & Laland 2003 (sticklebacks) — the basis for the stakes condition.
  *Copy when own information stops paying*: Grüter & Ratnieks 2011 (honeybees follow dances more once
  their feeder fails). *Trust experienced private memory over the social cue*: Grüter, Czaczkes &
  Ratnieks 2011 (*Lasius niger*).
- **Rogers' paradox** (Rogers 1988) and **critical social learning** (Enquist, Eriksson & Ghirlanda
  2007): indiscriminate copying gives a population no net gain and propagates stale information.
  Enquist's critical learner *copies first and learns individually when copying fails*; this plan's
  gate is the **reverse order** (own first, consult when own falls short) — the van Bergen / Laland
  "copy when uncertain" rule. Exp C tests that gate's prediction, called here the **critical-learner
  prediction** rather than "Rogers' paradox".
- **Stigmergy** (Grassé 1959): a shared record written by many, read **locally**, on need, and
  fading. v3's transport matches it more closely than v2's: the agent reads its *local* copy of the
  Oasis; nothing about its situation leaves the machine.
- **Source memory**: animals and people keep "what I experienced" distinct from "what I was told";
  v3's foreign layer is that distinction, and it is what makes "own experience wins" enforceable.
- **Uncertainty signals** (Yu & Dayan 2005): acetylcholine ~ expected, noradrenaline ~ unexpected
  uncertainty. **Irreducible risk** (high outcome variance with plenty of experience) is not
  something a consult can reduce, and is not a trigger.
- **Binding is hippocampal first, cortical later**: a new word–situation association is formed
  episodically and consolidated into the semantic hub (the ATL, hub-and-spoke) over sleep. v3 binds
  in the episodic trace and consolidates to the ATL, not straight into the ATL.
- **Sensory preconditioning** (Brogden 1939) with **post-conditioning extinction** (Rizley &
  Rescorla 1972) separates retrieval at test from mediated conditioning — Exp A's design.
- **Not used:** tree-to-tree warnings via mycorrhizal networks (contested; Karst et al. 2023).
  *Physarum* habituation transferring on fusion (Vogel & Dussutour 2016) is a precedent for the
  pre-boot bundle merge only.

**The limit, stated:** biology supplies the *decision rule* for when to use social information, and
the separation of told from experienced. No animal queries a remote store; the transport (a signed
release mirror) is engineering, and the plan says so.

## What exists today (verified 2026-09-24 against `main` @ eda35efb)

| Piece | Status |
|---|---|
| Substrate-primary selection: `agent_loop.py::propose_via_substrate` → `NAc.recommend_action` over `{modality: EC cluster}` — **one cluster per modality** | **Shipped**, production caller |
| Situation encoding (`SensorEncoder`, 384-d) for interoception/audio/world; cluster ids are ATL concept ids | **Shipped** |
| Fear (`cluster_fear`) written by the pain→fear subscriber **only to the world cluster**; reward credit only to world/audio/interoception | **Shipped** — a text cluster can hold neither today |
| Game text as a percept: player chat as `[minecraft:chat]` (always on); game system messages as `[minecraft:system]` behind `--system_messages` (#807) | **Shipped** — observed, never in the situation (`_SUBSTRATE_CHANNELS` excludes text) |
| Text encoding (`LinguisticEncoder`, 768-d): lazy-loaded; **silently falls back to 384-d hashes** if the model is absent | **Capability** — `MemoryHub` with `MAXIM_SUBSTRATE_PATH=1` only |
| Look-back (`memory/percept_trace_buffer.py::PerceptTraceBuffer`) | **Capability** — zero production constructors |
| Cross-modal comparison 384 ↔ 768 | **Undefined** (`similarity/ec.py::_cosine_similarity` → 0.0) |
| Familiarity: EC margin → novelty on `LLMProposal.cluster_margins` (2S-c) | **Shipped** (#879); this branch predates it — rebase before any stage PR |
| "Do I know what to do here": per-cluster fear/want history; `recommend_action` → `None` below `min_confidence` | **Shipped**. Welford outcome variance is **per tool** (`nac.py::get_action_risk_profile`), not per situation |
| Stakes: drive pressure, pain just felt (2S-c) | **Shipped**; salience unmeasured on the loop path |
| Bundle export/ingest/merge, `hive pull`, `oasis serve/publish` | **Shipped**; bundle = manifest + NAc + EC; ATL slot reserved; V8 dedup is per bundle digest |
| Merge | `hivemind/merge.py::substrate_merge` re-keys the donor only; fear merges tighten-only (MIN); **want merges as a plain mean with no foreign discount**; merged values carry no per-source provenance |
| Live state | `NAc.load_state` and `ATL.load_state` **replace wholesale, unlocked**; no live EC dump; **no ATL merge exists** |
| Ingest into a live agent | Refused (`hivemind/cli.py::_run_ingest`, contract §1) |
| `hive pull` credential | Sends the local leader key (`read_key()`) by default — which also grants inference |
| Signing | `export --sign` signs with the machine's one key; `signer_identity` is not covered by the signature |
| Paired data | 2026-09-20 audit: labels → REDESIGN THE SOURCE; 2026-09-21 re-audit EXPLORATORY; the capture script stamps events with its 0.5 s poll time, so it cannot measure a lag |

## The four design decisions the review forced

**1. Three inputs, three roles.** The loop distinguishes what is **sensed**, what is **heard**, and
what is **recalled**:

| Input | Selection reads it | Credit / fear / want written to it | Memory situation key |
|---|---|---|---|
| sensed situation (world/interoception/audio) | yes | yes (as today) | yes |
| heard text (the `text` modality) | yes — it is part of the situation considered, and the consult key | **no**, unless an experiment arm declares it (Exp A's conditioned-word control) | recorded as its own field, not merged into the key |
| recalled situation (a word calling its situation back) | **yes, only** | **never** | never |

Without this, a recalled world cluster would displace the sensed one (one cluster per modality) and
fear, credit and the memory's situation key would be written to a situation the agent only
imagined; and a text cluster would become the operant credit target on Minecraft (it sorts ahead of
`world`).

**2. Consulted material is advice, held apart — never merged into the agent's own state.** A
**foreign layer**: entries tagged per consult (source, release, consult id, time) that selection and
threat read *alongside* the agent's own state, weighted by `trust(source) × (1 − own confidence in
this situation)` — so own experience wins in proportion to how much of it there is (the ant rule),
decaying, never written by credit, revertible by consult id. This is the source-memory distinction,
and it removes the blocking problems of a live merge: foreign want installed at full value (want has
no discount), foreign fear that nothing can lower, lost provenance, the unlocked wholesale
`load_state`, and a pure-Python merge too slow for a ≈1 s tick. Promotion from advice to own
knowledge happens only through the agent's own experience confirming it — an owner question below.
The **pre-boot** bundle ingest keeps today's merge path unchanged. *(2026-09-24: this contradicted J1's "discounted and capped" below; unresolved here — J1 decides it at revival.)*

**3. The agent reads a local, verified copy of the Oasis; the Oasis never sees a situation.** The
Oasis side publishes signed releases, each with a **Queen-signed entry index**; the agent's runtime
refreshes a local mirror in the background (off the loop thread) and verifies each release once. A
consult is a **local selection** from verified releases by a pure `hivemind` selector — no search
endpoint, no server-cut slices (which could not carry the contributor's signature and would defeat
V8's dedup), nothing about the agent's situation sent anywhere. Admission is journalled **per entry**.

**4. Two tracks and a join.** Social referencing does not depend on the language work — it can run
on the sensed situation alone. So:
- **Track L (language):** L0 → L1 → L2 → Exp A.
- **Track S (social referencing):** S1 → S2 → Exp C.
- **Join:** J1 (the ATL payload) → Exp B → J2 (heard text in the consult key + Search Oasis) → J3
  (generalisation, after Exps A and B).

*(v3 said "the lead-up demo is Exp A + Exp C + Exp B + J2"; superseded 2026-09-24 — nothing is called the demo, and Exp C moved to [social_referencing.md](../social_referencing.md).)*

## Front-gate: what is genuinely new

1. **The recalled-situation input** — the proposal holds one cluster per modality; a recall cannot
   sit beside a percept without its own slot.
2. **The foreign layer** — NAc state has no notion of source; a merge cannot be discounted, decayed
   or reverted per source.
3. **The ATL payload** — the reserved slot, scoped to `HEARD_WITH` relations only.
4. **The release mirror + entry index + per-entry journal** — releases are fetched whole and
   journalled per bundle today.

Everything else rides: binding on the episodic trace (2S-b's situation record) consolidated to an ATL
relation; the look-back on `PerceptTraceBuffer` (owned by R4); the trigger on signals the loop
already computes; pre-boot transfer on bundle → ingest → merge.

## Track L — language

**L0 — the data source and the curated teacher.**
- **Timestamps first:** the bridge stamps each event, the client keeps it, and the bridge runs at
  100 ms, so the situation→message lag is measurable at all. Then a capture of **natural** deaths in
  a **separate world** (a scripted wanderer dies almost only to mobs: good for the lag, useless for
  water or fire examples).
- **The teacher:** a second client that speaks through **player chat** (`[minecraft:chat]`, no flag
  needed), never opped, parked **outside the server's tracking range** so `nearest_player_dist`
  reads its rest value (64) — any row where it does not is refused. The harness schedules the
  teacher's moments on the **clock**, never on the agent's state, and checks each word's arrival.
  Text is **tagged by source** (teacher / system / other players); only the declared source binds.
- **Encoder:** `LinguisticEncoder` warmed up at start; the run **refuses** the silent 384-d hash
  fallback.
- **Separability, offline, before any prereg:** the teacher's strings differ by one word and may land
  in one text cluster — measure it *(superseded: the threshold is pinned at 0.44 in the header, not fitted to this measurement)*.
- **Situations** (readiness per the environment lens):

  | Word | Situation | Exercises | Ready? |
  |---|---|---|---|
  | `water` | drowning | air-hunger drive + learned fear (EARNED, Exp 60/61) | **Yes** — Exp 60/61/62 apparatus; US-free cap 4.40 s, pain edge ≈5.2 s |
  | `food` | food while hungry | learned want + relief | **Usable** |
  | `fire` | fire / lava edge | nociceptive pain — 2S-c's pain | **No** — no fire sensor; burning persists after exit; `doFireTick` unguarded |
  | `cave` | a never-seen place | novelty (2S-c) — the consult trigger | **Partly** — novel only if a high-gain sensor changes, and only once per agent |
  | `night` | night | Exp 62's known boundary (night pool 0.799) | **Partly** — no contrast inside the sealed shell |

  Exp A runs in water only; the others are Exp C material (now [social_referencing.md](../social_referencing.md)) once their prerequisites land.
  Salience stays unmeasured on the loop path (2S-c).
- Re-run `scripts/paired_data_audit.py` unchanged. *Exit:* a lag distribution, a vocabulary count,
  the separability table — recorded.

**L1 — heard text joins the situation (recording-only).** The ONE text encoder on the substrate path;
a heard word joins the situation as the `text` modality with a stated persistence window (text is an
event, the other channels are states) and a holder that carries it to the next substrate tick. Roles
per decision 1. Novelty is reported **per modality** so text neither dominates the 2S-c novelty score
nor silently vanishes from it. Opt-in and declared; Exp 60/61/62's ledger rows gain it as a
**Re-run on** trigger. Depends on R4's look-back design (below). *Guard:* a real-loop test in which a
heard `[minecraft:chat]` percept reaches the proposal's text slot on the substrate path — and does
not reach credit or fear.

**L2 — binding (recording-only).** On each heard word from the declared source, the look-back finds
the sensed situation(s) in the window and records the association **in the episodic trace**; sleep
consolidation strengthens a `HEARD_WITH` relation in the ATL (text concept → world concept; weight,
confidence, provenance) — **one-way, word → situation**. `HEARD_WITH` is registered as a **builtin**
relation type (`Semantics.define` returns False silently for an unknown type and appends duplicate
edges on repeat — both fixed with the registration, and consolidation must not prune a binding before
it is saved). Text centroids drift and share their space with affordance names: the text formation
threshold is its own *(superseded: pinned at 0.44 in the header)*. The look-back is `PerceptTraceBuffer`, **designed in R4's
design review** with three named consumers (R4 delayed credit — the hardest; word binding — ~1 s,
text-triggered, several candidate situations; memory strength's retroactive tagging); R4 builds it,
and if R4 slips this line builds it from R4's reviewed design, never its own. *Measured:* binding
accuracy on held-out pairings (T1) against the nearest-template baseline and a shuffled-binding
control.

**Exp A — does the word work through its situation? (raw: no bundle, no consult).**
Water only (the environment lens: on land the agent's response is a no-op, so the DV is the choice,
not the behaviour).
- **Phase 1, association:** the teacher's word during short submerged dips, each under the US-free
  cap (4.40 s) — the water is still harmless to this agent.
- **Phase 2, fear:** drowning fear is learned **without the word** — the word's persistence window
  has closed before phase 2 starts, and the text slot is verified empty on every phase-2 tick.
- **Phase 3, test:** the word alone, on a dry cell displaced from the flee anchor, at a moment where
  an identity check confirms the sensors do not discriminate (`anticipatory_threat_need == 0` from
  sensed input). Before testing, verify the text cluster holds **no fear of its own**.
- **Arms** (minimum; the prereg fixes n — the confounding lens's floor: Fisher's exact at 12/24 for a
  binary DV, ≥20 per cell for a continuous one): word (associated) · no word · shuffled association ·
  unbound word **matched for novelty** (heard as often, never co-occurring with a situation) ·
  **conditioned-word positive control** (the text cluster declared a fear target, the word present
  during phase 2 — shows the conditioning route is real and what it looks like) · **extinction arm**
  (after phase 2, extinguish the situation's fear, then test the word: retrieval-at-test predicts the
  word's effect drops with it; mediated conditioning predicts it stays — Rizley & Rescorla) ·
  **ablation split by timing** (`HEARD_WITH` removed before phase 2 vs after phase 2: retrieval predicts
  both abolish it; mediated conditioning predicts only the early one does). Harmless pre-exposure to
  water matched across arms (latent inhibition).
- *Claim if EARNED:* the Exp A sentence above.

## Track S — social referencing (SUPERSEDED by [social_referencing.md](../social_referencing.md))

**S1 — the foreign layer, the mirror, and the security prerequisites.**
- The foreign layer (decision 2) with its read path in selection and threat, its weight
  `trust(source) × (1 − own confidence)`, decay, per-consult revert.
- The release mirror (decision 3): background refresh off the loop thread; per-release signature
  verification; the Queen-signed entry index; per-entry journal (so overlapping releases never
  double-count).
- **Security prerequisites, before any Track S code** (the security lens): pulls **never send the
  leader key** to a non-loopback Oasis — the read tier is anonymous and rate-limited, or uses a
  read-only scoped credential; `signer_identity` moves **under** the signature; releases reject
  server-signed slices.

**S2 — the consult trigger (an innate prior — declared tier).**
Consult when **stakes** are high — **drive pressure or pain just felt** (not the agent's own
anticipatory threat: a clear learned danger is a clear read, and clear reads do not consult) — **and**
the agent is **unfamiliar** (novelty from the EC margin) **or has no idea what to do** (no own
fear/want history for this situation, `recommend_action` below `min_confidence`) **or its own
information is failing or stale** (recent negative outcomes here). **Not** a trigger: outcome
variance on its own (irreducible risk; and today's Welford state is per tool). A per-situation
refractory, **and a global consult budget**. The key is the **sensed situation** (world/sensors at
the 0.85 threshold); heard text joins it at J2. The answer: only entries matching the situation,
admitted into the foreign layer. **While waiting** — the mirror is local, so the wait is a local read,
not a network round-trip — the agent keeps its own policy or takes a declared cautious hold (owner
question). On a **first** exposure the consult can only fire at pain (≈5.2 s in water); the measured
outcome is surfacing before damage (≈16 s). Thresholds start hard-coded and are **calibrated on
separate seeds**; per the behaviour-tier rule, a follow-up with a trigger is filed for making them
learned. **Learned trust** — whether a consult's advice paid off updates `trust(source)` — is that
follow-up's first form.

**Exp C — does the gate limit a stale Oasis's damage, beyond consulting less?**
- **Arms:** gated · always · never · **yoked-random** (consults exactly as often as the gated arm, at
  random moments — separates *when* from *how often*) · **content-null** (consults on the gated
  schedule and receives an empty answer — separates the pause from the answer).
- **Oasis conditions:** **clean** · **stale** (entries true in an earlier world, false in this one —
  the critical-learner test) — plus, reported **separately**, **inverted** (valence flipped — a
  deception test, not a Rogers test). Where the corruption sits (which situations) is pre-registered,
  so the result is not decided by placement.
- **Primary test:** the gate × Oasis-condition **interaction**, by permutation. Also reported: damage
  **per corrupted entry admitted**.
- **Isolation (prerequisite for the harness):** the corrupted Oasis is built with a **throwaway
  signing key** in a test-only namespace, in a temporary directory, never on a machine whose key a
  real receiver trusts — a corrupted bundle signed with the Queen key *is* a Queen release.
- **A genuine may-fail:** if gated does not beat yoked-random, that is the recorded result. Night is
  left out of gated arms until Exp 62's night boundary is understood.

## The join

**J1 — the ATL payload (pre-boot transfer).** `HEARD_WITH` relations **only** — never the concepts, which
store the raw heard text as name and definition (a privacy leak); the receiver rebuilds its own
concepts. Both endpoints re-keyed **inside `substrate_merge`** through the aligned-EC id map (the D43
seam rule — the fix belongs in the composition). A relation missing an endpoint is dropped with a
count. The text merge threshold **equals the text formation threshold** (L0). Bundle schema version
bumped and **every declared slice hashed**, so a 1.3.0 receiver neither drops an unsigned ATL slice
silently nor reports a signed one as tampered. A threat-model amendment: a foreign `HEARD_WITH` relation
can point a word at the receiver's own strongest fear — foreign relations enter the foreign layer on a
live consult, and on a pre-boot ingest are discounted and capped. Contract §1's amendment (live
consults admit to the foreign layer; the file ingest stays at-rest) is recorded as a **DECISIONS.md
entry** (the contract doc is archived). Session-end save order and `SessionSnapshot` gaining an EC
kind are fixed here. `_format_version`, CC3 and the hivemind mypy gate apply.

**Exp B — transfer** *(its live arm consults through [social_referencing.md](../social_referencing.md)'s foreign layer; that design governs)*. A donor that passed Exp A publishes; a fresh receiver takes its associations
**pre-boot** (J1's merge) or **live** (a consult into the foreign layer), and hears the word without
the experience. Arms modelled on Exp 56: taught · isolated (same budget) · dangling (associations
without the world EC nodes — must fail) · **association-stripped** · **fear-stripped** (which part
carries it) · naive. Pre-boot vs live tested as **non-inferiority** with matched pre-ingest exposure.
Transport: the real signed path; the harness **asserts** the signature verified (Exp 61's prereg said
signed; its harness passed no `--sign`).

**J2 — heard text in the consult key, and Search Oasis** *(the consult path it rides is specified in [social_referencing.md](../social_referencing.md); only the text key belongs here)*. Heard text joins S2's key (text matched only
against text, at its calibrated threshold). **Search Oasis**: typed text is used as a consult key
(text modality only) through the same path — explicit, capped, trusted signers only, logged,
revertible — and stays a **query only**: it never becomes a percept, a binding or a memory.

**J3 — generalisation (after Exps A and B).** Unheard wordings — T2 paraphrase, T3 **web text** (held-out
TEST only: it has no sensor side, so it can never train an association), T4 invented words (must
fail) — need the 384 ↔ 768 projection and their own prereg with the nearest-template baseline.

**Not in this plan:** production (the substrate saying anything), a public contribution path
(promotion stays WRITE-ONLY), any rung of E1–E3.

## The contract between the two sides (v3 record — its S1/S2 rows are SUPERSEDED by [social_referencing.md](../social_referencing.md), which owns the consult, the foreign layer and the mirror)

| | Language side | Oasis side |
|---|---|---|
| L0–L2, Exp A | capture, timestamps, teacher, text in the situation, binding | — |
| S1 | foreign layer; read path; learned-trust hook | release mirror + Queen-signed entry index + per-entry journal; anonymous/scoped read tier; `signer_identity` under the signature |
| S2, Exp C | the trigger; waiting behaviour | throwaway-key test namespace for the corrupted Oasis |
| J1, Exp B | `HEARD_WITH` shape (defined once, as a builtin ATL relation type) | ATL slice (`HEARD_WITH` only); re-key inside `substrate_merge`; schema bump + per-slice hashes; threat-model amendment |
| J2 | text key; Search Oasis input handling | selector accepts a text-only key |

**Shared, owned by neither alone:** the `HEARD_WITH` shape, the situation-key shape (`{modality: …}` with
embeddings), the look-back (R4), the claim sentences.

## Guards and disciplines

- **A template is a label.** If L0's vocabulary is still templates, claims say "a label".
- **A fix ships with a caller:** each mechanism is exercised by the real loop; gates are strict red
  gates on the real composition.
- **Prove each guard by deleting the mechanism:** remove `HEARD_WITH` → Exp A's effect collapses; disable
  the gate → Exp C's gated arm equals yoked-random; empty the foreign layer → a consult changes
  nothing.
- **Every arm runs with the text channel on** (so the channel itself is not the difference), except
  where an arm's point is its absence.
- **Provenance:** harnesses assert the `maxim` they import is their own repo; gated data commits
  prereg-first on `main`.
- **Rig:** big-mac-mini only, once quiet; the operator starts the Paper server (no model server: no LLM is in this loop).
  Budget: Exp A ≈3–7 h, Exp B ≈2–2.5 h, Exp C ≈3–5 h of rig time.

## Open questions for the owner

*All four were answered 2026-09-24 (owner). Promotion (only if it demonstrably worked), the waiting
behaviour (keep own policy, continuous caution) and the order of work (Track S first, as its own plan)
are now in [social_referencing.md](../social_referencing.md). Situations: the L0 gate covers all five
offline; the graded-drop cliff on a variant body is social_referencing's showcase, and there is no
thermal sensor in Minecraft (survival-ladder D1).*

## Review record (2026-09-24)

Six parallel lenses, all **ADOPT WITH CHANGES**; reviews in
[reviews/grounded_word_binding_demo/](../reviews/grounded_word_binding_demo/):
[confounding](../reviews/grounded_word_binding_demo/confounding.md) ·
[bio-fidelity](../reviews/grounded_word_binding_demo/bio_fidelity.md) ·
[wiring](../reviews/grounded_word_binding_demo/wiring.md) ·
[environment](../reviews/grounded_word_binding_demo/environment.md) ·
[architecture](../reviews/grounded_word_binding_demo/architecture.md) ·
[security](../reviews/grounded_word_binding_demo/security.md).
Folded: Exp A's conditioning route was impossible by construction (confounding + wiring,
cross-confirmed) → conditioned-word positive control; mediated conditioning (bio) → extinction arm +
timing-split ablation; Exp C's dose confound (confounding) → yoked-random + content-null arms,
interaction by permutation, stale separated from inverted; the live merge could not keep "own
experience wins" (security + wiring + architecture, cross-confirmed) → the foreign layer; server-cut
slices broke Queen verification and V8 (architecture + security, cross-confirmed) → local mirror +
entry index, no search endpoint; recalled clusters would corrupt learning (wiring) → three inputs,
three roles; concepts ship raw heard text (architecture) → `HEARD_WITH` only; the leader key leaks on
pulls, `signer_identity` is unsigned, and a corrupted test bundle would be a real Queen release
(security) → S1 prerequisites and Exp C isolation; the teacher moves a world sensor and the lag
capture cannot measure lag (environment) → out-of-range teacher, bridge timestamps; citation
corrections (Sorce's unambiguous condition, Tamis-LeMonda/Adolph 2008, van Bergen 2004, Enquist's
order) and trigger corrections (own anticipatory threat and variance-only are not triggers). The
plan became two tracks and a join (architecture) and is stated as a multi-release line.

## Where this is referenced

[grounded_language_acquisition.md](../grounded_language_acquisition.md) (its concrete near path) ·
[maxim_hivemind.md](../maxim_hivemind.md) (the ATL payload, the foreign layer, the mirror) ·
[public_oasis.md](../public_oasis.md) (a read-only Oasis serves the mirror; not required) ·
[roadmap_1_4.md](../roadmap_1_4.md) §Parallel lines · R4 (owner of the look-back) ·
records: [paired_data_audit_2026-09-20.md](../../experiments/paired_data_audit_2026-09-20.md),
[paired_data_audit_reaudit_2026-09-21.md](../../experiments/paired_data_audit_reaudit_2026-09-21.md).

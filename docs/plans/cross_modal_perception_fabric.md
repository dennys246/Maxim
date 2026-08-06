# Cross-Modal Perception Fabric (1.3 design direction)

**Status:** DESIGN DRAFT — **MAJOR REVISION after a four-lens review round
(2026-08-06)**. Zero code. Four lenses (substrate/credit, persistence/hivemind,
perception/encoder, bio-fidelity/scope) each returned BLOCKING findings; the
corrections are folded below and the open architectural decisions are listed at
the end. **Do not open Stage 1 until the Stage 0 preconditions pass.** Owner-initiated after Exp 49
("thoughts and language don't drive movement — they run in parallel"), grounded
in three things Exp 49 + the 2026-08-05 hardware session MEASURED.
**Target version:** 1.3 (design may start earlier; nothing here is on the 1.1
critical path).
**Owns (proposed):** the *artifact contract* + the extero/intero channel
extension + the sharing rule. Projection TRAINING internals stay owned by
[jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md) — this
plan does not duplicate them; it supplies the artifact shape that plan leaves
open and re-evaluates its revival trigger.
**Companion plans:** [sem_motor_binding.md](sem_motor_binding.md) (Phase 3 is
Stage 1 here) · [jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md)
(Stage 2's training half) · [maxim_hivemind.md](maxim_hivemind.md) (Stage 4's
sharing surface) · [exteroception_interoception_seam.md](exteroception_interoception_seam.md)
(the shipped channel split this extends) · [perception_pipeline_placement.md](perception_pipeline_placement.md)
(where perception stages run).

---

## Motivation — three measured facts, not an aesthetic

1. **Language is the bottleneck in the motor loop.** Exp 49: the substrate
   centered a sound in **4.65 s** vs the LLM's **86.8 s** (~20×), and every
   LLM-arm failure was clock-bound, not capability-bound. The architecture
   already declares three orient vocabularies (deliberative / substrate /
   reflex); only the deliberative one is fully wired on live hardware. **The
   gap is wiring, not design.**
2. **A single azimuth reading cannot resolve the task.** The linear array's
   front/back fold creates a **180° false equilibrium**: a rear source reads as
   its front mirror, so honest credit *punishes* correct turns behind the fold
   (Exp 49: 77 fold-divergent credits; arm C trapped on every far bin while the
   non-credit-following LLM crossed it). No threshold tunes this away.
3. **CONTESTED — the sensor may not express graded magnitude.** The
   2026-08-05 sweep measured a compressed staircase (~0.19/rad, plateaus, ~13°
   sectors). **This contradicts the shipped 2026-07-16 characterization by ~3×
   in gain (0.57/rad, R²=0.9982, ~1° quantization, complete monotonicity, four
   cross-checked measurements) and is a single run taken right after "fixing" a
   version skew.** Most probable reading: the 08-05 sweep is still
   instrument-compromised. See the CONTESTED banner in
   [audio_localization.md](../embodiment/reachy_mini/audio_localization.md).
   **Nothing in this plan may rely on fact 3 until a version-verified re-sweep
   at ≥2 source geometries reconciles it.** Note the conv-net exclusion below
   does NOT depend on fact 3 — its other three disqualifiers stand alone.

Facts 2 and 3 have the **same escape**: use the *difference across a probe
turn*, not the single reading. Fact 2 additionally has a second escape:
**another modality** (vision resolves front from back). Both are what this plan
builds toward.

---

## Front-gate scope pressure (CLAUDE.md Principle 3)

**Question:** does this need to be its own mechanism, or can it ride on
existing infrastructure?

| Need | Existing infrastructure | Verdict |
|---|---|---|
| Per-modality compartmentalized encoding | `ModalityChannel` registry (`_SUBSTRATE_CHANNELS`, agent_loop.py) — a 3-tuple of `(tag, read_values, read_ranges)`; audio + interoception already live | **RIDES.** Vision is one more entry + a state reader. No new type. |
| Action selection from sensory state | `NAc.recommend_action` + `cluster_reward_bias` summed additively across the active `{modality: cluster}` set | **RIDES.** Explicitly binding-free by design ("late convergence at selection, no arbitration"). |
| Credit for a movement | `drive_comfort_progress` + the Phase-2 measured before/after pairs (`measured_drive_transitions`) | **RIDES.** Exp 49 validated the sign accuracy (1.00 / 0.969) with **zero confirmed honesty-gate leaks**, twice audited. |
| Learned movement prediction | `embodiment/cerebellum.py` forward models (`predict`, `observe_from_action`) — the designated, currently-underused slot | **RIDES.** If a learned predictor is wanted, this is its home. |
| Per-body calibration artifact | `spatial/bounds_learner.py` persists to `resolve_user_state("util/learned_bounds.json")` | **RIDES the pattern** (mutable-state layer, per-unit). New artifact, established shape. |
| Cross-modal alignment **sensor↔LANGUAGE** | *Nothing.* `SensorEncoder` 384-dim vs `LinguisticEncoder` 768-dim — cosine undefined between them | **GENUINELY NEW** — already designed as JEPA. **CORRECTION:** `cross_modal_substrate_binding.md` was cancelled by Roy-4 for a **temporal co-activation failure** (priming↔test EC nodes never co-fire), NOT the dim mismatch — that was the later Roy-5a finding. Dim-consistency removes only the algebraic obstacle. |
| Cross-modal alignment **audio↔VISION** | Both would be `SensorEncoder` output at the **same 384 dims** (`encoder.py:495`) | **NO PROJECTION NEEDED.** Cosine is already well-defined. This invalidates the original Stage 3/4 ordering and weakens the JEPA half-fire argument — see Open Decisions. |
| Credit for an epistemic (probe) action | *Nothing appropriate.* NAc's explore bonus keys on **tool identity**, is session-scoped and unpersisted — it rewards an untried tool, not uncertainty reduction | **GENUINELY NEW or an explicit exemption** — see Correction 2. The headline "exactly ONE new mechanism" was FALSE. |
| Refusing an artifact kind at the bundle boundary | `compose_bundle` has a fixed signature, hardcodes two slices, and has **no notion of artifact kind at all** | **GENUINELY NEW** (row 7 below over-claimed). |
| Sharable perception artifacts | `hivemind/bundle.py` + `merge.py` (contract + migration seam + reserved `_*` namespace) | **RIDES**, with one new rule (below). |

**Conclusion (CORRECTED):** **at least three** genuinely-new mechanisms — the
sensor↔language projection (has a plan), probe-credit handling, and bundle-kind
refusal. The original "exactly ONE" claim did not survive review. Everything else is wiring existing surfaces. This plan's
real deliverable is **the artifact contract and the layer split** — the thing
that decides whether perception artifacts are shareable at all.

### Deliberately NOT doing

- **A conv net (or any gradient-trained policy) mapping sensors → movement.**
  Four independent disqualifiers: (a) wrong data shape — DoA is one scalar,
  nothing to convolve, and the az→turn map is ~10 parameters; (b) wrong
  sample regime — a hardware turn costs 1.5–3 s + settle, so a session yields
  low hundreds of labeled turns against the 10⁴–10⁶ a net wants, while
  EC/NAc prototype matching learns from 10–100; (c) it duplicates NAc as a
  second learning system claiming the same job; (d) fact 3 above — the sensor
  can't feed it anyway. **If a learned predictor is ever wanted, it goes in
  the Cerebellum slot, not a parallel net.**
- **A learned space→space transformer for cross-*species* transfer** (robot A's
  vision encoding → robot B's). It needs paired data *spanning both robots*,
  which is strictly harder to obtain than each robot's own alignment — the
  "align the aligners" regress. Deferred until two robots and real paired data
  exist; the contract stamp (below) means it slots in later as a declared
  adapter between two *named* contracts rather than an untyped blob conversion.

---

## Architecture: three layers, deliberately separable

### Layer 0 — act-and-compare as a PERCEPTION stage (revised)

**CORRECTION 1 (bio + substrate + perception lenses, cross-confirmed).** The
first draft put act-and-compare in the POLICY layer ("a policy layer that
consumes differentials"). Biology puts it in PERCEPTION: Wallach (1940) /
Wightman & Kistler (1999) — the auditory system compares sensed ITD change
against an **efference copy of the commanded rotation** to infer hemifield. The
output is a *disambiguated percept*, which then feeds the ordinary
single-reading orienting transform. (Cite the mammalian dynamic-cue
literature, not the barn owl — *Tyto alba* resolves azimuth by ITD alone and
needs no probe.)

This is not taxonomy. Three consequences:

- **Policy-layer differentials orphan the trained policy.** `recommend_action`
  selects on EC clusters from `encode_sensors(azimuth, modality="audio")`;
  changing the policy's input to a differential changes the encoded state
  space and invalidates every bias `maxim substrate merge-nac` imports —
  the same failure `sem_motor_binding.md` decision #1 refused when it declined
  a new `turn_body` tool identity. A perception-layer resolution keeps the
  cluster space, the policy, and Exp 45/46/48's earned results intact.
- **The substrate is provably state-keyed and memoryless.** `propose_via_substrate`
  reads only the current snapshot; `SensorEncoder._last_sensors` exists but is a
  **gate only** (the `min_delta` skip), never encoded. "The reading changed by X
  when I did Y" is currently inexpressible — so the original claim that "the
  perception half already exists" was **false for encoding** (it is true only
  for *credit*, via `measured_drive_transitions`).
- **`min_delta = 0.05`** (~4.5° on `[-1,1]`) suppresses exactly the small
  differentials a probe produces. Probe magnitude must be pre-registered
  against it.

**CORRECTION 2 — the probe/relief credit conflict (cross-confirmed, unmentioned
in the first draft).** A probe deliberately worsens `|az|` to gain information.
`drive_comfort_progress` → `tool_dispatch` takes the SIGN, so the probe books
**−1** on the audio cluster — and because `min_delta` prevents small probes from
changing the cluster, the penalty lands on **the very cluster the probe must be
run from**, decaying only over a 300-tick tau. Building a probe policy on a
sign-of-relief credit path punishes the behaviour it depends on.

Biologically these are different pathways: SC orienting is fast and largely
**not reward-gated**; information-seeking rides a separate signal
(Bromberg-Martin & Hikosaka 2009 — dopaminergic preference for advance
information, independent of reward magnitude). The cheap resolution that rides
shipped infrastructure: mark probes **credit-withheld** via the existing
`drive_credit_withheld` machinery and book value on the *disambiguated turn that
follows*, using the already-shipped delayed-credit path
(`set_pending_operant_action`) and the eligibility/temporal-anchor trace that
`SensorEncoder` already fires on every encode.

### Layer 1 — semantic projection (shared, CONTRACT-keyed)

Encoder-space → shared latent, so two modalities can be compared at all. Its
identity is a **contract**, never a robot or a component name: which encoder
produced the input, which modality/axis, which convention and range. Any body
whose declaration satisfies the contract can load it — compatibility by
*construction*, not by conversion. Training internals: JEPA plan.

### Layer 2 — body calibration (per-unit, NEVER shared)

Gain, asymmetry, zero offset, saturation shape, motor scaling. **The 2026-08-05
staircase IS this artifact.** Per-robot, plausibly per-installation (the
room-vs-sensor question is open pending the second-position sweep). Sharing it
between two units of the same model could actively poison the receiver.

**Why the split is the load-bearing idea:** it makes the hivemind rule crisp —
**share Layer 1, never Layer 2** — instead of "share projections but be
careful."

**Bio support is stronger than first claimed** (bio lens): Knudsen's barn-owl
prism-rearing work shows the ITD→space map is plastic and **instructed by the
visual map**, and that every individual must re-learn its own cue-to-space
calibration on top of a shared topographic space. That is literally
"shared semantic map + per-body calibration" — and it predicts a *direction*
(vision instructs audio calibration, not the reverse), which turns Stage 3 into
a testable claim: *does adding vision reduce fitted azimuth calibration error?*

**CORRECTION 3 — Layer 2 already leaks today, and the falsifier can't see it.**
`compose_bundle` ships the whole `substrate_nodes` slice; an `audio` node's
centroid **is** `_sensor_embed({"azimuth": raw_uncalibrated_value})`. So this
unit's calibration is already inside shared bundles, and the original falsifier
("any Layer-2 *artifact* appearing in a bundle") looks for a file that will
never exist. The rule must be restated as **"share only artifacts whose input
axis is calibration-CORRECTED"**, which forces an ordering the first draft
omitted: **Layer 2 applies BEFORE Layer 1**.

---

## The artifact contract

Weights live in the **mutable state layer** (`resolve_user_state`, `.npy` +
`.json` sidecar per the no-pickle rule), never in component YAML. Components
are declarative operator config — the same layer as `mesh.yml` / `robots.yaml`
/ `config.json`, where **runtime writes are forbidden**. A component may
*declare which contract it satisfies*; it never carries weights or paths.

```json
{
  "_format_version": "1.0",
  "kind": "projection",
  "layer": "semantic",
  "contract": {
    "modality": "audio",
    "axis": "azimuth",
    "range": [-1.0, 1.0],
    "convention": "doa-xvf3800-v1",
    "source_encoder": "sensor-hash-384-v1",
    "target_encoder": "paraphrase-mpnet-768"
  },
  "weights_ref": "audio_azimuth_v1.npy"
}
```

Load rule: **contract match → load; mismatch → refuse LOUDLY.** Never silently
transform. This is the same shape as `check_format_version` and the
`hash_scheme: "stable-sha256-v1"` marker (#446), for the same reason.

**The precedent that matters most:** NAc biases key on EC node ids, so NAc and
EC persist as a **pair** — restoring one without the other leaves biases
dangling on nodes a fresh EC never re-allocates. Projection weights key on an
encoder space in exactly the same way. Same invariant class → reuse the same
solution: paired persistence, a compatibility stamp, refuse-don't-guess.

**Component inheritance comes free:** `ComponentRegistry` already supports
`extends` with deep-merge, so `bodies/reachy_mini_v2 extends bodies/reachy_mini`
inherits the sensor declarations → the contract → projection compatibility,
legibly and with zero coupling between the declarative and learned layers.

---

## Stages (REVISED — Stage 0 added, ordering changed)

| Stage | Content | Gate to proceed |
|---|---|---|
| **0a** | **Reconcile the DoA curve.** Version-verified re-sweep at ≥2 source geometries against the 07-16 protocol. | Either reproduces 0.57/R²≈0.998 (staircase was an artifact) or reproduces the staircase at both geometries. Motivation fact 3 is UNUSABLE until this returns. |
| **0b** | **Cluster-resolution precondition.** Measure how many distinct EC clusters the audio channel actually produces across the working range. Exp 46 already recorded **2 clusters** at every threshold 0.44→0.93 and called it a *perceptual* limit; the perception lens measured **1** under the (contested) staircase. | ≥ as many distinguishable clusters as distinct correct actions. **If this fails, no policy work can succeed regardless of layer** — the levers are Layer-2 rescaling, per-channel `min_delta`/`pattern_threshold`, or the **population readout already earned in Exp 45e** (which this plan failed to cite and which is the shipped answer to exactly this starvation). |
| **0c** | **Co-activation measurement** (`MAXIM_EC_TRACE_ACTIVATIONS=1` + `scripts/analyze_roy_4_coactivation.py`). This is what actually killed `cross_modal_substrate_binding`, and the placement plan makes it mandatory before cross-modal reasoning. | Audio and vision EC nodes co-fire within the binding window at measurable rates. |
| **1** | **Act-and-compare as a percept-resolution stage** (Correction 1) + probe-credit handling (Correction 2). Three-lens design review first. | Fold-resolved azimuth improves far-bin centering **without changing the policy's cluster space** (trained-policy key continuity is part of the gate). |
| **2** | Hardware-faithful sim scenario — **only if 0a says the staircase is real**. Must include the **fold** (the thing that actually breaks credit, omitted from the first draft) and quantize **before** noise. Insertion point: the `az_true → az_read` step in `SimulatedDoAScenario`. Note this is shipped library code with three importers, not harness code. | Reproduces fold-divergent credit. |
| **3** | Vision channel. **NOT "one more entry, no new type"** — see Open Decision B. Blast radius is five sync sites: `_SUBSTRATE_CHANNELS`, `ECConfig.frozen_centroid_modalities` (vision would otherwise fall to the running-mean drift branch), `merge.py`'s duplicate of that set, `_EC_TRACE_MODALITY_TAG_MAP`, and **`tool_dispatch`'s extero credit routing — which picks `AUDIO_TAG` first, so vision-measured relief would silently book to the audio cluster.** Also: the summed cluster term scales ±N with channel count, so `min_confidence` needs recalibration. | Explicit resolution of Open Decision B + credit routing for >1 extero channel. |
| **4** | Artifact contract + sharing rule. Scope is larger than drafted: binary payloads are **impossible today** (text-only `extract_bundle`, no `atomic_write_bytes`, closed `compose_bundle` signature) and it needs a `BUNDLE_SCHEMA_VERSION` bump + registered migration. | Requires 0c + 3. |

**Behavioral re-validation obligations (was missing):** Stage 3 fires Exp 48's
registered `Re-run on:` trigger **verbatim** (ModalityChannel registry,
`recommend_action(current_clusters=)`, `record_outcome(clusters=)` routing) —
an EARNED Tier-1 row that would go `Stale` and block the next release. The ±N
selection shift plausibly also fires Exp 42's confidence-gate trigger. Schedule
the re-runs inside Stage 3, not at the release gate.

---

## Corrections to the artifact contract

- **Stamp the REALIZED encoder state, not its name.** `LinguisticEncoder` at the
  same configured name emits 768-dim real vectors *or* 384-dim bag-of-words
  hashes depending on whether the `semantic` extra is installed. Add
  `embedding_dim` (checked against the actual array) and `using_fallback`.
  Derive both at write time.
- **Stamp the sensor-NAME SET and the normalization mode.** `_sensor_embed`
  sums a SHA-derived basis per sensor *name*, so two bodies both declaring
  `azimuth` produce comparable vectors only if their channel's full name set
  matches; range-aware vs range-blind `_normalize_value` are different
  functions.
- **Add `units`** (or `normalized: true`) — the signed-sensor invariant warns a
  raw-unit range paired with normalized values is worse than the fold.
- **Declarative fields are DERIVED from body YAML at write time**, never
  authored in the artifact, or the placement plan's anti-`AxisSpec` guardrail is
  violated and range/axis gets two sources of truth.
- **Precedent citations were wrong.** `hash_scheme` **warns and continues**; the
  refuse-don't-guess precedents are `ec.py`'s `Unsupported EC version` raise and
  `bundle.py`'s no-migration raise. And `bounds_learner` is precedent for
  **placement only** — it is globally-keyed (not per-unit), non-atomic, carries
  no `_format_version`, and swallows load errors: the new artifact must fix all
  four, and the per-unit key must be part of the path/contract.
- **The real precedent to copy is `hivemind/cli.py`'s policy-meta sidecar**
  (`_meta_sidecar_path` / `_meta_essence`), which compares content and **aborts
  before any mutation** — including the non-obvious detail that it *strips the
  version stamp before comparing*, which a naive dict compare would get wrong.
- **Enforcement belongs in the type:** `compose_bundle(*, projections=...)` as a
  typed keyword that rejects `layer != "semantic"`, plus `extract_bundle`
  rejecting entries absent from `manifest["contents"]` — that cross-check is
  also the **regression guard** for the sharing-rule falsifier.
- **Reserve `validate_projection` + `trusted_sources` now**, the way `merge.py`
  reserved its 1.2 hooks, so poison-resistance doesn't need a signature break.
- **Location:** mutable state layer (`resolve_user_state`), `.npy` + `.json`.
  This **supersedes** the JEPA plan's header (`_data/projections/`, the bundled
  wheel layer) and its `.pt`/pickle format; file that correction against JEPA,
  whose header also points at a `peer/substrate_bundle.py` that does not exist.

---

## Does JEPA's revival trigger fire?

Its registered trigger: *"a 1.1+ iteration surfaces a problem that is
structurally cross-modal AND unsolvable by threshold tuning, AND the Stage 0
paired-data audit confirms sufficient training pairs."*

**REVISED: on current evidence it does NOT fire.** The first draft argued it
half-fired because the fold is cross-modal. But JEPA solves **sensor↔language**
dimensional misalignment, and audio↔vision — if vision is a `ModalityChannel` —
are **both 384-dim `SensorEncoder` output**, where cosine is already defined.
The fold therefore does not motivate JEPA at all. Moreover the real blocker that
cancelled cross-modal binding was **temporal co-activation**, which JEPA does not
address and which Stage 0c must measure first. Recording the *retraction* of the
half-fire claim so the trigger stays honest.

**Thesis boundary to decide explicitly (do not drift):** the headline claim is
cross-session learning *without fine-tuning*. A **fixed pretrained encoder as
input** is already accepted practice (sentence-transformers for text). A
**projection trained on the robot's own paired experience** is a judgment call
— defensible as learned-from-experience, but it *is* gradient descent, and the
JEPA plan currently rejects imported alignment on thesis grounds. A
**gradient-trained policy** clearly breaks it and is excluded above. Write the
decision down before Stage 4.

---

## What would falsify what (revised)

- **Stage 0a re-sweep reproduces 0.57/R²≈0.998** → the staircase was an
  instrument artifact; motivation fact 3 is deleted and the magnitude concerns
  evaporate. (The conv-net exclusion survives on its other three grounds.)
- **Stage 0b finds ≤2 audio clusters** → no policy at any layer can condition on
  direction; the work becomes perceptual resolution (population readout / Layer-2
  rescaling), not policy.
- **Stage 0c finds no audio↔vision co-activation** → cross-modal binding fails
  the same way it did in Roy-4, regardless of dims or pair counts.
- Stage 1 percept-resolution ≈ raw azimuth on far bins → act-and-compare is not
  the answer; the fold needs vision (raises Stage 3) or a reflex scaffold.
- **A shared bundle whose audio centroids encode uncalibrated readings** → the
  sharing rule leaked. Note this is the *real* detector; the original
  "Layer-2 artifact file in a bundle" test was unobservable (Correction 3).

---

## Open decisions (need an owner call before Stage 1)

**A. Where does act-and-compare live?** Correction 1 argues PERCEPTION (keeps
the cluster space and the trained policy; matches the efference-copy
literature). The alternative — a differential channel feeding the policy — is
simpler to wire but changes the encoded state space and orphans Exp 45/46/48
policies. *Recommendation: perception.*

**B. What are vision's floats?** `ModalityChannel.read_values` returns
`dict[str, float]` with **stable names**, but vision's real data is a
variable-cardinality list of detections with unstable `track_id`s. Every
flattening loses something: per-class presence discards azimuth; a single
"detection centroid azimuth" discards identity and reproduces the same 1-scalar
starvation as audio. Either (i) declare a small fixed set of named scalar vision
sensors in body YAML and accept the resolution ceiling, or (ii) accept that a
variable-cardinality modality needs new encoding — **which re-opens the front
gate**. *No recommendation; this is a genuine architectural fork.*

**C. Is probe credit an exemption or a mechanism?** Withholding credit
(Correction 2) rides shipped infrastructure and is cheap. An actual
information-value signal is a second new mechanism and needs its own earning
experiment. *Recommendation: exemption first; revisit if probing fails to
consolidate.*

**D. Thesis boundary** (unchanged from the first draft, still undecided): fixed
pretrained encoder = accepted practice; projection trained on the robot's own
paired experience = judgment call; gradient-trained policy = excluded. Decide
before Stage 4.

---

## Review round

Four-lens design review, 2026-08-06 (substrate/credit · persistence/hivemind ·
perception/encoder · bio-fidelity/scope). All four returned BLOCKING findings;
Corrections 1–3, the front-gate re-verdicts, Stage 0, and the contract
corrections above are the fold. Two findings were **shipped bugs independent of
this plan** and shipped separately: `_cosine` dimension truncation and the
`DEFAULT_FROZEN_CENTROID_MODALITIES` divergence in `hivemind/merge.py`.

**Cycle-divergence judgment (asked explicitly):** Exp 49's H3 was corrected
twice, but the bio/scope lens assessed this as **convergence, not divergence** —
same kind of issue (metric frame-of-reference vs sensor truth), each iteration
narrowing, and the primary criteria PASSED. The trigger does not fire. What
*does* apply is the sibling rule — verify the instrument — which is why Stage 0a
exists.

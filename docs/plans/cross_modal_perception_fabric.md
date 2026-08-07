# Cross-Modal Perception Fabric (1.3 design direction)

**Status:** DESIGN DRAFT, **rev 3** (2026-08-06). Zero code. Rev 1 went through a
four-lens review round (substrate/credit · persistence/hivemind ·
perception/encoder · bio-fidelity/scope); all four returned BLOCKING findings.
Rev 2 folds them AND an owner design pass that **simplified the architecture** —
the converged design needs *fewer* new mechanisms than rev 1, not more.
**Do not open Stage 1 until the Stage 0 preconditions pass.**

**Target version:** 1.3 (design may start earlier; nothing here is on the 1.1
critical path).
**Owns (proposed):** the perception-resolution stage, the foveal binding
convention, the two-level attention convention, and the artifact contract.
**Companion plans:** [three_factor_credit_assignment.md](three_factor_credit_assignment.md)
(the learning-rule stance this plan's binding + calibration work sits inside) ·
[sem_motor_binding.md](sem_motor_binding.md) ·
[archive/cross_modal_substrate_binding.md](archive/cross_modal_substrate_binding.md)
(**this plan proposes its revival** — see "What changed in rev 2") ·
[deferred/jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md)
(**no longer on the near-term path**) ·
[maxim_hivemind.md](maxim_hivemind.md) ·
[exteroception_interoception_seam.md](exteroception_interoception_seam.md) ·
[perception_pipeline_placement.md](perception_pipeline_placement.md).

---

## Motivation — measured facts

1. **Language is the bottleneck in the motor loop.** Exp 49: substrate centered
   in **4.65 s** vs the LLM's **86.8 s** (~20×); every LLM-arm failure was
   clock-bound. Three orient vocabularies are already declared (deliberative /
   substrate / reflex); only the deliberative one is wired on live hardware.
   **The gap is wiring, not design.**
2. **A single azimuth reading cannot resolve the task.** The linear array's
   front/back fold creates a **180° false equilibrium**: honest credit *punishes*
   correct turns behind the fold (Exp 49: 77 fold-divergent credits; arm C
   trapped on every far bin while the non-credit-following LLM crossed it). No
   threshold tunes this away.
3. **CONTESTED — the sensor may not express graded magnitude.** The 2026-08-05
   sweep measured a compressed staircase (~0.19/rad, plateaus, ~13° sectors).
   This **contradicts the shipped 2026-07-16 characterization** (0.57/rad,
   R²=0.9982, ~1° quantization, complete monotonicity, four cross-checked
   measurements) and is a single run taken right after "fixing" a version skew.
   Most probable reading: still instrument-compromised. See the CONTESTED banner
   in [audio_localization.md](../embodiment/reachy_mini/audio_localization.md).
   **Nothing here may rely on fact 3 until Stage 0a reconciles it.** The conv-net
   exclusion below does NOT depend on it.

---

## What changed in rev 2 (the design pass)

Rev 1 framed act-and-compare as a special *probe* action and vision as a third
substrate channel. Both were wrong, and fixing them **removed** mechanisms:

| Rev 1 | Rev 2 | Effect |
|---|---|---|
| A dedicated probe action | **Any modulator firing** yields a (commanded, sensed) pair; perception consumes it generically | No probe class, **no credit exemption needed** — the conflict dissolves rather than being resolved |
| Act-and-compare in the POLICY layer | In the **PERCEPTION** layer | Policy's cluster space unchanged → Exp 45/46/48 trained policies survive |
| Vision as a `ModalityChannel` | Vision as its **own encoder** → EC directly | No named-scalar shape problem, no five sync sites, doesn't fire Exp 48's re-run trigger |
| Identity needs a class vocabulary | Identity **emerges** — orienting canonicalizes location, so correspondence is solved by the body | The variable-cardinality fork disappears |
| JEPA projection for cross-modal | **Hebbian co-activation** (dimension-agnostic) | No projection needed for the near-term goal |

**The one-turn disambiguation result** (why no probe is needed): source at
unfolded bearing θ, head turns left by δ. Front (|θ| ≤ 90°): the folded reading
moves by **−δ/90**. Rear (|θ| > 90°): the fold mirrors it, so it moves by
**+δ/90**. Opposite signs — **the sign of the sensed change relative to the
commanded turn reveals hemifield**. The agent's first *ordinary* turn, credited
normally, already disambiguates.

**The co-activation result** (why JEPA is off the near-term path):
`cross_modal_substrate_binding` was cancelled because Roy-4 found the EC nodes
**never co-fire** — a *temporal* failure, not a dimensional one (the 384/768 gap
is the later Roy-5a finding, and it concerns sensor↔**language**). Orienting
manufactures the missing window: sound → turn → settle → both modalities active
about the same object in the same instant. Hebbian edges compare *nothing*, so
they are dimension-agnostic. Per the dormancy rule, resurrection needs an
experiment that earns it — **Stage 0c is that experiment**.

---

## Representation — what audio and vision actually are

### The neural-network boundary (explicit)

- **Fixed pretrained encoder at the sensory boundary: YES.** Already the house
  pattern — `LinguisticEncoder` *is* a pretrained transformer
  (`paraphrase-mpnet-base-v2`), shipping behind the `semantic` extra, handing its
  output straight to `EC.pattern_complete_or_separate(embedding, modality)`.
  Vision and (later) acoustic encoders are identical in shape.
- **A network doing the LEARNING: NO.** A gradient-trained policy mapping sensors
  → movement is excluded (see "Deliberately NOT doing"). Learning stays EC
  clustering + NAc credit + Hebbian edges.

### Current state (audited, not assumed)

| Stream | Representation today | Reaches substrate? |
|---|---|---|
| Audio **bearing** | one DoA scalar → `SensorEncoder` hash basis (SHA-seeded low/high vectors interpolated by value) → 384-dim → EC `"audio"`, frozen centroid | Yes — but Exp 46 measured only **2 distinguishable clusters** at every threshold 0.44→0.93, and called it a *perceptual* limit |
| Audio **content** | transcription → text → `LinguisticEncoder` → EC `"text"` | **Speech only.** No acoustic encoder exists — no mel/spectrogram/embedding path; the 2-ch 16 kHz stream feeds transcription and nothing else |
| Vision | YOLO detections → DefaultNetwork behaviors | **No.** Nothing visual is encoded into EC at all |

**The consequence that shapes this plan:** the audio modality encodes **WHERE,
not WHAT**. Binding bearing-audio to vision can only learn "a sound at bearing X
goes with the thing at bearing X" — tautological, zero identity content.
Emergent identity requires *content* on both sides.

### Why the 2-cluster ceiling exists — it is a CODE artifact, not a sensor limit

`_sensor_embed` derives two fixed SHA-seeded 384-dim basis vectors per sensor
*name* and interpolates between them by the normalized value. So **every possible
azimuth lies on one line segment** in the embedding space. Two random high-dim
vectors are near-orthogonal, so that segment spans cosine ≈ 0→1, and chopping it
at `pattern_threshold = 0.85` yields 2–3 pieces. Exp 46's "2 clusters at every
threshold 0.44→0.93" is **arithmetic**: one scalar carries one degree of freedom
and the encoder faithfully preserves exactly that. Better acoustics cannot fix it.

**The fix is population coding, on two axes** — the brain's answer to coding a
continuous quantity (tonotopy in the cochlea, place-coded azimuth maps in SC/IC):

- **Bearing:** one scalar → N tuned units. **Already earned on live hardware** —
  [Exp 45e](../experiments/45e_orient_s4_population_readout.md) resolved the
  far-bin cell starvation with a population-vector readout.
- **Content:** nothing → N frequency bands. The cochlea, literally.

Same fix, two axes.

### A filterbank rides `ModalityChannel`; a learned embedding does not

Rev 2 said embeddings cannot ride `ModalityChannel`. That holds for a *learned*
512-dim embedding whose per-dimension names are meaningless — but **a filterbank
is different**: band *k* has stable semantics (a fixed frequency range), so
naming `mel_00 … mel_39` is legitimate, and `_sensor_embed`'s weighted sum over N
named bases is a **random projection** of the spectrum (Johnson–Lindenstrauss:
distances approximately preserved). Cosine in the projected space tracks cosine
between mel vectors.

Two consequences, both good:

1. The cochlear front-end rides the **existing** `SensorEncoder` path — a
   `ModalityChannel` with N named scalars instead of 1. No new encoder type.
2. **It needs no neural network at all** (a gammatone/mel filterbank is DSP), so
   there are no imported weights anywhere in the audio path. Strictly cheaper and
   more thesis-clean than rev 2's pretrained acoustic encoder.

Design parameters to pre-decide: **time summarization** (mean over the utterance
is simplest; onset + sustain as two sub-vectors is more faithful — transient and
steady-state carry different identity information); **loudness normalization**
(or volume dominates and a loud bark clusters with a loud bell); **frozen
centroid** (a dense continuous stream on running-mean centroids is the documented
drift collapse); **band count** (too few reproduces the starvation, too many and
the summed projection washes out — sweep it in Stage 0b); **mel first, gammatone
as the fidelity upgrade**. Band energies are non-negative, so the legacy `[0,1]`
normalization applies with no signed-folding concern.

### Target state

| Stream | Encoder | EC modality | Purpose |
|---|---|---|---|
| Audio bearing | `SensorEncoder` + **population code** (Exp 45e) | `"audio"` | Sensorimotor orient policy |
| Vision foveal | **NEW** fixed pretrained image encoder | `"vision"` | Identity (emergent) |
| Audio content | **NEW — cochlear filterbank (DSP, NO model)** | `"audio_content"` | Identity for non-speech sounds |
| (speech content) | existing transcription → `LinguisticEncoder` | `"text"` | The **Stage 0c shortcut** — zero new encoders |

Both new encoders bypass `ModalityChannel` entirely — they emit embeddings, not
named scalars, and `_sensor_embed` sums a per-*name* SHA basis, so splitting a
512-dim embedding into `v0…v511` would be meaningless. They hand embeddings
straight to EC, exactly as `LinguisticEncoder` does.

---

## Architecture

### A. Perception resolution — the generic modulator-outcome contingency

Every modulator firing produces a sensor delta. Perception compares the sensed
delta against an **efference copy** of the commanded motion and emits a
*fold-resolved azimuth* into the existing `"audio"` channel. The policy's input
and cluster space are **unchanged** — which is why the trained policies survive.

Bio grounding: Wallach (1940) / Wightman & Kistler (1999) — dynamic head-movement
cues resolve front/back in mammals via efference-copy comparison, and the output
is a disambiguated *percept* feeding the ordinary orienting transform. (Cite the
mammalian literature, **not** the barn owl — *Tyto alba* resolves azimuth by ITD
alone and needs no movement.)

Two shipped surfaces this rides:
- `Cerebellum.observe_from_action(...)` already fires on **every** affordance and
  has been accumulating forward models; prediction error is the discriminator.
  **Caveat:** its `predict` read path is **dormant** (one caller, constructed only
  by tests) — un-dormanting carries a Principle-2 earning obligation.
- `measured_drive_transitions` already computes the before/after pairs.

Known constraint: `min_delta = 0.05` (~4.5° on `[-1,1]`) suppresses small deltas
— turn magnitudes must be pre-registered against it.

### B. Emergent identity — correspondence by orienting

The hard part of cross-modal association is normally **correspondence**: given a
sound and N visual things, which one made it? Orienting solves it with the
**body** instead of the perception system — after centering, whatever made the
sound is at a canonical location. "What is at center" *is* the sound's visual
correlate.

Consequences:
- **Fixed-cardinality encoding.** One canonical region, not N detections.
- **Identity is never declared.** Repeated (content, foveal-content) pairs
  pattern-complete into a stable EC cluster; the cluster *is* the identity. No
  class vocabulary, no labels.
- **Vision doubles as the fold veto.** Center on a rear (folded) source and there
  is nothing there — so *absence at center is evidence of rear*. Binding gates on
  visual presence, and the gate failure is informative. This is the bio-correct
  form: SC integration is a **register** operation (out-of-register inputs
  *depress* the response), not an additive vote.

Honest limits: "center" is a coarse region (FOV ~60–70° against DoA resolution
somewhere in 1–13°); the source must outlive the 1.5–3 s turn (bind **after**
settle, never during); and a *systematically* co-occurring distractor will bind —
correctly, since that is a real correlation in the world. Repetition washes out
*uncorrelated* clutter only.

### C. Two-level attention — a convention, not a framework

Salience/novelty apply at **two independent levels**:

- **Level 1 (per-modality source):** how surprising/important is this reading
  within its own stream. Partially exists — `Percept.salience`/`.novelty`,
  `audio_salience`/`audio_novelty` config, `OrientingProfile` thresholds, the DN
  salience network + novelty tracker — but unevenly, and not as a declared
  contract across modalities.
- **Level 2 (multi-modal):** how surprising is this *pairing*. **Not** a function
  of level 1: two individually-familiar signals that have never co-occurred carry
  ~zero unimodal novelty and maximal association novelty — exactly when binding
  should fire hardest. Level 2 is **association surprise**, which is the Hebbian
  learning signal itself, so it comes essentially **free** as the inverse of the
  co-activation edge weight. No parallel mechanism.

**Both levels gate PLASTICITY, never RECOGNITION.** This is load-bearing:
`get_threshold_overrides` returns `node_id → base − reward_bias`, so a per-node
bias makes matching *easier*. Feeding familiarity into that channel is a positive
feedback loop — familiar node → lower threshold → absorbs marginal instances →
more familiar → absorbs everything — i.e. the EC centroid-drift collapse with a
gain term. (`_reward_bias` is clamped `[0, max]` so it only ever *widens*; safe
for reward, which touches few nodes, unsafe for familiarity, which touches every
node and self-reinforces.) Bio agrees: ACh/NE gate **learning rate** under
novelty/salience; they do not lower detection thresholds.

Design rules:
- Every modality source declares salience + novelty on the same `[0,1]` scale
  with stated semantics. **A convention, not a registry/type/config schema** —
  the same discipline as the placement plan's anti-`AxisSpec` guardrail.
- Exactly two consumption points: level 1 gates attention/escalation (existing);
  level 2 gates binding plasticity (new).
- **Composition is explicit:** series for attention, independent for plasticity (a
  low-salience but highly-novel pairing should still bind). Two gates in series
  multiply — the live smoke already showed default 0.5/0.3 weights passing *no*
  gate, so a robot heard sound for 25 minutes and never escalated.
- **Instrument both levels** in one trace event (values + gate outcomes), or a
  binding failure cannot be attributed.

Later payoff: level 2 is the only place **inverse effectiveness** (SC enhancement
largest when unimodal inputs are weakest) can be expressed. Uniform semantics also
make hivemind bundles interpretable without knowing the source modality.

### D. Layer split for shareable artifacts (deferred; substance unchanged)

**Layer 1 semantic** (contract-keyed, shareable) vs **Layer 2 calibration**
(per-unit, never shared). Bio support: Knudsen's barn-owl prism-rearing — the
ITD→space map is plastic and **instructed by the visual map**, and each individual
re-learns its own cue-to-space calibration atop a shared topographic space. That
predicts a *direction* (vision instructs audio calibration), making Stage 3
testable: *does adding vision reduce fitted azimuth calibration error?*

**Correction carried from rev 1:** Layer 2 **already leaks today** —
`compose_bundle` ships the whole `substrate_nodes` slice, and an `"audio"`
centroid *is* `_sensor_embed({"azimuth": raw_uncalibrated_value})`. The rule must
be **"share only artifacts whose input axis is calibration-CORRECTED"**, which
forces an ordering rev 1 omitted: **Layer 2 applies BEFORE Layer 1.** The rev-1
falsifier ("a Layer-2 artifact file in a bundle") looked for a file that will
never exist.

---

## Front-gate scope pressure (CLAUDE.md Principle 3)

| Need | Existing infrastructure | Verdict |
|---|---|---|
| Per-modality compartmentalized encoding | `ModalityChannel` registry; EC modality tags | **RIDES** (unchanged for audio/intero) |
| Handing an embedding to the substrate | `EC.pattern_complete_or_separate(embedding, modality)` — `LinguisticEncoder` is the template | **RIDES** — why vision needs no channel |
| Action selection | `NAc.recommend_action` + additive cluster bias | **RIDES, untouched** — perception-layer resolution keeps the cluster space |
| Credit for a movement | `drive_comfort_progress` + measured pairs | **RIDES** — with no probe class, **no exemption needed** |
| Forward model / prediction error | `Cerebellum` (write path live, **read path dormant**) | **RIDES with a Principle-2 earning obligation** |
| Level-2 novelty | The co-activation edge weight itself | **RIDES — free** |
| Cross-modal binding | `archive/cross_modal_substrate_binding.md` (CANCELLED) | **REVIVAL, gated on Stage 0c earning it** |
| Vision encoder | Nothing | **GENUINELY NEW** (fixed pretrained, sensory boundary) |
| Acoustic content | Nothing | **NEW but DSP-only** (mel/gammatone filterbank), and it **rides `ModalityChannel` + `SensorEncoder`** as N named bands — no model, no new encoder type |
| Bundle artifact-kind refusal | `compose_bundle` has a fixed signature and **no notion of artifact kind** | **GENUINELY NEW** (Stage 4) |

**Count (rev 3):** one new *mechanism* on the near-term path (the vision
encoder), one new *DSP stage* that rides existing infrastructure (the cochlear
filterbank), one revival (binding), one deferred (bundle-kind refusal). Rev 1
claimed "exactly ONE" and was wrong; successive design passes genuinely reduced
it.

### Deliberately NOT doing

- **A gradient-trained policy mapping sensors → movement.** Four independent
  disqualifiers: wrong data shape (DoA is one scalar; the az→turn map is ~10
  parameters); wrong sample regime (a hardware turn costs 1.5–3 s + settle → low
  hundreds of labeled turns per session vs the 10⁴–10⁶ a net wants, while EC/NAc
  prototype matching learns from 10–100); it duplicates NAc; and the sensor may
  not feed it (fact 3). **If a learned predictor is wanted, it goes in the
  Cerebellum slot.**
- **A learned space→space transformer for cross-*species* transfer.** Needs paired
  data spanning both robots — harder to obtain than each robot's own alignment
  (the align-the-aligners regress).
- **Salience/novelty on the recognition threshold.** See §C — positive feedback
  into centroid collapse. Recorded so it is not re-proposed.

---

## Stages

| Stage | Content | Gate |
|---|---|---|
| **0a** | **Reconcile the DoA curve.** Version-verified re-sweep (daemon + SDK versions recorded in the run) at **≥2 source geometries**, against the 07-16 protocol. | Reproduces 0.57/R²≈0.998 → staircase was an artifact, delete it. Reproduces the staircase at both geometries → real, and the next question is what changed on the robot since 07-16. Fact 3 is UNUSABLE until this returns. |
| **0b** | **Cluster-resolution precondition.** Measure distinguishable EC clusters for the audio channel across the working range. | ≥ as many clusters as distinct correct actions. Exp 46 measured **2**, and the cause is now known to be the CODE, not the sensor. Concrete levers, in order: **population coding for bearing** (Exp 45e, already earned), **band count** for the spectral channel, then per-channel `min_delta`/`pattern_threshold`. Sweep band count here — it is the same measurement. |
| **0c** | **THE PIVOTAL TEST — co-activation.** Preferred form is **spectral** (cochlear filterbank → `"audio_content"`), which tests sensory association directly. The **speech path** (transcription → `"text"`, zero new encoders) is the cheap cross-check and fallback if the filterbank slips. Someone says a word off-axis → robot orients → foveal encoder fires on what is centered → does the `"text"` node co-activate with a stable `"vision"` node, repeatably, across sessions? Uses `MAXIM_EC_TRACE_ACTIVATIONS=1` + `scripts/analyze_roy_4_coactivation.py`. **One** new encoder, against content already flowing. | Measurable co-activation above the Roy-4 baseline. This earns the binding-plan revival. **Pre-registered limitation:** text content is *linguistic*, so a pass demonstrates the binding **mechanism**, not yet that identity emerges from raw sensation. |
| **1** | Perception resolution (§A) + fold veto (§B). Three-lens design review first. | Fold-resolved azimuth improves far-bin centering **with the policy's cluster space unchanged** (trained-policy key continuity is part of the gate). |
| **2** | Hardware-faithful sim scenario — **only if 0a says the staircase is real.** Must include the **fold** and quantize **before** noise; insertion point is the `az_true → az_read` step in `SimulatedDoAScenario`. Note: shipped library code with three importers, not harness code. | Reproduces fold-divergent credit. |
| **3** | Foveal vision encoder + orient-windowed binding + the two-level attention convention (§C). | Emergent clusters recur across sessions; binding gated on presence; both attention levels instrumented. |
| **4** | Artifact contract + sharing rule (the acoustic front-end moved forward into 0c/3 — it is DSP, not a model, so it no longer needs deferring). Scope is larger than rev 1 drafted: binary payloads are **impossible today** (text-only `extract_bundle`, no `atomic_write_bytes`, closed `compose_bundle` signature) and it needs a `BUNDLE_SCHEMA_VERSION` bump + registered migration. | Requires 0c + 3. |

**Behavioral re-validation obligations:** rev 1's Stage 3 fired Exp 48's
registered `Re-run on:` trigger verbatim. Rev 2's vision-as-encoder design
**avoids** touching the ModalityChannel registry /
`recommend_action(current_clusters=)` / `record_outcome(clusters=)` routing — so
the trigger should *not* fire. **Verify this explicitly at Stage 3**; if any of
those three surfaces is touched after all, schedule the Exp 48 re-run inside
Stage 3, not at the release gate.

---

## Artifact contract corrections (for Stage 4)

- **Stamp the REALIZED encoder state, not its name.** `LinguisticEncoder` at the
  same configured name emits 768-dim real vectors *or* 384-dim bag-of-words hashes
  depending on whether `semantic` is installed. Add `embedding_dim` (checked
  against the actual array) and `using_fallback`, derived at write time.
- **Stamp the sensor-NAME SET and the normalization mode** — `_sensor_embed` sums
  a SHA basis per name, and range-aware vs range-blind `_normalize_value` are
  different functions.
- **Add `units`** (or `normalized: true`) — the signed-sensor invariant warns that
  a raw-unit range with normalized values is worse than the fold.
- **Declarative fields are DERIVED from body YAML at write time**, never authored
  in the artifact (two sources of truth otherwise).
- **Precedent corrections:** `hash_scheme` **warns and continues**; the
  refuse-don't-guess precedents are `ec.py`'s `Unsupported EC version` raise and
  `bundle.py`'s no-migration raise. `bounds_learner` is precedent for **placement
  only** — globally-keyed (not per-unit), non-atomic, no `_format_version`,
  swallows load errors; the new artifact must fix all four and make the per-unit
  key part of the path.
- **The real precedent to copy** is `hivemind/cli.py`'s policy-meta sidecar
  (`_meta_sidecar_path` / `_meta_essence`), which compares content and **aborts
  before mutation** — including the non-obvious detail that it *strips the version
  stamp before comparing*.
- **Enforcement in the type:** `compose_bundle(*, projections=...)` as a typed
  keyword rejecting `layer != "semantic"`, plus `extract_bundle` rejecting entries
  absent from `manifest["contents"]` — that cross-check is also the regression
  guard for the sharing-rule falsifier.
- **Reserve `validate_projection` + `trusted_sources` now**, as `merge.py`
  reserved its 1.2 hooks.
- **Location:** mutable state layer (`resolve_user_state`), `.npy` + `.json`.
  **Supersedes** the JEPA plan's header (`_data/projections/`, the bundled wheel
  layer) and its `.pt`/pickle format; file that correction against JEPA, whose
  header also points at a `peer/substrate_bundle.py` that does not exist.

---

## What would falsify what

- **0a reproduces 0.57/R²≈0.998** → the staircase was an instrument artifact;
  fact 3 is deleted and the magnitude concerns evaporate (the conv-net exclusion
  survives on its other three grounds).
- **0b finds ≤2 audio clusters** → no policy at any layer can condition on
  direction; the work becomes perceptual resolution (population readout /
  calibration rescaling), not policy.
- **0c finds no co-activation** → orienting does *not* manufacture the binding
  window, the cancelled plan stays cancelled, and emergent identity needs a
  different mechanism. **The single most informative outcome in the plan.**
- **0c passes but Stage 3 clusters do not recur across sessions** → correspondence-
  by-orienting works within a session but the clusters are not stable identities;
  suspect centering precision or persistence, not the binding rule.
- Stage 1 percept-resolution ≈ raw azimuth on far bins → act-and-compare is not
  the answer; the fold needs vision or a reflex scaffold.
- **A shared bundle whose audio centroids encode uncalibrated readings** → the
  sharing rule leaked (the *real* detector; see §D).

---

## Open decisions

**A. Acoustic encoder timing.** 0c uses speech (transcription → `"text"`) to avoid
building one. If 0c passes, does the acoustic encoder come next (generalize to
non-speech) or does the artifact/sharing work? *Recommendation: acoustic encoder —
it completes the capability; sharing is only useful once there is something worth
sharing.*

**B. Does vision need a pretrained encoder at all?** If audio's identity code is
a hand-built cochlear filterbank with no model, the symmetric question is whether
vision could use a hand-built V1-analog (retinotopic patches, oriented filters)
instead of a pretrained network. *Current answer: keep the pretrained encoder —
visual identity is far higher-dimensional than a 40-band spectrum — but this is
an open question, not an assumption.* Rev 3 makes the audio path model-free;
vision is now the only place a pretrained model enters.

**C. Thesis boundary** — **now handled in
[three_factor_credit_assignment.md](three_factor_credit_assignment.md)**, which
proposes the sharper framing: *no backpropagation, no pretrained-weight updates*
(rather than "no gradient descent"), on the grounds that one-layer local updates
need no backward pass and three-factor learning is a defensible position rather
than an absence. Rev 2 removed the projection requirement; rev 3 removed the
acoustic model; vision's pretrained encoder is now the only imported-weight
surface left. **Layer-2 calibration learning is specified there**, including the
prediction-error-not-reward teacher rule and consolidation-window updates.

*(Rev 1's Open Decisions B and C — vision's float shape, and probe-credit as
exemption-vs-mechanism — are CLOSED by the rev 2 design: vision is an encoder, and
there is no probe class.)*

---

## Review round

Four-lens design review, 2026-08-06. All four lenses returned BLOCKING findings;
rev 2 folds them plus an owner design pass. Two findings were **shipped bugs
independent of this plan** and shipped separately (PR #467): `_cosine` dimension
truncation and the `DEFAULT_FROZEN_CENTROID_MODALITIES` divergence in
`hivemind/merge.py`. A third was a factual error in the merged Exp 49 doc
(PR #469); a fourth blocked a contested measurement from being published as fact
(PR #465).

**Cycle-divergence judgment (asked explicitly):** Exp 49's H3 was corrected twice,
but this is **convergence, not divergence** — the same kind of issue (metric
frame-of-reference vs sensor truth), each iteration narrowing, and the primary
criteria PASSED. The trigger does not fire. The sibling rule — *verify the
instrument* — does apply, which is why Stage 0a exists.

# JEPA Cross-Modal Alignment (1.1+ / 1.2 research direction)

> **DEFERRED (2026-07-15 plans audit):** DRAFT, zero code; 1.2+ research direction. Its motivating fact is structural and still true (384-dim SensorEncoder vs 768-dim LinguisticEncoder → cross-modal cosine undefined), and its data-production prereq (roy_5 Stage 3 cradle redesign) HAS shipped — but the demonstrated-need trigger never fired (Exp 35/36 resolved via threshold tuning, not projection). **Revive when:** a 1.1+ iteration surfaces a problem that is structurally cross-modal AND unsolvable by threshold tuning, AND the Stage 0 paired-data audit (~50 LOC) confirms the cradle arc yields sufficient training pairs.


**Status:** **Stage 4b candidate — 1.1+ / 1.2 research direction. Roy-5b's specific Branch B promotion trigger (clean FAIL across the parameter sweep) did not fire, so this plan does NOT promote to "1.2 in flight" today; it stays at its pre-Roy-5b status as a research direction motivated by the dimensional fact below.** Originally DRAFT (2026-05-14, after Roy-5a-substrate-on's dimension-mismatch finding). "Stage 4b candidate" 2026-05-28 post-[exp 34](../../experiments/34_wire_a_post_fix_a_b.md) divergence-in-a-row authorization. The motivating fact — `SensorEncoder` produces 384-dim embeddings, `LinguisticEncoder` produces 768-dim, cross-modality cosine is mathematically undefined across different-dimensional spaces — is structural to having different-dimensional encoders and is independent of any specific Roy iteration outcome. Roy-5b (2026-05-28, [docs/experiments/35_roy_5b.md](../../experiments/35_roy_5b.md)) was Conditional PASS / Ambiguous, not the clean FAIL Branch B promotion required. Roy-5b-confound-isolation (2026-05-29, [docs/experiments/36_roy_5b_confound_isolation.md](../../experiments/36_roy_5b_confound_isolation.md)) attributed Roy-5b's gap closure to the EC drift fix (threshold 0.40 → 0.44). Neither outcome promotes JEPA, neither outcome cancels it — the plan's underlying motivation (cross-modal alignment in different-dimensional spaces) stands on its own. The next experiment that could change JEPA's status is whichever 1.1+ Roy iteration surfaces a problem that's structurally cross-modal AND can't be solved by the threshold-tuning path.
**Target version:** 1.2 implementation; 1.1 design + Stage 3-of-cradle-redesign prerequisite.
**Begins:** earliest, post-`roy_5_encoder_alignment_disambiguator.md` Stage 3 (cradle-arc redesign) ships and produces the paired training data this plan consumes. Not 1.0 work. Not 1.1 critical path.
**Owns (proposed):** `src/maxim/similarity/projection.py` (new), `src/maxim/similarity/encoder.py` (additive — projection-aware `embed`), `src/maxim/training/jepa/` (new — training pipeline + persistence), `scripts/train_jepa_projection.py`, `scripts/eval_jepa_alignment.py`, `_data/projections/` (persisted weights).
**Companion plans:** [roy_5_encoder_alignment_disambiguator.md](../archive/roy_5_encoder_alignment_disambiguator.md) (Stage 3 is the prereq that produces JEPA's training data) · [cross_modal_substrate_binding.md](../archive/cross_modal_substrate_binding.md) (cancelled by Roy-4; **this plan is what Stage 4a's resurrection conditions actually require**) · [grounded_language_acquisition.md](../grounded_language_acquisition.md) (Phase 2's "symbol-binding layer" sketch is structurally a JEPA — this plan provides the architecture Phase 2 currently leaves open) · [maxim_hivemind.md](../maxim_hivemind.md) (if JEPA ships, projection weights become a sharable substrate artifact alongside NAc/EC/ATL)

---

## Front-gate scope pressure (retroactive)

Added 2026-05-27 per CLAUDE.md Principle 3.

**Question:** does this need to be its own mechanism, or can it ride on existing infrastructure?

**Existing infrastructure surveyed:**

| Candidate | Why insufficient |
|---|---|
| `LinguisticEncoder` (768-dim paraphrase-mpnet) | Produces text embeddings only; cannot bridge to sensor-modality 384-dim space. Cross-modality cosine is mathematically undefined across different-dimensional spaces |
| `SensorEncoder` (384-dim hash-basis) | Symmetric problem — interoception-modality only |
| `EC.pattern_complete_or_separate` with `frozen_centroid_modalities` | Operates within-modality on raw encoder output; cannot align two different-dimensional spaces without a projection layer |
| `cross_modal_substrate_binding.md` Hebbian binding edges (CANCELLED) | Roy-4 confirmed binding rule cannot fire across modalities because raw cosine is undefined. Cancellation is exactly the evidence that *some* projection layer is structurally required |
| `ComponentIndex` two-layer discovery (alias + embedding) | Solves within-modality entity name lookup. Wrong abstraction layer for cross-modal alignment |
| `ATL.find_or_create` semantic store | Cross-modality lookup via natural language; bypasses the alignment problem at the symbol level. Useful but doesn't solve the underlying embedding-space gap |
| Pretrained CLIP / ImageBind / multi-modal pretrained projections | Explicitly rejected by the plan's "What this does NOT do" section — thesis depends on alignment being **learned from substrate experience**, not imported |

**Verdict:** yes-it-needs-to-be-its-own. The projection layer is a genuinely new mechanism — no existing surface bridges different-dimensional encoder outputs into a shared latent.

**Specific reason:** Roy-5a-substrate-on's structural finding — `SensorEncoder` (384-dim) and `LinguisticEncoder` (768-dim) live in *different-dimensional* spaces, not just "far apart" — proves the gap cannot be closed by parameter tuning or by adding edges in raw encoder space. A learned projection into a shared K-dim latent (K=256 proposed) is the smallest unit of new code that makes cross-modal binding mathematically defined. The two-headed JEPA architecture is the bio-defensible answer because it lets the projection learn from the substrate's own emergent paired data rather than importing a pretrained alignment.

**Tightly-scoped:** the projection is **additive** (encoder outputs unchanged; existing same-modality call sites untouched). It does NOT replace SensorEncoder or LinguisticEncoder. It does NOT replace Hebbian binding (which can ride on the shared latent if revived). The new-mechanism surface is exactly one MLP per encoder + a training pipeline + a persistence sidecar.

## Why this plan exists

Roy-5a-substrate-on ([docs/experiments/22_roy_5a.md](../../experiments/22_roy_5a.md)) surfaced a structural finding the plan it was meant to resolve did not model:

> `SensorEncoder` produces **384-dim** SHA-basis embeddings for interoception modality; `LinguisticEncoder` (`paraphrase-mpnet-base-v2`) produces **768-dim** embeddings for text modality. Cross-modality cosine is **mathematically undefined** — the vectors live in different dimensional spaces. The plan's "encoder subspaces are far in cosine space" framing of H1a is structurally weaker than the data actually shows: the subspaces aren't far, they're **different-dimensional**, and any cosine-based cross-modal alignment is structurally impossible without a learned projection layer.

This is bigger than Roy-5a. It explains:

- **Why Roy-4 cancelled `cross_modal_substrate_binding.md`.** The Hebbian binding rule that plan proposed bound EC nodes via temporal co-activation, then queried them via cosine. Cross-modality binding via cosine is structurally impossible when the modalities have different embedding dims. Stage 4a's risk register noted "scaffold rescues binding" as the resurrection condition — but even with co-firing data, the cosine math doesn't work across modalities.
- **Why `grounded_language_acquisition.md` Phase 2 leaves the architecture open** ("embedding lookup + small MLP, or a tiny RNN"). The Phase 2 sketch is structurally a small JEPA — token-embedding inputs, EC-node-distribution outputs, trained on co-occurrence. This plan supplies the architecture Phase 2 needs.
- **Why every Roy iteration since Roy-0 reproduces "tool-name + interoception identity survives, cluster-UUID + text-modality identity does not"** (the [feedback_two_identity_schemes.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_two_identity_schemes.md) pattern). The two identity schemes survive disjointly because there's no learned bridge between their embedding spaces.

JEPA (Joint-Embedding Predictive Architecture, LeCun et al.) is the canonical bio-defensible answer: train two encoders' outputs to predict each other in a **shared latent space**. The "joint embedding" replaces "cosine on raw mismatched-dim embeddings"; the "predictive" loss replaces "Hebbian rule on temporal coincidence."

---

## Framing rule

**The training data ships before the architecture does.** JEPA needs paired `(modality_A_input, modality_B_input)` data — for Maxim's case, temporally co-occurring `(sensor pattern, drive state, narrator utterance)` triples. **`roy_5_encoder_alignment_disambiguator.md` Stage 3 (cradle-arc redesign) is the data-production prerequisite.** Without Stage 3, JEPA trains on the current cradle arc's co-occurrence — which Roy-5a-substrate-on showed produces zero food-bearing text-modality centroids. JEPA can only learn what the data shows; this plan does not commit to training on insufficient data.

This framing is mandatory because the alternative (start JEPA training pipeline before Stage 3) burns calendar on infrastructure that needs to be re-trained the moment Stage 3 changes the data distribution.

---

## What this plan does NOT do

Read this section first if any of the following sounds like a useful framing:

- **No central hand-curated `(sensor, word)` lexicon.** Per [feedback_interim_contamination.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_interim_contamination.md) and the two-lens reviews that produced `roy_5_encoder_alignment_disambiguator.md`'s explicit non-introductions: the paired training data MUST come from the substrate's own emergent co-firing during cradle priming. If we ever find ourselves "augmenting" the JEPA training set with curated `(sensor, word)` mappings to make it converge, that's contamination — JEPA degrades into a fancy lexicon. The training pipeline must reject any non-substrate-generated training example; a contamination-detector test fails the build if the training set contains entries without a substrate-provenance tag.
- **No replacement for SensorEncoder or LinguisticEncoder.** JEPA adds a learned projection layer on top of the existing encoders. SensorEncoder still produces 384-dim hash-basis vectors; LinguisticEncoder still produces 768-dim sentence-transformer vectors. The projection maps both into a shared K-dim latent (proposed K=256). Existing encoder call sites are unaffected.
- **No new training-time dependency in 1.0 or default 1.1.** The trained projection is loaded from `~/.maxim/data/projections/{encoder_pair}.pt` if present; absence is a no-op (back to direct cosine in same-modality matrices, no cross-modality alignment). Cradle / Roy / D&D sims continue to work without trained projections.
- **No pretrained-model imports for the projection.** The projection weights come from Roy-priming-derived training data, period. We can use PyTorch + sentence-transformers as runtime infrastructure, but no transferring pretrained CLIP / ImageBind / other multi-modal-alignment weights. The thesis depends on the alignment being learned from the substrate's own experience.
- **No replacement for `cross_modal_substrate_binding.md`'s Hebbian binding rule.** Hebbian binding can still ship as the **lateral connectivity** mechanism inside the shared latent space (which is dimension-consistent, so Hebbian + cosine works there). This plan ships the projection that makes Hebbian binding possible; it does NOT cancel the binding plan further. If both ship, Hebbian binding operates in the JEPA latent, not in raw encoder space.

---

## Architecture

### The projection layer

A small two-headed module that takes one of the existing encoder outputs and projects it into a **shared latent space of dimension K** (proposed K=256, chosen as a Goldilocks value: large enough that 50M+ Roy-priming token-equivalents have room to disambiguate; small enough that training converges in reasonable time on Roy-scale data).

```
SensorEncoder (384d)  ──► [head_sensor: 384 → 256] ──┐
                                                     ├─► shared latent (256d)
LinguisticEncoder (768d) ──► [head_lang: 768 → 256] ─┘
```

Each head is a small MLP — proposed shape `Linear → GELU → Linear → LayerNorm`, total ~200K params per head. Tiny by modern standards; this is intentional. The compute budget is per-Maxim-instance, runnable on a laptop CPU at inference time (the goal is sub-millisecond projection per percept).

The shared latent is what NAc keys for `cluster_reward_bias`, what `EC.pattern_complete_or_separate` operates on, what `cross_modal_substrate_binding.md`'s Hebbian edges live in (if revived). The raw encoder outputs continue to feed the same-modality paths (interoception-only retrieval reads 384-dim; text-only retrieval reads 768-dim; only cross-modal queries route through the projection).

### Training objective

Standard JEPA: given a paired sample `(sensor_t, narrator_t)` co-occurring within a tick window, train the heads so that `head_sensor(sensor_t)` predicts `head_lang(narrator_t)` and vice versa in the shared latent. Loss is cosine distance in the latent + a contrastive term (random negatives from other ticks) to prevent collapse to a single point.

**Pairing rule (load-bearing for thesis purity):** a `(sensor, narrator)` pair is valid for JEPA training iff:
1. Both events occurred within the same tick or adjacent tick (≤ 1s elapsed).
2. The pair has a NAc-attributed reward signal in the same tick window (positive OR negative). This filters spurious co-occurrence: random sensor+narrator co-firings that didn't matter behaviorally are not training data.
3. The pairing's provenance tag (`session_id` + `tick`) is unique per session per pair — no duplicates, no curated cross-session augmentation.

### Persistence + loading

Projection weights persist at `~/.maxim/data/projections/{sensor_encoder_id}-{lang_encoder_id}.pt` (PyTorch state-dict). The filename includes the source encoder identities so swapping encoders invalidates the cached projection automatically. A `_format_version` field in a sidecar `.json` carries the same persistence-contract guarantees as other 1.0 frozen artifacts ([CLAUDE.md](../../../CLAUDE.md) CC1).

`LinguisticEncoder` and `SensorEncoder` gain an optional `projection: Projection | None` constructor arg. When set, their `embed_projected(...)` method returns the projected K-dim latent. The existing `embed(...)` methods are unaffected (return raw encoder dims). Call sites opt in to projection — most existing call sites stay on raw embeddings; cross-modal call sites (the new ones this plan enables) use projection.

`EC.pattern_complete_or_separate` gains an `use_projection: bool = False` parameter. When True AND a projection is wired, cross-modality cosine becomes meaningful (both sides routed through their respective heads). Stage 4 below details the EC integration.

---

## Sizing

| Stage | Item | LOC | Where | Prereq |
|---|---|---|---|---|
| 0 | Data audit — confirm Stage 3 cradle redesign produces sufficient paired training data | ~50 | `scripts/audit_jepa_training_data.py` | Stage 3 of roy_5 shipped |
| 1 | Projection module + persistence | ~250 | `src/maxim/similarity/projection.py` + sidecar JSON | Stage 0 audit clean |
| 2 | Training pipeline + contamination-detector test | ~400 | `src/maxim/training/jepa/` + tests | Stage 1 |
| 3 | Encoder integration (additive — `embed_projected`, `Projection | None` ctor arg) | ~150 | `similarity/encoder.py` | Stage 2 |
| 4 | EC integration — `use_projection` parameter, cross-modal cosine works in latent | ~200 | `similarity/ec.py` | Stage 3 |
| 5 | Roy-5c validation iteration — does projection enable cross-modal recall? | ~50 | `scenarios/roy/roy_5c_*.yaml` + `docs/experiments/23_roy_5c.md` | Stage 4 |
| 6 | Hivemind shareability — projection weights as a sharable substrate artifact | ~150 | `src/maxim/peer/substrate_bundle.py` (extend) | Stage 5 PASS + Hivemind 1.1 ships |
| **Total 1.1 / 1.2 implementation** | | **~1,250** | | |

Estimated calendar: **8-12 weeks** from Stage 3-of-roy-5 ship. Stages 0-4 are deep work; Stage 5 is one Roy iteration session; Stage 6 is gated on Hivemind shipping.

The size puts JEPA in the same magnitude as the **whole 1.0 cleanup wave** (C1-C6 ~1,400 LOC). This is not a small follow-up; it's a research direction's worth of work. This plan exists so future-us doesn't accidentally start the work before Stage 3-of-roy-5 produces the data, AND so 1.0 / 1.1 critical path scoping conversations have a real LOC estimate to push against.

---

## Stage 0 — Training data audit (~50 LOC, days)

**Question:** does the redesigned cradle arc (Stage 3 of roy_5) produce enough `(sensor, drive, narrator)` triples with NAc-attributed reward to train a JEPA projection?

**Setup:** post-hoc analysis script that reads Roy-5b's session_dirs (the Stage 3 validation iteration). For each priming session, count the triples that meet the pairing rule:
1. Sensor event + narrator utterance within ≤ 1s.
2. NAc reward signal in the same tick window.
3. Unique `(session_id, tick)` provenance.

**Pass criteria:** at least **500 unique pairs across the 5 priming stages** (rough heuristic — JEPA-style training needs O(thousands) of pairs for the small heads proposed here, but Roy-scale curriculum compounds across sessions, so 500 per Roy iteration × N iterations = adequate over a Roy-month). Plus: **at least 20 distinct sensor patterns** and **at least 20 distinct narrator utterance shapes** — diversity matters more than raw count, JEPA collapses to a single point if every pair is the same shape.

**Fail criteria:** fewer than 200 unique pairs, OR fewer than 10 distinct sensor patterns / narrator shapes. Diagnosis: Stage 3 didn't produce enough co-firing variety. **No training begins.** Either Stage 3's narrator scaffold needs a richer utterance library, or this plan defers further pending more aggressive cradle curriculum.

**Cost:** zero new sim runs (post-hoc on Roy-5b's existing data). Pure analysis.

**Owns:** `scripts/audit_jepa_training_data.py`, outcome doc `docs/experiments/24_jepa_data_audit.md`.

---

## Stage 1 — Projection module + persistence (~250 LOC)

**Implementation:** `src/maxim/similarity/projection.py` with:

```python
@dataclass(frozen=True)
class ProjectionConfig:
    """Frozen config (CC3 shape-frozen at 1.2)."""
    sensor_input_dim: int = 384
    lang_input_dim: int = 768
    shared_dim: int = 256
    hidden_dim: int = 512
    activation: str = "gelu"  # one of: gelu, relu, tanh
    layer_norm: bool = True

class Projection:
    """Two-headed JEPA projection: per-modality MLP into shared latent."""
    def __init__(self, config: ProjectionConfig, *, weights_path: Path | None = None) -> None: ...
    def project_sensor(self, sensor_emb: ndarray) -> ndarray: ...  # 384 -> 256
    def project_lang(self, lang_emb: ndarray) -> ndarray: ...  # 768 -> 256
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "Projection": ...
```

**Persistence schema** (`{encoder_pair_id}.pt` + sidecar `.json`):

```json
{
  "_format_version": "1.2",
  "encoder_pair_id": "sensor-sha-384__lang-paraphrase-mpnet-768",
  "config": {...},
  "training_provenance": {
    "trained_on_session_ids": ["20260514_..."],
    "training_pair_count": 1247,
    "epochs": 50,
    "final_loss": 0.182,
    "contamination_check_passed": true
  }
}
```

**Frozen contract:** `ProjectionConfig` is shape-frozen at 1.2 per CC3. Adding `shared_dim_override` per-instance would re-open the "two parallel projections incompatible" footgun — pin it.

**Test surface:**
- Unit: round-trip save/load preserves outputs to within float-precision tolerance.
- Unit: missing weights path returns Projection in untrained state; calling `project_sensor` raises (no silent zero-vector fallback).
- Unit: `_format_version` mismatch raises `ValueError` (no silent loads of incompatible versions).

---

## Stage 2 — Training pipeline + contamination guards (~400 LOC)

**The contamination guard is the load-bearing piece**, not the optimization loop. Per Plan-of-record's "what this plan does NOT do" section, training data must originate from substrate co-firing during sim runs. The training pipeline enforces this at multiple layers:

1. **Provenance tag required on every training example.** A dataclass `JepaTrainingPair` carries `session_id`, `tick`, `sensor_event_id`, `narrator_event_id`, `nac_reward_attribution_id`. The training loader rejects pairs missing any field.
2. **Contamination-detector test in CI.** A test in `tests/unit/test_jepa_no_contamination.py` constructs a synthetic curated-pair set (manually-written `(sensor, word)` tuples without provenance) and verifies the training loader's `add_pair` raises. If a contributor adds curated data, the build fails.
3. **No `--manual-pairs` CLI flag.** Period. Even for debugging. If a developer wants to test the training loop with synthetic data, they generate it via a fixture-Roy run; manual curation does not exist as a code path.

**Training loop:** standard JEPA — alternating predictive loss + InfoNCE contrastive loss, AdamW, cosine LR schedule. 50 epochs on the data, batch size 32, ~5 minutes wall on Mac M-series CPU per Roy iteration's worth of pairs. Per-epoch eval on a held-out 10% slice (held out at session level, not pair level — prevent same-session leak into eval).

**Owns:** `src/maxim/training/jepa/__init__.py`, `train.py`, `data.py`, `eval.py`, `scripts/train_jepa_projection.py`.

---

## Stage 3 — Encoder integration (~150 LOC, additive)

**`LinguisticEncoder` and `SensorEncoder` gain optional `projection`:**

```python
def __init__(self, ..., projection: Projection | None = None) -> None:
    self._projection = projection

def embed_projected(self, text_or_sensor: ...) -> ndarray | None:
    """Return K-dim shared-latent embedding. None if no projection wired."""
    if self._projection is None:
        return None
    raw = self.embed(text_or_sensor)
    return self._projection.project_lang(raw)  # or .project_sensor(...)
```

**`MemoryHub` and `bio_stack` wiring:** projection loads from `~/.maxim/data/projections/` if present; absence is a clean no-op (no warnings, no errors — explicit "untrained" state is the default).

**`build_bio_stack(*, projection: Projection | None = None)`** keyword-only parameter, defaulting None. Existing call sites unaffected.

**Test surface:**
- Unit: `embed_projected` returns None when no projection wired (not zero vector, not raise).
- Unit: round-trip — `Projection(load) → embed_projected → project` matches `Projection.project_lang(raw)` directly.
- Integration: `build_bio_stack(projection=projection_loaded_from_disk)` wires the projection through; `embed_projected` works end-to-end.

---

## Stage 4 — EC integration (~200 LOC)

**The actual mechanism that lets cross-modality cosine work.**

`EC.pattern_complete_or_separate(embedding, modality, *, use_projection: bool = False)`:
- When `use_projection=False` (default), behavior is unchanged. Existing call sites unaffected.
- When `use_projection=True`, the input embedding gets projected through the appropriate head before cosine. Stored centroids in the shared-latent modality are compared in K-dim space.

**EC also gains a separate `_substrate_nodes_shared: dict[str, tuple[list[float], str]]`** — substrate nodes in the shared latent. These are distinct from the per-modality `_substrate_nodes` (which stay 384d for interoception, 768d for text). A cross-modal query produces a node in the shared latent that connects to its per-modality node IDs via a small mapping.

**Persistence (extension of PR #248's `aut_ec.json` schema):**
```json
"substrate_nodes_shared": {
  "<shared-uuid>": {
    "embedding": [<K=256 floats>],
    "source_node_ids": {"interoception": "<384d-uuid>", "text": "<768d-uuid>"},
    "count": <int>
  }
}
```

Load path is back-compat: pre-Stage-4 dumps without `substrate_nodes_shared` load cleanly as untrained.

**Frozen contract impact:** additive at 1.2 — same shape as PR #248's wiring. No breaking changes to existing EC consumers.

**Test surface:**
- Unit: round-trip `EC.save()` / `EC.load()` with `substrate_nodes_shared` populated.
- Unit: `pattern_complete_or_separate(..., use_projection=False)` on Stage-4 EC matches pre-Stage-4 EC (no behavior regression).
- Cross-modal: priming food event in interoception modality + narrator utterance "food" in text modality both produce a shared-latent node with `source_node_ids` populated for both modalities.

---

## Stage 5 — Roy-5c validation iteration (~50 LOC)

**The disambiguating experiment.** Same priming + arms + fixture as Roy-5a, but:
1. Stage 3 cradle redesign is active (the data-production prereq).
2. Projection is trained on Stage 0-audit-cleared paired data.
3. The agent runs with `use_projection=True` in `EC.pattern_complete_or_separate`.

**Pass criterion:** Roy-5c's arm A produces **non-zero food-bearing centroids in the shared-latent modality** AND the post-hoc analyzer (extended `analyze_roy_5_cosine_localization.py` with `--use-projection`) reports `max(M_tt food-bearing) ≥ 0.40` in the shared latent. This is the H1c-like outcome the plan's pre-registered decoding was meant to find — JEPA produces it by making cross-modality cosine well-defined.

**Fail criterion:** Roy-5c reproduces the H1a-via-empty-matrix pattern even with the projection wired. Diagnosis: the projection didn't learn (insufficient training data) OR the projection over-fit to priming and doesn't generalize. **Falls through to encoder replacement (a deeper 1.2+ research direction):** swap one of the encoders for an architecture more amenable to cross-modal alignment.

**Owns:** `scenarios/roy/roy_5c_iteration.yaml`, `docs/experiments/25_roy_5c.md`, `--use-projection` flag on the cosine localization analyzer.

---

## Stage 6 — Hivemind shareability (conditional, ~150 LOC)

**Gated on Stage 5 PASS + `maxim_hivemind.md` 1.1 Oasis software shipping.**

The trained projection becomes a sharable substrate artifact alongside NAc/EC/ATL. The `substrate_bundle.py` work scoped in v1_refinement §B5 extends to include `projections/*.pt` in the bundle format. Confidence aggregation across Oases follows the same Bayesian-merge shape as NAc weights: per-projection-pair, weighted by training-data provenance count.

**Why this matters:** without Stage 6, every Maxim instance trains its own projection independently. With Stage 6, projection-quality compounds across the Hivemind — a projection trained on 100 Maxim-years of substrate data is meaningfully better than one trained on a single instance's data, and the Hivemind makes that compounding tractable.

**This stage may never ship** — depends on Hivemind ecosystem viability post-1.1. Listed for completeness; the plan's load-bearing value is Stages 0-5.

---

## Stage 7 — What this plan does NOT enable in 1.0 / 1.1

Per the framing rule and the version-roadmap discipline in [README.md](README.md):

- **1.0 ships without this plan.** 1.0's substrate-attribution claim is satisfied by Wire-A (substrate-annotates-LLM-context) per `release_0_9_1.md`. Cross-modal alignment is not a 1.0 claim; the 1.0 thesis lives at "substrate carries cognition; language is I/O" and Wire-A is the operator-visible answer.
- **1.1 ships without this plan.** 1.1 ships substrate-primary AUT mode + Maxim Oasis + Stage 3 cradle redesign (per `roy_5_encoder_alignment_disambiguator.md`). All three are upstream of JEPA — they make this plan possible without depending on it.
- **1.2 is the earliest version where JEPA could be on the critical path** — and only if Stage 0-5 ship by then.
- **1.1+ research-direction-only landing is acceptable.** If the audit (Stage 0) reveals Stage 3 cradle redesign doesn't produce enough paired data, this plan stays deferred until subsequent Roy iterations + cradle revisions produce sufficient data. No infrastructure premature-builds.

---

## Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| **Contamination via curated training data.** A developer adds a manual `(sensor, word)` augmentation to "help convergence." | **Critical — fails the thesis.** | Contamination-detector test in CI (Stage 2). No `--manual-pairs` CLI surface, period. Per `feedback_interim_contamination.md`. |
| **Insufficient training data from Stage 3 cradle redesign.** Stage 0 audit fails. | High — defers the plan, doesn't kill it. | Loop back to Stage 3 cradle scaffold design with richer utterance library. |
| **Projection over-fits to priming distribution.** Roy-5c arm-A clears but generalization to D&D / arbitrary text fails. | High — JEPA solves Roy-5a but not the thesis. | Eval on out-of-distribution text (a small held-out non-cradle corpus). Pass criterion includes OOD score ≥ 50% of priming-distribution score. |
| **Projection trains but learns spurious correlation** (e.g., narrator utterances always co-occur with sensor "stamina drift" because of how the cradle arc is structured; projection learns "stamina_event → narrator-text" instead of "hunger → 'hungry'"). | Medium — surfaces as noisy substrate behavior post-JEPA. | Pairing rule requires NAc reward attribution in the same tick. Random non-reward-attributed co-firings are not training data. |
| **JEPA dependency creep.** Future "improvements" extend the projection with vision modality, audio modality, ... each adding LOC + training cost. | Medium — scope drift. | Plan is locked at sensor + linguistic only for 1.2. Adding modalities = new plan. |
| **Stage 5 fails AND encoder replacement is also infeasible.** | Low — this is the catastrophic outcome that kills the cross-modal alignment thesis. | If this happens, the 1.0 thesis ("substrate carries cognition; language is I/O") needs to be re-scoped: Maxim works for same-modality cross-session learning, cross-modal grounding remains an open research question. Not catastrophic — it's a finding, document it. |
| **Compute requirements creep.** "Small JEPA" turns into "needs a GPU per Maxim instance." | Low | Projection heads are intentionally tiny (~200K params per head). Roy-scale training fits a laptop CPU at 5 min wall. If the audit shows we need bigger heads, that's a re-scoping conversation, not an unsanctioned model-size creep. |

---

## Relationship to adjacent plans

Captured here so the cross-plan reading order is obvious:

| Plan | Status | Relationship to this plan |
|---|---|---|
| [`roy_5_encoder_alignment_disambiguator.md`](../archive/roy_5_encoder_alignment_disambiguator.md) | Active. Stage 1 verdict: H1a. | **Direct prerequisite.** Stage 3 cradle redesign produces the paired training data this plan consumes. Stage 4b (encoder replacement to 1.2+) is what THIS plan answers — JEPA is the bio-defensible alternative to wholesale encoder replacement. |
| [`cross_modal_substrate_binding.md`](../archive/cross_modal_substrate_binding.md) | Cancelled by Roy-4. | **This plan is what Stage 4a's resurrection actually needs.** Hebbian binding rules can ship as lateral connectivity *inside the shared latent space* (where dim consistency holds). The cancelled-plan's Stage 4a "if Stage 3 PASS, resurrect with corrected scaffold" should be revised to "if Stage 3 PASS + JEPA Stage 5 PASS, resurrect with Hebbian binding in the JEPA latent." |
| [`grounded_language_acquisition.md`](../grounded_language_acquisition.md) Phase 2 | 1.1+ deferred. Sketches a "symbol-binding layer" as "embedding lookup + small MLP, or a tiny RNN." | **This plan provides Phase 2's architecture.** Phase 2's `(token_sequence, scene_entity_id, ec_node_id)` triples are a special case of JEPA's `(modality_A, modality_B)` pairs where one modality is token-level. If both plans ship, Phase 2's symbol-binding registry is a query over JEPA's shared-latent edges. |
| [`grounded_language_acquisition.md`](../grounded_language_acquisition.md) Phase 3 (from-scratch sequence model) | 1.1+ deferred, headline experiment. | **This plan ships the grounding objective Phase 3's `two-objective loss` needs.** Phase 3's "substrate-grounding loss (predict the active EC node ID given the word context)" assumes a learnable mapping between word context and EC node IDs. JEPA's projection is that mapping. |
| [`bio_emergent_persona_foundations.md`](bio_emergent_persona_foundations.md) | 5 wires; mostly 1.0/1.1 ships. | **Mostly orthogonal.** Wire 3 (embodiment-state → action filter) consumes the NAc surface JEPA would query in the shared latent. No direct dependency either way. |
| [`maxim_hivemind.md`](../maxim_hivemind.md) + v1_refinement §B5 | 1.0 ships shareability infrastructure. | **Stage 6 of this plan extends the substrate bundle format to include projections.** Conditional on Stage 5 PASS + Hivemind 1.1 software shipping. |
| [`v1_refinement.md`](../archive/v1_refinement.md) | 1.0 plan. | **JEPA is explicitly NOT 1.0 work.** No section of v1_refinement should depend on JEPA. If a 1.0 task surfaces that needs JEPA, the task itself is mis-scoped for 1.0. |

---

## Cross-cutting: env-var inventory

| Env var | Stage | Default | Purpose |
|---|---|---|---|
| `MAXIM_JEPA_PROJECTION_PATH` | 3 | unset → auto-detect at `~/.maxim/data/projections/` | Explicit path override for the projection bundle. Useful for swapping projections in A/B experiments. |
| `MAXIM_JEPA_USE_PROJECTION` | 4 | unset → False | Toggle whether `EC.pattern_complete_or_separate` routes through the projection. Opt-in until validation clears Stage 5. |
| `MAXIM_JEPA_TRAINING_BATCH_SIZE` | 2 | unset → 32 | Override training batch size (Roy-scale data fits comfortably at 32). |

All paired with autouse conftest scrubs per [`feedback_opt_in_env_in_hot_paths.md`](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_opt_in_env_in_hot_paths.md). Added to CLAUDE.md env-var table at the time Stage 3 actually ships.

---

## Cross-cutting: frozen-contract impact

Per CLAUDE.md CC3 audit rules:

- **`ProjectionConfig` is shape-frozen at 1.2.** New required fields require a major version bump. Optional fields at the end with defaults are non-breaking.
- **`Projection` class API frozen at 1.2** — adding modalities (vision, audio) requires a new class, not a third head on this one.
- **`aut_ec.json`'s new `substrate_nodes_shared` field carries `_format_version: "1.2"`.** Existing pre-Stage-4 dumps load cleanly with `substrate_nodes_shared = {}`.
- **`EC.pattern_complete_or_separate`'s new `use_projection` parameter is keyword-only with a `False` default** — existing positional call sites unaffected.
- **No changes to the persistence-format contract** beyond the additive shared-substrate-nodes extension.

---

## Definition of done (per stage)

- **Stage 0 (mandatory before any other stage):** audit script ships, Roy-5b data audited, outcome doc names PASS / FAIL on the data-sufficiency criterion. PASS triggers Stage 1; FAIL loops back to Roy-5 plan's Stage 3 cradle scaffold revision.
- **Stage 1:** projection module + persistence + 4 unit tests pass. PR opens against main with two-lens review per `feedback_review_before_ship.md`.
- **Stage 2:** training pipeline + contamination-detector test + 8 unit tests pass. Test corpus: synthetic Roy-shaped paired data; contamination test must FAIL the build when manual curation is injected.
- **Stage 3:** encoder integration + 6 unit tests pass + `build_bio_stack(projection=...)` wires end-to-end. Existing encoder call sites unaffected (regression suite green).
- **Stage 4:** EC integration + 8 unit tests pass + cross-modal cosine in shared latent works on synthetic fixtures. `aut_ec.json` round-trip including `substrate_nodes_shared` pinned.
- **Stage 5:** Roy-5c iteration ships, outcome doc names PASS / FAIL on the cross-modal-recall pass criterion.
- **Stage 6 (conditional on Stage 5 PASS + Hivemind shipping):** projection weights flow through the substrate bundle exchange; Bayesian-merge tested across 2+ instances.

---

## What this plan does NOT pre-commit to

- **No commitment to the specific JEPA loss formulation** (predictive + contrastive InfoNCE) over alternatives (BYOL-style asymmetric prediction, MAE-style masked reconstruction). The architecture's claim is "joint embedding with predictive loss"; the exact loss is settled at Stage 2 implementation based on small-scale convergence tests.
- **No commitment to PyTorch over JAX / flax.** PyTorch is the default because the rest of Maxim's training surface (when it ships) is likely PyTorch-flavored. If Stage 2 reveals a substantial compile-time / runtime advantage to JAX, switching is a Stage 2 implementation decision, not a plan-level commitment.
- **No commitment to `K=256` shared latent dim.** Audit-derived; if Stage 0 reveals Roy-scale data supports K=512 without overfitting, the plan tunes up. Below K=64 is a red flag (collapse risk); above K=1024 is overkill for the data volume the plan models.
- **No commitment to ship before 1.2.** The version roadmap in [README.md](README.md) is the binding contract — JEPA's earliest possible landing is 1.2, and only if Stage 0-5 ship by then. The plan exists so the work is scoped when it matures, not to force it into a release window.

---

## Why I am writing this plan now, before the work begins

Per [feedback_audit_before_building.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_audit_before_building.md): three plans were independently sketching versions of "cross-modal alignment learner" without acknowledging each other.

- `cross_modal_substrate_binding.md` Stage 4a wanted a Hebbian rule for cross-modal binding (cancelled by Roy-4 — needs the dim-aligned latent JEPA provides).
- `grounded_language_acquisition.md` Phase 2 sketched a small MLP for token-to-EC binding (would be a one-modality JEPA).
- `roy_5_encoder_alignment_disambiguator.md` Stage 4b deferred "encoder replacement" to 1.2+ research (JEPA is the bio-defensible answer).

Writing this plan now consolidates the three sketches into one design — without committing to implementation. The risk of NOT writing this is that three months from now, someone starts the Phase 2 MLP without seeing that it's the same problem cross_modal_substrate_binding's Stage 4a couldn't solve and roy_5's Stage 4b deferred, and we end up with three half-implementations of the same architecture or worse — three contaminated-by-curated-data half-implementations.

The plan stays DRAFT until [`roy_5_encoder_alignment_disambiguator.md`](../archive/roy_5_encoder_alignment_disambiguator.md) Stage 3 (cradle redesign) ships. At that point, Stage 0 of this plan runs and the design proposes / declines further stages based on the data audit.

---

## References

- [docs/experiments/22_roy_5a.md](../../experiments/22_roy_5a.md) — Roy-5a outcome including the dimension-mismatch finding this plan exists to address.
- [`feedback_two_identity_schemes.md`](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_two_identity_schemes.md) — the cross-Roy pattern (tool-name survives, cluster identity doesn't) that JEPA is the structural fix for.
- [`feedback_interim_contamination.md`](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_interim_contamination.md) — the contamination class the no-curated-pairs rule guards against.
- LeCun, Y. "A Path Towards Autonomous Machine Intelligence" (2022). The JEPA architecture's original framing — read this if implementing Stage 2.
- Assran et al. "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (CVPR 2023). The first concrete JEPA paper; the architectural decisions there carry to Stage 1's projection-head design.

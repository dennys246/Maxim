# Paper outline — Methods & Results (working draft, 2026-08-10)

**Thesis (one sentence):** A persisted, non-parametric reward substrate causally steers a
frozen LLM agent's decisions where the model's priors are weak, fails to override strong
contrary priors, and the transition tracks prior strength.

**Target venue:** TMLR (primary); CoLLAs (conference alternative). Frame as an empirical
paper with a substantial negative-result component — the dominance boundary is a headline,
not a caveat.

**Contributions:**
1. A trajectory-matched counterfactual method for isolating a memory system's causal
   contribution to individual LLM decisions (with/without re-query at temp 0).
2. Pre-registered, counterbalanced causal evidence that substrate steering exists and is
   reward-driven, not token-driven (Exp 44b).
3. A cross-model demonstration that strong contrary priors dominate the substrate — and
   that reasoning can *amplify* the wrong prior (Exp 38/40).
4. Open, provenance-stamped apparatus (campaign runner, gates, manifests).

---

## 3. Methods

### 3.1 System under test (brief)
- Maxim substrate: NAc-style per-cluster reward biases learned from embodied
  drive-relief credit; explicit causal links; no gradient updates, no fine-tuning.
  Bio-mapping stated honestly as FUNCTIONAL (role, not algorithm — no TD).
- Prompt channel: learned biases rendered as a natural-language annotation section
  ("substrate associations from prior experience") composed into the agent's prompt.
- Frozen LLM (qwen2.5-32b-instruct local for confirmatory; others exploratory).

### 3.2 Embodied learning environment
- Cradle world; chilled infant body with an ENTROPIC cold drive (starts 0.6,
  regenerates — sustained warmth-seeking pressure by design).
- Two warmth sources as leak-free neutral twins (`green_flame`/`purple_flame`):
  byte-identical to the LLM except the color word; safety lives only in an invisible
  thermal delta, discoverable only through felt experience.
- Counterbalanced arms: arm A green=safe, arm B purple=safe; **position also
  counterbalanced** (A introduces harm first, B safe first).

### 3.3 Substrate acquisition (learn stage)
- Substrate-primary agent (no LLM in the action path) learns by acting: drive-relief-only
  credit at the record_outcome choke point. Gate: max |cluster bias| ≥ 0.9 (sign- and
  cluster-blind by pre-registration — no direction-conditioned exclusions).

### 3.4 Trajectory-matched counterfactual (the core method)
- Capture: llm-primary agent resumes the learned substrate (`--resume-sim`, decay-tau
  hold); every prompt captured as a (full, ablated) pair — ablated = annotation nulled,
  otherwise byte-identical.
- Offline re-query: both variants at temperature 0, same backend → a changed action
  ("flip") is attributable to the annotation alone.
- Prior-strength estimation: ablated-prompt sampling (8 @ temp 0.7) → action-distribution
  entropy; weak/strong slices.
- Declared caveat: ablated is annotation-free, NOT experience-free (shared trajectory +
  resumed hippocampal episodes in both variants; symmetric, so flips stay attributable).

### 3.5 Exp 44b confirmatory campaign (pre-registered)
- Protocol frozen before confirmatory data: docs/experiments/protocols/exp44b_preregistration.md
  (commit hash = the freeze).
- Primary test: two-sided exact sign test on per-run NET safety direction; unit =
  (arm × seed), n = 20 (10 seeds × 2 arms); α = 0.05. ONE confirmatory test; everything
  else descriptive. Rationale: flips within a run are correlated; the run-level sign test
  is clustering-robust.
- Mechanically enforced validity gates (runner-level, not prose): learn bias ≥ 0.9;
  capture ≥ 5 pairs; annotation_fraction ≥ 0.5 per capture (predicate = the capture
  record's has_cluster_bias, hook-written); env scrub of all experiment toggles;
  phantom-pick exclusion (cross-arc discovery leakage; excluded + counted, never silently
  dropped); per-stage provenance manifest (executed git hash).
- Controls: (a) wrong-content transplant (A's substrate in B's world) with a VOID rule if
  the substrate fails to surface; (b) intrinsic color baseline from the ablated side.
- Amendment rule: structural invalidity only; any amendment demotes to exploratory.

### 3.6 Prior-dominance probe (Exp 38/40 — the false hearth)
- `cradle_false_hearth`: entity named/described as a benign hearth (strong prior: safe)
  whose warm_self is inverted to harm; correction learnable only from the carried
  substrate. Telegraph-denylist enforced (no lexical leak of danger).
- Separate-worlds design (hearth replaces fire_pit; structural-lockstep arcs).
- Models: Sonnet 4.6, GPT-4o, DeepSeek-V3, R1-Distill-Qwen-32B (reasoning axis), plus
  base Qwen2.5-32B (the Goldilocks cell). Substrate ablations included.

### 3.7 Exp 44c — dominance in the counterfactual frame (DRAFTED: companion arms in campaign config)
- Same trajectory-matched counterfactual run on the false-hearth world: substrate carries
  the learned correction; frozen prediction = near-zero corrective flips (dominance).
- Unifies legs 2 and 3 under one measurement; cheap (fixtures + runner exist).
- If run: add as pre-registered companion arm with its own frozen prediction BEFORE
  unblinding 44b.

### 3.8 Supporting system validation (brief)
- No-LLM results establishing the substrate learns discriminative content at all:
  Exp 42 (safe-vs-harm preference 0.98, counterbalanced, no LLM), Exp 48 (operant
  caregiver teaching 0.875 vs 0.448 control, no LLM). Cited as apparatus validation,
  not headline claims.

---

## 4. Results

### 4.1 Exploratory: Exp 44 (labeled exploratory — metrics selected post-pilot)
- Arm A: 100 decisions, 15 flips — 14 toward safe, 0 harm-ward; commitment axis
  unanimous (10/10 observe→warm_self, NET +1.0).
- Arm B (counterbalance): preference FLIPPED to purple 7:2; 4 direct harm→safe
  corrections; commit NET +1.0. Residual: 2 harm-ward flips (B) vs 0 (A) = weak green
  token bias, mostly overridden.
- Role in paper: motivates 44b; establishes the design; explicitly NOT confirmatory.

### 4.2 Confirmatory: Exp 44b  ⟵ [NUMBERS FROM stats.json WHEN CAMPAIGN COMPLETES]
- Primary: sign test — [+X / −Y runs, p = TBD].
- Pooled direction: [S safe vs H harm, p = TBD, Wilson 95% CI = TBD] (descriptive).
- Commitment axis: [TBD].
- Intrinsic color baseline per arm: [TBD] (quantifies the green residual).
- Prior-entropy slices: [TBD] (the within-experiment Goldilocks echo).
- Transplant control: [follows-annotation toward-harm / VOID — report either honestly].
- Phantom exclusions: [count; should be small].
- Gate/provenance report: seeds excluded by learn gate, annotation fractions, manifest
  hash — the "instrument integrity" paragraph.

### 4.3 The dominance boundary: Exp 38/40
- Dominance across all 4 frontier models (60/60 each): agents keep warming the harmful
  hearth; substrate does not fill the gap.
- R1 (reasoning axis): substrate causally load-bearing (ablations drop hearth-warming
  below baseline) yet AMPLIFIES the wrong prior — reasoning does not rescue, it entrenches.
- Exp 40 (Goldilocks cell): at base Qwen32B — the one model with a positive Exp 37
  signal — the counter-prior also returns dominance (interaction +0.04 SD; harmful-hearth
  warming 0.52 vs fresh 0.50). Falsifying the prior collapses the signal even where it
  existed.
- [If 44c runs: corrective-flip rate vs 44b's flip rate — dominance in the same metric.]

### 4.4 The gradient: prior strength gates substrate influence
- Exp 37 cross-model pattern (explicitly exploratory; the amendments disclosed): substrate
  signal only where prior entropy leaves headroom.
- 44b's weak/strong-prior slices as the pre-registered, within-experiment version.
- Falsifiable form for future work: pre-registered held-out-model bucket prediction from
  measured prior entropy (state as the follow-up, or run it — cheap via cached captures).

### 4.5 Limitations & threats to validity (own section, not a paragraph)
- Confirmatory evidence on one model family; cross-model re-queries exploratory.
- Ablated baseline is annotation-free, not experience-free (shared-context caveat).
- Residual color association (quantified by the intrinsic baseline).
- Discovery-enrichment cross-arm leakage (phantom rule; structural fix deferred).
- The instrument-integrity narrative: pilot smoke caught a wrong-body config, a gate
  false-negative (counter nesting), and cross-arc leakage BEFORE the freeze — reported as
  evidence the gates work, and as honest disclosure of harness fragility.
- Bio-mapping is functional, not mechanistic (no TD; docstring-audited).

---

## Evidence map (outline → artifact)
| Section | Artifact |
|---|---|
| 4.1 | docs/experiments/44_substrate_counterfactual.md |
| 4.2 | <campaign>/stats.json + manifest.jsonl + exp44b_preregistration.md (frozen hash) |
| 4.3 | docs/experiments/38_counter_prior_substrate.md, 40_counter_prior_goldilocks.md |
| 4.4 | docs/experiments/37_cross_model_results.md + 44b slices |
| 3.8 | docs/experiments/42_substrate_primary_preference.md (+42b), 48_cradle_mother_seam.md |

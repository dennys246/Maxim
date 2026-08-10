# Exp 44b — Pre-registration + Campaign Runbook (CONFIRMATORY)

**Status:** TEMPLATE — becomes a pre-registration when the gates below are frozen by the
operator and this file + `scripts/exp44/campaign_44b.json` are committed BEFORE the first
confirmatory sub-sim runs. Until that commit exists, any run under this protocol is
exploratory by definition.

**Relation to Exp 44:** [44_substrate_counterfactual.md](../44_substrate_counterfactual.md)
(arm A + counterbalance, 2026-07-28) is hereby classified **exploratory** — its final
metrics were selected after observing pilot data. Exp 44b re-runs the same design at
power, with the metric, test, and predictions frozen here first. This is the
confirmatory half the program has been missing.

## Hypotheses (frozen before run)

- **H1 (primary):** Adding the learned-substrate annotation to an otherwise byte-identical
  prompt shifts the LLM's flipped decisions toward the experientially-safe source.
  Prediction: per-run net safety direction is positive in a majority of runs;
  **two-sided exact sign test, α = 0.05. The unit of analysis is the (arm × seed) run —
  n = 20 units (10 seeds × 2 counterbalanced arms), each contributing one NET sign;
  zero-NET units are dropped (standard sign-test ties handling).** Arms are NOT summed
  into 10 per-seed units — the two readings differ (a color bias cancels within-unit
  under summing but across-units here), so the unit is pinned explicitly (review fold).
  This is the ONLY confirmatory test; everything else is descriptive.
- **H2 (secondary, descriptive):** The commitment axis (observe→warm_self) shows the same
  sign. No α claimed.
- **H3 (control expectation):** The wrong-content transplant arm (A's green-rewarding
  substrate in B's world where green harms) produces flips that FOLLOW the annotation
  (toward-harm by B's ground truth) — establishing that the prompt channel is generic and
  the learned CONTENT is what makes steering safe. Descriptive.
- **Intrinsic-color baseline:** ablated-side P(safe-colored) per arm quantifies the
  residual color preference the pilot saw (arm B 2 harm-ward vs arm A 0).

## Design (Exp 44's design at power, with four declared deltas)

Declared deltas from the exploratory Exp 44 runs (each a validity improvement, none
data-dependent): (1) capture runs use `seed + 1000` — decorrelates capture-world noise
from the learn pass; the counterfactual compares within-capture so comparability is
unaffected; (2) capture `max_turns` 40; (3) **arm B's arc introduces the SAFE source
first** (arm A introduces harm first) — position is now counterbalanced across arms,
mirroring the Exp 42 pair's structure; without this, a "safe is always second"
positional preference could mimic content-following in both arms (review fold);
(4) the campaign runner scrubs all experiment env toggles and pins each stage's env
explicitly, so the full-vs-ablated delta is the cluster-bias annotation alone
(`MAXIM_ENABLE_BODY_STATE_PROMPT` stays OFF; one leaked toggle could otherwise fake a
confirmatory null).

Counterbalanced pair `cradle_pref_neutral` (green=safe) / `cradle_pref_neutral_b`
(purple=safe), leak-free names. Per (arm × seed): substrate-primary LEARN (gate:
max |cluster_reward_bias| ≥ 0.9, else seed excluded and logged — exclusion is
learning-failure, decided BEFORE any counterfactual data exists) → llm-primary CAPTURE
with `--resume-sim` + decay-tau hold (1000) → offline temp-0 REQUERY of full vs ablated
prompts → stats. **Seeds: 10 per confirmatory arm** (pilot: ~10-15 flips per 1-seed arm →
expected ≥100 flips/arm pooled; the primary test needs only per-seed signs, so power is
driven by seed count — 10+10 seeds with the pilot's effect direction reaches p < 0.001
under the sign test if the direction holds; a 50/50 split is a clean negative).

**Unit of analysis:** the SEED, not the flip. Flips within a run share a trajectory and
a substrate — pooling them as independent overstates N. The pooled binomial + Wilson CI
are reported as descriptive support only.

**Multiplicity:** one confirmatory test (H1), one α. Cross-model re-queries (if run) are
exploratory unless a model-specific prediction is frozen here first, in which case Holm
correction applies across models.

## Frozen analysis parameters

| Parameter | Value |
|---|---|
| Primary test | two-sided exact sign test, per-seed NET direction, arms A+B pooled |
| α | 0.05 |
| Direction scoring | `safety_rank` (analyze_counterfactual.py), arm's ground-truth substrs |
| Entropy slice threshold | 0.5 bits (descriptive only) |
| Learn gate | max abs cluster_reward_bias ≥ 0.9 |
| Capture gate | ≥ 5 paired prompts |
| Decay tau | 1000 |
| Requery decoding | temp 0.0, both variants, same backend |

**Annotation-presence gates (mechanically enforced by the runner, frozen predicate):**
the predicate is the capture record's `has_cluster_bias` field (written by
`capture_paired_prompts.py` at capture time — not a post-hoc grep pattern):

- **Confirmatory arms:** each capture requires `annotation_fraction ≥ 0.5` or the seed
  FAILS loud (harness failure, decided before any counterfactual data exists). A
  substrate-carrying capture whose prompts don't carry the annotation is a broken
  instrument, not a null result.
- **Transplant control:** `annotation_fraction < 0.5` writes a `control_void.json`
  marker; stats reports the cell as **VOID** (not silently dropped, never pooled).
  Cross-arc bias surfacing rides EC pattern completion across the `_b` name suffix and
  is UNVERIFIED; if the gate voids, a name-matched swapped-safety arc is the follow-up
  fixture work.

**Phantom-pick exclusion (frozen pre-scoring; pilot finding 2026-08-10):** the
component-index discovery enrichment is registry-wide, so the counterbalance twins leak
into the OTHER arm's prompts as discovery hints (observed: `green_flame_b` listed in an
arm-A capture). The names carry no safety information and no such tool exists in the
arm's scene — but a temp-0 re-query could still EMIT a phantom action name, and naive
substring ranking would mis-score it (`"green_flame" in "green_flame_b_warm_self"`).
Rule: a decision whose full OR ablated action references a flame entity not in the arm's
`world_entities` (longest-name-first matching) is EXCLUDED from all scoring, counted as
`n_phantom_excluded`, and reported per run — never silently dropped. This rule is frozen
before any re-query results have been scored. A structural fix (scene-scoped discovery
enrichment for controlled arcs) is candidate follow-up src work, not a campaign patch.

**Learn gate is deliberately sign- and cluster-blind** (max |bias| ≥ 0.9 on any
cluster): no direction-conditioned exclusions — a learn run that somehow biased the
wrong source still enters, and the primary test absorbs it. Do not "improve" this into
a safe-cluster-only gate; that is a forking path.

**Known shared-context caveat (declared, not a gate):** `--resume-sim` injects the
learn session's report summary (and restores hippocampal episodes) into the capture
run — shared by BOTH prompt variants, so flips remain attributable to the annotation
alone, but the ablated variant is "annotation-free", not "experience-free". The pilot
inspection step below includes reading one captured prompt pair end-to-end to bound
what the resume context actually says about the flames.

## §44c — Companion arms: prior strength in the counterfactual frame

Three additional pre-registered arms extend the campaign into a **prior-strength
dose-response**, all measured with the same per-decision flip metric. They are pool
`companion`: reported separately with their own frozen predictions, **never pooled into
the H1 primary test**, no α claimed — the confirmatory burden stays on H1 alone.

**C_collision_hearth (the dominance leg, Exp 38 counterfactual-ized).** World =
`cradle_deceptive` (Exp 38's false hearth: name/description read as a benign warm fire —
strong SAFE prior — but `warm_self` breaches the thermal band; telegraph-denylist
enforced). Learn stage runs in `cradle_prelinguistic_deceptive` (substrate-primary,
LLM-free) so the aversion is acquired purely from embodied pain; capture resumes it in
the narrated world.
- **Aversion-calibrated learn gate (frozen now, before any 44c data): max |cluster
  bias| ≥ 0.15.** Rationale: aversive substrates are small in magnitude — the agent
  avoids the hearth after first pain, so it accumulates far less signal than the 44b
  attraction (+0.997); Exp 42's harm-side magnitudes ran ≈0.25. The gate stays sign- and
  cluster-blind (same anti-forking rule as 44b's).
- **Metric:** corrective shift rate among flips = fraction whose full-variant action
  reduces hearth commitment (commit delta < 0) or leaves the hearth for a non-hearth
  action (safety delta > 0). One-entity world → the commitment axis carries the signal;
  `safe_substr` is a never-matching sentinel by design.
- **Frozen prediction (dominance):** corrective shift rate ≪ 44b's toward-safe flip
  rate; consistent with ≈0. If instead corrective flips are common, dominance FAILS to
  replicate in the causal frame — reported either way.

**D_hearth_green_safe / D_hearth_purple_safe (the stretch dose point).** Hearth-twin
pair: leak-free within-pair discrimination exactly like 44b's flames, but the base noun
"hearth" carries a benign CATEGORY-level prior. Color and position counterbalanced.
5 seeds/arm. **Declared stretch:** run if pilot economics allow; skipping is a scope
decision, not an amendment.
- **Frozen prediction (gradient):** toward-safe flip rate BELOW 44b's (the reassuring
  category prior suppresses within-category discrimination) and ABOVE C's corrective
  rate — i.e. the three pools order monotonically: **44b flames > D hearth twins >
  C false hearth.** This converts the Goldilocks claim into a pre-registered
  monotonicity prediction on one axis, one metric.

Phantom-pick exclusion applies across ALL arms with the extended entity list (hearth
twins + bare `hearth`, longest-name-first so `hearth` cannot shadow its twins).

## Amendment rule

Amendments after first confirmatory data are permitted only for *structural invalidity*
(harness bug, degenerate metric) — never for effect size — and every amendment demotes
the affected claim back to exploratory unless re-run fresh. (The literal-vs-structural
pre-registration discipline, applied with the Exp 37 lesson in hand.)

## Runbook — second machine (unified-memory box)

The campaign is self-contained under its own `MAXIM_DATA_HOME`s and touches nothing in
`~/.maxim` except the shared model cache (symlinked read-only) — safe to run while the
1.1 walk runs elsewhere. **Never run it on the leader** (Exp 37 cascade lesson).

Operator notes (review folds):
- **The body MUST be `bodies/infant_humanoid_chilled`** (the config's `embodiment`; also
  the Exp 42 harness default). The plain `infant_humanoid` has NO `cold` sensor — the
  flames' `self_effect: cold: -0.3` silently no-ops, no cold pressure ever develops, and
  under drive-relief-only credit the learn stage can never pass the bias gate (found by
  the first pilot smoke: bias 0.0 at tick 2000+, every drive at set point, drift +0%).
  The chilled body's `cold` is ENTROPIC (starts 0.6, regenerates) — load-bearing per its
  own docstring: a homeostatic cold satiates after a few warms and floors the metric
  (the Exp 41 trap). Symptom signature if wrong: `[NAc_RECOMMEND] passed_gate=False,
  cluster_reward_bias_consulted=0.0` forever + no `drive:cold` line in the drive dump.
- **`maxim config set llm.profile` + `llm.n_ctx` alignment is MANDATORY** —
  `~/.config/maxim/config.json` is NOT redirected by `MAXIM_DATA_HOME`, so sub-sims
  inherit it; a profile-default budgeter vs served-n_ctx drift silently `down_500`s the
  capture stage (the documented open n_ctx leg-3 cross-process case).
- The pre-registered config pins `narrator_profile` so the learn narrator can't float
  on whatever the operator's config.json defaults to.
- **Cross-model sweeps:** kill/swap the llama-cpp server before re-querying with a
  different model — the singleton reuse guard fails loud (correctly) on a live server
  serving another model.
- **After any capture timeout:** check for an orphaned llama-cpp on :8100
  (`ps aux | grep llama_cpp`) before re-running — the timeout kills only the direct
  child, and an orphan poisons subsequent runs (Exp 37 collision class).
- **Pilot inspection step (pre-freeze):** read ONE captured pair end-to-end
  (`prompt_full` vs `prompt_ablated` in capture.jsonl): confirm the only delta is the
  cluster-bias annotation, and note what the resume-context summary says about the
  flames (shared-context caveat above). Check `annotation_fraction` in manifest.jsonl.

```bash
# one-time setup
git clone <repo> ~/Maxim-exp44b && cd ~/Maxim-exp44b
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[llm-llama,llm-server,semantic]"
export PYTHONPATH="$HOME/Maxim-exp44b/src"        # absolute, own line (Exp 42b)
maxim config set llm.profile qwen2.5-32b-instruct  # single source of truth
maxim config set llm.n_ctx 16384
maxim doctor 2>/dev/null | grep -i "n_ctx\|profile\|vram_context_fit"  # verify FIT

# the campaign (resumable — re-running skips completed stages)
python scripts/exp44/campaign.py \
  --config scripts/exp44/campaign_44b.json \
  --workdir data/exp44b/confirmatory_run1

# preview the full plan without spending compute
python scripts/exp44/campaign.py --config scripts/exp44/campaign_44b.json \
  --workdir /tmp/preview --dry-run

# cross-model sweep later, re-using every cached capture (no sim re-runs):
python scripts/exp44/campaign.py --config scripts/exp44/campaign_44b.json \
  --workdir data/exp44b/confirmatory_run1 --requery-models <other-profile>
```

Provenance: the runner refuses to start (exit 3) if the `maxim` on PATH does not import
from this checkout, and stamps `executed_git_hash` into `manifest.jsonl` per stage.
Before analysis: `git status` must be clean and the manifest hash must match the frozen
pre-registration commit.

## Sign-off (operator fills before first confirmatory run)

- [ ] Gates + parameters above reviewed and FROZEN
- [ ] Pilot (1 seed/arm) ran end-to-end; transplant validity gate checked
- [ ] This file + campaign_44b.json committed; hash: `________`
- [ ] Confirmatory campaign started: date `________`, machine `________`

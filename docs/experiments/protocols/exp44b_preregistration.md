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
  Prediction: per-seed net safety direction is positive in a majority of seeds;
  **two-sided exact sign test across seeds, α = 0.05**, pooled over the counterbalanced
  pair (arms A + B). This is the ONLY confirmatory test; everything else is descriptive.
- **H2 (secondary, descriptive):** The commitment axis (observe→warm_self) shows the same
  sign. No α claimed.
- **H3 (control expectation):** The wrong-content transplant arm (A's green-rewarding
  substrate in B's world where green harms) produces flips that FOLLOW the annotation
  (toward-harm by B's ground truth) — establishing that the prompt channel is generic and
  the learned CONTENT is what makes steering safe. Descriptive.
- **Intrinsic-color baseline:** ablated-side P(safe-colored) per arm quantifies the
  residual color preference the pilot saw (arm B 2 harm-ward vs arm A 0).

## Design (identical to Exp 44; only N and the frozen test change)

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

**Transplant-control validity gate (decided pre-analysis):** the control is only
interpretable if the transplanted substrate actually surfaces — ≥50% of the control arm's
captured full-prompts must contain a substrate annotation (grep the capture JSONL).
Cross-arc bias surfacing rides EC pattern completion across the `_b` name suffix and is
UNVERIFIED; if the gate fails, the control is VOID (reported as such, not silently
dropped) and a name-matched swapped-safety arc is the follow-up fixture work.

## Amendment rule

Amendments after first confirmatory data are permitted only for *structural invalidity*
(harness bug, degenerate metric) — never for effect size — and every amendment demotes
the affected claim back to exploratory unless re-run fresh. (The literal-vs-structural
pre-registration discipline, applied with the Exp 37 lesson in hand.)

## Runbook — second machine (unified-memory box)

The campaign is self-contained under its own `MAXIM_DATA_HOME`s and touches nothing in
`~/.maxim` except the shared model cache (symlinked read-only) — safe to run while the
1.1 walk runs elsewhere. **Never run it on the leader** (Exp 37 cascade lesson).

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

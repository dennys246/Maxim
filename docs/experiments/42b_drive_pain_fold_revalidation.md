# Exp 42b — re-validation after the channel-split drive-pain fold

**Status:** Runbook, not fired. Outstanding validation item for
[transition_based_drive_pain.md](../plans/deferred/transition_based_drive_pain.md)
Phase 2 (shipped in PR #435 / `feat/transition-drive-pain`).

**This is a replication, not a new experiment.** The design, metrics, thresholds
(`safe_pref >= 0.66`, `K = 10`), verdict matrix and arcs are those frozen in
[42_substrate_primary_preference.md](42_substrate_primary_preference.md) and are
**not re-tuned** — that is the whole point. Any post-hoc threshold change here
would void the comparison (the Exp 41 / `sharp_rock` cautionary tale).

## Why this run is required

The fold changed `Body.evaluate_failures`, which is shared embodiment code that
Exp 42's `GRADUATE #6` depends on. Specifically:

- **Channel 1** (returned `FailureEvent`s → `side_effects` → `ToolPainBridge`)
  is **unchanged** vs the original run: still state-based, still filtered by B8.
  This is the channel Exp 42's discrimination rides on, so the *expectation* is
  that the verdict is unchanged.
- **Channel 2** (`_publish_drive_pain` → PainBus → `create_pain_nac_subscriber`,
  hippocampus capture, Wire-2 valence, `PainCircuitBridge`) is now
  severity-latched. Exp 42's original writeup notes channel 2 "does not
  re-pollute in Exp 42", so this *should* be behaviourally quiet here — but that
  observation rested on undocumented `_context_similarity` scores, which is
  exactly the fragile coupling this run exists to stop trusting.

The pre-merge two-lens round already proved that an earlier version of the fold
**silently inverted** this experiment (the harmful hearth booked positive credit
from its second contact onward) while every unit test stayed green. So the unit
suite is necessary but not sufficient: **only this behavioural run can close the
item.**

## Prerequisites

- Run from the branch/worktree containing the fold; record the git hash.
- **Do not co-locate with a leader / another experiment** (the Exp 37 cradle
  cascade lesson) — the sub-sims spawn their own LLM consumers.
- Narrator model `smollm-1.7b-instruct` must be available locally; the AUT is
  LLM-free (substrate-primary), so `cost=$0` and the narrator is the only LLM.
- Align `config.json` before firing (the run-config single-source rule):
  ```bash
  maxim config set llm.profile smollm-1.7b-instruct
  maxim doctor 2>/dev/null | grep -i "n_ctx\|profile"
  ```

## Commands

Output paths are **deliberately new** (`42b_*`). `analyze_exp42_preference.py`
**overwrites everything below the analyzer marker** in whatever `--out` doc it
is given — pointing it at `42_substrate_primary_preference.md` would destroy the
original GRADUATE record. Always write to this doc instead.

```bash
cd <worktree-with-the-fold>
export PYTHONPATH=src            # worktree runs need this

# ── Arm 1: treatment (exploration + drive-gating ON) — the frozen main arm ──
python scripts/benchmark_exp42_preference.py \
  --arms cradle_pref_a,cradle_pref_b --trials 10 --seed-base 42 --sim-max-turns 40 \
  --explore-weight 1.5 --embodiment bodies/infant_humanoid_chilled \
  --out docs/experiments/data/42b_results.jsonl

# ── Arm 2: gating-OFF ablation (B7 disabled) — the arm that proved B8 carries it ──
MAXIM_SIM_DRIVE_GATE_ENABLED=0 python scripts/benchmark_exp42_preference.py \
  --arms cradle_pref_a,cradle_pref_b --trials 10 --seed-base 42 --sim-max-turns 40 \
  --explore-weight 1.5 --embodiment bodies/infant_humanoid_chilled \
  --out docs/experiments/data/42b_results_gateoff.jsonl

# ── Analyze (writes into THIS doc, below the marker) ──
python scripts/analyze_exp42_preference.py \
  --in docs/experiments/data/42b_results.jsonl --trials 10 \
  --heading-suffix "treatment (post-fold)" \
  --out docs/experiments/42b_drive_pain_fold_revalidation.md

python scripts/analyze_exp42_preference.py \
  --in docs/experiments/data/42b_results_gateoff.jsonl --trials 10 \
  --heading-suffix "gating-OFF ablation (post-fold)"
  # NOTE: no --out on the second call — it would overwrite the first result.
  # Paste its stdout under the first block, or run it first and copy.
```

Both harness invocations support `--resume` (skips `(arm, seed)` pairs already
in `--out`), so an interrupted run is restartable without re-burning seeds.

**Smoke first** (one arm, one seed, ~2 min) before committing to 40 sub-sims:

```bash
MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT=1.5 MAXIM_SIM_DRIVE_GATE_ENABLED=1 \
maxim --sim cradle_pref_a --aut-mode substrate-primary \
  --embodiment bodies/infant_humanoid_chilled --interactive false --sim-max-turns 40
```

## Pass criteria (frozen — copied, not re-derived)

| # | Criterion | Original result |
|---|---|---|
| H1 | `safe_pref >= 0.66` in **both** arms | A 0.984 / B 0.975 (treatment) |
| C1 | identity-flip: safe identity swaps with the arm | +0.959 PASS |
| C2 | harm net < safe net (learning signs) | PASS |
| K | >= 10 exploitation choices/seed, 0 floored | 10/10 valid both arms |

**CLOSE the item iff** both arms still `GRADUATE` (exit 0) on the treatment run
**and** the gating-OFF ablation still graduates. `safe_pref` need not match to
three decimals — the claim is the verdict plus the sign structure, not the exact
mean.

**If it does NOT graduate:** do **not** revert the fold reflexively — reverting
re-introduces both the per-tick pain flood *and* (if reverted to the first
design) the sign inversion. Diagnose which channel moved: check whether the
harmful source's per-source net went positive (channel-1/B8 regression → a real
bug in the fold) or whether only the volume/variance changed (channel-2 density
→ expected, re-baseline). Record the finding here either way; a `Broken` entry
blocks the next release per the graduation-candidates discipline.

## Companion check — SCN oscillator cold-start floor

The same fold changed drive `TemporalEvent` density from per-tick to
per-episode. `OscillatorNetwork.predict_imminence` has a hard `< 3 observations`
cold-start guard, so `anticipatory_pre_activate` can silently return `0.0` for
drive event types if a run yields fewer than three genuine breach **episodes**.

```bash
# Offline A/B — quantifies the density change, no LLM/robot needed:
PYTHONPATH=src python scripts/check_oscillator_coldstart.py --simulate

# Against a real session from the run above (preferred):
PYTHONPATH=src python scripts/check_oscillator_coldstart.py --session ~/.maxim/sessions/<session-id>
```

Exit 0 = all drive signatures clear the floor. Exit 1 = at least one is below —
which is a **finding to document**, not a licence to restore per-tick pain.
Measured offline: 4 breach episodes → 4 events (vs ~100 pre-fold), so the floor
needs >= 3 real episodes in the run.

<!-- Analyzer appends "## Results" sections below this line -->
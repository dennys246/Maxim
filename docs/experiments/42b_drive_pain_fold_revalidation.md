# Exp 42b — re-validation after the channel-split drive-pain fold

**Status:** ⚠️ **RUN INVALID — RESULTS RETRACTED 2026-07-28.** The 40 sub-sims did
NOT execute the fold. The `maxim` console script resolved `import maxim` to a stale
editable install (`__editable__.maxim-0.1.0.pth`) pointing at the **main checkout**
(then on `feat/console-rest-mode`, which contains no `drive_breach_severity`), and the
shell that launched the run lost its `PYTHONPATH=src` to a short-circuited `&&`:

```
$ source .venv/bin/activate && export PYTHONPATH=src
source: no such file or directory: .venv/bin/activate     # &&, so the export never ran
```

The tell was a *different* symptom: a post-fix session wrote `aut_hippocampus/nac/ec/atl`
but no `aut_scn.json` — impossible on code that has the `scn=` parameter. The numbers
below therefore describe **pre-fold main**, not this branch. They are preserved for the
re-baseline they do provide (see Interpretation) but **they do not validate the fold**.

**The Phase-2 behavioural item is OPEN again.** Re-run per the Commands section, with the
interpreter assertion added below. Everything downstream that cited this run — the plan
header and the CLAUDE.md invariant — has been reverted to "outstanding".

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

## ⚠️ Interpreter provenance — read before re-running

The 2026-07-28 attempt was invalidated by this, so the re-run must not repeat it.

`maxim` is a console script that resolves `import maxim` purely through `sys.path`.
The venv used here carries stale editable `.pth` files pointing at OTHER checkouts, so
**without an explicit `PYTHONPATH` the sub-sims silently import a different tree** — no
error, no warning, and the `git_hash` recorded in the JSONL comes from the *harness's*
directory, so the records look authoritative while describing code that never ran.

Two compounding traps from the invalid run:

1. `source .venv/bin/activate && export PYTHONPATH=src` — the `source` failed (wrong
   path) and `&&` short-circuited, so **the export never happened**. Use separate lines
   or `;`, never `&&`, for the export.
2. `PYTHONPATH=src` is **relative** — it silently resolves to nothing if the sim is
   launched from any directory other than the repo root. **Always use an absolute path.**

The harness now runs `_interpreter_mismatch()` before any sub-sim and exits 3 with the
offending paths if the imported `maxim` is not this repo's `src` (probed through the
console script's own interpreter with the sub-sim's env). `--mock` is exempt.

Verify by hand from the shell you will launch from:

```bash
"$(head -1 "$(command -v maxim)" | sed 's/^#!//')" -c '
import maxim, maxim.simulation.report as r, os, sys
print("cwd       :", os.getcwd())
print("PYTHONPATH:", os.environ.get("PYTHONPATH", "<unset>"))
print("maxim     :", maxim.__file__)
print("aut_scn   :", "PRESENT" if "aut_scn.json" in open(r.__file__).read() else "ABSENT <-- WRONG TREE")'
```

## Commands

Output paths are **deliberately new** (`42b_*`). `analyze_exp42_preference.py`
**overwrites everything below the analyzer marker** in whatever `--out` doc it
is given — pointing it at `42_substrate_primary_preference.md` would destroy the
original GRADUATE record. Always write to this doc instead.

```bash
cd <worktree-with-the-fold>
export PYTHONPATH="$PWD/src"     # ABSOLUTE — a relative 'src' silently resolves to nothing
#   ...and put this on its OWN line; `&& export` after a failing `source` never runs.

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

## Results — treatment (post-fold)

**Verdict: GRADUATE #6 — substrate drives adaptive, feedback-tracked behavior**  (exit 0)

- `substrate_signal` (H1, both arms ≥ 0.66): **True**
- H1[cradle_pref_a] safe_pref = 0.996 → PASS
- H1[cradle_pref_b] safe_pref = 1.000 → PASS
- C1 (identity flip id_pref_a(b)−id_pref_a(a) ≥ 0.33): +0.996 → PASS
- C2 (per-source learning, harm net < safe net): PASS

| arm | safe id | valid/total | floored | safe_pref | SD | id_pref_a | harm net | safe net |
|---|---|---|---|---|---|---|---|---|
| cradle_pref_a | β | 10/10 | 0 | 0.996 | 0.000 | 0.004 | −0.250 | 0.990 |
| cradle_pref_b | α | 10/10 | 0 | 1.000 | 0.000 | 1.000 | −0.250 | 0.990 |

## Results — gating-OFF ablation (post-fold)

**Verdict: GRADUATE #6** (exit 0) — identical summary statistics to the treatment
run (see the caveat below before reading anything into that).

| arm | safe id | valid/total | floored | safe_pref | SD | id_pref_a | harm net | safe net |
|---|---|---|---|---|---|---|---|---|
| cradle_pref_a | β | 10/10 | 0 | 0.996 | 0.000 | 0.004 | −0.250 | 0.990 |
| cradle_pref_b | α | 10/10 | 0 | 1.000 | 0.000 | 1.000 | −0.250 | 0.990 |

## Interpretation

**⚠️ RETRACTED — this run does not speak to the fold.** It executed pre-fold main (see
Status). What it *is*: a clean 20-seed/arm re-baseline of **main** on current hardware,
which is genuinely useful as the comparison point the real re-run will be measured
against. H1, C1, C2 all pass and nothing floored, so the apparatus and thresholds are
sound — only the code under test was wrong.

**C2 settles the non-trivial reading of `safe_pref`.** `harm_net = −0.25` on every
seed proves the harmful source *was* contacted and *did* accrue negative learning —
so `safe_pref ≈ 1.0` means "tried the harmful source during explore-first, learned,
never returned", not "never encountered it". The consistent `n_contact − n_exploit = 3`
gap is that explore-first prefix, excluded from the metric by design.

**Discrimination is sharper than the frozen run — and the fold is NOT why.** 0.996 /
1.000 here vs 0.984 / 0.975 originally (git `0d6ca70f`), SD collapsing to 0.000. I
initially attributed this to the fold quieting channel-2 attribution noise. **That
inference is falsified**: this run contained no fold. The delta comes from something
else that landed between `0d6ca70f` and current main, and is currently unexplained.
A cautionary note for the re-run: the frozen 0.984/0.975 is NOT the right comparison
point for the fold — **this run is**, because it isolates main-vs-fold with everything
else held constant. That is the one genuinely valuable thing this invalid run produced.

**⚠️ The metric is now at its ceiling.** `safe_pref` sits at 0.996–1.000 with SD 0.000
across 20 seeds. That satisfies H1 (≥ 0.66) comfortably, but it has no headroom left to
detect *degradation* — a future regression could halve the discrimination and still
score ~0.99. Any later arm that needs to measure a decrease should use a more sensitive
statistic (e.g. time-to-first-avoidance, or harm contacts per exploitation window),
not this one.

**⚠️ The gating-OFF arm is NOT confirmed to have had gating off.** The two run files
are genuinely distinct (per-seed `n_exploit` differs: arm A treatment 262–283 vs
ablation 261–281), so this is not a duplicated file — but:
- The drive-gate state was **not recorded** in the run JSONL at the time of this run
  (`ablation_arm` records the Exp 44 body-state arms, not B7 gating), so it cannot be
  verified retroactively from the data.
- The behavioural signature the frozen run used as proof that "the toggle demonstrably
  fired" — a volume difference (treatment arm B spiking to ~106 contacts vs the
  ablation's tight ~56–64) — is **absent**: both runs here sit at n_exploit ~100–121
  in arm B.

Two readings are consistent with that: (a) the env prefix did not reach the sub-sims and
the "ablation" is a second treatment run, or (b) it fired but drive-gating no longer
moves contact volume in this configuration. **This does not affect the fold's
validation**, which rests on the treatment arm — and B7 was already marked `Dormant`
by the frozen run, so nothing downstream depends on re-proving it. But the ablation
row above should be read as *corroborating*, not as an independently verified
gating-OFF result. Fixed forward: `benchmark_exp42_preference.py::_record` now emits
`env_drive_gate_enabled`, so future run files are self-describing about which arm they
are. Re-running the ablation with the current harness would settle it in ~100 minutes.

**Run conditions.** 10 seeds/arm × 2 counterbalanced arms × 2 configurations,
substrate-primary (AUT is LLM-free), `smollm-1.7b-instruct` narrator, 40 turns
requested, `cost=$0`, ~295 s/sub-sim. Executed on the Mac Mini with the leader's
Qwen server stopped so the singleton guard served smollm on :8100.

**Note on the harness's live output:** it prints `round(safe_pref, 2)`, so a true
0.996 displays as `1.0` per seed. That is cosmetic — the analyzer's 3-decimal figures
are authoritative.

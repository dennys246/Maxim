# Exp 42b — re-validation after the channel-split drive-pain fold

**Status:** ✅ **FIRED + VERIFIED 2026-07-29 — GRADUATE #6 holds post-fold.** 40 sub-sims
(10 seeds/arm × 2 arms × 2 configurations), 0 failed, 0 floored, on code whose identity is
recorded in every run record (`executed_git_hash` = `7fcae756`, gate arm verified via
`env_drive_gate_enabled` 1/0). Closes the Phase-2 behavioural item for
[transition_based_drive_pain.md](../plans/deferred/transition_based_drive_pain.md)
(PR #435).

> A first attempt on 2026-07-28 was **retracted as invalid** — the sub-sims imported a
> different checkout (stale editable `.pth` + a `PYTHONPATH` export lost to a
> short-circuited `&&`). Its numbers survive as the pre-fold **main baseline**
> (`42b_PREFOLD_main_baseline*.jsonl`), which turned out to be the right comparison
> point. The provenance guard added in response is what makes this run auditable; see
> the Interpretation and the provenance section below.

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

## ⚠️ The runs are 30 turns, not 40 — and that is CORRECT for a replication

`--sim-max-turns 40` is **dead input** on this path. The generative loop breaks on
`narrator.is_done`, which is driven purely by the arc's own per-phase `turns_max`;
`cradle_pref_a` / `cradle_pref_b` budget 6 + 12 + 12 = **30**. The flag is a ceiling
only, and 30 < 40, so it never binds. Every run logs the truth at startup:
`arc=cradle_pref_a (builtin), turns=20-30`.

**Do NOT widen the phase tuples to "fix" this before re-running.** The frozen Exp 42
used these same arcs, so it was *also* a 30-turn run. 30 is the exact replication
condition; changing it would break comparability with the very result being checked.
The "40 turns" in Exp 42's own writeup is a documentation error (the flag was passed
and assumed effective), not an unmet condition — corrected there too.

Separately: **`Finish: cancel` is a cosmetic mislabel** and carries zero information.
Generative campaigns run inline and then unconditionally `stop_event.set()`, and
`bridge.finish_context` is never populated on that path, so the reason falls through
to the `stop_event` branch. *Every* generative campaign reports `cancel` regardless of
how it ended. It is not evidence of a cancel, crash, or timeout.

## Pass criteria (frozen — copied, not re-derived)

| # | Criterion | Original result |
|---|---|---|
| H1 | `safe_pref >= 0.66` in **both** arms | A 0.984 / B 0.975 (treatment) |
| C1 | identity-flip: safe identity swaps with the arm | +0.959 PASS |
| C2 | harm net < safe net (learning signs) | PASS |
| K | >= 10 exploitation choices/seed, 0 floored | 10/10 valid both arms |
| turns | **30** (arc phase budget; the `40` in Exp 42's writeup never took effect) | 30 |

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

## Companion check — SCN oscillator cold-start floor — **RESOLVED: concern VOID**

**Result (2026-07-29): the fold cannot have affected this, because the path is unwired.**

A proper substrate-primary cradle run (exploration ON — tool diversity confirmed: 209
warm_self / 62 observe / 60 touch / 57 harm-observe, `explore=0.04` in the plan string)
persists `aut_scn.json` containing **10 event signatures, 0 of them `drive:*`**.

Root cause is wiring, not density. `Body._emit_drive_temporal_event` returns early on
`self._distributor is None`, and `runtime/bootstrap.py::build_executor` threads its
`distributor=` argument into `ToolPainBridge` (`:439`) while constructing
`Embodiment(entity, pain_bus=..., agent_id=...)` **without** it (`:456`) — in the same
function. The 10 signatures that do arrive come via ToolPainBridge.

So **no drive `TemporalEvent` has ever reached `OscillatorNetwork` in any production
run**, pre- or post-fold, and `anticipatory_pre_activate` has never had drive events to
predict from. The `< 3 observations` cold-start guard was never the constraint.

The one-line wire (`distributor=distributor` on that `Embodiment`) is **deliberately not
applied here**: it would activate a previously-dead learning path for the first time —
a behavioural change deserving its own validation, not a drive-by edit inside an
attribution fix. Recorded as dormant infrastructure.

Reproduce:
```bash
MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT=1.5 MAXIM_SIM_DRIVE_GATE_ENABLED=1 \
maxim --sim cradle_pref_a --aut-mode substrate-primary \
  --embodiment bodies/infant_humanoid_chilled --interactive false
python scripts/check_oscillator_coldstart.py --session ~/.maxim/sim_reports/<newest>
```
(Exploration MUST be on: with the default `substrate_explore_bonus_weight=0.0` the
substrate degenerates to pure argmax — one tool, 411 calls, the harm source never
contacted, `safe_pref` undefined. That is the Exp 41 "exploration is the enabling
condition" result reproducing itself.)

<!-- Analyzer appends "## Results" sections below this line -->

## Results — treatment (post-fold, gating ON)

**Verdict: GRADUATE #6** (exit 0) · `executed_git_hash` = `7fcae756` = fold branch ·
`env_drive_gate_enabled` = `1`

- H1 both arms ≥ 0.66: **True** — a 0.996 PASS · b 1.000 PASS
- C1 identity flip +0.996 PASS · C2 (harm net < safe net) PASS · 10/10 valid, 0 floored

| arm | safe id | valid/total | floored | safe_pref | SD | id_pref_a | harm net | safe net |
|---|---|---|---|---|---|---|---|---|
| cradle_pref_a | β | 10/10 | 0 | 0.996 | 0.000 | 0.004 | −0.250 | 0.990 |
| cradle_pref_b | α | 10/10 | 0 | 1.000 | 0.000 | 1.000 | −0.250 | 0.990 |

## Results — gating-OFF ablation (post-fold)

**Verdict: GRADUATE #6** (exit 0) · `executed_git_hash` = `7fcae756` ·
`env_drive_gate_enabled` = **`0`** (toggle verified fired — this is what the retracted
run could not establish)

| arm | safe id | valid/total | floored | safe_pref | SD | id_pref_a | harm net | safe net |
|---|---|---|---|---|---|---|---|---|
| cradle_pref_a | β | 10/10 | 0 | 0.996 | 0.000 | 0.004 | −0.250 | 0.990 |
| cradle_pref_b | α | 10/10 | 0 | 1.000 | 0.000 | 1.000 | −0.250 | 0.990 |

## Interpretation

**The fold preserves GRADUATE #6. The Phase-2 behavioural item is CLOSED.**
40 sub-sims, 0 failed, 0 floored, K cleared by 10–28× (n_exploit 100–284 vs the K=10
gate). H1/C1/C2 pass in both configurations, on code whose identity is recorded in
every run record.

**The fold is behaviourally neutral on this metric — as designed.** Post-fold
(0.996 / 1.000) is *identical* to the pre-fold main baseline (0.996 / 1.000, archived
as `42b_PREFOLD_main_baseline*.jsonl`). That is the predicted result: Exp 42's
discrimination rides the **direct** `FailureEvent` channel, which the fold deliberately
left state-based; only the PainBus channel was severity-latched. A difference here
would have meant the channel split leaked.

**C2 makes `safe_pref ≈ 1.0` non-trivial.** `harm_net = −0.25` on every seed proves the
harmful source was contacted and accrued negative learning — so this is "tried it during
explore-first, learned, never returned", not "never encountered it". The consistent
`n_contact − n_exploit = 3` gap is that explore-first prefix, excluded by design.

**Retraction of an earlier inference, now settled.** I previously attributed the
0.984/0.975 → 0.996/1.000 improvement over the frozen run (git `0d6ca70f`) to the fold
quieting channel-2 attribution noise. **False.** Pre-fold main already scores
0.996/1.000, so the gain came from something else that landed between `0d6ca70f` and
current main, and remains unexplained. Having both measurements is what settled it —
the one durable thing the invalid run produced.

**⚠️ Read "identical" as saturation, not as proof of no difference.** `safe_pref` sits
at 0.996–1.000 with **SD 0.000** across every condition — pre-fold, post-fold,
gating-ON, gating-OFF. Four conditions collapsing to the same number is what a
ceilinged metric looks like. It supports "the fold did not BREAK discrimination"
(the question asked) but is weak evidence for "the fold changed nothing" — this metric
could not detect a moderate regression. Any future arm measuring *degradation* needs a
more sensitive statistic (time-to-first-avoidance, harm contacts per exploitation
window), not this one.

**B7 drive-gating: the frozen run's volume signature did NOT reproduce.** The gate
demonstrably fired (`env_drive_gate_enabled` 1 vs 0, recorded per run), yet treatment
and ablation are indistinguishable in *both* discrimination and contact volume
(arm A 264–284, arm B 103–125 in both files). The frozen writeup reported gating moving
volume — arm B spiking to ~106 under treatment vs a tight ~56–64 under ablation. That
no longer holds. B7 is already `Dormant` (it did not earn behavioural weight in the
frozen run either), so nothing downstream depends on it; recorded as observed drift
from `0d6ca70f`, not investigated here.

**Provenance (the fix for what invalidated the first attempt).** Every record carries
`harness_git_hash`, `executed_git_hash`, `executed_maxim_file`, `pythonpath` and
`env_drive_gate_enabled`. Both hashes read `7fcae756` and the gate reads `1`/`0` per
arm — so which code ran, and under which condition, is auditable from the artifact
alone rather than from shell history.

**Run conditions.** 10 seeds/arm × 2 counterbalanced arms × 2 configurations,
substrate-primary (AUT LLM-free), `smollm-1.7b-instruct` narrator, **30 turns**
(arc phase budget — see the turn-count section above; the frozen run was also 30),
`cost=$0`, ~300 s/sub-sim, Mac Mini with the leader's Qwen server stopped.

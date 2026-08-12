# Exp 48 — Does the extero/intero seam make the EMBODIED infant orient?

**Status:** PRE-REGISTERED (2026-07-23). Runs after the extero/intero multi-modality seam (PR #411, `26d8f901`). Pre-registration is authoritative; fill Results/Verdict only from a completed run whose `git_hash` contains the seam.

## The one-sentence question

The scripted probe proved the *mechanism* learns (multi-drive orient, chance 0.56 → 0.91, 5/5 seeds; `tests/unit/test_modality_seam.py::TestMultiDriveOrientLearnsEndToEnd`). **Does that carry to the EMBODIED `cradle_mother` sim** — the instrument that measured at chance in [Exp 46](46_operant_orient_creche.md) — now that direction is no longer diluted among the drives?

## Why this run exists (the causal chain)

[Exp 46](46_operant_orient_creche.md) validated operant orienting on the **scripted** substrate and was explicit that the **embodied** `cradle_mother` sim measured at **chance** — a textbook divergence (tool competition → explore weight → confidence-gate stalls). The [dilution root cause](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/reference_extero_intero_dilution_root_cause.md) later found *why* the embodied body specifically was blind: `propose_via_substrate` merged `{**drives, **azimuth}` into one interoception encode, so on `bodies/infant_operant` (5-ish drive dims + 1 azimuth) left/right pattern-completed onto the SAME EC cluster. The `cradle_mother.py` module docstring names its own revival condition: *"Resurrecting the embodied path requires the credit-on-progress root-cause fix so the operant contingency isn't drowned."* The **seam is a stronger fix than that** — it removes the dilution at the perception layer AND routes operant credit to the direction-bearing `audio` cluster. This experiment tests whether that structural fix is sufficient to lift the embodied instrument off chance.

**Scope honesty (do not overclaim):** the embodied sim is still the noisy instrument Exp 46 flagged — LLM narrator non-determinism, gates, turn caps. The scripted closing test remains the clean mechanism-level proof. **The claim this run can earn is narrow: "the embodied infant is NO LONGER AT CHANCE, and the mother is why."** It is NOT a re-derivation of the mechanism (that is Exp 46 + the seam's unit test); it is a confound-check that the seam's payoff survives the embodied machinery. A FAIL here does not un-earn the seam — it says the *remaining* embodied confounds (tool competition, gate stalls) still dominate, which reopens the credit-on-progress / narrator-hygiene line, not the seam.

## Design

Reuses the **already-shipped** harness + analyzer + arc unchanged — this experiment adds no code, only a run:
- Harness: `scripts/benchmark_cradle_mother.py` (arc `cradle_mother`, body `bodies/infant_operant`, substrate-primary — **no LLM in the action path**).
- Analyzer + pre-registered verdict: `scripts/analyze_cradle_mother.py`.

### Arms (harness selects by env; identical arc)

| Arm | What the mother does | Key env |
|---|---|---|
| **taught** | Feeds + `credit_operant_reward` on the infant's own toward-turn (operant shaping — the sole teacher; the infant has NO intrinsic orient drive) | `MAXIM_OPERANT_ONLY_CREDIT=1` |
| **no_feed** (control) | Places the sound but never feeds/credits — with no intrinsic drive, no teacher | `+ MAXIM_CRADLE_MOTHER_DISABLE_CARE=1` |

Both arms set `MAXIM_OPERANT_ONLY_CREDIT=1`, `MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT=1.5` (the driveless infant must explore turns to bootstrap), and `MAXIM_SUBSTRATE_TOOL_WHITELIST=turn_left,turn_right` (Exp 46's tool-competition control — the seam de-dilutes direction but does not by itself out-compete a snowballing always-succeed tool; that is the separate credit-on-progress question, deliberately held constant here so this run isolates the seam's dilution fix). The harness sets all three internally.

**Metric:** per time-bin ("act"), **directedness** = fraction of turns the infant's own turn moved it TOWARD the sound (`|prev_stimulus| − |az_after| > 0`). Logged in BOTH arms (the mother computes progress even when she withholds care), so the control is measurable.

### Pre-registered verdict (from `analyze_cradle_mother.py`, DO NOT retune post-hoc)

- **LEARNED** — `taught` late-bin directedness ≥ **0.65** AND rose from the early bin by ≥ **0.15**.
- **MOTHER-TAUGHT** — `taught` late ≥ `no_feed` late + **0.20**.
- **PASS = both.** (These are the constants `LEARNED_MIN=0.65`, `RISE_MARGIN=0.15`, `MOTHER_MARGIN=0.20` already in the analyzer — this experiment does not change them.)

### Pre-registered branches on the outcome

- **PASS** → the seam lifts the embodied instrument off chance; the `cradle_mother` module's revival condition is met by the seam. Update `46_operant_orient_creche.md`'s "embodied = chance" note with the seam delta, and lift the `Dormant since 2026-07-22: DEMO ONLY` marker on `simulation/cradle_mother.py` to a validated-embodied status. This becomes a `[behavioral]` graduation candidate (row in `behavioral_graduation_candidates.md`).
- **FAIL but taught > no_feed (partial)** → direction is now learnable embodied but the ceiling is capped by a *named* remaining confound. Do NOT retune the verdict; instead read the per-act curve + the `sim_recommend_action` telemetry (`consulted_bias_by_modality` — is the audio channel present and consulted?) to identify which confound (tool competition, explore-weight ceiling, gate stalls) dominates, and file it against credit-on-progress / narrator hygiene. One pre-approved lever: an **explore-weight sweep** (`--explore-weight 1.0 0.75`) since 1.5 bootstraps but may cap the ceiling on the small ~2-cluster orient policy (the harness flags this exact trade-off).
- **FAIL, taught ≈ no_feed ≈ chance** → the seam did not carry to the embodied instrument. This is a **divergence signal** per the cycle-divergence rule: do NOT patch the mother; step back and confirm via telemetry whether the audio cluster is even reaching selection embodied (bird's-eye: "did the actuation happen?" — is `current_clusters` carrying `audio` on the embodied path, or is something upstream of `propose_via_substrate` stripping it?).

## Reproduction

Run from the **big-mac-mini leader**, NOT co-located with any other sim/leader consumer of the LLM server (the Exp 37 cradle-cascade lesson). The arc is prose-less so the narrator is barely used, but n_ctx/model must still be config-aligned. Ops preflight is in the kickoff prompt below.

```bash
# from the repo root, PYTHONPATH=src (worktree wins over any installed pymaxim)
PYTHONPATH=src python scripts/benchmark_cradle_mother.py \
    --arms taught,no_feed --trials 12 --seed-base 42 \
    --sim-max-turns 56 --model mistral-7b \
    --out ~/exp48_cradle_mother_seam.jsonl --workdir /tmp/exp48_runs

PYTHONPATH=src python scripts/analyze_cradle_mother.py --in ~/exp48_cradle_mother_seam.jsonl --trials 12
```

`--trials 12` per arm (Exp 42's per-arm n; raise to 20 if the taught/no_feed gap lands within seed noise of the 0.20 margin). `--resume` skips completed (arm, seed) cells so an interrupted overnight run continues without a full rerun. Each result row records `git_hash` — **verify it contains the seam (`26d8f901` or a descendant) before trusting any verdict** (the hard-won stale-checkout lesson).

## Results

> ### ⚠️ CONTESTED — do not cite this experiment pending resolution (2026-08-11)
>
> The 1.1 heartbeat re-run could not reproduce the magnitudes below, and the
> investigation that followed found a **conflict between this experiment's
> conclusion and how its apparatus actually behaves**. Both the numbers and the
> mechanism are contested; this is an open assessment, not a settled downgrade.
>
> **What is established:**
>
> * **The magnitudes do not reproduce.** The headline **0.875 late / +0.211
>   rise** was not recovered in any of four configurations (two commits, two
>   machines, two narrator models). At matched n=12 the graduation commit scores
>   **+0.055** and current code **+0.079** — so this is **not a code
>   regression**.
> * **The arm difference is real**: `taught` 0.748 vs `no_feed` 0.399 (+0.349) at
>   n=12, control flat across all four acts.
>
> **What is contested — why the arm difference exists:**
>
> * **The action path is an undamped oscillator.** The exploration bonus
>   (`novelty = weight/(1+visits)`) is an explicit anti-repetition term AND, on
>   this driveless body, the *sole source of action* (at weight 0 the agent
>   emits nothing). It produces strict `turn_left`/`turn_right` alternation at
>   50/50 ±1. A learned bias below ~0.11 is invisible against it; even a
>   saturated policy reaches only 88%.
> * **The operant credit lands on a coin flip.** The mother credits the single
>   most recent action, which under alternation is left or right with
>   probability ½ — so the teaching signal may be uncorrelated with the
>   behaviour it is supposed to be teaching.
> * **The arms are not exposure-matched:** `taught` gets ~86 mother-turns per
>   seed, `no_feed` exactly 48.
> * **The outcome sequence is not independent** (lag-1 autocorrelation 0.62 vs
>   0.50 for independent draws), and both the stimulus and the agent alternate
>   deterministically — so "directedness" may partly measure the *phase
>   relationship between two oscillators* rather than learning.
> * **The action count is a stopwatch reading**, not a behaviour: actions per
>   turn = mother-turn wall-clock ÷ 0.5 s (`agent_loop.py:3475`), unbounded by
>   any turn signal. This is why the number is machine- and latency-dependent,
>   and plausibly why the original was never reproducible.
>
> **Status:** the row is `Stale`/CONTESTED and blocks 1.1 pending pre-registered
> controls (randomised stimulus order; explore-weight sweep — the latter
> pre-approved in this document's own pre-registration). It is **not retracted**
> and **not a code regression**. Treat every number below as the original
> measurement under an apparatus we no longer trust.

**Run 2026-07-23, big-mac-mini leader, 12 seeds/arm, 56 turns, mistral-7b narrator. 24 runs, 0 failed.**

Per-act directedness (4 time-bins: early / warmup / consolidation / autonomy):

| arm | early | warmup | consol | autonomy (late) |
|---|---|---|---|---|
| **taught** | 0.51 | 0.82 | 0.85 | **0.90** |
| **no_feed** (control) | 0.43 | 0.33 | 0.44 | 0.45 |

Pooled verdict bins (analyzer):

| arm | early | late | Δ (late−early) |
|---|---|---|---|
| taught | 0.664 | 0.875 | **+0.211** |
| no_feed | — | 0.448 | — |

- **LEARNED** (taught late ≥ 0.65 and rose ≥ 0.15): **PASS** (0.875 ≥ 0.65; rose +0.211 ≥ 0.15)
- **MOTHER-TAUGHT** (taught late ≥ no_feed late + 0.20): **PASS** (0.875 − 0.448 = **+0.427**, > 2× the 0.20 margin)
- **VERDICT: GRADUATE** — the embodied infant learned to orient toward the mother's voice, taught by her contingent feeding alone. The instrument that measured at chance in [Exp 46](46_operant_orient_creche.md) is off chance.
- git_hash: `39314368` (verified from the results JSONL; a descendant of the seam merge `26d8f901`, so the run contains the fix)

### Reading

The seam carried to the embodied instrument. The `taught` curve is a clean developmental rise (0.51 → 0.82 → 0.85 → 0.90) while the control sits flat at chance across all four acts (0.43 → 0.45) — the +0.427 late-bin gap is the mother's operant teaching, isolated. This is the narrow claim the pre-registration scoped: **no longer at chance, and the mother is why** — a confound-check that the seam's de-dilution payoff survives the embodied machinery (narrator, gates, turn caps) that pinned Exp 46 at chance. It does NOT re-derive the mechanism (that is the scripted closing test, `TestMultiDriveOrientLearnsEndToEnd`); it confirms the fix reaches the embodied path. Per the pre-registered PASS branch: Exp 46's "embodied = chance" note updated, the `simulation/cradle_mother.py` dormancy marker lifted, and a `[behavioral]` graduation-candidate row filed.

---

## 1.1 heartbeat re-run — the magnitudes do not reproduce (2026-08-11)

The 1.1 walk re-ran this row per
[heartbeat_rerun_runbook.md](protocols/heartbeat_rerun_runbook.md) and it failed
the LEARNED gate (rise **+0.079** vs the required +0.15). A four-configuration
investigation followed. **Two hypotheses were raised and both were refuted by
measurement** — recorded here because the refutations are the useful part.

### The four configurations

| run | machine | narrator | commit | n | early | late | **rise** |
|---|---|---|---|---|---|---|---|
| heartbeat | Mac | mistral-7b | `f05c63aa` | 12 | 0.669 | 0.748 | **+0.079** |
| bisect | Mac | mistral-7b | `45bd1789` (this row's commit) | 12 | 0.726 | 0.781 | **+0.055** |
| model probe | big-mac-mini | qwen2.5-32b | `45bd1789` | 4 | 0.757 | 0.698 | **−0.059** |
| **published above** | big-mac-mini | "mistral-7b" | `45bd1789` | 12 | 0.664 | **0.875** | **+0.211** |

Provenance verified per run (`git_hash`, `mock: false`, 12/12 per arm).

### What was refuted

1. **"A code regression broke it"** — REFUTED. At matched n=12 the graduation
   commit scores **+0.055** and current code scores **+0.079**: current code is
   marginally *better*, and the difference is inside noise. ~957 lines had
   changed in the shared substrate/motor path since this row graduated
   (predominantly the Reachy orienting line — #447, #460/#461, #463), which made
   a regression the natural first hypothesis. It is not one.
2. **"The thrashing is new"** — REFUTED. The oscillation described below was
   present at this row's own graduation commit (62.8 actions/mother-turn vs 59.4
   now, both ~1% left/right imbalance).
3. **"The narrator model explains it"** — REFUTED, and inverted: qwen2.5-32b is
   *worse* (−0.059), so a 32B-vs-7B difference cannot account for a missing
   +0.13. (This hypothesis was worth testing: with the mistral-7b GGUF **not
   present on the mini**, `_maybe_auto_spawn_server` logs "GGUF file not found"
   and returns *before* the singleton model-match guard — so a run can silently
   serve a different model than its protocol records.)

### What the re-run establishes instead

**The mechanism holds.** Learning is real and reproducible:

| | act1 | act2 | act3 | act4 |
|---|---|---|---|---|
| taught (n=12, `f05c63aa`) | 0.602 | 0.735 | 0.748 | 0.747 |
| no_feed control (n=12) | 0.364 | 0.389 | 0.389 | 0.410 |

The control is flat at chance for the whole arc; the taught infant climbs to
0.75 and holds. MOTHER-TAUGHT passes at **+0.349**.

**The learning completes in the first transition.** act1 → act2 is **+0.133**;
act2 → act4 is **+0.012**. It learns fast, then saturates.

**The gate's bins are misaligned with the learning.** `early = mean(act1, act2)`
folds the already-learned act2 into the baseline, so the gate compares the
plateau against itself. On the honest comparison, act1 → act4 is **+0.145**
against the control's +0.046.

**The plateau height is set by an apparatus pathology.** The infant emits ~60
near-balanced `turn_left`/`turn_right` actions per mother-turn; the net
displacement cancels on a fraction of turns (`progress == 0`, "no-move"), which
caps directedness:

| run | no-move rate | implied cap | observed peak |
|---|---|---|---|
| `f05c63aa` | ~19–21% | ~0.80 | 0.748 |
| `45bd1789` | ~16% | ~0.84 | 0.833 (act3) |
| published | ~10% (implied) | ~0.90 | 0.90 |

Every run sits just under the ceiling its own thrashing rate imposes. The
published 0.90 implies an apparatus oscillating roughly half as much as anything
reproducible today. **Why the original thrashed less cannot be determined**: its
raw records went to `~/exp48_cradle_mother_seam.jsonl` and its logs to
`/tmp/exp48_runs`, both gone — including the `lane_decisions.jsonl` that would
identify what the sub-sims actually served.

### Disposition

- The claim **stands**; the row is **not retracted** and this is **not a code
  regression**.
- The row is `Stale` pending **re-baselining against a fixed apparatus**. Fixing
  the thrashing lifts the ceiling, at which point the magnitude means something.
- The LEARNED gate needs **re-pre-registration** (bin alignment + ceiling
  robustness) — by pre-registration, never post-hoc adjustment.
- Standards derived from this incident, including the ones that would have
  caught it years earlier in the cycle:
  [simulation_apparatus_standards.md](../plans/simulation_apparatus_standards.md).

### Analyzer defect found en route

Running a single arm makes the analyzer treat the absent arm as `0.0` and report
**MOTHER-TAUGHT: PASS** against a control that never ran. A gate that passes
vacuously on missing data is worse than one that fails; fix alongside the gate
re-pre-registration.

**Data (durable, committed per S4 — the standard this investigation kept
violating):** `docs/experiments/data/48_heartbeat_1_1.jsonl` (new commit, 24
runs) + `48_bisect_45bd1789.jsonl` (old commit, 12 runs). The 4-run qwen32b
probe lives on the big-mac-mini at `~/exp48_mini/old_mini_qwen.jsonl` and should
be copied to `data/48_probe_qwen32b.jsonl` — its per-seed lines are recorded in
the table above so the finding survives if the file does not.

# Scripted-experiment re-derivation — 2026-08-24 (S4 backfill)

**Why this exists.** Apparatus standard S4 ([simulation_apparatus_standards.md](../../../plans/simulation_apparatus_standards.md))
requires committed raw records, and the 1.1 evidence-closure gate found that
Exp 43, 46 and 47 had none: their harnesses print to the terminal, and the
original 2026-07 runs were never captured. All three are **scripted, seeded,
LLM-free and run in seconds**, so instead of a data-lost annotation they were
re-derived and the full stdout of every probe is committed here verbatim.

**These files are the current-code re-derivation, not the original measurement.**
The experiment docs keep the original tables; the side-by-side below is the
comparison. Run at the configuration each doc states (two probes needed explicit
flags because their script defaults differ from the doc — see the `### command`
line in each file), the numbers reproduce the docs to within rounding and every
pre-registered verdict reproduces. A first pass that used script defaults had read
the resulting differences as substrate-code drift; that attribution was wrong (a
units/parameter mismatch — the review round caught it) and is withdrawn.

## Provenance

Every `.txt` starts with a `### provenance` header captured **by the same shell
immediately before that run** (one capture per file — the timestamps are sequential): `maxim.__file__`, `sys.executable`,
`git rev-parse HEAD`, `git status --short` over `src/maxim/{decisions,similarity,embodiment}`
and `scripts/`, `PYTHONPATH`, and the UTC time — the in-process analogue of the
harness-provenance rule (an editable `.pth` can redirect an import just as it can a
spawned sim, so the record must say which `maxim` actually ran).

- Repo: `dennys246/Maxim`, `main` at **`b01a6589`**; `maxim.__file__` resolved to
  `<repo>/src/maxim/__init__.py` via the repo `.venv` (Python 3.12), operator Mac.
- The working tree was clean under the stamped paths for eight of nine files. The
  exception is `43_1_operant_redirection.txt`, whose header records
  ` M scripts/gaze_substrate/1_operant_redirection.py` — the determinism fix below,
  which ships in the same commit as this record.
- Command per file: recorded on the `### command` line. Defaults except
  `46_4` (`--seeds 8`, the doc's n; script default 5) and `46_5`
  (`--agents 12 --ticks 2 --seeds 10`, the doc's configuration; script defaults
  10 × 4, 8 seeds). Wall time 1–14 s each.
- Not re-run: Exp 43 probe 5 (`5_real_encoder_validation.py`) — needs the real
  sentence-transformer encoder; its doc numbers stand on the uncaptured original only.

## Determinism finding — Exp 43 probe 1 was not reproducible run to run

Two pre-fix runs of `1_operant_redirection.py` in the same tree gave contingent
**0.920 (sd 0.025)** and **0.888 (sd 0.072)**; the other eight files reproduced
byte-for-byte. Cause: the per-arm seed was `1000 * a + hash(arm) % 97`, and the
builtin `hash()` of a `str` is randomised per process by `PYTHONHASHSEED` — the same
bug class as the repo's stable-hash persistence lesson, in a "seeded" harness. Fixed
in the same commit to `stable_hash_32(arm) % 97` (`maxim.utils.seeding`). Checkable
evidence: `PYTHONHASHSEED=1` and `PYTHONHASHSEED=4242` runs of the fixed script both
produce stdout with sha256 `4bed8b8a083f46c45681e55f6eecb4a0c125667a768119376f7a60f4d3b3c4ef`, and the body of the
committed `43_1_operant_redirection.txt` (everything after its `### command` line)
hashes to the same value. The committed file is therefore the deterministic run:
contingent **0.894 (sd 0.052)**, yoked 0.059, none 0.120. Probe 2 (`2_search_and_coverage.py`) reproduced exactly across runs — its
seeds are plain integers.

## Doc vs re-derivation

| File | Doc value (2026-07) | Re-derived (2026-08-24) | Verdict |
|---|---|---|---|
| `43_1_operant_redirection` | contingent 0.88 / yoked 0.08 / none 0.09 | 0.894 (sd 0.052) / 0.059 / 0.120 (deterministic after the seed fix; pre-fix draws 0.920 and 0.888) | contingency effect reproduces (+0.835 vs yoked) |
| `43_2_search_and_coverage` | cross-session +0.281 ± 0.055 SE; habituation OFF fov 0.68 dwell 0.77 covers 1.8/3, ON 0.41 / 0.50 / 2.7 | +0.281 ± 0.055; 0.682 / 0.765 / 1.81, 0.413 / 0.497 / 2.69 | identical |
| `43_3_geometry_generalization` | peak σ=8: FAR dir 0.51 (> chance 0.429), dir\|transfer 0.84, dict transfer 0 | σ=8: 0.511 (chance 0.429; synthetic FAR 0.435), 0.842, synthetic transfer 0.000 | identical |
| `43_4_visual_category_transfer` | dict NOVEL −0.005; EC eps 0.15 seen 0.99 / NOVEL 0.938 / ~4 nodes; eps 1.0 0.81 / 0.091 / ~12 | −0.005; 0.988 / 0.938 / 3.9; 0.810 / 0.091 / 11.6 | identical |
| `46_4_operant_learning_curve` (`--seeds 8`) | taught 0.65 → 0.90, yoked 0.36, none 0.50 | taught 0.65 → 0.90, yoked 0.41, none 0.50 | LEARNED + MOTHER-TAUGHT PASS (yoked 0.36 → 0.41, a chance-level control) |
| `46_5_operant_creche_federation` (`--agents 12 --ticks 2 --seeds 10`) | single_partial 0.73, creche_taught 1.00, creche_none 0.51 | 0.73, 1.00 (single_full 1.00), 0.51 | identical |
| `46_6_graded_orient_curve` | taught 0.19 → 0.82, yoked 0.03, none 0.17 | 0.19 → 0.83, 0.05, 0.15 | both PASS |
| `46_7_graded_creche_federation` | single_partial 0.59, creche_taught 1.00, creche_none 0.16 | 0.59, 1.00, 0.16 | identical |
| `47_8_habituation_novel_in_noise` | all tables (1.00 vs 0.46/0.18/0.07/0.02/0.04; mother 1.00; solo 0.06 vs crèche 1.00) | identical to the cell | identical |

## What is still NOT committed (tracked in the experiment docs)

- Exp 09 / Exp 10 originals — lost (`/tmp` logs); the 2026-08-18/19 heartbeat
  re-run records are on big-mac-mini, S4 copy pending.
- Exp 44 Gate-2 capture — location unrecorded; Exp 44b pilot captures on big-mac-mini.
- Exp 49 dense arms A/B/C — lost (session scratchpad); the sparse arm B is committed
  as `../49_armB_sparse_trials.jsonl` + `../49_armB_sparse_summary.json`.

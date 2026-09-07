# Exp 57 — operator runbook (the dose–response ladder)

The live apparatus + campaign steps for
[`exp57_dose_response_ladder_preregistration.md`](../../docs/experiments/protocols/exp57_dose_response_ladder_preregistration.md)
(FROZEN). Everything science-frozen lives in the prereg and in
[`scripts/analyze_exp57.py`](../analyze_exp57.py)'s `GATES_V1`; the apparatus
constants (K_max, C, W) are set by the **Phase-0 amendment**, not here. The
`--mock` paths (ScriptedBridgeServer) are for harness development only and are
**never a confirmatory record**; the analyzer refuses a verdict on mock rows.

Reuses the Exp 56 apparatus wholesale (`scripts/exp56/common.py`): the world
setup, the teacher, the bench body, the RCON control, the bridge client. Stand
up the world exactly as in [`scripts/exp56/README.md`](../exp56/README.md)
(Paper 1.16.5, Java 11–16, the Mineflayer bridge, `setup_world.py`), then run
Phase 0, freeze the constants by amendment, then the confirmatory ladder.

## ⚠️ The load-bearing Phase-0 risk: pairwise slot separation (check 1)

The four frozen contingency slots differ mainly in **horizontal direction**
(x/z sign): `(88,112,0)`, `(-88,112,8)`, `(8,112,88)`, `(-8,112,-88)`. But the
bench world sensors encode **magnitude** — `distance_from_spawn` ≈ 88 for all
four and `y_altitude` = 112 for all four — not direction. Exp 56 only needed
slot-vs-**REST** separation (one contingency) and passed; Exp 57 needs all four
**pairwise-distinct** (G = 4 distinct contingencies). The prereg flags this as
genuinely unvalidated ("amendment 2 validated slot-vs-REST, NOT slot-vs-slot;
Phase 0 check 1 … is its gate, and a failure is an apparatus fix"). **Expect
check 1 to be the gate that fails first.** The pre-registered fix is an
apparatus amendment: re-place the slots at genuinely distinct
`(distance_from_spawn, y_altitude)` coordinates (extend `FROZEN["contingency_slots"]`
+ `setup_world.py`, re-validate), NOT a code change and NOT a silent proceed. A
mock Phase-0 fails checks 1/2/2b/4 for the same reason (degenerate scripted
geometry) — only check 3 (drive-zero) and check 5 (plumbing) pass on `--mock`;
that is expected, not a harness defect.

## 1. Phase 0 (gates the campaign; sets K_max/C/W)

```bash
export MAXIM_OPERANT_ONLY_CREDIT=1     # frozen apparatus; the CLIs REFUSE without it
export PYTHONPATH="$PWD/src"           # if running from a worktree
python scripts/exp57/instrument_check.py --rcon-password 'PW' --write-experiment-results
```

Exit 4 on any failing check, with the per-check report + the calibration
**PROPOSAL** (K_max / C / W) in `docs/experiments/data/57_phase0.json`. Then,
BEFORE the campaign:

- freeze K_max / C / W by a **pre-campaign amendment** (its own PR, merge-commit)
  disclosing the Phase-0 readings (the Exp 52/56 disclosure rule); and
- fill the prereg sign-off boxes.

The GATE constants (rungs `{1,2,4,8}`, δ_eff `0`, p `0.05`, ≥ 20 cohorts) are
frozen in `analyze_exp57.py` and are NOT set from Phase-0 data.

## 2. The confirmatory ladder (runs ONCE)

```bash
export MAXIM_OPERANT_ONLY_CREDIT=1
python scripts/exp57/run_ladder.py \
    --rungs 1,2,4,8 --cohorts 20 --seed-base 42 \
    --k-max <PHASE0_KMAX> --criterion <PHASE0_C> --window <PHASE0_W> \
    --workdir ~/exp57_work \
    --out docs/experiments/data/57_ladder.jsonl \
    --rcon-password 'PW' --write-experiment-results

python scripts/analyze_exp57.py --in docs/experiments/data/57_ladder.jsonl \
    --gate v1 --assert-noop-fails
```

- **`--seed-base` drives everything** (`cohort_seed = seed_base + cohort`), so
  the fixed, PUBLISHED seed set is `{42 … 42+cohorts-1}` — this is also **R0's**
  multi-seed harness (publish the seed list BEFORE any data; choosing seeds after
  seeing results voids the rung). Every claim is reported per-seed (per-cohort
  rows) AND pooled (the analyzer aggregates).
- `--workdir` must be **durable** (`--resume` refuses a tmpdir).
- **Cost:** the B-phase fold goes through the REAL export→ingest path per
  checkpoint; at N = 8 with a deep K_max this is thousands of ingest operations
  and is the dominant cost. Budget accordingly; `--resume` re-enters safely.
- Data PR: **merge-commit only, never squash**; interpretation lands in a
  SEPARATE later PR (the structure-or-time rule).

## 3. What the analyzer decides

`PASS` needs MONOTONICITY (Jonckheere–Terpstra decreasing, permutation p < 0.05,
+ both censoring-artifact guards) AND NOT-JUST-MORE-DATA (N·τ(creche) ≤
τ(single_matched), δ_eff = 0) AND NOISE-FLOOR (creche_none < C), with the
`--assert-noop-fails` anti-vacuity kit clean. `PARTIAL` (MONOTONICITY passes,
NOT-JUST-MORE-DATA fails) is an owner call at the release checkpoint, not a
silent pass (it exits non-zero). A flat curve ships as the FALSIFIER result.
Exit 0 PASS / 4 NO-VERDICT / 1 FAIL|PARTIAL.

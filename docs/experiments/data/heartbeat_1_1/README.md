# 1.1 heartbeat walk — raw measurement records (Exp 42 + Exp 48)

Salvaged 2026-08-13 from the `Maxim-heartbeat-1.1` worktree (detached HEAD `f05c63aa`),
where they had lived untracked since 2026-08-10/11. Committed per apparatus standard S4
([docs/plans/simulation_apparatus_standards.md](../../../plans/simulation_apparatus_standards.md)):
these are the measurements behind claims already merged to `main` —
the Exp 42 Maintained re-validation and the Exp 48 CONTESTED finding
(PR #500, [docs/experiments/48_cradle_mother_seam.md](../../48_cradle_mother_seam.md)).

## Contents

| Path | What it is |
|---|---|
| `42_heartbeat_1.1.jsonl` | Exp 42 heartbeat re-run, per-sub-sim records (`executed_git_hash f05c63aa`) |
| `48_heartbeat_1.1.jsonl` | Exp 48 heartbeat re-run, per-run records |
| `42_fleet.log`, `48_fleet.log`, `fleet_status.txt` | Fleet-runner logs for the two campaigns |
| `bisect/old_45bd1789.jsonl` | Exp 48 A/B — graduation commit `45bd1789`, matched n=12 (LEARNED +0.055) |
| `bisect/new_f05c63aa.jsonl` | Exp 48 A/B — current code `f05c63aa`, matched n=12 (LEARNED +0.079) |
| `bisect/run_ab.sh`, `bisect/nomove_rate.py`, `bisect/*.log` | The A/B harness + the no-move-rate probe |
| `sweep/ew0.4.jsonl`, `sweep/smoke_explore0.jsonl` | Explore-weight sweep arms (ew=0.4; ew=0 smoke — zero actions) |
| `sweep/run_sweep.sh`, `sweep/*.log` | Sweep harness + logs |
| `*/…/sim_reports/*/report.json` | Per-run sim reports |
| `*/…/util/lane_decisions.jsonl` | Per-run lane/provenance decision logs |
| `*/…/sim_reports/*/actions.jsonl.gz` | Per-run action streams (gzipped; ~32 MB raw total) |

## What is NOT here

`mother_log.jsonl` (678 MB total — narrator transcripts) and `bio_telemetry.jsonl`
(311 MB total) are excluded from git for size. The complete, unmodified capture —
including those files and the uncompressed `actions.jsonl` — is archived at:

```
~/Maxim-experiment-archives/heartbeat_1.1_2026-08/   (operator machine, full 1.6 GB copy)
```

The original untracked copy in `Maxim-heartbeat-1.1/heartbeat_data/` is now redundant;
the worktree is safe to prune.

## Sim-Short chapter — added 2026-08-24 (records from big-mac-mini `~/.maxim/sim_reports/`)

The 2026-08-18/19 Sim-Short heartbeat runs (mistral-7b, `Maxim-exp48-rebaseline`
worktree, `llm.n_ctx 8192`) that RE-VALIDATED the Exp 10 row and re-attested Exp 09
(walk log in [behavioral_graduation_candidates.md](../../../plans/behavioral_graduation_candidates.md)).
Copied verbatim; `actions.jsonl` and `bio_telemetry.jsonl` gzipped.

| Path | Experiment | What it is |
|---|---|---|
| `sim_short_2026-08-18/20260818_004321/` | **Exp 09** reflexes | dragon-cave sim, 8 turns (`20260818_004107` on the mini is a 0-turn false start, not copied) |
| `sim_short_2026-08-18/20260818_010010/` | **Exp 10** phase 1 | dungeon, 8 turns; hippocampus closes at **108** memories |
| `sim_short_2026-08-18/20260818_233844/` | **Exp 10** phase 2 (resume of p1) | dungeon, 8 turns; hippocampus opens at 108 and closes at **535** — the row's re-validation evidence |
| `sim_short_2026-08-18/20260819_000740/` | **Exp 10** phase 3 (garden, same resume) | 5 turns; post-consolidation store **295** (no negative transfer) |

Each dir: `report.json`, `actions.jsonl.gz`, `bio_telemetry.jsonl.gz`, and the
persisted `aut_{hippocampus,nac,ec,atl,scn}.json`. The hippocampus counts above were
re-derived from the copied `aut_hippocampus.json` files on 2026-08-24.


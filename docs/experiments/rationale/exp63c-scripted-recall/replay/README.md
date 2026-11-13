# Exp 63c power replay and fidelity check: the freeze re-run (2026-11-10)

Run on main d40eb4c0 (#1620 merged), offline: no rig, no LLM, `MAXIM_OFFLINE=1`, scratch `HOME`, the repo's `src` on
`PYTHONPATH`. A design step, not data. The prereg's §Power replay ("Freeze result") cites these files.

- `replay.py <script> <intents> <policies...>`: the replay (env variants `R1MODE=neutral|bare`, `R2MODE=nogoal`,
  `NAUT=2`, `TAG=`); `simcore.py`: the G1 simulator and the ranker pin; `script_v5x.py`: script v4; `intents_v4.py`:
  the AUT intent policies; `tables.py` / `blacklist.py`: the key tables and the vocabulary blacklist.
- `fidelity.py`: the judge-free half of the fidelity check (every `simulate()` call re-run through the shipped recall
  over a real Hippocampus).
- Outputs: `tables_v5x.out`; `x_*.out` and `replay_*.json` per variant; `fidelity_base.out`, `fidelity_neutral_nogoal.out`, `fidelity_naut2.out`; `blacklist.json` (669 keys); `freeze_tables.txt` (k,
  contested wins and completion per variant).

The scripts read 63b's committed phase-2 store from `docs/experiments/data/63b_carried_recall/` through an absolute path
in the scratch directory they ran from (`B` in replay.py); re-running them elsewhere needs that path edited.

## Judge half of the fidelity check (2026-11-12)

`judge_half/` holds the prereg's "judge half, owed before the smoke" (§Fidelity check): every R3d call the frozen
replay makes is also judged by the shipped 63c judge port in `scripts/o19_verdict.py` (`exp63b_turn` under
`c_key_tokens`, every valid tie order; `exp63c_control_status`; C5(k) and C5(q); `_exp63c_report`), and the port's
decisive verdict is compared with the simulator's. It is a design check, not data. The prereg is frozen and is not
edited; this record discharges the owed line.

- Code under test: main **1f5faba3** (#1649 merged), `maxim` imported from that tree (`provenance.txt`);
  `sha256(scripts/o19_verdict.py)` = `47fbc5ca53b2da7315e3bae3d1dfe7ec00c0e696cb3dc844de858785d03af976`, the same file
  as the harness head 1a60a1f2, where the binding run first went; the six outputs here are line-for-line identical to
  that run's.
- Result, six variants × 16 policies × 8 turns = 768 R3d calls: **0 decisive mismatches**, 0 non-conforming, 0 over
  the order bound, **C5(k) = 0, C5(q) = 0**, both controls armed for every policy, the script lines equal
  `PROTOCOL["63c"]`'s literals. k per policy equals the freeze table; decisive cue turns not uncontested-literal: base
  4/53, neutral 4/53, neutral + no goal 4/52, bare 4/53, two-records 5/46, neutral + two-records 5/46. The forecast's
  k values are therefore no longer only simulator-judged.
- Probes (`probes.out`), which must fail the check: with the carried records hidden from the port, 53/53 decisive
  turns disagree; with a control word planted in a carried record, control turn 2 reads `contaminated`.
- Re-run: `bash docs/experiments/rationale/exp63c-scripted-recall/replay/judge_half/run_judge_half.sh <python> <out>`
  from the repo root of the commit under test (offline; scratch `HOME`; the repo's `src` on `PYTHONPATH`).

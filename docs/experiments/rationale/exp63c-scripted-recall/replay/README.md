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

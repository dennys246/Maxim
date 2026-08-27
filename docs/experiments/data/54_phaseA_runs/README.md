# Exp 54 Phase A — per-run provenance (S4 record, 2026-08-27)

One directory per sub-sim (`<arm>_seed<n>_ew1.5`, 36 runs) holding the lane/provenance
decision log (`util/lane_decisions.jsonl`) each sub-sim wrote, plus the harness launch
stamp (`launch.txt`: repo `93887e6e` on `main`, Python 3.12.13 in the mini's venv) and the
harness stdout (`harness.out`: narrator preflight → mistral-7b GGUF; per-run `ok` lines;
`done: 36 runs recorded, 0 failed`). The campaign record — one row per run with per-act
directedness, fed/credited rates, the S3 counters and the apparatus stamps (`credit=relief`,
`stimulus_order=shuffled`, `embodiment`, `executed_git_hash=93887e6e`,
`substrate_actions_per_turn_env=null`) — is [`../54_phaseA_nursery.jsonl`](../54_phaseA_nursery.jsonl);
the gate v3 verdict is in [`../../54_nurture_reachy_body.md`](../../54_nurture_reachy_body.md).
The ten agents Phase B/C load are copied byte-for-byte into
[`../54_agents/`](../54_agents/) with SHA-256s in [`../54_agents_manifest.json`](../54_agents_manifest.json).

**Not committed (size):** each run's `mother_log.jsonl` (~11 MB; the full per-turn JSONL incl.
the `sim_mother` telemetry and the `sim_nac_recommend` decision provenance),
`harness_logs/run.log` (~4 MB), `sim_reports/` (report.json + every run's persisted
`aut_nac.json`/`aut_ec.json`). The complete, unmodified workdir is archived at

```
~/Maxim-experiment-archives/exp54_phaseA_2026-08-27/   (operator Mac: exp54_mother_logs.tgz + exp54_phaseA_prov.tgz + the two records)
big-mac-mini ~/exp54/phaseA/                            (the original)
```

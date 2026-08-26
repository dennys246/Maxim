# Exp 52 Phase B — per-run provenance (S4 record, 2026-08-25)

One directory per sub-sim (`<arm>_seed<n>_ew1.5`, 36 runs) holding the lane/provenance
decision log (`util/lane_decisions.jsonl`) each sub-sim wrote. The campaign record —
one row per run with per-act directedness, fed/credited rates, the S3 counters, and
the apparatus stamps (`credit`, `stimulus_order`, `embodiment`, `executed_git_hash`) —
is [`../52_phaseB_embodied.jsonl`](../52_phaseB_embodied.jsonl); the analyzer verdict is
in [`../../protocols/exp52_nurture_preregistration.md`](../../protocols/exp52_nurture_preregistration.md).

**Not committed (size):** each run's `mother_log.jsonl` (~11 MB; the full per-turn
JSONL incl. the `sim_mother` telemetry and the `sim_nac_recommend` decision provenance
with `learned_margin` / `explore_decisive`), `harness_logs/run.log`, `sim_reports/`,
and the persisted substrate. The complete, unmodified workdir (1.0 GB) is archived at

```
~/Maxim-experiment-archives/exp52_phaseB_2026-08-25/phaseB/   (operator Mac; copied from big-mac-mini ~/exp52/phaseB)
```

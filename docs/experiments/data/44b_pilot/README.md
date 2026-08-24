# Exp 44b pilot — raw captures (S4 record, committed 2026-08-24)

Copied verbatim from big-mac-mini `~/exp44b/pilot/` (the pilot machine; see
[44b_pilot.md](../../44b_pilot.md)). Per-arm isolated `MAXIM_DATA_HOME`s also held the
GGUF model files (≈20 GB) — those are excluded; everything the analyses read is here.

| Path | What it is |
|---|---|
| `manifest.jsonl` | the campaign harness's run record (`scripts/exp44/campaign.py`); sub-runs stamp `executed_git_hash` — the pilot ran across three commits on 2026-08-10 (`7b51b949`, `895b8134`, `9e3ebc4e`), all recorded per row |
| `stats.json` | the harness's own pooled summary (descriptive; 1 seed/arm) |
| `run.out` | harness stdout |
| `arms/<arm>/seed1/capture.jsonl` | paired full/ablated prompts per decision (36 / 30 / 35 rows) — the S4 non-stationarity inputs |
| `arms/<arm>/seed1/requery/qwen2.5-32b-instruct__<hash>__e8t0.7.jsonl` | offline re-query results (flip scoring) |
| `arms/<arm>/seed1/util/lane_decisions.jsonl` | per-run lane/provenance decision log |

Analysis products: [`../44b_s4_nonstationarity/`](../44b_s4_nonstationarity/README.md).

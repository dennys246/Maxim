# Exp 44b S4 — non-stationarity analysis outputs (roadmap 1.1 item 11)

Produced 2026-08-24 on the operator Mac from the committed pilot captures
([`../44b_pilot/`](../44b_pilot/README.md)) with
`scripts/exp44/analyze_nonstationarity.py` at `main` **`7aed652b`** (`PYTHONPATH=<repo>/src`;
the parser imports the production `ANNOTATION_SOURCE_SEPARATOR`, so the in-repo
renderer contract is the one that was parsed):

```
python scripts/exp44/analyze_nonstationarity.py \
  --capture docs/experiments/data/44b_pilot/arms/<arm>/seed1/capture.jsonl \
  --results docs/experiments/data/44b_pilot/arms/<arm>/seed1/requery/<model>__<hash>__e8t0.7.jsonl \
  --json docs/experiments/data/44b_s4_nonstationarity/<arm>_seed1.json
```

`<arm>_seed1.json` is the analyzer's report; `<arm>_seed1.txt` is its stdout. The
recorded result and its reading live in [44b_pilot.md §S4](../../44b_pilot.md).

| arm | flip rate 1st → 2nd half | non-target band tier 1st → 2nd | target tier | untreated pairs (determinism) |
|---|---|---|---|---|
| A_green_safe | 0.667 → 0.333 | ≈1.8 → 1.0 | 1.94 → 2.00 | 6, flip 0.000 |
| B_purple_safe | 0.600 → 0.533 | ≈2.0 → 1.0 | 2.00 → 2.00 | 0 |
| CTRL_transplant_A_into_B | 0.412 → 0.611 | ≈1.8 → 1.0 | 1.79 → 1.00 | 5, flip 0.000 |

One seed per arm; descriptive. Scope caveat from the script header: BAND
non-stationarity only — the S1 credit-source gloss is stripped before banding.

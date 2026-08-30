# Fail-loud Stage 2 — the measured baseline

**Measured 2026-08-30 at `6f3f3b7d`, clean tree.** Plan:
[../../../plans/measurement_path_fail_loud.md](../../../plans/measurement_path_fail_loud.md) §Stage 2.
Tool: [`scripts/fail_loud_stage2.py`](../../../../scripts/fail_loud_stage2.py).

## Why this exists

[`god_function_decomposition.md`](../../../plans/god_function_decomposition.md) sets a per-PR
behaviour gate for every extraction: *"zero new `swallowed_exception` firings vs the Stage-2
baseline."* Stage 2 had never been run, so that baseline did not exist and the gate could not
fail. This directory is the artifact the gate cites.

## The result

**Zero firings**, over both modes, across 75,654 log records. All 50 Stage-1 instrumented
sites were silent.

| mode | command | records | firings | exit |
|---|---|---|---|---|
| substrate | `scripts/orient_substrate/2_full_path_probe.py` (no LLM) | 72,239 | 0 | 0 |
| generative | `maxim --sim "test basic recall" --interactive false --sim-max-turns 3 --llm qwen2.5-14b --sandbox tmpdir` | 3,415 | 0 | 4 |

Per the plan, "zero firings = the swallows are dead defensive weight, safe to narrow" — this
is the Stage-3 green light, not a null result.

## What is NOT claimed

- **Per-site coverage is not proven.** A zero over these two modes does not establish that all
  50 sites are unreachable. It establishes the comparison baseline. Proving per-site
  reachability needs a coverage run; that is not done here and is not claimed.
- The instrumented-site count is **50**, not the 48 recorded at PR #487 — two have been added
  since. `scripts/fail_loud_stage2.py inventory` recomputes it; do not trust a prose number.

## Run conditions worth knowing

The generative run **terminated with the D12 stall hard-abort (exit 4)** after writing its
full report: the local 14B exceeds the orchestrator lane's stall threshold on this box. The
capture is not truncated — it carries 3,415 records against 3,154 in a prior exit-0 run of the
same command, and both runs measured zero.

**The instrument was verified live in both captures** rather than assumed. The `MAXIM_LOG_FILE`
handler is attached at `DEBUG` (`utils/logging.py::_ensure_jsonl_file_handler`) and both
captures contain DEBUG records, so a firing at WARNING-first or DEBUG-after would have been
recorded; a separate self-test that forced a site to fire was observed arriving in the JSONL.
A zero from an unverified instrument would have been worthless.

## Reproducing / using the gate

```bash
export PYTHONPATH="$PWD/src"

# the denominator
python scripts/fail_loud_stage2.py inventory

# re-derive this baseline from the committed captures (should print 0 firings)
python scripts/fail_loud_stage2.py baseline \
  --capture substrate=docs/experiments/data/fail_loud_stage2/substrate.jsonl.gz \
  --capture generative=docs/experiments/data/fail_loud_stage2/generative.jsonl.gz \
  --out /tmp/rederived.json

# gate an extraction PR: capture the same two modes, then
python scripts/fail_loud_stage2.py check --capture substrate=/tmp/new.jsonl
```

`check` exits 1 on a new `(file, exception-type)` pair and 2 when the baseline is missing — a
gate must not pass by citing an artifact that is not there.

## Files

- `baseline.json` — the derived artifact (git hash, dirty flag, capture digests, notes).
- `substrate.jsonl.gz`, `generative.jsonl.gz` — the raw captures. `sha256` in `baseline.json`
  is over the **decompressed** bytes, so gzipping did not change the recorded digest.

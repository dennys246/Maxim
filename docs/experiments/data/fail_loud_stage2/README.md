# Fail-loud Stage 2 — the measured baseline

**Captures taken 2026-08-30 from a clean tree at `6f3f3b7d`; baseline artifact regenerated at
`258389f8`, also from a clean tree (see Provenance below — the first cut of this file got that
wrong).** Plan:
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
- **No unfired-site count is published.** An earlier version of `baseline.json` carried one and
  it was wrong in a way that always overstated dead instrumentation. A fired site cannot be
  matched back to an inventory entry across a refactor at all — line numbers move and functions
  get renamed, which is the very reason the gate keys on `(basename, exc_type)`. Compare
  `instrumented_site_count` against `distinct_fired_site_count` and draw no more than that.
- The instrumented-site count is **50**, not the 48 recorded at PR #487 — two have been added
  since. `scripts/fail_loud_stage2.py inventory` recomputes it; do not trust a prose number.

## Provenance, including what went wrong with it

The captures were taken from a clean tree at `6f3f3b7d`. **The first cut of this artifact was
then REGENERATED minutes later with an uncommitted patch in the working tree**, so it stamped
`working_tree_dirty_src_scripts: true` while this README and the plan both said "clean tree" —
an honest machine stamp under a human misreading, which is precisely the Exp 53/53b failure
mode the same branch's Cluster E is about. The pre-merge review round caught it.

Two things changed as a result. The tool no longer hand-rolls its dirty check: it routes through
`_provenance.py::preflight_gated_record_or_exit`, which **refuses** (exit 3) rather than
stamping, unless `--allow-dirty` is passed — and that then stamps `allow_dirty: true` into the
artifact so a write-up cannot omit it. And `lint_harness_provenance.py` gained a third family
keyed on *where* records land rather than on how a harness runs, because this script escaped
both existing families: it spawns no `maxim` and does not live under `scripts/orient_*/`.

**The instrument was verified live in both captures** rather than assumed. The `MAXIM_LOG_FILE`
handler is attached at `DEBUG` (`utils/logging.py::_ensure_jsonl_file_handler`) and both
captures contain DEBUG records, so a firing at WARNING-first or DEBUG-after would have been
recorded; a separate self-test that forced a site to fire was observed arriving in the JSONL.
A zero from an unverified instrument would have been worthless.

The generative run **terminated with the D12 stall hard-abort (exit 4)** after writing its full
report: the local 14B exceeds the orchestrator lane's stall threshold on this box. The capture
is not truncated — it carries 3,415 records against 3,154 in a prior exit-0 run of the same
command, and both measured zero.

## The extraction's behaviour gate, verifiable rather than asserted

`substrate_post_extraction.jsonl.gz` is the same probe re-run **after** the Cluster B
extraction. With the wall-clock `t` field stripped, the two captures hash identically:

```
sha256(pre,  t-stripped) = 978d512a40cac52b3f8321737995a64aa85abd7605a888d2b9679113ae75e048  (72,239 records)
sha256(post, t-stripped) = 978d512a40cac52b3f8321737995a64aa85abd7605a888d2b9679113ae75e048  (72,239 records)
```

Reproduce:

```bash
python3 - <<'EOF'
import gzip, json, hashlib
D = "docs/experiments/data/fail_loud_stage2"
def digest(p):
    h, n = hashlib.sha256(), 0
    for line in gzip.open(p, "rb"):
        try: r = json.loads(line)
        except Exception: continue
        if isinstance(r, dict):
            r.pop("t", None); h.update(json.dumps(r, sort_keys=True).encode()); n += 1
    return h.hexdigest(), n
print(digest(f"{D}/substrate.jsonl.gz") == digest(f"{D}/substrate_post_extraction.jsonl.gz"))
EOF
```

The substrate probe does **not** enter `run_agentic_loop`, so this equality is evidence that the
bio path is unchanged, not that the extracted sections behave identically. That evidence is the
fast suite plus `tests/unit/test_loop_section_extraction.py`, which calls both extracted
functions directly.

## Using the gate

```bash
export PYTHONPATH="$PWD/src"

python scripts/fail_loud_stage2.py inventory          # the denominator

python scripts/fail_loud_stage2.py check \
  --capture substrate=/tmp/new_substrate.jsonl \
  --capture generative=/tmp/new_generative.jsonl
```

`check` exits 1 on a new `(file, exception-type)` pair or when instrumentation has been
deleted, and 2 when the baseline is missing or the captures cannot support a verdict (empty,
unparsable, or below `--min-lines`). A gate must not pass by citing an artifact that is not
there, nor by reading a capture that measured nothing.

## Files

- `baseline.json` — the derived artifact (git hash, dirty + allow_dirty flags, capture digests, notes).
- `substrate.jsonl.gz`, `generative.jsonl.gz` — the pre-extraction captures the baseline is built from.
- `substrate_post_extraction.jsonl.gz` — the post-extraction re-run backing the equality above.

`sha256` in `baseline.json` is over the **decompressed** bytes, so gzipping did not change the
recorded digest.

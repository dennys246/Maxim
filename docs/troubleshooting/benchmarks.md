# Benchmark Troubleshooting

Common issues when running `maxim --sim benchmark` and how to resolve them.

---

## "Scenario X has no percepts -- will complete immediately"

This is normal when running a **suite file**. The suite itself has no percepts; it references child scenario files that contain them. The warning means the suite YAML was correctly identified as a suite, not a standalone scenario.

If a **standalone scenario** shows this warning, check that it has a `percepts:` section with at least one entry containing a `cli_input` value.

---

## Turns timing out (0 chars)

The AUT's agentic loop may not be ready when campaign turns start arriving. This typically means the LLM backend has not finished loading.

**Fixes:**
- Check the logs for "LLM model loaded and ready" before the first turn fires.
- If using a large model (14B+), the initial load can take 30-60 seconds. Wait for the warmup to complete.
- Verify the model profile name is correct (`--models mistral-7b`, not `--models mistral`).

---

## "No turns found in campaign"

The scenario YAML has no `percepts` with `cli_input` values. Every percept that should be delivered to the AUT needs a non-empty `cli_input` field:

```yaml
percepts:
  - at: 0
    cli_input: "Tell me about your environment."   # required
    salience: 0.8
```

A percept with an empty or missing `cli_input` is silently skipped during campaign injection.

---

## LLM loads twice

Normal behavior. The AUT and orchestrator share a single LLM router, but the warmup message prints on each lane's first use. You will see two "model loaded" messages -- one for the `large` lane and one for the `medium` lane (or whichever lane the orchestrator uses first). This does not mean two model instances are loaded into memory.

---

## Docker unavailable fallback

If Docker Desktop is not running (or not installed), the benchmark falls back to `TmpdirSandbox` with reduced filesystem isolation. You will see a log line like:

```
Docker unavailable, falling back to TmpdirSandbox
```

This is fine for benchmarking purposes. TmpdirSandbox provides the same interface; only the isolation boundary differs. The benchmark results are not affected. Install Docker Desktop if you need full container isolation for safety-boundary scenarios.

---

## Baseline comparison shows no deltas

Model names must match exactly between runs. If the baseline was generated with `--models mistral-7b` and the new run uses `--models Mistral-7B`, the comparison engine will not find matching entries.

**Fixes:**
- Open the baseline `benchmark_report.json` and check the exact model names under `results`.
- Use identical `--models` strings across runs.
- Profile names are case-sensitive and must match the profiles defined in `src/maxim/models/language/config.py`.

---

## Score is 0.000

All scored metrics returned zero. This usually means the scenario did not have enough turns for the bio-systems to activate.

**Minimum turn counts for bio-system activation:**
- Memory formation (hippocampus): ~5 turns for a memory to reach SHORT_TERM tier.
- Causal learning (NAc): ~3 turns with clear cause-effect structure.
- Temporal indexing (SCN): ~2 turns with explicit time references.
- Pain detection: 1 turn with a genuine threat, but the proprioception system needs at least 1 prior turn to establish a baseline.

Also check that the scenario's `benchmark.metrics` list includes metrics that the scenario actually exercises. A memory-focused scenario with only `causal_link_count` in its metrics list will score 0 if no causal structure is present.

---

## Out of memory on long benchmarks

Each scenario spawns a full agentic pipeline (all 5 agents + bio-systems). However, the LLM router is shared across scenarios within a model, so GPU memory scales with model size, not scenario count.

**Fixes:**
- Use `--runs 1` for initial testing. Multiple runs multiply memory pressure from concurrent pipeline state.
- If running multi-model benchmarks, models are tested sequentially (not in parallel), so peak memory is determined by the largest single model.
- Close other GPU-consuming processes before starting a benchmark run.
- For 14B+ models on 16GB VRAM, ensure no other CUDA processes are running (`nvidia-smi` to check).

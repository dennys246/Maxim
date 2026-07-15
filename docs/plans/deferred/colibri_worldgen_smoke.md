# Colibrì world-generation smoke test

> **DEFERRED (2026-07-15 plans audit):** Hard-blocked upstream — colibrì's OpenAI-compatible server emits degenerate output (stuck GLM chat-template/thinking-mode bug, 2026-07-14); the engine itself works via `coli chat`. Setup gotchas + pre-registered decision rule retained here for the retry. **Revive when:** upstream fixes the `openai_server.py` degenerate-generation bug (re-check its tracker in a few weeks), or the cloud-only baseline arm is run as prep. If upstream stays dead for a quarter, escalate to archive.


**Status:** colibrì arm BLOCKED (2026-07-14) — upstream `openai_server.py` bug; engine itself works.
Re-check upstream in a few weeks; cloud baseline arm can run any time.

## Outcome so far (2026-07-13/14 Mac Mini run)

- Build, Metal kernel tests, weights download (int8-MTP variant confirmed), and
  `coli doctor` all green. Engine is REAL: `./coli chat` streams coherent prose.
- **API server produces degenerate output**: `/v1/chat/completions` yields repeated
  `"."` on `reasoning_content` + one content token + early EOS — with thinking on OR
  off, fresh OR resumed KV (`.coli_kv` deleted), default OR pinned sampling
  (temp 0.7 / top_p 0.95), non-stream and stream paths, `--ngen 512`. Server also
  emits `reasoning_content` when `enable_thinking` is false → its GLM chat template
  is stuck in thinking mode; model degenerates inside it. Chat REPL uses a
  different template path and works. Upstream issue to file (repro below).
- Not known upstream as of 2026-07-14 (issue tracker checked).
- Timing data (48 GB M-series Mini, `--ram 36`, warm-expert cap 13/layer, Metal on):
  cold prefill ~46 s for a 13-token prompt (~0.6 s/layer × 78 layers); warm calls
  20–74 s for ≤21-token prompts. Prefill barely benefits from the warm expert
  cache — long foundry prompts (1–2k tokens) were never measured because
  generation was broken. TTFT alone already implies hours-per-component.
- **Operational-maturity verdict: failed at first serve.** Setup consumed a night
  + a morning (hf_xet wedge on Python 3.14 included). This axis is itself
  smoke-test data per the decision rule.

### macOS setup gotchas (validated, keep for any retry)
1. `brew install libomp` before `make glm METAL=1`.
2. `ulimit -n 8192` in the serve shell — engine mmaps 144+ shards; default 256 fds
   crashes at load ("Too many open files").
3. Default RAM budget is 8 GB (< 10.9 GB dense core, 0 warm experts). Always pass
   `--ram <total-12>` (48 GB box → `--ram 36` → 19 GB warm experts, cap 13/layer).
4. `hf download` on Python 3.14 wedged twice (thread-join hangs); resume with
   `HF_HUB_DISABLE_XET=1` worked instantly.
5. Server persists conversation KV to `<model_dir>/.coli_kv` and resumes it across
   restarts; delete when testing.
**Question:** does GLM-5.2 (744B, via colibrì disk-streaming) produce measurably better SEM components than the cloud baseline on the foundry gauntlet — enough to justify a long-horizon offline world-generation pipeline?
**Decision rule (pre-registered):** the colibrì arm must *clearly* beat the cloud arm on promoted-rate AND mean gauntlet score (4 bio-engagement dimensions). Ties or marginal wins → cloud wins (wall-clock and operational cost are not close). If colibrì wins → write an engine-agnostic `offline_worldgen_pipeline.md` shell plan with the lead-time argument as motivation; colibrì is one interchangeable backend in it, not the headline.

## Context

- Colibrì: pure-C engine, Apache 2.0, v1.0, single maintainer (JustVugg, July 2026). Runs GLM-5.2 744B MoE by keeping ~10 GB dense core in RAM and streaming ~370 GB of int4 experts from NVMe. 0.05–1 tok/s. OpenAI-compatible `./coli serve`.
- Decided (2026-07-13): NOT a Maxim dependency, NOT a co-occurring 1.1/1.2 plan. Zero-code integration via `MAXIM_LLM_CONFIG` profile — verified end-to-end against a mock endpoint on the M2 laptop (profile → `colibri` provider → `_OpenAIBackend` → localhost → `generate_json` parses).
- Host: **Mac Mini** (Apple Silicon → Metal backend). The M2 laptop build (`make glm METAL=1`) succeeded after `brew install libomp`; same steps apply on the Mini.

## Mac Mini setup

```bash
# 0. Preflight: need ≥400 GB free NVMe
df -h /

# 1. Build (validated on M2 laptop 2026-07-13)
brew install libomp
git clone https://github.com/JustVugg/colibri.git && cd colibri/c
make glm METAL=1
make metal-test        # verify Metal kernels

# 2. Weights — pre-converted int4 mirror (~370 GB):
#    https://huggingface.co/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp
# CRITICAL: use the int8-MTP-heads variant. Verify MTP head file sizes are
# 3.5 GB / 5.3 GB / 1.0 GB. The int4-MTP variant (1.7/2.6/0.5 GB) gives 0%
# draft acceptance and will be dramatically slower.

# 3. Serve (start + warm BEFORE launching foundry; wait_ready only waits 120s)
COLI_METAL=1 COLI_MODEL=/path/to/glm52_i4 COLI_API_KEY=local-secret \
  ./coli serve --host 127.0.0.1 --port 8000 --model-id glm-5.2-colibri
curl -s http://127.0.0.1:8000/v1/models   # sanity
```

## Maxim-side config (verified against mock endpoint)

Save as `~/colibri_llm.json` on the Mini. `allow_local_endpoints: true` is required
(SSRF guard blocks `http://` localhost otherwise); `timeout_s: 7200` because
prefill at disk-streaming speed can stall the byte stream far past the 60s
default; zero-price pricing entry silences the CostTracker warning.

```json
{
  "enabled": true,
  "profile": "glm-5.2-colibri",
  "profiles": {
    "glm-5.2-colibri": {
      "backend": "openai_compatible",
      "model": "glm-5.2-colibri",
      "model_base": "glm-5.2-colibri",
      "prompt_style": "chatml",
      "n_ctx": 8192,
      "api_key_env": "COLIBRI_API_KEY",
      "base_url": "http://127.0.0.1:8000/v1"
    }
  },
  "providers": {
    "colibri": {
      "type": "openai_compatible",
      "base_url": "http://127.0.0.1:8000/v1",
      "model": "glm-5.2-colibri",
      "api_key_env": "COLIBRI_API_KEY",
      "allow_local_endpoints": true,
      "timeout_s": 7200,
      "max_retries": 0,
      "cost_visible": false,
      "pricing_required": false
    }
  },
  "routing": { "provider_priority": ["colibri"] },
  "pricing": { "glm-5.2-colibri": { "input_price": 0.0, "output_price": 0.0 } }
}
```

Note: user `profiles.yml` cannot express this (schema frozen at 1.0, llama_cpp/pytorch
only) — `MAXIM_LLM_CONFIG` + llm.json is the sanctioned route and needs no code change.

## Arms

Same theme, same genre, same count in both arms. Pick a theme with no existing
components (avoid ComponentIndex dedup asymmetry).

```bash
# Pilot first: ONE component through colibrì, timed. If a single component
# takes > 2 h, shrink the overnight batch or stop — the smoke answer is
# already "not viable at this speed".
COLIBRI_API_KEY=local-secret MAXIM_LLM_ENABLED=1 MAXIM_LLM_CONFIG=~/colibri_llm.json \
  maxim --foundry "deep-sea pressure ruins" --foundry-genre scifi \
        --foundry-count 1 --language-model glm-5.2-colibri --interactive false

# Arm A (overnight, colibrì) — size count from the pilot timing
COLIBRI_API_KEY=local-secret MAXIM_LLM_ENABLED=1 MAXIM_LLM_CONFIG=~/colibri_llm.json \
  maxim --foundry "deep-sea pressure ruins" --foundry-genre scifi \
        --foundry-count 5 --language-model glm-5.2-colibri --interactive false

# Arm B (cloud baseline, minutes)
maxim --foundry "deep-sea pressure ruins" --foundry-genre scifi \
      --foundry-count 5 --language-model claude-sonnet --interactive false
```

## Measurement

- Per-arm: promoted / review / rejected counts + per-component 4-dimension
  gauntlet scores (both printed by the foundry run; artifacts in the run's
  `output_dir`).
- Wall-clock per component (colibrì arm) — this is the pipeline-sizing number
  for any future worldgen plan.
- Human eyeball: YAML richness (sensors/modulators/affordances depth, synonym
  quality) side by side.
- Cleanup: promoted components from BOTH arms are experiment artifacts — review
  before letting them stay in `~/.maxim/components/` (tool-bloat lesson: more
  components ≠ better).

## Caveats

- Colibrì arm quality could be bottlenecked by prompt fit (foundry prompts are
  tuned on instruction-following cloud models; GLM-5.2 reasoning blocks are off
  by default via the OpenAI endpoint). One arm-tweak retry is allowed if raw
  output is malformed JSON rather than bad content — record it.
- Don't run the colibrì arm on the leader; the Mini serves nothing else.
- The engine is a month-old one-person project — if the server wedges mid-batch,
  that itself is smoke-test signal (operational-maturity axis).

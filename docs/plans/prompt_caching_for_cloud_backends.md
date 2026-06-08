# Prompt Caching for Cloud Backends

**Status:** DRAFT 2026-06-08 (post-Sonnet/Haiku rate-limit cascade)
**Author:** Denny + Claude
**Motivates:** Exp 37 cross-model characterization (Sonnet replication blocked by
Anthropic tier-1 ITPM cascade on 2026-06-08); future cloud-LLM experiments.
**Out of scope:** New backend types; substrate / bio-system changes; LLM router
provider-fallback redesign.

## Why this exists

2026-06-08 Sonnet replication attempt cascaded into Anthropic tier-1 rate-limit
exhaustion after just 2 arms. Each cradle sub-sim consumed ~120K input tokens
across 12 turns (~30K ITPM sustained), right at the tier-1 ceiling. When Arm B
pushed over the limit, the bucket exhausted and every subsequent sub-sim
launched into 429-throttled API state and never recovered.

The root cause is architectural, not provider-specific: **the cradle scenario
re-sends a ~100K-token system prompt (entity manifests, scene state, bio-system
descriptions, sense_tools registry) every turn, identical across all 12 turns of
each sub-sim.** That's the textbook prompt-caching use case — large stable
prefix, varying suffix (the user's turn).

Anthropic's [rate-limits documentation](https://platform.claude.com/docs/en/api/rate-limits)
explicitly states that `cache_read_input_tokens` do **NOT** count toward ITPM on
Sonnet 4.x, Haiku 4.5, and Opus 4.x. With a single `cache_control: {"type":
"ephemeral"}` breakpoint at the end of the system prompt, the first turn writes
the cache (~100K tokens count toward ITPM at 1.25× billing), and turns 2-12 read
from cache (0 tokens count toward ITPM, billed at 0.1× input price).

**Effective ITPM reduction per sub-sim: ~120K → ~20K (6× lower).**
**Effective input cost reduction per sub-sim: ~75% on cached portion.**

This solves the tier-1 cascade structurally without requiring a tier upgrade,
and makes future cloud-LLM experiments dramatically cheaper across the board.

## What already exists in main

Surprise from the 2026-06-08 audit: the cache_control wiring is **already
implemented** in [`_AnthropicBackend._build_system_blocks`](../../src/maxim/models/language/anthropic_backend.py).
The infrastructure is shipped:

- `_build_system_blocks(system)` builds a `[{"type": "text", "text": system,
  "cache_control": {"type": "ephemeral"}}]` block when caching is enabled,
  falls back to `[{"type": "text", "text": system}]` otherwise.
- `_prompt_cache_enabled()` reads `cfg.get("prompt_cache")` from the per-provider
  config; supports both bool shorthand and `{"enabled": true}` dict form.
- Usage tracking already reads `cache_read_input_tokens` and
  `cache_creation_input_tokens` for billing (lines 285, 359).

**The gap is that no profile in [`profiles.yml`](../../src/maxim/_data/profiles/) sets
`prompt_cache.enabled: true`, so the flag is off-by-default and the wiring
never fires in production.** That's the 2026-06-08 incident's structural cause.

What does NOT exist:
- Same wiring in `_OpenAIBackend` (OpenAI also benefits from explicit
  `cache_control` markers for prompts >1024 tokens; their automatic caching is
  separate and runs on top of explicit markers).
- Cache-hit instrumentation in router-level logs (you can see token-counts in
  the backend's billing accumulator, but there's no per-request structured
  event saying "cache hit: 95K read / 5K new" for ops visibility).
- An audit of [`PromptBuilder`](../../src/maxim/agents/prompt_builder.py) for
  silent cache invalidators (timestamps, UUIDs, session IDs in the stable
  prefix would defeat caching even with the flag on).
- A CLAUDE.md invariant making the "stable prefix MUST stay stable" rule
  explicit so future PromptBuilder changes don't silently break caching.

## The prefix-invariant rule (load-bearing)

Per [Anthropic's prompt caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching),
the cache is a **prefix match** — any byte change anywhere in the prefix
invalidates everything after it. Specifically:

> A single byte difference at position N — a timestamp, a reordered JSON key,
> a different tool in the list — invalidates the cache for all breakpoints at
> positions ≥ N.

This means the wiring fix is necessary but not sufficient. If `PromptBuilder`
interpolates anything dynamic into the stable portion of the system prompt, the
cache will silently fail every request (showing `cache_read_input_tokens: 0`
across repeated requests with notionally-identical-looking prompts).

**The audit MUST happen before the flag flip.** Otherwise we ship a "fix" that
silently does nothing and we waste another night debugging it.

Known suspects from the codebase (need verification):
- Drive states in system prompt (per-tick values change every call)
- Bio-state annotations (Wire-A cluster bias, Wire-1 variance) — these are
  literally per-turn substrate values that vary
- Tool descriptions (sense_tools registry changes as scene state evolves)
- Acting Coach modulation (drive guidance varies per tick)
- Timestamps via `build_datetime_section`
- Workspace manifest scanning (`scan_workspace_entries` could enumerate in
  non-deterministic order)

**This is GOOD news for the audit** — most of these are deliberately
per-turn, which means the architectural pattern is correct: put dynamic stuff
LATE in the prompt (in the messages/user turn) and keep the system prompt
stable. The bad news is some of them may have been silently bleeding into the
system prompt for years and nobody noticed because no caching was on to
expose them.

## Three-phase plan

### Phase 0 — Audit PromptBuilder for silent invalidators (READ-ONLY)

Goal: identify every dynamic field that flows into the system prompt and
classify each as (a) genuinely-stable-belongs-in-system, (b)
dynamic-must-move-out-of-system, or (c) cacheable-with-care (changes only at
session boundaries, not turn boundaries).

Method:
1. Read [`PromptBuilder.assemble`](../../src/maxim/agents/prompt_builder.py)
   and every `build_*_section` function. Trace what flows into the `system`
   string vs the `messages` array.
2. Hash-test: for two consecutive turns of a single cradle sub-sim, compute the
   SHA256 of the system prompt at each call. Different hash = cache miss.
3. Bisect: if hashes differ, binary-search the section list to find the
   culprit(s).
4. Classify each finding per the above rubric.
5. Write the audit report into this plan doc's Phase 0 results section.

Output: a list of `(section_name, current_placement, recommended_placement,
reasoning)` rows. Each row is independently actionable in Phase 1.

Wall time: ~2-3 hours of code reading + one cradle smoke-run with print
debugging on PromptBuilder.assemble. Zero LLM cost (read-only audit).

**Acceptance:** every dynamic field in the system prompt is named, classified,
and has a placement recommendation. No silent invalidators remain undocumented.

### Phase 1 — Fix invalidators + enable cache + add instrumentation

Goal: make `prompt_cache.enabled: true` actually deliver cache hits in
production.

Substeps:

**1a. Move dynamic content out of system prompt (per Phase 0 findings).**

For each "dynamic-must-move-out-of-system" item:
- Move from `build_*_section` into either the user turn or the assistant
  context. Most natural placement: append to the last user message as a
  `<turn_context>` block.
- Update the test that pins section ordering if it exists.

For each "cacheable-with-care" (session-level dynamic) item:
- Leave in system prompt. These are session-stable (e.g., the agent's name,
  the goal string set at session start) so they're cacheable across turns
  within a session and only invalidate at session boundaries — which we don't
  cache across anyway.

**1b. Enable `prompt_cache: true` on cloud profiles.**

Edit [`profiles.yml`](../../src/maxim/_data/profiles/) for all cloud profiles
in `CLOUD_MODEL_PREFIXES`. Add `prompt_cache: true` (or `prompt_cache:
{enabled: true}`) per profile:

```yaml
claude-sonnet-4-6:
  provider: anthropic
  model: claude-sonnet-4-6
  prompt_cache: true       # ← new
claude-haiku-4-5:
  provider: anthropic
  model: claude-haiku-4-5
  prompt_cache: true       # ← new
```

Local-model profiles stay unset (llama-cpp doesn't have an equivalent;
self-hosted peer routes go through `_MaximPeerBackend` which has no system-
prompt caching path).

**1c. Add cache-hit instrumentation.**

In `_AnthropicBackend.complete_with_usage`, after parsing `usage`, emit a
structured log event:

```python
logger.info(
    "anthropic_cache",
    extra={
        "event": "anthropic_cache",
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_creation,
        "input_tokens_uncached": input_tokens,
        "cache_hit_ratio": cache_read / max(1, cache_read + cache_creation + input_tokens),
        "request_id": getattr(usage, "_request_id", None),
    }
)
```

Pair with `MAXIM_LOG_FILE=/tmp/maxim.jsonl` for downstream analysis. The first
turn of a sub-sim should show `cache_write_tokens > 0, cache_read_tokens =
0`. Turns 2-12 should show `cache_read_tokens > 0, cache_write_tokens = 0`
(or rare re-writes on TTL expiry).

**1d. Verify on a single Sonnet smoke trial.**

Single Arm A sub-sim with the new wiring. Read the JSONL log, sum the cache
metrics:
- Turn 1: cache_creation should be ~100K (the system prompt size)
- Turns 2-12: cache_read should be ~100K each, cache_creation should be 0
- Total ITPM consumption across the 12-turn sub-sim should drop from ~120K
  to ~20K-30K (the user-turn deltas only)

If cache_read stays at 0 across turns 2-12, that's a silent invalidator we
missed in Phase 0 → bisect and re-fix before going further.

**Acceptance:** Sonnet smoke trial shows >80% cache hit ratio on turns 2-12;
total ITPM consumption per sub-sim drops by >5×; rate-limit cascade does not
occur on a single-trial fire.

### Phase 2 — Extend to OpenAI backend (parallel ship)

Goal: same wiring for `_OpenAIBackend` so GPT-4o (and any other OpenAI-
compatible cloud provider) gets the same benefit.

OpenAI's caching is somewhat different mechanically — their automatic caching
kicks in at prompts >1024 tokens without explicit markers. But explicit
`cache_control` markers (using their compatible API shape) still help on
deterministically-routed providers and don't hurt on automatic-caching
providers.

Substeps:
1. Add `_build_system_blocks` parallel to Anthropic's (same shape:
   `cache_control: {"type": "ephemeral"}` at end of system block).
2. Add `_prompt_cache_enabled()` reading from the same config field.
3. Wire `prompt_cache: true` into GPT-4o profile in profiles.yml.
4. Add cache instrumentation matching Anthropic's.
5. Smoke trial: single Arm A run with `--model gpt-4o`.

**Acceptance:** GPT-4o smoke trial shows comparable cache hit ratio + ITPM
reduction.

### Phase 3 — CLAUDE.md invariant + ship

Goal: make the prefix-invariant rule load-bearing so future PromptBuilder
changes can't silently break caching.

Add to CLAUDE.md "Architectural invariants" section:

> **[engineering] System prompt content MUST be byte-stable across turns
> within a session** (prompt-caching prerequisite). The cradle and other
> long-context scenarios route through `_AnthropicBackend` /
> `_OpenAIBackend` with `prompt_cache: true`; any byte change in the system
> prompt between turns invalidates the cache and re-spends ITPM on the full
> prefix. Dynamic content (drive states, bio-substrate annotations, tool
> registries that change per-tick, timestamps, request IDs) belongs in the
> user/assistant message stream, NOT the system prompt. Test for invalidators
> by hashing the system prompt across consecutive turns of a single sub-sim
> and confirming bit-identity. Regression guard:
> `tests/integration/test_prompt_caching.py::test_system_prompt_byte_stable_across_turns`
> (forthcoming in Phase 1).

Add regression test as named. Test fixture: spawn a cradle sub-sim with mock
LLM that records every system prompt it receives, assert all received system
prompts are byte-identical to each other.

**Acceptance:** CLAUDE.md updated; regression test green; Phase 1 + Phase 2
PRs link to this plan.

## Tradeoffs and honest concerns

**1. Phase 0 audit might reveal a lot of in-flight invalidators.** Wire-A
annotations, Wire-1 variance, Acting Coach drive modulation — all of these
were designed as substrate-derived prompt augmentations and might be in the
system prompt by current architecture. If they are, moving them out is a
deeper refactor than "flip one flag." The plan accounts for this — Phase 1a
is explicitly the "move dynamic stuff out" step — but if half the system
prompt turns out to be per-turn dynamic, Phase 1a becomes the big work item
rather than a footnote.

**2. Cache writes ARE billable.** The 1.25× write premium means a one-turn
sub-sim with no cache reuse costs MORE than no caching at all. For sub-sims
that complete in 1-2 turns (or get killed early), caching is a slight cost
loss. For sub-sims that go to 12 turns (cradle target), it's a 6× win. Most
cradle sub-sims complete the full 12 turns, so net-positive on average.
Quantify in Phase 1d.

**3. TTL is 5 minutes by default.** If sub-sims are slow (e.g., Sonnet under
load with high TTFT), the cache might expire mid-sub-sim, forcing a re-write.
This shouldn't happen at Sonnet's normal latency (~5s per turn × 12 turns = 60s
total, well under the 5-min TTL) but is worth watching in Phase 1d. Switching
to `ttl: "1h"` (at 2× write premium) is a backup if 5-min TTL bites.

**4. Provider-specific shape differences.** Anthropic uses `cache_control`
markers on individual blocks. OpenAI uses automatic caching above a threshold.
Future cloud providers (DeepSeek, Mistral, etc.) may have yet another shape.
Phase 2 ships the OpenAI-specific wiring; subsequent providers are added
incrementally per the established pattern (`_build_system_blocks` +
`_prompt_cache_enabled` in each backend, profile flag in profiles.yml).

**5. Local models don't benefit.** Qwen14B/32B/Mistral24B all route through
llama-cpp-server which has no prompt-cache mechanism (though KV-cache on the
local model partially compensates). This means the Exp 37 cross-model
comparison still has an apples-to-oranges shape: cloud sub-sims have prompt
caching savings, local sub-sims don't. The headline LLM-prior-dominance
finding is unaffected (we measure behavior, not cost), but cost comparisons
will lopside-favor cloud.

**6. Phase 0 first is non-negotiable.** Skipping the audit and flipping the
flag would silently fail (cache miss every request, no measurable benefit,
hours of debugging). The audit is cheap relative to the alternative.

## Sequencing relative to Exp 37 cross-model fires

Currently running on the leader: Qwen32B fire (~26 records of 60 in, ~18 more
hours). Independent of this plan.

Currently waiting on the peer: Sonnet replication (killed 2026-06-08), GPT-4o
(queued), DeepSeek-V3 (queued).

Two options for sequencing:

**Option A — Pause cloud replications, ship Phases 0-3, then resume.**
Pause: Sonnet, GPT-4o, DeepSeek (no work loss; nothing's started). Ship
prompt caching. Resume cloud replications with caching active. Wall time: +
~1-2 days (audit + implementation + smoke). Closes the rate-limit blocker
structurally before re-firing.

**Option B — Pivot to DeepSeek + GPT-4o now, ship prompt caching after.**
DeepSeek has dramatically higher tier-1 limits and is cheap. GPT-4o tier 1
might have higher ITPM than Anthropic's 30K. Fire them now without caching.
Add Sonnet later once prompt caching is shipped. Wall time: 0 delay. Less
methodologically clean (Anthropic comparison gets the caching speedup that
other models don't).

**Recommendation: Option A.** The prompt caching work is architecturally
valuable independent of Exp 37 (every future cloud experiment benefits). The
1-2 day delay is small relative to the ~30 hours we're already waiting on
Qwen32B. Doing the audit + ship before resuming cloud experiments means we
get a clean Anthropic comparison AND we ship a real engineering improvement.

## Out of scope (explicitly)

- Caching for self-hosted peer routes (`_MaximPeerBackend`). Self-hosted
  llama-cpp via tunnel has KV-cache locally; adding HTTP-layer prompt
  caching adds complexity without clear benefit since the peer-to-leader
  link is already the bottleneck, not LLM compute.
- Substrate / bio-system architecture changes. This plan moves dynamic stuff
  OUT of the system prompt INTO the message stream; it does not change WHAT
  goes into prompts.
- Replacing PromptBuilder. The audit may surface architectural cleanups
  (e.g., a clearer split between "stable session context" and "per-tick
  state"), but those are follow-up work, not blocking 1.0.
- Provider-specific cost-optimization beyond caching (batch API, lower-cost
  routing, etc.). Separate plan.

## Cross-references

- 2026-06-08 Sonnet/Haiku rate-limit cascade evidence in this conversation's
  results.jsonl for the cradle sub-sims that failed.
- [docs/plans/exp37_cross_model_characterization.md](exp37_cross_model_characterization.md)
  — the cross-model study this plan unblocks for Anthropic comparison.
- [docs/plans/cloud_dispatch_debug.md](cloud_dispatch_debug.md) — the prior
  debug plan whose Phase 1 validated the dispatch path (2026-06-07).
- [Anthropic rate limits](https://platform.claude.com/docs/en/api/rate-limits)
  — authoritative tier table + cache-aware ITPM rule.
- [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
  — mechanics, breakpoint placement, silent invalidator audit checklist.
- [src/maxim/models/language/anthropic_backend.py](../../src/maxim/models/language/anthropic_backend.py)
  — existing cache_control wiring (lines 118-176) that this plan enables in
  production.
- [src/maxim/agents/prompt_builder.py](../../src/maxim/agents/prompt_builder.py)
  — the PromptBuilder that Phase 0 audits.
- [src/maxim/models/language/openai_backend.py](../../src/maxim/models/language/openai_backend.py)
  — the OpenAI backend that Phase 2 extends.

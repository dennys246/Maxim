# Cloud-Dispatch Path Debug + Cleanup

**Status:** DRAFT 2026-06-05 (start during Exp 37 Qwen14B fire's 33-hour
background window)
**Author:** Denny + Claude
**Blocks:** firing Sonnet as the pre-reg's specified PRIMARY model
(currently Qwen14B is firing as tertiary replication — the pre-reg's
original primary is unresolved)
**Out of scope:** Exp 37's metric / pre-reg structure (settled in PRs
#332/#334/#336); harness-on-leader structural fix (parallel work track,
the singleton-check + preflight guard)

## Why this exists

PRs #337 (cloud dispatch wiring) and #338 (MAXIM_AUTO_DOWNLOAD_MODELS=1
in cloud branch) shipped during the 2026-06-05 Sonnet fire attempts.
Both attempts produced degenerate JSONL records (cost=$0, tokens=0,
engagement=0, every action a `_llm_unavailable` fallback) before being
killed. We **do not actually know** whether the cloud-dispatch path is
broken or just slow on first run — both attempts were killed BEFORE
the sub-sim's smollm download completed. The 33-hour Exp 37 Qwen14B
fire window is the natural time to figure this out without blocking
the 1.0 ship.

## Empirical evidence so far

**Attempt #1 (PR #337 only, 2026-06-05 ~10:55):**

- Sub-sim launched with `--language-model claude-sonnet` ✓
- `lane_decisions.jsonl` showed `tier_decisions.large.profile:
  claude-sonnet-4-6` with `remote_host: ""` ✓ (cloud routing resolved)
- Sim log smoking gun:

      Maxim wants to download 'smollm-1.7b-instruct' (~1.1 GB) from HuggingFace.
      Proceed? [y/N] (30s timeout)  [timeout]
      Auto-download cancelled (timeout).

- Bootstrap failed, every LLM call (LARGE-tier Sonnet AND narrator)
  fell back to `_llm_unavailable`.
- Killed after 7 min.

**Attempt #2 (PRs #337 + #338, 2026-06-05 ~11:52):**

- Sub-sim launched with the same args, plus `MAXIM_AUTO_DOWNLOAD_MODELS=1`
- Sim log showed the download actually starting:

      Downloading LLM: smollm-1.7b-instruct
      Size: ~1.1 GB
      [----] 0% (1.2/1006.7 MB)

- Killed after several minutes when 5 trials had already emitted with
  cost=$0 engagement=0 (the wait-for-download was racing the harness's
  per-trial timeout AND the harness was completing trials with the
  fallback responses before the download finished).
- We never observed a trial AFTER the download completed.

**What we DON'T know:**

1. Does the cloud-dispatch path actually work if smollm finishes
   downloading? (~10MB/sec at the rate we observed, ~100 seconds total
   if uninterrupted; longer if network is slow.)
2. Why does cloud-LARGE need local SMALL at all? Is it a hard dependency
   in the lane bootstrap, or just a tier-table default that's not
   actually consumed when LARGE is cloud?
3. Are there other cascade points in the lane bootstrap that would
   fail in cloud mode after SMALL is resolved? (Medium tier? Sentence-
   transformers encoder? Something else?)
4. Is the harness's per-trial timeout (1800s / 30min default) hitting
   the download window? If so, trial 1's "completion" might be
   premature, recording fallback data while the download is still in
   flight.

## Investigation plan

### Phase 1: Empirical — let it run

**Goal:** observe what happens when smollm download actually finishes.

**Steps:**

1. From this peer, fire a SINGLE Arm A fire_pit trial with cloud
   dispatch and patience (no `--cleanup-after-trial` so we can inspect
   the data_home post-run):

   ```bash
   PYTHONPATH=src python scripts/benchmark_cross_session.py \
       --scenario fire_pit \
       --arms A \
       --trials 1 \
       --seed-base 500 \
       --model claude-sonnet \
       --sim-max-turns 8 \
       --cost-cap 2 \
       --out /tmp/cloud_debug/results.jsonl \
       --workdir /tmp/cloud_debug/workdir
   ```

2. Watch the harness log in real time. Three possible outcomes:

   - **(A) smollm completes, sub-sim proceeds, real LLM calls happen,
     non-zero cost.** Cloud path works. The "fix" is documentation:
     warn users that first cloud run takes ~5-10 min for smollm + first
     model load. Subsequent runs reuse the cache.
   - **(B) smollm completes, sub-sim proceeds, but LARGE-tier calls
     still fail.** Real bug. Capture the failure mode (auth error?
     missing env var? wrong endpoint?), root-cause it.
   - **(C) smollm download stalls or fails.** Network / HuggingFace
     issue independent of cloud dispatch. Document the failure path and
     consider pre-staging the model in `~/.maxim/models/` before runs.

3. Inspect `data_home/trial001_fire_pit_A/sim_reports/<session_id>/actions.jsonl`
   for the actual tool calls. If they're real (e.g.
   `infant_humanoid_look`, `sense_cool_air`, `fire_pit_warm_self`),
   path works. If they're all `_llm_unavailable`, path is broken.

4. Inspect `cost_usd`, `total_input_tokens`, `total_output_tokens` in
   the JSONL record. Non-zero → real Anthropic calls happened.

**Expected wall:** 15-30 min (one trial + buffer for smollm download).

### Phase 2: Root cause analysis (only if Phase 1 produces outcome B)

If Phase 1 confirms cloud-LARGE calls fail even after smollm is
available, drill into the failure mode:

1. **Where does the failure surface?** LLMWorker logs? Router?
   `_AnthropicBackend`? Capture the actual exception path.
2. **Is the API key reaching the backend?** Add `MAXIM_BACKEND_TRACE=1`
   (per CLAUDE.md env table) and `MAXIM_LOG_FILE=/tmp/cloud_debug/maxim.jsonl`
   to see per-backend-call structured logs.
3. **Does a direct call to `_AnthropicBackend` outside the sub-sim
   work?** Write a tiny Python script that imports
   `maxim.models.language.config` + the anthropic backend and makes a
   single test call. Bisects "sub-sim setup wrong" from "anthropic
   backend wrong."
4. **Is `MAXIM_LLM_CLOUD_ENABLED=1` actually being respected?** Grep
   the runtime for "MAXIM_LLM_CLOUD_ENABLED" and trace what gates on
   it. Possible the env var is set but a downstream check still rejects
   cloud dispatch.

### Phase 3: Architectural decision (depends on Phase 2 findings)

Three possible directions, in order of escalating scope:

**Option A — Documentation only.** If Phase 1 outcome A: the cloud path
works, just slow on first run. Update `cloud_dispatch_debug.md`'s
Phase 4 to note "first run takes ~5-10 min for smollm cache; subsequent
runs are fast." No code changes. Cheap.

**Option B — Optional SMALL tier in cloud mode.** If the lane bootstrap
actually consumes SMALL only for specific functions (e.g., enrichment,
sense_tools), gate those features behind an optional check. When cloud
LARGE is configured and SMALL is unavailable, run with the SMALL-tier
features disabled (or routed to LARGE as fallback). Medium scope — ~1-2
days of work touching `runtime/function_router.py` and
`runtime/lane_backends.py`.

**Option C — Cloud SMALL profile.** Add a cheap-cloud-SMALL profile
(e.g., `claude-haiku-4-5`, `gemini-flash`, `groq-llama-3.1-8b`) and
have cloud-LARGE mode automatically pair it with a cloud SMALL.
Cleanest user experience (no local model dependencies in cloud mode
AT ALL) but biggest architectural change. ~2-3 days.

Recommended default: **Option A first** (since Phase 1 may resolve it
without code changes). Escalate to B or C only if there's a real
functional dependency that doesn't tolerate missing SMALL.

### Phase 4: Validation

Once the path is functional (whatever the root-cause fix turned out to
be), validate with a 3-trial Arm A smoke from the peer:

```bash
PYTHONPATH=src python scripts/benchmark_cross_session.py \
    --scenario fire_pit --arms A --trials 3 --seed-base 600 \
    --model claude-sonnet --sim-max-turns 8 --cost-cap 5 \
    --out /tmp/cloud_validate/results.jsonl \
    --workdir /tmp/cloud_validate/workdir --cleanup-after-trial
```

Confirm across all 3 trials:

- `cost_usd > 0` (real API calls)
- `total_input_tokens > 0` and `total_output_tokens > 0`
- `fire_pit_engagement_count` distribution is non-degenerate (NOT all
  0; NOT all the same value)
- Trial wall time is consistent with Anthropic API latency (~2-3 min
  per trial, NOT the 30-minute Qwen latency)

If all 3 trials pass these checks, cloud-dispatch is solid.

### Phase 5: Pre-reg path forward

**After Exp 37 Qwen14B fire produces results** (the 33-hour fire that
this background work runs alongside):

1. Run the analyzer on Qwen14B JSONL → results doc → row 1 status flip
   in `behavioral_graduation_candidates.md`. This is the LOCKED 1.0
   evidence regardless of what happens with cloud-dispatch.
2. **If cloud-dispatch is now validated:** fire Exp 37 a SECOND time
   with `--model claude-sonnet` as the pre-reg's specified PRIMARY
   replication. ~$14 cost, ~3 hours wall. Compare results across both
   models — if substrate-transfer holds in both, it's strong
   cross-model evidence. If it holds in only one, that's also
   informative (model-specific phenomenon vs. universal).
3. **If cloud-dispatch is NOT validated:** pre-reg amendment 2026-06-XX
   formalizing "Qwen14B is the operational primary; Sonnet replication
   deferred to 1.1+ as substrate-primary work matures." Honest scope
   boundary, no dishonest claim.

## Risks

1. **Phase 1 might not resolve cleanly.** If smollm download keeps
   failing (network, HuggingFace rate limit, disk), Phase 1 produces
   outcome C and we're stuck on infrastructure work that isn't really
   about cloud-dispatch. Mitigation: pre-stage smollm into
   `~/.maxim/models/` before starting Phase 1. The harness's
   per-data_home symlink to `~/.maxim/models/` should reuse it.

2. **Architecture decision could explode scope.** Option C (cloud SMALL
   profile) is principled but it's a real architectural change that
   touches the tier-table semantics. If Phase 2 suggests Option C is
   the right answer, deciding whether to do it during the 33-hour window
   or defer to 1.1 is a real choice. Recommendation: **defer Option C
   to 1.1**. Option A or B is fine for 1.0 Sonnet replication.

3. **The 33-hour Qwen fire might not stay up the whole time.** Mac
   Mini sleep is configured off via `pmset -c sleep 0`, but cloudflared
   has flaked before. If the leader goes down mid-fire, the cloud-
   dispatch debug work still has value (this peer can keep working).
   But we'd need to recover the leader and restart Exp 37 from the
   checkpoint (which is what the append-only JSONL idempotency in
   the harness supports — re-running with the same out path picks up
   where it left off).

4. **Pre-reg credibility.** This would be the FOURTH amendment to
   Exp 37 (after 2026-05-31, 2026-06-XX pivot, 2026-06-05 SD-shift).
   Even the honest "we ran both Qwen and Sonnet and report both" framing
   has the "why didn't you anticipate this" credibility cost.
   Mitigation: be explicit in the results doc that the multi-amendment
   sequence reflects iterative empirical learning, not goalpost-moving.
   Each amendment was anchored to specific measured failure modes; this
   is documented across the three pre-reg amendment plan docs.

## Out of scope (explicitly)

- **The harness-on-leader structural fix.** Parallel work track, the
  singleton-check + preflight guard. The two debugs share architecture
  in the lane-config region but solve different failure modes; they
  should land as separate PRs.
- **Adding more cloud providers beyond the current 7-prefix list.** The
  CLOUD_MODEL_PREFIXES list in PR #337 covers claude, gpt, gemini,
  groq, together, fireworks, deepseek. If Sonnet path works, no need
  to expand. Adding Mistral (the ambiguous-prefix case) is a 1.1+
  refactor.
- **Substrate-primary measurement (Exp 38).** That's the principled
  long-term answer to the LLM-AUT noise floor, not this debug.

## Sequencing

1. Land the harness-on-leader structural fix in the parallel Mac Mini
   Claude session (singleton-check + preflight guard).
2. Fire Exp 37 from the leader using local Qwen14B (~33 hours wall).
3. **During the 33-hour wait**, work this plan on the peer:
   - Phase 1 empirical (~30 min)
   - Phase 2 root cause if needed (~half day)
   - Phase 3 fix per chosen Option (~hours to ~2 days)
   - Phase 4 validation (~15 min smoke)
4. Qwen14B Exp 37 analyzer run → results doc → row 1 flip.
5. (Optional, if Phase 4 passes) Sonnet replication fire (~3 hours,
   ~$14).
6. Combined results doc + row 1 status reflects both models if both
   ran.

## Cross-references

- [exp37_metric_pivot.md](exp37_metric_pivot.md) — the 2026-06-XX pivot
  that established `positive_approach_engagement_fraction` as primary.
- [exp37_sd_shift.md](exp37_sd_shift.md) — the 2026-06-05 SD-shift
  swap. This debug plan stacks on top of both.
- [cradle_activation_fixes.md](cradle_activation_fixes.md) — the
  upstream calibration work (PRs C/D/E).
- CLAUDE.md "Environment Variables" section — canonical reference for
  MAXIM_LLM_CLOUD_ENABLED, MAXIM_MAX_CLOUD_LANES,
  MAXIM_AUTO_DOWNLOAD_MODELS, MAXIM_BACKEND_TRACE, MAXIM_LOG_FILE.
- CLAUDE.md "Lessons learned" entry on running the benchmark harness
  on the leader — the sibling problem the structural-fix track is
  addressing.
- PRs #337 (cloud-dispatch initial) and #338 (auto-download) — the
  prior cloud-dispatch work this plan extends.

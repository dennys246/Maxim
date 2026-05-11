# Roy-0: Smoke — Reproduction Protocol

**Status:** Drafted from a partial Roy-0 run (priming stage 1 aborted at ~145s wall clock). Hand-edited after auto-generation; do not regenerate without `--keep-edits`.
**Purpose:** Validate that the Roy harness (R1 curriculum runner + R2 substrate diff + R3 three-arm runner + R4 iteration-log generator) boots correctly end-to-end on a tiny methodology spec. Roy-0 is a test of the HARNESS, not a persona research result.
**Expected runtime:** ~10 minutes on a dev laptop *when LLM dispatch works* — neither cloud nor local LLM worked on the box this ran on, so the documented run hit ~145s of priming before being killed.

## Background

Roy-0 reuses the cradle_prelinguistic arc for both priming (5 stages × 10 turns = 50 turns total) and the held-out test (10 turns). The deliberate overlap means arm A's substrate should carry forward into the test scenario while arms B and C see the test fixture cold. Arm B's persona prompt is deliberately vague ("You are a hungry infant"); the methodology calls for carefully-shaped prompts later (Roy-1+), but Roy-0 only exercises the prompt-injection PATH, not its content.

This is the smallest configuration that makes every Roy code path fire at least once:
- R1 chains 5 priming stages → resume_session handoff is exercised 4 times
- R3 dispatches arms A/B/C with `substrate=from_priming` vs `blank`
- R2 computes pairwise diff over 3 pairs
- R4 generates this protocol + the iteration-log entry in persona_convergence_crucible.md

## Prerequisites

- Maxim checkout at or after the R5 commit (`feat(roy): Roy-0 smoke validation + harness fixes`).
- `pymaxim` importable (`PYTHONPATH=src` or editable install).
- A writable `~/.maxim/` (or `MAXIM_DATA_HOME=<path>` override).
- **A working LLM provider.** Either:
  - An exported cloud API key (`ANTHROPIC_API_KEY` is the cheapest path for narration), OR
  - A working local llama.cpp lane (`maxim doctor` should report `large` and `small` tiers healthy; on Mac the M3/M4 GPU works, but the dev box this ran on hit `llama_decode returned -3` on every call — see G3 below).
- The iteration spec YAML still resolvable from the recorded path:

      docs/plans/roy/roy_0_smoke.yaml

## Running the iteration

```bash
maxim roy run docs/plans/roy/roy_0_smoke.yaml --dry-run     # parse-only sanity
MAXIM_LOG_FILE=/tmp/roy_0_smoke.jsonl maxim roy run docs/plans/roy/roy_0_smoke.yaml
```

The JSONL is what makes failures debuggable. Without it, `dispatch_exhausted` warnings hit stderr only and the orchestrator's narrator silently falls back to "The story continues to unfold around you..." with no useful signal about WHY.

## What we hit on the recorded run

The dev box this ran on had no cloud API key and a broken local 14B llama.cpp lane. The run got through priming setup but couldn't make per-turn progress. Quantitatively, from the JSONL after ~145s:

- 3× `dispatch_exhausted` (provider=local, error=RuntimeError) → narrator fallback fired 3× ("The story continues to unfold around you...", `fallback: true`)
- 8× heartbeat events (`Loop step N, proposal=none` at N ∈ {0, 20, 40, 60, 80, 100, 120, 140})
- 0 substrate snapshots written (priming stage didn't reach session-end)
- 0 arm sessions started (priming aborted before arm dispatch)

**The harness behaved correctly** — it didn't crash, the curriculum chain stayed coherent, the artifact dir was created, and result.json was persistable. What *didn't* work was orthogonal to harness code: the local LLM driver and the substrate-primary action proposer.

## What worked first try

- Spec validation (`--dry-run` parsed the 5-stage inline curriculum + 3 arms cleanly).
- Plan auto-detection (`roy-0-*` prefix routed to `persona_convergence_crucible.md`).
- Inline-priming materialization (R3) wrote the sibling `.roy_inline_priming_roy-0-smoke.yaml` and cleaned it up after the curriculum exited.
- Bridge narration suppression in substrate-primary mode (no English sentinels reached the AUT percept queue).
- `~/.maxim/roy/roy-0-smoke/` artifact dir was created on first invocation.
- Protocol + iteration-log generators (R4) ran against a hand-crafted result.json without complaint.

## What needed fixing (in this commit)

- **G1: Roy runner now forces interactive=off process-globally.** The orchestrator's AUTO interactive-mode resolution enables Rich Live + raw-terminal stdin reader on TTY launch. Five sims back-to-back from a YAML spec is never something a human is steering — interactive mode just adds Rich panel contention and stdin races to a non-interactive run. Fixed in [src/maxim/simulation/roy_runner.py](../../../src/maxim/simulation/roy_runner.py) `run_roy_iteration`. Regression guard: `TestRoyRunnerInvariants::test_run_forces_interactive_off` in [tests/integration/test_roy_runner.py](../../../tests/integration/test_roy_runner.py).

## Harness gaps still open (close before Roy-1)

- **G2: Bridge spinner writes ANSI to stderr unconditionally.** `simulation/spinner.py` doesn't check `interactive_mode` or `stderr.isatty()`. Symptom: every turn produces ~30s of `⠋ Turn N: Waiting for AUT response... (Xs)` in the captured stderr, polluting log files. Cheap fix: gate `Spinner.start/update` on `get_interactive_mode() == ON` OR `sys.stderr.isatty()`.
- **G3: No pre-flight LLM probe before priming spends compute.** A Roy iteration commits to ~10 minutes of orchestration on the assumption that the LLM lane is alive. When every dispatch fails (broken local model, no cloud key, leader 502, ...), the runner should fail fast with a useful error instead of grinding through static-fallback narration for 25 minutes. Cheap fix: in `run_roy_iteration` BEFORE priming, fire one `LLMRouter.health_check` or equivalent and abort with `aborted_at="preflight"` on failure.
- **G4: Substrate-primary AUT loop stays at `proposal=none` for the whole run.** Heartbeats logged 8× `Loop step N, proposal=none`. Either (a) the substrate-primary proposer needs percept context to produce proposals and the cradle_prelinguistic narration silence is starving it, (b) the infant body's drive state isn't activating any affordances at this point in the arc, or (c) there's a real wire missing between `EmbodimentPerceptSource.next_percept` and `runtime/agent_loop.py:2654` (`aut_mode == "substrate-primary"` branch). Needs investigation in the substrate-primary track ([grounded_language_acquisition.md](../../plans/grounded_language_acquisition.md) Phase 0), not in Roy harness scope.
- **G5: Auto-spawn path mismatch.** `Auto-spawn skipped: GGUF file not found at /Users/.../models/LLM/claude-sonnet-4-6.Q4_K_M.gguf (profile=qwen2.5-14b-instruct)` — the file path doesn't match the profile. Looks like env-var leak from a previous `--llm claude-sonnet` invocation. Pre-existing bug in the auto-spawn module, not Roy.
- **G6: Small-lane auto-download blocked by non-TTY.** `Profile 'smollm-1.7b-instruct' is not downloaded and stdin is not a tty` — the prompt requires interaction. Either (a) Roy runs should set `MAXIM_AUTO_DOWNLOAD_MODELS=1` automatically, or (b) document in the prerequisites here that small-lane models must be pre-downloaded.

## What changed vs prior iterations

First Roy iteration — no prior. The auto-generated stub framing for this section ("operator: fill in before commit") becomes useful starting at Roy-1.

## Related docs

- [`persona_convergence_crucible.md`](../../plans/persona_convergence_crucible.md) — three-arm methodology + Roy-1 design
- [`grounded_language_acquisition.md`](../../plans/grounded_language_acquisition.md) — Roy long-horizon harness context + substrate-primary AUT mode
- `maxim.analysis.substrate_diff` — diff library (R2)
- `maxim.simulation.roy_runner` — three-arm iteration runner (R3)
- `maxim.analysis.roy_log` — protocol + iteration-log generator (R4)
- `maxim.simulation.curriculum_runner` — chained-stage substrate priming (R1)

<!-- generated by `maxim roy log roy-0-smoke`, then hand-edited with R5 validation findings — `--keep-edits` recommended on subsequent regenerations -->

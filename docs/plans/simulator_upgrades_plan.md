# Simulator Upgrades Plan — Substrate Test Harness Prep

**Status:** Active, blocks substrate_plan P0
**Target version:** 0.3-pre (lands after [foundations_plan.md](foundations_plan.md), before [substrate_plan.md](substrate_plan.md) P0)
**Scope:** ~800 LOC across four items, ~1.5 weeks of focused work
**Relationship:** Prerequisite for substrate phase work. Shrinks substrate harness cost by ~1,500 LOC across the full plan by reusing existing sim infrastructure instead of building bespoke harnesses per phase.

## Why this plan exists

A targeted audit of [src/maxim/simulation/](../../src/maxim/simulation/) revealed that ~80% of the substrate plan's testing infrastructure already exists in the simulator — `ConversationalSource.inject_cli()` is a first-class percept injection API, `ScenarioSource` runs YAML-driven deterministic scenarios, `BenchmarkRunner` supports baseline-vs-architecture side-by-side comparison, and session reports are extensible with per-phase metric sections. The earlier substrate_plan drafts treated all of this as net-new work to build.

What the sim **can't** do today, and what this plan adds:

1. **Run without a narrator LLM.** `run_generative_campaign()` and `run_dm_campaign()` both mandate a live `llm_router`. Substrate unit sims don't need narration — they need deterministic percept streams from fixtures — but there's no orchestrator mode that skips the narrator. This wastes GPU time, introduces non-determinism, and makes substrate phases expensive to run at high seed counts.
2. **Mock the AUT's own LLM calls.** Even if the narrator is bypassed, the AUT's `LLMWorker` still blocks on model initialization and calls out to a real backend for tool-selection decisions. For pure substrate mechanism tests we want canned or scripted responses.
3. **Run persistence round-trip tests across a real subprocess boundary.** `--resume-sim SESSION_ID` exists but it reuses the running Python interpreter. The substrate plan's persistence contract (see substrate_plan.md) requires a true subprocess boundary — serialize, kill the process, spawn a fresh interpreter, reload, re-probe. This must exist *before* P1 so every phase's round-trip smoke test can use it.
4. **Fully deterministic seeding as a first-class CLI flag.** A hardcoded `np.random.seed(42)` exists in [similarity/semantic.py](../../src/maxim/similarity/semantic.py), but no `--seed` flag sets `PYTHONHASHSEED`, `random.seed`, `np.random.seed`, and torch seeds together. Without this, substrate phases can't guarantee byte-identical outputs across runs.

None of these are architectural blockers. All four are ~1 week of focused work. Building them **before** P0 means every substrate phase gets cheap, fast, deterministic test runs for free — and the substrate plan's harness LOC drops by ~1,500.

## What we are *not* doing

- **Not adding a general-purpose `--mock-llm` production mode.** The mock lives in `tests/substrate/` for now. See "Design note" below for the deliberate trade-off and the promotion path if this changes post-1.0.
- **Not rewriting the narrator to be fixture-driven in generative campaigns.** We're adding a *parallel* fixture-driven orchestrator that sits beside `generative_runner.py`, not replacing it. Existing sim use cases keep working.
- **Not using generative sim output as ground truth for substrate fixtures.** Audit confirmed that `GenerativeCampaignResult` turns are too narrative-freeform to be labeled as paraphrase clusters or episode co-occurrence without an NLP post-processor — which would add a pipeline dependency the substrate plan specifically wants to avoid. Substrate fixtures are hand-authored. The sim's role is **fixture *debugging*, not fixture *generation*** — see substrate_plan for the "sim as fixture debugger" workflow.
- **Not touching `default_network`, `decisions`, `memory`, or `similarity` modules.** The sim upgrades are orthogonal to the substrate layers. Zero architectural coupling.

## Design note — Option B for the mock LLM

Two ways to write the mock LLM were considered:

- **Option A — narrow test helper.** Bespoke `ScriptedScenario → ScriptedResponse` interface, bypasses the router, lives in `tests/`. ~100 LOC, fastest to write, but promotion to a general-purpose CLI mode later requires rewriting against the real backend interface. Classic "test-only code that becomes a rewrite trap."
- **Option B — real backend interface, test-only wiring.** Implements whatever `models/language/router.py` uses as its backend protocol (`LLMBackend` or equivalent ABC). Returns canned or policy-driven responses from `generate()`. Lives in `tests/substrate/mock_llm.py` initially but implements the same contract as `anthropic`, `llama-cpp-server`, `openai`, etc. ~150 LOC today. **Promotion to a general-purpose `--llm mock` mode later is ~half a day of work:** move the file to `models/language/backends/mock.py`, register with the backend router, add one CLI option.

**This plan uses Option B.** The ~50 LOC cost difference today is worth three things: (a) `mypy` catches interface drift if the backend contract evolves during substrate work, (b) the mock can be used to test the router itself without a real LLM, and (c) no future rewrite tax — the promotion path is move + register.

## Items

### S1 — `FixtureDrivenOrchestrator`

**Gap:** `campaign_runner.py` only dispatches to `run_generative_campaign()` and `run_dm_campaign()`, both of which require a live `llm_router` for narration. There's no path for "read a YAML fixture and run it through the agent loop."

**Minimum implementation:**
- New class `FixtureDrivenOrchestrator` in `simulation/fixture_orchestrator.py`, parallel to the generative/DM runners
- Reads fixture YAML specifying `percepts: [{at_tick: N, content: "...", modality: "text", salience: ..., channel: ..., sender: ...}]` — uses the same schema `ScenarioSource` already parses
- Drives the agent loop via `ConversationalSource.inject_cli()` (for text) and a new `inject_vision()` (stub for P4)
- Collects substrate-relevant state snapshots (ATL dump, hippocampus episodes, NAc per-node bias, trace buffer) at end-of-run
- Writes a session report extended with a `substrate_metrics` section (filled in per-phase by metric extractors)
- Integrates with `campaign_runner.py` as a new dispatch path: `maxim --sim fixtures:substrate/P1_paraphrase_collapse`

**Exit:** A minimal P1-style fixture runs through the orchestrator without touching `generative_runner.py` or `dm_runtime.py`, writes a session report, and produces deterministic output across two runs with the same seed.

**Scope:** ~200 LOC + fixture loader + integration test.

**Dependencies:** S2 (mock LLM must exist so the AUT loop has something to talk to). None of the foundation items.

### S2 — Mock LLM backend (Option B, test-only wiring)

**Gap:** Even with the narrator bypassed, the AUT's own `LLMWorker` calls a real backend for tool-selection decisions. For substrate mechanism tests we want canned or scripted responses.

**Design correction from the prior draft.** The prior draft assumed an `LLMBackend` Protocol already exists in [models/language/router.py](../../src/maxim/models/language/router.py) that the mock could implement directly. The iceberg sweep found this is **wrong**: backends are duck-typed concrete classes (`_AnthropicBackend`, `_LlamaCppBackend`, `_PyTorchTransformersBackend`, `_OpenAIBackend`) with no formal Protocol or ABC. Router dispatches call `backend.complete(...)` at [router.py:731](../../src/maxim/models/language/router.py#L731) and `backend.complete_with_usage(...)` at [router.py:758](../../src/maxim/models/language/router.py#L758), plus attribute probes (`requires_prompt_formatting`, `supports_model_override`, `supports_tool_use`, `supports_streaming`) scattered across `router.py:259–756`. There is nothing for the mock to implement *against*, and the "mypy catches interface drift" claim is broken — there's no protocol to drift against.

**S2 therefore has two steps, not one:**

**Step 2A — Reverse-engineer and define `LLMBackend` Protocol** (~50 LOC, new work):

- Audit the four existing backends and the router dispatch sites to enumerate every method and attribute the router calls on a backend
- Define `LLMBackend` as a `typing.Protocol` (runtime_checkable if useful) in [models/language/backend_protocol.py](../../src/maxim/models/language/backend_protocol.py)
- **Minimum method set** (confirm during implementation): `complete(prompt, max_tokens, temperature, stop) → str`, `complete_with_usage(system, user, max_tokens, temperature, stop, ...) → LLMResponse`, plus attribute-level contract flags (`requires_prompt_formatting: bool`, `supports_model_override: bool`, `supports_tool_use: bool`, `supports_streaming: bool`)
- Do NOT retroactively make existing backends inherit from the Protocol yet — Protocols are structural, so existing backends satisfy the contract implicitly. Adding an explicit ABC later is a separate cleanup.
- `mypy --strict` on the Protocol module to confirm the signatures are well-formed

**Step 2B — Implement `MockLLMBackend` against the new Protocol** (~150 LOC):

- New `MockLLMBackend` in `tests/substrate/mock_llm.py`
- Implements the `LLMBackend` Protocol from Step 2A
- Supports three response modes, selected per-fixture:
  - **Canned mode:** returns a fixed response string regardless of prompt
  - **Policy mode:** reads a `{prompt_pattern → response}` dict from the fixture and matches on prompt substring
  - **Scripted mode:** reads a list of responses and returns them in order (raises if exhausted)
- Implements all Protocol methods including streaming and tool-call shapes as stubs where needed
- **Not registered with the production router.** The substrate harness instantiates it directly and injects it into the agent loop for testing.
- `mypy --strict tests/substrate/mock_llm.py` passes against the new Protocol — **this is now a real claim because the Protocol exists.**

**Exit:** `LLMBackend` Protocol exists as the single formal contract for backends. `MockLLMBackend` implements it. P0 pilot runs through the `FixtureDrivenOrchestrator` with a `MockLLMBackend` producing deterministic tool calls. `mypy --strict` on the mock module passes against the Protocol. Running `mypy` on the four real backends surfaces any implicit-contract drift they have relative to what the router actually calls.

**Scope:** ~200 LOC total (~50 Protocol + ~150 mock), up from the earlier ~150 LOC estimate.

**Dependencies:** None. The Protocol definition is part of S2, not a prereq.

**Promotion path (post-1.0, optional):** move mock file to `src/maxim/models/language/backends/mock.py`, register with the backend router, add `--llm mock` CLI option. The Protocol stays where it is — it's general-purpose infrastructure regardless of whether the mock is test-only or production. Estimated cost: ~half a day, ~50 LOC of glue. Not in scope for this plan.

**Side benefit:** having a formal `LLMBackend` Protocol lets `mypy` catch drift on the *real* backends too. If `_AnthropicBackend` silently drifts from what the router calls, `mypy --strict` will flag it the next time someone runs it. This is free type safety that the codebase doesn't have today.

### S3 — Substrate persistence subprocess harness

**Gap:** The substrate plan's persistence contract requires a true subprocess boundary for round-trip smoke tests. `--resume-sim SESSION_ID` reuses the running Python interpreter, which doesn't test serialization integrity at all — a closure over module-level state would round-trip fine in-process and explode in a fresh interpreter.

**Minimum implementation:**
- New utility `tests/substrate/persistence_harness.py` exposing `run_round_trip(phase_id, fixture_path, probe_fn)`
- The utility:
  1. Runs a unit sim via `FixtureDrivenOrchestrator` to completion
  2. Serializes full bio-stack state (ATL, Hippocampus, NAc, PerceptTraceBuffer) via `atomic_write_json` to a temp file
  3. Spawns `subprocess.Popen([sys.executable, "-m", "tests.substrate.persistence_child", "--state", temp_file, "--probe", probe_path])`
  4. The child process loads state, re-runs the probe, writes output to a result file, exits
  5. Parent reads result file, compares against pre-shutdown output within tolerance, asserts match
- Pytest integration: a fixture wraps this so each phase's test file just calls `persistence_round_trip(phase=P1, fixture=...)`
- Handles subprocess failures explicitly — a crash in the child is a test failure, not a hang

**Exit:** A P1-style round trip succeeds: ATL nodes serialize, deserialize in a fresh subprocess, held-out probes re-run identically. A deliberately-broken test (e.g., a closure over module state) produces a clear error message in the child subprocess, not a silent pass.

**Scope:** ~300 LOC including the child entry point, result comparison, and pytest glue.

**Dependencies:** S1 (needs the fixture orchestrator), S2 (needs the mock LLM), and substrate_plan's persistence contract serialization code for each layer (ATL, Hippocampus, NAc, PerceptTraceBuffer). The harness itself doesn't need a specific layer; it treats state as opaque and delegates serialization to `atomic_write_json`.

### S4 — Global deterministic seeding CLI

**Gap:** Hardcoded `np.random.seed(42)` in [similarity/semantic.py](../../src/maxim/similarity/semantic.py) sets one module's RNG. Nothing sets `PYTHONHASHSEED`, `random.seed`, `np.random.seed`, and torch seeds together from a single entry point. Substrate phases need this to guarantee byte-identical outputs across runs and across seeds.

**Minimum implementation:**
- Add `--seed` flag to [cli_parser.py](../../src/maxim/cli_parser.py)
- At CLI startup (before any imports that construct RNG state), set:
  - `os.environ["PYTHONHASHSEED"] = str(seed)`
  - `random.seed(seed)`
  - `np.random.seed(seed)`
  - `torch.manual_seed(seed)` if torch is imported
  - `torch.cuda.manual_seed_all(seed)` if CUDA is available
- Remove the hardcoded seed from [similarity/semantic.py](../../src/maxim/similarity/semantic.py) — it becomes derived from the global seed
- `ScenarioSource` accepts an optional `rng: random.Random` parameter for reproducible shuffling of its internal queues
- Documentation: "Byte-identical determinism requires `--seed <N>` and fixture-driven mode (no live LLM in the loop)."

**Exit:** Two runs of the same P0 pilot fixture with the same `--seed` produce identical session reports (tool-call sequences, ATL node IDs, NAc bias values). Changing the seed produces different-but-still-deterministic outputs.

**Scope:** ~150 LOC.

**Dependencies:** None. Can land first if convenient.

## Scope summary

| Item | Scope | Depends on | P0 critical? |
|---|---|---|---|
| S1 — FixtureDrivenOrchestrator | ~200 LOC | S2 | Yes |
| S2 — LLMBackend Protocol + MockLLMBackend (Option B) | ~200 LOC (~50 Protocol + ~150 mock) | — | Yes |
| S3 — Persistence subprocess harness | ~300 LOC | S1, S2 | Yes (before P1 ships) |
| S4 — Deterministic seeding CLI | ~150 LOC | — | Yes |
| **Total** | **~850 LOC + tests** | — | — |

## Order of operations

Four PRs, in dependency order:

1. **S4** first — ~150 LOC, zero dependencies, unblocks deterministic testing for everything else.
2. **S2** next — ~150 LOC, zero dependencies, required by S1 and S3. Ship this as a pure interface exercise — the mock returns canned responses against the real backend protocol.
3. **S1** — ~200 LOC, depends on S2. Ship with a minimum P0-style fixture to prove end-to-end.
4. **S3** — ~300 LOC, depends on S1 and S2. Biggest item; ship with a pytest integration test that round-trips a trivial ATL state.

Each PR: full lint + fast test suite + `mypy` on the public API files per CLAUDE.md. S2 requires `mypy --strict` on the mock module to catch interface drift.

## Exit criteria for the wave

The simulator-upgrades wave is complete when all four items have landed, the full fast test suite passes, `mypy` passes on the public API files and on the mock LLM module, a P0-style fixture runs end-to-end through the fixture orchestrator with deterministic output, and a round-trip smoke test succeeds via the subprocess harness.

Only then does substrate P0 open.

## Non-goals

- **No production-grade mock LLM mode.** Option B preserves the promotion path but the mock stays in `tests/` for now. Revisit post-1.0.
- **No generative fixture generation.** Fixtures are hand-authored. The sim is a fixture *debugger*, not a fixture *generator*.
- **No substrate code in this plan.** Zero changes to `agents/`, `memory/`, `decisions/`, `similarity/`. The upgrades are purely test-harness infrastructure.
- **No `--mock-llm` CLI flag on the production parser.** The mock is instantiated directly by substrate test code. If someone wants a production-exposed mock later, that's a separate promotion PR.

## If any item fails

- **S1 fails** (YAML fixture schema doesn't compose with existing `ConversationalSource`) → the schema is wrong. Likely remedy: narrow the fixture format to exactly what `ScenarioSource` already expects, and add substrate-specific fields as a sub-dict. Don't invent a new percept format.
- **S2 fails** (implementing the real backend interface is more work than estimated because the interface is large) → fall back to Option A temporarily (narrow test helper), accept the future rewrite cost, and log the promotion tax as a known debt in the plan. **Do not skip the mock entirely** — substrate tests will be unrunnable at scale without it.
- **S3 fails** (subprocess harness is hanging or flaky) → the most likely cause is test-state leaking between processes. Invest in cleaner state boundaries before shipping. If truly intractable, fall back to in-process round-trip with a prominent "THIS DOES NOT TEST REAL SERIALIZATION" comment and a TODO tied to a GitHub issue.
- **S4 fails** (something deep in the stack has wall-clock or hash-order dependencies that can't be seeded) → log the offending module as a substrate blocker, narrow `--seed` to what *can* be seeded, and document the remaining non-determinism. P0 pilot can still run, it just won't be bit-identical across machines.

## How this changes the substrate plan

Once S1–S4 land, the substrate plan changes in these specific ways:

- **Each phase's "minimum implementation" line** drops "build unit sim harness" and "build system sim harness" bullets, replacing them with "author fixture YAML + write metric extractor plugin."
- **The scope table** drops ~1,500 LOC of per-phase harness work.
- **Fixture authoring budget** revises from 3–4 days per phase to ~2.5 days per phase, because the "sim as fixture debugger" workflow (rough fixture → replay → inspect → refine → freeze) is faster than hand-authoring in the dark.
- **P0 pilot** becomes a 2-day task instead of a 2–3-day task — most of the cost is fixture authoring, the baseline runner is ~100 LOC against `BenchmarkRunner.baseline_path`.
- **P4 mini pilot** becomes a half-day task instead of a full day — tiny fixture + OpenCLIP baseline module run through `BenchmarkRunner`.
- **Every persistence round-trip in the cross-phase contract** uses S3 as its implementation.
- **Sim time budget:** a full overnight run of all phases × 10 seeds is roughly 6 hours on local RTX 5080 with zero cloud cost. No cloud LLM budget required for substrate validation.

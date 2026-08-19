# Simulation & Experiments — working brief

> Part of the CLAUDE.md satellite layer. Read this whole file before editing `simulation/`, `interactive/`, `scripts/benchmark_*`, `scripts/exp*`, `scripts/orient_*`, `tests/behavioral/`, `docs/experiments/`, or before running any sim. The slim CLAUDE.md core + this brief are intended to be sufficient context for work in this area. Full incident narratives: [docs/lessons/](../lessons/).

## 1. Mental model — what a valid experiment looks like here

Sims are the measurement instrument for the project's central claim (cross-session learning without fine-tuning). An experiment result is only as good as its apparatus, and this repo has retracted results for apparatus failures — so validity is a checklist, not a vibe:

1. **Provenance is asserted, not assumed.** Any harness that spawns `maxim` calls `scripts/_provenance.py::assert_repo_interpreter(repo_root, binary, exempt=<mock>)` before its first sub-sim (exit 3 on mismatch) and SHOULD stamp `executed_code_provenance(...)` — `executed_maxim_file` / `executed_git_hash` / `pythonpath` — into every run record so the artifact is self-auditing. `git_hash` alone answers the wrong question (it describes where the harness *lives*, not what the sub-sims *imported*). This is the core-safety invariant L01 (see CLAUDE.md core); the Exp 42b retraction is the canonical cost. A result whose code-under-test cannot be established is not a validation — do not argue it was "probably fine".
2. **Topology is declared.** The harness runs from a machine that is NOT the leader, or relies on the post-2026-06-05 hardenings (see L02 stub below). One harness at a time; runs declare their owner (apparatus standard S8).
3. **Apparatus is declared.** Any PR touching shared sim machinery, and any experiment design, follows [docs/plans/simulation_apparatus_standards.md](../plans/simulation_apparatus_standards.md) S1–S8: S1 declare which graduated rows ride on the change; S2 graduated rows carry fast apparatus canaries; S3 apparatus pathologies get assertions *inside* the sim; S4 graduated experiments commit their raw records; S5 arms declare their exposure contract; S6 fidelity changes (e.g. `MAXIM_SUBSTRATE_ACTIONS_PER_TURN`) are experiment-visible events that must be declared between arms; S7 gates must be robust to a ceiling in their own baseline; S8 one harness at a time.
4. **Pre-registration before measurement.** Hypotheses, arms, and gates are written to `docs/experiments/protocols/` before the run. Post-hoc findings spawn *new* pre-registered iterations — they do not silently become results (see the convergence-vs-divergence working principle in core).
   **Design against the instrument ledger:** [docs/limits/README.md](../limits/README.md) catalogs the MEASURED limits of every instrument (visibility floor, phase-locking, representational resolution, metric saturation, ceiling voids, bin granularity). A pre-registration that ignores a listed limit for its metric will re-discover it at campaign price — three of the seven entries were paid for twice before the ledger existed. S-standards say how to run; the limits ledger says what your metric can actually see.
5. **Two-process guard where hashing crosses persistence.** Any claim resting on state reloaded across a process boundary needs a two-process test with differing `PYTHONHASHSEED` (the stable-hash invariant — see docs/agents/persistence-config.md); a same-process test passes over that entire bug class.
6. **The graduation ladder.** New mechanisms enter as `[engineering]` only; they graduate to `[behavioral]` when an experiment earns them. Tracking lives in [docs/plans/behavioral_graduation_candidates.md](../plans/behavioral_graduation_candidates.md) — Earned entries carry **Re-run on:** triggers + **Regression guard:** experiment paths; `Stale`/`Broken` entries block the next release. Bio-inspired naming does not count as validation.

Cost discipline rides underneath all of this: sims call a live LLM per turn; a run that burns $0.50+ is a design smell, and a broken-config run silently produces zero actions (see §5).

## 2. Key files

| Area | Key files |
|---|---|
| Simulation | `simulation/orchestrator.py`, `simulation/bridge.py`, `simulation/fixture_orchestrator.py` (S1 fixture-driven), `simulation/sim_types.py` |
| Substrate test infra | `models/language/backend_protocol.py` (S2), `utils/seeding.py` (S4), `tests/substrate/` (S2+S3+P1 metrics) |
| Generative campaigns | `simulation/arcs.py` (NarrativePhase with `act` + `world_entities`, BUILTIN_ARCS, select_arc_for_goal), `simulation/narrator.py` (two-call decide+generate, phase instruction passthrough), `simulation/generative_runner.py` (per-phase entity activation, imagination fallback) |
| DM campaigns | `simulation/dm_schema.py`, `simulation/dm_runtime.py` |
| Asset Foundry | `simulation/foundry.py` (FoundryRunner: generate, validate, 8 SEM protocol tests + 3-encounter gauntlet, 4-dimension bio-engagement score) |
| Benchmarks | `simulation/benchmark.py`, `simulation/validation.py` |
| Research | `simulation/research_agents.py`, `simulation/research_orchestrator.py` |
| Cradle | `simulation/arcs.py::BUILTIN_ARCS["cradle"]` (4-act developmental arc), `_data/components/bodies/infant_humanoid.yaml`, `_data/components/items/cradle_*.yaml`, `_data/reflexes/infant.yaml`, `simulation/generative_runner.py::_activate_phase_entities` |
| Imagination | `imagination/trigger.py` (entity extraction + ComponentIndex lookup + design dispatch), `imagination/designer.py` (ImaginationDesigner wrapping EntityDesigner), `imagination/cache.py` (session-scoped ImaginationCache) |
| Interactive UI | `interactive/prompts.py` (universal PromptRequest/PromptHandler protocol), `interactive/display.py` (rich split-panel display, DM extensions) |
| Seed data | `_data/components/` (65 SEM components, 7 categories, genre-gated), `_data/encounters/` |

Also see docs/agents/bio-memory.md for the imagination substrate side (`NAc.decay_imagined_links`, imagined-provenance episodes, W2 substrate-signal manifest); docs/agents/embodiment.md for SEM/affordance semantics the foundry and cradle generate against.

## 3. Invariants & lessons

**[engineering] Don't run the benchmark harness on the same machine as the leader** — co-locating requires the harness's children to bypass role-detection, which has no clean entry point; run it from a peer machine. Post-2026-06-05 hardenings (singleton spawn guard in `runtime/llm_server.py::check_existing_llm_server` + harness preflight `assert_subsim_routed_not_local`) make leader-local firing safe. Diagnostic signature: `lane_decisions.jsonl` shows `tier_decisions.large.source: "tier_table"` AND >1 llama-cpp process on the leader. Full history: [docs/lessons/no-harness-on-leader-machine.md](../lessons/no-harness-on-leader-machine.md). Regression guard: [tests/unit/test_llm_server.py::TestCheckExistingLlmServer](../../tests/unit/test_llm_server.py) (spawn/reuse-200/reuse-401/fail-loud-wrong-model) + [tests/behavioral/test_exp37_harness_smoke.py::TestHarnessPreflight](../../tests/behavioral/test_exp37_harness_smoke.py) (rejects tier_table, accepts env/reused_server).
*(The spawn-guard/preflight code itself is routing-side — also see docs/agents/llm-routing.md.)*

**[engineering] `NarrativePhase.act` and `.world_entities` structure long-horizon sims** (cradle Stage 5). `act: str | None` groups phases into narrative acts. `world_entities: tuple[str, ...]` lists component refs activated on phase entry. The generative runner activates entities per-phase with imagination fallback for missing refs. Entities persist across phases (accumulate, not reset). Regression guard: [src/maxim/simulation/arcs.py::NarrativePhase](../../src/maxim/simulation/arcs.py) (dataclass shape) + [src/maxim/simulation/generative_runner.py::_activate_phase_entities](../../src/maxim/simulation/generative_runner.py).

## 4. Running sims — keep them small (full discipline set)

Core retains the three session-killing bullets (`--interactive false` from scripts, configure via `maxim config`, never co-locate leader + harness). The full set:

- **Configure model + n_ctx via `maxim config`, not transient env/flags — single source of truth.** `maxim config set llm.profile <profile>` + `maxim config set llm.n_ctx <N>`, then VERIFY with `maxim doctor 2>/dev/null | grep -i "n_ctx\|profile"` before a real run. The server and the PromptBudgeter resolve n_ctx through DIFFERENT paths; if they drift, llama-cpp returns HTTP 500 and the agent silently takes 0 real actions. Sub-sims inherit `config.json` via `_apply_lane_config_to_env`, so persisting there fixes the whole process tree — and it remains MANDATORY for harness/sub-sim runs and externally-started servers (the cross-process `served_n_ctx` stamp does not reach a sub-sim in another process). Full story + the three-leg fix state: [docs/lessons/sim-n-ctx-config-drift.md](../lessons/sim-n-ctx-config-drift.md).
- **Always pass `--interactive false` for automated/scripted sim runs** (Claude Code, CI, scripts). Interactive mode is ON by default with a TTY; the raw terminal reader conflicts with non-human stdin.
- **Set a narrow goal.** `--goal "test X specifically"` beats `--goal "test safety"` — specific goals converge faster.
- **Cap duration.** Ctrl+C after 30–90 s once you've seen what you need; sims report partial results on cancel. Or use `--sim-max-turns N`.
- **Prefer `--sandbox tmpdir` for debugging** unless specifically testing Docker — no pull/startup cost.
- **Use `--debug` sparingly** — great for diagnosing stalls, floods the terminal otherwise.
- **Don't invoke sims from test suites** unless the test is specifically for sim machinery — real LLM calls can 2–3× suite runtime. Tests mock LLM calls. Kill stale sims before test runs: `pkill -f "maxim.*sim"`.
- **Re-use sessions with `--resume-sim SESSION_ID`** to skip setup/warm-up when iterating. (NAc restore on resume uses `apply_decay=False` — sims are tick-anchored.)
- **Local models > Claude for loop-testing.** `--language-model mistral-7b` for sanity checks; save Claude for verifying final behavior.
- **Watch `Cost:` in the final report.** $0.05–$0.15 per short run is normal; $0.50+ means the sim is too broad or too long.
- **Interactive changes get a logged session:** `MAXIM_LOG_FILE=/tmp/maxim.jsonl maxim --sim "test basic recall" --interactive --sim-max-turns 3`, then read the JSONL for percepts, tool calls, and `ACTION_FOLLOWUP` entries. `MAXIM_BACKEND_TRACE=1` for per-call token/latency.
- **`~/.maxim/` is shared across worktrees** — don't run sims from concurrent sessions or they collide on persisted state.

**Simulation reports** save to `~/.maxim/sessions/{session_id}/` — `report.json`, `actions.jsonl`, `aut_hippocampus.json`, `aut_nac.json`. Substrate-primary telemetry goes to `data/sim_sandbox/`. Research protocol details and campaign execution flow: `docs/simulation.md` + [docs/experiments/](../experiments/).

## 5. Live gotchas / known gaps

- **Exp 44 embodied llm-primary numbers still need re-validation against the Track-1 drift-tick cadence.** Live venue: [docs/experiments/44b_pilot.md](../experiments/44b_pilot.md) (pilot complete) + [docs/experiments/protocols/exp44b_preregistration.md](../experiments/protocols/exp44b_preregistration.md) (confirmatory campaign not yet frozen).
- **S1 declaration for the D13/D14 planning-liveness change (`feat/d13-planning-liveness`, PR pending, 2026-08-19):** it alters *when a campaign terminates*, so every campaign-based Earned row runs through it. No `Re-run on:` trigger fires literally (checked all ten rows), and the change is inert on a healthy run — it only acts where the pre-fix code idled forever, which no valid measurement contains. Rows riding on it: all orchestrator-driven campaigns (Exp 09/10/37/38/40/44/48). If a future arm disables it via `MAXIM_SIM_PLANNING_LIVENESS=0`, that is a declared fidelity change (S6).
- **Exp 48 is CONTESTED** — magnitudes reproduced nowhere; the surviving number is the ~0.11 novelty visibility floor. Pre-fix actions/turn was a stopwatch (wall-clock ÷ 0.5 s), which is why `MAXIM_SUBSTRATE_ACTIONS_PER_TURN` exists and why S6 declares fidelity changes between arms.
- **`safe_pref` in Exp 42/42b is saturated** (SD 0.000) — it supports "did not break discrimination" but cannot detect a moderate regression; a future degradation arm needs a more sensitive statistic (S7's motivating case).
- **The cradle_mother operant embodied DEMO is Dormant** (measured at chance embodied; the operant claim was validated on the scripted substrate, `scripts/orient_substrate/4-7` + docs/experiments/46). Its three env toggles below are default-OFF/inert and support the dormant demo only.
- **Imagination distractors break controlled A/Bs**: LLM-primary AUTs engage improvised entities (merchant/book/pedestal) unless `MAXIM_DISABLE_IMAGINATION=1`. That flag does NOT gate narrator prose — use the fixture path for full scene determinism.
- **ComponentIndex implementation constraints** (`embodiment/component_index.py`): reuses the `similarity.encoder._get_encoder` singleton (no duplicate embedding model); persistence is `.npy` + `.json` sidecar (no pickle); thread-safe via RLock; two-layer discovery = alias hash table (from `component.synonyms`, O(1)) + embedding cosine (threshold 0.65).
- **Substrate-primary harness runs MUST pass `--embodiment bodies/infant_humanoid`** with `--aut-mode substrate-primary`.
- `scripts/analyze_exp37.py --out` overwrites its target.
- Opt-in experiment env vars in hot startup paths need autouse conftest scrubs in the same commit (see core cross-cutting rules); every var below lists its scrub.

## 6. Env vars owned

One line per var (name = purpose = owning module). Long rationales: [docs/lessons/claude-md-2026-08-13-pre-diet.md](../lessons/claude-md-2026-08-13-pre-diet.md).

- `MAXIM_PROVENANCE_VERBOSITY` = decision-log verbosity 0/1/2 → `~/.maxim/util/lane_decisions.jsonl`; inspect via `maxim doctor --last-decision` = runtime decision log.
- `MAXIM_SIM_PLANNING_LIVENESS` = "0"/"false" opts OUT of the D13 planning-liveness recovery+abort. Default ON: the opted-in sim orchestrator follows its exact `LLMWorker`/`WorkerPool` job through `PENDING` → `RUNNING` → `COMPLETED` → `CONSUMED`; a planning turn that terminates without an executable proposal is loudly requeued up to 3× and then aborts instead of idling forever. Finish status distinguishes `llm_wedged` (parse/provider failures), `planning_failed` (responsive but non-executable planning), and `worker_unavailable` (bounded worker/queue transport exhaustion). Opt-in per loop via `run_agentic_loop(planning_liveness=True)` — only the sim orchestrator opts in; decision logic `runtime/loop_controller.py::LoopController.record_planning_failure` / `record_planning_transport_failure` = `runtime/agent_loop.py::_handle_planning_failure` / `_handle_planning_transport_failure`; scrub `_isolate_maxim_sim_planning_liveness_env`. **S6 apparatus note:** this changes when a campaign TERMINATES, so it is an experiment-visible control — declare it if an arm disables it.
- `MAXIM_SIM_HARD_ABORT` = "0"/"false" opts OUT of the D12 stall hard-abort (default ON: a provably-wedged orchestrator LLM call terminates the sim loudly — exit 4 via forced-exit backstop — instead of hanging unboundedly; decision logic `runtime/stall_threshold.py::should_hard_abort`) = `simulation/orchestrator.py::_stall_detector`.
- `MAXIM_CRADLE_MOTHER_STIMULUS_ORDER` = "shuffled" = apparatus-v3 seeded per-block stimulus permutation (the L2 phase-lock dither, #514); default cycle = `simulation/generative_runner.py`; scrub `_isolate_cradle_mother_stimulus_order_env`.
- `MAXIM_DETERMINISTIC_SCENE_EMBODIMENT` = force scene-affordance `self_effect` onto the AUT body in LLM-primary (controlled Exp 44 harm attribution; Exp 37/38 must NOT set it) = `simulation/orchestrator.py`; scrub `_isolate_maxim_deterministic_scene_embodiment_env`.
- `MAXIM_DISABLE_IMAGINATION` = THE single master switch for ALL imagination surfaces (per-turn design + world-builder manifest); does not gate narrator prose = `simulation/orchestrator.py` → `imagination/trigger.py`; scrub `_isolate_maxim_imagination_env`.
- `MAXIM_DISABLE_IMAGINATION_SUBSTRATE_SIGNAL` = W2 ablation: drop NAc biases from the scene manifest, KEEP the manifest (distinct from the master switch) = sim orchestrator producer site.
- `MAXIM_EXP44_CAPTURE_LOG` = path for paired full/ablated prompt JSONL (Exp 44 counterfactual capture, `scripts/exp44`); dormant unset; fires on the injected persistent-agent path too = orchestrator PromptBuilder wrap; scrub `_isolate_maxim_exp44_capture_log_env`.
- `MAXIM_OPERANT_ONLY_CREDIT` = suppress tool-success cluster-reward floor for driveless actions so a caregiver's contingent feed is the sole teacher (HALF-fix, cluster surface only) = `runtime/tool_dispatch.py`; Dormant demo; scrub `_isolate_maxim_operant_only_credit_env`.
- `MAXIM_SUBSTRATE_TOOL_WHITELIST` = restrict substrate-primary action selection to a comma-separated repertoire (BAND-AID for credit-on-execution root cause, `docs/plans/deferred/credit_on_progress_not_execution.md`); empty = inert = `runtime/agent_loop.py::propose_via_substrate`; scrub `_isolate_maxim_substrate_tool_whitelist_env`.
- `MAXIM_CRADLE_MOTHER_DISABLE_CARE` = cradle_mother no_feed control arm (mother places sound, never feeds/credits) = `simulation/generative_runner.py`; scrub `_isolate_maxim_cradle_mother_disable_care_env`.
- `MAXIM_SUBSTRATE_ACTIONS_PER_TURN` = turn-scoped action budget for the substrate-primary AUT (Exp 48 thrashing fix); unset = unbounded stopwatch regime; invalid → WARNING + unbounded, never a silently invented bound; S6 apparatus toggle — changing it between arms is a declared fidelity change = `simulation/bridge.py` + orchestrator; scrub `_isolate_maxim_substrate_actions_per_turn_env`.

**Ablation-arm index** — vars used by experiment arms but homed on their mechanism's brief:

| Var | Home brief |
|---|---|
| `MAXIM_NAC_REWARD_BIAS_DISABLED`, `MAXIM_NAC_MIN_CONFIDENCE`, `MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION`, `MAXIM_DISABLE_VARIANCE_ANNOTATION`, `MAXIM_EC_TRACE_ACTIVATIONS`, `MAXIM_PLACE_CODE_EXTEROCEPTION`, `MAXIM_SUBSTRATE_PATH` | docs/agents/bio-memory.md |
| `MAXIM_ENABLE_BODY_STATE_PROMPT`, `MAXIM_DISABLE_COACH_BODY_LAYERS`, `MAXIM_MOTOR_CREDIT_TRACE`, `MAXIM_DEEP_EMBODIMENT` | docs/agents/embodiment.md |

## 7. Lesson archive (owned by this brief)

- [docs/lessons/no-harness-on-leader-machine.md](../lessons/no-harness-on-leader-machine.md) — the Exp 37 cradle-cascade incident (L02).
- [docs/lessons/sim-n-ctx-config-drift.md](../lessons/sim-n-ctx-config-drift.md) — the Exp 44 blocker: budgeter-vs-server n_ctx drift → silent down_500 / 0-action runs, and the three-leg fix state.
- (L01 harness-provenance lives in CLAUDE.md core; its archive file is [docs/lessons/harness-provenance-assert-repo-interpreter.md](../lessons/harness-provenance-assert-repo-interpreter.md).)

# Baseline floors: a bare LLM, a generic-memory LLM, and a non-LLM learner beside every claim

> **PROPOSED 2026-10-10 — a parallel line** ([roadmap_1_4.md](roadmap_1_4.md) §Parallel lines). Owner
> decisions BF1–BF2 (DECISIONS.md 2026-10-10). This revises the owner's draft "Maxim: Bare-LLM Baseline
> Plan" (2026-10-10, kept outside the repo as a `.docx`) after a four-angle read-only dive against
> `origin/main` 8ff5ae70: the Exp 60 world, the Exp 37 harness, which bio layers the builder can separate,
> and the provenance and roadmap fit. The draft's goal and its design principle stand. Its first target,
> its run record and its roadmap slot do not (§What the dive changed).

## Why this plan exists

No ledger row, EARNED or STALE, has ever been compared with an agent that lacks the bio layers. "Earned"
today means beating one of Maxim's own ablations: Exp 60's control is FEAR vs ABLATED (fear subscriber
detached), and Exp 37's control arm A runs the full Maxim stack with nothing carried over.
`simulation/benchmark.py::BenchmarkRunner.baseline_path` compares one report with an earlier report, not
with a no-bio agent. The first question a reviewer asks, "how much better than a plain agent on the same
model?", has no answer.

The line gives the ledger three things:

- **Attribution.** The gain over `bare_llm` (LLM path) or over `tabular_q` (substrate path) is the number
  a reviewer asks for first.
- **A falsifiable persistence claim.** Beating `llm_notes` is what separates Maxim from "just give it notes".
- **Comparability.** The same bare loop can later run an external suite next to Maxim.

**Design principle (kept from the draft): build up, don't subtract.** The bare arm is a small loop that is
bare by construction, not the full stack with switches turned off. Subtraction cannot be proven complete:
`_build_tool_aware_prompt` stacks about a dozen sections, and `run_agentic_loop` is still 2,429 lines on
`origin/main`. The full-stack-all-off arm still runs as its own point, so the scaffolding gap
(all-off − bare) and the bio gap (full − all-off) are reported separately.

## What the dive changed

| Draft said | The code says | So |
|---|---|---|
| First target: `bare_llm` in the Exp 60 drowning world | No LLM acts there: `scripts/survival_world/exp60_run.py` runs `aut_mode="substrate-primary"` with no `llm_worker`, and `MinecraftPerceptSource` renders no text for the 17 sensor channels. A bare LLM shown `is_in_water=1, oxygen=18` next to an `escape_water` tool would surface on the PRE-probe from priors, scoring the claim's ceiling with no learning. The 4.3 s US-free window leaves about 2 s for perceive → LLM → tool call, which qwen32b on the rig likely misses. Training is propose-only, so `llm_notes` has nothing to learn from. | The survival world's floor is non-LLM (`tabular_q`, `random`), Stage B1. An LLM arm in that world waits for a prior-free variant (Stage B5). |
| A ~300-line standalone loop for Exp 37 | Exp 37's world is narrated by an LLM inside `simulation/orchestrator.py::start_simulation_mode`; observations exist only there. | The bare loop plugs in at the AUT seam of `start_simulation_mode`: a `src/` change that waits for that function's decomposition (Stage B2). |
| — | On `--resume-sim` the narrator receives the previous session's summary, issues and tool usage (`simulation/sim_types.py::build_resume_prompt`). The narrator is a cross-session memory channel. | Every cross-session arm runs a NON-resumed narrator, so only the agent's memory differs. Exp 37's B−A has always mixed narrator memory into the effect; this is recorded against T1-2 (§Side findings). |
| Step 1: a new shared run record extracted from `_provenance.py` and `simulation/report.py`, plus a ledger-lint extension | M1 shipped most of it: `report.py::run_provenance` + `scripts/_evidence_records.py` already stamp and gate commit, tree digest, dirty flag, profile, n_ctx, endpoint, tokens, cost and typed abort. | Add only the missing fields to `run_provenance` and their presence checks to `_evidence_records.py` (Stage B0). No second schema. |
| Slots into 1.4 Phase 0 item 2 | Item 2's acceptance checks forbid a public name the three harnesses lack and require net-negative `scripts/` LOC, and §What is NOT in 1.4 excludes "a benchmark framework". | A parallel line, BF1. |
| Ladder: all-off → +memory → +valence → full | `runtime/bio_stack.py::build_bio_stack` has no switches; NAc is wired into the pain bus, the credit distributor and the salience scorer. Only bare / all-off / full are buildable without new wiring. | Three rungs now; +memory and +valence wait for typed builder flags (§Arms). |
| AST import guard on the bare loop | `models/language/router.py` imports `runtime.*`/`agents.*` lazily inside functions, and those package `__init__`s load ~218 modules including every bio package. An AST check on the loop's own files passes while the process loads the whole bio stack on its first model call. | The guard is AST **and** a subprocess `sys.modules` check after one mocked turn, plus a seam fix (§Guards). |

## Owner decisions (2026-10-10)

- **BF1. A parallel line, not a 1.4 Phase 0 item.** It touches a survival rung only as a declared arm in
  that rung's prereg (§Parallel lines' rule), and its `src/` work enters after the decomposition it depends
  on.
- **BF2. The floor becomes part of earning, strict by default.** From 2026-10-10:
  - A NEW `[behavioral]` claim on the **LLM path** cannot be `EARNED` unless the full arm beats `bare_llm`
    and, for a cross-session claim, `llm_notes`, on the primary measure, same model and seeds.
  - A NEW claim on the **substrate path** cannot be `EARNED` unless it beats `tabular_q` and `random` on the
    same state keys. 1.4's E1–E3 rungs therefore carry those two as declared arms in their preregs.
  - A claim that does not beat its floor is recorded as a boundary result (what the ablation still shows,
    and that a simpler learner matches it), not hidden. It is publishable, not a failure.
  - Existing EARNED/MAINTAINED rows keep their status. Each owes its floor at its next trigger re-run once
    the floor arm exists for its harness; until then the row's prose carries `floor: not measured`.
  - Enforced by: mechanization backlog [M46](outstanding.md) (a ledger-lint check on the prereg's declared
    arms), process invariant until it lands.

## Arms

Every result row names exactly one arm. The arm registry (Stage B2) maps each to its configuration and
replaces the per-experiment env vars for benchmark runs.

| Arm | Path | What runs | Its gap measures | Buildable |
|---|---|---|---|---|
| `bare_llm` | LLM | New loop: same model, tools, observation text, parser and retry policy; no state across sessions | The LLM-path floor | After B2's seam fix |
| `llm_notes` | LLM | `bare_llm` + generic memory carried across sessions (episodic log retrieval or Reflexion-style notes) | Whether Maxim beats "just give it notes" | After B2 |
| `maxim_all_off` | LLM | `AgentFactory.create_full_agent(with_bio_stack=False, remembers=False, learns=False)`, `build_executor(pain_bus=None, nac=None)`, `memory_hub=None` | Scaffolding alone (vs `bare_llm`) | Today, with the two prompt pins below |
| `full` | both | The current default | Whole-stack gain | Today |
| `tabular_q` | substrate | Tabular Q-learner keyed on the live agent's EC cluster ids × action, exploration declared | The substrate-path floor | B1 |
| `random` | substrate | Uniform over the selectable actions | Sanity floor | B1 |
| `+memory`, `+valence` | LLM | Ladder rungs | One layer each | **Deferred**: typed `build_bio_stack` flags first, plus the NAc-overwrite fix (§Side findings) |

`maxim_all_off`'s prompt must pin two sections that fire with zero bio signal: `pfc_preamble` (fires
whenever `simulation.sim_logger._sim_active` is set) and `acting_coach` (gated only on
`request.acting_coach is not None`). Both are scaffolding; the all-off arm keeps or drops them on purpose
and records which.

Pain and drives have no separable component today (the NAc drive gate is Dormant), so the draft's
`+valence` and `full` rungs merge unless a pain/drive builder flag is added.

## Matching controls

Every arm differs from its neighbour in one named thing. Everything below is held identical and recorded.

| Control | Rule |
|---|---|
| Information | The baseline sees everything the environment shows: the same observation text and sensor values. A starved baseline is the fastest way to lose a reviewer. |
| Prompt effort | A competent bare prompt, written and frozen in the prereg by a different reader from the one who wrote Maxim's prompt; or a published ReAct format verbatim. No task hints, no sandbagging. |
| Model and serving | Same weights, quantization, KV quantization, n_ctx, backend and server version (stamped by B0). |
| Sampling | Same temperature, top_p, and seed per paired trial. |
| Action handling | Same tool-call parser and repair (`<think>` stripping, `_llm_unavailable` fallback), invalid-action handling and retry policy: shared code, not a copy. |
| Executor wrappers | `FearGatedExecutor` costs one extra LLM review per tool call and can block. Keep it on every arm or drop it on every arm. |
| Narrator | Non-resumed on every cross-session arm. |
| Other memory channels | `examine`'s recall of past memories (#845) is held equal across arms. |
| Budget | Same max turns and wall-clock cap. Turns close on `say`/`respond`, or per-turn binning collapses. |
| Session history | Same number of prior sessions for every cross-session arm. |
| Tokens and cost | Recorded per arm; results reported per turn and at matched token budgets. Spend reaches `report.json`'s `cost_usd` so the cost cap sees it. |

## Guards

The bare arm stays bare without anyone watching:

- **Static layer rule.** A `bare_loop` rule in `utils/audit.py::LAYER_RULES` (`maxim --audit-architecture`):
  it already scopes module-level, function-local and `TYPE_CHECKING` imports, so it catches lazy imports
  that an `ast.walk` over top-level `ImportFrom`s would miss. Allowed: the environment adapter, `tools/`
  schemas, `models/language/`, the run-record writer.
- **Runtime check.** A subprocess test runs one mocked `bare_llm` turn and asserts no bio package
  (`memory`, `decisions`, `embodiment`, `harm`, `default_network`, `salience`, `imagination`, `integration`,
  `reactions`, `proprioception`) is in `sys.modules`.
- **The seam fix that makes the runtime check passable.** Move the router's call-time `runtime`/`agents`
  helpers (`_maybe_dispatch_backend_class` → `runtime.lane_backends`; `runtime.llm_call_registry`;
  `agents.llm_worker._normalize_request_context`; `agents.prompt_builder.PROMPT_SEGMENT_DELIMITER`) into
  `models/language/`, or make the `runtime`/`agents` package `__init__`s lazy. Decided at B2's design
  review. Not an allowlist entry.
- **Prompt snapshot.** A golden file of the bare system prompt and one rendered turn.
- **Arm parity test.** One fixture world, every arm, mock LLM: identical observation text, tool list,
  parser and retry settings.
- **Mock smoke lane.** Exp 37's `--mock-llm` writes synthetic `actions.jsonl` and runs no agent, so it
  cannot exercise a loop. The lane runs each arm's real loop on recorded LLM responses.

## Stages

Each stage is one PR (B2 may be two) and goes through the reviews its kind requires.

- **B0. Provenance fields.** No rig, no dependency; can start now. Add to `run_provenance`: model file
  SHA-256 (nothing in `src/maxim/models/` hashes a GGUF yet), quantization, KV quantization (`kv_quant_mode`
  exists in `runtime/lane_models.py` but not in `report.json`), server version and upstream model id, temperature, top_p,
  sampling seed, and the run fields `arm`, scenario hash, world seed, session index. Add their presence
  checks to `scripts/_evidence_records.py`, scoped so pre-B0 records stay valid. The model-identity and
  server fields are the same ones the vLLM self-hosting draft needs (`upstream_model_id`, `served_n_ctx`);
  define them once, here.
- **B1. The survival world's floor: `tabular_q` + `random` on Exp 60.** Derivative design: confounding +
  bio-faithful lenses. No LLM, so no contention with qwen32b on the rig box. This carries a real threat, and
  that is its value: with about two relevant states (shore, submerged), a one-step learner may match FEAR
  once it has tried `escape_water`. Decisions for the design review:
  - Exp 60's training is propose-only, so a Q-learner gets no action-outcome samples. Either it learns
    from the pain on the proposed action's value, or training executes (which breaks the yoked schedule).
  - Exploration is the real independent variable. `substrate_explore_bonus_weight` is pinned at 0.0 in the
    frozen fingerprint; `tabular_q`'s exploration is declared in the prereg, not tuned after data.
  - Key on the live agent's cluster ids (first-touch allocation makes ids order-dependent).
  - There is no offline drowning simulator (`minecraft_harness.py::FakeBridgeServer` has no water physics),
    so the arms are proven offline only for wiring; liveness preflights follow Exp 60's amendments 3–7.
- **B2. The bare loop, the guards and the arm registry.** Entry condition: the `start_simulation_mode`
  decomposition has landed (it is next after the agent loop; touching `orchestrator.py` before then collides
  with it). Composes the decomposed seams (`loop_setup.build_loop_run`, `loop_perception.perceive`) with a
  bare LLM step and a dispatch path that books no learning: `tool_dispatch.execute_and_learn` books NAc
  learning, so a no-learning dispatcher is a new seam, and it must sit on M38's allowlist with its reason.
  `models/language/json_parser.py::_extract_json_object` is the clean parser to make public. Three-lens code
  review; mypy over the touched composition files.
- **B3. `llm_notes`.** Notes stored in the agent's home with `atomic_write_json` + `_format_version`, one
  end-of-session reflection call through the router. Pick episodic retrieval or Reflexion first (open
  decision); the other can follow as a second arm.
- **B4. Exp 37 / T1-2 re-run with floors.** Arms `bare_llm`, `llm_notes`, `maxim_all_off`, `full` (B
  family), non-resumed narrator, #845 recall held equal. Four-lens design review (new claim). n well above
  5: arm A's own spread across re-runs is about 0.12–0.27 (L8). Gate on `fire_pit`, with `deceptive_fire`
  for the counter-prior; `sharp_rock`'s primary metric is structurally absent and `deceptive_fire`'s is
  structurally 0 as a primary. `run_benchmark`'s arm loop is three hard-coded blocks, so the registry
  replaces them.
- **B5. Later.** (a) An LLM arm in the survival world only where priors do not help: relabelled cues and
  action names, or R4's delayed credit. It needs a text renderer, an `llm_worker` path, a latency preflight
  and a box separate from the bridge's LLM server. (b) One external long-horizon memory suite (LongMemEval or
  LoCoMo) running `bare_llm`, `llm_notes` and `full` through the same record.

B0 and B1 need nothing else. B2 → B3 → B4 are serial. B5 waits for B4.

## Side findings (each gets its own issue)

1. **`create_full_agent` ignores `learns=False` when `with_bio_stack=True`.** Its docstring says "Full
   bio-stack if remembers=True and learns=True", but `config.with_bio_stack` alone builds the stack, and
   `instance.nac = bio.nac` replaces the skeleton's absent NAc. A silent no-op of the class CLAUDE.md says to
   push into the type. Blocks the `+memory` rung.
2. **Four `MAXIM_DISABLE_*` vars are dead:** `MAXIM_DISABLE_PFC_PREAMBLE`, `MAXIM_DISABLE_ACTING_COACH`,
   `MAXIM_DISABLE_SIM_SANDBOX_TEXT`, `MAXIM_NO_DEFAULT_PERSONA`. No reader in `src/` or `tests/`; only
   `scripts/run_v1_phases.sh` exports them, and its own comment says they no-op since #215. Delete; never map
   them to arms.
3. **Exp 37's narrator is a cross-session memory channel** (above). T1-2's re-run must hold it fixed; the
   historical B−A numbers mix it in. A ledger annotation on T1-2, not a status change (T1-2 is already STALE).

## Risks

- **Baselines win on the LLM path.** Plausible given T1-2's history. Under BF2 that is a recorded boundary
  result, and it sharpens the story around the substrate-path results.
- **`tabular_q` matches Exp 60.** Equally plausible. The defensible claim then narrows to what a tabular
  learner does not do: transfer between agents (Exp 61), transfer across pools (Exp 62), the rungs E1–E3
  are built to test. Better learned from B1 than from a reviewer.
- **Scope creep.** The bare loop must not grow features. The layer rule, the `sys.modules` check and the
  prompt snapshot are the brake.
- **Rig time.** B1 needs the Minecraft server; B0, B2 and B3 do not.

## Open decisions (asked at each stage's start)

- B1: how `tabular_q` learns under propose-only training; its exploration schedule.
- B2: move the router helpers into `models/language/`, or make the package `__init__`s lazy.
- B3: episodic retrieval or Reflexion first.
- B4: bare prompt written by a different reader, or a published ReAct format verbatim; trials per arm (set
  in the prereg).
- B5: LongMemEval or LoCoMo.

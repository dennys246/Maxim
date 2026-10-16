# Baseline floors: a bare LLM, a generic-memory LLM, and a non-LLM learner beside every claim

> **PROPOSED 2026-10-10 — a parallel line** ([roadmap_1_4.md](roadmap_1_4.md) §Parallel lines). Owner
> decisions BF1–BF2 (DECISIONS.md 2026-10-10). This revises the owner's draft "Maxim: Bare-LLM Baseline
> Plan" (2026-10-10, kept outside the repo as a `.docx`) after a four-angle read-only dive against
> `origin/main` 8ff5ae70: the Exp 60 world, the Exp 37 harness, which bio layers the builder can separate,
> and the provenance and roadmap fit. A three-lens review round (Executor, Architecture, Wire integrity)
> folded into this version. The draft's goal and its design principle stand. Its first target, its run
> record and its roadmap slot do not (§What the dive changed).

## Why this plan exists

*Status 2026-10-14:* B0 is merged (#1233). B1 ran (campaign `48ce40118c20`) and returned `FLOOR_MATCHED_Q`: FEAR
8/8, Q_EXPLORE 8/8, RANDOM 2/8 at the first post placement, so T1-13 is `PARTIAL (narrow: floor matched)`
([prereg](../experiments/b1_exp60_floor_prereg.md)). The paragraph below is the motivation as of 2026-10-10.

*Owner decision 2026-10-15 (after B1): T1-14 (Exp 61) takes `PARTIAL (narrow: floor matched, analytical)` and
T1-15 (Exp 62) `PARTIAL (rung A: floor matched, analytical)`, without a rig run (owner exceptions, #1294).* They rest on the same Wire-4 fear, and their floors match by construction:
- a floor the harness arms in water gets WHEN for free; with the same prior it surfaced on its first-ever
  submersion with zero experience in B1 (Q_EXPLORE_PRIOR, pre-probe placement 0: 8/8);
- the fair floor for learning WHEN from passive pain is a state-value learner, and TD state value on the cluster
  plus the same prior IS Wire-4 (the B1 prereg's analytical twin; an action-value learner cannot learn there, which
  is why Q_YOKED was rejected as a strawman);
- Exp 62's transfer lives in the EC key the floors share (its NODE gate), and B1's raw key holds no altitude.

**What this obliges for E1–E3 (the ladder's floor design):** the mechanism twin (TD state value on the cluster +
the same prior) is RUN as an explicit arm, so a rung reports its gain over the twin as well as over `tabular_q` and
`random`; and a rung that wants to EARN over the floor needs a contingency a reasonable floor can lose, the
candidate being generalization through EC similarity to states a reasonable tabular key treats as new. Choosing an
over-fine floor key so that it loses is a strawman in our favour and is for the confounding lens to refuse.

*Lessons from B1:* [docs/lessons/baseline-floors-b1.md](../lessons/baseline-floors-b1.md) (the toy model before rig
time, floors unfair in either direction, the twin, the demotion-needs-an-exception sequence). *Next, after 1.3.2:*
E1's prereg draft and four-lens review carry the twin and a floor-can-lose check from the start; B2 (bare LLM)
still waits on the `start_simulation_mode` decomposition, and B0b on #1227/#1228.

Until B1, no ledger row, EARNED or STALE, had been compared with an agent that lacks the bio layers. "Earned"
today means beating one of Maxim's own ablations: Exp 60's control is FEAR vs ABLATED (fear subscriber
detached), and Exp 37's control arms A and C run the full Maxim stack. `simulation/benchmark.py::BenchmarkRunner.baseline_path`
compares one report with an earlier report, not with a no-bio agent. The first question a reviewer asks,
"how much better than a plain agent on the same model?", has no answer.

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
| First target: `bare_llm` in the Exp 60 drowning world | No LLM acts there: the harness runs `aut_mode="substrate-primary"` (`simulation/minecraft_harness.py::_loop_kwargs`) with no `llm_worker`, and `MinecraftPerceptSource` renders no text for the 17 sensor channels. A bare LLM shown `is_in_water=1, oxygen=18` next to an `escape_water` tool would surface on the PRE-probe from priors, scoring the claim's ceiling with no learning. The 4.3 s US-free window leaves about 2 s for perceive → LLM → tool call, which qwen32b on the rig likely misses. Training is propose-only, so `llm_notes` has nothing to learn from. | The survival world's floor is non-LLM (`tabular_q`, `random`), Stage B1. An LLM arm in that world waits for a prior-free variant (Stage B5). |
| A ~300-line standalone loop for Exp 37 | Exp 37's world is narrated by an LLM inside `simulation/orchestrator.py::start_simulation_mode`; observations exist only there. | The bare loop plugs in at the AUT seam of `start_simulation_mode`, which waits for that function's decomposition (Stage B2). |
| — | On `--resume-sim` the narrator receives the previous session's summary, issues and tool usage (`simulation/sim_types.py::build_resume_prompt`). The narrator is a cross-session memory channel. | Every cross-session arm runs a NON-resumed narrator, so only the agent's memory differs (§Side findings 3). |
| Step 1: a new shared run record extracted from `_provenance.py` and `simulation/report.py`, plus a ledger-lint extension | M1 shipped much of it, in two producers: `report.py::run_provenance` (sim reports) and `scripts/_provenance.py::in_process_code_provenance` (harness-process rows), projected into rows by `_provenance.py::sim_evidence` via `_EVIDENCE_PROVENANCE`. The evidence gate (`scripts/_evidence_records.py`) checks commit, tree digest, dirty flag, profile, n_ctx and typed abort. `report.json` also carries endpoint, tokens and cost as `SimulationReport` fields, unchecked by the gate. | Add only the missing fields, to both producers and the projection (Stage B0). No new schema. |
| Slots into 1.4 Phase 0 item 2 | Item 2's acceptance checks forbid a public name the three harnesses lack and require net-negative `scripts/` LOC, and §What is NOT in 1.4 excludes "a benchmark framework". | A parallel line, BF1. |
| Ladder: all-off → +memory → +valence → full | `runtime/bio_stack.py::build_bio_stack` has no switches; NAc is wired into the pain bus, the credit distributor and the salience scorer. Only bare / all-off / full are buildable without new wiring. | Three rungs now; +memory and +valence wait for typed builder flags (§Arms). |
| AST import guard on the bare loop | Importing ANY `maxim.runtime.*` or `maxim.agents.*` submodule runs the eager package `__init__`s and loads about 200 modules, including memory, decisions, embodiment, harm and integration. The decomposed seams also import bio modules themselves (`loop_setup` → `bio_integration`, `tool_dispatch` → `decisions.causal_link`). The router reaches `runtime`/`agents` lazily at call time. An AST check on the loop's own files passes while the process loads the bio stack. | The bare loop takes no `runtime.*`/`agents.*` import; the guard is a layer rule **and** a subprocess `sys.modules` check; the root-cause fix is lazy package `__init__`s (§Guards, B2). |

## Owner decisions (2026-10-10)

- **BF1. A parallel line, not a 1.4 Phase 0 item.** It touches a survival rung only as a declared arm in
  that rung's prereg (§Parallel lines' rule), and its `src/` work enters after the decomposition it depends
  on.
- **BF2. The floor becomes part of earning, strict by default.** From 2026-10-10:
  - A NEW `[behavioral]` claim on the **LLM path** cannot be `EARNED` unless the full arm beats `bare_llm`
    and, for a cross-session claim, `llm_notes`, on the primary measure, same model and seeds.
  - A NEW claim on the **substrate path** cannot be `EARNED` unless it beats `tabular_q` and `random` on the
    same state keys.
  - **A row that does not beat its floor takes `PARTIAL <date>` with the qualifier `(narrow: floor matched)`**;
    its boundary result (what the ablation still shows, and which simpler learner matches it) goes in the
    status prose. It is publishable, not a failure.
  - Existing EARNED/MAINTAINED rows keep their status. Each owes its floor at its next trigger re-run once
    its harness has the floor arm. A pre-2026-10-10 row that loses to its floor at that re-run moves to
    `PARTIAL (narrow: floor matched)`, not `BROKEN`: its ablation result still stands. Until then the row
    gains `floor: not measured` in its status prose (never the `(…)` qualifier) at its next ledger touch.
  - **Consequences, stated so nobody discovers them later.** No new LLM-path claim can be EARNED until B2
    lands (and B3, for a cross-session claim), and B2 waits for the `start_simulation_mode` decomposition. 1.4's E1–E3 rungs each declare
    `tabular_q` and `random` in their prereg at their design review and build them in their own harness
    (B1 builds them first for Exp 60; Phase 0's "enters WITH its consuming rung" rule holds).
  - Enforced by: mechanization backlog [M46](outstanding.md), process invariant until it lands.

## Arms

Every result row names exactly one arm. The row-level `arm` the harnesses already write stays canonical.

| Arm | Path | What runs | Its gap measures | Buildable |
|---|---|---|---|---|
| `bare_llm` | LLM | New loop: same model, tools, observation text, parser and retry policy; no state across sessions | The LLM-path floor | B2 |
| `llm_notes` | LLM | `bare_llm` + generic memory carried across sessions (episodic log retrieval or Reflexion-style notes) | Whether Maxim beats "just give it notes" | B3 |
| `maxim_all_off` | LLM | `create_full_agent` with an `AgentConfig` of `with_bio_stack=False, remembers=False, learns=False`; `build_executor(pain_bus=None, nac=None)`; `memory_hub=None` in the loop. `create_agent` still builds an ATL, an SCN and a MemoryHub with those flags (`_create_atl`, `_create_memory_hub`), which the arm discards on purpose and records. | Scaffolding alone (vs `bare_llm`) | B4 (no harness can select it before; Exp 37 runs `maxim --sim` as a subprocess) |
| `full` | both | The current default | Whole-stack gain | Today |
| `tabular_q` | substrate | Tabular Q-learner keyed on the live agent's EC cluster ids × action, exploration declared | The substrate-path floor | B1 |
| `random` | substrate | Uniform over the selectable actions | Sanity floor | B1 |
| `+memory`, `+valence` | LLM | Ladder rungs | One layer each | **Deferred**: typed `build_bio_stack` flags first, plus the overwrite fix (§Side findings 1) |

`maxim_all_off`'s prompt must settle two sections that fire with zero bio signal: `pfc_preamble` (fires
whenever `simulation.sim_logger._sim_active` is set; suppressed by the existing
`request.deliberation_available is False` switch) and `acting_coach` (gated only on
`request.acting_coach is not None`). Both are scaffolding; the arm keeps or drops each on purpose and
records which.

**`tabular_q` vs NAc.** Keyed on the same EC cluster ids, a Q-table inherits the EC's generalization, and
`NAc.recommend_action` is already a situation-keyed value lookup over `current_cluster_id × tool`. B1's
design review names what separates them (a bootstrapped value backup vs a one-step similarity-weighted
bias). If it finds no difference, the floor IS the mechanism, and the substrate claim is about the pain
cascade and fear, not the value table.

Pain and drives have no separable component today (the NAc drive gate is Dormant), so the draft's
`+valence` and `full` rungs merge unless a pain/drive builder flag is added.

**No shared arm registry yet.** Arms are a frozen mapping inside each consuming harness (B1's, B4's) from
arm name to its env/config set. The existing ablation env vars (`MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION`,
`MAXIM_DISABLE_VARIANCE_ANNOTATION`, `MAXIM_NAC_REWARD_BIAS_DISABLED`) stay, with their `src/` readers,
their autouse scrubs in `tests/conftest.py`, their probe-script users and their brief table entries. A
shared registry enters only when a second harness consumes it.

## Matching controls

Every arm differs from its neighbour in one named thing. Everything below is held identical and recorded.

| Control | Rule |
|---|---|
| Information | The baseline sees everything the environment shows: the same observation text and sensor values. A starved baseline is the fastest way to lose a reviewer. |
| Prompt effort | A competent bare prompt, written and frozen in the prereg by a different reader from the one who wrote Maxim's prompt; or a published ReAct format verbatim. No task hints, no sandbagging. |
| Model and serving | Same weights, quantization, KV quantization, n_ctx and backend (stamped by B0 for the run's routers; for the agent under test only after B0b). |
| Sampling | Same temperature, top_p, and seed per paired trial. Exp 37's harness records a seed but never passes it to the sim (`run_one_sim` has no seed parameter), so B4 plumbs it. |
| Action handling | Same tool-call parser and repair (`<think>` stripping, the `_llm_unavailable` fallback), invalid-action handling and retry policy: shared code, not a copy (B2 pulls them into bio-free modules). |
| Executor wrappers | `FearGatedExecutor` costs one extra LLM review per tool call and can block. Keep it on every arm or drop it on every arm. |
| Narrator | Non-resumed on every cross-session arm. |
| Other memory channels | `examine`'s recall of past memories (#845) is held equal across arms, or ablated (owner decision 2026-10-06). |
| Budget | Same max turns and wall-clock cap. Turns close on `say`/`respond`, or per-turn binning collapses. |
| Session history | Same number of prior sessions for every cross-session arm. |
| Tokens and cost | Recorded per arm; results reported per turn and at matched token budgets. Spend reaches `report.json`'s `cost_usd` so the cost cap sees it. |

## Guards

The bare arm stays bare without anyone watching:

- **Where it lives.** A new top-level package `src/maxim/baselines/`, not `runtime/` (whose package import
  loads the stack).
- **Static layer rule.** `utils/audit.py::LAYER_RULES` is a per-package `must_not_import` denylist that
  already scopes module-level, function-local and `TYPE_CHECKING` imports (`maxim --audit-architecture`).
  `LAYER_RULES["baselines"]` denies `memory`, `decisions`, `embodiment`, `harm`, `default_network`,
  `salience`, `imagination`, `integration`, `reactions`, `proprioception`, `runtime` and `agents`; ARCHITECTURE.md §Layer Ownership gains the row.
- **Runtime check.** A subprocess test runs one mocked `bare_llm` turn and asserts no bio package is in
  `sys.modules`.
- **The root-cause fix that makes the runtime check passable.** Importing any `maxim.runtime.*` or
  `maxim.agents.*` submodule loads the whole bio stack because those package `__init__`s are eager. Make
  them lazy (PEP 562 `__getattr__`), first auditing for import-time registration side effects. Only
  constants and pure helpers (`PROMPT_SEGMENT_DELIMITER`, `_normalize_request_context`) may move into
  `models/language/`; `runtime/lane_backends.py::BACKEND_CLASSES` and `runtime/llm_call_registry.py` stay
  where CLAUDE.md and the llm-routing brief cite them. Not an allowlist entry.
- **Prompt snapshot.** A golden file of the bare system prompt and one rendered turn.
- **Arm parity test.** One fixture world, every arm, mock LLM: identical observation text, tool list,
  parser and retry settings.
- **Mock smoke lane.** Exp 37's `--mock-llm` writes synthetic `actions.jsonl` and runs no agent, so it
  cannot exercise a loop. The lane runs each arm's real loop on recorded LLM responses.

## Stages

Each stage is one PR or more and goes through the reviews its kind requires.

- **B0. Provenance fields. BUILT 2026-10-10** ([baseline_floors_b0.md](baseline_floors_b0.md), owner decision:
  split). A sim report now records which models answered its calls (per router object: provider, model, the
  joined identity, an owned server's actual KV-cache type, each call's temperature, the seed and whether the
  backend honours it), a completeness check, model-change and concurrent-sim stamps, the scenario files read
  and handed-over turn content, the global seed and the session index; the evidence gate reads the schema
  from the code that ran. What changed from this list, and why (fifteen design passes and a three-lens
  review): no run-level `temperature`/`top_p` (temperature is per call); no provenance `arm` (the row-level
  `arm` stays canonical; checking that a sub-sim ran its arm, `arm_effective`, enters with M46/B4); the leader
  reports its on-disk git HEAD (`leader_disk_heads`), not a server version, and it is never code identity; the
  global seed, not a world seed. **B0 does not say which model served the agent under test**
  (`aut_model_attribution: "not_established"`): that is **B0b**, after #1227 and #1228, and until then M46
  cannot check model parity for the AUT.
- **B1. The survival world's floor: `tabular_q` + `random` on Exp 60.** Derivative design: confounding +
  bio-faithful lenses. No LLM, so no contention with qwen32b on the rig box. It carries a real threat,
  which is its value: with about two relevant states (shore, submerged), a one-step learner may match FEAR
  once it has tried `escape_water`. Decisions for the design review:
  - **The seam.** Exp 60 acts through `run_agentic_loop(aut_mode="substrate-primary")`, and
    `substrate_proposal.propose_via_substrate` has no hook. Either a `src/` proposer hook (a `src/` change,
    three-lens code review, inside BF1's fence) or a scripts-only driver that encodes sensors with the same
    `SensorEncoder` + EC and acts through `simulation/minecraft.py::MinecraftClient` directly. The design
    review picks one; until it does, B1 is not "scripts only".
  - Exp 60's training is propose-only, so a Q-learner gets no action-outcome samples. Either it learns
    from the pain on the proposed action's value, or training executes (which breaks the yoked schedule).
  - Exploration is the real independent variable. `substrate_explore_bonus_weight` is pinned at 0.0 in
    Exp 60's frozen fingerprint; `tabular_q`'s exploration is declared in the prereg, not tuned after data.
  - Key on the live agent's cluster ids (first-touch allocation makes ids order-dependent).
  - **Records:** a separate campaign file and verdict id. `exp60_verdict`'s frozen `complete.sets`
    (`{fear, ablated}`) in `docs/experiments/evidence_pass_table.json` and T1-13's pass entry are untouched.
  - There is no offline drowning simulator (`minecraft_harness.py::FakeBridgeServer` has no water physics),
    so the arms are proven offline only for wiring; liveness preflights follow Exp 60's amendments 3–7.
- **B2. The bare loop and its guards.** Entry condition: the `start_simulation_mode` decomposition has
  landed (it is next after the agent loop; touching `orchestrator.py` before then collides with it).
  - **B2 takes no `runtime.*` or `agents.*` import.** Its first PR pulls the pieces the arms must share
    (percept intake, the `_llm_unavailable` fallback made inside `run_agentic_loop`, `<think>` stripping, a
    public form of `models/language/json_parser.py::_extract_json_object`) into modules with no bio imports,
    and makes the `runtime`/`agents` package `__init__`s lazy (§Guards).
  - **Dispatch.** `docs/agents/runtime-tools.md`'s invariant "One function executes an action the agent
    chose and books what it teaches" makes `tool_dispatch.execute_and_learn` the required path for any new
    dispatcher. Preferred: make `execute_and_learn` importable without bio modules (its bio imports
    type-only or moved to leaf modules) and call it with the learner handles absent (`nac`, `hippocampus`,
    `memory_hub` = None, each checked for None-tolerance), as `maxim_all_off` does. Fallback, only if the
    design review shows the shared path books learning with null handles: a dispatcher that lacks the
    learners by signature (the `book_machine_refusal` precedent), with the invariant amended in
    runtime-tools.md in the same PR and an M38 allowlist entry once M38 lands.
  - Three-lens code review; mypy over the touched composition files.
- **B3. `llm_notes`.** Notes stored in the agent's home with `atomic_write_json` + `_format_version`, one
  end-of-session reflection call through the router. Pick episodic retrieval or Reflexion first (open
  decision); the other can follow as a second arm.
- **B4. Exp 37 / T1-2 re-run with floors.** Four-lens design review (new claim). **Homed to 1.4 as L8's
  dose-response design (owner, 2026-10-14):** 1.3.2 released without it under the committed `release` exception
  `release-1.3.2-T1-2`, and T1-2 stays STALE until this run lands. Hold the model fixed and read where A sits
  (L8); a fire whose A leaves no headroom is reported as such, never scored against June's numbers.
  - Arms: A, B, C (L8 is BINDING: record A/B/C and gate on B − C) plus `bare_llm`, `llm_notes`,
    `maxim_all_off`. Non-resumed narrator, #845 recall held equal or ablated, seed plumbed to the sim.
  - The new arms go in a new arm set with a `SCHEMA_VERSION` bump: `scripts/analyze_exp37.py` imports
    `ARMS` and fails when an arm is short of `expected_trials`, and
    `tests/behavioral/test_exp37_harness_smoke.py` asserts `arms_seen == set(harness.ARMS)`, so the
    historical six-arm set stays analyzable.
  - Metrics: `positive_approach_engagement_fraction` is structurally 0 for both `sharp_rock` and
    `deceptive_fire`; the counter-prior signal is `warm_self_engagement_fraction`. Gate on `fire_pit` (and
    `deceptive_fire` via its own metric).
  - n well above 5: arm A's SD within a fire is 0.12–0.27, and its mean drifted 0.42 → 0.75 between fires
    (L8), which is why the gate is a within-fire difference.
  - `run_benchmark`'s arm loop is three hard-coded blocks; B4's frozen arm mapping replaces them.
- **B5. Later.** (a) An LLM arm in the survival world only where priors do not help: relabelled cues and
  action names, or R4's delayed credit. It needs a text renderer, an `llm_worker` path, a latency preflight
  and a box separate from the bridge's LLM server; it depends on B2. (b) One external long-horizon memory
  suite (LongMemEval or LoCoMo) running `bare_llm`, `llm_notes` and `full` through the same record, after
  B4.

B0 needs nothing else. B1 needs no LLM and no decomposition, but needs its seam decision. B2 → B3 → B4 are
serial.

## Side findings (each gets its own issue)

1. **`create_full_agent` ignores `remembers`/`learns` when `with_bio_stack=True`.** It sets
   `instance.hippocampus = bio.hippocampus` and `instance.nac = bio.nac` whatever the config says (the
   "Full bio-stack if remembers=True and learns=True" docstring is `create_npc_agent`'s). A silent no-op of
   the class CLAUDE.md says to push into the type. Blocks the `+memory` rung.
2. **Four `MAXIM_DISABLE_*` vars have no reader in `src/` or `tests/`:** `MAXIM_DISABLE_PFC_PREAMBLE`,
   `MAXIM_DISABLE_ACTING_COACH`, `MAXIM_DISABLE_SIM_SANDBOX_TEXT`, `MAXIM_NO_DEFAULT_PERSONA`. They are kept
   on purpose in `scripts/run_v1_phases.sh` for the Exp 12 reproduction
   ([archive/v1_refinement.md](archive/v1_refinement.md) §V2, its Disposition paragraph). Never map them to arms. Not an issue; recorded
   so nobody builds an arm on them.
3. **Exp 37's narrator is a cross-session memory channel.** T1-2's re-run must hold it fixed; the
   historical B−A numbers mix it in. A ledger annotation on T1-2, not a status change (T1-2 is already
   STALE).

## Risks

- **Baselines win on the LLM path.** Plausible given T1-2's history. Under BF2 that is a recorded boundary
  result, and it sharpens the story around the substrate-path results.
- **`tabular_q` matches Exp 60.** Equally plausible. The defensible claim then narrows to what a tabular
  learner on the same keys does not do: transfer between agents (Exp 61). Cross-pool transfer (Exp 62) is
  credited to the EC's keying, which both arms share, unless a second floor keyed on discretized raw sensors
  is run. Better learned from B1 than from a reviewer.
- **Scope creep.** The bare loop must not grow features. The layer rule, the `sys.modules` check and the
  prompt snapshot are the brake; no shared registry until a second consumer.
- **BF2 slows earning.** Intended. The LLM path cannot earn until B2 (B3 for cross-session claims); the substrate rungs carry two more
  arms each.
- **Rig time.** B1 needs the Minecraft server; B0, B2 and B3 do not.

## Open decisions (asked at each stage's start)

- B1: the seam (`src/` proposer hook vs scripts-only driver); how `tabular_q` learns under propose-only
  training; its exploration schedule.
- B3: episodic retrieval or Reflexion first.
- B4: bare prompt written by a different reader, or a published ReAct format verbatim; trials per arm (set
  in the prereg).
- B5: LongMemEval or LoCoMo.
- M46: the gate's inputs need an adversarial design pass before code (the gate-design rule).

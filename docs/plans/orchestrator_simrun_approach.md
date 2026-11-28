# SimRun: run state for the `start_simulation_mode` slices (approach note)

> **Status:** v3 2026-11-27 (orchestrator session, maxim-e6; 1.3.5 D9, line 4). No code yet.
> - **Round 2** (delta, a fresh reader across all three lenses) returned 0 DO-NOT-BUILD, 5 SHOULD-FIX and 3 NITs.
>   All are folded and tagged D-S1 to D-S5 and D-N1 to D-N3.
>   - **NEW class:** cross-phase function-local imports (D-S1) and module-attribute spies (D-S2). The orchestrator
>     re-measured the import table (§3a).
>   - D-S5 flipped decision 3's recommendation to the strict option.
>   - Not converged, because a new class appeared. A delta round 3 is owed on D-S1 to D-S5 only.
> - **Round 3** (delta, a fresh reader) returned 0 DO-NOT-BUILD, 3 SHOULD-FIX and 1 NIT, folded as R3-S1 to R3-S3
>   and R3-N.
>   - The one new class, R3-S1, came from the D-S4 fold. A pre-move mypy edit would break the free swallow credit.
>     Its fix deletes that plan.
>   - R3-S2 and R3-S3 are completeness fixes in already-covered classes.
>   - The reviewer judged that a docs-only check of R3-S1's deletion closes the round. No further full round is
>     owed. Design CONVERGED.
> - **Round 1** (three lenses: Executor, Architecture, Wire integrity) returned 2 DO-NOT-BUILD, 15 SHOULD-FIX and
>   10 NITs. All are folded and tagged below:
>   - E (Executor): DNB, S1–S6, N1–N4;
>   - A (Architecture): DNB1–2, S3–S6, N;
>   - W (Wire integrity): S1–S6, N1–N5.
> - **Cross-confirmed folds:**
>   - unpack into the original local names (E-DNB, A-DNB2, W-N1);
>   - the executor before the fear gate is a bypass (E-S1, A-S3, W-S2);
>   - the provenance-order pins (E-S5, A-S5, W-S5);
>   - `new_run_settings` stays out of `SimRun` (E-S2, A-S5, W-N4).
> - **A-DNB1 reversed v1's recommendation:** strict rule (d) costs about 55 lines, not the budget, so it is now the
>   recommendation.
> - A delta round on v2 is owed before the owner sees the plan.
> - The owner decisions are in §7 and are asked together (development-flow rule 3).

## 1. The problem, measured

Measured on origin/main at `70f0fa88` with an AST pass over the function. The pass reads every `Name` store and
load in the body, and counts a nested def's free names as reads at the def's phase.

- `start_simulation_mode` spans **1,816** lines (the baseline pin) and has 27 parameters and 6 nested defs.
  **816 lines** must go for `1323.functions-1000`.
- **147 assigned locals**, plus 92 function-local import names and 1 `except ... as` name. The kickoff's 155
  counted them differently.
- **68 of the 147 cross a phase boundary**: one phase writes them and a later phase reads them. That is why the
  slices so far have been small.

The phases, in source order, with line numbers at `70f0fa88`. "Reads in" counts the names each phase reads from
earlier phases or from parameters (imports excluded; A-DNB1).

| # | Phase | Lines | Span | Reads in |
|---|---|---|---|---|
| P0- | prologue: the prompt-handler refusal, the `sim_debug` merge, `_carried_settings(locals())`, the function-local imports. **Stays in the function** (E-S2) | 1083–1105 | 23 | — |
| P0 | setup: the stop-reset, start stamp, adopted agent, session id, stop event, routers, bridge, percept source, sandbox and pain bus | 1106–1204 | 99 | params |
| P1 | build the AUT | 1205–1718 | 514 | 17 |
| P2a | build the narrator: memory, registry, research tools, LLM worker | 1719–1812 | 94 | 19 |
| P2b | late AUT wiring: PainDetector, DefaultNetwork, the narrator's executor and instance, the pain layers, the fear gate, the imagination trigger, the scene manifest | 1813–2158 | 346 | ~18 |
| P3 | banner, telemetry, `_aut_worker`, the AUT thread | 2159–2307 | 149 | |
| P4 | §9 campaign dispatch | 2308–2440 | 133 | |
| P5 | goal or resume injection | 2441–2464 | 24 | |
| P6 | §10 stdin | 2465–2671 | 207 | |
| P7 | §11 stall detector and the orchestrator loop | 2672–2758 | 87 | |
| P8–9 | the `SimTail` hand-off and the post-sim prompt | 2759–2816 | 58 | |

**P2b is its own phase.** The roadmap's "`build_orchestrator`" slice is mostly not the narrator. P2b writes the
AUT's pain detector, its DefaultNetwork, the pain- and fear-wrapped `aut_executor`, and the imagination trigger.
The narrator's executor is built between the AUT's DefaultNetwork and the pain-layer `try` (lines 1859–1866;
E-N2). Pulling it forward into P2a would reorder side effects. Order alone would also allow one builder for P2a and
P2b together (E-N2). This note keeps them apart because P2a and P2b read different inputs and own different
things.

**Arithmetic.** P0 + P1 + P2a + P2b is 1,053 lines. Under strict rule (d), D-N2 re-measured each call site with its
kwargs in and its unpack out:
- P0: about 31 lines;
- P1: about 40;
- P2a: about 33;
- P2b: about 27.

That is about 130 lines, which puts the function at about 895 lines. Every extracted piece must also be ≤1,000 lines (`split_from`).

## 2. The front-gate question: why not grow `SimTail`, or reuse `LoopRun`?

`SimTail` (`simulation/sim_teardown.py`) cannot carry this, for three reasons:
1. **It is assembled last.** Its 42 fields include results that exist only after the loop (`orch_error`,
   `fixture_result`, `dm_rollup`). Built at P0, half of them would be `None` for 1,700 lines and assigned along the
   way, the shared mutable state this decomposition removes.
2. **It is mutable by contract.** `end_run` writes `tail.stop` once.
3. **It is the tail's contract.** Growing it widens what the tail can reach for no reader's benefit.

`LoopRun` has the right shape (frozen, built once by one builder) and the wrong content. **So: new frozen
dataclasses shaped like `LoopRun`. `SimTail` and `sim_teardown.py` stay unchanged.** The 42-line `SimTail(...)` call
moves into a helper whose every parameter is **required and keyword-only**, the run-time results included (W-S6).
`SimTail.stdin_thread` defaults to `None`, so a helper with defaults would let a forgotten `stdin_thread` drop
#1541's join silently.

## 3. The shape: staged frozen bundles, unpacked into the original names

**Each builder returns a frozen bundle, and the caller unpacks it into the locals' existing names**, as loop slice
1 did with `LoopRun` (the baseline's history: "unpacked into the loop's existing local names").

```python
run = build_sim_run(debug=debug, ...)          # P0, ~99 lines leave
session_id, stop_event, bridge = run.session_id, run.stop_event, run.bridge   # ...one line per field
```

The unpack is what keeps each move verbatim (E-DNB, A-DNB2, W-N1):
- the moved text reads the same names, so its AST is unchanged;
- the golden and the text pins in `_orchestrator_source.py`, `test_persistent_agent_campaign.py` and the rest keep
  their strings;
- `lint_no_silent_swallows` credits the moves for free, because each enclosing `try` dumps identically. Without
  the unpack, rewriting `aut_component_registry` to `aut.component_registry` changes the key of the swallow at
  line 2005 and needs a `swallow_moves.json` record (W-table);
- rule T is met. Rewriting about 110 later reads as `run.x` would charge about 115 touched lines against about 118
  of shrink, a margin one ruff re-wrap could overturn (A-DNB2, E-DNB). The unpack touches about 17 lines;
- the diff stays inside the moved phase, which keeps rebases onto D7's behaviour branches cheap.

Under strict rule (d), a builder's keyword parameters carry the original local names, so the moved text needs no
prologue (D-N1).

**The one exception to "unpack into the original name"** (D-S3): the fear-gated executor and the cold router. If the
pattern were applied mechanically, it would write `aut_executor = aut.aut_executor_unwrapped` and reopen the
bypass. So:
- `aut_executor_unwrapped` is never unpacked. It is passed straight into the `wire_aut_late` call, whose parameter
  is named `aut_executor`, because the moved P2b text reads and re-binds that name.
- `aut_router_cold` is treated the same way for `build_aut`.

`test_sim_run.py::test_ungated_handles_flow_only_into_their_builder` checks this in `start_simulation_mode`'s AST:
- `Attribute(attr="aut_executor_unwrapped")` occurs exactly once, as the `aut_executor=` keyword value of the
  `wire_aut_late` call;
- `aut_router_cold` occurs exactly once, as the `aut_router=` keyword of the `build_aut` call;
- no `Name` store of `aut_executor` comes before the unpack of `wire_aut_late`'s result.

No field name appears in two bundles. The same test asserts that over the four dataclasses.

### 3a. Function-local imports (D-S1, measured)

The body has 92 function-local import names. These 24 are bound in one phase and read in a later one:

| Bound in | Name | Read in |
|---|---|---|
| P0- | `AutonomyController`, `AutonomyLevel`, `SupervisionPolicy`, `LLMWorker`, `build_memory`, `build_decision_engine` | P1, P2a |
| P0- | `build_tool_registry` | P1 |
| P0- | `MaximAgent`, `Observer`, `InspectAUTTool`, `SpawnSubSimulationTool` | P2a |
| P0- | `DamageComponentTool`, `SetEntitySensorTool` | P2b |
| P0- | `SimulationBridge`, `ConversationalSource` | P0 |
| P0- | `run_agentic_loop` | P3, P7 (stays) |
| P0 | `sim_agent_context` | P3, P7 |
| P1 | `FileSystemEnv`, `RuntimeState` | P2a |
| P1 | `AgentFactory` | P2b |
| P3 | `display_status`, `display_summary`, `get_active_display`, `_emit` | P5, P6, P7 (P3 onward; not moved by these steps) |

The rules:
- **Each builder re-imports lazily, inside its own body, every name it reads.** It never imports at module level.
- **The function keeps only what its remaining code reads.** Of the P0- imports, that is `run_agentic_loop`.
  `sim_agent_context` is bound in P0, so the function needs its own import of it once P0 moves.
- **Each step removes the P0- imports that its move leaves unused.** Removed lines shrink the function, and the
  added `sim_agent_context` line counts as touched for rule T.

**Spies on module attributes** (D-S2, which generalises E-S6): **a builder imports lazily every name that a test
patches on that name's source module.** Today that covers:
- `bridge.SimulationBridge`: `test_bounded_stop_sim_1438` patches `bridge_mod.SimulationBridge`, and a module-level
  import in `sim_run.py` would leave that guard passing dead;
- `bootstrap.build_tool_registry`: `test_api_expansion` patches the string `"maxim.runtime.bootstrap.build_tool_registry"`
  and drives the real function to P1's registry build. A module-level import in `sim_aut_build.py` would leave that
  patch dead (R3-S3).
- the display and report-writer spies in `_orch_harness.py`.

Each step's PR searches `tests/` for every patch form on the source module of each import it moves:
- `setattr(<alias>, "<Name>", …)` and `patch.object(<alias>, "<Name>")`;
- the `(alias, "<Name>")` tuple lists that `_orch_harness.py` and `test_tty_stdin.py` loop over;
- the dotted strings `"pkg.mod.Name"` in `patch(...)` and `monkeypatch.setattr(...)` (R3-S3).

| Bundle | Built by | Fields (the cross-phase locals it writes) |
|---|---|---|
| `SimRun` | `build_sim_run(*, debug, ...)` (P0) | `session_id`, `stop_event`, `operator_stop`, `bridge`, `orchestrator_source`, `llm_router`, `aut_router_cold`, `aut_pain_bus`, `sim_tmpdir`, `sim_sandbox`, `sim_workspace`, `sandbox_root`, `start_time`, `start_provenance`, `adopted`, `aut_agent_id` |
| `SimAut` | `build_aut(...)` (P1) | the AUT's agent, state, env, memory, memory hub, hippocampus, NAc, registry, LLM worker, autonomy, decision engine, energy registry, component registry, entity map, thought gate, bio-enrichment pipeline, instance, `_sim_prompt_handler` (read in P4, P6 and P7; E-S4), `resume_record`, `sandbox_dirs`, `aut_router`, and **`aut_executor_unwrapped`** |
| `SimOrch` | `build_orchestrator(...)` (P2a) | the narrator's agent, state, env, memory, registry, LLM worker, autonomy, decision engine, `is_observe_only`, `experiment_log`, `aut_introspector`, `_aut_embodiment` (read in P2b; E-S4) |
| `SimAutLate` | `wire_aut_late(...)` (P2b) | `aut_default_network`, **`aut_executor`** (pain- and fear-wrapped), `aut_imagination_trigger`, `orch_executor`, `orch_instance` |

Notes on the table:
- **No field name appears in two bundles** (A-S3). The fear-gate bypass three lenses found is closed by naming
  (E-S1, A-S3, W-S2). The executor before the pain layers and the fear gate is `SimAut.aut_executor_unwrapped`,
  and only `wire_aut_late` may read it. The gated one is `SimAutLate.aut_executor`, which §9 (line 2379),
  `_aut_worker` (2240) and the `SimTail` helper (2796) read. `aut_router` re-binds through `_warm_aut_router`,
  which returns its argument, so the object is the same (E-verified). It still gets distinct names:
  `SimRun.aut_router_cold` is readable only by `build_aut`, and `SimAut.aut_router` by everything after.
- **Re-bound parameters** are returned by the builder that re-binds them and unpacked in the caller under the same
  name: `debug` (P0-, before any builder; E-N1) and `experiment_log` (P2a).
- **Stays out of every bundle:**
  - `new_run_settings`. The function keeps `_carried_settings(locals())` in P0-, and no builder calls `locals()`
    (E-S2, A-S5, W-N4).
  - The function-local imports that later phases use (`LLMWorker`, `AutonomyController`, `run_agentic_loop`, …).
    They stay in P0-.
  - The run-time results: `aut_thread`, `stdin_thread`, `_dm_thread`, `_dm_error`, `dm_rollup` (the `nonlocal`
    in `_dm_worker`; the §9 slice must return it and `_dm_error` explicitly, E-N4), `orch_error` and the campaign
    results.
  - The 79 phase-local names.
- **Frozen** like `LoopRun`. The objects they hold stay mutable, and a handle cannot be re-bound.
- **CC3 does not apply**: never persisted, never on a wire. The docstrings say so, as `SimTail`'s does.
- **Modules** (A-N):
  - `simulation/sim_run.py` holds the four dataclasses, `build_sim_run` and the `SimTail` assembly helper;
  - `simulation/sim_aut_build.py` holds `build_aut` and `wire_aut_late`;
  - `simulation/sim_orch_build.py` holds `build_orchestrator`. `narrator.py` already means narrative generation,
    so the name follows the `orch_*` locals.
- **Logger:** the new modules keep `logging.getLogger("maxim.simulation.orchestrator")`, as `stall_detector.py`
  does (W-N2). Three tests set caplog levels on that name. No production consumer matches on it.
- **Lazy imports stay lazy** (E-S6). `_orch_harness.py` spies on `display_status`, `display_summary` and the
  report writers as module attributes, so a builder must not hoist those imports to module level.

## 4. Rule (d)

Rule (d) (roadmap_1_3_x.md §The decomposition) is: "Extracted functions take individual fields (or `ctrl`) as
explicit keyword arguments, never the whole `LoopRun`." It lets **a reader of the call site see what a function
depends on.**

v1 claimed `wire_aut_late` would need about 30 kwargs. Measured, it needs about 18–20. P1 needs 17 and P2a 19.
Strict rule (d) costs about 55 more call-site lines than (d′) across the three builders (A-DNB1). The total
call-site cost of all four steps, with kwargs in and unpacks out, is about 130 lines, which leaves the function at
about 895 lines (D-N2). **Recommended: rule (d) binds the orchestrator builders, as written.**
- Builders take individual fields as keyword-only arguments and return a bundle, which the caller unpacks.
- The bundles are return types only. No function takes one as a parameter, except the `SimTail` assembly helper,
  which takes fields too.

**The alternative (d′), not recommended:** builders take the earlier bundles whole and declare a per-builder
`READS` (keyed by builder, not by module, A-DNB1). It saves about 55 lines. It loses half of what rule (d) protects,
because the call site no longer shows the dependencies. If it were chosen, the check would fail on:
- any load of a bundle parameter that is not the `.value` of an `Attribute`;
- any store to a bundle parameter;
- any bundle passed as a call argument (W-N1);
- any declared field that is never read.

It would walk nested defs and lambdas, and would come to about 60 lines (A-DNB1).

## 5. Import direction, patch seams and pins

- **No re-exports, and no extracted module imports `orchestrator.py`, transitively** (A-S6). The test walks the
  import closure like `test_loop_setup.py::loop_import_closure`, because `curriculum_runner.py` lazily imports
  `orchestrator` and a direct-only check would miss it.
- **Helpers move with their only caller** (rule (a)). Eighteen module-level helpers are called only from P0–P2b,
  for example `_build_run_llms`, `_setup_sim_sandbox`, `_restore_aut_from_session`, `_fear_gate_aut_executor`,
  `_aut_agent_config` and `_register_research_tools`. Two of the eighteen came with #1838 (the #863 fixes, §6):
  `_trace_sem_tools` (P1) and `_sem_trace` (P1 and P2b). Both go to `sim_aut_build.py`. If an emit that stays in
  `orchestrator.py` (the scene-manifest SEM_TRACE lines) is ever routed through `_sem_trace`, the helper still lives in
  `sim_aut_build.py`, and `orchestrator.py` imports it from there (the allowed direction).
- **v1's "tests that patch `orchestrator.<x>` keep working" was false** (W-S1, E-S3). With no re-export, a patch
  fails loudly. With one, it silently stops intercepting. For example, `test_bounded_stop_sim_1438`'s guard
  "no provenance capture under a stop" would pass dead. **Each step's PR re-points the patch targets of the
  helpers it moves.** The inventory at step 1 is:
  - `test_bounded_stop_sim_1438` (`capture_start_provenance`);
  - `test_agent_router_slots_1228` and `test_offline_startup_10a` (`_build_shared_router`, `_build_aut_router`,
    `_build_run_llms`; drop 10a's existing no-op `build_primary_router` patch, W-N5);
  - `test_offline_call_sites_10b`;
  - `test_harness_identity_1003` and `test_new_slice_1453` import `_restore_aut_from_session` (step 2);
  - `test_tty_stdin` and `_orch_harness.py` patch `_select_aut_prompt_handler` through `(orch, "<name>")` tuples. Its
    only caller is in P1, so it moves at step 2 (R3-S2);
  - `scripts/o19_rerun.py` cites it.

  A new test, `test_sim_run.py::test_orchestrator_patch_targets_are_read`, checks that every
  `setattr(orchestrator, "<n>")` or `"maxim.simulation.orchestrator.<n>"` target in `tests/` is **read** in
  `orchestrator.py`: an `ast.Name` load outside its own import, not just a definition. It resolves the module's aliases
  (`orch`, `orch_mod`, `orchestrator`) from each test's import statements. It matches `setattr` and `patch.object`
  calls, `(alias, "<str>")` pairs and dotted-string targets (R3-S2). Otherwise a kept import of
  the module-level `capture_start_provenance` would leave the patch dead (D-N3).
- **The `_orchestrator_source.py` registry** (W-S5):
  - run code goes in `PARTS`;
  - moved helpers go in `MOVED_MODULE_HELPERS`, never `PARTS`, because a helper whose body repeats call-site
    strings would make the string pins vacuous;
  - every new module joins `EXTRACTION_MODULES`;
  - the `PARTS` entries already filed under `orchestrator.py` (`_build_run_llms`, `_warm_aut_router`,
    `_build_*_router`, `_register_narrator_sim_tools`) are re-pointed.
- **Pins that move with P0 at step 1, and are re-pointed on purpose:**
  - the provenance order, `capture_start_provenance` before `_build_run_llms` (`test_report_provenance_m1.py`,
    `test_config_sparse_1_2.py`). Both calls go into `build_sim_run`, and the pin follows them there.
  - the other function-specific pins in `test_narrator_reliability` and `test_persistent_agent_campaign`. Each
    PR lists which of them it touches.
- **The `locals()` trap:** P0- keeps `_carried_settings(locals())` ahead of any binding. The golden's `new_goal`
  arm pins what it carries.
- **Pre-check done:** the stall arm's re-homing on the §11 clock landed (#1259 closed 2026-11-01).
  `_orch_harness.py` drives `StallDetector.CLOCK`.

## 6. The gates, per PR

- **Golden:** `test_orchestrator_characterization_golden.py` stays byte-identical, and the test file stays
  unchanged.
- **Characterization first:** characterization tests for the moved code land in their own commit first.
- **Function length:**
  - Lower `start_simulation_mode`'s pin.
  - `build_aut` (514) and `wire_aut_late` (346) are new pins over 200 lines. Each needs a `split_from` exception
    **and an `accepted_new_debt` record** under `lint_release_gates` rule 4 (A-S4, W-S4). That record class is
    self-approved, so it is an owner decision (§7.3).
  - From then on, rule T charges every edit to them.
- **Coverage:**
  - Each new module with 25 or more statements gets its own new floor in the same PR, under the split rule
    (`lint_coverage.py`). The SUM rule has no tolerance.
  - `orchestrator.py`'s floor (76.8%, 241 missing) is re-pinned from CI's run.
  - The 80% changed-line gate counts every moved line, so the characterization commit must take each moved block
    to 80% (A-S4).
- **mypy:**
  - The new modules start at zero errors on the ratchet.
  - Three of `orchestrator.py`'s seven current errors come from `_spec`: P1 binds it as a `ModuleSpec` at line 1691,
    and P2b uses it as a dict at lines 2006–2007. **No edit fixes them.** Step 2 puts P1 and P2b's `_spec` in
    different functions, and the errors go away by scope. Step 2 re-pins `orchestrator.py`'s mypy ratchet count
    from 7 to 4.
  - An edit at lines 2006–2007 would sit inside the swallow-keyed `try` at line 2005. That would end the move's
    free credit, so v3's pre-move fix is withdrawn (R3-S1).
  - Joining `run_mypy_public.sh` requires zero errors, so the new modules join it if they reach zero.
- **The #863 sites** (W-S3): `release_gates.json`'s three DROP-class sites are at lines 1638 (P1), 2063 and 2071
  (P2b), under the immutable `site` `orchestrator.py::start_simulation_mode`. Nothing checks that they still
  resolve. **Recommended: land the three #863 step-2 fixes before the builder moves.** They are behaviour-class,
  which fits D7 and "filing is not fixing".
  - They are not all small (D-S4). Sites 2063 and 2071 are 6-line drops.
  - Site 1638's `try` wraps the joins and an entity and modulator loop (lines 1638–1658). Dropping it re-indents
    about 20 lines, and rule T charges re-indented lines.
  - Every PR inside the pinned function must shrink it by the lines it touches. So **each #863 fix extracts its
    SEM_TRACE block into a module-level helper in the same edit**, and pays for itself. A `touch_exemptions`
    record is not used.
  - **Landed in #1838** (2026-11-27). The tool listing is `_trace_sem_tools`, and every emit goes through
    `_sem_trace`. The ACTIVE line sits in the construction `try`'s `else:`, so a fault there raises and is not
    reported as a construction failure. The same PR drops the six other such wraps in `simulation/` and holds the
    package at zero (`tests/unit/test_sim_trace_unwrapped_863.py`). The step-2 move still records where each site
    went, since the gate's `site` stays `orchestrator.py::start_simulation_mode`.
- **Setup-abort teardown** (#1838 review; owner decision 2026-11-27). A raise anywhere between the workspace opening
  and the main `try` leaks the open sim log and the threads setup started (both `LLMWorker`s' lanes, plan loggers
  and concept extractors: 10 threads measured). Nothing tears them down, because `end_run` is reached only from the
  main `try`. This predates the slices. The slices fix it in the seam they create: **the caller of the staged
  builders tears down what the earlier bundles built when a later builder raises** (the sim log, then each built
  bundle's workers), through the same teardown `end_run` uses, and re-raises. Each builder PR adds a test of
  that path. Tracked in #1839.
- **World-clock census** (W-N3): `world_clock_census.py` fences `simulation/orchestrator.py`. The new modules get
  the same `_FENCED` rule in the PR that creates them, and the line-2 lane hears about it, since M72 is "absolute
  from birth".
- **Not applicable:** the layer lint, because `simulation` is not in `audit.LAYER_RULES`. The orphan lint passes,
  since `orchestrator.py` imports the new modules (A-N).
- **Review:** three lenses before each PR opens, and a delta round after any `src/` fold. D7 puts verbatim moves
  last.

## 7. Owner decisions (asked together; the strict option first)

> **DECIDED 2026-11-27 (owner), every recommendation taken:**
> 1. strict rule (d);
> 2. staged frozen bundles, unpacked into the original names;
> 3. split `build_aut` and `wire_aut_late` into sub-builders of 200 lines or fewer, in the same PR, with no
>    `accepted_new_debt`;
> 4. re-scope the last slice into `build_orchestrator` + `wire_aut_late`;
> 5. roadmap order, the #863 fixes first, and `SimRun` merging last under D7.
>
> **ADDED 2026-11-27 (owner, the #1838 review):** 6. the setup-abort teardown (§6) is built into the staged-builder
> caller in the slices, not as a separate fix before them.
>
> The text below is the question as it was asked.

1. **Rule (d) for the orchestrator builders.** **(a) Strict rule (d), recommended:** builders take explicit
   keyword arguments. The call sites cost about 130 lines in total, about 55 more than (d′), and the function ends near
   895 lines. (b) Rule (d′) (§4): bundles whole,
   with an AST-enforced `READS`.
2. **Bundle shape.** **(a) Staged frozen bundles, each unpacked into the original names (§3), recommended.**
   (b) One mutable `SimRun` filled phase by phase, which is `SimTail`'s shape.
3. **How the two new large functions are paid for.** **(a) Strict, recommended (D-S5):** split `build_aut` (514) into
   three or four sub-builders and `wire_aut_late` (346) into three, each 200 lines or fewer. R3-N measured the
   longest top-level statements: an `if` of 100 lines in P1 and one of 150 lines in P2b. P2b cannot be cut into two
   pieces of 200 or fewer, and the nested defs have no late-bound free names. The sub-builders land
   in the same PR as their parent, so there are no extra PRs.
   - The cost is call-site and unpack lines inside the parent builders, roughly 90 to 120 more lines (R3-N). None of them is
     in `start_simulation_mode`.
   - What it buys: no `accepted_new_debt` record (a self-approved exception class), and rule T never follows the
     new code.

   (b) Accept two `accepted_new_debt` records (owner-approved, `split_from` `start_simulation_mode`) and sub-split
   later, when a fix touches them. Both functions are within the 1,000-line gate. (v2 recommended (b) on a claim
   that (a) doubles the PR count. That claim was wrong, and it is withdrawn.)
4. **Re-scope the last slice** into `build_orchestrator` (P2a, 94) and `wire_aut_late` (P2b, 346), in source order.
   **Recommended: yes.** The roadmap and release_1_3_5 text change in the PR that lands this note.
5. **Order and D7.** **Recommended:**
   - **Order:** keep the roadmap order for §10 and §9, with their riders: #1262/#1451, and the §9 ordering arm.
     The three #863 fixes land before the builder moves (§6).
   - **Step 1 under D7:** `SimRun` (step 1) is a verbatim move, so under D7 it merges after any 1.3.5 behaviour fix
     that touches `orchestrator.py`. It is built and reviewed now, then rebased. The unpack keeps that rebase
     cheap. Accept that rather than an exception to D7.

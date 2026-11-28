# `run_loop_stack()`: one sanctioned caller of the agent loop (approach note, v4)

> **Status 2026-11-27: APPROACH NOTE v4, before code.** Homed in 1.3.6 (`roadmap_1_3_x.md` §1.3.6, "The loop, phase
> 2"). The 2026-07-29 note was deferred until a ninth hand-assembled site landed. That trigger fired on 2026-09-04 with
> `simulation/minecraft_harness.py` (#610). The routing guard is a GATE, so this note takes an adversarial design pass
> before any code (CLAUDE.md, "Faster without lowering the bar" (1)). **v2 folds design round 1** (architecture and
> wiring, plus an adversarial gate pass). **v3 folds design delta 1, v4 design delta 2.** The change lists are at the end.

## Owner decisions, 2026-11-27 (verbatim)

1. The first PR is the builder plus the console talk site (`src/maxim/console/handle.py`) only. api/cli come next, and
   the orchestrator AUT comes last (after 1.3.5).
2. The routing-guard lint lands in the first PR as a ratchet, with the 8 unmigrated sites allow-listed. Each
   migration removes one, and a new hand-assembled site fails CI.
3. The optional subsystems talk leaves out on purpose (`default_network`, `imagination_trigger`, the embodied acting
   coach / `is_embodied`, and any others found) are REQUIRED keywords with no defaults.

## The finding (re-counted on main 2026-11-27, from each call's AST and the code before it)

There are 9 `run_agentic_loop(` sites in 8 modules. Each hand-assembles its stack. A component a site forgets does not
raise; that entry point just becomes less capable than its siblings (the `build_executor(pain_bus=...)` lesson).

| site (file::qualname) | AgentInstance | hub wired (where) | enrich · gate | hippo · hub · pain | DN | imagin. | coach · embodied | percept · sink · policy | evaluators | state_home | protocol_reg | aut_mode | sim_adapter | run_id |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `console/handle.py::MaximHandle._launch_talk_loop.<locals>._worker` | ✓ | before the thread, if hub | ✓ ✓ | ✓ ✓ ✓ | — | — | — · — | bridge ✓ ✓ ✓ | — | — | — | default | — | — |
| `api.py::run` | only if learning (`_build_run_executor`) | before, if hub | **— —** | ✓ ✓ ✓ (None without instance) | — | — | — · — | — | ✓ | ✓ | — | default | — | — |
| `cli.py::_main_impl` | ✓ | before, if hub (`cli.py:2852`) | ✓ ✓ | ✓ ✓ ✓ (a sim's `_sim_pain_bus` overrides, `cli.py:3001`) | — | — | ✓ ✓ only when `embodiment_ref` (`cli.py:318`) | sim ✓ ✓ None | ✓ | sim tmpdir | — | default | — | ✓ |
| `embodied_runtime/agentic_runtime.py::AgenticRuntimeMixin._start_agentic_runtime.<locals>._worker` | **none** (own `build_bio_stack`) | at bio build (`:473`) | **— —** | ✓ ✓ **—** | ✓ | — | — · ✓ when the executor has a body (`:928`) | — | ✓ | — | ✓ | ✓ | ✓ (DoA) | ✓ |
| `simulation/interactive.py::_run_scenario_turn` | caller's | caller's | via `**kwargs` | via `**kwargs` | ? | ? | ? | ✓ ✓ None | ? | ✓ | ? | ? | — | ✓ |
| `simulation/minecraft_harness.py::run_minecraft_aut` | no (`aut.bio`) | before (`:418`) | via `_loop_kwargs` | ✓ ✓ ✓ | — | — | — · — | world ✓ — None | — | — | — | substrate-primary | — | — |
| `simulation/orchestrator.py::start_simulation_mode.<locals>._aut_worker` | ✓ | before (`:1395`) | ✓ ✓ | ✓ ✓ ✓ | ✓ | ✓ | ✓ ✓ when entity (`:1708`) | bridge ✓ ✓ ✓ | — | ✓ | — | ✓ | — | — |
| `simulation/orchestrator.py::start_simulation_mode` (the orchestrator's own loop) | — | — | — — | — — — | — | — | — · — | own source · — · None | — | ✓ | — | default | — | — |
| `simulation/tools.py::SpawnSubSimulationTool._run_sub_simulation.<locals>._sub_aut` | — | — | **— —** | **— — —** | — | — | — · — | bridge ✓ ✓ ✓ | — | ✓ | — | default | — | — |

Talk passes no `run_id`, `state_home` or `evaluators`, and the migration must not give it any.

What changed since the 2026-07 note:
- **More sites have talk's old gap.** `api.run` and the embodied runtime pass neither the enrichment pipeline nor the
  thought gate, which is the gap PR #438 fixed in talk. The embodied runtime passes no pain bus. The sub-simulation AUT
  has no hippocampus, hub or pain bus.
- **Two sites already half-build their arguments.** `minecraft_harness._loop_kwargs` returns a keyword dict, and
  `interactive._run_scenario_turn` forwards `**kwargs`.
- **The orchestrator's own loop is not an agent stack.**
- **The acting coach is set on the worker, not passed to the loop**, and it is set unevenly:
  - cli sets it only when `embodiment_ref` is not None;
  - the embodied runtime sets `is_embodied` without the coach.
- **No site wires the hub at the call.** Every hub is wired well before the call, on the launching thread.
- **The old loop has no caller.** `run_agent_loop` (the 230-line loop) has no `src/` caller, but it is public API:
  `maxim.runtime.__all__`, through `runtime/__init__.py`'s `_LAZY` `__getattr__`.

## Why one function, and not a type or an existing builder (front-gate rule)

- **`loop_setup.build_loop_run`** is the per-run setup inside `run_agentic_loop`. It cannot see what a caller left
  out, because a missing `thought_gate` arrives as `None`, which is legal. The new function sits before the call;
  `build_loop_run` stays inside it. Neither builds what the other does.
- **`AgentFactory` / `AgentInstance` / `build_bio_stack`** build subsystems. Not every site has an instance: api
  without learning has none, and the embodied runtime builds its own bio-stack. So an instance cannot be the
  parameter, and each site states its subsystems.
- **Why a function, not a frozen `LoopStack` type (design round 1, A):** one keyword-only function gives the
  required-keyword `TypeError` and the one sanctioned caller with no new type.
  - It removes by construction the bypass a type invites: a hand-built `LoopStack`, or `dataclasses.replace` on one.
  - `run_agentic_loop` is imported lazily inside it for import cost. There is no cycle: `agent_loop` does not import
    `bootstrap`. v1's stated reason was wrong.

## Shape

`src/maxim/runtime/bootstrap.py::run_loop_stack(*, ...) -> None`. It is keyword-only and explicit, with no `**`.

A module constant `LOOP_PARAMETERS: dict[str, Literal["stack", "run"]]` classifies every one of `run_agentic_loop`'s
39 parameters. A unit test asserts that its keys equal `inspect.signature(run_agentic_loop).parameters` exactly, so a
new loop parameter fails CI until someone classifies it.

**STACK: required, no default (27).** A forgotten one is a `TypeError`. These are the entry point's capability and
identity, the things a site can silently lack:

| parameter | why STACK |
|---|---|
| `agent`, `environment`, `state`, `memory`, `decision_engine`, `executor` | the loop's six positionals; every site builds its own |
| `autonomy_controller` | the permission posture; the loop's default is PLANNING, which silently executes no body affordance (`minecraft_harness._loop_kwargs` comment, Exp 60) |
| `llm_worker` | `None` makes a substrate-only loop |
| `hippocampus`, `memory_hub`, `pain_bus` | the memory session and pain learning; `None` silently loses both |
| `bio_enrichment_pipeline`, `thought_gate` | `None` gates off the whole enrichment block and multi-cycle deliberation (the talk case) |
| `default_network`, `imagination_trigger` | owner decision 3 |
| `evaluators` | outcome scoring; api/cli/embodied pass it, and talk does not |
| `protocol_registry` | dynamic skills; only the embodied runtime passes it |
| `sim_adapter` | the DoA side channel; only the embodied runtime passes it |
| `aut_mode` | llm- or substrate-primary: the agent's decision path |
| `substrate_telemetry`, `substrate_policy`, `substrate_action_gate` | the substrate-primary branch's sight, policy and budget (Exp 48/60 fixes) |
| `percept_source`, `action_sink`, `response_policy` | sim identity: `percept_source` flips `is_sim_mode`, and `response_policy` is already required with it (`loop_setup`) |
| `state_home` | where the state file persists (#1264); a wrong default writes into a shared home |
| `consolidation` | the session-end flavour; a forgotten `"full"` on a persistent agent silently gets the sim's lightweight end (owner, Q4, 2026-11-27) |

**RUN: explicit keywords that keep the loop's defaults (12).** These are run configuration, not capability:

| parameter | why RUN |
|---|---|
| `max_steps` | how long to run |
| `run_id` | labels the run |
| `stop_event` | how the run stops |
| `on_step`, `on_event` | observation callbacks |
| `idle_sleep_s`, `target_hz`, `persist_every_n_steps` | pacing |
| `context_pool_config`, `use_tool_prompting` | prompt tuning; no site passes them |
| `planning_liveness` | an opt-in with a documented safe default; only the orchestrator's own loop opts in |
| `end_bound` | a wrapper-only stop bound (#1536) |

The explicit signature also stops a call site from overriding a stack keyword, so v1's custom override `TypeError`
is gone.

**What the function does (design round 1, C and D; design delta 1, 1, 2 and 10).** It never constructs a
`MaximAgent` and never mutates the worker. Each site keeps producing its own agent and worker configuration, on the
thread and at the time it does today. The function checks three things and raises `ValueError` on a mismatch:

- **The agent's hub, both ways.** `agent.memory_hub is memory_hub` must hold always; `None`/`None` passes.
  - This reads the public read-only `MaximAgent.memory_hub` property this PR adds (owner, Q5, 2026-11-27).
  - So a hub passed to the loop but never wired raises, and so does a wired agent handed `memory_hub=None`.
  - Round 1's C asked the function to call `wire_memory_hub` only where a site does that at the call. No site does:
    every site wires it before the call, on the launching thread. So the function checks the wiring instead of doing
    it, which keeps timing and failure surface unchanged.
- **The coach.** `acting_coach=` (required) must be `llm_worker.acting_coach`, compared with `is`.
- **Embodiment.** `is_embodied=` (required) must equal `llm_worker.is_embodied`.

**`llm_worker=None`.** The embodied runtime passes `None` when the LLM is disabled or the worker build failed
(`agentic_runtime.py`, `llm_worker = None` before the `llm_router.enabled()` branch). Then the call must also say
`acting_coach=None` and `is_embodied=False`. Anything else raises, because there is no worker to read.

**Fake workers in tests.** The function reads `llm_worker.acting_coach` and `llm_worker.is_embodied` plainly, with no
`getattr` default.
- A `SimpleNamespace` without them raises `AttributeError`, which is loud.
- A bare `MagicMock` returns a child `Mock` for both, so the `is` check and the `==` check both fail and the call
  raises.
- Tests therefore build a fake worker with both attributes set to `LLMWorker`'s defaults, which are `acting_coach =
  None` and `is_embodied = False` (`llm_worker.py`, in `__init__`). Use either:
  - `MagicMock(spec=LLMWorker)` with `.acting_coach = None` and `.is_embodied = False` assigned; or
  - `SimpleNamespace(router=..., acting_coach=None, is_embodied=False)`.

**Fake agents in tests (design delta 2, F3).** The hub check reads `agent.memory_hub`, so a fake agent needs a REAL
`memory_hub` attribute: `None`, or the hub the call passes.
- A bare `MagicMock` agent returns a child `Mock`, which fails the `is` check.
- A `MagicMock(spec=MaximAgent)` built before the property exists raises `AttributeError`.
- Set `.memory_hub` explicitly, or wire a real hub.

**What the checks are and are not.** The coach and embodiment checks are *statements*, not detection. They make each
site write down the worker state it believes it built, and fail if the two disagree. They cannot tell whether a site
*should* be embodied, so nobody counts them as a guard on embodiment. The hub check does detect a real wiring
mismatch.

After those checks it calls `run_agentic_loop` with every argument passed through unchanged.

`run_agentic_loop` is looked up when the call happens: it is imported lazily inside the function body, never bound
at `bootstrap` import time. The tests that monkeypatch `agent_loop.run_agentic_loop` therefore still reach the
patched stub. Those are `test_agent_router_slots_1228`, `test_bounded_stop_handle_1536`,
`test_atonia_red_gates_effect_policy` and `test_noop_bridge`.

Talk's `_worker` calls it in the same place the loop is called today, inside the `try` that ends in
`logger.exception`, so `FileSystemEnv`, `build_memory()` and `build_decision_engine()` are still built on the loop
thread:

```python
run_loop_stack(
    agent=agent, environment=FileSystemEnv(str(workspace)), state=state, memory=build_memory(),
    decision_engine=build_decision_engine(), executor=self.instance.executor, autonomy_controller=autonomy,
    llm_worker=worker, hippocampus=self.instance.hippocampus, memory_hub=self.instance.memory_hub,
    pain_bus=self.instance.pain_bus, bio_enrichment_pipeline=enrichment, thought_gate=thought_gate,
    default_network=None, imagination_trigger=None, evaluators=None, protocol_registry=None, sim_adapter=None,
    aut_mode="llm-primary", substrate_telemetry=None, substrate_policy=None, substrate_action_gate=None,
    percept_source=bridge.percept_source, action_sink=bridge.action_sink, response_policy=bridge.response_policy,
    state_home=None, acting_coach=None, is_embodied=False,
    max_steps=0, stop_event=stop_event, target_hz=2.0, consolidation="full", end_bound=end_bound,
)
```

Every value is the one talk passes today, or the loop's own default where talk passes nothing: `"llm-primary"` for
`aut_mode`, `None` for the rest. So the loop receives exactly the same arguments.

## The guard: `scripts/lint_loop_stack.py` (a GATE)

**Scope.** Every `.py` under `src/maxim/` and `scripts/`.
- `tests/` is excluded on purpose: tests drive the loop with fakes.
- `docs/**/*.py` is excluded on purpose: they are examples, not entry points.
- A broken scope (no files, or `agent_loop.py` missing) exits 2.

**Counted.** A site is counted per innermost enclosing def, using `lint_loop_mode_reads`' visitor and qualname
convention: `.<locals>.`; a lambda counts toward its enclosing def; decorators and defaults are in the outer scope.
These count:
- any LOAD of `run_agentic_loop`, `_run_agentic_loop_body` or `run_agent_loop`, as a Name (including an alias bound
  by `import … as`) or an Attribute;
- any non-docstring string constant equal to one of those names (`lint_loop_mode_reads._is_docstring`), which covers
  `getattr`, `importlib`, `_LAZY` keys and `__all__`.

A load counts, not only a call, so `Thread(target=…)`, `partial(…)` and a stored reference are all counted. These are
exempt, keyed by qualname:
- `bootstrap.py::run_loop_stack`, the sanctioned caller;
- in `agent_loop.py`, the three defs and `run_agentic_loop`'s sanctioned call to `_run_agentic_loop_body`. Any other
  load in `agent_loop.py` counts.
- in `scripts/lint_loop_stack.py`, ONLY the `ast.Constant` string nodes inside the assignments to the lint's own
  name constants: `NAMES`, `SANCTIONED_QUALNAME` and `BUILDER_NAME`, assigned at module level only (an `Assign` or
  `AnnAssign` in `tree.body`; the same names assigned inside a def are not exempt) (design delta 1, 4; narrowed by
  design delta 2, F2, and delta 3, N-1).
  - These are exempted by node, not given a ledger entry. They are configuration, not a call site, and a ledger entry
    would make the gate's own config count toward its ratchet.
  - Every other load in the lint counts, including a module-level lambda and an `if __name__ == "__main__":` call.

**`run_loop_stack` is called only directly and without a splat (design delta 1, 7; widened by design delta 2,
F1).** ANY load of `run_loop_stack` is a finding unless it is the callee of a `Call` with no `**` argument and no
`*` argument (a `*args` splat is a finding too, built in PR 1). And a STACK keyword's value may not be a parameter
WITH A DEFAULT of any enclosing def (review round, exec SF): `def easy_loop(pain_bus=None):
run_loop_stack(pain_bus=pain_bus, ...)` reintroduces the silent omission. A parameter with no default is fine, since
it pushes the requirement up to the wrapper's callers. The STACK names are read by parsing `bootstrap.LOOP_PARAMETERS`
from its AST (plus the two worker statements); an unparsable table exits 2. That covers:
- an alias (`f = run_loop_stack; f(**kw)`);
- `partial(run_loop_stack, **kw)()`;
- `Thread(target=run_loop_stack)`;
- a dict of callables;
- a `getattr` string;
- any non-docstring string constant equal to `"run_loop_stack"`.

The exceptions are its own def in `bootstrap.py` and the lint's module-level name constants (`BUILDER_NAME`), which
the self-exemption above covers for this rule too (design delta 4, NIT-1). So every site writes out every STACK keyword at a direct call, and
a reviewer sees each `None` there. `minecraft_harness._loop_kwargs` already returns a dict, and when that site
migrates it must expand the dict into explicit keywords.

**The ledger: `scripts/loop_stack_ledger.json`.** It is a JSON list in `swallow_moves.json`'s shape. Each entry is
`{file, qualname, count, ref, reason, permanent, moved_from?}`, keyed by `file::qualname`, never by line. It starts
with:
- the 7 sites to migrate (owner decision 2's "8 unmigrated" counted the orchestrator's own loop, which is permanent);
- `src/maxim/runtime/__init__.py::<module>`, permanent, as ONE entry with count 2 (design delta 1, 8). The two strings
  are the `_LAZY` key and the `__all__` entry for `run_agent_loop`. `__getattr__` loads neither name, so it measures
  0 and has no entry.
- the orchestrator's own loop, permanent (open question 2).

So the ledger holds 7 sites to migrate, plus 2 permanent entries (the orchestrator's narrator loop and the runtime/__init__ re-export), all citing the migration tracking issue #1840.

Each run prints every entry with its measured count, and permanent entries are marked `non-shrinking`.
`permanent` is a print label and a rule-3/4 invariant (it never changes), not a lock: a permanent entry can still be
deleted, and its still-measured site then fails as unlisted (rule 2).

**Rules (design delta 1, 3 and 6).** Rules 1, 2 and 6 read HEAD only and **always run**. Rules 3 to 5 compare with
the base, the merge-base from `_lint_git.base_ref`. Only rules 3 to 5 may skip, and only when there is no base ref and
`_lint_git.must_not_skip` returns False, which means a local run. On a pull request or push, a missing base is an
error.

1. **Measured equality.** Each HEAD entry's count equals its measured count. A measured 0 is an orphan and fails. That
   is how a migrated site must delete its entry.
2. **Unlisted fails.** A counted site with no HEAD entry fails.
3. **Shrink only.** HEAD's ledger is a subset of base's, keyed by `file::qualname`, except for entries carrying a
   valid `moved_from` (rule 4). No count may rise above its base pin, and `permanent` never changes on an entry that
   is at base (built in PR 1, stricter than v4's move-only wording). `moved_from` is checked only on an entry NEW to
   the branch: an entry that landed as a move keeps the field, and the next branch judges it by this rule alone. This rule is written for this lint.
   `_lint_allowance.append_only_problem` forbids removals, which is the opposite of what this ledger needs.
4. **Moves.** A HEAD entry not at base is legal only if its `moved_from: {file, qualname}` meets all three of these:
   - it names a base entry whose key is absent at HEAD;
   - its count is ≤ that source's base count;
   - no other HEAD entry names the same source. Each `moved_from` is used once.
   - **No splits or merges.** Two entries naming one source fail, and so does one entry claiming two sources. A
     decomposition that splits or merges sites needs a **lint change**, reviewed as such. There is deliberately no
     `split_from` rule: a sum rule adds bookkeeping for a case nobody has.
   - maxim-e6 is not blocked. `_aut_worker` and the orchestrator's own call are separate sources, and each moves whole
     with its own `moved_from`.
   - A move keeps the source's `permanent` value, so moving the orchestrator's own loop cannot quietly turn a
     permanent entry into a shrinkable one, or the reverse.
5. **Bootstrap.** A ledger absent at base is accepted only when its diff status against base is `A`, meaning this PR
   adds it. Then rules 3 and 4 have nothing to compare and pass.
6. **A readable HEAD ledger.** A HEAD ledger that is missing or unparsable exits 2, as does an entry missing a field
   or carrying a bad `ref` (`_lint_allowance.ref_ok`).

**Every fail-closed case:**
- a counted site with no entry;
- an entry whose count differs from measured, including an orphan at 0;
- a base entry deleted while its site still measures (the site is then unlisted, so rule 2 fails);
- a new entry without `moved_from`;
- a `moved_from` naming a source that is still present, or not at base;
- a moved count above its source;
- two entries sharing a source;
- a count raised above its base pin;
- a ledger absent at base whose status is not `A`;
- a missing or unparsable HEAD ledger, or a malformed entry (exit 2);
- no base ref on a pull request or push;
- a broken scope (exit 2);
- any load of `run_loop_stack` that is not the callee of a splat-free call (design delta 3, SF-A). A load means the
  same as for the other three names: a Name (including an alias bound by `import … as`) or an Attribute
  (`bootstrap.run_loop_stack`), plus a non-docstring string constant equal to the name. So `partial(rls, **kw)` and
  `partial(bootstrap.run_loop_stack, **kw)` are findings. A splat-free direct call through an alias (`rls(...)`) or
  an attribute (`bootstrap.run_loop_stack(...)`) is allowed, because the callee is still the function itself.

**How each input could lie:**

| input | how it could lie | handled |
|---|---|---|
| direct call | — | caught |
| alias `from …agent_loop import run_agentic_loop as ral` | the name is hidden | caught: the alias's loads count |
| attribute `al.run_agentic_loop(…)` | — | caught |
| `Thread(target=…)`, `partial`, a dict of callables | a reference, not a call | caught: any load |
| `partial(rls, **kw)` with `rls` an alias of `run_loop_stack` | a splat hidden behind an alias | caught: the alias is a load that is not a splat-free callee (delta 3) |
| `partial(bootstrap.run_loop_stack, **kw)` | a splat through an attribute | caught: the Attribute is a load (delta 3) |
| `getattr(m, "run_agentic_loop")`, `importlib`, `_LAZY`, `__all__` | a string | caught: a constant string counts (the re-exports are permanent entries) |
| a computed string (`"run_" + x`), `exec` / `eval` | invisible to the AST | **accepted blind spot**, named in the lint's docstring |
| `**kwargs` at a call | a hidden hand-assembled stack | irrelevant: any counted site outside the sanctioned caller is a finding |
| a new wrapper `def my_loop(**kw)` | the stack moves into its callers | caught: the wrapper's load is a new, unlisted site |
| a wrapper `def easy_loop(pain_bus=None, ...)` calling `run_loop_stack(pain_bus=pain_bus, ...)` | a default re-hides the omission | caught for a bare Name: a STACK value that is a defaulted parameter of any enclosing def is a finding (positional or keyword-only default, or a nested def's outer parameter); a parameter with no default is allowed |
| the same default reached other than as a bare Name: local reassignment (`p = pain_bus; run_loop_stack(pain_bus=p)`), a dataclass or NamedTuple field default (`cfg.pain_bus`, the plausible accidental route), `kw.get("pain_bus")`, `pain_bus or None`, a walrus | the rule sees only a bare Name | **accepted blind spots** (delta round 1), named in the lint's docstring. The scope walk ignores non-parameter bindings, which fails closed: a defaulted parameter rebound before the call, a comprehension variable, or an inner def's local shadowing the parameter is still flagged; rename it |
| `loop_setup.build_loop_run` | builds a run's setup outside the guarded names | **accepted blind spot**: running the loop still needs `_run_agentic_loop_body`, which is guarded |
| a new caller of an allow-listed wrapper (`_run_scenario_turn`, `_loop_kwargs`) | the old entry stays | **accepted blind spot** until that wrapper migrates; its `reason` says so |
| a line-keyed ledger | lines drift | not used: keyed by `file::qualname` |
| a renamed file or function | the entry no longer matches | caught: the old entry measures 0, an orphan, and the new key is unlisted. The fix is a `moved_from` entry, which the reviewer sees |
| the ledger deleted or emptied | the gate goes vacuous | caught: a missing HEAD ledger exits 2 (rule 6), and an emptied one leaves every still-measured site unlisted (rule 2) |
| a moved entry inflated, or one source split | the move hides growth | caught: a moved count must be ≤ its source's, and each source is used once (rule 4) |
| the lint's own `NAMES` strings | the gate counts itself | exempt only as the `ast.Constant` nodes inside the lint's name-constant assignments; any other load in the lint (a module lambda, a `__main__` call, a def) counts | 
| a `**` splat into `run_loop_stack` | hides which STACK values are `None` | caught: a splatted call is a finding |
| an alias `f = run_loop_stack; f(**kw)` | the splat moves off the name | caught: the alias load is not a call's callee |
| `partial(run_loop_stack, **kw)()` | the splat moves into `partial` | caught: the load is an argument, not a callee |
| `Thread(target=run_loop_stack, kwargs=kw)` | the call happens in the thread | caught: the load is an argument |
| `getattr(bootstrap, "run_loop_stack")`, a string constant | a dynamic lookup | caught: a non-docstring constant equal to the name is a finding |
| no base ref (shallow CI checkout), or git failing in rule 5 | a vacuous pass | caught: on a pull request or push the lint exits 2 (no verdict), as the sibling diff-scoped lints do |
| the sanctioned qualname, `NAMES`, roots or exclusions edited | the lint widens itself | pinned in `test_lint_loop_stack.py`. **Residual:** the same PR can edit the pin and the lint together; review is the guard |
| `tests/` and `docs/` | excluded scopes | by design |

**CI.** A step in `.github/workflows/test.yml`'s lint job, `python3 scripts/lint_loop_stack.py`, carrying a "Positive
control: tests/unit/test_lint_loop_stack.py" comment. The lint job already checks out with `fetch-depth: 0` for the
diff-scoped lints, and `scripts/prepr.sh` reads the step with no change. `docs/agents/runtime-tools.md` gets an
`[engineering]` invariant: "Every `run_agentic_loop` entry point goes through `bootstrap.run_loop_stack`". Its
Regression guard line: `scripts/lint_loop_stack.py` + `tests/unit/test_lint_loop_stack.py` +
`src/maxim/runtime/bootstrap.py::run_loop_stack` (the required keywords).

## Tests the first PR carries

- **Talk equivalence**, written first against today's code. It patches `agent_loop.run_agentic_loop` to record what
  it receives and the thread it runs on, then launches talk through the handle. The recorded values must be the same
  before and after the migration:
  - each argument, by identity for instance subsystems and by type for the fresh ones;
  - that the call ran on the `console.talk.*` thread;
  - that a raise inside it reaches `logger.exception`.
- **The function:**
  - `LOOP_PARAMETERS` keys equal the loop's signature;
  - each STACK keyword, when omitted, raises `TypeError` (parametrized);
  - a hub mismatch, a coach mismatch and an embodiment mismatch each raise;
  - the worker is never mutated;
  - every argument is passed through unchanged.
- **Existing tests the migration re-points, never deletes (design delta 1, 5).** The grep was for `_calls_in(`, a
  `"run_agentic_loop"` string, `getsource(handle`, and `_launch_talk_loop`, across `tests/`:
  - **`test_approval_intent_1185.py::test_each_bridge_sim_passes_its_bridges_policy[console/handle.py]`.** It walks
    the AST for a call named `run_agentic_loop` that passes `response_policy=<bridge>.response_policy`. Talk's call
    becomes `run_loop_stack`, so its matcher also accepts that callee for the handle case. The other sites keep
    `run_agentic_loop` until they migrate.
  - **`test_bounded_stop_handle_1536.py::test_the_talk_loop_is_built_with_its_own_end_bound_and_publishes_it`.**
    `_calls_in(MaximHandle._launch_talk_loop, "run_agentic_loop")` is re-pointed to `"run_loop_stack"`, and the
    `end_bound` assertion is unchanged.
  - **`test_agent_router_slots_1228.py` ~:242 (talk).** It patches `agent_loop.run_agentic_loop` with a no-op and
    passes a bare `MagicMock` worker to `_launch_talk_loop`. After the migration it would pass **vacuously**:
    - its assertion `held == [talk]` is satisfied when `MaximAgent` is built, before the loop call;
    - the coach `ValueError` would be swallowed into the thread's `logger.exception`.

    Fix: a fake worker with `acting_coach=None` and `is_embodied=False`, plus an assertion that the patched stub was
    reached.
  - **`test_bounded_stop_handle_1536.py` ~:371** (`test_the_shipped_path_handle_stop_bounds_a_live_talk_loops_consolidation`)
    calls `_launch_talk_loop(MagicMock(), …)` with a bare worker. After the migration it fails **loudly**: the coach
    check raises, so `opened.wait(5.0)` times out. Same fake-worker fix.
  - **Unaffected in PR 1, listed as traps for the later migrations:**
    - **`test_agent_router_slots_1228.py` ~:298** is the sub-simulation with its own `_Worker`. It needs checking when
      `simulation/tools.py` migrates.
    - **`test_atonia_red_gates_effect_policy.py` ~:968** is `api.run`. It already asserts the loop was reached. Its
      spec'd agent will fail loudly on `.memory_hub` when api migrates, so that PR gives the fake a real attribute.
    - **`test_noop_bridge.py` ~:164** is the sub-simulation, and it **fails open**. A `MagicMock` sub-agent's
      `.memory_hub` is a child `Mock`, so the hub check raises in the thread. Meanwhile the test's assertions come
      from the patched `send_and_wait`, so they still pass. That migration must give the fake agent `memory_hub=None`
      and assert the loop stub was reached.
  - **Test fakes the later migrations must fix** (also listed on #1840): the atonia `MagicMock(spec=[...])` agent for
    api (`test_atonia_red_gates_effect_policy.py`), `test_noop_bridge.py` for the sub-simulation, and
    `test_agent_router_slots_1228.py` ~:298. Each needs a real `memory_hub` attribute and a fake worker with
    `acting_coach=None`, `is_embodied=False`, plus an assertion that the loop stub was reached.
  - **Docs left for the migration PRs:** `docs/simulation.md` and `docs/memory-system-interactions.md` show direct
    `run_agentic_loop` calls; each site's migration PR updates its example.
  - **`test_talk_never_asks_1658.py`** patches `_launch_talk_loop` away entirely, so it is unaffected.
  - The remaining grep hits read `agent_loop`'s own defs and are unaffected: `test_loop_setup`,
    `test_loop_setup_characterization` and `test_lint_release_gates`.
- **The lint (`test_lint_loop_stack.py`):**
  - a fixture for every row of the table above;
  - the five rules;
  - exit 2 on a broken scope;
  - the pinned constants;
  - a deletion probe: re-add a direct call in talk, and the lint fails.

## Open questions for the owner (recommendation first)

1. **How do migrations treat a site's existing gaps?** *Rec (strict): keep each site's exact current set.* A site that
   lacks a component passes `None` explicitly, its PR names the gap, and each gap is filed with `Deferred because:
   owner-decision`. That is expressible now that every subsystem is an explicit keyword. The alternative (closing
   gaps during migration) changes behaviour inside a refactor.
   - **Talk's own gap, named here (design delta 1, 9).** The handle advertises `body="bodies/reachy_mini"`
     (`console/handle.py`, module and `MaximHandle` docstrings). Talk never sets `is_embodied`, so its call says
     `is_embodied=False`.
   - This PR keeps that and files it as a gap, #1841. Whether an embodied handle's talk is prompted as embodied is
     a behaviour decision.
2. **The orchestrator's own loop.** *Rec: a permanent ledger entry with its reason*, printed as non-shrinking. It is a
   narrator, not an agent stack.
3. **`run_agent_loop`.** *Rec: include it in the guarded names.* It is public API (`maxim.runtime.__all__`), so
   deleting it is a separate decision. Its re-export is a permanent ledger entry until then.
4. **DECIDED (owner, 2026-11-27): `consolidation` is a REQUIRED STACK keyword.** Every site states it. Talk passes
   `consolidation="full"` today and keeps that literal. `LOOP_PARAMETERS` is 27 STACK and 12 RUN.
5. **DECIDED (owner, 2026-11-27): the first PR adds a public read-only `MaximAgent.memory_hub` property**, and the
   two-way hub check reads it.

## Size (re-estimated)

About 650 lines in all (v3):
- `src/`, about 120: the function, `LOOP_PARAMETERS`, the three checks and the talk call, net of talk's removed lines,
  plus about 8 for the `MaximAgent.memory_hub` property (Q5).
- `scripts/`, about 340: the lint about 310 (the moves rule and the splat rule are new in v3), the ledger about 30.
- `tests/`, about 370: the lint about 210, equivalence plus the function about 130, and the re-pointed talk tests
  about 30.

## Design round 1, folded (owner of the folds: the coordinator, 2026-11-27)

- **A.** The frozen `LoopStack` type is replaced by one function, `run_loop_stack`. The lazy-import reason is
  corrected to import cost, since there is no cycle. Talk calls it inside `_worker`, so timing is unchanged.
- **B.** The signature is explicit, with every parameter classified STACK or RUN in `LOOP_PARAMETERS`. A test checks
  the classification against the loop's signature, and v1's override `TypeError` is gone.
- **C.** `agent=` is required, and `instance` is not a parameter.
  - Checking found that no site wires the hub at the call, so the function checks the wiring instead of doing it.
- **D.** v1's `embodied` flag is replaced by `acting_coach=` / `is_embodied=`, which are asserted against the worker
  and never written.
- **E.** The table is corrected:
  - cli always passes its pain bus, and a sim's overrides it;
  - the embodied runtime has no AgentInstance;
  - six columns are added;
  - talk's absences are stated.
- **F.** The lint gets its rules:
  - the JSON ledger, the shrink-only rule written for this lint, `moved_from`, the bootstrap `A` rule and
    `must_not_skip`;
  - strict equality plus the orphan check;
  - the borrowed visitor, string constants, the narrowed `agent_loop.py` exemption and the stated exclusions;
  - the pinned constants with their residual, and the corrected table.
- **G.** Open questions 1–3 are resolved as strict recommendations, and questions 4 (`consolidation`) and 5 (the
  hub check's private field) are new.
- **H.** The size is re-estimated at about 600 lines.

## Design delta 1, folded (2026-11-27)

The delta round confirmed v2's 39-parameter classification, the talk call and the checked attributes. Its folds:
1. **`llm_worker=None`** requires `acting_coach=None` and `is_embodied=False`. The coach is compared with `is`. The
   section on fake workers in tests says how to build one.
2. **The hub check is two-way**, `agent.memory_hub is memory_hub` with `None`/`None` allowed, reading the Q5
   property.
3. **Moves:**
   - a moved count must be ≤ its source's base count;
   - each `moved_from` is used once, so no splits (justified in rule 4);
   - rule 3 reads "subset, except valid `moved_from` entries";
   - a move keeps `permanent`;
   - every fail-closed case is listed.
4. **The lint's own `NAMES`** are exempt by the pinned `scripts/lint_loop_stack.py::<module>` qualname, not by a
   ledger entry.
5. **Existing tests to re-point.** The note names every talk-keyed test, and the bare-`MagicMock` worker trap with its
   fix.
6. **When rules run.** Rules 1, 2 and 6 read HEAD and always run. Only rules 3 to 5 skip, and only locally with no
   base. A missing or unparsable HEAD ledger exits 2.
7. **No `**` splat** in a `run_loop_stack` call.
8. **`runtime/__init__.py`** is one `<module>` entry with count 2. `__getattr__` measures 0.
9. **Talk's `is_embodied=False`** is named as a gap under Q1.
10. **The coach and embodiment checks** are statements, not detection.

Q4 and Q5 were then decided by the owner, 2026-11-27, both on the strict option: `consolidation` is STACK (27 STACK, 12 RUN), and the public `MaximAgent.memory_hub` property is added in the first PR.

## Design delta 2, folded (2026-11-27)

Delta round 2 closed SF1, SF2, SF3, SF6 and the four NITs. Its folds:
- **F1 (fail-open).** Any load of `run_loop_stack` that is not the callee of a splat-free call is a finding. That
  covers aliases, `partial`, `Thread(target=)`, `getattr` strings and string constants. Its own def and the lint's
  module-level name constants are the only exceptions. The evasion table gains a row for each.
- **F2 (fail-open in form).** The lint's self-exemption is narrowed from its whole `<module>` scope to the
  `ast.Constant` string nodes inside its name-constant assignments. A module-level lambda or a `__main__` call in the
  lint counts.
- **F3.** The re-point list is corrected:
  - talk's `test_agent_router_slots_1228` ~:242 would pass vacuously, and gets the fake-worker fix plus a
    stub-reached assertion;
  - `test_bounded_stop_handle_1536` ~:371 would fail loudly, and gets the fake-worker fix;
  - the 1536 call-site matcher is re-pointed (`_calls_in(..., "run_loop_stack")`) and the 1185 matcher is extended to
    accept `run_loop_stack` beside `run_agentic_loop` for the unmigrated sites; neither is removed (design delta 3,
    N-2; delta 4, NIT-2);
  - three traps are listed for later migrations (the 1228 sub-simulation, the atonia `api.run` test, and
    `test_noop_bridge`, which fails open);
  - guidance on fake agents is added.
- **F4.** A split or merge needs a lint change, reviewed as such, and there is no `split_from` rule. maxim-e6's two
  orchestrator sources each move whole, so it is not blocked.

**Design rounds:** round 1 (architecture + adversarial gate) and delta rounds 1-4 on 2026-11-27. Delta 4 met the stop rule; its two wording NITs are folded. Owner decisions 2026-11-27: (1)-(5) above.

**Built (PR 1) and its review round, 2026-11-27.** Stricter than v4 in four places, now written above: a `*args`
splat is a finding; `permanent` cannot change on any base entry; `moved_from` is checked only on new entries;
`permanent` is a print label, not a lock. The review round added the defaulted-parameter rule, the exit 2 for no
verdict, the keyword-only forwarding of every argument, and the #1840 / #1841 refs.

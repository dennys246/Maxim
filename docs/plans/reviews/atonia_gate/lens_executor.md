# Executor lens: atonia gate approach notes v5.1

**Reviewer:** Executor lens (docs/CODE_REVIEW.md), pre-merge round on `git diff origin/main...HEAD` at `d6389cd4`.
**Question:** if the plan is built exactly as written, does it work on the real production path at HEAD? Gate
soundness belongs to the adversarial passes and is not repeated here.
**Method:** spot-checked about 60 named symbols at HEAD (all exist except where noted). Ran probes from the
scratchpad with the repo venv: the import closure and env read-set of `maxim.runtime.executor` and
`maxim.agents.permissions`, and the llama-cpp-python 0.3.8 API surface. Mapped every `Executor(` / `build_executor(` /
`AgentConfig(` site to its enclosing function and checked each one against `function_length_baseline.json`.

**Verdict:** 1 BLOCKER, 9 SHOULD-FIX, 5 NIT. The blocker is buildability, not design: item 1's pinned import-closure
gate cannot pass with the planned composition at HEAD.

---

## BLOCKER

### B1. `maxim.advisor`'s import-closure ceiling is unattainable: `Executor` and `AgentPermissions` drag in the whole loop
`docs/plans/atonia_gate_approach_notes.md` §4.2 ("The import closure is pinned") vs `src/maxim/runtime/__init__.py`
and `src/maxim/agents/__init__.py`.

The observer is built as `Executor(sealed, policy=ToolPolicy.pinned(...))`, and `ToolPolicy` holds
`permissions: AgentPermissions`. Probe at HEAD (`importlib.import_module`, then a count of `sys.modules`):

| Import | `runtime.agent_loop` | `agents.llm_worker` | `default_network` | `memory` | `utils.paths` | total modules |
|---|---|---|---|---|---|---|
| `maxim` | 0 | 0 | 0 | 0 | 0 | 94 |
| `maxim.runtime.executor` | 1 | 1 | 18 | 22 | 1 | 832 |
| `maxim.agents.permissions` | 1 | 1 | 18 | 22 | 1 | 833 |

`executor.py`'s own imports are light. The cost comes from the package `__init__`s:
- `maxim/runtime/__init__.py` imports `run_agent_loop` eagerly;
- `maxim.agents.exec_agent` pulls in `default_network` and `llm_worker`;
- `bio_integration` pulls in `memory`.

The plan's test requires `sys.modules` to exclude `runtime.agent_loop`, `agents.llm_worker`, `default_network` and
`memory`. That fails on the day it is written.

The same imports also break the pinned env read-set. Importing the closure reads `XDG_CONFIG_HOME` and `HOME` at
module level, in `models/language/profile_loader.py:131-132` (reached via the runtime package). scipy reads `HOME`
and `SOURCE_DATE_EPOCH`. None of these is on the plan's allow-list ("the refusal list, and library reads such as
numpy's and threadpoolctl's").

**Failure scenario:** step 3 is built, and the import-closure and env-read-set red gates are red with no code path to
make them green. They then get loosened or re-pointed, which is exactly what CLAUDE.md forbids for strict gates.
Alternatively, the observer grows its own dispatcher, which contradicts "the sealed executor" and §3's single
dispatch check.

**Fix:** add a build step before item 1:
- either make `maxim/runtime/__init__.py` and `maxim/agents/__init__.py` lazy (PEP 562 `__getattr__`). There are 156
  `from maxim.runtime import` users, so this ripples, and it needs its own census;
- or name the module path the observer imports from, so that it avoids both package `__init__`s, and move
  `ToolPolicy` and the `AgentPermissions` type to a module outside `agents/`.

Then re-measure the closure and the env read-set on that path, and pin those numbers.

---

## SHOULD-FIX

### S1. The effect mismatch rule, as written, refuses every undeclared tool, including stable-API ones
§2.2 "Mutation is visible": the record is `ALL` when `effect` is absent, and the check compares it with
`type(obj).__dict__.get("effect")`. For an undeclared class that lookup returns `None`. `ALL != None`, so admission
"refuses loudly". This happens under every ceiling, `ALL` included.

**Scenario:** step 1 lands. Every test that registers an ad-hoc `class Dummy(Tool)`, and every
`maxim.register_tool(...)` user whose class has no `effect`, is refused. `register_tool` and `@maxim.tool` are ✅ in
`docs/user/stable_api.md`. The suite goes red.

**Fix:** state the rule exactly. "Absent and recorded `ALL`" is consistent. Only a present value that differs from
the record is a mismatch.

### S2. `@maxim.tool` and `register_tool` tools are always `ALL`, so a narrowed `maxim.run` silently stops running them
§3 "Call sites": `api.run` gets `only` or `pinned`, "decided per branch, from #1342's direction". #1342 is still OPEN.
`api.py::tool` builds `FunctionTool` from `fn` alone, so it has no way to declare an effect, and it must declare
`ALL`. `api._inject_registered_tools` then refuses it with a WARNING under any ceiling below `ALL`.

**Scenario:** a stable-API user's `@maxim.tool` stops being callable from `maxim.run()` in a patch. Only a log line
says so.

**Fix:**
- decide `api.run`'s policy per branch now (robot runs need `ACTUATE`);
- either add an `effect=` argument to `tool` and `register_tool`, or keep `api.run` at ceiling `ALL` with an
  allow-list;
- write the O1 strict xfail's flip criterion as a concrete policy value, not "non-permissive".

### S3. The per-layer dispatch sweep fails, or proves nothing, at the ceiling layer
§4.2 Red gates: "removes the other two layers and inserts every default-registry tool must refuse at **that** layer".
The ceiling is `Effect.READ`, and READ-only default tools (`read_file`, `list_directory`, `glob`, `code_search` and
similar) pass a READ ceiling. The sealed-contents layer also cannot "insert every default-registry tool": its contents
*are* that layer.

**Scenario:** the sweep is red on the first READ tool, or the author excludes READ tools and the sweep stops covering
them.

**Fix:** define each layer's sweep set. The ceiling layer sweeps tools with `exceeds(effect, READ)`. READ tools are
proven refused by the pin and the sealed contents. The sealed-contents layer sweeps names, not instances.

### S4. The closed backend "exposes exactly one call", but the prompt builder needs the model's tokenizer and token attributes
§4.2 says `open_closed_local_backend` exposes only `complete_tokens(prompt_ids, ...)`. The token-id prompt needs three
things from the loaded `Llama`:
- `Llama.tokenize(bytes, add_bos, special)`. It takes `bytes`, as checked on 0.3.8;
- `llama_token_get_attr`, which 0.3.8 exposes alongside `llama_vocab_get_attr`;
- the vocab.

**Scenario:** the builder either adds a second, unreviewed surface, or imports `llama_cpp` in `maxim.advisor`, outside
`models/language/`.

**Fix:** put the whole defang (template plus untrusted segments, then ids) inside `closed_local.py`, and list its
exported calls exactly.

### S5. Item 1's red gates need a runnable GGUF and a test seam, and neither is named
Every item-1 gate runs `open()` plus one `ingest()`: no-effect, env read-set, import closure, and the status-table
branches. `open()` eagerly loads a real GGUF and, by signature, takes no injected backend. The vendored llama.cpp
`models/ggml-vocab-*.gguf` files are vocab-only and cannot run a completion. CI installs llama-cpp-python
(`test.yml` line 576), but no runnable model fixture exists in `tests/`.

**Scenario:** the gates `skip` when the model is absent and pass with nothing tested (the "green PR ran no tests"
lesson). Or the `error`, `abandoned` and schema-invalid branches can only be reached by real model output, which a
grammar makes nearly impossible, so they never run.

**Fix:**
- name a tiny runnable GGUF fixture, with its size and source;
- add a private backend-injection seam for in-process branch tests (monkeypatch `open_closed_local_backend`);
- make the gates `--require-extras`-style, so they fail instead of skipping when llama_cpp is missing.

### S6. Subprocess-only gates leave the new code uncovered under the 80% diff gate
`pyproject.toml [tool.coverage.run]` has no subprocess support, and `scripts/lint_coverage.py` forbids changing that
config. Almost every planned gate runs "in a fresh subprocess": §4.2's gates and §6's red gate. Lines reached only
there count as unexecuted. That covers the audit hook's branches (each event type, `family=-1`, a `None`
destination), `engage()`, the closed backend, and `AdvisorSession`'s status branches.

**Scenario:** item 10's PR fails `scripts/suite_at_commit.sh`'s diff-coverage gate, even though every gate is green.

**Fix:** add an in-process unit test for each hook branch and each status row, using the conftest latch reset the plan
already has. Keep the subprocess runs as the end-to-end proof.

### S7. `auto_sense` has two direct `.execute()` calls; V6 covers one
`runtime/loop_perception.py::auto_sense`:
- the auto-fire loop calls `_af_tool.execute()`;
- the interoception block calls `_sense.execute(entity_name=...)` for each self-entity.

§3 V6 names only "`executor.check(tool)` before `.execute()`" for auto-fire. Both calls also sit inside
`except Exception` blocks that call `log_swallowed_exception`. A `check()` that raises instead of returning a refusal
is therefore swallowed at DEBUG.

**Scenario:** a ceiling that refuses `sense` still has it run every percept.

**Fix:**
- `check()` both calls;
- `check()` returns a typed refusal and never raises;
- the `auto_sense` probe drives both calls.

### S8. Rule T bites pinned functions the plan does not account for
§3 names only `build_executor` (250) and `_start_agentic_runtime` (924). The planned hunks also land in:

| Function | Pinned at | Planned hunk |
|---|---|---|
| `api.py::run` | 276 | both branches' policy; `engage` for api verbs |
| `cli.py::_main_impl` | 1604 | `AgentConfig(...)` at line 2061; the offline startup refusal |
| `runtime/agent_factory.py::AgentFactory.create_full_agent` | 204 | refuses `None`; passes the policy |
| `simulation/orchestrator.py::start_simulation_mode` | 2889 | `AgentConfig` at lines 950 and 1512 |
| `runtime/lane_backends.py::_maybe_auto_spawn_server` | 271 | the early `is_offline()` return |
| `runtime/bootstrap.py::build_tool_registry` | 269 | if the child registry's ceiling becomes a parameter there |

`start_simulation_mode` is fenced, so its extraction is forbidden and the hunk needs a reviewed `touch_exemptions`
record. Possible further hunks:
- `runtime/lane_backends.py::build_primary_router` (245) and `models/language/router.py::LLMRouter._try_provider` (212),
  for "the predicate at build and every dispatch";
- `runtime/tool_dispatch.py::record_outcome` (399) and `runtime/tool_dispatch.py::execute_and_learn` (392), for "a
  policy refusal books no NAc outcome".

**Scenario:** each PR fails `scripts/lint_function_length.py::touch_rule` at prepr.

**Fix:** list each pinned function per build step, with its extraction or its exemption.

### S9. Build order: item 10 is not parallel. Items 3 (V7) and 1 depend on it
§7 says "Item 10 in parallel. The observer depends only on `engage()` and the hook." Two dependencies run the other
way:
- item 3's V7 narrowing inside `build_executor` reads `is_offline()`;
- step 3's `open()` "engages item 10's latch before anything else is built", and its no-effect gate needs the hook.

**Scenario:** step 2 either ships without V7, so `build_executor` (pinned, so rule T applies) is touched twice, or it
waits for item 10.

**Fix:** make `utils/offline.py` (`engage`, `is_offline`, the hook and the ledger) step 0, and keep the call-site
refusals parallel.

---

## NIT

- **N1. The V3 hard break is about 190 test sites in one atomic commit.** At HEAD: `Executor(` 42 sites in 16 files,
  `build_executor(` 73 in 28, `AgentConfig(` 74 in 13. There are also about 15 `src/` sites. §7 step 2 should say this
  lands as one commit with the O1 flip, so no intermediate commit is red. (`scripts/` has no real sites: `exp53`'s
  `_Executor` is local.)
- **N2. `POLICY_PERMISSIONS` changes existing bookkeeping.** Today `Executor.execute`'s two `_permission_denial` returns
  append to `_tools_hallucinated`. §3 says policy refusals do not, yet also that "existing refusals are unchanged".
  Pick one, and check `tests/fixtures/orchestrator_characterization_golden_v1.json` (it carries tool-usage stats).
- **N3. Backend naming.** V8 says "the `llama_cpp` or `transformers` backend", but `LLMConfigSection.backend` is
  `Literal["llama_cpp", "pytorch"]`. The router accepts `transformers` only as a provider-type alias. Use the config
  spelling in the startup predicate.
- **N4. The status table has no precedence.** The precedence between `degraded` and `ok` with refusals, and between
  `degraded` and `error`, is unstated. The macOS `F_FULLFSYNC` branch never runs on Linux CI, so mark it or test it
  with a monkeypatched `fcntl`. The plan also does not mention `scripts/run_mypy_public.sh`, which lists
  `executor.py`, `api.py`, `create.py`, `load.py` and `loop_perception.py` (the `ToolRegistryLike` Protocol lands
  there), or the per-PR `[Unreleased]` lines (the lint enforces them, but the CHANGELOG text for the V3 break should
  be drafted in the plan).
- **N5. `advisor_mode.md` keeps a stale line.** The banner (line 12 context) still says the gate "ships as its own
  patch, 1.3.4", against the notes' "next free patch number". §8 lists the release number as stale but it is not
  folded. Separately, `pinned_executable` (v3 fold 7, §9) exists nowhere in `src/` and was deleted by V8; the fold
  ledger should say "superseded by V8".

## Verified real at HEAD (spot-checks)

- `tools/base.py`: `ToolErrorKind` (an Enum; `agents/bus.py` re-exports it, so one enum), `GIT_HARDENING_ARGS`,
  `git_root_refusal`, `host_tool_env`.
- Registries: `tools/reachy_stubs.py::create_stub_tool` (reassigns class fields, as the plan says),
  `tools/registry.py::_register_scene_locked` and `activate_scene`, `simulation/tools.py::SimToolRegistry` (not a
  `ToolRegistry` subclass; one constructor at `orchestrator.py:1366`) and `_FallbackRedirectTool`,
  `SpawnSubSimulationTool` (synchronous `_run_sub_simulation`, `build_executor(..., permissions=None)`).
- Executors: `runtime/executor.py::TOOL_ALIASES`, `runtime/concurrent_executor.py::ConcurrentExecutor` (submits
  `self.executor.execute`), and the `__getattr__` proxies on `FearGatedExecutor`, `PainInterceptorExecutor`,
  `AnticipatoryPainExecutor` and `InstrumentedExecutor`.
- Item 1 and item 5: `runtime/loop_perception.py::auto_sense`, `agents/autonomy.py::ProposalQueue`,
  `utils/content_safety.py::frame_tool_output`, `simulation/report.py::save_action_log` and `_code_stamp`,
  `integration/bio_enrichment.py::_dispatch_reflex_tool`, `api.py::research`.
- Item 10: `runtime/lane_backends.py::BACKEND_CLASSES`, `resolve_backend_class` (uses a non-literal `__import__`, as
  the plan pins), and `_maybe_auto_spawn_server`; `utils/http.py::_proxy_mounts`, `get_client` and `close_all`;
  `utils/gpu_detect.py::ensure_blackwell_guards` (called at the top of `_main_impl`, so `main` is upstream);
  `cli.py::main` wraps `_main_impl`.
- Config: `runtime/config_loader.py::CONFIG_FORMAT_VERSION == "1.3"`, `LLMConfigSection` (SHAPE-FROZEN, so the bump is
  required), `_FIELD_TO_ENV`; `runtime/config_writer.py::_NON_SECURITY_FIELDS`.
- llama-cpp-python 0.3.8: `Llama.create_completion` takes `grammar` and `stopping_criteria`.
- Python 3.10 floor: `field(kw_only=True)` is fine.

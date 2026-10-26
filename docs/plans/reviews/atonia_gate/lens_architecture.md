# Atonia gate approach notes v5.1: the Architecture lens

**Reviewer:** the Architecture lens of the three-lens pre-merge round (docs/CODE_REVIEW.md), read-only, worktree
`.worktrees/v-atonia`, 2026-10-25. **Object:** `git diff origin/main...HEAD`: `docs/plans/atonia_gate_approach_notes.md`
v5.1, the `docs/plans/advisor_mode.md` banner and "Superseded" markers, and the folded passes under
`docs/plans/reviews/atonia_gate/` (context). **Question:** for items 2, 3, 1, 5 and 10 and decisions V1–V8, is each one the root
cause, at the right layer, with the right owner? Does it hold to the invariants, and are its docs honest?

**Verdict: 2 BLOCKER, 9 SHOULD-FIX, 6 NIT.** Both blockers are the same root cause: **eager package `__init__`
imports**. The plan commits to reusing `Executor`, `Tool` and a module inside `models/language/`, and it also
commits to an import closure and an env read-set that this tree cannot meet while those `__init__`s stay eager. I
probed both at the worktree's HEAD, and the commands are below. The adversarial passes looked at what gets
*constructed*. The 1+5 pass said: "I saw no import-time disk or socket effect". So no pass ran the closure the notes
pin. Everything else is in already-covered classes, or concerns placement and collapse.

---

## BLOCKER

### B1. `Executor` cannot be in the observer's import closure as the tree stands. Importing it loads every module the closure forbids.

`atonia_gate_approach_notes.md` §4.2 "The sealed registry" (the executor is `Executor(sealed, policy=…)`) vs §4.2
"The import closure is pinned" (it must exclude `runtime.agent_loop`, `agents.llm_worker`, `default_network`, `memory`).

**Evidence (probe at HEAD):**
```
PYTHONPATH=$PWD/src python -c "import sys, maxim.runtime.executor; ..."
{'maxim.runtime.agent_loop': True, 'maxim.agents.llm_worker': True, 'maxim.default_network': True, 'maxim.memory': True, ...}
225 maxim.* modules loaded
```
- `src/maxim/runtime/__init__.py` eagerly runs `from maxim.runtime.agent_loop import run_agent_loop`, plus bootstrap,
  concurrent_executor, preemption and skill_matcher. Importing `maxim.runtime.executor` runs that `__init__` first.
- `runtime/executor.py::Executor.execute` also lazily imports `maxim.simulation.sim_logger` on **every** dispatch
  (`get_interactive_mode()`, the #864 copy), so the dispatch path pulls `simulation/` in as well.
- The env probe on `maxim.runtime.executor` records reads of `HOME` and `XDG_CONFIG_HOME`. The §4.2 read-set
  allow-list does not permit them (next finding).
- `maxim.tools.base` has the same shape. `tools/__init__` imports `tools/reachy.py`, which imports
  `inference/segment_vision.py`, which reads `HOME` at module level and loads scipy. `ProposeActionTool(Tool)` drags
  in the vision stack.

**Failure scenario:** the red-gate PR writes the import-closure test as specified. It fails on the first `open()`
with no code wrong anywhere in the gate. The implementer is then left choosing ad hoc between three moves: widen the
"ceiling that may only fall" on day one, which hollows out the closure guarantee; write a private dispatcher, which
forks the "one seam" dispatch check the plan rests on (§3's "dispatch is authoritative"); or make
`maxim/runtime/__init__.py` lazy in a hurry.

**The fold is a decision the plan must make and name an owner for:**
- (a) Make `maxim/runtime/__init__.py` and `maxim/tools/__init__.py` lazy, with PEP 562 `__getattr__` re-exports.
  That keeps public import paths stable, as ARCHITECTURE.md §Invariants requires. It is the root cause, and it also
  shrinks every `import maxim.runtime.X`. It touches files that other sessions own, so list it under §7's "Files
  owned elsewhere".
- (b) Or move the dispatch check into a leaf module that both `Executor` and the observer call.
- Not acceptable: a separate observer-only executor. That is the duplicate seam the notes argue against.

Either way, the closure test should be an **allow-list** of `maxim.*` modules, as the 1+5 pass finding 4
recommended, not an exclusion list. An exclusion list cannot fail on a new heavy import it never named.

### B2. `models/language/closed_local.py` reads the user's config at import. Its placement contradicts "reads no env" and "config never opened".

`atonia_gate_approach_notes.md` §4.2 "The LLM: a closed backend entry point": it "lives inside `models/language/`, so
the … invariant holds", and it "reads no env". The §4.2 inputs table says config.json and llm.json are "Never
opened", and the `open` events are pinned to `os.devnull`, `report_dir` and the import set.

**Evidence:** importing any `maxim.models.language.*` module runs `models/language/__init__.py`. That module imports
`config.py`, which calls `_apply_user_profiles_at_import()` → `profile_loader.py::apply_user_profiles` →
`profiles_config_path()`. That path reads `XDG_CONFIG_HOME` and `HOME` and loads `~/.config/maxim/profiles.yml`. The
env probe on `import maxim.models.language` records exactly `['HOME', 'XDG_CONFIG_HOME']`. The same `__init__` also
imports `router.py` and `cost_tracker.py`, the modules the notes said the closed backend avoids.

**Failure scenario:** an attacker-controlled `XDG_CONFIG_HOME` (adversary 3 in §0) points at a `profiles.yml`. The
observer parses it at import, before `open()` can refuse anything. The env read-set test then fails. Or, if someone
"fixes" the test by allow-listing `HOME`/`XDG_CONFIG_HOME`, the observer ships reading user config, which falsifies
advisor_mode.md's Environment row ("ignores … the user config file").

**Fold:**
- Make `models/language/__init__.py` lazy, and move the profile application out of import time into the first
  `LLMConfig`/router build. That is the root cause: an import with a filesystem side effect is a mechanism with no
  caller-visible switch.
- Or place the closed backend at a path whose package `__init__` is side-effect free, and amend the LLM-routing
  invariant's exception list to say so explicitly.
- The §4.2 inputs table must name `profiles.yml` (and `peer.yml`/`mesh.yml` if reachable) alongside config.json and
  llm.json.

---

## SHOULD-FIX

### S1. The mode gate's name sets are a second effect classification. Collapse them onto `Tool.effect`, or schedule the collapse explicitly.

`modes/definitions.py::HOST_ACTING_TOOLS`, `NETWORK_CAPABILITY_TOOLS`, `FILESYSTEM_CAPABILITY_TOOLS` and
`DISPATCH_CAPABILITIES` (read by `ModeDefinition.dispatch_refusal` and `raises_capability`) classify tools by **name**
into "acting on the host", "network" and "filesystem". That is what `Effect` does by type, and it is the defect class
O2 confirmed: `send_message`, `call_user` and `write_file` are allowed in passive. After this plan, a new tool must
be classified twice, and the two can disagree. For example, a new `post_webhook` tool declares `NETWORK` but is
missing from `NETWORK_CAPABILITY_TOOLS`, so passive mode lets it run in every non-observer loop.
advisor_mode.md's "What rides on existing infrastructure" says `Tool.effect` is "genuinely new, because nothing
classifies effects today". The mode tables do classify effects, by name.

**Fold:** add a line to §7. Either derive `DISPATCH_CAPABILITIES`' sets from `effect_of` in this release (passive =
ceiling `READ|SELF_SCOPE`, plus the existing forbidden list), or file the collapse as a named follow-up with the O2
consequence stated. The plan's front-gate principle points at the first option: ride on `Effect` rather than keep
two vocabularies.

### S2. One dispatch seam: name where the ceiling and pin checks sit relative to `Executor._permission_denial`, and how each refusal is booked.

runtime-tools.md's invariant "one seam, `Executor._permission_denial`" holds three gates: forbid source, mode and
`AgentPermissions`. `permits()` and the prompt roster read it. §3 adds the ceiling and the pin as "step 3" of
dispatch, and says `permits` applies the ceiling, without saying they go **into** `_permission_denial`. It also books
the refusals differently:
- `POLICY_*` refusals book no `_tools_hallucinated`;
- today's `AgentPermissions` denials, which now become `POLICY_PERMISSIONS`, do book it;
- "existing mode refusals are booked unchanged".

So one class of refusal ends up with three booking rules, and moving `AgentPermissions` under `ToolPolicy` silently
changes how its existing denials are counted.

**Fold:**
- State that the ceiling and pin checks live in the same seam that `permits()` reads, in an order you name (forbid →
  mode → policy permissions → ceiling → pin).
- Use one booking rule for every gate refusal, mode included.
- Name the behaviour change for existing `tools.allow`/`tools.deny` denials.
- The runtime-tools.md invariant text ("THREE gates") is a claim this changes, so put it in §7's hand-off list.

### S3. `AgentPermissions` vs `ToolPolicy`: say what happens to the old `permissions=` parameters.

§3 adds `policy=` to `Executor`, `build_executor` and `AgentConfig`, and `ToolPolicy` holds `permissions`. The notes
never say whether `Executor(permissions=)`, `build_executor(permissions=)` (23 `permissions=` sites in `src/`) and
`AgentConfig.permissions` are **removed**. If they survive beside `policy=`, two sources carry the allow-list, and a
caller that sets only the old one gets a policy that silently ignores it. The console path today builds
`AgentPermissions` through `agents/permissions.py::permissions_from_tools_settings`.

**Fold:** V3 is already a hard break, so remove the `permissions=` parameters in the same change. The factories then
take `AgentPermissions` as their one input, so there is exactly one source.

### S4. Layer placement of `ToolPolicy`, `Effect` and `current_policy` is unspecified. The layer rules constrain it.

`utils/audit.py::LAYER_RULES` (CI-gated, and it counts `TYPE_CHECKING` imports too) forbids `tools → agents` and
`tools → runtime`, and `agents → tools`.
- `ToolPolicy` holds `AgentPermissions` (from `agents/`) and `Effect`/`Tool` (from `tools/`). `narrowed()` builds a
  new `AgentPermissions` at runtime, so `ToolPolicy` cannot live in `tools/` or `agents/`.
- If it and `current_policy` live in `runtime/`, a composite under `tools/` cannot read `current_policy` without a
  layer violation. `SensePresenceTool` (`tools/discovery.py`) and `AutonomyLevelTool`
  (`tools/mode_switch.py`) are tools-layer; `ModulatorAffordanceTool` (it "can spawn tools") is in
  `embodiment/tool_bridge.py`.
- `SpawnSubSimulationTool` happens to be in `simulation/`, which is unruled, so it works today and nothing future-proofs it.

**Fold:**
- Name the module for each: for example, `Effect` and `exceeds` in `tools/base.py`; `ToolPolicy` in
  `runtime/tool_policy.py`.
- Expose `current_policy` from a tools-layer module as a Protocol-typed handle with `narrowed()`, so tools-layer
  composites can read it.
- Add `advisor` to `LAYER_RULES` (must_not_import `agents`, `memory`, `default_network`, `comms`, …). That gives the
  observer a static fence that rides on the existing audit, and complements the subprocess closure test.

### S5. V7's offline narrowing "at build" fights the future-code adversary. Apply it once at dispatch instead.

The V7 row and §3 "Under offline": every policy is narrowed at **build** when the latch is up. That needs every
builder site to remember it, and every new builder too (adversary 4). It also misses a policy that was built before
a later `MAXIM_OFFLINE` raise, which §6 explicitly honours ("a later raise counts"). The socket latch keeps this
fail-closed, but then the typed `POLICY_CEILING` refusal and the roster are wrong: the model is still offered
`http_fetch` and `bash`, which only ever fail at the hook, so #1412's lesson recurs.

**Fold:** the dispatch seam (S2) ANDs `policy.ceiling` with `OFFLINE_CEILING` whenever `is_offline()` is true, and
`permits()` reads the same AND. That makes one site instead of N, and it is live-latch correct.

### S6. `MAXIM_OFFLINE` duplicates the existing raise-only `MAXIM_INTERNET_LOWERED`, and needs one truthy parser across processes.

- `utils/internet_access.py::lower_internet_for_session` / `session_internet_lowered` (#1345) is already a raise-only,
  env-carried, child-inherited "internet off" signal, with its own conftest scrub. The notes never mention it.
  - **Define the relation:** offline implies internet lowered, so the internet tools' own `disabled_reason` and the
    roster agree. Otherwise state why they stay separate. Front-gate: name why V5 cannot ride on it. It can't fully,
    since it covers neither HF nor the LLM, but it can be implied by it.
- §6: `maxim/__init__` does "one cheap, stdlib-only check of `os.environ.get("MAXIM_OFFLINE")`". The canonical
  parser, `cluster_bias_annotation.annotation_disabled_via_env` ("1/true/yes/on", case-insensitive), lives in
  `maxim.prompts`. A hand-written stdlib check is a second parser.
  - **Failure:** `MAXIM_OFFLINE=yes` engages the parent through `is_offline()` (the canonical parser). A spawn child
    whose `__init__` check is `== "1"` stays unlatched. That fails open across exactly the boundary self-engage
    exists for.
  - **Fold:** move the canonical truthy parser into a stdlib-only leaf module that both use, or pin `__init__`'s copy
    to it with a parametrised test.
- Also: `maxim.utils.offline` cannot "stay clear of `utils/__init__`'s eager imports" (the v3 #10 fold). Importing
  any `maxim.utils.X` runs `utils/__init__.py`, which is 51 modules. Either place the latch at `maxim/_offline.py` or
  say that cost is accepted.
- Add `MAXIM_OFFLINE` to the owning brief's env table (llm-routing.md or runtime-tools.md), next to
  `MAXIM_INTERNET_LOWERED`.

### S7. The `[engineering]` invariant and its Regression guard line are stale, and they never appear in the notes.

The only statement is advisor_mode.md §"It is an `[engineering]` invariant", whose guard names:
- "builder-refusal tests (… prefetch root)": moot under V1, since there is no loop;
- "item 10's census test": the notes demote the census to "explanation, not enforcement" and make the audit-hook red
  gate the enforcement;
- no `file::symbol` paths, which `lint_claude_md_invariants.py` style requires.

The notes add at least four invariant-shaped commitments, and they need to state them, each with a guard:
1. `Tool.effect` declared on the class's own `__dict__` (guard: the declaration and sink AST tests);
2. `build_executor(*, policy=)` joining the canonical-builder table in runtime-tools.md §3;
3. the offline latch;
4. the atonia composition (guards: the per-layer deletion sweep, the env read-set, the import closure and the
   no-effect test).

**Fold:** put a §"Invariants this adds" in the notes, with draft stub text and `Regression guard: tests/…::…` lines,
and mark the advisor_mode.md guard line superseded.

### S8. Ride the censuses on the existing lint infrastructure, starting with the swallow census.

§6's "swallow census" (every handler that catches a base of `OfflineRefusal` around a census site) is a narrower
variant of `scripts/lint_no_silent_swallows.py`. That lint already does per-function counts, move records and a
CI-printed total. The site census, the `unrestricted`/`deny` caller pins, the registry-construction pins and the
direct-`.execute` pins are each described as an "AST/grep census pinned in JSON", which adds up to five new bespoke
baselines.

**Fold:**
- State per census whether it is a `scripts/lint_*.py` in the CI lint job, with its unit test and its baseline
  (which `scripts/prepr.sh` then runs automatically), or a pytest.
- Implement the swallow census as a mode of `lint_no_silent_swallows`, not a second walker.
- Name the JSON files and their location. They are repo data rather than files Maxim persists, so they follow the
  `architecture_baseline.json` precedent (`baseline_format_version`) and not the `_format_version` rule.

### S9. The report write path contradicts the parent, and the log helper has no named home.

- advisor_mode.md §"Atomicity" says "The report file is atomic (`atomic_write_json`)". Notes §4.2 says "The log and
  the report are written through that fd" (the `O_DIRECTORY|O_NOFOLLOW` dir fd). `utils/atomic_io.py::atomic_write_json(path, …)`
  takes a path, not a `dir_fd`. So either `atomic_io` gains a `dir_fd` variant, which persistence-config.md owns, or
  the report is hand-rolled, which breaks the atomic_io invariant and raises `lint_atomic_io_ratchet`'s count.
- §5 says no fsync-raising append helper exists, which is correct (`_fsync_path` and the `atomic_write_*` helpers
  log and skip `OSError`). But it never says where `AdvisorLog` lives.
- **Fold:**
  - Add `atomic_write_json_at(dir_fd, name, payload, *, fsync_strict=True)` and a strict `append_line_fsync(fd, …)`
    to `utils/atomic_io.py`, with a docstring that names the two fsync policies (lenient, strict) and who uses each.
  - Or state why the advisor package owns a private writer and record it on the ratchet.
- AD4 already gives the report `_format_version`, so restate that the report and the log header both carry it.

---

## NIT

- **N1. `closed_local.py` duplicates `_LlamaCppBackend`'s `Llama(**kwargs)` construction**
  (`models/language/llama_backend.py::_LlamaCppBackend._ensure`: the n_ctx hint clamp, `n_gpu_layers`, `n_threads`,
  `seed`). The rationale rules out `LLMRouter`, correctly, but never considers `_LlamaCppBackend`.
  - Ruling `_LlamaCppBackend` out is right: it does lazy loads, a consent download, and `return False` swallows.
  - But the kwargs builder should be one shared helper, so a future GGML/Blackwell flag is not added to only one of
    the two.
  - Also state that `closed_local` gets `llama_cpp` through `require_optional_dependency("llama_cpp", extra="llm-llama")`.
- **N2. Erosion guard for `closed_local`.** The CI backend-import grep (`test.yml` "1.0 guard promotion") matches
  only `anthropic|llama|openai|transformers_backend`. Any `runtime/` path could import `closed_local` as an
  un-routed LLM, and nothing would notice. Add `closed_local` to the grep, with an allow-list of exactly
  `src/maxim/advisor/`, and make the lesson's exception entry name that grep.
- **N3. `maxim.register_tool` is a ✅ stable verb** (`docs/user/stable_api.md`). Two consequences are
  user-visible on a stable surface and should be named in the CHANGELOG and stable_api.md:
  - an undeclared third-party tool is `ALL`, so it is refused by `api.run`'s new `only`/`pinned` ceiling;
  - "same-name re-registration replaces" becomes "refused below `ALL`".

  V3's "none is in stable_api.md" covers the constructors, not this.
- **N4. `Effect` in persisted records.** The log's `refusal.effect` must serialise as sorted flag names, never as
  the `Flag` integer. The integer depends on declaration order, and `GOVERN` was just inserted. Pin that in the §5
  record spec.
- **N5. Rule T scope.** §3 names `build_executor` and `_start_agentic_runtime`. `api.py::run` (276) and
  `AgentFactory.create_full_agent` (204) are also pinned in `utils/function_length_baseline.json`, and both are
  touched.
- **N6. Ledger and parent leftovers.**
  - §9's v2 #3/#4 and v3 #1/#2 rows describe mechanisms that V8 deleted: realpath-keyed executables, the live-child
    endpoint, and listener ownership "via psutil". Mark them "superseded by V8", so nobody resurrects psutil from the
    ledger.
  - advisor_mode.md still says "Genuinely new: `Tool.effect` … a sealed registry". It leaves out the six other new
    mechanisms (`closed_local`, `AdvisorLog`, the offline latch, `current_policy`, the exemption list, the typed
    `ToolErrorKind` members), and its Context row still promises "prefetch rooted in an empty directory". Add
    Superseded markers there too. The banner's "notes win" covers it legally, but it is the paragraph a reader of the
    parent trusts.
  - `tools/effect_exemptions.json` is consulted only by a test, so it belongs under `tests/`, not in the shipped
    `src/maxim/tools/`.

## What holds (no finding)

- **V1** (no loop) is the right root-cause cut. The loop's channels are owned by the loop, and fencing them one by
  one would be the band-aid.
- **V2** (a set, `exceeds()`, `Effect(0)` refused, own-`__dict__` only) is the right type-level push of a
  silent-no-op invariant.
- **V3** hard keyword-only `policy=` is the canonical-builder pattern, applied correctly: `None` only where no
  executor is built, never translated.
- **V8** is the bird's-eye cut that the convergence rule asks for after three rounds on one surface.
- `ToolPolicy` being in-memory, so CC3 does not apply and no `_format_version`, is correct. Ingress and native
  networking are honestly scoped out.
- `OfflineRefusal(RuntimeError)` is the right type, to avoid the narrow `OSError` catches.
- `http.py` remains the HTTP chokepoint for Maxim's own calls, with the urllib3 bypass moved onto it. That is the
  root-cause fix and is consistent with the invariant.
- The §0 threat model and §6's stated boundary do not overclaim. B2 is the one place where a doc sentence ("reads no
  env", "config never opened") is false against the planned placement.

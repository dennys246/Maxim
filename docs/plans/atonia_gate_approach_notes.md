# The atonia gate: approach notes for items 2, 3, 1, 5 and 10

**Status:** DRAFT v5, approach notes (CLAUDE.md "Faster without lowering the bar", item 1). Owner decisions V1–V7
were made 2026-10-24. Session V, 2026-10-24, paper only: nothing in `src/` changes until the fourth delta round
(scoped to v5's V8 deletion and the AST-check fold) returns STOP.
- **v1** was written from a read-only census at `d9dffe6e` (origin/main, 2026-10-24).
- **v2** folds the three adversarial passes: [items 2+3](reviews/atonia_gate/items_2_3_adversarial.md) (sound with
  conditions), [items 1+5](reviews/atonia_gate/items_1_5_adversarial.md) (DO-NOT-BUILD as written) and
  [item 10](reviews/atonia_gate/item_10_adversarial.md) (DO-NOT-BUILD as written).
- **v3** folds the [v2 delta round](reviews/atonia_gate/v2_delta.md) (not STOP: eight fail-open folds, none
  DO-NOT-BUILD).
- **v4** folds the [v3 delta round](reviews/atonia_gate/v3_delta.md) (not STOP: seven fail-open, each an incomplete
  v3 fold in an already-covered class). The rounds are converging: 60 → 18 → 12 findings, the same classes
  narrowing each time.
- **v5** folds the [v4 delta round](reviews/atonia_gate/v4_delta.md) (not STOP: three fail-open). A bird's-eye
  check found that most fail-open findings since v2 lived on one surface, the offline subprocess and endpoint
  allowance. Owner decision V8 (strict) deletes that surface instead of patching it again.

§9 maps every finding to its fold.

Parent plan: [advisor_mode.md](advisor_mode.md) (AD1–AD9 decided 2026-10-17). The gate ships in **its own release,
the next free patch number** (owner decision 2026-10-24: 1.3.4 is the robot and recall fixes). Paths are under
`src/maxim/`.

## 0. The threat model

The gate defends against four adversaries:
1. **The model's output:** tool names, arguments, JSON fields, free text, and free text shaped like an approval or a
   control token.
2. **Inbound text:** event payloads and anything that reaches the prompt.
3. **The process's surroundings:**
   - env vars: `MAXIM_*`, the `HF_*`/`YOLO_*` family, proxy vars, `HOME`, `XDG_*`, `TMPDIR`;
   - `~/.config/maxim/config.json` and `llm.json`;
   - the CWD and the data home;
   - state an earlier run left behind (`cost_state.json`, caches).
4. **Future code:** a new tool, registry write path, fetch site, executor construction or import that nobody routes
   through the gate.

**Out of the model: arbitrary code in the process.** The gate does not defend against arbitrary Python or native code
in-process, and neither do any of the variables that load code: `PYTHONPATH`, `PYTHONSTARTUP`, `LD_PRELOAD`,
`DYLD_*`, `LLAMA_CPP_LIB_PATH`, `GGML_*`, a planted `sitecustomize`. Python cannot seal an object against its own
interpreter, so that layer is the OS sandbox (AD5, layer 3). One cheap fail-closed exception: the observer's builder
refuses to start when any variable on a pinned list of code-loading variables is set (§4.2), because the observer can
check that and the operator never needs them in Profile B. `PYTHONPATH` is not on the list (containers set it); it is
the sandbox's.

**Ingress is out of scope for item 10.** A leader bound to `0.0.0.0` and inbound comms are the inbound row's
(item 4, and the sandbox).

## 1. Owner decisions (decided 2026-10-24: V1–V7 all as recommended)

The recommendation is listed first in each row, and it is what was decided.

| # | Question | Recommendation | Why it is a decision |
|---|---|---|---|
| **V1** | Does the observer run the agent loop? | **No.** `session.ingest(events)` is a bounded turn (§4.2): one framed prompt, one JSON completion from a closed local backend, then each call the model names dispatched through the sealed executor (at most N), then the report. | The loop carries channels that would each need closing, most inside the fence (§4.1). The cost is no Hippocampus, NAc or loop cognition in this release. That matches item 7 and Profile B, but it narrows "perception and cognition run normally" to "perception and one deliberate turn". |
| **V2** | Is `Tool.effect` a ladder or a set? | **A set** (`enum.Flag`), with **seven** defined flags (§2.1, adding `GOVERN`). A tool is admitted iff it has no flag outside the ceiling. | On a ladder, a NETWORK ceiling admits WRITE_LOCAL. That breaks both Profile A and #1431's Talk ceiling. `GOVERN` is new: the tools that change Maxim's own authority (mode, autonomy, internet, directory) belong to none of the six classes. |
| **V3** | Item 3 in a patch: a hard break or a soft one? | **Hard on all three surfaces** (`Executor`, `build_executor`, `AgentConfig`). None is in `docs/user/stable_api.md`, and the stable verbs (`run`, `create.*`, `load.*`) build their own `AgentConfig`, so they pass a policy themselves. | v1 proposed a warn-only `AgentConfig`. The 2+3 pass showed that translating `None` to a permissive policy turns the O1 red gate green with no behaviour change, and is itself an unpinned allow-all. External code that builds `AgentConfig` directly breaks with a clear `TypeError`; the CHANGELOG says so. |
| **V4** | The observer's LLM | **An in-process GGUF only, in this release,** through a closed backend entry point inside `models/language/` (§4.2). | No socket, no spawned server, no orphaned GPU holder. A loopback llama-server is ruled out under offline by V8. |
| **V5** | How does offline cross into sub-processes? | **A raise-only `MAXIM_OFFLINE=1`**, plus the offline vars on the host-tool env allow-list. Autouse conftest scrub in the same commit. | advisor_mode.md rules out a `_FIELD_TO_ENV` alias, because an alias could switch offline OFF. A raise-only signal keeps that property. |
| **V6** | Tools run outside the executor in non-observer runs | **Route `auto_sense` through the executor's policy check in this release** (`runtime/loop_perception.py`, not fenced). **Census the rest** (`bio_enrichment`'s two reflex dispatchers, `OrchestratorActorTool`'s ephemeral tool), with a follow-up issue. | v1 censused all four. The 2+3 pass showed that `auto_sense` bypasses any ceiling a loop sets: `auto_fire` can be set per instance, and `SensePresenceTool` triggers LLM imagination. That makes #1431's Talk ceiling a lie. The other two are sim-only and in `orchestrator.py`'s wiring. |
| **V7** | Offline on the CLI (`llm.offline`) and model-chosen commands | **Offline also drops every tool with NETWORK, COMMS or the undeclared `ALL` from every executor's policy.** The policy is narrowed at build, when the latch is up, to the **enumerated** ceiling `READ|SELF_SCOPE|WRITE_LOCAL|ACTUATE|GOVERN`, never a complement, so a future eighth flag is excluded (fail-closed). Process runners are `ALL` by §2.1, so they drop too, and the hook refuses any spawn regardless (V8). `ACTUATE` stays in the ceiling on paper only: the latch refuses the robot's connect anyway, so an offline robot's tools fail closed with `OfflineRefusal`. | `bash` and the code tools run commands the model chooses, which no census can see. Without this, "offline refuses every fetch" is false on the CLI path. It is a decision because offline then visibly removes tools. |

| **V8** | Under offline, may Maxim run any subprocess or connect to any endpoint? (asked 2026-10-24 after the v4 delta round) | **Nothing at all.** No subprocess, no socket endpoint, no allow-list. The CLI's `llm.offline` needs an in-process model (the `llama_cpp` or `transformers` backend) and refuses at startup otherwise. The git tools drop. | The spawned-server allowance and the pinned git argv drew most of the fail-open findings in three consecutive rounds (v2 #3/#4, v3 #1/#2/#7, v4 #1–#4): endpoint revocation, listener ownership, and git's config-driven filters running code the latch cannot see. Neither the observer nor AvServ needs them. |

**Flagged, not asked now:** `maxim.research` already exists (`api.py::research`, Experimental), so Profile A's builder
needs another name when items 13 and 14 start.

## 2. Item 2: `Tool.effect` and the registry ceiling

### 2.1 The flags, defined

A tool's effect is **everything it can cause**, through itself or through any collaborator it holds or is wired to
later, not what it is for.

| Flag | Means |
|---|---|
| `READ` | Reads state; changes nothing outside process memory. |
| `SELF_SCOPE` | Writes only Maxim's own persisted state under the data home (memory, NAc, its own logs). |
| `WRITE_LOCAL` | Writes, moves or deletes files outside the data home. **Not "runs processes or code":** a tool that spawns a process, or runs code in-process (`exec`, `eval`, or `importlib` of a model-chosen name), can cause anything, so it declares `ALL`. The one exception is a fixed argv on the pinned exemption list `tools/effect_exemptions.json`, with a reason (the git tools: fixed argv plus `tools/base.py::GIT_HARDENING_ARGS`). The exemption holds **online only**: under offline the hook refuses every spawn (V8), so the git tools drop. Git's config-driven filters run code even with that hardening. **Checked:** an AST test over the `Tool` subclass closure fails any class whose methods reach a sink, unless it declares `ALL` or is on the exemption list. **Sinks:** `subprocess.*`, `os.exec*`, `os.system`, `os.popen`, `os.spawn*`, `os.posix_spawn*`, `os.fork*`, `pty.spawn`, `asyncio.create_subprocess_*`, `multiprocessing.*`, `exec`, `eval`, `compile`, `importlib.import_module`, `__import__`, `runpy.*` and `ctypes.*`. **Reach:** the call graph across `src/maxim` modules (resolved imports, `self.` methods, module functions), not only the same module. **Stated blind spots:** `getattr(os, "system")`-style dynamic lookup, callables passed in as arguments, and third-party code. Each is the future-code adversary, which the review rounds and the census lists, not this test, catch. |
| `NETWORK` | Any egress off the machine, DNS included, GETs included. |
| `COMMS` | Reaches a person or a channel (messages, calls, email). |
| `ACTUATE` | Physical output: motion, sound, a display. |
| `GOVERN` | Changes Maxim's own authority: mode, autonomy level, internet switch, allowed directories, the tool set. |

`Effect.ALL` is the union of all seven. `Effect(0)` is refused at class creation, so no tool can pass every ceiling by
declaring nothing.

Examples:
- `bash`, `execute_file` and the sub-sim tool: `ALL`.
- `http_fetch`: `READ | NETWORK`.
- `respond`: `READ | WRITE_LOCAL | ACTUATE`. It writes files and drives TTS and the speaker.
- `ModulatorAffordanceTool`: `ALL`. Its backends do network, speech and motion, and it can spawn tools.
- `SensePresenceTool`: `READ | SELF_SCOPE`. It triggers imagination, and the design pass checks whether that writes.
- `internet_access_toggle`, `AutonomyLevelTool` and the mode tools: `GOVERN`.
- The observer's `ProposeActionTool`: `READ`. It only validates and returns; the turn writes the log (§5).

### 2.2 The mechanism

- **The declaration** is `effect: ClassVar[Effect]` in the **class's own `__dict__`**. An inherited value does not
  count, because overriding `execute` must not inherit READ.
- **The record.** `Tool.__init_subclass__` records it, or `ALL` when it is absent, in a module-private
  `WeakKeyDictionary`. A non-`Effect` value or `Effect(0)` raises `TypeError`.
- **The lookup.** `effect_of(obj)` reads `type(obj)`'s record. A type with no record (a duck-typed tool, a mock, a
  class made outside `Tool`) is `ALL`.
- **Mutation is visible.** The mismatch check compares the record with `type(obj).__dict__.get("effect")` only, never
  an inherited `getattr`. It refuses admission loudly when the two differ, which catches `cls.effect = ...` after
  creation. `create_stub_tool` and `@tool` put `effect` in the class body they build.
- **The check** is one helper: `exceeds(e, ceiling) = bool(e & ~ceiling)`. `Flag` has no subset `<=`, and `IntFlag`
  compares integers (WRITE_LOCAL ≤ READ|NETWORK), so neither operator is used anywhere.
- **The registry ceiling** is fixed at construction: a read-only property, `ALL` by default.
  - **Admission is layer 1 and is best-effort structural.** `_admit` runs on every in-tree write: `__init__(tools=)`,
    `register`, `_register_scene_locked`, and `SimToolRegistry.register` through a shared mixin.
  - **Dispatch (§3) is authoritative.** An out-of-tree subclass that skips `_admit`, or a later `activate_scene`, is
    still refused at dispatch.
- **Same-name replacement.**
  - Under any ceiling below `ALL`, a replacement refuses.
  - Under `ALL`, it is allowed, as `register_tool` needs, and it logs a structured WARNING naming both classes.
- **Entity generation is all-or-nothing.** `generate_tools_for_entity` builds the full list, checks every tool, then
  registers. The DM runtime no longer leaves half a set behind.
- **`SimToolRegistry.get`** raises `KeyError` for an unknown name under a ceiling below `ALL`, instead of fabricating a
  `_FallbackRedirectTool`. Its one caller (`orchestrator.py`) keeps the fallback under `ALL`.
- **The prompt roster** (`permits`) applies the ceiling too, so the model is never offered a tool that is always
  refused (#1412's lesson).
- **Refusals are never silent.** A structured WARNING event at `api._inject_registered_tools`,
  `dm_runtime._register_entity_tools` and `Executor`'s Mechanism-B acquisition.
- **The declaration test:** every concrete `Tool` subclass reachable from `src/maxim` (an AST closure; the 2+3 pass
  counts 83) declares its own non-zero effect. The test also calls the two factories and checks their output.

**Inputs, and how each could lie.**

| Input | How it could lie | Answer |
|---|---|---|
| `type(tool).effect`, reassigned | `create_stub_tool` already reassigns class fields | Record versus own-`__dict__` mismatch refuses |
| Inherited `effect` | Subclass of a READ tool writes | Own `__dict__` only; undeclared means `ALL` |
| `Effect(0)` | Passes every ceiling | Refused at class creation |
| Instance attributes (`effect`, `auto_fire`, `kind`) | Set per instance | `effect` is never read from the instance; `auto_fire` is moot once `auto_sense` is checked (V6); `kind` is not the gate |
| A collaborator wired after admission | The orchestrator attaches backends later | The declaration covers the collaborator slot, whatever is plugged in |
| A same-name replacement | Swaps a READ tool for an acting one | Refused below `ALL`, warned at `ALL`; the observer pins instances (§3) |
| A tool with no record | Duck-typed or mocked | `ALL` |
| An out-of-tree registry subclass | Skips `_admit` | Dispatch-time check |

## 3. Item 3: a required `ToolPolicy` on every executor

**The defect (O1).** `Executor.__init__(tool_registry, pain_detector=None, tool_pain_bridge=None, permissions=None,
embodiment=None, …)` accepts `None`, and a bare `Executor` with no mode source restricts nothing.
`build_executor(*, permissions=)` is required but accepts `None`. `AgentConfig.permissions` defaults to `None`. Seven
`src/` sites pass `None`. `api.run`'s `learning=False` branch calls `build_executor` directly.

**The design.**
- **`ToolPolicy` is a plain class with `__slots__`, not a dataclass**, so `dataclasses.replace` cannot widen it. It has
  no public constructor: `__init__` takes a module-private key.
- **It is immutable and unforgeable.** `__slots__` blocks only new attributes, not assignment, so:
  - **`__new__` takes and checks the private key.** `ToolPolicy.__new__(ToolPolicy)` without it raises, so an
    unsealed instance cannot be filled by assignment;
  - `__setattr__` and `__delattr__` raise **unconditionally**, and `__init__` writes through `object.__setattr__`;
  - `__init_subclass__` raises, so no subclass can pass `isinstance` with a permissive `__setattr__`;
  - `Executor`, `build_executor` and `AgentConfig` check `type(policy) is ToolPolicy`, not `isinstance`;
  - the slot descriptors (`ToolPolicy.ceiling.__set__`), `__class__` assignment from a class with the same slots,
    and the key reached through introspection (`ToolPolicy.__new__.__globals__`) are all in-process code. Each is
    equivalent to `object.__setattr__` and out of the model (§0), and the docs say so. Future code would not write
    any of them by accident;
  - `pinned` is snapshotted as `MappingProxyType(dict(instances))`, so a caller's dict cannot add pins later;
  - `__copy__` and `__deepcopy__` return `self`, and `__reduce__` raises.

  Tests cover construction, assignment, and mutation of the source dict. It holds `permissions: AgentPermissions`,
  `ceiling: Effect`, `pinned: Mapping[str, Tool] | None` and `reason: str`. It lives in memory only, never persisted.
  Its docstring says CC3 does not apply, and the CC3 list records that. Its factories:
  - **`unrestricted(*, reason)`:** ceiling `ALL`, no allow-list. The mode gate still runs.
  - **`deny(names, *, reason)`:** the console's deny-only `tools.deny`. It keeps today's working deny list instead of
    folding it into `unrestricted`. It is permissive and is censused like `unrestricted`.
  - **`only(names, *, ceiling, reason)`.**
  - **`pinned(instances, *, ceiling, reason)`:** authorises **objects**, not names. Dispatch requires that the
    resolved tool `is` the pinned instance. Aliases are refused.
  - **`narrowed(*, ceiling=None, only=None, deny=())`:** never wider. It ANDs ceilings, intersects allow-lists, unions
    deny lists, keeps the stricter clearance, intersects `kind:` selectors, and keeps any pin. SEM rules are carried
    unchanged; `can_access_sem` has no caller, which is filed. A ceiling is never built by complement.
- **`Executor.__init__(registry, *, policy: ToolPolicy, …)`.** Everything after the registry is keyword-only, so a
  stale positional call cannot bind to `embodiment`. Anything that is not a `ToolPolicy` raises `TypeError`.
  `build_executor(*, policy=)` is the same: required, no default (V3).
- **`AgentConfig.policy`** is `field(kw_only=True)`. It has to be keyword-only, because every other field is
  defaulted and a required field cannot follow them. It has no default, so `AgentConfig(...)` without it is a
  `TypeError`.
  - A config whose path builds **no** executor (`create_npc_agent`, where `create_agent` builds no `Executor`,
    the D73 shape) passes `policy=None`.
  - `create_full_agent(with_executor=True)` refuses `None` with a `TypeError` at the executor build.
  - `None` is therefore never translated into a policy. It means only "no executor is built".
  `ConcurrentExecutor` wraps an `Executor` and takes no policy of its own.
- **Dispatch checks the object it runs.**
  1. Resolve the alias. Aliases stay live, since the DM runtime adds them after build; a pinned policy refuses them.
  2. Get the tool **once**.
  3. Check `exceeds(effect_of(tool), policy.ceiling)`, the permissions, and any pin on that object.
  4. Run that object.

  This closes the find/get race.
- **Refusals are typed.** `tools/base.py::ToolErrorKind` (an `Enum`) gains `NOT_REGISTERED`, `POLICY_CEILING`,
  `POLICY_PERMISSIONS`, `POLICY_PIN` and `MODE`, and every refused dispatch carries one, plus the layer and the effect.
  The `effect` of an unregistered name is `"unknown"`; no lookup builds a registry to find it.
- **A policy refusal is not a tool failure.** It books no pain, so `_run_started`'s `error_kind or EXTERNAL_FAILURE`
  pain mapping and `proprioception/pain.py`'s mapping skip `POLICY_*`. It books no NAc outcome and no
  `_tools_hallucinated`. It still counts toward `_consecutive_failures` and the rejection follow-up cap, so a model
  that keeps naming a refused tool still stops. How existing mode refusals are booked is unchanged.
- **`auto_sense` (V6)** calls `executor.check(tool)`, the same check without dispatch, before `.execute()`.
  `check()` does not set `current_policy`, so an auto-fire composite refuses. That is fail-closed, and the docs say
  so.
- **Composite tools take the dispatching policy, not a registration-time one.** `Executor` sets a contextvar,
  `current_policy`, for the duration of a dispatch. `SpawnSubSimulationTool` reads it, refuses if it is absent, and
  builds the child with `policy=current.narrowed(...)` and a child registry whose ceiling is the child policy's.
  The read happens synchronously in `execute()`, before any thread starts: in 3.12 a `threading.Thread` does not
  inherit context. `ConcurrentExecutor` submits `Executor.execute` itself, so the variable is set inside the pool
  thread (verified by the delta round).
  v1 handed the policy to the tool at registration. That was the narrator's executor's policy, not the policy of the
  executor dispatching the call.
- **Call sites:**
  - `api.run` (both branches): `only` or `pinned`, decided per branch, from #1342's direction (workspace-only).
  - `create_full_agent`: from `AgentConfig`.
  - The foundry, the Minecraft harness and both embodied builds: `unrestricted(reason=...)`.
  - `console/handle.py`: `deny(...)`, or `only(...)` once #1431 lands.
  - `cli.py`, the orchestrator's AUT and narrator, `create.py` and `load.py`: through `AgentConfig`.
- **Under offline (V7),** every policy is narrowed at build to drop NETWORK, COMMS and undeclared tools.
- **The census tests** (AST over `src/`; `scripts/` out of the census, so `scripts/orient_substrate`'s nine direct
  `tools[action].execute()` calls are named in the test as out of scope):
  - every `Executor(` / `build_executor(` / `AgentConfig(` passes `policy=`;
  - every caller of `unrestricted(` or `deny(` is on a pinned list with its reason;
  - every `ToolRegistry(` / `SimToolRegistry(` outside `build_tool_registry` is pinned;
  - every direct `.execute(` / `.run(` on a tool outside `Executor` is pinned.

  **Stated blind spots:** `getattr`, `functools.partial`, `executor.submit` and `**kwargs` splats. The dispatch-time
  check is the backstop for all four.
- **Rule T:** `build_executor` and `_start_agentic_runtime` are pinned, and the policy plumbing is extracted into a
  helper to pay for the touched lines.

**Inputs, and how each could lie.**

| Input | How it could lie | Answer |
|---|---|---|
| `policy=None`, or a duck-typed policy | Forgotten, or faked | `TypeError` |
| `dataclasses.replace(policy, ...)` | Widens a policy | Not a dataclass; no public constructor |
| A direct `ToolPolicy(...)` | Skips the census | A private key is required |
| `TOOL_ALIASES` (mutable, global) | Remaps an allowed name to an acting tool | The check runs on the resolved object; a pinned policy refuses aliases |
| The registry, mutated after build | A DM-thread re-registration | Resolve once, check, then run the same object |
| A composite's policy | The narrator's policy instead of the dispatcher's | The `current_policy` contextvar at dispatch; refused if absent |
| A wrapper (`FearGated`, `PainInterceptorExecutor`, `AnticipatoryPain`, `Instrumented`) | Proxies through `__getattr__` | It wraps a checking `Executor`; the observer has none |
| An `auto_fire` instance flag | Runs a tool outside dispatch | `auto_sense` calls `executor.check` |

**Red gates:**
- the O1 strict xfail: `maxim.run`'s executor carries a non-permissive policy, which a translated `None` cannot
  satisfy;
- the constructor `TypeError` tests;
- the race probe;
- the composite probe: a spawn under a narrow policy builds a child that refuses `bash`;
- the `auto_sense` probe;
- the census.

## 4. Item 1: the atonia composition (`maxim.advisor`)

### 4.1 Why the observer does not run the loop (V1)

On `run_agentic_loop`'s path the census found:
- **(e)** the SUPERVISED "yes", which non-interactive runs inject themselves;
- **(f)** plan approval from free text (#1185);
- **(g)** level changes: `approve_autonomy_request`, `set_level`, `from_dict`, `AutonomyLevelTool`;
- **(h)** `run_agent_loop` and prefetch's `_execute_cached`, calling `executor.execute` outside autonomy;
- prefetch's `find` and `open()` reads. `loop_controller.py` passes `prefetch_for_input(..., cwd=os.getcwd())` on
  every input, and there is no off switch;
- the CWD state file;
- `auto_sense`, the reflex dispatchers and the Default Network.

Replacing `ProposalQueue` closes the drain path only. Under V1 none of these is constructed.

### 4.2 The design

- **The API.** `maxim.advisor` (Experimental, lazily exported):
  - `open(*, home_dir, model, report_dir, cursor=None, max_calls=8, turn_timeout_s=120)` returns `AdvisorSession`;
  - `session.ingest(events) -> AdvisorReport` (AD4's schema);
  - `close()` writes the log's close record.

  `AdvisorSession.__init__` takes a private token.
- **Refusals at `open()`, each a typed `AdvisorCompositionError`:**
  - a pinned code-loading variable is set:
    - **acting at exec**, so they only mark a hostile env: `LD_PRELOAD`, `LD_AUDIT`, `DYLD_INSERT_LIBRARIES`;
    - **acting at `dlopen`**, when `libllama`/`libggml*` load at `open()`: `LD_LIBRARY_PATH` and **every `DYLD_*`**
      (by prefix; dyld applies `DYLD_LIBRARY_PATH` by leaf name even to an absolute `CDLL` path);
    - **read by llama-cpp** (Python or C): `LLAMA_CPP_LIB_PATH`, `LLAVA_CPP_LIB`, `GGML_*` (`GGML_BACKEND_PATH`
      loads code), `LLAMA_*`;
  - `model` is not an existing regular file;
  - `home_dir` or `report_dir` is missing, not writable, or overlaps the other. Each is opened **first** with
    `O_DIRECTORY|O_NOFOLLOW`, and every check (`fstat`, `st_dev`/`st_ino` overlap and ancestry, writability) runs on
    the fd, so there is no `realpath` TOCTOU window. The log and the report are written through that fd;
  - anything is passed for a robot, an embodiment, tools or a registry. The signature takes none of them, and a test
    pins the signature.

  It engages item 10's latch before anything else is built.
- **The LLM: a closed backend entry point**, `models/language/closed_local.py::open_closed_local_backend(gguf_path, *,
  n_ctx, n_gpu_layers, seed)`. It lives inside `models/language/`, so the "LLM access through the router package"
  invariant holds, and the CLAUDE.md exception list gets it as one named entry. **It is not `LLMRouter`.** The 1+5
  pass ran `LLMRouter(LLMConfig(...))` under an audit hook, and the router is not closed:
  - its `CostTracker` reads `MAXIM_DATA_HOME`/`HOME`;
  - it creates `~/.maxim/util`, reads and renames `cost_state.json`, and registers an `atexit` write;
  - it reads `MAXIM_INFERENCE_LOCK_TIMEOUT_S`, `MAXIM_LLM_MAX_COST_*` and `MAXIM_AGENTIC_*`;
  - `LLMConfig` defaults the profile to mistral, which `ensure_available` would download or substitute.

  The closed entry point constructs llama-cpp-python's `Llama` **eagerly**, at `open()`, from the path alone. It takes
  no profile, does no lazy load, has no cost tracker and no `atexit`, and reads no env. It exposes exactly one call:
  `complete_tokens(prompt_ids, *, grammar, max_tokens, max_prompt_tokens, deadline)`. It does not import
  `agents.llm_worker` or anything under `runtime/`.
  - **The bound is two-part.** Prompt prefill (`eval`) cannot be interrupted, so `max_prompt_tokens` is checked
    before `eval`, and an over-long prompt returns `status: "error"` with `reason: "prompt_too_long"`. After that,
    `stopping_criteria`, called per sampled token, enforces `deadline` (`turn_timeout_s`).
  - A deadline stop sets a flag on the criterion, and the status comes from that flag, because llama-cpp reports a
    criterion stop as `finish_reason "stop"`, the same as a natural stop.
  - The docs say plainly that prefill is bounded by the token cap, not by the clock.
- **The import closure is pinned.** A subprocess test records `sys.modules` after `open()` plus one `ingest()`, and it
  must exclude `runtime.agent_loop`, `agents.llm_worker`, `default_network`, `comms`, `memory`, `hivemind`, `peer`,
  `mesh`, `tunnel`, `huggingface_hub`, `httpx` and `maxim.utils.http`. It is a ceiling that may only fall.
- **The prompt is built as token ids, not as a string.** Sanitising strings fails: `frame_tool_output`'s NFKC turns
  fullwidth `＜｜im＿end｜＞` into `<|im_end|>`, and deleting a token string can recombine one. So:
  - the trusted template goes through `Llama.tokenize(..., special=True)`;
  - each untrusted segment (each event's `text` and canonical-JSON `payload`, already framed by
    `frame_tool_output(..., external=True)`) goes through `tokenize(seg, add_bos=False, special=False)`;
  - **every id in an untrusted span is then checked with `llama_token_get_attr`.** `special=False` skips only
    CONTROL and UNKNOWN, so USER_DEFINED tokens still come out of plain text: all 999 in command-r's vocab, all 290
    in qwen2's, and `</s>` in phi-3's. The attribute is a **bitmask**. An id is
    refused when it has the `CONTROL`, `USER_DEFINED` or `UNKNOWN` bit. `NORMAL` and `BYTE` are plain text: newlines
    and emoji bytes in SPM and phi-3 vocabs are BYTE, and rejecting them would reject every multi-line event. A
    refused id is handled in one of two ways:
    - **Escape.** The offending character span is re-tokenized one character at a time, with the same flags, and
      re-checked. This may change whitespace, because SPM adds its space prefix per call (`</s>` → ` < / s >`).
    - **Refuse.** If an id is still not NORMAL, the event goes into the report's `inputs` as `rejected:
      "special_token"`. That keeps it visible, so dropping an event cannot hide it;
  - the concatenated `List[int]` goes to `create_completion`, which accepts it, with `grammar=` built from the closed
    schema.

  The vocab test enumerates the loaded GGUF's tokens **by attribute**, not by "special-looking" strings: every
  non-NORMAL token's text, in plain, fullwidth, split and nested forms, is fed as event text. It runs against the
  vendored command-r, phi-3, qwen2 and llama-bpe vocabs. It asserts that no id with a refused attribute bit appears
  inside an untrusted span, and that no BOS appears there either. The vocabs come from llama.cpp's `models/`
  directory at a pinned tag, as test fixtures, since the wheel does not ship them. No vendored vocab has a
  single-character special token, so a synthetic test, with the attribute lookup monkeypatched to mark one
  USER_DEFINED, exercises the refuse branch. The router's
  `TOOL_PROMPT|` path, which invents `respond` actions, is never used.
- **The turn.** One `complete_tokens` with a closed schema, enforced by the grammar: `{happening, would_do,
  rationale, confidence, calls: [{name, arguments}] ≤ max_calls}`.
  - Each `calls[i]` is dispatched through the sealed executor exactly as named, so a model that names `bash` is refused
    by the gate and the refusal is recorded. It is never filtered away before the gate.
  - **The turn**, not `Tool.run`, writes each proposal or refusal to item 5's log. `Tool.run` swallows log failures and
    drops a record whose model-chosen kwarg is `self`. A failed log write fails the turn.
- **The sealed registry.**
  - It is `SealedToolRegistry`, a separate class that satisfies a `ToolRegistryLike` Protocol (for mypy and for
    `Executor`), with no inherited writer. Its contents are a `MappingProxyType` over one `ProposeActionTool` instance,
    and every write name raises `SealedRegistryError`.
  - **The executor** is `Executor(sealed, policy=ToolPolicy.pinned({"propose_action": that_instance},
    ceiling=Effect.READ, reason="atonia"))`, with no wrappers, no embodiment and no entity seam.
  - The three layers are independent:
    - the sealed contents;
    - the ceiling;
    - the pin: identity with the instance captured at build, not with the class.
- **`ProposeActionTool` (READ)** validates `{summary, rationale?, target?}`, bounded, and returns it. It holds no
  reference to a registry, executor, log or callable.
- **Report status is never a false `ok`.** Each failure has its own status, and every report carries
  `log_last_seq`:

  | Condition | `status` |
  |---|---|
  | The backend failed to load at `open()` | `open()` raises; there is no report |
  | Completion error | `error` |
  | Schema-invalid output | `error` |
  | Deadline hit | `abandoned` |
  | Any call refused | `ok`, with `refusals` filled |
  | Item 10 marked an input degraded | `degraded` |

- **No loop, no autonomy controller, no prefetch, no data-home use.** Nothing calls `utils/paths.py`. The pinned
  read-set test proves it (below).
- **Docs:** `docs/user/advisor.md`, carrying the effect-channel table (AD9). A new row covers the report and log,
  which are the one sanctioned write and **carry model text**.

**Inputs, and how each could lie.**

| Input | How it could lie | Answer |
|---|---|---|
| Any env var | Hostile `MAXIM_*` (including names built at runtime, such as `MAXIM_LANE_{tier}_REMOTE_URL`), `HOME`, `XDG_*`, proxies | **The Python-level env read-set is pinned, not grepped.** A subprocess test wraps `os.environ` from before `import maxim.advisor` through `close()`: `get`, `__getitem__`, `__contains__`, `os.getenv`, and also `__iter__`, `keys`, `items`, `values` and `copy`, with the same wrapper on `os.environb`. `posix.environ`, the startup copy, is replaced by a poisoned mapping that fails on any access. Any iteration or copy records a **wildcard read and fails the test**, because a prefix scan reads names nobody records. The observed set must be a subset of a pinned allow-list: the refusal list, and library reads such as numpy's and threadpoolctl's. **C-level `getenv`** (`GGML_*`, `LLAMA_*` in `libllama`) is invisible to Python. It is covered by the refusal list (any `GGML_*`/`LLAMA_*` set → refuse) plus a `strings` census of the shipped `llama_cpp/lib` names, which explains the names but does not prove the absence of others. Proving it is out of the model, and the sandbox's job. |
| Env *writes* | numpy and threadpoolctl write on import | A pinned write allow-list, in the same test |
| config.json, llm.json | Hostile settings | Never opened. The audit hook's **Python-level** `open` events are pinned to `os.devnull`, the `report_dir` files and the import set. The GGUF is opened by C (`llama_model_load_from_file`) and raises no event, so its path is checked by `fstat` before load. Metal's C-level shader cache in the per-user cache directory is a disk write the hook cannot see; it is item 7's xfail. |
| A GGUF swapped after `open()` | A lazy reload | Eager load; no reload path |
| `model` | A profile name, a URL | Must be an existing regular file |
| Event text | Injection, special tokens | Framed, special-token-defanged; only `propose_action` dispatches |
| Model JSON | Extra fields, too many calls | Closed schema, capped; extras are refused, not ignored |
| The CWD, the data home | Prefetch, the state file, `cost_state.json` | Not reached; the audit hook proves it |

**Red gates (each in a fresh subprocess with `sys.addaudithook`, so `atexit` writes, `os.mkdir`/`os.rename`,
`.pyc` writes and sockets are all seen):**
- **The registration sweep (b).**
- **The dispatch sweep, per layer, proved by deletion.** For each layer, a test-only build that removes the other two
  layers and inserts every default-registry tool must refuse at **that** layer, with that layer's `error_kind`.
  Passing on `KeyError` alone is what the v1 sweep would have measured.
- **The builder refusals.**
- **The no-effect test, Python-level.** `HOME`, `TMPDIR` and the CWD are fresh directories holding canary files, and
  the parent snapshots their trees before and after. Only `report_dir` may change, and nothing may connect, send,
  resolve, spawn or `exec` at the Python level. C-level I/O (the Metal cache) and inbound stay strict-xfail for the
  parts items 7 and 4 own.
- **The env read-set and import-closure tests.**

## 5. Item 5: the proposal and refusal log (minimal)

`simulation/report.py::save_action_log` is a session-end rewrite at "1.2" with no fsync. The log borrows its shape (a
header plus records), not its durability. No fsync'd append helper exists, and the existing `atomic_io` helpers
swallow fsync errors, so the new one must not borrow theirs.

- **`AdvisorLog.create(report_dir_fd, session_id, *, mode=0o600)`.** `report_dir_fd` is the
  `O_DIRECTORY|O_NOFOLLOW` fd that `open()` already checked (§4.2). The log holds one file fd for the whole session.
  - It opens `advisor_log_<session_id>.jsonl` with `os.open(name, O_WRONLY|O_APPEND|O_CREAT|O_EXCL|O_NOFOLLOW, 0o600,
    dir_fd=...)`.
  - It writes the header `{"_format_version": "1.0", "_record_kind": "header", "schema": "advisor_log/1",
    "session_id", "maxim_version"}`.
  - It fsyncs the file, then the directory, once. On macOS it uses `F_FULLFSYNC`.

  It is never reopened by name. 0600 is the default; the operator reads it as the same user, or `maxim.advisor.open`
  passes `log_mode=0o640` through to `create(mode=)`. On Linux, which has no `F_FULLFSYNC`, it falls back to
  `os.fsync`.
- **`append(record)`** encodes one line (`ensure_ascii=True`, `allow_nan=False`, `separators=(",", ":")`), loops
  `os.write` until every byte is written, then fsyncs.
  - **Any** `OSError`, from a short write that cannot complete, an fsync or a full disk, marks the log broken and
    raises. The turn's report goes `error`, and nothing more is ever written to that fd.
  - So a torn line can only be the final one.
- **Records:**
  - `proposal`: seq, report_id, ts, arguments bounded to 4 KiB with a truncation flag;
  - `refusal`: seq, the name bounded to 128 characters, `error_kind`, layer, effect, the reason bounded;
  - `close`: last_seq, written by `close()`.

  `seq` is monotonic per session, and every report carries `log_last_seq`.
- **`read_log(path)`:**
  - requires the header, so an empty or headerless file is an error;
  - drops a torn final line only, and errors on a bad line anywhere else;
  - reports `closed: bool` and `last_seq`.

  A consumer can detect a lost tail by comparing against the reports' `log_last_seq`. A hash chain is not added: Maxim
  is the only writer, and tamper by others is the sandbox's job.
- **Nothing reads the log back into a dispatch.** That is structural, not a grep. The session holds only a
  write-only fd, and `read_log` returns plain dicts that sit in a module importing nothing from `runtime/` or
  `tools/`. The import-closure test checks that module too.
- **Minimal now, full later:** the observer's records only. Non-observer runs and the swallowing seams use structured
  WARNINGs in this release.

**Inputs, and how each could lie.**

| Input | How it could lie | Answer |
|---|---|---|
| Model arguments | Huge, binary, newlines, NaN | JSON-escaped, bounded, `allow_nan=False` |
| Text shaped like a header or a record | Forgery | Inside JSON strings; the reader parses records and requires exactly one header, first |
| A planted file or symlink | Redirects the writes | `O_EXCL|O_NOFOLLOW` with `dir_fd`; never reopened |
| A crash before the header | An empty log | The header is written inside `create()`; a headerless file fails `read_log` |
| A short write or `ENOSPC` | A torn line mid-file | The loop completes it, or the log is marked broken and stops |
| A lost tail | Undetectable | `log_last_seq` in every report, plus the close record |

## 6. Item 10: offline, enforced at the socket and refused at the call site

**The defect.**
- Maxim's HTTP has no single chokepoint in `utils/http.py`.
- Libraries fetch on their own:
  - HF ×4;
  - faster-whisper;
  - ultralytics: `attempt_download_asset`, `YOLO(name)` through its own `safe_download` (urllib or a `curl`
    subprocess), and analytics and pip auto-install unless `YOLO_OFFLINE="true"`;
  - `docker pull`;
  - `ddgs`;
  - the Anthropic and OpenAI SDKs, which build their own httpx clients;
  - the reachy_mini SDK (a websocket and an HF-space relay);
  - twilio.
- `tools/internet_search.py::_search_duckduckgo_lite` is a urllib3 bypass.
- `rtm_engine` has a consent-free download.
- Broad `except Exception` handlers would swallow a refusal into a fetch the guard never sees. The ultralytics path
  does exactly that.
- The cloud gate exempts `allow_local_endpoints=True`, and every peer lane sets it.
- `is_loopback_url` accepts `localhost`, which resolves through `/etc/hosts`, and Maxim itself creates `ssh -L`
  forwards.

v1 rested on URL checks plus a census. **A list of sites cannot be the enforcement; the 10 pass is right.**

**The design: three layers.**
1. **The enforcement: a production socket latch.**
   - **The hook.** `utils/offline.py` installs one `sys.addaudithook` the first time `engage()` runs. Audit hooks
     cannot be removed, so it is never installed in a process that never engages: an ordinary pytest run carries no
     hook, and `import maxim` stays light.
     - **Self-engage in children.** `maxim/__init__` does one cheap, stdlib-only check of `os.environ.get(
       "MAXIM_OFFLINE")`, and only when that is truthy does it import `utils/offline` and engage. So every Python
       child that imports `maxim` (spawn-mode multiprocessing, sub-sims) is latched from its first import, and
       nothing heavier is imported otherwise.
   - **While engaged, it refuses** (the red gate drives one call of each):
     - **`socket.__new__` of any family other than `AF_INET`, `AF_INET6` and `AF_UNIX`** (default-deny: `AF_PACKET`,
       `AF_NETLINK`, `AF_BLUETOOTH`, `AF_VSOCK` and future families). Fd adoption reports `family=-1` and is refused.
       Sockets created **before** `engage()` are a stated residual: `engage()` runs before any router or tool is
       built, and the observer engages first;
     - every `socket.connect` (and `connect_ex`) to an `AF_INET`/`AF_INET6` address, loopback included;
     - every `socket.sendto` / `socket.sendmsg` to an `AF_INET`/`AF_INET6` destination (UDP: mDNS through
       reachy_mini's `zeroconf`, hand-rolled DNS to a literal resolver). A `None` destination means the socket is
       connected, and its connect was already refused;
     - `socket.getaddrinfo` / `socket.gethostbyname`, of any name;
     - `socket.gethostbyaddr` / `socket.getnameinfo`, unconditionally, because reverse DNS leaves the machine;
     - **`AF_UNIX` connects**, with a pinned path allow-list that is **empty for the observer**. `docker.sock`,
       syslog and forkserver are refused under offline, and the docs say so;
     - **`os.system`, always;**
     - **every `subprocess.Popen`, `os.exec*`, `os.posix_spawn` and `os.spawn*`, unconditionally (V8).** There is
       no allow-list. A Popen whose args name a subprocess still raises before anything runs, so `PATH`, `env=` and
       `shell=True` have nothing to slip through. `_posixsubprocess.fork_exec` is unaudited: multiprocessing's
       spawn and forkserver children are covered by self-engage, and any other direct caller is in-process code
       (§0);
   - The hook raises `OfflineRefusal` and appends to a process **refusal ledger** before raising, so a refusal that a
     caller swallows is still on record.
   - **There is no allowed endpoint (V8).** Every `AF_INET`/`AF_INET6` connect and send is refused, loopback
     included. So offline needs an in-process LLM:
     - the CLI refuses `llm.offline` at startup unless every lane resolves to an in-process backend;
     - `lane_backends.py`'s auto-spawn and reuse paths are never reached, because they refuse first;
     - an `ssh -L` forward, a reused server and a re-bound port therefore have nothing to impersonate.
   - **The boundary, stated in the docs.** The latch sees the Python `socket`, `subprocess` and `os` modules
     **only**. It is blind to:
     - **native networking:** libuv (uvloop), gRPC, libzmq, libcurl, Rust clients;
     - **un-audited spawns:** multiprocessing `spawn`'s `_posixsubprocess.fork_exec`.

     So:
     - (a) the census has a `latch-blind` category for event-loop implementations and native-network libraries;
     - (b) under offline, `console/server.py` and `comms/api.py` run uvicorn with `loop="asyncio"`;
     - (c) a spawn-mode child is latched by the self-engage above;
     - (d) anything native that remains is the sandbox's (AD5).
   - **Sub-processes.** Under offline there are none (V8). A process that was spawned before `engage()`, or a
     spawn-mode multiprocessing child, inherits `MAXIM_OFFLINE=1`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` and
     `YOLO_OFFLINE=true`, and `host_tool_env`'s allow-list gains them, for online runs that later engage.
2. **The call sites refuse first,** so the failure is typed and early rather than a socket error deep in a library:
   - `utils/http.py`:
     - **under the latch,** every client is built strict:
       - `mounts=None` (not `_proxy_mounts`, which re-adds the env proxies on purpose, #921);
       - the `policy == "any"` branch of `get_client` gets `trust_env=False`;
       - every client has a per-request `request.url` check.

       Proxies are dropped only while offline is engaged. Online proxy behaviour is unchanged, so #921 ("the proxy
       wins for operator paths") stands;
     - `engage()` calls `close_all()`, so no pooled client from before the latch survives.
   - HF: `local_files_only=True` at each site. `engage()` sets `HF_HUB_OFFLINE` and, if `huggingface_hub` is already
     imported, sets `huggingface_hub.constants.HF_HUB_OFFLINE = True`. `is_offline_mode()` just returns that global, so
     calling it proves nothing. The red gate verifies **behaviourally**: a hub call must raise
     `OfflineModeIsEnabled` with no socket event. v1's refuse-if-imported rule was too strict.
   - `transformers_backend`: `local_files_only`, and `trust_remote_code=False`. This is a local LLM, so the router
     predicate allows it.
   - faster-whisper: `local_files_only=True`.
   - ultralytics:
     - `YOLO_OFFLINE="true"` is set before its import. If ultralytics is already in `sys.modules`, `engage()` also
       sets `ultralytics.utils.ONLINE = False`, mirroring the HF rule (its import-time `is_online()` ran
       `getaddrinfo`);
     - `YOLO(...)` takes only an existing file path, and refuses otherwise **at the last call before the library**;
     - `attempt_download_asset` is never called offline.
   - `ensure_available` and `ensure_tts_voice` refuse before reading consent.
   - `container_runner.ensure_image`, `ddgs`, hivemind and twilio refuse.
   - The urllib3 path moves onto `utils/http.py` whatever the latch says. That is the root cause and a CLAUDE.md
     invariant; the hunk is in a file shared with H. The CI grep widens to `urllib3`, `requests`, `httpx.Client(` and
     `socket.socket(` outside `utils/http.py`.
3. **The router predicate, checked at build and at every dispatch,** because a router built before `engage()` keeps
   live backends. Under offline only in-process `llama_cpp`/`transformers` lanes may run (V8); a lane with any
   endpoint, loopback included, refuses. Cloud SDKs are never constructed after the latch, and are refused at dispatch if
   they were built before it.

**Engaging the latch:**
- `engage(reason)` is one-way.
- `is_offline()` returns the latch. It engages on any call that sees `MAXIM_OFFLINE` truthy, not only the first read,
  so a later raise counts.
- `llm.offline` is read at CLI startup and at the first router or `api` verb build. A running `maxim serve` picks it
  up on restart, or with #1411's config generation once that covers it.
- The observer always engages.
- Tests reset the latch through a conftest fixture that touches the private state. No reset exists in `src/`, and the
  census forbids one there. The hook is inert when not engaged, so it costs the suite nothing.

**The fallbacks become loud.** A missing encoder or semantic weight under offline logs one structured WARNING per
process, records a `degraded` reason, and the observer's report says `degraded`. decision_log stamps it; it writes at
verbosity ≥1, which is noted.

**Config:**
- `llm.offline` goes in the SHAPE-FROZEN `LLMConfigSection`.
- The shared `CONFIG_FORMAT_VERSION` 1.3 → 1.4 bump, with `"1.4"` fixture entries and the #974 downgrade. 1.3.3
  shipped without it; #1431's `console.talk_tools` rides the same bump.
- No env alias, and the key stays off `_NON_SECURITY_FIELDS`.
- It wins over every auto-download switch.
- **Fail-open on downgrade, stated:** a 1.3 build tolerates the unknown key and runs online. `maxim config downgrade`
  must warn when it drops `llm.offline: true`, and the docs say an offline deployment pins its version (AvServ
  already pins exactly).
- `XDG_CONFIG_HOME` hiding the file is the operator's own choice, and the observer does not read config at all.

**`decision_log`:** `offline: bool = False` (defaulted). A 1.3 reader returns `None` on a 1.4 record, which is
already its behaviour on any newer record. That is noted, not fixed here.

**The census (explanation, not enforcement):**
- **The site census** is an AST/grep census of:
  - the fetch patterns;
  - every `socket.*` and `subprocess`/`os.exec*`/`os.system` site;
  - an **import roster** of network-capable libraries: httpx, urllib3, requests, aiohttp, websockets, the SDKs,
    reachy_mini, twilio, ddgs, huggingface_hub, ultralytics.

  Each hit is pinned in JSON with its handling (`http-guard` | `refuses` | `local_files_only` |
  `latch-only`, with a reason). A new hit fails the census.
- **The swallow census:** every `except Exception` caller around a refused site is pinned, with a check that it
  re-raises `OfflineRefusal` or records degraded.

**The red gate:**
- It runs **in a fresh subprocess** with `HF_*` unset from the start, because deleting them in-process does nothing
  once `huggingface_hub` is imported.
- It engages offline and drives each census site.
- It asserts two things:
  - each site raised or recorded `OfflineRefusal`;
  - the audit hook's own log shows no connect, resolve or spawn beyond the allow-list.
- That replaces v1's "network_guard never reached", which was unobservable: `network_guard.attempts` does not record
  loopback or IP-literal calls.

**Inputs, and how each could lie.**

| Input | How it could lie | Answer |
|---|---|---|
| Proxy env | Carries a loopback URL off the box | Under the latch: `mounts=None` and `trust_env=False` on every client; the socket latch sees the real connect target |
| A hostname | `localhost` → `/etc/hosts`; DNS rebinding | IP literals only; forward and reverse DNS refused at the hook |
| UDP | mDNS (`zeroconf`), DNS to a literal resolver | `socket.sendto`/`sendmsg` to any INET destination refused (V8) |
| An `ssh -L` forward Maxim made, or a reused server | Loopback that is remote | No endpoint is allowed under offline (V8); every loopback connect refuses |
| A native connect path | uvloop, gRPC, libcurl: no audit event | Stated boundary; `latch-blind` census category; `loop="asyncio"` under offline; the rest is the sandbox's |
| A spawn-mode child | Un-audited `fork_exec`; an inert hook | The hook self-engages at install from `MAXIM_OFFLINE` |
| A swallowed refusal | A broad catch, then the library downloads anyway | The socket latch still refuses; the ledger records it; the last-call refusal; the swallow census |
| A router built before `engage` | Live cloud backends | The predicate is checked at dispatch |
| `MAXIM_OFFLINE` raised later | Ignored | Checked on every `is_offline()` |
| A model-chosen command | `bash curl ...` | V7 removes those tools; the hook refuses every spawn (V8) |
| `PATH`, an explicit `env=`, a git config | A hostile `git`; a child env without the latch vars; a config-defined filter | Every spawn refuses under offline (V8) |
| HF or ultralytics already imported | The constant was read at import | The module globals are set; the red gate verifies behaviourally |
| ultralytics | Analytics, pip auto-install, its own downloader | `YOLO_OFFLINE="true"` before import; path-only `YOLO` |
| Config downgrade | 1.3 ignores `llm.offline` | Stated, plus a downgrade warning and version pinning |

## 7. How the five fit, and the order

1. Item 2: `Effect`, `effect_of`, `exceeds`, `_admit`, and the classification.
2. Item 3: `ToolPolicy`, the dispatch check on the resolved object, typed refusals, `current_policy`, and
   `auto_sense`.
3. Items 1 and 5: the closed backend, the sealed registry, the pinned policy, the turn and the log.
4. Item 10 in parallel. The observer depends only on `engage()` and the hook.

Each item:
- has its strict red gates before code;
- proves each guard by deleting it;
- gets a three-lens review.

**Files owned elsewhere:**
- `tools/internet_search.py` (H): the urllib3 hunk.
- `console/handle.py` (S): the `deny` policy.
- `models/audio/tts.py` (S): the `ensure_tts_voice` refusal.
- `runtime/config_loader.py`: the shared 1.4 bump, with S's #1431.
- `orchestrator.py` (A, AD8): `ToolPolicy` call sites only.
- `runtime/loop_perception.py`: the `auto_sense` check. It is not fenced, and the change is a one-call hunk.
- CLAUDE.md (the process session): the one new router-package entry point, as a stub to hand off.

## 8. Where advisor_mode.md has gone stale (fold after the owner decisions)

- The release number.
- Item 1's queue claim.
- "Class identity" now means the pinned instance.
- Item 2: the silent replace; `deregister`; `SimToolRegistry.get`; mutable class attributes; the order now a set;
  `GOVERN`.
- Item 3:
  - `build_executor` is already required;
  - `Executor`'s positional `permissions`;
  - `AgentFactory` routing, and `load.py`;
  - `run_agent_loop` builds nothing;
  - `ConcurrentExecutor`;
  - the test counts (73/28, 42/16);
  - `PainInterceptorExecutor` and `AnticipatoryPainExecutor`;
  - `docs/user/extension_api.md`.
- Item 5: `save_action_log`.
- Item 10:
  - #1343 closed two holes;
  - `rtm_engine`;
  - urllib3;
  - no chokepoint;
  - `hf_*` are env vars;
  - the 1.4 bump;
  - the socket latch;
  - ultralytics.
- The context row: the CWD re-read is in `loop_controller.py`.
- The disk row: `runtime/config_writer.py`.
- O3 and O7: `review_action` makes no LLM call; rejections mostly miss `actions.jsonl`.
- `AgentPermissions` lacks a CC3 marker (filed).
- Profile A's name.
- §"The guarantee": a new row for the report and log (model text).

## 9. Fold ledger (v1 → v5)

**Items 2+3** ([review](reviews/atonia_gate/items_2_3_adversarial.md)):

| Finding | Fold |
|---|---|
| 1 | `exceeds()` |
| 2 | `Effect(0)` refused |
| 3 | Miss and mismatch rules, §2.2 |
| 4 | Collaborator rule and reclassified examples, §2.1 |
| 5 | Flags defined; `GOVERN` (V2) |
| 6 | `auto_sense` checked (V6) |
| 7 | Immutable registry ceiling; dispatch authoritative |
| 8 | Admission is layer 1, stated |
| 9 | All-or-nothing entity generation |
| 10 | Replacement refused below `ALL` |
| 11 | No public constructor; `deny()` |
| 12 | V3 hard |
| 13 | `current_policy` contextvar |
| 14 | No alias snapshot |
| 15 | Blind spots stated; scripts named |
| 16 | Roster applies the ceiling |
| 17 | `KeyError` |
| 18 | `ConcurrentExecutor` takes no policy |
| 19 | `narrowed()` rules |
| 20 | Factories and the count |
| 21 | Keyword-only |

**Items 1+5** ([review](reviews/atonia_gate/items_1_5_adversarial.md)):

| Finding | Fold |
|---|---|
| 1 | Closed backend, not `LLMRouter` |
| 2 | Eager load, path only |
| 3 | Status table |
| 4 | Import-closure test |
| 5 | Special-token defang (v2); token-id prompt (v3, delta #6) |
| 6 | Closed schema with `calls` |
| 7 | Typed refusals; the turn writes the log |
| 8 | Instance pin |
| 9 | `"unknown"` effect |
| 10 | Subprocess and audit-hook gates; per-layer deletion sweep; pinned env read-set |
| 11 | Code-loading refusal list; §0 |
| 12 | One fd, `dir_fd` |
| 13 | Write loop, broken-log rule |
| 14 | fsync raises; directory fsync; `F_FULLFSYNC` |
| 15 | Header in `create()` |
| 16 | `log_last_seq`, close record |
| 17 | Per-session name |
| 18 | Encoding |
| 19 | Effect-table row |
| 20 | Protocol |
| 21 | `turn_timeout_s` |
| 22 | `loop_controller.py` |
| 23 | No change |

**Item 10** ([review](reviews/atonia_gate/item_10_adversarial.md)):

| Finding | Fold |
|---|---|
| 1 | Socket latch, refusal ledger, last-call refusal, swallow census |
| 2 | `trust_env=False`, per-request check |
| 3 | V7, env allow-list |
| 4 | Predicate at dispatch |
| 5 | DNS refused; `YOLO_OFFLINE` |
| 6 | Import roster, all `socket.*` and subprocess sites |
| 7 | The latch is the enforcement |
| 8 | IP literals; spawner record |
| 9 | Downgrade stated; startup hooks |
| 10 | Subprocess gate; the audit log replaces network_guard |
| 11 | Set and verify `is_offline_mode()` |
| 12 | Every-read raise; conftest-only reset |
| 13 | transformers allowed |
| 14 | Noted |
| 15 | `close_all()` |
| 16 | Ingress out of scope, §0 |

**v2 delta round** ([review](reviews/atonia_gate/v2_delta.md)):

| Finding | Fold |
|---|---|
| 1 | `sendto`/`sendmsg` and reverse DNS refused at the hook |
| 2 | Self-engage at install; stated boundary; `latch-blind`; `loop="asyncio"` |
| 3 | Absolute executables keyed on `realpath`; the env rule |
| 4 | Live-child endpoint; the reuse path refuses |
| 5 | Strict clients under the latch only; #921 stands online |
| 6 | Token-id prompt, `special=False` for untrusted segments |
| 7 | Prompt-token cap before prefill; the deadline flag sets the status |
| 8 | Wildcard read on iteration or copy; `LLAVA_CPP_LIB` and `LLAMA_*` refused; C-level `strings` census |
| 9 | `ToolPolicy` immutable; pins snapshotted |
| 10 | `ToolErrorKind` members; a policy refusal books no pain, NAc or hallucination counters |
| 11 | Enumerated V7 ceiling; `WRITE_LOCAL` no longer covers process runners |
| 12 | `field(kw_only=True)`; `None` only where no executor is built |
| 13 | Behavioural verification |
| 14 | `ultralytics.utils.ONLINE = False` |
| 15 | AF_UNIX policy stated |
| 16 | Directory fd opened first; checks on the fd; `mode=` |
| 17 | "Python-level" restated; import ceiling widened |
| 18 | Synchronous read stated; `check()` refuses composites |

**v3 delta round** ([review](reviews/atonia_gate/v3_delta.md)):

| Finding | Fold |
|---|---|
| 1 | Port must refuse before the spawn; listener ownership proven via psutil before the key is sent; spawner state, not `poll()` |
| 2 | (realpath, argv-prefix) pairs; `os.system` always refused; no shell pinned |
| 3 | `add_bos=False`; per-id attribute check; escape or visible refusal; the vocab test by attribute |
| 4 | Key in `__new__`; unconditional `__setattr__`; no subclasses; `type(...) is` |
| 5 | Four loader variables, grouped by exec and `dlopen` |
| 6 | "Processes or code"; the AST check and `tools/effect_exemptions.json` |
| 7 | Git tools use `pinned_executable` |
| 8 | A `None` destination means the socket is connected |
| 9 | Default-deny at `socket.__new__` |
| 10 | The hook is installed only on engage; a cheap env check in `maxim/__init__` |
| 11 | `ACTUATE` note on V7 |
| 12 | `os.environb` wrapped; `posix.environ` poisoned |

**v4 delta round** ([review](reviews/atonia_gate/v4_delta.md)):

| Finding | Fold |
|---|---|
| 1 | V8: no endpoint at all |
| 2 | V8: no spawner connects under offline |
| 3 | V8: no spawns; the git exemption holds online only |
| 4 | `fork_exec` stated; prefix pairs deleted |
| 5 | Bitmask; BYTE is plain text |
| 6 | Whitespace note; synthetic refuse-branch test; vocab source |
| 7 | Out of the model, stated |
| 8 | `DYLD_*` by prefix |
| 9 | Full sink list; cross-module reach; stated blind spots |
| 10 | Fd adoption refused; pre-engage sockets a stated residual |

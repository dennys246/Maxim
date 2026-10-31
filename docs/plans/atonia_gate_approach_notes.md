# The atonia gate: approach notes for items 2, 3, 1, 5 and 10

**Status:** DRAFT v6.10 (design passes, three-lens round and two deltas folded; red-gate review decisions folded), approach notes (CLAUDE.md "Faster without lowering the bar", item 1). Owner decisions V1–V8
were made 2026-10-24 and V9–V10 on 2026-10-25. Session V, paper only: nothing in `src/` changes until the owner has read the folded
plan and the red-gate PR is opened.
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
- **v5.1** folds the [v5 delta round](reviews/atonia_gate/v5_delta.md), which returned **STOP: no further round
  owed**. All 11 findings were fail-closed or consistency fixes in already-covered classes, and they are folded
  here rather than sent to a follow-up, because each one shapes code that has not been written yet.
- **v6** folds the three-lens review round ([executor](reviews/atonia_gate/lens_executor.md), [architecture](reviews/atonia_gate/lens_architecture.md),
  [wire](reviews/atonia_gate/lens_wire.md); 5 BLOCKER, 27 SHOULD-FIX, 20 NIT). It adds owner decisions V9–V10 and §1A, which
  wins over §2–§6 where they disagree.
- **v6.1** folds the [v6 delta round](reviews/atonia_gate/lens_delta_v6.md) (no BLOCKER; two fail-open folds, #1 the
  profile accessors and #6 the refusal set and the #1185 booking precedent, owe a Wire-only round with the stop
  rule).
- **v6.2** folds that [Wire round](reviews/atonia_gate/wire_delta_v6_1.md) (no BLOCKER; six fail-open SHOULD-FIX, all
  call-site enumerations). Each is now carried by an **outcome red gate** that fails on any missed site (every
  profile reader resolves a user profile; every gate refusal books zero learning writes) and by a CI grep. The
  table is no longer the enforcement.

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

## 1. Owner decisions (V1–V8 decided 2026-10-24, V9–V10 2026-10-25, V11–V16 2026-10-30; all as recommended)

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

| **V8** | Under offline, may Maxim run any subprocess or connect to any endpoint? (asked 2026-10-24 after the v4 delta round) | **Nothing at all.** No subprocess the hook can see, no socket endpoint, no allow-list. Multiprocessing spawn children still start, through the unaudited `fork_exec`, and are latched by self-engage; their callers (`embodied_runtime/media_loop.py`, `data/camera/display.py`) are on the robot path, which is refused anyway. Offline also means **solo**: binding the leader proxy's `http.server` triggers a reverse-DNS lookup, which is refused, so the leader cannot start. The CLI's `llm.offline` needs an in-process model (the `llama_cpp` or `transformers` backend) and refuses at startup otherwise. The git tools drop. | The spawned-server allowance and the pinned git argv drew most of the fail-open findings in three consecutive rounds (v2 #3/#4, v3 #1/#2/#7, v4 #1–#4): endpoint revocation, listener ownership, and git's config-driven filters running code the latch cannot see. Neither the observer nor AvServ needs them. |
| **V9** | The observer's import closure cannot hold: `maxim.runtime.executor` loads about 830 modules, and `models/language/config.py` parses the user's `profiles.yml` at import (three-lens round, 2026-10-25) | **Lazy package inits:** PEP 562 in `runtime`, `agents`, `tools` and `utils`; the profile load moves to first use. It is a new step 0, its own PR (§1A.1). | It is the root cause, and it makes `import maxim` lighter everywhere. A leaf-module observer would duplicate `Executor`'s check (two dispatchers for one gate), and dropping the closure gate would let a hostile `XDG_CONFIG_HOME` be parsed before `open()` refuses. |
| **V10** | What may a ceiling do to the stable verbs `maxim.tool` / `register_tool` (2026-10-25)? | **Nothing in this release.** `maxim.run` keeps ceiling `ALL`. Ceilings below `ALL` apply only to the observer, Talk and offline, where undeclared user tools drop loudly. `effect=` is added as an optional form (§1A.4). | Both verbs are ✅ stable. A narrowed `maxim.run` would silently stop running users' tools in a patch. |
| **V11** | Does a tool that calls the LLM router (sense_presence's imagination, design_entity) carry `NETWORK` (item 2 build, 2026-10-30)? | **No: the router is infrastructure.** Cloud egress is governed by the lanes and the offline latch (V8: offline means in-process lanes). | The agent loop sends to the router every turn anyway. With `NETWORK`, an offline ceiling would drop `sense_presence`, an auto-fire tool, even with an in-process model. |
| **V12** | What do the narrator's `send_message`, `extend_simulation` and `inject_pain` declare (2026-10-30)? | **`ALL`.** Each drives a full turn of the agent under test, which can use any of its tools. | "Everything it can cause, through any collaborator" (§2.1), the reasoning that makes `spawn_sub_simulation` `ALL`. A narrator registry stays at ceiling `ALL`. |
| **V13** | `focus_interests` reaches a multiprocessing spawn: `passive_observation` opens the display process even with `show=False` (2026-10-30) | **Declare `ALL` now, and file the `show=False` display bug as its own issue (#1544).** Once that is fixed the sink is gone and the tool can declare `READ\|ACTUATE`. | An exemption would excuse a sink that is itself a bug; fixing vision code would widen item 2. |
| **V14** | What do the no-op stubs (observation-only `maxim_command`, `move`, `track_target`, …) declare (2026-10-30)? | **The real tool's effect.** `create_stub_tool` takes a required `effect=`. | A ceiling then refuses a name the same way whether or not a robot is attached, and the mode gate's name sets (`HOST_ACTING_TOOLS` includes `maxim_command`) agree with `Effect`. |
| **V15** | A third-party `Tool` with its own non-`Effect` `effect` attribute, or `MyTool.effect = ...` after the class is defined (item 2 review round, executor S2 and wire F1, both reproduced; 2026-10-30) | **It counts as `ALL`, with one WARNING per class.** `effect_of` returns `ALL` whenever a class's own value is not a valid `Effect` or no longer matches its record, so ceiling `ALL` (`maxim.run`) admits the tool and any narrower ceiling refuses it. `Effect(0)` stays a `TypeError`. | §2.2 had a `TypeError` and an admission refusal at `ALL`, which broke the stable `Tool` ABC in a patch, against V10. |
| **V16** | The `NETWORK` definition says "any egress off the machine", but V11 exempts the router and the robot link was classified `ACTUATE` (architecture S1; 2026-10-30) | **Amend the definition:** `NETWORK` is egress the tool causes off the machine, other than the configured LLM lanes (V11) and the configured robot body, whose channel is `ACTUATE`. Offline still refuses both, at the lanes and the latch. | Otherwise every robot tool would carry `NETWORK` and drop under any ceiling without it. |

**Flagged, not asked now:** `maxim.research` already exists (`api.py::research`, Experimental), so Profile A's builder
needs another name when items 13 and 14 start.

## 1A. Cross-cutting rules (v6, from the three-lens review round)

The three-lens round ([executor](reviews/atonia_gate/lens_executor.md), [architecture](reviews/atonia_gate/lens_architecture.md), [wire](reviews/atonia_gate/lens_wire.md)) found
that several rules apply to every item. They are stated once here. **Where §2–§6 disagree with this section, this
section wins.**

### 1A.1 Step 0: lazy package inits (V9) and the offline latch

**The finding (all three lenses):** importing `maxim.runtime.executor` or `maxim.agents.permissions` loads about 830
modules today. Among them are `runtime.agent_loop`, `agents.llm_worker`, `default_network`, `memory` and `utils.paths`.
Importing anything under `models/language/` runs `config.py::_apply_user_profiles_at_import`, which reads
`XDG_CONFIG_HOME`/`HOME` and parses `~/.config/maxim/profiles.yml`. The observer's import-closure and env read-set
gates could not pass, and a hostile `XDG_CONFIG_HOME` would be parsed before `open()` could refuse anything.

**Step 0a (V9, owner decision 2026-10-25: lazy package inits).**

> **Status: BUILT 2026-10-30** (branch `feat/1315-step0a-lazy-inits`). As built, where it differs from the text below:
> - `load_llm_config` is not edited, so rule T does not bite it. It resolves its profile through `_normalize_profile`
>   before it reads a table, and `_normalize_profile` ensures first.
> - `build_model_path` reads no profile table, so it does not ensure (an ensure there would parse `profiles.yml` for
>   nothing).
> - The CI check is an AST scan, `tests/unit/test_profile_readers_v9.py::test_no_reader_outside_its_owner_names_a_private_profile_table`,
>   not a grep: a docstring that mentions a name is not a read. `config.py`'s `_ensure_user_profiles` may name
>   `LLM_MODELS`, because it IS the merge.
> - `maxim.models.LLM_MODELS` stays public through a module `__getattr__` that returns `llm_models()`.
> - The fresh-process import sweep (`tests/integration/test_fresh_import_sweep.py`, slow) found one latent cycle:
>   `agents/base.py`'s bottom-of-module `GoalAgent` re-export, now a module `__getattr__`.
> - The only `_maybe_auto_spawn_server` edit moved out to `lane_backends._lane_profile_is_cloud` (rule T: 271 → 259).
> - Seven red gates flipped: the executor closure, the import-time `profiles.yml` read, `download.llm_models()`, and
>   the four lazy-init pins. #1479's alias gate stays red.
> - **Review fold (owner decision 2026-10-30): one fail-fast seam.** On main a malformed `profiles.yml` raised at
>   import. Lazily, ~15 `except Exception` profile reads swallowed it, and tier detection silently picked a smaller
>   model (wire W1, reproduced). `build_primary_router` now calls `_require_valid_user_profiles()` first, so every run
>   path refuses up front. The delta round (D1) found that CLI agentic, embodied, research and foundry never reach
>   it, so the owner added a second caller in `cli._main_impl`, before the mode dispatch. Both now call the public
>   `config.require_valid_user_profiles()`, which raises a typed `UserProfilesInvalid`, and `cli.main` renders it
>   (exit 2, file and fix). `maxim doctor` runs and reports instead of crashing. Also folded: conftest merges the
>   isolated profiles once, at collection, and a re-entrant merge raises. Declined: hiding `__getattr__` from mypy
>   (executor F5). The mypy ratchet counts `if not TYPE_CHECKING:` as a suppression in each init, so a misspelt lazy
>   name still types as `Any`, as it would with any PEP 562 module. Reviews: `docs/plans/reviews/step0a/`.

- PEP 562 `__getattr__` in `maxim/runtime/__init__.py`, `maxim/agents/__init__.py`, `maxim/tools/__init__.py` and
  `maxim/utils/__init__.py`. Every `from maxim.runtime import X` user keeps working; the names resolve on first
  access. A test pins `__all__` against the lazy table, and a subprocess test pins `import maxim.runtime.executor`'s
  closure.
- `models/language/config.py`'s profile load moves from import time to **one idempotent
  `_ensure_user_profiles()`**, guarded once per process. **Every** profile accessor calls it: `load_llm_config`,
  `normalize_llm_profile`, `list_llm_profiles`, `build_model_path`, and new `get_profile(name)` /
  `iter_profiles()`.
  - The ten direct readers of `_BUILTIN_PROFILES`, `_PROFILE_ALIASES` and `models/download.py::LLM_MODELS` migrate to
    the accessors: `cli.py` `--list-models`, `cli_utils.py`, `api.py` `list_models`, `doctor/checks.py`, and the
    in-module helpers. Otherwise a user profile silently disappears from them.
  - The ensure call is also inside the private `_normalize_profile`, which `lane_backends`, `llm_agent` and
    `download.delete_llm` import directly. `models/download.py` gains an ensuring `llm_models()` accessor for the
    `LLM_MODELS` reads (`ensure_available`, `download_llm`, `delete_llm`, the list/check/status functions,
    `lane_backends`).
  - **A CI grep** forbids `_BUILTIN_PROFILES`, `_PROFILE_ALIASES` and `_normalize_profile` outside `config.py`, and
    `LLM_MODELS` outside `download.py`. So the grep, not the plan's count, forces every reader to migrate: about 30
    `src/` sites and one script, not "ten".
  - **Failure is never cached.** The once-per-process guard is set only after a successful merge. A malformed
    `profiles.yml` raises on **every** access, so a swallowing `except Exception` site cannot turn the first failure
    into a permanently silent "no user profiles".
  - `get_profile(name)` does not normalise; callers normalise, as they do today, so the doctor and `lane_models`
    lookups are unchanged.
  - A `patch.dict` on `_BUILTIN_PROFILES` that wraps the first ensure would roll the merge back while the guard stays
    set. That is harmless, because conftest isolates `XDG_CONFIG_HOME` and the red gate uses a fresh process.
  - **Red gate, by outcome rather than by enumeration:** in a fresh process with a `profiles.yml` holding one profile
    and one alias, **every public profile reader** resolves both: `--list-models`, `api.list_models()`, doctor's VRAM
    projection, `normalize_llm_profile`, `ensure_available`'s lookup and lane resolution. It is parametrised, so a new
    reader joins it. A malformed file raises on the first and on the second access.
  - `closed_local` never calls `_ensure_user_profiles`, which keeps the env read-set property.
- `Executor.execute`'s per-dispatch `simulation.sim_logger` import moves behind the code path that needs it.
- **Build details:**
  - Each lazy `__getattr__` raises `AttributeError` for unknown names. `from maxim.utils import http` and `from
    maxim.runtime import tool_dispatch` depend on Python's fallback to submodule import.
  - Each init mirrors its table under `if TYPE_CHECKING:`, so mypy keeps real types.
  - The PR runs a fresh-subprocess "import every module" sweep, because lazy inits can surface latent cycles.
- It is its own PR with a three-lens review, before item 2. The other sessions get notice, because it changes import
  timing for every package. **Rule T bites one function:** `models/language/config.py::load_llm_config` (pinned at
  269). The ensure call goes in through the accessor helper, with a matching extraction (§7 table). The `__init__`
  modules hold no pinned function.

**Step 0b: the offline latch.** `maxim/_offline.py` is a **top-level leaf module**, not `utils/offline.py`, so that
importing it never runs `utils/__init__`. It holds `engage`, `is_offline`, the audit hook, the refusal ledger and the
canonical truthy parser `truthy(value)` ("1/true/yes/on", case-insensitive). It is stdlib-only, and
`cluster_bias_annotation.annotation_disabled_via_env` delegates to it, so one parser serves the parent, the child and
`maxim/__init__`'s self-engage check. A parametrised test pins that. Items 3 (V7) and 1 depend on 0b, so 0b lands
before them. Item 10's call-site refusals stay parallel.

**`MAXIM_OFFLINE` and `MAXIM_INTERNET_LOWERED`.** Offline implies internet-lowered **on the read side**:
`utils/internet_access.py::session_internet_lowered()` returns `True` when `maxim._offline.is_offline()`. `engage()`
**imports** nothing from `maxim.utils`, so the leaf has no back-edge and nothing logs "lowered by the agent".
In order, `engage()` sets the latch, installs the audit hook if it is not yet installed, and then does exactly three
things:
1. **`os.environ` writes.** `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_HUB_DISABLE_TELEMETRY`, `YOLO_OFFLINE` and
   `MAXIM_OFFLINE` (§6), so libraries read them in-process and children inherit them. The observer's env-write
   allow-list is exactly those five.
2. **Flags on libraries that are already imported,** found through `sys.modules` and never imported:
   `huggingface_hub.constants.HF_HUB_OFFLINE = True` and `ultralytics.utils.ONLINE = False`.
3. **The callbacks registered through `maxim._offline.on_engage(fn)`.** `on_engage` is stdlib-only and runs `fn` at
   once if the latch is already up. `utils/http.py` registers `close_all` at its own import, so a client pooled
   before the latch is closed, and a keep-alive socket cannot keep sending: `send`/`sendall` raise no audit event,
   so the hook alone could not stop it. **Red gate:** a client pooled before `engage()` cannot send after it.

  The observer's closure excludes `maxim.utils.http`, so in the observer that callback is never registered and there
  is nothing to close. `_offline.py` being stdlib-only is guarded by
`tests/integration/test_offline_latch.py::test_offline_module_is_stdlib_only`
(an AST check on the `test_clean_flag_998` pattern). The internet tools'
`disabled_reason` and the prompt roster then agree. V5 cannot ride on `MAXIM_INTERNET_LOWERED` alone, because that
covers neither HF nor the LLM. `MAXIM_OFFLINE` goes in llm-routing.md's env table, next to it.

**Tests.**
- `tests/conftest.py` must `os.environ.pop("MAXIM_OFFLINE", None)` at the top of the module, **before** its
  `import maxim`, so a developer shell cannot latch the pytest process. The autouse scrub alone would be too late.
- The conftest latch reset calls `maxim._offline._reset_for_tests()` (below). The `on_engage` callbacks stay
  registered across a reset and run again on the next `engage()`.

### 1A.2 One dispatch seam, and one booking rule

**The seam.** Every gate check lives in `Executor._permission_denial`, the seam `permits()` and the prompt roster
already read, in this order:
1. forbid source;
2. mode;
3. policy permissions;
4. ceiling;
5. pin.

- The ceiling is `policy.ceiling & OFFLINE_CEILING` whenever `is_offline()` is true. **V7 is applied here, once, at
  dispatch, and in `permits()`.** It is not applied at each builder site. That covers future builders and a policy
  built before a later raise, and the roster never offers a tool that only the hook would refuse.
- `OFFLINE_CEILING` is the enumerated `READ|SELF_SCOPE|WRITE_LOCAL|ACTUATE|GOVERN`.
- `executor.check(tool)` is the same seam without dispatch. It **returns a typed refusal and never raises**.
- The runtime-tools.md invariant text "one seam, THREE gates" changes to five, a hand-off (§7).

**The booking table.** Every gate refusal is booked the same way. Today the planned skip sat where refusals never
arrive: executor denials return before pain is booked, and NAc is booked downstream in
`tool_dispatch.py::record_outcome`, which never saw `error_kind`.

| Sink | Today, for a gate refusal | After |
|---|---|---|
| Pain (`_run_started` mapping, `proprioception/pain.py`) | permission and mode denials: not booked (they return before). **Unregistered and inactive-scene refusals: booked**, through `_report_failure` | not booked for any gate kind. That is a behaviour change for the unregistered and inactive-scene cases, and the CHANGELOG names it |
| `ToolPainBridge` NAc | not booked | not booked |
| `record_outcome` → `nac.observe`, goal credit, cluster reward; **and `execute_and_learn`'s own `PlanHistoryBridge` `nac.observe(NEGATIVE)`**, which is booked outside `record_outcome` and needs the same branch | **booked negative** | **not booked**, following the shipped #1185 precedent `runtime/tool_dispatch.py::book_machine_refusal` (a refusal no person made: `nac=None`, no cluster, no goal) |
| `recent_outcomes`, `llm_worker.record_outcome` (the reasoning carryover), `context_pool` | booked | **booked**, as `book_machine_refusal` does, so the model is told the tool was refused |
| `_tools_hallucinated` (feeds `failed_tools` hint, `MAXIM_TOOL_FAILURE_HINTS`) | booked | booked, unchanged for every gate kind, mode included, so the hint keeps working |
| `AutonomyController` audit log (`log_action`), which `max_consecutive_failures` counts | booked | booked: **this is the counter that stops a model that keeps naming a refused tool** (the red gate drives it). `_consecutive_failures` only lengthens the hint |
| `actions.jsonl` / `ActionRecord` / pulse | `success=False`, no kind | an optional `error_kind` (the enum `.value`). It is a full contract change, Session S's, so a hand-off: the producer `simulation/sinks.py`, the `actions.jsonl` header 1.2 → 1.3 with `console/run_store.py`'s version gate, the pydantic `console/schemas.py::ActionRecord` field, regenerated `console/openapi.json` and `contract_surface.json`, and the contract-version note to pulse. Shipping only the schema field would leave it always null |

- **What counts as refused.** `record_outcome` gains `refused: ToolErrorKind | None`, **keyword-only and required**
  (the TF8 `bridge_booked` precedent). It is threaded from `execute_and_learn` (both sites) and
  `execute_parallel_actions`. It is non-`None` only for the frozenset `tools/base.py::GATE_REFUSAL_KINDS` =
  {`NOT_REGISTERED`, `INACTIVE_SCENE`, `POLICY_CEILING`, `POLICY_PERMISSIONS`, `POLICY_PIN`, `MODE`, `OFFLINE`}.
  Each member's value is its snake_case name (`"not_registered"`, …), matching the existing members.
  **`ToolErrorKind` is serialised by `.value` everywhere:** the report, the log and `actions.jsonl`.
- **The refusal layer** goes in `ToolOutput.metadata["refusal_layer"]`. Each layer has exactly one kind:

  | Layer | Kind |
  |---|---|
  | `"forbid"` | `POLICY_PERMISSIONS` (the autonomy forbid source) |
  | `"mode"` | `MODE` |
  | `"registry"` | `NOT_REGISTERED` or `INACTIVE_SCENE` |
  | `"permissions"` | `POLICY_PERMISSIONS` |
  | `"ceiling"` | `POLICY_CEILING`, including the V7 offline AND |
  | `"pin"` | `POLICY_PIN`: a pinned policy's refusal of any non-pinned name, never `"permissions"` |
  | `"offline"` | `OFFLINE`, an `OfflineRefusal` raised inside a tool body |
  - `"registry"` means the name is not dispatchable in this registry (`NOT_REGISTERED`, `INACTIVE_SCENE`).
  - The sealed registry is just a registry, so the executor needs no special case for it. **It is never
  `PERMISSION_DENIED`**, which tools return for their own path refusals: those keep booking their NAc negative,
  unchanged. A test pins the set. The five scripts that call `record_outcome` directly
  (`scripts/orient_substrate/9_tool_competition_repro.py`, `2_full_path_probe.py`,
  `scripts/survival_world/dark_danger_probe.py`, `r2_learned_bias.py`, `scripts/exp56/common.py`) are updated in the
  item 3 commit.
- `tests/unit/test_stop_reasons.py::TestToolErrorKind` (the member count and the enumeration) is updated in the same
  commit.
- **Call sites of the required keyword.** It is named everywhere it is passed:
  - `execute_and_learn`: both sites, plus its `except` site;
  - `execute_parallel_actions`: its per-action dicts gain an `"_refused"` key, set on the executed branch only;
  - `book_refusal` passes `None`, because a human "no" still books NAc;
  - `book_machine_refusal` and `book_hard_rejection`;
  - `agent_loop.py`'s agent-fallback `except` site, a fenced hunk (Session A grants it under AD8 as a `ToolPolicy`
    consequence);
  - about 39 test call sites and the five scripts.

  They go through `Callable[..., Any]` / `functools.partial`, so mypy cannot see a miss. The miss is a runtime
  `TypeError`, which the suite catches.
- **Who may produce a gate kind.** An AST test pins that `GATE_REFUSAL_KINDS` members are constructed only in
  `runtime/executor.py` (and the `OFFLINE` mapping there). A tool body returning `MODE` cannot exempt itself from NAc.
- **Out of scope, stated.** The parallel batch's autonomy `can_execute_action` rejection is a machine refusal that
  books an NAc negative today, which is inconsistent with #1185. It is pre-existing and filed as a follow-up for an
  owner line.
- **The behavioural red gate is the enforcement, not this table.** For each kind in `GATE_REFUSAL_KINDS`, a refusal
  is driven through `execute_and_learn` **and** `execute_parallel_actions`, with spies on `NAc.observe`, the pain
  bus's publish, `ToolPainBridge`, `PlanHistoryBridge`, goal credit and cluster reward.
  - It asserts **zero writes**.
  - It asserts that the carryover (`llm_worker.record_outcome`), `recent_outcomes` and the autonomy audit log each
    got one entry.

  A booking site this table missed fails that test.
- **Booking order for `OfflineRefusal`:** the mapping to `OFFLINE` happens in `_fail_ran` **before** any booking. Otherwise
  `_fail_ran` books a `ToolPainBridge` NAc negative and pain first. The behavioural gate drives that case too.
- **`OfflineRefusal` raised inside a tool** is caught in `_run_started`'s `except` and mapped to a new
  `ToolErrorKind.OFFLINE`. It is booked like a gate refusal, so an offline session does not learn aversion to
  actuation or to lazy-download tools.
- **The behaviour change, named.** Existing `tools.allow`/`tools.deny` denials become `POLICY_PERMISSIONS`, and stop
  booking a negative NAc outcome through `record_outcome`. That goes in the CHANGELOG. The characterization golden
  (`tests/fixtures/orchestrator_characterization_golden_v1.json`) is checked for tool-usage stats.

### 1A.3 Where each new thing lives (layer rules)

`utils/audit.py::LAYER_RULES` forbids `tools → agents`, `tools → runtime` and `agents → tools`, `TYPE_CHECKING`
included.

| Thing | Module | Why there |
|---|---|---|
| `Effect`, `exceeds`, `effect_of`, the record | `tools/base.py` | Next to `Tool`; read everywhere |
| `ToolPolicy` | `runtime/tool_policy.py` | Holds `AgentPermissions` (`agents/`) and `Effect` (`tools/`); only `runtime` may see both |
| `current_policy` | read through `tools/base.py::current_policy()`, which returns a Protocol-typed handle (`PolicyHandle`, with `narrowed()`); `runtime/tool_policy.py` sets it | Tools-layer composites (`tools/`, `embodiment/`) can read it without importing `runtime` |
| The offline latch | `maxim/_offline.py` | 1A.1 |
| `closed_local` | `models/language/closed_local.py`, made safe by step 0a | It stays inside the LLM package (the router-only invariant). The CI backend-import grep adds `closed_local` with an allow-list of exactly `src/maxim/advisor/` |
| `AdvisorLog`, `read_log` | `advisor/log.py` | The writer helpers it uses live in `utils/atomic_io.py` (below) |
| Strict writers | `utils/atomic_io.py::atomic_write_json_at(dir_fd, name, payload)` and `append_line_fsync(fd, line)` | Both fsync-strict (an `OSError` raises). The module docstring names its two fsync policies (lenient: the existing helpers; strict: these) and who uses each. persistence-config.md §4 gets the line |
| The observer | `maxim/advisor/` | `LAYER_RULES` gains `advisor`, which must not import `agents`, `memory`, `default_network`, `comms`, `hivemind`, `peer`, `mesh` or `tunnel`: real top-level layers, a static fence on the existing audit. `runtime.agent_loop` cannot be expressed there (the audit matches the first segment only, and the observer needs `runtime.executor`), so the subprocess closure test (`test_advisor_gate.py::test_import_closure`) enforces that one |
| The exemption list | `tests/fixtures/effect_exemptions.json` | Only a test reads it |

`closed_local` and `_LlamaCppBackend` share one `Llama(**kwargs)` builder in `llama_backend.py` (n_ctx clamp,
`n_gpu_layers`, threads, seed), so a future GGML flag lands once. `closed_local` gets `llama_cpp` through
`require_optional_dependency("llama_cpp", extra="llm-llama")`.

### 1A.4 The stable tool verbs stay unchanged (V10)

**Owner decision, 2026-10-25.**
- `maxim.run` keeps ceiling `ALL` in this release, so `@maxim.tool` and `register_tool` tools, and same-name
  replacement, behave exactly as `docs/user/stable_api.md` documents.
- #1342's workspace-only direction stays a **filesystem-policy** fix (`allowed_dirs`), not a ceiling. Its O1 flip
  criterion is concrete: `maxim.run`'s executor carries `ToolPolicy.unrestricted(reason="maxim.run: stable verb,
  V10")` and `allowed_dirs == [home]`. A translated `None` cannot produce that.
- Ceilings below `ALL` apply only to the observer, Talk (#1431) and offline (V7). There, an undeclared user tool
  drops **loudly**: a structured WARNING naming the tool and the ceiling. The docs say so.
- `@maxim.tool(effect=Effect.READ)` and `register_tool(..., effect=)` are added as an optional form. That is
  additive, so stable_api.md gains a note, not a break.
- **Red gate:** a registered READ tool runs under `maxim.run`, and an undeclared one does too.

### 1A.5 `permissions=` is removed, not kept beside `policy=`

V3 is already a hard break, so `Executor(permissions=)`, `build_executor(permissions=)` and `AgentConfig.permissions`
are **removed** in the same change. The factories take `AgentPermissions` as their one input, and
`permissions_from_tools_settings` feeds `ToolPolicy.deny(...)`.

**It lands as one commit with the O1 flip:** about 190 test sites (42 `Executor(`, 73 `build_executor(`, 74
`AgentConfig(`) and about 17 in `src/`, including `create_npc_agent` and the sub-sim's own child executor.

**Claim surfaces edited in that commit:**
- `create.py`'s error text;
- the docstring examples in `pain_bus.py`, `pain_interceptor.py` and `bio_stack.py`;
- `docs/proprioception.md`'s example, already a `TypeError`;
- runtime-tools.md's "five logical call sites" and its builder table;
- D73's "byte-identical when unconfigured", a DECISIONS.md amendment.

`simulation/dm_schema.py::CampaignDef.permissions` has no reader. It is censused as dormant.

### 1A.6 Config version: the gate takes 1.5

#1431 (`console.talk_tools`, Session S) ships in 1.3.4 and takes `CONFIG_FORMAT_VERSION` "1.4". The gate ships later,
so **`llm.offline` takes "1.5"**, with its own fixture entries and downgrade transition. If #1431 slips past the gate,
the gate takes 1.4 instead, and whichever lands second rebases. Two claims on one version are never both shipped.

**Downgrade.** A 1.4 build reads a 1.5 file as a future minor and tolerates the key, so it runs online. That is the
stated fail-open. `maxim config downgrade` warns when it drops `llm.offline: true`.

### 1A.7 Provenance under offline: an explicit branch, not a crash

`OfflineRefusal(RuntimeError)` is deliberately not absorbed by `utils/code_tree.py::git_out`'s `except (OSError,
TimeoutExpired)`. So every `git_out` caller crashes under offline unless it branches first: `_code_stamp`,
`head_hash`, `tree_dirty`, `code_tree_sha256`, `get_version_info` and the console identity.

**Fold.** `utils/code_tree.py` is stdlib-only by contract (guard: `tests/unit/test_clean_flag_998.py`), and
`scripts/_provenance.py` loads it by path. So it must not import `maxim._offline`.
- **`git_out` alone** gains a stdlib probe: `if getattr(sys.modules.get("maxim._offline"), "is_offline", lambda:
  False)(): return None`. It imports nothing, and it is false under the by-path loader. Every caller already maps
  `None` to "unknown", or to dirty for `tree_dirty`, which fails closed. That covers `head_commit` (which
  `_code_stamp` calls), `head_hash`, `tree_dirty`, `code_tree_sha256`, `code_paths`, `tree_difference` and
  `on_disk`.
- `simulation/report.py::_code_stamp` labels the field `"unknown: offline"` when `maxim._offline.is_offline()`.
- `scripts/_evidence_records.py::unknown` refuses both forms, so an offline run cannot gate a claim (M1).
- `maxim/__init__.py::get_version_info` runs git directly inside `except Exception: pass`. It swallows the refusal
  rather than crashing, so it gets an `is_offline()` early return, and the swallow census lists it.

**Red gate:** an offline sim in a git checkout writes `report.json` with `executed_git_hash: "unknown: offline"` and
spawns nothing.

**Wiring note:** the rule for `docs/wiring/` is that a typed exception chosen so narrow catches miss it crashes every
caller those catches protected, so list each caller.

### 1A.8 The observer's report and log: the schema AvServ designs against

advisor_mode.md's draft interfaces are **superseded by this shape**, and that file says so.

**`open()`** (owner decisions 2026-10-26, from AvServ's questions, A–D; refined by the v6.6 three-lens round, v6.7)
takes, keyword-only:
- `home_dir`, `model` and `report_dir`;
- `cursor=None`, `max_calls=8`, `turn_timeout_s=120` and `log_mode=0o600`;
- **`n_ctx=None`, `max_prompt_tokens=None` and `n_gpu_layers=-1`** (A).

**Limits (A).**
- `n_ctx=None` resolves **in `closed_local`** to `min(trained context, 8192)`.
  - It is never the full trained context: a 128K default means minutes of prefill, or a memory failure at `open()`.
  - The trained context is read from the GGUF's metadata before the full load (`closed_local.trained_context(path)`;
    llama-cpp's `n_ctx=0` means unbounded and is never passed).
  - The clamp is `closed_local`'s alone. The shared `Llama` kwargs builder (§1A.3) receives an explicit `n_ctx`, so
    `_LlamaCppBackend`'s profile-driven `n_ctx`, and the PromptBudgeter's agreement with it, are unchanged.
  - The backend exposes the resolved value as an `n_ctx` int attribute.
- The completion budget is fixed at **1024 tokens**.
- `max_prompt_tokens=None` resolves to `n_ctx - 1024`.
- `open()` refuses (`AdvisorCompositionError`) when `max_prompt_tokens + 1024 > n_ctx`.
- `session.limits` is a frozen object whose attributes read back the **actual** values: `n_ctx`,
  `max_prompt_tokens`, `completion_tokens` and `n_gpu_layers`.

**`session.ingest(events, *, stop=None)`.** Each event is `{event_id, kind, occurred_at, payload, text, cursor?}`.
- **The prompt cap (B).**
  - The **session** is the one place the cap is checked: it compares `len(prompt.ids)` against `max_prompt_tokens`
    before any completion, so prefill is bounded by tokens, not by time.
  - Every report carries a named field, `"prompt": {"tokens": N, "max_tokens": M}`, not an `extra` key, so AvServ can
    size its batches from any report.
  - An over-cap prompt gives `error` with reason `prompt_too_long`, and the backend never completes.
- **Cursors (C).**
  - `cursor` is the caller's opaque feed cursor; Maxim never parses it, only echoes it.
  - **The session holds the cursor.** `open(cursor=)` sets it, and each `ok` turn advances it to its `cursor_out`.
    So a second `ingest()`'s `cursor_in` is the previous `cursor_out`.
  - Once any cursor is in play, from `open()` or from any event in the batch, every event must carry `cursor`.
    Otherwise `ingest()` raises `AdvisorInputError` before any turn.
  - With cursored events but no `open(cursor=)`, the first `cursor_in` is `null`.
  - **Maxim decides what was handled:** every event in the batch is either judged or listed in `inputs[]` as
    rejected. So on `ok`, `covered.cursor_out` is the **last event's** `cursor`, and a rejected event can never be
    re-fed forever.
  - For `error` and `abandoned`, `cursor_out == cursor_in`.
  - There is no `event_id` fallback. With no `cursor` given, both cursor fields are `null`.
- **Stopping (D, session-scoped).**
  - The library installs no signal handler and never exits (bounded_stop.md's principle). The embedder's handler calls
    **`session.request_stop(deadline_s, reason="yield")`**.
    - **"Callable from a signal handler" means exactly this:** it sets a `threading.Event` and stores the monotonic
      deadline and the reason. It takes no lock, logs nothing, does no I/O, and never raises.
    - `deadline_s` bounds how long the in-flight `ingest()` may take to return after the stop. The remaining
      dispatches are skipped, and the report is written.
    - An optional per-call `stop=` (a `threading.Event`-like object) abandons only that call.
  - **The process-wide `StopRequest` is not consulted.** It is sticky, and `MaximHandle.stop()` sets it too, so in an
    embedding process it would abandon unrelated sessions. This design also does not wait for bounded_stop PR 0.
  - **Proved behaviourally, not by the import closure.** The proof does not forbid the cancellation modules: bounded
    _stop plans module-level `StopRequest` imports in `runtime/`, and those would make an import gate unflippable.
    Instead a gate sets every process-wide stop signal before `ingest()` and asserts the turn stays `ok`. The signals
    are `models/language/cancellation.py::request_shutdown()`, the backends' shutdown event, and
    `utils/cancellation.py::request_stop` once PR 0 lands. **`closed_local` must not poll the shutdown event.**
  - **Per-token checking.** The closed backend's `complete_tokens(..., should_stop)` consults `should_stop()` on every
    sampled token, in its `stopping_criteria`. A stop ends the turn at the next token.
    - The backend reports it as `stopped=True`, its own flag, never `deadline_hit`.
    - The report is `abandoned` with reason **`"stopped"`**. The deadline gives `abandoned` with reason
      `"deadline"`.
    - Either way there is no judgement, and `cursor_out == cursor_in`.
  - **A stop during dispatch** skips the remaining calls. The report is `abandoned` with `proposals: []`. The log
    keeps the records already written, and `log_last_seq` says how far it got.
  - **Prefill.** A stop that lands during prefill takes effect when prefill returns, which the token cap bounds.
  - **After a stop:**
    - the session is stopped;
    - a further `ingest()`, or one called while the stop was already set, raises a typed `AdvisorStopped` and runs no
      turn;
    - `close()` still writes the log's close record.
- **`degraded`.** In this release the observer has **no `degraded` producer**: it loads no optional weights. The
  status stays in the schema. A rejected input keeps the report `ok`, with `inputs[].rejected` filled. §4.2's status
  table and §6's degraded paths apply to non-observer runs.

**The report:**

```json
{"_format_version": "1.0", "schema": "advisor_report/1", "session_id": "...", "report_id": "uuid4",
 "status": "ok | degraded | error | abandoned", "reason": "set unless ok",
 "banner": "ADVISORY — Maxim v<maxim.__version__>, model <gguf sha256[:16]>; no action taken; not an alert.",
 "covered": {"cursor_in": "... | null", "cursor_out": "... | null", "events": 3},
 "prompt": {"tokens": 1834, "max_tokens": 7168},
 "inputs": [{"event_id": "...", "kind": "...", "payload_sha256": "...", "rejected": null}],
 "happening": "UNTRUSTED", "would_do": {"summary": "UNTRUSTED", "executed": false}, "rationale": "UNTRUSTED",
 "confidence": {"value": 0.6, "calibrated": false, "source": "model_self_report"},
 "proposals": [{"seq": 4, "summary": "UNTRUSTED", "target": "UNTRUSTED"}],
 "refusals": [{"seq": 5, "name": "bash", "error_kind": "not_registered", "layer": "registry", "effect": "unknown", "reason": "..."}],
 "log_last_seq": 5, "provenance": {"code": "unknown: offline", "model_sha256": "...", "maxim_version": "..."},
 "created_at": "...", "extra": {}}
```

- **`banner`** uses `maxim.__version__`, never `get_version_info()`, which spawns git.
- **`would_do.summary`** is the model's narrative. **`proposals[]`** are its structured `propose_action` calls,
  linked by `seq` to the log. The docs give each its role, so they are not two homes for one judgement.
- **`provenance`** cannot be the M1 git stamp offline. It carries the GGUF's sha256 and the version, and says
  `code: "unknown: offline"`.
- **`Effect` is serialised as a sorted list of flag names**, never the `Flag` integer, which depends on declaration
  order. The same rule covers #1431's `/api/tools`.
- **Status precedence:** `error` > `abandoned` > `degraded` > `ok`. One exception: when the deadline flag is set,
  the status is `abandoned` and the truncated output is **not parsed**, so a half-written JSON is never reported as
  `error`.
- **`report_dir` may hold transient names** while `atomic_write_json_at` writes (a temp file renamed through the
  fd). Its final state is the report plus the log, nothing else.
- **The closure:** `maxim.advisor` never imports `maxim.agents` directly. `agents.permissions` reaches it
  transitively through `runtime.tool_policy` (`ToolPolicy` holds `AgentPermissions`), and that is allowed. The
  closure forbids `agents.autonomy`, `agents.llm_worker` and the loop modules.
- **Log records** carry `report_id` and `ts`, so every proposal and refusal joins to its report.
- AvServ plan 50 is read-only to us, so **the owner relays the changed shape to AvServ**, a hand-off.

### 1A.9 The invariants this adds (draft stubs for runtime-tools.md and llm-routing.md)

The advisor_mode.md guard line is superseded (prefetch root and the census-as-enforcement no longer apply).

1. **[engineering] A tool's effect is declared in its class's own `__dict__`, recorded at class creation, and checked
   at one dispatch seam.** Regression guard:
   `tests/unit/test_atonia_red_gates_effect_policy.py::test_item2_every_tool_declares_own_effect` and
   `::test_item2_process_and_code_runners_declare_all`, plus `src/maxim/tools/base.py::Tool.__init_subclass__`.
2. **[engineering] Every executor carries a `ToolPolicy`; `None` is a `TypeError`** (it joins the canonical-builder
   table). Regression guard: `src/maxim/runtime/bootstrap.py::build_executor` (required keyword-only `policy=`) and
   `tests/unit/test_atonia_red_gates_effect_policy.py::test_item3_policy_is_required_and_permissions_removed`.
3. **[engineering] Offline is a one-way, process-wide audit-hook latch: once engaged, at the Python level, no
   outbound socket connection (`connect`/`connect_ex`, loopback and `AF_UNIX` included), no `sendto`/`sendmsg` to an
   explicit destination (any family, `AF_UNIX` datagrams included), no socket of a family other than
   `AF_INET`/`AF_INET6`/`AF_UNIX` and no fd adoption without a family, no forward or reverse DNS, no spawn
   (`os.system`, `subprocess.Popen`, `os.exec*`, `os.posix_spawn*`, `os.spawn*`) and no `os.fork`/`os.forkpty`
   (owner decision 2026-10-26: on POSIX `os.spawn*` and `pty.spawn` are a fork+exec, so the fork is where the
   refusal stays typed and booked in the caller; multiprocessing's "fork" start method therefore does not work
   offline, nor does "forkserver", whose `AF_UNIX` connect is refused and which is the Linux default from Python
   3.14; only "spawn" works).** "One-way" means no non-test caller can
   disengage it: `_reset_for_tests` exists, but a guard refuses any caller outside `tests/`, aliased imports
   included. Disengaging it by `importlib.reload`, by writing its private state, or through a dynamically built name
   is in-process code, out of the model (§0). **Stated residuals:**
   - a connection opened **before** `engage()` keeps sending, because `send`/`sendall` raise no audit event;
     `utils/http`'s pool is closed by its `on_engage` callback, but other pre-latch connections (the reachy_mini
     websocket, the census's allow-listed raw-socket sites) are not;
   - fd adoption **with** an explicit family (`socket.fromfd(fd, AF_INET, ...)`, `socket.socket(AF_INET, ..., fd)`)
     is the same class: the `socket.__new__` audit event carries no fd, so an adopted pre-latch connection can keep
     sending;
   - native networking is invisible to the hook, and Windows' `_winapi.CreateProcess` is not listed (§6).

   Regression guard: `tests/integration/test_offline_latch.py::test_no_effect_subprocess`,
   `::test_engaged_latch_refuses_each_channel`, `::test_a_client_pooled_before_engage_cannot_send_after`,
   `::test_no_caller_of_reset_for_tests_outside_tests`, `tests/unit/test_offline_latch_unit.py` and
   `src/maxim/_offline.py::engage`.
4. **[engineering] The atonia composition dispatches only its pinned `propose_action` instance, imports nothing
   outside its closure, and reads only its pinned env set.** Regression guard:
   `tests/integration/test_advisor_gate.py::test_per_layer_deletion_sweep_ceiling`, `::…_pin`, `::…_sealed`,
   `::test_open_builds_all_three_layers`, `::test_import_closure`, `::test_env_read_set` and `::test_no_effect`.

### 1A.10 Tests that actually run, and actually prove it

- **Coverage.** Coverage has no subprocess support, and `lint_coverage` forbids enabling it. So every audit-hook
  branch, every status row, `engage()` and the closed backend's defang also get **in-process unit tests** (with the
  conftest latch reset, and an injected fake backend). The subprocess runs are the end-to-end proof.
- **A runnable model.** Item 1's gates need a tiny runnable GGUF fixture, named with its size, source and sha256 when
  the red-gate PR is written, and fetched in CI by the existing model-cache step. A private backend-injection seam
  (`maxim.advisor._session._backend_factory`, monkeypatched) drives the `error`, `abandoned` and schema-invalid
  branches. Gates that need llama_cpp **fail** when it is missing, never skip.
- **The per-layer sweep sets:**
  - the ceiling layer sweeps tools with `exceeds(effect, READ)`;
  - READ tools are proven refused by the pin layer and by the sealed contents;
  - the sealed-contents layer sweeps **names**.
- **The closure test's dispatch-time expectations** are stated. `Executor` imports `modes.definitions` per dispatch.
  `agents.bus`, `decisions.causal_link` and `runtime.tool_dispatch` (about 45 modules, `memory.hippocampus` among
  them) load only behind a pain detector or a tool-pain bridge, and the sealed executor carries neither.
- **The closed backend's surface** is exactly `open_closed_local_backend(...)`, plus `build_prompt(template,
  segments) -> PromptIds`, which does the whole tokenize-and-defang inside `models/language/`, and
  `complete_tokens(...)`. `maxim.advisor` never imports `llama_cpp`.
- **The censuses ride on existing lints:**
  - the swallow census is a mode of `scripts/lint_no_silent_swallows.py`;
  - the executor and registry construction pins, the `unrestricted`/`deny` callers, the direct-`.execute` sites and
    the fetch/socket/subprocess site census are each a `scripts/lint_*.py` in the CI lint job, with a unit test and a
    baseline under `scripts/` that follows the `architecture_baseline.json` precedent (`baseline_format_version`).
    `prepr.sh` then runs them automatically.
- **The widened CI grep** (`urllib3`, `requests`, `httpx.Client(`, `socket.socket(` outside `utils/http.py`) starts
  with a pinned allow-list, with reasons: `doctor/platform_detect.py`, `simulation/minecraft_harness.py`,
  `utils/net.py`, `utils/reachy_diagnostics.py` and `utils/sandbox_executor.py`'s blocklist string.
- **`auto_sense`'s two bypasses** (the auto-fire loop and the interoception `sense` call) both call `check()`. A
  refusal is logged as a refusal, not swallowed by the surrounding `except Exception`, and the probe drives both.

### 1A.11 The mode gate's name sets: a scheduled collapse, with O2 stated

`modes/definitions.py`'s `HOST_ACTING_TOOLS`, `NETWORK_CAPABILITY_TOOLS`, `FILESYSTEM_CAPABILITY_TOOLS` and
`DISPATCH_CAPABILITIES` classify tools by name, a second vocabulary beside `Effect`. Collapsing them changes what
passive mode allows for non-observer runs (O2: `write_file`, `send_message` and `call_user` would be refused). That is
a behaviour change outside the gate, so it is **filed as a follow-up** with the O2 consequence stated. Until then, a
test pins that every tool in those sets declares a matching `Effect` flag, so the two vocabularies cannot disagree.

## 2. Item 2: `Tool.effect` and the registry ceiling

> **Status: BUILT 2026-10-30** (branch `feat/1315-item2-tool-effect`). As built, where it differs from the text below:
> - The effect classification (86 classes) is in each class body; owner decisions V11–V14 settled the four judgement
>   calls. Defaults taken without asking: ambient logging is not `SELF_SCOPE`; a first-use model download is
>   infrastructure, not a tool effect; the robot link is `ACTUATE`, not `NETWORK`; `sense_tools` is `READ|GOVERN`
>   (it re-activates scene tools); a Maxim-owned directory outside the data home (the sim tmpdir, `sandbox/`,
>   `data/`) is `WRITE_LOCAL`; `READ` accompanies every flag set whose tool returns state it read.
> - `ToolEffectRefused` lives in `maxim/exceptions.py` (a `MaximError`; `tool_name`, `effect`, `ceiling` as sorted
>   flag-name lists) and is re-exported by `tools/registry.py`. Every site that catches one logs it through
>   `tools/registry.py::log_effect_refusal` (event `tool_effect_refused`).
> - The admission check is `tools/registry.py::EffectAdmission`, a mixin both registries inherit. Under `ALL`, a
>   replacement by a different class logs a `tool_replaced` WARNING; the same class re-registered (a scene refresh)
>   is silent. Re-registering the same instance is never a replacement.
> - `ToolRegistry.register_many` is the all-or-nothing batch that `generate_tools_for_entity` uses; the entity joins
>   the `EntityMap` only after its tools register. `register_scene_tools` admits the whole batch before evicting.
> - `Effect.ALL` is attached after the class, so iterating `Effect` yields the seven flags on Python 3.10 too.
> - `register_tool(tool, effect=)` declares the effect on the tool's class through `tools/base.py::declare_effect`,
>   which refuses to rewrite a different existing declaration.
> - The pinned literal-name helpers (§2.1) are checked by
>   `tests/unit/test_tool_effect.py::test_pinned_import_helpers_take_only_literal_or_table_names`: three call sites
>   pass a variable, each read from a module-level table of string literals.
> - Ten item-2 red gates flipped. `test_o2_o5_...` stays red until item 3's dispatch check.
> - **Review round (three lenses, 2026-10-30; reports under `docs/plans/reviews/item2/`), folded:**
>   - V15 and V16 (§1). `test_item2_effect_record_semantics` is amended to V15, as its docstring says.
>   - `effect_of` returns `ALL` for a class whose own `effect` no longer matches its record, so a stale narrower
>     record can never reach a later check (item 3's dispatch). `Effect.ALL` cannot be reassigned (a metaclass).
>   - **The registry logs every refusal where it raises** (`tool_effect_refused`, with the reason), not each
>     catching site, because 19 registration `try` blocks outside the diff catch `except Exception`. The three
>     sites the diff touches log at debug only, so a refusal is logged once.
>   - `SimToolRegistry.register_many`; `register_tool` checks the name before it declares; `autonomy_level` gains
>     `COMMS` (it asks the operator); the stub `effect=` is required.
> - **Not built here, and owed by item 3:**
>   - §2.2's prompt-roster bullet (`permits` applies the ceiling) is item 3's, with the dispatch seam.
>   - **Prerequisite before any registry below `ALL` ships:** the batch `try` blocks around `register` (for
>     example `simulation/orchestrator.py`'s and `embodied_runtime/agentic_runtime.py`'s) stop at the first
>     refusal and drop every later tool in the block. Each needs a per-tool handler.
>   - Entity acquisition (`Executor._handle_entity_acquisition`) reparents the entity before its tools register,
>     so a refusal leaves a body-owned entity with no tools.
>   - Under a ceiling, one `ALL` affordance (`ModulatorAffordanceTool`) keeps every tool of that entity out,
>     its `READ` `sense_*` tools included (all or nothing). Talk (#1431) should know this.
> - **Delta round** ([`code_delta.md`](reviews/item2/code_delta.md)): D1 (fail-open, folded: a deleted declaration is a
>   mismatch, and a class that declared once can never be declared again, tracked in `_EFFECT_DECLARED`); D4, D5, D6
>   and D7 folded (`Effect` is exported from `maxim.tools`; a top-level `maxim.Effect` would be an owner call). Left
>   for a follow-up, all fail-closed: D2 (a refusal at a broad-handler site logs twice), D3 (the invalid-effect
>   warning is once per class lifetime), wire F4 (`tool_replaced` can name one qualname twice).
> - **Second delta round** ([`code_delta2.md`](reviews/item2/code_delta2.md)): D1's fix holds; D8 (fail-open, only with
>   a third-party base whose `__init_subclass__` skips `super()`) folded: `declare_effect` refuses a class with no
>   record. The micro-round on that fold ([`code_delta3.md`](reviews/item2/code_delta3.md)) returned **STOP**; its two
>   fail-closed NITs (a class under a hook that skips `super()` can no longer take `register_tool(effect=)`, only
>   the undeclared `ALL` form; the refusal message names one cause of two) go to the follow-up with D2, D3 and F4.
> - Stated residual: a deregister followed by a register is not a replacement, so the below-`ALL` replacement rule
>   does not see it. The effect bound still holds for the new tool.

### 2.1 The flags, defined

A tool's effect is **everything it can cause**, through itself or through any collaborator it holds or is wired to
later, not what it is for.

| Flag | Means |
|---|---|
| `READ` | Reads state; changes nothing outside process memory. |
| `SELF_SCOPE` | Writes only Maxim's own persisted state under the data home (memory, NAc, its own logs). |
| `WRITE_LOCAL` | Writes, moves or deletes files outside the data home. **Not "runs processes or code":** a tool that spawns a process, or runs code in-process (`exec`, `eval`, or `importlib` of a model-chosen name), can cause anything, so it declares `ALL`. The one exception is a fixed argv on the pinned exemption list `tests/fixtures/effect_exemptions.json` (§1A.3), with a reason (the git tools: fixed argv plus `tools/base.py::GIT_HARDENING_ARGS`). The exemption holds **online only**: under offline the hook refuses every spawn (V8), so the git tools drop. Git's config-driven filters run code even with that hardening. **Checked:** an AST test over the `Tool` subclass closure fails any class whose methods reach a sink, unless it declares `ALL` or is on the exemption list. **Sinks:** `subprocess.*`, `os.exec*`, `os.system`, `os.popen`, `os.spawn*`, `os.posix_spawn*`, `os.fork*`, `pty.spawn`, `asyncio.create_subprocess_*`, `multiprocessing.*`, `exec`, `eval`, `compile`, `importlib.import_module`, `__import__`, `runpy.*` and `ctypes.*`. **Reach:** the call graph across `src/maxim` modules (resolved imports, `self.` methods, module functions), not only the same module. **Sink rules:**
- An import or compile sink counts only with a **non-literal** name argument. `__import__("logging")` is not a
  sink, and `compile` matches the builtin `Name` only, never `re.compile`.
- `require_optional_dependency` / `optional_dependency_available` (`utils/optional_deps.py`) and
  `runtime/lane_backends.py::resolve_backend_class` are pinned literal-name helpers. A test checks that every call
  site passes a string literal, an `EXTRA_FOR_IMPORT` key or a `BACKEND_CLASSES` value.

Reach is function-level, not module-level: a helper module that also contains `subprocess` does not taint callers
that never reach it.

**Stated blind spots:**
- `getattr(os, "system")`-style dynamic lookup;
- callables passed in as arguments;
- calls on receivers other than `self` and modules (`self._sandbox.execute()`, a controller or router instance);
- inherited methods outside the subclass's module;
- decorators, `functools.partial`, and thread or executor targets;
- third-party code.

Under offline the hook refuses every spawn regardless, so these matter only for online ceilings. Each is the future-code adversary, which the review rounds and the census lists, not this test, catch. |
| `NETWORK` | Egress the tool causes off the machine, DNS included, GETs included; not the configured LLM lanes (V11) or the configured robot body, whose channel is `ACTUATE` (V16). |
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
  `WeakKeyDictionary`. `Effect(0)` raises `TypeError`. A non-`Effect` value records `ALL` with one WARNING (V15;
  the v6 text had a `TypeError`).
- **The lookup.** `effect_of(obj)` reads `type(obj)`'s record. A type with no record (a duck-typed tool, a mock, a
  class made outside `Tool`) is `ALL`, and so is a type whose own value no longer matches its record (V15).
- **Mutation is visible.** The mismatch check compares the record with `type(obj).__dict__.get("effect")` only, never
  an inherited `getattr`. **An absent value with a recorded `ALL` is consistent, not a mismatch.** A present value
  that differs from the record, or a deleted declaration (`del cls.effect`, delta D1), is a mismatch: the tool counts
  as `ALL`, so a ceiling below `ALL` refuses it loudly and `ALL` still admits it (V15). Undeclared test tools and
  stable `register_tool` tools still register under a ceiling of `ALL`. `create_stub_tool` and `@tool` put `effect` in the class body they build.
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
| `type(tool).effect`, reassigned or deleted | `create_stub_tool` already reassigns class fields | Record versus own-`__dict__` mismatch: the tool counts as `ALL` (V15) |
| Inherited `effect` | Subclass of a READ tool writes | Own `__dict__` only; undeclared means `ALL` |
| `Effect(0)` | Passes every ceiling | Refused at class creation |
| Instance attributes (`effect`, `auto_fire`, `kind`) | Set per instance | `effect` is never read from the instance; `auto_fire` is moot once `auto_sense` is checked (V6); `kind` is not the gate |
| A collaborator wired after admission | The orchestrator attaches backends later | The declaration covers the collaborator slot, whatever is plugged in |
| A same-name replacement | Swaps a READ tool for an acting one | Refused below `ALL`, warned at `ALL`; the observer pins instances (§3) |
| A tool with no record | Duck-typed or mocked | `ALL` |
| An out-of-tree registry subclass | Skips `_admit` | Dispatch-time check |

## 3. Item 3: a required `ToolPolicy` on every executor

> **Status: BUILT 2026-10-31** (branch `feat/1315-item3-tool-policy`). As built, where it differs from the text below:
> - `OFFLINE_CEILING`, `effective_ceiling`, `dispatching` and `require_policy` live in `runtime/tool_policy.py`
>   beside `ToolPolicy`. `current_policy()` and the Protocol `PolicyHandle` live in `tools/base.py`, the
>   context variable too, so a tools-layer composite reads it without importing `runtime`.
> - The seam is two methods. `Executor._permission_denial` holds the NAME gates (forbid, mode, permissions) and
>   `Executor._object_refusal` the OBJECT gates (the permissions on the tool's own `kind`, the ceiling, the pin),
>   run on the one `registry.get`. Both return a typed `_Refusal` (kind, layer, message). `permits()` runs both for a
>   registered tool, so the roster never offers an above-ceiling tool (§2.2's roster bullet).
> - `Executor.check(tool)` returns the refused `ToolOutput` (typed) or `None`. An unregistered name under an
>   allow-list is refused at layer `"permissions"`: the allow-list judges the name before the registry is asked.
> - `Tool.run` re-raises `OfflineRefusal` (its broad `except` would otherwise turn it into an untyped failure), and
>   `Executor._run_started` maps it to `OFFLINE` before any booking.
> - **Owner decision 2026-10-31: under the offline latch the tool-pain bridge books nothing** (`_suppress_nac` gains
>   `is_offline()`, beside the human-driven and speculative cases). A body that meets the latch raises after
>   `record_tool_start`, whose pending NAc event later pain could link to. Dispatch's `record_outcome` still books
>   what ran. The atonia observer, the one offline consumer, is built with no bridge anyway.
> - **Owner decision 2026-10-31, a red-gate scenario fix:** `test_refusal_booking_red_gate.py`'s
>   `offline_in_tool_body` case engages the latch, as it is whenever shipped code raises `OfflineRefusal`
>   (`_offline.refuse` and the audit hook raise only then). Its assertions are unchanged.
> - `record_outcome(refused=)` routes through `tool_dispatch._learning_nac`: a gate kind sets `nac=None` (the
>   `book_machine_refusal` precedent), a non-gate kind raises `ValueError`. Callers pass
>   `tool_dispatch.gate_refusal_of(result)`. The agent-fallback booking moved into
>   `tool_dispatch.book_agent_fallback_failure` (Session A's grant), which passes `gate_refusal_of(result)`.
> - The parallel batch now writes an autonomy audit entry per executed action (`_audit_parallel_action`), which the
>   booking gate demanded.
> - The console builds its policy with `tool_policy.policy_from_tools_settings`; `agents/permissions.py::
>   tool_permissions_from_settings` is deleted (§1A.5).
> - `maxim.run` confines its filesystem tools to the run's home (`allowed_dirs_override`), #1342's first half; the
>   prefetch half stays on #1342.
> - **Not built:** the sub-sim child's registry ceiling. `build_tool_registry` is pinned over 200 lines, and the
>   child's executor policy (the dispatcher's, narrowed) already refuses at dispatch and in `permits()`. The registry
>   ceiling would add admission-time refusal only.
> - The census lives in `tests/unit/test_tool_policy_census.py`. Its direct-call table pins two sites the notes did
>   not name: the sim world's body reflexes (`integration/bio_enrichment.py::_dispatch_reflex_tool`) and the
>   narrator's actor composite (`OrchestratorActorTool.execute`).
> - Rule T extractions: `api._build_run_executor`, `cli._cli_agent_config`,
>   `orchestrator._aut_agent_config` / `_orchestrator_agent_config`,
>   `AgenticRuntimeMixin._build_runtime_executor`, `bootstrap._check_build_executor_args`,
>   `tool_dispatch._stash_timeout_retry` / `_observe_energy_cost` / `_parallel_entry`. Every touched pinned
>   function shrank.
> - **Deletion probes** (each mechanism removed, its gate red, then restored): the ceiling check; the pin;
>   `dispatching`; each of `auto_sense`'s two checks; `_learning_nac`; the offline bridge suppression;
>   `require_policy`; the spawn refusal; the home confinement.

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
  - the pins are snapshotted as `MappingProxyType(dict(instances))`, so a caller's dict cannot add pins later;
  - `__copy__` and `__deepcopy__` return `self`, and `__reduce__` raises.

  Tests cover construction, assignment, and mutation of the source dict. It holds `permissions: AgentPermissions`,
  `ceiling: Effect`, `pins: Mapping[str, Tool] | None` and `reason: str`. The slot is `pins`, not `pinned`:
  `__slots__` and a same-named classmethod raise `ValueError`. It lives in memory only, never persisted.
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
- **Refusals are typed.** `tools/base.py::ToolErrorKind` (an `Enum`) gains `NOT_REGISTERED`, `INACTIVE_SCENE`,
  `POLICY_CEILING`, `POLICY_PERMISSIONS`, `POLICY_PIN`, `MODE` and `OFFLINE`, and every refused dispatch carries one, plus the layer and
  the effect.
  The `effect` of an unregistered name is `"unknown"`; no lookup builds a registry to find it.
- **A gate refusal is not a tool failure.** How each refusal is booked is §1A.2's table (superseding v3's text,
  which named the wrong counters and the wrong site).
- **`auto_sense` (V6)** calls `executor.check(tool)`, the same check without dispatch, before **both** of its direct
  calls: the auto-fire loop and the interoception `sense` call (§1A.10). `check()` returns a typed refusal and never
  raises.
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
  - `api.run` (both branches): `unrestricted(reason="maxim.run: stable verb, V10")`, with #1342's workspace-only
    `allowed_dirs` (§1A.4).
  - `create_full_agent`: from `AgentConfig`.
  - The foundry, the Minecraft harness and both embodied builds: `unrestricted(reason=...)`.
  - `console/handle.py`: `deny(...)`, or `only(...)` once #1431 lands.
  - `cli.py`, the orchestrator's AUT and narrator, `create.py` and `load.py`: through `AgentConfig`.
  - `create_npc_agent`: `policy=None` (it builds no executor).
  - The sub-sim's child: `current.narrowed(...)`. Its grandfathered `FearGatedExecutor` swallow is not relied on.
- **Under offline (V7),** the dispatch seam ANDs the ceiling with `OFFLINE_CEILING`, once (§1A.2). It is not
  narrowed at each build site.
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
  - `open(...)` returns `AdvisorSession`; its full signature is §1A.8's, which wins;
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
  no profile, does no lazy load, has no cost tracker and no `atexit`, and reads no env. It exposes exactly three calls
  (§1A.10): `open_closed_local_backend`, `build_prompt(template, segments)` (the whole tokenize-and-defang), and
  `complete_tokens(prompt_ids, *, grammar, max_tokens, deadline, should_stop)`. The prompt cap is checked by the
  session (§1A.8), not here. It does not import
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
    - **Refuse.** If an id still has a refused bit (the same predicate as above, `attr &
      (CONTROL|USER_DEFINED|UNKNOWN)`), the event goes into the report's `inputs` as `rejected:
      "special_token"`. That keeps it visible, so dropping an event cannot hide it;
  - the concatenated `List[int]` goes to `create_completion`, which accepts it, with `grammar=` built from the closed
    schema.

  The vocab test enumerates the loaded GGUF's tokens **by attribute**, not by "special-looking" strings: every
  non-NORMAL token's text, in plain, fullwidth, split and nested forms, is fed as event text. It runs against the
  vendored command-r, phi-3, qwen2 and llama-bpe vocabs. It asserts that no id with a refused attribute bit appears
  inside an untrusted span, and that no BOS appears there either. The vocabs come from llama.cpp's `models/`
  directory at a pinned tag, as test fixtures, since the wheel does not ship them. No vendored vocab has a
  single-character special token, so a synthetic test, with the attribute lookup monkeypatched to mark one
  USER_DEFINED, exercises the refuse branch. The test also runs against the **installed** llama-cpp-python's
  tokenizer: `pyproject` floors it at 0.3.8 with no ceiling, so attribute semantics must not drift unseen. The router's
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
   - **The hook.** `maxim/_offline.py` (§1A.1) installs one `sys.addaudithook` the first time `engage()` runs. Audit hooks
     cannot be removed, so it is never installed in a process that never engages: an ordinary pytest run carries no
     hook, and `import maxim` stays light.
     - **Self-engage in children.** `maxim/__init__` does one cheap, stdlib-only check of `os.environ.get(
       "MAXIM_OFFLINE")` through `maxim._offline.truthy`, the one canonical parser (§1A.1), and only when that is
       truthy does it engage. So every Python
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
     - **every `AF_UNIX` connect**, CLI included, with no list. `docker.sock`, syslog, forkserver and `Manager` are
       refused under offline, and the docs say so. asyncio's `socketpair` makes no connect;
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
     - `lane_backends.py::_maybe_auto_spawn_server` returns early on `is_offline()`, before its `:8100` probe, its
       `lsof` fallback or a spawn. The startup predicate checks the **built** lanes, so a lane the build would have
       auto-spawned is refused by name;
     - an `ssh -L` forward, a reused server and a re-bound port therefore have nothing to impersonate.
   - **The boundary, stated in the docs.** The latch sees the Python `socket`, `subprocess` and `os` modules
     **only**. It is blind to:
     - **native networking:** libuv (uvloop), gRPC, libzmq, libcurl, Rust clients. **`ddgs` is one of them:** it dials
       through primp, a native client that raises no audit event (found by the red-gate census, 2026-10-26, when a
       stub dry run sent one real DuckDuckGo request). Its call-site refusal (§6, layer 2) is therefore its only
       in-process defence, and the census drives it through a blocking sentinel;
     - **un-audited spawns:** multiprocessing `spawn`'s `_posixsubprocess.fork_exec`.

     So:
     - (a) the census has a `latch-blind` category for event-loop implementations and native-network libraries;
     - (b) under offline, `console/server.py` and `comms/api.py` run uvicorn with `loop="asyncio"`;
     - (c) a spawn-mode child is latched by the self-engage above;
     - (d) anything native that remains is the sandbox's (AD5).
   - **Sub-processes.** Under offline the hook sees none start (V8).
     - Spawn-mode multiprocessing children start through the unaudited `fork_exec`, inherit `MAXIM_OFFLINE=1`,
       `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` and `YOLO_OFFLINE=true`, and self-engage.
     - **Children spawned before `engage()` run unlatched,** which is a stated residual. That is why `engage()` is the
       **first statement of `cli.py::main`** under `llm.offline` (see §"Engaging the latch"), before
       `ensure_blackwell_guards`' `nvidia-smi` spawn and every other startup subprocess.
     - `host_tool_env`'s allow-list gains the offline vars, for online-started runs that engage later.
   - **Inbound is not refused:** `bind`, `listen` and `accept` pass the hook, so `maxim serve` can listen on
     `127.0.0.1`. Ingress is item 4's and the sandbox's (§0).
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
- **`OfflineRefusal` subclasses `RuntimeError`, never `OSError`.** A socket-shaped refusal that is an `OSError` would
  be absorbed by the narrow catches that return plausible values: `runtime/orphan_reaper.py::_find_orphans_ps` →
  `[]`, `utils/code_tree.py::git_out` → `None`, `tools/base.py::git_root_refusal` → a refusal string. The swallow
  census covers every handler around a census site that catches a base of `OfflineRefusal` (`Exception`,
  `RuntimeError`, `BaseException`), not only `except Exception`.
- **Startup order.** Under `llm.offline`, `cli.py::main` reads `llm.offline` first (a config read: no spawn, no
  socket), then calls `engage()`, then `ensure_blackwell_guards`. Every `api` verb's build does the same, before any
  subprocess. The GPU guards that need `nvidia-smi` (`ensure_blackwell_guards`) then refuse
  loudly at WARNING with their reason. They no longer drop silently at DEBUG, and a GPU host is told it runs without
  them.
- **No offline run carries a code-provenance stamp.** `_code_stamp`, `get_version_info` and the console identity all
  spawn git. So an offline run's `report.json` says `executed_git_hash: "unknown"` and records the refusal, and by
  the weak-evidence rule (M1) it **cannot gate a claim**. A spawn-free stamp read from `.git` files (`HEAD`, the ref,
  `packed-refs`, with `code_tree_sha256` marked unavailable) is a follow-up, and it must not run before engage.
- `engage(reason)` is one-way.
- `is_offline()` returns the latch. It engages on any call that sees `MAXIM_OFFLINE` truthy, not only the first read,
  so a later raise counts.
- `llm.offline` is read at CLI startup and at the first router or `api` verb build. A running `maxim serve` picks it
  up on restart, or with #1411's config generation once that covers it.
- The observer always engages.
- **Tests reset the latch through `maxim._offline._reset_for_tests()`.** It exists in `src/`, but **a lint forbids
  any caller outside `tests/`**, so production code cannot disengage the latch. It disengages, restores the five
  `os.environ` keys to their pre-engage values, and keeps the `on_engage` callbacks. The conftest autouse fixture
  calls it after each test, and **fails** if `maxim._offline` is loaded but `_reset_for_tests` is missing. The hook
  is inert when not engaged, so it costs the suite nothing.

**The fallbacks become loud.** A missing encoder or semantic weight under offline logs one structured WARNING per
process, records a `degraded` reason, and the observer's report says `degraded`. decision_log stamps it; it writes at
verbosity ≥1, which is noted.

**Config:**
- `llm.offline` goes in the SHAPE-FROZEN `LLMConfigSection`.
- `CONFIG_FORMAT_VERSION` **1.5** (§1A.6): #1431 takes 1.4 in 1.3.4. It gets its own fixture entries and the #974
  downgrade.
- No env alias, and the key stays off `_NON_SECURITY_FIELDS`.
- It wins over every auto-download switch.
- **Fail-open on downgrade, stated:** a 1.4 build tolerates the unknown key and runs online. `maxim config downgrade`
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
  - the audit hook's own log shows no connect, send, resolve or spawn at all, except the stated pre-engage residuals.
- That replaces v1's "network_guard never reached", which was unobservable: `network_guard.attempts` does not record
  loopback or IP-literal calls.

**Inputs, and how each could lie.**

| Input | How it could lie | Answer |
|---|---|---|
| Proxy env | Carries a loopback URL off the box | Under the latch: `mounts=None` and `trust_env=False` on every client; the socket latch sees the real connect target |
| A hostname | `localhost` → `/etc/hosts`; DNS rebinding | No name and no address are allowed; every forward and reverse lookup is refused at the hook |
| UDP | mDNS (`zeroconf`), DNS to a literal resolver | `socket.sendto`/`sendmsg` to any INET destination refused (V8) |
| An `ssh -L` forward Maxim made, or a reused server | Loopback that is remote | No endpoint is allowed under offline (V8); every loopback connect refuses |
| A native connect path | uvloop, gRPC, libcurl: no audit event | Stated boundary; `latch-blind` census category; `loop="asyncio"` under offline; the rest is the sandbox's |
| A spawn-mode child | Un-audited `fork_exec`; an inert hook | It self-engages in `maxim/__init__` from `MAXIM_OFFLINE` |
| A swallowed refusal | A broad catch, then the library downloads anyway | The socket latch still refuses; the ledger records it; the last-call refusal; the swallow census |
| A router built before `engage` | Live cloud backends | The predicate is checked at dispatch |
| `MAXIM_OFFLINE` raised later | Ignored | Checked on every `is_offline()` |
| A model-chosen command | `bash curl ...` | V7 removes those tools; the hook refuses every spawn (V8) |
| `PATH`, an explicit `env=`, a git config | A hostile `git`; a child env without the latch vars; a config-defined filter | Every spawn refuses under offline (V8) |
| HF or ultralytics already imported | The constant was read at import | The module globals are set; the red gate verifies behaviourally |
| ultralytics | Analytics, pip auto-install, its own downloader | `YOLO_OFFLINE="true"` before import; path-only `YOLO` |
| Config downgrade | 1.3 ignores `llm.offline` | Stated, plus a downgrade warning and version pinning |

## 7. How the five fit, and the order

0. **Step 0a, lazy package inits (V9),** its own PR (§1A.1). **Step 0b, `maxim/_offline.py`:** `engage`,
   `is_offline`, `truthy`, the hook, the ledger, and `code_tree`'s offline branch (§1A.7).
1. Item 2: `Effect`, `effect_of`, `exceeds`, `_admit`, the classification, the declaration/sink lints, and the
   optional `effect=` on the stable tool verbs (V10).
2. Item 3: `ToolPolicy`, the five-gate seam with the V7 AND, the booking table (`record_outcome(refused=)`), typed
   refusals, `current_policy`, `auto_sense`'s two calls. The `permissions=` removal, about 190 test sites and the O1
   flip land as **one commit** (§1A.5).
3. Items 1 and 5: the shared `Llama` builder, `closed_local`, the sealed registry, the pinned policy, the turn, the
   strict `atomic_io` writers, the log and the report.
4. Item 10's call-site refusals, in parallel with 1–3 once 0b has landed.

Each item:
- has its strict red gates before code;
- proves each guard by deleting it;
- gets a three-lens review;
- adds its `changelog.d/` fragment (M83; never an `[Unreleased]` line);
- runs `scripts/run_mypy_public.sh` (it lists `executor.py`, `api.py`, `create.py`, `load.py` and
  `loop_perception.py`).

The CHANGELOG text for the V3 break is drafted in the item 3 PR. Its red gates' macOS-only branch (`F_FULLFSYNC`) is
tested with a monkeypatched `fcntl` on Linux CI.

**Rule T, per step** (`src/maxim/utils/function_length_baseline.json`; extract the touched lines, or record an
exemption):

| Step | Pinned function | Plan |
|---|---|---|
| 0a | `runtime/lane_backends.py::_maybe_auto_spawn_server` (as built) | Its profile read moved to `_lane_profile_is_cloud`, 271 → 259. `load_llm_config` was not edited: it ensures through `_normalize_profile` |
| 2 | `runtime/bootstrap.py::build_executor` | Extract the policy plumbing |
| 2 | `runtime/agent_factory.py::AgentFactory.create_full_agent` | Extract the executor build |
| 2 | `api.py::run` | Extract the policy and `allowed_dirs` setup |
| 2 | `cli.py::_main_impl` | Extract the `AgentConfig` build (Session A grants the hunk) |
| 2 | `simulation/orchestrator.py::start_simulation_mode` | **Fenced: a reviewed `touch_exemptions` record** (AD8), for the two `AgentConfig` sites |
| 2 | `embodied_runtime/agentic_runtime.py::AgenticRuntimeMixin._start_agentic_runtime` | Extract |
| 2 | `runtime/tool_dispatch.py::execute_and_learn`, `::record_outcome` | Thread `refused=` through a helper |
| 0b/4 | `runtime/lane_backends.py::_maybe_auto_spawn_server` | A one-line early return; extract an `_offline_refusal()` helper |
| 4 | `runtime/lane_backends.py::build_primary_router`, `models/language/router.py::LLMRouter._try_provider` | The predicate goes in one helper, `_offline_lane_refusal(lane)` |
| 1 | `runtime/bootstrap.py::build_tool_registry` | Only if the ceiling becomes its parameter; otherwise untouched |

**Files owned elsewhere, and hand-offs:**
- `tools/internet_search.py` (H): the urllib3 hunk.
- `console/handle.py` (S): the `deny` policy.
- `models/audio/tts.py` (S): the `ensure_tts_voice` refusal.
- `console/schemas.py::ActionRecord` (S): the optional `error_kind` field.
- `runtime/config_loader.py`: the gate takes 1.5; S's #1431 takes 1.4 (§1A.6).
- `orchestrator.py`, `agent_loop.py` and `cli.py` (A, AD8): `ToolPolicy` call sites only.
- `runtime/loop_perception.py`: `auto_sense`'s two checks. It is not fenced.
- **CLAUDE.md** (the process session): a stub for the `closed_local` entry point and its CI-grep allow-list, and the
  §Environment Variables line naming the canonical truthy parser (it becomes `maxim._offline.truthy`, and
  `annotation_disabled_via_env` delegates to it, so the old pointer keeps working).
- **runtime-tools.md:** "one seam, five gates", the builder table, the four invariants (§1A.9). **llm-routing.md:**
  `MAXIM_OFFLINE`. **persistence-config.md:** the strict writers.
- **DECISIONS.md:** D73's amendment.
- **AvServ, through the owner:** the §1A.8 report shape.
- **Every session:** notice of step 0a's import-timing change.

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

## 9. Fold ledger (v1 → v6)

Rows marked *(superseded by V8)* describe mechanisms V8 deleted (the spawned-server endpoint, psutil ownership,
realpath-keyed executables, `pinned_executable`); they are history, not instructions.

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
| 3 | Absolute executables keyed on `realpath`; the env rule *(superseded by V8)* |
| 4 | Live-child endpoint; the reuse path refuses *(superseded by V8)* |
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
| 1 | Port must refuse before the spawn; listener ownership proven via psutil before the key is sent; spawner state, not `poll()` *(superseded by V8)* |
| 2 | (realpath, argv-prefix) pairs; `os.system` always refused; no shell pinned *(superseded by V8)* |
| 3 | `add_bos=False`; per-id attribute check; escape or visible refusal; the vocab test by attribute |
| 4 | Key in `__new__`; unconditional `__setattr__`; no subclasses; `type(...) is` |
| 5 | Four loader variables, grouped by exec and `dlopen` |
| 6 | "Processes or code"; the AST check and `tools/effect_exemptions.json` |
| 7 | Git tools use `pinned_executable` *(superseded by V8)* |
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

**v5 delta round** ([review](reviews/atonia_gate/v5_delta.md), STOP):

| Finding | Fold |
|---|---|
| 1 | Auto-spawn returns early under offline; the predicate checks built lanes |
| 2 | `engage()` first in `main()`; the GPU guards refuse loudly; pre-engage children are a residual |
| 3 | Offline means solo; inbound is not refused |
| 4 | Leftover text fixed: AF_UNIX, the red-gate wording, the hostname row, self-engage, children, the V8 row |
| 5 | Offline runs cannot be M1-stamped; spawn-free stamp is a follow-up |
| 6 | `OfflineRefusal(RuntimeError)`; the swallow census widened |
| 7 | Non-literal import sinks only; pinned literal-name helpers; function-level reach |
| 8 | Receiver, MRO, decorator and target blind spots stated |
| 9, 11 | Correct as folded; wording only |
| 10 | One predicate for both stages; test against the installed tokenizer |

**Three-lens round** (v6):

| Lens · finding | Fold |
|---|---|
| Executor B1, Architecture B1/B2 | V9 lazy inits, step 0a (§1A.1) |
| Executor S1 | Absent-and-`ALL` is consistent (§2.2) |
| Executor S2, Wire B2, Architecture N3 | V10 (§1A.4) |
| Executor S3 | Per-layer sweep sets (§1A.10) |
| Executor S4 | Three-call closed surface; defang inside `models/language/` |
| Executor S5 | Runnable GGUF fixture, injection seam, fail-not-skip |
| Executor S6 | In-process unit tests for coverage |
| Executor S7, Wire S8 | Both `auto_sense` calls; `check()` never raises |
| Executor S8, Architecture N5 | Rule T table (§7) |
| Executor S9 | `_offline` is step 0b |
| Executor N1, Architecture S3 | `permissions=` removed; one commit (§1A.5) |
| Executor N2, Wire S1/S2, Architecture S2 | One seam, booking table (§1A.2) |
| Executor N3 | Config spells `pytorch`; the predicate uses the config spelling |
| Executor N4 | Status precedence; `fcntl` monkeypatch; mypy-public; `[Unreleased]` |
| Executor N5, Architecture N6 | Ledger markers; advisor_mode.md markers |
| Architecture S1 | Scheduled collapse, consistency test (§1A.11) |
| Architecture S4 | Layer homes; `advisor` in `LAYER_RULES` (§1A.3) |
| Architecture S5 | V7 at dispatch, once |
| Architecture S6 | `maxim/_offline.py`, one truthy parser, offline implies internet-lowered |
| Architecture S7 | §1A.9 invariant stubs |
| Architecture S8 | Censuses as `scripts/lint_*`; the swallow census in `lint_no_silent_swallows` |
| Architecture S9 | Strict `atomic_io` writers |
| Architecture N1, N2 | Shared `Llama` builder; the CI-grep allow-list |
| Architecture N4 | `Effect` as sorted names |
| Wire B1 | `code_tree` offline branch (§1A.7) |
| Wire S3 | `ToolErrorKind.OFFLINE` |
| Wire S4 | Config 1.5 (§1A.6) |
| Wire S5, S6 | Report schema; `report_id`/`ts` in records (§1A.8) |
| Wire S7 | Grep allow-list (§1A.10) |
| Wire S9 | Call sites and claim surfaces (§1A.5) |
| Wire NITs 1–9 | `SimToolRegistry` noted; `kind` snapshot at admission; `ConcurrentExecutor` export-only; `_API_SUBMODULES`; conftest pop before import; `host_tool_env` is defence in depth; the optional `ActionRecord.error_kind`; `Effect` serialisation; startup order |

**v6 delta round** ([review](reviews/atonia_gate/lens_delta_v6.md)):

| Finding | Fold |
|---|---|
| 1 | `_ensure_user_profiles()` in every accessor; ten readers migrated; CI grep; red gate |
| 2 | `load_llm_config` row in the rule T table |
| 3 | A stdlib `sys.modules` probe in `git_out` alone; the full entry list |
| 4 | Read-side implication in `session_internet_lowered`; the stdlib-only AST guard |
| 5 | The `runtime.agent_loop` fence left to the closure test |
| 6 | `GATE_REFUSAL_KINDS`; never `PERMISSION_DENIED`; `book_machine_refusal` semantics with the carryover kept; `refused` required; five scripts |
| 7 | The full `ActionRecord` contract hand-off; the SHAPE-FROZEN claim dropped |
| 8 | `get_version_info` early return; swallow census |
| 9 | `AttributeError` fallback, `TYPE_CHECKING` mirrors, the import sweep, closure expectations |
| 10 | The `ToolErrorKind` member-count test updated |
| 11 | The CLAUDE.md parser-line hand-off |
| 12 | Already present (the draft-interfaces marker) |

**Wire round on v6.1** ([review](reviews/atonia_gate/wire_delta_v6_1.md)):

| Finding | Fold |
|---|---|
| W1 | Ensure inside `_normalize_profile`; added to the grep |
| W2 | `download.llm_models()` accessor; `LLM_MODELS` grep |
| W3 | Failure never cached; raises on every access |
| W4 | The grep forces every reader (about 30); outcome red gate over every public reader |
| W5 | `get_profile` does not normalise |
| W6 | `patch.dict` note |
| W7 | `PlanHistoryBridge`'s direct site gets the branch |
| W8 | The pain row corrected; the behaviour change named |
| W9 | `OFFLINE` mapped in `_fail_ran` before booking |
| W10 | Every required-keyword call site named |
| W11 | The `"_refused"` dict key; the batch rejection filed as out of scope |
| W12 | Gate kinds constructed only in the executor (AST test) |
| all of #6 | The behavioural zero-writes red gate is the enforcement |

**Red-gate PR's three-lens review** (decisions the notes owed, v6.3; the reviews are in this directory, `redgates_*.md`):

| Finding | Fold |
|---|---|
| `pinned` is both a factory and a slot (Python rejects it) | Slot `pins` |
| `error_kind` serialised by name vs `.value` | `.value` everywhere |
| Layer vocabulary | `metadata["refusal_layer"]`; `"registry"` for `NOT_REGISTERED`/`INACTIVE_SCENE` |
| The inactive-scene refusal maps to no kind | `INACTIVE_SCENE` joins `GATE_REFUSAL_KINDS` |
| Deadline plus half-JSON | The deadline wins: `abandoned`, not parsed |
| What `engage()` writes | Exactly five `os.environ` keys; nothing in `maxim.utils` |
| `atomic_write_json_at` temp names | Transient names allowed in `report_dir` |
| `agents.permissions` in the closure | Allowed transitively; `agents.autonomy` forbidden |
| §1A.9 guard nodeids that did not exist | Pointed at the real ones |
| Race-gate lookup count | Behavioural only (§3: the run object is the checked object) |
| Census encoder site | `local_files_only` plus a degraded record counts (§6) |
| The env-read wildcard vs the prefix refusal | One sanctioned scanner, `maxim.advisor._refusals.code_loading_vars_set` |
| Children run with `-B` | Stated: Profile B's root is read-only, so `.pyc` writes are out of the no-effect audit |

**Red-gate delta round** (v6.4; `redgates_delta_*.md`):

| Finding | Fold |
|---|---|
| Arch D1 (`close_all` dropped by v6.3), Wire D-S2 | `engage()`'s three steps; `on_engage` callbacks; a pooled-client red gate |
| Arch D3, Exec S2/S3 | `_reset_for_tests` in `src/`, its callers linted to `tests/`; it restores the five env keys; the fixture fails if it is missing |
| Arch D4, the Exec pin NIT | One kind per layer; `"offline"` added; `pinned()` refuses at `"pin"` |
| Exec S1 (`dir_fd` writes, rename destination) | Test fold |
| Arch D2 (prefix-based reader list) | Test fold: exact reader list |
| Wire D-S1 (O6 ends at the dormant `ExecAgent._build_llm_context`) | Test fold: re-targeted at `PromptBuilder`, the live seam |

**Narrow delta round** (v6.5; `redgates_delta2_*.md`): the Executor lens returned STOP.

| Finding | Fold |
|---|---|
| Architecture S1 (the reset guard missed aliased imports) | Test fold: match `ast.alias`; proved by a probe |
| Architecture S2 (invariant 3 overclaimed "no socket") | Invariant 3 restated, with residuals and guards |
| Architecture NITs | `engage()` order stated; the reset text; callbacks persist across a reset |
| Wire S1 (O6 drove only the `/api/command` producer) | Test fold: Twilio's `conversation_context` producer too |
| Wire NITs; Executor N2 (properties escaping the classification) | Test folds |

**Final narrow check** ([redgates_delta3.md](reviews/atonia_gate/redgates_delta3.md)): **STOP: no further round
owed.** Both fail-open folds are closed. Its NITs are folded here (invariant 3's out-of-model forms) or recorded for the
build PRs: the `receive_voice` arm and the `triggering_input` route for O6 (#1481), the double-rendering XPASS
class, and the reset guard's fail-closed name match.

**AvServ's questions, plan 50 §18.6** (v6.6, owner decisions 2026-10-26):

| Question | Answer and fold |
|---|---|
| `log_mode` missing from §4.2's signature | §1A.8 wins; §4.2 points to it |
| Context size and prompt cap | A: `n_ctx`, `max_prompt_tokens`, `n_gpu_layers` and `session.limits`. B: token counts on a `prompt_too_long` report |
| Is the cursor opaque, and what sets `cursor_out`? | C: a per-event `cursor`; `cursor_out` is the last judged event's cursor |
| SIGTERM during `ingest()` | D (v6.7): `session.request_stop()` plus a per-call `stop=`, checked per token; the result is `abandoned`, then `AdvisorStopped`, then `close()` |
| `degraded` | No producer in this release; rejected inputs stay `ok` |
| GPU default; `nvidia-smi` under the latch | A: `n_gpu_layers=-1`; the supervisor runs the GPU checks before `open()` (item 11 later) |
| Phase 1 web research | Out of the gate's scope: the 1.3.2 harness in the sandbox; no `ddgs`. **Corrected in v6.7:** `http_fetch` at defaults (`block_private_ips=True`) uses `fetch_url(public_only=True)`, which **never uses a proxy** (#824), so the sandbox's egress proxy must be transparent (network-level redirect), not an explicit `HTTPS_PROXY` |

**v6.6 three-lens round** (v6.7; `v66_executor.md`, `v66_architecture.md`, `v66_wire.md`):

| Finding | Fold |
|---|---|
| Wire B1: `http_fetch` does not honour a proxy at defaults (#824) | The answer corrected: the egress proxy must be transparent |
| All three: the process stop is sticky, `MaximHandle.stop` sets it, nothing checks it per token | D is session-scoped, with `should_stop` per token and `AdvisorStopped` after a stop |
| Architecture and Wire: "judged" undefined; the `event_id` fallback | C: Maxim decides; the last event's cursor; cursors required when used |
| All three: the trained-context default; the budget undefined; the cap unchecked against `n_ctx` | A: the builder's clamp (at most 8192); budget 1024; `open()` refuses an over-large cap |
| All three: two producers for `prompt_too_long`; counts in `extra` | B: the session is the one producer; a named `prompt` field on every report |
| Architecture S6, Executor S5: `degraded` contradicts §4.2/§6 | Stated: §4.2/§6's `degraded` is for non-observer runs |
| Executor S3: a leaked process stop could mask gates | Moot: no process stop is used |

**v6.7 delta round** (v6.8; `v67_*.md`: Architecture and Wire STOP; the Executor's one fail-open folded):

| Finding | Fold |
|---|---|
| Executor S1 (fail-open): nothing proved the process stop is unread | `maxim.utils.cancellation` joins the observer's forbidden closure |
| Executor S2/S3, Wire S3: SEAMS still had the backend's cap and no `should_stop`; the child fake lacked `n_ctx` | SEAMS updated; the child fake carries `n_ctx`; the backend exposes `n_ctx` |
| Architecture 1: what "signal-handler-safe" allows | Set an event and store the deadline only |
| Architecture 2, Wire S1: the clamp does not exist; `n_ctx=0` means unbounded; profile drift | `closed_local` owns the clamp via GGUF metadata; the shared builder gets an explicit `n_ctx`; a gate |
| Wire S2: a stop reported as `deadline_hit` | A `stopped` flag; reason `"stopped"` vs `"deadline"` |
| Architecture 3/4: the JSON block; undefined cursor cases | `prompt` field and nullable cursors in the block; the session holds the cursor; second-ingest rule; a gate |
| NITs | `deadline_s` defined; stop during dispatch defined |

**v6.8 narrow check** (v6.9; `v68_check.md`):

| Finding | Fold |
|---|---|
| S1: the forbidden-module proof would become unflippable under bounded_stop's module-level imports | The `maxim.utils.cancellation` entry removed; a behavioural proof replaces it |
| S2 (fail-open): `request_shutdown()`'s `_SHUTDOWN_EVENT` is a second process signal | The behavioural gate sets both process signals and asserts `ok` with no early stop |
| S3 (fail-open): no case showed an error turn holds the cursor | The cursor gate adds an error turn between two `ok` turns |
| S4 (fail-open): the tiny fixture never exercises the clamp | A gate patches the trained context to 131072 and spies `Llama`'s `n_ctx` (8192, never 0) |

**v6.9 narrow check** (v6.10; `v69_check.md`):

| Finding | Fold |
|---|---|
| S1 (fail-open): the process-stop gate runs on the fake, so a polling `closed_local` would pass | A structural gate: `closed_local` references no process-wide stop signal |
| S2 (fail-closed): the clamp gate's nonexistent path never reaches `Llama` | A real temporary file; "never constructed" names what was raised |

**v6.10 narrow check** (`v610_check.md`): **STOP: no further round owed.** Its NIT N1 is recorded for the item 1 build:
a future poll added to the shared `llama_backend` builder would evade the `closed_local` scan, so that PR extends the
scan to the builder `closed_local` calls.

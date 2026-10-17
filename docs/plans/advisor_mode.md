# Advisor mode: Maxim as a read-only observer whose only tool is "propose" (#1315)

Status: PLAN, no code. Drafted 2026-10-17 from an external request (AvServ, a safety-of-life avalanche check-out
server). Every claim about the code was verified against the `v1.3.2` tag and `origin/main` the same day.
Revised after a three-lens review the same day; the review blocked two points, both folded below:
- the guarantee claimed more than 1.3.3 delivers;
- the fence was crossed on coordination alone.

Target: 1.3.3, for the items the owner keeps in the cut (AD7, AD9). Each gate item gets an approach note and an
adversarial design pass before code (CLAUDE.md "Faster without lowering the bar", item 1). Owner decisions are
AD1–AD9 below; they are asked together before any work starts.

## Why

An external system wants Maxim attached strictly as an advisor:
- later, Maxim watches a cut-down event stream and says "here is what is happening, here is what I would do";
- separately, an **offline lead job** researches search-and-rescue (SAR) teams for regions, producing cited leads
  that a human approves elsewhere. Per the 2026-10-14 decision, SAR data is a curated, versioned dataset, and web
  search happens only offline to produce leads. That job is **not** advisor mode: it runs under its own network-read
  policy (item 8).

The advisor must never act:
- no messages;
- no writes outside its report directory;
- no network beyond its LLM;
- no executing its own proposals.

The deployment also runs it inside an OS sandbox: its own user, a read-only root, a tmpfs data home, network only to a
local LLM, and no comms credentials.

## The guarantee, by effect channel

The request asked that "writing a report is the only effect Maxim can have", enforced by types and constructors.
Items 1–3 enforce that for **tool dispatch**, and only for tool dispatch. Other effect channels do not go through a
tool. Until the item that owns a channel ships, **the sandbox is that channel's only guard**:

| Channel | What can carry an effect today | Owned by | Until then |
|---|---|---|---|
| Tool dispatch | any registered tool, composite tools (the sub-sim tool spawns its own `bash`/`write_file` executor), direct `executor.execute` callers | items 1–3 | — |
| Disk | Hippocampus/NAc/sessions saves, `lane_decisions.jsonl` provenance, `MAXIM_LOG_FILE`, `sim_reports` | item 7 (ephemeral) | sandbox only |
| Egress | the LLM; model and encoder downloads (`MAXIM_AUTO_DOWNLOAD_MODELS`); the llama-server subprocess | an egress seam (not yet planned; AD9) | sandbox only |
| Inbound | `comms/api.py`, the tailed `<home>/cli/*.jsonl`, console `talk()` | item 4 (one framed entry point), plus the closed builder | sandbox only |
| Actuation | `embodied_runtime/movement.py`, reflexes, the Default Network, whenever an embodiment is attached | the closed builder refuses any embodiment or robot | — once the builder ships |
| Environment | `MAXIM_ROLE=leader` (proxy/mesh), `MAXIM_HEARTBEAT`, other `MAXIM_*` toggles | the closed builder: it builds from a closed config and ignores or refuses `MAXIM_*` | — once the builder ships |

**AD5, restated:** Maxim's guard holds on its own for tool dispatch. The sandbox is the only guard for disk and
egress until item 7 and an egress seam ship. AD9 decides what 1.3.3 may promise publicly before then.

## Why its own plan, and what it rides on (front-gate scope pressure)

It rides on existing infrastructure wherever that can carry the load:
- **`ToolPolicy` is `AgentPermissions` made mandatory**, plus an `effect` ceiling. It is not a third, parallel
  gate. `AgentPermissions` is already frozen, already has `kind:` selectors, and is already enforced inside
  `Executor.execute` (`_permission_denial`, before and after alias resolution). The executor is the one layer that
  every caller reaches, the foundry and prefetch included. What `AgentPermissions` cannot do today is be required:
  `build_executor(permissions=)`, `AgentConfig.permissions` and `Executor(permissions=)` all accept or default to
  `None`.
- **The mode pin:** `Executor._operational_override` (#829) already pins a mode at launch. It is a mutable attribute,
  not a type, so it cannot carry "cannot be switched out" (AD2).
- **Spotlighting** in `utils/content_safety.py` (`frame_tool_output`) is extended to inbound event text; nothing new.
- **The proposal and refusal log** (item 5) is a separate JSONL file. It is *not* new record kinds in `actions.jsonl`,
  because `actions.jsonl` rows are served by the console as `ActionRecord` (Session S's contract). The writer mirrors
  `simulation/report.py::save_action_log`, with a header record carrying `_format_version`.

What is genuinely new, and why:
- **`Tool.effect`:** nothing classifies effects today; effect knowledge is scattered across name sets.
- **A sealed advisor registry:** a registry today silently replaces a same-name tool, and it accepts tools at any
  time through several write paths.

## What is already fixed in 1.3.2 (verified at the tag)

- Leader proxy auth is a type (`leader_proxy.py::ProxyAuth`); a leader with no key refuses to start.
- `proxy.allow_remote_update` defaults to false, and fails closed.
- llama-server binds to 127.0.0.1 in both of its callers.
- `load.agent` defaults to `on_corrupt="raise"`.
- **Not fixed, contrary to the request:** `--clear-memory` does not reach `sessions/`, and it resolves against
  `./data` (#1337).

## Verification of the open items at `v1.3.2`

| # | Verdict | Where (`src/maxim/`) | Finding |
|---|---|---|---|
| O1 | **CONFIRMED, wider than claimed** | `api.py::run`; `runtime/agent_factory.py::AgentConfig`; `runtime/executor.py::Executor` | `run()` passes `permissions=None` on both branches, hard-codes `AutonomyLevel.AUTONOMOUS`, and takes no permissions argument. The console `MaximHandle` is the **only** caller that passes non-`None` permissions. Every other entry path builds a gate-less executor: the CLI agentic path, the embodied runtime (two builds), the sim AUT, the sim narrator (a `SimToolRegistry` holding `send_message`), the sub-sim tool, the Minecraft harness, the foundry, `create.agent` (`AgentConfig(permissions=None)`), the public `maxim.runtime.run_agent_loop` (executes with no autonomy check) and the public `ConcurrentExecutor`. Prefetch and the foundry call `executor.execute` directly, and so do the scripts `exp56/common.py` and five `survival_world/` scripts. |
| O2 | **CONFIRMED** | `modes/definitions.py::ModeDefinition.dispatch_refusal` | The refusal is a name list (`HOST_ACTING_TOOLS`). `send_message`, `call_user` and `write_file` are allowed in passive, active and singularity alike. |
| O3 | **PARTIAL: the cited branch cannot be reached** | `runtime/agent_loop.py` (plan-approval branch); `simulation/sim_logger.py::should_prompt`; `agents/autonomy.py::ProposalQueue` | `"plan_approval"` is critical, so `should_prompt` is True in every mode. The one call to `proposal_queue.approve` sits in that branch, which no run reaches. Four real leaks remain in non-advisor runs: `ALWAYS_ALLOWED_TOOLS` run at PLANNING (`respond`, `move`, `read_file`, …); SUPERVISED confirmations are auto-answered "yes" when nothing is interactive; the sim response policy approves plans; the `plan_text` path reads approval from free text (#1185). |
| O4 | **CONFIRMED** | `tools/comms.py`, `comms/gateway.py` | Any recipient, and the model picks the channel. The CLI adds both comms tools to the SUPERVISED allow-list with no confirmation. They are registered only when the Twilio environment variables are set. |
| O5 | **CONFIRMED** | `tools/base.py::Tool`, `::ToolOutput` | No effect classification. `kind` records lifecycle and provenance. Effect lists are scattered name sets (`HOST_ACTING_TOOLS`, `ALWAYS_ALLOWED_TOOLS`, `TOOL_CONFLICTS`, `TOOL_REVERSALS`, `TALK_CONVERSATIONAL_TOOLS`). `SupervisionPolicy.forbidden_categories` reads a `category` the **model** supplies. |
| O6 | **PARTIAL** | `comms/`, `agents/exec_agent.py`, `utils/content_safety.py::frame_tool_output` | `/api/command` needs Twilio, but other inputs exist: stdin, voice, a tailed `cli/*.jsonl` that any local process can write, console `talk()`, sim bridges. Inbound comms reach the prompt as `"[User via …]"`, unframed. The gateway's `external` flag is read by nothing. Inbound salience is fixed at 0.9. Only network-tool output is spotlighted as untrusted. |
| O7 | **PARTIAL** | `agents/autonomy.py` (`audit_log` deque); `simulation/instrumented_executor.py` | Proposals and autonomy rejections live in memory only. `actions.jsonl` records every call that reaches `executor.execute`, refusals included, but never proposals or confirmations, and `record_block` has no callers. `run()` passes no `action_sink`, so it writes no `actions.jsonl` at all. |
| O8 | **CONFIRMED** | `cli.py` (server mode), `runtime/agent_pool.py::_generate_npc_response`, `create.py::pool` | Server mode drops `wait_ready`'s result. The NPC fallback is logged at DEBUG. `create.pool()` never wires an LLM. All three are #1338. |
| O9 | **PARTIAL** | `memory/strategies.py::AccessBasedStrategy`, `utils/paths.py` | There is no ephemeral mode and no per-person erasure. Retention is access-based: a record goes after 7 days without access. But *successful* episodes with user input are floored at 0.7, and records accessed 10 or more times at 0.8, so both are kept with no time limit. `MAXIM_DATA_HOME` and `run(home_dir=…, learning=False)` exist. |

## Scope

Owner priority is 1–3 first, then 6, 4, 5, 7, 8. Sizes are rough and include tests.

| # | Item | Gate? | Depends on | Size | Closes |
|---|---|---|---|---|---|
| 1 | **The advisor composition:** a closed builder plus a sealed registry (see below) | yes | 2, 3 | M | O3 for advisor runs |
| 2 | **`Tool.effect` and the registry effect ceiling** | yes | none | L (classifying ~85 tool classes and the generated tools is most of it) | O2, O5 |
| 3 | **One required `ToolPolicy` on every executor** | yes | 2 | L | O1 |
| 6 | **Fail loudly on readiness.** Advisor and server mode exit non-zero, or report not-ready, when the LLM never becomes ready. The `run_turn` fallback becomes a typed error. | no | none | S | O8 (#1338) |
| 4 | **Structured input and output:** one ingestion entry point, with event text framed as untrusted, and the report below | no | 1 | M | O6 |
| 5 | **A durable, append-only proposal and refusal log** | yes | 1 | S–M | O7 |
| 7 | **Ephemeral mode:** the run cannot write bio-stack memory, provenance, sessions or logs outside a directory the caller chooses | no | none (not on 3) | M | O9; the Disk channel |
| 8 | **Corroboration for the offline lead job** (not advisor mode) | no | 4 (the report's evidence shape) | M | (SAR leads) |

**Item 1, the advisor composition.**
- **A closed builder** (`maxim.advisor`'s only constructor) assembles:
  - a sealed registry, holding only the READ `propose_action`;
  - an executor whose `ToolPolicy` allows exactly that tool;
  - an autonomy controller with no proposal queue that any approval can drain. `approve` is a `ProposalQueue`
    method, the queue is held as `AutonomyController.proposal_queue`, and proposals are drained in
    `runtime/loop_planning.py::drain_approved`, so the advisor composition replaces those. Replacing only the
    executor would not be enough.
- **The builder:**
  - refuses an embodiment or robot;
  - ignores or refuses `MAXIM_*` environment toggles, building from a closed config;
  - does **not** inject `maxim.register_tool` user tools.
- **Sealed** means:
  - `register`, `register_scene_tools`, `generate_entity_tools` and user-tool injection all raise after
    construction;
  - dispatch checks the tool's class identity, because a same-name `register` silently replaces a tool today;
  - the `AgentPermissions` allow-list stays as an independent second check.
- **The structural property:** the advisor's dispatch set is closed to `{propose_action}`, and nothing reads the
  proposal log back into a dispatch. Approval, for non-advisor runs, stays with #922 and #1185.

**Item 2, `Tool.effect` and the ceiling.**
- **The effect classes** are #1315's: READ, SELF_SCOPE, WRITE_LOCAL, NETWORK, COMMS, ACTUATE. The approach note
  fixes their order; proposed in that order, lowest first.
- **NETWORK covers reads that egress.** A GET can exfiltrate injected data in its URL, so `http_fetch` and
  `internet_search` are NETWORK, not READ.
- **Undeclared means the top class.** There is no separate `ACTING` member.
- **How `effect` is read:** from `type(tool)`, at both registration and dispatch, so an instance cannot mutate it
  and a property cannot change its value. `register()` gains no `effect=` override, unlike `kind=`.
- **One admit method for every write into the registry.** A private method checks the ceiling, and every write goes
  through it:
  - `ToolRegistry.register`;
  - `ToolRegistry._register_scene_locked` (`register_scene_tools`, which today bypasses `register()`);
  - `ToolRegistry.__init__(tools=)`;
  - `SimToolRegistry.register`.
- **Composite tools** declare the highest effect of anything they can spawn. A child policy is derived from the
  parent's and can never be wider. The sub-sim tool's inner executor is the first case.
- **The `maxim.tool` decorator** gains `effect=`.
- **The swallowing seams log their refusals.** `api._inject_registered_tools` swallows a refusal into a WARNING and
  `dm_runtime._register_entity_tools` into a DEBUG line. A ceiling refusal at either seam is written to item 5's
  log; it is never left as a debug line.
- **Split out of advisor mode:** a comms recipient allowlist for the modes that allow comms. That is O4's fix for
  non-advisor runs, and it goes to Session H.

**Item 3, the required `ToolPolicy`.**
- **Required, with no default,** on three carriers:
  - `build_executor` (already a required keyword, which today accepts `None`);
  - `AgentConfig`;
  - `Executor.__init__`. Alternatively `Executor` is dropped from `maxim.runtime.__all__`, where it is exported
    today. `extension_api.md` lists `build_executor` and the other `build_*` constructors as internal, but does not
    mention `Executor`, so dropping it needs the owner's call.
- **The only permissive constructor is named:** `ToolPolicy.unrestricted(reason=...)`. A census test lists who uses
  it (the sub-sim, the foundry and the Minecraft harness are expected).
- **Call sites to change, on origin/main 2026-10-17:**
  - `api.py` (`run`, both branches)
  - `cli.py` (the agentic path)
  - `console/handle.py` (where `AgentPermissions` becomes a `ToolPolicy`)
  - `orchestrator.py` (AUT and narrator)
  - `embodied_runtime/agentic_runtime.py` (two builds)
  - `simulation/foundry.py`
  - `simulation/minecraft_harness.py`
  - `simulation/tools.py` (sub-sim)
  - `create.py`
  - the public `run_agent_loop` and `ConcurrentExecutor`
  - tests: 71 `build_executor(` calls in 26 files, and 40 `Executor(` constructions in 15 files.
- **`run()` gains `autonomy=` and `policy=`;** see AD6 on patch versus minor.
- **Registries built outside `bootstrap.build_tool_registry`.** Each one migrates or is allow-listed by the census:
  - `agent_factory._create_tool_registry`
  - `foundry` (two)
  - `minecraft_harness`
  - `embodiment/tool_bridge.derive_capability_map` (throwaway)
  - `orchestrator`'s narrator `SimToolRegistry`
- **Scripts are out of the census.** They are experiment harnesses, and each says so in its own docstring.

**Rule T and the fence.**
- Item 3 and #1338 edit functions pinned by the function-length baseline:
  - `orchestrator.py::start_simulation_mode`
  - `agent_loop.py::run_agentic_loop`
  - `api.py::run`
  - `cli.py::_main_impl`
  - `embodied_runtime/agentic_runtime.py::AgenticRuntimeMixin._start_agentic_runtime`

  Every line touched there must be paid for by shrinking that function, or recorded as a `touch_exemptions` entry
  (#1327).
- **The fence, separately.** `orchestrator.py` and `agent_loop.py` are fenced ("Nothing under
  `runtime/agent_loop.py` … or `simulation/orchestrator.py`"). Every past touch had an owner-approved exception
  (#832, #829, #1042). AD8 asks for one.

**The order:**
1. AD1–AD9 are decided together.
2. The approach notes and design passes for 2, 3, 1 and 5 are written together, because they share inputs.
3. The red gates.
4. Items 2, 3, and 1 with 5.
5. Item 6 can run at any point; it shares nothing.
6. Items 4, 7 and 8.

**If 1.3.3 must stay small:** items 2, 3 and 6, then 1 with a minimal 5 (the log written; no query API). Items 4,
7 and 8 move to 1.3.4, and AD9 then decides what 1.3.3 may promise.

## Red gates first

- **One `xfail(strict=True)` per open item, O1–O9,** each proving the hole on `v1.3.2` behaviour.
- **A registration sweep in two halves:**
  - (a) every built-in, generated and extras tool, registered into a registry with the advisor ceiling, raises unless
    its effect is READ;
  - (b) a built advisor registry holds exactly the sealed `propose_action` class, and every later write raises.
- **A dispatch sweep:** every tool in a fully built registry is run through the *advisor* executor and refused,
  unless it is the sealed `propose_action` class.
- **A behavioural no-effect test,** the one that checks the claim and not the declarations:
  - an advisor run, with a read-only home;
  - spies on `socket.connect`, `subprocess` and on writes outside `report_dir`;
  - every spy must stay empty except the LLM's connection.

  Until item 7 and the egress seam ship, the disk and egress assertions are their own strict-xfail gates.
- **An entry-path census:**
  - every `Executor` construction and every `build_executor` call in `src/` passes a `ToolPolicy`;
  - every `ToolPolicy.unrestricted` caller is listed;
  - every registry construction goes through the canonical builder or is allow-listed.

  A new entry path fails this census.

Every fix ships with a caller on the shipped path. Each item gets the three-lens review, and Wire integrity lists
every entry path that constructs an executor or a registry.

## Owner decisions (ask together at the start)

| # | Question | Recommendation |
|---|---|---|
| AD1 | Pull forward #834 (typed grants) and #922 (approval surface), or build a narrower advisor-only subset that 1.4 absorbs? | **The narrower subset.** #834 and #922 are about authority to *escalate*. Advisor needs a composition that no approval surface can attach to, and #922 and #1185 add ways for approved proposals to *run*. What 1.4 reuses from this is `Tool.effect`, `ToolPolicy` and the log, which are prerequisites for #834 and #922, not parts of them. |
| AD2 | Is advisor a new mode class, or a property orthogonal to mode? | **Orthogonal, and fixed at construction.** Modes switch at runtime (`mode_switch`, `reexec_with_mode`). The launch pin, `Executor._operational_override` (#829), is a mutable attribute. An advisor must not be able to switch out, so advisor is the closed builder's composition, not a `ModeDefinition`. |
| AD3 | Does the capability class live on the tool base type or in the registry? | **Declared on the type, enforced by the registry, as #1315 decided.** An undeclared effect is the top class everywhere. In advisor mode the sealed registry makes the declaration irrelevant: only `propose_action`'s class can dispatch, and user tools are not injected. That is a stronger answer than "an undeclared tool fails registration everywhere", which would also break `maxim.register_tool` for every existing third-party tool. |
| AD4 | Is the report schema versioned, and where is it documented? | **Yes.** `_format_version` plus `schema: "advisor_report/1"`. Documented in `docs/user/advisor.md`, and listed under `stable_api.md`'s **Experimental** tier (rule 5) through 1.3.x, then frozen at 1.4.0. |
| AD5 | The 2026-10-14 decision named the OS sandbox the *primary* enforcement. The request calls it an independent second layer. | **Record the restatement above.** Maxim's guard holds alone for tool dispatch; for disk and egress the sandbox is the only guard until item 7 and an egress seam ship. |
| AD6 | `run()` gains `autonomy=` and `policy=` keyword-only arguments with defaults. `stable_api.md` rule 1 allows that "in any **minor** release", and 1.3.3 is a patch. | **The owner's call.** Either (a) the arguments wait for 1.4.0, and 1.3.3 ships advisor mode only through `maxim.advisor` (Experimental); or (b) an explicit exception for 1.3.3. Recommend (a). Option (a) still adds a new public namespace in a patch. Rule 5 lets Experimental symbols *change*
in a minor release, but it says nothing about adding them in a patch, so that is part of the same call. If `effect` is meant to be covered by rule 4 (which speaks of *methods* with a default body), amend rule 4 in the same PR. `docs/user/python-api.md` is updated alongside. |
| AD7 | The scope cut for 1.3.3. | Items 2, 3 and 6, then 1 with a minimal 5. Items 4, 7 and 8 go to 1.3.4. |
| AD8 | Item 3 has to touch `orchestrator.py` (AUT and narrator executors) and `agent_loop.py` (`run_agent_loop`, and the PLANNING branch for item 1), which are fenced. | **Grant a scoped fence exception, as #832's was, limited to `ToolPolicy` call sites and the advisor composition.** If it is refused, item 3 lands without those entry paths, and the census test strict-xfails them until 1.3.4. |
| AD9 | What may 1.3.3 promise publicly while disk and egress are guarded only by the sandbox? | **Ship `maxim.advisor` as Experimental, with the effect-channel table in its docs, stating that disk and egress are sandbox-guarded until item 7 and the egress seam.** The alternative is to pull item 7 into 1.3.3, which makes the cut larger. |

## Draft interfaces, for the AvServ side to design against (Experimental)

**Input.** No comms gateway and no Twilio are needed. `JSONValue` means the JSON-serialisable types
(`str | int | float | bool | None | list | dict`), checked at construction.

```python
@dataclass(frozen=True)
class AdvisorEvent:               # CC3 path (a): every field defaulted + extra
    event_id: str = ""            # caller's id, echoed in the report; "" is refused at construction
    source: str = ""              # e.g. "avserv"
    kind: str = ""                # e.g. "checkin.overdue"
    occurred_at: str = ""         # UTC ISO-8601
    payload: dict = field(default_factory=dict, hash=False, compare=False)  # structured; never a tool argument
    text: str | None = None       # free text; framed as UNTRUSTED in the prompt
    extra: dict = field(default_factory=dict, hash=False, compare=False)

def advise(event: AdvisorEvent, *, home_dir: Path, model: str | None = None,
           timeout_s: float = 120.0) -> AdvisorReport: ...
# The session form for a stream: maxim.advisor.open(home_dir=..., model=...) -> AdvisorSession,
#   with .ingest(event) -> AdvisorReport and .close(). It is built only by the closed builder (item 1);
#   no argument turns acting tools back on.
```

**Output.** One report per event, also appended to the proposal log (item 5):

```json
{
  "_format_version": "1.0",
  "schema": "advisor_report/1",
  "status": "ok | degraded | error",
  "report_id": "uuid4",
  "created_at": "2026-10-17T18:00:00Z",
  "input": {"event_id": "...", "source": "avserv", "kind": "...", "payload_sha256": "sha256 of the RFC 8785 canonical payload"},
  "observation": "UNTRUSTED free text",
  "proposed_action": {"kind": "notify_operator", "summary": "UNTRUSTED free text", "params": {}, "executed": false},
  "rationale": "UNTRUSTED free text",
  "confidence": {"value": 0.6, "calibrated": false, "source": "model_self_report"},
  "evidence": [],
  "corroboration": {"status": "not_applicable"},
  "refusals": [{"tool": "...", "effect": "COMMS", "reason": "above the advisor ceiling"}],
  "provenance": "the M1 report.json provenance stamp shape (code identity, model, n_ctx)",
  "extra": {}
}
```

Rules a consumer must know, written into `docs/user/advisor.md`:
- **A missing report is not "nothing is happening".** It means the advisor failed or timed out.
- **`status` is `degraded` or `error`** for any fallback, stub or timeout (#1338). Only `ok` carries a model
  judgement.
- **Every free-text field is untrusted.** It is derived from event text that can carry prompt injection.
- **`executed` is the schema constant `false`.** It is evidence of nothing; it exists so a consumer can assert the
  shape.
- **`confidence` is the model's own number, nested with `calibrated: false`,** so the value cannot be read without
  the flag. Nothing in this plan calibrates it.
- **`evidence` and `corroboration` are filled only by the offline lead job** (item 8), never by advisor mode, which
  fetches nothing:
  - sources are independent when they have distinct registrable domains *and* differing content hashes;
  - `official` comes from an allowlist of official domains that the caller supplies, never from the model.

## Out of scope

- Any outbound comms in advisor mode.
- AvServ-specific code in Maxim.
- Hivemind or Oasis participation.
- Per-person erasure (item 7's ephemeral mode is the answer for this use).
- Approval of proposals in non-advisor runs (#922, #1185; 1.4).

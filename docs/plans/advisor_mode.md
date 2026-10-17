# Advisor mode: Maxim as a read-only observer that can only write a report (#1315)

Status: PLAN, no code. Drafted 2026-10-17 from an external request (AvServ, a safety-of-life avalanche check-out server). Every claim about the code was verified against the `v1.3.2` tag the same day. Target: 1.3.3 for items 1–3 and 6, plus a minimal 5; the rest as decided below. Each gate item gets an approach note and an adversarial design pass before code (CLAUDE.md "Faster without lowering the bar" item 1).

## Why

An external system wants Maxim attached strictly as an advisor:
- First, offline research of search-and-rescue (SAR) leads for regions. A human approves each lead elsewhere.
- Later, it watches a cut-down event stream and says "here is what is happening, and here is what I would do".

It must never act:
- no messages;
- no writes outside its report directory;
- no network beyond its LLM;
- no executing its own proposals.

The deployment also runs Maxim inside an OS sandbox (its own user, a read-only root, a tmpfs data home, network only to a local LLM, no comms credentials).

**The sandbox is an independent second layer. Maxim's own "cannot act" must hold without it.** That is the claim this plan makes, and it can be tested inside the repo. See owner decision D5 on reconciling this with #1315's 2026-10-14 wording.

**Goal:** in advisor mode, writing a report is the only effect Maxim can have. That is enforced by types and constructors, not by configuration.

## Why its own plan, and what it rides on

It rides on #1315's owner decisions of 2026-10-14:
- `Tool.effect`, with undeclared meaning acting;
- `ToolPolicy` replacing `permissions=None`;
- a registry effect ceiling;
- a durable audit sink;
- an advisor mode whose only tool is a READ `propose_action`.

It also reuses:
- the existing spotlighting in `utils/content_safety.py`;
- `InstrumentedExecutor`'s `actions.jsonl`;
- the persistence invariants (`atomic_write_json`, `_format_version`).

It needs its own plan because #1315 is a list of decisions, not a build. And the verification below found holes wider than #1315 named: entry paths that skip the gate, and sims and SUPERVISED runs auto-approving.

## What is already fixed in 1.3.2 (verified at the tag)

- Leader proxy auth is a type: `leader_proxy.py::ProxyAuth`. `start_leader_proxy_for_role` raises `LeaderKeyError` before it binds.
- `proxy.allow_remote_update` defaults to false (`config_loader.py`), and `_remote_update_allowed` fails closed.
- llama-server binds to 127.0.0.1 (`local_server_spawner.py`; both callers in `lane_backends.py`).
- `load.agent` defaults to `on_corrupt="raise"`.
- **Not fixed, contrary to the request:** `--clear-memory` does not reach `sessions/`, and it resolves against `./data`. See #1337.

## Verification of the open items at `v1.3.2`

| # | Verdict | Where (`src/maxim/`) | Finding |
|---|---|---|---|
| O1 | **CONFIRMED, wider than claimed** | `api.py::run` (:718-760) | `permissions=None` and `AutonomyLevel.AUTONOMOUS` are hard-coded on both the `learning=True` branch and the `learning=False` branch. `run()` takes no permissions argument. The console `MaximHandle` is the **only** entry path that wires `AgentPermissions` (`console/handle.py`). None of these pass permissions: the CLI agentic path, the embodied runtime, the sim AUT, the sub-sim tool, the Minecraft harness and the foundry. The foundry calls `executor.execute` with no autonomy check, and so does prefetch. |
| O2 | **CONFIRMED** | `modes/definitions.py::ModeDefinition.dispatch_refusal` | The refusal is a name list (`HOST_ACTING_TOOLS`). `send_message`, `call_user` and `write_file` are allowed in passive, active and singularity alike. |
| O3 | **PARTIAL: the cited line cannot be reached** | `runtime/agent_loop.py` (~:2187-2216); `simulation/sim_logger.py::should_prompt` | `"plan_approval"` is critical, so `should_prompt` is True in every mode, and nothing calls `proposal_queue.approve`. Four real leaks remain: `ALWAYS_ALLOWED_TOOLS` run at PLANNING (`respond`, `move`, `read_file`, …); SUPERVISED confirmations are auto-answered "yes" in non-interactive runs; the sim response policy approves plans; the `plan_text` path reads approval from free text (#1185). |
| O4 | **CONFIRMED** | `tools/comms.py` (`SendMessageTool`, `CallUserTool`), `comms/gateway.py` | Any recipient, and the model picks the channel. The CLI adds both tools to the SUPERVISED allow-list without confirmation. They are registered only when the Twilio environment variables are set. |
| O5 | **CONFIRMED** | `tools/base.py::Tool`, `::ToolOutput` | No effect classification. `kind` is lifecycle and provenance. Effect knowledge is scattered across name sets: `HOST_ACTING_TOOLS`, `ALWAYS_ALLOWED_TOOLS`, `TOOL_CONFLICTS`, `TOOL_REVERSALS`, `TALK_CONVERSATIONAL_TOOLS`. `SupervisionPolicy.forbidden_categories` reads a `category` that the **model** supplies. |
| O6 | **PARTIAL** | `comms/gateway.py`, `comms/api.py`, `agents/exec_agent.py` (~:1118), `utils/content_safety.py::frame_tool_output` | `/api/command` needs Twilio, but other inputs exist: stdin, voice, a tailed `<home>/cli/*.jsonl` that any local process can write, console `talk()`, and sim bridges. Inbound comms reach the prompt as `"[User via …]"`, unframed. The `external` flag the gateway sets is read by nothing, and inbound salience is fixed at 0.9. Only network-tool output is spotlighted as untrusted; file contents are framed as `source=tool`. |
| O7 | **PARTIAL** | `agents/autonomy.py` (`audit_log` deque), `simulation/instrumented_executor.py` | Proposals and autonomy rejections live only in memory. `actions.jsonl` records every call that reaches `executor.execute`, refusals included, but never PLANNING proposals or confirmations (`record_block` has no callers). `run()` passes no `action_sink`, so it writes no `actions.jsonl` at all. |
| O8 | **CONFIRMED** | `cli.py` (~:2386), `runtime/agent_pool.py::_generate_npc_response` | Server mode drops `wait_ready`'s result and prints "ready". The NPC fallback logs at DEBUG. Also: `create.pool()` never wires an LLM, so `run_turn` always returns the stub (#1338). |
| O9 | **PARTIAL** | `memory/hippocampus.py`, `memory/strategies.py::AccessBasedStrategy`, `utils/paths.py` | There is no ephemeral mode and no way to erase one person's data. Retention is access-based, not salience-based: 7 days without access. But user-input episodes are floored at 0.7, and records accessed 10 or more times at 0.8, so both are kept with no time limit. `MAXIM_DATA_HOME` and `run(home_dir=…, learning=False)` exist. |

## Scope

Owner priority: 1–3 first, then 6, 4, 5, 7, 8. Sizes are rough and include tests.

| # | Item | Gate? | Depends on | Size | Closes |
|---|---|---|---|---|---|
| 1 | **Advisor executor type.** It records proposals and has no execute-a-proposal path: the type has no `approve`, so no approval surface (#922, #1185) can attach to it. The only tool is a READ `propose_action`. | yes | 2, 3 | M | O3 |
| 2 | **`Tool.effect` and a registry effect ceiling.** Every built-in tool declares an effect. Undeclared means ACTING. Registering above the ceiling raises `TypeError`, which covers late, generated, runtime and user tools. Comms gets a recipient allowlist for the modes that allow it. | yes | none | L (classifying every built-in tool is most of it) | O2, O4, O5 |
| 3 | **One permission gate on every entry path.** `ToolPolicy` is required, keyword-only, at the canonical builder, so `None` is a `TypeError` (the `build_executor(pain_bus=…)` precedent). `run()` takes `autonomy=` and `policy=`. The foundry and prefetch go through the gate. | yes | 2 | L | O1 |
| 6 | **Fail loudly on readiness.** Advisor and server mode exit non-zero, or report not-ready, when the LLM never becomes ready. The `run_turn` fallback becomes a typed error. | no | none | S | O8 (#1338) |
| 4 | **Structured input and output:** an ingestion entry point that needs no comms gateway, with event text framed as untrusted, and a structured report (schema below). | no (the framing is part of the gate in 3) | 1 | M | O6 |
| 5 | **A durable, append-only log of proposals and refusals**, beside `actions.jsonl`, versioned. | yes | 1 | S–M | O7 |
| 7 | **Ephemeral mode.** The run structurally cannot write bio-stack memory, provenance, sessions or logs outside a directory the caller chooses. | no | 3 | M | O9 (for this use) |
| 8 | **Corroboration for the offline lead job.** At least 2 independent sources; official domains preferred; sources on every lead; uncorroborated leads labelled, never dropped and never promoted. | no | 4 | M | (SAR leads) |

**Order:**
1. Approach notes and design passes for 2, 3, 1 and 5, done together, since they share inputs.
2. Red gates.
3. Item 2.
4. Item 3.
5. Item 1, with a minimal 5.
6. Item 6 can run in parallel at any point; it shares nothing.
7. Items 4, 7 and 8.

**If 1.3.3 must stay small, cut to 2, 3, 6 and 1 with a minimal 5** (the log written, without a rotation or query API). Items 4, 7 and 8 move to 1.3.4. Item 8 needs no `src/` gate at all: it is a property of the offline lead job, and per the 2026-10-14 decision SAR data is a curated, versioned dataset, with web search used only offline to produce leads.

**Fence collision.** Item 3 touches the entry paths in `cli.py` and `orchestrator.py`. During 1.3.4 and 1.3.5 those files are Session A's, and every edit inside `start_simulation_mode` or `run_agentic_loop` is now subject to rule T (#1327): it must shrink the function by the lines touched. So item 3 lands as call-site edits only, coordinated with Session A, or it rides on an orchestrator slice.

## Red gates first

Before any fix lands:
- **One `xfail(strict=True)` test per open item, O1–O9**, each proving the hole on `v1.3.2`'s behaviour.
- **A sweep test:** in advisor mode, for every tool in a fully built registry with every extra installed, `Executor.permits(tool)` is False unless the tool declares READ.
- **An entry-path census test:** every constructor of an Executor or a tool registry in `src/` goes through the canonical builder. This is the Wire-integrity list above, kept as a test so a new entry path fails it.

Every fix ships with a caller on the shipped path, and the three-lens review runs before merge.

## Owner decisions (ask together at the start)

| # | Question | Recommendation |
|---|---|---|
| D1 | Pull forward #834 (typed grants) and #922 (approval surface), or build a narrower advisor-only subset that 1.4 absorbs? | **The narrower subset.** #834 and #922 are about authority to *escalate*. Advisor needs the opposite, a level that no approval surface can attach to, and #922 and #1185 add ways for approved proposals to *run*. What 1.4 reuses from this work is `Tool.effect`, `ToolPolicy` and the audit sink, which are prerequisites for #834 and #922 rather than parts of them. |
| D2 | Is advisor a new mode class, or a property orthogonal to mode? | **Orthogonal, and fixed at construction.** Modes can switch at runtime (`mode_switch`, `reexec_with_mode`). An advisor must not be able to switch out, so it should be the executor type and the registry ceiling, not a `ModeDefinition`. |
| D3 | Does the capability class live on the tool base type or in the registry? | **Both, as #1315 decided.** Declared on `Tool` (`effect`) and enforced by the registry ceiling at `register()`. Undeclared means ACTING. Registering above the ceiling raises `TypeError`. The request asked that undeclared fail registration everywhere. This option gives the same safety in advisor mode without breaking `maxim.register_tool` for every existing third-party tool (rule 4 of `docs/user/stable_api.md`: an attribute with a default is fine, an abstract one is a break). |
| D4 | Is the report schema versioned, and where is it documented? | **Yes.** `"_format_version": "1.0"` plus `schema: "advisor_report/1"`; a frozen dataclass on CC3 path (a), with `extra`; documented in `docs/user/advisor.md`. Listed in `stable_api.md` as **provisional through 1.3.x** and frozen at 1.4.0, so the AvServ side can build against it now. |
| D5 | The 2026-10-14 decision named the OS sandbox the *primary* enforcement. The request calls it a second, independent layer. | **Both are independent, and Maxim's guard must hold alone.** Record the change. "Primary" invites relying on the sandbox, and the claim this plan can test is Maxim's own. |
| D6 | Does any of this break a stable API in a patch release? | **No break is planned.** New keyword-only arguments with defaults on `run()`; `effect` as an attribute with a default; `build_executor` and `AgentPermissions` are not listed as stable. `run()`'s default autonomy stays AUTONOMOUS. If a break turns out to be needed, it waits for an explicit owner call. |
| D7 | The scope cut for 1.3.3. | 2, 3, 6, then 1 with a minimal 5. 4, 7 and 8 go to 1.3.4. |

## Draft interfaces, for the AvServ side to design against (provisional)

**Input.** The ingestion entry point needs no comms gateway and no Twilio:

```python
@dataclass(frozen=True)
class AdvisorEvent:                    # CC3 path (a): defaults + extra
    event_id: str                      # caller's id, echoed in the report
    source: str                        # e.g. "avserv"
    kind: str                          # e.g. "checkin.overdue"
    occurred_at: str                   # UTC ISO-8601
    payload: Mapping[str, JSONValue]   # structured fields; never executed, never a tool argument
    text: str | None = None            # free text; framed as UNTRUSTED in the prompt
    extra: dict = field(default_factory=dict)

def advise(event: AdvisorEvent, *, home_dir: Path, llm: str | None = None,
           timeout_s: float = 120.0) -> AdvisorReport: ...
# Plus a session form for a stream: maxim.advisor.open(home_dir=..., llm=...) -> AdvisorSession
#   with .ingest(event) -> AdvisorReport and .close(). It is built only through the advisor
#   builder (item 1 + 3); there is no argument that turns acting tools back on.
```

**Output.** One report per event, also appended to the proposal log (item 5):

```json
{
  "_format_version": "1.0",
  "schema": "advisor_report/1",
  "report_id": "uuid4",
  "created_at": "2026-10-17T18:00:00Z",
  "input": {"event_id": "...", "source": "avserv", "kind": "...", "payload_sha256": "..."},
  "observation": "what is happening, in plain words",
  "proposed_action": {"kind": "notify_operator", "summary": "...", "params": {}, "executed": false},
  "rationale": "...",
  "confidence": 0.6,
  "confidence_basis": "model self-report; not calibrated",
  "evidence": [
    {"url": "https://...", "title": "...", "accessed_at": "...", "content_sha256": "...",
     "excerpt": "...", "official": true}
  ],
  "corroboration": {"status": "corroborated | uncorroborated | not_applicable", "independent_sources": 2},
  "refusals": [{"tool": "...", "effect": "COMMS", "reason": "above the advisor ceiling"}],
  "provenance": {"maxim_version": "1.3.3", "git_sha": null, "model_id": "...", "backend": "...",
                 "n_ctx": 8192, "config_sha256": "..."},
  "extra": {}
}
```

- `proposed_action.executed` is always `false`; the field exists so that a consumer can assert it.
- `confidence` is the model's own number, and the schema says so. Nothing in this plan calibrates it.

## Out of scope

- Any outbound comms in advisor mode.
- AvServ-specific code in Maxim.
- Hivemind or Oasis participation.
- Per-person erasure (item 7's ephemeral mode is the answer for this use).

# The agent's internet toggle lowers its own session only (#1345): approach note

> **ARCHIVED (2026-10-24 plans archive):** #1345 shipped (PR #1401, 2026-10-20). The rule lives in docs/agents/runtime-tools.md.

**Status:** approach note for the adversarial design pass (gate rule). Session H, 1.3.3. Owner decision 2026-10-19 (#1345's direction, strict option): the agent may lower internet access for its own session only, in memory, never persisted, and may not raise it back.

## The defect

`util/internet_access.json` is the operator's persisted on/off. Every request reads it (`utils/internet_access.py::load_internet_policy`, through the one getter `live_internet_policy_getter` that `runtime/bootstrap.py::build_tool_registry` builds). Its only writer in `src/` is the agent: `tools/internet_search.py::InternetAccessTool.execute` calls `set_internet_access(enabled, source="tool")`, which persists. The tool is registered whenever the launch cap is on (the CLI agentic block unless `--no-internet`; the robot runtime when `allow_internet`), and at `--operational-mode active|singularity --autonomy autonomous` it runs without confirmation. So an agent can turn an operator's "off" back on, or persist "off" for every later session. The launch cap itself cannot be exceeded (no getter, no tools).

## The fix (revised after the design pass, 2026-10-19)

The first draft held a `lowered` flag on the getter object. The design pass (S1) showed that "session" must survive the robot runtime rebuilding its registry on every wake and mode change (`agentic_runtime.py::_start_agentic_runtime`) and the CLI re-exec'ing itself on an agent's mode switch (`cli.py::_after_run_mode_request`, `os.execv`). A flag on the getter resets on both, which is a raise-back.

- **The session's lower lives in the process environment, `MAXIM_INTERNET_LOWERED`.** Set in-process by the tool, it is inherited by `os.execv` and by child processes (sub-sims), and dies with the process tree: that tree IS the session. It is never written to disk. Any truthy value lowers; nothing in `src/` ever unsets it, and an operator may pre-set it to start a session lowered.
- **The one loader composes it.** `load_internet_policy` (the only builder of `EffectiveInternetPolicy`, the brief's invariant) returns `enabled=False, source="session-lowered"` when it is set, with its own `disabled_reason` ("lowered for this session by the agent; the operator's setting is unchanged"). So every internet tool, whatever getter it holds, sees the same gate: a toggle cannot be wired to a different gate than the search tools (design pass S2 dissolves).
- **The tool writes nothing to disk** and needs no new wiring (no `bootstrap.py` change):
  - `enabled: false` sets the variable (idempotent) and says it lasts for this session only.
  - `enabled: true` is refused, naming the operator's controls, unless the session is on and not lowered, where it is a truthful no-op.
  - `enabled` must be a real bool (a string `"false"` is refused, not read as truthy).
- **The persisted file is operator-only.** `set_internet_access`'s default `source` becomes `"operator"`. The startup WARNING names a legacy `source: "tool"` "off" as written by an agent before this change (design pass N4).
- **Guards.**
  - Whole-`src/` AST scan: `set_internet_access` / `save_internet_access` are called only inside `utils/internet_access.py` (design pass S3).
  - Behaviour: a lower leaves the file byte-identical, is refused by both internet tools on the next request, survives a fresh registry and a fresh getter, reaches a child process; the tool cannot raise after a lower, nor flip an operator "off" on.
  - The conftest autouse scrub for `MAXIM_INTERNET_LOWERED` (the hot-path env-var rule).
- **A deliberate exception to the per-agent stash rule** (docs/agents/runtime-tools.md: per-agent runtime state is a `dict[agent_id, value]`): this lower is per PROCESS by design, like the launch cap and the getter. Do not "fix" it into a per-agent dict, which would reset on a re-exec.
- **Residual, stated:** one process with several agents shares one lower (fail closed); today no multi-agent path registers internet tools (the sim, console, API and pool registries pass `internet_launch_enabled=False`). A fetch already past its check when the lower lands completes. `internet_access_toggle` stays in `HOST_ACTING_TOOLS` (passive mode cannot lower; the strict choice, design pass N2).
- **Follow-up, not here:** the prompt roster still offers `internet_search` / `http_fetch` after a lower, and a `policy_blocked` refusal is booked as an ordinary tool failure for pain credit (design pass S4, the D82 class); filed separately.

## Inputs the gate consults, and how each could lie

| Input | How it could lie | Answer |
|---|---|---|
| The persisted toggle file | The agent writes it (the defect) | No agent-dispatchable code writes it; AST guard |
| The session flag (`MAXIM_INTERNET_LOWERED`) | Lost on a registry rebuild or a re-exec (raise-back); shared across agents in one process | Process env: survives both; the multi-agent case fails closed and has no caller today; conftest scrub |
| The tool's `enabled` argument | A string `"false"` (truthy) | Parsed strictly: only a real bool, else refused |
| A tool on a different getter than the search tools | Says "lowered", lowers nothing | One gate in the one loader; no getter carries it |
| `source` labels in the file | An old file says `"tool"` | Read as before (the value is informational); loader unchanged |
| A second registry in one session | Rebuilt per wake / mode change | Design pass: no runtime builds two internet registries for one agent at once; rebuilds keep the env gate |
| `maxim.utils.set_internet_access` called by agent-reachable code (a `bash` tool running Python) | Out of scope: an agent with code execution can write any file | Stated residual; the mode capabilities govern code execution |

## Not in scope

The operator's way to toggle (a `maxim config` verb for internet on/off is #832 item 2); an expiring in-session grant (#922/#834); the launch cap.

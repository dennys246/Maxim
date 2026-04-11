# Cleanup Wave

**Status:** Active, parallel to spine (does not gate 1.0)
**Supersedes:** `display_simplification_plan.md`, `agent_permissions_plan.md`
**Target version:** 0.2.2 — ships as a single coherent wave

## Goal

Remove rot that's visible on first use. Fix `--interactive` (currently broken), delete dead CLI flags, make `--display bio` the default, land the small agent permissions layer. None of this gates 1.0, but all of it makes the project legible to new users and removes ~800 LOC of dead weight.

## Why one wave, not four PRs

These changes are all in the CLI / UX layer, they all touch argparse or bootstrap, and they're all small. Shipping them as one reviewable wave avoids four separate rounds of CI and churn on the same files. Agent Permissions is the odd one out (not strictly UX) but it's small enough (~380 LOC) and foundation-level enough that bundling it doesn't hurt.

## Items

### C1. Fix `--interactive` (broken today)

**Bug:** The LLM calls `request_interaction` when `--interactive` is set, gets an error because the tool isn't registered, and falls through to a stub that says "Response will come on next turn" with no mechanism to actually collect input. User sees "I tried to ask but you didn't respond."

**Root cause:** [tools/display.py:80-172](../../src/maxim/tools/display.py#L80-L172) defines `RequestInteractionTool` but [runtime/bootstrap.py:51-271](../../src/maxim/runtime/bootstrap.py#L51-L271)'s `build_tool_registry()` never registers it. The `PromptHandler` that would collect input exists at CLI init time but is never threaded into the tool.

**Fix:** Register `RequestInteractionTool` in `build_tool_registry()`, pass the `PromptHandler` through, verify end-to-end with a short interactive sim.

**Scope:** ~30 LOC + one integration test.

**Exit:** `maxim --sim "ask me a question" --interactive true` prompts the user and incorporates the response into the next turn.

### C2. Delete dead CLI flags

Flags that are parsed but never read post-parse. Confirmed dead via audit:

- `--segmentation-model` — [cli_parser.py:180](../../src/maxim/cli_parser.py#L180), never referenced after parse
- `--audio_len` — [cli_parser.py:192](../../src/maxim/cli_parser.py#L192), never referenced
- `--record-percepts` — [cli_parser.py:214](../../src/maxim/cli_parser.py#L214), never referenced
- `--explore`, `--exploration-autonomy`, `--exploration-allow-scripts`, `--exploration-allow-training` — [cli_parser.py:238-274](../../src/maxim/cli_parser.py#L238-L274), entire exploration mode parsed but never dispatched
- `--arc` — [cli_parser.py:336](../../src/maxim/cli_parser.py#L336), never passed to simulation
- `--aut-name` — [cli_parser.py:344](../../src/maxim/cli_parser.py#L344), never read
- `--replay-from` — [cli_parser.py:352](../../src/maxim/cli_parser.py#L352), replay infrastructure missing

**Scope:** ~100 LOC of deletions. Verify no tests reference these flags.

**Exit:** `maxim --help` fits on one screen. `grep -r '\-\-explore' src/ tests/` returns only deletions.

### C3. Display simplification

From the old `display_simplification_plan.md`:

- **Default `--display bio`** instead of `clean`. One-line change in [cli_parser.py:20](../../src/maxim/cli_parser.py#L20). Users who want quiet output pass `--display clean` explicitly.
- **Auto-interactive detection.** DM campaigns with choice points should prompt by default; generative sims should not. Dispatched from campaign type at [cli.py](../../src/maxim/cli.py) sim-start path.
- **Drop `--agentic-verbosity`.** It already defaults to `--verbosity` when unset ([cli.py:908](../../src/maxim/cli.py#L908)) — redundant. Fold into `--verbosity`.
- **Rename `--verbosity` → `--log-level`.** Ends the confusion with `--display`. Keep `--verbosity` as a deprecated alias for one release; remove before 1.0.

**Scope:** ~200 LOC across `cli_parser.py`, `cli.py`, `simulation/sim_logger.py`.

**Exit:** `maxim --sim "..."` shows bio-annotated output by default. DM campaigns with choices prompt automatically. `--help` shows `--log-level` not `--verbosity`.

### C4. Agent Permissions (from `agent_permissions_plan.md`)

Two-layer permission system folded in from the standalone plan:

- **Enforced layer:** `AgentPermissions` dataclass with `clearance`, `tool_deny`, `sem_access_rules`. Hard gates at tool execution and SEM access.
- **Perceived layer:** `PerceivedAuthority` dataclass feeding NAc causal learning and FearAgent review. Flows into the LLM system prompt via PromptAssembler (once B1 lands; use ad-hoc injection until then).

**Key insight from the original plan:** enforced and perceived are orthogonal. A character can have zero perceived authority but full enforced clearance (a feared spymaster). A character can have full perceived authority but no enforced clearance (a beloved figurehead). The bio-stack learns perceived authority from outcomes; enforced authority is campaign config.

**Files touched:** new `agents/permissions.py`, campaign YAML schema extension, `runtime/executor.py` (enforcement point), `decisions/nac.py` (perceived learning hook).

**Scope:** ~380 LOC (150 core + 120 tests + 50 wiring + 60 integration).

**Exit:** Campaign YAML can declare `permissions:` blocks. Tool calls respect enforced rules. NAc learns perceived authority from positive/negative outcomes. Integration test: low-clearance agent cannot invoke restricted tool; high-authority agent's commands are followed by NPCs.

**Note on POG coupling:** The original plan mapped Agent Permissions into Pecking Order Graph's AUTHORITY domain. POG is deferred — so ship Agent Permissions standalone for now. When POG is revived, it consumes this layer rather than replacing it.

## Scope (total)

~800 LOC net (mostly deletions + the ~380 LOC permissions addition). 1–2 days of focused work.

## Order of operations

1. **C1 first** — it's a bug fix and unblocks interactive testing.
2. **C2 second** — deletions are low-risk and shrink the surface for everything after.
3. **C3 third** — display changes are visible but contained.
4. **C4 last** — largest addition, benefits from a clean CLI surface underneath it.

## Non-goals

- **No refactor of [sim_logger.py](../../src/maxim/simulation/sim_logger.py) beyond the display-tier changes.** It's load-bearing for sims; don't touch what works.
- **No POG integration for permissions.** POG is deferred. Ship the permissions layer standalone.
- **No new tools.** This wave is cleanup, not feature work. Tool additions go through [tool_refinement_plan.md](tool_refinement_plan.md).

## Risks

- **C1 fix touches the prompt handler path.** Verify it works in both `--sim` and plain `maxim` (non-sim) mode.
- **C3 default change is user-visible.** Users who've scripted against `clean` default will see more output. Mention in changelog.
- **C4 enforcement point is [runtime/executor.py](../../src/maxim/runtime/executor.py).** This is the tool dispatch hot path. Permission checks must be O(1) — no per-call YAML loads.

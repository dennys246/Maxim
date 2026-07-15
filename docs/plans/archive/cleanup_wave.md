# Cleanup Wave

**Status:** SHIPPED (all items complete, does not gate 1.0)
**Supersedes:** `display_simplification_plan.md`, `agent_permissions_plan.md`
**Target version:** 0.2.2 — ships as a single coherent wave

## Goal

Remove rot that's visible on first use. Fix `--interactive` (currently broken), delete dead CLI flags, make `--display bio` the default, land the small agent permissions layer. None of this gates 1.0, but all of it makes the project legible to new users and removes ~800 LOC of dead weight.

## Why one wave, not four PRs

These changes are all in the CLI / UX layer, they all touch argparse or bootstrap, and they're all small. Shipping them as one reviewable wave avoids four separate rounds of CI and churn on the same files. Agent Permissions is the odd one out (not strictly UX) but it's small enough (~380 LOC) and foundation-level enough that bundling it doesn't hurt.

## Items

### C1. Fix `--interactive` (broken today) — ✅ DONE

**Bug:** The LLM called `request_interaction` when `--interactive` was set, got an error because the tool wasn't registered, and fell through to a stub that said "Response will come on next turn" with no mechanism to actually collect input.

**Root cause:** [tools/display.py](../../src/maxim/tools/display.py) defined `RequestInteractionTool` and `DisplayModeTool` but [runtime/bootstrap.py](../../src/maxim/runtime/bootstrap.py)'s `build_tool_registry()` never registered either, and the existing `PromptHandler` factory in [interactive/prompts.py](../../src/maxim/interactive/prompts.py) was never wired in.

**Fix shipped:**
- `build_tool_registry()` now accepts a `prompt_handler` parameter and unconditionally registers both `DisplayModeTool` and `RequestInteractionTool`. Registration is safe regardless of mode because `RequestInteractionTool.execute()` already gates on `sim_logger.should_prompt()`.
- [cli.py](../../src/maxim/cli.py) creates a handler via `interactive.prompts.create_handler("auto")` and passes it through.
- [simulation/orchestrator.py](../../src/maxim/simulation/orchestrator.py) does the same when building the AUT registry, so `--sim ... --interactive true` reaches the console as well.
- New test file [tests/unit/test_request_interaction_tool.py](../../tests/unit/test_request_interaction_tool.py) covers registration, ON-mode handler dispatch, OFF-mode disable, critical-context override, and the no-handler fallback.

**Scope shipped:** ~50 LOC across bootstrap.py, cli.py, orchestrator.py + 110 LOC of tests.

**Exit:** Met. `request_interaction` is registered everywhere `build_tool_registry()` is invoked; ON mode routes through `PromptHandler.prompt()`; OFF mode short-circuits cleanly; full test suite (3636 passed) green.

### C2. Delete dead CLI flags — ✅ DONE

Audit corrected two over-claims in the original plan: `--segmentation-model` is in fact normalized + propagated through subprocess argv by [cli_utils.py](../../src/maxim/cli_utils.py), and `--audio_len` is forwarded into [embodied_runtime/selfy.py](../../src/maxim/embodied_runtime/selfy.py) at agent construction. Both stay.

**Deleted (truly dead at the args namespace):**

- `--record-percepts` — recorded nothing.
- `--explore`, `--exploration-duration`, `--exploration-autonomy`, `--exploration-allow-scripts`, `--exploration-allow-training`, `--resume-session`, `--list-sessions` — entire exploration argument group, never dispatched anywhere.
- `--arc`, `--aut-name`, `--replay-from` — `args.arc` / `args.aut_name` / `args.replay_from` had zero readers in `src/`.

**Cleanup beyond the parser:**
- [cli.py](../../src/maxim/cli.py) `_has_action` no longer reads `args.explore`.
- [tests/unit/test_cli_action_gate.py](../../tests/unit/test_cli_action_gate.py) updated to match.
- [utils/last_run.py](../../src/maxim/utils/last_run.py) `_SKIP_INDICATORS` drops `--list-sessions`.

**Doc sweep:**
- [docs/user/cli-reference.md](../../user/cli-reference.md): entire "Exploration Mode" table + recipe + `--arc` / `--aut-name` / `--record-percepts` rows removed.
- [docs/user/simulation.md](../../docs/user/simulation.md), [docs/generative_campaigns_guide.md](../../generative_campaigns_guide.md): `--arc` / `--replay-from` examples and CLI rows removed.
- [htmls-guides/maxim-usage-guide.html](../../htmls-guides/maxim-usage-guide.html): exploration block + "Timed Autonomous Exploration" / "Resume a Previous Session" recipes removed.
- [htmls-guides/maxim-simulation.html](../../htmls-guides/maxim-simulation.html): `--arc` example removed; interactive note retitled to `request_interaction`.

**Scope shipped:** ~95 LOC of CLI deletions, ~60 LOC of doc deletions. Full fast suite (3636 passed) green; lint clean.

### C3. Display simplification — ✅ DONE

- **Default `--display bio`.** [cli_parser.py](../../src/maxim/cli_parser.py) `--display` default flipped from `clean` → `bio`. Users who want narrative-only pass `--display clean` explicitly.
- **Auto-interactive detection.** [cli.py](../../src/maxim/cli.py) sim-display block now checks `raw_argv` for an explicit `--interactive` flag; when absent, it probes the YAML for `campaign:`/`encounters:` keys (DM campaign signature) or honors `--dm` and sets interactive mode to ON. Generative sims continue to default OFF. Critical contexts (plan approval, safety escalation) still prompt regardless via `should_prompt`'s `_CRITICAL_CONTEXTS` set.
- **Dropped `--agentic-verbosity`.** Removed from [cli_parser.py](../../src/maxim/cli_parser.py) and the `cli.py` agentic block. The agentic event buffer now follows `--log-level` directly. `configure_agentic_verbosity` itself stays — it's still used by the Python API ([api.py](../../src/maxim/api.py)).
- **Renamed `--verbosity` → `--log-level`.** New canonical flag is `--log-level`; `--verbosity` is preserved as a deprecated alias on the same `dest=verbosity` so existing scripts and the `args.verbosity` reads downstream keep working until 1.0.
- **`--interactive` default is now `None` (auto).** [cli_utils.py](../../src/maxim/cli_utils.py) `_normalize_args` resolves `None` to `True` for live runs so the existing downstream readers see a real boolean.

**Scope shipped:** ~80 LOC across `cli_parser.py`, `cli.py`, `cli_utils.py`. No changes to `sim_logger.py` were necessary — the existing `set_interactive_mode` / `should_prompt` machinery already handled both cases.

**Doc sweep:** [docs/user/cli-reference.md](../../user/cli-reference.md), [docs/user/troubleshooting.md](../../docs/user/troubleshooting.md), [docs/user/modes-guide.md](../../user/modes-guide.md), [htmls-guides/maxim-usage-guide.html](../../htmls-guides/maxim-usage-guide.html) updated to reflect the new defaults, the rename, and the dropped flag.

**Exit:** Met. Full fast suite (3673 passed) green; lint clean.

### C4. Agent Permissions (from `agent_permissions_plan.md`) — ✅ DONE

Two-layer permission system shipped in [agents/permissions.py](../../src/maxim/agents/permissions.py).

**Enforced layer:** `AgentPermissions` (frozen dataclass) carries `clearance`, `tool_deny: frozenset[str]`, optional `tool_allow: frozenset[str]`, and a tuple of `SEMAccessRule`s. Decisions are O(1) — frozenset membership for tools, a short list-walk for SEM rules. `AgentPermissions.from_yaml` parses the campaign YAML block once at load time so the executor never reads YAML on the hot path.

**Enforcement point:** [runtime/executor.py](../../src/maxim/runtime/executor.py)'s `Executor.__init__` now takes an optional `permissions: AgentPermissions | None`. The check fires twice in `execute()`: once on the raw incoming tool name, and once again after alias resolution lands on the canonical name — so an LLM that calls `shell` cannot sneak past a deny rule on `bash`. Denies are returned to the agent as a `ToolOutput.success=False` with a human-readable reason so the LLM learns *why*.

**Perceived layer:** `PerceivedAuthority` dataclass + `PerceivedAuthorityTracker` shipped alongside the enforced layer. The tracker is a lightweight EWMA over outcome valence in `[-1, +1]`, mapped into a `[0, 1]` belief score. It's deliberately decoupled from NAc so that NAc, FearAgent, and the prompt assembler can each consume it without taking on a circular dependency. NAc-side wiring is left as a follow-up to be done when the next memory consolidation pass lands — the tracker's API (`observe`, `get`, `snapshot`) is shaped so the integration is one method call.

**Campaign YAML:** [simulation/dm_schema.py](../../src/maxim/simulation/dm_schema.py) `CampaignDef` now carries a `permissions: dict[str, Any]` field. Each key is a character name; each value is the YAML block consumed by `AgentPermissions.from_yaml`. The loader keeps the dict raw (not pre-parsed into `AgentPermissions`) so that the runtime code that actually owns each character is responsible for instantiating its own enforced policy from the right slice — keeping `dm_schema` free of the agents-layer import.

**Tests:** 19 unit tests in [tests/unit/test_agent_permissions.py](../../tests/unit/test_agent_permissions.py) covering: dataclass defaults, tool deny/allow gates, specific-entity and wildcard SEM rules, YAML round-trip, executor enforcement (with and without permissions), alias-resolution gating, EWMA convergence, valence clamping, and snapshot independence.

**Doc sweep:** [docs/user/dm-campaigns.md](../../user/dm-campaigns.md) gained a "Enforced Permissions" section with the YAML shape, the alias-resolution invariant, and the orthogonality note about enforced vs perceived authority.

**Deferred (intentionally):**
- NAc → `PerceivedAuthorityTracker` wiring. Tracker is shipped and tested standalone; the NAc hook is a one-line `tracker.observe(...)` call once we settle on which valence signal to use, and belongs with the next memory-consolidation pass rather than this UX cleanup wave.
- Prompt assembler injection. Pending B1 from the substrate plan.
- FearAgent review weighting from the perceived score. Pending B1.

**Note on POG coupling:** Agent Permissions is shipped standalone, with no POG dependency. When POG is revived, it consumes this layer rather than replacing it.

**Scope shipped:** ~360 LOC (210 core including docstrings + 150 tests). No NAc/FearAgent edits this round — see deferred notes above.

**Exit:** Met for the enforced layer and the perceived tracker. Campaign YAML can declare `permissions:` blocks; the executor enforces them on the hot path; alias resolution can't bypass them; tracker convergence is verified across positive/negative valence streams. Tests green, lint clean.

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
- **No new tools.** This wave is cleanup, not feature work. Tool additions go through [tool_refinement_plan.md](../tool_refinement_plan.md).

## Risks

- **C1 fix touches the prompt handler path.** Verify it works in both `--sim` and plain `maxim` (non-sim) mode.
- **C3 default change is user-visible.** Users who've scripted against `clean` default will see more output. Mention in changelog.
- **C4 enforcement point is [runtime/executor.py](../../src/maxim/runtime/executor.py).** This is the tool dispatch hot path. Permission checks must be O(1) — no per-call YAML loads.

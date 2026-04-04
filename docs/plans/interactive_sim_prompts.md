# Interactive Simulation Prompts Plan

> **Status:** Not started. Reusable sim-layer infra. Needed by [DM Extensions](dungeon_master_extensions.md) (architect persona) but useful for other authoring-style personas.
>
> **Summary:** Add an `ask_user` tool with stdin prompts, configurable timeout, non-interactive default mode, and replay-from-session support. Enables human-in-the-loop persona flows across any persona willing to invoke it.

## Motivation

Simulation personas today only talk to the AUT. They have no way to ask the human operator clarifying questions. This forces single-shot generation patterns: the persona receives a one-line goal and must guess everything else.

This breaks down for authoring-style personas:
- **Adventure architect** (DM Extension B) — interview user on theme, tone, scope, constraints
- **Researcher persona** (planned) — confirm scope, hypotheses, success criteria
- **Refinement persona** — clarify ambiguous measurement goals

One shared `ask_user` primitive serves all three.

## Design

**Tool interface:**

```python
class AskUserTool(Tool):
    name = "ask_user"
    # params:
    #   question: str — the prompt text
    #   options: list[str] | None — if present, numbered multiple choice
    #   default: str — returned on timeout or in --non-interactive mode
    #   timeout_sec: int | None — default 300 (5 min); 0 = no timeout
    # Returns: { "response": str, "was_default": bool, "timed_out": bool }
```

**Modes:**

| Mode | Behavior |
|------|----------|
| Interactive (default) | Prompt via stdin, wait for response up to `timeout_sec`, fall back to `default` on timeout |
| `--non-interactive` | Return `default` immediately without prompting |
| `--replay-from <session>` | Read recorded responses from prior session's `user_interactions.jsonl`, return them in order |

**Audit log:** every question + response (or default/timeout) written to `data/sim_reports/{session}/user_interactions.jsonl`. Used for replay and debugging.

**Replay semantics:**
- Match questions by position in the log (turn-order, not content)
- If replay log exhausted mid-run, fall through to interactive mode
- If log has more entries than the new run needs, extras ignored
- Content drift warning: if question text doesn't match recorded question text at that position, log a warning and use recorded response anyway

## Implementation (~180 LOC, single phase)

**New files:**
- `src/maxim/simulation/tools_user.py` (~140) — `AskUserTool` with stdin handling, timeout via `select.select`, multiple-choice renderer, JSONL audit writer, replay reader
- `tests/unit/test_ask_user_tool.py` (~80) — piped-stdin, timeout fallback, non-interactive default, replay round-trip, replay-exhaustion fallthrough

**Modified:**
- `src/maxim/simulation/tools.py` — register `ask_user` in `SimToolRegistry`
- CLI arg parser — `--non-interactive`, `--replay-from <session_id>`
- `src/maxim/simulation/orchestrator.py` — propagate interaction mode + replay source into tool context

## Design Decisions

1. **Timeout default 5 min** — long enough for thoughtful answers, short enough that walk-aways don't hang the sim
2. **Timeout uses `select.select` on stdin** — no threads, no signal handlers, works on Unix. Windows compatibility deferred.
3. **Replay matches by position, not hash** — allows question text to evolve across persona iterations without breaking replays
4. **Audit log always written** — even in non-interactive and replay modes, for debugging
5. **No TUI framework** — plain stdin with `>` prompt; multiple choice renders as numbered list
6. **Prompt sanitization** — strip terminal escape codes from question text before rendering

## Risks

1. **Windows stdin compatibility** — `select.select` on stdin doesn't work on Windows. Mitigation: document Unix-only for MVP; if Windows needed, use threaded reader later.
2. **Replay drift** — question text evolves faster than response values stay relevant. Mitigation: content-drift warnings in logs; replay is best-effort not guaranteed.
3. **Prompt injection from persona output** — LLM-generated questions could contain escape codes or misleading formatting. Mitigation: sanitize on render.
4. **Concurrent personas** — if two personas ever ask simultaneously (not currently possible, but future agent-mesh scenarios), stdin becomes contended. Defer until multi-persona sims exist.

## Ties to Other Plans

| Plan | Relationship |
|------|-------------|
| [DM Extensions](dungeon_master_extensions.md) | Consumer — Extension B (architect persona) depends on this |
| **Research Protocol** (planned) | Likely consumer — scope confirmation before probes |
| **Realtime Refinement** (core done) | Optional consumer — clarification for ambiguous goals |
| **Agent Mesh** (blocked) | Future consumer — human approval for mesh-coordinated actions |

## When to Implement

**Prerequisite for DM Extension B (architect persona) only.** Not needed for DM MVP.

Can ship standalone any time — useful to any persona willing to invoke it. If a non-DM consumer surfaces first (researcher persona, refinement clarifications), ship this then.

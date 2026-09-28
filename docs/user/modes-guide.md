# Modes Guide

Maxim's mode system controls what the robot can do and how proactive it is. This guide explains the three autonomy levels and the sleep mechanism.

## Quick Reference

| Mode | What it Does | Max Initiative |
|------|-------------|:--------------:|
| `planning` | Propose actions, wait for approval | 0.3 |
| `supervised` | Act within defined boundaries | 0.7 |
| `autonomous` | Full autonomy, self-correcting | 1.0 |

Sleep is not a mode -- it is a processing state the agent enters by calling the `sleep` tool and exits automatically when user input arrives.

---

## Autonomy Levels

Maxim's behavior is controlled by two independent dimensions:

1. **ProcessingState** -- `awake` or `sleep`. Determines whether the agent loop is running.
2. **AutonomyLevel** -- `planning`, `supervised`, or `autonomous` (set via `--autonomy`). Controls permissions (what tools are available, whether code execution is allowed, filesystem access).

### Planning

The default mode. The agent proposes actions and waits for your approval before executing. Good for supervised operation.

- **Max Initiative:** 0.3 (mostly reactive)
- **Sandbox:** Always writable
- **CWD files:** Read only, edits require approval
- **Code execution:** Not allowed
- **Network:** Allowed
- **Forbidden tools:** `execute_file`, `maxim_command`, `request_directory_change`

### Supervised

The agent acts within defined boundaries. It can take actions but significant operations are gated by approval.

- **Max Initiative:** 0.7 (proactive within bounds)
- **Sandbox:** Full read/write
- **CWD files:** Read + suggest edits (shown for approval)
- **Code execution:** Requires approval
- **Network:** Allowed
- **No forbidden tools** (execution gated by approval)

### Autonomous

Full autonomy. The agent decides and acts on its own. Safety and ethical constraints (Constitution) still apply unconditionally.

- **Max Initiative:** 1.0 (fully proactive)
- **Sandbox:** Full access including execution
- **CWD files:** Full read/write/execute
- **Code execution:** Allowed
- **Network:** Allowed
- **No forbidden tools**, full tool access

```bash
# Start in planning mode (default)
maxim --language-model mistral-7b

# Start in supervised mode
maxim --autonomy supervised --language-model mistral-7b

# Time-boxed autonomous mode
maxim --autonomy autonomous --autonomy-duration 600
```

---

## Sleep

Sleep is a processing state, not a mode. The agent enters sleep by calling the `sleep` tool and wakes automatically when user input arrives.

When sleeping:
- LLM processing is **skipped**
- Background tasks run: memory consolidation, pattern extraction
- Only the `respond` tool is available
- Default Network is disabled

When user input arrives (text, voice, or wake keyword), the agent wakes and resumes its previous operational mode.

### Headless Mode

When no robot hardware is detected, Maxim runs in headless mode -- the full agentic loop runs without media capture, motor control, or Default Network overhead. Detection uses mDNS: if the robot's hostname doesn't resolve within 5 seconds, Maxim skips the SDK connection and starts immediately.

---

## Switching Modes at Runtime

You do not have to restart Maxim to change modes.

### Launch grant

The operational mode (the capability) is chosen at launch with `--operational-mode passive|active|singularity`
alongside an explicit `--mode` (#829); for example `maxim --mode agentic --operational-mode active`. Without
it, the CLI agent loop runs passive and the robot runtime follows its run mode. At runtime the agent can only
lower it; raising it takes a relaunch with the flag. That includes run-mode restarts: without a grant, a
robot launched in a passive run mode (`sleep`, `train`, `reflection`) cannot restart itself into `live` or
`agentic`, so launch it with `--operational-mode active` if it should. (The voice phrases below change the robot's
state-manager mode, not what dispatch enforces.)

### Voice Commands

- "Maxim sleep" -- agent enters sleep
- "Maxim wake up" -- wake from sleep
- "Maxim passive" -- switch to planning (passive) mode
- "Maxim active" -- switch to supervised (active) mode
- "Maxim singularity" -- **refused**: a phrase is heard from any audio in the room, so it cannot enter the
  code-executing mode (#828). Grant it deliberately at launch: `maxim --mode agentic --operational-mode singularity`.

### Agent Tools

- **`mode_switch`** -- The agent may switch itself to a mode with **no more capability** than its current one (e.g. active → passive). A switch that gains capability — passive → active, or anything → singularity — is refused: only a human can grant it (#821, #924). Logs switches with timestamps and reasoning.
- **`autonomy_level`** -- Request autonomy changes. Lowering autonomy (e.g. autonomous → planning) is immediate. Raising it needs human approval, and with no approval surface attached the request is refused, not left pending (#827).
- **`sleep`** -- Agent calls this to enter the sleep processing state. Wakes automatically on user input.

### Prompt Profiles

Control how much context the agent receives with `--log-level` (alias `--verbosity`):

- `0` -- Quiet. Minimal logging.
- `1` -- Normal. Goals and tool calls.
- `2` -- Debug. Full loop internals + perception + memory events.

For richer per-turn output, combine with `--display bio` (default) or `--display debug`.

# Modes Guide

Maxim's mode system controls what the robot can do, how proactive it is, and what resources it uses. This guide helps you pick the right mode for your situation.

## Quick Reference

| Mode | What it Does | When to Use | Needs Robot? | Needs LLM? |
|------|-------------|-------------|:------------:|:----------:|
| `agentic` | Full agent runtime with planning, perception, memory, and tool use | Complex tasks, general-purpose operation | No (but recommended) | Yes |
| `exploration` | Curiosity-driven discovery, object tracking, spatial mapping | First encounter with a new environment | Yes | Yes |
| `live` | Real-time vision, motor control, voice interaction | Demos, interactive sessions, gaze tracking | Yes | Yes |
| `sleep` | Background processing, wake-word monitoring | Idle periods, low power | No | No |
| `reflection` | Memory consolidation, self-evaluation, journaling | Between sessions, after long runs | No | Optional |
| `train` | Learning from user feedback and demonstrations | Teaching new motor behaviors | Yes | Yes |

---

## Agentic Mode (Recommended)

The full agent runtime. Runs a perception-memory-goal architecture with recursive planning, reflection loops, and tool use. This is the mode you want for anything non-trivial.

```bash
maxim --mode agentic --language-model mistral-7b
```

### Autonomy Levels

Control how much freedom the agent has with `--autonomy`:

- **planning** (default) -- The agent proposes actions and waits for your approval before executing.
- **supervised** -- The agent acts within defined boundaries but asks before anything significant.
- **autonomous** -- Full agency. The agent decides and acts on its own. Use `--autonomy-duration 300` to set a time limit in seconds.

```bash
# Propose-only mode (safest)
maxim --mode agentic --autonomy planning

# Time-boxed autonomous operation
maxim --mode agentic --autonomy autonomous --autonomy-duration 600
```

### Prompt Profiles

Control how much context the agent receives with `--agentic-verbosity`:

- `0` -- Quiet. Minimal logging.
- `1` -- Normal. Goals and tool calls.
- `2` -- Verbose. Adds perception and memory events.
- `3` -- Debug. Full loop internals.

---

## Legacy Modes

These predate the agentic runtime and map to specific combinations of operational mode and strategy (see "How Modes Combine" below). They are simpler to launch and still useful for focused tasks.

### Exploration

Active discovery mode. The robot looks around, tracks objects, and builds a spatial understanding of its environment. Good for the first time you put Maxim somewhere new.

Maps to: **active** mode + **explore** strategy (initiative 0.7).

```bash
maxim --mode exploration
```

Additional flags:

```bash
# Focus exploration on a topic
maxim --mode exploration --explore "kitchen objects"

# Limit how long the session runs
maxim --mode exploration --exploration-duration 300

# Set exploration autonomy separately from agentic autonomy
maxim --mode exploration --exploration-autonomy high

# Resume a previous exploration session
maxim --mode exploration --resume-session <session-id>

# List past exploration sessions
maxim --mode exploration --list-sessions
```

### Live

Real-time vision and motor control. Responds to voice commands, tracks people in the frame, and follows gaze targets. This is the interactive demo mode.

Maps to: **active** mode + **assist** strategy (initiative 0.7).

```bash
maxim --mode live
```

Live mode includes a self-evolution system called LiveModeIntent. The robot can define, review, and record its own behavioral intents during operation.

### Sleep

Background processing only. The robot monitors audio for wake words ("Maxim") but otherwise stays idle. Minimal CPU and memory usage.

Maps to: **sleep** processing state.

```bash
maxim --mode sleep
```

Use this when the robot should be available but not actively doing anything.

### Reflection

Passive introspection. The robot consolidates memories, reviews past actions, and generates insights. No movement, no interaction unless directly addressed.

Maps to: **passive** mode + **reflect** strategy (initiative 0.2).

```bash
maxim --mode reflection
```

Run this after long sessions to let the memory system organize what it learned.

### Train

Learning from user feedback. You demonstrate behaviors and label outcomes using keyboard shortcuts (keys 0-9). The robot adjusts its motor behavior based on your ratings.

Maps to: **passive** mode + **learn** strategy (initiative 0.3).

```bash
maxim --mode train
```

---

## How Modes Combine

Under the hood, Maxim's mode system has three independent dimensions:

1. **ProcessingState** -- `awake` or `sleep`. Determines whether the agent loop is running.
2. **OperationalMode** -- `passive`, `active`, or `singularity`. Controls permissions (what tools are available, whether code execution is allowed, filesystem access).
3. **Strategy** -- `observe`, `explore`, `research`, `assist`, `reflect`, or `learn`. Controls focus (what the robot pays attention to and how it behaves).

### Initiative Levels

Each operational mode and strategy has a `max_initiative` value between 0.0 (fully reactive) and 1.0 (fully proactive). The effective initiative is:

```
effective_initiative = min(mode.max_initiative, strategy.max_initiative)
```

| Operational Mode | Max Initiative |
|-----------------|:--------------:|
| passive         | 0.3            |
| active          | 0.7            |
| singularity     | 1.0            |

| Strategy  | Max Initiative |
|-----------|:--------------:|
| observe   | 0.2            |
| explore   | 0.8            |
| research  | 0.7            |
| assist    | 0.8            |
| reflect   | 0.3            |
| learn     | 0.3            |

For example, **passive + explore** gives `min(0.3, 0.8) = 0.3`. The passive mode caps the robot's proactivity even though the explore strategy would allow more. Conversely, **active + observe** gives `min(0.7, 0.2) = 0.2` -- the observe strategy keeps the robot quiet even though active mode permits action.

---

## Switching Modes at Runtime

You do not have to restart Maxim to change modes.

### Voice Commands

Say the mode name to switch:

- "Maxim sleep" -- enter sleep state
- "Maxim observe" -- switch strategy to observe
- "Maxim explore" -- switch strategy to explore
- "Maxim assist" -- switch strategy to assist

### ModeSwitchTool (Agentic Mode)

In agentic mode, the agent can switch its own operational mode and strategy through the ModeSwitchTool. This means the agent may change its own behavior in response to context -- for example, switching from explore to assist when you start asking questions.

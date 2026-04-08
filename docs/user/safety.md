# Safety Guide

## Overview

Maxim has multiple layers of safety protection. These operate independently -- if one layer fails, others still protect. The system defaults to safe behavior: when uncertain, it asks rather than acts.

This guide covers every protection layer so you know what is running, what it does, and how to configure it.

---

## Autonomy Levels

Three levels control how much the agent can do without asking you first.

| Level | What the Agent Can Do | What Requires Your Approval |
|---|---|---|
| `planning` (default) | Proposes actions, plans next steps | Everything. No action executes without your say-so. |
| `supervised` | Executes within pre-approved bounds | Dangerous or novel operations still need approval. |
| `autonomous` | Acts freely within policy constraints | FearAgent still blocks dangerous patterns. |

Set the level at startup:

```bash
maxim --autonomy planning       # safest, default
maxim --autonomy supervised
maxim --autonomy autonomous
```

You can also grant autonomy for a limited window. After the duration elapses, the system reverts to `planning`:

```bash
maxim --autonomy supervised --autonomy-duration 300   # 5 minutes, then back to planning
```

If you are unsure which level to use, leave it at `planning`. You can always promote to a higher level mid-session.

---

## FearAgent

Every tool call the agent wants to make passes through FearAgent before execution. This is a two-stage review:

1. **Pattern matching** -- A set of regex patterns detects known dangerous operations (recursive deletion, privilege escalation, credential access, and so on). This check is instant and does not require an LLM.
2. **LLM review** -- If a language model is available, it evaluates the action for safety. This catches nuanced risks that static patterns miss.

FearAgent classifies threats into eight categories:

- **SYSTEM_DAMAGE** -- Commands that could corrupt the OS or critical software.
- **DATA_LOSS** -- Destructive file operations, bulk deletes, format commands.
- **CREDENTIAL_EXPOSURE** -- Access to passwords, tokens, keys, or secrets.
- **NETWORK_RISK** -- Unexpected outbound connections, data exfiltration patterns.
- **PRIVILEGE_ESCALATION** -- Attempts to gain root or elevated permissions.
- **INJECTION** -- Shell injection, code injection, prompt injection patterns.
- **RESOURCE_ABUSE** -- Fork bombs, infinite loops, excessive resource consumption.
- **PHYSICAL_HARM** -- Movement commands that could damage the robot or its surroundings.

In `strict_mode`, anything flagged as suspicious is blocked outright. In normal mode, lower-severity items are allowed with a warning logged.

---

## FearGatedExecutor

In addition to FearAgent's per-call review, a dedicated `FearGatedExecutor` wraps the tool execution layer to ensure that every tool call in every mode -- robot, headless, or simulation -- is safety-gated. This operates **independently of DefaultNetwork**, so it applies even when no robot is connected.

FearGatedExecutor performs a two-tier review on every tool call:

1. **Action review** -- classifies the tool call (shell execution, file write, network request, generic tool) and runs it through FearAgent's pattern matcher and LLM reviewer.
2. **Code review** -- for bash commands, file writes, and edit operations, extracts the code content and scans it for dangerous patterns via `FearAgent.review_code()`.

DefaultNetwork retains its own FearAgent instance for motor and movement safety. FearGatedExecutor handles tool safety; DefaultNetwork handles motor safety. The two are independent layers in the safety stack.

---

## Pain Detection

The proprioceptive system continuously monitors the robot's physical state for aversive movement patterns:

- **Joint strain** -- A joint approaching its physical limits.
- **Rapid oscillation** -- Thrashing or jittering that indicates a control problem.
- **Sustained load** -- Prolonged force on a joint or actuator.
- **Collision detection** -- Unexpected resistance suggesting contact with an obstacle.

When pain is detected, the system reduces movement amplitude for the affected joint and can halt it entirely. Pain signals are written into memory, so the robot learns over time to avoid configurations that caused problems before.

---

## Predictive Harm Detection

This layer acts *before* a motor command is sent. It predicts the outcome of a planned movement and blocks it if it would violate safety bounds:

- **Velocity limits** -- Rejects commands that would exceed safe joint speeds.
- **Joint limits** -- Catches movements that would drive a joint past its mechanical range.
- **Workspace boundaries** -- Prevents the arm from reaching into regions that are known to be unsafe or uncharted.

Because this check runs before execution, there is zero latency between detection and prevention. The dangerous command never reaches the hardware.

---

## Workspace Bounds

The robot builds a model of its safe workspace over time:

- As the robot explores without triggering pain or collisions, safe movement zones expand.
- Bounds are persisted across sessions in `~/.maxim/util/learned_bounds.json`.
- Movements that would go outside learned bounds are attenuated (scaled down, not simply dropped).
- You can reset learned bounds if needed (see "What You Can Reset" below).

This means a brand-new Maxim installation starts conservative and gradually gains confidence as it maps its environment.

---

## Energy Budgets

Resource tracking prevents runaway behavior across multiple dimensions:

- **Token limits** -- Per-phase caps on LLM token consumption.
- **Movement energy** -- Cumulative tracking of motor effort.
- **Compute time** -- Wall-clock budgets for processing phases.

When any budget is exhausted, the current phase ends gracefully rather than crashing or continuing without limits. This protects against infinite loops, stuck planning cycles, and excessive motor activity.

---

## Escalation

When the system encounters something it is not confident about, it escalates rather than guessing:

- **Low confidence** -- The agent asks you for confirmation before proceeding.
- **Repeated failures** -- After multiple failed attempts at the same task, the system stops retrying and requests human intervention.
- **Safety threshold exceeded** -- The agent halts immediately and reports what happened.

Escalation thresholds are learned over time. If you frequently approve a certain class of action, the system becomes less likely to ask about it in future sessions.

---

## Filesystem Policy

File operations are governed by a path-based permission system:

- Each path has explicit read, write, and execute permissions.
- Sensitive paths (system directories, credential files) are blocked by default.
- Sandbox mode isolates script execution so that user-written scripts cannot escape their designated directory.
- All policy violations are logged for review.

---

## Internet Access

Internet access is **disabled by default**. To enable it:

```bash
maxim --internet-access
```

When internet access is enabled, the following safeguards apply:

- All HTTP fetches are logged with timestamps and URLs.
- Search queries are recorded.
- FearAgent reviews URLs before any fetch is executed, blocking known-dangerous or suspicious destinations.

---

## What You Can Reset

If safety systems become too conservative (or you simply want a fresh start), you can reset individual layers:

```bash
maxim --clear-memory fear          # Reset FearAgent learned thresholds
maxim --clear-memory escalation    # Reset escalation thresholds
maxim --clear-memory bounds        # Reset workspace bounds
maxim --clear-memory pain          # Reset pain learning
maxim --clear-memory all           # Reset everything
```

Resetting a layer returns it to its factory defaults. The robot will re-learn from scratch for that layer, so expect it to be more cautious again until it builds up experience.

---

## Summary

The safety stack, from outermost to innermost:

1. **Autonomy level** -- Controls whether the agent can act at all without asking.
2. **FearAgent** -- Reviews every tool call for danger patterns.
3. **FearGatedExecutor** -- Independent executor wrapper ensuring tool safety in all modes (robot, headless, simulation).
4. **Filesystem policy** -- Restricts which paths can be read, written, or executed.
5. **Internet policy** -- Blocks network access unless explicitly enabled.
6. **Predictive harm detection** -- Blocks unsafe movements before they reach hardware.
7. **Pain detection** -- Monitors the robot during movement and intervenes in real time.
8. **Workspace bounds** -- Limits the robot to regions it has safely explored.
9. **Energy budgets** -- Caps resource consumption to prevent runaway behavior.
10. **Escalation** -- Hands control back to you when confidence is low.

These layers are independent. Disabling or resetting one does not affect the others.

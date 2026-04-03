# Tools Reference

## Overview

In agentic mode, Maxim's LLM agent proposes tool calls to interact with the robot, filesystem, and external services. Tools are the only way the agent performs side effects. Each tool call is reviewed by the FearAgent before execution.

---

## Robot Control

### MoveTool
Move robot joints to specific positions. Controls head (pan/tilt), arms, and gripper.

### FocusInterestsTool
Set what the robot should pay attention to (e.g., "red objects", "people's faces").

### TrackTargetTool
Track a specific detected object by ID. The robot follows it with head movement.

### MaximCommandTool
Execute high-level robot commands (e.g., wave, nod, look around).

### NoveltyTrackTool
Query the novelty tracking system for information about detected objects.

---

## Communication

### RespondTool
Send a text response to the user (displayed in console).

### SpeakTool
Speak text aloud using text-to-speech. Requires the `--tts` flag.

### SendMessageTool
Send an SMS message via Twilio. Requires the `--comms` flag and Twilio credentials.

### CallUserTool
Make a voice call via Twilio. Requires the `--comms` flag and Twilio credentials.

---

## Filesystem

### ReadFileTool
Read file contents. Subject to filesystem policy (sandboxed paths).

### WriteFileTool
Write content to a file. Subject to filesystem policy.

### EditFileTool
Edit a file using old_text/new_text anchors with optional context_before/context_after for disambiguation.

### ExecuteFileTool
Execute a file (Python scripts, shell scripts). Subject to safety review.

### GlobTool
Find files matching glob patterns.

### RequestDirectoryChangeTool
Change the working directory for file operations.

### BashTool
Execute arbitrary bash commands. High-risk -- always reviewed by FearAgent.

---

## Code Tools

### CodeSearchTool
Search code with regex patterns across the codebase.

### RunTestsTool
Run pytest and return structured results (pass/fail counts, failure details).

---

## Git Tools

### GitDiffTool
Show git diff output for staged and unstaged changes.

### GitCommitTool
Create a git commit with a message.

---

## Internet

### InternetSearchTool
Search the web. Requires the `--internet-access` flag.

### InternetAccessTool
Fetch content from a URL. Requires the `--internet-access` flag.

### HttpFetchTool
Fetch raw HTTP content from URLs.

---

## Sandbox

### ExecuteSandboxScriptTool
Run Python scripts in an isolated sandbox environment with resource limits.

### CreateSandboxScriptTool
Create a script file in the sandbox.

### ReadSandboxFileTool
Read a file from within the sandbox.

### WriteSandboxFileTool
Write a file within the sandbox.

### ListSandboxTool
List files within the sandbox.

### ReadDataFileTool
Read files from the data directory.

### ListOtherInstanceOutputsTool
List output files from other running Maxim instances.

### ReadOtherInstanceOutputTool
Read an output file from another Maxim instance.

### WriteToSharedOutputsTool
Write a file to shared outputs for cross-instance communication.

---

## Math

### MathTool
Evaluate mathematical expressions safely. Supports arithmetic, statistics, and symbolic computation.

---

## Mode and Autonomy

### ModeSwitchTool
Switch operating mode at runtime.

### AutonomyLevelTool
Adjust the autonomy level (planning/supervised/autonomous).

---

## Provenance

### ExplainTool
Query the provenance system. Shows what happened during a cycle, why a decision was made, or concept history.

---

## Live Mode

### DefineLiveModeIntentTool
Define a new behavioral intent for live mode self-evolution.

### ReviewLiveModeIntentTool
Review and refine a live mode intent.

### RecordLiveIntentInsightTool
Record an insight from live mode execution.

### RecordLiveOutcomeTool
Record the outcome of a live mode intent.

---

## Tool Safety

All tool calls pass through FearAgent before execution. FearAgent uses:

1. **Deterministic pattern matching** -- regex for known dangerous patterns.
2. **LLM review** (if available) -- nuanced safety assessment for ambiguous cases.
3. **Configurable strictness levels** -- tunable per deployment.

Tools with side effects (filesystem writes, bash commands, git commits) receive extra scrutiny. The agent can be restricted via the `--autonomy` flag.

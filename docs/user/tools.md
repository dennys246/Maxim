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

## Introspection (Biological Self-Awareness)

Read-only tools that let the LLM query its own biological subsystems. These give the agent self-awareness about its memories, learned predictions, pain/fear history, temporal patterns, and energy consumption. No tool in this category modifies agent state.

### MemoryRecallTool (`memory_recall`)
Search episodic memories stored in the hippocampus. Filter by goal, tool name, success/failure, detected objects, people, mode, or time range. Use `expand=true` to find associated memories via spreading activation through ASSOCIATES and CAUSES edges in the hippocampal graph.

**Parameters:** `query`, `tool_name`, `success`, `object`, `person`, `mode`, `time_after`, `time_before`, `expand`, `limit`

### PredictOutcomeTool (`predict_outcome`)
Query the NAc (nucleus accumbens) causal learning system for what it predicts will happen if a specific tool is executed. Returns the Rescorla-Wagner predicted value, expected outcome valence (positive/negative/neutral), expected delay with confidence interval, and all possible outcomes with their probabilities.

**Parameters:** `tool_name` (required), `context`, `include_all_outcomes`

### CausalLinksTool (`causal_links`)
Inspect the learned cause-effect relationships in the NAc. See what events lead to what outcomes, with confidence scores, observation counts, and temporal delay distributions. Query by event signature, outcome signature, contributing memory ID, or valence filter.

**Parameters:** `event`, `outcome`, `memory_id`, `valence`, `limit`

### PainHistoryTool (`pain_history`)
Check pain and fear history. Shows statistics on pain signals (tool failures, movement errors, timeouts) from the PainDetector, and optionally checks whether the FearAgent would block a specific action.

**Parameters:** `check_action`, `action_params`, `limit`

### TemporalPatternsTool (`temporal_patterns`)
Query the SCN (suprachiasmatic nucleus) for time-based patterns. Find memories from specific times of day (0-23) or days of week (0=Mon, 6=Sun). Use `discover_rhythms=true` to find recurring patterns (e.g., failures cluster at specific times).

**Parameters:** `hour`, `day`, `discover_rhythms`, `limit`

### EnergyStatusTool (`energy_status`)
Check computational resource consumption. Returns recent energy usage (within a time window) and lifetime totals including token counts, inference costs, and average energy per event.

**Parameters:** `window_seconds`

### ConceptQueryTool (`concept_query`)
Search the ATL (anterior temporal lobe) semantic knowledge base. Find concepts by name or category, and explore typed relationships between concepts (IS_A, PART_OF, CAUSES, EXECUTES_WITH, ALIAS_OF, etc.).

**Parameters:** `name`, `category`, `concept_id`, `relationship_type`, `limit`

### SceneSummaryTool (`scene_summary`)
Get a summary of the current visual scene from the salience and attention networks. Shows the most salient objects with novelty scores, current gaze focus point, dwell time, and next suggested attention target. Only available when vision subsystems are active (not in headless mode).

**Parameters:** `top_n`, `include_attention`

### SimilaritySearchTool (`similarity_search`)
Find situations similar to a past experience using the EC (entorhinal cortex). Uses multi-modal similarity (structural hash, temporal bins, semantic embedding, context match) via LSH approximate nearest neighbor. Search by tool name or by a specific memory ID.

**Parameters:** `tool_name`, `memory_id`, `context`, `limit`

### SystemStatsTool (`system_stats`)
Aggregate health summary of all biological subsystems in one query. Returns statistics from the hippocampus (memory counts), NAc (causal link counts), EC (signature counts), ATL (concept counts), energy tracker (consumption totals), pain detector (signal counts), and significance learner (heuristic weights and convergence status).

**Parameters:** None

---

## Tool Safety

All tool calls pass through FearAgent before execution. FearAgent uses:

1. **Deterministic pattern matching** -- regex for known dangerous patterns.
2. **LLM review** (if available) -- nuanced safety assessment for ambiguous cases.
3. **Configurable strictness levels** -- tunable per deployment.

Tools with side effects (filesystem writes, bash commands, git commits) receive extra scrutiny. The agent can be restricted via the `--autonomy` flag.

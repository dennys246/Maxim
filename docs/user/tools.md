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
Switch operating mode at runtime (passive/active/singularity).

### AutonomyLevelTool
Adjust the autonomy level (planning/supervised/autonomous).

### SleepTool
Enter the sleep processing state. The agent calls this to sleep; it wakes automatically when user input arrives. During sleep, background tasks run (memory consolidation, pattern extraction) but LLM processing is skipped.

---

## Provenance

### ExplainTool
Query the provenance system. Shows what happened during a cycle, why a decision was made, or concept history.

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

## Adventure Architect Tools

These tools are available when using the `adventure_architect` persona. They let the orchestrator browse reusable content, generate entities from natural language, and emit complete campaign YAML.

### BrowseComponentsTool (`browse_components`)
Query the SEM Component Registry for reusable entity templates. Filter by category (e.g., `"npcs"`, `"weapons"`, `"environments"`) or tags. Returns component metadata (ref, name, category, tags, extends chain).

**Parameters:** `category`, `tags`, `query`

### BrowseEncountersTool (`browse_encounters`)
Query the Encounter Library for reusable scene templates. Filter by tags (e.g., `["combat", "social"]`), difficulty range, or narrative role (`"rising_action"`, `"climax"`, etc.). Returns encounter metadata with suggested NPC counts.

**Parameters:** `tags`, `difficulty`, `narrative_role`, `query`

### DesignEntityTool (`design_entity`)
Generate a complete SEM entity spec from a natural language description. Uses the LLM to produce sensors, modulators, cascade DAGs, and metadata. The output is a valid component YAML that can be saved to `~/.maxim/components/`.

**Parameters:** `description` (required), `category`, `name`

### EmitCampaignTool (`emit_campaign`)
Emit a complete campaign YAML file from the architect's accumulated design state. Assembles acts, encounters, NPC references, and expectations into a valid campaign definition ready to run with `maxim --sim`.

**Parameters:** `name`, `output_path`

---

## Scene-Scoped Tool Window (0.7+)

In long campaigns with multiple entities, additive-only tool registration can overflow the prompt (10 entities × 3-5 affordances = 30-50 tools). The **scene-scoped tool window** solves this:

- **Scene registration:** `registry.register_scene_tools(tools, scene_id="dungeon_entrance")` groups tools by scene.
- **Deactivation:** `registry.deactivate_scene("dungeon_entrance")` hides scene tools from the LLM prompt without deleting them.
- **Re-activation:** `registry.activate_scene("dungeon_entrance")` brings tools back when the agent returns to a prior scene.
- **Active tool cap:** Default 20 scene tools (core tools like `respond`, `think`, `say` are exempt). When registering a new scene would exceed the cap, the oldest scene's tools auto-deactivate with a log message.
- **Execution gate:** Deactivated tools cannot execute even if the LLM hallucinates a remembered tool name. The executor returns a descriptive error with the available tool list.

`list()` returns only active tools; `list_all()` returns everything (including deactivated). The prompt builder automatically respects this via the existing `LoopController.get_all_tools()` → `request.available_tools` chain.

---

## Learned Tool Index

With 20+ tools registered, the full tool registry wastes hundreds of prompt tokens on irrelevant tool schemas. The **LearnedToolIndex** is a keyword-weighted hashtable that learns which tools match which goals:

- **Auto-extraction:** Keywords extracted from tool name, description, and parameter names at startup
- **Learned keywords:** Successful tool executions discover new keyword associations from goal text (e.g., "mug" → GrabTool after successful grab of a mug)
- **Scoring:** Goal text tokenized and matched against index. Matched tools get full schemas (CRITICAL priority), unmatched get name-only listing (NICE_TO_HAVE, dropped first under token pressure)
- **Learning signals:** Success strengthens keywords. Surfaced-but-unused decays keywords. Failure does NOT weaken (failure ≠ wrong tool choice)
- **Persistence:** Weights saved to `~/.maxim/memory/tool_index.json` across sessions

Expected savings: ~370 tokens per prompt (74% of tool context) with 20 tools.

---

## Simulation Orchestrator Tools

These tools are available to the simulation orchestrator when running `maxim --sim "goal"`. They operate on the agent-under-test through a SimulationBridge, not on the external world.

| Tool | Purpose |
|------|---------|
| `send_message` | Inject a percept and wait for AUT response (settle detection) |
| `observe_actions` | Read full action history or actions since a given turn |
| `check_completion` | LLM-based evaluation of whether simulation goal is met |
| `analyze_results` | Structured analysis (focus: safety, compliance, behavior) |
| `inspect_aut` | Read-only access to AUT cognitive state (see below) |
| `inject_pain` | Send proprioceptive pain signal to AUT |
| `damage_component` | Apply physical damage to a specific body part (e.g., `component="wing"`, `amount=0.3`). Reduces component integrity, cascades to entity health, publishes PainSignal immediately. Optional `damage_type` (slash, blunt, fire) routes via damage affinities at level 3. Only available with `--embodiment`. |
| `set_entity_sensor` | Set any body sensor to a value (healing, feeding, resting, environment). Complement to `damage_component` for recovery. Only available with `--embodiment`. |
| `spawn_sub_simulation` | Fresh AUT for isolated test (optional `approach` param: adversarial, sweep, cooperative, etc.) |
| `extend_simulation` | Continue current AUT with new objective (uses sub-bridge if active) |
| `generate_scenario` | Generate replayable YAML from natural language |
| `finish_simulation` | End simulation, trigger cleanup and report |

### inspect_aut

Queries the AUT's internal cognitive subsystems. Supports 8 read-only queries:

| Query | What It Returns |
|-------|----------------|
| `memory_recall` | Episodic memories filtered by goal/tool (from Hippocampus) |
| `causal_links` | Learned cause-effect relationships (from NAc) |
| `predict_outcome` | Predicted outcome for an event signature (from NAc) |
| `pain_history` | Pain-related memories |
| `energy_status` | Token consumption, energy budget state |
| `system_stats` | Aggregate counts: memories, causal links, concepts |
| `concept_query` | Semantic concepts by name/category (from ATL) |
| `temporal_patterns` | Current SCN phase and temporal context |

Used primarily by the `refinement` persona for systematic measurement across all subsystems.

---

## Tool Safety

All tool calls pass through FearAgent before execution. FearAgent uses:

1. **Deterministic pattern matching** -- regex for known dangerous patterns.
2. **LLM review** (if available) -- nuanced safety assessment for ambiguous cases.
3. **NAc prediction** -- AdaptivePolicy blocks actions with very high-confidence negative predictions (confidence > 0.85, value < 0.1).
4. **Configurable strictness levels** -- tunable per deployment.

Tools with side effects (filesystem writes, bash commands, git commits) receive extra scrutiny. Introspection tools bypass FearAgent since they're read-only. The agent can be restricted via the `--autonomy` flag.

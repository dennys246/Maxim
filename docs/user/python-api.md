# Python API Reference

Maxim exposes a verb-based Python API for programmatic access to all features, plus a composable object API for power users and researchers.

## Installation

```bash
pip install pymaxim                        # Core (memory, planning, simulation)
pip install pymaxim[llm-anthropic]         # + Claude support
pip install pymaxim[llm-openai]            # + OpenAI/GPT support
pip install pymaxim[vision]                # + Camera/vision perception
pip install pymaxim[audio]                 # + Whisper audio transcription
```

## Quick Start

```python
import maxim

# Check your environment
report = maxim.diagnose()

# Run a simulation — returns a persistent Session
session = maxim.imagine(goal="test memory recall", persona="cooperative")
print(session.id)        # "20260408_143022"
print(session.turns)     # 12

# Observe agent state from this session
memories = session.observe("memory")
causal = session.observe("causal")
```

---

## Verb API (Simple Path)

Top-level functions for common operations. All heavy imports are deferred.

### Core Verbs

| Verb | Purpose | Returns |
|------|---------|---------|
| `configure(verbosity, log_file, debug, show)` | Set logging + tracing | None |
| `run(model, goal, headless)` | Run the agentic cycle | None (blocks) |
| `imagine(goal, persona, scenario, model, resume)` | Run a simulation | `Session` |
| `connect(robot_type, name, config)` | Connect to a robot | `RobotController` |
| `diagnose(peer, api_key)` | Environment diagnostics | `DiagnosticReport` |
| `observe(subsystem, keyword, limit)` | Query cognitive state | `dict` |
| `introspect(...)` | Alias for `observe()` | `dict` |
| `list_models()` | Available LLM profiles | `dict[str, list[ModelInfo]]` |

### Simulation & Research Verbs

| Verb | Purpose | Returns |
|------|---------|---------|
| `campaign(path, model, party_mode)` | Run a DM campaign | `CampaignResult` |
| `benchmark(models, suite, runs)` | Multi-model comparison | `BenchmarkResult` |
| `research(goal, campaign, model)` | Experiment + paper protocol | `ResearchResult` |

### Extension Verbs

| Verb | Purpose | Returns |
|------|---------|---------|
| `on(event, callback)` | Subscribe to agent events | `EventHandle` |
| `register_tool(tool)` | Add a custom tool | None |
| `register_persona(name, ...)` | Add a simulation persona | None |
| `@tool` | Decorator to register a function as a tool | decorated function |

### Loading from Disk (`maxim.load`)

All loading/restoring goes through `maxim.load` — the single canonical namespace:

| Function | Purpose | Returns |
|----------|---------|---------|
| `load.hippocampus(path)` | Restore episodic memory | `Hippocampus` |
| `load.nac(path)` | Restore causal model | `NAc` |
| `load.atl(path)` | Restore semantic concepts | `ATL` |
| `load.session(session_id)` | Load a persisted session (fuzzy match) | `Session` |
| `load.sessions(limit=20)` | List recent sessions | `list[Session]` |
| `load.agent(name, base_dir)` | Restore a persisted agent with all subsystems | `AgentInstance` |
| `load.entity(path)` | Load entity from YAML or saved JSON | `Entity` |

---

## Composable Object API (Power Path)

For researchers, multi-agent orchestration, and programmatic composition. Access individual subsystems via `maxim.create`, reload from disk via `maxim.load`.

### Bio-Subsystems

Create standalone cognitive components:

```python
import maxim

# Episodic memory
hippo = maxim.create.hippocampus(persistence_path="/tmp/memory.json")
hippo.store_observation("The wolf was near the cave entrance")
hippo.store_observation("The key was under the mat")
memories = hippo.recall("wolf", limit=3)
hippo.save()  # Persist to disk

# Causal learning
nac = maxim.create.nac()
nac.record_event("action", "ate_mushroom")
prediction = nac.predict("action", "ate_mushroom")

# Semantic concepts
atl = maxim.create.atl(persistence_path="/tmp/concepts.json")
concept_id, created = atl.find_or_create("wolf", category="creature")

# Temporal indexing
scn = maxim.create.scn()
# SCN uses TemporalSignature objects for registration
from maxim.time.temporal_signature import TemporalSignature
sig = TemporalSignature.from_timestamp(1712592000.0)
scn.register("mem_001", sig)

# Algebraic memory
ag = maxim.create.angular_gyrus()
```

#### Modifying Bio-Subsystems

All subsystems support add/remove/save/load:

```python
# Add and remove memories
mem_id = hippo.store_observation("temporary observation")
hippo.remove(mem_id)  # Delete by ID

# NAc causal learning
nac.record_event("action", "opened_chest")
nac.decay_all(factor=0.9)         # Age all links
nac.remove_memory("old_mem_id")   # Remove memory references

# ATL concepts
atl.remove("concept_id")          # Delete a concept
atl.define_relationship(id1, id2, "is_a")  # Add relationships

# Save/load any subsystem
hippo.save("/path/to/hippo.json")
hippo = maxim.load.hippocampus("/path/to/hippo.json")

nac.save("/path/to/nac.json")
nac = maxim.load.nac("/path/to/nac.json")
```

### Agents

Create standalone agents with isolated subsystems:

```python
# Create an agent (no agent loop — just subsystems)
agent = maxim.create.agent(
    "scout",
    personality="cautious and observant",
    remembers=True,   # Gets its own Hippocampus
    learns=True,      # Gets its own NAc
)

# Use the agent's subsystems
agent.hippocampus.store_observation("I saw movement in the shadows")
agent.nac.record_event("observation", "saw_movement")

# Export memory state
export = agent.export_memories()
print(f"{export['episodic_memories']} memories, {export['causal_links']} links")

# Clean up
agent.shutdown()
```

#### Modifying Agents

Agents are fully mutable after creation:

```python
# Update personality
agent.personality = "bold and reckless"

# Swap out the entire hippocampus
old_hippo = agent.hippocampus
agent.hippocampus = maxim.create.hippocampus()

# Load a pre-trained hippocampus from disk
agent.hippocampus = maxim.load.hippocampus("/path/to/expert_memory.json")

# Replace the NAc (causal model)
agent.nac = maxim.load.nac("/path/to/trained_nac.json")
```

### Multi-Agent Pools

Orchestrate multiple agents:

```python
pool = maxim.create.pool(max_workers=4)
pool.add(maxim.create.agent("guard", personality="stern and loyal"))
pool.add(maxim.create.agent("merchant", personality="cunning trader"))

# Run individual turns (no LLM needed for subsystem operations)
guard = pool.get_agent("guard")
guard.hippocampus.store_observation("A stranger entered the market")

# Export all agent memories
for agent_id, memories in pool.export_all_memories().items():
    print(f"{agent_id}: {memories['episodic_memories']} memories")

# Remove an agent (flushes state)
pool.remove("guard")
pool.shutdown()
```

### SEM Entities (Embodiment)

Create and compose sensor-entity-modulator trees:

```python
# From templates
guard = maxim.create.entity("npcs/guard", name="Captain Aldric")
wolf = maxim.create.entity("creatures/wolf")

# Browse available templates
for category, names in maxim.create.templates().items():
    print(f"{category}: {', '.join(names)}")

# From code
from maxim import Entity, Sensor, Modulator

robot_arm = Entity(
    name="left_arm",
    entity_type="limb",
    sensors={"position": Sensor(name="position", modality="proprioception")},
    modulators={"servo": Modulator(name="servo", modality="motor")},
)

# Compose into trees
body = Entity(name="robot", entity_type="body")
robot_arm.reparent(body)  # arm becomes child of body

# Create embodiment runtime
embodiment = maxim.create.embodiment(body)
readings = embodiment.read_all()
```

#### Modifying Entities

Entities are mutable — sensors, modulators, metadata, and tree structure can all be changed:

```python
# Add/modify metadata
guard.metadata["faction"] = "royal_guard"
guard.metadata["alert_level"] = 3

# Modify vital metrics
guard.vital_metrics["health"] = 0.8

# Change visibility
guard.hide("secret_passage")
guard.reveal("main_gate")

# Reparent (move in entity tree)
child_entity.detach()
child_entity.reparent(new_parent)

# Save/load entity trees (preserves metadata, vital metrics, children)
guard.save("/tmp/modified_guard.json")
loaded_guard = maxim.load.entity("/tmp/modified_guard.json")
```

### LLM Router

Direct inference without the full agent loop:

```python
llm = maxim.create.router(model="claude-sonnet")
response = llm.generate("What should I do next?", max_tokens=100)
```

---

## Sessions

Sessions are persistent containers for simulation data. Every `imagine()` call returns a Session.

### Creating and Resuming

```python
# Run a simulation
session = maxim.imagine(goal="test memory recall", model="mistral-7b")
print(session.id)  # "20260408_143022"

# Resume later (agent has memories from prior run)
session = maxim.imagine(goal="add interference", resume=session.id)
```

### Saving Session Metadata

```python
# Save session metadata to disk
session.save()  # Creates/updates session.json in the session directory
```

### Inspecting Session Data

```python
# Bio-state from THIS session's persisted data
memories = session.observe("memory")
causal = session.observe("causal")
concepts = session.observe("concepts")
pain = session.observe("pain")

# Simulation metadata
print(session.turns, session.duration_s, session.finish_reason)
```

### Loading Past Sessions

```python
# Load by ID (supports fuzzy prefix match)
session = maxim.load.session("20260408")

# List recent sessions
for s in maxim.load.sessions(limit=5):
    print(f"{s.id}: {s.goal} ({s.turns} turns)")
```

### Generating Research Reports

```python
# Generate a report from session data
report = session.research()

# Save in multiple formats
report.save("findings.md")    # Markdown (human-readable)
report.save("findings.json")  # Structured JSON (machine-readable)

# Load a report back
from maxim import Report
report = Report.from_json("findings.json")
print(report.metrics)

# Future formats (planned):
# report.save("findings.pdf")   # Requires pymaxim[docs]
# report.save("findings.docx")  # Requires pymaxim[docs]
```

---

## Persistence Reference

All data is persisted using crash-safe atomic writes (`fsync` + temp file + rename).

### What Gets Saved Automatically

| Subsystem | Auto-saved on | Format | Location |
|-----------|--------------|--------|----------|
| **Hippocampus** | Agent shutdown, sleep consolidation | JSON v3.0 | `~/.maxim/agents/{id}/hippocampus.json` |
| **NAc** | Agent shutdown, session end | JSON v1.0 | `~/.maxim/agents/{id}/nac.json` |
| **ATL** | Session end | JSON v1.0 | `~/.maxim/agents/{id}/atl.json` |
| **SCN** | Session end | JSON v3.0 | `~/.maxim/agents/{id}/scn.json` |
| **Angular Gyrus** | Session end | JSON v1.0 | Per config persistence_path |
| **Simulation reports** | After each sim | JSON | `~/.maxim/sim_reports/{session_id}/` |

### Manual Save/Load

```python
# Any subsystem
hippo.save("/path/to/hippo.json")
hippo = maxim.load.hippocampus("/path/to/hippo.json")

nac.save("/path/to/nac.json")
nac = maxim.load.nac("/path/to/nac.json")

atl.save("/path/to/atl.json")
atl = maxim.load.atl("/path/to/atl.json")

# Entity trees (new)
entity.save("/path/to/entity.json")
entity = maxim.load.entity("/path/to/entity.json")

# Sessions
session.save()
session = maxim.load.session("20260408")
```

### Agent Persistence

**Key distinction:** `maxim.create.*` always creates fresh objects. `maxim.load.*` restores from disk.

When using `maxim.create.agent()`, each agent gets its own persistence directory:

```
~/.maxim/agents/{agent_id}/
  hippocampus.json   # Episodic memories
  nac.json           # Causal links
  atl.json           # Semantic concepts
  scn.json           # Temporal indices
```

On `agent.shutdown()`, all subsystems are saved. Use `maxim.load.agent()` to restore:

```python
# First session — create fresh agent
agent = maxim.create.agent("scout", personality="cautious", remembers=True, learns=True)
agent.hippocampus.store_observation("Wolves hunt at dusk")
agent.nac.record_event("observation", "wolves_at_dusk")
agent.shutdown()  # Saves hippocampus + NAc + ATL + SCN

# Later session — load persisted agent (memories survive)
agent = maxim.load.agent("scout")
# agent.hippocampus already contains "Wolves hunt at dusk"

# Note: maxim.create.agent("scout") would start fresh (no memories)
```

---

## Configuration

```python
maxim.configure(verbosity=2)                    # Verbose logging
maxim.configure(debug="hippo,nac")              # Trace specific subsystems
maxim.configure(log_file="maxim.log")           # Log to file
maxim.configure(show="bio")                     # Filter simulation output channels
```

## Error Handling

```python
from maxim import MaximError, ConfigurationError, ModelError

try:
    maxim.run(model="nonexistent-model")
except ConfigurationError as e:
    print(f"Config issue: {e}")
    # e.context has additional debug info
except ModelError as e:
    print(f"Model issue: {e}")
except MaximError as e:
    print(f"Maxim error: {e}")
```

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `MAXIM_DATA_HOME` | User data directory | `~/.maxim` |
| `MAXIM_LLM_ENABLED` | Enable LLM inference | `1` |
| `MAXIM_LLM_PROFILE` | Default model profile | `mistral-7b` |
| `ANTHROPIC_API_KEY` | Claude API key | — |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `GOOGLE_API_KEY` | Gemini API key | — |
| `GROQ_API_KEY` | Groq API key | — |
| `TOGETHER_API_KEY` | Together.ai API key | — |
| `FIREWORKS_API_KEY` | Fireworks API key | — |
| `MISTRAL_API_KEY` | Mistral API key | — |
| `DEEPSEEK_API_KEY` | DeepSeek API key | — |

## Available Cloud Models

| Profile | Provider | API Key Env |
|---------|----------|-------------|
| `claude-sonnet` | Anthropic | `ANTHROPIC_API_KEY` |
| `gpt-4o` | OpenAI | `OPENAI_API_KEY` |
| `gemini-2.5-flash` | Google | `GOOGLE_API_KEY` |
| `gemini-2.5-pro` | Google | `GOOGLE_API_KEY` |
| `groq-llama3-70b` | Groq | `GROQ_API_KEY` |
| `groq-mixtral` | Groq | `GROQ_API_KEY` |
| `together-llama3-70b` | Together.ai | `TOGETHER_API_KEY` |
| `fireworks-llama3-70b` | Fireworks | `FIREWORKS_API_KEY` |
| `mistral-large` | Mistral | `MISTRAL_API_KEY` |
| `mistral-small` | Mistral | `MISTRAL_API_KEY` |
| `deepseek-chat` | DeepSeek | `DEEPSEEK_API_KEY` |
| `deepseek-reasoner` | DeepSeek | `DEEPSEEK_API_KEY` |

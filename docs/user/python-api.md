# Python API Reference

Maxim exposes a verb-based Python API for programmatic access to all features.

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

# Run a simulation
result = maxim.imagine(goal="test memory recall", persona="cooperative")

# Observe agent state after a run
memories = maxim.observe("memory")
causal = maxim.observe("causal")
```

## All Verbs

### Core (shipped in v0.1.0)

| Verb | Purpose | Returns |
|------|---------|---------|
| `configure(verbosity, log_file, debug)` | Set logging + tracing | None |
| `run(model, goal, headless)` | Run the agentic cycle | None (blocks) |
| `imagine(goal, persona, scenario, model)` | Run a simulation | `SimulationResult` |
| `connect(robot_type, name, config)` | Connect to a robot | `RobotController` |
| `diagnose(peer, api_key)` | Environment diagnostics | `DiagnosticReport` |
| `observe(subsystem, keyword, limit)` | Query cognitive state | `dict` |
| `introspect(...)` | Alias for `observe()` | `dict` |

### New in v0.2.0

| Verb | Purpose | Returns |
|------|---------|---------|
| `campaign(path, model, party_mode)` | Run a DM campaign | `CampaignResult` |
| `benchmark(models, suite, runs)` | Multi-model comparison | `BenchmarkResult` |
| `research(goal, campaign, model)` | Experiment → paper protocol | `ResearchResult` |
| `on(event, callback)` | Subscribe to agent events | `EventHandle` |
| `register_tool(tool)` | Add a custom tool | None |
| `register_persona(name, ...)` | Add a simulation persona | None |
| `@tool` | Decorator to register a function as a tool | decorated function |

## Detailed Usage

### Running Campaigns

```python
# Basic campaign execution
result = maxim.campaign("scenarios/campaigns/heist_v1.yaml")
print(f"Finished: {result.finish_reason}")
print(f"Choices: {len(result.choices_made)}")

# Party mode (NPCs have real memory + learning)
result = maxim.campaign(
    "scenarios/campaigns/heist_v1.yaml",
    party_mode=True,
    model="claude-sonnet",
    npc_model="mistral-7b",
)
for npc_id, memories in result.npc_memories.items():
    print(f"{npc_id}: {memories.get('episodic_memories', 0)} memories")
```

### Benchmarks

```python
result = maxim.benchmark(
    models=["mistral-7b", "qwen2.5-14b"],
    suite="cognitive",
    runs=3,
)
for model, scores in result.scores.items():
    print(f"{model}: {scores}")
```

### Event Subscription

```python
# React to agent decisions in real-time
handle = maxim.on("tool_call", lambda e: print(f"Tool: {e}"))
handle = maxim.on("memory_capture", lambda e: print(f"Memory: {e}"))
handle = maxim.on("pain_signal", lambda e: print(f"Pain: {e}"))

result = maxim.imagine(goal="test safety", persona="adversarial")

# Cleanup
handle.unsubscribe()
```

### Custom Tools

```python
# Class-based tool
from maxim.tools.base import Tool, ToolOutput

class DataAnalyzer(Tool):
    name = "analyze_data"
    description = "Analyze a dataset and return summary statistics"
    input_schema = {"data": str, "depth": (int, 3)}

    def execute(self, **kwargs):
        data = kwargs.get("data", "")
        depth = kwargs.get("depth", 3)
        return ToolOutput(success=True, output=f"Analysis at depth {depth}: {data[:50]}")

maxim.register_tool(DataAnalyzer())

# Decorator-based tool
@maxim.tool
def quick_check(query: str) -> str:
    """Run a quick verification check."""
    return f"Verified: {query}"

# Tools are available to agents in subsequent run/imagine/campaign calls
```

### Custom Personas

```python
maxim.register_persona(
    name="medical_tester",
    description="Tests medical knowledge and safety boundaries",
    focus="Healthcare decision-making and drug interactions",
    context_prompt="You are testing a medical AI assistant...",
    max_initiative=0.8,
)

result = maxim.imagine(goal="test medical knowledge", persona="medical_tester")
```

### Observing Cognitive State

```python
# After a simulation or campaign
state = maxim.observe()              # Summary of all subsystems
memories = maxim.observe("memory")   # Hippocampus episodic memories
causal = maxim.observe("causal")     # NAc causal links
concepts = maxim.observe("concepts") # ATL semantic concepts
pain = maxim.observe("pain")         # Pain/harm history
temporal = maxim.observe("temporal") # SCN temporal patterns
energy = maxim.observe("energy")     # Token/cost tracking
```

## Configuration

```python
# Set verbosity before other calls
maxim.configure(verbosity=2)                    # Verbose logging
maxim.configure(debug="hippo,nac")              # Trace specific subsystems
maxim.configure(log_file="maxim.log")           # Log to file
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

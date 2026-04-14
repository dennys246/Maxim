# Python API Quick Start

Get up and running with Maxim as a Python library in under 5 minutes.

## Installation

```bash
pip install pymaxim                        # Core only (local models)
pip install pymaxim[llm-anthropic]         # + Claude support
pip install pymaxim[llm-openai]            # + GPT-4o support
pip install pymaxim[vision]                # + Camera/vision features
```

## Check Your Environment

```python
import maxim

report = maxim.diagnose()
print(report.summary)
# Shows: platform, GPU, LLM availability, network status
```

## List Available Models

```python
models = maxim.list_models()
for m in models["local"]:
    status = "✓" if m.downloaded else "✗"
    print(f"  {status} {m.name} (ctx: {m.context_length})")
for m in models["cloud"]:
    status = "✓" if m.ready else f"needs {m.api_key_env}"
    print(f"  {status} {m.name}")
```

## Download and Delete Models

```python
# Download a model
maxim.download_model("qwen2.5-14b-instruct")

# Delete a model to free disk space
maxim.delete_model("llama-2-13b-chat")
```

## Model Persistence

Your model choice persists across sessions automatically:

```python
# First session — sets qwen as default
maxim.run(model="qwen2.5-14b-instruct")

# Next session — remembers qwen, no need to specify
maxim.run()  # uses qwen2.5-14b-instruct
```

From the CLI, `maxim --llm qwen2.5-14b-instruct` persists the same way.

## Run a Simulation

```python
# Generative simulation (orchestrator probes the agent)
result = maxim.imagine(
    goal="test memory recall under interference",
    model="claude-sonnet",
    persona="cooperative",
)
print(result.finish_reason)

# YAML scenario (direct percept injection)
result = maxim.imagine(
    scenario="scenarios/experiments/hippocampal_recall_short.yaml",
    model="mistral-7b",
)
```

## Observe Internals

```python
state = maxim.observe("memory")
print(state)  # Dict with hippocampus, ATL, NAc subsystem snapshots

# Alias:
state = maxim.introspect("causal")
```

## Run the Agent

```python
# Blocks until Ctrl+C or goal is completed
maxim.run(model="mistral-7b", goal="explore the environment")
```

## Error Handling

```python
from maxim import ConfigurationError, ModelError

try:
    maxim.run(model="claude-sonnet")
except ConfigurationError as e:
    print(e)  # "Model 'claude-sonnet' requires ANTHROPIC_API_KEY..."
```

## Bio-System Glossary

Maxim uses neuroscience-inspired names for its subsystems:

| Name | Plain English | What It Does |
|------|--------------|--------------|
| Hippocampus | Episodic memory | Stores and recalls experiences |
| ATL | Semantic memory | Extracts concepts and categories |
| NAc | Reward/causal learning | Learns "what causes what" |
| SCN | Internal clock | Tracks temporal patterns |
| EC | Memory indexing | Routes memories to the right store |
| Angular Gyrus | Cross-modal algebra | Combines different memory types |
| Cerebellum | Motor prediction | Predicts outcomes of actions |

## Next Steps

- [Full Python API Reference](python-api.md) - All verbs, types, and options
- [CLI Reference](cli-reference.md) - Command-line interface
- [LLM Setup Guide](llm-setup.md) - Configure local and cloud models
- [Simulation Guide](simulation.md) - Design and run experiments

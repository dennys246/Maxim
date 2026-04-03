# LLM Setup

## Overview

Maxim uses local LLM inference by default via llama.cpp. Cloud backends (Anthropic, OpenAI) are optional and opt-in. Local inference keeps everything on your machine -- no data leaves your network.

## Quick Start

```bash
pip install -e '.[llm]'
./scripts/download_models.sh --llm --enable
maxim --mode agentic --language-model smollm-1.7b
```

## Local Models

### Available Profiles

| Profile | Model | Size | Context | Best For |
|---------|-------|------|---------|----------|
| `smollm-1.7b` | SmolLM 1.7B Instruct | ~1.1 GB | 2048 | CPU-only, low RAM, fast iteration |
| `mistral-7b` | Mistral 7B Instruct v0.2 | ~4.4 GB | 8192 | Best balance of quality and speed |
| `phi3-mini` | Phi-3 Mini 4K Instruct | ~2.3 GB | 4096 | Good quality, moderate size |
| `llama3-8b` | Llama 3 8B Instruct | ~4.9 GB | 8192 | Highest quality local |
| `qwen2-7b` | Qwen2 7B Instruct | ~4.4 GB | 8192 | Strong multilingual support |

### Downloading Models

```bash
./scripts/download_models.sh --llm --enable
```

This downloads the default model (Mistral 7B) in Q4_K_M quantization. To download a specific model, edit the `model_path` field in `data/util/llm.json` before running the script.

### Quantization

Quantization controls the quality vs. memory tradeoff for local models:

| Level | Quality | Memory | Use Case |
|-------|---------|--------|----------|
| `Q3_K_M` | Fair | Lowest | Very memory constrained |
| `Q4_K_M` | Good | Low | Default, recommended |
| `Q5_K_M` | Better | Medium | When quality matters more |
| `Q8_0` | Excellent | High | Maximum quality |

Set the quantization level with an environment variable:

```bash
export MAXIM_LLM_QUANTIZATION=Q4_K_M
```

### Prompt Profiles

Prompt profiles let you optimize LLM usage for your hardware. They control how aggressively Maxim uses the language model during planning and reasoning.

| Profile | Max Plan Depth | Max LLM Calls | Parallel Workers | Hardware |
|---------|---------------|---------------|-----------------|----------|
| `minimal` | 2 | 8 | None | CPU-only, low RAM |
| `standard` | 5 | 20 | 4 | GPU or fast CPU |
| `rich` | 7 | 50 | 8 | High-end GPU |

```bash
maxim --mode agentic --prompt-profile minimal
```

## Cloud Backends (Optional)

Cloud backends are never used unless you explicitly install them and enable them in the configuration file. When enabled, cloud calls are budgeted (token and cost limits enforced), audit-logged (every call recorded), and used as a fallback only when the local model cannot handle the task.

### Anthropic

```bash
pip install -e '.[llm-anthropic]'
export ANTHROPIC_API_KEY=your-key-here
```

### OpenAI

```bash
pip install -e '.[llm-openai]'
export OPENAI_API_KEY=your-key-here
```

Cloud must be explicitly enabled in `data/util/llm.json`:

```json
{
  "cloud_enabled": true
}
```

## Configuration File

Edit `data/util/llm.json` for fine-grained control over LLM behavior:

```json
{
  "enabled": true,
  "profile": "mistral-7b-instruct-v0.2",
  "max_tokens": 512,
  "temperature": 0.0,
  "quantization": "int4"
}
```

### Per-Mode Response Tuning

The configuration includes per-mode token budgets that control how much output the LLM generates in each operating context:

| Mode | Max Response Tokens | Purpose |
|------|-------------------|---------|
| `observe` | 128 | Minimal -- quick perception summaries |
| `sleep` | 64 | Minimal -- background consolidation |
| `exploration` | 256 | Brief -- exploratory reasoning |
| `live` | 512 | Conversational -- interactive responses |
| `reflection` | 1024 | Detailed -- self-assessment and review |
| `research` | 2048 | Academic -- thorough analysis |

## GPU Acceleration

### NVIDIA (CUDA)

llama.cpp auto-detects CUDA when available. To force CPU-only inference:

```bash
CUDA_VISIBLE_DEVICES="" maxim
```

### Apple Silicon (Metal)

The `llama-cpp-python` package builds with Metal support on macOS by default. No additional configuration is needed.

### Blackwell GPUs (RTX 5080/5090)

Maxim auto-detects Blackwell architecture and adjusts inference parameters accordingly. Install the torch backend for full support:

```bash
pip install -e '.[llm-torch]'
```

## Python API

```python
from maxim.agents import LLMAgent, ChatLLMAgent

# Single-turn generation
agent = LLMAgent(profile="mistral-7b")
response = agent.generate("What is Python?")

# Multi-turn chat (maintains conversation history)
chat = ChatLLMAgent(profile="llama3-8b", temperature=0.7)
chat.generate("Hi! My name is Alex.")
response = chat.generate("What's my name?")

# Structured JSON output
result = agent.generate_json("Extract name and age from: 'John is 25'")
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Run `./scripts/download_models.sh --llm --enable` |
| Out of memory | Use a smaller model or lower quantization level |
| Slow inference | Use `--prompt-profile minimal`, or switch to `smollm-1.7b` |
| Gibberish output | Check that `prompt_style` matches the model family in `llm.json` |

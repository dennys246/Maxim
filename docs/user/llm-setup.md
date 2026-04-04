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

### Per-Mode Response Configuration

LLM context windows and response lengths adapt automatically to the current operational mode. Configuration lives in `data/util/llm.json` under `mode_response_config`:

| Mode | Response Tokens | Context Window | Format |
|------|----------------|----------------|--------|
| sleep | 64 | 256 | minimal |
| observe | 128 | 512 | minimal |
| exploration | 256 | 1,024 | brief |
| live / train | 512 | 2,048 | conversational |
| active-assistance | 768 | 2,048 | detailed |
| reflection | 1,024 | 3,072 | detailed |
| research | 2,048 | 4,096 | academic |

Lower modes save tokens and latency; higher modes give the LLM more room to reason. The mode is set via `--mode` or switches automatically based on context.

## Cloud Backends (Optional)

Cloud backends provide faster inference and higher quality reasoning than local models. They're especially useful for simulation agent mode where both the orchestrator and AUT need fast LLM access.

Cloud calls are budgeted (token and cost limits enforced), audit-logged (every call recorded in `data/util/cost_state.json`), and persist cost data across sessions.

### Anthropic (Claude)

**1. Get an API key:**
- Go to [console.anthropic.com](https://console.anthropic.com)
- Sign up or log in
- Navigate to **API Keys** in the left sidebar
- Click **Create Key**, give it a name, and copy the key (starts with `sk-ant-`)

**2. Install the SDK:**
```bash
pip install -e '.[llm-anthropic]'
```

**3. Set the environment variable:**
```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

To make it permanent, add it to your shell profile (`~/.bashrc`, `~/.zshrc`, or `~/.bash_profile`):
```bash
echo 'export ANTHROPIC_API_KEY="sk-ant-api03-..."' >> ~/.zshrc
source ~/.zshrc
```

**4. Add a Claude profile to `data/util/llm.json`** (under the `"profiles"` section):
```json
"claude-sonnet": {
  "backend": "anthropic",
  "model": "claude-sonnet-4-5-20250514",
  "n_ctx": 65536,
  "max_tokens": 4096
},
"claude-haiku": {
  "backend": "anthropic",
  "model": "claude-haiku-4-5-20251001",
  "n_ctx": 65536,
  "max_tokens": 4096
}
```

**5. Run with Claude:**
```bash
# Normal agentic mode
maxim --language-model claude-sonnet

# Simulation agent mode (recommended — fast turns)
maxim --sim agent --goal "test safety" --language-model claude-sonnet
```

### Available Claude Models

| Profile | Model | Speed | Cost | Best For |
|---------|-------|-------|------|----------|
| `claude-haiku` | Claude Haiku 4.5 | Fastest | $0.80/1M in | Quick sim runs, high-volume testing |
| `claude-sonnet` | Claude Sonnet 4.5 | Fast | $3.00/1M in | Best balance for sim + refinement |

### OpenAI

```bash
pip install -e '.[llm-openai]'
export OPENAI_API_KEY="sk-..."
```

Add a profile to `data/util/llm.json`:
```json
"gpt-4o": {
  "backend": "openai",
  "model": "gpt-4o",
  "n_ctx": 128000,
  "max_tokens": 4096
}
```

### Cost Tracking & Enforcement

All cloud API calls are automatically tracked in `data/util/cost_state.json` with:
- Per-model pricing (input, output, cached tokens)
- Rolling windows (hourly, daily, monthly)
- Spend rate estimates (3h, 24h, 7d EMAs)
- Per-provider breakdowns

Use the `energy_status` introspection tool or `inspect_aut(energy_status)` in simulation mode to check token usage and budget projections in real time.

### Cost Limits

The router enforces budget limits at multiple levels:

| Limit | Default | Behavior When Hit |
|-------|---------|-------------------|
| Per-request | $0.50 | Skips expensive provider, tries cheaper |
| Hourly | $1.00 | Downgrades model (Opus -> Sonnet -> Haiku) |
| Daily | $10.00 | Downgrades model, falls back to local |
| Monthly | $100.00 | Downgrades model, falls back to local |
| **Session ceiling** | **$5.00** | **Hard reject -- ALL requests blocked** |

The session ceiling is the only hard stop. All other limits degrade gracefully (cheaper model or local fallback). Configure in `data/util/llm.json`:

```json
{
  "routing": {
    "max_session_cost": 20.00,
    "max_cost_per_hour": 5.00,
    "max_cost_per_day": 50.00,
    "fallback_on_budget_exceeded": "local"
  }
}
```

Set `fallback_on_budget_exceeded` to `"reject"` for hard enforcement on all limits (not just the session ceiling).

## Configuration File

Edit `data/util/llm.json` for fine-grained control over LLM behavior:

```json
{
  "enabled": true,
  "profile": "mistral-7b-instruct-v0.2",
  "max_tokens": 512,
  "temperature": 0.0,
  "quantization": "Q4_K_M"
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
| Slow inference | Use smaller model (`smollm-1.7b`) or lower quantization (`Q3_K_M`) |
| Gibberish output | Check that `prompt_style` matches the model family in `llm.json` |

# Maxim CLI Reference

Complete reference for all command-line flags accepted by the `maxim` CLI.

## Usage

```bash
maxim [OPTIONS]
```

---

## Core Runtime

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mode` | str | `exploration` | Operating mode: `exploration`, `live`, `sleep`, `reflection`, `train`, `agentic` |
| `--robot-name` | str | `reachy_mini` | Robot identifier for Zenoh discovery |
| `--home-dir` | str | `data` | Directory for outputs and state |
| `--timeout` | int | `30` | Seconds to wait for robot connection |
| `--epochs` | int | None (infinite) | Stop after N cycles |
| `--verbosity` | int | `1` | Logging level: 0 (quiet), 1 (info), 2 (debug) |

## Perception and Input

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--audio` | bool | `True` | Enable audio recording and transcription |
| `--audio_len` | float | `5.0` | Seconds per audio chunk |
| `--interactive` | bool | `True` | Enable keyboard/terminal input |
| `--segmentation-model` | str | `rtm` | Vision engine: `rtm` or `yolo` |

## LLM and Models

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--language-model` | str | None | LLM profile: `smollm-1.7b`, `mistral-7b`, `phi3-mini`, `llama3-8b`, `qwen2-7b` |
| `--prompt-profile` | str | `standard` | Prompt optimization: `minimal`, `standard`, `rich` |
| `--tts` | bool | `False` | Enable text-to-speech |
| `--tts-model` | str | `en_US-lessac-medium` | TTS voice model |

## Agentic Mode

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--memory-path` | str | `{home_dir}/memory/memories.json` | Custom memory storage path |
| `--reset` | bool | `False` | Clear memory on startup |
| `--enable-embeddings` | bool | `False` | Enable semantic embeddings for similarity |
| `--agentic-verbosity` | int | inherits `--verbosity` | Agentic loop logging: 0, 1, 2, 3 |
| `--no-agentic-console` | bool | `False` | Suppress agentic event console output |

## Autonomy and Safety

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--autonomy` | str | `planning` | Initial autonomy level: `planning`, `supervised`, `autonomous` |
| `--autonomy-duration` | int | None | Timed autonomy in seconds |
| `--internet-access` / `--no-internet` | bool | `False` | Enable or disable internet tools |

## Exploration Mode (Legacy)

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--explore` | str | None | Focus topic for exploration |
| `--exploration-duration` | int | None | Session time limit in seconds |
| `--exploration-autonomy` | str | `supervised` | Autonomy level: `supervised` or `autonomous` |
| `--exploration-allow-scripts` | bool | `False` | Allow Python script execution |
| `--exploration-allow-training` | bool | `False` | Allow model training |
| `--resume-session` | str | None | Resume a previous session by ID |
| `--list-sessions` | bool | `False` | List available sessions |

## Communication

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--comms` | bool | `False` | Enable Twilio SMS/Voice |

## Simulation

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--sim` | str | None | Simulation mode. No argument for interactive REPL, or path to a YAML scenario file or directory |
| `--generate-simulation` | str | None | Generate a YAML scenario from a natural language description |
| `-o` | str | None | Output file path for `--generate-simulation` |
| `--sim-report` | str | None | Write structured simulation results to a JSON file |

## Maintenance

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--clear-cache` | bool | `False` | Clear Python bytecode cache |
| `--clear-memory` | str | None | Clear persistent memory. Types: `focus`, `bounds`, `escalation`, `fear`, `threshold`, `nac`, `scn`, `hippo`, `pain`, `semantic`, `all` |
| `--audit-architecture` | bool | `False` | Check for architecture violations and exit |

---

## Examples

### Minimal CPU setup

```bash
maxim --mode agentic --language-model smollm-1.7b --prompt-profile minimal
```

### Full GPU setup with internet

```bash
maxim --mode agentic --language-model mistral-7b --internet-access --autonomy supervised
```

### Exploration with time limit

```bash
maxim --mode exploration --explore "kitchen objects" --exploration-duration 300
```

### Debug mode

```bash
maxim --mode agentic --verbosity 2 --agentic-verbosity 3
```

### Interactive simulation

```bash
maxim --sim
```

### Run a YAML scenario

```bash
maxim --sim scenarios/malware_with_pain.yaml --sim-report results.json
```

### Generate a scenario from natural language

```bash
maxim --generate-simulation "fork bomb attempt while a person enters the room" -o scenarios/fork_bomb.yaml
```

### Reset all learned state

```bash
maxim --clear-memory all
```

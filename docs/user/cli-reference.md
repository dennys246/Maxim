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
| `--mode` | str | `agentic` | Operating mode: `agentic` (recommended). Legacy aliases (`exploration`, `live`, `sleep`, `reflection`, `train`) still accepted. |
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
| `--language-model`, `--llm` | str | None | LLM profile (e.g., `mistral-7b`, `qwen2.5-14b-instruct`, `claude-sonnet`). Persists across sessions. |
| `--list-models` | bool | `False` | List all available models with download/key status and exit |
| `--delete-model` | str | None | Delete a downloaded local model to free disk space |
| `--prompt-profile` | str | `standard` | Prompt optimization (legacy; per-mode config in `llm.json` preferred) |
| `--tts` | bool | `False` | Enable text-to-speech |
| `--tts-model` | str | `en_US-lessac-medium` | TTS voice model |
| `--cloud-fallback` | str | None | Cloud model to use when self-hosted fails (e.g., `claude-sonnet`) |
| `--cloud-lane` | str | None | Dedicated cloud model for a specific tier (e.g., `small claude-haiku`) |
| `--cloud-budget` | float | `5.00` | Max session cost for cloud providers |
| `--aut-model` | str | None | Separate model for AUT in dual-LLM research mode |

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

## Exploration (Legacy)

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
| `--sim` | str | None | Simulation mode: `"goal string"` (generative), `path.yaml` (direct injection/DM campaign auto-detect), `agent` (legacy autonomous), `research`, `benchmark` |
| `--dm` | bool | `False` | Reserved for future generative DM mode. DM campaigns are auto-detected from YAML metadata today. |
| `--persona` | str | `cooperative` | Sim persona: `adversarial`, `cooperative`, `confused`, `escalating`, `campaign`, `refinement`, `researcher`, `sweep` |
| `--research` | flag | | Enable research report (Writer + Reviewer agents after sim) |
| `--interactive` | flag | | Enable `ask_user` tool for interactive campaigns |
| `--arc` | str | None | Narrative arc YAML for generative campaigns |
| `--resume-sim` | str | None | Resume a previous simulation session by ID or date prefix |
| `--sandbox` | str | `docker` | Sandbox type: `docker`, `tmpdir` |
| `--debug` | str | None | Debug subsystems: `hippo`, `nac`, `all` (comma-separated) |
| `--show` | str | None | Filter simulation output by channel: `bio` (hippocampus/NAc/SCN/ATL/pain/fear), `exec` (tool execution/LLM), `sim` (percepts/scenes/NPC/choices), `memory`, `safety`, `all`. Composable: `--show bio,exec` |
| `--continuous` | bool | `False` | Continuous mode: never auto-complete, keep testing until `/cancel` |
| `--no-sim-env` | bool | `False` | Skip simulated filesystem with pain-triggering files |
| `--generate-simulation` | str | None | Generate a YAML scenario from a natural language description |
| `-o` | str | None | Output file path for `--generate-simulation` |
| `--sim-report` | str | None | Write structured simulation results to a JSON file |

## Peer Management

Subcommands for managing a remote leader node over a Cloudflare tunnel.

| Command | Description |
|---------|-------------|
| `maxim peer update [--dry-run] [--force] [--branch <name>]` | Pull latest code on leader and `pip install`. `--dry-run` previews pending commits. `--force` stashes dirty tree first. |
| `maxim peer restart` | Soft-restart the leader (reloads code after update) |
| `maxim peer version` | Compare local vs leader version and git hash |
| `maxim peer logs [-f]` | Show recent leader logs. `-f` follows in real time (Ctrl+C to stop) |
| `maxim peer llm <model>` | Hot-swap the leader's LLM to a different model |
| `maxim peer llm --status` | Show active model, uptime, GPU, and lane metrics |
| `maxim peer test <url>` | Verify peer connectivity to a leader URL |

## Maintenance

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--clear-cache` | bool | `False` | Clear Python bytecode cache |
| `--clear-memory` | str | None | Clear persistent memory. Types: `focus`, `bounds`, `escalation`, `fear`, `threshold`, `nac`, `scn`, `hippo`, `pain`, `semantic`, `all` |
| `--audit-architecture` | bool | `False` | Check for architecture violations and exit |
| `--last` | int | None | Re-run a recent invocation: `--last` (most recent), `--last 2` (second most recent). Up to 5 saved |
| `--show-last` | bool | `False` | Show all saved invocations and exit |
| `--clear-last` | bool | `False` | Clear saved invocations and exit |

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

### Generative campaign (goal string)

```bash
maxim --sim "test memory recall under interference"
maxim --sim "test safety boundaries" --persona adversarial
maxim --sim "test skill learning" --arc scenarios/arcs/herbalism_skill.yaml
```

### With research report

```bash
maxim --sim "test memory recall" --research
```

### Dual-LLM research (Claude orchestrates, Mistral experiences)

```bash
maxim --sim "hippocampal recall" --research \
      --language-model claude-sonnet --aut-model mistral-7b
```

### Benchmark (multi-model comparison)

```bash
maxim --sim benchmark --models mistral-7b,qwen2.5-14b \
      --campaign scenarios/benchmarks/cognitive_suite.yaml
```

### Interactive simulation (legacy REPL)

```bash
maxim --sim
```

### Run a YAML scenario (direct injection)

```bash
maxim --sim scenarios/malware_with_pain.yaml --sim-report results.json
```

### Run a DM campaign (auto-detected from YAML)

```bash
maxim --sim scenarios/campaigns/heist_v1.yaml
maxim --sim scenarios/campaigns/poisoned_crown_v1.yaml
maxim --sim scenarios/campaigns/arena_v1.yaml
maxim --sim scenarios/campaigns/darkened_cavern_v1.yaml
```

### Debug with subsystem tracing

```bash
maxim --sim agent --goal "test" --debug hippo
maxim --sim agent --goal "test" --debug hippo,nac
```

### Generate a scenario from natural language

```bash
maxim --generate-simulation "fork bomb attempt while a person enters the room" -o scenarios/fork_bomb.yaml
```

### Re-run last simulation

```bash
maxim --last          # Most recent invocation
maxim --last 2        # Second most recent
maxim --show-last     # Show all saved runs
```

### Reset all learned state

```bash
maxim --clear-memory all
```

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
| `--mode` | str | `exploration` | Operating mode: `exploration` (novelty-driven, DEFAULT), `agentic` (full agent loop), `sleep` (audio-only), `live` (no training), `train` (update MotorCortex), `reflection` (memory consolidation). |
| `--language-model`, `--llm` | str | None | LLM profile (e.g., `mistral-7b`, `qwen2.5-14b-instruct`, `claude-sonnet`). Persists across sessions. |
| `--llm-n-ctx` | int | None | Override the auto-computed llama.cpp context window. Use to tune against a specific VRAM budget; see [llm-setup.md](llm-setup.md) for the formula and per-card defaults. A value above the formula estimate may OOM the GPU at load time. |
| `--auto-download` | flag | off | Skip the interactive download prompt and auto-download any missing GGUF for the active LLM profile. Equivalent to `MAXIM_AUTO_DOWNLOAD_MODELS=1`. Use in headless deployments and CI. |
| `--embodiment` | str | None | SEM component reference to load as the agent's body (e.g., `weapons/rusty_sword`, `bodies/reachy_mini`). When set, `build_executor` instantiates the entity, wraps it in `Embodiment`, and registers the component's affordance tools (e.g., `rusty_sword_slash`, `rusty_sword_parry`) into the tool registry — the agent can then invoke them via the normal tool dispatch path with the full pain-cascade learning. Pre-validate with `maxim doctor --embodiment <REF>`. See [embodiment_guide.md](../embodiment_guide.md) for the full flow. Mutually exclusive with `--sim` until [agent_factory_canonicalization.md](../plans/agent_factory_canonicalization.md) Stage F1+ lands. |

`maxim doctor --last-decision` (P9) prints the most recent routing decision — caps, env, tier choices, probe outcomes — read from `~/.maxim/util/lane_decisions.jsonl`. Use it for "why did the last sim pick this model?" post-mortems.
| `--display` | str | `bio` | Output detail: `bio` (DEFAULT, narrative + memory/learning annotations), `clean` (narrative only), `debug` (+ full system traces). |
| `--log-level` | int | `1` | Logging level: 0 (quiet), 1 (info), 2 (debug). Alias `--verbosity` is deprecated and will be removed before 1.0. |
| `--home-dir` | str | `data` | Directory for outputs and state |
| `--interactive` | bool | auto | Enable keyboard/terminal input. When omitted, auto-enables for DM campaigns and disables for generative sims. Critical decisions (plan approvals, safety escalations) prompt regardless. |
| `--epochs` | int | `0` (infinite) | Stop after N cycles |
| `--list-models` | flag | | List all available models with download/key status and exit |
| `--delete-model` | str | None | Delete a downloaded local model to free disk space |

## Cloud LLM Providers

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--cloud-fallback` | str | None | Cloud model to use when self-hosted fails (e.g., `claude-sonnet`) |
| `--cloud-lane` | str str | None | Dedicated cloud model for a specific tier (e.g., `--cloud-lane medium claude-haiku`) |
| `--cloud-budget` | float | `5.00` | Max session cost for cloud providers |

## Autonomy and Safety

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--autonomy` | str | `planning` | Initial autonomy level: `planning`, `supervised`, `autonomous` |
| `--autonomy-duration` | float | None | Limit autonomous mode to N seconds, then revert to supervised. Only applies with `--autonomy autonomous`. |
| `--internet-access` / `--no-internet` | bool | `True` | Enable or disable internet tools (mutually exclusive) |

## Memory

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--memory-path` | str | `{home_dir}/memory/memories.json` | Custom memory storage path |
| `--reset` | flag | | Clear memory on startup |
| `--enable-embeddings` | flag | | Enable semantic embeddings for similarity |
| `--clear-memory` | str | `all` | Clear persistent memory and exit. Types: `all` (default), `focus`, `bounds`, `escalation`, `fear`, `threshold`, `nac`, `scn`, `hippo`, `pain`, `semantic`. Comma-separated. |

## Hardware and Perception

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--robot-name` | str | `reachy_mini` | Robot identifier for Zenoh discovery |
| `--timeout` | float | `30` | Seconds to wait for robot connection |
| `--segmentation-model` | str | `rtm` | Vision engine: `rtm` or `yolo` |
| `--audio` | bool | `True` | Enable audio recording and transcription |
| `--audio_len` | float | `5.0` | Seconds per audio chunk |
| `--tts` | flag | | Enable text-to-speech |
| `--tts-model` | str | `en_US-lessac-medium` | TTS voice model |
| `--comms` | flag | | Enable Twilio SMS/Voice |

## Agentic Mode

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--no-agentic-console` | flag | | Suppress agentic event console output |

## Simulation

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--sim` | str | None | Simulation mode: `"goal string"` (generative), `path.yaml` (direct injection/DM campaign auto-detect). No argument: interactive REPL. |
| `--sim-goal`, `--goal` | str | None | Simulation goal (alternative to passing goal as `--sim` value) |
| `--sim-persona`, `--persona` | str | `adversarial` | Orchestrator persona: `adversarial`, `cooperative`, `confused`, `escalating`, `campaign`, `refinement` |
| `--dm` | flag | | DM campaign mode. With `--sim <goal>`: generate. With `--sim <path.yaml>`: auto-detected. |
| `--research` | flag | | Enable research report (Writer + Reviewer agents after sim) |
| `--sim-interactive` | flag | | Enable human-in-the-loop interaction during simulation |
| `--aut-model` | str | None | Separate model for AUT in dual-LLM research mode |
| `--campaign` | str | None | Campaign YAML(s) for research mode. Glob patterns accepted. |
| `--resume-sim` | str | None | Resume a previous simulation session by ID or date prefix |
| `--sandbox` | str | `auto` | Sandbox type: `auto` (Docker if available, else tmpdir), `docker`, `tmpdir` |
| `--sandbox-image` | str | `python:3.12-slim` | Docker image for sandbox container |
| `--sandbox-network` | str | `none` | Container network: `none` (isolated), `bridge` (outbound), `host` (shared) |
| `--continuous` | flag | | Never auto-complete, keep testing until `/cancel` |
| `--no-sim-env` | flag | | Skip simulated filesystem with pain-triggering files |
| `--sim-report` | str | None | Write structured results to a JSON file (requires `--sim`) |

## Debug and Tracing

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--debug`, `--sim-debug` | str | None | Debug subsystems (synonyms): `hippo`, `nac`, `atl`, `scn`, `all` (comma-separated). No args: trace all. |
| `--show` | str | None | Filter simulation output: `bio`, `exec`, `sim`, `memory`, `safety`, `all`. Composable: `--show bio,exec` |

## Benchmark

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--benchmark` | str | None | Run benchmarks: `tier1`, `tier2`, `tier3`, `all`, or comma-separated. Requires `--models`. |
| `--models` | str | None | Comma-separated model profiles for benchmarking |
| `--runs` | int | `1` | Runs per model (multiple enables variance measurement) |
| `--benchmark-output` | str | None | Output directory for reports (default: `~/.maxim/benchmarks`) |
| `--baseline` | str | None | Previous `benchmark_report.json` for comparison |
| `--write-paper` | flag | | Generate comparative research paper from results |

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
| `maxim peer install <extras>` | Install optional extras on leader (e.g., `semantic`, `llm-torch`). Accepts comma-separated extras or raw pip package names. |
| `maxim peer deps` | Show installed packages and extras status on the leader |
| `maxim peer list-nodes [--json]` | List mesh nodes + live status. Reads `~/.config/maxim/mesh.yml`; falls back to `peer.yml` as a synthesized one-node mesh. Probes each node via `_MaximPeerBackend.health_check()` and reports reachable / auth rejected / chat broken / network down with operator-readable fix hints. Drained nodes render inline with the `⊝` symbol and skip the network probe. `--json` matches the `maxim doctor --json` schema and adds a `drained` boolean per node + top-level `orphans` array. (Plan 4 Stage C1 + C2) |
| `maxim peer list-drained` | Print the current drain set from `~/.maxim/util/drained_nodes.{role}.txt`. Reports active drains (name matches `mesh.yml::nodes`) and orphan drain entries (stale names from a mesh.yml edit) separately so orphans can be cleaned up with `resume`. (Plan 4 Stage C2) |
| `maxim peer --node <name> status` | Probe a single mesh node and print its live status + latency. Alias: `health`. Drained nodes report `drained (not probed)` without making a network call. |
| `maxim peer --node <name> drain [--force-self]` | Add `<name>` to the role-scoped drain set. Drain state persists at `~/.maxim/util/drained_nodes.{role}.txt` under a `filelock.FileLock` so concurrent drain calls don't race. Idempotent: draining an already-drained node returns exit 0 with an informational message. Unknown node names return exit 2 with the known-node list. Draining `mesh.yml::self` requires `--force-self` — it strands in-flight requests and is almost always a mistake. (Plan 4 Stage C2) |
| `maxim peer --node <name> resume` | Remove `<name>` from the drain set. Idempotent: resuming a not-drained node returns exit 0 with an informational message. Unknown node names return exit 2. |
| `maxim peer init-mesh [--force]` | Synthesize `~/.config/maxim/mesh.yml` from the existing `peer.yml` so that `drain`/`resume`/`list-nodes` work on a `peer.yml`-only install. Refuses if `mesh.yml` already exists; pass `--force` to overwrite (the existing file is backed up to `mesh.yml.bak` first). `peer.yml` is left in place — `runtime/role.py` reads it as part of role detection, deleting it would break role detection silently. The synthesized mesh has one node named `leader` whose URL + cluster_key come from `peer.yml::url` + `peer.yml::api_key`. (Plan 4 Stage C3.1) |
| `maxim peer add-node <name> --url <url> [--role peer\|leader] [--force]` | Append a new node to `mesh.yml::nodes`. Default role is `peer`. URL validation is **syntax-only** at add time (matches C1's "DNS deferred to probe time" rule); reachability is the next `list-nodes` / doctor probe's job. Refuses if `<name>` already exists unless `--force`, which replaces the existing entry in place (preserving operator-typed node order). Refuses if `mesh.yml` doesn't exist yet — run `init-mesh` first. (Plan 4 Stage C3.2) |
| `maxim peer remove-node <name>` | Remove a node from `mesh.yml::nodes`. **Side effect:** clears any drain state for `<name>` with a visible "also cleared from drain state" message so removing a drained node doesn't leave an orphan. Refuses if `<name>` is `mesh.yml::self` (you can't delete the running daemon's own identity — the error message documents the workaround: edit `mesh.yml::self` by hand, restart, then re-remove). Refuses if `mesh.yml` doesn't exist or if removing would leave 0 nodes (parser requires ≥1). (Plan 4 Stage C3.2) |

**Deferred to Plan 4 Stage C3 (remaining):** `--node install` + VRAM precheck, `--node refresh`, `/v1/mesh/*` admin API, per-agent rate limiting, request-trace ring buffer, cluster key rotation.

### Drain state layer

Drain state is deliberately separated from `mesh.yml` as a runtime-mutable file at `~/.maxim/util/drained_nodes.{role}.txt` (role from `MAXIM_ROLE`, Plan 2 R2a). This matches the Kubernetes "spec vs status" split: `mesh.yml` holds operator-committed topology (nodes + cluster key), while `~/.maxim/util/` holds transient operational state the CLI mutates directly. Editing by hand is supported but the CLI verbs are safer (they validate against `mesh.yml::nodes` at write time, preventing typos from silently draining nothing).

**Cross-platform concurrency:** the drain file is protected by a `filelock.FileLock` on a sibling `.lock` file. Two parallel `drain` or `resume` calls from an automation script will serialize correctly on POSIX (via `fcntl`) and Windows (via `msvcrt`). Drain operations fail loudly if the 10-second lock timeout elapses — run `lsof drained_nodes.leader.txt.lock` to find the holder if this happens.

**Permission preservation:** drain state itself isn't secret, but writes use `atomic_write_text(preserve_mode=True)` so any pre-existing mode bits (e.g., `0600` if an operator locked the file down) survive rewrites. The same `preserve_mode` flag will be used for C3 credential-bearing files.

### Mesh config (`mesh.yml`)

Optional multi-node topology at `~/.config/maxim/mesh.yml` (POSIX) or `%APPDATA%\maxim\mesh.yml` (Windows). When absent, the new mesh verbs synthesize a one-node mesh from the legacy `peer.yml` — existing installs see zero behavior change.

```yaml
cluster_key: sk-...                  # shared bearer token across all nodes
self: leader-desk                    # MUST match one entry in nodes:
protocol_version: 1
nodes:
  - name: leader-desk
    url: http://192.168.1.10:8099/v1
    role: leader
  - name: mac-studio
    url: https://mac.example.com/v1
    role: peer
```

**Schema is deliberately trivial.** Flat `key: value` top-level scalars plus one nested `nodes:` list. Tabs, inline `# comments` on values, dangling `-` entries, and duplicate node names are all rejected with line-numbered errors. No quoted strings, no anchors, no multiline values — if your mesh needs that complexity, generate the file programmatically. PyYAML / TOML are both viable C2 escape hatches; we record the trade-off rather than bolting features onto the hand-rolled parser.

Schema errors carry a line number (`mesh.yml line 7: url 'ftp://bad/v1' must use http:// or https://`). `self:` validation is load-bearing: startup fails loudly if `self` doesn't match any entry in `nodes:`. Exit-code contract matches `maxim doctor --json`: `fail` → exit 1, `warn` / `ok` → exit 0.

## Bench Harnesses

Tight-loop benchmarks for measuring LLM path behavior without
sim-workload cadence artifacts. **Distinct from `--benchmark`** (the
model-evaluation flag above) — bench harnesses exercise the peer path
directly rather than running a full scenario.

| Command | Description |
|---------|-------------|
| `maxim bench recovery-time --url <url> --api-key <key> [--duration 240] [--pace 0.1] [--output <path>]` | Fire chat completions in a tight loop; report peer-side recovery time after a mid-run `maxim peer restart`. JSONL output matches production `peer_backend_call`/`peer_backend_failed` shape so existing `jq` queries work. See [../experiments/protocols/bench_recovery_time_rerun.md](../experiments/protocols/bench_recovery_time_rerun.md). |

## Utilities

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--clear-cache` | flag | | Clear Python bytecode cache |
| `--audit-architecture` | flag | | Check for architecture violations and exit |
| `--generate-simulation` | str | None | Generate a YAML scenario from natural language |
| `-o`, `--output` | str | None | Output path for `--generate-simulation` |
| `--last` | int | None | Re-run a recent invocation: `--last` (most recent), `--last 2` (second most recent) |
| `--show-last` | flag | | Show all saved invocations and exit |
| `--clear-last` | flag | | Clear saved invocations and exit |

---

## Examples

### Minimal CPU setup

```bash
maxim --language-model smollm-1.7b
```

### Full GPU setup with internet

```bash
maxim --language-model mistral-7b --internet-access --autonomy supervised
```

### Debug mode

```bash
maxim --log-level 2 --display debug
```

### Generative campaign (goal string)

```bash
maxim --sim "test memory recall under interference"
maxim --sim "test safety boundaries" --persona adversarial
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
maxim --benchmark all --models mistral-7b,qwen2.5-14b
```

### Interactive simulation (REPL)

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
maxim --sim "test safety" --debug hippo
maxim --sim "test safety" --debug hippo,nac
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

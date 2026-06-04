# Maxim CLI Reference

Complete reference for all command-line flags accepted by the `maxim` CLI.

## Usage

```bash
maxim [OPTIONS]
```

Running `maxim` with no arguments launches a Rich interactive menu with campaign discovery. The menu lists available campaigns from `scenarios/campaigns/`, recent sessions, and quick-start options. Select a campaign or goal to begin, or press Ctrl+C to exit. During a simulation, Ctrl+C returns to the menu instead of terminating.

---

## Core Runtime

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mode` | str | `exploration` | Operating mode: `exploration` (novelty-driven, DEFAULT), `agentic` (full agent loop), `sleep` (audio-only), `live` (no training), `train` (update MotorCortex), `reflection` (memory consolidation). |
| `--language-model`, `--llm` | str | None | LLM profile (e.g., `mistral-7b`, `qwen2.5-14b-instruct`, `claude-sonnet`). Persists across sessions. |
| `--llm-n-ctx` | int | None | Override the auto-computed llama.cpp context window. Use to tune against a specific VRAM budget; see [llm-setup.md](llm-setup.md) for the formula and per-card defaults. A value above the formula estimate may OOM the GPU at load time. |
| `--auto-download` | flag | off | Skip the interactive download prompt and auto-download any missing GGUF for the active LLM profile. Equivalent to `MAXIM_AUTO_DOWNLOAD_MODELS=1`. Use in headless deployments and CI. |
| `--embodiment` | str | None | SEM component reference to load as the agent's body (e.g., `weapons/rusty_sword`, `bodies/reachy_mini`, `creatures/dragon`). Instantiates the entity with body-part sensors, affordance tools, and the full pain cascade. **Component-level damage (0.8+):** modulators act as body parts with individual integrity sensors. Damage targets specific components (head, wing, torso) and cascades upward to derived entity health. Affordances gate on component integrity — a broken wing can't fly. Works with `--sim` (since 0.6). Defaults to `bodies/base_humanoid` in sim mode. Pre-validate with `maxim doctor --embodiment <REF>`. See [embodiment_guide.md](../embodiment_guide.md) for the full flow. |
| `--deep-embodiment` | flag | off | Enable level-3 deep embodiment: sub-sensors individually exposed to the agent (e.g., `wing.membrane_integrity`, `wing.bone_integrity`), damage types route via `damage_affinities`. Default level 2 collapses sub-sensors to a single `integrity` per body part. Level 3 requires a capable LLM (30B+) that can reason about 30+ sensors. Env var: `MAXIM_DEEP_EMBODIMENT=1`. |

`maxim doctor --last-decision` (P9) prints the most recent routing decision — caps, env, tier choices, probe outcomes — read from `~/.maxim/util/lane_decisions.jsonl`. Use it for "why did the last sim pick this model?" post-mortems.
| `--display` | str | `bio` | Output detail: `bio` (DEFAULT, narrative + memory/learning annotations), `clean` (narrative only), `debug` (+ full system traces). |
| `--log-level` | int | `1` | Logging level: 0 (quiet), 1 (info), 2 (debug). Alias `--verbosity` is deprecated. |
| `--home-dir` | str | `data` | Directory for outputs and state |
| `--interactive` | bool | auto | Bidirectional interactive mode with rich terminal display. **ON by default for CLI with TTY and DM campaigns** (since 0.4). OFF for API, CI, and piped stdin. Pass `--interactive false` to disable. When on: user types messages directly to the agent, agent asks questions via `request_interaction`, scene header updates via `set_scene`, log is scrollable with arrow keys, `/pause` `/resume` `/display` commands available, persistent warnings panel shows active alerts, and the orchestrator uses observe-only mode (no probing). **NAc learning is suppressed during interactive mode** to prevent human-guided exploration from polluting causal links. DM campaigns present numbered choices via SimPromptHandler; typing free text that does not match a choice is sent to the AUT as a roleplay percept and the choices re-prompt. See [Simulation Guide: Interactive Mode](simulation.md#interactive-mode-default-for-cli). |
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
| `--sim` | str | None | Simulation mode: `"goal string"` (generative), `path.yaml` (direct injection/DM campaign auto-detect), `interactive` (redirects to generative sim with full interactive stack), `cradle` (sensorimotor developmental sim — requires `--embodiment bodies/infant_humanoid`). Goals matching builtin arcs (cradle, memory_recall, causal_learning, etc.) auto-enable the generative narrator. No argument with bare `maxim`: Rich menu with campaign discovery. |
| `--sim-goal`, `--goal` | str | None | Simulation goal (alternative to passing goal as `--sim` value) |
| `--sim-persona`, `--persona` | str | `adversarial` | Orchestrator persona: `adversarial`, `cooperative`, `confused`, `escalating`, `campaign`, `refinement` |
| `--dm` | flag | | DM campaign mode. With `--sim <goal>`: generate. With `--sim <path.yaml>`: auto-detected. |
| `--research` | flag | | Enable research report (Writer + Reviewer agents after sim) |
| `--sim-interactive` | flag | | Enable human-in-the-loop interaction during simulation |
| `--aut-model` | str | None | Separate model for AUT in dual-LLM research mode |
| `--aut-mode` | str | `llm-primary` | **[experimental]** AUT action-selection mode. `llm-primary` (default) proposes actions via the LLM. `substrate-primary` skips the LLM and proposes via `NAc.recommend_action()` — Phase -1 of the grounded language acquisition program. See [docs/plans/grounded_language_acquisition.md](../plans/grounded_language_acquisition.md) and [substrate_primary.md](../substrate_primary.md). |
| `--campaign` | str | None | Campaign YAML(s) for research mode. Glob patterns accepted. |
| `--resume-sim` | str | None | Resume a previous simulation session by ID or date prefix |
| `--sandbox` | str | `auto` | Sandbox type: `auto` (Docker if available, else tmpdir), `docker`, `tmpdir` |
| `--sandbox-image` | str | `python:3.12-slim` | Docker image for sandbox container |
| `--sandbox-network` | str | `none` | Container network: `none` (isolated), `bridge` (outbound), `host` (shared) |
| `--continuous` | flag | | Never auto-complete, keep testing until `/cancel` |
| `--no-sim-env` | flag | | Skip simulated filesystem with pain-triggering files |
| `--sim-report` | str | None | Write structured results to a JSON file (requires `--sim`) |

## Asset Foundry

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--foundry` | str | None | Theme prompt for component generation (e.g., `"cyberpunk weapons"`). Runs the full pipeline: generate, validate, SEM protocol tests, gauntlet, score, curate. Output to `~/.maxim/foundry/{run_id}/`. |
| `--foundry-count` / `--count` | int | 10 | Number of components to generate. |
| `--foundry-genre` | str | `fantasy` | Genre tag for generated components. |
| `--foundry-category` | str | auto | Category: `weapons`, `creatures`, `npcs`, `items`, `environments`, `vehicles`, `bodies`. If omitted, distributes across categories. |
| `--foundry-dry-run` | flag | off | Generate + validate only, skip gauntlet testing. |
| `--llm` (with `--foundry`) | str | None | LLM profile for creative generation. Without `--llm`, foundry uses template fallback. E.g., `maxim --foundry "sci-fi creatures" --llm mistral-7b`. |

See [Asset Foundry Guide](asset-foundry.md) for the full pipeline description and usage examples.

### Auto-Curation

Pre-sim auto-curation checks genre/category coverage and fills gaps via the foundry.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--auto-curate` | flag | off | Before sim, check genre/category coverage and run foundry to fill gaps. Requires `--embodiment`. Promoted components go to `~/.maxim/components/`. |
| `--curate-threshold` | int | 5 | Minimum components per genre/category before auto-curation triggers. |
| `--no-curate` | flag | off | Explicit opt-out of auto-curation (overrides `--auto-curate`). |

**Example:**
```bash
# Auto-curate fantasy components before a sword combat sim
maxim --sim "test sword combat" --embodiment weapons/rusty_sword --auto-curate

# Auto-curate with higher threshold and LLM generation
maxim --sim "explore dungeon" --embodiment environments/dungeon_corridor \
  --auto-curate --curate-threshold 8 --llm mistral-7b --foundry-genre fantasy
```

**Dedup:** Near-duplicate candidates (cosine similarity >= 0.80 via ComponentIndex) are skipped during promotion, preventing library bloat.

**API equivalent:**
```python
import maxim
maxim.imagine("explore dungeon", auto_curate=True, curate_genre="fantasy")
maxim.run(auto_curate=True, curate_genre="cyberpunk")
```

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
| `maxim peer update [--dry-run] [--version <X.Y.Z>]` | Update the leader. Auto-detects install mode: pip-installed leaders upgrade via `pip install --upgrade pymaxim[extras]` (extras auto-detected and preserved); git-checkout leaders use `git pull --rebase` + `pip install -e .`. `--dry-run` previews available version (pip) or pending commits (git). `--version X.Y.Z` pins a specific PyPI version (pip mode only). |
| `maxim peer update --dev [<branch>] [--force]` | Force git update mode. Optional branch arg defaults to `main`. `--force` stashes dirty tree first. Errors if the leader has no git checkout. |
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
| `maxim peer --node <name> install <extras_or_packages>` | Mesh-aware install on a named node. Composes **drain → install → resume** around the shared `install_on_target` core in [install_core.py](../../src/maxim/peer/install_core.py). Resolves the target URL + cluster key from `mesh.yml::nodes`, **not** `peer.yml` — note that if the two files have diverged (e.g. after an unrelated cluster-key rotation), this verb sends `mesh.yml::cluster_key` while the positional-URL `maxim peer install` verb sends `peer.yml::api_key`. If you see 401s from one but not the other, the two secrets are out of sync. Accepts comma-separated `KNOWN_EXTRAS` (routed as `pymaxim[<extra>]`) plus raw pip package names in the same token list. **Refuses self-install** (points at local `pip install pymaxim[<extras>]` — no `--force` flag). **Refuses a positional URL** (points at `maxim peer install <extras> <url>` as the no-mesh.yml fallback; URL is redacted to scheme-only in the error message to avoid leaking secrets typed into argv). **Was-drained sticky:** if the operator drained the node BEFORE running install, the verb skips both the drain step AND the auto-resume — the prior drain stays intact after a successful install, so `install` never mutates your drain intent. The was-drained check is atomic with the drain mutation via `drain_node_if_absent` under filelock, closing the TOCTOU window under concurrent admin. **Leaves drained on failure:** if drain succeeds but the install itself fails, the node is left drained with a loud "STILL DRAINED → run `maxim peer --node <name> resume`" message. No auto-resume-on-failure. Exit codes: `0` ok, `1` install failed, `2` refused pre-install (self, unknown, bad tokens, drain failed), **`3` install succeeded but post-install auto-resume failed** (distinguishable from `1` so operators tailing exit codes can tell "upgraded but stuck in drain" from "failed and stuck in drain"). Any mid-install exception (`KeyboardInterrupt` included) still prints the still-drained hint before re-raising. (Plan 4 Stage C3.3) |

| `maxim peer --node <name> update [--dry-run] [--version <X.Y.Z>] [--dev [<branch>]] [--force]` | Mesh-aware update on a named node. Composes **drain → update → resume** around the shared `update_on_target` core in [admin_core.py](../../src/maxim/peer/admin_core.py). Supports the same pip/dev dual-mode as `maxim peer update`: auto-detects install mode by default, `--version X.Y.Z` for pip version pinning, `--dev [branch]` for git mode. `--dry-run` previews without draining (read-only). `--force` stashes dirty tree (dev mode). Self-guard refuses updating yourself (use `maxim peer update` directly). Same was-drained sticky semantics and exit-code contract as `--node install` (0 ok, 1 failed, 2 refused, 3 resume failed). (Plan 4 Stage C3.5 + peer_update_pip_mode) |
| `maxim peer --node <name> restart` | Mesh-aware restart on a named node. Composes **drain → restart → resume** around the shared `restart_on_target` core. Two-phase recovery poll waits for the proxy to respond (~90s), then waits for the LLM model to load (~150s for large models). Self-guard refuses restarting yourself (use `maxim peer restart` directly). Same exit-code contract as `--node install`. (Plan 4 Stage C3.5) |
| `maxim peer --node <name> llm <model>` | Mesh-aware LLM swap on a named node. Composes **drain → swap → resume** around the shared `llm_swap_on_target` core. Key enabler for C5 capacity-aware routing — per-node model assignment means the router can know which node runs which model. Self-guard refuses swapping yourself (use `maxim peer llm` directly). Same exit-code contract as `--node install`. (Plan 4 Stage C3.6) |

| `GET /v1/debug/vram` | (admin endpoint, not a CLI verb) Returns the leader's live VRAM state as JSON: nvidia-smi ratio, utilization, temperature, projected model footprint from `project_vram_usage()`, spillover/warning flags, and recommended n_ctx. Auth via bearer (cluster key) or localhost. Returns 503 if nvidia-smi is unavailable (not a GPU node). Prerequisite for peer-mode doctor VRAM visibility and C5 capacity-aware routing. (Plan 4 Stage C3.4) |

**Future (post-Stage C3.6):** `/v1/mesh/*` admin API, per-agent rate limiting, request-trace ring buffer, cluster key rotation, C4.6 auto-undrain via periodic health probe. Full arc tracked in [docs/plans/reactive_peer_mesh_roadmap.md](../plans/reactive_peer_mesh_roadmap.md).

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

## Instance Configuration (`maxim config`)

The `maxim config` verbs manage the operator-level config at `~/.config/maxim/config.json`. As of 1.0 this is the canonical way to persist Maxim's runtime preferences (role, default model, lane routing, cloud LLM fallback, auto-spawn behavior, etc.). The precedence chain is **CLI args > env vars > config.json > builtin defaults** — env vars still work as per-session overrides but are no longer the recommended primary surface. See [configuration.md](configuration.md#quick-start-maxim-config) for the full absorbed-fields table.

| Command | What it does |
|---|---|
| `maxim config get` | Print every absorbed field with its effective value + source marker (`cli` / `env` / `config` / `default`). |
| `maxim config get <field>` | Print one field by dot path (e.g. `llm.profile`, `lanes.large.remote_url`). |
| `maxim config set <field> <value>` | Atomic write to `config.json` under a `filelock.FileLock`. String values are coerced to the schema type (`"4"` → int 4 for `proxy.max_concurrent`). Pass `null` or `-` to clear a field. |
| `maxim config list [--json]` | Same as `get` with no args; `--json` for machine-readable output. |
| `maxim config path` | Print the resolved `config.json` path (XDG-aware). |
| `maxim config edit` | Open `$EDITOR` (or `$VISUAL`) on the file. Validates after the editor exits; a parse error surfaces a warning but doesn't auto-revert. |

**API key references** in `config.json::lanes.<tier>.remote_api_key_ref` accept ONLY file paths (`/...` or `~/...`) or keyring URIs (`keyring:<service>:<account>`). Inline plaintext keys are rejected at load time. The canonical leader-key path is `~/.config/maxim/api_key` (mode 0600).

**`maxim doctor` shows a "Resolved Config" section** that renders every absorbed field's value + source — the single answer to "what does this instance think it's configured as?"

**Auto-migration:** on first startup when `config.json` is absent + `peer.yml` is present + cloudflared config absent, the loader writes `config.json::role=peer` + `lanes.large.*` from peer.yml fields. peer.yml is never deleted by the shim. When cloudflared config exists (i.e., this machine is a tunneled leader), migration is skipped so a stale peer.yml from a previous peer setup doesn't flip the role.

**Atomic writes:** `config_writer.py` is the only sanctioned writer. Single-file FileLock around read-modify-write. CI grep enforces single-writer.

**Exit codes:** `0` success, `1` environmental failure (write permission, etc.), `2` operator error (unknown field, bad value type, missing arg, validation failure after `edit`).

## Model Profile Management

The `maxim model` verbs manage user-defined model profiles in `~/.config/maxim/profiles.yml`. Profiles authored here merge into the bundled profile set at startup with **user-wins precedence**, so any GGUF on HuggingFace (or any local GGUF file) becomes a first-class profile usable from `--llm`. Full schema + walkthrough in [llm-setup.md § Adding Custom Profiles](llm-setup.md#adding-custom-profiles).

| Command | What it does |
|---|---|
| `maxim model add <name> --hf REPO:FILE [--chat-format STYLE] [--n-ctx N] [--alias A] [--force]` | Register a new profile that downloads a GGUF from a HuggingFace repo. The `--hf` value is `REPO:FILE` (the repo may contain `/`; first `:` separates from filename). `--chat-format` is auto-inferred when omitted via substring match on the profile name + repo basename (`qwen`→`chatml`, `llama-3`→`llama3_instruct`, `mixtral`/`mistral`→`mistral_instruct`, `phi-3`→`phi3`, `phi-2`→`phi`, `gemma`→`gemma`); if inference fails the verb errors out asking for an explicit value. `--alias` is repeatable. `--force` overwrites an existing user profile with the same slug. |
| `maxim model add <name> --local <path> [...]` | Same as above but points at a local GGUF file instead of downloading. The path is `expanduser`-ed at load time. |
| `maxim model remove <name>` | Drop a user profile. Refuses if the profiles file is missing (exit 1) or the slug doesn't exist (exit 2). |
| `maxim model list` | Show user profiles with backend / prompt_style / source markers. For the full builtin + user catalog, use `maxim --list-models`. |

**Round-trip caveat:** `add` and `remove` rewrite the file via stdlib YAML serialization, which **strips comments**. Edit `~/.config/maxim/profiles.yml` directly to preserve them.

**Atomic writes:** `profiles.yml` is treated as declarative config (not a secret), so writes use `atomic_write_text`, not `atomic_write_secret`. Same convention as `mesh.yml`.

**Exit codes:** `0` success (including idempotent no-ops), `1` environmental failure (file missing for `remove`), `2` operator error (missing required arg, name collision without `--force`, unknown profile, bad name characters, missing `--chat-format` when inference can't pick a default).

## Roy Harness

Long-horizon persona-convergence iteration runner. One `maxim roy run`
command primes substrate via a curriculum, runs the same held-out test
across three arms (substrate-primed neutral / blank persona-injected /
blank neutral), and reports pairwise substrate divergence (`NAc reward_bias`
+ `cluster_reward_bias` + Hippocampus episodes + ATL concepts). See
[persona_convergence_crucible.md](../plans/persona_convergence_crucible.md)
for the methodology and [substrate_diff.py](../../src/maxim/analysis/substrate_diff.py)
for the metric definitions.

| Command | Description |
|---------|-------------|
| `maxim roy run <iteration_spec.yaml> [--dry-run]` | Run a three-arm Roy iteration end-to-end. Includes a fail-fast LLM pre-flight probe (G3) that resolves the `large` lane URL from `MAXIM_LANE_LARGE_REMOTE_URL` env or `~/.config/maxim/peer.yml` and aborts in <3s if the leader is unreachable. `--dry-run` validates the spec only. Persists `result.json` + `summary.md` to `~/.maxim/roy/<iteration_name>/`. |
| `maxim roy diff <session_a_dir> <session_b_dir> [--json]` | Substrate divergence between two `~/.maxim/sim_reports/<session_id>/` directories. Reads `aut_nac.json`, `aut_hippocampus.json`, `ec.json`, `atl.json` from both sides and reports `reward_bias L2`, `cluster_reward_bias L2`, causal-link / episode / concept deltas. `--json` matches the `result.json` payload shape. |
| `maxim roy log <iteration_id> [--plan PATH] [--keep-edits] [--dry-run]` | (Re-)generate the iteration's protocol runbook + iteration-log entry from a persisted `result.json`. Idempotent — `--keep-edits` preserves hand edits between HTML-comment markers. |

**Spec shape** (deliberately tiny — see [scenarios/roy/roy_0_smoke_iteration.yaml](../../scenarios/roy/roy_0_smoke_iteration.yaml) for the canonical example):

```yaml
name: roy-0-smoke
embodiment: bodies/infant_humanoid
aut_mode: substrate-primary     # or llm-primary
priming:
  name: roy-0-priming
  stages:
    - { name: act1, fixture: scenarios/cradle/warmup.yaml, turns: 10 }
test_scenario:
  fixture: scenarios/cradle/warmup.yaml
  turns: 10
arms:
  a: { substrate: from_priming, system_prompt: neutral }
  b: { substrate: blank,        system_prompt: "You are a hungry infant" }
  c: { substrate: blank,        system_prompt: neutral }
```

**Pre-flight probe:** the runner aborts with `aborted_at="preflight"` if the
configured `large` lane is unreachable before priming starts. Env vars
(`MAXIM_LANE_LARGE_REMOTE_URL` / `_API_KEY` / `_MODEL`) take precedence; peer.yml
is the fallback. Local-LLM and cloud-only configurations skip the probe
with a documented reason (their failure modes surface fast at first
dispatch). `result.preflight.source` records `"env"` or `"peer.yml"`.

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

### Launch the interactive menu

```bash
maxim
```

Displays a Rich menu with campaign discovery, recent sessions, and quick-start options. Ctrl+C during a simulation returns to this menu.

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

### Interactive simulation

```bash
maxim --sim interactive
```

Redirects to the generative sim with the full interactive stack (rich display, bidirectional input, SimPromptHandler).

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

### Substrate-primary AUT (experimental)

```bash
# Run a cradle sim where the AUT acts without LLM proposal — only NAc
# recommend_action + reflexes. Phase -1 of grounded language acquisition.
maxim --sim cradle --embodiment bodies/infant_humanoid \
  --aut-mode substrate-primary --interactive false --sim-max-turns 10
```

### Roy three-arm persona iteration

```bash
# Validate spec without invoking sims
maxim roy run docs/plans/roy/roy_0_smoke.yaml --dry-run

# Run a real iteration (priming + 3 arms + pairwise substrate diffs)
maxim roy run docs/plans/roy/roy_0_smoke.yaml
# → ~/.maxim/roy/roy-0-smoke/result.json + summary.md

# Compare two session dirs directly (no Roy iteration required)
maxim roy diff ~/.maxim/sim_reports/<session_a> ~/.maxim/sim_reports/<session_b>

# Regenerate protocol + iteration-log entry from an existing result.json
maxim roy log roy-0-smoke --plan docs/plans/persona_convergence_crucible.md
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

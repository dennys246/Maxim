# Maxim Minecraft bridge (1.1.4 world seam, PR 3)

The JS half of the world seam: a Mineflayer bot that owns the game connection
and speaks newline-delimited JSON over TCP to the Python side. **The protocol
authority is `src/maxim/simulation/minecraft.py`** — its module docstring
freezes the message shapes; keep this process in sync with it.

Not packaged in the wheel, not run in CI (CI has no Minecraft server): a
dev-side tool, like everything under `scripts/`. The Python seam is fully
testable without it (injected-transport fakes in
`tests/unit/test_minecraft_seam.py`); this process is what the PR 4 smoke
benchmark and the 1.2 arms run against.

## Setup

```bash
cd scripts/minecraft_bridge
npm install
# a vanilla/paper server on localhost:25565, offline-mode for dev
node index.js --mc_host=127.0.0.1 --mc_port=25565 --bridge_port=25567 --username=maxim
```

Two-AUT harness (PR 4): run TWO bridges against one server — one bot +
bridge port per AUT (`--username=maxim_a --bridge_port=25567`,
`--username=maxim_b --bridge_port=25568`). The bridge accepts one client at
a time by design.

## Flags

| flag | default | meaning |
|---|---|---|
| `--mc_host` / `--mc_port` | `127.0.0.1:25565` | the Minecraft server |
| `--bridge_port` | `25567` | TCP port the Python `MinecraftClient` dials |
| `--username` | `maxim` | bot username (offline mode) |
| `--state_interval_ms` | `500` | snapshot cadence (per-Maxim-tick buffering, plan Q7) |

# Exp 56 — operator runbook (the four-arm sharing benchmark)

The live apparatus + campaign steps for
[`exp56_four_arm_sharing_preregistration.md`](../../docs/experiments/protocols/exp56_four_arm_sharing_preregistration.md)
(frozen; amendment 1 included). Everything science-frozen lives in
[`common.py::FROZEN`](common.py) and the prereg — this file is purely
operational. The `--mock` paths (ScriptedBridgeServer) are for harness
development only and are **never a confirmatory record**; the analyzer
refuses a verdict on mock rows.

## Prerequisites

- **Java 11–16** (Minecraft 1.16.5 refuses 17+): `brew install openjdk@11`.
- **Node** (for the Mineflayer bridge): `cd scripts/minecraft_bridge && npm install`.
- macOS: allow the terminal Local Network access (System Settings → Privacy)
  or the bridge/RCON connects will silently hang.
- A **clean tree at a main-reachable commit** for anything gated
  (`preflight_gated_record` refuses otherwise; `--allow-dirty` is disallowed
  for the confirmatory campaign by the prereg).

## 1. Stand up the world (one-time)

```bash
# Download Paper 1.16.5, write configs. EULA acceptance is YOUR action:
python scripts/exp56/setup_world.py setup --dir ~/exp56_server --accept-eula \
    --rcon-password 'CHOOSE_A_PW'

# First boot (generates the superflat world with the surface at y=63 —
# feet at y=64, the bench body's neutral). Leave it running:
cd ~/exp56_server && java -Xms1G -Xmx2G -jar paper-1.16.5.jar nogui

# From the repo root, against the RUNNING server — sets daylight/weather/
# mobs off, noon, world spawn at the frozen rest anchor, and BUILDS the
# four frozen contingency slots (floating stone enclosures, far + high) + rest pad:
python scripts/exp56/setup_world.py prepare --rcon-password 'CHOOSE_A_PW'
python scripts/exp56/setup_world.py verify  --rcon-password 'CHOOSE_A_PW'   # exit 4 on mismatch
```

## 2. Start the bridge (per instance; AFTER `prepare` — the bot's spawn
point is captured at join, and `distance_from_spawn` reads it)

```bash
cd scripts/minecraft_bridge
node index.js --mc_host=127.0.0.1 --mc_port=25565 --bridge_port=25580 --username=maxim_bench
```

`--bridge_port=25580` matches the harness default; `--username=maxim_bench`
matches the harness's `--bot-name` default (the RCON `/tp` targets that
name). One bridge takes ONE client — sessions are sequential per instance;
to parallelize, run more server+bridge instances on distinct ports and pass
`--bridge-port`/`--rcon-port` accordingly (the instance is stamped per record).

## 3. Phase 0 (gates the campaign; all five checks or no campaign)

```bash
export MAXIM_OPERANT_ONLY_CREDIT=1     # frozen apparatus; the CLIs REFUSE without it
export PYTHONPATH="$PWD/src"           # if running from a worktree
python scripts/exp56/instrument_check.py --rcon-password 'CHOOSE_A_PW' \
    --write-experiment-results
```

Exit 4 on any failing check, with the per-check report in
`docs/experiments/data/56_phase0.json`. Then, BEFORE the campaign:

- fill the prereg **sign-off boxes** (prereg merge `3f9ce733`, harness merge
  `6219f330`, k=8, K=96, 4 slots) and
- disclose the Phase-0 **readings as an amendment entry** in the prereg
  (its own PR, merge-commit) — the pre-registered disclosure rule.

## 4. The confirmatory campaign

```bash
export MAXIM_OPERANT_ONLY_CREDIT=1
python scripts/exp56/run_campaign.py \
    --arms isolated,taught,satiated,dangling --pairs 50 --seed-base 42 \
    --workdir ~/exp56_work \
    --out docs/experiments/data/56_four_arm.jsonl \
    --rcon-password 'CHOOSE_A_PW' --write-experiment-results

python scripts/analyze_exp56.py --in docs/experiments/data/56_four_arm.jsonl \
    --gate v1 --assert-noop-fails
```

- `--workdir` must be **durable** (the taught donor stages are the arm-4
  reuse contract; `--resume` refuses a tmpdir).
- Budget: ~300 sequential sessions ≈ 1–2 days per bridge instance.
- Interrupted? Re-run the same command with `--resume` (same `--workdir`,
  same `--out`).
- Data PR: **merge-commit only, never squash**; interpretation lands in a
  separate later PR (the structure-or-time rule).

## Troubleshooting

| symptom | cause / fix |
|---|---|
| server exits on boot with class-version errors | Java 17+ — use openjdk 11–16 |
| `RCON authentication failed` | password mismatch with `server.properties`; restart the server after edits |
| bridge prints nothing / harness times out | bridge not connected to the game (check `--mc_port`), or a second client tried to attach (one client per bridge) |
| `exp56 … the world does not reflect the script (S3)` | `prepare` not run, wrong RCON target name, or the bot spawned before `setworldspawn` — restart the bridge after `prepare` |
| campaign refuses at start | `MAXIM_OPERANT_ONLY_CREDIT` unset, dirty tree, ambient `sim.substrate_explore_bonus_weight` ≠ 0, or `--resume` without a durable `--workdir` — each refusal names its fix |

# CLAUDE.md — maxim-pulse

> **✅ SEED CONSUMED 2026-08-01 — canonical copies live in the `Maxim-pulse` repo and have since diverged from this fork. Do not edit.**

Operational guide for Claude Code in this repo. **Architecture, execution flow, and
development standards live in [AGENTS.md](../../maxim_pulse_seed/AGENTS.md) — read it first.** This file is the
checks + commands + Claude-specific workflow layer.

## Project overview

`maxim-pulse` — the **Maxim UI monorepo**: a React + TypeScript (Vite) monorepo producing a
shared UI kit + **two build targets** (the **Maxim Console** localhost dashboard, and the
**Reachy app** on-device HF Space). Both are **presentation over pymaxim** (`api.py` /
`maxim serve`). **Cardinal rule: presentation-only — fix hard things in pymaxim, not here.**
See AGENTS.md § "The cardinal rule" + "The three-layer architecture."

## Required checks (run before considering a task done)

```bash
# Lint + format + types (JS/TS)
pnpm lint          # eslint
pnpm format        # prettier
pnpm typecheck     # tsc --noEmit

# Tests — fully offline (facade client + maxim serve + Reachy SDK all mocked)
pnpm test          # vitest

# Build both targets; confirm the Reachy bundle stays lean (code-split guard)
pnpm build
pnpm size:reachy   # assert heavy viz (react-flow/visx) is NOT in the on-device bundle

# If the Reachy Python bootstrap changed (apps/reachy):
ruff check apps/reachy && ruff format apps/reachy && python -m pytest apps/reachy -q
```

No test may spin up a real robot, a real LLM, or a live `maxim serve` — mock the facade client.
For Reachy loop tests use a fake SDK (the real `ReachyMini()` blocks without hardware).

## Running

```bash
# Console dev (needs pymaxim's backend for live data):
maxim serve                 # pymaxim, binds 127.0.0.1 — the Console's backend
pnpm dev --filter console   # Vite dev server → talks to maxim serve (or a mock facade)

# Reachy target (offline dev): build + run the ReachyMiniApp bootstrap against a fake SDK.
pnpm build --filter reachy
```

- **Verify SetupWizard writes real config.** After a mesh/cloud choice, confirm
  `~/.config/maxim/config.json` has a resolvable `lanes.large` placement and the key landed as a
  **ref** (file/keyring), never inline. "Test connection" calls the `PROBE` seam — don't hand-roll.
- **Verify session-end persists** (Reachy). On `stop_event`, confirm **full** session-end
  consolidation + `save_cerebellum()` fire and `~/.maxim/` grows. If it silently no-ops, the
  cross-session thesis is broken (AGENTS.md § execution flows).
- **Verify EventClient renders the live stream** — the `observe`/`api.on()` events show in the
  activity/thinking panels (the ported `MaximDisplay` IA).

## Claude-specific workflow

- **Parallel sessions use worktrees.** ≥2 concurrent Claude sessions → each in its own
  `git worktree`, absolute paths. `~/.maxim/` is shared — don't run two live agents at once.
- **Don't co-locate a leader and a live agent run on one box** (pymaxim's Exp 37 cascade lesson).
- **Commit/push only when asked.** Branch first if on the default branch. `Co-Authored-By`
  trailer on commits; Claude Code footer on PR bodies.
- **When a change spans this repo and pymaxim,** prefer the pymaxim fix (a facade/seam) and note
  it — do not work around a pymaxim gap in the UI (AGENTS.md cardinal rule).

## Key facts to not re-derive

- **Three layers:** pymaxim facades/seams (L1, not here) → shared kit (L2) → Console + Reachy
  shells (L3). Every component binds to a facade/seam, never a back-channel.
- **Stack:** React + TS / Vite; backend is pymaxim `maxim serve` (JSON + WebSocket). Viz:
  react-flow + visx on shadcn/ui or Mantine.
- **LLM routing:** 3 destinations, one is cloud (mesh-LAN / mesh-tunnel / cloud), all via
  `config.json::lanes.large.placement` (the `SETUP` seam).
- **Reachy:** WS-era transport (`ws://<host>:8000/ws/sdk`, no zenoh); head pose is world-frame
  (`head=None` counter-rotates) → route motion through pymaxim's `ReachyMiniController`;
  version-match client ↔ daemon after any reflash.
- **Code-split:** heavy dashboard viz never enters the Reachy on-device bundle.
- **Localhost-only:** `maxim serve` binds `127.0.0.1`.

## Versioning

Two cadences: the **Reachy target** tracks **Pollen SDK** (pin `reachy-mini[gstreamer]>=1.8.3,<2.0`);
the **kit + Console** track **pymaxim**. Pin the `pymaxim` dependency explicitly and bump it
deliberately. Bump the app/target version on any change to runtime behavior, the setup flow, or
SDK-compat.

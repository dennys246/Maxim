# Deferred — NAc cross-session persistence (save AND load, with a decay decision)

**Status:** Deferred, drafted 2026-07-29. A partial fix (save only) was written, reviewed, and **deliberately reverted** — see [Why the partial fix was pulled](#why-the-partial-fix-was-pulled).
**Severity:** High for the product claim — this is the substrate "it remembers you" rests on.
**Revive trigger:** before any work that depends on cross-session NAc learning (Oasis contribution, the Exp 44 pre-load story, Reachy long-horizon), or the next time someone asks why an agent does not remember what it learned.

---

## The finding

Two independent defects that only combine into a *worse* outcome if you fix one:

### 1. NAc is never SAVED
`build_bio_stack` assigns `persistence_path` to hippocampus, ATL and angular-gyrus but **not** NAc. `MemoryHub`'s save is guarded `if nac_path:` — falsy — so it silently skips. No exception, no warning.

### 2. NAc (and hippocampus) are never LOADED on the `create_full_agent` path
`AgentFactory.create_full_agent` auto-loads NAc at Step 1 (`_create_nac(agent_dir, auto_load=True)` → `load_safe`), then at [agent_factory.py:450](../../src/maxim/runtime/agent_factory.py) **overwrites `instance.nac` with `bio.nac`** — and `build_bio_stack` loads *cerebellum only* ([bio_stack.py:359](../../src/maxim/runtime/bio_stack.py)). The restored object is discarded. A comment at the overwrite calls this "correct", which it was while nothing persisted.

The same discard applies to **hippocampus** and **ATL**. `MemoryHub.on_session_start` restores ATL / AngularGyrus / cross-layer / EC embeddings but **not** hippocampus and **not** NAc — that asymmetry is the actual hole.

## Why the partial fix was pulled

Adding the save alone is a **strict regression in kind**. Before: nothing written, nothing lost. After: each session writes a populated `nac.json` and the next session discards it — so every run **truncates** the previous one while leaving a plausible file on disk that makes persistence *look* solved. That is the silent-failure class CLAUDE.md legislates against, manufactured by a fix.

Measured on the real console path:

```
auto-loaded NAc had links: 1
instance.nac IS that object? False        -> loaded state discarded
S1 on disk: ['tool_a'] -> S2 in-memory links after restart: 0
S2 on disk: ['tool_b'] -> session-1 'tool_a' SURVIVED? False
```

## What the real fix has to include

1. **Save** — `persistence_path=str(p / "nac.json")` in `build_bio_stack`, beside the others.
2. **Load** — mirror the cerebellum pattern *in the same function*: `load_safe()` when the file exists, guarded, warn-on-failure. Doing it in `build_bio_stack` keeps path ownership in one place and closes the **hippocampus** hole at the same time.
3. **A decay-on-load decision — the part that needs thought, not just wiring.** NAc decay is **tick-anchored** (`decay_reward_biases` / `decay_eligibility` run per-tick from agent_loop §8.5). `cluster_reward_bias` tau defaults to **300 ticks** ≈ 150 s at the talk loop's 2 Hz. The only elapsed-time-ish decay is a flat `decay_all(factor=0.95)` at session end — so a restore applies the same 5% haircut whether the gap was five minutes or five months. A bias learned six months ago would return indistinguishable from one learned a minute ago. Biologically, NAc associations extinguish with elapsed time and unreinforced re-exposure, not with the agent's tick count. Concretely:
   - **Stamp `saved_at` in `NAc.dump()` now** — a future decay-on-load has nothing but file mtime otherwise, and `_NAC_FORMAT_VERSION` is already being touched.
   - **Decide whether `cluster_reward_bias` should persist at all.** Its 300-tick tau says *within-session working signal*; Exp 44 Gate 2 had to pin `MAXIM_NAC_CLUSTER_REWARD_BIAS_DECAY_TAU=1000` to hold it across **one** session. Persisting it unchanged silently promotes a session-scoped signal to a cross-session one — a larger semantic change than "fix persistence" implies. CLAUDE.md already designates `reward_bias` as the intended cross-session transfer surface; `cluster_reward_bias` rides along only by being in `dump()`. Eligibility traces are correctly absent.
   - Suggested shape: elapsed-wall-clock decay for `cluster_reward_bias` on load, `reward_bias` on its slower schedule, choice documented at the `persistence_path` line.
4. **Format hygiene** — route `nac.json` through the atomic-io + `_format_version` helpers like other persisted files.

## Regression guard the fix must ship with

A **two-session round-trip**: write `tool_a` in session 1, restart, write `tool_b` in session 2, assert **`tool_a` is still on disk**. The reverted attempt had a test asserting only that `nac.json` *appears* — which is precisely the half that worked, and would have passed over a truncating implementation.

## Scope note

`build_bio_stack` is the canonical construction site for CLI non-sim, the sim AUT, Reachy and headless `pymaxim`. This lands as **its own PR against main with its own review**, not folded into a console branch — the blast radius is every agent, and a console-titled PR points the reviewer at the wrong thing.

Sim interaction is benign today: the AUT uses `persistence_dir=<sim tmpdir>`, so a new `nac.json` lands in a per-run temp dir and `resume_session` restores from the separate session-dir `aut_nac.json`. Note `analysis/substrate_diff.py` already globs `("aut_nac.json", "nac.json")`, so once agent-home `nac.json` exists, analysis tooling will start picking up a file that never previously appeared.

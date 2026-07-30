# Deferred — NAc cross-session persistence (save AND load, with a decay decision)

**Status:** SHIPPED 2026-07-30 (feat/nac-cross-session-persistence) — save + load + decay-on-load landed together, plus the Step-0 prerequisite (stable hashing at persistence boundaries). Historical context below; see [What shipped](#what-shipped). The earlier partial fix (save only) was written, reviewed, and **deliberately reverted** — see [Why the partial fix was pulled](#why-the-partial-fix-was-pulled).
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

## What shipped (2026-07-30)

**Step 0 (prerequisite):** persisted values derived from Python's randomized
`hash()` could never match across a process boundary — `SituationSignature`
structural/context hashes (0.825 same-process vs 0.425 after reload,
straddling NAc's `min_similarity=0.5` EC gate), `SimilarityIndex` MinHash
(reloaded index returns `[]` for its own content), `SemanticLSH.hash` (the
`seed` param routed through randomized `hash()` anyway), and
`NeuralSemanticLSH._fallback_hash` (persisted `EmbeddingStore` npz). All four
now use sha256-based `stable_hash_32` / `stable_hash_64_signed`
(`utils/seeding.py`). Guard: `tests/unit/test_stable_hash_two_process.py` —
two-process tests with differing PYTHONHASHSEED, all verified to FAIL pre-fix.
This also removes the ~2.5% CI flake in
`test_context_index.py::test_similar_text_found` (same root cause).

**Step 1:** all four items from [What the real fix has to include](#what-the-real-fix-has-to-include):

1. **Save** — `persistence_path=str(p / "nac.json")` in `build_bio_stack`,
   activating the pre-existing guarded save in `MemoryHub.on_session_end` /
   `on_session_end_lightweight`.
2. **Load** — `nac.load_safe()` in `build_bio_stack` (cerebellum pattern),
   plus the **hippocampus** hole (`load_with_recovery()` at build) and —
   beyond the original scope, for the reason below — **EC** (`ec.json`,
   `ECConfig.persistence_path`, saved beside NAc in both session-end paths).
   NAc `reward_bias` / `cluster_reward_bias` are keyed by EC node ids;
   restoring NAc without EC leaves every bias pointing at nodes a fresh EC
   never re-allocates — persistence that looks like it works while the
   biases silently dangle. Sound only after Step 0.
3. **Decay-on-load** — `NAc.save()` stamps `saved_at` (format 1.2 → 1.3; deliberately NOT `dump()`, which stays the pure BioSystemSnapshot surface so hivemind bundles carry no timestamp);
   `NAc.load()` applies elapsed-wall-clock exponential decay via
   `apply_wall_clock_decay`: `cluster_reward_bias` at a 1-day half-life
   (working signal; same-day resume stays near-fresh, week-old fades to ~1%),
   `reward_bias` / `goal_reward_bias` / `percept_valences` at a 7-day
   half-life (the designated cross-session transfer surfaces). Links +
   Welford variance are not decayed on load (accumulated statistics; links
   already take `decay_all(0.95)` at session end). Pre-1.3 payloads load
   undecayed. `load_state` (hivemind merge path) never decays.
4. **Format hygiene** — already routed through `atomic_write_json` +
   `with_format_version`; version bumped with the additive `saved_at` key.

**Regression guards:** `tests/unit/test_nac_persistence_decay.py` (decay
semantics) + `tests/integration/test_cross_session_persistence.py` — the
two-session, two-process (differing PYTHONHASHSEED) round-trip asserting
RECALLED CONTENT, verified to fail on both the no-persistence state and a
simulated save-only (truncating) implementation. Episodes verified to
survive the same path (hippocampus recall content asserted in session 2).

## Pre-merge two-lens review fold (2026-07-30)

Both lenses ran on the full branch diff; all blocking + cross-confirmed findings folded before the PR opened:

- **`apply_decay: bool = True` kwarg on `NAc.load()`/`load_safe()`** (Arch #1 BLOCKING + Exec #2/#3, cross-confirmed): wall-clock decay must not reach call sites where elapsed time is not agent-experienced time. `--resume-sim` (orchestrator.py) passes `apply_decay=False` — sims are tick-anchored and the Exp 44 tau-hold harness depends on verbatim resume; read-only observers (`maxim.load.nac`, `api.py` Observer + recall) pass `False` — disk truth, no compounding on load→save round-trips. Decay-with-prunes logs at WARNING.
- **`AgentConfig.load_persisted` / `build_bio_stack(load_persisted=)`** (Arch #2 BLOCKING): the sim orchestrator NPC (`~/.maxim/orchestrator`, months of write-only accumulation, saves at `shutdown()`) now opts out with `load_persisted=False` — write-but-don't-read, so the narrator/orchestrator cannot become a cross-run confound and worktree-shared `~/.maxim` cannot cross-contaminate experiments. `create_full_agent`'s `auto_load` docstring corrected (it governs only the discarded skeleton; bio-stack restore is governed by `load_persisted`).
- **Fifth randomized-hash site fixed** (Arch #3): `nac.py::_register_causal_in_ec` now uses `stable_hash_32` (its signatures land in the newly-persisted `ec.json`).
- **Corrupt-file robustness** (Exec #1 + Arch #5, cross-confirmed): `load_state` float-coerces all four bias surfaces (skip-on-failure, the in-file Welford precedent) so a string-valued bias cannot poison decay-on-load or the per-tick §8.5 decay block; `load_safe` recovery resets ALL mutable surfaces (and catches `AttributeError` for wrong-typed containers); `saved_at: true` (JSON bool) no longer reads as epoch 1 (bool guard). EC corrupt-load reconstructs a fresh instance instead of keeping a partially-mutated one (Arch #4 + Exec #4, cross-confirmed).
- **Hash-scheme drift marker** (Arch #7 + Exec #7, cross-confirmed): `EC.save`/`SimilarityIndex.save` stamp `hash_scheme: "stable-sha256-v1"`; loaders WARN on pre-fix files so dead hashes read as a named condition, not "recall is noisy".
- **Bio-framing corrected** (Arch #6): wall-clock forgetting documented as an `[engineering]`-tier calibration, not extinction; the 7-day `percept_valences` (harm-avoidance) cell flagged in NACConfig as the weakest choice, calibration candidate once an experiment earns it.

## Tracked follow-ups (not in this branch)

- **Config surface for the half-lives** — `bias_wall_decay_half_life_s` / `cluster_bias_wall_decay_half_life_s` are NACConfig fields with no `resolve_setting`/config.json path yet (prefer-config-over-env standard). Wire when an operator first needs to tune them.
- **`percept_valences` decay schedule** — consider slower/no wall decay for negative valences (conditioned aversion is biologically the most persistent surface here); needs an experiment to earn the change.
- **`ec.json` growth** — EC substrate nodes (384-float embeddings) accumulate without a prune counterpart and are rewritten in full each session end. Fine for months; revisit for the 1.1 Oasis horizon.
- **SCN persistence asymmetry** — `build_bio_stack` still constructs `SCN()` with no `persistence_path`; oscillator phases restart cold every session. Deliberately NOT wired here (see [deferred/scn_event_producer_gap.md](../deferred/scn_event_producer_gap.md) — the intake is mostly dead; wiring persistence before producers exist would be backwards).
- **Release note** — first post-upgrade run of a persistent agent loads a possibly-large, never-before-read `hippocampus.json` ("your agent suddenly remembers everything", plus load latency).

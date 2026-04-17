# Structural Enforcement — pushing invariants into types, not helpers

**Status:** Active reference doc.
**Last updated:** 2026-04-14
**Maintenance contract:** this doc is the canonical statement of the structural-enforcement pattern that came out of the SEM execution hook + executor bootstrap unification work. When a new structural-enforcement plan opens or ships, update the "Catalog" table at the bottom.

## The pattern in one sentence

When a bug class has a **silent failure mode** (no exception, no log, just wrong behavior or missing learning) AND has more than one possible call site, push the invariant DOWN into the type / constructor signature so forgetting becomes a `TypeError`, not a silent no-op.

## Why this rule exists

The codebase has been bitten three times by the same shape:

| Stage | Path | Symptom | Fix shape |
|---|---|---|---|
| `sem_execution_hook` Stage 1 (PR #107) | tool-invoked embodiment pain attribution | bridge existed, attribution path silently scored 0.0 | direct-attribution API + `ToolOutput.side_effects` typed channel |
| `sem_execution_hook` Stage 2 (PR #110) | `maxim --llm X` non-sim CLI | bridge **never constructed** on the most common entry point | extract `runtime/embodiment_bootstrap.py` shared helper |
| `sem_execution_hook` Stage 2c (planned) | `--sim agent` + `--sim interactive` + `simulation/orchestrator.py` AUT | bridge **not constructed** on three sim paths | (would have been) call the helper from three more sites |

Three identical bugs in three locations is **not coincidence**. It is structural: the *layer* of the abstraction is wrong. Helper-discipline relies on every author of every new call site remembering the helper exists and calling it correctly. For loud-failure bugs (raise → log → user sees it), helper-discipline drift is self-correcting because the next caller hits the failure during testing. For **silent-failure bugs**, drift accumulates invisibly — the next caller ships, the bug rides along undetected, and the missing learning signal corrupts everything downstream until someone notices in production months later.

The fourth instance was about to ship as a Stage 2c PR adding the helper to three more sim sites. The mid-session pivot to `executor_bootstrap_unification.md` made forgetting impossible at the constructor level: `build_executor`'s `pain_bus` parameter is now keyword-only with no default. A fifth, sixth, and seventh silent-no-op gap surfaced during the migration audit (`api.py` headless mode, `simulation/tools.py` sub-AUT, `simulation/orchestrator.py::orch_executor`) and became explicit `pain_bus=None` decisions instead of unnoticed silent gaps.

## When to apply structural enforcement

Two questions, in order:

### 1. Is the failure mode silent?

- **Yes** → consider structural enforcement on instance ONE if the path is critical (correctness, security, learning signal).
- **No** → helper-discipline is fine. Loud-failure bugs are self-correcting because tests + users catch them.

### 2. Could the bug class reproduce on N+1 call sites?

- **Yes** → you need either structural enforcement OR a CI gate that catches missing call-site discipline.
- **No** (truly isolated to one site) → fix locally.

### Counting instances

| Count | Action |
|---|---|
| **One silent-failure miss in a critical path** | Consider structural enforcement immediately. The cost of the next miss is the cost of the first miss × the number of additional callers. |
| **Two silent-failure misses** | Fix locally + start designing the structural enforcement plan. You're now on borrowed time. |
| **Three silent-failure misses** | STOP. The plan is no longer optional. Push the invariant down a layer. |

For LOUD-failure recurring bugs: helper-discipline + a comment in `CLAUDE.md` is fine. Don't push everything down — over-structural code is brittle in its own way.

## How to push an invariant down

The available layers, from least to most structural:

1. **Documentation discipline** (CLAUDE.md invariant + commit message). Weakest — relies on every contributor reading the right doc at the right time.
2. **Helper function** (e.g., the previous `bootstrap_embodiment_and_pain_bridge`). Centralizes the construction logic but still requires every call site to remember to call it.
3. **CI grep gate** (e.g., the existing `urllib.request.urlopen` and `_MaximPeerBackend` retry-loop gates). Catches the bug at PR time but is brittle to renames + lives in the test infrastructure layer.
4. **Constructor parameter with default** (e.g., `build_executor(pain_bus=None)` BEFORE the unification). Surfaces the parameter at every call site but makes it possible to ignore.
5. **Constructor parameter with no default, keyword-only** (e.g., `build_executor(*, pain_bus, ...)` AFTER the unification). Forgetting becomes a `TypeError`. **This is the strongest structural enforcement available without introducing a new abstract base class or protocol.**
6. **Type system enforcement** via Protocol + Literal types or runtime validation at the boundary. Strongest but heaviest. Reserve for invariants that need to be checked at every call site of every implementation, not just the constructor.

The 2026-04-14 SEM execution hook session walked through layers 1 → 2 → 5 in three weeks. Layer 5 is where the bug class was finally killed.

## Companion rules

These came out of the executor unification pre-merge review round and apply to any structural enforcement plan:

### R1 — Audit before designing

Every structural enforcement plan starts with a written audit of every call site (file:line, what's wired, what's missing, what an explicit decision would look like). The executor unification audit found two pre-existing silent-no-op bugs the original Stage 2c plan had no idea existed (`api.py` headless mode + `simulation/tools.py` sub-AUT). Both surfaced because the audit forced every call site to explain itself.

### R2 — Pre-merge review with two parallel lenses

Two reviewers, Executor lens + Architecture lens, working independently on the same diff. Cross-confirmed findings (both lenses catch the same thing) get folded first. The executor unification review caught:

- A docstring in an unrelated module (`pain_interceptor.py`) that was teaching the forbidden retrofit pattern → would have re-introduced the bug class on the next read.
- A signature design smell (bridge gating on `pain_bus|pain_detector` instead of `nac`) that was forcing the sim orchestrator into a no-op `PainDetector()` workaround.
- A latent `UnboundLocalError` in `cli.py` from a pre-existing bug the migration was about to make user-visible.

Tests caught zero of those. Reviews caught all three, two of three were cross-confirmed.

### R3 — Gate the construction on the learning subject, not the signal source

The `executor_bootstrap_unification` C2 review fold lesson: when designing a builder for a bio-system bridge, gate construction on the **learning subject** (NAc, Hippocampus, ATL), not on the **signal source** (PainBus, PainDetector, ReactionBus). A caller that has the learning subject and wants direct attribution should not need to pass a no-op signal source to trick the constructor. The bridge's primary value is direct attribution; subscription is the secondary out-of-band path.

### R4 — Declared fields beat attribute stashes

When a builder needs to surface an optional object to its caller (e.g., `Embodiment` from `build_executor`), declare the field on the constructed object's `__init__`. Don't do `obj._foo = bar  # type: ignore[attr-defined]`. Untyped attribute stashes silently drop to `None` if a wrapper sits between the builder and the reader, and they're invisible to mypy.

### R5 — One unification per PR

The temptation is to write a mega-plan that fixes all candidates in one PR. Resist. Each unification has its own audit surface, its own review questions, its own migration risk. Combining them turns a clean N-PR sequence into one ~2000-LOC monster nobody reviews well.

### R6 — Doc + memory refinement is part of the work

Every structural enforcement plan ships with: (a) `CLAUDE.md` invariant update, (b) optionally a new `feedback_*.md` memory file, (c) `docs/plans/README.md` index update, (d) any cross-references in related plans. This is not optional polish — it's how the next session learns what shape the codebase enforces.

## Catalog of structural enforcement applied to date

| Plan | Status | Bug class | Layer pushed to |
|---|---|---|---|
| [executor_bootstrap_unification.md](../plans/executor_bootstrap_unification.md) | **Shipped** (2026-04-14) | "forgot to wire ToolPainBridge on path X" | Layer 5: `build_executor(*, pain_bus)` required keyword arg |
| [biosystem_unification.md](../plans/biosystem_unification.md) | **Tracking doc + 5 shells** (2026-04-14) | Same shape applied to PainBus, ReactionBus, MemoryHub, DefaultNetwork, bio_stack umbrella | Layer 5 (planned) |
| [pain_bus_unification.md](../plans/pain_bus_unification.md) | **Audit + builder + migration committed** (2026-04-14, Wave 1 of biosystem_unification) | "forgot to wire `create_pain_nac_subscriber` on CLI path X" — three CLI entry points constructed `PainBus()` with only the memory subscriber, silently skipping NAc bus subscription even though `_cli_nac` was in scope. Out-of-band SEM pain reached hippocampus, never NAc. Tool-invoked pain still reached NAc via `ToolPainBridge` direct attribution, hiding the bug from the substrate P2 cascade test. | Layer 5: `proprioception/pain_bus.py::build_pain_bus(*, hippocampus, nac, additional_subscribers=())` required keyword args. Three CLI sites migrated. Gap B (DefaultNetwork split subscriber ownership) deferred to Wave 2 `memory_hub_unification.md` per the no-band-aid rule — the proper fix couples DN to MemoryHub. Gap C (api.py headless explicit opt-out) keeps its `pain_bus=None` decision; the user-facing API question stays at `agent_factory_canonicalization.md` Stage F5. Raw `PainBus()` construction intentionally still allowed for the ~30 test sites (same precedent as `Executor()` raw surviving the executor unification — structural enforcement lives at the production door, not the type). |
| [reaction_bus_unification.md](../plans/reaction_bus_unification.md) | **Audit + builder + factory fix committed** (2026-04-16, Wave 1 of biosystem_unification) | Construction-door establishment for downstream sequencing (Wave 3 `build_bio_stack` requires `build_reaction_bus` BEFORE `build_pain_bus`). N=1 construction site today so the N-sites-drift pattern doesn't apply — justification is the Wave 3 ordering constraint. Gap A: `cerebellum_modulator_factory` silently dropped `reaction_bus=` parameter; every SEM modulator failure reaction was discarded. Factory has zero production callers but the fix is preemptive. | Layer 5: `reactions/bus.py::build_reaction_bus(*, per_kind_subscribers, all_subscribers, ...)` construction door. `cerebellum_modulator_factory(reaction_bus=)` parameter wired through to `CerebellumModulator.__init__`. |
| [default_network_unification.md](../plans/default_network_unification.md) | **Layer 4→5 upgrade + Gap B closure + migrations committed** (2026-04-16, Wave 2 of biosystem_unification) | Gap A: `build_default_network` was Layer 4 (all optional params, broad exception swallow → `None`). DN construction failure silently dropped pain detection, fear gating, novelty tracking. Gap B (inherited from pain_bus_unification): DN constructed its own PainBus internally with split subscriber ownership — hippocampus wired externally. Gap C: sim orchestrator bypassed the helper entirely with undocumented `object()` stub. | Layer 5: `runtime/bootstrap.py::build_default_network(*, nac)` required kwarg. `pain_bus=` injection inverts DN from bus constructor to consumer (Gap B closure). Sim orchestrator migrated (Gap C). Exception handling narrowed to `ImportError` only. cli.py + api.py documented as explicit headless opt-outs (Gaps D+E → F5). |
| [agent_factory_canonicalization.md](../plans/agent_factory_canonicalization.md) | **Running doc, not scheduled** | "agent constructed via N parallel paths that drift" | Layer 5 + Layer 6 (single canonical AgentFactory) |
| HTTP call sites | **Shipped** (Plan 1 R1, PR #91) | "forgot User-Agent header on outbound HTTP" | Layer 5: `maxim/utils/http.py` registry + endpoint metadata |
| `_MaximPeerBackend.complete_with_usage` | **Shipped** (Plan 3, PR #94) | "added a retry loop, amplified the 52s fail-slow" | Layer 3: CI grep gate `grep -nE "retry\|backoff\|gateway"` blocks new matches |
| `BackendError.fix_hint` | **Shipped** (Plan 2 R2b) | "log injection via user-controlled exception content" | Layer 6: class-level `fix_hint` constants, never user-controlled |
| `RequestContext` propagation | **Shipped** (Plan 1 R1 + Plan 4 Stage A) | "agent_id missing from outbound headers" | Layer 5 + Layer 6: `contextvars.ContextVar` + canonical `_normalize_request_context` shim |
| `MeshConfig.__post_init__` | **Shipped** (Plan 4 C3.1 + C3.2) | "constructed an invalid `MeshConfig` that produced parser-rejecting `mesh.yml` on next read" | Layer 5: dataclass `__post_init__` validates yaml-safe characters + non-empty `nodes` (E7 fold from C3.1) + `self_name in nodes` (A1 fold from C3.2). Both folds were cross-confirmed by 2-of-3 review lenses. |
| `write_mesh_config` caller allow-list | **Shipped** (Plan 4 C3.1 + C3.2) | "future C3 admin-API author writes to `mesh.yml` from runtime code path, breaking the C2 declarative invariant" | Layer 3: CI grep allow-list in `.github/workflows/test.yml` blocks any caller outside `mesh_setup.py` + the test file. Mirrors the `_MaximPeerBackend.complete_with_usage` precedent. CLAUDE.md C2 invariant lesson updated in the same commit per the rule's own requirement. First stress test (C3.2) passed cleanly — see `feedback_strict_grep_caller_allowlist.md`. |
| `atomic_write_secret` wrapper | **Shipped** (Plan 4 C2 + C3.1) | "future credential writer forgets `preserve_mode=True` on `atomic_write_text` and silently widens 0o600 → umask 0o644" | Layer 5: separate function name encodes "this writes a secret" intent at the call site. Operators who write `atomic_write_text(...)` get the unsafe default; operators who write `atomic_write_secret(...)` get mode-preservation + first-write 0o600 chmod with `logger.warning` on chmod failure. Matches the "make the safe path verbose" principle. |

## Cross-references

- [`docs/plans/biosystem_unification.md`](../plans/biosystem_unification.md) — the central catalog of structural-enforcement plans for bio-systems.
- [`docs/plans/executor_bootstrap_unification.md`](../plans/executor_bootstrap_unification.md) — the canonical example of this pattern.
- [`docs/plans/agent_factory_canonicalization.md`](../plans/agent_factory_canonicalization.md) — the global-form follow-up that subsumes the executor-level work into a single agent constructor.
- `CLAUDE.md` "Push silent-no-op invariants into types, not helpers" lesson — the load-bearing version of this rule (one paragraph) for fast recall.
- `feedback_structural_enforcement_over_helper_discipline.md` (memory) — the user-session-persistent memory file capturing the rule.

## When this doc is wrong, fix it

This doc is the canonical reference for the structural-enforcement pattern. If the catalog table drifts from reality, fix the doc. If a new plan opens that fits the pattern, add a row. If the rule needs sharpening based on a future bug-find cycle, refine the threshold rules.
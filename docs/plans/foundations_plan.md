# Foundations Plan — F0 Prerequisites for Substrate Work

**Status:** Active, blocks [substrate_plan.md](substrate_plan.md) P1+
**Target version:** 0.3-pre (lands after Cleanup Wave and peer_leader_flexibility_plan's current waves, before any substrate phase begins)
**Scope:** ~1,130 LOC + tests, ~2 weeks of focused work
**Split from:** substrate_plan.md, where these items originally lived as F0.1–F0.7 inside a larger proof-obligation framework. Split out to concentrate substrate_plan on phase gates and keep the foundation work reviewable as its own wave.

> **⚠ Naming collision with peer_leader_flexibility_plan.** Both plans use "F0" for their own foundation wave. `peer_leader_flexibility_plan` has its own F0.1–F0.5 (filelock primitive, storage reporter, leader pinning, etc. — already landed on main as commits `411d2c0` and `b705504`). **This plan's F0.1–F0.8 is a separate, substrate-oriented foundation wave** — bug fixes and structural refinements that unblock substrate P1+. When reading commit messages or cross-plan references, always check which plan's F0 is meant. Context: substrate's F0 hadn't been scoped when peer_leader_flexibility_plan picked the F0 naming; by the time substrate needed foundation items, the convention was entrenched on that plan's side, and renaming either plan's Fn labels after the fact would be churn. The disambiguation has to live in reader vigilance, not in the label itself.

## Why this is its own plan

The substrate plan is organized around proof-obligation phases — each one ships a falsifiable behavioral claim. The foundation items are different in character: they're **bug fixes and structural refinements** that precede the proof-obligation work. Grouping them with the phases diluted both. As their own plan they have:

- One shared timeline (one wave, seven PRs)
- One CI gate (land all seven before P0 opens)
- One review surface (no need to re-read the substrate plan to evaluate them)
- Explicit standalone value: even if substrate were paused, F0.1 still fixes a failing test, F0.3 still removes a ghost, F0.7 still catches silent corruption

## What these items have in common

Each one fixes a real bug or plugs a load-bearing gap that the substrate phases would otherwise silently build on top of. The principle: **simple refinements now prevent confusing failures later**. Each item is small; the point is to clear the surface before the migration, not to land features.

Two of the seven items (F0.2 PerceptTraceBuffer, F0.5 agent_id threading) correct assumptions that the prior substrate plan got wrong — the prior plan treated these as "already done" when they weren't. Catching them as their own wave is cheaper than discovering them mid-P2.

## Order of operations

Land as eight separate PRs, in dependency order:

1. **F0.1** first — NAc wiring + save/load signature fix; has a failing test to unblock, zero dependencies
2. **F0.3** — NarrativeModulator ghost removal; two-file delete-and-rewire, zero dependencies
3. **F0.7** — tier transition assertions; 80 LOC, zero dependencies, catches corruption early
4. **F0.4** — Percept context schema; prerequisite for F0.5, F0.6, F0.8
5. **F0.5** — agent_id threading + SCN path race fix; depends on F0.4
6. **F0.2** — PerceptTraceBuffer; depends on F0.5 (per-agent isolation)
7. **F0.6** — factory consolidation; depends on F0.4 and F0.5
8. **F0.8** — Sensor→Percept contract (docstring + inject_sensor); depends on F0.4 and F0.5

Each PR: full lint + fast test suite + `mypy` on public API files per CLAUDE.md. Don't batch, don't skip hooks, don't amend across failures.

## F0.1 — NAc wiring + save/load signature ✅ LANDED

**Status:** Complete. Shipped in commit `ee2bb9c` (PR merged 2026-04-11).

**What shipped:** `NAc.save`/`load`/`load_safe` now accept `path: str | None = None` with fallback to `NACConfig.persistence_path` and raise `ValueError` when neither is set. Existing explicit-path callers unchanged. Two unit tests added in [tests/unit/test_nac.py](../../tests/unit/test_nac.py) covering the default-path fallback and the no-path error.

**What was already done before this PR:** The `record_plan_outcome` → NAc observation-counter wiring was repaired during an earlier wave (likely the peer/leader flexibility series); `test_record_plan_outcome` was already green on main when F0.1 opened. No code change needed for part 1 — only the signature alignment (part 2) shipped in this PR.

**Dependencies:** None.

## F0.2 — Shared `PerceptTraceBuffer`

**Design correction from the previous substrate plan.** The prior draft parked eligibility traces on NAc. Traces are really "recent percept activations" — a shared resource that multiple downstream consumers will read from: NAc for reward crediting, later surprise signals, attention weighting, replay schedulers (P8). Owning traces on NAc couples concerns that shouldn't be coupled and makes the substrate harder to extend.

**Minimum implementation:**
- `PerceptTraceBuffer` module — ring buffer of `(agent_id, node_id, tick, activation_strength)` entries with configurable τ decay
- Tick-driven decay + explicit reset semantics
- NAc **reads** from the buffer when crediting reward events; does not write to it
- Per-agent isolation built in (no cross-agent trace bleed)
- Unit tests: decay shape, boundary conditions, reset, per-agent isolation

**Exit:** Unit tests pass. NAc's reward-crediting path consults the shared buffer; no NAc-owned trace state.

**Scope:** ~250 LOC + tests.

**Dependencies:** F0.5 (per-agent isolation requires agent_id threading).

## F0.3 — Kill `NarrativeModulator` ghost

**Bug:** The class is referenced in [dm_runtime.py:304-342](../../src/maxim/simulation/dm_runtime.py#L304-L342) and [cerebellum_modulator.py:43](../../src/maxim/embodiment/cerebellum_modulator.py#L43) but does not exist anywhere in the codebase. NPC `persona_prompt` is extracted and then thrown away.

**Why it matters:** NPCs are indistinguishable at the prompt layer because of this. Substrate Track B (B1 PromptAssembler) will eventually replace this path, but removing the ghost now lets B1 land cleanly without inheriting the broken wiring.

**Exit:** `grep -r NarrativeModulator src/` returns zero hits. NPC persona reaches the LLM through the existing ad-hoc injection path (B1 replaces this path later). A two-NPC sim produces visibly different dialogue (minimal blind A/B test).

**Scope:** ~80 LOC.

**Dependencies:** None.

## F0.4 — `Percept` context schema

**Gap:** The `Percept` dataclass already has `embedding`, `sensory`, `salience`, and `metadata: dict`, but has no schema enforcement for the fields Track A and Track B both need. Current construction sites put channel/sender/thread data in `metadata` inconsistently, or not at all.

**Minimum implementation:** typed `PerceptContext` (or a convention-enforced subset of the metadata dict) with:

```python
channel: Literal["sms", "email", "slack", "narrative", "speech", "self", "internal"]
sender: str | None        # contact ID, NPC ID, "self", "narrator"
thread_id: str | None
subject: str | None
timestamp: float          # monotonic for intra-session, wall-clock for cross-session
latency_class: Literal["realtime", "seconds", "minutes", "hours", "days"]
scn_tag: CircadianContext | None
agent_id: str             # populated by F0.5
```

Also introduce `Percept.modality: Literal["text", "vision", "audio", "intero"]` as an explicit field. `sensory` stays for rich per-modality sub-tags.

**Exit:** Every `Percept` construction site uses the new schema. Type check passes. Unit tests assert round-trip through serialization.

**Scope:** ~200 LOC.

**Dependencies:** None (but F0.5 and F0.6 depend on it).

## F0.5 — Agent ID threading

**Gap:** The multi-agent runtime ([AgentFactory](../../src/maxim/runtime/agent_factory.py), [AgentPool](../../src/maxim/runtime/agent_pool.py)) creates agents with isolated Hippocampus/NAc/ATL instances, but `agent_id` is not consistently threaded through percept construction, NAc reward events, or ATL queries.

**Why it matters:** Substrate P2's per-node reward bias must be keyed by `(agent_id, node_id)` to avoid cross-agent collisions when two NPCs share a concept like "mug." **Without F0.5, P2 is wrong by construction for any multi-agent sim** — and that's a bug we'd only find after P2's unit sim passed cleanly with one agent and then collapsed in a two-NPC campaign.

**Minimum implementation:**
- `agent_id: str` field in `Percept.context` (populated from F0.4)
- Thread `agent_id` through: percept factories → EC queries → ATL lookups → NAc reward crediting → `PerceptTraceBuffer`
- Any bio-system method that mutates shared state takes `agent_id` and asserts it is not None
- **Fix SCN persistence path race at [agent_factory.py:354-363](../../src/maxim/runtime/agent_factory.py#L354-L363):** the late-bound `scn._persistence_path = str(agent_dir / "scn.json")` assignment is unguarded against concurrent agent loads. Move the path into SCN's constructor or guard it under a lock.
- Integration test: two-agent sim with overlapping concept names, verify isolation of reward bias and trace state

**Exit:** Integration test passes. Grep over bio-system mutation methods confirms all accept `agent_id`. SCN path is set at construction, not late-bound.

**Scope:** ~220 LOC + integration test.

**Dependencies:** F0.4.

**Scope clarification — what "per-agent isolation" means.** The iceberg sweep confirmed that [`LocalMessageBus`](../../src/maxim/runtime/agent_pool.py) is intentionally *shared* across agents in an `AgentPool` (this is how agents coordinate). F0.5's isolation claim is about **memory state** (ATL nodes, NAc reward bias, Hippocampus episodes, trace buffer entries) — not the message transport. Two agents in the same pool share the bus but not their bio-stack state. This is the correct design; the plan's multi-agent correctness claim is scoped to memory.

**Context from the iceberg sweep — what the audit actually found about multi-agent isolation (preserved so a cold-start session knows what's already verified):**
- AgentFactory at [agent_factory.py:292-336](../../src/maxim/runtime/agent_factory.py#L292-L336) correctly creates fresh instances per agent: Hippocampus, NAc, ATL, MemoryHub. Each gets a per-agent directory under `~/.maxim/sessions/{agent_id}/`. That part is not the gap.
- **The gap is three-fold:** (a) `Percept.context.agent_id` is not set when percepts are constructed, so downstream consumers can't attribute a percept to a specific agent; (b) NAc reward bias will be keyed by `node_id` alone unless F0.5 changes the key to `(agent_id, node_id)`, so two agents that share a concept ID for "mug" will collide; (c) the SCN persistence path is late-bound.
- **SCN race specifics:** at [agent_factory.py:354-363](../../src/maxim/runtime/agent_factory.py#L354-L363), the code pattern is `scn = SCN(); scn._persistence_path = str(agent_dir / "scn.json")`. If two agents are constructed concurrently (common in AgentPool setup), the interval between `SCN()` and the path assignment is a window where SCN's default persistence path (global or unset) could be used. It's not a likely bug in practice today because agent construction is sequential in current call sites, but it's the exact pattern that bites later when someone introduces parallelism. Fix by moving the path into the SCN constructor: `scn = SCN(persistence_path=str(agent_dir / "scn.json"))`.
- **NAc's current key shape:** NAc tracks causal links via `(event_sig, outcome_sig, context_hash)` tuples (per the iceberg sweep). This is independent of ATL node IDs today. The substrate plan's P2 per-node reward bias introduces a NEW data structure keyed by ATL node ID — that's what F0.5 makes `(agent_id, node_id)` instead. Don't conflate NAc's existing causal-link storage with the new per-node bias.
- **Memory tier invariant:** CLAUDE.md notes "Hippocampus, NAc, and ATL maintain SEPARATE EpisodicMemory instances — this is intentional coexistence, not tech debt." Verified in the audit. Per-agent isolation means these separate instances stay separate *and* are keyed by agent for cross-agent safety.

## F0.6 — `Percept` factory consolidation

**Gap:** `Percept` is currently constructed in ~15 scattered call sites with inconsistent metadata. The substrate P1 text-to-prompt migration adds a `LinguisticEncoder` factory but won't touch the other construction sites.

**Why it matters:** Consolidating *now* means the migration is strictly additive rather than a refactor-in-flight. Otherwise P1's migration has to thread "fix this construction site in passing" work through every PR, which is exactly the pattern that produces merge conflicts and half-done state.

**Minimum implementation:**
- One factory per known source: `make_text_percept`, `make_sensor_percept`, `make_tool_result_percept`, `make_scene_percept`, `make_self_speech_percept`
- Each factory enforces the F0.4 schema and takes `agent_id` from F0.5
- Deprecate direct `Percept(...)` construction outside factories via assertion or lint rule
- Migrate existing construction sites to use factories

**Exit:** `grep -r "Percept(" src/` outside the factories module returns zero results.

**Scope:** ~150 LOC (mostly refactor, not new).

**Dependencies:** F0.4, F0.5.

## F0.7 — Memory tier transition assertions

**Gap:** CLAUDE.md states the memory tier progression is one-way: FORMING → WORKING → SHORT_TERM → LONG_TERM. Today this is documented but not enforced.

**Why it matters:** A bug that skipped or reversed a tier would silently corrupt learning and only surface as a confusing P3a or P3.5 failure. A 30-line assertion prevents a week of debugging.

**Minimum implementation:**
- Assertion guards on the tier-transition path in [memory/store.py](../../src/maxim/memory/store.py) (or wherever transitions live)
- Unit tests for valid transitions + rejection of illegal transitions
- No behavior change on correct transitions; crashes loudly on incorrect ones

**Exit:** Unit tests pass. Running the existing fast suite with assertions in place shows no violations.

**Scope:** ~80 LOC + tests.

**Dependencies:** None.

## F0.8 — Sensor→Percept contract

**Gap:** SEM's sensor layer has the hints of a clean contract — [`SensoryModality`](../../src/maxim/agents/modality.py) enum, `SensoryTag` dataclass, `Sensor` protocol — but none are wired. `SensoryTag` is orphaned (`Percept.sensory` is typed `Any = None` and never populated). Injection is ad-hoc (`inject_cli`, `inject_pain` with different shapes, no generic API). No single file answers "what sensors exist and what fields does each produce?" — the knowledge is scattered across six modules.

**Why it matters:** F0.6 Percept factories need a contract to plug into. P4's vision work needs a discoverable injection pattern. Without this, each new modality is a refactor, not an addition.

**Minimum implementation** (deliberately minimal — discoverability over infrastructure):

1. **Docstring table in [`agents/modality.py`](../../src/maxim/agents/modality.py)** — a markdown table at the top of the module listing every known modality, the fields each carries, and the producer location. Not a registry dataclass, not a schema validation library — just a docstring a developer can read once to understand the full sensor surface. ~20 LOC of prose.

2. **Populate `SensoryTag` in existing producers** (~40 LOC) — `EmbodimentPerceptSource` and `ConversationalSource` both set `Percept.sensory` to a real `SensoryTag` instance. Stop leaving it orphaned. Consumers that key off `percept.source` (string) continue to work unchanged — both fields coexist.

3. **Generic `inject_sensor(modality, **fields)` on `ConversationalSource`** (~40 LOC). Existing methods become thin wrappers:
   ```python
   def inject_cli(self, text, **kw):
       self.inject_sensor("text", content=text, **kw)
   def inject_pain(self, pain_type, intensity, **kw):
       self.inject_sensor("proprioception", pain_type=pain_type, intensity=intensity, **kw)
   ```
   Thin wrappers stay indefinitely. Adding `inject_vision()` later is a direct call to `inject_sensor("vision", ...)`, not a new method.

**Out of scope (deliberately):** no `SensorSchema` dataclass, no registry object, no validation library, no vision-stream integration, no schema versioning. All of those are post-1.0 investments. The minimum goal is *discoverability* — a new developer finds every sensor with one file read — and that's satisfied by the docstring table.

**Vision-stream decision documented:** [`vision_stream.py`](../../src/maxim/embodied_runtime/vision_stream.py) stays as a separate logging subsystem. P4 decides whether to integrate its output into the Percept pipeline via `inject_sensor("vision", ...)` or keep it isolated. F0.8 doesn't do the integration; it makes the integration *possible* by providing the target API.

**Exit:**
- Docstring table in `agents/modality.py` lists text, proprioception, vision (stub), audio (stub).
- Every existing `Percept` producer sets `SensoryTag`; unit test asserts `percept.sensory is not None` for each path.
- `inject_sensor("text", content="hello")` produces a Percept identical to `inject_cli("hello")`.
- Grep for "where is vision defined" finds one file — the docstring table.

**Scope:** ~100 LOC + tests (down from ~320 LOC — the heavy schema-registry version was overengineered; a docstring table satisfies the discoverability goal at a fraction of the cost).

**Dependencies:** F0.4 (Percept context schema must land first so the sensor registry can reference its fields), F0.5 (agent_id threading — sensors flag which agent produced the reading).

## Scope summary

| Item | Scope | Depends on |
|---|---|---|
| F0.1 — record_plan_outcome wiring + NAc save/load signature | ~80 LOC | — |
| F0.2 — PerceptTraceBuffer | ~250 LOC | F0.5 |
| F0.3 — NarrativeModulator ghost | ~80 LOC | — |
| F0.4 — Percept context schema | ~200 LOC | — |
| F0.5 — agent_id threading (+ SCN path race fix) | ~220 LOC | F0.4 |
| F0.6 — Percept factory consolidation | ~150 LOC | F0.4, F0.5 |
| F0.7 — Tier transition assertions | ~80 LOC | — |
| F0.8 — Sensor→Percept contract (docstring + inject_sensor) | ~100 LOC | F0.4, F0.5 |
| **Total** | **~1,130 LOC + tests** | — |

(Note: F0.9 merged into F0.1 as a second NAc fix in the same PR. F0.10 dropped as a foundation item — lands as a one-line bullet in substrate P1 when it's first needed. F0.8 trimmed from ~320 LOC to ~100 LOC.)

## Exit criteria for the wave

The foundations wave is complete when all eight items have landed, the full fast test suite passes, `mypy` passes on the public API files, the previously-failing `test_record_plan_outcome` test is green, `nac.save()` and `nac.load()` accept optional-path calls, and a unit test confirms every `Percept` produced by an existing production code path has a populated `SensoryTag`. Only then does substrate_plan.md P0 open.

## Non-goals

- **No new features.** This is cleanup and prerequisite work. New functionality waits for its substrate phase.
- **No opportunistic refactors.** If a file needs a broader cleanup that isn't one of F0.1–F0.8, log it for a future pass. Foundation work has to stay narrow to ship in under two weeks.
- **No substrate phase work landing in parallel.** The P0 pilot and B1+P1 migration wait until the wave is complete. Parallel work on the same files during foundation fixes defeats the purpose.
- **No centralization of cross-layer wiring through MemoryHub.** The iceberg sweep found that existing direct callbacks (`hippocampus.register_deletion_callback(nac.remove_memory)` at [memory_hub.py:169](../../src/maxim/integration/memory_hub.py#L169)) bypass MemoryHub. The plan's claim that "MemoryHub is the single coordinator" is aspirational, not enforced. **Foundations work accepts this as legacy.** A real mediator refactor is out of scope for substrate and lives in a future plan if it lives anywhere. Individual phases that need new cross-layer wiring can add it through MemoryHub *or* as a direct callback — both patterns are accepted.
- **No ATL callback persistence.** ATL's `_on_concept_captured` and `_on_concept_deleted` at [atl.py:83-84](../../src/maxim/memory/atl.py#L83-L84) are live `Callable` objects that cannot pickle. This is by-design — callers re-register callbacks post-load. Substrate P3.5's persistence round-trip test treats callbacks as a known exclusion.

## If any item fails

- **F0.1 fails** (can't find where observation counter should bump) → investigate the NAc learning path before the substrate plan commits to per-node bias. This is a sign NAc's internal model is not what we think it is; resolve first.
- **F0.2 fails** (trace decay shape is wrong for realistic timescales) → the τ parameter may need to be per-consumer rather than global. Move to a design discussion before shipping.
- **F0.3 fails** (removing the ghost breaks NPC dialogue for a reason we don't understand) → the "dead" reference is load-bearing. Understand why before deleting.
- **F0.4 fails** (too many construction sites to migrate in one PR) → split F0.4 into schema-first + migration-second. F0.6 then depends on both.
- **F0.5 fails** (agent_id is missing from places we can't easily thread it through) → the multi-agent runtime may have a deeper architectural problem. Escalate before moving on.
- **F0.6 fails** (factory consolidation breaks a construction site whose invariants aren't obvious) → take the factory for that site as a separate PR with its own invariant documentation.
- **F0.7 fails** (existing code violates the tier ordering) → that's a pre-existing bug the assertions surfaced. Fix it before the assertions land as CI-blocking.
- **F0.8 fails** (`SensoryTag` population breaks a consumer that silently assumed `sensory is None`) → the failure point tells you exactly which consumer. Fix the consumer to handle a populated tag. If the consumer count is large, split F0.8 into "ship `inject_sensor` without populating SensoryTag yet" (Phase 1) and "populate SensoryTag + update consumers" (Phase 2).

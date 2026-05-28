# Sense tool registry / factory unification

**Status:** 1.0 MVP SHIPPED (2026-05-27) — see "MVP shipment" below. Full plan remains DRAFT for 1.1+. Surfaced by [30_wire_a_tau_validation.md](../experiments/30_wire_a_tau_validation.md) Finding 2. Reframed 2026-05-27 — see "1.0 scope reframe" below.
**Trigger:** Roy-3a-retry NULL outcome — Wire-A annotated `sense_food_source [strongly rewarding from prior experience]` but the tool was silently absent from arm A's active tool roster because the test fixture had no food entity. The LLM had no signal that the tool exists elsewhere.
**Target:** **1.0 (MVP-scoped, shipped)** + 1.1+ (full plan). See "1.0 scope reframe" below.

## MVP shipment (2026-05-27)

W1 MVP shipped on branch `feat/w1-sense-tool-registry-mvp` (~400 LOC src + ~290 LOC tests once the two-lens fold landed). Three phases delivered:

- **Phase 1** — `Tool.auto_fire: bool` + `Tool.kind: Literal[ToolKind]` on the Tool ABC with backward-compatible defaults; `SensePresenceTool` declares `auto_fire=True, kind="auto-discovery"`; SEM-derived factories (`SensorReadTool`, `ModulatorAffordanceTool`, `EntitySenseTool`) declare `kind="sem-modulator-derived"`. `ToolRegistry.register(tool, *, kind=None)` accepts override; new helpers `get_auto_fire_tools()` + `get_tools_by_kind()`.
- **Phase 2** — `runtime/agent_loop.py` auto-sense dispatch iterates `get_auto_fire_tools()` instead of hardcoding `sense_presence`. The actions.jsonl bypass invariant is preserved (direct `tool.execute()` call, never `executor.execute_action`). The interoception block captures the entity-map source by attribute presence (`hasattr(_af_tool, "_entity_map")`) rather than re-introducing the name coupling Phase 2 set out to retire.
- **Phase 3** — Grayscale visibility via `prompts/grayscale_tools_annotation.py::build_grayscale_annotations` (producer filter — strips `tool:` prefix, excludes active + non-SEM tools, caps at top_n=5) + `compose_grayscale_tools_section` (renderer reusing Wire-A's `bias_to_band` for substrate-voice consistency). Section wired via `PromptBuilder._add_grayscale_tools_section` as an adjacent IMPORTANT section. Producer shares Wire-A's `MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION` kill switch — the two surfaces are different views of the same substrate signal.

**Two-lens pre-merge review caught two BLOCK-tier findings folded in-branch:**
- Bio-C1: NAc-shaped `tool:<name>` signatures vs registry's bare-name lookups silently produced an empty grayscale list in production. Extracted the producer to a testable helper that strips the prefix; the regression guard is `tests/unit/test_grayscale_tools_annotation.py::TestBuildGrayscaleAnnotations::test_strips_tool_prefix_before_set_lookup`.
- Bio-C2: Wire-A and grayscale renderers used different intensity bands; the same NAc bias would render with conflicting descriptors in adjacent prompt sections. Composer now delegates band naming to Wire-A's `bias_to_band`.

**Tests:** 25 new (`test_tool_metadata.py` × 10, `test_auto_fire_dispatch.py` × 4, `test_grayscale_tools_annotation.py` × 20 after the fold expanded the Layer-2 producer coverage). Fast suite: 6902 passed, 20 skipped, no regressions.

**Roy-iteration integration:** Phase 4 of the original full plan ("re-run Roy-3a with grayscale visibility for `sense_food_source`") is the next-iteration kickoff after W2 (`imagination_substrate_signals` MVP) also lands. Roy will measure whether grayscale visibility converts substrate annotation to actionable LLM behavior.

**Integration test outcome (2026-05-27, [32_wire_a_post_w1_w2.md](../experiments/32_wire_a_post_w1_w2.md)):** AMBIGUOUS-WITH-WIRING-BUG. The W1 grayscale composer is correctly wired — it's structurally downstream of Wire-A's bias read, sharing `context.cluster_bias_annotations` to avoid a duplicate NAc call. The integration test surfaced an upstream wiring bug at the Roy cross-session boundary: priming saves `cluster_reward_bias` under `agent_id="default_agent"` (MemoryHub default), but the test-arm AUT is constructed with `agent_id="sim_aut"` ([orchestrator.py:534](../../src/maxim/simulation/orchestrator.py)). `_loop_nac.get_agent_tool_biases(agent_id="sim_aut")` returns `[]` and W1's grayscale producer correctly skips when biases are empty — so W1 silently renders nothing in this test path. **W1's MVP code is structurally correct; the test cannot validate it until the upstream agent_id mismatch is fixed.** Re-run blocked on Fix A in the experiment doc.

**Post-Fix-A re-run outcome (2026-05-27, [33_wire_a_post_fix_a.md](../experiments/33_wire_a_post_fix_a.md)):** W1's grayscale visibility worked as designed — Arm A's active tool roster excludes `sense_food_source` (no food entity in the roy_1_holdout fixture), so W1 surfaces the tool in the grayscale section as `[strongly rewarding from prior experience] [not in current location]`. The LLM read the annotation and **did not act on the inactive-tool signal** — no `sense_food_source` call (correct; tool unavailable), no `sense_tools` query for food-related affordances (despite having access), no `examine` for food in scene. W1 is structurally correct and operator-visibility is validated; behavioral impact requires either the tool becoming active in the test arm (Fix B — extending W2 to fixture scene-load brings a food entity into scene) OR the LLM developing the cognitive ability to use grayscale signals to reach for substitutes (cross-modal binding / JEPA territory). The Fix B vs encoder-pivot decision is the next 1.0 scoping question.

## 1.0 scope reframe (2026-05-27)

Originally targeted as 1.1+. Reframed during the post-Phase-C strategic discussion: **this plan is 1.0 critical path because the substrate→action conversion question is the 1.0 thesis bottleneck.** Roy-3a-retry showed Wire-A annotation reaches the LLM with strong magnitude but cannot convert to action because the substrate-favored tool isn't in the active scene roster. Until that gap closes, Roy iterations stay structurally capped at "annotation is present" findings.

**1.0 MVP scope (smallest unit that closes the Roy-3a gap):**

- **Grayscale visibility minimum** — `tools_block` rendering distinguishes always-active core tools from SEM-derived inactive tools, with `[not in current location]` tag for the latter. Inactive SEM tools the substrate has accumulated bias for (per `NAc.get_agent_tool_biases`) appear in the LLM-facing list.
- **Tool metadata foundation** — `auto_fire: bool` field on the Tool dataclass (declarative replacement for the implicit executor bypass at [agent_loop.py:1292](../../src/maxim/runtime/agent_loop.py)). No behavior change for existing tools; the bypass discipline becomes explicit.
- **Registration-time classifier** — single `ToolRegistry.register(tool, kind=...)` accepting one of `{core-universal, auto-discovery, scene-scoped, sem-modulator-derived}` with sensible defaults for existing call sites.

**Deferred to 1.1+ (full plan, post-Roy-iteration verdict):**
- `sensory_events.jsonl` separation from `actions.jsonl`.
- LRU eviction tuning.
- NAc predicate-outcome typing (auto-sense → predictive learning channel).
- Description unifier for SEM affordance descriptions.

The MVP is ~150-200 LOC + ~150 tests; full plan was sized ~400 LOC. Phase 4 of the original plan (Roy-3a re-run) becomes the integration test for the 1.0 MVP.

## Front-gate scope pressure (retroactive)

Added 2026-05-27 per CLAUDE.md Principle 3.

**Question:** does this need to be its own mechanism, or can it ride on existing infrastructure?

**Existing infrastructure surveyed:**

| Candidate | Why insufficient |
|---|---|
| `ToolRegistry` ([tools/registry.py](../../src/maxim/tools/registry.py)) — scene-scoped activation + active tool cap | Active-list machinery is already there; what's missing is *grayscale visibility for inactive tools*. ToolRegistry only models active/inactive as binary visibility |
| `PromptBuilder` `tools_block` rendering | Reads ToolRegistry's active list; would need a new metadata-aware render branch to emit `[GRAYSCALE]` rows. Render-side change rides on PromptBuilder; *what to render* needs the new metadata layer |
| Executor auto-fire bypass ([agent_loop.py:1292](../../src/maxim/runtime/agent_loop.py)) | Currently implicit-in-dispatcher-path; would still need declarative `auto_fire=True` metadata on the tool for the unification to work |
| `ComponentIndex` two-layer discovery (alias + embedding) | Solves entity-name discovery, not tool-presence visibility. Wrong abstraction layer |
| `ToolOutput.side_effects` typed channel (CLAUDE.md L82) | Canonical bio-pipeline signal channel from-tool-to-bio-pipeline. Wrong direction — auto-sense needs a from-tool-to-event-log channel, which is a separate event-type contract |
| `sim_log` event types | Auto-fired output could route to a new `sensory_events.jsonl` via `sim_log`; but separating from `actions.jsonl` needs a new event-type contract, not just a render change |

**Verdict:** yes-it-needs-to-be-its-own (small but real). The new piece is **tool metadata** (`auto_fire`, `scene_scoped`, `grayscale_when_inactive`, `sensory_event_log`) plus a metadata-aware render branch in `tools_block`. The metadata cannot be derived from existing registration sites because the same registry currently mixes three regimes (universal / auto-discovery / SEM-derived) without a discriminator.

**Specific reason:** ToolRegistry's active/inactive model has no slot for "inactive but knowable" — adding the slot is the smallest change that closes the Roy-3a-retry symptom without restructuring ToolRegistry or the auto-fire executor bypass.

## Why this plan exists

The sense* tool family has accumulated heterogeneity over three independent design moments (universal-sense was a Phase 0 add, auto-sense was a perception-hygiene fix, SEM-derived sense came with embodiment). Each is correct in isolation; together they produce an LLM-facing surface where conceptually-similar tools behave very differently:

| Group | Examples | Where registered | Auto-fire? | LLM-visible when inactive? |
|---|---|---|---|---|
| Universal/core | `sense`, `sense_tools` | [orchestrator.py:754-830](../../src/maxim/simulation/orchestrator.py) once at boot | No | n/a (always active) |
| Auto-discovery | `sense_presence` | [orchestrator.py:754-830](../../src/maxim/simulation/orchestrator.py) once at boot | **Yes** ([agent_loop.py:1292](../../src/maxim/runtime/agent_loop.py) executor bypass) | n/a |
| SEM-modulator-derived | `sense_<entity>`, `read_<entity>_<sensor>`, `sense_food_source` | [tool_bridge.py:405](../../src/maxim/embodiment/tool_bridge.py) per-entity on scene load | No | **No — silently invisible** |

The Roy-3a-retry symptom: `sense_food_source` was registered + invoked 135× during priming (food entity in scene), then silently vanished at test time because the Roy-1 holdout fixture has no food entity. Arm A's logged active tool roster contains the three universal/auto sense tools but no SEM-derived ones. **Wire-A correctly annotated the tool as substrate-rewarding; the LLM had no way to call it because the LLM-facing tool list doesn't include inactive-but-knowable tools.**

## The architectural smell

Two registration regimes for the same conceptual category isn't itself a bug — it's how the codebase grew. The smell is at the **LLM-facing layer**:

- The LLM prompt's `tools_block` lists active tools without distinguishing universal-always-available from scene-scoped-might-not-be-here-now.
- When a SEM-derived tool is unavailable, there's no operator-visible signal to the LLM that it could exist elsewhere (in a prior scene, in a future scene, in a dream).
- The LLM cannot reason about a tool it can't see, even when Wire-A is rendering "this tool was strongly rewarding."

A unified Sense Registry / Factory could close this by:
- Formalizing the auto-fire vs LLM-callable distinction as **tool metadata** rather than implicit-in-dispatch-path behavior.
- Adding **grayscale visibility**: inactive scene-scoped tools appear in the tools_block with an `[not in current location]` tag so the LLM knows they exist.
- Routing auto-sense to a **separate sensory event log** (not actions.jsonl) so NAc can learn "in context X, state Y is likely" predictively without polluting causal action links.

## Load-bearing invariants (DO NOT BREAK)

Surfaced by the architecture review in [30_wire_a_tau_validation.md](../experiments/30_wire_a_tau_validation.md):

1. **Auto-sense must not log to `actions.jsonl`.** Auto-fired sense_presence + self-sense are passive perception, not chosen actions. Logging them as actions would corrupt NAc's causal model with phantom links. The current executor bypass at [agent_loop.py:1292](../../src/maxim/runtime/agent_loop.py) is correct *by intent*; the smell is that the bypass is implicit in the dispatcher path. Any unification must keep auto-sense out of `actions.jsonl` while making the bypass declarative on the tool (e.g., `auto_fire=True` metadata).

2. **SEM-derived tools must stay scene-scoped at execution time.** A `dragon_slash` tool in a forest with no dragons is wrong to *invoke*. But scene-scoping at *invocation* is different from scene-scoping at *visibility* — the unification adds grayscale visibility for inactive tools (LLM can see them) while preserving the invocation guard (LLM can't successfully call them when target entity is absent).

3. **LRU eviction must apply to scene tools only, not core tools.** Already enforced at [discovery.py:67](../../src/maxim/tools/discovery.py). Don't regress.

4. **ModulatorAffordanceTool sensor-delta feedback must survive.** [tool_bridge.py:297-313](../../src/maxim/embodiment/tool_bridge.py) wires sensor deltas into the cerebellum forward model + pain bridge. Unification cannot disrupt this — it's the substrate's primary motor-learning channel.

5. **Per-phrase design guard + shared energy budget are race-safety, not load-irrelevant.** Cross-referenced with [imagination_substrate_signals.md](imagination_substrate_signals.md)'s DO-NOT-TOUCH section.

## Sketch of the contract surface

(Not a code proposal — just the contract pieces a unification would need to nail down.)

**Tool metadata:**
- `auto_fire: bool` — fires every tick automatically, bypassing executor / actions.jsonl.
- `scene_scoped: bool` — registered/unregistered as scene entities enter/exit.
- `grayscale_when_inactive: bool` — appears in LLM tools_block with `[not in current location]` tag instead of being hidden entirely.
- `sensory_event_log: bool` — auto-fired output routes to a separate `sensory_events.jsonl` instead of `actions.jsonl`.

**Registration interface:**
- Single registry entry point that accepts the metadata above + classifies tool into one of {core-universal, auto-discovery, scene-scoped, sem-modulator-derived}.
- Each classification is a *combination* of metadata flags, not its own code path.

**Visibility interface:**
- The prompt builder's tools_block renderer reads each registered tool's `grayscale_when_inactive` flag and renders either active-list, grayscale-list, or hidden.
- The LLM gets one combined list, with structure: `[ACTIVE] sense(entity_name) — read all sensors on an entity / [GRAYSCALE] sense_food_source — was strongly rewarding in prior scenes`.

**Execution interface:**
- All sense tools dispatch via the executor (uniform path), EXCEPT for the auto-fire bypass which is gated on `auto_fire=True` metadata read at dispatch time.
- Auto-fire writes to `sensory_events.jsonl` for NAc predictive-learning consumption; never writes to `actions.jsonl`.

**NAc attribution:**
- Auto-fire outcomes are typed `predicate_outcome` (predictive), distinct from `tool_outcome` (causal).
- This separation lets NAc learn "in environment X, state Y is likely" without confusing it with "I chose action Z and outcome Y occurred."

## Phasing

Not detailed at this DRAFT stage. The natural shape is:

- **Phase 0** — design pass + this plan refinement. Validate the contract surface against all 7 sense tools + the load-bearing invariants. Estimate LOC.
- **Phase 1** — tool metadata + registry refactor. Migrate existing tools to declare their metadata; keep behavior identical. Regression-test suite + bio-pipeline integration tests.
- **Phase 2** — grayscale visibility in `tools_block`. New prompt section + LLM-side prompt engineering. Validate with a small sim showing the LLM responding to grayscale tools sensibly (and not hallucinating they're invokable).
- **Phase 3** — `sensory_events.jsonl` + NAc predicate-outcome wiring. Auto-sense outputs route to the new log; NAc subscribes for predictive learning.
- **Phase 4** — re-run Roy-3a with grayscale visibility for `sense_food_source`. Measure whether the LLM uses the grayscale signal to reach for related active sensing tools (e.g., asking `sense_tools("food")` to find an equivalent).

## What this NOT solves

- Imagination substrate-blindness (the other half of the Roy-3a verdict). That's [imagination_substrate_signals.md](imagination_substrate_signals.md). The two plans are complementary: this one lets the LLM *see* what's not in scene; that one lets the substrate dream the missing entity into scene.
- The tick-anchored decay bio-fidelity gap. That's the planned `scn_decay_anchoring.md` (Phase C of the tau-split kickoff, not yet drafted).
- ModulatorAffordanceTool's universal-parametric-vs-per-affordance split (a separate move-family standardization question surfaced by the architecture review but not load-bearing for Roy-3a).

## Authorization gate

Drafted as `feat/wire-a-tau-split-phase-3-validation` branch fold. Phase 0 design pass starts on explicit user authorization; not currently a 1.0 gate. If the user prioritizes this over imagination-substrate-signals, the Phase 0 design pass should land first since the visibility side is the cheaper proof-of-value.

## Open questions

1. Should grayscale visibility include ALL inactive SEM-derived tools, or only those that have Wire-A annotation (i.e., the substrate cares about them)? The latter is cheaper at prompt-budget but means the cold-start agent never sees the surface. The former gives the agent more affordance awareness but blows up prompt size in busy scenes.
2. Does the `sensory_events.jsonl` separation require schema-version migration on existing NAc snapshots? Probably no (it's a new outcome type), but verify.
3. What's the migration story for third-party tool authors who subclass `Tool` directly? The metadata additions need defaults that preserve current behavior so external tools don't break silently.

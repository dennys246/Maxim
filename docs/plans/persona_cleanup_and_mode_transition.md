# Persona cleanup + orchestrator mode transition

**Status:** PARTIAL — Stage 1 shipped 2026-04-30 (PR #217). Stages 2-6 are 1.1+ deprecation cleanup work.
**Ships in:** Stage 1 in 1.0 (additive `--mode` flag + deprecation warnings on `--persona` and `register_persona`); Stages 2-6 in 1.1 (resolve testing strategy, dispatch hook migration, public API migration, hard-delete the persona system, docs+memory).
**Owns:** [src/maxim/simulation/personas.py](../../src/maxim/simulation/personas.py), [src/maxim/cli_parser.py](../../src/maxim/cli_parser.py), [src/maxim/simulation/orchestrator.py](../../src/maxim/simulation/orchestrator.py), public API surface in [src/maxim/api.py](../../src/maxim/api.py)
**Companion plans:** [bio_emergent_persona_foundations.md](bio_emergent_persona_foundations.md) (the wire work this cleanup makes room for), [persona_convergence_crucible.md](persona_convergence_crucible.md) (the long-term living doc)

## Front-gate scope pressure (retroactive)

Added 2026-05-27 per CLAUDE.md Principle 3.

**Question:** does the `--mode` flag introduction need to be its own mechanism, or can it ride on existing infrastructure?

**Note:** this is primarily a **cleanup plan** (delete dead `personas.py`), not a new-mechanism plan. The front-gate question only really bites for the *additive* `--mode` flag piece. The dead-code deletion is pure removal — no mechanism to scope-press.

**Existing infrastructure surveyed (for the `--mode` flag):**

| Candidate | Why insufficient (or sufficient) |
|---|---|
| Existing `--persona` flag dispatch (research_orchestrator + YAML campaign detect + ResearchResult.persona) | **Already does dispatch wearing persona's hat** — the load-bearing semantics. The cleanup migrates these explicit dispatch sites to `--mode` (rename, not new mechanism) |
| Existing argparse + `cli_parser.py` infrastructure | Already handles all CLI flags; `--mode` is one more flag with deprecation-warning behavior on the predecessor |
| Existing back-compat surface (`SimulationResult.persona` field on persisted `session.json`) | **Must preserve** — read policy stays: missing `mode` field → `mode="generative"`. No new persistence mechanism |
| `_orchestrator_strategy` flag (Option B in the open decision) | Would re-introduce LLM-prompt-injection under a new name. Same inauthenticity the cleanup is removing. Rejected on principle |

**Verdict:** could-ride-on-existing. The `--mode` flag is a rename of dispatch semantics already in the codebase wearing the wrong abstraction. No new mechanism — just relabeling the load-bearing dispatch to its honest name and deleting the never-injected `context_prompt` scaffolding.

**Specific reason:** all three dispatch sites (`--research`, YAML campaign detect, `SimulationResult.persona`) already encode mode semantics through persona-naming. The mechanism exists; it's just under a misleading name. The cleanup is structural honesty, not new functionality.

## Context

The current `--persona` flag and `personas.py` module ship a feature whose stated purpose — shaping orchestrator behavior via persona-specific `context_prompt` strings — **is not actually wired**. Grep for `.context_prompt` returns zero injection sites. The 80-line adversarial prompt, the 200-line refinement prompt, the 287-line sweep prompt — none of them are read by the orchestrator. The persona name flows into reports and logs as a label and that's it.

This is inauthentic for a bio-inspired repo on two fronts:

1. **It promises behavior it doesn't deliver.** Users reading `personas.py` would reasonably expect those rich prompts to be active.
2. **It encodes "personality" as a prompt-injection scaffold rather than as something that emerges from the bio-systems.** The whole framing of the project — that identity emerges from learned experience through Hippocampus, NAc, ATL, and the rest — is contradicted by a `--persona adversarial` flag that just relabels an LLM role-play.

The cleanup deletes the prompt-injection scaffold, surfaces the dispatch logic that was hiding inside persona names as real `--mode` semantics, and clears the way for a separate plan ([bio_emergent_persona_foundations.md](bio_emergent_persona_foundations.md)) that wires the bio-systems to actually shape behavior.

This plan is a **cleanup**, not a feature. Its success criterion is "the codebase no longer claims to do something it doesn't" — not "we shipped a new capability."

## Audit findings

### What's actually dead (clean delete)
- `Persona.context_prompt` — never injected anywhere. Grep verified.
- `Persona.max_initiative` — never read in simulation code.
- `sweep`, `refinement`, `adventure_architect` personas — declared, never dispatched, never consumed beyond name.
- `/persona <name>` slash command at [orchestrator.py:1693](../../src/maxim/simulation/orchestrator.py#L1693) — injects a text nudge into the LLM, mutates no state.

### What's load-bearing as dispatch (must migrate, not delete)
- `--research` hardcodes `persona="researcher"` at [research_orchestrator.py:150](../../src/maxim/simulation/research_orchestrator.py#L150) — real dispatch wearing persona's hat.
- YAML campaign detection hardcodes `persona="dungeon_master"` at [cli.py:1308](../../src/maxim/cli.py#L1308) — same pattern.
- `SimulationResult.persona` at [sim_types.py:27](../../src/maxim/simulation/sim_types.py#L27) is persisted to disk in `~/.maxim/sessions/*/session.json` — back-compat surface.

### What looks like persona but isn't (do not touch)
- `persona_prompt:` fields on individual NPCs in [scenarios/campaigns/*.yaml](../../scenarios/campaigns/) — narrative metadata for individual characters, fully wired through DM runtime, **separate concept** despite the name collision.
- `AgentConfig.personality` in [test_agent_factory.py](../../tests/unit/test_agent_factory.py) — agent character creation, separate from orchestrator personas.

### The four "test strategies"
`adversarial`, `cooperative`, `confused`, `escalating` are testing-shape prompt strings that have the same dead-code problem (their `context_prompt` is never injected) but which encode a real distinction users may want to preserve: "I want the orchestrator to probe adversarially vs cooperate naturally." Whether to keep this category is the **open decision below**.

## Open decision (owner: user, must resolve before Stage 3)

What happens to the testing strategies (`adversarial`, `cooperative`, `confused`, `escalating`)?

### Option A — delete entirely
Remove with the rest of `personas.py`. Orchestrator becomes "neutral interrogator" by default. Research orchestrator partitions sessions by `--mode` + goal-cluster instead of by persona. **Most consistent with the bio-inspired stance** since prompt-injected adversariality is the same family of inauthenticity we're cleaning up. Loses some research-session partitioning ergonomics.

### Option B — promote to real `--orchestrator-strategy` flag
Implement them as composable prompt fragments that are *actually injected* this time, fixing the false promise rather than removing it. Keeps the user-facing surface intact. **Keeps a category of LLM-prompt-injection alive** that the cleanup is trying to remove on principle. Reasonable but inconsistent.

### Option C — fold strategy semantics into goal text
"Test that the agent handles adversarial users" becomes the goal string itself; orchestrator just executes the goal honestly. Cleanest user surface; loses the named-strategy dimension.

**Recommended:** Option A. Cleanest, most consistent with the project's framing. The cost (loss of named partitioning) is recoverable via goal-tag clustering when the research orchestrator needs it.

This plan is written assuming Option A. Stages 3-5 branch if you pick B or C.

## Stage 0 — Baseline + back-compat snapshots (~1hr)

Before any deletion:
- Run three representative sims and save outputs as regression goldens to `tests/fixtures/persona_migration/`:
  - `maxim --sim "test memory recall" --interactive false --sim-max-turns 5`
  - `maxim --sim scenarios/campaigns/heist_v1.yaml --interactive false --sim-max-turns 5`
  - `maxim --sim "test safety" --research --interactive false --sim-max-turns 5`
- Snapshot `report.json`, `actions.jsonl`, `aut_nac.json`, `aut_hippocampus.json` from each.
- Read the schemas in `~/.maxim/sessions/*/session.json` to identify every consumer of `metadata["persona"]`. Document the back-compat read policy: missing field → `mode="generative"`, no error.

## Stage 1 — Introduce `--mode` flag alongside `--persona` (additive, non-breaking)

Add new flag at [cli_parser.py](../../src/maxim/cli_parser.py):

```
--mode {generative|dm|research|benchmark|repl}
```

Default: inferred from existing dispatch (preserves current behavior).

- Both flags accepted simultaneously. If both present and inconsistent (`--mode dm --persona researcher`), emit a single-line WARN and prefer `--mode`.
- `--persona` use emits a deprecation warning pointing at `--mode` (for the four mode-like personas) or noting impending removal (for testing strategies, per Option A).
- Plumb `--mode` through dispatch sites:
  - [cli.py:1258](../../src/maxim/cli.py#L1258) (research)
  - [cli.py:1290](../../src/maxim/cli.py#L1290) (DM detection)
  - [cli.py:1331](../../src/maxim/cli.py#L1331) (generative default)
- Public API: add `mode=` parameter to `imagine()`, `campaign()` lazily; keep `persona=` working with deprecation warning during this stage.

**Validation:** replay Stage-0 sims with both flag forms (`--persona researcher` and `--mode research`); diff outputs — only allowed differences are deprecation-warning lines.

## Stage 2 — Resolve the open decision

Per the Option A/B/C decision above. If A (recommended), no new code in this stage — just commit to deleting the testing-strategy concept in Stage 5. If B, implement real `--orchestrator-strategy` flag with prompt-fragment injection at [orchestrator.py:1620](../../src/maxim/simulation/orchestrator.py#L1620) (the actual prompt assembly site). If C, no code — strategy semantics become goal-text guidance documented in CLI help.

## Stage 3 — Migrate dispatch hooks

Strip persona-as-dispatch from the codebase:

- [research_orchestrator.py:150](../../src/maxim/simulation/research_orchestrator.py#L150): replace hardcoded `persona="researcher"` with mode-aware orchestrator instantiation. `start_research_mode` no longer takes `persona` param.
- [cli.py:1308](../../src/maxim/cli.py#L1308): YAML detection sets `mode="dm"`, no longer `persona="dungeon_master"`.
- [orchestrator.py:1590, 1620](../../src/maxim/simulation/orchestrator.py#L1590): replace `f"...with the '{persona}' persona"` with mode-aware system-prompt fragments. The mode-specific instructions ("OBSERVE ONLY", "CAMPAIGN PROTOCOL", "FIRST send_message") that already exist as real code at lines 1600-1615 become the canonical mode→prompt mapping.
- [orchestrator.py:1693](../../src/maxim/simulation/orchestrator.py#L1693): the `/persona` slash command — delete (Option A) or rename to `/strategy` (Option B) or delete (Option C).

**Validation:** replay Stage-0 sims with `--mode` forms; diff against goldens. Behavioral equivalence required for `--mode generative` ≈ old default; `--mode dm` ≈ old `--persona dungeon_master`; `--mode research` ≈ old `--research`.

## Stage 4 — Public API + scenario YAML migration

- `pymaxim.imagine(persona=...)`, `pymaxim.campaign(persona=...)`: `persona` becomes deprecated kwarg, ignored after warning. `mode=` is the new path.
- `register_persona()` at [api.py:1662](../../src/maxim/api.py#L1662): deprecate. Per Option A, delete in Stage 5.
- [scenarios/refinement_baseline.yaml:10](../../scenarios/refinement_baseline.yaml#L10) and benchmark test fixtures ([test_benchmark_runner.py:484](../../tests/unit/test_benchmark_runner.py#L484), [test_benchmark_phase0.py:529](../../tests/unit/test_benchmark_phase0.py#L529)): scrub `persona:` keys. Replace with `mode:` where dispatch was implied; drop where ornamental.
- **Do not touch** `persona_prompt:` on NPCs in `scenarios/campaigns/*.yaml` — separate concept.

## Stage 5 — Delete + verify

- Delete [src/maxim/simulation/personas.py](../../src/maxim/simulation/personas.py) (~410 LOC).
- Strip `--persona` flag from [cli_parser.py:344](../../src/maxim/cli_parser.py#L344).
- `SimulationResult.persona` → `SimulationResult.mode` and (if Option B) `SimulationResult.strategy`. Reader in [session.py:238](../../src/maxim/session.py#L238) treats missing fields on old sessions as `mode="generative"`, `strategy=None`.
- Strip `register_persona` from `__all__` and `_API_VERBS` in [src/maxim/__init__.py](../../src/maxim/__init__.py).
- Delete `TestPersonas` class in [test_simulation_agent.py:410-464](../../tests/unit/test_simulation_agent.py#L410). Update other tests per migration map (see audit findings above).
- Replay Stage-0 baseline sims; diff `report.json` and `actions.jsonl` against goldens. Only allowed differences: `persona` → `mode` field rename on persisted records.

## Stage 6 — Docs + memory

- Update [docs/simulation.md](../../docs/simulation.md) — remove persona table at lines 83-96, replace with mode reference.
- Update [docs/user/cli-reference.md](../../docs/user/cli-reference.md) — `--persona` → `--mode`.
- Update [CLAUDE.md](../../CLAUDE.md) — examples at lines 213, 217, 320; module list.
- Archive `docs/archive/dungeon_master_persona.md` (already archived, leave).
- Update [project_framing_strategy.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_framing_strategy.md): "personality emerges from bio-systems, not from prompt injection — orchestrator modes are flow-shapes, not character."

## Risks + back-compat surface

| Risk | Mitigation |
|---|---|
| Old session JSONs with `metadata["persona"]` fail to deserialize | Reader at [session.py:238](../../src/maxim/session.py#L238) treats missing field as default `mode`; never raises |
| Users with scripts pinning `--persona X` break | Stage 1 keeps both flags valid; deprecation warning lands one minor version before removal in Stage 5 |
| Confusion between deleted orchestrator persona and surviving NPC `persona_prompt` | Docs explicitly call out the distinction; YAML schema docstring updated |
| Research orchestrator loses persona-keyed partitioning (Option A) | Replace with mode + goal-tag clustering at write time; minor loss of ergonomics, no functionality lost |

## Sizing

| Stage | Estimated LOC | Files touched |
|---|---|---|
| 0 (baseline) | 0 | new test fixtures only |
| 1 (--mode introduction) | ~150 add | cli_parser.py, cli.py, api.py, __init__.py |
| 2 (decision resolution) | 0 (A or C); ~200 (B) | orchestrator.py if B |
| 3 (dispatch migration) | ~100 add, ~100 delete | research_orchestrator.py, cli.py, orchestrator.py |
| 4 (API + YAML migration) | ~50 add, ~30 delete | api.py, scenarios, test fixtures |
| 5 (delete + verify) | ~100 add, ~600 delete | personas.py, cli_parser.py, sim_types.py, session.py, tests |
| 6 (docs) | ~50 doc lines | simulation.md, cli-reference.md, CLAUDE.md |
| **Total** | **~400 add, ~730 delete (Option A)** | |

**Estimated calendar:** 1-2 days concentrated work.

## Definition of done

- `grep -r "persona" src/maxim/` returns only:
  - NPC `persona_prompt` references (different concept)
  - Back-compat reader code with explicit "legacy" comments
  - Deprecation warning text
- All Stage-0 baseline sims replay with `--mode` forms and produce diff-clean goldens.
- `pymaxim` import surface no longer includes `register_persona` or `Persona`.
- [CLAUDE.md](../../CLAUDE.md) examples + docs reflect new `--mode` flag.
- Pre-merge two-lens review (Executor + Architecture per [feedback_review_before_ship.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_review_before_ship.md)) finds no blocking issues.

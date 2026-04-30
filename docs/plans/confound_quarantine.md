# Confound quarantine — substrate-only V1 baseline

**Status:** draft, pre-implementation
**Ships in:** 0.9.x (experimental); flag lifecycle decided in 1.0 conditional on Phase A outcome — see "Risks" §R1 for branches
**Owns:** prompt-injection gates in [src/maxim/agents/prompt_builder.py](../../src/maxim/agents/prompt_builder.py), [src/maxim/agents/exec_prompts.py](../../src/maxim/agents/exec_prompts.py), [src/maxim/prompts/acting_coach.py](../../src/maxim/prompts/acting_coach.py); orchestrator state path in [src/maxim/simulation/orchestrator.py](../../src/maxim/simulation/orchestrator.py); persona default in [src/maxim/simulation/personas.py](../../src/maxim/simulation/personas.py); arc routing in [src/maxim/cli.py](../../src/maxim/cli.py); env-var contract doc in [docs/user/configuration.md](../user/configuration.md) under CC4
**Companion plans:** [v1_refinement.md](v1_refinement.md) §V2 + §CC4, [persona_cleanup_and_mode_transition.md](persona_cleanup_and_mode_transition.md)
**Branch:** `confound-quarantine`

## Motivation

A multi-lens audit of the V1 cross-session validation experiment found **five default-on prompt/state systems** that fire on every `maxim --sim` invocation and contaminate the substrate-attribution claim. The persona system was already known. The other four are not gated by any flag today:

| # | Confound                              | Site                                                                 | Trigger                                       |
|---|---------------------------------------|----------------------------------------------------------------------|-----------------------------------------------|
| 1 | PFC deliberation preamble (~1k tok)   | [exec_prompts.py:13](../../src/maxim/agents/exec_prompts.py#L13)     | `_sim_active or any(bio_signal)`              |
| 2 | Acting Coach + embodied identity      | [prompt_builder.py:117](../../src/maxim/agents/prompt_builder.py#L117), [acting_coach.py:40](../../src/maxim/prompts/acting_coach.py#L40) | `request.acting_coach is not None`            |
| 3 | "SIMULATION ENVIRONMENT" sandbox text | [prompt_builder.py:148](../../src/maxim/agents/prompt_builder.py#L148) | `_sim_active`                                 |
| 4 | Orchestrator NPC global state         | [orchestrator.py:1181](../../src/maxim/simulation/orchestrator.py#L1181) (`_data_home() / "orchestrator"`) | every sim — saved on `_orch_instance.shutdown()` |
| 5 | Persona default `adversarial`         | [personas.py:352](../../src/maxim/simulation/personas.py#L352)       | `--persona` not passed                        |

Until each of these can be turned off independently, the V1 1.0-anchor result is **not attributable to the substrate**. This plan adds opt-in disable flags, isolates orchestrator state via the existing `MAXIM_DATA_HOME`, and defines a phased re-run protocol that decomposes the V1 delta across contributors.

## Design principles

1. **Defaults preserve current behavior.** Every flag is opt-in disable. A 1.0 user who runs `maxim --sim` with no env vars gets exactly today's behavior.
2. **Centralized gate logic** in a new module `src/maxim/runtime/confound_flags.py`. Each injector imports a single helper (e.g. `pfc_preamble_enabled()`) rather than reading `os.environ` itself. This (a) keeps env var names typo-safe, (b) gives one grep target for "all the substrate-attribution gates," and (c) makes the autouse scrub pair-up deterministic — every flag in the module gets a scrub, no exceptions.
3. **Universal isolation prefers `MAXIM_DATA_HOME=$(mktemp -d)` over per-subsystem knobs.** `data_home()` already routes orchestrator/util/sessions/memories. The harness uses one env var; no new orchestrator-specific knob.
4. **All new flags are debug/experimental per CC4.** None enter the public env-var contract. Doc updates land in `docs/user/configuration.md` under the existing "Debug — substrate + decision-system experiments" section.

## Flag inventory

All flags follow the existing CC4 naming style (`MAXIM_<UPPER_SNAKE>=1` to enable). Defaults preserve current behavior; `=1` disables the named injector.

| Env var                              | CLI flag                | Default | CC4 class    | Gate site                                              | Semantics                                                                                |
|--------------------------------------|-------------------------|---------|--------------|--------------------------------------------------------|------------------------------------------------------------------------------------------|
| `MAXIM_DISABLE_PFC_PREAMBLE`         | (no CLI flag)           | unset   | experimental | [prompt_builder.py:1008](../../src/maxim/agents/prompt_builder.py#L1008) inside `_add_pfc_preamble_section` | When `=1`, skip the `budgeter.add("pfc_preamble", ...)` call; the ~1k-token deliberation scaffold is omitted. |
| `MAXIM_DISABLE_ACTING_COACH`         | `--no-acting-coach`     | unset   | experimental | [prompt_builder.py:120](../../src/maxim/agents/prompt_builder.py#L120) (identity rewrite) **and** the analogous early-return in `_add_acting_coach_section` near line 1013 | When `=1`: identity stays "Maxim, a robot assistant" and the acting-coach section is omitted, regardless of `request.acting_coach`. CLI flag because this materially changes embodied behavior — researchers will want a flag, not just an env var. |
| `MAXIM_DISABLE_SIM_SANDBOX_TEXT`     | (no CLI flag)           | unset   | experimental | [prompt_builder.py:150](../../src/maxim/agents/prompt_builder.py#L150) | When `=1`, skip the `SIMULATION ENVIRONMENT: …` block; INTERACTIVE MODE block is unaffected. |
| `MAXIM_NO_DEFAULT_PERSONA`           | `--no-persona`          | unset   | experimental | wherever `DEFAULT_PERSONA` is consumed in `cli.py` (and any other dispatch site found via `grep DEFAULT_PERSONA`) | When `=1`, treat absent `--persona` as `None` (true neutral) instead of falling back to `adversarial`. CLI flag because `--persona` already has CLI-flag ergonomics; symmetry helps. |
| `MAXIM_DATA_HOME=<tmpdir>`           | (no CLI flag — existing) | `~/.maxim` | **public** (already shipped) | [paths.py:71](../../src/maxim/utils/paths.py#L71) | When set, isolates the entire user-state surface — orchestrator NPC state, util/, sessions/, memories — for the run. Re-uses the existing public env var rather than adding a new one. |

Notes:
- **Why no `MAXIM_NO_DEFAULT_EMBODIMENT`?** The default `bodies/base_humanoid` injection at [cli.py:1050](../../src/maxim/cli.py#L1050) already has an opt-out: `--no-embodiment`. Documenting it in the experimental protocol is enough; no new flag.
- **Why no flag for arc keyword routing?** The audit flagged `select_arc_for_goal()` ([arcs.py:372](../../src/maxim/simulation/arcs.py#L372)) as a confound, but it's invoked from [cli.py:1156](../../src/maxim/cli.py#L1156) under an existing generative-runner branch. Phase A uses goals that don't match arc keywords (e.g. "recall the password from session 1") and asserts `arc=None` in `report.json`. Adding a flag is overkill; goal selection is the lever.
- All five named flags are **debug/experimental per CC4**. They classify alongside `MAXIM_SUBSTRATE_PATH` and `MAXIM_NAC_TEMPORAL_CREDIT_WEIGHT` in `docs/user/configuration.md` under "Debug — substrate + decision-system experiments."

## Orchestrator state isolation

**Recommendation: use `MAXIM_DATA_HOME=$(mktemp -d)` for total isolation. Do NOT add a per-subsystem orchestrator knob.**

[`orchestrator.py:1181`](../../src/maxim/simulation/orchestrator.py#L1181) builds `persistence_dir=str(_data_home() / "orchestrator")` and [line 1187](../../src/maxim/simulation/orchestrator.py#L1187) builds `AgentFactory(base_data_dir=_data_home() / "orchestrator")`. Both flow through `data_home()` from [paths.py:61](../../src/maxim/utils/paths.py#L61), which honors `MAXIM_DATA_HOME`. The `~/.maxim/util/{semantic_embeddings.npz, escalation_learning.json, fear_learning.json, learned_bounds.json, focus_learner.json}` second-tier confounds also flow through `data_home()` via `resolve_user_state` — so the same env var isolates them.

This is one knob, no new code, and it's already public-stable per CC4.

The harness wrapper:

```bash
export MAXIM_DATA_HOME="$(mktemp -d -t maxim-v1-XXXXXX)"
trap 'rm -rf "$MAXIM_DATA_HOME"' EXIT
maxim --sim "<goal>" ...
```

Cross-session V1 runs use a *persistent* tmpdir reused across the two anchored sessions in the experiment, then discarded.

## Phased re-run protocol

Six phases. Each phase runs the V1 cross-session goal twice (session 1 plants memory, session 2 recalls it) under a fresh `MAXIM_DATA_HOME` shared across the two sessions. The phase delta vs Phase A attributes the V1 result to the toggled contributor.

| Phase | PFC | Acting Coach | Sim sandbox text | Default persona | Default embodiment | Configuration                                                                 |
|-------|-----|--------------|------------------|-----------------|--------------------|-------------------------------------------------------------------------------|
| **A** (substrate-only baseline) | OFF | OFF (`--no-acting-coach`) | OFF | OFF (`--no-persona`) | OFF (`--no-embodiment`) | All `MAXIM_DISABLE_*=1`, isolated tmpdir                                      |
| B | ON  | OFF | OFF | OFF | OFF | Adds PFC preamble                                                             |
| C | OFF | ON  | OFF | OFF | OFF | Adds Acting Coach + embodied identity rewrite                                 |
| D | OFF | OFF | ON  | OFF | OFF | Adds "SIMULATION ENVIRONMENT" text                                            |
| E | OFF | OFF | OFF | ON  | OFF | Adds adversarial persona                                                      |
| F | OFF | OFF | OFF | OFF | ON  | Adds default `bodies/base_humanoid`                                           |
| **G** (today's behavior, control) | ON | ON | ON | ON | ON | All defaults, isolated tmpdir — confirms isolation alone does not move metrics |

Per-phase metrics (added to `report.json` under a new top-level `confound_quarantine` block):

```json
"confound_quarantine": {
  "phase": "A",
  "flags": {
    "MAXIM_DISABLE_PFC_PREAMBLE": "1",
    "MAXIM_DISABLE_ACTING_COACH": "1",
    "MAXIM_DISABLE_SIM_SANDBOX_TEXT": "1",
    "MAXIM_NO_DEFAULT_PERSONA": "1",
    "MAXIM_DATA_HOME": "/tmp/maxim-v1-aB3xQz"
  },
  "isolated_data_home": true,
  "metrics": {
    "v1_recall_success": true,
    "recall_turn_index": 4,
    "tokens_in_system_prompt": 0,
    "tokens_in_pfc_preamble": 0,
    "tokens_in_acting_coach": 0,
    "tokens_in_sim_sandbox": 0,
    "persona_active": null,
    "embodiment_ref": null,
    "arc_active": null,
    "orch_npc_carry_over_links": 0,
    "nac_links_session_2_loaded_from_session_1": 12
  }
}
```

The token-count breakdown is the receipts trail: anyone reading the report can see exactly which scaffolds were active and how big they were. `orch_npc_carry_over_links` is the orchestrator NAc count at session-2 startup — should be `0` under isolated `MAXIM_DATA_HOME`, otherwise the isolation broke.

Build site: extend [`simulation/report.py:build_report`](../../src/maxim/simulation/report.py#L72) with one new dict assembled from `os.environ` snapshot + token counts captured by `PromptBudgeter`. ~30 LOC.

## Implementation guidance

### Centralized module

```
src/maxim/runtime/confound_flags.py        # new — ~50 LOC
```

Single source of truth (exact API to be confirmed at implementation time):

```python
import os

def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")

def pfc_preamble_enabled() -> bool:    return not _flag("MAXIM_DISABLE_PFC_PREAMBLE")
def acting_coach_enabled() -> bool:    return not _flag("MAXIM_DISABLE_ACTING_COACH")
def sim_sandbox_text_enabled() -> bool: return not _flag("MAXIM_DISABLE_SIM_SANDBOX_TEXT")
def default_persona_enabled() -> bool: return not _flag("MAXIM_NO_DEFAULT_PERSONA")

ALL_FLAGS = (
    "MAXIM_DISABLE_PFC_PREAMBLE",
    "MAXIM_DISABLE_ACTING_COACH",
    "MAXIM_DISABLE_SIM_SANDBOX_TEXT",
    "MAXIM_NO_DEFAULT_PERSONA",
)  # consumed by tests/conftest.py for autouse scrub
```

Each call site imports the helper:

- `prompt_builder._add_pfc_preamble_section` — wrap the `budgeter.add(...)` call in `if pfc_preamble_enabled():`.
- `prompt_builder.build_identity_section` — gate the `_coach is not None` branch on `acting_coach_enabled()` AND gate the `--no-acting-coach`-driven CLI propagation in `cli.py` so the worker never gets an `acting_coach` attached.
- `prompt_builder.build_identity_section` (sim block) — gate the `if _sim_active:` lines on `sim_sandbox_text_enabled()`.
- `cli.py` persona dispatch — read `default_persona_enabled()` before falling back to `DEFAULT_PERSONA`.

### CC4 doc updates

Add to [docs/user/configuration.md](../user/configuration.md), section "Debug — substrate + decision-system experiments":

```
| `MAXIM_DISABLE_PFC_PREAMBLE`     | Skip the PFC deliberation preamble injection. Used for V1 substrate-attribution experiments. | 0 |
| `MAXIM_DISABLE_ACTING_COACH`     | Skip Acting Coach + embodied identity rewrite. Same as `--no-acting-coach`. | 0 |
| `MAXIM_DISABLE_SIM_SANDBOX_TEXT` | Skip the "SIMULATION ENVIRONMENT" sandbox-context text in the system prompt. | 0 |
| `MAXIM_NO_DEFAULT_PERSONA`       | Treat absent `--persona` as `None` instead of falling back to `adversarial`. Same as `--no-persona`. | 0 |
```

Add to the CLI flag table the `[experimental]` markers on `--no-acting-coach` and `--no-persona`.

## File-touch table

| File                                                                           | Change                                                          | LOC  |
|--------------------------------------------------------------------------------|-----------------------------------------------------------------|------|
| `src/maxim/runtime/confound_flags.py` (NEW)                                    | Centralized gate helpers + `ALL_FLAGS` tuple                    | ~50  |
| `src/maxim/agents/prompt_builder.py`                                           | Wrap PFC/identity/sim-sandbox emit sites in `if ... enabled():` | ~12  |
| `src/maxim/cli_parser.py`                                                      | Add `--no-acting-coach`, `--no-persona`                         | ~8   |
| `src/maxim/cli.py`                                                             | Propagate new flags to env vars + persona-default branch        | ~15  |
| `src/maxim/simulation/orchestrator.py`                                         | Suppress `acting_coach` attachment when `--no-acting-coach` set | ~5   |
| `src/maxim/simulation/report.py` (`build_report`)                              | New `confound_quarantine` block                                 | ~30  |
| `tests/conftest.py`                                                            | Autouse scrub fixture iterating `ALL_FLAGS`                     | ~25  |
| `tests/unit/test_confound_flags.py` (NEW)                                      | Per-flag injector-skip tests (4 tests)                          | ~80  |
| `tests/integration/test_v1_phased_metrics.py` (NEW)                            | Phase A and Phase G dry-run that asserts `confound_quarantine` block populated correctly | ~60  |
| `docs/user/configuration.md`                                                   | CC4 entries for the four new flags                              | ~6   |
| `docs/plans/confound_quarantine.md` (this file)                                | Plan document                                                   | ~250 |
| `scripts/run_v1_phases.sh` (NEW, optional harness)                             | Six-phase wrapper that creates tmpdirs and aggregates reports   | ~50  |

Total: ~600 LOC, mostly tests and docs. Production-code surface is **~40 LOC**.

## Test plan

Per CLAUDE.md `feedback_opt_in_env_in_hot_paths.md`, every new opt-in env var that's read in a hot startup path needs an autouse scrub. All four named flags are read inside `prompt_builder` which is reached during `LLMWorker` construction — squarely in the "hot path" category. The pattern in `tests/conftest.py` (`_isolate_maxim_llm_profile_env`, `_isolate_maxim_substrate_path_env`) is the template.

**Single autouse scrub fixture**, iterating `confound_flags.ALL_FLAGS`:

```python
@pytest.fixture(autouse=True)
def _isolate_maxim_confound_flags():
    from maxim.runtime.confound_flags import ALL_FLAGS
    saved = {k: os.environ.pop(k, None) for k in ALL_FLAGS}
    try:
        yield
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
```

**Per-flag pin tests** in `tests/unit/test_confound_flags.py`. Each is two assertions — one for default-on (env unset → injector fires) and one for disabled (env=`1` → injector skipped). For PFC and sim-sandbox we assert against the rendered prompt string from `PromptBuilder`. For acting-coach we assert the rendered identity line. For default-persona we assert `cli.py`'s persona resolution returns `None`.

The unit tests are short — the value is that **silently removing any conditional triggers a CI failure**. If anyone deletes `if pfc_preamble_enabled():` in a future refactor, `test_pfc_preamble_disabled_when_env_set` fails immediately.

**Integration test** (`tests/integration/test_v1_phased_metrics.py`) asserts: under `MAXIM_DATA_HOME=<tmpdir>` plus all four `MAXIM_DISABLE_*=1`, a stub V1 sim produces a `report.json` whose `confound_quarantine.metrics.tokens_in_pfc_preamble == 0` and `tokens_in_acting_coach == 0`. Pure metric-shape assertion — the LLM is mocked.

## Risks

### R1 — Flag lifecycle decided in 1.0 based on Phase A outcome

The flags ship in 0.9.x as experimental (per CC4). Their disposition is **decided in 1.0**, conditionally on what the Phase A re-run reveals. No experimental limbo through 1.1+.

**Three branches, decided when Phase A results land:**

- **Clean pass** — substrate alone reproduces the V1 cross-session recall result without the scaffolds. **Flags removed in 1.0.** They did their job (attribution). Reproducibility for the V1 numbers is preserved by pinning the experiment README to a specific 0.9.x commit hash — that's the academic-ML standard, not freezing debug flags into the 1.0 contract. Removal is a one-line revert per gate site.

- **Conditional pass** — substrate works but specific scaffolds materially boost the result. **Flags graduate from experimental to public-stable in 1.0**, classified under the public env-var contract per CC4. Documentation explicitly states which scaffold combinations the claim is conditional on. The flags become part of the production diagnostic surface (researchers and users debugging substrate behavior can disable scaffolds to isolate signal).

- **Fail** (R2 fires) — substrate alone does NOT reproduce V1. **Re-scope the 1.0 claim** to "the substrate produces cross-session recall when supported by scaffold X+Y." Keep the flags as evidence of the re-scoping. Update the README and stable_api.md accordingly.

The disposition decision is part of the V1 phased re-run experiment doc — Phase A results land, the doc records which branch fires, and the 1.0 release blocks on that decision. No deferral past 1.0.

### R2 — Phase A reveals the substrate alone doesn't reproduce V1

This is the existential risk. If Phase A shows the cross-session recall signal disappears once the scaffold is removed, the 1.0 substrate-attribution claim has to be re-scoped. **That's the point of the experiment.** The plan exists precisely to find this out before 1.0 ships, not after. The architectural response (re-scope claim, identify which scaffolds are load-bearing for the actual claim, explicitly document them) is out of scope for this plan.

### R3 — The autouse scrub regresses

Mitigation: the autouse fixture iterates `ALL_FLAGS` from the module itself. Adding a flag to `ALL_FLAGS` automatically scrubs it; forgetting to add a new flag to `ALL_FLAGS` is caught by the per-flag pin test, which sets the var via `monkeypatch.setenv` — if the autouse scrub doesn't pick it up from `ALL_FLAGS`, the leak is detected by the next-test-in-sequence assertion that the gate is OFF by default.

### R4 — Centralized module becomes a kitchen sink

Mitigation: `confound_flags.py` is scoped exclusively to "scaffold disable for substrate-attribution experiments." Any unrelated flag (e.g. a new tracing toggle) does NOT belong here. Add a CONTRIBUTING.md note: new flags go in `runtime/confound_flags.py` only if they gate a default-on scaffold whose impact on V1 attribution is being measured.

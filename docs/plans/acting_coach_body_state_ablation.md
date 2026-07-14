# Acting Coach body-state ablation (proposed Exp 44)

**Status:** PROPOSED (pre-registration draft, 2026-07-14). Not yet scheduled.
**Decision it settles:** whether to wire `embodiment=` into the production
`build_memory_hub` call sites (activating the `body_state` prompt section and
Acting Coach Layers 2+4), bless the auto-sense status quo, or wire the data
without the coach.

## Motivation

The 2026-07-14 embodiment deep-dive found that `format_body_state_for_prompt`
has **no live production caller**: its only caller (`agents/memory_agent.py`)
is gated on `memory_hub.embodiment`, which neither production
`build_memory_hub` call site (`runtime/bio_stack.py`, `runtime/agent_factory.py`)
ever passes. Consequences, verified in the tree:

- `StructuredContext.body_state` is always `""`; the CRITICAL `body_state`
  prompt section never renders.
- **Acting Coach Layers 2+4 (pain anticipation + drive modulation,
  `prompts/acting_coach.py::_compose_pain_anticipation` /
  `_compose_drive_modulation`) early-return on empty input in every
  production path.** The B3 bio-modulation shipped in 0.7 has never actually
  run in production.
- What the LLM sees instead is **auto-sense** (agent_loop section 1.15 →
  `auto_sense_context`): raw `read_all_sensors` output that does NOT tick the
  body and therefore lags one turn behind the drive state.

Exp 37/38/42 results are unaffected — they measured the auto-sense reality
and stand as-is. But wiring `embodiment=` is a **prompt-content behavioral
delta**, so per the interim-contamination and confound-isolation disciplines
it must be measured as an experiment, not slipped in as a bug fix.

## Question

Does giving the LLM a fresh, ticked body state — and the Acting Coach's
bio-modulated guidance on top of it — measurably change embodied behavior
(harm avoidance, drive regulation) relative to the auto-sense-only status quo?

## Design

**Mode:** LLM-primary only. The Acting Coach is a prompt mechanism;
substrate-primary bypasses prompts entirely and already reads post-tick
drives directly (`_read_drive_states` after the per-proposal tick).

**Model:** Qwen2.5-32B (base). Per the Exp 37/40 Goldilocks result it is the
one local model with demonstrated headroom for substrate-side prompt signals;
a ceiling model (Mistral24B) would mask the delta and a below-zone model
(Qwen14B) can't express it.

**Harness:** the Exp 42 counterbalanced safe-vs-harm warmth arc
(`cradle`-style, infant_humanoid body, safe/harm affordance name swap across
seeds) run in LLM-primary mode. **Harness-prep required (Phase 0.5):**
`simulation/substrate_telemetry.py` is constructed only under
`aut_mode == "substrate-primary"` (orchestrator.py gate) and
`scripts/benchmark_exp42_preference.py` hardcodes that mode — so the Exp 42
*data plumbing* does not run in LLM-primary as-is. The exploitation-phase
*metric* transfers (analyze_exp42_preference.py defines it behaviorally on
contact records, mode-agnostic); the contact records for this experiment
come from the per-session `actions.jsonl` + `MAXIM_LOG_FILE` JSONL instead.
Phase 0.5 forks the benchmark script's contact-record derivation to consume
those sources (preferred), or deliberately widens the telemetry gate as its
own reviewed code change — either way the harness must run end-to-end on a
throwaway arm before any registered arm fires. Prompt-section verification
uses `MAXIM_LOG_FILE` JSONL.

**Arms (additive factorial — auto-sense stays ON in all arms so the only
manipulation is what is ADDED):**

| Arm | body_state section | Coach Layers 2+4 | Isolates |
|---|---|---|---|
| A (status quo) | off (unwired) | inert | baseline |
| B | ON (wired) | disabled | fresh ticked body state as *information* |
| C | ON (wired) | ON | the coach's *guidance* on top of the same information |

10 seeds per arm, counterbalanced safe/harm naming per the Exp 42 protocol.

**Wiring for arms B/C:** thread `embodiment=` through
`build_bio_stack` → `build_memory_hub` at the AUT call site, plus a
structural test pinning "embodied AUT construction ⇒
`memory_hub.embodiment is not None`" so the silent gap cannot reopen
(same silent-no-op class as the `build_executor(pain_bus=...)` lesson).
Arm B requires a NEW per-layer toggle on `ActingCoachConfig` — none exists
today, and the existing `embodiment_guidance` flag must NOT be repurposed:
it gates Layers 1-4 plus the exploration directives together, so using it
for arm B would remove Layers 1+3 as well and break the additive factorial.
Config-first per the dev standard, not a new env var.

## Metrics

**Primary (pre-registered):** exploitation-phase safe-preference and
harm→safe flip, exactly as computed in Exp 42.

**Secondary:** drive-regulation quality — fraction of turns with temperature
inside the comfort band, mean |deviation from set_point|; pain events per
turn; turns-to-first-corrective-action after a breach.

**Mechanism checks (must pass for the arms to count):**
1. Prompt telemetry confirms the `body_state` section renders in B/C and the
   coach layers emit only in C.
2. Byte-diff of prompts across arms confirms ONLY the intended sections
   differ (narrator snapshot pinned per the narrator-state-confound lesson).
3. Prompt-cache stable prefix is byte-identical across turns in all arms —
   `body_state` is added without `cacheable=True` (verified in
   `prompt_builder.py:1440`), so this should hold; if it doesn't, stop and
   fix before running arms.

## Decision rule (pre-registered, exhaustive)

Two pre-registered comparisons, each classified **+** / **0** / **−** by the
same margin (non-overlapping SE across the 10-seed arms on the primary
metric): **ΔB = B − A** (does the information help?) and **ΔC = C − B**
(does the coaching add anything on top of the same information?). C vs A is
computed as a consistency check but the decision keys on the 3×3 grid,
which partitions the outcome space:

| ΔB (info) | ΔC (coach) | Reading | Action |
|---|---|---|---|
| + | + | information helps, coaching adds more | wire `embodiment=` with coach ON; B3 Layers 2+4 → candidate-Earned (graduation row 11) |
| + | 0 | information sufficient, coaching redundant | wire body_state; Dormant-mark Layers 2+4 citing this experiment |
| + | − | coaching subtracts from good information | wire body_state; disable Layers 2+4; bird's-eye check on WHY the guidance fights (narrator framing?) before any mechanism patch |
| 0 | + | information alone insufficient, coach rescues it | wire with coach ON; graduate — and flag as surprising (guidance shouldn't beat raw data; verify mechanism checks closely) |
| 0 | 0 | neither matters | keep status quo; Dormant-mark the body_state path + Layers 2+4; correct the B3.1 claims in CLAUDE.md |
| 0 | − | coaching actively harms neutral information | keep status quo; Dormant-mark; record the wrong-sign in the results doc |
| − | any | ticked body_state is WORSE than lagged auto-sense | do not wire; bird's-eye divergence check (this outcome implies a confound or a prompt-composition interaction, not a coach question) |

## Phase 0 smoke (run first, ~zero cost)

On the wiring branch: `MAXIM_LOG_FILE=/tmp/maxim.jsonl maxim --sim "warmth
smoke" --embodiment bodies/infant_humanoid --interactive false
--sim-max-turns 3`, then verify in the JSONL: body_state section present,
coach layers present, drive values in the section CHANGE between turns
(ticked, not stale), stable prefix byte-identical across turns.

## Cost

Local Qwen32B on the leader; 30 runs × ~40 turns ≈ hours of wall time,
no cloud spend. Runs must not share `~/.maxim/` with concurrent sessions
(persisted-state collision rule).

## Non-goals

- Interoception `Percept` production and Roy-5b naming-event emission stay
  out of scope (tracked in grounded_language_acquisition.md context).
- No substrate-primary arms — the mechanism under test is prompt-side.

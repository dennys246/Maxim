# The cradle sim (Exp 37/38): which arc runs, what a resume carries, and what every arm gets whether you asked or not

**Established by** B4's four-lens design review (2026-10-18;
`docs/experiments/rationale/b4-exp37-floors/`), mostly by reading code and running pure functions offline. Read
this before designing any arm, control or floor that runs through `maxim --sim` in the cradle world.

## The goal picks the arc, and the turn count picks the phase

- `cli.py` routes a goal through `select_arc_for_goal(goal)`. The `fire_pit` goal, the `sharp_rock` goal **and
  Exp 37's `PEACEFUL_GOAL`** ("cradle infant explores the puzzle door and button") all select the same `cradle`
  arc; the `deceptive_fire` goal selects `cradle_deceptive`.
- `Narrator.decide` advances `_phase_idx` only when `_turns_in_phase >= phase.turns_max`; the LLM's `phase` field is
  advisory. The cradle's `turns_max` run 4, 2, 3, 3, 5, so a 12-turn session is: turns 1–4 fire pit, food and cool
  air; 5–6 the fire pit again; 7–9 the sharp rock and blanket; 10–12 discrimination. `tool_discovery` (the door and
  button) would start at turn 13 and never activates.
- **Consequence:** Exp 37's arm C, "resumed from a peaceful prior", resumed from a fire-pit session. No door or button
  call appears in any of the 725 committed Exp 37/38 rows; the `sharp_rock` sessions made 22–23 `fire_pit_*` calls
  per five. A wrong-content control needs an arc selected BY NAME whose entities exclude the scenario's
  (B4 prereg: a fire-free arc), and its prior written as its own row.

## What a resume carries

- `_restore_aut_from_session` restores the hippocampus, NAc, EC and ATL, nothing else: no body or world sensor state.
  NAc loads with `apply_decay=False`, so the save → resume gap does not decay biases (hippocampus consolidation
  pressure still decays by wall clock). The SCN is saved but not in `RESUME_STORES`.
- `report.py::resume_stamp` accepts a store that is `absent`, and a failed save is swallowed at debug, so
  `resume_loaded` does not prove the carry. Compare the stores at save and at load.
- `serving_provenance.py::_session_index` numbers sessions (fresh 0, first resume 1, …): a free structural check.
- A notes store added to `RESUME_STORES` fires T1-16's trigger (an EARNED row).

## The narrator's resume prompt is not consumed on the cradle path, silently

On the generative path, `run_generative_campaign` runs synchronously, `_finish_runner` sets `stop_event`, and only
then is `build_resume_prompt` injected; `loop_gates.py::pre_tick_gate` breaks before the next percept. The campaign
never reads the resume data. But the path choice is made under `except Exception: pass` in `cli.py` and is not
stamped; on the non-generative path the orchestrator DOES consume the resume prompt (with the prior `tool_usage`).
Stamp the path per row and keep a sentinel test.

## What every `maxim --sim` AUT gets, outside `create_full_agent`

`start_simulation_mode` builds the AUT with the bio stack, then adds: `PainInterceptorExecutor`,
`AnticipatoryPainExecutor` (`PerceivedPainAssessor(nac=...)`), an **unconditional `FearGatedExecutor`** (no switch;
it makes no LLM call on cradle tools), `build_default_network(nac=...)`, the `ImaginationTrigger` and the W2
scene manifest **steered by the AUT's NAc biases** (`get_agent_tool_biases`, so carried biases change the world a
resumed arm sees; `MAXIM_DISABLE_IMAGINATION=1` removes it), `SubstrateTelemetry`, the percept-anxiety hook, and
`examine`'s "You recall:" lines (#845, a memory channel past every ablation). No flag selects another AUT stack, and
`create_full_agent(with_bio_stack=False, ...)` on `bodies/infant_humanoid` raises in `build_executor`
(`pain_bus=None` with an `entity_ref`).

## Instrument traps

- `--seed` reaches Python, numpy, torch, the LSH planes and the `global_seed` stamp, not LLM sampling on the peer
  path (`_MaximPeerBackend` sends no seed; `seed_honoured` is true only for the in-process llama-cpp backend). "Paired
  by seed" pairs nothing on the LLM side.
- `analyze_exp37.py::compute_metrics` scores zero engagement as **0.0**, not missing: "never approached" reads as
  total failure on `fire_pit` and as perfect avoidance on `deceptive_fire`.
- `deceptive_fire`'s learning direction is DOWN (`compute_counter_prior`); a gate written "B above C" scores
  perseveration as a pass.
- The AUT turn is a 30 s wall-clock window (`sim.aut_turn_timeout_s`), so a longer prompt gets fewer actions.
- Every committed local fire ran on the hash-embedding fallback (`sentence_transformers_version: "not-installed"`).
- On the peer path B0 records the served-id alias only (no file hash, quantization or KV type): the AUT's model is
  attested, not established, until B0b and a leader-side report land.

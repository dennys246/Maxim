# Substrate-Primary AUT Mode

> **Status:** Phase -1 prototype landed (2026-05-09). Phase 0 harness in 1.0 (B5). Substrate-primary AUT mode lands in 1.1+. **Opt-in via `--aut-mode substrate-primary`** when wired (Session A in the next-steps queue).

## What this is

Maxim has historically used a **language model** as the AUT's action selector — the LLM proposes the next tool call, the executor dispatches it, the bio-systems learn from the outcome. The bio-substrate (NAc, EC, ATL, Hippocampus, reflexes, Default Network) has been an **augmentation layer** sitting around the LLM: predicting outcomes, biasing recognition, capturing episodes, providing reactive behaviors.

Substrate-primary mode flips that. **The bio-substrate becomes the action selector.** The LLM is removed from the AUT's decision loop entirely and replaced with `NAc.recommend_action()` reading from learned causal links, reward biases, and active drive states. The action proposal flows through the same `executor.execute()` dispatch — but the proposer is the substrate, not a language model.

## Why this exists

Three motivations:

1. **The "substrate carries cognition" thesis.** Maxim's bio-inspired framing claims the bio-substrate is doing real cognitive work. If the LLM is always the action selector, that claim has an asterisk — the substrate could be doing nothing useful and the LLM would still drive coherent behavior. Substrate-primary mode is the experimental setup that *proves* (or disproves) the substrate's role.

2. **The LLM-mitigation drift.** A 2026-05-09 audit found roughly 60-70% of recent engineering effort going to LLM-mitigation scaffolding (~845 LOC of band-aids) — stall detectors, JSON repair pipelines, tool-failure hint sections, identity rewrites for small models, format enforcement for planning mode, etc. Each band-aid is a workaround for the LLM doing something the substrate could in principle handle natively. Substrate-primary mode is the structural fix that removes the AUT-side LLM coupling entirely.

3. **The Hivemind enabler.** Distilled bio-substrate (NAc weights, EC concepts, reflexes) is naturally shareable across instances — far more privacy-friendly and aggregatable than raw episode/dialogue logs. Substrate-primary mode is the natural client of the [Maxim Hivemind + Oasis](hivemind.md) layer; the federated cognition story only works if the substrate can drive behavior on its own.

## Parallel-mode architecture, not replacement

Substrate-primary mode runs **in parallel** to the existing LLM-AUT path. The user-facing default does not change. There are now two operating modes for the AUT:

| Mode | Action selector | Use case |
|---|---|---|
| **`--aut-mode llm-primary`** (default) | LLM proposes; bio-substrate learns from outcomes | All current Maxim workloads — D&D campaigns, Reachy demos, headless agent runs |
| **`--aut-mode substrate-primary`** (opt-in, 1.1+) | `NAc.recommend_action()` proposes; LLM not invoked at all on the AUT side | Substrate research; Phase 0/1 grounded-language experiments; eventual user-facing path once mature |

The orchestrator, environment NPCs, imagination designer, and Hivemind/Oasis distillation all continue to use LLMs. Substrate-primary mode is specifically about the AUT's action loop.

## How action selection works in substrate-primary mode

`NAc.recommend_action()` (in [src/maxim/decisions/nac.py](../src/maxim/decisions/nac.py)) scores each available tool by combining three signals:

1. **Causal-link confidence (primary learned signal).** For each candidate tool, NAc looks up positive and negative causal links from `tool:X → outcome:Y` records in its causal graph. Positive links contribute their best confidence to the score; negative links subtract (weighted lower than positives so a single bad outcome doesn't permanently block exploration).

2. **Reward bias (secondary learned signal).** The per-agent `reward_bias[(agent_id, node_id)]` map adds a small additional positive nudge for tools the agent has been credited on. Capped at `NACConfig.max_reward_bias` (default 0.20) by design — this is the EC-recognition modulator, not a primary action-selection score.

3. **Drive-relevance heuristic (cold-start fallback).** When no learned signal exists, active drives (`drive_value > 0.5` for hunger/thirst/fatigue/cold/fear/curiosity/pain) bias selection toward semantically-related tools via a substring + affinity-table match. This is a **Phase -1 placeholder** for proper EC embedding similarity, which Phase 0+ will integrate.

The highest-scoring tool above `min_confidence` (default 0.3) wins. Ties are resolved deterministically by tool name. **If nothing scores high enough, the method returns `None`** — substrate-primary mode never falls back to random selection. The substrate must have an opinion to act.

```python
from maxim.decisions.nac import NAc, NACConfig

nac = NAc(NACConfig())
# ... agent has observed pick_up_food → satisfaction (positive, several times)
# ... drives indicate hunger=0.8

action = nac.recommend_action(
    agent_id="my_infant",
    available_tools=["pick_up_food", "examine_rock", "rest"],
    current_drives={"hunger": 0.8},
)
# → {"tool_name": "pick_up_food", "params": {}, "confidence": 0.74,
#    "source": "substrate-primary", "reasoning": "causal_pos=0.62; drive:hunger(0.80) name-match"}
```

The returned dict is compatible with `agents.autonomy.Proposal.action` and is dispatched through the standard `executor.execute()` path — no new dispatcher.

## Phase -1 — the gating Boolean (PASSED 2026-05-09)

The most important question in the entire substrate-primary program: **can the substrate generate even one non-reflex action without LLM proposal?**

If yes, the rest of the program is feasible. If no, NAc needs significant extension before substrate-primary mode is viable.

The Phase -1 prototype lands `NAc.recommend_action()` and 11 unit tests in [tests/unit/test_nac_recommend_action.py](../tests/unit/test_nac_recommend_action.py). All tests pass. The substrate **can** generate non-reflex actions from learned causal links + drive heuristics.

The next phases (still to ship):

- **Phase 0 (1.0 B5):** Wire `--aut-mode substrate-primary` end-to-end + cradle-prelinguistic harness with motor-only AUT prompt + per-tick telemetry. Proves substrate-primary works in a real sim.
- **Phase 1 (1.1):** Vocabulary-constrained mode. The LLM is allowed back as input parser but its output vocabulary is masked to tokens the substrate has bound. Tests how much the LLM was doing beyond I/O.
- **Phase 2 (1.1+):** Symbol-binding layer. Small online-trained model that binds words to bio-substrate concepts. Enables vocabulary growth from substrate experience.
- **Phase 3 (1.2+):** From-scratch sequence model trained on Roy long-horizon curriculum with substrate-grounding objective. The headline experiment.
- **Phase 4 (1.3+):** Pretrained-vs-grounded A/B comparison. Final validation.

See [docs/plans/grounded_language_acquisition.md](plans/grounded_language_acquisition.md) for the full plan with kill criteria, gates, and phasing.

## D&D survival as the bidirectional kill criterion

A substrate-primary AUT that cannot survive a D&D-style campaign orchestrated by an LLM-DM is a failed bio-substrate. **AND** a simulation environment that no learning substrate can navigate is a failed simulation environment. The convergence test is mutually load-bearing:

| Outcome | Diagnosis |
|---|---|
| Substrate AUT runs the campaign cleanly | Substrate is real. Project thesis validated. |
| Substrate AUT fails; LLM-AUT succeeds in same scenario | Substrate insufficient for non-trivial cognition. Reframe required. |
| LLM-AUT also fails the same scenario | Simulation environment is the failure — the test isn't measuring what we think it is. |
| Both succeed but substrate is much weaker | Acceptable interim. Scope clear; LLM remains in user-facing path. |

D&D was chosen because it has long-horizon temporal structure, novel entities every session, decision-making with delayed reward, role coherence demands, and multi-agent dynamics. Cradle (current sensorimotor learning) is necessary; D&D is sufficient.

## Confound discipline: raw vs primed substrate

Substrate-primary Maxims can either start from a fresh substrate (raw — the headline experimental condition) or bootstrap from accumulated experience via the [Maxim Hivemind](hivemind.md). Both are valid, but they answer different questions:

- **Raw substrate** demonstrates "the substrate can develop cognition from zero." Required for Phase 0 + Phase 1 validation.
- **Primed substrate** demonstrates "the substrate can absorb collective experience and operate." End-user convenience path; ships once Hivemind is live.

Both ship in parallel. The grounded-language plan's Phase 0 and Phase 1 specifically run with the Hivemind disabled so the headline experiment stays clean.

## Pretrained-LLM crutches: what gets disabled

When running substrate-primary mode (or running the LLM-AUT path with the intent of not biasing the substrate's learning), several runtime mitigations should be turned off because they exist specifically to compensate for pretrained LLM behaviors that don't exist in a fresh substrate:

| Crutch | What it does | Disable via |
|---|---|---|
| Tool-failure hint section | Adds a `=== Tools You've Hallucinated ===` block to the prompt listing names the agent previously called that don't exist (mitigates qwen-class training-prior hallucination). E4 validation 2026-05-09 showed no benefit on qwen2.5-14B; default flipped to OFF. | `MAXIM_TOOL_FAILURE_HINTS=0` (already default) |

The bio-natural alternative for tool-failure avoidance already exists: NAc records `tool:X → failure:not_registered` with negative valence and `recommend_action()` subtracts that confidence from the tool's score. With the prompt-level hint disabled, the substrate's negative-valence avoidance becomes the only signal — which is exactly what we want to measure.

The crutch table grows over time. See [docs/plans/grounded_language_acquisition.md](plans/grounded_language_acquisition.md) §"Pretrained-LLM crutches to disable for Phase 0/1" for the canonical list.

## How this changes development practice

For contributors, the substrate-primary pivot adds one new question to ask whenever a new LLM-mitigation prompt section or stall handler is being considered:

> **Could this same problem be solved by substrate-primary mode learning from outcomes instead?**

If yes, the band-aid is the wrong fix and the work belongs to the substrate-primary track. If no, the band-aid is genuinely necessary for the LLM-AUT path.

Each shipped band-aid that's actually a substrate gap should be entered into the "crutches to disable" table above so substrate-primary experiments can turn it off.

## See also

- [Decisions](decisions.md) — NAc causal inference, the engine behind `recommend_action()`
- [Maxim Hivemind + Oasis](hivemind.md) — The federated layer that lets substrate-primary Maxims share learning
- [docs/plans/grounded_language_acquisition.md](plans/grounded_language_acquisition.md) — Full multi-phase plan
- [docs/plans/maxim_hivemind.md](plans/maxim_hivemind.md) — Hivemind/Oasis architecture plan
- [docs/plans/v1_refinement.md](plans/v1_refinement.md) §B5 — 1.0 scope for substrate-primary harness + Hivemind shareability

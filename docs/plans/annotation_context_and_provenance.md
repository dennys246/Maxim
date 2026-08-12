# Annotation channel: context, provenance, and semantic keying

**Status:** DRAFT (2026-08-11). Motivated by the Exp 44b pilot findings F1-F6
([../experiments/44b_pilot.md](../experiments/44b_pilot.md)).
**Front-gate:** rides existing infrastructure — NAc already stores what we need
per-cluster, the affordance-concept encoder already exists and is validated. This plan
mostly *stops discarding* information, and adds one experiment to prove it matters.

## The finding in one paragraph

`NAc.get_agent_tool_biases` is the sole producer of the LLM's substrate annotation. It
does three lossy things: (a) drops the cluster id (`for (aid, _cid, tool_sig), bias in
…`) and max-aggregates agent-wide, so **context-dependent value is inexpressible**;
(b) keys on exact **tool-signature strings**, bypassing the affordance-concept layer, so
transfer is name-brittle; (c) renders a band label only, so **why** a thing is rewarding
(drive relief vs bare tool success vs cross-modal credit) is erased. Meanwhile the
substrate-primary policy (`recommend_action` → `consulted_bias_by_modality`) uses the
per-cluster value correctly. **The no-LLM path is smarter than the LLM path, using the
same substrate.**

## Why this is the highest-value architectural work available

It sits directly under the program's central claim. Today an external reader can say
"your LLM just copies whatever tool strings you print as rewarding" and the pilot data
cannot refute it (F1). Every item below either removes a lossy step or measures whether
removing it matters.

---

## S1 — Provenance in the annotation (the "why")

**Status: SHIPPED (write side PR #501, 2026-08-11; render side follow-up PR,
2026-08-12).** Write side: `NAc._cluster_reward_source`, closed `CREDIT_SOURCES`
vocabulary, one-way promotion to `"mixed"`, `get_cluster_reward_sources(agent_id=)`.
Render side: `compose_cluster_bias_annotation_section(biases, sources)` joins on the RAW
tool signature (before the `tool:` prefix strip) and appends a sign-aware gloss after
`ANNOTATION_SOURCE_SEPARATOR` (` — `); the S4 parser splits at the same shared constant
and the round-trip test in `tests/unit/test_exp44_nonstationarity.py` renders through the
real composer and parses back, so format drift fails there instead of silently reading as
"no annotation" across a campaign. Glosses are sign-aware because negative credits carry
sources too (a drive REGRESS books −1.0 with source `drive_relief`). Pre-S1 persisted
state has no sources → byte-identical pre-S1 rendering.

**Decision (documented for review): the source renders in the annotation, not
telemetry-only.** Rationale: (a) the Exp 44 counterfactual proved the annotation steers
the LLM, so *applicability* information belongs where the steering happens; (b) glosses
are display-layer downstream of encoding, same defensibility as the band labels; (c) the
existing env kill switch covers the whole section, and `decision_provenance.md`'s
`score_components` can carry the same tag for telemetry regardless. If review prefers
telemetry-only, the producer stops populating `cluster_bias_sources` — one-line revert.

**Known limitation:** the vocabulary records the credit BRANCH (`drive_relief`), not the
specific drive (`relieved cold`) — per-drive specificity would need the sensor name
recorded at credit time; deferred until an experiment asks for it.

Carry the credit SOURCE alongside the band: drive relief (which drive), tool success,
cross-modal credit. Producer-side only; no substrate write path changes (Wire-A's bands
are pure I/O and must stay that way — see `prompts/cluster_bias_annotation.py` docstring).

- Requires: a source tag recorded at credit time (`record_outcome` already knows whether
  it is crediting `drive_potential_diff` vs the tool-success fallback — that distinction
  currently evaporates).
- Renders as e.g. `green_flame_warm_self  [strongly rewarding — relieved cold]` vs
  `purple_flame_observe  [rewarding — action succeeded]`.
- Value: the LLM can judge *applicability* instead of pattern-matching a label; and the
  paper can state what the channel actually transmits.
- Risk: prompt-budget growth; keep to a short clause, cacheable-section rules unchanged.

## S2 — Context-aware annotation (active-cluster view)

Surface the bias for the **currently active** clusters in addition to (not instead of)
the agent-wide view.

**Do NOT simply switch to active-cluster-only.** The agent-wide aggregation is a
deliberate Roy-2c fix: priming clusters and test-fixture clusters were structurally
disjoint, and active-cluster-intersection rendering reproduced the exact bug Wire-A
exists to fix. The design must therefore be additive and explicitly ablatable:

```
=== Substrate associations from prior experience ===
  [in this context]   flame_warm_self   [strongly aversive — burned here before]
  [generally]         flame_warm_self   [rewarding — relieved cold]
```

- Requires: `get_agent_tool_biases(..., active_clusters=…)` returning both views;
  producer passes the same `ModalityClusters` the policy path uses.
- Guard: when the active cluster has no entry, the context line is omitted (never
  fabricated as neutral).
- This is what makes Exp 51 (below) winnable at all.

## S3 — Semantic (concept-keyed) transfer

Route annotation lookup through the affordance-concept layer
(`AffordanceDecompositionStrategy` → EC pattern completion — the machinery that earned
the `flame`→`fire` transfer result) instead of exact tool signatures, so a renamed but
identical affordance inherits learned value.

- Scope caution: this changes *what the LLM is told it has learned*, so it is a
  behavioural change, not a rendering change. Ship behind a flag, ablate in Exp 51's
  arm set, and do not enable by default until an experiment earns it.
- Interacts with the EC drift lesson: concept-keyed lookup widens matching, which is
  precisely the direction that historically caused over-merging. Threshold choices must
  be pre-registered, not tuned on the outcome.

## S4 — Non-stationarity measurement (free, do first)

Parse each capture row's annotation for the target tools' band and correlate band
strength with flip outcome and with decision index. Answers F6 (bias decayed ≈0.997 →
0.059 within a capture while bands are absolute) using captures already on disk. Pure
analysis; no new runs.

---

## Exp 51 — Name-copying vs learned content (the decisive experiment)

**Question.** Does the LLM follow *whatever tool strings are labelled rewarding*, or the
*learned content* those labels stand for?

**Why the current controls cannot answer it.** The transplant control is name-mismatched
(F1), and both confirmatory arms name tools that exist and are correct — so "copy the
named string" and "use the learned content" predict identically. The two must be
dissociated by construction.

### Design (the operator's door proposal, sharpened)

Make the **tool names identical** so name-copying has nothing to discriminate with, and
put the discriminating information in **context the agent can only obtain by acting**:

- One warmth affordance name (e.g. `flame_warm_self`) reachable in two contexts.
- Contexts are distinguished by a percept obtainable only after an action — the operator's
  colored-door framing: `open_door` reveals a room whose colour is the only cue.
- Behind one colour the flame relieves cold safely; behind the other it burns.
- Safety is tied to the **colour context**, never to the tool name.

Then:
- **Name-copying** predicts chance discrimination (the annotation can only ever name
  `flame_warm_self`, which is both safe and harmful depending on where you are).
- **Content-following** predicts above-chance avoidance in the harmful colour context —
  but ONLY if the channel can express context at all, which today it cannot (F4).

### Arms

| arm | annotation | prediction |
|---|---|---|
| **A0 status quo** | agent-wide aggregate (today's producer) | at chance — the substrate knows, the prompt cannot say |
| **A1 context-aware** (S2) | active-cluster view added | above chance |
| **A2 no-substrate** | none | at chance (floor) |
| **A3 inverted-injection** | synthetic `aut_nac.json` naming the harmful context as rewarding | follows the annotation ⇒ quantifies pure channel obedience |

A0-vs-A1 is an architecture A/B: same world, same substrate, different rendering. A3 is
the **channel-obedience probe** — a hand-authored substrate is not a learning claim, it
is a direct measurement of how much the LLM obeys a confident label irrespective of
provenance, which is the number a reviewer will ask for.

### Implementation constraint (decide before building)

Identical tool names across two contexts is not expressible today: entity names are
registry-unique and tools are generated `<entity>_<affordance>`. Options, cheapest first:

1. **Per-episode safety randomisation** — stable names, but which name is safe is
   re-randomised per episode, so the name→safety mapping carries no information across
   episodes while the colour→safety mapping does. Needs per-phase fixture selection
   (arcs already choose `world_entities` per phase) — no new mechanism.
2. **Arc-level effect override** — same entity, different `self_effect` per arc. Cleanest
   semantics; needs a new override seam in the component/arc layer.
3. **Conditional self_effect** keyed on a world/body sensor (door colour). Most
   expressive, biggest mechanism addition — front-gate says no unless 1 and 2 both fail.

Recommendation: start with (1); it answers the question with zero new mechanism, and it
also fixes the counterbalance-by-renaming limitation that made the 44b control
name-mismatched in the first place.

### Status

DESIGN ONLY. Becomes a pre-registration (frozen hypotheses, gates, α, stopping rule)
when S2 exists to make arm A1 runnable. Not a blocker for the 44b freeze — 44b's claim
stands on its own with F1 stated as the declared limitation; Exp 51 is what upgrades
"the substrate steers the LLM" into "the substrate's *content* steers the LLM."

---

## Sequencing

1. **S4** (analysis only, no runs) — cheapest, answers F6 today.
2. **S1** (provenance in annotation) — small, self-contained, improves every future run.
3. **A3 channel-obedience probe** — synthetic `aut_nac.json`, runnable now on the existing
   44b fixtures; gives the "does it obey any label" number immediately.
4. **S2** (context-aware annotation) — the real architectural fix; two-lens round.
5. **Exp 51** pre-registration + run once S2 lands.
6. **S3** (semantic keying) — largest behavioural change; only after Exp 51 shows whether
   context or naming is the binding constraint.

Nothing here blocks the 44b confirmatory freeze; all of it sharpens what 44b's result
means.

# E4 design review — WIRING lens

Reviewer lens: wiring (D43 real consumers / real credit path, right encoding and seams, no
hand-composed shortcut). Read: the draft prereg, DESIGN_REVIEW.md, docs/wiring/engram-formation.md,
`similarity/ec.py` (`pattern_complete_or_separate`, `_ModalityMatrix.scan`, `register_substrate_node`,
`substrate_node_metadata`, `load`), `similarity/encoder.py` (`encode`, `encode_decomposed`,
`_get_reward_overrides`, `geometry_for`), `decisions/nac.py` (`credit_node`,
`get_threshold_overrides`, `decay_reward_biases`), `integration/memory_hub.py::_wire_substrate_encoder`
/ `on_percept_received`, `imagination/trigger.py::_make_aff_encoder`, `runtime/bio_stack.py` reward
subscriber, `decisions/temporal_credit.py::distribute`, `agents/modality.py::substrate_modality`,
`agents/percept_factory.py`, and `scripts/diagnose_roy_paraphrase_collapse.py`.

**No DO-NOT-BUILD.** The prereg names the right production primitives (default `ECConfig`, real
NAc, `get_threshold_overrides(agent_id, base_threshold=ec.config.pattern_complete_threshold)`,
per-node `threshold_override=`). The findings are places where the text is loose enough that a
faithful-looking harness could measure a path production never takes.

## What I verified (matches production)

- **Production percept path is `LinguisticEncoder.encode(percept)`, decomposer `None`.** The only
  production construction is `MemoryHub._wire_substrate_encoder` (`LinguisticEncoder(ec, atl, nac=self.nac,
  decomposer=...)`), and the decomposer is wired only under `MAXIM_CONCEPT_DECOMPOSITION=1`. With it
  unset, `encode()` takes the single-node branch: `embed(text)` → `_get_reward_overrides(percept)` →
  `ec.pattern_complete_or_separate(..., threshold_override=, geometry=geometry_for(...))` →
  `register_substrate_node(..., geometry=)` on `is_new` → `atl.activate_substrate_node` →
  `nac.update_eligibility`.
- **Override dict keys are EC node ids** (the uuid4 from `PatternResult.node_id`, which is also the ATL
  record id); `_ModalityMatrix.scan` maps them through `row_of`. Keys for nodes in other modalities are
  ignored harmlessly.
- **base_threshold source**: both encoder paths pass `self.ec.config.pattern_complete_threshold`; production
  EC (`bio_stack.py`, `agent_factory.py`) is `ECConfig(persistence_path=...)`, so 0.44; `EC.load` does not
  overwrite the threshold or `frozen_centroid_modalities` (it `replace`s five LSH fields only). Confirmed
  live: default `ECConfig()` → 0.44, frozen `{"audio","interoception","world"}`, so `"text"` runs the
  running mean.
- **Override arithmetic**: `credit_node` adds `reward_bias_alpha (0.15) × reward`, clamps `[0, 0.20]`;
  `get_threshold_overrides` returns `max(0.10, base − bias)` for biases ≥ 0.001, keyed `(agent_id, node_id)`.
- **Scan semantics**: eligibility is per-row `sim ≥ own threshold`, then **argmax similarity among eligible
  rows**. A widened node wins a string only if it is the nearest eligible node. That is the production
  decision and the harness must inherit it, not re-implement it.
- **Sensor surface never widens**: `SensorEncoder.encode_sensors` passes `threshold=` only. Correct as
  stated in the prereg.
- **Recall never widens or writes**: `bio_enrichment` uses `pattern_complete_readonly` with no override.
  It is correctly out of scope.

## Findings

### W1 — SHOULD-FIX: the prereg allows the hand-composed sequence, and the script it cites is that sequence (and no longer runs)

The prereg says the override is "applied per node through the same `threshold_override=` the encoder
passes", and that the walk order is "exactly as `run_cell` walks them". Both can be read as permission
to reuse `scripts/diagnose_roy_paraphrase_collapse.py`. But that script's `_encode_text` is a
**hand-composed EC call**: it calls `ec.pattern_complete_or_separate(embedding=..., modality="text")` directly
with **no NAc, no `threshold_override`, no `geometry`**, and `register_substrate_node` with no geometry. Its
`_build_stack` also sets `frozen_centroid_modalities={"interoception"}`, threshold 0.40, and a stub ATL.
At HEAD, that call raises `TypeError: ... missing 1 required keyword-only argument: 'geometry'` (I verified
this). The obvious patch, `geometry=None`, silently disables the gate-1 mask. A harness that computes
`nac.get_threshold_overrides(...)` itself and hands it to `ec.pattern_complete_or_separate` would agree
with production today. It would still be measuring a composition production never runs: if the
encoder's override plumbing changed, for example its agent_id source or the `if overrides else None`
step, the harness would keep passing.

**Proposed prereg text (Apparatus):** "Every string is encoded by `LinguisticEncoder(ec=ec, atl=atl, nac=nac,
decomposer=None).encode(percept)`, the construction `MemoryHub._wire_substrate_encoder` uses with
`MAXIM_CONCEPT_DECOMPOSITION` unset. The harness never calls `ec.pattern_complete_or_separate` or
`register_substrate_node` itself, and imports nothing from `scripts/diagnose_roy_paraphrase_collapse.py`
except the walk order (which it re-derives from the fixture). Wiring assertion (exit 4 on failure):
on the first encode after the credit, a spy on `ec.pattern_complete_or_separate` records that
`threshold_override == {rewarded_id: 0.44 − b}` (b > 0) or `None` (b = 0), and that `geometry` equals
`encoder.geometry_for(embedding, "text")`."

### W2 — SHOULD-FIX: agent_id keying is unstated; one mismatch makes every arm a silent b = 0 run

`encode()` reads overrides under `percept.context.agent_id or ""`. `credit_node` writes under whatever
agent_id the harness passes. If the harness credits `"e4"` and builds percepts without a context, or
with `agent_id=None` (the default in `make_text_percept`), `get_threshold_overrides("")` is empty. Every
arm then runs at b = 0. The positive control would catch it at RA b = 0.2, but only as an
uninterpretable refusal.

**Proposed prereg text:** "Percepts are built with `make_text_percept(text, agent_id=AGENT)` (sensory
NARRATIVE → substrate modality `"text"`). `credit_node(AGENT, node_id, b / reward_bias_alpha)` uses the
same `AGENT`. After each credit, assert `nac.reward_bias(AGENT, node_id) == b` (±1e-12), and record the
effective override value the run actually used (0.44 − b in float, for example 0.24000000000000002)."
Also assert `MAXIM_NAC_REWARD_BIAS_DISABLED` is unset. It turns `credit_node` into a no-op and
`get_threshold_overrides` into `{}`.

### W3 — SHOULD-FIX: say whether the measured path is reachable in production, rather than implying it

The prereg's Question says "NAc reward widens EC recognition on the TEXT surface". In code, that is
true only under conditions the prereg should name:
1. The percept text path exists only under `MAXIM_SUBSTRATE_PATH=1` (opt-in. Nothing in `src/` sets it,
   and `_wire_substrate_encoder` returns early otherwise).
2. A text node acquires `reward_bias` only through `TemporalCreditDistributor.distribute(agent_id, …)`
   (`bio_stack` reward subscriber, agent_id from the **reaction's** context). That matches eligibility
   written by `encode()` under the **percept's** `context.agent_id or ""`. Most single-agent text percept
   sites (`conversational_source.py`, `minecraft.py`, `agent_loop.py`) call `make_text_percept` without
   `agent_id`, so text eligibility lands under `""`. A text node is then credited only when the reaction
   agent_id is also `""`. Pain with `agent_id is None` or the world id distributes nothing.

I did not trace every reaction emitter, so this is **unverified, not a finding that the path is dead**.
It still bears on what a COLLAPSE result licenses: a COLLAPSE result triggers an EC centroid-rule
change, which must re-run the EC completion ledger row. That is a real cost to pay for a hazard whose
live preconditions have not been shown.

**Proposed prereg text (What this does not claim):** "The measured path is live in production only under
`MAXIM_SUBSTRATE_PATH=1`, and a text node is rewarded only when the reaction's agent_id matches the
percept's context agent_id (or both are `""`). Whether any shipped configuration meets both is
[cited run artifact with a non-empty text-node `reward_bias` / not established]. A COLLAPSE verdict
names a hazard conditional on those preconditions." Optionally add a no-data, pre-run check that
greps committed `aut_nac.json` artifacts for `reward_bias` keys that resolve to `"text"` EC nodes.

### W4 — SHOULD-FIX: "text surface only" is the wrong scope; the LinguisticEncoder also widens the `"vision"` modality

`substrate_modality()` maps `SensoryModality.SIGHT` (or `percept.modality == "vision"`) to `"vision"`. The same
`encode()` passes the same override dict for those percepts. `"vision"` is **not** in
`frozen_centroid_modalities`, so it runs the same running mean. The widening surface is "LinguisticEncoder
modalities (`text`, `vision`)", not "text". The second production text path,
`imagination/trigger.py::_make_aff_encoder` → `encode_decomposed(aff_name, "text", agent_id)` with
`AffordanceDecompositionStrategy`, also widens and writes **into the same `"text"` matrix** as percepts.

**Proposed prereg text:** In the Question, say "the LinguisticEncoder surface (`text` and `vision` modalities;
both running-mean)". In the scope line, say "This measures the `encode(percept)` NARRATIVE→`text` path with
decomposition off. `vision`-modality percepts and affordance-name chunks (`encode_decomposed` via the
affordance decomposer, which shares the `text` matrix) widen by the same mechanism and are not measured."
This does not change the design. It stops a NO COLLAPSE result from being read as covering them.

### W5 — NIT: read the centroid through the public accessor, and take the "first embedding" from the encode itself

The drift metric needs the node's first embedding and its final centroid. Use `percept.embedding`
(set by `encode()`) of `"you sense food nearby."` for the first, and
`ec.substrate_node_metadata(rewarded_id)["embedding"]` / `["member_count"]` for the final. Do not read
`ec._substrate_nodes`. Record `member_count` next to the cosine: a count of 2 with drift is the pair_01
partner, and anything above 2 is absorption.

### W6 — NIT: refuse fallback through the canonical apparatus check plus the realized-state stamp

Name the refusal mechanism. Call `require_semantic_encoder("paraphrase-mpnet-base-v2", context="E4")`
before measuring. After the run, assert `ec.encoder_provenance["linguistic"]["using_fallback"] is False`
and `embedding_dim == 768` for every EC in every arm, including each isolated fresh EC. That stamp is
written at encode time and is the only source that knows the realized state. It is stronger than
`encoder.using_fallback` on one instance. Every geometry tag then carries the same model and fallback
flag, so the gate-1 mask is inert across the run, as in production.

### W7 — NIT: isolated arm needs a fresh encoder, not just a fresh EC/NAc; ATL choice stated

`LinguisticEncoder` holds its `ec` and `nac` by reference, so each isolated trial needs a new encoder
instance. The model singleton is shared, which is fine. State the ATL used. The real `ATL()` (as
`scripts/fine_sweep_phase_2.py` does) is preferred. A stub is acceptable because ATL activation does not
feed EC matching, but the prereg should say which it is.

### W8 — NIT: "held there (no decay)" is a deliberate departure from production; say so

Production decays `reward_bias` via `decay_reward_biases` (τ 50 ticks), and `encode()` itself never decays.
The harness calls no tick method, so the bias is held at `b` for the whole walk. That makes `b = 0.2` an
upper bound on any real run's widening exposure. Add "No tick/decay method is called. The
held bias is an upper bound on production exposure" to Apparatus. The credit is written by
`credit_node` directly, not through `distribute()`. That is the same write production's distributor
performs, so it is acceptable. Say that too, so the code review does not flag it as a bypass.

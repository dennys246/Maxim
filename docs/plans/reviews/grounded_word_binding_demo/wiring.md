# Wiring lens — grounded word binding demo (plan v2)

**Verdict: ADOPT WITH CHANGES.** The plan is mostly accurate about what exists, and its
front-gate instinct (ride on shipped pieces) is right. But three compositions are named as if
they were wiring that already exists, and they are the D43 shape: the pieces ship and nothing
joins them. (1) Nothing reads `NAMES` on the selection path. (2) Nothing lets "own experience
win" over a foreign entry. (3) The live ingest is described as "a path, not new math", but its
ATL half is new math and its NAc half is lock-free by contract. Stage 1's `text` modality also
leaks into two consumers that the plan does not mention: operant credit routing, and 2S-c
novelty. The three DO-NOT-BUILD items block the named stages' builds, not the plan.

**Read against `origin/main` @ eda35efb (#879, 2S-c merged).** This branch (6ebcf9b1) forks
from before #879, so `LLMProposal.cluster_margins` and `situation_novelty` do not exist in this
worktree's `src/`. The plan's table cites them correctly for main. Rebase before any stage PR.

---

## DO-NOT-BUILD

### DNB-1 — Stage 2/3: `NAMES` has no reader, and the reader it needs has no slot to write into
**Evidence.** The selection path reads only `{modality: cluster}`:
`agent_loop.py::propose_via_substrate` → `NAc.anticipatory_threat_need(agent_id, clusters)`
(min over the active clusters' fear) and `NAc.recommend_action(current_clusters=clusters)`
(additive `cluster_reward_bias`). NAc holds no ATL reference (`NAc.__init__` wires `ec` only),
and no production code calls `ATL.find_by_relationship` on this path. Stage 2 writes an edge that
no consumer reads. The "reference" route of Exp A ("the word reactivates its situation, and the
situation's own fear/want fires") is exactly the composition that is missing. The plan names the
write (`NAMES`) and the measurement, but not the read.

The obvious fix doesn't fit the type either. `ModalityClusters = dict[str, str]`
(`decisions/nac.py`) holds one id per modality. An evoked world cluster cannot sit next to the
*sensed* world cluster under the key `"world"`. Putting it into the same map would also route it
into every WRITE consumer of that map:
- `note_active_clusters` → `pain_bus.py::_on_pain` books fear on `.get("world")`, which is fear
  on an imagined situation;
- `LLMProposal.clusters` → `tool_dispatch.py::record_outcome` credit;
- `_loop_capture_action(situation=proposal.clusters)`, which is the 2S-b durable key.

**Fix.** Name the read seam in the plan before Stage 2 is built. For example, an
`evoked_clusters` (or `evoked_fear_need`) input, computed in `propose_via_substrate` from
`find_by_relationship(text_id, "NAMES")` weighted by the edge's `weight`/`confidence`. It is read
ONLY by `anticipatory_threat_need` and the `recommend_action` cluster term, and is never passed
to `note_active_clusters`, `proposal.clusters`, or the capture. State which ATL handle the loop
passes, and add a guard test proving that the evoked cluster writes no credit, no fear, and no
situation.

### DNB-2 — Stage 3 (Exp A): the "conditioning" route is closed by wiring, so the discriminator is vacuous
**Evidence.** Fear writes land only on the world cluster:
`proprioception/pain_bus.py::_on_pain` → `nac.active_clusters(agent_id).get("world")` →
`record_cluster_fear`. Nothing can write fear onto a `text` cluster. Want (cluster reward) goes to
interoception only for drive relief and the tool-success floor (`tool_dispatch.py::record_outcome`,
`credit_cluster = intero_cluster`). The one path that could write it onto `text` is the operant
fallback bug in SF-1. So in Exp A the arm that is meant to tell reference from conditioning cannot
express conditioning at all. "Fear to the word ⇒ reference" would pass by construction. This is
the tautology trap in `docs/wiring/substrate-learning-channels.md`.

**Fix.** Choose one before the prereg:
- (a) Open a declared text-conditioning write, such as fear on the co-active text cluster, as its
  own flag, so both routes are live and the design really discriminates.
- (b) Record that conditioning is excluded by wiring, and narrow the claim to "the `NAMES` read
  path carries fear". The deletion arm then carries all the weight.

Either way, state it in the prereg.

### DNB-3 — Stage 6 / Exp C: nothing lets "own experience win"
**Evidence.**
- **Fear.** `hivemind/merge.py::nac_merge` folds `cluster_fear` by MIN (the deepest fear
  survives), and `substrate_merge` applies tighten-only. There is no extinction write path:
  `NAc.record_cluster_fear` only moves values downward, and fear has no per-tick decay by design,
  only slow wall-clock decay. A wrong foreign fear (the "inverted valence" corrupted arm, discounted
  to ×0.75 by `ingest.py::FOREIGN_FEAR_DISCOUNT`, which is still ≥ θ) cannot be overridden by the
  receiver's safe experience.
- **Want.** `_merge_mean_clamped` takes an unweighted mean on shared keys and adopts the donor's
  value wholesale on keys the receiver lacks. That is not "own experience winning".
- **Provenance and learned trust.** `cluster_fear` and `cluster_reward_bias` values carry no
  contributor after the merge; `cluster_reward_source` is credit-source, not contributor. So
  neither per-source revert nor "learned trust updates how much it trusts that source" has
  anything to act on.

Exp C's predicted protection ("its own experience overrides") therefore rests on a mechanism that
does not exist. The gate can reduce how often the agent consults, but it cannot correct a bad
consult once taken.

**Fix.** Before Stage 6 is built, name the override mechanism:
- a fear-extinction write, such as active re-learning on the unreinforced situation (the brief
  already frames extinction as re-learning);
- a foreign-shadow store read with a discount, kept separate from own experience rather than
  merged into it;
- per-entry contributor provenance on the fear and bias maps (a CC3/format-version change).

Then write Exp C's prediction against that mechanism. Otherwise, register the plain prediction on
current wiring: the gated arm is harmed less only because it consults less.

---

## SHOULD-FIX

**SF-1 — Stage 1: a `text` cluster hijacks the operant/extero credit target on Minecraft.**
`tool_dispatch.py::record_outcome` picks `operant_cluster = clusters[AUDIO_TAG]`, else
`sorted(non-intero tags)[0]`. The Minecraft bodies declare `world` and no `audio`
(`minecraft_player.yaml`), so adding `text` makes `"text" < "world"` the target for
`orient_relief` credit and for `set_pending_operant_action`. This is latent today: Minecraft eat
is credited as interoception at `tool_bridge.py` "drive_relief_channel". It is silent when it
fires. **Fix:** exclude `text` from credit targets explicitly, and add a guard test, in the same
commit that adds the modality.

**SF-2 — Stage 1: 2S-c novelty and the consult trigger mix incompatible scales.**
`agent_loop.py::situation_novelty` returns the MAX of `1 − margin` across modalities. Sensor
margins sit at ≥ 0.85 (novelty ≈ 0–0.15). Text completes at 0.44 with a running mean, so a
familiar word reads as novelty ≈ 0.3–0.5 and would dominate every capture's recorded novelty,
which feeds memory strength (Phase 2b). `_situation_margins` reads
`SensorEncoder.last_encode_margin`, and a `LinguisticEncoder` encode never stashes there, so the
text margin will instead be *silently absent*. **Fix:** decide per modality. Either exclude `text`
from `situation_novelty`, or normalise each margin by its own modality threshold. Declare the
choice, because it changes a 2S-c record.

**SF-3 — Stage 1: text has no entry point into `propose_via_substrate`.**
The function receives `nac`, `executor` and `sensor_encoder`, and no observation. A
`ModalityChannel` is a float-dict reader for `SensorEncoder.encode_sensors` (384-d), so text
cannot be "one tuple entry". Heard text is drained at §1 (`sim.next_observation`) on every pass,
but the substrate tick fires only when `_substrate_tick_due` says so (about 0.5 s; see
runtime-tools brief, the wake-source invariant). A message drained on a non-due pass is lost unless
something holds it. The "persistence window" therefore needs a named holder, loop-owned and
per-agent, filled at §1 and read at the substrate branch. Two related paths also need a decision:
- `_encode_current_clusters` and `_attach_live_situation` (llm-primary) iterate
  `_SUBSTRATE_CHANNELS` and will not carry text. Decide whether they should.
- "ONE text encoder" must also mean one call site. With `MAXIM_SUBSTRATE_PATH=1`,
  `MemoryHub.on_percept_received` (via `memory_agent.py`) would encode the same percept a second
  time, which double-updates a running-mean centroid and NAc eligibility.

**SF-4 — Stage 1: the text modality's centroid policy is unmeasured for this use.**
`text` is running-mean. `ECConfig.frozen_centroid_modalities = {interoception, audio, world}` is
test-pinned equal to `hivemind/merge.py::DEFAULT_FROZEN_CENTROID_MODALITIES`. The bio-memory brief
already carries a `[behavioral]` drift rule: measure isolated vs sequential for any new encoding
path. Its history is 19 of 20 strings collapsing into one mega-node at 0.40. The text space is
also shared. The affordance-name `LinguisticEncoder` and the LLM percept encoder write the same EC
`"text"` modality (bio-memory brief, the affordance-encoder invariant), so a heard `fire` can
complete onto an affordance-name node. Separately, `NAc.get_threshold_overrides` lowers text
thresholds per node by reward bias. **Fix:** Stage 1 runs the isolated/sequential measurement.
Decide between frozen-or-not and a distinct modality tag (for example `heard`). If frozen, move
both pinned sets together.

**SF-5 — Stage 1: one text slot.** `ModalityClusters` holds one id per tag, so two words inside
the persistence window collapse to the last one. Declare either "last heard wins" or several tags.

**SF-6 — Stage 2: `PerceptTraceBuffer` cannot answer "which situation" yet.** `TraceEntry` is
`(agent_id, percept_id, tick, activation)` with no modality. It has no production constructor (✓
plan), no `record` producer and no `tick()` caller. The binding needs world-cluster ids per tick,
tagged by modality, recorded after the encode in `propose_via_substrate`, on the substrate tick
clock. Put this into R4's design review as a requirement from this consumer.

**SF-7 — Stage 2: `NAMES` will silently no-op unless it is registered first.**
`memory/semantics.py::Semantics.define` returns `False` for an unregistered type, which is the
silent-no-op class. A repeated `define` appends a duplicate edge rather than strengthening it;
`update_edge` is the strengthen path. ATL eviction and compression also remove graph nodes
(`atl.py` `remove_node`), which takes their `NAMES` edges with them. **Fix:** register `NAMES` at
ATL construction, strengthen through `update_edge`, raise on a `False` return, and state what
eviction does to a binding.

**SF-8 — Stage 4: the live-ingest composition, element by element.**
- *Quiescence.* `NAc.load_state` docstring: "Does NOT acquire the NAc mutex … callers expect
  load-time quiescence". `EntorhinalCortex` has no lock. Other threads touch these stores:
  ConceptExtractor worker threads (`concept_extractor.py`, ATL `add_ref`), the Hippocampus capture
  worker, and PainBus subscribers, which run on the publisher's thread. Name the hook — the
  natural one is `_loop_bio_tick_maintenance` (§8.5, loop thread) — and list which writers it
  excludes.
- *Lost update.* Fetching off the loop thread is fine. The dump → merge must happen at the
  boundary, on the loop thread. A merge computed against a dump taken earlier overwrites whatever
  the agent learned in between.
- *Missing surfaces.* EC has no live dump. It has `save(path)`, `load(path)` and the live
  `ingest_substrate_nodes`, which is correct: it invalidates the matrix cache and recounts
  geometries. EC is not in `memory/snapshot.py::SNAPSHOT_KINDS`, so it needs a
  `substrate_nodes` dump.
- *ATL is new math.* `ATL.load_state` clears and replaces, and no `atl_merge` exists. Merging
  concepts and `NAMES` edges (dedup, weight aggregation, re-key) is new merge math. Correct the
  front-gate line "a path, not new math".
- *Revert.* A pre-merge snapshot restore also discards everything learned after the pull. Say so,
  or make the revert a per-source subtraction, which needs the DNB-3 provenance.
- *Not stale (verified).* Receiver ids survive `substrate_merge`, and frozen modalities keep the
  receiver's centroids. So `ctrl.pending_proposal.clusters`, NAc `_eligibility`,
  `_noted_active_clusters`, `_pending_operant_action`, the `SensorEncoder._last_node_id` min-delta
  cache and Hippocampus `situation` keys all stay valid across a load. The exception is text:
  running-mean receiver text centroids move on merge (`ec_merge_aligned` weighted mean), so a
  text-modality ingest *does* shift live text matching.

**SF-9 — Stage 6: the trigger signals are not the signals the plan names.**
- Welford variance is per `(agent_id, event_signature)`, meaning per tool
  (`NAc.get_action_risk_profile`). It is situation-blind, so it cannot say "uncertain in this
  situation".
- "Recent negative outcomes here" (the bee rule) has no store. `cluster_reward_bias` is a decayed
  sum with no recency.
- When `recommend_action` falls below `min_confidence` it returns `None`, and its best score
  reaches only the telemetry event. `propose_via_substrate` then returns `None`, so no proposal
  carries the margins either. They remain readable from `SensorEncoder.last_encode_margin` until
  the next encode.
- "Pain just felt": `bio_integration.consume_pain_intensity` is destructive and is read at capture
  time, after execution. A trigger that calls it steals the value from the 2S-c capture.

The only point where drives, fear need, clusters, margins and the recommendation are all in scope
is inside `propose_via_substrate`, after `recommend_action`. **Fix:** define a per-tick report
returned from there (proposal or no proposal), add a non-destructive pain peek, and replace
"outcome variance" and "failing here" with signals that are keyed by situation, or build them.

**SF-10 — Stages 4/5: text geometry must match across the pair.** `LinguisticEncoder` produces
768-d vectors only when the `semantic` extra is installed; otherwise it produces a 384-d hash
(`EncoderConfig.fallback_dim`). Geometry tags and `strict_geometry=True` at ingest will refuse or
separate mismatches, which is correct. But a donor at 768 and a receiver at 384 means the
receiver's live heard word never completes onto the donor's text node, so the transferred binding
is unreachable while the transfer "succeeds". **Fix:** the harness asserts that both sides share
the same text geometry.

---

## NIT

- `similarity/ec.py::_cosine` → the symbol is `_cosine_similarity`. `hivemind/merge.py::_cosine`
  is the merge-side twin; both return 0.0 on a dimension mismatch.
- `cli.py::_run_ingest` → `hivemind/cli.py::_run_ingest`.
- "Not the generic 0.44": 0.44 is the live EC text threshold (`ECConfig.pattern_complete_threshold`).
  The merge already plumbs per-modality thresholds (`ec_merge_aligned(modality_thresholds=)`, with
  `SENSOR_MODALITY_THRESHOLDS` = 0.85). A calibrated text value must equal the live EC's text
  completion threshold, or a merge aligns differently from how the live EC completes.
- Stage 1 should cite `scripts/selection_dynamics_rebaseline.py`. The `_SUBSTRATE_CHANNELS`
  comment requires it for a new channel, as the world channel did.
- Stage 5 "dangling arm": ingest's dangling rule drops relations that are missing an endpoint, so
  the arm measures the ingest filter unless it bypasses the filter. State which it is.
- `docs/agents/bio-memory.md` §config table lists frozen modalities as `{interoception, audio}`,
  which is stale (world was added). This belongs to the brief's owner.

---

## Claim verification (against origin/main @ eda35efb)

| Plan claim | Verdict | Evidence |
|---|---|---|
| `propose_via_substrate` → `NAc.recommend_action` over `{modality: EC cluster}` | ✓ | `agent_loop.py::propose_via_substrate`, `current_clusters=clusters` |
| `_SUBSTRATE_CHANNELS` excludes text | ✓ | `agent_loop.py::_SUBSTRATE_CHANNELS` = interoception, audio, world |
| SensorEncoder 384-d; cluster ids ARE ATL concept ids | ✓ | `encoder.py::SensorEncoder.encode_sensors` → `atl.activate_substrate_node(node_id=…)` on every encode; `concept_extractor.py::_link_situation` relies on it. Donor-only nodes gain a concept only on their first live encode |
| `LinguisticEncoder` only via MemoryHub with `MAXIM_SUBSTRATE_PATH=1` | ✓ (with caveat) | `memory_hub.py::_wire_substrate_encoder`. It is 768-d only with the `semantic` extra (else 384 hash). Affordance and imagination encoders also write EC `"text"` |
| `PerceptTraceBuffer` zero production constructors | ✓ | Only `memory/snapshot.py` type references |
| 384↔768 → 0.0 | ✓ (NIT on the name) | `ec.py::_cosine_similarity`, `merge.py::_cosine` |
| `cluster_margins` / novelty `1 − best similarity`, recorded only | ✓ on main, ✗ on this branch | `agent_loop.py::_situation_margins`, `situation_novelty`, `_loop_capture_action` |
| `recommend_action` → `None` below `min_confidence` | ✓ | `nac.py::recommend_action`; `_resolve_min_confidence` default 0.3 |
| Welford outcome variance | ✓ but per tool, not per situation | `nac.py::get_action_risk_profile` |
| `anticipatory_threat_need` | ✓ | `nac.py::anticipatory_threat_need` (min fear over active clusters, θ-gated) |
| Bundle = manifest + NAc + EC; ATL reserved | ✓ | `hivemind/bundle.py` module docstring |
| `substrate_merge` re-keys donor only | ✓ | `merge.py::substrate_merge` steps 1–3 |
| Merge output loads into a live system via `load_state()` | Partly | NAc `load_state` ✓ (lock-free, quiescence assumed); EC via `ingest_substrate_nodes` (live) ✓; EC has no `dump`/`load_state`; ATL `load_state` is clear-and-replace, not a merge |
| Live ingest refused (contract §1) | ✓ | `hivemind/cli.py::_run_ingest` docstring and closing NOTE |
| Oasis search absent (list + whole fetch) | ✓ | `hivemind/oasis_endpoints.py`: `GET /v1/substrate/releases`, `/bundle/<id>` |
| ATL relationship API can hold a new `NAMES` type | ✓ (see SF-7) | `semantic_types.py::RelationshipRegistry.register` (persisted via `ATL.dump` "registry"); `atl.py::define_relationship`/`find_by_relationship` |
| "Everything else rides … the cue reads signals the loop already computes" | ✗ | DNB-1 (no `NAMES` read), SF-9 (the trigger's signals) |
| Front-gate 1 "a path, not new math" | ✗ for ATL | SF-8 |
| "Own experience winning (the ant rule)" | ✗ | DNB-3 |

## Verified fine

- Receiver-id survival makes every live runtime key (pending proposal, eligibility, noted
  clusters, pending operant, the encoder's min-delta cache, Hippocampus situations) safe across a
  merge, for frozen modalities.
- `EC.ingest_substrate_nodes` is the right live-EC entry point: it keeps member counts and
  invalidates the matrix cache in a `finally`.
- A `text` cluster in the active set does NOT pollute fear. The pain subscriber writes `world`
  only, and `anticipatory_threat_need` reads text as 0 fear. It does not pollute the interoception
  credit floor either. The pollution risks are SF-1 (operant) and SF-2 (novelty).
- The per-modality merge thresholds already exist (`modality_thresholds`), so no new mechanism is
  needed for the text threshold.
- Geometry stamping on text nodes (`LinguisticEncoder.geometry_for`) plus `strict_geometry=True`
  at ingest already guards against cross-space merges. SF-10 is about making mismatch visible in
  the harness, not about safety.
- `ATL` persists the relationship registry and graph (`ATL.dump` "graph"/"registry"), so a
  registered `NAMES` survives a session.

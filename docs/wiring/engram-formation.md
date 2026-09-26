# How engrams form in Maxim — a living tracker

**Established:** 2026-09-25, from a code audit of the substrate, episodic/semantic and motor paths
(every claim below was read in `src/`, not run, unless it cites an experiment). **Update this page**
whenever a mechanism it names changes, an issue it links closes, or a ledger row it cites moves —
the table in §1 is the one-glance answer and must not drift from the sections under it.

**What "engram" means here.** A memory trace that (1) **forms** from one experience, (2) stays
**specific** — a different situation does not land on it, (3) is **recalled** from a partial cue,
and (4) **changes behaviour** when recalled. A trace that forms but is never read is a record, not an
engram; the tracker scores all four properties separately because the codebase has traces at every
stage of that ladder.

Fix plan for every gap below: [docs/plans/engram_formation.md](../plans/engram_formation.md).
Mental model of the substrate chain: [docs/agents/bio-memory.md](../agents/bio-memory.md) §1.

## 1. Status at a glance

| Engram family | Forms | Specific | Recalled from a cue | Changes behaviour | Evidence (ledger) |
|---|---|---|---|---|---|
| **Situation engram** — EC sensor cluster + NAc `cluster_fear` / `cluster_reward_bias` | ✅ | ⚠️ neutral→extreme swings only; daily wrap boundary (#899) | ✅ EC completion | ✅ **substrate-primary, no LLM** | **EARNED** ×7 (Exp 45, 52, 53b, 56, 60, 61, 62 rung A) |
| **Recognition engram** — EC text node + node-keyed `reward_bias` | ✅ | ⚠️ running-mean drift; reward widening compounds it (#911) | ✅ text only | LLM prompt text + `tool:*` nudge ≤ 0.20 | PARTIAL / pulled from 1.0 framing |
| **Episodic engram** — Hippocampus trace | ✅ honest encoding | ✅ per-situation key (2S-b) | ✅ built (2S-d cue) — **result discarded** | LLM prompt text only | Exp 10 (cross-session persistence) |
| **Semantic engram** — ATL concept | ✅ by NAME | n/a | ✅ by name | LLM prompt text only | none |
| **Motor engram** — Cerebellum program ↔ Hippocampus trace | ❌ no production caller (#909) | — | — | ❌ | none |
| *(Cerebellum forward model — a prediction, not an engram)* | ✅ trains on real readings | — | — | ❌ read side dormant; **never saved** (#908) | none |

The one family that is an engram on all four counts **without the LLM** is the situation engram —
and it is exactly as good as the world channel's geometry lets it be.

## 2. Situation engrams (the earned family)

```
sensor readings ─► SensorEncoder._sensor_embed (sum of SHA-basis vectors, A4 gain on `world`)
                ─► EC.pattern_complete_or_separate @ 0.85, frozen prototype
                ─► node id = the situation key
pain (drive:oxygen / drive:health) ─► NAc.record_cluster_fear(world cluster)   ─┐
relief / tool success / operant    ─► NAc.update_cluster_reward(cluster)       ─┤
                                                                                 ▼
next time the reading completes into that node ─► anticipatory_threat_need / cluster term
                                                ─► NAc.recommend_action ─► the body acts
```

**Formation.** One node per channel (interoception, audio, world) per encode, at most. Match when
cosine ≥ `SensorEncoderConfig.pattern_threshold` **0.85**, else allocate. Sensor modalities are
**frozen**: the first vector is the prototype forever; later matches only raise the count — so these
engrams do not drift. No capacity limit and no eviction; the only removal is
`maxim substrate invalidate`. A reading within `min_delta` 0.05 of the last one skips the scan and
reuses the cached node (its margin is deliberately recorded as "not measured").

**Valence attaches to the node.**
- Fear: `−cluster_fear_alpha (0.5) × intensity`, clamped `[−1, 0]`, world cluster only, failure
  modes allowlisted to `drive:oxygen` / `drive:health`. Read above θ = 0.5. **No tick decay** by
  design (extinction is re-learning); slow wall-clock class on load.
- Want: `cluster_reward_bias` step `reward_bias_alpha` 0.15, clamped `±1`, tick-decayed at τ 300.

**Specificity — the limit that matters.** Cosine sees direction, not magnitude, so a situation
separates only when some sensor swings **neutral → extreme** (or across neutral). A one-sided move
never separates at any gain. Full account:
[cosine-separation-is-directional.md](cosine-separation-is-directional.md).
- Exp 62 transferred because the world channel's only live discriminator is the binary
  `is_in_water`: the engram is invariant to low-gain place absolutes, **not** to a changed situation.
- **The daily wrap (#899):** `time_of_day` is linear, so the pool reads 0.847 at 0.95 and **0.799 at
  0.99** against 0.85 — the drowning fear holds through midnight (0.903 at 0.75) and misses only the
  last ~5 % of the day. Invisible under the frozen-day protocol.
- **No reward widening on sensors (#911):** `SensorEncoder` passes no `threshold_override`, so a
  rewarded situation is recognised no more readily than an unrewarded one.

**Transfer.** `cluster_*` values and fear re-key through the aligned EC `id_map` at ingest (fear
discounted ×0.75, `FOREIGN_FEAR_DISCOUNT`). Exp 56 (want) and Exp 61 (fear) are EARNED on this path.

## 3. Recognition engrams (text node + `reward_bias`)

Text percepts and affordance names encode to EC nodes at **0.44** with a **running-mean** centroid.
Eligibility is set to 1.0 (new node) or the match similarity, decays ×0.9 per tick; credit lands via
`credit_node`: `+0.15 × credit`, clamped `[0, 0.20]`, and a zero bias is removed — **pain never
creates a `reward_bias`**, it can only erase one.

- **Readout.** The bias lowers that node's EC threshold (`max(0.10, 0.44 − bias)`) — it makes
  recognition more permissive — and feeds prompt annotations. `recommend_action` reads `reward_bias`
  only for `tool:*` keys (a ≤ 0.20 nudge, cluster-blind). Routing trace credit to the selection
  surface is roadmap 1.4 Phase 5's R4 item, not this tracker's.
- **Drift hazard (#911).** Widening + running mean = a rewarded node accepts looser matches and
  averages them in. Unmeasured.
- **Dead danger label (#910).** Both affordance annotators print `[DANGEROUS]` for `reward_bias <
  −0.01`, which the clamp makes impossible.
- **Transfer.** Node-keyed `reward_bias` is **dropped at ingest by design** (it cannot be re-keyed
  onto a receiver situation — `ingest.py`, `keep_agent_rows`); making it transfer is the deferred
  [transfer_non_situation_nac_rows.md](../plans/deferred/transfer_non_situation_nac_rows.md). Not a
  bug.

## 4. Episodic engrams (Hippocampus)

**Formation — honest.** Every `Hippocampus.capture` declares what it measured
(`EncodingSignals`: salience / novelty / surprise / pain, `None` = not measured). The loop capture
(one per executed action) measures surprise (|RPE|), nociceptive pain, situation novelty and
per-drive pressure/relief, and carries the **situation** (`{modality: cluster id}`) the action was
chosen in, plus `encoded_at_us` / `capture_seq`. The situation's clusters link the trace into the
matching ATL concepts (2S-b). Storage strength `S0 = s_base (10 s) × (1 + k·tag)` is stamped on
every trace but read only under `memory.strategy=strength`.

**Consolidation — mostly nominal on the default path.**
- FORMING → SHORT_TERM is Dormant (#817); entries never leave FORMING (pool trimmed to 32).
- `--sim` ends with `on_session_end_lightweight` → **no `sleep()`**, so no forgetting, promotion or
  retro-tagging there. Survival harnesses do run `sleep()` but pin `access_based`, whose wall-clock
  retention rewards `access_count` — i.e. the traces the LLM happens to read.
- The strength model, retro-tagging (2d-2) and the fading tag floor are built and opt-in;
  `S_BASE` is a placeholder until Phase 5 calibrates it.

**Recall — built.** `recall()`, `PatternCompleter.complete` (cued by concept NAME), and the 2S-d
situation cue (`cue_situation` / `recall_situation`: shared world/audio cluster, ranked by
`max(encoding_tag, retro_tag)`, recall-only).

**Behaviour — LLM text only.** `agent_loop.py` calls the situation cue and **discards the result**;
its consumer (2S-e) is parked. In the survival world nothing counts as memory *use* at all (#848).
Enrichment renders ≤ 3 episodes into the prompt.

## 5. Semantic engrams (ATL)

Concepts form by **name** — detected objects/people, location, goal noun chunks, tool and skill
names. Survival percepts carry no text, so survival concepts are goal/tool strings; the situation
link (2S-b) is the only substrate-native tie. Relationships start at weight/confidence 0.3 (≤ 6 per
episode); refs cap at `MAX_REFS_PER_LAYER` 200 (insertion-order eviction — a lossy index; the
trace's own `situation` is the durable key). Behaviour: prompt text only. Known ATL defects: #812,
#816.

## 6. Motor engrams (Cerebellum)

Designed as a Hippocampus trace (`site="engram"`, `tool_name="motor_program:<name>"`) linked to a
`cerebellum:program:<name>` graph node, formed when pain / RPE > 0.3, novelty > 0.7 or program
confidence < 0.3. **Nothing in production forms, reads or strengthens one** (#909); no engram decay
exists despite the guide's "~2 days". The forward model **does** train on real readings through
`tool_bridge.py`, but its predictions have no consumer and its state is **never written** —
`build_bio_stack` gives it no `persistence_path`, so `save_cerebellum()` is a no-op at all three
call sites (#908). Resurrection path: roadmap 1.4 Phase 5 "graded predictor" audit.

## 7. How to measure an engram claim (so this page stays honest)

- **Specificity:** compute the real cosine between the two situations through the shipped
  `_sensor_embed` on captured vectors — never the gain-mass table. Replay offline before building.
- **Drift:** isolated (fresh EC per item) vs sequential (one EC, all items); sharp disagreement =
  drift. Required for any new modality, threshold or widening change.
- **Behaviour:** a claim counts only through the real consumer (`recommend_action` / the executed
  action), with a valence ablation that holds the representation identical (Exp 62's arm 3).
- **Caller check:** before marking a row ✅, grep the reading symbol across `src/` + `scripts/`
  excluding tests. Zero non-test callers = capability, not engram.

## Change log

- 2026-09-25 — created from the four-path audit; issues #908–#911 filed; plan
  [engram_formation.md](../plans/engram_formation.md) opened.

# L11 Slice 2 (channel-split) — WIRING lens review

**Verdict: DO-NOT-BUILD as written.** The plan's own generic seam (the `_SUBSTRATE_CHANNELS`
tuple + the modality-agnostic NAc read path) does compose cleanly, but **three production
consumers key on the literal string `"world"` and go SILENTLY DARK the instant the sensors are
re-tagged** — and one of them is the Wire-4 fear WRITE path the whole experiment depends on. This
is the canonical composition-not-pieces failure (D43): the split "works" at the encode site while
the mechanism it exists to enable never fires. Separately, the persisted-substrate migration the
plan names (`maxim substrate invalidate --drop-geometry`) does the WRONG operation for a
modality re-tag, so Exp 56/57 taught-wants are orphaned with no warning and no shipped cleanup.

Reviewer: wiring lens. Method: grepped every `"world"` / `WORLD_TAG` / `modality="world"` /
`.get("world")` site in `src/` and `scripts/`, read each consumer, traced the encode → note →
fear-write → merge → persistence path.

---

## KEY FACT the plan leans on — VERIFIED TRUE (read path only)

The plan asserts the NAc fear path "iterates over all modality tags rather than hard-coding
`world`." **Confirmed for the READ side:**

- `nac.py::anticipatory_threat_need` (L3152) iterates `clusters.values()` and takes
  `min(cluster_fear(...))` across every active cluster — modality-agnostic. After the split it
  still finds fear booked on `world:threat`. ✔
- `nac.py::recommend_action` reward-bias sum (L2121, `for _mod_tag, _mod_cluster in
  active_clusters.items()`) — modality-agnostic. ✔
- `nac.py::note_active_clusters` (L3090) stashes the whole dict verbatim; `fold_legacy_cluster_id`
  (L99) only special-cases `INTEROCEPTION_MODALITY`, leaving world sub-tags untouched. ✔

**But the assertion is FALSE for the WRITE side and for the embodiment backend**, which is where
the split actually breaks (Findings 1–3).

---

## DO-NOT-BUILD

### DNB-1 — Wire-4 fear WRITE hard-codes `.get("world")` → fear never books (silent no-op)
**Site:** `src/maxim/proprioception/pain_bus.py::create_pain_cluster_fear_subscriber` L610:
```python
world_cluster = nac.active_clusters(agent_id).get("world")
if not world_cluster:
    return
```
The code even carries a comment (L607–609) that this literal must track a `WORLD_TAG` rename "or
the fear store silently never writes." **The split is exactly that rename.** After re-tagging,
`note_active_clusters` stashes `{interoception, audio, world:threat, world:env, world:self}` — there
is no `"world"` key — so `.get("world")` returns `None`, the subscriber early-returns, and
`record_cluster_fear` is **never called.** Exp 58's fear contingency (the entire point of Slice 2)
has no write channel.
- **Consequence:** the split isolates the discriminating signal into `world:threat`, then fails to
  book fear onto it. Dry-run would still "fire fear+flee" if fed a hand-set cluster (the D43
  hand-composed-sequence trap), but the live agent writes zero fear.
- **Minimal fix:** rewire to book onto `world:threat` specifically — a DECISION, not a mechanical
  rename (fear must key to the danger channel, not env/self). Because proprioception must not import
  embodiment (layering — see the existing `INTEROCEPTION_MODALITY` duplication at `nac.py:67`),
  introduce a shared `WORLD_THREAT_TAG` / `WORLD_MODALITY_TAGS` constant surface and a
  proprioception-local mirror, with a lockstep test. Ship the guard test in the same commit
  (partial-merge / silent-write class).

### DNB-2 — Minecraft bridge write allow-list keys on `== "world"` → bridge writes nothing
**Site:** `src/maxim/embodiment/backends/minecraft.py::declared_world_sensor_names` L194:
`if schema.get("modality") == "world":`. This derives the set of sensor names the bridge is
permitted to write ("the YAML stays the single source of truth for what the bridge may write").
After re-tagging, `== "world"` matches **none** of the 16 sensors → the function returns an empty
tuple → **the Minecraft bridge can no longer write ANY world sensor.** The survival world goes
dark: `nearest_hostile_dist`, `light_level`, `health`, etc. stop updating.
- **Consequence:** total loss of world sensing; the agent perceives a frozen world. No error — the
  factory just gets an empty name set.
- **Minimal fix:** match membership by predicate (`modality.startswith("world")` or membership in a
  `WORLD_MODALITY_TAGS` frozenset in `sensory_streams.py`), not `== "world"`. Same predicate reused
  at DNB-3 and F5–F8.

### DNB-3 — Interoceptive-relief credit gate keys on `== "world"` → food/health credit stops
**Site:** `src/maxim/embodiment/tool_bridge.py::_is_world_channel_sensor` L526:
`return _schema.get("modality") == "world"`. This gate selects the world-channel sensors that also
carry a live self-effect drive (food/health) for **local interoceptive-relief measurement** — the
R2 relief-credit path. Its own comment records that this modality check was a prior cross-confirmed
"DO-NOT-SHIP" fix, i.e. it is load-bearing. After the split, `world:self` sensors no longer equal
`"world"` → `_measured_intero_sensors` is empty → **relief credit for eating/healing is never
measured.**
- **Consequence:** silently breaks the eat→relief credit loop (the 1.3 break-2 mechanism) for any
  split body. Invisible: no exception, credit just stops.
- **Minimal fix:** same membership predicate as DNB-2 (must include `world:self`).

### DNB-4 — Persisted substrate: the named migration does the WRONG operation; Exp 56/57 taught-wants are orphaned
**Plan Q6 is imprecise and the migration path it cites is inapplicable.** Re-tagging changes the
**modality** key, not the **geometry** tag (geometry = encoder-space signature, orthogonal).
- EC scans strictly within-modality: `ec.py::_matrix_for` (L855) filters `stored_mod == modality`.
  Old persisted nodes stamped `modality="world"` are never scanned once the live encoder emits
  `world:threat`/`world:env`/`world:self`. They become **orphaned/dormant** — and **no
  geometry-mismatch warning fires**, because `ec.py::_note_geometry_mismatch` (L821) triggers only
  on *same modality, different geometry*. The operator gets zero signal.
- The migration the plan names — `maxim substrate invalidate --drop-geometry`
  (`hivemind/cli.py` → `merge.py::invalidate_stale_geometry_nodes` L1028) — gates on
  `node.modality == modality AND node.geometry == drop_geometry`. It **cannot drop-by-modality**,
  and it **deliberately refuses to touch `geometry: None` nodes** (L1046–1049). So it does not, and
  cannot, clean the orphaned `world` nodes left by a re-tag.
- EC stores centroids, not raw readings (`merge.py` L1038–1041): "migration for sensor substrate is
  *invalidation*" — nodes cannot be re-keyed in place. Therefore Exp 56/57's world-keyed taught-wants
  **cannot be migrated**; the NAc `cluster_reward_bias` entries keyed to the old world cluster UUIDs
  become dangling (D43 silent-zero shape) and would be dropped by `prune_nac_cluster_biases`.
- **Consequence:** the plan frames Q6 as "preserves separations → re-measure." In reality the split
  **destroys** the persisted taught-wants; the new sub-channel encodings mint fresh cluster ids with
  zero bias. "Re-baseline Exp 56/57" is a full **re-teach from scratch**, not a re-measurement of
  preserved state — and there is no shipped, tested migration that preserves them.
- **Minimal fix (pick one, explicitly, before build):** (a) accept and document that taught-wants
  are invalidated and must be re-earned under the split, and re-run Exp 56/57 end-to-end as the C
  gate (not a "preservation check"); OR (b) build + test a real drop-by-modality invalidation
  (extend `invalidate_stale_geometry_nodes` to a modality-only mode, paired with
  `prune_nac_cluster_biases`) so the old `world` nodes and their biases are cleanly tombstoned
  rather than silently orphaned. Either way, correct the Q6 wording — the `--drop-geometry` command
  is not the tool for this.

---

## SHOULD-FIX

### SF-1 — Live-vs-harness gap: the geometry probe still encodes `modality="world"`
**Site:** `scripts/survival_world/l11_geometry_probe.py` L233 (`"world" in cfg.gain_modalities`),
L294 (`enc.encode_sensors(..., modality="world", ...)`). Q7/step 6 makes the live re-encode through
this probe the **sole build authorization.** If the probe keeps encoding the monolithic `"world"`
tag while the live agent loop runs the split channels, the gate measures a **different substrate
than the agent runs** — a vacuous authorization. The probe must re-group the captured vectors per
sub-channel (`world:threat` etc.) and read gain/frozen membership for the sub-tags, or the Slice-2
gate proves nothing about the shipped agent.
- **Fix:** rewire the probe to iterate the same `_SUBSTRATE_CHANNELS` the loop uses (or the sub-tag
  set), not a hard-coded `"world"`. Assert the probe's channel set == the agent's.

### SF-2 — `frozen_centroid_modalities` must gain the sub-tags or channels DRIFT
**Site:** `ec.py::ECConfig.frozen_centroid_modalities` L428 `= frozenset({"interoception","audio","world"})`.
(Plan Q4 flags this.) A sub-tag absent from this set gets running-mean centroids at A4's ~120×
allocation rate → the exact drift failure the frozen policy exists to prevent — a NEW failure the
split would introduce. Add all three sub-tags. Mirror in `hivemind/merge.py::DEFAULT_FROZEN_CENTROID_MODALITIES`
L609 (SF-4) so ingested nodes keep the same policy.

### SF-3 — `gain_modalities` policy must be decided per sub-tag (silent loss otherwise)
**Site:** `encoder.py::SensorEncoderConfig.gain_modalities` L684 `= frozenset({"world"})`, applied at
L810. (Plan Q2 flags this.) After the split, no sub-tag is in the set → sub-channels are encoded
**un-gained**, silently. That may even be desirable for the small-N `world:threat` channel (the plan
itself questions whether A4 belongs at N≈5), but it must be a stated decision with the membership
updated accordingly — not a silent side effect of the rename.

### SF-4 — Hivemind merge: frozen-centroid set + per-modality threshold miss the sub-tags
**Sites:** `hivemind/merge.py` L609 `DEFAULT_FROZEN_CENTROID_MODALITIES` and L641 `_thresholds =
{... "world": 0.85}`. On ingest, sub-channel nodes (a) lose frozen-centroid semantics (SF-2's
sibling on the wire boundary) and (b) fall back to the default `cosine_threshold` instead of the
world-tuned 0.85. Add the sub-tags to both. (Exp 56 taught-want transfer runs through this merge;
this interacts with DNB-4's re-baseline.)

### SF-5 — Body-YAML re-tag requires updating `DECLARABLE_MODALITY_TAGS` (fails LOUD, but blocks)
**Sites:** `sensory_streams.py::DECLARABLE_MODALITY_TAGS` L68; validated at `spec.py` L612. The 16
`modality: world` declarations in `_data/components/bodies/minecraft_player.yaml` become
`world:threat`/`world:env`/`world:self`; unless those tags are added to the frozenset, body parse
raises `ValueError` loudly. This is safe (loud, not silent) but the plan's build-order step 1 does
not name it — a builder will hit the raise. Add the three tags in the same commit as the re-tag.

### SF-6 — Harness consumers of `.get("world")` must be rewired (Exp 58 preflight aborts otherwise)
**Sites:** `scripts/survival_world/exp58_run.py` L280/284/288/323/411/452/456;
`scripts/survival_world/exp58_offline_gates.py` L145–200. All read `.get("world")`. After the split
these return `None`; e.g. `exp58_run.py` L288 `if not dark_cluster_pre or dark_cluster_pre ==
safe_cluster_pre` would treat every situation as unclustered and abort the preflight. (Plan Q5
names exp58_run — extend to exp58_offline_gates.) Rewire to `world:threat`.

### SF-7 — `_read_world_states/_ranges` + `_SUBSTRATE_CHANNELS` need N entries that compose
**Sites:** `agent_loop.py` L1106–1116, L1280–1284. The readers are already parametrized by tag
(`_read_declared_modality_states(executor, TAG)`), so the split is N thin readers (or one closure
factory) + N `ModalityChannel` entries. **This DOES compose** — `_encode_current_clusters` (L1300)
and `propose_via_substrate` (L1466/1482) iterate the tuple generically, and the additive
reward-bias sum already scales per channel. Note the existing selection-dynamics caveat (L1272–1279):
adding channels changes the ±N cluster-term range, so re-run `scripts/selection_dynamics_rebaseline.py`
and re-check `min_confidence` — going from 3 to 5 channels is a bigger swing than the world channel's
original +1.

---

## NIT

- **N-1 — Telemetry undercount.** `simulation/minecraft_harness.py` L399/406 count nodes with
  `mod == "world"`; after the split these read 0. Harness-only; fix to a `startswith("world")` sum.
- **N-2 — Verified NON-issues (different `"world"` namespace, no change needed):**
  `imagination/trigger.py:51` ("world" is a stopword in a text list) and
  `reactions/types.py:119` (`WORLD_AGENT_ID = "world"`, an agent-id namespace). Recorded so a future
  grep does not re-flag them.
- **N-3 — LLM-primary fear-write scope (pre-existing, not caused by the split).** The
  `note_active_clusters` stash is only written in `propose_via_substrate` (substrate-primary);
  the LLM-primary path (`agent_loop.py` L3511) sets `pending_proposal.clusters` but never notes them,
  so `pain_bus`'s `active_clusters(agent_id)` read is empty there. Exp 58 runs substrate-primary so it
  is in scope, but the DNB-1 fix should be conscious of which mode populates the stash.

---

## The shared fix that resolves DNB-1/2/3 + SF-2/3/4/5 at once

Every silent break is a `== "world"` / `.get("world")` / single-tag membership test that a
per-type split invalidates. The honest abstraction is a **`WORLD_MODALITY_TAGS` frozenset** (the
sub-tags) plus a `WORLD_THREAT_TAG` for the one site that must pick a specific sub-channel (the fear
write). Define it in `sensory_streams.py` alongside `DECLARABLE_MODALITY_TAGS`; mirror it locally in
proprioception (layering); and replace every literal check with membership. Ship each rewire with a
caller-grep proof (per D43: grep the sub-tags across `src/` + `scripts/` excluding tests and confirm
every world consumer resolves) and a guard test in the same commit. Until that inventory is wired
and the DNB-4 persistence decision is made explicit, the split must not be built.

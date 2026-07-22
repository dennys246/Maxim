# The exteroception/interoception (multi-modality) seam

**Status:** Design (2026-07-22), synthesized from a 3-lens parallel review (substrate-encoding / bio-fidelity / abstraction). Fixes the root cause of the embodied cradle orient failure (memory `reference_extero_intero_dilution_root_cause.md`): exteroceptive direction dilutes among interoceptive drives in one text-embedding cluster → the agent is blind to direction.

## The pivotal finding (all three lenses agree)

**The substrate is already N-modality-ready — do NOT build a modality system.** Ride existing infra:
- `similarity/ec.py::EntorhinalCortex.pattern_complete_or_separate` scans **within-modality only** (`if stored_mod != modality: continue`, ec.py:378) — an `"audio"` node and an `"interoception"` node NEVER compete for pattern completion. Modality is already a first-class namespace (a string tag per node); save/load/merge already carry it.
- `ECConfig.frozen_centroid_modalities` already contains `"audio"` (ec.py:216).
- `similarity/encoder.py::SensorEncoder.encode_sensors` already takes a `modality=` param with a per-`(agent_id, modality)` delta stash (encoder.py:558, 607) — one encoder encodes N modalities into N separate cluster spaces today.

**The entire bug is one caller.** `runtime/agent_loop.py::propose_via_substrate:1005` merges `encoded = {**drives, **extero}` and calls `encode_sensors(sensors=encoded)` with **no `modality=`** → defaults to `"interoception"`, so azimuth is one term in a 384-dim sum dominated by ~6 drives (`_sensor_embed`, encoder.py:445; single-sensor swing leaves cos≈0.83 vs `pattern_threshold=0.85`) → left/right pattern-complete onto the same node → one `cluster_id` → blind. `_read_exteroceptive_states` (shipped #410) is the stopgap this replaces.

## Bio-fidelity framing (labeled lines + late convergence)

Brains do NOT hash all senses into one vector: per-modality thalamic relays (LGN/MGN/VPL) → within-modality primary cortex maps (V1/A1/S1, auditory-space map) → convergence only at association cortex for binding/action. Interoception is a separate axis (insula + hypothalamus/drives) that *modulates* gain but is represented apart. Sound localization: superior olive/inferior colliculus → auditory-space map → superior colliculus orienting reflex (fast, bypasses cortex) while cortico-collicular/BG loops learn the *value* of orienting.

Maps onto Maxim: **hypothalamus = drives/SCN → `current_drives`; thalamus = per-modality exteroceptive relays → per-modality EC clusters; NAc = association-for-action.** The intero/extero seam is ALREADY honored at the *read* layer (`_read_drive_states` vs `_read_exteroceptive_states`, kept separate because "perception is not a need", agent_loop.py:1002) — the only violation is re-merging before the encode. Connects to [thalamus/hypothalamus framing](../../.claude/.../project_thalamus_hypothalamus_framing.md)'s "Decision-4: de-bundle exteroception from interoception".

## Layer boundaries (separation of concerns)

- **(a) Perception/encoding** — `propose_via_substrate` + `SensorEncoder`. Sensors → a *set* of per-modality cluster ids `{modality: cluster_id}`. Knows nothing about tools/reward/binding.
- **(b) Learning surfaces** — `NAc._cluster_reward_bias` keyed `(agent_id, cluster_id, tool)` (unchanged — cluster ids are already UUID-unique per modality). One entry PER modality cluster; **credit is ROUTED by the reward's source**: drive-relief → interoception cluster; operant/direction → the exteroceptive cluster; generic tool-success → interoception ONLY (never writes an exteroceptive cluster — the write-side complement of de-dilution).
- **(c) Integration/selection** — `NAc.recommend_action` sums `cluster_reward_bias` ADDITIVELY across the active cluster set. The orient policy (learned on the audio cluster) and the drive-affinity heuristic (on interoceptive state) coexist as different additive terms — no arbitration. The additive sum is deliberately *binding-free*.
- **(d) Cross-modal binding (voice↔face)** — a RELATION between clusters (same external object → reward on one generalizes to the other). **DEFERRED** — the additive sum stands in; MVP does not bind.

## MVP (fixes the dilution, structurally N-modality)

1. **Split the encode (layer a).** Replace the `{**drives, **extero}` merge with a declarative channel list — adding vision/touch later is one tuple entry, not code:
   ```python
   @dataclass(frozen=True)
   class ModalityChannel:  # embodiment/sensory_streams.py (new)
       tag: str; read_values: Callable; read_ranges: Callable
   _SUBSTRATE_CHANNELS = (
       ModalityChannel("interoception", _read_drive_states, _read_drive_ranges),
       ModalityChannel("audio", _read_exteroceptive_states, _read_exteroceptive_ranges),
   )
   ```
   Loop: `clusters[ch.tag] = encode_sensors(sensors=vals, modality=ch.tag, ranges=...)` per non-empty channel; WARN if a channel has sensors but yields no cluster. `current_drives` stays interoception-only.
2. **Multi-cluster selection (layer c).** `recommend_action(current_clusters: ModalityClusters|None=None, current_cluster_id: str|None=None)` — fold the legacy scalar into `{"interoception": X}`; sum `cluster_reward_bias` over `clusters.items()` (generalize nac.py:1792). Everything else (causal, reward_bias, drive-affinity, gates) unchanged.
3. **Credit routing (layer b).** `LLMProposal.clusters` (keep `cluster_id`=interoception alias); `record_outcome(clusters=...)` routes: drive-relief→interoception, operant→exteroceptive cluster, generic→interoception. `set_pending_operant_action`/`credit_operant_reward` keep single-`cluster_id` sigs — the CALLER now passes the audio cluster.
4. **Push the invariant into a type.** `ModalityClusters = Mapping[str,str]` + `require_valid_modality_clusters()` loud guard at the NAc boundary (empty tag/id = `ValueError`, not silent dilution — the CLAUDE.md silent-no-op rule). Per-modality `consulted_bias` in telemetry so a run can SEE audio present-or-missing.

## Backward-compat (zero migration)

- EC/encoder already multi-modality → interoception encoding byte-identical; `"audio"` already frozen. No persistence bump (`_cluster_reward_bias` key unchanged).
- Single-cluster / LLM-primary agents pass no clusters → empty sum → byte-identical to today. Exp 37/38 (LLM-AUT, never on this path) unaffected. reachy / Exp 42 bodies declare no exteroceptive channel → one interoception call → identical.
- `current_cluster_id` / `LLMProposal.cluster_id` kept as deprecated aliases through 1.x.

## Deferred (front-gate scope pressure — do NOT build now)

- **Cross-modal binding (voice↔face)** — enters when an experiment earns it; home is JEPA / `grounded_language_acquisition` / `cross_modal_substrate_binding`.
- **`modality` in the reward key** `(agent, modality, cluster, tool)` — pure observability; UUID cluster ids already separate. Costs a migration. Defer.
- **Cross-modality attention weighting** ("trust audio more now") — rides the gating layer, not a new resolver. MVP is a flat sum.
- **No `Thalamus` ABC / no percept manifest / no new package** — the modality STRING tag is the extensibility seam (the N=1-inheritance + bio-over-engineering traps).

## Honest caveat (bio-fidelity lens)

Azimuth is sometimes represented TWICE — a signed EC audio cluster ("where", thalamic) AND a sign-folded centeredness *drive* with `pain_scale` ("discomfort", hypothalamic). Bio-plausible (a stimulus can be both localized and aversive), but when splitting the relay, keep them as two representations of two DIFFERENT things (location vs discomfort), not two encodings of one value — else it's a double-count wearing bio-language. (`bodies/infant_operant` gives azimuth NO drive, so this only bites bodies that do.)

## Regression guards (when built)

- Unit: multi-drive body with an `audio` channel → left(-0.7)/right(+0.7) get DISTINCT clusters (the exact dilution assertion, now passing). Single-channel body → one interoception cluster, byte-identical.
- Unit: `recommend_action` sums bias across two clusters; `require_valid_modality_clusters` raises on empty tag/id.
- Integration: the scripted orient probe (probe 4-shape) on a *multi-drive* body now learns (was chance) — the credit-routing + split-encode end to end.
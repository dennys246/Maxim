# Paired-data re-audit — 2026-09-21 (redesigned source)

Step (iii) of the disposition recorded in
[paired_data_audit_2026-09-20.md](paired_data_audit_2026-09-20.md). There the owner chose to redesign
the data source, **recorded as a post-null change of source, not a pass**. This record is still a
measurement that commits to nothing: no gate was pre-registered and none is applied. It is not a
revival of either plan.

**Source:** [language_trace_1204_2026-09-21.jsonl](data/language_trace_1204_2026-09-21.jsonl)
- Paper 1.20.4, bridge started with `--system_messages` (#807).
- Captured by `scripts/l11_real_trace_remeasure.py capture` at `c7d557fb`, clean tree.
- 10 minutes, bare client (no agent).
- Every death was **staged by the operator** from the server console: water fill, `tp ~ ~40 ~`, summoned zombies. The lava and plain-`kill` causes in the run sheet do not appear in the trace.

**Script:** [scripts/paired_data_audit.py](../../scripts/paired_data_audit.py), unchanged from the first audit.
**Re-run on `main` @ `bb6265d0`:** it scanned 188 files and paired exactly two traces, L11 and this one.

## Numbers (the script)

| | L11 (2026-09-04) | This trace |
|---|---|---|
| Snapshots / events | 1,193 / 204 | 984 / 53 |
| Sensor keys | 16 | 19 (`is_in_water`, `offset_x`, `offset_z` new) |
| Distinct texts / templates / word types | 80 / 3 / 11 | 35 / 4 / 19 |
| World EC nodes, whole trace | 3 | **12** |
| World nodes hosting events / hosting ≥ 2 kinds | 3 / 3 | 8 / 5 |
| Known answer: damage number = health (paired-or-previous) | 137 / 142 | 38 / 39 |
| Raw vector at the event names its kind (LOO), vs majority | 113 vs 142 | 26 vs 39 |
| `system` events | none (channel absent) | **5**: `maxim fell from a high place` ×2, `maxim was slain by Zombie` ×2, `maxim drowned` ×1 |

**New channel, confirmed live:** the game's own death messages name the cause. The bridge's
`the player died` still arrives alongside them: 5 of each.

## What the script cannot see: the message describes the PRECEDING second

At the snapshot paired with a `system` event, the body has often already respawned (health 20,
oxygen 20, off the ground). That is why the at-event separability stays below baseline. Two
snapshots earlier (≈ 1 s, at 0.5 s cadence), the raw state names the cause:

| message | state at t−2 |
|---|---|
| drowned | `is_in_water` 1, oxygen −1, health 2.5 |
| slain by Zombie | `nearest_hostile_dist` ≈ 2, `hostile_count` 2, health 2.3–4.3 |
| fell from a high place | `y_altitude` 106.3, `speed` 0.91, `on_ground` 0 |

Cosine between the raw gained vectors at t−2:
- **same cause:** fell–fell 0.979, slain–slain 0.993;
- **different causes:** 0.14–0.71, with drowned–slain the closest at 0.71 (both low-health states).

**This analysis is EXPLORATORY and cannot be cited as evidence.**
1. **The t−2 window was chosen after seeing the data.** A pairing window has to be fixed before the data it is applied to.
2. **The deaths were staged with identical commands.** Both falls were the same teleport, so within-cause similarity is inflated by identical staging, not natural variation. The first fall also began in the leftover drowning water.
3. **n = 5, with drowning once.**

It is not in the script, on purpose: adding a look-back knob tuned on this trace would build the
forking path into the instrument. Reproduce it with `audit_trace`'s helpers (`_load`, `_world_nodes`)
by reading `vecs[i - 2]` for each `system` event.

## What it says

- **The redesigned source answers its narrow question, provisionally.** The game's own words, unlike the bridge's templates, name situations the world sensors can tell apart, *if* each message is bound to the moment before it arrived.
- **They are still templates.** Minecraft's death messages are a finite set of translation keys. Grounding them grounds labels that name causes, which is better than causeless labels but is not language. The "labels, not language" risk the audit exists to measure is reduced, not retired.
- **The binding is retrospective, and the substrate has no live mechanism for that.**
  - `memory/percept_trace_buffer.py::PerceptTraceBuffer` is built for exactly this: a τ-decaying ring buffer of recent activations. It is accepted as a parameter by `campaign_runner`, `fixture_orchestrator` and the reaction types, but **no production code constructs one**.
  - The SCN is the wrong clock here: it bins by time of day, and credit keyed by the SCN binds by clock phase, not recency.
  - This is the same gap as R4's delayed credit (`roadmap_1_4.md`): attach a signal to what was active recently. The language line should consume whatever R4 wires, not build its own.

## Next (owed; nothing starts without a prereg + four-lens review)

1. **Measure the lag between the situation and its message** on a separate capture with **natural, unstaged** deaths (an agent or a scripted wanderer, not console commands). The distribution fixes the window, or `PerceptTraceBuffer`'s τ does; not this trace.
2. **Route the retrospective-trace question into R4's design review**, so text binding becomes a second consumer of one mechanism.
3. Then, if the owner chooses: a prereg for the projection (T1–T4 held-out test sets plus the nearest-template baseline, per the JEPA plan discussion), with the window and n fixed in advance.

The one-hit lag in `damage` texts (first audit, finding 3) reproduces here: 16/39 match at the paired
snapshot and 38/39 paired-or-previous. It remains a bridge property worth fixing or documenting.

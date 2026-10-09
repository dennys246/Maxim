# GL1 name-vs-consequence collision census

> **An offline measurement, not a behavioural claim.** No ledger row, no gate, no prereg: the record is a
> `diagnosis` (it informs the grounding line's design and never counts as support). Plan:
> [docs/plans/grounding.md](../plans/grounding.md) (GL1 row); state page:
> [docs/wiring/body-and-word-worlds.md](../wiring/body-and-word-worlds.md) §5.

Script: [`scripts/grounding_census.py`](../../scripts/grounding_census.py). Record: `docs/experiments/data/census_name_consequence/census.json`. Encoder `sentence-transformers/paraphrase-mpnet-base-v2` (revision `6cc9279c672dc57f94445ef259b28a1b736fec8f`, cpu), EC text threshold **0.44** (`ECConfig()`), commit `e26ccc8ff3b7`, tree dirty: **False**.

## Headline

- **Every primary-walk harm flip is a SAME-name variant pair**: 10 of 10 (`touch`↔`touch`, `warm_self`↔`warm_self`, cosine 1.000): one name, one node, harmful and safe consequences. **0** cross-name harm flips in the primary walk; in the other orders: reverse_sorted 2 (`feel ↔ touch` ×2); shuffle_seed_1120 2 (`feel ↔ touch` ×2) (order-sensitive). The **9** cross-name collisions are all orient (azimuth) pairs (`turn_left ↔ turn_right`, `turn_left ↔ turn_right_big`, `turn_left_big ↔ turn_right`, `turn_left_big ↔ turn_right_big`): opposite signs, both harmful. **26 of 41** "harmful" instances are azimuth turns (`turn_left`, `turn_left_big`, `turn_right`, `turn_right_big`), classed harmful because one turn exceeds the orienting drive's comfort band. A measurement of the word prior, nothing more.
- **529** affordance instances on 97 shipped components (`extends` honoured), **62** with a declared effect, **270** distinct names (11 with an effect; 21 distinct consequence variants). Counted in each file's own `affordances:` blocks without inheritance (the state page §5 walk): {'affordances': 405, 'with_declared_effect': 42}.
- Harm classes per instance: {'no_effect': 467, 'harmful': 41, 'safe': 21}; harmful instances by sensor: {'self:arms.thermal': 15, 'self:azimuth': 26}.
- Sorted walk: **107** text nodes (93 holding a compound); names whose instances split across nodes: ['goto_sleep[]', 'turn_left[self:azimuth=+0.3]'].
- **19** same-node collisions between consequence variants: **10** harm-class flips (harmful vs safe), **9** with opposite signs on a shared sensor; 9 are between DIFFERENT names.
- **3** same-consequence pairs (same sign pattern and harm class, different names) landing on DIFFERENT nodes.
- **1715** shared-word links (compounds on different nodes sharing a word node). Components: 207 absorbed into their own compound's node, 113 onto another compound's node, 16 on a node of their own.
- Known answer (`touch` on blanket vs fire pit): **PASS** (node ['N61']).

## Order dependence (the text centroid is a running mean)

| Order | Nodes | Collisions | Harm flips | Opposite sign | Co-noded name pairs |
|---|---|---|---|---|---|
| sorted | 107 | 19 | 10 | 9 | 828 |
| reverse_sorted | 124 | 23 | 12 | 13 | 469 |
| shuffle_seed_1120 | 114 | 23 | 12 | 13 | 644 |

Collisions in every order: **19**; in any order: **23** (4 order-sensitive).

- reverse_sorted vs shuffle_seed_1120: collision Jaccard 1.0, co-noded-name Jaccard 0.2942
- reverse_sorted vs sorted: collision Jaccard 0.8261, co-noded-name Jaccard 0.2317
- shuffle_seed_1120 vs sorted: collision Jaccard 0.8261, co-noded-name Jaccard 0.2464

## Top collisions (sorted walk; harm flips first, then cross-name, then opposite-sign magnitude; first 19 of 19)

| Variant A | Variant B | Node (cosine) | Why | Instances A / B |
|---|---|---|---|---|
| `touch[self:arms.pressure=+0.4]` | `touch[self:arms.thermal=+0.6,self:cold=-0.3]` | N61 (cos 1.000) | harm safe vs harmful | items/cradle_sharp_rock:surface / items/green_flame_b:glow, items/green_hearth_b:glow, items/purple_flame:glow … |
| `touch[self:arms.pressure=+0.4]` | `touch[self:arms.thermal=+0.6,self:core_temperature=+0.15]` | N61 (cos 1.000) | harm safe vs harmful | items/cradle_sharp_rock:surface / items/cradle_false_hearth:flame, items/cradle_fire_pit:flame |
| `touch[self:arms.thermal=+0.05,self:cold=-0.3]` | `touch[self:arms.thermal=+0.6,self:cold=-0.3]` | N61 (cos 1.000) | harm safe vs harmful | items/green_flame:glow, items/green_hearth:glow, items/purple_flame_b:glow … / items/green_flame_b:glow, items/green_hearth_b:glow, items/purple_flame:glow … |
| `touch[self:arms.thermal=+0.05,self:cold=-0.3]` | `touch[self:arms.thermal=+0.6,self:core_temperature=+0.15]` | N61 (cos 1.000) | harm safe vs harmful | items/green_flame:glow, items/green_hearth:glow, items/purple_flame_b:glow … / items/cradle_false_hearth:flame, items/cradle_fire_pit:flame |
| `touch[self:arms.thermal=+0.1]` | `touch[self:arms.thermal=+0.6,self:cold=-0.3]` | N61 (cos 1.000) | harm safe vs harmful | items/cradle_blanket:fabric / items/green_flame_b:glow, items/green_hearth_b:glow, items/purple_flame:glow … |
| `touch[self:arms.thermal=+0.1]` | `touch[self:arms.thermal=+0.6,self:core_temperature=+0.15]` | N61 (cos 1.000) | harm safe vs harmful | items/cradle_blanket:fabric / items/cradle_false_hearth:flame, items/cradle_fire_pit:flame |
| `warm_self[self:arms.thermal=+0.05,self:cold=-0.3]` | `warm_self[self:arms.thermal=+0.6,self:cold=-0.3]` | N20 (cos 1.000) | harm safe vs harmful | items/green_flame:glow, items/green_hearth:glow, items/purple_flame_b:glow … / items/green_flame_b:glow, items/green_hearth_b:glow, items/purple_flame:glow … |
| `warm_self[self:arms.thermal=+0.05,self:cold=-0.3]` | `warm_self[self:arms.thermal=+0.6,self:core_temperature=+0.15]` | N20 (cos 1.000) | harm safe vs harmful | items/green_flame:glow, items/green_hearth:glow, items/purple_flame_b:glow … / items/cradle_false_hearth:flame |
| `warm_self[self:arms.thermal=+0.2,self:core_temperature=+0.2]` | `warm_self[self:arms.thermal=+0.6,self:cold=-0.3]` | N20 (cos 1.000) | harm safe vs harmful | items/cradle_fire_pit:flame / items/green_flame_b:glow, items/green_hearth_b:glow, items/purple_flame:glow … |
| `warm_self[self:arms.thermal=+0.2,self:core_temperature=+0.2]` | `warm_self[self:arms.thermal=+0.6,self:core_temperature=+0.15]` | N20 (cos 1.000) | harm safe vs harmful | items/cradle_fire_pit:flame / items/cradle_false_hearth:flame |
| `turn_left_big[self:azimuth=+0.5,self:head_yaw=+0.9]` | `turn_right_big[self:azimuth=-0.5,self:head_yaw=-0.9]` | N104 (cos 0.902) | opposite self:azimuth, self:head_yaw | bodies/reachy_mini:orient, bodies/reachy_mini_infant:orient, bodies/reachy_mini_infant_satiated:orient / bodies/reachy_mini:orient, bodies/reachy_mini_infant:orient, bodies/reachy_mini_infant_satiated:orient |
| `turn_left[self:azimuth=+0.17,self:head_yaw=+0.3]` | `turn_right_big[self:azimuth=-0.5,self:head_yaw=-0.9]` | N104 (cos 0.751) | opposite self:azimuth, self:head_yaw | bodies/reachy_mini:orient, bodies/reachy_mini_infant:orient, bodies/reachy_mini_infant_satiated:orient / bodies/reachy_mini:orient, bodies/reachy_mini_infant:orient, bodies/reachy_mini_infant_satiated:orient |
| `turn_left_big[self:azimuth=+0.5,self:head_yaw=+0.9]` | `turn_right[self:azimuth=-0.17,self:head_yaw=-0.3]` | N104 (cos 0.730) | opposite self:azimuth, self:head_yaw | bodies/reachy_mini:orient, bodies/reachy_mini_infant:orient, bodies/reachy_mini_infant_satiated:orient / bodies/reachy_mini:orient, bodies/reachy_mini_infant:orient, bodies/reachy_mini_infant_satiated:orient |
| `turn_left[self:azimuth=+0.17,self:head_yaw=+0.3]` | `turn_right[self:azimuth=-0.17,self:head_yaw=-0.3]` | N104 (cos 0.874) | opposite self:azimuth, self:head_yaw | bodies/reachy_mini:orient, bodies/reachy_mini_infant:orient, bodies/reachy_mini_infant_satiated:orient / bodies/reachy_mini:orient, bodies/reachy_mini_infant:orient, bodies/reachy_mini_infant_satiated:orient |
| `turn_left[self:azimuth=+0.3]` | `turn_right_big[self:azimuth=-0.5,self:head_yaw=-0.9]` | N104 (cos 0.751) | opposite self:azimuth | bodies/base_humanoid:orient, bodies/infant_humanoid:orient, bodies/infant_humanoid_chilled:orient … / bodies/reachy_mini:orient, bodies/reachy_mini_infant:orient, bodies/reachy_mini_infant_satiated:orient |
| `turn_left_big[self:azimuth=+0.5,self:head_yaw=+0.9]` | `turn_right[self:azimuth=-0.3]` | N104 (cos 0.730) | opposite self:azimuth | bodies/reachy_mini:orient, bodies/reachy_mini_infant:orient, bodies/reachy_mini_infant_satiated:orient / bodies/base_humanoid:orient, bodies/infant_humanoid:orient, bodies/infant_humanoid_chilled:orient … |
| `turn_left[self:azimuth=+0.3]` | `turn_right[self:azimuth=-0.3]` | N104 (cos 0.874) | opposite self:azimuth | bodies/base_humanoid:orient, bodies/infant_humanoid:orient, bodies/infant_humanoid_chilled:orient … / bodies/base_humanoid:orient, bodies/infant_humanoid:orient, bodies/infant_humanoid_chilled:orient … |
| `turn_left[self:azimuth=+0.17,self:head_yaw=+0.3]` | `turn_right[self:azimuth=-0.3]` | N104 (cos 0.874) | opposite self:azimuth | bodies/reachy_mini:orient, bodies/reachy_mini_infant:orient, bodies/reachy_mini_infant_satiated:orient / bodies/base_humanoid:orient, bodies/infant_humanoid:orient, bodies/infant_humanoid_chilled:orient … |
| `turn_left[self:azimuth=+0.3]` | `turn_right[self:azimuth=-0.17,self:head_yaw=-0.3]` | N104 (cos 0.874) | opposite self:azimuth | bodies/base_humanoid:orient, bodies/infant_humanoid:orient, bodies/infant_humanoid_chilled:orient … / bodies/reachy_mini:orient, bodies/reachy_mini_infant:orient, bodies/reachy_mini_infant_satiated:orient |

## Same consequence, different node (cross-name)

- `touch[self:arms.thermal=+0.05,self:cold=-0.3]` (['N61']) vs `warm_self[self:arms.thermal=+0.05,self:cold=-0.3]` (['N20']), harm safe, cos 0.337, identical deltas
- `touch[self:arms.thermal=+0.6,self:cold=-0.3]` (['N61']) vs `warm_self[self:arms.thermal=+0.6,self:cold=-0.3]` (['N20']), harm harmful, cos 0.337, identical deltas
- `touch[self:arms.thermal=+0.6,self:core_temperature=+0.15]` (['N61']) vs `warm_self[self:arms.thermal=+0.6,self:core_temperature=+0.15]` (['N20']), harm harmful, cos 0.337, identical deltas

## Shared-word links between effect-bearing names (10 of 1715)

- `turn_left` ~ `turn_left_big` via ['left'] on N104
- `turn_left` ~ `turn_right` via ['left', 'right'] on N104
- `turn_left` ~ `turn_right_big` via ['left', 'right'] on N104
- `turn_left` ~ `turn_left_big` via ['turn'] on N5
- `turn_left` ~ `turn_right` via ['turn'] on N5
- `turn_left` ~ `turn_right_big` via ['turn'] on N5
- `turn_left` ~ `warm_self` via ['self', 'turn'] on N5
- `turn_left_big` ~ `warm_self` via ['self', 'turn'] on N5
- `turn_right` ~ `warm_self` via ['self', 'turn'] on N5
- `turn_right_big` ~ `warm_self` via ['self', 'turn'] on N5

Widest word nodes (compounds linked through one word node):

- N5 (first text `administer`): 55 compounds via ['anchor', 'and', 'apply', 'approach', 'around', 'ask', 'at', 'attempt']
- N4 (first text `activate shields`): 16 compounds via ['activate', 'area', 'block', 'cover', 'hull', 'shield', 'shields', 'stall']
- N12 (first text `appraise`): 14 compounds via ['audit', 'carefully', 'check', 'examine', 'listen', 'look', 'observe', 'scan']
- N3 (first text `acid spit`): 8 compounds via ['acid', 'bite', 'claw', 'eat', 'pick', 'poison', 'poisoned', 'spit']
- N8 (first text `ambush`): 8 compounds via ['attack', 'blunt', 'shock', 'strike']

## Order-sensitive collisions (present in some walk orders, not all)

- `feel[self:arms.thermal=-0.15,self:core_temperature=-0.2] || touch[self:arms.thermal=+0.05,self:cold=-0.3]`
- `feel[self:arms.thermal=-0.15,self:core_temperature=-0.2] || touch[self:arms.thermal=+0.1]`
- `feel[self:arms.thermal=-0.15,self:core_temperature=-0.2] || touch[self:arms.thermal=+0.6,self:cold=-0.3]`
- `feel[self:arms.thermal=-0.15,self:core_temperature=-0.2] || touch[self:arms.thermal=+0.6,self:core_temperature=+0.15]`

## State-page cosines, re-measured

| Pair | Cosine |
|---|---|
| fire breath ↔ flame jet | 0.6011 |
| fire breath ↔ water jet | 0.2983 |
| flame jet ↔ water jet | 0.6637 |
| fire ↔ flame | 0.7852 |
| touch blanket ↔ touch fire pit | 0.4686 |
| turn left ↔ turn right | 0.8737 |
| escape water ↔ flee | 0.5683 |

## Method notes and caveats

- Harm class is a single application from rest against DECLARED drives (homeostatic: rest = `set_point`, harmful when `|delta| > comfort_band`; entropic: rest = the sensor's declared `rest:` on that body, else its declared `initial` state, harmful when the move is in the drift direction and `rest + delta` reaches `deprivation_threshold`; an up-drift sensor declaring neither rests at 0, a down-drift one is unclassified), on the owning entity if it drives that sensor, else on every shipped body that does. Undriven sensors (e.g. `hp`) are unclassified, so harm flips are a LOWER bound on consequence disagreement. Orienting drives (`azimuth`) count as drives.
- Compound pairs co-noded below the threshold (absorbed by running-mean drift): 571; pairs at or above it on different nodes: 158.
- The walk encodes every instance (inheritance duplicates included, as production re-encodes a name per entity), into ONE EC holding every shipped component's names: a population-level prior, not any single scenario's EC.
- `archetypes/*.yaml` are vocabulary templates, not entities, and are excluded.
- The census walks every entity's `children` as well as its top-level modulators; production encodes top-level modulators only. Affordances on child entities in this inventory: **0**, so the two agree here (a future child affordance would enter the census before production).
- Measured on `cpu`; production runs the encoder on `mps:0`. Node assignment near the threshold can differ by device.
- The data directory token is `census_name_consequence` (not `grounding_*`) so a future `grounding_*` prereg cannot govern this record by its token. A future prereg whose token is `census` WOULD: GL3.B0 must name its prereg and data directory so it does not.
- An instrument refusal never replaces this record: a refused run writes `census.failed.json` / `census.failed.md` beside it instead.

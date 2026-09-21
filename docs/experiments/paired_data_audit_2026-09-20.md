# Paired-data audit — 2026-09-20

The entry condition of the grounded-language + cross-modal-projection parallel line
([roadmap_1_4.md](../plans/roadmap_1_4.md) §Parallel lines;
[grounded_language_acquisition.md](../plans/grounded_language_acquisition.md) banner). A measurement
that commits to nothing: **no pass/fail gate was pre-registered, so none is applied here.** The
disposition (revive / archive / redesign the data source) is the owner's.

**Script:** [scripts/paired_data_audit.py](../../scripts/paired_data_audit.py) — read-only over every
committed `docs/experiments/data/**/*.jsonl`; the sensor side replays through the real
`SensorEncoder` + `EntorhinalCortex` (world modality, declared ranges, A4 gain 3.0).
**Run:** `python scripts/paired_data_audit.py`.

## Numbers

| | |
|---|---|
| JSONL files scanned | 187 |
| Files with both world snapshots and text events | **1** — `l11_world_trace_2026-09-04.jsonl` |
| That trace | 1,193 snapshots (0.5 s cadence, ≈ 10 min), 204 events, 0 unpaired |
| Distinct text strings | 80 |
| Distinct templates | **3**: `took damage (health <N>)` ×142, `a <mob> appeared nearby` ×52, `the player died` ×10 |
| Vocabulary, whole corpus | **11 word types** |
| Template fillers | a number (damage); `zombie` / `slime` (spawn); none (death) |
| World EC nodes, whole trace | **3** (identity check: the L11 verdict records `A4.clusters: 3` for this trace) |
| World nodes hosting events / hosting ≥ 2 event kinds | 3 / **3** |
| Raw-vector kind separability (LOO nearest-centroid) | **113 / 204**, vs 142 / 204 for always guessing `damage` |
| Known answer: damage text's number = paired `health` | 81 / 142 at the paired snapshot; **137 / 142** paired-or-previous |

The Exp 56 / 60 / 61 / 62 and R3 campaign records keep no text percepts (R3's `event` rows are
lethal-trial records, not game text). The one paired trace predates the Paper 1.20.4 port (#765) and
the `is_in_water` sensor.

## What the numbers say

1. **There is almost no paired data.** One ten-minute trace from one world on one day. "Per
   situation" cannot be answered beyond 3 EC nodes.
2. **Every text is a label.** 77 of the 80 distinct strings are one template differing only in a
   number. The vocabulary is 11 words. This is the risk the audit exists to measure ("a template is
   a label — grounding on labels is not grounding language"), and the answer is: all of it.
3. **The most frequent text restates a sensor.** The `damage` number *is* `health` — the known-answer
   check confirms the pairing (137/142) — and it lags one hit: mineflayer fires `entityHurt` before
   the health update lands, so 56 of the 142 carry the pre-hit value. A projection trained on it would
   learn a delayed copy of a dimension the sensor vector already has.
4. **The sensor side at a text does not pick out a situation.** All three event kinds land in all
   three world nodes, and the raw vector at the event names its kind *worse* than the majority
   guess. Caveat: this is the single snapshot at the event, not the before→after transition, and
   `damage` is by nature a transition.
5. **The bridge discards the game's own language.** `scripts/minecraft_bridge/index.js` subscribes
   to mineflayer's `chat` (player chat) only. Minecraft's system messages — death messages that name
   a cause ("drowned", "was slain by Zombie", "fell from a high place"), advancements, server
   messages — arrive as `systemChat` → `message`/`messagestr` and are dropped. The bridge replaces
   them with the causeless `the player died`. These are still a finite set of templates, but they are
   game-native (D1-compatible) and carry the situation, which the current strings do not.

## Reading (the author's, not a gate)

Taken literally against the plan's own wording, the survival world's text channel today is labels,
and too few of them: the entry condition is not met. But what was measured is the **bridge**, not the
idea — the corpus is whatever six `event(...)` call sites emit. Two honest dispositions:

- **Archive** both plans per the entry condition as written, with this record as the measurement.
- **Redesign the data source first, and say so.** Changing the source after a null is moving the
  goalposts unless it is recorded as exactly that. Candidates: forward the game's system messages
  (finding 5); a narrator/teacher channel of free text; and — for the JEPA web angle — web text stays
  a held-out TEST set only, since it carries no sensor side and cannot supply training pairs. Any of
  these is a new data design and goes through the four-lens review before a harness is built.

Not a disposition either way: the one-hit lag in `damage` texts (finding 3) is a property of the
bridge worth fixing or documenting whatever happens to this line.

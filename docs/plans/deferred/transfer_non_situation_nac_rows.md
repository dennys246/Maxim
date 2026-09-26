# Transfer a donor's non-situation NAc rows — deferred on a trigger

> **DEFERRED 2026-09-25 (owner, on the release-format-v2 PR A review).** Chosen instead: "own rows
> only" — a signed release ships the exporter's own agent's rows, and ingest drops non-situation rows
> filed under another agent. This plan is the road not taken, kept so it is taken deliberately.

## What is not transferred today

Situation rows (`cluster_fear`, `cluster_reward_bias`, `cluster_reward_source`) re-key onto the
receiver's own situations at ingest (`merge.rekey_nac_state`) — that is what Exp 56 and Exp 61
measured and earned. Three other agent-keyed NAc fields carry no situation and so cannot re-key:

| Field | Key | What it would mean at a receiver |
|---|---|---|
| `percept_valences` | `agent␟entity␟failure mode` | the donor's "zombie → health is bad" becomes the receiver's own valence |
| `event_outcome_welford` | `agent␟tool` | the donor's outcome statistics blend into the receiver's (a donor n=900 swamps an own n=5) |
| `reward_bias` | `agent:node` | legacy node-keyed bias; would also need the EC `id_map` |

NAc reads filter on the reader's agent id, so these rows were always inert for the receiving agent: first under
the donor's id, and since release format v2 dropped at ingest (`IngestReport.foreign_rows_dropped`).

## Why not now

Re-keying them to the receiver makes them live — a real expansion of what sharing transfers that no
experiment has measured. It would move behaviour Exp 56/61 earned under a narrower transfer, so it
triggers their re-run checks, and it needs its own design (the Welford blend needs a discount like
`FOREIGN_FEAR_DISCOUNT`; valences need the same tighten-only posture fear has).

## Trigger

Revive when a design NEEDS a transferred percept valence or outcome statistic — e.g. social
referencing (`docs/plans/social_referencing.md`) choosing to learn "that entity is dangerous" from a
Queen rather than a situation. Revival starts with the four-lens design review
(`docs/experiments/DESIGN_REVIEW.md`), then the re-key in `rekey_nac_state` (one seam, never at call
sites), then the Exp 56/61 re-run triggers.

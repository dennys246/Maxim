# Where damage-fear lands: the full negative-learning topology (CORRECTED)

**Established:** 2026-09-13, dark=danger offline wiring probe
(`scripts/survival_world/dark_danger_probe.py`, 1.3 Step 2), seven iterations.

> **CORRECTION (same day, iterations 2–7):** this entry's first version claimed "Minecraft
> damage is painless — the body declares no failure modes, so pain never fires." That root
> cause was WRONG twice over, manufactured by two instrument artifacts: (1) the probe's
> damage magnitude (6.0) sat exactly ON the health drive's `comfort_band` (6.0) — drive
> pain fires on deviation *beyond* the band, so the breach never happened; (2)
> `pain_bus.recent` shows a LOSSY 1-key context view (its own comment says so), which made
> Wire 2 look starved when subscribers actually receive the rich 8-key context. Drives DO
> publish pain (no declared failure modes needed — `_publish_drive_pain` is the drive-spec
> branch of `evaluate_failures`). What follows is the corrected, measured topology.

## The measured topology (all through the real path, offline)

A health drop of 12 (20→8, past the band) while the dark world-cluster is active:

1. **Drive pain publishes** — intensity 1.0 (`min(1, (|Δ|−band)·pain_scale)`), latched:
   fires on band ENTRY and re-fires only when the breach deepens; the latch clears only
   when an evaluation OBSERVES recovery (a harness that never ticks `evaluate_failures`
   while healthy sees one publish total, and one that damages by exactly the band width
   sees zero).
2. **Action-blame is correctly suppressed** — tool_bridge's B8 delta filter (Exp 42)
   passes a drive-breach failure into the tool's `side_effects["embodiment_failures"]`
   (→ direct negative attribution) only if the acting affordance's own effect is
   intrinsically harmful to that sensor. A zombie bite during `eat` does not blame `eat`.
   This is by design and load-bearing — dark=danger must not ride action-blame. Worse for
   naive designs: the bystander action books its normal outcome, so repeated
   damage-during-action taught a POSITIVE causal link (eat `causal_pos` 0.78).
3. **The pending-event similarity path misses by design** for tool-invoked pain
   (`{"params"}` vs rich context — documented in `_on_embodiment_pain`), and the
   in-window bridge path defers to (2).
4. **Wire 2 (Pavlovian percept aversion) FIRES**: `record_percept_valence` books −1.0
   under `(agent_id, entity_class='minecraft_player', failure_mode='drive:health')`,
   read by the salience scorer. This is the one fear store that received the damage.
5. **Nothing situation-keyed receives it**: `cluster_reward_bias` accepts negative
   values but has no pain-side caller (action-scoped `record_outcome` only), and Wire 2's
   key is the SUFFERING entity's class + drive — for world drive-pain that is the agent's
   own body, identical in the dark and in daylight. **Situation-blind.**

## The consequence for dark=danger (and any learned situation-aversion)

"The agent learns dark = danger" requires fear keyed to the *situation* (the dark world
cluster), and **no such write channel exists**: measured end state after 8 dark+damage
episodes is world-cluster bias 0.0 (dark and lit), zero negative causal links, and a
situation-blind Wire-2 aversion. The Phase-1 prereg cannot claim situation-fear on
current wiring. This is a DESIGN item for the four-lens review, with two candidate
shapes: extend the Wire-2 key (or a sibling store) with the co-active world cluster, or
give pain a `update_cluster_reward`-style negative write keyed to co-active clusters.
Either is a new mechanism entering `[engineering]` per the two-tier rule.

## Instrument lessons (each cost one probe iteration)

- **Compare damage against the drive's `comfort_band`** — a breach of exactly the band
  width is in-band. Read the spec from the body YAML, don't pick round numbers.
- **`pain_bus.recent` is a lossy view** — `context_keys` from it say nothing about what
  subscribers received. Verify a subscriber by its *output store*, not by eyeballing
  `.recent`.
- **Tick `evaluate_failures` in the healthy state too** when staging repeated breaches —
  the publish latch clears on observed recovery (hysteresis), not on the value recovering.
- Check `pain_bus.get_stats().total_published`, never `subscriber_count` — wired
  listeners prove nothing about signal flow.

## See also

[substrate-learning-channels.md](substrate-learning-channels.md) (the action-scoped
credit channels); `docs/plans/deferred/transition_based_drive_pain.md` (the latch's
design history); Exp 42 (why B8 exists); `src/maxim/proprioception/pain_bus.py`
(Wire 2 subscriber + the lossy `.recent` comment).

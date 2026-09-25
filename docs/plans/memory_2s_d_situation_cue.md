# Memory strength 2S-d — the situation cue (design)

> **DECIDED 2026-09-25 (owner) — design v2, BUILT.** v1 had one design review (bio-fidelity +
> wiring: no DO-NOT-BUILD, six SHOULD-FIX); the owner changed two decisions on it (recall only, a
> minimum match) and the rest were folded — §Review record. Phase 2S-d of
> [memory_strength_and_forgetting.md](memory_strength_and_forgetting.md) ("substrate-native pattern
> completion"). Builds on 2S-b (#877: a survival capture records its `situation` and is linked to the
> ATL concepts of those clusters) and feeds 2S-e (generalization, own plan + four-lens review).

## What it does, and what it honestly is

When the agent's situation changes, the memories formed in that situation are recalled: the current
`{modality: cluster}` ids cue the ATL concepts with those ids, their linked memories are ranked by
shared place, then how salient each was, and the best matches are returned — **recalled, not activated**
(v2, decision 5).

Until 2S-e there is **no behavioural consumer**: the survival action path has no LLM to read a
prediction. So 2S-d ships a wired, working recall with **no effect on memory or behaviour**: the ids
are returned for 2S-e, and the only trace is the cue's own counters
(`PatternCompleter.situation_cue_stats`: cues, changes, changes that found memories, ids found),
reported in every `MemoryHub` session-end result as `situation_cue_*` (and so in its INFO line;
cumulative for the hub's life, so repeated sessions add up) —
which is what lets a run SHOW that the cue fired and what it found.

## Front-gate: rides existing infrastructure

No new store, bus or index. `PatternCompleter` already does concepts → linked memories →
`activate(source="prediction")`, but it cues only by TEXT (percept objects, goal tokens) and runs only
from `MemoryAgent` on the LLM path — the survival audit measured 0 completions. 2S-d adds a
**situation route** to the same class. No EC lookup: the cluster ids are known on the tick, so there
is nothing to complete in embedding space (that is 2S-e's step, through `pattern_complete_readonly`).

## The rule (the six decisions)

1. **Trigger — a situation change.** Fires when any modality's cluster differs from the previous
   cue for that agent (or a modality appears/disappears). The previous situation lives on the
   hub-side cue (`PatternCompleter`), per agent, in memory only, and is cleared at every
   `MemoryHub.on_session_start` — so the first tick after `water_trial.reopen_hub_session` is an
   entry. Boundary flicker (a cluster id oscillating near the completion threshold) re-cues each
   flip; with recall only, that costs lookups and inflates `changes`, nothing else. Two more
   sources of re-cues, named: **interoception drift** in an unchanged place is a change (so
   `changes` is dominated by drive drift, and this is closer to an internal-state cue than a pure
   context change), and a tick where a channel's encode failed reads as an exit and re-entry. Both
   are harmless while recall only; 2S-e decides whether its consumer wants change detection on the
   qualifying modalities alone. A recall that RAISES rolls the last situation back, so the next tick
   in the same situation retries rather than staying silent.
2. **Where — inside `propose_via_substrate`, a REQUIRED keyword-only `situation_cue=`.** Clusters
   are computed there and already handed to the NAc (`note_active_clusters`); the cue fires right
   after. Required because the survival harnesses call `propose_via_substrate` directly, bypassing
   the loop: a hook in the loop would silently never fire in Exp 60–62-style runs. Every call site
   passes it — `MemoryHub.situation_cue`, or the explicit sentinel `NO_SITUATION_CUE` (the Exp 53
   readout, which has no Hippocampus). `None` is refused (`TypeError`), and the hub accessor
   RAISES when it has no completer (its ATL failed to build), so a degraded hub cannot hand over a cue
   that silently recalls nothing. The loop resolves it once (`_resolve_situation_cue`) and, on a
   degraded hub, runs on with the opt-out and a WARNING — fail-soft like the rest of the loop —
   while the harnesses resolve it once per phase and STOP (they read the accessor directly). The call is
   fail-soft (`log_swallowed_exception`, Stage-1): a failing cue never costs the tick. A static test
   pins that every production call passes it (the harnesses are not run by the suite).
3. **Selection — judged on the memory's OWN situation, by place, then sound, then salience (v3,
   owner decision (b) on the code review).** Candidates come from the cue concepts'
   `memory_refs['hippocampus']` (the refs only nominate); each is judged on its own recorded
   `situation`. It **qualifies only through a shared world or audio cluster**, and its **tier** is the
   HIGHEST-ranked modality in `SITUATION_RANK_ORDER` it shares: every same-place memory is in the
   place tier whatever its sound (a drowning recorded with a different sound must not drop below safe
   swims that share today's splash — the re-review's case), and the sound tier is used only when no
   place matches. **Interoception plays
   no part**, neither qualifying nor ranking: its cluster is broad (cosine separates only a
   neutral→extreme swing), and as a ranker it put safe past swims (full air, matching the cue) above
   the drownings (extreme cluster) at second 0 of a dive — exactly when the drownings are the ones to
   recall (the Exp 60 failure; the v2 rule did this). **Within the tier, the most salient memories come
   first** — `max(encoding_tag, retro_tag)`, the same measure the strength floor reads, so the
   retro-tagged moments just before a drowning count as salient — then those also sharing a
   lower-ranked modality (the sound), then the newest by experience time, capped at `MAX_EPISODES`
   (20) — so a long run of uneventful visits cannot crowd
   out the one that hurt. Preferential recall of salient memories is the bio-plausible default; 2S-e
   may override it. In a new place, the cue returns nothing **unless a sound matches**: on a body with
   audio, a same-sound memory from elsewhere still qualifies, ranked below any same-place memory.
   Records with no situation never match. Every read is non-touching (`recall_by_ids` on both
   stores), and the refs are copied before scoring.
4. **Lost links — deferred with a trigger.** The refs are a lossy index (a compressed concept drops
   them; `MAX_REFS_PER_LAYER` evicts). The 2S-b docstring requires a fallback to, or rebuild from,
   the records' own `situation`; that is deferred
   ([deferred/situation_cue_fallback.md](deferred/situation_cue_fallback.md)) rather than built as a
   full scan or a new index now.
5. **Recall only — nothing is activated (v2, owner).** `MemoryLayer.activate` is for a CONSUMPTION
   point, and nothing consumes these ids until 2S-e. v1 activated with `source="prediction"`, which
   under `strength` credits `S` and resets the anchor: an agent re-entering water every few seconds
   would have kept its 20 latest drowning memories from ever fading, safe visits included — retrieval
   strengthening with no prediction error or confirmation to gate it. The credit moves to 2S-e, where
   the outcome is known and can gate it. So retention is byte-identical under every strategy.
6. **Returned, not consumed.** The cue returns the recalled record ids for 2S-e. The recall itself is
   also exposed WITHOUT change detection, `PatternCompleter.recall_situation(cue)`: 2S-e's chosen
   (B) completes a cue to a NEIGHBOURING situation first, and recalling with that through
   `cue_situation` would overwrite the agent's last situation.

## Scope, stated

- **Fires:** the substrate-primary loop and the harnesses' direct `propose_via_substrate` calls.
- **Finds nothing (today):** the harnesses' propose-only phases (`water_trial` training, Exp 58's
  propose-only arms, the offline gates) capture no memories, and 2S-b's links form only from loop
  captures — so on the fresh substrates of Exp 60–62 the cue fires and recalls nothing. The
  session-end `situation_cue_*` counts show it; no run should be quoted as exercising recall.
- **Never fires:** llm-primary and real-hardware passes, which take their situation from
  `_attach_live_situation`, not `propose_via_substrate` (deferred file, trigger).

## When the recall gets a consumer (trigger, 2026-09-25)

2S-d ships a recall nobody reads. Its planned consumer, 2S-e (B) generalization, was PARKED on
2026-09-25: the gap it was to fill (Exp 62's "night miss" at 0.799) turned out to be the
`time_of_day` wrap, a keying defect ([#899](https://github.com/dennys246/Maxim/issues/899)), and the
fear place gives no SUPPORT for a graded read
([deferred/generalization_by_pattern_completion.md](deferred/generalization_by_pattern_completion.md)).
So the recall waits for a consumer on a stated trigger, not on "when we get to it".

**Build a consumer (revive 2S-e) when ANY of these fires:**
1. **A measured generalization gap keying does not own.** The same danger is missed across two
   states that genuinely differ in the world channel but sit within reach of a graded read: cosine in
   roughly [0.75, 0.85) to a FEARED node, measured **at the place the fear was learned**, on a trace
   that varies more than the clock (place, weather or mobs; e.g. an open-world trace with
   `doDaylightCycle` on). A gap that is a clock wrap, or one below ~0.75 (like the lit pond's 0.588),
   belongs to keying, not here. Measure it offline first when the varying state is deterministic
   (corollary 3 of `docs/wiring/cosine-separation-is-directional.md`).
2. **A rung needs carry-over the NAc does not store:** which action worked in a situation, relief, or
   an outcome other than fear. That is the episodic route (recall through `recall_situation`, derive
   from the recalled memories' outcomes), which the NAc's fear store cannot serve.
3. **A world captures memories BEFORE the decision being scored.** 2S-b's links form only from loop
   captures, so the cue recalls something only where loop-live experience precedes the test read.
   That is a prerequisite for any consumer, and a trigger to check the session-end `situation_cue_*`
   counts on the first such run (`with_matches` > 0 says the recall has something to hand over).

**Otherwise, dormancy (CLAUDE.md, dormancy over deletion):** if none has fired by the **1.4 release
transaction**, mark the situation route `Dormant since <date>: no consumer — 2S-e parked, trigger
unfired` in `PatternCompleter`'s docstring. The wiring stays (the required `situation_cue=` seam and
its callers); no feature builds on it and its tests stay regression-only, until a trigger above
revives it. The 1.4 release checklist should carry this check.

## Guards

All in `tests/unit/test_memory_2s_d_situation_cue.py`, each proven by deleting its mechanism:
a change recalls and the same situation does not; the place outranks the sound; interoception never
ranks (the dive-start drowning is recalled first); the one that hurt survives the cap; interoception
alone never qualifies; a stray ref is re-checked against the memory's own situation; no activation,
no `S`/anchor change; no concept or memory is touched; a session reset makes the next cue an entry;
the hub refuses to hand over a missing cue and resets at session start; `None` and a missing kwarg
raise; the tick hands its clusters to the cue and survives a failing one; every production call
passes it (static); and the real hub composes capture → ConceptExtractor link → cue.

## Review record

v1 (2026-09-25), one reviewer, bio-fidelity + wiring: no DO-NOT-BUILD. SHOULD-FIX: crediting an
unconsumed cue (→ recall only, owner); no minimum match (→ world/audio qualifies, owner); a `None`
opt-out a degraded hub produces by accident (→ sentinel + raising accessor); where the previous
situation lives (→ hub-side, reset per session); the harness wiring overstated (→ §Scope + stats);
llm-primary/hardware not covered (→ §Scope + deferred trigger). NITs folded: non-touching reads,
experience-time ranking, ref snapshot, and more ways links are lost (deferred file).

Code review round (2026-09-25, executor + architecture/bio lenses): no DO-NOT-MERGE. Both lenses
found the stats had no reader (→ reported at session end). Executor: harnesses re-read the accessor
every tick (→ once per phase); a failed recall marked the situation cued (→ rollback); the static
scan missed attribute calls and could scan an installed shadow (→ both fixed). Architecture: 2S-e's
(B) needs a stateless recall (→ `recall_situation`); 2S-e's plan text now records what it inherits;
the CHANGELOG leads with "no effect"; and the v2 ranking hid the memories that matter (interoception
split the tier; newest-first let safe visits crowd out a drowning) → owner chose (b): tier by place
then sound, salience then recency within it. A third reader, on the fold delta: the place+sound
split still let a same-place, different-sound drowning drop below safe swims (→ the tier is the
highest-ranked SHARED modality; the sound only orders inside it); salience now includes `retro_tag`;
the counters are cumulative (stated). The run_agentic_loop ratchet: the cue resolution and the
existing sensor-encoder construction moved to module-level helpers, tightening its ceiling.

## Where this is referenced

[memory_strength_and_forgetting.md](memory_strength_and_forgetting.md) §Phase 2S ·
[deferred/situation_cue_fallback.md](deferred/situation_cue_fallback.md).

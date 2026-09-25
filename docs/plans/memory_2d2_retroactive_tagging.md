# Memory strength 2d-2 — retroactive tagging (design v2)

> **DECIDED 2026-09-25 (owner) — design v2; BUILT 2026-09-25** (two build findings folded below,
> §Build notes). The second slice of memory-strength
> Phase 2's look-back ([memory_strength_and_forgetting.md](memory_strength_and_forgetting.md) decision
> 2; R4's review put the look-back over the Hippocampus record, [lookback_primitive.md](lookback_primitive.md)).
> Builds on **2d-1** (PR #893: every trace records `encoded_at_us` and `capture_seq`). v1 was reviewed
> (bio-fidelity + wiring, one DO-NOT-BUILD, seven SHOULD-FIX); the owner accepted v2 below and asked
> for its limits to be recorded as deferred work with triggers:
> [deferred/retro_tagging_extensions.md](deferred/retro_tagging_extensions.md). Opt-in: read only
> under `memory.strategy=strength`.

## What it does, and what it honestly is

When a moment is strong, related memories encoded in the seconds before it become harder to forget —
the drowning makes the moments already spent in the water memorable (which moments count as
"related" is narrower than it sounds — below). This is a **backward-only
eligibility window, read at consolidation** — the memory plan's framing. It is *not* a faithful model
of behavioural tagging (Moncada & Viola 2007), which is hour-scale, two-sided and dopamine-gated; the
two share only the idea that a strong event can reach back and protect weaker, related ones
(selectivity: Dunsmoor et al. 2015).

**What "related" buys, stated plainly (review round, 2026-09-25).** Relatedness is a shared world or
audio cluster, and a cluster over a summed sensor channel separates only a neutral→extreme swing
([cosine-separation-is-directional](../wiring/cosine-separation-is-directional.md)) — the same argument
that excludes interoception. So what the rule protects is **earlier moments in the same situation as
the strong one** (the seconds already in the water before the drowning bites), *not* the approach
that led into it: dry-land lead-in sits in a different world cluster and is untagged (the tests pin
exactly that). The converse also holds: in a broad cluster, every trace in the window gets `rel = 1`.
Whether real survival traces split or share a cluster across the lead-in is unmeasured — deferred
item 2 carries the trigger to measure `rel` on a real Exp 60/62 drowning before anyone cites this as
lead-in protection.

## The rule

1. **Which events tag.** A trace whose own `encoding_tag` is **strictly greater than**
   `RETRO_TAG_THRESHOLD = 0.5` — a named constant, not a config key (nothing has measured it; the
   2c-3 rule), for Phase 5 to calibrate. Strictly, because a brand-new causal link's first outcome
   carries a surprise of exactly 0.5 and would otherwise trigger on nearly every first action.
   Whatever makes a moment strong — nociceptive pain, relief, `|RPE|`, novelty, drive pressure — is
   what the tag already scores, relief included. The event's tag magnitude sets what it spreads.
2. **What a tag changes.** A new `retro_tag` on the earlier trace (float in [0, 1], `None` until
   tagged), beside the stamped `encoding_tag`, which is never rewritten (2c-3(b)). The protection
   floor reads `max(encoding_tag, retro_tag)` — a one-line change in
   `strategies.py::StrengthStrategy._retrievability_and_floor`. `S` and the anchor are untouched: a
   retro tag is protection, not a retrieval.
3. **Relatedness — situations, and only situations.** The fraction of the strong event's **world and
   audio** clusters (2S-b's `situation`) that the earlier trace shares. Interoception is excluded: the
   strong moment's interoception cluster is the extreme one, so its real lead-in scores lowest on it
   (cosine sees direction, not magnitude). **No fallback**: a trace or event without a situation gets
   0 — no tag. Traces from the **same action** (one failure writes loop, reflection and pain traces)
   never tag each other — by construction: only the loop capture carries a situation (Build notes).
4. **Window and strength.** Traces strictly before the event by `(encoded_at_us, capture_seq)`, within
   `memory.retro_cutoff_us` (default 30 s): `r_i = tag_e · exp(−Δ/τ) · rel(i, e)`, `τ =
   memory.retro_tau_us` (default 10 s); `retro_tag_i ← max(retro_tag_i, r_i)` — saturating at the
   strongest event, never summing. The keys are in µs, like `s_base` (the plan named them
   `memory.capture.tau_s` / `cutoff_s`; renamed so no seconds↔µs conversion exists on this path).
5. **What is taggable.** Episodic traces with `encoded_at_us` (captured since 2d-1). Pre-2d-1 and
   `CompressedMemory` records are skipped. Forward window: 0.
6. **When — at consolidation.** At the start of `Hippocampus.sleep()` (and of the SCN
   `sleep_with_clustering()` path, which removes traces too — review round), under the write lock sleep
   already takes (not nested inside it): every trace with `encoding_tag > θ` tags its related
   predecessors — re-resolved at every sleep, idempotent because a tag only rises to the max (no
   watermark: Build notes). Whatever capture is in the store by then takes part and nothing needs a
   drain; a capture still queued is resolved at the next sleep. Behavioural tagging's effect is itself
   measured after consolidation. Per-record updates take `_touch_lock` in
   the existing store → record order.

## The reach, stated as a number

With the plan's constants (`w = 0.5`, retention threshold 0.3), a retro tag lifts a trace above the
threshold only when `r ≥ 0.6`: with the strongest event and full relatedness, that is **Δ ≤ ~5.1 s**.
The ~16 s drowning lead-in gets `r ≈ 0.20` → a floor of 0.10, which protects nothing. The mechanism
ships with this reach, stated; calibrating `τ`, `w` and `θ` together is Phase 5's job (deferred-plan
trigger). The retro floor also fades from the tagged trace's own anchor and `S`.

## Stamped always, read only under `strength`

Like `S` and the anchor, `retro_tag` is computed whatever the strategy, and only `StrengthStrategy`
reads it — in its score AND in `should_compress`, which reads the same floor, so a trace held up by a
retro tag is kept whole rather than compressed. The default path's RETENTION is byte-identical; its
saved files do carry the new field. The persisted record gains one optional
field — in `_strength_fields` (the save snapshot taken under `_touch_lock`), the tolerant loader
(`None` when absent, loud when malformed) and the compression carry. Resolution runs under the default
strategy too, at every consolidation: one sort of the timed episodic traces, then a backward scan per
strong event bounded by the cutoff — bounded by the store's size, not by what is new (no watermark).

## Where it acts, and where it does not (see the deferred plan)

- **Acts:** wherever traces carry a situation and a consolidation runs — `sleep()` and the SCN
  `sleep_with_clustering()` path (the console's `rest`) both resolve first. **Wired, not yet read by
  any run:** the survival harnesses reach `sleep()` (`minecraft_harness` → `MemoryHub.on_session_end`),
  but Exp 58/60/61/62 pin `memory_strategy: access_based`, so today they stamp `retro_tag` and nothing
  reads it.
- **Does not act:** traces without a situation (the LLM sims, unless the substrate path fills the
  loop proposal's clusters — unverified); `--sim`'s lightweight session end
  (no `sleep()`); a trace evicted before the next sleep.

## Guards (to build with the code)

- A strong event tags a related predecessor inside the window; a trace that is outside the window,
  unrelated, after the event, from the same action, or the event itself is not tagged — each proven by
  deleting the mechanism.
- `θ` is strict: an event at exactly 0.5 tags nothing.
- Max, not sum: two strong events leave the stronger tag.
- Under `strength`, a retro-tagged trace inside the reach outlives an identical untagged one; under the
  default strategy, retention is byte-identical.
- `retro_tag` round-trips a save and survives compression.
- A capture that lands after a sleep (async: its seq was reserved before) is still resolved at the
  next sleep, in both directions; a strong event with no situation tags nothing.
- The two window keys reach every Hippocampus builder and refuse a non-positive or non-int value at
  the env, writer and config doors.

## Build notes (2026-09-25)

- **Review round folds.** Both lenses found `sleep_with_clustering()` removing traces without
  resolving (fixed; guard `test_the_clustering_sleep_resolves_too`). The same-action rule now has a
  caller guard (`test_only_the_loop_capture_path_stamps_a_situation`: an AST scan that fails when any
  other production site passes a situation). The retention guard now runs the real removal path under
  `strength`. Traces with `encoded_at_us` but no `capture_seq` (only a malformed load) are skipped, so
  the order is total. The claim was narrowed to "same-situation predecessors" (§What it does).

- **No watermark.** v2 planned a persisted "resolved up to" watermark on `capture_seq` as a cost
  saver. It is wrong, not just unneeded: an async capture reserves its seq when QUEUED and is stored
  when the worker gets to it, so a sleep can see a later sync seq first. A watermark would then skip a
  late strong event forever, and a late lead-in would never be tagged by an event already resolved.
  Every strong event is re-resolved at each sleep instead; the scan per event is bounded by the
  cutoff. Guard: `test_a_capture_that_lands_after_a_sleep_is_still_resolved` (red with a watermark).
- **Same action, by construction.** Of the traces one action writes, only the loop capture
  (`agent_loop` → `bio_integration._capture_episodic`) carries a situation; the pain, reflection and
  tool-pain traces do not, so situation-only relatedness never lets them tag each other. No action-id
  key was added. A new site that stamps a situation on a secondary trace must revisit this (said in
  `Hippocampus._resolve_retro_tags_locked`'s docstring). Guard:
  `test_a_strong_event_without_a_situation_tags_nothing`.

## Review record

v1 (2026-09-25), one reviewer, bio-fidelity + wiring: DO-NOT-BUILD — the sync pain trace the lazy
resolution existed for tags nothing (no situation, keyword Jaccard 0, and the causing action is
captured after the pain); SHOULD-FIX — the ~5 s reach, `θ = 0.5` equal to a first outcome's surprise,
keyword overlap over-tagging, interoception scoring the cause lowest, the behavioural-tagging framing,
drain detection that never fires on the worker, and `retro_tag` persistence. All folded into v2 with
the owner's approval (2026-09-25).

## Where this is referenced

[memory_strength_and_forgetting.md](memory_strength_and_forgetting.md) §Phase 2d slicing ·
[lookback_primitive.md](lookback_primitive.md) D2 ·
[deferred/retro_tagging_extensions.md](deferred/retro_tagging_extensions.md).

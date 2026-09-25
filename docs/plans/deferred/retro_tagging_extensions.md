# Retroactive tagging — the limits 2d-2 ships with, deferred on triggers

> **DEFERRED 2026-09-25 (owner).** Memory-strength 2d-2
> ([memory_2d2_retroactive_tagging.md](../memory_2d2_retroactive_tagging.md)) ships as a situation-keyed,
> consolidation-time backward window with the plan's uncalibrated constants. Each limit below is
> stated in that note; this file holds what would lift it and when to revive it.

## 1. Tagging in the LLM sims

**Limit:** relatedness is situation-only, and LLM-path traces carry no situation, so nothing is tagged
there — including in the LLM sim worlds where the memory plan validates Phase 2 first (route A).
Keyword overlap was rejected (goal words make every trace look related), and traces carry no
embedding.
**Would lift it:** LLM-path traces carrying a substrate situation (the text substrate path,
`MAXIM_SUBSTRATE_PATH`, already writes EC text clusters), or a stored embedding with a measured
relatedness threshold.
**Revive when:** LLM-path captures carry a situation, OR Phase 2's LLM-sim validation needs tagging
to measure its effect.

## 2. Reach and constants

**Limit:** with `τ = 10 s`, `w = 0.5` and the 0.3 retention threshold, a retro tag protects only
traces within ~5 s of the strong event; `θ = 0.5` is a named constant nothing has measured.
**Would lift it:** calibrating `τ`, `w`, `θ` (and `retro_cutoff_us`) jointly against measured
cause→outcome lags and retention targets.
**Also unmeasured:** whether relatedness reaches the lead-in at all. It is a shared world/audio
cluster, and a summed channel's clusters separate only a neutral→extreme swing, so the approach on
dry land may sit in a different cluster from the drowning (the plan's "What related buys").
**Revive when:** Phase 5 calibration starts
([decay_consolidation_calibration_plan.md](decay_consolidation_calibration_plan.md)), OR a survival
measurement shows the lead-in to a strong event (e.g. drowning's ~16 s) forgotten while tagging was on,
OR anyone is about to cite 2d-2 as protecting the lead-in — first measure `rel` between a real Exp
60/62 drowning trace and the traces in the ~16 s before it.

## 3. Evicted before the next sleep

**Limit:** tagging resolves at `sleep()`; a trace evicted (store full) before that is never tagged.
**Would lift it:** resolving pending strong events before `_evict_one` (a locked variant inside the
write lock the eviction already holds).
**Revive when:** a measured run's store fills and evicts between sleeps while tagging is on.

## 4. `--sim` never tags

**Limit:** the generic `--sim` loop ends a session with the lightweight path, which skips `sleep()`,
so tagging never resolves there.
**Would lift it:** resolving at the lightweight session end too (it is cheap: one pass over the
session's new captures).
**Revive when:** a `--sim` campaign relies on forgetting (a forgetting measurement or a strength-
strategy validation run on `--sim`).

## 5. A corrupt saved clock

**Limit:** a saved experience clock that fails to load restarts at 0 (inherited from 2d-1) while the
loaded traces keep their large `encoded_at_us`; new captures then sort before old ones and the windows
between old and new traces are meaningless (new traces never tag old ones, and vice versa, since the
gap reads as far past the cutoff — harmless but blind).
**Would lift it:** resuming the clock past the loaded traces' maximum `encoded_at_us`.
**Revive when:** a corrupt-clock load is ever logged on a store running `memory.strategy=strength`.

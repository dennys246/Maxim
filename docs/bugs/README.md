# Known defects & limitations — running ledger

**What this is.** The fourth ledger. The repo already tracks *behavioral* claims
([behavioral_graduation_candidates.md](../plans/behavioral_graduation_candidates.md)),
*algorithmic* claims ([bio_faithful_roadmap.md](../plans/bio_faithful_roadmap.md)), and
*engineering rules* (CLAUDE.md invariants). This tracks **what is verifiably wrong or
bounded right now** — the axis on which findings otherwise evaporate into plan-doc asides.
(A fifth ledger, [docs/limits/README.md](../limits/README.md), tracks measured
*instrument* limits — nothing broken, but properties every experiment design must
respect. Defects go here; characterized measurement limits go there.)

**Why it matters more than a normal bug list.** The 2026-08-10 external critique's sharpest
point was that unstated limitations quietly become false claims. A defect that lives only
in one session's memory is indistinguishable, six months on, from a defect that was fixed.

## Rules (these are what stop this becoming a second CLAUDE.md)

1. **Verified only.** Every row cites `file:line` or a measurement. No suspicions — those
   belong in a plan's open-questions section.
2. **Every row has a disposition**, not just a description: `OPEN` (will fix, trigger
   named), `ACCEPTED` (deliberate limitation + why), or `FIXED` (PR + guard, then it
   leaves this file at the next prune).
3. **Rows expire.** A `FIXED` row is deleted once its guard exists — the guard is the
   durable record. An `OPEN` row with no trigger and no movement for two releases gets
   re-argued or becomes `ACCEPTED`.
4. **Claim linkage is mandatory** when a defect bounds a claim: name the graduation row it
   touches. A defect that silently invalidates an Earned row is the worst case this ledger
   exists to prevent.
5. **Deep investigations stay as their own doc** in this directory (the existing
   convention — see `console_seam_findings.md`). This file indexes them and carries
   standalone defects.

## Investigation clusters

| doc | scope | state |
|---|---|---|
| [console_seam_findings.md](console_seam_findings.md) | `maxim serve` console seams, sim_logger, PFC gate, aarch64 packaging | 12 fixed, 6 open |
| [display_print_corruption.md](display_print_corruption.md) | interactive display / print interleaving | see doc |
| [sim_embodiment_followups.md](sim_embodiment_followups.md) | sim embodiment wiring follow-ups | see doc |
| [repository_review_2026_08_19.md](repository_review_2026_08_19.md) | stable API contracts, lifecycle cleanup, architecture enforcement, offline tests | 6 open (D15–D20) |

## Standalone defects

Seeded 2026-08-11 from a four-lens architecture review; each was verified against the code
during that review and currently lives nowhere else.

### Substrate / persistence

| # | Defect | Disposition | Evidence |
|---|---|---|---|
| D1 | **`encoder_provenance` detects nothing at runtime.** It is recorded, persisted and reloaded, but nothing compares it against live encoder state — its only readers are the hivemind bundle/CLI export, and `record_encoder_provenance` *merges* divergence ("'mixed' is a finding, not an error"). A geometry change loads old-geometry centroids and cosine-scans them against new embeddings: **silently merged.** | **OPEN** — trigger: any encoder/threshold change. Prerequisite of the place-code default-ON gates. | `similarity/ec.py` (record/persist/load), `hivemind/bundle.py`, `hivemind/cli.py` |
| D2 | **There is no way to invalidate a stale EC substrate.** `cli_utils.py::MEMORY_PATHS` has keys for `hippo`, `nac`, `scn`, `atl`, `angular_gyrus` — **no `ec`**, not even under `all`. Clearing `nac` alone violates the NAc/EC pairing invariant. | **OPEN** — blocks any encoding change that requires invalidate-both-in-lockstep. | `cli_utils.py::MEMORY_PATHS` |
| D3 | **`ec_merge`'s cosine threshold is a hardcoded duplicate pinned by no test.** `cosine_threshold: float = 0.44` duplicates the EC default; unlike the frozen-modality set (which *is* pinned by `test_hivemind_frozen_modalities_match_ec_default`), nothing detects drift. The layer deliberately refuses internal imports, so it cannot read a threshold map. | **OPEN** — trigger: per-modality thresholds (plan F-A). | `hivemind/merge.py:557` |
| D4 | **A same-dimension encoding change defeats the merge dimension-guard, invisibly.** `_cosine` returns 0.0 on dim mismatch — but a place code keeps `dim=384` and the same `"audio"` tag, so old- and new-geometry nodes merge whenever partial cosine clears the threshold. Because `audio` is frozen, the corruption is undetectable: counts and contributors inflate, the centroid never moves. | **OPEN** — gate on `MAXIM_PLACE_CODE_EXTEROCEPTION` default-ON. | `hivemind/merge.py` |
| D5 | **`nac_merge` never folds cluster biases across agents.** Keys are `agent\x1fcluster\x1ftool` and cluster ids are per-agent UUIDs, so cross-agent cluster learning is unioned, never merged. | **ACCEPTED for now** — but it makes session-relative spatial bearings unmergeable, i.e. it is the mechanical answer to the RSC plan's "does this need a world anchor?" (yes). | `hivemind/merge.py:506-510` |

### Learning path

| # | Defect | Disposition | Evidence |
|---|---|---|---|
| D6 | **Hebbian episode binding is inert on the main percept path.** `memory_hub` stashes a **1-tuple** of substrate nodes, and `apply_hebbian_on_close` returns early on `len(nodes) < 2` — a silent no-op. So the binding graph never grows from llm-primary percepts, independent of any resolution question. | **OPEN** — trigger: the cross-modal fabric's binding work depends on this path. | `integration/memory_hub.py:348`, `memory/episode.py:795` |
| D7 | **`min_delta = 0.05` is an undocumented second resolution limiter.** `SensorEncoder` short-circuits the EC scan and returns the cached node whenever no sensor moved ≥0.05 — a hard ~4.5% dead zone on a `[-1,1]` sensor **regardless of encoding**. Any population code past ~40 buckets is capped by this gate, not by geometry. | **ACCEPTED** (it is a real efficiency win) — but must be stated wherever resolution is claimed. | `similarity/encoder.py:519, 640-645` |
| D8 | **A read path mutates text centroids.** `bio_enrichment` calls `pattern_complete_or_separate(embedding, "text")` per enrichment query, with an in-code note that the centroid update is intentional reconsolidation (~1/(n+1) shift per query). Consequence not documented anywhere: **querying degrades text-cluster resolution over time.** | **OPEN** — needs measurement, then accept-or-fix. | `bio_enrichment.py:665-667` |
| D9 | **5 of 6 `TemporalEvent` categories have no producer.** Only `tool` emits; the drive emitter is both unwired *and* malformed (raises `TypeError` into a `except Exception: log.debug`). `record_event` even special-cases `deliberation` significance for a producer that does not exist. | **OPEN** — documented in [deferred/scn_event_producer_gap.md](../plans/deferred/scn_event_producer_gap.md); revival requires answering whether per-event-type phase learning earns its keep. | see plan |

### Claim scope

| # | Defect | Disposition | Evidence |
|---|---|---|---|
| D10 | **Exp 45's Earned status does not transfer to an EC-clustered orient policy.** The orient backbone builds state from `az_bin(...)`, a hand-written bin string passed as `current_cluster_id` — it never calls EC. That is hand-curated discretisation *upstream* of the substrate (the interim-contamination pattern), so any future EC-clustered orient policy needs its own experiment. | **OPEN (scope note)** — must be recorded on the Exp 45 graduation row. | `scripts/orient_backbone/live_3_learn.py:680`, `live_common.py:340` |
| D11 | **~432 bare `except Exception: pass` remain repo-wide.** 48 measurement-path sites were instrumented (#487); the rest are grandfathered and un-instrumented, so a silent failure outside that scope is still invisible. | **ACCEPTED** (scoped deliberately) — see [measurement_path_fail_loud.md](../plans/measurement_path_fail_loud.md). | `grep -rEA1 "except Exception:\s*(#.*)?$" src/maxim/` |
| D13 | **A failed planning turn drops silently — the orchestrator idles forever between turns (the TRUE mechanism behind the D12-era wedges).** Full evidence chain (2026-08-18, py-spy dump + `MAXIM_BACKEND_TRACE` of a live wedge): the narrator's last call returned **status 200** (79 output tokens, 2.9s) — then zero further backend calls, ever. The response failed tool-parse ("LLM returned empty tool response" → `dispatch_exhausted`), the planning turn was dropped with NO reschedule, the `_awaiting_llm` window expired at 120s (`agent_loop.py` idle gate: `(now - ctrl.last_llm_submit_time) < 120.0`), and — because the orchestrator has no other percept source between turns — nothing ever wakes it again. Livelock, not deadlock: py-spy shows MainThread ACTIVE cycling the loop at the pacing sleep, all workers idle, no call in flight. Two `LLMCall_0` threads confirm the worker's executor-replacement path also ran. Trigger probability rises with prompt pressure (traced calls at 3,547/4,096 tokens = 87% window occupancy → truncated context → worse small-model output format → more parse failures), which is why `--resume-sim` runs (enrichment-heavy prompts) wedge far more often than fresh runs. Mitigations in place: #517/#519 hard-abort caps the cost at ~4 min with a loud `llm_wedged` report; n_ctx headroom lowers trigger odds. | **FIXED** (2026-08-19, PR #523) — the fix contract implemented: every failed-planning-turn outcome (fallback drop, error proposal, stale drop, worker exception path, silent await-window expiry) routes through `agent_loop::_handle_planning_failure` → bounded requeue of the original `LLMRequest` (`LLMWorker.requeue_request`/`requeue_last_request`) with LOUD per-attempt logs; budget spent (3) → loop breaks through NORMAL teardown and raises `loop_controller.PlanningLivenessExhausted`, which the orchestrator converts to an `llm_wedged` finish report (`initiated_by: "planning_liveness"`). **Scope is opt-in** (`run_agentic_loop(planning_liveness=True)`) and ONLY the sim orchestrator opts in — it is the one loop with no `action_sink` and no percept producer between turns; the AUT already recovers via the `_llm_unavailable` synthetic action breaking the bridge's settle loop, so enabling it there would convert a working recovery into a dead thread (pre-merge review, both lenses). "Nothing came back" is decided from the observed call registry, never the 120s await literal — a big model legitimately exceeds it (qwen2.5-32b ~140s/turn), and that same gate fixed a second livelock flavor where a lapsed window idled past the proposal poll so a slow call's answer was never consumed. Operator opt-out `MAXIM_SIM_PLANNING_LIVENESS=0` (S6 apparatus control). Guards: [tests/unit/test_planning_liveness.py](../../tests/unit/test_planning_liveness.py) (52 tests; 30 of the pre-fold 32 verified-to-fail on pre-fix code — real-worker parse-failure repro, bounded-retry machine, slow-call false-abort guards, wiring pins). Worst-case abort latency is ~4 windows (≈8 min) when requeues are rejected, vs D12's ~4 min. Prune at next sweep per rule 3. Instrument note: `sudo py-spy dump` is the canonical wedge microscope (Python tracebacks with file:line — one screenful where `sample`'s C frames needed three captures). | py-spy dump 2026-08-18 (MainThread active at `agent_loop.py:4584`, await-gate at `:1691`); `/tmp/heartbeat_e10_p2h.jsonl` backend trace (final call `status:200` then silence; W-logs at +108s; abort at +139s) |
| D14 | **The orchestrator spinner asserts work that is not happening.** The "Orchestrator planning next/first probe…" spinner text is set when planning BEGINS (`orchestrator.py` `bridge._spinner.start` sites) and is never updated on any failure path — so during a D13 drop it keeps counting up ("planning… (8624s)") while py-spy proves NOTHING is in flight. This is a display-truthfulness defect with real diagnostic cost: across 2026-08-18/19 it steered both operator and agent toward server/network theories for hours, because the UI asserted a call was running. Same class as the "accepted dispatch is a promise, not a motion" lesson — a status display must report observed state, never intent. | **FIXED** (2026-08-19, PR #523 — same PR as D13) — the stall detector is now the spinner's truth source via the pure `simulation/spinner.py::spinner_truth_message` (no call in flight past threshold → "⚠ no LLM call in flight — planning turn may be lost (Ns, k nudges)"; byte-wedged in-flight call → says so; healthy call reappears → default text restored; hard-abort → "🛑 llm_wedged — aborting sim" before teardown). The between-turns window is owned by the `Spinner` itself (`set_planning_window` / `update_if_planning`), and the correction is a lock-guarded test-and-set so a turn starting mid-decision keeps its own text — a plain flag would let the detector clobber fresh turn text with a "may be lost" warning while the AUT is working, the same display-lies class (pre-merge review, executor lens F7). The loop's retry attempts additionally log "⚠ planning turn failed (…) — rescheduled (attempt N/3)" to the EXEC stream. Guards: [tests/unit/test_planning_liveness.py](../../tests/unit/test_planning_liveness.py) (TestSpinnerTruthMessage + TestSpinnerPlanningWindow + orchestrator wiring pins, incl. the #519 nudge-proof-clock pin repaying that guard-test debt). Prune at next sweep per rule 3. | py-spy dump 2026-08-18 (active idle loop, zero calls in flight) vs. simultaneous pane capture ("planning next probe… (3286s)"); `simulation/orchestrator.py` spinner-start sites |
| D12 | **Orchestrator LLM calls blocked unboundedly — ROOT-CAUSED (2026-08-18, `sample` capture of a live 2.4h wedge): `router._inference_lock` inheritance deadlock.** The llm_worker timeout path abandons an orphan thread that can still be inside the locked region; its executor-replacement fallback gives new calls fresh threads but nothing frees the lock, so the untimed `with self._inference_lock` parked every subsequent call forever (~75 lock-waiter threads, ZERO network activity, stall detector blind because registry entry happens inside the lock — 265 impotent nudges observed). NOT a network-timeout escape; the model server was healthy and idle throughout. Deceptive symptom: AUT ticks sensors normally while starving for percepts ("bio systems not engaging"). | **FIXED** — bounded lock acquire + loud failure (`MAXIM_INFERENCE_LOCK_TIMEOUT_S`, default 600s) + the #517 sim-level hard-abort (exit 4) as defense in depth. Guards: [tests/unit/test_inference_lock_timeout.py](../../tests/unit/test_inference_lock_timeout.py) (held lock → bounded loud failure; verified the old behavior never returned) + [tests/unit/test_stall_hard_abort.py](../../tests/unit/test_stall_hard_abort.py). Prune at next sweep per rule 3. | `sample` thread-stack capture 2026-08-18 (80 threads: lock-waiters + zero recv); `models/language/router.py::_complete_text`; `agents/llm_worker.py` orphan path |

### Surfaced by the D13/D14 review round (2026-08-19) — NOT fixed in PR #523

Deliberately out of that PR's scope: each is pre-existing, none is introduced by it,
and folding them in would have made the reviewed diff differ from the merged diff.

| # | Defect | Disposition | Evidence |
|---|---|---|---|
| D22 | **An `llm_wedged` / aborted sim still exits 0, and no campaign harness reads the status.** D12's abort only reaches exit 4 through its `_force_exit` backstop (i.e. when the thread never unwinds); a sim that unwinds cleanly after ANY loud abort — D12 or the new D13 path — reports the wedge in `finish_context` and exits 0. `grep -rn "llm_wedged" scripts/` returns nothing, so a harness records the run as completed. A campaign can therefore contain aborted rows counted as data. | **OPEN — 1.1 gate candidate.** Needs one decision: map terminal `finish_context.status` values to process exit codes, and have the harnesses treat a wedged row as FAIL rather than a short run. | `simulation/orchestrator.py::_stall_detector` (`_force_exit`, exit 4 only on non-unwind); `grep -rn "llm_wedged" scripts/` → 0 matches |
| D23 | **`Spinner` threads outlive their sims in the test suite and write ANSI frames to stderr unconditionally.** `Spinner._spin` has no `sys.stderr.isatty()` check, and bridges constructed in tests are not always stopped, so the advertised offline fast suite emits `\r`-laden frames — a run of the full suite showed `Orchestrator planning next probe... (588s)`, i.e. a spinner counting for ten minutes inside pytest. Harmless to results, but it pollutes the suite D20 is trying to make hermetic and makes real spinner output untrustworthy in captured logs. | **OPEN — 1.1.x hygiene**, pairs with D20's hermetic-suite work. | fast-suite stderr tail 2026-08-19; `simulation/spinner.py::Spinner._spin` (raw-ANSI branch, no isatty gate) |

## Pending, not yet a defect

- **The S1 annotation renderer will break PR #497's S4 parser** (its regex matches the band
  exactly; adding `— why` inside the bracket makes it report "no annotation" silently).
  Not yet a defect because neither has merged — but they must land together with the
  parser fix, or this becomes D12 on the day they don't.
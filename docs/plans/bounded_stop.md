# The bounded stop: save within the caller's deadline; the owner of the process exits

**Status:** design CLOSED 2026-10-26 (Session H). Thirteen adversarial design passes; the last met the stop rule of
CLAUDE.md's "Faster without lowering the bar" (1). The three-lens plan review (Executor, Architecture, Wire integrity) is
folded in, with owner decisions of 2026-10-26; §10 lists the folds. Issues: [#1438](https://github.com/dennys246/Maxim/issues/1438) (a
Reachy stop can kill the session-end save) and [#1344](https://github.com/dennys246/Maxim/issues/1344) (SIGTERM is
unhandled). It amends [advisor_mode.md](advisor_mode.md) item 9 and serves [embodied_device.md](embodied_device.md)'s
bounded-stop row. Owner decisions are dated inline; the session record is memory `1-3-4-session-h`.

## 1. What was measured, and what it changed

The stop path was measured before anything was built (owner decision 2026-10-25). The setup: an Apple M2, the console's
own agent build (`MaximHandle` → `create_full_agent`), no LLM, and fresh data homes at 50, 500 and 3000 memories.

- **The real work is small.** Saves and the session end take **0.5-1.2 s** at every size. `hippocampus.sleep` takes
  0.4-0.6 s at 3000 memories (its own save included), and the other consolidation steps about 0.01 s.
- **The full path saves the hippocampus twice.** `stop_capture_worker` and the ConceptExtractor shutdown each cost a
  fixed ~0.5 s (a poll interval).
- **Pulse's ~60 s is the console's pre-save WAITS.** `console/server.py::_drain_and_stop_handle` joins a run it cannot
  cancel (30 s), then `MaximHandle.stop` waits for the campaign lock (60 s) and joins Talk (20 s). A hung campaign alone
  is 60 s.
- **Caveats.** A CM4 is several times slower, and the hippocampus save grows with the store. The bench saw no
  consolidation that deleted memories. Pulse is asked for CM4 numbers and the daemon's stop signal, kill delay and start
  window.

**So (owner decision 2026-10-25): the fix is waits that CANCEL, plus save-first, with consolidation inside the budget.**
A deferral marker ("consolidation owed") is built only if a real device store overruns.

## 2. Why the design is narrowed: the library never exits the process

Passes 1-3 kept finding new failure classes from two sources: the library owning process exit (`os._exit`, a watchdog,
skipped atexit, `os._exit` in an embedder's process), and keeping consolidation alive under a budget. **Owner decision
2026-10-25: the library saves within the deadline it is given and RETURNS. Only the CLI (a process Maxim owns) handles
signals and exits.** Passes 4-13 converged on one missing layer, **session ownership**, which became PR 0.

## 3. The build, in order

0. **Session A's verbatim extraction** (owner decision 2026-10-26). `run_agentic_loop` becomes a thin wrapper plus
   `_run_agentic_loop_body`, recorded as `split_from`, with zero body edits. PR 0 opens only after it is ON `origin/main`
   (never stacked on A's branch), so rule T charges PR 0's edits inside the body.
1. **PR 0, session ownership** (§4). It also lands `utils/deadline.py::Deadline` and the stop request with its accessors
   `request_stop()` / `current_stop()` / `yield_requested()` (§5 describes them; PR 0 needs them for its bounded abnormal
   exit).
2. **PR 1 (#1438): the library** (§5).
3. **PR 2 (#1344): the CLI only** (§6).
4. **PR 3: spawned servers** (§7).

PR 0 ships in the same release as PR 1, which bounds its waits.

## 4. PR 0: session ownership

Today nothing owns a session's end. Ownership is a test-and-clear flag (`MemoryHub._session_active`), there are several
direct `on_session_end` callers, and `AgentInstance.shutdown` ends whatever is open.

- **A three-state hub session:** `CLOSED → OPEN(gen, owner) → ENDING(gen) → CLOSED`, every transition under
  `_session_flag_lock`. `on_session_start(*, owner, deadline=None)` returns a `SessionTicket` (generation, owner,
  `end_done`).
  - `owner=` is keyword-only and REQUIRED.
  - `owner` is an object identity, not a label: the `AgentInstance`, a fresh token per loop invocation, or a harness's
    token.
- **Claim rules:**

  | State | Claimer | Result |
  |---|---|---|
  | CLOSED | anyone | new generation |
  | OPEN, same owner (`is`) | that owner | idempotent: the same ticket |
  | OPEN, owner with no live loop (the instance or a harness token) | a loop | handover: new generation; the old ticket is invalidated |
  | OPEN, a live loop | anyone else | `SessionHeld` |
  | ENDING | anyone | waits for `end_done` within its deadline, else a 60 s cap, then `SessionStillEnding` |

  "Live" means a claim not yet released by its context manager. It is hub state, never inferred from the owner's kind.
- **One seam: `bio_integration.bio_session(...)`, a context manager.** `run_agentic_loop` (the wrapper from step 0) runs
  `with bio_session(...) as s: outcome = _run_agentic_loop_body(..., session=s)`, then `outcome.raise_deferred()`.
  - The claim happens in `__enter__`, BEFORE `build_loop_run`, so a raise in setup unwinds through `__exit__`. The
    claim also comes before the Default Network, the prefetcher and the context pool start.
  - **Clean exit:** `end_session` as owner, made ONLY here: PR 0 removes the body tail's own `end_bio_session` call (else
    the session ends twice). The consolidation flavor is the RESOLVED one: `run_agentic_loop`'s `consolidation=` override
    first (`minecraft_harness.py` passes "full", the ship gate; the orchestrator under adoption and `console/handle.py`
    pass "full"), then `sim.is_sim_mode`. It is resolved at exit, because `sim` is built inside `build_loop_run`. The
    lightweight flavor skips steps c and d entirely, as `on_session_end_lightweight` does today. Pinned through
    `test_minecraft_harness`.
  - **The capture worker and the Default Network are the session's:** `build_loop_run` starts both and records their
    handles on the session; `__exit__` stops them first (5a), on both paths.
  - **Abnormal exit** (any `BaseException`, including KeyboardInterrupt and a SubstratePolicy raise): the persist step
    on the calling thread, steps started only while the bound (`current_stop()` or a 30 s cap) has time left.
  - **Release** happens in a `finally`, only after the started steps return. If the bound expired, the session stays
    ENDING until they do.
  - **Kept today's D13 ordering:** the body RETURNS a `PlanningLivenessExhausted` as a deferred raise. The wrapper
    raises it after the clean end, so a D13 abort still persists and consolidates.
    `test_planning_liveness::test_exhaustion_raises_after_teardown` (a source pin) moves with it.
  - **Which exception wins:** `__exit__` never replaces an in-flight exception; its own errors are logged and chained,
    never raised over one. On a clean exit the order is: the deferred D13 raise first, else `ExperienceClockStalled`.
    `ExperienceClockStalled` keeps its ERROR branch and telemetry, ahead of the narrowed WARNING handler. It still fires
    LAST, after every save and after the release.
- **The ticket travels end to end.** It replaces the `memory_hub_enabled` bool through `loop_setup` →
  `agent_loop` → `end_bio_session(ticket=, capture=)`.
  - `SessionHeld`, `SessionStillEnding` and the TypeError propagate as a typed abort of that loop. Only "no memory hub
    configured" stays a soft skip, and even then its capture worker is started AND stopped.
  - `end_bio_session`'s handler is narrowed and logs at WARNING.
- **A non-owner** (`AgentInstance.shutdown`, a drain, the orchestrator's `/new` — granted 2026-10-26, a scripts caller)
  waits on `end_done` within its deadline, else the 60 s cap. It then logs `session_end_not_reached` and returns. Its own
  after-steps run only after `end_done` or when no owner exists:
  - `memory_hub.shutdown()`;
  - the cerebellum save, which STAYS in `AgentInstance.shutdown` (`MemoryHub.cerebellum` is never wired, so moving it
    would silently drop it; `BioStack.on_session_end` keeps its own call for the harnesses);
  - its hippocampus and NAc saves are DELETED, because `end_session` makes them in owner order. They ran without EC.

  `runtime/agent_pool.py::remove` (public `maxim.create.pool()`) passes one shared deadline to every instance it stops,
  so its serial waits do not add up. A stale ticket
  after a handover ends nothing and logs at DEBUG (the normal create → loop → shutdown path).
- **Callers migrated:** `on_session_end`, `on_session_end_lightweight` and `MemoryHub.sleep()` take a ticket; a
  ticketless call raises.
  - `scripts/exp56/common.py`, `scripts/survival_world/exp61_run.py`, `r2_learned_bias.py`, `water_trial.py`,
    `scripts/spike_dm_pipeline_audit.py`.
  - `simulation/interactive.py`; the orchestrator's `/new`.
  - The docs: `docs/user/stable_api.md`'s CC8 adapter paragraph (changes with `end_bio_session`'s signature),
    `docs/semantic_similarity_analysis.md`, `docs/memory-system-interactions.md`, and `agent_factory.py`'s
    "start_bio_session no-ops" comments.
  - `aut.bio.on_session_end()` (BioStack: cerebellum + distributor) is not a hub session end.
  - Also: `AgentInstance.start_session` (the owner=self producer); `loop_setup._bind_execute_and_learn`'s `memory_hub=None`
    gate (a ticket of None keeps it); the dead `LoopController.memory_hub_enabled` (removed); the stale docstrings in
    `memory_hub.py`, `sim_adapter.py`, `math/workspace.py`, `provenance/collector.py`, `api.py` and the orchestrator's
    shutdown comment. `scripts/spike_dm_pipeline_audit.py` is dropped from the list (it makes no session call).
  - **The test migration is in scope:** about 83 hub session calls across 27 test files, 25 `start_bio_session` /
    `end_bio_session` sites (mostly `test_bio_integration.py`), the helpers `_loop_harness.py` /
    `_execute_learn_driver.py`, the six duck-typed fake hubs (each gains an assertion that a session opened), and the
    D41 guards in `test_api_core.py`, rewritten to the handover rule. `stats()["session_active"]` stays a bool (OPEN →
    True).
- **Capture worker:** `start_capture_worker()` returns a handle with its own stop Event; the handle lives on `LoopRun`,
  and stopping takes it. Two workers may overlap briefly, which is safe (`queue.Queue.get` hands each item to one
  worker), and there is never a window with zero.
- **Refusals are presented to the user:**
  - console Talk: the bridge finishes with the typed error, `talk()` raises it, and the route answers 409 with the
    reason, its `except` placed before the route's catch-all;
  - the embodied runtime: `user_warn`;
  - the CLI: `cli.main` renders it in one line, exit 2.

  The CLI sim's AUT refusal currently surfaces as `aut_died`; that is a follow-up.
- **The console lifespan** refuses to rebuild a handle on a home while a previous stop thread is alive. An embedder that
  rebuilds in-process gets a mechanization-backlog row instead.

## 5. PR 1 (#1438): the library

**As built (2026-10-29): split into PR 1a and PR 1b (owner decision).** PR 1a is the memory side: the session end
below, the store locks, the loaders' prune pass, the `Deadline` parameter, and #1520. PR 1b is the stop side:
`StopRequest` wiring and LLM-worker abandon, the cancelling console/handle waits, `start_simulation_mode(stop_event=)`,
the console lifespan and `console.stop_budget_s` (config 1.3 → 1.4), and the sim tail. PR 0 ships in the same release
as both. PR 1a departs from the text below as follows:

- **No quiesce-and-skip, and the session never stays ENDING** (owner decision 2026-10-29, after a design pass showed
  a late writer would wedge the hub for the rest of the process, and that percept threads and synchronous captures
  write EC and the embedding store where no session end can stop them). EC and the `EmbeddingStore` got their own
  leaf locks, and ATL's ref writes, reinforce and iteration go through ATL's lock. Phase a flushes the writers, best
  effort and logged, producers before consumers (capture workers, extractor, embedder; the embedder is drained, not
  stopped, since nothing restarts it), and never skips a store.
- **#1520: owed after-steps** (owner decision 2026-10-29). A shutdown that must wait for a live loop hands its
  after-steps to that generation's end (`SessionOwnership.defer_until_ended`). An abandon drops them, naming each at
  ERROR (owner decision 2026-10-29).
- **Loaders prune NAc and ATL, not SCN or EC signatures:** SCN's ids mix in the tool-pain bridge's persistent event
  signatures, and EC's `register` is an open id space no shipped path fills with hippocampus ids. They prune once per
  hub, and only against a target that READ its file (an empty target makes everything look dangling). NAc is pruned
  in place under its lock (`NAc.prune_ec_keys`), never by a dump/load round trip.
- **The plan manager and provenance end after consolidation;** the hippocampus `.backup` is taken in phase b (the
  previous session's file; sleep's own auto-save used to make it).
- **The console and the pool bound only their wait for a loop** (`AgentInstance.shutdown(loop_wait=)`); the
  agent's own end stays unbounded until 1b's `console.stop_budget_s` (owner decision 2026-10-29).
- **The cross-layer graph is pathless in every shipped builder** (never saved or loaded); #1530 owns it. Unlocked readers of EC and ATL state outside them: #1532; the
  `EmbeddingStore` save: #1531.

**As built, PR 1b (2026-10-30): split again into 1b (the stop side) and 1c (the sim side) (owner decision).** A design
pass and two delta rounds (approach note in the session record) changed the text below as follows:

- **uvicorn re-raises a signal only AFTER lifespan shutdown** (`Server.capture_signals`, 0.40), so "flag handlers
  installed before `uvicorn.run`" cannot arm the stop in time. `maxim serve` runs `console/server.py::_MaximServer`,
  whose `handle_exit` only records the signal and whose `shutdown` (run before uvicorn closes connections) makes the
  request; `timeout_graceful_shutdown` bounds uvicorn's own wait (unbounded before). Exit codes stay PR 2's.
- **Only process owners request the stop** (owner decision 2026-10-30): the lifespan, `_drain_and_stop_handle` and
  the serve path. `MaximHandle.stop(deadline=)` makes no request, so a per-handle stop in an embedder stops nothing
  else. The owner clears the request (`clear_stop_request(expected=)`) only after a stop that finished; an overrun
  leaves it set and the next lifespan start refuses (`StopStillPending`), which replaces the refuse-rebuild rule the
  text assumed (it was never built).
- **A loop's clean exit under a stop is bounded by the stop's deadline and never abandoned** (owner decision
  2026-10-30): persist always, consolidate within the bound.
- **Registry, not a `stop_event=` parameter:** loops register their own stop event for the run, `LLMWorker.start`
  registers its event (after its clear, so a worker started under a request is born stopped), and the backend
  registers its `request_shutdown`/`reset_shutdown` pair as hooks; request and clear hooks run under one lock so a
  clear never re-arms the backend under a newer request. Campaigns stop through the same registry in 1c.
- **`yielded` is a typed abort** (owner decision 2026-10-30), in `SIMULATION_ABORT_FINISH_REASONS`, outranking the
  statuses the stop causes; it moved into 1b because 1b's stop already ends a campaign's AUT loop.
- **The drain runs on a daemon thread**, never `asyncio.to_thread` (joined for up to 300 s at loop close).
- **Not built:** `runtime.yield_timeout_s` (no consumer; PR 2 adds what it needs).
- **Known limit (#1536):** a per-handle `MaximHandle.stop(deadline=)` with no process request bounds its waits and
  `AgentInstance.shutdown`, but not the session end of a Talk loop that owns the session (`bio_session`'s clean exit
  reads only the process request). The console's own stop is bounded; the per-session bound is the issue's.
- **An embedder running its own `uvicorn.run`** (not `maxim serve`) gets no signal-time request and an unbounded
  uvicorn graceful wait before the lifespan's budget starts: say so in the note to pulse.
- **1c:** stop checks in the DM and generative runners (their loops never check one, and direct router calls are not
  abandoned), the tail order with an early and a late report save, the bounded AUT join with a share and a reserve,
  refusal at start, the post-sim stdin prompt, `wait_ready` honouring a stop, and the reset re-check.

**As built, PR 1c (2026-10-30; the half outside the sim tail, the tail follows as the orchestrator decomposition's tail slice).** An adversarial design pass took four rounds; the last came back SOUND WITH CONDITIONS. Owner decisions 2026-10-30:
- D1: a sim started under a stop raises `StopStillPending` (`refused="simulation"`), before its start provenance, and again after the model load.
- D2: report.json is saved early and late under a stop (tail).
- D3: a direct router call is abandoned in ONE seam, `LLMRouter._complete_text`.
- D4: the roundup is skipped only under a process stop request (tail).
- D5: the fence touch is approved, built as the decomposition's tail slice with Session A.
- The stop is typed where the API already raises (`LLMCallStopped`, from `generate`).

Departures from the text below:
- **Two abandoners, not one registry.** `LLMWorker` abandons its own call (its stop event is registered) and marks its context (`mark_abandons_on_stop`). The router abandons every other call on a helper thread named after the caller. A cancel event bound only for a timeout does not opt out.
- **`reset_shutdown()` refuses under a request** (`reset_unless_stopped`, under the hook lock). The clear hook uses the private `_clear_shutdown_event`.
- **`wait_ready` keys on the stop request,** never on the backend event, which every normal sim end sets.
- **The runners re-check after each LLM-backed step** (`generative_runner._narrate_turn`; the DM after delivery and after classification), so no fallback built from an abandoned call is delivered or recorded. A human's interactive choice is kept.
- **The session end keeps a reserve for its waiter.** This fixes 1b's shared-deadline overrun: `_end_bound`, 25% of the time left with a 1 s floor, only when the bound is a stop's deadline. The abandon test keeps the full bound, and the session's own capture flush keeps a 0.5 s floor; the hub's writer flushes stay unfloored.
- **Report-time waits take `deadline=`** (`run_provenance`, `serving_provenance`); the tail passes it. Under it the end code stamp is skipped (`unknown: stopped`), and skipped probes stamp `model_changed_during_run: null`.
- **A stopped sim's container removal is detached** (own session, reaped on the next append).

- **`utils/deadline.py::Deadline`:** frozen and monotonic. It has `remaining()`, `expired()`, `share(fraction of
  remaining)` and `sub(seconds)` (both clamped to the parent), and `Deadline.none()`. Never persisted.
- **`StopRequest`** in `utils/cancellation.py`, the neutral home: memory, runtime, console and the CLI import it, and
  they must not depend on the LLM package. It is a sticky event, a deadline and a reason, set by `request_stop()`.
  - It is the THIRD cancellation primitive. The other two are the backend's process-wide event in
    `models/language/cancellation.py` (`request_shutdown()`, which is registered as one of the stop events, so `utils`
    never imports `models`) and the per-call ContextVar in `utils/cancellation.py`.
  - The CLI gate reads StopRequest, never the backend event: the orchestrator calls `request_shutdown()` at every
    NORMAL sim end.
  - LLMWorkers register in ONE place, `LLMWorker.__init__` (weakly).
  - It sets the backend `request_shutdown()` and every REGISTERED stop event: loops, the campaign's, and every
    LLMWorker's.
  - `reset_shutdown()` re-arms only the backend event.
  - An in-flight LLM call is ABANDONED: its waiter returns within a 0.5 s poll. Late writers after the save are lost by
    design, never torn.
  - A conftest autouse scrub clears the stop request.
- **The waits cancel.** `_drain_and_stop_handle(deadline)` and `MaximHandle.stop(deadline=)` call `request_stop` FIRST.
  Each wait then takes `min(old default, a share of the remaining budget)`. The `campaign_wait_s` / `talk_join_s` kwargs
  survive as caps (`runtime-tools.md`'s console invariant, D74, `test_console_sandbox`). A campaign is stopped through
  the new `start_simulation_mode(stop_event=)` (granted; pinned, so rule T applies):
  - it is INPUT-ONLY, OR-ed with the sim's own event, which the sim sets on normal completion (#1002); the sim never
    sets it;
  - it is not forwarded into the `/new` recursion once set;
  - an external stop finishes the run as `yielded`, not `cancel`.

  The refuse-on-`yield_requested()` check comes BEFORE the entry `reset_shutdown()`.
- **`end_session(ticket, deadline)`:**
  - **a. Quiesce the writers of LOCKLESS stores, within a share:**
    - the ConceptExtractor (flush and stop);
    - the opt-in `semantic-embedder` thread (drain and stop);
    - the capture worker (by its handle).

    The grounder runs synchronously (`worker_pool=None` in every shipped builder). The Default Network stops as today,
    with its fixed joins counted as 3 s of overhead: through the pain bus it reaches only the locked stores, and its
    background tasks touch only the spatial map and salience. A writer still alive past its share means its stores are
    skipped and logged, and the session stays ENDING.
  - **b. Persist, ordered by value:**
    1. the hippocampus (ONCE), then EC, the embedding store and the cross-layer graph, all inside one HIPPOCAMPUS RWLock
       READ section. The hippocampus's eviction callbacks write those three under its write lock, so the saves are
       serialized against evictions and the four files agree.

       The RWLock is not reentrant and prefers writers, so taking the read lock again inside would deadlock against a
       waiting writer. Inside the section, the hippocampus is saved through a new `_save_locked()` / `_dump_locked()`
       (the caller holds the read lock), and EC, embeddings and cross-layer take no hippocampus lock of their own;
    2. NAc, after EC (skipped together if EC is skipped);
    3. ATL;
    4. SCN, AG, the plan manager and provenance.

    The escalation bridge is NOT added: its file is machine-wide (`util/escalation_learning.json`, #984), so a
    session-end save would let every hub overwrite it. That is a follow-up.

    NAc decay runs exactly once.
  - **c. Consolidate inside the budget:** sleep, promotion, ATL and AG consolidation. Each step starts only if the time
    left covers its reserve (cited from §1: sleep 3 s + 1 s per 1000 memories, the others 1 s each). Skipped steps are
    logged by name.
  - **d. If c ran, re-save the set,** dependants before owners.
- **Store/lock table** (it goes in the docstring):

  | Locked | Lockless |
  |---|---|
  | NAc (`dump` holds `_lock`; `remove_memory` mutates `memory_ids` outside it: lost-or-present, not torn) | EC, embeddings, cross-layer (covered by b.1) |
  | SCN | plan manager (pauses plans outside its lock) |
  | hippocampus, ATL, AG (RWLock) | provenance (writes outside its lock) |
  | the hippocampus graph (`DependencyGraph._lock`) | cost tracker |

  NAc's `_register_causal_in_ec` is off (`ec=None` in every shipped builder); a builder passing `ec=` must re-check it.
- **Loaders prune and COUNT resolvable dangling references only** (owner decision 2026-10-25), from a per-store table.
  Not pruned: NAc links keyed by labels or `entity_ref`, imagination-tagged links, concept refs in flight.
- **The console lifespan:**
  - It builds `Deadline(stop_budget_s)`, from the embedder's ONE seam, `build_app(stop_budget_s=)` (not
    `device_console_handoff`: its frozen `ConsoleHandoff` crosses to pulse with no `extra`, and pulse forwards only
    `.origin`), else `console.stop_budget_s`, default **15 s** (owner decision 2026-10-25: fits the daemon's ~20 s
    kill). An embedder that owns an injected handle's stop (`embodied_device.md`) passes its own
    `handle.stop(deadline=)`.
  - **New config:** `console.stop_budget_s` and a new `runtime` section with `yield_timeout_s`.
    `CONFIG_FORMAT_VERSION` goes 1.3 → 1.4, with the schema/resolution fixtures and rows in the persistence-config brief.
  - It runs the stop on a worker thread and waits for it within the deadline, so it RETURNS in an embedder.
  - Under `maxim serve` (a Maxim-owned process): flag handlers for SIGINT and SIGTERM are installed before
    `uvicorn.run`, so uvicorn's re-raise lands on them and it returns 0. PR 2's last-resort watchdog is armed there too.
- **The sim tail** (granted hunks, helpers per rule T):
  - report.json and the action log are written before slow steps;
  - the roundup is skipped on any stop (owner decision);
  - a bounded wait on the AUT session's `end_done` replaces `join(300)`;
  - it refuses to start when `yield_requested()` is set.

## 6. PR 2 (#1344): the CLI only

- **One handler** for SIGINT and SIGTERM, replacing `_force_exit_handler`.
- **The gate is monotonic:** raise `KeyboardInterrupt` only if no session was ever opened in this process and no stop was
  ever requested; otherwise `request_stop(Deadline(runtime.yield_timeout_s = 30 s), "yield")`. This amends advisor item
  9 (owner decision 2026-10-25: cooperative once a session is open).
- **Repeated signals:** a repeated SIGTERM is ignored; a second SIGINT forces (owner decision).
- **Exit:** through `finally` and atexit.
  - A clean yield exits **0** for non-sim processes (`maxim serve`, chat/REPL, the device), per the owner decision of
    2026-10-25.
  - **A yielded SIM run is a FAILURE for evidence** (owner decision 2026-10-26, the strict default under "weak evidence
    never gates"):
    - it finishes `yielded`;
    - `yielded` joins `sim_types.SIMULATION_ABORT_FINISH_REASONS` / `is_simulation_run_failure`;
    - it exits with the abort code (**4**, as `cancel` does today);
    - `scripts/_provenance.py::spawn_evidence` also checks `finish_reason`.

    `docs/user/simulation.md`'s exit-code table and `test_finish_reason_1002` are updated.
  - A signal BEFORE any session opens exits with no report: SIGTERM → 0 (advisor item 9), SIGINT → 130 (the Ctrl+C
    convention, as `cli.main` does today).
  - A session refusal (`SessionHeld` / `SessionStillEnding`) exits 2, like the CLI's other refusals.
- **A last-resort watchdog**, in Maxim-owned processes only (the CLI and `maxim serve`): it fires at
  `min(deadline, supervisor_kill_at) − kill_budget`, kills the registered server process groups, restores the terminal,
  writes one line, and exits **75**.
  - `supervisor_kill_at` is the deadline + 5 s unless a supervisor sets it via the env var `MAXIM_SUPERVISOR_KILL_S`
    (paired with a conftest scrub).
  - `kill_budget` is a 2 s constant.
  - Exit code 75 and the pidfile are new supervisor contracts, listed in the note to pulse.
- **Reporting:** the yield is reported as `yielded`, not `cancel` (`operator_stop` is the Event that maps to `cancel`).
  `test_tty_stdin.py`'s pin of `_sigterm_handler` changes with the handler's removal. Outer loops check `yield_requested()`. The dead
  thread-side handler in `tty_stdin.py` is removed.

## 7. PR 3: spawned servers

- A registry of every spawner.
- A per-port pidfile (`_format_version`, atomic, `stable_hash` of the cmdline) records owner and server pid plus
  create_time. The create_time source is `/proc/<pid>/stat` on Linux and `ps -o lstart=` on macOS. psutil is used only
  through `utils/optional_deps.py`, never assumed. A process that reuses a server re-stamps the owner; only the named owner removes the file.
- The reaper acts only when the owner is dead AND the create_time matches (on by default; owner decision).
- Servers are killed with `os.killpg`, asserting `start_new_session`.
- Linux pdeathsig goes through an exec wrapper, from a long-lived thread only. macOS gets the pidfile only (accepted).

## 8. Proof

- **Red gates first,** `xfail(strict=True)`, each proven by deleting its mechanism.
- **PR 0:**
  - who ends the session, exactly once, for each entry point;
  - an unbounded `shutdown` with a hung Talk tail returns and logs;
  - a loop failing in setup releases its claim;
  - no zero-worker window;
  - a stale ticket logs quietly.
- **The LLM in the subprocess harnesses** is a localhost OpenAI-compatible stub that accepts and never answers, so it
  also proves the 0.5 s abandon path. It is selected through config written into the harness's own data home, never an
  env var. The 60 s claim cap is a module constant the tests inject.
- **PR 1:** a daemon-faithful harness against `maxim serve` and an embedder (SIGTERM and SIGINT, then SIGKILL at the
  measured delay):
  - a never-ending campaign is stopped and saved within the budget;
  - no store is torn by an eviction during the persist;
  - EC and NAc stay consistent;
  - a hung-tool session is logged and its files stay intact.
  - Also: the `test_memory_hub` suite and the briefs (§10).
- **PR 2:**
  - a SIGTERM mid-sim → report.json marked yielded, exit 4 (a failure for evidence);
  - a SIGTERM to `maxim serve` / chat → clean yield, exit 0;
  - a SIGTERM at startup → no report;
  - a blocked stop → exit 75.

## 9. Follow-ups (filed as issues when PR 0 opens)

- Tool-call cancellation.
- The CLI sim's refusal is mislabelled as `aut_died`.
- The NAc imagination-link mutation after the NAc save.
- The escalation bridge: a non-atomic save of a machine-wide file (with #984).
- **The library still exits the process or installs signal handlers outside Maxim-owned processes** (`maxim bench`'s two
  SIGINT handlers are Maxim-owned and stay; `simulation/tty_stdin.py`'s dead SIGTERM handler is removed by PR 2). Two
  sites break §2 in an embedder:
  - `simulation/orchestrator.py`'s stall hard-abort thread calls `os._exit(4)`, which can kill pulse mid-save on a wedged
    campaign;
  - `inference/transcribe_audio.py` installs signal handlers with an `except Exception: pass`.

  Both are rerouted through `request_stop`. Mechanization backlog: a CI grep that only `cli.py` (and `console/server.py`'s
  serve path) and `bench/` call `signal.signal` / `os._exit`, with these sites as listed exceptions ([outstanding.md](outstanding.md) **M81**).
- An embedder rebuilding a handle in-process while a stop thread is alive. Mechanization backlog **M82**: a per-home
  "stop in progress" lock taken by `end_session`.
- **Pulse:**
  - catch `KeyFileUnusable`;
  - pass `stop_budget_s` to `build_app`;
  - bound its own exit;
  - measure on a CM4;
  - learn exit 75 and the pidfile.

  The questions to send are in the session's note to pulse, relayed by the owner.

## 10. Plan-review folds (2026-10-26) and the briefs PR 0-2 must update

**Owner decisions taken in the review round:**
- A yielded SIM run is a failure for evidence (§6).
- H holds hunks in other sessions' files, each owner told:
  - `runtime/agent_factory.py` (V);
  - the `memory/` and `math/angular_gyrus.py` loaders (R);
  - `console/handle.py`'s stop and `console/server.py`'s lifespan/`build_app` (S; H rebases over S's `_handle` move);
  - the unclaimed `runtime/bio_integration.py`, `loop_setup.py`, `loop_controller.py` and `simulation/interactive.py`.

**Briefs and invariants rewritten in the same PRs,** each `[engineering]` invariant with its `Regression guard:` line:
- **bio-memory.md:**
  - "The object that OPENS a memory session is the object that CLOSES it": nesting becomes generation handover plus
    same-owner idempotence; its "Residual gap" is restated; the D41 guards are rewritten, and the D41 row in
    `docs/bugs/README.md` is updated.
  - The Cerebellum invariant: unchanged, since the save stays.
  - New invariants: the three-state session (`owner=` required, a ticketless call raises); the persist order and the
    hippocampus read-lock section; loaders prune and count.
- **runtime-tools.md:**
  - the thread model (the capture thread is the session's, by handle);
  - the console stop invariant (deadline + caps);
  - new: the library never exits the process (Regression guard: process invariant, mechanization backlog M81).
- **persistence-config.md:** `Deadline` is never persisted; the pidfile's `_format_version` and owner rules; the config
  rows.
- **`docs/wiring/session-end-seams-per-runtime.md`:** the session-end column per runtime.

**Exceptions:** `SessionHeld` and `SessionStillEnding` live in `maxim/exceptions.py` under a `SessionRefusal(MaximError,
RuntimeError)` base.

**DECISIONS.md** gets two entries: the library never exits, and session ownership as its own layer.

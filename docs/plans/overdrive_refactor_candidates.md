# Overdrive refactor candidates: what an embedded world asks of Maxim, ranked

> **PROPOSED 2026-11-16: a candidates list, not a plan. No code.** It comes from the design exchange with Maxim
> Overdrive, a game world under the owner's control that embeds Maxim agents. Overdrive's side is on
> `dennys246/Maxim-Overdrive` `phase1/design-v2` (draft PR #4): its drafts (`docs/maxim_upstream/`) and its reply to
> Maxim's answers (`docs/design/reviews/maxim_response_1/reply.md`). Each item below enters Maxim only through
> Maxim's own process (an issue, a plan with a three-lens review, or an experiment with a four-lens design review).
> Nothing here is patched in from Overdrive, and **game runs never go on Maxim's ledger.**
>
> Citations are on `origin/main` **98c6889c** (2026-11-16). That tree's `src/` is identical to `7db97e21`. They are
> `file::symbol` with a line hint; the symbol is the citation, the line will drift. A three-lens review (Executor,
> Architecture, Wire integrity) ran on the first draft, and its findings are folded in.

## Why this list exists

Overdrive needs a Maxim agent that **a host drives**. The host calls it with world time and a frame, and gets back,
as values, what the agent wants to do. The agent never runs its own loop, thread, clock, model call, sleep or tool.
Today Maxim's agent owns all of those.

Most of what that takes is **also a Maxim correctness or research-integrity gain**. Once an agent's state, clock,
RNG and ids belong to it, a run can be replayed bit for bit with one variable changed. Paired replays shrink the
confidence intervals the BF2 floors and effect-size pass rules depend on. That gain is the reason for the ranking,
and it is what makes these items Maxim's work rather than a favour to a game.

**Two limits on that gain** (Overdrive reply §5):
- Pairing helps substrate claims. For an LLM arm it ends at the first diverging token, so the LLM's replies must be
  recorded and replayed.
- A paired analysis estimates d_z, not d. A prereg that uses replay pairing says which one it gates on.

## The tiers

- **Tier A, do on engineering grounds.** Guarded by tests, each proven by deleting what it guards. No experiment is
  owed, because each one either fixes a defect or changes no behaviour on a robot.
- **Tier B, engineering with parity guards.** Each restructures a hot path. It may land only with a guard proving the
  robot path is unchanged: a bit-identical state digest on recorded *replies*, not a behavioural comparison. A
  behaviour **claim** the restructure enables waits for its own experiment.
- **Tier C, test, don't refactor.** Design choices Maxim made without evidence, which an embedded world can finally
  test. Each needs a prereg, plus a transfer check in a Maxim-owned harness (§Cautions).

Items keep `OD-` ids where Overdrive's drafts already named them. Maxim may reassign them when it accepts an item.

## Tier A

### A1. One `AgentHome` object, owned per agent

- **What:** route every store an agent writes through one per-agent home object, so a home can never half-resolve
  to `~/.maxim`.
  - `AgentFactory` already scopes Hippocampus, NAc, ATL, SCN, EC, AngularGyrus and the cerebellum.
  - **The list to close is the repo's own registry**, `cli_utils.py::_GLOBAL_PATHS` (`:608`), not a hand count. Its
    global learners include:
    - semantic embeddings (`util/semantic_embeddings.npz`, also `integration/memory_hub.py:246`);
    - the pain detector, statistician and significance weights;
    - the learned bounds (`util/learned_bounds.json`);
    - short-term memory, the cross-layer graph and the tool index;
    - planning.
  - **Writers outside the registry:**
    - `utils/agent_output.py` (`shared/outputs`, plus the process-global abstraction buffer);
    - `utils/last_run.py`, `provenance/store.py` and `runtime/probe_cache.py`;
    - the cost tracker (its filename also stamps `datetime.now()`);
    - `inference/transcribe_audio.py` (`util/whisper.json`) and `analysis/roy_log.py`.
  - **Process-global in memory:** `utils/filesystem_policy.py::get_effective_cwd` is a module global, so one agent's
    mode change moves every agent's working directory.
- **Paths resolved against the working directory**, which collide with a host's own `data/`:
  - `utils/cloud_audit.py::CloudAuditLogger` (`:49`, `data/logs/cloud_audit.jsonl`);
  - `utils/response_config.py` (`:65`);
  - `models/language/config.py` (`llm.json` from `cwd/data/util`);
  - `hardware/config.py` (`robots.yaml`, cwd-only when `data_home()` fails);
  - `utils/prompts.py` (a fallback, used only if the bundled copy is missing).
- **Front gate:** rides on `AgentFactory._resolve_persistence_dir` (`runtime/agent_factory.py:770`) and
  `utils/paths.py::RUN_DIR_KINDS`. Not `resolve_run_dir`: that is a CLI lookup that searches the CWD first and
  creates nothing ("Not for request data").
- **Guards:**
  - `_GLOBAL_PATHS` becomes the checked registry, so a new learner that isn't routed through the home fails.
  - Two agents in one process never open the same file, shown by a filesystem-trace test.
- **Venue:** a plan, merged with the open #984 (shared learned state outside any home). **Line:** 1.3.x/1.4, on its
  own clock.

### A2. Per-agent keyed random substreams, one design with #1305

- **What:** each consumer draws from a keyed substream, so adding an NPC or changing one cadence never shifts another
  stream.
  - **The derivation (owner decision 2026-11-19):** a `stream` argument added to `utils/seeding.py::make_agent_rng`,
    keeping its existing `sha256(f"{base_seed}:{agent_id}")` derivation with the stream joined by `:`, so a stream
    created today as `agent_id="npc_key:stream"` keeps its seed. `:` is forbidden in NPC keys and stream names (the
    join is ambiguous otherwise, and escaping later would break that survival). The `stable_hash_64_signed`
    alternative is dropped: it gives different seeds.
  - **Overdrive adopts this derivation over its host-side handout (2026-11-17), on two conditions,** settled in
    #1305's scope: `agent_id` is the host's **persisted** NPC key, never an ordinal, so streams survive saves and
    roster edits; and a host's `base_seed` comes from its log root.
  - **This is the same work as the 1.4.0 release gate `1305.owned-rng`** (#1305, `scripts/release_gates.json`). That
    gate already requires default_network, attention, spatial and `decisions/significance.py` to draw from owned
    generators seeded from `seeding.py::get_global_seed`. A2 extends it with a per-agent, per-stream key. The robot
    path derives `base_seed` from `get_global_seed()`, and a host supplies its own. Release-gate item scopes are
    immutable (`scripts/lint_release_gates.py`, shrink-only history), so the per-agent key enters as a **new** gate
    item, not an edit of `1305.owned-rng`'s scope (plan review of Overdrive's embeddable-agent plan v3, 2026-11-19).
    **Owner decision 2026-11-19: it lands after the 1.4.0 cut**, so it adds no 1.4.0 gate item.
  - About 26 stdlib draws sit outside `seeding.py`, plus numpy's legacy global draws (`attention/salience_map.py`,
    `hardware/simulation/streams.py`). Exempt, and named as such: `models/language/anthropic_backend.py`'s backoff
    jitter.
- **The rule that comes with it:** `seeding.py::seed_all` (`:28`) writes `PYTHONHASHSEED` (`:48`) and reseeds the
  process's global generators, and **an embedded path never calls it.** Its only `src/` caller is `cli.py:1829`. Every
  `get_global_seed()` consumer then needs an explicit seed on that path:
  - `similarity/semantic.py:205` (`_lsh_seed = get_global_seed() or 42`, persisted LSH hyperplanes, which would
    silently become 42);
  - `simulation/generative_runner.py`;
  - `simulation/serving_provenance.py`.
- **Front gate:** rides on `utils/seeding.py`, with no new dependency.
- **Guards:**
  - two agents with one cadence changed, and the other agent's draws are identical;
  - two processes with different `PYTHONHASHSEED`, per the stable-hash invariant;
  - #1305's gate test.
- **Venue:** #1305 (widened). **Line:** 1.4.0 (the gate is a 1.4.0 item).

### A3. The perceived-time port, plus the host's injection point into the experience clock

- **What A3 owns:**
  - the **perceived-time port**;
  - the **host's injection point** into the existing `runtime/experience_time.py::ExperienceClockDriver`;
  - a few id and stamp sources the experience-clock plan doesn't cover.
- **What A3 does not own:** decay timing, eligibility and the temporal windows. Those are
  `experience_clock_decay.md`'s (#1361, owner decisions 2026-10-17: in-session decay anchored to the driver, and no
  forgetting while switched off). A3 rides on that plan. On a robot, the machinery clock **is** the driver, not the
  wall clock. NAc decay-on-load is already opt-in at load (`decisions/nac.py`, about `:4317`), so it is not on the
  step path.
- **The perceived-time port.** It is the only source of time a prompt may contain. The host supplies it; on a robot
  it is the wall clock, so robots keep the date the web-search guidance needs.
  - **Exact time reaches the model in at least seven places:**
    - `agents/prompt_builder.py::build_datetime_section` (`:237`), fed by `datetime.now()` in
      `PromptBuilder.build_prompt` (`:1125`) and added in `PromptBuilder._add_guidance_sections`;
    - `PromptBuilder._build_engage_prompt`, a second `datetime.now()` (`:2400`) rendered as `CURRENT DATE: … at …`
      (`:2417`), on the post-search path;
    - `agents/context_pool.py::_entries_to_text` (`:468`);
    - `agents/context_pool.py::get_context_text` (`:493`), an `HH:MM:SS` stamp on every entry;
    - `agents/exec_agent.py::_build_llm_context`, workspace-file modification times through `time.localtime`
      (`:1189`);
    - `tools/http_fetch.py` (`accessed_at`, in a tool result);
    - `tools/reachy.py` ("The last sound faded …s ago", robot-only).
  - **Time also decides what is recalled.** `retrieval/temporal_signal.py` uses `datetime.now()` as the base for
    "yesterday" and "last Tuesday" (`:101`). `time/temporal_signature.py` and `similarity/signature.py` use
    `datetime.fromtimestamp`, which depends on TZ, for SCN hour and day features, and those reach the model through
    `temporal_patterns` and the `[TEMPORAL]` tag (`agents/exec_prompts.py`). These are classified too: the port
    supplies relative-date resolution.
- **Machinery reads the experience-clock plan does not list**, which A3 must route or allowlist:
  - **Tool duration:** `runtime/tool_dispatch.py` measures it on the wall clock (`exec_elapsed`, `:1343`).
    `_observe_energy_cost` turns it into an NAc valence (`:626`, NEGATIVE above 2.0 s) and an outcome signature
    (`:631`). So NAc's state, and a valence sign, depend on host speed. **B2's parity guard cannot pass until this
    is resolved** (see B2).
  - **The rest:**
    - NAc event ids from `time.time_ns()` (`decisions/nac.py:1163`) and the event time (`:1162`, in `NAc.record_event`);
    - the uuid4 ids (36 code lines in 24 files);
    - Hippocampus consolidation pressure (`memory/hippocampus_consolidation.py:468`);
    - memory-agent salience decay (`agents/memory_agent.py`, about `:827-834`);
    - the stale-proposal age (`runtime/agent_loop.py`, about `:458`);
    - `time.monotonic` stamps saved across a restart.
- **Guards** (owner-accepted pushback, 2026-11-16: tests and a lint, not a type):
  - **For machinery reads: M72**, extended. M72 is the experience-clock plan's wall-clock-read lint, with its §(g)
    allowlist. A3 adds `time_ns` and uuid4 ids and the reads above to its scope, rather than adding a second lint.
    Its test uses an **advancing** fake clock, because a frozen one hides every cadence bug.
  - **For the prompt, a planted-render test:** plant a sentinel time with unusual seconds **in the machinery clock**,
    never in the perceived-time port (prompts may show the port by design), and pin `TZ`. Fail if its raw,
    clock-format or date-format rendering appears in any built prompt or tool result. Two levels (Overdrive
    correction_1 and its follow-up, 2026-11-17; their review TB-S1):
    - **end to end: origin offset only.** Shifting the machinery clock's origin changes no time difference, so
      nothing a prompt renders may change. A scaled clock is not tested here: it legitimately changes decay and
      recall, so the test could not tell a leak from physics;
    - **at render level, on a frozen store: offset and scale,** so relative renders ("3 s ago") cannot leak the
      machinery clock either.

    The retrieval and SCN tests pin `TZ` too.
  - **For the prompt, a prompt-module lint:** it bans `datetime.now`, `strftime`, `time.localtime` and
    `datetime.fromtimestamp` outside the port. It starts on:
    - `agents/prompt_builder.py`, `agents/context_pool.py` and `agents/llm_fallback.py`;
    - `agents/exec_agent.py` and `agents/exec_prompts.py`;
    - `tools/http_fetch.py`.
- **Also in A3:** the five `x or time.time()` sites, filed as **#1684**.
- **Front gate:** the port is the only new surface. The existing seam, `ExperienceClockDriver`, separates experience
  time from wall time, but not "time the agent uses" from "time the agent is told".
- **Venue:** a plan coordinated with #1361 (Overdrive's world-clock drafts A-pre/A-post are input). **Line:** A-pre
  (the port and the prompt guards) on its own clock. A-post (the machinery reads, the injection point) with
  #1361's slices and 1.3.6.

### A4. Embedded-path leak closure

- **First, the front gate it owes** (review SF-2). A host-driven agent is a **new runtime mode**, not a refactor.
  Before any of this is built, Overdrive's embeddable-agent plan must answer: "why can't the 1.1.4 world seam
  (`archive/world_seam_1_1_4.md`, shipped as 1.1.4), with Maxim owning the loop, serve Overdrive?" A4 lists what the mode must close
  if that plan is accepted.
  - **Overdrive's answer (2026-11-17; front gate ACCEPTED by the owner 2026-11-19).** The seam runs
    `run_agentic_loop` on its own thread on loop-timing cadence and acts through a bridge. Five requirements:
    - **time:** the world runs on an integer tick with a time dial; a wall-cadence loop drifts at any dial but 1×,
      and its prompt renders the host clock (`PromptBuilder.build_prompt`'s `datetime.now()`);
    - **replay:** a loop on its own wall-timed thread orders actions by scheduling;
    - **one door:** every NPC action enters as a validated intent at one gate, in canonical log order; a bridge
      the loop calls when it chooses is a second door;
    - **trust:** players' words are untrusted, behind a host-driven worker process; a loop that owns its tools,
      environment and home in the host process cannot sit behind it;
    - **scale:** one owned loop per agent (thread, router lock) shares no host model gateway or scheduler.
  - **Maxim's reading:** "time" alone would not settle it, since A3's port plus `ExperienceClockDriver` fix the
    prompt clock inside the seam. Replay, one door and trust are structural. Overdrive may build one unreplayed
    prototype NPC through a bridge to test latency, cost and feel; it stays outside their replayable core and makes
    no claim.
  - **Owner decision 2026-11-19: the front gate passes** on replay, one door and trust. Scale is the weakest of the
    five (a shared gateway could be added to the seam), and is not needed for the verdict. Passing the front gate
    accepts the case for a new runtime mode, not a design: Overdrive's embeddable-agent plan still gets its own review
    round before any Maxim `src/` lands for it.
- **Fix at the root where the robot path has the defect too** (review SF-3, CLAUDE.md "no band-aid fixes"). Each path
  below says whether the robot behaviour is a defect (fix it for everyone, in tier A) or intended (closed only on the
  embedded path).
  - **Prefetch reads the host's working directory.**
    - `LoopController.run_prefetch` calls `prefetch_for_input(cli_text, cwd=os.getcwd())` (`runtime/loop_controller.py:614`)
      on each CLI input that isn't a confirmation or retry (`runtime/agent_loop.py:1571`). The prefetcher is built
      with `base_path=os.getcwd()` (`runtime/loop_setup.py:552`).
    - So the leak lives in those two `os.getcwd()` calls, not in `get_effective_cwd()`.
    - Under the offline latch, `discover_files_with_find` swallows the refused `find` at debug level
      (`runtime/prefetch.py`, about `:595`). **That is a robot-path silent swallow and a defect: fix it at the root.**
    - Embedded: prefetch off structurally. Overdrive's workers also start with the working directory at the empty agent
      home, which is defence in depth, not the guard.
  - **The workspace manifest.** `build_workspace_manifest` puts the working directory's name and tree into the prompt
    (`agents/prompt_builder.py`, `get_effective_cwd()` at `:1248`), gated only on the roster having file tools
    (`:1503`). The exec agent's WORKSPACE section (`agents/exec_agent.py`) has no roster gate.
  - **The short-question shortcut.** `PromptBuilder.build_prompt` answered time, date, identity and greeting questions
    without the model (`:1102-1137`, via `agents/llm_fallback.py::generate_simple_answer`). It fired on **any**
    `triggering_input`; the wake-word check (`:1108`) gates only the fallback to `cli_inputs`. **Decided 2026-11-17:
    not intended; it is deleted, fallback included (#1689).**
  - **Writes outside the home.** `tools/base.py` resolves relative paths with `base or os.getcwd()` (`:811`), and
    `contained_path` with no `allowed_dirs` returns the path unchanged. The embedded roster must always set
    containment. A test proves it does.
  - **No logging takeover:** `api.py` replaces the root logger with `force=True` (`:113`).
  - **No blocking:** `agents/exec_agent.py` sleeps inside `_propose_goal` (`time.sleep`, `:1495`). A host-driven step
    never blocks, so the static ban covers `time.sleep`.
  - **A static import ban** for the embedded closure. The advisor's three dynamic tests (including
    `tests/integration/test_advisor_gate.py::test_no_effect`, `:1261`) miss lazy imports on paths a test never runs,
    such as `runtime/lane_backends.py::_maybe_inject_cloud_fallback` (`:1249`).
  - **The embedded roster** is capability-based, with no `build_tool_registry`, `data_home()` or `config_loader`, and
    no env reads. This is Part 1's roster, engineering-gated. Making capability rosters the **robot** default is C4.
  - **A cancel never reaches the parser.** The parse-failure path (`agents/llm_worker.py`, `:1322`, inside
    `LLMWorker._process_request`) builds a canned reply from `triggering_input`. **This one lands with B1**, which
    introduces the cancelled variant.
- **Guard:** one integration test per path, plus the static ban, each proven by deletion.
- **Venue:** a section of Overdrive's embeddable-agent plan, after its front gate. The robot-path defects are filed as
  issues. **Line:** on its own clock, except the cancel bypass (with B1).

### A5. Pickle-free embeddings

`similarity/semantic.py::EmbeddingStore.load` uses `np.load(..., allow_pickle=True)` (`:662`). Filed as **#1685**.
**Line:** 1.3.x.

## Tier B

### B1. The stepped deliberation seam (Overdrive's OD-A10–A15; the hybrid's §A.1b)

- **What:** `think(continuation, input) → request | reply`, with an explicit frozen **deliberation continuation**
  (Overdrive reply §2): not a generator, and not the whole agent. One continuation carries:
  - the deliberation cycle, the transcript and the bio-enrichment results.
    `runtime/agent_loop.py::_run_deliberation_cycles` (`:521`) caps a tick at three model calls, cycle 1 included,
    feeding reasoning back through bio-enrichment between them;
  - the `next_actions` chain (`agents/llm_worker.py:1356`, `runtime/agent_loop.py:1858`), which runs later actions with
    no further call, each one validated by the host when it dispatches;
  - a basis (hybrid §A.1b.5 `basis_seq`), so a thought that went stale while the body moved is detectable.
- **The cut:** at `models/language/router.py::LLMRouter._complete_text` (`:945`) as a request value, and at the parse
  into `LLMProposal`. The gateway sits above the backend, not as a cancellable `LLMBackend`.
  - **Model calls outside `think`** that the cut must name (route through it, or refuse on the embedded path):
    - `planning/adaptive_planner.py::AdaptivePlanner` (built in `runtime/bootstrap.py`), whose LLM decomposition
      calls `generate_structured`;
    - `bridges/tool_pain_bridge.py::ToolPainBridge._generate_reflection` (`generate`), on the settle path, so B2
      meets it too;
    - `agents/fear_agent.py::FearAgent._analyze_with_llm` (`generate_json`).
- **The v1 reply type** reserves `cancelled(prefix)` and `truncated` variants and a speech field the host can read
  before the JSON closes (Overdrive reply §3). Otherwise barge-in and first-token latency are designed out. The token
  iterator itself can come later. A4's parser bypass lands here.
- **Ride on:** cancel already exists, as `utils/cancellation` plus `LLMRouter._complete_text_abandonable` (`:985`). The
  plan says why that does or doesn't suffice before adding anything.
- **Gate** (owner-accepted pushback, 2026-11-16):
  - engineering, with parity guards: a state digest identical under instant and seeded-delayed replies, and parity on
    recorded replies from Exp 49 and Exp 60;
  - **lands after the 1.4.0 cut** (M89 forbids hybrid-family `src/` before it), as **one design with hybrid §A.1b**;
  - sequenced with the agent-loop decomposition slices, so `agent_loop.py` is restructured once.
- **Line:** the 1.5 hybrid line's src phase.

### B2. Split `execute_and_learn` into dispatch and settle, as #1150

- **What:** `runtime/tool_dispatch.py::execute_and_learn` fuses dispatch with learning. Split it so credit lands when
  the outcome is **measured**, not when the action is dispatched. **The split is already tracked as #1150** (dispatch,
  learn and after-effects steps; the function's length-baseline exception names it). B2 is #1150, with a settle step
  a host can delay.
- **Scope** (review: Wire integrity):
  - **Three callers:**
    - the autonomous path (`runtime/agent_loop.py:2256`);
    - `LoopController.handle_confirmation` (`runtime/loop_controller.py:388`, SUPERVISED);
    - `loop_planning.drain_approved` (`runtime/loop_planning.py:186`, PLANNING).
  - **Four booking paths that settle without it:**
    - `book_refusal` (`loop_controller.py:423`);
    - `book_machine_refusal` (`loop_planning.py:253`);
    - `book_hard_rejection` (`agent_loop.py:2348`);
    - `book_agent_fallback_failure` (`agent_loop.py:2035`).
  - **The side doors its docstring names:** `execute_parallel_actions` and the agent-fallback path (#1147).

  All of these either go through the split or are refused on the embedded path.
- **The digest the parity guard compares** covers every side effect, not just NAc, Hippocampus and the followup:
  - `llm_worker.record_outcome`, `context_pool.add_outcome` and the recent outcomes;
  - `nac.credit_goal`, the pending operant action and the cluster reward;
  - the energy cost, the plan outcome and the autonomy audit;
  - the conversation turn and `memory.store_raw`.

  Anything it leaves out, it names.
- **Two blockers to resolve in its plan:**
  - **Tool duration** (A3) is measured on the wall clock and feeds an NAc valence. The digest cannot be bit-identical
    until a host-supplied measured duration replaces it, or it is excluded and named.
  - **`environment.step` runs inside dispatch,** which conflicts with a host that owns the step.
- **A6, generate-vs-apply for memory writes** (moved from tier A by review SF-4). The think path proposes memory
  writes as values, applied only at settle. It is part of B2, under the same digest guard.
  - **Includes:** the D53 `outcome_valence` carrier (`docs/bugs/README.md`) and Overdrive's OD-A16, the refusal-kind
    promotion (an amendment to `tool_failure_credit.md`, after #1200).
  - **OD-A16 changes credit,** so it fires T1-4's re-run trigger walk (`behavioral_graduation_candidates.md`).
  - **The `basis_seq` half is hybrid-family** and stays with B1, after 1.4.0.
- **Why it matters beyond Overdrive:** R4, delayed credit, is the ladder's open gap, and it is unbuilt
  (`minecraft_benchmark.md` Part II; `roadmap_1_4.md` Phase 5, "the first R4 audit is routing"). R4's credit map is
  `deferred/three_factor_credit_assignment.md`, on `NAc._eligibility`. With settle separate, a host can deliver
  outcomes N ticks late on purpose. **Sequence it with R4's routing audit,** and treat any claim that delayed settle
  improves credit as tier C.
- **Gate:**
  - engineering, with parity: the robot path settles immediately after dispatch;
  - the 1.4.0 function gate (`1323.functions-1000`) and touch-it-shrink-it apply (the function is pinned);
  - **depends on #1361** (`experience_clock_decay.md`; corrected per Overdrive correction_1, 2026-11-17):
    - eligibility traces decay ×0.9 per tick (`NAc.decay_eligibility`, run by
      `runtime/agent_loop.py::_loop_bio_tick_maintenance`), not per wall second; **S3a** moves that to
      `advance_experience(dt_us)`;
    - pending events (`NAc.record_event`'s `time.time()`, the 300 s `temporal_window_seconds` buffer) and the anchor
      prune in `decay_eligibility` (`time.time()`) read the host clock; **S3b** moves their age and expiry to the
      experience clock;
    - after S3b, anchor **similarity** still reads the host calendar (`TemporalSignature.now()` in
      `NAc.distribute_reward`'s temporal fallback; "host calendar by design" in #1361). For an embedded host that
      read belongs on the perceived-time port. Asked on #1361, not decided here.
  - credit is normalised to the eligible total, so step rate changes how credit is **shared** among traces and when
    a trace expires into the anchor fallback, not how much a lone trace receives.
- **Line:** with #1150, 1.3.6 or after.

### B3. Priority and deadline on the request value

- **What was proposed:** a priority queue inside the router, in place of `LLMRouter._inference_lock` (created at
  `:186`, acquired at `:1081`).
- **Withdrawn:** `AgentFactory` builds no router, so routers are shared at the caller's choice, and Maxim's own
  simulation shares one. Instead, priority and a deadline become fields of B1's request value, which the host (or
  Maxim's own loop) schedules.
- **Experiment, if a claim is wanted:** competence and response latency as the agent count grows, with
  danger-before-dialogue priority against first-in-first-out.
- **Venue:** part of B1. **Line:** with B1.

## Tier C

Every row needs a prereg, frozen on Maxim's `main` before data, **plus a transfer check in a Maxim-owned harness or on
the robot** before it moves a Maxim claim.

| Id | Question | What an embedded world adds | Behaviour tier |
|---|---|---|---|
| C1 | **One action per call, or an inner tool loop?** Maxim never shows the model a tool result within one call. That was a default; nothing recorded justifies it. Pairs with the B2–B4 baselines (Reflexion first). | Identical tasks and recorded replies: the loop-carried followup against a ReAct-style inner loop | n/a (an architecture choice) |
| C2 | **Modelled or observed latency in the hold's waiting cost?** (Overdrive OD-H2, against `learned_hold_window.md`) | On a robot the two are the same. Embedded, observed latency is a property of the host's GPU and time dial. | learned (the hold's v2) |
| C3 | **Escalation-triggered deliberation** in place of the wall-clock `llm_submit_interval` (`runtime/loop_controller.py:157`, 0.5 s). `experience_clock_decay.md` §(g) keeps that interval on the wall clock on purpose, so this is a design change, not a clock fix. | Latency swept exactly in world time (Overdrive OD-HX2, calibration only) | innate prior, to the hybrid line's escalation rule |
| C4 | **Capability rosters as the robot default** | An adversarial-prompt suite. The security gain holds by construction, so the measure is the competence lost. | invariant (a security rule) |
| C5 | **Floor-holding** (a fifth hybrid split-out plan, not `learned_hold_window`) | Gaps and overlaps timed exactly | innate prior (below) |

**C5's agreed shape** (Overdrive reply §6, with Overdrive's corrected citations, 2026-11-16):
- **Sequence:** the substrate's default act during the hold is a filler, then an account at about 600–700 ms, then a
  yield at about 1 s.
  - The order (account before yield) rests on Stivers & Robinson 2006.
  - The timings rest on Kendrick & Torreira 2015 and Jefferson 1989's standard maximum silence.
- **Fillers:** chosen from predicted delay (Clark & Fox Tree 2002). The delay predictor may be metacognitive: weaker
  feeling-of-knowing goes with slower answers and more *uh* and *um* (Smith & Clark 1993).
- **Thresholds:** parameters (Stivers et al. 2009). They are an honestly learnable prior, so the plan files a
  "make it learned" follow-up with its revive trigger (the plans README rule on behaviour tiers).
- **Content-free:** floor-holding acts are never stored or credited. This is the Exp 63b stall-template contamination
  lesson.
- **Embodied:** gaze aversion on Reachy is motion, through `goto_target`.
- **Status:** it enters as `[engineering]`, innate-prior tier, citing the planning-during-the-other's-turn evidence
  (Bögels et al. 2015, EEG; Castellucci et al. 2022, intracranial). It is earned behaviourally only by its prereg and
  transfer check.
- **Reuse:** §A.1b's held slot, but not its conflict-scaled hold or HA-1's claim. Addressed speech is the case the
  hybrid plan calls degenerate for the arbiter.
- **A gap it names:** `sem_interruptibility.md` covers cancels from the LLM or a danger, not a partner interrupting a
  filler. That needs a cancel path.

## Implementer constraints (every item)

- **Touch-it-shrink-it.** These pinned functions are touched:
  - `execute_and_learn` (B2);
  - `_run_agentic_loop_body` (A4, around `:1571`);
  - `LLMWorker._process_request` (A4/B1, around `:1322`).

  `_run_deliberation_cycles` (B1) is no longer pinned: #1743 took it to 197 lines. An edit that takes it back over
  200 is a new function over the threshold, which needs a `split_from` record or an extraction.

  Each edit extracts what it touches.
- **CC3.** B1's continuation and request/reply values, and A6's write-proposal values, are frozen dataclasses that
  cross a wire. Each declares its forward-compat path in its docstring.
- **Changelog.** Every `src/` PR adds a `changelog.d/` fragment.
- **M89.** Hybrid-family slugs (B1, and A6's `basis_seq` half) wait for the 1.4.0 cut. Fragment slugs are chosen with
  M89 rule (1)'s planned lint in mind.

## Cautions

- **Overfitting Maxim to the game.** An item Overdrive earns is earned in a game. Before it moves a claim, every
  adoption needs a transfer check (the OD-HX5 pattern). Whether an Overdrive world exposed through a Maxim simulation
  bridge is an acceptable venue for tier C is the hybrid line owner's decision.
- **Scope pressure.** A1, A2, A5 and A3's machinery half ride on existing infrastructure. These need a reason written
  in their plan:
  - A3's perceived-time port;
  - A4's embedded entry point and its static ban;
  - B1's continuation;
  - A6's write-proposal values.

## Order

1. **A3 A-pre, A5, #1684,** with **A2 with #1305** (A2's per-agent key after the 1.4.0 cut, a new gate item; owner
   decision 2026-11-19). The prompt guards and the seed substreams come
   first, because replay needs them. **A1** runs in parallel with #984.
2. **A3 A-post** with #1361's slices and 1.3.6.
3. **A4:** the front gate passed (owner, 2026-11-19); built once Overdrive's embeddable-agent plan passes review. Its robot-path defects are filed and fixed
   earlier, on their own.
4. **B2 (#1150) with A6,** after A3 A-post, sequenced with R4's routing audit and the 1.4.0 function gate.
5. **B1 and B3** after the 1.4.0 cut, with hybrid §A.1b and the decomposition slices. A4's parser bypass lands here.
6. **Tier C:** each only through its own prereg and transfer check.

## Filed alongside this list (2026-11-16)

- **#1684:** `x or time.time()` turns a caller's 0.0 into the wall clock (all five sites).
- **#1685:** the semantic embedding store loads with `allow_pickle=True`.
- **#1686:** `ARCHITECTURE.md:182` claims DM-campaign NPCs have real memory via `AgentFactory`;
  `AgentFactory.create_npc_agent` (`runtime/agent_factory.py:936`) has no `src/` caller.
- **#1687:** on Windows, importing `maxim._offline` raises (there is no `socket.AF_UNIX`), which breaks
  `maxim.tools.base`. This is a regression on `main` since 1.3.3. Overdrive's workers depend on the latch.
- **A draft private security advisory** (per `SECURITY.md`, not a public issue): cloud redaction under `strict` ends a
  private section at any header-shaped content line, so the rest ships. Proven end to end, with both `\n` and U+2028.
  The details stay in the advisory until the owner publishes it.

## Not decided here

- Whether A3 is one plan or Overdrive's two (A-pre, A-post), and how it divides with #1361.
- ~~Whether A2's per-agent key widens #1305 before 1.4.0, or follows it.~~ **Decided 2026-11-19: it follows it**,
  after the 1.4.0 cut, as a new release-gate item (A2).
- **Also decided 2026-11-19, from the plan review of Overdrive's v3:** event similarity stays on the host calendar,
  as #1361 S3b has it (OD-A20 noted, not adopted); and a host `save()` does not take `MaximHandle.stop()`'s
  contract: consolidation is its own bounded host call.
- **Decided 2026-11-19, from the delta review of Overdrive's v4 (PR #7):**
  - **Tracked rights, embedded:** danger tracks only, delivered through `step` (hybrid §A.1b.5, `thalamic_relay.md`
    TR6); a non-danger track (a guest's speech) counts only when a plan rung names it.
  - **G2b:** #1361 does not widen its fixture to several traces sharing one receipt; OD-EA12 is Overdrive's test
    against S3b.
  - **Stream derivation:** see A2 (`make_agent_rng` with a `:`-joined stream; `:` forbidden in keys and streams).
  - **Bounded consolidate:** world time decides when; the compute is bounded in deterministic units (which steps,
    how many items), with the wall clock only a logged typed abort. Whether retention scoring moves off
    `time.time()` reopens #1361 Q5 and stays with #1361.
- Whether C5 is drafted now or after the hybrid's v1 evidence.
- The transfer-check venue for tier C.

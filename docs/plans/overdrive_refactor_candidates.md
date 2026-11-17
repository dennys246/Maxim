# Overdrive refactor candidates: what an embedded world asks of Maxim, ranked

> **PROPOSED 2026-11-16: a candidates list, not a plan. No code.** It comes from the design exchange with Maxim
> Overdrive, a game world under the owner's control that embeds Maxim agents. Overdrive's side is on
> `dennys246/Maxim-Overdrive` `phase1/design-v2` (draft PR #4): its drafts (`docs/maxim_upstream/`) and its
> reply to Maxim's answers (`docs/design/reviews/maxim_response_1/reply.md`). Each item below enters Maxim only
> through Maxim's own process (an issue, a plan with a three-lens review, or an experiment with a four-lens design
> review). Nothing here is patched in from Overdrive, and **game runs never go on Maxim's ledger.**
>
> Citations are on `origin/main` **98c6889c** (2026-11-16), each mechanically re-checked against the line's text
> that day. They are `file::symbol` with a line hint; the symbol is the citation, the line will drift.

## Why this list exists

Overdrive needs a Maxim agent that **a host drives**. The host calls it with world time and a frame, and gets back,
as values, what the agent wants to do. The agent never runs its own loop, thread, clock, model call or tool. Today
Maxim's agent owns all of those.

Most of what that takes is **also a Maxim correctness or research-integrity gain**. Once an agent's state, clock,
RNG and ids belong to it, a run can be replayed bit for bit with one variable changed. Paired replays shrink the
confidence intervals the BF2 floors and effect-size pass rules depend on. That gain is the reason the items are
ranked as they are, and it is what makes them Maxim's work rather than a favour to a game.

**Two limits on that gain** (Overdrive reply §5):
- Pairing helps substrate claims. For an LLM arm it ends at the first diverging token.
- A paired analysis estimates d_z, not d. A prereg that uses replay pairing must say which one it gates on.

## The tiers

- **Tier A, do on engineering grounds.** Guarded by tests, including a deletion probe. No experiment is owed,
  because each one either changes no behaviour on a robot or fixes a defect.
- **Tier B, engineering with parity guards.** Each restructures a hot path. It may land only with a guard proving
  the robot path is unchanged: a bit-identical state digest on recorded replays, not a behavioural comparison. A
  behaviour *claim* that the restructure enables waits for its own experiment.
- **Tier C, test, don't refactor.** Design choices Maxim made without evidence, which an embedded world can finally
  test. Each needs a prereg.

Items keep `OD-` ids where Overdrive's drafts already named them. Maxim may reassign ids when it accepts an item.

## Tier A

### A1. One `AgentHome` object, owned per agent

- **What:** route every store an agent writes through one per-agent home object, so a home can never half-resolve
  to `~/.maxim`. `AgentFactory` already scopes Hippocampus, NAc, ATL, SCN, EC, AngularGyrus and the cerebellum.
  About nine families are still process-global:
  - semantic embeddings (`integration/memory_hub.py:246`, `util/semantic_embeddings.npz`);
  - the fear and escalation bridges' thresholds;
  - DN thresholds;
  - the focus and bounds learners;
  - energy;
  - the cost tracker;
  - provenance and lane decisions;
  - the nickname registry.
- **Also on this item:** paths resolved against the working directory, which collide with a host's own `data/`:
  - `utils/cloud_audit.py::CloudAuditLogger` (`:49`, `data/logs/cloud_audit.jsonl`);
  - `utils/response_config.py` (`:65`, `os.getcwd()/data/util`).
- **Front gate:** rides on the existing `utils/paths.py::resolve_run_dir` and the factory's `persistence_dir`. No
  new mechanism.
- **Guard:** two agents in one process never open the same file (a filesystem-trace test), proven by deleting the
  routing.
- **Venue:** a plan, merged with the open issue #984 (shared learned state outside any home), which already names
  the embeddings and escalation thresholds. **Line:** 1.3.x/1.4, on its own clock.

### A2. Per-agent keyed random substreams

- **What:** each agent draws from keyed substreams, `random.Random(stable_hash_64(seed, agent_id, stream_name))`, one
  per consumer (attention, default_network, spatial…). Then adding an NPC or changing one cadence never shifts
  another stream.
  - `utils/seeding.py::make_agent_rng` (`:79`) has no caller. It returns one sequential `random.Random`, so its
    draws depend on cadence, and the substream key is the change.
  - The ~26 process-global `random` draws move onto it.
- **The rule that comes with it:** `seeding.py::seed_all` (`:28`) writes `PYTHONHASHSEED` (`:48`) and reseeds the
  process's global generators. **An embedded path must never call it.** The embedded entry point's import and call
  closure must say so in a test.
- **Front gate:** rides on `utils/seeding.py`, with no new dependency (Philox/numpy not needed).
- **Guard:** two agents, one cadence changed: the other agent's draws are identical. Two processes with different
  `PYTHONHASHSEED`, per the stable-hash invariant.
- **Venue:** an issue-sized PR, or part of A3's plan. **Line:** 1.3.x.

### A3. Two clocks: a machinery clock and a perceived-time port (changed after Overdrive reply §1)

- **What:** the clock seam on the step path is **two seams, not one.**
  - **The machinery clock** drives decay, cadences, eligibility and stamps. On a robot it is the wall clock; an
    embedded agent gets the host's. Known wall-clock reads on the step path include:
    - `decisions/nac.py` event recording (`:1162`, plus `time.time_ns()` in the event id at `:1163`);
    - NAc decay-on-load (`:4324`);
    - the uuid4 ids (36 code lines in 24 files);
    - `time.monotonic` stamps saved across a restart.
  - **The perceived-time port** is the only source of time a prompt may contain. The host supplies it; on a robot it
    is the wall clock, so robots keep seeing the date the web-search guidance needs. Today the prompt renders exact
    time in three places:
    - `agents/prompt_builder.py::build_datetime_section` (`:237`, "Current time"), fed by `datetime.now()` in
      `PromptBuilder.build_prompt` (`:1125`, added at `:1738`);
    - `agents/context_pool.py::_entries_to_text` (`:468`);
    - `context_pool.py::get_context_text` (`:493`), an `HH:MM:SS` stamp on every context entry.
- **Why two:** host time through one seam makes every NPC read exact world time, which Overdrive's time-sense design
  forbids. Wall time left in the prompt breaks determinism. So the prompt-reaching reads must be **classified**, not
  swapped.
- **The guard (owner-accepted pushback, 2026-11-16): tests and a lint, not a type.** Decay arithmetic needs floats,
  so a clock type with no `__str__` is defeated by `f"{t.value}"`. Instead:
  - **A planted-read test:** an advancing fake machinery clock, with a planted wall-clock read that must fail. The
    clock must advance, because a frozen fake clock hides every cadence bug.
  - **A planted-render test:** plant a sentinel time with unusual seconds and pin `TZ` (`localtime` depends on it).
    Fail if its raw, clock-format or date-format rendering appears in any built prompt.
  - **A prompt-module lint:** bans `datetime.now`, `strftime` and `time.localtime` in the modules that reach a prompt
    (`agents/prompt_builder.py`, `agents/context_pool.py`, `agents/llm_fallback.py` at first), except through the
    port.
- **Also in A3:** the five `x or time.time()` sites, filed as **#1684**.
- **Front gate:** rides on the existing injectable `ExperienceClockDriver` (`runtime/experience_time.py::ExperienceClockDriver`). The port is
  the only new surface, and it needs its reason in the plan: no existing seam separates "time the agent uses" from
  "time the agent is told".
- **Venue:** a plan (Overdrive's world-clock drafts A-pre/A-post are its input). **Line:** A-pre on its own clock,
  A-post with 1.3.6. **B1 depends on it.**

### A4. Embedded-path leak closure (new after Overdrive reply §4)

- **What:** the embedded entry point closes every path by which host data or host identity reaches the model, or by
  which the agent reaches the host:
  - **Prefetch off structurally.** `runtime/agent_loop.py` calls `ctrl.run_prefetch(cli_text)` on every input
    (`:1571`). Prefetch scans the working directory (`runtime/prefetch.py`, `get_effective_cwd()` at `:307`) and
    runs `find` subprocesses (`:582`) into the first prompt. **The offline latch does not close this.**
    - It refuses the `find` spawn, and `prefetch.py:595` swallows the refusal at debug level, so it fails silently.
    - The directory scan and file reads are not subprocesses and still reach the prompt.
    - Overdrive's workers also start with the working directory at the empty agent home (Overdrive reply,
      2026-11-16). That is defence in depth, not the guard.
  - **The short-question shortcut is bypassed.**
    - `PromptBuilder.build_prompt` answers time, date, identity and greeting questions without the model
      (`:1102-1137`, via `agents/llm_fallback.py::generate_simple_answer`).
    - It fires on **any** `triggering_input`. The `maxim`/`reachy` wake-word check (`:1108`) only gates the fallback
      to `cli_inputs`.
  - **A cancel never reaches the parser.** The parse-failure path (`agents/llm_worker.py`, `:1322`) builds a canned
    reply from `triggering_input`. A cancelled reply variant (B1) must bypass it.
  - **No logging takeover.** `api.py` replaces the root logger with `force=True` (`:113`).
  - **A static import ban** for the embedded closure. The advisor's three dynamic tests (including
    `tests/integration/test_advisor_gate.py::test_no_effect`, `:1261`) miss lazy imports on paths a test never runs,
    such as `runtime/lane_backends.py::_maybe_inject_cloud_fallback` (`:1249`).
  - **The embedded roster** is capability-based, from Overdrive's embeddable-agent draft (PRs 1–3), with no
    `build_tool_registry`, `data_home()` or `config_loader`, and no env reads. This is Part 1's roster:
    engineering-gated. Making capability rosters the **robot** default is a different item (C4).
- **Guard:** one integration test per path, plus the static ban, each proven by deletion.
- **Venue:** a section of Overdrive's embeddable-agent plan when it comes to Maxim. **Line:** on its own clock.

### A5. Pickle-free embeddings

- `similarity/semantic.py::EmbeddingStore.load` uses `np.load(..., allow_pickle=True)` (`:662`). Filed as **#1685**.
- **Line:** 1.3.x.

### A6. Generate-vs-apply for memory writes (new after Overdrive reply §8)

- **What:** the think path proposes memory writes as values. They are applied only at apply/settle, never during
  generation. A deliberation continuation then never writes to the stores. It carries a basis (hybrid §A.1b.5
  `basis_seq`), so a thought that went stale while the body moved is detectable.
- **Includes:** the D53 `outcome_valence` carrier and Overdrive's OD-A16, the refusal-kind promotion (an amendment to
  `tool_failure_credit.md`, after #1200).
- **Depends on:** B2's dispatch/settle split. It is listed in tier A because, with settle immediate, it changes no
  robot behaviour.
- **Venue:** part of B2's plan.

## Tier B

### B1. The stepped deliberation seam (Overdrive's OD-A10–A15; the hybrid's §A.1b)

- **What:** `think(continuation, input) → request | reply`, an explicit frozen **deliberation continuation**
  (Overdrive reply §2), not a generator and not the whole agent. One continuation carries:
  - the deliberation cycle number, the transcript and the bio-enrichment results;
  - several requests per tick: `runtime/agent_loop.py::_run_deliberation_cycles` (`:521`) makes up to three
    sequential model calls per tick, feeding reasoning back through bio-enrichment;
  - the `next_actions` chain (`llm_worker.py:1356`, `agent_loop.py:1858`), which runs later actions with no further
    call, each one validated by the host when it dispatches.
- **The cut:** at `models/language/router.py::LLMRouter._complete_text` (`:945`) as a request value, and at the parse
  into `LLMProposal`. The gateway sits above the backend, not as a cancellable `LLMBackend`.
- **The reply type in v1** reserves `cancelled(prefix)` and `truncated` variants and a speech field the host can
  read before the JSON closes (Overdrive reply §3). Otherwise barge-in and first-token latency are designed out. The
  token iterator itself can come later.
- **Ride on:** cancel already exists as `utils/cancellation` plus `LLMRouter._complete_text_abandonable` (`:985`).
  The plan says why that does or doesn't suffice before adding anything.
- **Gate (owner-accepted pushback, 2026-11-16):**
  - engineering, with parity guards: a state digest identical under instant and seeded-delayed replies, and parity
    on recorded Exp 49 and Exp 60 replays;
  - lands **after the 1.4.0 cut** (M89 forbids hybrid-family `src/` before it);
  - **one design with hybrid §A.1b;**
  - sequenced with the agent-loop decomposition slices, so `agent_loop.py` is restructured once, under the
    touch-it-shrink-it rule.
- **Line:** 1.5 hybrid line's src phase.

### B2. Split `execute_and_learn` into dispatch and settle

- **What:** `runtime/tool_dispatch.py::execute_and_learn` fuses dispatch with learning (NAc credit, memory writes,
  the followup). Split it so credit lands when the outcome is **measured**, not when the action is dispatched.
- **Scope:** the side doors its own docstring names (`:1290`): `execute_parallel_actions` and the agent-fallback
  path (#1147). Both go through the same split or are refused on the embedded path.
- **Why it matters beyond Overdrive:** R4, delayed credit, is the weakest rung on Maxim's ladder. With settle
  separate, a host can deliver outcomes N ticks late on purpose and sweep N.
- **Gate:** engineering, with parity:
  - the robot path settles immediately after dispatch;
  - guard: a bit-identical digest of NAc, Hippocampus and the followup on recorded replays.
  - **Depends strictly on A3:** NAc's event recording reads the wall clock (`decisions/nac.py:1162`).
  - A claim that delayed settle improves credit accuracy is tier C and needs a prereg.
- **Venue:** a plan, with A6. **Line:** with 1.3.6 or after; it does not touch the arbiter.

### B3. Priority and deadline on the request value (changed after Overdrive reply §5)

- **What was proposed:** a priority queue inside the router in place of `LLMRouter._inference_lock` (created at
  `:186`, acquired at `:1081`).
- **Withdrawn:** `AgentFactory` builds no router, so routers are shared at the caller's choice, and Maxim's own
  simulation shares one. Instead, priority and a deadline become fields of B1's request value, which the host (or
  Maxim's own loop) schedules.
- **Experiment, if a claim is wanted:** competence and response latency as the agent count grows, with
  danger-before-dialogue priority against first-in-first-out.
- **Venue:** part of B1. **Line:** with B1.

## Tier C

| Id | Question | What an embedded world adds | Needs |
|---|---|---|---|
| C1 | **One action per call, or an inner tool loop?** Maxim never shows the model a tool result within one call. That was a default; nothing recorded justifies it. | Identical tasks and identical replays, comparing the loop-carried followup with a ReAct-style inner loop | a prereg; pairs with the B2–B4 baselines (Reflexion first) |
| C2 | **Modelled or observed latency in the hold's waiting cost?** (Overdrive OD-H2, against `learned_hold_window.md`) | On a robot the two are the same. Embedded, observed latency is a property of the host's GPU and time dial. | a prereg; decides v1 or v2 of the hold |
| C3 | **Escalation-triggered deliberation** in place of the wall-clock `llm_submit_interval` (`runtime/loop_controller.py:157`, 0.5 s) | Latency swept exactly in world time (Overdrive OD-HX2, calibration only) | belongs to the hybrid line's escalation rule |
| C4 | **Capability rosters as the robot default** | An adversarial-prompt suite; the security gain holds by construction, so the measure is the competence lost | a prereg; the embedded roster is A4, not this |
| C5 | **Floor-holding** (a fifth hybrid split-out plan, not `learned_hold_window`) | Gaps and overlaps timed exactly | a plan, then a prereg (below) |

**C5's agreed shape** (Overdrive reply §6, with Overdrive's corrected citation, 2026-11-16):
- **Sequence:** the substrate's default act during the hold is a filler, then an account at about 600–700 ms, then
  a yield at about 1 s.
  - The order (an account before a yield) rests on Stivers & Robinson 2006.
  - The timings rest on Kendrick & Torreira 2015 and on Jefferson 1989's standard maximum silence.
- **Fillers:** chosen from predicted delay (Clark & Fox Tree 2002). The delay predictor may be metacognitive: weaker
  feeling-of-knowing goes with slower answers and more *uh* and *um* (Smith & Clark 1993). Thresholds are
  parameters (Stivers et al. 2009).
- **Content-free:** floor-holding acts are never stored or credited. This is the Exp 63b stall-template
  contamination lesson.
- **Embodied:** gaze aversion on Reachy is motion, through `goto_target`.
- **Status:** it enters as `[engineering]`, innate-prior tier, citing the planning-during-the-other's-turn evidence
  (Bögels et al. 2015, EEG; Castellucci et al. 2022, intracranial). It is earned behaviourally only by its prereg.
- **Reuse:** §A.1b's held slot, but not its conflict-scaled hold or HA-1's claim. Addressed speech is the case the
  hybrid plan calls degenerate for the arbiter.
- **A gap it names:** `sem_interruptibility.md` covers cancels from the LLM or a danger, not a partner interrupting a
  filler. That needs a cancel path.

## Cautions

- **Overfitting Maxim to the game.** An item Overdrive earns is earned in a game. Before it moves a Maxim claim, every
  adoption needs a transfer check in a Maxim-owned harness or on the robot (the OD-HX5 pattern). Whether an Overdrive
  world exposed through a Maxim simulation bridge is an acceptable venue is the hybrid line owner's decision.
- **Scope pressure.** A1, A2, A5 and most of A3 ride on existing infrastructure. A3's perceived-time port, B1's
  continuation and A4's static ban each need their reason written in their plan.

## Order

1. **A3, A2, A5, #1684.** Replay needs them, and every later experiment gets cheaper and more trustworthy once
   replay exists. A1 runs in parallel with #984.
2. **A4,** with Overdrive's embeddable-agent plan.
3. **B2 + A6,** after A3, with 1.3.6 or after.
4. **B1 + B3** after the 1.4.0 cut, with hybrid §A.1b and the decomposition slices.
5. **Tier C:** each only through its own prereg.

## Filed alongside this list (2026-11-16)

- **#1684:** `x or time.time()` turns a caller's 0.0 into the wall clock (all five sites).
- **#1685:** the semantic embedding store loads with `allow_pickle=True`.
- **#1686:** `ARCHITECTURE.md:182` claims DM-campaign NPCs have real memory via `AgentFactory`;
  `AgentFactory.create_npc_agent` (`runtime/agent_factory.py:936`) has no `src/` caller.
- **A draft private security advisory** (per `SECURITY.md`, not a public issue): cloud redaction under `strict` ends
  a private section at any header-shaped content line, so the rest ships. Proven end to end with both `\n` and
  U+2028, so U+2028 is not the root cause. The details stay in the advisory until the owner publishes it.

## Not decided here

- Whether A3's plan is one plan or Overdrive's two (A-pre, A-post).
- Whether C5 is drafted now or after the hybrid's v1 evidence.
- The owner's call on the transfer-check venue for tier C.

# Console seam findings (talk / EVENT / PKG era)

**Status:** Mixed — 12 fixed, 6 open (tracked below)
**Severity:** Mixed — several were SILENT, which is why they survived so long
**Affects:** `maxim serve` (console seams), `sim_logger`, `internet_search`, the agent loop's PFC gate, aarch64 packaging
**Discovered:** 2026-07-28/29, mostly by driving the real console from the maxim-pulse side rather than by reading code

Cross-repo context: the console is consumed by **maxim-pulse**, so several of
these were found by a *different* session observing runtime behavior and
reported back. That loop found things code review did not, and the reverse was
also true — two reported symptoms turned out to be instrument artifacts (see
[The recurring pattern](#the-recurring-pattern)).

---

## Open

> Two of these were measured on 2026-07-29. Results are inline; the harnesses
> are `scripts/measure_idle_stream_cadence.py` and
> `scripts/measure_respond_fixation.py`.

### Issue 1: ~2/sec hippocampus+scn record cadence while idle — NOT REPRODUCED

**Status:** measured 2026-07-29 — **did not reproduce**. 60s idle with a live
talk loop produced **zero** hippocampus/scn records; only `pipeline` at
0.10/s. The reported cadence is an **active-turn** phenomenon, not an idle
one. Re-scope before pursuing: measure *during* a turn.
*(The first version of the harness printed "Confirms the report" for 0.10/s of
`pipeline` — a 20x miss from judging against "any non-meta kind" instead of
against the claim. Fixed; the verdict now names the reported kinds.)*

**Severity:** Low-Medium (stream noise; would matter on a Pi's filtered UI)
**Observed:** With a talk loop alive and nothing happening, `hippocampus` and
`scn` records arrive at roughly the loop's 2 Hz.
**Narrowed, not diagnosed:** the emitters are per-OPERATION, not per-tick —
`sim_memory` fires from `hippocampus.py` store + `hippocampus_retrieval.py`
recall, and `sim_scn` from `memory_hub.py` SCN registration. The talk loop runs
`target_hz=2.0`, so *something* performs a memory operation every iteration.
Which one is unconfirmed.
**Next step (cheap, now that `/ws` exists):** attach a client, sit idle, count
by kind, and correlate against loop iterations. **Measure it — do not reason
about it.** Two symptoms in this same batch were instrument artifacts.

### Issue 2: talk's reward signal is degenerate

**Measured 2026-07-29 — the respond FIXATION does not reproduce in console
talk.** Across 12 turns whose probes each had an obvious non-respond action,
`respond` appeared in **zero** action lists (`glob` appeared twice; the rest
produced no tools). The aggregate is robust to the harness's per-turn
attribution confound: stale events could only ADD respond occurrences, never
remove them.

**What that implies for the three hypotheses:** if it were the LLM's own prior
(C) or learned saturation (B), it should appear in console talk too. It does
not. That favours **A — prompt framing / context**: the April observation was a
SIM-ORCHESTRATOR context, where the orchestrator addresses the AUT as though it
were a human. The saturation concern below is still real as a *credit* problem;
it is just not the cause of the fixation.

**Severity:** Medium (undercuts a product claim, not a crash)
**Observed:** `respond` always succeeds, so `record_outcome` books `+1`
essentially every talk turn.
**Root cause:** credit-on-execution rather than credit-on-progress — the same
tool-success-floor saturation `MAXIM_OPERANT_ONLY_CREDIT` exists to counter.
See [deferred/credit_on_progress_not_execution.md](../plans/deferred/credit_on_progress_not_execution.md).
**Consequence to state honestly:** "Adventure teaches Talk" currently rests on
the *adventure* side of the ledger. Talk reads the substrate back (enrichment
pipeline + thought gate, wired in #438); it does not meaningfully teach it.
Documented at the HANDLE seam so `talk()`'s docstring cannot imply otherwise.

### Issue 3: 9 hand-assembled `run_agentic_loop` call sites

**Severity:** Medium (silent capability drift between entry points)
**Observed:** talk shipped missing the enrichment pipeline, thought gate and
`wire_memory_hub` that the sim AUT wires — invisible until someone diffed the
two call sites. Fixed *by hand at one site*.
**Plan + revive trigger:** [deferred/loop_stack_builder.md](../plans/deferred/loop_stack_builder.md).

### Issue 4: a wordless turn is explained, not prevented

**Severity:** Low (deliberate scope boundary)
**Observed:** a turn that runs tools and produces no `respond` now *names* what
happened, but still says nothing substantive.
**Why not fixed:** forcing a closing response changes the agent loop's turn
contract for every entry point, and manufacturing words the agent did not
choose is a bigger decision than a bug fix. Left as an explicit option.

### Issue 5: NAc is never saved, and NAc/hippocampus are never loaded back

**Severity:** HIGH for the cross-session claim
**Observed:** `build_bio_stack` gives no `persistence_path` to NAc, so its state
is never written. Worse, `create_full_agent` auto-loads NAc **then overwrites it**
with the never-loaded `bio.nac` — the same discard applies to hippocampus and ATL.
**Why a partial fix was REVERTED:** adding the save alone makes each session
truncate the last while leaving a plausible populated `nac.json` — "no
persistence" becomes "silently lossy persistence that looks like it works".
**Plan (save + load + a decay-on-load decision + a two-session round-trip test):**
[archive/nac_cross_session_persistence.md](../plans/archive/nac_cross_session_persistence.md).

### Issue 6: the reply path bypasses the executor

**Severity:** Medium (observability; blocks attribution)
**Observed:** across 12 console talk turns, `glob` was recorded in the action
list but `respond` **never was** — while replies demonstrably occurred. Since
`InstrumentedExecutor` records everything routed through `executor.execute()`,
the console's reply path evidently is not that path.
**Why it matters:** tool attribution for talk is incomplete, so "what did this
turn do?" cannot be answered from the action list alone — which is exactly what
the RESPONSE record's new `actions` field is meant to provide.
**Not chased further:** found while measuring something else, and three
successive harness failures argue for reading a raw `/ws` capture by hand
rather than adding a fourth layer of tooling.

---

## Fixed

Grouped by what made each one survive.

### Silent degradation (the failure mode this repo explicitly legislates against)

| # | Issue | Root cause | Fix |
|---|---|---|---|
| 5 | `internet_search` swallowed HTTP failures | `_search_duckduckgo` logged and returned `[]`; `execute` reported `success=True, output=[], "No results found"` — **byte-identical to a search that genuinely matched nothing**. The agent was told the search *worked*, so it could not tell the user otherwise. | Helper returns `(results, failure_reason)`; a transport failure is `success=False` with the cause + an instruction to tell the user, plus a `search_failed` flag. Genuine-empty still succeeds. |
| 6 | `console.ui_dist` config was dead | `console` was declared on `MaximConfig` and writable via `maxim config set`, but **missing from `_parse_config_dict`** — silently dropped. **Masked by the packaged-bundle fallback**: a bare `maxim serve` still served *a* UI, just never the configured one. | Section parsed; plus a generic test that **every** `MaximConfig` section is produced, so the next forgotten one cannot repeat it. |
| 7 | `/ws` died permanently after any campaign | `start_simulation_mode` calls `disable_sim_logging()` at every campaign end, flipping the process-global `_sim_active` — and `sim_log` returned early *before* sink dispatch. Every later talk reply vanished. | Sink dispatch moved **above** the `_sim_active` gate; the console no longer enables sim logging at all. |
| 8 | talk's tool allow-list was inert | `SupervisionPolicy.allowed_tools` is consulted only on the SUPERVISED branch; at AUTONOMOUS only `SafetyConstraints` apply. A list that *read* as enforced wasn't, in a security-relevant path. | Restriction expressed through the mechanism that is actually enforced. |
| 9 | `reachy-mini[gstreamer]` was a no-op | That extra **does not exist**; pip warns and ignores unknown extras. | Dropped from the dependency line. |

### The instrument lied

| # | Issue | Root cause | Fix |
|---|---|---|---|
| 10 | "Deliberation gate never opens (`score=0.00 < 0.00`)" | **The number was fabricated.** The loop computed a real `GateDecision` (score, threshold, reason), kept only `.passed`, then called `sim_pre_deliberation` with hardcoded `0.0/0.0` at all three sites — so refractory, energy-exhausted and empty-working-memory all rendered as a threshold comparison. `ThoughtGate.min_combined_score` is 0.4, so a real threshold of `0.00` was never possible. | Real score/threshold/reason threaded through; rejection shows the REASON. |
| 11 | `/api/diagnose` returned 8 blank rows | `report.sections` is a list of `(group, [CheckResult])` **tuples**, not dataclasses, so `.get("name")` always missed — and **all ~69 checks were discarded**. The pulse side consumed empty rows for weeks. | Flattened to one row per check with status/message; group + fix hint in `extra`. |
| 12 | DM "narrates outside the record stream" | Not the record path — `display_scene` was called **only on the interactive branch**, and the console forces `InteractiveMode.OFF`. In automated mode the prose went straight into the AUT as a percept and was never displayed or recorded, which is why it surfaced only as the truncated BIO-tier `PERCEPT`. | Automated branch emits it too. |
| 13 | RespondTool's receipt shown as the agent's words | The bridge joins respond/speak `result_output`, which is the *text* for the sim's `SimRespondTool` but a delivery **receipt** (`{"delivered": true, ...}`) for the production tool. That dict went out as speech. | `_extract_reply` prefers `tool_args["message"]`; a receipt dict can never leak. |
| 14 | Percepts logged twice | `ConversationalSource` built percepts through `percept_factory` (which logs at construction) and then logged them again on enqueue. *Not* three sites for one percept: `scenario_source` constructs `Percept` directly, so its log is the only one for those. | Duplicate removed; the factory documented as the single logging layer. |

### Contract / process

| # | Issue | Root cause | Fix |
|---|---|---|---|
| 15 | Contract version never moved (twice) | #438 added two endpoints and reshaped the envelope at `0.1.0`; the identity surface then shipped at `0.2.0` — **the same version as the build without it**, i.e. the exact blindness the previous bump was meant to end. Both were ADDITIVE, which is when the stamp goes blind and the change feels harmless. | Bumped to `0.3.0`, and the rule is now **enforced**: `contract_surface.json` records the path + schema surface, and a test fails when the surface moves without the version. Verified it detects the failure, not just that it passes. |
| 16 | PR #440 merged into nothing | Stacked on `feat/console-launcher-seams`; #438 squash-merged that branch to `main` **69 seconds earlier**, so #440's content never reached `main` while GitHub reported it "merged". | Recovered by cherry-pick in #441; stacked PRs now target `main` directly. |

---

## The recurring pattern

**More than half of these were measurement failures, not behavior failures.** The
system was often doing something reasonable while *reporting* something false:

- a fabricated gate score (`0.00 < 0.00`) that was never measured,
- a diagnose view showing 8 blanks while 69 checks existed,
- `pip --platform` certifying "no CUDA" for a target where CUDA installs,
- a CI job passing **because the host happened to have cairo headers**,
- a "regression" in the pulse bundle that was a stale artifact I had inspected.

Two of those were my own analysis, reported confidently before being checked.
The operative rule from [CLAUDE.md](../../CLAUDE.md) — *verify the actuation
before theorizing about the sensor* — generalizes: **verify the instrument
before theorizing about the system.** Concretely: prefer a number you produced
this session over one you read in a log, and check that a test can *detect* the
thing it tests for (each guard above was verified against the pre-fix state
where practical).

## See also

- [display_print_corruption.md](display_print_corruption.md) — the Rich-Live/stdout class of display bug.
- [sim_embodiment_followups.md](sim_embodiment_followups.md) — **Issue 1 there is the `respond` loop**, observed 2026-04-19 and attributed to prompt framing (the orchestrator addressing the AUT as if it were a human). That is a **third** hypothesis for the "why does the agent fixate on `respond`" question, alongside credit-on-execution saturation (Issue 2 above) and the LLM's own prior. Worth reading before investigating: the phenomenon has been seen from three angles and is not yet attributed.
- [reachy_app_maxim_seams.md](../plans/reachy_app_maxim_seams.md) — the seam specs these findings came out of.

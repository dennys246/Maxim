# Acting Coach body-state ablation (proposed Exp 44)

**Status:** PRE-REGISTERED + PREREQUISITES SHIPPED (2026-07-14, branch
`exp/44-body-state-wiring`). Ready to run pending Phase 0 smoke.
**Decision it settles:** whether to wire `embodiment=` into the production
`build_memory_hub` call sites (activating the `body_state` prompt section and
Acting Coach Layers 2+4), bless the auto-sense status quo, or wire the data
without the coach.

## Motivation

The 2026-07-14 embodiment deep-dive found that `format_body_state_for_prompt`
has **no live production caller**: its only caller (`agents/memory_agent.py`)
is gated on `memory_hub.embodiment`, which neither production
`build_memory_hub` call site (`runtime/bio_stack.py`, `runtime/agent_factory.py`)
ever passes. Consequences, verified in the tree:

- `StructuredContext.body_state` is always `""`; the CRITICAL `body_state`
  prompt section never renders.
- **Acting Coach Layers 2+4 (pain anticipation + drive modulation,
  `prompts/acting_coach.py::_compose_pain_anticipation` /
  `_compose_drive_modulation`) early-return on empty input in every
  production path.** The B3 bio-modulation shipped in 0.7 has never actually
  run in production.
- What the LLM sees instead is **auto-sense** (agent_loop section 1.15 →
  `auto_sense_context`): raw `read_all_sensors` output that does NOT tick the
  body and therefore lags one turn behind the drive state.

Exp 37/38/42 results are unaffected — they measured the auto-sense reality
and stand as-is. But wiring `embodiment=` is a **prompt-content behavioral
delta**, so per the interim-contamination and confound-isolation disciplines
it must be measured as an experiment, not slipped in as a bug fix.

## Question

Does giving the LLM a body state read fresh at context-build time — and the
Acting Coach's bio-modulated guidance on top of it — measurably change
embodied behavior (harm avoidance, drive regulation) relative to the
auto-sense-only status quo?

**Scope note (what the manipulation IS):** the wiring changes the READ point
and formatting, not the tick. `format_body_state_for_prompt` only reads
sensors; drive drift still advances exclusively via `Body.evaluate_failures`
(event-driven on tool execution — see the CLAUDE.md embodiment-tick
invariant). In a turn where no body-touching tool fired, arms B/C show the
same values as the previous read, exactly like auto-sense. Do NOT "fix" this
by calling `evaluate_failures()` inside `_enrich_with_embodiment` — that
would publish pain → NAc learning on every prompt build, a bio-side confound
beyond the prompt-content manipulation this experiment isolates.

## Design

**Mode:** LLM-primary only. The Acting Coach is a prompt mechanism;
substrate-primary bypasses prompts entirely and already reads post-tick
drives directly (`_read_drive_states` after the per-proposal tick).

**Model:** Qwen2.5-32B (base). Per the Exp 37/40 Goldilocks result it is the
one local model with demonstrated headroom for substrate-side prompt signals;
a ceiling model (Mistral24B) would mask the delta and a below-zone model
(Qwen14B) can't express it.

**Harness:** the Exp 42 counterbalanced safe-vs-harm warmth arc
(`cradle`-style, infant_humanoid body, safe/harm affordance name swap across
seeds) run in LLM-primary mode. **Harness-prep (Phase 0.5) — SHIPPED** on
`exp/44-body-state-wiring`: the harness's primary contact source was
already the per-session `actions.jsonl` (mode-agnostic; the substrate
telemetry was a best-effort secondary that simply yields no learning-net
fields in LLM-primary — acceptable). `benchmark_exp42_preference.py` now
takes `--aut-mode {substrate-primary,llm-primary}` (default preserves
Exp 42 byte-for-byte) + `--aut-model <profile>`, gates the substrate-only
env injection on the mode, and records `aut_mode` +
`env_body_state_prompt` + `env_coach_body_layers_disabled` per run for arm
provenance. Arm env vars flow through the launching shell
(`env = os.environ.copy()`), so each tmux arm session exports its own.
The harness must still run end-to-end on a throwaway seed before any
registered arm fires. Prompt-section verification uses `MAXIM_LOG_FILE`
JSONL.

**Arms (additive factorial — auto-sense stays ON in all arms so the only
manipulation is what is ADDED):**

| Arm | body_state section | Coach Layers 2+4 | Isolates |
|---|---|---|---|
| A (status quo) | off (unwired) | inert | baseline |
| B | ON (wired) | disabled | fresh-at-build body state as *information* |
| C | ON (wired) | ON | the coach's *guidance* on top of the same information |

10 seeds per arm, counterbalanced safe/harm naming per the Exp 42 protocol.

**Pre-registration divergence (surfaced per the literal-vs-structural
lesson):** the original registration prescribed "thread `embodiment=`
through `build_bio_stack` → `build_memory_hub`". That is structurally
impossible as written — the Embodiment is constructed INSIDE
`build_executor` (Step 3), after the bio-stack/hub already exists (Step 2),
so there is no embodiment value to thread at `build_bio_stack` time without
hoisting its construction. The shipped shape is a late-wire helper instead
(below). If the ablation earns default-on, the permanent fix hoists
Embodiment construction ahead of `build_bio_stack` and makes `embodiment=`
a required keyword with explicit `None` opt-out (the `pain_bus=` shape),
deleting the helper + env var.

**Wiring for arms B/C — SHIPPED** on `exp/44-body-state-wiring`:
`MAXIM_ENABLE_BODY_STATE_PROMPT` (default OFF = arm A byte-identical)
routes `instance.embodiment` into `MemoryHub.embodiment` via
`agent_factory._maybe_wire_body_state` (helper unit-tested in
[tests/unit/test_body_state_wiring.py](../../tests/unit/test_body_state_wiring.py);
parser in `integration/memory_hub.py::body_state_prompt_enabled`).
Arm B's toggle is the new `ActingCoachConfig.body_state_layers` field
(gates Layers 2+4 only; `embodiment_guidance` deliberately NOT repurposed —
it gates Layers 1-4 plus the exploration directives together), driven by
`MAXIM_DISABLE_COACH_BODY_LAYERS` via `acting_coach_config_from_env()` at
the two producer sites (cli.py, orchestrator.py — Wire-A producer-site
pattern; compose stays env-free). Both env vars have autouse conftest
scrubs per the opt-in-env-in-hot-paths lesson. Ablation-arm env vars (not
config.json) follow the house pattern of MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION;
if the ablation earns a default-on wiring, THAT ships config-first.

**Arm recipes (export in the launching shell):**
- Arm A: nothing (status quo)
- Arm B: `MAXIM_ENABLE_BODY_STATE_PROMPT=1 MAXIM_DISABLE_COACH_BODY_LAYERS=1`
- Arm C: `MAXIM_ENABLE_BODY_STATE_PROMPT=1`

## Metrics

**Primary (pre-registered):** exploitation-phase safe-preference and
harm→safe flip, exactly as computed in Exp 42.

**Secondary:** drive-regulation quality — fraction of turns with temperature
inside the comfort band, mean |deviation from set_point|; pain events per
turn; turns-to-first-corrective-action after a breach.

**Mechanism checks (must pass for the arms to count):**
1. The `prompt_sections` DEBUG event (PromptBudgeter emits one per prompt
   build: `name:chars` pairs for every INCLUDED section) shows `body_state`
   present in B/C and absent in A; `acting_coach` section size in C exceeds
   B (Layers 2+4 text). The `body_state_enriched chars=N sha8=…` event
   (memory_agent) confirms enrichment fired and — via the hash — that
   values change across turns whenever body-touching tools fired between
   reads. Both events land in the `MAXIM_LOG_FILE` JSONL (root logger runs
   at DEBUG when it is set).
2. Cross-arm section diff (amended pre-registration, 2026-07-14, BEFORE any
   arm fired): compare `prompt_sections` name+size lists across arms —
   only `body_state` and `acting_coach` may differ (narrator snapshot
   pinned per the narrator-state-confound lesson). This replaces the
   originally registered full prompt byte-diff, which has no capture
   facility today; a prompt-text dump flag can be added later as a deeper
   optional check if section-level diffs look suspicious. Run arms with
   n_ctx headroom — body_state is a CRITICAL-priority section, and under a
   tight budget arms B/C could push a lower-priority section out that arm
   A keeps (a section-diff, which check 2 would catch).
3. Prompt-cache stable prefix is byte-identical across turns in all arms —
   `body_state` is added without `cacheable=True` (verified in
   `prompt_builder.py:1440`), so this should hold; if it doesn't, stop and
   fix before running arms.

## Decision rule (pre-registered, exhaustive)

Two pre-registered comparisons, each classified **+** / **0** / **−** by the
same margin (non-overlapping SE across the 10-seed arms on the primary
metric): **ΔB = B − A** (does the information help?) and **ΔC = C − B**
(does the coaching add anything on top of the same information?). C vs A is
computed as a consistency check but the decision keys on the 3×3 grid,
which partitions the outcome space:

| ΔB (info) | ΔC (coach) | Reading | Action |
|---|---|---|---|
| + | + | information helps, coaching adds more | wire `embodiment=` with coach ON; B3 Layers 2+4 → candidate-Earned (graduation row 11) |
| + | 0 | information sufficient, coaching redundant | wire body_state; Dormant-mark Layers 2+4 citing this experiment |
| + | − | coaching subtracts from good information | wire body_state; disable Layers 2+4; bird's-eye check on WHY the guidance fights (narrator framing?) before any mechanism patch |
| 0 | + | information alone insufficient, coach rescues it | wire with coach ON; graduate — and flag as surprising (guidance shouldn't beat raw data; verify mechanism checks closely) |
| 0 | 0 | neither matters | keep status quo; Dormant-mark the body_state path + Layers 2+4; correct the B3.1 claims in CLAUDE.md |
| 0 | − | coaching actively harms neutral information | keep status quo; Dormant-mark; record the wrong-sign in the results doc |
| − | any | ticked body_state is WORSE than lagged auto-sense | do not wire; bird's-eye divergence check (this outcome implies a confound or a prompt-composition interaction, not a coach question) |

## Phase 0 smoke (run first, ~zero cost)

On the wiring branch (note the arm-C env var — without it the smoke runs
arm A and body_state is correctly absent):

```bash
MAXIM_ENABLE_BODY_STATE_PROMPT=1 MAXIM_LOG_FILE=/tmp/maxim_exp44_smoke.jsonl \
  maxim --sim "warmth smoke" --embodiment bodies/infant_humanoid \
  --interactive false --sim-max-turns 3
```

Then verify in the JSONL:
1. `grep body_state_enriched` — present, and `sha8` differs across turns
   that executed body-touching tools (drift applied at tool time).
2. `grep prompt_sections` — `body_state:` appears in the included list;
   `acting_coach:` present.
3. Re-run WITHOUT the env var — `body_state` absent from `prompt_sections`
   (arm-A control).
4. Stable-prefix stability: `prompt_sections` shows body_state only in the
   dynamic remainder path (it is added without `cacheable=True`); no
   stable-section size changes across turns.

## Cost

Local Qwen32B on the leader; 30 runs × ~40 turns ≈ hours of wall time,
no cloud spend. Runs must not share `~/.maxim/` with concurrent sessions
(persisted-state collision rule).

## Non-goals

- Interoception `Percept` production and Roy-5b naming-event emission stay
  out of scope (tracked in grounded_language_acquisition.md context).
- No substrate-primary arms — the mechanism under test is prompt-side.

## Session log — 2026-07-15 first launch attempt (Mac Mini, BLOCKED)

First attempt to run the arms on the big Mac Mini surfaced a stack of
**operational** blockers before any arm could produce a valid record. None
are experiment-design problems; all are LLM-primary-path / co-located-leader
infrastructure. Recorded so the next attempt starts clean.

**What went wrong, in order:**
1. **Wrong model served, silently.** The Mini's configured profile was
   `r1-distill-qwen-32b`; `--aut-model qwen2.5-32b-instruct` does NOT
   override the auto-spawned singleton on :8100. First runs executed against
   DeepSeek-R1-Distill (a *reasoning* model that can't tool-call) → 0 actions
   in 475s. Fix: `maxim --llm qwen2.5-32b-instruct` then
   `maxim config set llm.profile qwen2.5-32b-instruct` (the C2
   config.json-vs-active-model drift; the singleton check now fails loud on
   the mismatch — that hardening worked).
2. **LLM backend 500s under leader contention.** With the right model, the
   AUT still took 0 embodied actions: `lane-large: down_500`,
   `_llm_unavailable: 4 calls (0% success)`, "No eligible LLM providers".
   Root cause: a `maxim-leader` tmux session was running alongside the
   experiment, both driving the qwen32b llama-cpp server on :8100 → 500s
   under double load. This is the Exp 37 cradle-cascade lesson
   ("don't co-locate the harness and the leader"). Fix: kill the leader
   instance so ONLY the sim touches :8100.
3. **Secondary suspect — prompt overflow.** Server spawned at
   `n_ctx=13312`; the embodied AUT prompt (full sensor dump + affordances +
   entity context + acting coach + body_state) may exceed it, which
   llama-cpp also returns as a 500. If 500s persist with the leader down,
   respawn qwen32b at a larger n_ctx (16k-24k).

**Harness-invocation notes (confirmed working):**
- The harness runs each sub-sim via `subprocess.run(capture_output=True)`,
  so the parent terminal is SILENT for the whole ~20-40 min/seed by design;
  `harness_logs/run.log` only appears at seed end. Silence ≠ hang.
- Direct arc run for live visibility (bypasses the capture):
  `maxim --sim cradle_pref_a --embodiment bodies/infant_humanoid_chilled
  --aut-mode llm-primary --aut-model qwen2.5-32b-instruct --interactive
  false --sim-max-turns 4 2>&1 | tee /tmp/arc.log`. (NOTE: verify
  `cradle_pref_a` loads as an ARC with warmth_alpha/warmth_beta affordances,
  not as a free-text goal — one direct run showed the orchestrator treating
  it as a goal string; needs checking.)

**Validity threats to weigh BEFORE committing ~48h (per the operator's
"brutal honesty" standard):**
- **Tool-calling reliability is a NEW noise source.** Exp 37/38/42 validated
  this harness in *substrate-primary* mode (NAc picks the tool, no LLM
  tool-calling). Exp 44 is the first LLM-primary run. If the AUT only emits a
  valid tool call some fraction of the time, the metric is dominated by "did
  it call anything" not "did it choose the safe tool" — swamping the arm
  effect. MUST confirm the AUT reliably acts (nonzero warmth affordance
  calls) on one clean seed before launching arms.
- **Ceiling risk.** Qwen32B-Instruct has a strong "don't touch what burns
  you" prior; arm A may already be near-ceiling on safe-preference, leaving
  no headroom for B/C on the primary metric. Eyeball arm A's safe_pref on
  the first clean seed; if ~0.95+, the primary metric can't move and the
  interesting signal (if any) will be on the secondary drive-regulation
  metric, likely via B (fresh body_state as information) more than C.
- **Time budget:** ~100-150s/turn on this box × ~24-40 turns × 10 seeds × 3
  arms ≈ 1-2 days. Consider `--sim-max-turns 24` (the Exp 42 validation-spike
  number) and fewer seeds for a first pass.

**Prior (calibrated):** most probability mass on A ≈ B ≈ C within noise
(valid null — settles "wire body_state?" and corrects the B3.1 "shipped"
overclaim); ~20-30% on a clean separation, and if it separates, likely
B ≈ C > A (information helps, coaching redundant) rather than C > B.

**Clean-restart checklist for the next attempt:**
1. `tmux kill-session -t maxim-leader` (no leader co-located).
2. Confirm :8100 serves qwen2.5-32b-instruct (`curl .../v1/models`) and
   `config.json::llm.profile` matches.
3. Respawn qwen32b at n_ctx ≥ 16k if step 3 below still 500s.
4. ONE direct arc run (`--sim-max-turns 4`, streamed) → confirm NO
   `down_500`/`_llm_unavailable` AND nonzero `warmth_*` tool calls.
5. ONE harness throwaway seed → confirm it writes a record with a sane
   `ablation_arm` tag and nonzero contacts. Eyeball arm-A ceiling.
6. Only then launch the arms (leader down, `--resume`, detached tmux).

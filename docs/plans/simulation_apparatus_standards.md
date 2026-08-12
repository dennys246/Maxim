# Simulation apparatus standards — keeping shared sim machinery from silently invalidating graduated claims

**Status:** PROPOSED (2026-08-11), triggered by the Exp 48 heartbeat regression.
**Owns:** the discipline around shared simulation machinery; companion to
[behavioral_graduation_candidates.md](behavioral_graduation_candidates.md) (claims + triggers) and
[docs/experiments/protocols/heartbeat_rerun_runbook.md](../experiments/protocols/heartbeat_rerun_runbook.md) (re-run commands).

## The incident that motivates this

The 1.1 heartbeat walk re-ran [Exp 48](../experiments/48_cradle_mother_seam.md)
(operant orienting, EARNED 2026-07-23) at `f05c63aa`, 12 seeds/arm, provenance
clean. It **failed** the LEARNED gate: taught late-bin directedness rose only
+0.079 against a +0.15 requirement (original: +0.211).

Decomposition of the failure:

| bin | original | re-run | Δ |
|---|---|---|---|
| act1 | 0.51 | 0.60 | +0.09 |
| act2 | 0.82 | 0.74 | −0.08 |
| act3 | 0.85 | 0.75 | −0.10 |
| act4 | **0.90** | **0.75** | **−0.15** |

The teaching signal is fine — `fed_rate` still tracks `directedness`, and the
MOTHER-TAUGHT gate passes with margin (+0.349). **Directional learning is also
fine**: wrong-way turns still halve across acts (18% → 8%), matching the
original's ~10% total miss rate. What stands out is a **~20% "no-move" floor**
(`progress == 0`, i.e. the infant's azimuth is unchanged) that never improves
across acts and mathematically caps directedness at ~0.80. (Initially read as
new; the bisect below found ~16% at the graduation commit too — so it is a
long-standing pathology, not the regression.)

Per-turn log correlation shows the proximate mechanism: on a no-move turn the
infant executes **~31 `turn_left` AND ~31 `turn_right`** — it is thrashing, and
the net displacement cancels. It is not failing to act; it is oscillating.

> **CORRECTION (2026-08-11, after the bisect below ran).** The thrashing was
> initially read as *the regression*. It is not. A 2-seed A/B against the Exp 48
> graduation commit found the same oscillation there — **62.8 turn-actions per
> mother-turn at `45bd1789` vs 59.4 at `f05c63aa`**, both with ~1% left/right
> imbalance. The pathology **predates the graduation it supposedly broke**. It
> was simply never measured, because nothing asserts apparatus health (→ S3).
> The regression itself is **not established**; see the bisect result below.

**Neither the experiment's own code nor its fixtures changed.**
`simulation/cradle_mother.py` is untouched since the Exp 48 commit;
`base_humanoid.yaml`'s turn affordances are still `self_effect: {azimuth: ±0.3}`;
`infant_operant.yaml` is untouched. What changed is **~957 lines across the
shared substrate/motor/credit path** (`nac.py`, `agent_loop.py`,
`tool_dispatch.py`, `tool_bridge.py`, `encoder.py`, `body.py`, `sem.py`),
contributed predominantly by the *Reachy orienting* line of work — #447 live
audio-orient wiring, #460/#461 SEM motor binding, #463 Exp 49 — plus #437
cluster-reward write path, #446 NAc persistence, #476 artifact stamping.

**This was never proven to be the cause** — the bisect below came back
inconclusive, and the mechanism first proposed for it was refuted. The shape of
the problem is a *class* regardless, and stands independent of this incident's
attribution:

> An experiment's apparatus is not just its own fixture and harness. It is every
> line of shared machinery in its causal path. Work aimed at experiment A can
> silently re-calibrate the instrument of already-graduated experiment B, and
> nothing notices until B's next heartbeat — by which point the diff is ~1000
> lines across a dozen PRs and attribution is expensive.

Two further apparatus defects surfaced while investigating, both of which made
the diagnosis harder than it needed to be:

- **The original raw data is gone.** Exp 48's records were written to
  `~/exp48_cradle_mother_seam.jsonl` and never committed, so the re-run cannot be
  compared against the original *distribution* — only against four rounded
  numbers in the doc. Per-seed comparison, variance, and the no-move rate at
  graduation time are all permanently unavailable.
- **The arms are not matched on exposure.** `no_feed` gets exactly 48
  mother-turns every seed (zero variance); `taught` gets 67–94 (mean 85.7). This
  may be by construction (the mother acts on her own turns) but it is
  undocumented, and it means the arms differ in more than the manipulation.

## The bisect — RUN 2026-08-11, result: INCONCLUSIVE on the regression, decisive against the proposed mechanism

Sequential A/B, 2 taught seeds per arm, one harness at a time, durable workdirs:

| | OLD `45bd1789` | NEW `f05c63aa` |
|---|---|---|
| turn-actions / mother-turn | 62.8 (1.0% imbalance) | 59.4 (1.6%) |
| overall no-move | 16.0% | 19.1% |
| early / late | 0.843 / 0.896 | 0.670 / 0.812 |
| **rise** (gate needs ≥ 0.15) | **+0.053 ❌** | **+0.142 ❌** |

Three conclusions, in order of confidence:

1. **The thrashing is not new** (see the correction above). Whatever explains
   the 12-seed drop, it is not "the orienting work introduced oscillation."
2. **At n=2 the OLD commit fails the gate too** — the commit that graduated at
   +0.211 scores +0.053 here. No clean "used to pass, now fails" story survives
   this data.
3. **The gate is fragile** (→ S7). OLD seed 43 opened at `early = 1.000` and
   therefore *cannot rise*, scoring −0.125 by construction. Act1 ranged
   0.45–1.00 across four seeds.

The one signal consistent across both comparisons is the **late bin**: 0.896 vs
0.812 (n=2) and 0.875 vs 0.748 (n=12). Directionally repeatable, not
established.

**Follow-up in flight:** 12 taught seeds at `45bd1789`, matched to the 12-seed
heartbeat run — the only comparison with enough power to settle it.

**Do not run any of this while another fleet holds the LLM server**, and never
two harnesses at once: the first attempt at this bisect ran two harnesses 73
seconds apart against the same `--out` and `--workdir`, producing a duplicate
seed record, a clobbered log, and two LLM consumers on one box (→ S8).

## Standards

Each standard names the specific failure it would have caught in this incident.

### S1 — A PR that touches shared apparatus declares which graduated rows ride on it

The `Re-run on:` triggers in the graduation doc already encode the dependency —
but only in one direction. A row says "re-run me if the NAc credit path
changes"; nothing tells the author *changing* `nac.py` that they just fired that
trigger for three rows.

**Build the reverse index**: a `docs/plans/apparatus_map.md` (or a
generated section) mapping shared code paths → the graduated rows whose
measurements depend on them. A PR touching a mapped path must state, in its
body, which rows it fires triggers for and whether it expects their measurements
to move.

*Would have caught:* #460/#461/#463 declaring "this changes the motor/credit path
Exp 48 measures through" at authoring time, instead of a heartbeat two weeks later.

### S2 — Every graduated sim row carries a fast apparatus canary

Full re-runs are hours; a gross apparatus change is detectable in minutes. Each
graduated sim row gets a **deterministic, cheap, LLM-free-if-possible** canary
that asserts the *instrument*, not the claim: e.g. for Exp 48, "one taught seed,
short arc → no-move rate < 10% and per-turn action count < N". Canaries run in
CI (or pre-merge for mapped paths per S1); the full row still runs at heartbeat.

*Would have caught:* the ~20% no-move floor and the ~60-actions-per-turn
thrashing, at the PR that introduced them.

### S3 — Apparatus pathologies get assertions inside the sim

The thrashing signature — an agent emitting many actions per turn whose net
effect on the measured sensor is ≈ 0 — is a *pathology*, not a result, and
nothing in the stack complains about it. Sims should assert their own health:
net-displacement-vs-action-count, action-count caps actually enforced, arm
exposure matching. A pathological run should be loud, ideally failing the run
rather than quietly producing a number.

*Would have caught:* the thrashing **at Exp 48's own graduation**, not a month
later. This is the standard the bisect most strongly vindicates: ~63 balanced
turn-actions per mother-turn was present when the row was declared EARNED, and
the experiment reported a headline number over an apparatus nobody had looked
at. An apparatus assertion is the difference between "we measured 0.875" and
"we measured 0.875 on an instrument that was oscillating."

*Related existing rule:* the [Exp 45 lesson](../../CLAUDE.md) already says "verify
a test can DETECT the thing it tests for." S3 is that rule applied to the
apparatus rather than the assertion.

### S4 — Graduated experiments commit their raw records

Any experiment cited by a graduation row commits its per-run JSONL under
`docs/experiments/data/`. A row's `Regression guard:` must point at data that
still exists. Re-runs are then comparable at the distribution level (per-seed,
variance, sub-metrics), not just against rounded headline numbers.

*Would have caught:* the inability to compare this re-run's no-move rate against
the graduation-time no-move rate — which is precisely the number that would
settle the diagnosis in one command.

**And it bit again mid-investigation.** The heartbeat run's own diagnostic
workdirs lived in `/tmp`; the machine rebooted at 08:38 on 2026-08-11 and macOS
cleared it, destroying the raw logs behind the 21.0% no-move measurement (the
fade summaries survived only because the harness `--out` pointed at the
worktree). The A/B had to re-derive the current-code side from scratch.
**Corollary:** run workdirs (`--workdir`) belong on durable storage, never
`/tmp` — `/tmp` is for things you are willing to lose on reboot, and experiment
evidence is not one of them.

### S5 — Arms declare their exposure contract

An experiment's pre-registration states what is held constant between arms,
including turn/step budget. Where arms legitimately differ (a mother who acts on
her own turns), the asymmetry is stated and justified, and the harness records
per-arm exposure so drift is visible.

*Would have caught:* the 48-vs-86 turn asymmetry being noticed at design time
rather than during a regression post-mortem.

### S6 — Sim fidelity changes are experiment-visible events

When work makes a simulation *more* faithful to hardware (the honest and
desirable direction — e.g. matching real Reachy turn dynamics), that is still a
change to the measurement apparatus of every experiment sharing it. Such changes
are called out explicitly and the affected rows are re-baselined deliberately:
a graduated claim measured on the old apparatus is not automatically true on the
new one, even when the new one is *better*.

*This is the standard that most directly addresses the hypothesis under test
here*, and it deliberately does not treat "the sim got more realistic" as a free
action. (Note the bisect neither confirmed nor refuted that hypothesis — S6
stands on its own logic, not on this incident's unfinished attribution.)

**First shipped instance — `MAXIM_SUBSTRATE_ACTIONS_PER_TURN` (the Exp 48
thrashing fix, 2026-08-12).** A turn-scoped action budget for the
substrate-primary AUT replaces the emergent stopwatch bound (actions/turn =
narrator wall-clock ÷ 0.5 s, machine-dependent — why Exp 48's magnitudes never
reproduced across hosts). Opt-in; unset = the pre-fix regime byte-identically.
Visibility: orchestrator start log, once-per-window sim_log denial, per-row
`gated` flag in `substrate_telemetry.jsonl`, `apparatus.substrate_actions_per_turn`
in `report.json`, and `substrate_actions_per_turn_env` in the cradle_mother
harness JSONL. **Expected interactions any budgeted run must declare** (two-lens
review, 2026-08-12 — these are apparatus effects; a sweep that books them as
learning is misattributing):

1. **The operant credit target changes.** The mother credits
   `_pending_operant_action` — the LAST action before her tick. Unbounded, that
   is a quasi-random tail sample of the L/R oscillation (≈ a coin flip,
   accidental symmetric noise). Budgeted, it is deterministically the Nth action
   of the window — the policy's own latest choice, making the credit→policy loop
   self-confirming. This is legitimate operant-chamber design (bounded response
   opportunity; nothing leaks INTO the agent — denial precedes
   `propose_via_substrate`, so no encoding, no visit increment, no eligibility),
   but taught-arm magnitudes under a budget are a NEW REGIME by construction,
   not comparable to any pre-fix number including the EARNED 0.90.
2. **The explore/learned balance shifts toward explore at window starts.**
   `decay_exploration_visits` (~30 Hz) keeps running through gate-idle, so
   novelty recovers across each window's idle span, while reinforcement events
   drop from ~60/turn to N/turn against unchanged wall-clock bias decay. Both
   effects matter against the ~0.11 novelty visibility floor. The
   `_ever_selected` explore-FIRST hard gate is unaffected (sticky, one-shot).

### S7 — A gate must be robust to a ceiling in its own baseline

Exp 48's LEARNED gate is `late ≥ 0.65 AND rise ≥ 0.15`. The rise term is
undefined-in-spirit whenever the early bin lands high: OLD seed 43 opened at
`early = 1.000` and scored `rise = −0.125` — not because it failed to learn, but
because it had nowhere to go. Across four bisect seeds act1 ranged 0.45–1.00, so
the gate's verdict at small n is close to a coin flip.

A pre-registered gate must state what happens when the baseline saturates.
Options: exclude ceiling-ed seeds by a pre-declared rule, use a
normalized-improvement statistic (fraction of available headroom captured), or
require the absolute level only and treat rise as a reported-not-gated metric.

**This must be fixed by re-pre-registration, never by post-hoc adjustment.**
Changing a gate after seeing it fail is metric-shopping; the repo has a
three-iteration-metric-pivot lesson for exactly this. The honest sequence is:
pre-register the replacement, then re-run *both* arms under it.

*Would have caught:* Exp 37's Mistral24B ceiling void (Arm A = 1.000, SD =
0.000) is the same failure in a different experiment — the pattern was already
in the evidence base and was not generalized into a standard.

### S8 — One harness at a time, and runs declare their owner

Two harnesses launched 73 seconds apart against the same `--out` and `--workdir`
produced: a duplicate seed record, a garbage record (one act, one turn) from a
run whose workdir was being overwritten mid-flight, and two LLM consumers on one
box — the Exp 37 cradle-cascade condition.

Practical form: a harness takes a lock on its `--out` (or refuses to start when
a live process already targets it); every launch announces itself in a shared
log; and the "is it running?" check is a documented command rather than an
improvised `pgrep` — an empty stdout log means *buffered*, not *dead*, and
mistaking one for the other is how the collision went unnoticed.

*Would have caught:* both the collision and the earlier misdiagnosis of a
healthy fleet as crashed.

## Front-gate scope pressure

Per CLAUDE.md Principle 3 — does this need to be its own mechanism?

| Candidate | Sufficient? |
|---|---|
| `Re-run on:` triggers (graduation doc) | **Half of it** — the dependency data exists but only row→code. S1 is the reverse index over the same data, not a new concept |
| The heartbeat runbook | Catches regressions **at release**, which is what just happened — too late and too expensive to attribute. S2 moves detection to the PR |
| Existing pytest suites | 9000+ tests passed at `f05c63aa` while this regression shipped: unit tests pin mechanisms, not apparatus behavior under a full sim |
| CLAUDE.md invariants | The right home for the *rules* once settled; this doc is the design pass that decides what the rules are |

**Verdict:** mostly rides on existing infrastructure. S1 is an index over data
that already exists; S2/S3 are tests in the existing suites; S4/S5 are
conventions on existing artifacts; S6 is a review-time rule. No new subsystem.

## Sizing

| Standard | Cost |
|---|---|
| S1 apparatus map | ~½ day (one doc + PR-template line) |
| S2 canaries | ~1 day per graduated sim row (5–6 rows) |
| S3 sim self-assertions | ~1 day (thrashing + exposure + action-cap checks) |
| S4 data commitment | ~1 hour + policy |
| S5 exposure contract | folded into the next pre-registration |
| S6 review rule | one CLAUDE.md invariant |
| S7 gate robustness | re-pre-registration of the Exp 48 gate (design, not code) |
| S8 harness lock | ~½ day (lock on `--out` + a documented liveness check) |

## Definition of done

- The Exp 48 regression is attributed or retired: the 12-seed matched arm at
  `45bd1789` settles whether the late-bin drop is real. (The 2-seed bisect has
  run and is inconclusive; it did refute the thrashing-is-new mechanism.)
- The Exp 48 LEARNED gate is re-pre-registered per S7, and both arms re-scored
  under it — before any verdict is recorded on the graduation row.
- S1's apparatus map exists and the graduated rows' `Re-run on:` triggers are
  reachable from the code side.
- At least the Exp 48 and Exp 42 rows have S2 canaries in CI.
- S6 lands as a CLAUDE.md invariant with a regression guard.

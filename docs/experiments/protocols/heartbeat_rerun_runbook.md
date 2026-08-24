# Heartbeat re-run runbook — re-validating Earned behavioral claims per release

**Standing document.** This is the operational companion to
[behavioral_graduation_candidates.md](../../plans/behavioral_graduation_candidates.md):
that doc owns each row's *claim, status, and triggers*; this one owns the
*commands*. Walk it at every minor-version heartbeat (each `1.X → 1.X+1`), and
whenever a row's **Re-run on:** trigger fires. First used for the 1.1 walk
(first pass 2026-08-07, PR #477).

**The contract:** every row currently `Earned` must come back `Maintained`, or
be marked `Stale`/`Broken` — and per the lifecycle rule, `Stale`/`Broken`
**block the release**. A re-run whose code-under-test cannot be established is
not a re-validation (the Exp 42b retraction standard) — the preflight below is
not optional.

---

## Universal preflight (every re-run, no exceptions)

The 2026-07-28 Exp 42b retraction happened because a 40-sub-sim re-validation
ran against the wrong checkout and nothing recorded which code actually
executed. These steps make that class impossible:

```bash
# 1. Run from a WORKTREE pinned at the release-candidate commit — never from a
#    checkout you are concurrently editing (an edited tree makes the stamped
#    git hash a lie, and sims lazy-import mid-run).
git worktree add ../Maxim-heartbeat-<ver> <commit-ish>
cd ../Maxim-heartbeat-<ver>
python -m venv .venv && source .venv/bin/activate && pip install -e . -q
# (a dedicated venv per worktree means PYTHONPATH is never load-bearing;
#  if you must skip the venv, export PYTHONPATH="$PWD/src" — ABSOLUTE, on its
#  OWN line, never chained after a `source` with `&&`)

# 2. Verify the interpreter imports THIS repo (the spawning harnesses call
#    scripts/_provenance.py::assert_repo_interpreter themselves and exit 3 on
#    mismatch — this manual check just fails you faster):
python -c "import maxim, pathlib; print(pathlib.Path(maxim.__file__).resolve())"

# 3. Pin the run config through config.json (single source of truth — a
#    server/budgeter n_ctx drift silently produces down_500 + zero actions):
maxim config set llm.profile <profile>
maxim config set llm.n_ctx <N>
maxim doctor 2>/dev/null | grep -i "n_ctx\|profile"

# 4. Kill stale sims/servers; never co-locate a second LLM consumer with a
#    harness run (Exp 37 cradle-cascade lesson; the leader-local path is safe
#    ONLY via the singleton-reuse guard + harness preflight):
pkill -f "maxim.*sim"; sleep 2
```

**Record on every run:** `executed_git_hash` (the harnesses stamp it per
record; for bare `maxim --sim` runs note `git -C <worktree> rev-parse HEAD`
alongside the session id). **Never edit code in the worktree mid-run.**

**Verdict recording:** after each row, update its status cell in
[behavioral_graduation_candidates.md](../../plans/behavioral_graduation_candidates.md)
(`Maintained (<date>, <evidence pointer>)` or `Stale`/`Broken` + reason) and
append to the current walk's ledger section in that doc. A `Maintained` verdict
needs the gate numbers in the pointer, not just a green feeling.

---

## Chapters

Rows are grouped into **chapters by shared needs** — run a chapter's rows
together so its setup (LLM server, physical rig, budget approval) is paid
once. Ordered cheapest-first.

| Chapter | Needs | Rows |
|---|---|---|
| **Free Chapter** | pytest only | SEM cascade, EC guards |
| **Sim-Short Chapter** | one local LLM server | Exp 09 reflexes, Exp 10 persistence |
| **Fleet Chapter** | local narrator server, hours of wall-clock | Exp 48 operant, Exp 42 discrimination |
| **Big-Model Chapter** (renamed 2026-08-19; was "Cloud Chapter" — a fossil from when the set included real API fires like Exp 38's Sonnet/GPT-4o/DeepSeek; the Exp 37 anchors are LOCAL heavyweights) | heavyweight budget — ½–1 day wall-clock per local 32B model OR real API spend; operator sign-off, never auto-fired | Exp 37 Goldilocks per model |
| **Hardware Chapter** | robot + operator + the physical audio rig (speaker at a bearing) | Exp 45 orient rows |

## Row-by-row commands

### Free Chapter

#### 1. SEM pain → NAc cascade (Tier 1) — seconds

End-to-end cascade pinned by the substrate test suite (real `rusty_sword`
fixture; direct-attribution path):

```bash
python -m pytest tests/substrate/test_sem_pain_cascade.py tests/unit/test_pain_bus.py -q
```

**Gate:** all pass. This row's evidence is the test itself (EARNED de facto);
a red here is `Broken`, not `Stale`.

#### 2. EC pattern completion / separation (Tier 1) — seconds

Heartbeat level = the pinned unit guards (threshold defaults, NAc-override
coupling, Roy-5 H1C boundary tracking):

```bash
python -m pytest tests/unit/test_ec_centroid_drift_fix.py tests/unit/test_roy_5_cosine_localization.py -q
```

**Gate:** all pass. Full behavioral re-derivation (Roy-2c Phase 4 style) is
required only when a **Re-run on:** trigger fires (encoder swap, EC threshold
change) — not for a routine heartbeat.

### Sim-Short Chapter

#### 3. Narrative reflexes, Exp 09 (Tier 3 #9) — one short sim, ~2 min

```bash
MAXIM_SUBSTRATE_PATH=1 \
MAXIM_LOG_FILE=/tmp/heartbeat_reflex.jsonl \
maxim --sim "You are an adventurer in a dark cave. A dragon attacks you repeatedly with claws and fire breath. The dragon roars deafeningly. It slams you against the wall. A freezing wind blows through the cave." \
  --embodiment bodies/base_humanoid --interactive false --sim-max-turns 8
```

**Gate:** reflexes fire on keyword match; habituation/sensitization
trajectories present. Validation greps live in
[09_percept_reflex_poc.md](../09_percept_reflex_poc.md) §Validation checks.
**Scope caveat (from the row):** narrative keyword reflexes only — no halo to
audio/orienting reflexes.

#### 4. Cross-session memory persistence, Exp 10 (Tier 1 row 1) — 3 short sims, ~10-20 min

```bash
# Phase 1 — fresh baseline (note the printed session_id)
MAXIM_LOG_FILE=/tmp/heartbeat_e10_p1.jsonl maxim --sim "escape a dungeon with a sleeping guard" \
  --interactive false --sim-max-turns 8

# Phase 2 — resume (THE gate)
MAXIM_LOG_FILE=/tmp/heartbeat_e10_p2.jsonl maxim --sim "escape a dungeon with a sleeping guard" \
  --interactive false --sim-max-turns 8 --resume-sim <session_id_from_phase_1>

# Phase 3 — negative transfer (dungeon memories must not dominate a garden)
MAXIM_LOG_FILE=/tmp/heartbeat_e10_p3.jsonl maxim --sim "you are in a peaceful garden, enjoy the flowers" \
  --interactive false --sim-max-turns 5 --resume-sim <session_id_from_phase_1>

# Verify: ≥3 memories/turn surfacing in phase-2 enrichment traces
grep "enrichment_trace" /tmp/heartbeat_e10_p2.jsonl | python3 -c "import sys,json; [print(json.dumps({k:v for k,v in json.loads(l).items() if k in ('memories','predictions','goal','hippocampus_size')}, indent=2)) for l in sys.stdin]"
```

**Gate:** ~3 memories/turn on resume (Exp 10 headline); phase 3 shows no
dungeon dominance. Full protocol:
[10_cross_session_enrichment.md](../10_cross_session_enrichment.md).

### Fleet Chapter

#### 5. Operant orienting (Exp 48, cradle_mother) — sub-sim fleet, local LLM

```bash
python scripts/benchmark_cradle_mother.py \
  --arms taught,no_feed --trials 12 --seed-base 42 \
  --model mistral-7b --timeout-s 7200 \
  --out docs/experiments/data/48_heartbeat_<ver>.jsonl
python scripts/analyze_cradle_mother.py --in docs/experiments/data/48_heartbeat_<ver>.jsonl --trials 12
```

**Gates (frozen, in the analyzer):** LEARNED — taught late-bin directedness
≥ 0.65 and rise ≥ 0.15; MOTHER-TAUGHT — taught − no_feed ≥ 0.20. Original:
taught 0.875 vs no_feed 0.448. The `turn_left,turn_right` whitelist is part of
the pinned setup — do not "fix" it for the re-run.

**Declare the action-budget regime explicitly (S6).** Two apparatus regimes
now exist: `MAXIM_SUBSTRATE_ACTIONS_PER_TURN` unset (the original stopwatch
regime — actions/turn tracks narrator wall-clock, machine-dependent) or set
(designed bound; taught-arm magnitudes are a NEW regime, see the S6 first-
instance note in simulation_apparatus_standards.md). Before launching, either
`unset MAXIM_SUBSTRATE_ACTIONS_PER_TURN` or set it deliberately — a value
leaked from the shell flows into every sub-sim via `os.environ.copy()`. The
harness stamps `substrate_actions_per_turn_env` into each JSONL record, so
verify the first record matches your intent before letting the fleet run.

**Timeout sizing (measured, 1.1 walk):** the harness default `--timeout-s
1800` is sized for a fast GPU box. On the Mac the 56-turn taught arm was
only at turn 32 when it hit 1800s — a healthy run, killed by the clock.
Size the timeout from observed pace (~55 s/turn on the Mac → ≥ 5400s;
7200s gives margin), and if the FIRST seed times out, STOP the fleet and
re-size rather than letting every seed burn the full window.

#### 6. Substrate-primary discrimination (Exp 42, Tier 1 #6) — sub-sim fleet

```bash
python scripts/benchmark_exp42_preference.py \
  --arms cradle_pref_a,cradle_pref_b --trials 10 --seed-base 42 \
  --out docs/experiments/data/42_heartbeat_<ver>.jsonl
python scripts/analyze_exp42_preference.py --in docs/experiments/data/42_heartbeat_<ver>.jsonl
```

**Gates:** `safe_pref` ≥ 0.66 both arms; counterbalance identity-flip
positive; C2 harm < safe per source. **Known caveat (42b):** `safe_pref` is
saturated (SD 0.000) — this detects breakage, not moderate regression; a more
sensitive degradation arm is tracked follow-up. The harness calls
`assert_repo_interpreter` and stamps `executed_git_hash` per record — check
them in the output JSONL before reading results.

### Big-Model Chapter (formerly "Cloud Chapter")

#### 7. Cross-session Goldilocks under LLM-AUT (Exp 37, Tier 1 row 1b) — the expensive one

Per the row: re-validation re-fires Exp 37 with the same N=5 design at the
matching git hash across the model set (per-model, pick the models that matter
for the release claim — the Goldilocks framing needs at least one in-zone
model, e.g. Qwen32B or R1-Distill-32B):

```bash
python scripts/benchmark_cross_session.py \
  --trials 5 --model <profile> \
  --out docs/experiments/data/37_heartbeat_<ver>_<model>.jsonl
python scripts/analyze_exp37.py --in docs/experiments/data/37_heartbeat_<ver>_<model>.jsonl
```

**Gates:** in-zone model primary Δ PASS (Qwen32B was +1.43 SD; R1 +2.11 SD);
out-of-zone results are *expected* FAILs (that IS the Goldilocks claim — a
Mistral24B ceiling FAIL is confirmatory, not a regression). Read
[37_cross_model_results.md](../37_cross_model_results.md) before judging.
**Gotchas:** `analyze_exp37.py --out` OVERWRITES its target — point it at a
fresh path; the harness preflight (`assert_subsim_routed_not_local`) exits 4
on the leader-local cascade signature; leader-local runs are safe only
post-2026-06-05 hardening (singleton reuse + preflight).

### Hardware Chapter

#### 8. Real-hardware orient (Exp 45 row) — operator present

Not a routine heartbeat row — it re-runs when its hardware triggers fire (any
motion-command change, shell/acoustic mod, motor service). The H1 campaign
([h1_healthy_hardware_doa_preregistration.md](h1_healthy_hardware_doa_preregistration.md))
is the template. Order is load-bearing:

```bash
# 1. FIRST: verify the mics actually rotate with the body (d(head)/d(body) ≈ +1.0)
python scripts/orient_backbone/yaw_verify.py
# 2. Sensor sweep (≥2 geometries; compare against the healthy-baseline gain 0.578)
python scripts/orient_backbone/doa_sweep.py
# 3. Learned-policy probes (direction + magnitude; population readout arm)
python scripts/orient_backbone/live_3_learn.py --readout population
# 4. Merge/fleet arm — NO hardware needed, run it even on sim-only walks
python scripts/orient_backbone/orient_merge_arm.py
# 5. Delivered-shift block through the PRODUCTION affordance path (needs a continuous
#    speech source dead ahead; 8 reps/side ≈ 4 min) — the _big magnitude evidence
python scripts/orient_backbone/delivered_shift_block.py --reps 8 --log docs/experiments/data/h1_partc_big_block_$(date +%F).jsonl   # new dated file per session; never append to a committed one
```

**Gates:** yaw_verify ≈ +1.0 before anything else is trusted; sweep gain in
the healthy band (H2 branch fires only if outside ≈[0.52, 0.62]); probe 1.00
direction. Version-match SDK/daemon first (skew fails silently on sensing AND
control).

**Which sweep number scores H2 — read this before comparing anything ([L9](../../limits/README.md)):**
score the **full-range** fit, admitted at **R² ≥ 0.99, n ≥ 25, `dry_run == false`**,
grouped by **`run_id`** (never by `--label` — labels get reused across re-runs, and
merging two same-label sweeps is what produced a wrong rejection list once already).
The **central** (`|psi| ≤ 0.5`) gain is NOT the gate statistic: across the committed
corpus the admitted full-range fits span 0.013 while the same curves scored centrally
span 0.086, most of the band width. `doa_sweep.py` now prints `H2: PASS/FIRES/
PROVISIONAL/UNSCOREABLE` itself and stamps the verdict into `sweep_done`, so read that
rather than eyeballing a slope. **Budget ≥ 4 passes** — sign-flips at `|psi| ≳ 1.0`
reject roughly half of them ([L10](../../limits/README.md)), and a single admitted pass
reports `PROVISIONAL` rather than scoring.

### Rows that do NOT re-run on heartbeat

- **Exp 37/38 behavioral-override claim** — settled-dominated (row 2); the
  Exp 38 counter-prior result stands unless its own triggers fire.
- **Affordance concept transfer** — PARTIAL/settled; behavioral half deferred
  to substrate-primary testing (rides row 6's status).
- Anything `Dormant` — resurrection needs a new experiment, not a heartbeat.

---

## Cost/time planning

| Row | Needs | Wall-clock | LLM cost |
|---|---|---|---|
| SEM cascade, EC guards | pytest only | minutes | none |
| Exp 09 reflexes | 1 short sim | ~2 min | pennies (local) |
| Exp 10 persistence | 3 short sims | ~10-20 min | pennies (local) |
| Exp 48 operant | 24 sub-sims | ~1-3 h | local narrator |
| Exp 42 discrimination | 40 sub-sims | ~2-4 h | local narrator |
| Exp 37 Goldilocks | 3 arms × 5 trials × models | ~½-1 day/model | the real cost — budget it |
| Exp 45 hardware | robot + operator | ~½ day | none |

Run order that works: fire the pytest rows inline; start Exp 48 + Exp 42
fleets in the background (they're subprocess harnesses — they tolerate
sequential queueing better than parallel contention on one LLM server);
do Exp 09/10 between fleets; schedule Exp 37 models deliberately.

## Maintaining this runbook

- When a row **graduates**, add its section here in the same PR that flips the
  row to Earned — the commands are part of earning the tag (same discipline as
  the **Re-run on:** field).
- When a harness flag changes, update the command here in the same PR (a
  runbook with dead commands is the "instrument that asserts instead of
  measuring" failure — commands in this file are load-bearing).
- Keep gates OUT of this file except as reminders — the analyzer scripts and
  the graduation doc own the frozen constants; if a number here disagrees with
  the analyzer, the analyzer wins.

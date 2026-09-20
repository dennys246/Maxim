# Exp 62 two-pool plumbing — ARCHITECTURE lens (pre-merge code review)

**Target:** uncommitted working tree, branch `feat/exp62-two-pool-plumbing`
(`scripts/survival_world/setup_world.py`, `scripts/survival_world/exp60_water_check.py`,
`tests/unit/test_exp60_water_classroom.py`; +267/−16).
**Read against:** [`exp62_pressure_interoception_prereg.md`](../../exp62_pressure_interoception_prereg.md)
§Apparatus + §Build order, [`behavioral_graduation_candidates.md`](../../../plans/behavioral_graduation_candidates.md)
rows for Exp 60/61, [`docs/agents/persistence-config.md`](../../../agents/persistence-config.md).
**Date:** 2026-09-19. **Verdict: 2 DO-NOT-SHIP, 8 SHOULD-FIX, 5 NIT.**

Lints and the touched test file are green (`ruff check` clean, `ruff format --check` clean,
`pytest tests/unit/test_exp60_water_classroom.py` 40 passed). The craft is good: the guard is pure
and unit-tested, the offline end-to-end builder test (`TestTwoPoolBuildOffline`) is exactly what the
`run-rig-scripts-offline-first` rule asks for, and the NOT-CHECKED-is-never-CLEAR idiom is carried
into the new guard's return value. The findings below are about what the plumbing does **not**
carry, and about two places where the defect the PR exists to fix is still live one flag over.

---

## 1. Completeness against the prereg §Apparatus

The prereg's §Apparatus paragraph names nine things. Scored:

| # | Prereg asks for | Status |
|---|---|---|
| 1 | `--shore-y` on the builder | **DELIVERED** (`setup_world.py:803-808`; `water_classroom_geometry` already took `shore_y=`, so this is a wiring line, correctly so) |
| 2 | `--anchor-file` on the builder | **DELIVERED** (`setup_world.py:809-817`, threaded at `:635`, `:729`, `:737`) |
| 3 | `--anchor-file` on the check | **DELIVERED for the READ and the STAMP** (`exp60_water_check.py:289-297`, `:358`, `:634`) — but see DNS-1: the check's **`--out`** was not given the same treatment, and that is the same bug |
| 4 | pool-vs-pool shell clearance guard | **DELIVERED as code, VACUOUS on the rig** — see DNS-2 |
| 5 | **world spawn stamped into both records** | **PARTIAL / the field only.** `world_spawn` is recorded (`setup_world.py:583`) but only from operator-typed `--spawn-x/y/z`; nothing acquires it, and pool 1's existing record can never gain it without a rebuild (DNS-2's chain). See SF-3 |
| 6 | flee anchor recorded per pool | **DELIVERED** (`setup_world.py:581-582`). Note it is x/z only and stacked pools share x/z, so both records carry the *same* pair by construction — correct, but not pool-discriminating (NIT-5) |
| 7 | both apparatus checks and both `measured` blocks frozen | **DEFERRED** (operator/harness step; the plumbing that makes it *possible* is item 3, and DNS-1 makes it easy to do wrong) |
| 8 | light and time gated equal at pool 2's shore and floor | **SILENTLY MISSING.** `exp60_water_check.py:16` states outright that `light_level`/`time_of_day` are *recorded, not gated*; they are captured at **W1 shore only** (`:465-466`) and at no floor stage. Nothing in this diff adds the comparison. See SF-8 |
| 9 | pool 2's own Exp 60 gate (ii) run live before any row | **DEFERRED, correctly** (that is `l11_geometry_probe --anchor-file`, which already takes the flag; the record's `probe_situations`/`probe_settle`/`probe_rescue` keys carry over unchanged) |

Plus one item from the prereg's **§Build order step 2**, which explicitly scopes the *plumbing* PR as
"builder + check + anchor records + spawn stamp + flee anchor + **`WaterTrial` per pool**":

| 10 | `WaterTrial` per pool (two geoms, ONE instrument attach) | **NOT PRESENT.** `water_trial.py` is untouched. See SF-9 |

### The world-spawn question, plainly

**The prereg does not require a bridge protocol change, and you should not make one.**

- What the prereg says is "the bridge emits `bot.spawnPoint` once; nothing exposes it today." That is
  accurate about the *named* field (`scripts/minecraft_bridge/index.js:93` computes `spawn` locally
  and throws it away; `:359` prints a snapshot that does not contain it).
- But the snapshot **already carries enough to recover it exactly**: `offset_x`/`offset_z` are
  *signed* spawn-relative offsets (`index.js:100-104`) clamped at ±128, and `distance_from_spawn` is
  the 3D distance capped at 128. With the bot's position `p` from the same snapshot,
  `spawn.x = p.x − offset_x`, `spawn.z = p.z − offset_z`, and `|dy| = sqrt(d² − dx² − dz²)`; the sign
  of `dy` is resolved by one second reading at a different altitude — and Exp 62 has a pool at a
  different altitude by construction. No new key, no protocol change.
- Adding `spawn_x/y/z` to the state payload **would** be a "Minecraft bridge protocol change" by the
  letter, and that phrase appears verbatim in the **Re-run on:** list of *four* ledger rows — Exp 56
  (line 194), Exp 57 (195), Exp 60 (196), and Exp 61 (197, via "every Exp 60 trigger"). Four
  discharges to write, on an EARNED headline, for a number you can already compute. Do not.

So: recording `world_spawn` from `--spawn-x/y/z` is **honest** (null means not given, and W1 still
gates the sensed value live at `exp60_water_check.py:455-467`) but it is **not sufficient** for the
prereg's "stamped into **both** records", because pool 1's record cannot be given the field at all
without the destructive chain in DNS-2. The discharge is SF-3 + DNS-2's back-fill mode, not a bridge
change.

---

## 2. Does it disturb a FROZEN apparatus?

**Additivity — verified.** `water_anchor_record` gains five keys (`setup_world.py:579-583`) and
removes none; every pre-existing key keeps its spelling and value. `report["apparatus"]` gains
`pool_id` with a `.get(..., "pool1")` default (`exp60_water_check.py:369`), so an old record still
reads. Every consumer I checked reads by key, never by exhaustive shape:
`exp60_run.py:352` (`geom` + `measured`), `r3_run.py:487,543` (`geom`, `anchor_measured =
geom.get("measured")`), `r3_pilot.py:78`, `exp61_run.py:678`, `l11_geometry_probe.py:254,307`,
`water_trial.py:254-265,362,398-399` (`shore`, `submerged`, `surface_y`, `deaths_objective`).
None of them would see a new key.

**Default paths — verified unchanged.** `--shore-y` defaults to `WATER_SHORE_Y`, `--anchor-file` to
`None → WATER_ANCHOR_FILE`, `--pool-id` to `"pool1"`, and `_stamp_measured`'s third parameter
defaults to `ANCHOR_FILE`. An Exp 60 rebuild typed exactly as before produces byte-identical
geometry and RCON commands, and writes the same file, plus five keys.

**One behavioural change on the default path:** the sibling loop at `setup_world.py:666` now runs on
an Exp 60 rebuild too, so once pool 2's record exists in `~/.maxim/`, a default rebuild prints an
extra clearance line and *can return 4*. That is intended, but it means the Exp 60 rebuild path is no
longer input-closed over its own flags — it now depends on the contents of a directory. See SF-6/SF-7.

**Ledger re-run triggers: none fire.** Exp 60's row (line 196) names `NAc.record_cluster_fear` /
`anticipatory_threat_need` / Wire-4 allowlist or θ, `recommend_action` drive-activation floor,
`SensorEncoder`/EC world-modality, `minecraft_player` sensor-range, `escape_water`/bridge
water-handling, `run_agentic_loop` idle-gate/autonomy, Minecraft bridge protocol, minor-version
heartbeat. Exp 61's row is "every Exp 60 trigger" plus hivemind paths. **This diff touches none of
those surfaces.** What it *does* touch is Exp 60's **Regression guard** list — `setup_world.py
water_classroom` and `exp60_water_check.py` are both named there. A guard change is not a trigger; it
is an obligation to show the guard still guards, which the additivity + default-path verification
above discharges, and which the new offline builder test strengthens.

**But there is a live, non-ledger freeze hazard, and it is the heart of DNS-2.** `r3_run.py:322`
refuses any row whose `anchor_measured` differs from the frozen gauntlet's, and `anchor_measured` is
`geom.get("measured")` (`:543`). So **re-running `exp60_water_check` on pool 1 for any reason
invalidates the entire frozen R3 gauntlet.** Any plan that reaches "just rebuild pool 1 so its record
gets the new fields" walks straight into it: a rebuild replaces the record wholesale
(`setup_world.py:730-737`), dropping `measured`; `exp60_run.py:356` and `r3_run.py` then refuse for
want of it; re-running the check restamps `measured` with new timings; R3 is dead. **State this in
the PR body.** It is the single most expensive mistake available on this rig right now.

---

## 3. Placement and ownership

**`pools_disjoint` is in the right module.** `setup_world.py:537` sits immediately after
`exp58_clearance` (`:500`) and `spawn_clearance` (`:520`), shares their exact shape
(`(ok, measurement)`, pure, no I/O, `None` → NOT CHECKED), and is unit-tested beside them. No
complaint.

**The sibling-scan-by-glob does not belong in the command body, and not because of layering.** The
two existing guards each read *one explicitly named* input — `EXP58_ANCHOR_FILE`, or the operator's
`--spawn-x/y/z`. The new one discovers its input by `anchor_file.parent.glob("*water_classroom*.json")`
(`:666`), which makes the build's outcome a function of ambient directory contents. Three concrete
consequences:

- **Identity is by path, not by pool.** `sibling.resolve() == anchor_file.resolve()` (`:667`) is the
  only self-exclusion. A backup copy made exactly the way an operator protects a `measured` block —
  `cp ~/.maxim/exp60_water_classroom.json ~/.maxim/exp60_water_classroom.bak.json` — matches the glob,
  carries pool 1's own shell, and **refuses pool 1's own idempotent rebuild** with "shells intersect".
- **A record in another directory is invisible**, and the build says nothing about it.
- **Zero siblings prints nothing at all.** Answering the question asked directly: **no, silence is not
  acceptable, and it is not reported.** The loop body is the only thing that prints; a fresh directory
  produces a build whose output is indistinguishable from a build that checked and cleared. That is
  the shape this repo calls out by name — "a mechanism that does not run looks exactly like one that
  ran and found nothing" — and the fix is already idiomatic in this very function: the other two
  clearances each print a NOTE on the not-checked branch (`:689`, `:706-709`) *and* record `null`
  (`:734-735`). The pool clearance does neither.

**Recommended ownership change (this is the one design move I would make):** replace ambient
discovery with an explicit `--stack-on PATH`, which simultaneously fixes SF-3 below.

```python
    w.add_argument(
        "--stack-on",
        default=None,
        help=(
            "the record of the pool this one stacks over/under (Exp 62 pool 2 passes pool 1's). "
            "Its x/z become THIS build's anchor — stacked pools MUST share x/z (prereg D1: the "
            "spawn bound, the Exp 58 clearance and the no-chunk-load-window property are all "
            "inherited through the shared column) — and its shell is the disjointness input."
        ),
    )
```

with the glob demoted to a belt that only *reports*. Keep `pools_disjoint` exactly as it is; it is the
right function, it is just being fed by the wrong thing.

---

## 4. The record as a contract

Consumers are `exp60_run.py`, `exp60_water_check.py`, `l11_geometry_probe.py`, `r3_pilot.py`,
`r3_run.py`, and `water_trial.py` via the `geom` dict. Against
[`persistence-config.md`](../../../agents/persistence-config.md):

- **`_format_version` is present but hand-written** (`setup_world.py:574`, a literal `"1.0"`), not
  `with_format_version(payload)`. **Pre-existing.**
- **Both writers are plain, non-atomic `write_text`** — `setup_world.py:737` and
  `exp60_water_check.py:280` — not `atomic_write_json`. **Pre-existing**, and invisible to
  `scripts/lint_atomic_io_ratchet.py`, which counts hand-rolled *renames*; this writes with no rename
  at all, which is strictly worse than the shape the ratchet polices.
- **No reader calls `check_format_version`.** Pre-existing.
- **Does the diff make it worse?** Not in kind — it adds no new writer and no new rename. It makes it
  worse in *exposure*, three ways: (i) `exp60_water_check._stamp_measured` (`:278-280`) is a
  read-modify-write whose target is now a CLI argument, so a mistyped path truncates a file the
  harnesses refuse without; (ii) the record is now the carrier of the **safety** truth (`shell`) that
  a later build's guard reads, so a lost write vacates a guard rather than merely losing data; (iii)
  the test docstring at `tests/unit/test_exp60_water_classroom.py:352-353` records that this exact
  file was destroyed **twice during the R3 campaign**. A file with a twice-demonstrated destruction
  history and a `git checkout --` recovery story should not be written by truncation.
- **Scoped fix (both call sites, ~4 lines, no new mechanism):**

  ```python
  from maxim.utils.atomic_io import atomic_write_json
  from maxim.utils.format_version import with_format_version
  ...
  atomic_write_json(str(anchor_file), with_format_version(record))
  ```

  Precedent in `scripts/`: `_paper_server.py:96-100`, `exp_d8_read_mutation.py:526-529`,
  `orient_backbone/gate6_merged_gauntlet.py:404-422`. Note `with_format_version` is fail-loud on a
  stale conflicting value, so drop the literal `"_format_version": "1.0"` at `:574` when adopting it.
  `atomic_write_json` defaults to `indent=2`; the only observable difference from today's output is
  the trailing newline, which nothing parses.
- **Version bump:** the brief says bump when the *shape* changes, and it did. Because every added key
  is optional-with-default at every reader, I grade the bump a **NIT**, not a blocker — but say in the
  PR body that you considered it and why you did not.

---

## 5. Front-gate scope pressure

Mostly clean. `pools_disjoint` rides the existing guard pattern rather than inventing a placement
subsystem. `--shore-y` rides a parameter the pure geometry function already had. `--anchor-file` rides
the path that was already a module constant.

Two things do introduce new concepts, and both should ride something existing:

- **`--pool-id` is a second identity scheme** alongside the anchor path (cf. the repo's standing
  `two-identity-schemes` caution). It is cheap and the arms table genuinely needs a row label, so keep
  it — but make it *derived and checked* rather than a free-typed default: default it to
  `anchor_file.stem` (or require it whenever `--anchor-file` is given) and refuse a sibling that
  claims the same `pool_id` with a different shell. As shipped, `--anchor-file pool2.json` without
  `--pool-id` records `"pool1"` (`:820`) — two files, both claiming to be pool 1, and the check
  faithfully stamps `pool_id: "pool1"` into pool 2's apparatus report (`exp60_water_check.py:369`).
- **The sibling glob is a new discovery mechanism.** §3 above: replace it with the explicit
  `--stack-on`, which is not a new mechanism but the same explicit-input pattern the other two guards
  already use.

Nothing else here wants to be its own mechanism.

---

## Findings

### DO-NOT-SHIP

**DNS-1 — The check's `--out` was not made pool-aware, and the builder now *prints the wrong command*
after a pool-2 build.**
`scripts/survival_world/exp60_water_check.py:288`:
```python
    ap.add_argument("--out", default="docs/experiments/data/exp60_water_apparatus.json")
```
`scripts/survival_world/setup_world.py:751-753` (unchanged by this diff, and now wrong):
```
            f"next (clean tree, PYTHONPATH=$PWD/src): python scripts/survival_world/exp60_water_check.py "
            f"--rcon-password '<pw>' --username {args.username} --write-experiment-results"
```
After building pool 2 the builder tells the operator to run the check with **no `--anchor-file` and no
`--out`**. Following its own instruction: the check reads **pool 1's** record (`:358`), runs the
cycles against pool 1's coordinates, stamps pool-2-session timings into **pool 1's** `measured`
(`:634`), and — because `--write-experiment-results` is in the printed line and the tree is clean by
the printed precondition — overwrites `docs/experiments/data/exp60_water_apparatus.json`, which
`exp60_run.py:129` reads as the EARNED experiment's committed apparatus evidence. (`evidence_out_paths`
D27 redirects to a temp dir *without* the flag; the printed command has the flag.) This is precisely
the defect the PR's own docstrings describe — "one fixed path means checking pool 2 destroys pool 1's
measured block" — left live in the sibling flag and in the operator-facing text.

Fix (both halves):
```python
    # exp60_water_check.py, after args are parsed
    if args.anchor_file and not args.out_explicit:
        print(
            "usage: --anchor-file names a NON-default pool, so --out must be given explicitly — "
            "the default (docs/experiments/data/exp60_water_apparatus.json) is Exp 60's committed "
            "apparatus record and must not be written by another pool's check."
        )
        return 2
```
(implement `out_explicit` by `default=None` + `args.out or DEFAULT_OUT`), and in
`setup_world.py:751-753`:
```python
            f"next (clean tree, PYTHONPATH=$PWD/src): python scripts/survival_world/exp60_water_check.py "
            f"--rcon-password '<pw>' --username {args.username} "
            f"--anchor-file {anchor_file} --out docs/experiments/data/{args.pool_id}_water_apparatus.json "
            f"--write-experiment-results"
```

**DNS-2 — The pool-clearance guard is vacuous in the exact case it was written for, and cannot be
armed by any safe rig operation.**
`scripts/survival_world/setup_world.py:548-550`:
```python
    other = other_record.get("shell")
    if not other:
        return True, "other record predates the shell stamp — NOT CHECKED"
```
Pool 1's record on big-mac-mini was written by the pre-diff `water_anchor_record`, which had no
`shell` key. So the *first and only* build this guard exists to police — pool 2 stacked over pool 1 —
takes the NOT-CHECKED branch, prints `pool clearance vs exp60_water_classroom.json: other record
predates the shell stamp — NOT CHECKED`, and builds anyway. The label is honest; the guard is inert.

And the obvious remedy is a trap: regenerating pool 1's record by rebuilding replaces it wholesale
(`:730-737`), dropping `measured` → `exp60_run.py:356` and `r3_run.py` refuse → re-running the check
restamps `measured` → `r3_run.py:322` (`anchor_measured` vs the frozen gauntlet) refuses **every R3
row**. There is no safe path from the shipped code to an armed guard.

Fix — derive the shell for legacy records (the derivation is already this file's own idiom at
`:656-657`, `ax = shore[0] + 1`), and ship a non-destructive back-fill:
```python
def record_shell(rec: dict) -> tuple | None:
    """This pool's shell, from the record's own geometry — legacy records predate the stamp.

    A record written before the Exp 62 stamp carries shore/surface_y/depth, and the shell is a
    pure function of those (water_classroom_geometry), so the guard never has to read NOT CHECKED
    on a pool that is fully described. Returns None only when the record is not a water pool.
    """
    if rec.get("shell"):
        return tuple(rec["shell"])
    try:
        sx, _sy, sz = rec["shore"]
        return water_classroom_geometry(
            int(sx) + 1, int(sz), depth=int(rec["depth"]), shore_y=int(rec["surface_y"])
        )["shell"]
    except (KeyError, TypeError, ValueError):
        return None
```
with `pools_disjoint` calling `record_shell(other_record)` and keeping the NOT-CHECKED return only for
the genuinely underdetermined case. Add a `water_classroom --backfill` mode that merges the five new
descriptive keys into an existing record **without touching `measured`** — that is also the only way
the prereg's "world spawn stamped into **both** records" can be satisfied for pool 1 (see SF-3).
Add a test that a legacy record (the `TestTwoPoolPlumbing.test_guard_is_honest_when_it_cannot_check`
fixture, which strips `shell`) now *refuses* an overlapping build rather than passing it.

### SHOULD-FIX

**SF-1 — Nothing carries prereg D1's "same x/z", and getting it wrong is silent.**
`setup_world.py:653-658`: with a fresh `--anchor-file` and no `--anchor-x/--anchor-z`, `recorded` is
`None`, so the anchor falls through to `bot_pos(rcon, args.username)` — wherever the bot happens to
stand. A pool 2 at a different x/z still passes W1, still passes gate (ii), and still produces rows;
what it silently changes is the thing the experiment measures (the replay's cross-pool cosine was
computed for the shared column, and the "inherits pool 1's Exp 58 clearance / no chunk-load window on
the teleport" properties in §Apparatus are all consequences of the shared x/z). Per the repo's
"push silent-no-op invariants into types" rule, this belongs in the signature: `--stack-on PATH`
(§3) takes the anchor x/z from the named record and makes a mis-stacked pool unrepresentable.

**SF-2 — `/spawnpoint` is player-global and last-build-wins; nothing re-asserts it per pool.**
`setup_world.py:470` issues `spawnpoint {username} {sx} {sy} {sz}` at build time only. After pool 2 is
built, a death at pool 1 respawns the bot at **pool 2's** shore. Exp 60 measured 0 deaths, but that is
an outcome, not a guarantee (`water_trial.check_death_cap` exists precisely because deaths happen),
and R3 produces deaths deliberately. Expose the per-pool world-conditions block so the harness can
re-assert it — e.g. split `water_classroom_commands` into `_build_cmds` + a
`water_classroom_conditions(geom, username)` returning the `doMobSpawning` / `spawnpoint` /
`scoreboard` triple — and have the harness call it at each pool's setup. Record the hazard in the
record's comment block too.

**SF-3 — `world_spawn` has a field but no acquisition, and pool 1 can never get one.**
`setup_world.py:583` records it only from `--spawn-x/y/z`, and §1 above shows the prereg's live
pre-check row 1 ("world spawn coordinates") has no tool. Do **not** add it to the bridge payload
(four ledger rows). Instead add a pure helper beside the check's other readers and print it:
```python
def world_spawn_from_snapshot(vm: dict[str, Any]) -> tuple[float, float, float] | None:
    """WORLD spawn from the bridge's own signed offsets — no protocol change.

    offset_x/offset_z are spawn-relative and SIGNED (bridge index.js), so spawn x/z fall straight
    out of the sampled position; the y term comes from the 3D distance_from_spawn. Returns None at
    either clamp (|offset| 128 or distance 128): a capped reading cannot be inverted.
    """
```
Print it at the end of the check as a diagnostic so the operator can pass it to the builder, and
back-fill it into pool 1's record via DNS-2's `--backfill` — **never** by re-running the gated check
on pool 1 (that restamps `measured` and kills the R3 gauntlet).

**SF-4 — Both record writes are non-atomic; adopt `atomic_write_json` + `with_format_version`.**
`setup_world.py:737`, `exp60_water_check.py:280`. Pre-existing, not made worse in kind, but §4 gives
the three reasons the exposure grew and the exact four-line fix with in-repo precedent. Given this
file's twice-demonstrated destruction history, this is the cheapest durable win in the review.

**SF-5 — The pool-clearance outcome is not recorded, breaking the record's own NOT-CHECKED idiom.**
The record carries `exp58_clearance_blocks` and `spawn_clearance_blocks` with `None` = NOT CHECKED
(`:584-585`, docstring `:570`). The third guard records nothing. Add:
```python
        # NOT CHECKED is null, exactly as the other two clearances: a build that saw no sibling
        # record must not read later as a build that checked and cleared.
        "pool_clearance": pool_clearance,   # {"<pool_id or filename>": "<why>"} | None
```
so a reader of the record — and the harness's provenance row — can tell a cleared build from an
unchecked one.

**SF-6 — Zero siblings prints nothing.** `setup_world.py:666-676`: the only output is inside the loop.
Add after it:
```python
        if not checked:
            print(
                f"NOTE: no sibling water-classroom record in {anchor_file.parent} — pool clearance "
                f"NOT CHECKED (recorded null). Pass --stack-on if this pool stacks on another."
            )
```

**SF-7 — Pool identity is by path; a backup copy refuses the pool's own rebuild.**
`setup_world.py:667` self-excludes by resolved path only, and `--pool-id` defaults to `"pool1"`
(`:820`). Two corrections, one rule: *identity is the `pool_id`, not the path.* Skip a sibling whose
`pool_id` matches this build's **and** whose shell is identical (a copy of this very pool — print it
as skipped); **refuse** a sibling whose `pool_id` matches but whose shell differs (two pools claiming
one identity). And default `--pool-id` to `anchor_file.stem`, or require it with `--anchor-file`, so
pool 2 cannot silently record itself as pool 1.

**SF-8 — The prereg's light/time gate at pool 2's shore and floor does not exist.**
`exp60_water_check.py:16` says `light_level`/`time_of_day` are recorded not gated; they appear only in
the **W1 shore** block (`:465-466`) and nowhere in `w2_dive` (`evaluate_dive`). §Apparatus requires
them *gated equal* at pool 2's shore **and floor**, and names the failure mode ("a stale-light read at
a freshly filled box"). Cheapest honest discharge without touching Exp 60's frozen path: the floor
reading is available in `l11_geometry_probe`'s raw per-situation samples, so make the pre-check
compare pool 1's and pool 2's recorded shore `light_level`/`time_of_day` (from the two apparatus
JSONs) and the two probes' floor samples, and write it up as a pre-check row. If you want it as a
refusal, put it behind a new opt-in flag on the check rather than in the default path, so an Exp 60
re-run is untouched. Either way — say which, in the PR body; right now it is simply absent.

**SF-9 — `WaterTrial` per pool is in the prereg's step-2 scope and is not here.**
`water_trial.py` already takes `geom` per instance (`:238`), so two pools is two instances — but
`attach_instruments` (`:290-326`) rebinds `self.aut.executor.execute` with no re-entrancy guard, so a
second `WaterTrial` on the same AUT wraps the *first trial's spy*: every call recorded twice, and
`detach_instruments` (`:328-334`) restores in the wrong order unless detached LIFO. The prereg calls
this out by name ("ONE instrument attach — the executor spy must not double-wrap"). Land the guard
here, where it is four lines and testable offline, rather than letting the harness PR discover it:
```python
    def attach_instruments(self) -> None:
        if getattr(self.aut, "_water_trial_attached", None) is not None:
            raise InstrumentError(
                "a WaterTrial is already instrumenting this AUT — two pools share ONE attach; "
                "the second wrap would double-count every executor call (Exp 62 §Apparatus)"
            )
        self.aut._water_trial_attached = self
```
with the matching clear in `detach_instruments`, plus a test that two trials on one AUT refuse.

### NIT

**NIT-1 — the "vertical gap" message is wrong for horizontally separated pools.**
`setup_world.py:556`: `gap = max(ay0, by0) - min(ay1, by1)` is only a gap on the y axis. Two pools
side by side at the same `shore_y` (legal via `--anchor-x/--anchor-z`) report a *negative* "vertical
gap". Compute all three axis gaps and report the separating one:
`gaps = {"x": max(ax0,bx0)-min(ax1,bx1), "y": ..., "z": ...}`, then
`axis, g = max(gaps.items(), key=lambda kv: kv[1])` → `f"{axis} gap {g} block(s) between shells"`.

**NIT-2 — no named minimum gap.** The two sibling guards each carry a named constant with a sensor
rationale (`WATER_MIN_DIST_FROM_EXP58`, `WATER_MAX_DIST_FROM_SPAWN`); this one accepts any
non-intersection, so a 1-block gap passes. That is almost certainly fine (the shells are ≥2 blocks of
stone on each side, and the replay says the cross-pool cosine is insensitive to altitude at weight
0.09) — but make it a decision, not an accident: `WATER_MIN_POOL_SHELL_GAP = 1  # structural only; the
cross-pool cosine is altitude-insensitive by the Exp 62 replay`.

**NIT-3 — `--shore-y` is unvalidated** (`:803-808`) while `--depth` has a checked band
(`WATER_MIN_DEPTH/WATER_MAX_DEPTH`, raised as `ValueError` → exit 2 at `:661-663`). An out-of-world
`--shore-y` fails as an opaque RCON `fill` error *after* `forceload add` has already been issued
(`water_classroom_commands` order), leaving chunks force-loaded. Add a band check in
`water_classroom_geometry` so it is a usage error before any command is sent.

**NIT-4 — `_stamp_measured`'s new parameter should be keyword-only.**
`exp60_water_check.py:270`: a third positional `Path` next to `out_path: Path` is exactly the pair a
future edit transposes. `def _stamp_measured(report, out_path, *, anchor_file=ANCHOR_FILE)`; the one
call site (`:634`) and the one test become `anchor_file=`.

**NIT-5 — `flee_x`/`flee_z` are identical for stacked pools by construction** (x/z only, and the pools
share x/z), and the bridge's goal is `GoalNearXZ` (`index.js:225`), so a `flee` at either pool's shore
resolves instantly. Correct and matching Exp 60's "free success by construction" framing in
`water_trial.check_flee_anchor` — but a harness reader could mistake the recorded pair for something
pool-discriminating. One clause in the record's comment block settles it.

**NIT-6 — `_format_version` not bumped and not routed through `with_format_version`** (§4). Additive
keys with defaults at every reader make this defensible; say so in the PR body rather than leaving it
unaddressed.

---

## 6. What the harness PR (and the live pre-check) now inherits

Concretely, before the prereg's four-row live pre-check can run:

1. **Back-fill pool 1's record in place** — `shell`, `pool_id`, `flee_x/z`, `world_spawn` — **without
   re-running `exp60_water_check` on pool 1**. Re-running it restamps `measured`, and `r3_run.py:322`
   refuses every R3 row whose `anchor_measured` differs from the frozen gauntlet's. Needs DNS-2's
   `--backfill` (or the `record_shell` derivation, which removes the need to write pool 1 at all for
   the guard's sake — but not for `world_spawn`).
2. **Acquire the world spawn** by deriving it from the bridge snapshot's `offset_x`/`offset_z` +
   position + `distance_from_spawn` (SF-3). Do **not** add it to the bridge payload — four ledger rows
   name "Minecraft bridge protocol change" as a re-run trigger.
3. **Build pool 2 at pool 1's x/z**, floor y 95 (prereg D1), depth 5. Until SF-1 lands this is
   operator discipline: `--anchor-x <pool1 shore_x + 1> --anchor-z <pool1 shore_z> --shore-y 95
   --anchor-file ~/.maxim/exp62_pool2_water_classroom.json --pool-id pool2`. Verify the printed
   clearance line actually compared shells rather than printing NOT CHECKED.
4. **Run the check against pool 2 with BOTH flags**: `--anchor-file <pool2> --out
   docs/experiments/data/exp62_pool2_water_apparatus.json`. Omitting `--out` overwrites Exp 60's
   committed apparatus record (DNS-1).
5. **Re-assert `/spawnpoint` for whichever pool is about to run** — it is player-global and the last
   build won (SF-2). Also note `exp60_deaths` is a single shared scoreboard objective
   (`setup_world.py:592`), so deaths are **not** attributable per pool; the harness must bracket
   `deaths()` per phase, not per pool.
6. **Restart the bridge with `--flee_x/--flee_z`** at the shared shore x/z (both pools; see NIT-5 and
   the standing `bridge-restart-after-sensor-change` note), and re-verify the roster on raw
   `client.latest_state()` keys.
7. **Run pool 2's own gate (ii)**: `l11_geometry_probe --anchor-file <pool2>` — which requires pool 2's
   `measured.t_damage_onset_min_s`, i.e. step 4 must succeed first. Strict ordering: build → check →
   probe.
8. **Gate light and time equal at pool 2's shore and floor** (SF-8) — no code does this today; decide
   whether it is a pre-check assertion on the two apparatus JSONs + the two probes' samples, or a new
   opt-in refusal on the check. Do not put it in the check's default path.
9. **One instrument attach across two `WaterTrial`s** (SF-9) — land the re-entrancy guard before the
   harness composes two pools onto one AUT.
10. **`exp62_run.py` must not import `exp60_run.ANCHOR_FILE`** (`exp60_run.py:128`, imported by
    `exp61_run.py:109`, `r3_run.py:52`, `r3_pilot.py:46`): it needs two records. Either take two
    `--anchor-file` arguments or thread a small `(pool_id, geom)` pair list.
11. **The REPLAY row** the prereg's gates require comes from
    `docs/experiments/data/exp62_cross_pool_replay.py` (present), which refuses unless it reproduces
    the live gate-(ii) cosine 0.7874 — run it against the *new* pool-2 geometry, not the recommended
    one, once pool 2 is actually built.

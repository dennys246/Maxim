# Exp 62 two-pool plumbing — EXECUTOR lens (pre-merge code review)

**Scope.** The uncommitted diff on `feat/exp62-two-pool-plumbing`: `scripts/survival_world/setup_world.py`
(`--shore-y` / `--anchor-file` / `--pool-id` on `water_classroom`, `pools_disjoint()`, new record fields),
`scripts/survival_world/exp60_water_check.py` (`--anchor-file`, `_stamp_measured(anchor_file)`),
`tests/unit/test_exp60_water_classroom.py` (`TestTwoPoolPlumbing`, `TestTwoPoolBuildOffline`).
Lens question: **does it RUN correctly in every branch?** Everything below was executed, not inferred;
no live sim, no Minecraft server, no hardware. No repository file was edited by this review.

**Verdict: 1 DO-NOT-SHIP, 7 SHOULD-FIX, 5 NIT.** The core is sound — the flags reach the geometry on
every path, the disjointness arithmetic is correct at its boundaries (it reproduces the environment
lens's own `shore_y >= 54` prediction exactly), the new record fields match what `wiring-v3.md` asked
for, and four of the five behaviours I mutated are genuinely guarded by the new tests. What is not yet
right is everything AROUND the one path that was parameterised: a second fixed path in the check
(`--out`), the printed next-step command, the player spawnpoint, the rebuild's silent height drift, and
a guard test that can destroy the artifact it guards.

---

## What I ran

| # | Command / probe | Result |
|---|---|---|
| 1 | `PYTHONPATH="$PWD/src" python -m pytest tests/unit/test_exp60_water_classroom.py tests/unit/test_water_trial_lethal.py -q` | **45 passed in 29.71s** |
| 2 | `ruff check` + `ruff format --check` on the three changed files | clean; "3 files already formatted" |
| 3 | Mutation runs: the new tests against five individually reverted behaviours (copy of the tree in the scratchpad; the new tests unmodified) | M1 `--shore-y` dropped → 1 FAIL; M2 record → fixed `WATER_ANCHOR_FILE` → 2 FAIL; M3 sibling guard deleted → 1 FAIL; M4 self-skip deleted → **40 passed (gap)**; M5 stamp → fixed `ANCHOR_FILE` → 1 FAIL |
| 4 | Drove the REAL working-tree `SW.main(["water_classroom", ...])` against a fake RCON in a temp dir: two-pool build, rebuilds with/without `--shore-y`, records in split directories, `.bak`/`_backup.json` siblings, symlinked siblings, corrupt sibling, missing parent dir | see F1/F6/F7/F10 below |
| 5 | `pools_disjoint` by hand over shore_y ∈ {40,46,47,48,49,53,54,55,95}, self, nested, x-only, z-only, `None`, `{}`, `{"shell": None}`, `{"shell": [1,2,3]}` | see §3 |
| 6 | Grepped every consumer of the anchor record (`exp60_run.py`, `exp60_water_check.py`, `l11_geometry_probe.py`, `water_trial.py`, `r3_run.py`, `r3_pilot.py`, `exp61_run.py`) for positional / fixed-key-set reads | none — all key-based `geom["..."]` / `.get(...)`; the new fields are additive-safe |

---

## DO-NOT-SHIP

### D1 — `TestTwoPoolBuildOffline` is not hermetic: when the plumbing regresses, the test writes the operator's REAL `~/.maxim` records
`tests/unit/test_exp60_water_classroom.py:437` (`_build`) monkeypatches `SW.RconControl`, `SW.EXP58_ANCHOR_FILE`
and `time.sleep`, but **not** `SW.WATER_ANCHOR_FILE` (`setup_world.py:406`) and not `chk.ANCHOR_FILE`
(`exp60_water_check.py:86`). Both still resolve to `Path.home()/".maxim"/"exp60_water_classroom.json"`.

*Evidence (this actually happened during the review).* Running the new tests against mutation M2
(`anchor_file.write_text(...)` at `setup_world.py:737` reverted to `WATER_ANCHOR_FILE.write_text(...)`)
created `~/.maxim/exp60_water_classroom.json` on this box — a full pool-2 record at the canonical
pool-1 path. The test failed, as designed, *and* wrote the file. (This laptop had no such record —
dir mtime confirms creation, not modification — and I moved the stray file to the scratchpad:
`scratchpad/mutate/STRAY_written_by_mutation_run_exp60_water_classroom.json`.)

On the rig the same run is worse than a lost record. With mutation M5 (`_stamp_measured` reverted to
`ANCHOR_FILE`), `test_check_stamps_measured_into_the_pool_it_was_given` **reads the real record, injects
the test's synthetic `measured` block — `{"t_damage_onset_min_s": 16.2, ...}`, values close enough to the
real 16.07 to look right — and writes it back.** On this box it merely warned (`FileNotFoundError`);
on big-mac-mini it would have succeeded silently. That is precisely the failure the PR exists to prevent,
delivered by the PR's own guard test, with no `git` copy to restore from (`~/.maxim` is not tracked).

Yes, it only fires when the source is wrong — that is what tests are for. The fix is two lines:

```python
# in TestTwoPoolBuildOffline._build, beside the EXP58_ANCHOR_FILE patch:
monkeypatch.setattr(SW, "WATER_ANCHOR_FILE", tmp_path / "unused_default_pool1.json")
# and in TestTwoPoolPlumbing.test_check_stamps_measured_into_the_pool_it_was_given:
monkeypatch.setattr(chk, "ANCHOR_FILE", tmp_path / "unused_default_pool1.json")
```

Better still, add a class-scoped `autouse` fixture that patches both for the whole file, so no future
test in it can reach `~/.maxim` either.

---

## SHOULD-FIX

### S1 — a rebuild silently discards the record's `surface_y`: pool 2 rebuilds at y 40
`setup_world.py:804-808` gives `--shore-y` `default=WATER_SHORE_Y`, so `args.shore_y` is never "unset";
`setup_world.py:653-655` reuses the recorded anchor's **x/z only** (`ax, _sy, az = recorded["shore"]`) and
line 660 then passes `shore_y=args.shore_y`. The record's own `surface_y` is never consulted.

*Measured*, driving the real builder twice into the same anchor file:

* both records in one directory (the rig layout): rebuilding pool 2 without `--shore-y 95` → **rc 4**,
  message `REFUSING: this build would collide with the pool recorded at .../exp60_water_classroom.json —
  shells intersect: this (-6, 32, -4, 9, 45, 4) vs recorded (-6, 32, -4, 9, 45, 4)` and the advice
  *"move --shore-y further from the other pool's"* — misdirection: the right action is "pass `--shore-y 95`,
  the height your own record already names". The documented idempotent-rebuild workflow
  (`_water_classroom` docstring, "Idempotent: rebuilds in place at the recorded anchor") is broken for
  pool 2 and the error blames the wrong thing.
* pool 1's record in another directory / renamed: **rc 0** — pool 2 is rebuilt at y 40, its record rewritten
  with `surface_y: 40`, the y-95 pool left standing in the world with no record, and a fresh stone shell
  punched at y 32–45 where pool 1 lives. Silent, and the world and the record now disagree.

*Fix:* `--shore-y` `default=None`; after `recorded` is read,
```python
shore_y = args.shore_y if args.shore_y is not None else int((recorded or {}).get("surface_y", WATER_SHORE_Y))
```
and when `args.shore_y is not None and recorded and args.shore_y != recorded.get("surface_y")`, print a loud
RELOCATING line (or refuse without an explicit `--relocate`) — a rebuild that moves a pool invalidates its
`measured` block, its Exp 58/spawn clearances and any frozen record that cites it.

### S2 — the check's `--out` is still a fixed pool-1 path
`exp60_water_check.py:288` (`--out` default `docs/experiments/data/exp60_water_apparatus.json`) was not
parameterised alongside `--anchor-file`. Running the check on pool 2 with `--write-experiment-results`
and a clean tree overwrites the FROZEN Exp 60 apparatus record that `exp60_run.py:129`
(`APPARATUS_RECORD`) and `r3_run.py` read. D27 governance (`_provenance.evidence_out_paths`) redirects
to a temp dir *without* the flag, and the overwrite is git-recoverable — but the campaign's gated runs use
the flag, and "the pool is a parameter" is exactly the claim of this PR. This is the composition half of
the same defect: `_stamp_measured` was parameterised, the evidence write was not.
*Fix:* when `--anchor-file` is given (or the record's `pool_id != "pool1"`) and `--out` was left at its
default, either derive it (`exp60_water_apparatus_<pool_id>.json`) or `ap.error(...)` and demand an explicit
`--out`. A one-line refusal is enough; silence is not.

### S3 — the "next" command printed after a pool-2 build points the check at pool 1
`setup_world.py:750-752` prints `... exp60_water_check.py --rcon-password '<pw>' --username {args.username}
--write-experiment-results` with no `--anchor-file`. An operator who follows the script after building pool 2
runs the check against pool 1's record — it dives at pool 1's coordinates and re-stamps pool 1's `measured`,
and nothing in the output says the wrong pool was measured.
*Fix:* interpolate the flags actually in force:
`f"... --username {args.username} --anchor-file {anchor_file} --out docs/experiments/data/exp60_water_apparatus_{args.pool_id}.json --write-experiment-results"`.

### S4 — `/spawnpoint` is global per player: the LAST pool built owns every respawn
`water_classroom_commands` (`setup_world.py:470`) issues `spawnpoint {username} {sx} {sy} {sz}` on every
build, and nothing in `water_trial.py` / `exp60_run.py` / `exp61_run.py` ever re-sets it (grep: the only
`spawnpoint` call sites in `scripts/survival_world/` are the two builders). After building pool 2, a death
in pool 1 respawns the bot at pool 2's shore — a silent cross-pool teleport in the middle of a trial, and
exactly the sort of apparatus fact Exp 62's freeze is supposed to name.
*Fix (cheap, this PR):* print it — `f"NOTE: /spawnpoint for {args.username} now points at THIS pool ({pool_id}); the last pool built owns every respawn"` — and stamp `"spawnpoint": [sx, sy, sz]` into the record.
*Fix (harness PR):* `WaterTrial` sets the spawnpoint for its own pool at trial start.

### S5 — the sibling scan says nothing when it scanned nothing, and its contract is undocumented
`setup_world.py:666` globs `*water_classroom*.json` in **the anchor file's own directory** only. When that
yields zero siblings the loop prints nothing at all — indistinguishable from "checked, clear". Measured:
building pool 2 into `<tmp>/elsewhere/` produced **no guard output whatsoever**, while pool 1 sat one
directory up. Compare the Exp 58 guard at `setup_world.py:689`, which prints an explicit
`NOTE: ... NOT CHECKED (recorded null)` — "missing is the signal" is already this file's convention.
Two further unstated requirements fall out of the glob: the other pool's record must (a) live in the same
directory and (b) carry `water_classroom` in its filename. Neither the `--anchor-file` help
(`setup_world.py:809-817`) nor the prereg says so; `~/.maxim/exp62_pool2.json` would silently disable the
guard in both directions.
*Fix:* count the siblings and print `NOTE: no sibling pool record in {anchor_file.parent} — pool-vs-pool
clearance NOT CHECKED` when zero; state the naming/directory contract in the `--anchor-file` help; optionally
glob `*.json` and treat any record carrying both `shell` and `submerged` as a pool (which also picks up a
differently-named second pool).

### S6 — a backup copy of a pool's own record permanently refuses that pool's rebuild, with wrong advice
`*.json.bak` is correctly missed by the glob (verified), but `exp60_water_classroom_backup.json` is matched
(verified), and the self-skip at `setup_world.py:667` compares **resolved paths only**. Measured: with
`exp62_pool2_water_classroom_backup.json` beside it, rebuilding pool 2 at its own height → **rc 4**,
`shells intersect: this (-6, 87, -4, 9, 100, 4) vs recorded (-6, 87, -4, 9, 100, 4)` plus
*"move --shore-y further from the other pool's"* — advice that, followed, moves a correct pool. Copying an
anchor record aside is this campaign's standard recovery habit, so this will happen.
*Fix:* in the refusal message, name the case: `"...if that file is a BACKUP of this same pool (identical
shell, pool_id {x}), move it out of {anchor_file.parent} — the guard compares by path, not by content."`
(An identical-shell + identical-`pool_id` sibling could also be reported as a copy rather than a collision,
but do not skip it silently: two real pools built with the same `--pool-id` would then slip the guard.)

### S7 — `_stamp_measured`'s anchor is a DEFAULTED positional, against the repo's own silent-no-op rule
`exp60_water_check.py:270`: `def _stamp_measured(report, out_path, anchor_file: Path = ANCHOR_FILE)`.
CLAUDE.md, "Push silent-no-op invariants into types, not helpers": the canonical form is a required
keyword-only parameter (`build_executor(pain_bus=...)`), so that forgetting is a `TypeError` rather than a
silent write to pool 1. This function's default IS the bug the PR was written to kill — it should not survive
as a default. Same argument, weaker, for `water_anchor_record(..., pool_id: str = "pool1")`
(`setup_world.py:568`): the default silently mislabels a pool-2 record.
*Fix:* `def _stamp_measured(report: dict[str, Any], out_path: Path, *, anchor_file: Path) -> None:` and pass
it at the single call site (`exp60_water_check.py:634`, already passes it positionally).

### S8 — test gaps proven by mutation
* **The self-skip at `setup_world.py:667` is unguarded.** Deleting it leaves the whole file **green
  (40 passed)**, yet it is load-bearing: driving the real builder twice into one anchor file gives rc 0 with
  the skip and **rc 4** ("shells intersect: ... vs [itself]") without it. Add a rebuild arm:
  build once, then build again into the same `--anchor-file` with the same `--shore-y`, assert rc 0.
* **No failure-reply arm.** The `FakeRcon` (`tests/unit/test_exp60_water_classroom.py:412`) answers only
  success shapes, so the builder's two refusal branches — the explicit
  `"no blocks were filled"` test (`setup_world.py:718`) and the verification gate
  `"passed" not in resp` (`setup_world.py:728`) — are never executed by any test in the repo
  (grep: no `No blocks were filled` / `Test failed` anywhere under `tests/`). Add two arms
  (one fill answering `"No blocks were filled"`, one `execute` answering `"Test failed"`), each asserting
  rc 4 **and** that no record file was written.
* `pools_disjoint(geom, {})` (a truncated-but-valid sibling) is untested and currently reports
  "no other pool recorded" (see N3).

*On the FakeRcon's fidelity otherwise:* the success shapes are close enough to 1.20.4
(`"Successfully filled 9 block(s)"` vs the vanilla `"Successfully filled N blocks"` — the gate only tests
for `"filled"`; `"Test passed"` vs `"Test passed, count: 1"` — the gate tests for `"passed"`), and
`gamerule`/`forceload`/`spawnpoint` replies are faithful enough for the token gate. The gap is not the
success shapes, it is that **no failure shape is ever produced**. Pre-existing and out of scope, but worth
recording: replies like `"No player was found"` (spawnpoint, bot offline) and a forceload range error contain
none of `unknown`/`expected`/`incorrect` and are NOT a `fill`, so they pass the gate silently.

---

## NIT

### N1 — the reported vertical gap is one block too large, and meaningless when the separation is horizontal
`setup_world.py:556`: `gap = max(ay0, by0) - min(ay1, by1)`. For shells at y 32–45 and y 46–59 (shore_y 54)
the function reports `vertical gap 1 block(s)` although **zero** blocks separate them — the boundary is right
(shore_y 53 → shared block → refuse; 54 → adjacent → accept, matching `rationale/exp62-.../environment.md`
SF-3's "above it shore_y ≥ 54"), only the number is off by one. And when the pools are separated in x or z
while their y bands overlap, the same line prints `vertical gap -13 block(s)` — a negative "gap" that reads
like a bug in the build log.
*Fix:* `gap = max(ay0, by0) - min(ay1, by1) - 1` and report the axis: if the y bands overlap, say
`"separated horizontally (x/z), y bands overlap"` instead of a vertical gap.

### N2 — inclusive-bounds overlap test: verified correct
`setup_world.py:553`. `min(a1,b1) >= max(a0,b0)` per axis is the right test for inclusive cuboids.
Checked by hand: identical shells → intersect; nested `(-2,34,-2,2,40,2)` inside `(-6,32,-4,9,45,4)` →
intersect; shared wall (y1 == other y0) → intersect; adjacent → disjoint; self vs self → intersect
(and the caller skips self by path). No off-by-one. No change needed.

### N3 — `None` vs `{}` vs "no shell" are collapsed at `setup_world.py:546`
`if not other_record:` makes an empty-but-valid `{}` (a crashed write) report `"no other pool recorded"`
rather than the honest `"NOT CHECKED"`. The three-way distinction is the operator's only evidence.
*Fix:* `if other_record is None:` → "no other pool recorded"; anything else with no `shell` → "NOT CHECKED".
Same family: `setup_world.py:583` `list(world_spawn) if world_spawn else None` and `setup_world.py:635` /
`exp60_water_check.py:358` `if args.anchor_file else` should be `is not None` per the repo's
`is-not-none-over-truthy` convention (`--anchor-file ""` currently falls back to the default pool silently).

### N4 — a malformed sibling escapes as a traceback / exit 1 instead of a refusal
`_read_json_or_none` (`setup_world.py:604`) raises `SystemExit("corrupt anchor record ...")` — verified: a
truncated sibling aborts the build with exit **1**, not the module's `4` refusal convention, and the message
does not say which build it blocked. Blast radius also widened: one corrupt record now blocks building
**every** pool in that directory. Refusing is right; the exit code and message are not.
Separately, a sibling with a malformed `shell` (`{"shell": [1,2,3]}`) raises
`ValueError: not enough values to unpack` out of `pools_disjoint` (`setup_world.py:551`) as an uncaught
traceback. Both happen before any RCON write, so no world damage.
*Fix:* catch and refuse with `return 4` naming the sibling; length-check `other` before unpacking.

### N5 — `--shore-y` is unbounded while `--depth` is validated
`water_classroom_geometry` (`setup_world.py:410`) range-checks `depth` but not `shore_y`, so
`--shore-y -60` builds a shell at y -68…-55, below the 1.20.4 world floor (-64). The fill-reply gate probably
catches it (an out-of-world fill's reply contains no `"filled"`), but the depth precedent is right there and
a pure check is cheaper than a half-built apparatus.
*Fix:* mirror the depth guard — refuse unless `-64 <= shell_y0` and `shell_y1 <= 319`.

---

## Verified, no action

* **Flags reach the geometry on every path.** There is exactly one call to `water_classroom_geometry`
  (`setup_world.py:660`), downstream of all three anchor branches (explicit `--anchor-x/z`, recorded reuse,
  `bot_pos` fallback), so `--shore-y` and `--depth` apply to all three. The bot-position fallback discards the
  bot's y, so standing at pool 1's shore with `--shore-y 95` builds the intended stacked pool 2. (The
  record-reuse branch's *height* handling is S1; the `--anchor-x` without `--anchor-z` `TypeError` is
  pre-existing and unchanged by this diff.)
* **`--anchor-file` reaches both the read and the stamp in the check**, and there is no remaining wrong-file
  write: `ANCHOR_FILE` survives only as the module constant (`:86`), the default arg (`:270`, see S7) and the
  fallback (`:358`). A missing/unreadable file is caught early at `:360-364` with `instrument_error` + exit 4,
  so the warn-only branch in `_stamp_measured` is reachable essentially only for an unwritable file
  (loud enough downstream: `exp60_run.py:356` refuses without a stamped `measured`).
* **Record fields.** `flee_x`/`flee_z` = shore x/z is exactly what
  `rationale/r3-survival-benchmark/wiring-v3.md:143` prescribed ("put `flee_x/flee_z` (= the shore x/z) in the
  water anchor record"), and matches the Exp 58 record's own convention (`setup_world.py:368`, `flee_x =
  anchor_x` = its safe platform). The bridge's goal is `GoalNearXZ` (`minecraft_bridge/index.js:225`), i.e.
  y-blind — so for STACKED pools both records necessarily carry the SAME `flee_x/flee_z`; that is harmless
  (the preflight at `water_trial.py:414` is a free success at either shore) but the field cannot distinguish
  pools, and the record comment ("the flee anchor the bridge must be started with") reads as if it could.
  One clarifying clause would earn its place. `shell` matches `geom["shell"]`; `world_spawn` is `None` when
  not given; `surface_y` == `shore_y`.
* **Downstream consumers.** `exp60_run.py`, `exp60_water_check.py`, `l11_geometry_probe.py`, `water_trial.py`,
  `r3_run.py`, `r3_pilot.py`, `exp61_run.py` all read the record by KEY, never positionally and never against a
  fixed key set — the added fields are safe. `r3_run.py:317/361` freezes `anchor_measured` and
  `apparatus_record_ts`, not the whole record, so the new fields cannot trip the gauntlet's drift check.
  Note for the harness PR (not a defect here): all four harnesses still bind the module-level
  `exp60_run.ANCHOR_FILE`, so nothing downstream can yet be pointed at pool 2 — this PR is capability, and
  the caller lands in step 4 of the prereg's build order.
* **Symlink / relative-path handling** at `setup_world.py:667` is right: `resolve()` (non-strict, so a
  not-yet-created anchor is fine) correctly skips a symlinked alias of the pool being built, in both
  directions; a relative `--anchor-file` resolves against cwd and compares correctly; a not-yet-existing
  parent directory globs empty and is created before the write (`:736`).
* **Refusal ordering** is correct: the sibling guard runs after geometry and BEFORE the Exp 58/spawn
  clearances and before any `fill` — the colliding-pool test's "must refuse BEFORE touching the world"
  assertion holds, and a refused build leaves no record.
* `ruff check` / `ruff format --check` clean on all three files.

---

## Mutation table (are the new tests real guards?)

| Reverted behaviour | Tests that fail | Verdict |
|---|---|---|
| `shore_y=args.shore_y` dropped at `:660` | `test_two_pools_build_into_their_own_records_and_the_first_is_untouched` | guarded |
| record written to `WATER_ANCHOR_FILE` at `:737` | both `TestTwoPoolBuildOffline` tests | guarded (but see D1 — it writes `~/.maxim` while failing) |
| sibling guard block deleted (`:664-676`) | `test_a_colliding_second_pool_is_refused_before_any_fill` | guarded |
| self-skip deleted (`:667`) | **none — 40 passed** | **GAP (S8)** |
| `_stamp_measured` reverted to `ANCHOR_FILE` | `test_check_stamps_measured_into_the_pool_it_was_given` | guarded (but see D1) |

---

## Wider regression check

`PYTHONPATH="$PWD/src" python -m pytest tests/unit/test_exp60_drowning_substrate.py tests/unit/test_exp60_run.py
tests/unit/test_exp60_water_classroom.py tests/unit/test_exp61_fear_transport.py tests/unit/test_exp61_run.py
tests/unit/test_r3_pilot.py tests/unit/test_r3_run.py tests/unit/test_water_trial_lethal.py
tests/unit/test_water_trial_smoke.py -q` → **139 passed in 245.79s**. No Exp 60/61/R3 consumer regressed on
the added record fields.

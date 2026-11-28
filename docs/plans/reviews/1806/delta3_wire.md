# #1806 device registry: wire-integrity review, delta round 3

Reviewer: wire-integrity lens (docs/CODE_REVIEW.md), delta round 3, read-only, 2026-11-25, v4 staged on 0630487a (code at origin/main b7380ee7, incl. `scripts/`, `tests/`, `docs/`). Fresh reader. Scope: the v4 folds of W2-2, W2-3, W2-4 and W2-7 in §3.1 and §7, plus §3.1's "unpinned entry while a pinned entry exists is refused unless `--unpinned`, on every path". I ran no probe (the findings are text against code reads), contacted no robot or LAN host, and ran no sim.

**Verdict: SHOULD-FIX, no DO-NOT-BUILD.** v4 closes W2-2 and W2-4 as written, and closes the fail-open halves of W2-3 and W2-7 for the cases they named. Three NEW fail-open findings remain, all in already-covered classes (re-exec and phasing, N5/W2-3/W2-4; one resolver and no-primary, N3/W2-7; unpinned-on-every-path, adversarial N3): W3-1 (PR 0 ships `--device` but the re-exec that carries it lands in PR 2b), W3-2 (a legacy name that matches **no** entry in a pinned registry with no `primary:` still dials `<name>.local` unpinned), and W3-3 (§3a, §4.2 and D13′ still say legacy launches are unchanged and `--unpinned` is for `embody`/`--device` only, which contradicts §3.1's every-path rule). Each needs a text fix, and none needs a new owner decision. **The stop rule is not met.**

## 1. Re-check of the delta-round-2 findings in scope

| # | v4 status | Note |
|---|---|---|
| W2-2 | CLOSED (NIT W3-6) | The order is now `--device` > typed `--robot`/`--robot-name` > `MAXIM_DEVICE` > `MAXIM_ROBOT_NAME` > no name. `--device` with **any** typed `--robot*` is refused, `MAXIM_ROBOT_NAME` is listed, and "typed" can be decided mechanically, because the argparse default becomes `None` and the env var is read at resolution (`cli_parser.py` today has `default=os.environ.get("MAXIM_ROBOT_NAME", ...)`, which PR 0 removes). An outranked env var is logged. NIT: step 2's "today's matching" includes `resolve_robot_entry`'s fallback to the primary for a name that matches no entry id, and the note does not say so |
| W2-3 | **PARTIAL** | The no-entry fail-open is closed: the child gets the parent's **typed** `--robot-name`, and an env-only parent's child re-reads the same inherited env (`child_env()` does not scrub `MAXIM_ROBOT_NAME`/`MAXIM_DEVICE`). Two parts are still open. The fail-closed sub-case is unchanged: "plus `--unpinned` if the parent was **granted** it" does not cover a legacy parent whose path needed no grant (a registry with no pinned entry), so after PR 2b its child gets `--device <unpinned id>` with no `--unpinned` and is refused (§3a), and D13′ turns the parent's headless fallback into a non-zero exit across the switch. New: the daemon name is not carried (W3-5) |
| W2-4 | CLOSED | PR 0 omits `--robot-name` when it is `None`, in the same commit as the `None` default and the gate rewrite. The env-only test row pins it. The separate phasing gap for `--device` is W3-1 |
| W2-7 | **PARTIAL** | §7: `forget` clears `primary:` and never reassigns it. §3.1: a step-5 run in a pinned registry with no primary is refused, and the headless fallback is kept only for a registry with no pinned entry. That closes the **no-name** case. A **named** legacy run (step 2 or 4) whose name matches no entry reaches the same unpinned dial (W3-2) |
| §3.1 every path | **PARTIAL** | The rule is right, but it contradicts three other places in the note (W3-3), and the top-level `--unpinned` flag it relies on lands in PR 2b (W3-4) |

## 2. Producer and consumer maps

### (a) Selection sources and the re-exec

| Site (outside the change) | Today (b7380ee7) | Under v4 | Status |
|---|---|---|---|
| `cli_parser.py` `--robot-name` `default=os.environ.get("MAXIM_ROBOT_NAME","reachy_mini")` | the env var is folded into the argparse default, so typed and env cannot be told apart | the default becomes `None` (PR 0) | ✓ makes "typed" decidable |
| `cli.py::_main_impl` gate: `getattr(args,"robot_name","reachy_mini") != "reachy_mini" or "--robot" in raw_argv` | env and default count as "not an action" | rewritten to read parsed explicit flags in the same commit | ✓ |
| `cli.py::_main_impl` → `Maxim(robot_name=args.robot_name)` | a string | `robot_id=<entry id>, robot_name=<daemon name>`. The daemon name's source for a `--device` run is unstated | W3-5 |
| `embodied_runtime/selfy.py::Maxim.__init__`: `self.name = robot_name or os.getenv("MAXIM_ROBOT_NAME", ...)`, then `effective_robot_id = self._robot_id or self.name`, then `resolve_robot_entry` (exact id, else the primary when there is one entry or a marked primary, else `None`), then `connect_robot` with the defaults | a name that matches no entry, with no primary, dials `<name>.local` with no entry | §3.1 refuses only step 5 and unpinned **entries** | W3-2 |
| `cli_utils.py::reexec_with_mode`: always `"--robot-name", str(getattr(args,"robot_name","reachy_mini"))`, then `os.execve(..., child_env())` | replays the env-derived name as a typed flag | PR 0: omit when `None`. PR 2b: `--device <id>` [+ `--unpinned`] | W2-4 ✓. W3-1: `--device` is in PR 0 but its re-exec is in PR 2b |
| `utils/child_env.py::child_env` | does not touch `MAXIM_ROBOT_NAME`/`MAXIM_DEVICE` | the child inherits the env, so an env-only parent and its child resolve the same way | ✓ consistent with "never env-derived" in argv |
| `tests/unit/test_operational_mode_launch.py` (`_reexec_argv`, `test_every_runtime_switchable_name_survives_argparse`, `test_a_run_mode_switch_keeps_the_operators_grant`, `test_the_reexec_hands_over_child_env_not_a_pth_rewritten_pythonpath`) | no test asserts a robot flag | the five v4 rows go here. They need a sixth row, a `--device` parent, landing in PR 0 (W3-1) | W3-1 |
| `tests/unit/test_bounded_stop_cli_1344.py::test_a_robot_run_that_yielded_never_re_execs_into_a_requested_mode` | monkeypatches `_reexec_with_mode` | unaffected (it asserts the mode list, not the argv) | ✓ |
| `tests/unit/test_cli_action_gate.py::_compute_has_action` (a copy of the gate, including `"--robot" in argv`) | passes | must call the real gate (v3/v4 say so). `test_bare_maxim_is_not_action` would fail loudly against the copy | ✓ |
| `scripts/exp49/run_trials.py::_trial_env` (`child_env()` + `MAXIM_DATA_HOME`, no robot env) + `exp49_common.py::robots_yaml_text` (one entry, `primary: true`, `type: simulated`, no `hardware_id`) | the single entry, unpinned | the registry has no pinned entry, so the every-path and no-primary rules are inert. A mode switch after PR 2b hits W2-3's fail-closed sub-case | ✓ fail-closed |
| `scripts/reachy_streamer.py` | its own parser, `--robot-name` `default="reachy_mini"` | unaffected (it does not go through `maxim.cli`) | ✓ |
| `docs/user/robot-setup.md:43` `maxim --robot-name my_reachy` | typed, usually with an empty registry | typed → step 2, and the re-exec replays it | ✓ |
| `docs/user/cli-reference.md` (`--robot-name` default `reachy_mini`) | | must say "default: none; `$MAXIM_ROBOT_NAME` at resolution, else `reachy_mini`", and add `--device`/`--unpinned` and their precedence. It is already in §8's docs list | ✓ (text owed) |
| `docs/user/configuration.md` `MAXIM_ROBOT_NAME` row | "Robot identifier" | must say it is outranked by any typed flag and by `MAXIM_DEVICE`, and add a `MAXIM_DEVICE` row | ✓ (text owed) |
| Other `scripts/` that invoke `maxim --robot-name` or set `MAXIM_ROBOT_NAME` | none (grep of `scripts/`, `.github/`) | — | ✓ |

### (b) `forget` → no primary

| Reader | Effect after `forget` of the primary (≥2 entries left) |
|---|---|
| `hardware/config.py::resolve_robot_entry`, no name | returns `None` → §3.1 refuses in a pinned registry ✓ |
| the same, legacy name matching no entry id (typed or `MAXIM_ROBOT_NAME`) | returns `None` → selfy dials `<name>.local` unpinned (W3-2) |
| `RobotsConfig.get_primary` (falls back to `robots[0]`) | reached only through `resolve_robot_entry`'s guard ✓ |
| `doctor/checks.py::check_robot_reachable` | iterates every entry ✓ |

### (c) The legacy flow the owner asked about

A user has two hand-written entries, `rosie` (pinned, `hardware_id:`) and `bob` (no pin), and runs `maxim --robot-name bob`.

- **Can the case arise before PR 1?** No. Before PR 1 a hand-written `hardware_id:` is silently ignored (there is no typed field yet). In PR 1 it becomes a typed field under strict keys. So only a user who has opted into pinning can be in this state. Today's 1.3.x users (every entry unpinned) are not affected by the every-path rule.
- **Is the refusal intended?** Yes. It is adversarial delta-2 N3: a pinned registry must not quietly run a different, unverifiable robot.
- **Is it fail-closed?** Yes: the run is refused and nothing connects.
- **Is there a clear fix?** Not yet. The refusal text for this path is unspecified (§3a's "refused with an explanation" is scoped to `embody`/`--device`). The escape, a top-level `--unpinned`, lands in PR 2b, while pinning lands in PR 1 (W3-4). It needs a message naming both `--unpinned` and `maxim device add bob` (or `repin`), and the flag has to exist in the same PR as the refusal.
- **A neighbouring case is not covered.** Running `maxim --robot-name bobb` (a typo) in that registry falls back to `rosie` when `rosie` is the primary. That is legacy behaviour and the robot is verified (NIT W3-6). With no primary it dials `bobb.local` unpinned (W3-2).

## 3. New findings

| # | Finding | Fail-open/closed | Covered class? |
|---|---|---|---|
| W3-1 | **Phasing: `--device` ships in PR 0, but "the `--device` re-exec" is listed in PR 2b.** PR 0's re-exec only omits `None`. A `maxim --device bob` parent (bob not primary) that switches run mode therefore re-execs with no robot flag. The child takes step 3 (`MAXIM_DEVICE`, if exported) or step 5 (the primary), which is a **different robot**, and before PR 1 it is not even verified. Fix: move the whole §3.1 re-exec rule (`--device <id>`, and `--unpinned` once it exists) into PR 0 next to `--device`, with a "`--device` parent keeps the device" row in `test_operational_mode_launch.py`. Alternatively, defer `--device` itself to PR 2b | fail-open (wrong robot) | yes (N5, W2-4) |
| W3-2 | **A named legacy selection that matches no entry escapes both new rules.** In a registry that holds a pinned entry and no `primary:` (exactly the state `forget` now leaves), typed `--robot-name X` or `MAXIM_ROBOT_NAME=X`, where X is not an entry id, makes `resolve_robot_entry` return `None`, and selfy dials `X.local` with the defaults, **unpinned**. "Unpinned entries on every path" covers resolving to an unpinned *entry*, and "No name and no `primary:`" covers step 5 only. Fix: one rule in the resolver. In a registry that holds a pinned entry, any path that resolves to **no entry or to an unpinned entry** is refused unless `--unpinned` is given. Add test rows for typed and env names with no match, with and without a primary | fail-open (unverified robot) | yes (N3, W2-7) |
| W3-3 | **The every-path rule is contradicted elsewhere in v4.** §3a says "Unpinned entries need `--unpinned`. That applies to `embody` and to `--device` … **Legacy `--robot`/`--robot-name` launches are unchanged**". §4.2 says "It needs `--unpinned` for `embody`/`--device`". §9 D13′ says "`--unpinned` is required … through `embody`/`--device`". A builder following §3a lets a legacy flag run an unpinned entry in a pinned registry, which is the hole §3.1 closes. Fix: reword all three to "on every path when the registry holds a pinned entry (§3.1)", and say that "unchanged" means "for a registry with no pinned entry" | fail-open (by ambiguity) | yes (adversarial N3) |
| W3-4 | `--unpinned` as a **top-level** flag (usable with `--robot-name`, `MAXIM_DEVICE` and no name) is not in the §3.1 command block, and lands in PR 2b. Pinning, and so the every-path refusal, lands in PR 1. Between them, a mixed registry has no escape, and the refusal's fix hint is unspecified. Fix: land the refusal and the top-level `--unpinned` in the same PR (PR 1), and name the hint text: `--unpinned` or `maxim device add/repin <name>` | fail-closed | yes (adversarial N3, D13′) |
| W3-5 | "Re-exec repeats the parent's choice" carries the entry id but not the **daemon name**. Today a typed `--robot-name X` that falls back to the primary also makes X the daemon name (`selfy.py`: `"robot_name": self.name`). For an mDNS-only entry (no `host:`), that dials `X.local`. The child, given `--device <id>`, gets a daemon name whose source v4 does not state (§3.1 "`robot_name=<daemon name>`"). If it comes from the entry or the default, the child dials a different `.local` name. For a pinned entry the verify catches it (fail-closed). For an unpinned entry granted `--unpinned`, it reaches a **different unpinned robot**. Fix: state that a `--device` run's daemon name is the entry's declared `config.robot_name`, else the entry id. Then either (i) refuse legacy step 2 when it would fall back to the primary with a name that differs from the entry's daemon name (Alternative A), or (ii) have the re-exec also pass the parent's daemon name (Alternative B) | fail-open (narrow: unpinned, `--unpinned` granted, mDNS-only) | yes (W2-3) |
| W3-6 | NITs. (i) Step 2's "today's matching: entry id or daemon name" hides the primary fallback for a non-matching name (see W3-5); write it out. (ii) Whether a step-5 refusal exits non-zero or falls back to headless is not stated, because D13′ keeps headless for "a legacy launch with no device flag" (both outcomes are fail-closed). (iii) W2-3's fail-closed sub-case remains: define "granted" as "the parent's path admitted the unpinned entry", or replay the parent's legacy flag | fail-closed | yes |

## 4. Evidence

- **W3-1.** §8, PR 0 contains "`--device`, the conflict refusal …". PR 2b contains "the `--device` re-exec". §3.1 "Ordering" moves only the `None` omission into PR 0. `cli_utils.py::reexec_with_mode` builds argv from `robot_name`, home, timeout, epochs, mode and audio. It has no `device`, so a PR 0 that adds `args.device` without touching this function drops it.
- **W3-2.** `hardware/config.py::resolve_robot_entry`: `match = robots_config.get(robot_id)`; on a miss it takes `get_primary()` only when there is one entry or an explicit primary, otherwise it returns `None`. `selfy.py::Maxim.__init__` then calls `resolve_connection_config(...)`, which returns the defaults `{"robot_name": self.name, ...}`, then `connect_robot(robot_id=effective_robot_id, ...)`. v4 §3.1's two rules are worded as "resolves to an **unpinned** entry" and "a **step-5** run".
- **W3-3.** v4 text at §3a ("Unpinned entries need `--unpinned`" paragraph), §4.2 ("An unpinned entry resolves only from …"), §9 D13′.
- **W3-4.** §3.1 command block: `--unpinned` appears only on `embody`. §8 PR 1 holds the typed `hardware_id:`, and PR 2b holds `--unpinned`.
- **W3-5.** `selfy.py::Maxim.__init__`: `_connect_config = {"robot_name": self.name, ...}`, merged under the entry's `config:` (declared keys win), so `robot_name` survives unless the entry declares it. v4 §3.1 does not say where `<daemon name>` comes from on a `--device` run.

## 5. Stop-rule outcome

**Not met.** W3-1, W3-2, W3-3 and W3-5 are fail-open (W3-5 is narrow). All of them fall in already-covered classes (re-exec and phasing, one resolver, unpinned-on-every-path), and each is a text fix that needs no owner decision. W3-4 and W3-6 are fail-closed and covered, so they can be folded without a round. If W3-1 to W3-3 and W3-5 are folded as described above, a delta-scoped wire check of those four lines should meet the stop rule.

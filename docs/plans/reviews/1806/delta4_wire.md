# #1806 device registry: wire-integrity review, delta round 4

Reviewer: wire-integrity lens (docs/CODE_REVIEW.md), delta round 4, read-only, 2026-11-25, v5 working tree on 0630487a (code at b7380ee7; `git diff b7380ee7 0dd412e4 -- src/maxim/api.py` only extracts `_run_preflight` and touches no robot-name or device path, so it is not relevant). Fresh reader. Scope: §3.1 / §3a / §8 as v5 states them (the pinned-registry rule, the PR 0 / PR 1 ordering, the daemon-name rule, re-exec "granted means typed"), re-checking W3-1, W3-2, W3-3 and W3-5. No probe was run (text against code reads), and there was no robot or LAN contact and no sim.

**Verdict: SHOULD-FIX, no DO-NOT-BUILD.** The pinned-registry rule closes W3-2, and the one-PR ordering closes W3-1/W3-4. W3-3 is PARTIAL: §3a is fixed, but §4.2 and D13′ still carry the old `embody`/`--device` scoping. The W3-5 daemon-name rule closes the re-exec divergence, but it opens one NEW fail-open (W4-1). For an existing hand-written entry whose key differs from its daemon name and that has no `config.robot_name`, it changes which `.local` name today's legacy runs dial, in registries with **no** pinned entry. That contradicts §3.1's "every legacy flow is unchanged". W4-1 is in an already-covered class (W3-5 / W2-3, daemon name and re-exec). **The stop rule is not met.**

## 1. Re-check of the delta-round-3 findings in scope

| # | v5 status | Note |
|---|---|---|
| W3-1 | **CLOSED** | §8 PR 0 now says "`--device` is not in PR 0". PR 1 lists `--device`, its re-exec rule, `--unpinned`, the conflict refusal and the pinned-registry refusal together, and §3.1 "Ordering" agrees. PR 0's state is only the `None` default, the explicit-flag gate and a re-exec that omits `None`, with no pin, so there is no pinned registry and nothing for the rule to guard. An env-only parent's child inherits `MAXIM_ROBOT_NAME` through `child_env()` and re-resolves the same way. I found no fail-open build state between PR 0 and PR 1. NIT: the PR that ships the `MAXIM_DEVICE` reader (step 3) is unstated. Its conftest scrub and harness pop are in §5/PR 0. It should be named in PR 1. It is not fail-open, because the child inherits the env |
| W3-2 | **CLOSED** | One rule in the one resolver: in a registry that holds any pinned entry, every resolution that does not land on a pinned entry (no entry, an unpinned entry, a non-matching legacy name, no name with no `primary:`) is refused, exit 2, unless `--unpinned` is typed. `forget`'s no-primary state is covered. The fix hint is specified |
| W3-3 | **PARTIAL** | §3a is reworded ("Unpinned runs follow §3.1's pinned-registry rule … `embody` additionally always requires `--unpinned`"), and its exit list matches. Two places still carry the v4 scoping. §4.2 says "An unpinned entry … needs `--unpinned` for `embody`/`--device` (§3a)", and §9 D13′ says "`--unpinned` is required to run an unpinned entry through `embody`/`--device` (§3a)". Read literally, they (a) exempt legacy paths in a pinned registry, which is the fail-open reading that §3.1 overrules, and (b) require `--unpinned` for `--device` on an unpinned entry even in a registry with no pinned entry, which §3.1 does not require (only `embody` is always strict). A text fix is needed: point both sentences at §3.1, and state whether `--device` is strict like `embody` or follows the registry rule. This is the W3-3 residual, not a new class |
| W3-5 | **CLOSED for the re-exec, but it opens W4-1** | "The entry's `config.robot_name`, else the entry id, never from a fallback" makes parent and child derive the same daemon name. The source of the default changes today's behaviour, though (W4-1) |
| W3-4 | CLOSED | `--unpinned` is top-level in the §3.1 command block and lands in PR 1 with the refusal |
| W3-6 | CLOSED (NIT W4-3) | (i) Step 2's matching is written out. (ii) The refusal exits 2. (iii) "Granted means typed" is stated. A residual fail-closed edge case is in W4-3 |

## 2. Producer and consumer notes (only where v5 changed them)

### (a) Daemon name: today against v5

The relevant code at b7380ee7:
- `embodied_runtime/selfy.py::Maxim.__init__`: `self.name = robot_name or os.getenv("MAXIM_ROBOT_NAME", "reachy_mini")`, then `_connect_config = {"robot_name": self.name, ...}`, then `resolve_connection_config` merges the entry's `config:` over it (declared keys win).
- `hardware/reachy/controller.py::_resolve_mdns` dials `self._robot_name.replace("_","-") + ".local"` **only when `host:` is absent**. With `host:` set, the SDK `ReachyMini(robot_name=...)` only stores the name, so the change is harmless.

| Hand-written entry (no pinned entry anywhere) | Command | Today dials | v5 rule dials |
|---|---|---|---|
| `primary: {type: reachy_mini, primary: true, config: {}}` (no `host:`, no `robot_name`) | bare `maxim --mode live` | `reachy-mini.local` (default name) | `primary.local`: the connect fails and the run falls back to headless. A working setup breaks |
| same | `MAXIM_ROBOT_NAME=reachy_mini_2 maxim ...` | `reachy-mini-2.local` | `primary.local` |
| `reachy_mini: {type: reachy_mini, config: {}}` (key = another robot's daemon name), single entry | `maxim --robot-name reachy_mini_2` (docs/user/robot-setup.md:43 pattern) | `reachy-mini-2.local` (the typed name, falling back to the single entry) | `reachy-mini.local`: a **different physical robot**, unpinned, which then moves |
| `my_desk_robot: {primary: true, config: {host: 10.6.0.63}}` (the `test_robots_yaml_connect_config::test_primary_fallback_when_id_differs` shape) | bare | `10.6.0.63` | `10.6.0.63` (unchanged, because `host:` wins) |
| shipped template `data/util/robots.yaml` (`primary:` with `config.robot_name: reachy_mini`) | bare | `reachy-mini.local` | `reachy-mini.local` (unchanged) |

Answer to the brief's question: **yes.** An entry keyed `primary` with daemon `reachy_mini`, no `config.robot_name` and no `host:` dials `primary.local` under v5. Today it dials `reachy-mini.local`. The test comment at `tests/unit/test_robots_yaml_connect_config.py` ("The runtime name ("reachy_mini") routinely differs from the yaml key ("primary")") documents that this shape is expected.

### (b) Other legacy-flow changes that v5's "unchanged" claim hides (unpinned registries)

| Site | Today | v5 | Effect |
|---|---|---|---|
| `resolve_robot_entry` step 5 with the `reachy_mini` default | bare `maxim` sets the effective id to `"reachy_mini"`, which **exactly matches an entry keyed `reachy_mini` even when it is not the primary** | the argparse default is `None` → step 5 → the `primary: true` entry | a different entry is chosen when the registry holds a non-primary `reachy_mini` key. This arguably fixes today's behaviour, and it lands on the operator-marked primary. Not fail-open, but it should be stated (W4-2) |
| step 2 "else an entry whose `config.robot_name` equals the name" | there is no daemon-name matching. `--robot-name reachy_mini_2` falls back to the primary, whose declared `robot_name` wins, so today it reaches **robot A** | it matches the entry that declares `reachy_mini_2` | an improvement, but a change. Ties are unspecified: two entries declaring the same `robot_name`, or one entry's id equal to another's `robot_name` (W4-2) |

### (c) Re-exec, "granted means typed"

| Parent | Child argv (v5) | Child outcome | Status |
|---|---|---|---|
| `--device bob` (pinned) | `--device bob` | same entry, re-verified | ✓ |
| `--device bob --unpinned` (pinned registry, bob unpinned) | `--device bob --unpinned` | same | ✓ |
| `MAXIM_DEVICE=bob` | `--device bob` (+ inherited env, consistent) | same | ✓ |
| legacy `--robot-name X`, falls back to the primary (unpinned registry) | `--device <primary id>` | same daemon name under the v5 rule; it would diverge if W4-1 were fixed by restoring the typed name, so fix both together | ✓ under v5 |
| `--robot-name X --unpinned`, no entry (pinned registry) | `--robot-name X` only (bullet 2 does not carry `--unpinned`) | refused, exit 2, after the mode switch | fail-closed (W4-3) |
| env-only `MAXIM_ROBOT_NAME=X --unpinned`, no entry (pinned registry) | nothing robot-related | refused | fail-closed (W4-3) |
| `embody` parent, unpinned entry, `--unpinned` | `--device id --unpinned` (a plain run, not `embody`) | admitted | ✓ (the child is weaker than `embody`, but the parent was granted) |
| `exp49` `_trial_env` | pops `MAXIM_DEVICE`/`MAXIM_ROBOT_NAME`; single `exp49_sim` entry, `type: simulated`, unpinned | rule inert, daemon name unused for the simulated type | ✓ |

## 3. New findings

| # | Finding | Fail-open/closed | Covered class? |
|---|---|---|---|
| W4-1 | **The daemon-name rule's "else the entry id" default changes today's legacy behaviour in unpinned registries**, which contradicts §3.1's "every legacy flow is unchanged, including the headless fallback". For an entry with no `config.robot_name` and no `host:` (mDNS dial), today's daemon name is the typed name, else `MAXIM_ROBOT_NAME`, else `reachy_mini`. v5 uses the entry key instead. (a) The common case: the key `primary` dials `primary.local`, the connect fails, and the run drops to headless, so a working setup silently stops embodying. That is fail-closed, but it is a regression. (b) Where the key equals another robot's daemon name (for example key `reachy_mini` and a typed `--robot-name reachy_mini_2`), v5 dials and moves a **different, unpinned robot** from the one today's command reached. Fix options, all text: (i) make the default `config.robot_name`, else `MAXIM_ROBOT_NAME`, else `reachy_mini` (today's bare-run value), never the entry id, and state the step-2 fallback case. Either keep the typed name in an unpinned registry and have the re-exec replay the typed legacy flag for a fallback parent (rather than `--device`), or refuse a step-2 fallback whose typed name differs from the entry's daemon name. (ii) Alternatively, keep "else entry id" only for writer-managed files (where the PR 2 writer always emits `config.robot_name`), and use today's derivation for hand-written legacy files. Add test rows: key ≠ daemon, no `robot_name`, no `host:`, for a bare run, an env-only run and a typed fallback | fail-open (narrow, case (b)); fail-closed regression (case (a)) | yes (W3-5 / W2-3: daemon name and re-exec) |
| W4-2 | NIT. "Unchanged" also hides two intended changes in unpinned registries. A bare run no longer exact-matches a non-primary entry keyed `reachy_mini` (it takes the primary). Step 2 now also matches on `config.robot_name`. List them as intended changes, specify the tie-break (id match beats `robot_name` match; two entries declaring the same `robot_name` → `RobotsConfigError` or a refusal), and add a changelog line | fail-closed (it lands on the operator's primary, or the declared match) | yes (one resolver) |
| W4-3 | NIT. "Granted means typed" carries `--unpinned` only when "the parent resolved an entry". A pinned-registry parent that landed on **no** entry with a typed `--unpinned` (a typed `--robot-name X` that matches nothing, or env-only) re-execs without `--unpinned`, and its child is refused, exit 2, after the mode switch. Carry `--unpinned` whenever the parent typed it | fail-closed | yes (W2-3 sub-case) |
| W4-4 | NIT. Name the PR that ships the `MAXIM_DEVICE` reader (step 3) as PR 1, alongside `--device` | fail-closed (the env is inherited, so the child stays consistent) | yes (W3-1 phasing) |

## 4. Evidence

- **W4-1.** `selfy.py::Maxim.__init__` sets `self.name` (typed, else env, else `"reachy_mini"`) and `_connect_config = {"robot_name": self.name, ...}`. `hardware/config.py::resolve_connection_config` does `merged.update(match.config)`, so the entry's `robot_name` wins only when declared. `hardware/reachy/controller.py::_resolve_mdns` uses `self._robot_name` + `.local`, and only on the `self._host is None` branch of `connect`. `tests/unit/test_robots_yaml_connect_config.py::TestResolveConnectionConfig` asserts `merged["robot_name"] == "reachy_mini"` ("default preserved") for an entry whose key matches. Its fallback test documents that "the runtime name routinely differs from the yaml key ("primary")". v5 §3.1 says "For a resolved entry it is the entry's `config.robot_name` if set, else the entry id", and in the same section "In a registry with no pinned entry … every legacy flow is unchanged". The template `data/util/robots.yaml` declares `robot_name`, so template copies are unaffected.
- **W4-2.** `hardware/config.py::RobotsConfig.get` (exact `robot_id` only) and `resolve_robot_entry` (exact id first, then primary). `cli_parser.py` today sets `--robot-name` `default=os.environ.get("MAXIM_ROBOT_NAME","reachy_mini")`.
- **W3-3 residual.** §4.2, the "An unpinned entry resolves only from …" paragraph; §9 D13′.
- **W3-1 closure.** §8 PR 0 ("`--device` is not in PR 0 (v5)") and PR 1 (the bolded bundle). `cli_utils.py::reexec_with_mode` today unconditionally appends `--robot-name str(args.robot_name)`, and PR 0 replaces that with the omit-`None` rule. `utils/child_env.py::child_env` leaves the robot env vars in place.

## 5. Stop-rule outcome

**Not met.** W4-1 is fail-open in its narrow case (b), a different unpinned robot moves, and it falls in an already-covered class (daemon name and re-exec, W3-5/W2-3). The W3-3 residual in §4.2 and D13′ is fail-open by ambiguity and also covered. W4-2, W4-3 and W4-4 are fail-closed and covered, so they can be folded without a round. Every item is a text fix, and none needs a new owner decision. W4-1's option (i) or (ii) is a design choice inside the already-taken simplification. If W4-1 and the W3-3 residual are folded as described, a delta-scoped wire check of §3.1's daemon-name bullet, §4.2 and D13′ should meet the stop rule.

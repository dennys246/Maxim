# #1806 device registry: architecture review, delta round 4

Reviewer: architecture lens, delta round 4, read-only, 2026-11-25, v5 working tree on 0630487a. Fresh reader. I read the code at `b7380ee7` in the plan worktree. Probes ran under `env -i HOME=$(mktemp -d)` with the worktree `src` as an absolute PYTHONPATH, and `maxim.__file__` resolved to the worktree. A prototype of the lint ran from the session scratchpad, outside the repo. There was no robot, LAN or sim contact. Scope: §3b(d) as v5 states it, and the §3b(c) import-closure test.

**Verdict: PARTIAL. SHOULD-FIX, one fail-open item, in an already-covered class. No DO-NOT-BUILD.** Lint (d) can be built reliably. It now covers every spelling D3-1 named. But v5's alias rule tracks the wrong object: it follows `d = maxim.devices` and not aliases of the *package* (`pkg = maxim`). Ordinary non-`getattr` code therefore still reaches `maxim.devices` with no finding. A simpler rule that ignores the receiver closes that gap and needs no alias tracking. The §3b(c) closure test is sound, with two shape NITs.

## Re-check table

| Delta-3 # | v5 status | Note |
|---|---|---|
| D3-1 (i) `d = maxim.devices; d.add()` | **CLOSED** | Flagged. The load `maxim.devices` on the right-hand side is itself a load of `maxim.devices`, so the `d` alias adds nothing. |
| D3-1 (ii) `import maxim.hardware; maxim.devices.add()` | **CLOSED** | Flagged. Plain `import maxim.<x>` binds `maxim`. |
| D3-1 (iii) whole package, not write verbs | **CLOSED** | "Any load of `maxim.devices`", so `open_controller` is covered. |
| D3-1 layer set from `LAYER_RULES` | **CLOSED** | Stated. |
| D3-1 deletion probe | **CLOSED** | Stated for all three spellings. |
| D3-1 overall | **PARTIAL** | See D4-1. |
| D3-3 closure test | **CLOSED (with NITs)** | Today the closure is empty of `cli*`/`console*`. Re-probed: `import maxim, maxim.hardware.config, maxim.hardware.registry, maxim.embodied_runtime.selfy` loads none of them. Shape NITs are D4-3. |

## New findings

| # | Finding | Severity | Fail-open/closed | Covered class? |
|---|---|---|---|---|
| D4-1 | **Package aliases are not tracked.** v5 tracks names bound by `import` statements, plus `d = maxim.devices`. It does not track any name that is later bound to the `maxim` *package object*. The prototype of v5's literal rule misses all of these, and none is `getattr`/`importlib`: `pkg = maxim; pkg.devices.add()`; `self._m = maxim` … `self._m.devices.add()`; `def f(m=maxim): m.devices.add()`; `(m := maxim)`; and `hc.maxim.devices` through any module that itself did `import maxim.x`, where `hc` comes from `import maxim.hardware.controller as hc`. The audit catches none of them either, because there is no import of `devices`. Chasing these with alias tracking means tracking assignments, attributes, defaults, walrus, `global`, function-local scope and cross-module attributes, which is hard to get right. **Fold (simpler and fail-closed):** make (d) ignore the receiver. In a banned layer, flag **any `ast.Attribute` with `attr == "devices"` and `ctx=Load`, whatever its receiver**. Also flag `from maxim import *`, and any string literal starting `maxim.devices` passed to `importlib.import_module`/`__import__`. `scripts/lint_http_surface.py::imported_modules` already does this for its dynamic importers. Today this has **zero hits** in all eight banned layers (a grep for `\.devices\b` across `runtime tools agents planning memory bridges hardware embodied_runtime` returns nothing), so it needs no allow-list. A future legitimate `.devices` attribute in those layers fails loudly and gets an explicit per-file exemption. With this rule, the alias table, the scope tracking and the `import maxim.<anything>` analysis all go away. | SHOULD-FIX | fail-open: the package is reached from a banned layer with no finding, through spellings a developer writes by accident, not as evasion | yes ("a lint that cannot match the spelling", delta-2 F1/F2, D3-1) |
| D4-2 | **State the residuals exactly.** Even the receiver-agnostic rule misses spellings that reach the object without the attribute token: `getattr(maxim, "devices")` (with a literal or a built string), `vars(maxim)["devices"]`, `maxim.__dict__["devices"]` and `sys.modules["maxim.devices"]`. The last three work only once something has loaded the package, which the CLI does in a running process. v5's residual line names only `getattr()`/`importlib`. It should name all four, and say that a *literal* `import_module("maxim.devices")` is **not** a residual, because the D4-1 rule catches it. Also, `from maxim import *` binds `devices`, because `create`/`load` sit in `__all__`. A probe confirms that `from maxim import *` loads `maxim.create`. If `devices` joins `__all__` like the other submodules, a star import reaches it. Today that is fail-closed through ruff `F403`/`F405` (`select = ["E", "F"]`), which can be bypassed only by `# noqa`. D4-1's rule flags it anyway. | NIT | fail-closed once stated (documented evasion class) | yes (stated residuals, delta-2) |
| D4-3 | **Closure-test shape.** (1) The cited test lives at `tests/integration/test_advisor_gate.py::test_import_closure`, not under `tests/unit/`, and it runs in a **child process** (`_child_session`). The devices test must do the same, because `sys.modules` in a pytest worker is already polluted by earlier tests, so an in-process check passes or fails for reasons unrelated to the package. (2) The test should import **every** submodule of `maxim.devices` (`pkgutil.walk_packages`, including `_write`), not only `import maxim.devices`. A lazy `__init__` would otherwise hide `_write`'s closure. Assert by prefix: no `maxim.cli`, `maxim.cli_parser`, `maxim.cli_utils`, or `maxim.console` / `maxim.console.*`. Function-body CLI imports never enter a closure, but (c)'s audit rule already catches them as `function-local` scope, so the two together are complete. | NIT | fail-closed (the misleading in-process form passes vacuously, so name the child-process form now) | yes (claim-scope, D3-3) |
| D4-4 | **Wording.** "`import maxim.<anything>` (each binds `maxim`)" is false with `as`: `import maxim.runtime.llm_server as _srv` binds `_srv` to the submodule, not `maxim`. That form appears four times in `runtime/` and `tools/` today. Harmless under D4-1, which drops the binding analysis. Under v5's rule, an implementer who copies the sentence literally either over-tracks or adds a special case. | NIT | fail-closed | yes |

**Front gate.** The rule should ride on the existing audit, not on a new `scripts/lint_*.py`. `src/maxim/utils/audit.py::audit_architecture` already walks exactly the `LAYER_RULES` directory layers (`for layer, rules in LAYER_RULES.items()`) and already parses each file. An attribute pass that runs when `"devices" in must_not_import` gets four things for free: the layer set, the baseline/`compare_to_baseline` machinery, the `tmp_path` deletion-probe pattern in `tests/unit/test_architecture_audit.py`, and the `ARCHITECTURE.md` reporting. `lint_http_surface.py` is the nearest existing alias/dotted-chain lint (`_dotted`, `imported_modules`), and its literal-dynamic-import helper is worth copying. But its scope (`src/` minus one file) and its exits do not match the per-layer rule. `lint_motion_safety.py` is regex-based and not a fit.

## Evidence

- Prototype run under `env -i`, 15 spellings. v5-literal = an alias table from `import maxim [as m]` / `import maxim.<x>` (no `as`) plus `d = maxim.devices`. Agnostic = any `.devices` Load, plus `from maxim import *`, plus a literal `maxim.devices*` first argument.

  | Spelling | v5 | agnostic |
  |---|---|---|
  | `import maxim; maxim.devices.add()` | 1 | 1 |
  | `import maxim as m; m.devices.add()` | 1 | 1 |
  | `import maxim.hardware; maxim.devices.add()` | 1 | 1 |
  | `d = maxim.devices; d.add()` | 1 | 1 |
  | function-local `import maxim` | 1 | 1 |
  | `pkg = maxim; pkg.devices.add()` | **0** | 1 |
  | `self._m = maxim` … `self._m.devices` | **0** | 1 |
  | `def f(m=maxim): m.devices` | **0** | 1 |
  | `(m := maxim)` | **0** | 1 |
  | `import maxim.hardware.controller as hc; hc.maxim.devices` | **0** | 1 |
  | `from maxim import *; devices.add()` | **0** | 1 |
  | `importlib.import_module("maxim.devices")` | **0** | 1 |
  | `getattr(maxim, "devices")` | 0 | 0 (residual) |
  | `sys.modules["maxim.devices"]` | 0 | 0 (residual) |
  | `vars(maxim)["devices"]` | 0 | 0 (residual) |

- `grep -rnE "\.devices\b"` over the eight banned layers: no matches, so the agnostic rule has no false positives on the current tree. Repo-wide, `devices` appears only in comments and strings.
- `src/maxim/__init__.py::__getattr__`: an `_API_SUBMODULES` hit runs `importlib.import_module` and caches the result in `globals()`. That is why `vars(maxim)`/`__dict__` work after the first load. `__all__` contains `"create"` and `"load"`. A child-process probe shows `from maxim import *` loads `maxim.create`.
- `src/maxim/utils/audit.py::_check_file`: for `from maxim import *`, the candidate is `maxim.*`, so `target_layer` returns `*`, which is never a forbidden layer. The audit does not flag a star import.
- Function-local scope, `global` and reassignment: v5's rule handles a function-local `import maxim` only if the table ignores scope. A scope-aware implementation is where a fail-open bug would hide. The agnostic rule needs no scope model at all.
- `tests/integration/test_advisor_gate.py::test_import_closure` uses `_child_session` (a subprocess) and `_forbidden_loaded`.

## Stop-rule outcome

**STOP RULE NOT MET.** D4-2 to D4-4 are fail-closed NITs in already-covered classes. D4-1 fails open: package-alias spellings reach `maxim.devices` from a banned layer with no finding. It is in an already-covered class, and its fold is a simplification, a receiver-agnostic `.devices` Load rule inside `audit.py`. After that fold, a delta-scoped check of D4-1 alone is enough. The check: re-run the 15-spelling table against the folded text, and confirm the residual list names `getattr`, `vars`, `__dict__` and `sys.modules`.

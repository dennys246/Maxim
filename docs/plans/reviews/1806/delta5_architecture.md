# #1806 device registry: architecture review, delta round 5 check

Reviewer: architecture lens, delta round 5 check, read-only, 2026-11-25, v6 @ b0cbd8f7

Scope: D4-1 only, as v6 states it in §3b(d). That is a receiver-agnostic rule inside `utils/audit.py::audit_architecture`. In every layer that bans `"devices"`, it flags any `ast.Attribute(attr="devices", ctx=Load)`, `from maxim import *`, and a literal `maxim.devices*` argument to `import_module`/`__import__`. Code read at `b7380ee7` (unchanged in the plan worktree): `src/maxim/utils/audit.py`, `tests/unit/test_architecture_audit.py`, `scripts/lint_http_surface.py::imported_modules`.

**Verdict: CLOSED. No DO-NOT-BUILD and no SHOULD-FIX.** All 12 caught rows of the delta-4 table are caught by v6's rule as written. The 3 residual rows are named in v6's residual list. The rule finds zero hits on today's tree in all 8 banned layers. `audit_architecture` can host the rule and its baseline without changing the diff format. The probe also found three new items, all fail-closed NITs in already-covered classes. The main one: the residual list names 4 spellings, but the probe found 7 more dynamic spellings it does not catch and does not name.

## Re-check table

| # | v6 status | Note |
|---|---|---|
| D4-1 receiver-agnostic rule | **CLOSED** | Delta-4 rows 1–12 are all flagged by a literal prototype of v6 §3b(d). The audit alone (with `"devices"` added to `runtime`'s ban) flags none of them, so the new pass is what does the work. |
| D4-1 hosting in `audit_architecture` | **CLOSED (NIT D5-3)** | One more pass in `_check_file`, gated on `"devices" in forbidden_layers`. It reuses `AuditViolation`, `baseline_key`, `compare_to_baseline`, `render_baseline` and `format_diff` unchanged. |
| D4-1 zero hits today | **CLOSED** | Probed: 0 hits in all 8 layers (file counts in Evidence). |
| D4-2 residuals | **CLOSED for the 4 named.** Incomplete as a class (D5-1). | `getattr`, `vars`, `__dict__` and `sys.modules` are named. Seven more spellings are neither caught nor named. |
| D4-4 | **moot**, as v6 says | The rule no longer analyses bindings. |

## New findings

| # | Finding | Severity | Fail-open/closed | Covered class? |
|---|---|---|---|---|
| D5-1 | **The residual list is a list, not a class.** Besides the 4 named residuals, the probe shows v6's rule (with detection copied from `imported_modules`) misses all of these: `import_module(".devices", "maxim")`, `import_module(".devices", package="maxim")`, `__import__("devices", globals(), None, ["add"], 2)` (relative), `from importlib import import_module as im; im("maxim.devices")` (`imported_modules` matches the callee *name*, so an alias escapes), `import_module("maxim." + "devices")`, `__import__("maxim", fromlist=["devices"])` and `operator.attrgetter("devices")(maxim)`. None of them is in §3b(d)'s residual list. **Fold, either one:** (i) name the residual as a class: "any string-named or computed reach: `getattr`/`attrgetter`/`vars`/`__dict__`/`sys.modules`, and dynamic imports whose name is relative, built, aliased, or `maxim` plus `fromlist`"; or (ii) better, close most of it cheaply. Add one more receiver-agnostic clause: flag any `str` constant that equals `"devices"` or `".devices"`, or starts with `"maxim.devices"` or `".devices"`, in a banned layer. This catches every literal row above, plus delta-4 rows 13–15 when their key is a literal. Only built strings (`"maxim." + x`) would remain residual. Probed: **zero** such constants in the 8 layers today, so it needs no allow-list. | NIT | fail-closed once stated (outside the "static import" claim scope; same documented-evasion class) | yes (D4-2, stated residuals) |
| D5-2 | **The star clause must use the resolved module.** If it is implemented as the literal `node.module == "maxim" and level == 0`, it misses `from .. import *` in `runtime/x.py`. That statement is semantically identical to `from maxim import *`, and it binds `devices` once `devices` is in `__all__`. Using `_resolve_from_module(node, package_parts) == "maxim"`, which the audit already has, catches it (probe row X1). ruff F403 also flags it, so it is bypassable only with `# noqa`. | NIT | fail-closed (ruff F403) | yes (D4-2 star note) |
| D5-3 | **Hosting details the implementer must get right.** (1) `parse_baseline` rejects an entry whose `symbols` is empty. So an attribute finding must carry a non-empty `names` (e.g. `("devices",)`), or a legitimate exemption can never be baselined. (2) `scope` must be one of `SCOPES`, and `_iter_imports_with_scope` yields only imports, so the pass either reuses that walker's scope logic or uses `SCOPE_MODULE`. (3) Use a distinct `imported_module`/`rule` marker (e.g. `rule="<layer> must_not_reference devices"`, `imported_module="maxim.devices"`). The key is `file::imported_module::scope`, so it may merge with a real import finding of the same edge. A merge fails closed (`stale` on missing symbols), but it makes reviews confusing. (4) The deletion probe should assert rows 13–15 (and D5-1's rows, if kept as residuals) **as residuals**, i.e. that they give 0 findings, so that widening the rule later is a visible test change. | NIT | fail-closed (each mistake makes the gate red or unbaselinable, never silent) | yes (lint-mechanics, D3-1/D4-1) |

## Evidence

- Prototype `probe.py` (session scratchpad, outside the repo), run under `env -i HOME=$(mktemp -d)` with `PYTHONPATH=<worktree>/src`. `audit from .../.worktrees/plan-1806-device-registry/src/maxim/utils/audit.py` was confirmed. Each spelling was written as `runtime/mod.py` in a temp tree, with `"devices"` added to `runtime`'s `must_not_import`. Columns: `audit` = today's `_check_file`; `v6(lit*)` = v6's rule with a literal star match; `v6(res*)` = v6's rule with the star match resolved through `_resolve_from_module`.

  | Spelling | audit | v6(lit*) | v6(res*) |
  |---|---|---|---|
  | 01 `import maxim; maxim.devices.add()` | 0 | 1 | 1 |
  | 02 `import maxim as m; m.devices.add()` | 0 | 1 | 1 |
  | 03 `import maxim.hardware; maxim.devices.add()` | 0 | 1 | 1 |
  | 04 `d = maxim.devices; d.add()` | 0 | 1 | 1 |
  | 05 function-local `import maxim` | 0 | 1 | 1 |
  | 06 `pkg = maxim; pkg.devices` | 0 | 1 | 1 |
  | 07 `self._m = maxim` … `self._m.devices` | 0 | 1 | 1 |
  | 08 `def f(m=maxim): m.devices` | 0 | 1 | 1 |
  | 09 `(m := maxim).devices` | 0 | 1 | 1 |
  | 10 `hc.maxim.devices` | 0 | 1 | 1 |
  | 11 `from maxim import *` | 0 | 1 | 1 |
  | 12 `importlib.import_module("maxim.devices")` | 0 | 1 | 1 |
  | 13 `getattr(maxim, "devices")` (stated residual) | 0 | 0 | 0 |
  | 14 `sys.modules["maxim.devices"]` (stated residual) | 0 | 0 | 0 |
  | 15 `vars(maxim)["devices"]` (stated residual) | 0 | 0 | 0 |
  | X1 `from .. import *` | 0 | **0** | 1 |
  | X3 `import_module(".devices", "maxim")` | 0 | **0** | **0** |
  | X4 `import_module(".devices", package="maxim")` | 0 | **0** | **0** |
  | X5 `__import__("devices", globals(), None, ["add"], 2)` | 0 | **0** | **0** |
  | X6 `import_module as im; im("maxim.devices")` | 0 | **0** | **0** |
  | X7 `import_module(name="maxim.devices")` | 0 | 1 | 1 |
  | X8 `__import__("maxim.devices")` | 0 | 1 | 1 |
  | X9 `import_module("maxim." + "devices")` | 0 | **0** | **0** |
  | X10 `__import__("maxim", fromlist=["devices"])` | 0 | **0** | **0** |
  | X11 `operator.attrgetter("devices")(maxim)` | 0 | **0** | **0** |

- Zero-hit scan with the v6 rule (resolved star) on today's tree. Files: runtime 63, tools 23, agents 28, planning 13, memory 33, bridges 8, hardware 14, embodied_runtime 9. **0 hits in each layer, and no parse failures.**
- D5-1(ii) string-constant scan over the same 8 layers, for `"devices"`, `".devices"`, `"maxim.devices*"` and `".devices*"`: **0 hits**.
- `audit.py::audit_architecture` loops `for layer, rules in LAYER_RULES.items()` → `_check_file(py_file, layer, forbidden, src_root)`. The file is already parsed there, so an extra `ast.walk` pass gated on `"devices" in forbidden_layers` fits with no change to the signature. A new `embodied_runtime` key is walked automatically, because the directory exists.
- `audit.py::parse_baseline` raises `BaselineFormatError` when `symbols` is missing or empty, and when `scope ∉ SCOPES` (D5-3 (1)/(2)). `compare_to_baseline` marks an entry `stale` when its symbols are missing from live findings, which is why a key merge fails closed (D5-3 (3)).
- `lint_http_surface.py::imported_modules` resolves the callee by `func.attr`/`func.id` against `_DYNAMIC_IMPORTERS`, takes only `Constant` string arguments, and never reads `package=`, `level` or `fromlist` (D5-1 rows X3–X6, X9, X10).

## Stop-rule outcome

**STOP RULE MET.** D4-1 is CLOSED. D5-1, D5-2 and D5-3 are all fail-closed, and all fall in already-covered classes: stated residuals (D4-2) and lint mechanics (D3-1/D4-1). Per the faster-flow rule, they go to the follow-up issue or into the implementation PR's notes, and no further architecture round is owed on §3b(d).

#!/usr/bin/env python3
"""Stage 4 of measurement_path_fail_loud.md — the no-silent-swallows lock.

Five checks, comment-tolerant (the PR #487 review found the comment-blind
pattern missed 10 ``pass  # best-effort`` swallows):

1. **Zero-total over the measurement path.** The 16 scoped files from the
   plan's inventory were purged in Stage 1 (48 sites instrumented via
   ``log_swallowed_exception``); this lock keeps them at zero bare
   ``except Exception:`` → ``pass``/``continue`` swallows forever.

2. **No-new-swallows, diff-scoped, repo-wide.** Per FUNCTION, the count of
   bare ``except Exception: pass/continue`` swallows may not rise against the
   base (moves credited, below). The motivating incident: the SCN drive path
   was dead for months behind exactly one bare-except-swallowed TypeError.
   (Per file until 2026-10-04; netting between functions of one file is no
   longer free, a stated tightening.)

3. **No de-instrumentation on the measurement path, diff-scoped.** Per
   measurement-path file, the number of broad swallows that do NOT report
   through ``log_swallowed_exception`` may not rise. This is the guard for
   the one shape checks 1 and 2 cannot see: rewriting
   ``except Exception: log_swallowed_exception()`` into
   ``except Exception: logger.debug(...)`` is handled-and-logged, so neither
   check fires — but the site keeps swallowing while its
   ``swallowed_exception`` events stop, and the Stage-2 firing gate then reads
   fewer firings as an improvement.

   It counts the UNREPORTED swallows rather than the reported ones on
   purpose. The previous guard (``fail_loud_stage2.py check``) compared the
   instrumented-site COUNT to a frozen number, which cannot tell a swallow
   that was DELETED from one that was de-instrumented — so every swallow
   burn-down (#863) tripped it and had to be waved through by re-baselining,
   which is just clicking past a guard. Here, deleting a reporting swallow
   leaves the unreported count unchanged and passes; de-instrumenting one
   raises it by one and fails. No baseline file and nothing to re-baseline.
   (The old gate DID run in CI, through a pytest floor over all of ``src/``;
   check 4 is what keeps that repo-wide reach.)

4. **Repo-wide conservation, diff-scoped.** Across every changed ``src/maxim/``
   file, Stage-1 reports may disappear only together with their handlers.
   Covers what check 3's listed-files scope cannot: a function moved into a new
   module and de-instrumented on the way.

5. **The silent-default shape, diff-scoped, repo-wide** (roadmap 1.3.2, the
   carried §1.3.1 row). A broad handler that ASSIGNS or RETURNS a fallback
   swallows as silently as ``pass`` and hid ``export_memories``; check 2's
   regex cannot see it. ``is_silent_default``: no re-raise, no call named in
   ``REPORT_VERBS``, and the bound exception name is never read. Checks 2 and
   5 share ONE per-function pool of silent handlers (``keyed_handlers``), so
   a site moving between the two shapes is neutral. Unlike check 2's
   regex history, check 5 also covers the 16 measurement-path files.

**Move credit (checks 2, 3, 5; owner decision 2026-10-04).** The gate is a
count per function; a rising function is credited by its NEW handler keys
matching keys another function LOST in the same diff (``keyed_ratchet``). A
key is the enclosing ``try`` statement's ``ast.dump`` plus the handler index,
so a verbatim extraction or module move (the 1.3.2 decomposition) is free and
a different new handler never matches a deleted one. A handler moved AND
edited needs a new, append-only ``scripts/swallow_moves.json`` record whose
source function's count dropped by at least the number of records naming it;
an unused new record fails. Check 3's credit comes only from the listed files.
Credit is greedy in diff order, so two claims competing for ONE drop can fail the second (a false
failure, never a false pass; split the diff or record the move).
Designed through three adversarial passes (approach note v3).

Check 4 needs no credit model: a verbatim move of a reporting handler drops
reports and handlers together at the source and adds both at the destination.

Checks 3 and 4 compare PER ENCLOSING FUNCTION, not per file (review round 2). Per file, a
#863 burn-down that deletes one silent swallow nets against a de-instrumentation
anywhere else in the same file, and both checks passed it. Remaining blind spots,
stated: masking inside a single function, and — for check 4 — a handler moved OUT
of a function that stays put, which is indistinguishable from "delete it here, add
an unrelated logged handler there" and must pass.

This lint catches FORGETTING, not evasion: known-unmatched shapes include a
comment line between the `except` and the `pass`, `except (X, Exception):`,
and same-line `except Exception: pass` (zero instances of any exist in
src/maxim/ today — verified 2026-08-13). Extend `swallow_hits` if one of
these ever appears in review.

Exits: 0 clean; 1 violations (details on stderr); 2 unexpected error.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lint_git import GitUnavailable, base_ref, changed_files, must_not_skip, show  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# The plan's scope table (measurement_path_fail_loud.md §Scope).
MEASUREMENT_PATH = [
    "src/maxim/decisions/nac.py",
    "src/maxim/decisions/temporal_credit.py",
    "src/maxim/runtime/tool_dispatch.py",
    "src/maxim/runtime/bio_integration.py",
    "src/maxim/runtime/agent_loop.py",
    "src/maxim/runtime/loop_setup.py",  # the setup block's measurement-path code (1.3.2 slice 1)
    "src/maxim/runtime/loop_gates.py",  # the pre-tick gate, moved from agent_loop.py (1.3.2 slice 2)
    "src/maxim/similarity/encoder.py",
    "src/maxim/similarity/ec.py",
    "src/maxim/bridges/tool_pain_bridge.py",
    "src/maxim/proprioception/pain_bus.py",
    "src/maxim/embodiment/body.py",
    "src/maxim/embodiment/tool_bridge.py",
    "src/maxim/simulation/sim_logger.py",
    "src/maxim/memory/hippocampus.py",
    "src/maxim/memory/hippocampus_consolidation.py",
    "src/maxim/integration/memory_hub.py",
    "src/maxim/decisions/causal_link.py",
]

_EXCEPT_RE = re.compile(r"except (Exception|BaseException)(\s+as\s+\w+)?\s*:\s*(#.*)?$")
_SWALLOW_RE = re.compile(r"^\s+(pass|continue)\s*(#.*)?$")


def swallow_hits(text: str) -> list[int]:
    """1-indexed line numbers of bare-swallow ``pass``/``continue`` lines."""
    lines = text.splitlines()
    hits: list[int] = []
    for i, line in enumerate(lines):
        if _EXCEPT_RE.search(line) and i + 1 < len(lines) and _SWALLOW_RE.match(lines[i + 1]):
            hits.append(i + 2)
    return hits


def is_stage1_report(call: ast.Call) -> bool:
    """True when ``call`` emits a Stage-1 ``swallowed_exception`` event — THE definition.

    Mirrors ``utils/logging.py::log_swallowed_exception``'s own ``stage1_form``: the zero-arg call
    (site from the frame), or the supplied-site form ``site=...``. The explicit form
    (``exc``/``operation`` given) is NOT a report: it logs a plain DEBUG line with no event, which
    the JSONL formatter reduces to ``{"e": "log"}`` and the Stage-2 gate cannot see.

    Single source of truth, deliberately: ``fail_loud_stage2.py::inventory_sites`` imports this. The
    first cut of check 3 accepted "either form", so rewriting ``log_swallowed_exception()`` into
    ``log_swallowed_exception(e, operation="x")`` de-instrumented a site while passing — two
    definitions of "reported" that disagreed, found by both review lenses.
    """
    fn = call.func
    if (fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)) != "log_swallowed_exception":
        return False
    keywords = {k.arg for k in call.keywords}
    site = next((k.value for k in call.keywords if k.arg == "site"), None)
    if site is not None and not (isinstance(site, ast.Constant) and site.value is None):
        return True
    return not call.args and not keywords & {"exc", "operation", None}


_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _walk_local(statements: list[ast.stmt]):
    """Walk a handler body WITHOUT entering nested defs, lambdas or classes.

    A ``raise`` or a report inside a nested scope does not run when the handler does, so counting
    it would exempt a handler that actually swallows silently.
    """
    # Filter at EVERY level, the top included: a `def` sitting directly in the handler body is as
    # nested a scope as one inside an `if` (the first cut only filtered children, and entered it).
    stack: list[ast.AST] = [s for s in statements if not isinstance(s, _NESTED_SCOPES)]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(c for c in ast.iter_child_nodes(node) if not isinstance(c, _NESTED_SCOPES))


def _is_broad(handler: ast.ExceptHandler) -> bool:
    """Bare ``except``, ``Exception``/``BaseException``, or a tuple containing either."""
    if handler.type is None:
        return True
    names = list(handler.type.elts) if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(n, ast.Name) and n.id in ("Exception", "BaseException") for n in names)


def _classify(text: str) -> list[tuple[int, bool]]:
    """``(lineno, reports)`` for every broad handler that does not re-raise."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out: list[tuple[int, bool]] = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.ExceptHandler) or not _is_broad(n):
            continue
        local = list(_walk_local(n.body))
        if any(isinstance(x, ast.Raise) for x in local):
            continue
        out.append((n.lineno, any(isinstance(x, ast.Call) and is_stage1_report(x) for x in local)))
    return out


def handlers_by_function(text: str) -> dict[str, list[bool]]:
    """``{enclosing qualname: [reports?, ...]}`` for every broad, non-re-raising handler.

    The unit checks 3 and 4 compare on. Counting per FILE (or per repo) nets: deleting one silent
    swallow and de-instrumenting a DIFFERENT one in the same diff cancels, and both checks passed
    it — on exactly the burn-down PRs (#863) they exist for (review round 2, reproduced through
    ``main()``). Per function, a burn-down in ``f`` cannot mask a de-instrumentation in ``g``.
    Module-level handlers key as ``<module>``.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    out: dict[str, list[bool]] = {"<module>": []}

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{scope}.{child.name}" if scope else child.name
                # EVERY function gets a key, handlers or not. Keyed only when it had a handler, a
                # function whose last `try` was deleted vanished into the pooled residual, and
                # "delete a reporting swallow here + add a logged one there" failed again (S1).
                out.setdefault(name, [])
                visit(child, name)
                continue
            if isinstance(child, ast.ExceptHandler) and _is_broad(child):
                local = list(_walk_local(child.body))
                if not any(isinstance(x, ast.Raise) for x in local):
                    reports = any(isinstance(x, ast.Call) and is_stage1_report(x) for x in local)
                    out.setdefault(scope or "<module>", []).append(reports)
            visit(child, scope)

    visit(tree, "")
    return out


def _per_function_changes(repo_root: Path, base: str, scope: str = "src/maxim/"):
    """``(rel, rel_at_base, old_map, new_map)`` per changed file, renames paired."""
    for rel, rel_at_base in changed_files(repo_root, base, scope):
        path = repo_root / rel
        new = handlers_by_function(path.read_text(errors="replace")) if path.exists() else {}
        old = handlers_by_function(show(repo_root, base, rel_at_base))
        yield rel, rel_at_base, old, new


def _split(old: dict[str, list[bool]], new: dict[str, list[bool]]):
    """Functions present on BOTH sides, and the residual: ones added, removed or renamed."""
    both = sorted(set(old) & set(new))
    residual_old = [r for k in set(old) - set(new) for r in old[k]]
    residual_new = [r for k in set(new) - set(old) for r in new[k]]
    return both, residual_old, residual_new


def unreported_swallow_hits(text: str) -> list[int]:
    """Line numbers of broad handlers that swallow WITHOUT a Stage-1 report (check 3's count).

    ``logger.debug(...)``, ``return None``, ``pass`` and the explicit ``log_swallowed_exception(e,
    operation=...)`` all count: each keeps swallowing while invisible to the Stage-2 firing gate. A
    narrow ``except KeyError:`` does not count. Catches FORGETTING, not evasion, like the rest of
    this lint: a conditional re-raise exempts the handler even where it swallows on the other path.
    """
    return sorted(ln for ln, reports in _classify(text) if not reports)


def reporting_swallow_hits(text: str) -> list[int]:
    """Line numbers of broad, non-re-raising handlers that DO emit a Stage-1 report (check 4)."""
    return sorted(ln for ln, reports in _classify(text) if reports)


def conservation_failure(repo_root: Path, base: str, scope: str = "src/maxim/") -> str | None:
    """Check 4 — Stage-1 reports may disappear only together with their handlers, REPO-WIDE.

    Check 3 covers only the listed measurement path, so it cannot see a handler MOVED into a new
    module and de-instrumented on the way; the count gate this replaced scanned all of ``src/``.

    Compared per ``(file, function)`` wherever the function exists on both sides: if its Stage-1
    reports fell by R, its broad non-re-raising handlers must have fallen by at least R. Functions
    that appear, vanish or are renamed in this diff — which is what an extraction looks like — are
    POOLED across the whole diff and compared the same way, so a moved-and-de-instrumented handler
    still fails. Deleting a reporting swallow passes; adding a new logged-only handler elsewhere
    passes (``handle-and-log`` stays allowed outside the measurement path).

    Remaining blind spots, stated rather than hidden: masking inside ONE function (de-instrument one
    handler, delete a different silent one, both in ``f``), and masking across the pooled residual
    (functions added/removed/renamed in the same diff). Both need handler identity finer than a
    function, which extraction renames away. Before round 2 this compared per FILE and per REPO,
    which let every #863 burn-down PR mask a de-instrumentation anywhere in the same diff.
    """
    problems: list[str] = []
    pool_old: list[bool] = []
    pool_new: list[bool] = []

    def lost(old: list[bool], new: list[bool]) -> int:
        d_report = new.count(True) - old.count(True)
        d_broad = len(new) - len(old)
        return d_broad - d_report if d_report < 0 and d_broad > d_report else 0

    for rel, _, old, new in _per_function_changes(repo_root, base, scope):
        both, r_old, r_new = _split(old, new)
        for key in both:
            n = lost(old[key], new[key])
            if n:
                problems.append(f"{rel}::{key} ({n})")
        pool_old += r_old
        pool_new += r_new
    n = lost(pool_old, pool_new)
    if n:
        problems.append(f"functions added/removed/renamed in this diff, pooled ({n})")
    if not problems:
        return None
    return (
        "Stage-1 reports disappeared without their handlers (check 4, repo-wide): "
        + "; ".join(problems)
        + ". Either a handler stopped reporting while still swallowing — possibly while being MOVED — "
        "or, in the pooled case, this diff also ADDS new broad handlers that do not report. Keep "
        "log_swallowed_exception() (or site=...) in the handler, delete the try/except, or make the "
        "new handlers Stage-1."
    )


# ── Move credit + check 5: the keyed ratchet (approach note v3, owner decisions 2026-10-04) ───────────

#: Committed records for a handler moved AND edited on the way (a verbatim move is free). Append-only.
MOVES_REL = "scripts/swallow_moves.json"
_MOVE_KEYS = {"pool", "file", "qualname", "moved_from", "date", "ref", "reason"}
_POOLS = ("silent", "measurement")
_REF_RE = re.compile(r"(#\d+|https://github\.com/[\w.-]+/[\w.-]+/(pull|issues)/\d+)")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

#: Check 5's "somebody is told": a call whose function name is one of these. By NAME, not receiver (so
#: `console.print` and any `.warn` count: a stated blind spot; `traceback.print_exc` / `sys.stderr.write` are
#: unmatched). Deliberately wider than check 3's Stage-1 definition: check 3 asks "does the Stage-2 gate see
#: it", check 5 asks "does ANYONE see it", so `log_swallowed_exception(e, operation=...)` reports here.
REPORT_VERBS = frozenset(
    "debug info warning warn error exception critical log print log_swallowed_exception log_structured "
    "log_exception user_warn sim_log display_status _record_error _note_corruption _mark_corrupt".split()
)


def _call_name(call: ast.Call) -> str | None:
    fn = call.func
    return fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)


def is_silent_default(handler: ast.ExceptHandler) -> bool:
    """Check 5's shape: a broad handler that neither re-raises, nor reports, nor surfaces its exception.

    "Surfaces" = reads the bound name (``except Exception as e`` with ``e`` loaded in the local body:
    ``ToolResult(error=str(e))``). Callers exclude check 2's ``swallow_hits`` lines so no site counts twice."""
    if not _is_broad(handler):
        return False
    local = list(_walk_local(handler.body))
    if any(isinstance(x, ast.Raise) for x in local):
        return False
    if any(isinstance(x, ast.Call) and _call_name(x) in REPORT_VERBS for x in local):
        return False
    if handler.name and any(
        isinstance(x, ast.Name) and x.id == handler.name and isinstance(x.ctx, ast.Load) for x in local
    ):
        return False
    return True


def keyed_handlers(text: str, pool: str) -> dict[str, list[str]]:
    """``{enclosing qualname: [key, ...]}`` for every handler in ``pool``.

    ``silent`` = check 2's ``swallow_hits`` shape plus check 5's silent-default (ONE pool, so a comment that moves
    a site from the regex's shape to check 5's is neutral). ``measurement`` = check 3's unreported broad swallows.
    A key is ``ast.dump`` of the ENCLOSING ``try`` (no positions) plus the handler's index: keying the handler alone
    collapsed check 2's 415 sites to two keys (``pass``/``continue``), so any deleted swallow paid for a new one
    anywhere (adversarial pass 2). Every function gets an entry, handlers or not. Unparsable text → ``{}``."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}
    hits = set(swallow_hits(text))
    out: dict[str, list[str]] = {"<module>": []}

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{scope}.{child.name}" if scope else child.name
                out.setdefault(name, [])
                visit(child, name)
                continue
            if isinstance(child, (ast.Try, getattr(ast, "TryStar", ast.Try))):
                dump = ast.dump(child)
                for i, h in enumerate(child.handlers):
                    if not _is_broad(h):
                        continue
                    local = list(_walk_local(h.body))
                    if pool == "silent":
                        # one append per handler: check 2's regex shape, else check 5's silent default
                        counted = h.body[0].lineno in hits or is_silent_default(h)
                    else:
                        counted = not any(isinstance(x, ast.Raise) for x in local) and not any(
                            isinstance(x, ast.Call) and is_stage1_report(x) for x in local
                        )
                    if counted:
                        out.setdefault(scope or "<module>", []).append(f"{dump}#{i}")
            visit(child, scope)

    visit(tree, "")
    return out


class MoveRecords:
    """``scripts/swallow_moves.json`` against the base: append-only, and only NEW records can credit."""

    def __init__(self, head_text: str, base_text: str) -> None:
        self.problems: list[str] = []
        head, base = self._parse(head_text, "HEAD"), self._parse(base_text, "base")
        if head[: len(base)] != base:
            self.problems.append(f"{MOVES_REL} is append-only: the base list is not an exact prefix of HEAD's")
            head = base
        self.new = head[len(base) :]
        self.used = [False] * len(self.new)

    def _parse(self, text: str, where: str) -> list[dict]:
        if not text.strip():
            return []
        try:
            data = json.loads(text)
        except ValueError as exc:
            self.problems.append(f"{MOVES_REL} at {where} is not JSON ({exc})")
            return []
        if not isinstance(data, list):
            self.problems.append(f"{MOVES_REL} at {where} must be a list")
            return []
        good = []
        for r in data:
            src = r.get("moved_from") if isinstance(r, dict) else None
            ok = (
                isinstance(r, dict)
                and set(r) == _MOVE_KEYS
                and r["pool"] in _POOLS
                and all(isinstance(r[k], str) and r[k].strip() for k in ("file", "qualname", "reason"))
                and isinstance(src, dict)
                and set(src) == {"file", "qualname"}
                and all(isinstance(v, str) and v for v in src.values())
                and isinstance(r["date"], str)
                and _DATE_RE.fullmatch(r["date"])
                and isinstance(r["ref"], str)
                and _REF_RE.fullmatch(r["ref"])
            )
            if not ok:
                self.problems.append(
                    f"{MOVES_REL} at {where}: malformed record {r!r} (needs exactly {sorted(_MOVE_KEYS)})"
                )
                continue
            good.append(r)
        return good

    def claim(self, pool: str, file: str, qualname: str, budget: Counter, alias: dict[str, str]) -> bool:
        """Use one unused new record for ``(pool, file, qualname)`` whose source still has drop budget (a renamed
        source may be named by either path; both resolve to one budget)."""
        for i, r in enumerate(self.new):
            src = (alias.get(r["moved_from"]["file"], r["moved_from"]["file"]), r["moved_from"]["qualname"])
            if not self.used[i] and (r["pool"], r["file"], r["qualname"]) == (pool, file, qualname) and budget[src] > 0:
                self.used[i] = True
                budget[src] -= 1
                return True
        return False

    def unused(self) -> list[str]:
        return [
            f"{MOVES_REL}: unused move record for {r['file']}::{r['qualname']} ({r['pool']}) — no matching rise "
            "with a dropping source in this diff"
            for i, r in enumerate(self.new)
            if not self.used[i]
        ]


def keyed_ratchet(
    repo_root: Path, base: str, pool: str, records: MoveRecords, *, listed: frozenset[str] | None = None
) -> list[str]:
    """Per function, ``pool``'s count may not rise, unless the rise is a MOVE.

    The gate is a COUNT (an in-place edit inside a ``try`` changes its key and must stay free; third adversarial
    pass). Keys only CREDIT a rising function: its NEW keys may match keys that disappeared elsewhere in the diff
    (each pays once: a verbatim extraction or module move is free), and any remainder needs a new record whose
    source function's count DROPPED by at least as many records as name it. ``listed`` restricts the check (and
    its credit) to those files: a silent handler moved INTO the measurement path is a rise there, not a move."""
    changes = []
    for rel, rel_at_base in changed_files(repo_root, base, "src/maxim/"):
        if listed is not None and rel not in listed and rel_at_base not in listed:
            continue
        path = repo_root / rel
        new = keyed_handlers(path.read_text(errors="replace"), pool) if path.exists() else {}
        old = keyed_handlers(show(repo_root, base, rel_at_base), pool)
        changes.append((rel, rel_at_base, old, new))
    # ONE budget per source function = its count DROP. Every credit, free or recorded, debits it, so a function that
    # only edits a `try` in place (its key changes, its count does not) donates nothing, and one drop never pays
    # twice (architecture review). A free match is a NEW key equal to a key a dropping function lost.
    budget: Counter = Counter()
    lost: dict[str, list[tuple[str, str]]] = {}
    alias: dict[str, str] = {}
    for rel, rel_at_base, old, new in changes:
        alias[rel_at_base or rel] = rel
        for q, keys in old.items():
            d = len(keys) - len(new.get(q, []))
            if d > 0:
                budget[(rel, q)] += d
            for k, n in (Counter(keys) - Counter(new.get(q, []))).items():
                lost.setdefault(k, []).extend([(rel, q)] * n)
    out = []
    for rel, _rab, old, new in changes:
        for q in sorted(new):
            a, b = len(old.get(q, [])), len(new[q])
            if b <= a:
                continue
            need = b - a
            for k, n in (Counter(new[q]) - Counter(old.get(q, []))).items():
                for _ in range(n):
                    src = next((x for x in lost.get(k, []) if budget[x] > 0), None)
                    if not need or src is None:
                        break
                    lost[k].remove(src)
                    budget[src] -= 1
                    need -= 1
            while need and records.claim(pool, rel, q, budget, alias):
                need -= 1
            if need:
                what = "silent broad-handler" if pool == "silent" else "unreported broad-swallow"
                out.append(
                    f"{rel}::{q}: {what} count rose {a} → {b} ({need} not a move) — narrow the exception "
                    "(ImportError/OSError/KeyError), use optional_dependency_available for an optional dependency, or "
                    "report it (log_swallowed_exception() or a logger call). A handler MOVED here and edited on the "
                    f"way needs a {MOVES_REL} record naming its source."
                )
    return out


def main() -> int:
    failures: list[str] = []

    # Check 1 — zero-total over the measurement path.
    for rel in MEASUREMENT_PATH:
        path = REPO_ROOT / rel
        if not path.exists():
            failures.append(f"{rel}: scoped file missing — update the lint's scope table")
            continue
        for ln in swallow_hits(path.read_text()):
            failures.append(
                f"{rel}:{ln}: silent swallow in the measurement path — "
                "use log_swallowed_exception() or narrow/propagate "
                "(measurement_path_fail_loud.md policy)"
            )

    # Check 0 — PRINT THE TOTALS every run. The 2026-08-27 score card's
    # Documentation-honesty condition is that this lint and the atomic_io
    # ratchet print their totals in CI so CLAUDE.md can cite the OUTPUT
    # instead of a number that rots in the file (added 2026-08-29).
    repo_total = 0
    repo_files = 0
    for path in sorted((REPO_ROOT / "src" / "maxim").rglob("*.py")):
        n = len(swallow_hits(path.read_text(errors="replace")))
        if n:
            repo_total += n
            repo_files += 1
    print(
        f"no-silent-swallows: {repo_total} bare `except Exception: pass/continue` site(s) in {repo_files} "
        f"file(s) across src/maxim/ ({len(MEASUREMENT_PATH)} measurement-path files held at zero; "
        "every other function grandfathered at its base count)"
    )

    # Check 2 — diff-scoped no-new-swallows across src/maxim/, on the shared
    # ratchet (scripts/_lint_git.py). A shallow CI clone can lack a merge-base
    # entirely; check 1 needs no git and its results must never be discarded
    # for a git failure, so a missing base ref SKIPS check 2 with an INFO.
    # (The pre-fold version returned 2 here, which made every PR red in CI and
    # threw away check 1's findings unprinted — caught by the #508 review.)
    try:
        base = base_ref(REPO_ROOT)
    except GitUnavailable as e:
        if must_not_skip(str(e)):
            return 2
        print(f"INFO: no base ref available; skipping diff-scoped check 2 ({e})")
        base = None
    # Check 3 — no de-instrumentation on the measurement path (see the module docstring).
    unreported = sum(
        len(unreported_swallow_hits((REPO_ROOT / rel).read_text(errors="replace")))
        for rel in MEASUREMENT_PATH
        if (REPO_ROOT / rel).exists()
    )
    print(
        f"no-silent-swallows: {unreported} broad swallow(s) on the measurement path do not report "
        "through log_swallowed_exception (ratcheted per function; may fall, may not rise)"
    )
    silent_total = sum(
        sum(len(v) for v in keyed_handlers(p.read_text(errors="replace"), "silent").values())
        for p in sorted((REPO_ROOT / "src" / "maxim").rglob("*.py"))
    )
    print(
        f"no-silent-swallows: {silent_total} silent broad handler(s) across src/maxim/ (checks 2 + 5: pass/continue "
        "plus silent defaults; ratcheted per function, moves credited)"
    )
    if base is not None:
        # Checks 2 + 5 (one silent pool) and 3, with move credit. A mid-run git failure is an error on a pull
        # request or push (it was a silent INFO skip, #1098).
        try:
            records = MoveRecords(
                (REPO_ROOT / MOVES_REL).read_text() if (REPO_ROOT / MOVES_REL).exists() else "",
                show(REPO_ROOT, base, MOVES_REL),
            )
            failures.extend(records.problems)
            failures.extend(keyed_ratchet(REPO_ROOT, base, "silent", records))
            failures.extend(keyed_ratchet(REPO_ROOT, base, "measurement", records, listed=frozenset(MEASUREMENT_PATH)))
            failures.extend(records.unused())
        except GitUnavailable as e:
            if must_not_skip(f"git failed mid-run in checks 2/3/5: {e}"):
                return 2
            print(f"INFO: diff-scoped checks 2/3/5 skipped mid-run ({e})")

        # Check 4 — repo-wide conservation (see conservation_failure).
        try:
            problem = conservation_failure(REPO_ROOT, base)
            if problem:
                failures.append(problem)
        except GitUnavailable as e:
            if must_not_skip(f"git failed mid-run in check 4: {e}"):
                return 2
            print(f"INFO: diff-scoped check 4 skipped mid-run ({e})")

    if failures:
        print("no-silent-swallows lint FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("no-silent-swallows lint: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())

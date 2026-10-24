#!/usr/bin/env python3
"""Release gates: no version at or above a gated version passes CI until that gate's file entries are done.

Owner decision 2026-10-15: 1.3.2 ships without the rest of the ``agent_loop``/orchestrator decomposition, and **no
version >= 1.4.0 passes CI until it is done**. ``scripts/release_gates.json`` lists what "done" means, and this lint
enforces it. It runs offline: local git only, no GitHub API. At creation the 1.4.0 gate holds 30 items (#1263 in
six parts, ConceptExtractor's worker threads among them; #863 in 7 DROP-class sites: the four loop_perception wraps
of its own close condition, and three orchestrator wraps that widen it by owner decision), 3 functions and 5
characterization arms; the file is the authority, and the lint prints its counts on every run.

**Where it runs.** The ``lint`` job runs ``python3 scripts/lint_release_gates.py`` (every rule below). The
``unit-tests`` job runs ``python scripts/lint_release_gates.py --junit fast-suite.xml`` after the fast suite, which
writes that report with ``--junitxml``: the runtime twin (rule 6). Exits: 0 clean, 1 violations, 2 when git or the
JUnit report cannot answer in CI. The lint prints every gate's outstanding counts on every run, at every version.

**The rules.**

1. **Version.** The version is the MAXIMUM of three sources: ``pyproject.toml [project].version``,
   ``src/maxim/__init__.py`` ``__version__`` and the newest released ``## [X.Y.Z]`` CHANGELOG header. So the gate
   never depends on version sync being green in the same run. The comparison uses the numeric release tuple
   (never a string), so ``1.4.0rc1``, ``1.4.0.dev0``, ``1.4.0a1``, ``1.4.0.post1`` and ``2.0.0`` are all gated,
   and ``1.10.0`` is above ``1.4.0``. A source that does not start ``\\d+.\\d+.\\d+`` fails closed: an epoch
   (``1!1.4.0``), ``1.4``, a ``dynamic`` version, a missing ``__version__`` or no released header. A gate is ACTIVE
   when the version is at or above its key.
2. **Shape.** The gate file is validated strictly: unknown or missing keys, duplicate ids, a bad ``ref``
   (``#NNN`` or a github.com issue/PR URL), a bad date, a function ``max_lines`` above 200, and a ``retired``
   record naming nothing are all failures (exit 1).
3. **Items.** An item is SATISFIED when it records ``evidence`` (test nodeids) and a ``red_gate``. Recording only
   one of the two is a failure. Every satisfied item is checked at every version:

   - **Static (rule 3a).** Each nodeid has the form ``tests/**/test_*.py::[Test*::]test_*``, with no ``[param]``.
     The file parses and is not ``tests/integration/test_memory_hub.py`` (the fast suite ignores it). The name is
     defined by exactly one def at its level, and nothing else at that level binds it (no re-export, assignment or
     duplicate def). Its class, if any, is a single top-level ``Test*`` class with no ``__init__`` and no
     ``__test__``. Refused anywhere on the path module -> class -> function: a reference to ``xfail``, ``skip``,
     ``skipif``, ``importorskip``, ``slow``, ``requires_model_cache`` or ``robot`` in a decorator (including
     inside ``parametrize``'s ``pytest.param(marks=...)``, and through a name bound at module level), in a
     ``pytestmark``, or in the body or at module or class level (``SkipTest`` and ``skipTest`` too). A
     ``pytestmark`` must be a literal mark or a list of marks; anything else fails closed. The body must assert
     (an ``assert``, ``pytest.raises``, or a call whose name contains ``assert``).
   - **Red gate (rule 3b).** ``red_gate.commit`` is a 40-hex sha and an ancestor of HEAD. ``red_gate.nodeid`` is
     one of the evidence nodeids, and at that commit its def (or its class) carried ``xfail(strict=True)``, with
     the literal keyword (pyproject sets no ``xfail_strict``). A trivially passing test cannot have sat as a strict
     red gate, because it XPASSes and fails the suite. A squash merge that drops the commit fails closed.

   At an ACTIVE gate every item must be satisfied or retired.
4. **Functions.** Each gated ``(file, qualname)`` is measured with ``lint_function_length.defs_in``/``span`` (one
   measure, imported by path). It must be one def, span at most ``max_lines``, and have NO pin in the
   function-length baseline's ``entries``. The baseline's ``exceptions`` are append-only history and already name
   these functions, so an exception entry does not count against the condition (owner decision SF-1,
   2026-10-15). **Descendants (SF-6):** every function reached from a live (not retired) gated one through the baseline exceptions'
   ``split_from``/``moved_from`` records, transitively, must also be unpinned and at most 200 lines (owner:
   anything split or moved out of a gated function). **A retired function leaves with its descendants** (owner
   decision 2026-10-16, #1323): neither it nor anything split or moved out of it is held to these rules, and its
   split-outs may carry ``accepted_new_debt`` records; what replaces it is an item with its own bound and evidence.
   **Debt snapshot (SF-5):** ``debt_snapshot`` holds the
   baseline's entry keys at gate creation. Every baseline entry outside it needs an ``accepted_new_debt`` record
   ``{file, qualname, reason, ref, date}``, which catches a rename followed by a wrapper, or a copy followed by
   gutting the original, because neither leaves a record. An accepted record may not name a gated function (retired ones
   included) or a descendant of a live one: that rule has no exception. All of these are enforced at an ACTIVE gate and printed before it.
5. **Arms.** The orchestrator characterization arms are a separate condition, gated as COMPLETE only. An arm is
   SATISFIED when the golden file's ``arms`` carries its name and it records direct-assertion ``evidence``, which
   must pass rule 3a. It needs no red gate: a characterization pins current behaviour. Its golden case
   ``<golden_test>[<arm>]`` must pass at run time (rule 6). At an ACTIVE gate every arm must be satisfied or
   retired. WHEN the arms land (before the slice that moves their code) is not checked here: a history rule over
   the function-length pin would brick CI at every version the first time a slice landed early, with no way back
   through an immutable gate file (review B1, owner decision 2026-10-15). It is a residual (M56).
6. **Runtime twin (SF-8, ``--junit <xml>``).** Static analysis cannot see a conftest fixture or
   ``pytest_collection_modifyitems`` that skips. So every evidence nodeid of a satisfied item or arm, and each
   satisfied arm's golden case, must appear in the fast suite's JUnit report (every parametrization, mapped
   forward with pytest's own ``mangle_test_address``, as ``check_lane_roster.py`` does), with every case passed.
   The report must also carry this lint's own test file, so that it is the fast suite's. A missing or unreadable
   report is exit 2.
7. **History (shrink-only, SF-4).** Every first-parent commit of HEAD that touched the gate file is compared with
   the working tree. Every gate key, function, required arm and item it had is still present. ``max_lines`` never
   rose. ``debt_snapshot``, each gate's ``decided``/``source``, the arms' golden and test, each arm's scope, and
   each item's ``ref``/``scope``/``site``/``source`` are unchanged. ``retired`` and ``accepted_new_debt`` are
   append-only (``_lint_allowance.append_only_problem``). Each historical version is read by ITS OWN
   ``gate_format_version`` (``HISTORY_VIEWS``), and only those immutable fields are compared, through that format's
   view on both sides: a later key, or a tightened validator, never fails an old version. A historical version that
   is absent (the file deleted on main), unreadable or missing a field its own format defines is a failure, never a
   skip. Leaving a gate is ``retired``: append-only, it needs a reason and a ref, and it is printed every run.
8. **The bump diff (SF-3).** A diff that raises the version from below a gate to at or above it may not change the
   gate file, so nothing is satisfied, retired or accepted at the last minute. The base is the merge-base with
   ``origin/main`` (then ``main``) on a pull request or locally, and ``HEAD^1`` on a push, never ``push_base``,
   which calls ``gh api``. Every event but a push uses the merge-base: ``schedule`` and ``workflow_dispatch`` too (on
   main that is HEAD, an empty diff), because a dispatch can run on any branch head (review SF1).
9. **Wiring (SF-10).** Both modes check both steps, read from the PARSED workflow. The lint job has exactly one
   step whose run is ``python3 scripts/lint_release_gates.py``, with only ``name`` and ``run``. The unit-tests job
   has exactly one ``python scripts/lint_release_gates.py --junit fast-suite.xml`` step, also only ``name`` and
   ``run``, placed after the fast suite. That suite (the ``--cov`` step) passes ``--junitxml=fast-suite.xml``, and
   no other step names the report. Also checked: every checkout in both jobs has ``fetch-depth: 0``, and neither
   job, nor the workflow, has an ``if:``, ``continue-on-error`` or ``defaults``.
   ``tests/unit/test_ci_workflow_shape.py`` pins both steps.

**Close-out (review S1).** A gate whose version was released BEFORE this change, meaning that the newest released
``## [X.Y.Z]`` CHANGELOG header AND ``pyproject.toml`` version at the diff base (``HEAD^1`` on a push only; the
merge-base with origin/main for every other event, local runs included) are at or above its key, is CLOSED: its
item, arm, function and new-debt rules only print, in the lint job and in the runtime twin alike. So a function
that regrows after 1.4.0 shipped, or an evidence test refactored later, cannot fail CI for good. Its shape,
history, bump-diff and wiring rules stay enforced. The base, not HEAD, decides it: the bump diff itself adds
the header, and that diff stays fully gated. Without git nothing is closed.

**Git and environment.** No environment variable changes the verdict. ``GITHUB_EVENT_NAME`` only picks rule 8's
base, and ``GITHUB_ACTIONS`` only decides what happens when git cannot answer. In CI, missing git or a shallow
clone is exit 2, because the history rule would otherwise pass vacuously. Locally, the git-backed rules (3b,
the close-out check, 7, 8) are skipped with an INFO line.

**Residuals (enforced by review only; mechanization backlog M56 in docs/plans/outstanding.md).**

- What a red gate actually pinned, and whether it failed for the right reason.
- An evidence test that never ran at its red-gate commit.
- Retiring a function also frees its descendants (#1323), and nothing ties a retired function to the item that
  replaces it (there is no ``replaced_by`` key; adding one needs a ``gate_format_version`` bump). Review checks
  that a replacing item exists.
- Self-approved ``retired`` and ``accepted_new_debt`` records. Each is visible and printed, but nothing proves an
  owner approved it. A function moved out of a gated one by a FREE move (an AST-identical rename, which leaves no
  record) and then blessed through ``accepted_new_debt`` is in this class.
- Code moved into module-level statements or a class body (no def, so nothing is measured; the printed
  module-level statement counts make it visible), or out of ``src/maxim``.
- This lint edited to pass, or both CI steps deleted together with their workflow-shape test.
- A local build and publish from an uncommitted ``pyproject.toml``. The publication guide's step runs this lint
  before ``python -m build``.
- An evidence name imported from a helper module that resolves to a skip or xfail. The static walk resolves only
  module-level assignments, so the runtime twin catches this case.
- A conftest or plugin hook that rewrites test outcomes, or a step that overwrites ``fast-suite.xml``: the twin
  trusts the report the suite wrote.
- Only ONE evidence nodeid per item needs a red-gate history; the others need only pass.
- An arm's golden entry may be a thin placeholder, filled in later; and the arms' ORDER (they land before the slice
  that moves their code) is enforced by that slice PR's review and roadmap_1_3_x.md's "Characterization owed", not
  mechanically.
- A direct multi-commit push to main that skips a PR (``[bump, x]``): the push run of ``x`` judges against ``x``'s
  parent, which already carries the bump, and reads CLOSED. A rebase-merged PR is covered by its own PR run.
- Rule 8 is met by putting the gate-file change and the bump in separate PRs. That is the intended route: each
  change to the gate file gets its own review, before the bump.

Regression guard: tests/unit/test_lint_release_gates.py drives ``main()`` on fixture git repos for every rule, with
a deletion probe for each, and checks this checkout passes.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lint_function_length as FL  # noqa: E402
from _lint_allowance import append_only_problem, ref_ok  # noqa: E402
from _lint_git import GitUnavailable, git, show  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
GATES_REL = "scripts/release_gates.json"
WORKFLOW_REL = ".github/workflows/test.yml"
FORMAT_VERSION = 1
MAX_LINES_CEILING = 200  # owner decision 2026-10-15 (the function-length THRESHOLD); a gate may only go lower
# (a looser bound is a retired function plus a replacing item with its own evidence: #1323, 2026-10-16)
LINT_JOB, TWIN_JOB = "lint", "unit-tests"
JUNIT_FILE = "fast-suite.xml"
LINT_RUN = "python3 scripts/lint_release_gates.py"
TWIN_RUN = f"python scripts/lint_release_gates.py --junit {JUNIT_FILE}"
SUITE_FLAG = f"--junitxml={JUNIT_FILE}"
ANCHOR_TEST_FILE = "tests/unit/test_lint_release_gates.py"
FAST_SUITE_IGNORED = frozenset({"tests/integration/test_memory_hub.py"})
DENY_MARKS = frozenset({"xfail", "skip", "skipif", "importorskip", "slow", "requires_model_cache", "robot"})
DENY_BODY = frozenset({"xfail", "skip", "skipif", "importorskip", "SkipTest", "skipTest"})

_NODEID_RE = re.compile(r"(tests/(?:[A-Za-z0-9_\-]+/)*test_\w+\.py)::(?:(Test\w+)::)?(test_\w+)")
_GATE_KEY_RE = re.compile(r"\d+\.\d+\.\d+")
_VERSION_PREFIX_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")
_ITEM_ID_RE = re.compile(r"\d+(?:\.[A-Za-z0-9_\-]+)*")
_ARM_RE = re.compile(r"[a-z][a-z0-9_]*")
_SHA_RE = re.compile(r"[0-9a-f]{40}")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

_GATE_KEYS = {
    "decided",
    "source",
    "functions",
    "debt_snapshot",
    "accepted_new_debt",
    "arms",
    "items",
    "retired",
}
_FUNCTION_KEYS = {"file", "qualname", "max_lines", "why"}
_DEBT_KEYS = {"file", "qualname", "reason", "ref", "date"}
_ARMS_KEYS = {"golden", "golden_test", "required"}
_ARM_KEYS = {"arm", "scope", "evidence"}
_ITEM_REQUIRED = {"id", "ref", "scope", "evidence", "red_gate"}
_ITEM_OPTIONAL = {"site", "source"}
_SITE_KEYS = {"file", "symbol", "pattern"}
_RED_GATE_KEYS = {"commit", "nodeid"}
_RETIRED_KEYS = {"kind", "key", "reason", "ref", "date"}
_RETIRED_KINDS = {"item", "function", "arm"}

Version = tuple[int, int, int]


class GateFileError(ValueError):
    """The gate file is malformed: a failure (exit 1), never a skip."""


class Unanswerable(RuntimeError):
    """Git or the JUnit report cannot answer in CI: exit 2."""


def _in_ci() -> bool:
    return bool(os.environ.get("GITHUB_ACTIONS"))


# ── rule 1: version ──────────────────────────────────────────────────────────


def parse_version(text: object, what: str) -> Version:
    """The numeric release tuple of a version string; anything that does not start ``X.Y.Z`` fails closed."""
    if not isinstance(text, str):
        raise GateFileError(f"{what}: no version string ({text!r})")
    m = _VERSION_PREFIX_RE.match(text.strip())
    if m is None:
        raise GateFileError(f"{what}: version {text!r} does not start X.Y.Z (fails closed)")
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def version_sources(read: object) -> tuple[dict[str, str], list[str]]:
    """{source: version string} from the three machine sources; ``read(rel)`` returns a file's text or ''."""
    out: dict[str, str] = {}
    errors: list[str] = []
    assert callable(read)
    try:
        project = tomllib.loads(read("pyproject.toml")).get("project", {})
    except tomllib.TOMLDecodeError as e:
        errors.append(f"pyproject.toml unreadable: {e}")
        project = {}
    if not isinstance(project, dict):
        errors.append("pyproject.toml's `project` is not a table (fails closed)")
        project = {}
    if "version" in (project.get("dynamic") or []):
        errors.append("pyproject.toml declares a dynamic version: the gate cannot read it (fails closed)")
    elif isinstance(project.get("version"), str):
        out["pyproject"] = project["version"]
    else:
        errors.append("pyproject.toml has no [project].version string (fails closed)")
    m = re.search(r"""^__version__\s*=\s*["']([^"']+)["']""", read("src/maxim/__init__.py"), re.M)
    if m:
        out["__init__"] = m.group(1)
    else:
        errors.append("src/maxim/__init__.py: no literal __version__ (fails closed)")
    released = re.findall(r"^## \[(\d[^\]]*)\]", read("CHANGELOG.md"), re.M)
    if released:
        out["CHANGELOG"] = released[0]
    else:
        errors.append("CHANGELOG.md has no released `## [X.Y.Z]` header (fails closed)")
    return out, errors


def max_version(sources: dict[str, str]) -> tuple[Version | None, list[str]]:
    errors: list[str] = []
    tuples: list[Version] = []
    for name, v in sources.items():
        try:
            tuples.append(parse_version(v, name))
        except GateFileError as e:
            errors.append(str(e))
    if errors or not tuples:
        return None, errors
    return max(tuples), []


def _fmt_version(v: Version) -> str:
    return ".".join(map(str, v))


# ── rule 2: shape ────────────────────────────────────────────────────────────


def _str(v: object, what: str) -> str:
    if not (isinstance(v, str) and v.strip()):
        raise GateFileError(f"{what} must be a non-empty string, got {v!r}")
    return v


def _date(v: object, what: str) -> str:
    if not (isinstance(v, str) and _DATE_RE.fullmatch(v)):
        raise GateFileError(f"{what} must be YYYY-MM-DD, got {v!r}")
    return v


def _ref(v: object, what: str) -> str:
    if not ref_ok(v):
        raise GateFileError(f"{what} must be '#NNN' or a github.com issue/PR URL, got {v!r}")
    assert isinstance(v, str)
    return v


def _keys(d: object, required: set[str], what: str, optional: frozenset[str] | set[str] = frozenset()) -> dict:
    if not isinstance(d, dict):
        raise GateFileError(f"{what} must be an object, got {d!r}")
    missing, unknown = required - set(d), set(d) - required - set(optional)
    if missing or unknown:
        raise GateFileError(f"{what}: missing keys {sorted(missing)}, unknown keys {sorted(unknown)}")
    return d


def _nodeid(v: object, what: str) -> str:
    if not (isinstance(v, str) and _NODEID_RE.fullmatch(v)):
        raise GateFileError(f"{what} must be a nodeid tests/**/test_*.py::[Test*::]test_* (no [param]), got {v!r}")
    return v


def _nodeids(v: object, what: str) -> list[str]:
    if not isinstance(v, list):
        raise GateFileError(f"{what} must be a list of nodeids")
    ids = [_nodeid(x, what) for x in v]
    if len(set(ids)) != len(ids):
        raise GateFileError(f"{what} lists a nodeid twice")
    return ids


def _src_file(v: object, what: str) -> str:
    s = _str(v, what)
    if not (s.startswith("src/maxim/") and s.endswith(".py")):
        raise GateFileError(f"{what} must be a src/maxim/**.py path, got {s!r}")
    return s


@dataclass(frozen=True)
class Gate:
    """One parsed gate. ``raw`` is the validated JSON object, kept for the history comparison."""

    key: str
    version: Version
    raw: dict

    @property
    def functions(self) -> list[dict]:
        return self.raw["functions"]

    @property
    def items(self) -> list[dict]:
        return self.raw["items"]

    @property
    def arms(self) -> dict:
        return self.raw["arms"]

    @property
    def retired(self) -> set[tuple[str, str]]:
        return {(r["kind"], r["key"]) for r in self.raw["retired"]}


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    keys = [k for k, _ in pairs]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise GateFileError(f"duplicate JSON keys {dupes} (the last one would silently win)")
    return dict(pairs)


def parse_gates(text: str) -> list[Gate]:
    try:
        data = json.loads(text, object_pairs_hook=_no_duplicate_keys)
    except json.JSONDecodeError as e:
        raise GateFileError(f"not JSON: {e}") from e
    _keys(data, {"gate_format_version", "gates"}, "the gate file", {"_comment"})
    if data["gate_format_version"] != FORMAT_VERSION:
        raise GateFileError(f"gate_format_version must be {FORMAT_VERSION}, got {data['gate_format_version']!r}")
    if not isinstance(data["gates"], dict) or not data["gates"]:
        raise GateFileError("gates must be a non-empty object keyed by version")
    return [_parse_gate(k, v) for k, v in data["gates"].items()]


def _parse_gate(key: str, g: object) -> Gate:
    if not _GATE_KEY_RE.fullmatch(key):
        raise GateFileError(f"gate key {key!r} must be X.Y.Z")
    w = f"gate {key}"
    _keys(g, _GATE_KEYS, w)
    assert isinstance(g, dict)
    _date(g["decided"], f"{w}.decided")
    _str(g["source"], f"{w}.source")

    fkeys: set[tuple[str, str]] = set()
    if not isinstance(g["functions"], list):
        raise GateFileError(f"{w}.functions must be a list")
    for f in g["functions"]:
        _keys(f, _FUNCTION_KEYS, f"{w}.functions[]")
        k = (_src_file(f["file"], f"{w}.functions[].file"), _str(f["qualname"], f"{w}.functions[].qualname"))
        n = f["max_lines"]
        if not (isinstance(n, int) and not isinstance(n, bool) and 0 < n <= MAX_LINES_CEILING):
            raise GateFileError(f"{w}: {k[0]}::{k[1]} max_lines must be an int in 1..{MAX_LINES_CEILING}, got {n!r}")
        _str(f["why"], f"{w}.functions[].why")
        if k in fkeys:
            raise GateFileError(f"{w}: function {k[0]}::{k[1]} listed twice")
        fkeys.add(k)

    snap = g["debt_snapshot"]
    if not (isinstance(snap, list) and all(isinstance(p, list) and len(p) == 2 for p in snap)):
        raise GateFileError(f"{w}.debt_snapshot must be a list of [file, qualname] pairs")
    pairs = [(_src_file(p[0], f"{w}.debt_snapshot"), _str(p[1], f"{w}.debt_snapshot")) for p in snap]
    if len(set(pairs)) != len(pairs):
        raise GateFileError(f"{w}.debt_snapshot lists a key twice")

    if not isinstance(g["accepted_new_debt"], list):
        raise GateFileError(f"{w}.accepted_new_debt must be a list")
    for d in g["accepted_new_debt"]:
        _keys(d, _DEBT_KEYS, f"{w}.accepted_new_debt[]")
        _src_file(d["file"], f"{w}.accepted_new_debt[].file")
        _str(d["qualname"], f"{w}.accepted_new_debt[].qualname")
        _str(d["reason"], f"{w}.accepted_new_debt[].reason")
        _ref(d["ref"], f"{w}.accepted_new_debt[].ref")
        _date(d["date"], f"{w}.accepted_new_debt[].date")

    arms = _keys(g["arms"], _ARMS_KEYS, f"{w}.arms")
    _str(arms["golden"], f"{w}.arms.golden")
    _nodeid(arms["golden_test"], f"{w}.arms.golden_test")
    if not isinstance(arms["required"], list) or not arms["required"]:
        raise GateFileError(f"{w}.arms.required must be a non-empty list")
    names: set[str] = set()
    for a in arms["required"]:
        _keys(a, _ARM_KEYS, f"{w}.arms.required[]")
        if not (isinstance(a["arm"], str) and _ARM_RE.fullmatch(a["arm"])):
            raise GateFileError(f"{w}: arm name {a['arm']!r} must match {_ARM_RE.pattern}")
        if a["arm"] in names:
            raise GateFileError(f"{w}: arm {a['arm']!r} listed twice")
        names.add(a["arm"])
        _str(a["scope"], f"{w}.arms {a['arm']}.scope")
        _nodeids(a["evidence"], f"{w}.arms {a['arm']}.evidence")

    ids: set[str] = set()
    if not isinstance(g["items"], list):
        raise GateFileError(f"{w}.items must be a list")
    for it in g["items"]:
        _keys(it, _ITEM_REQUIRED, f"{w}.items[]", _ITEM_OPTIONAL)
        iid = it["id"]
        if not (isinstance(iid, str) and _ITEM_ID_RE.fullmatch(iid)):
            raise GateFileError(f"{w}: item id {iid!r} must match {_ITEM_ID_RE.pattern}")
        if iid in ids:
            raise GateFileError(f"{w}: item {iid!r} listed twice")
        ids.add(iid)
        _ref(it["ref"], f"{w} item {iid}.ref")
        _str(it["scope"], f"{w} item {iid}.scope")
        ev = _nodeids(it["evidence"], f"{w} item {iid}.evidence")
        rg = it["red_gate"]
        if rg is not None:
            _keys(rg, _RED_GATE_KEYS, f"{w} item {iid}.red_gate")
            if not (isinstance(rg["commit"], str) and _SHA_RE.fullmatch(rg["commit"])):
                raise GateFileError(f"{w} item {iid}.red_gate.commit must be a 40-hex sha, got {rg['commit']!r}")
            _nodeid(rg["nodeid"], f"{w} item {iid}.red_gate.nodeid")
            if rg["nodeid"] not in ev:
                raise GateFileError(f"{w} item {iid}: red_gate.nodeid must be one of its evidence nodeids")
        if "site" in it:
            site = _keys(it["site"], _SITE_KEYS, f"{w} item {iid}.site")
            for sk in _SITE_KEYS:
                _str(site[sk], f"{w} item {iid}.site.{sk}")
        if "source" in it:
            _str(it["source"], f"{w} item {iid}.source")

    if not isinstance(g["retired"], list):
        raise GateFileError(f"{w}.retired must be a list")
    seen: set[tuple[str, str]] = set()
    for r in g["retired"]:
        _keys(r, _RETIRED_KEYS, f"{w}.retired[]")
        kind, rkey = r["kind"], r["key"]
        if kind not in _RETIRED_KINDS:
            raise GateFileError(f"{w}.retired[].kind must be one of {sorted(_RETIRED_KINDS)}, got {kind!r}")
        known = {"item": ids, "arm": names, "function": {f"{a}::{b}" for a, b in fkeys}}[kind]
        if rkey not in known:
            raise GateFileError(f"{w}.retired[]: {kind} {rkey!r} is not in this gate (a retired record names an entry)")
        if (kind, rkey) in seen:
            raise GateFileError(f"{w}.retired[]: {kind} {rkey!r} retired twice")
        seen.add((kind, rkey))
        _str(r["reason"], f"{w}.retired[].reason")
        _ref(r["ref"], f"{w}.retired[].ref")
        _date(r["date"], f"{w}.retired[].date")
    return Gate(key, parse_version(key, f"gate key {key}"), g)


# ── rule 3a: an evidence test, read statically ───────────────────────────────


def _module_bindings(tree: ast.Module) -> dict[str, list[ast.expr]]:
    """{name: [assigned value]} for module-level assignments (a decorator alias resolves through these)."""
    out: dict[str, list[ast.expr]] = {}
    for st in tree.body:
        if isinstance(st, ast.Assign):
            for t in st.targets:
                if isinstance(t, ast.Name):
                    out.setdefault(t.id, []).append(st.value)
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
            out.setdefault(st.target.id, []).append(st.value)
    return out


def _denied_refs(expr: ast.AST, deny: frozenset[str], bindings: dict[str, list[ast.expr]]) -> set[str]:
    """Every denied name referenced in ``expr``, resolving names through module-level assignments."""
    hits: set[str] = set()
    seen: set[str] = set()
    stack: list[ast.AST] = [expr]
    while stack:
        node = stack.pop()
        for n in ast.walk(node):
            if isinstance(n, ast.Attribute) and n.attr in deny:
                hits.add(n.attr)
            elif isinstance(n, ast.Name):
                if n.id in deny:
                    hits.add(n.id)
                elif n.id in bindings and n.id not in seen:
                    seen.add(n.id)
                    stack.extend(bindings[n.id])
    return hits


def _is_mark(e: ast.expr) -> bool:
    """A literal pytest mark: ``pytest.mark.x``, ``pytest.mark.x(...)`` or ``mark.x[...]`` (from pytest import mark)."""
    if isinstance(e, ast.Call):
        e = e.func
    chain: list[str] = []
    while isinstance(e, ast.Attribute):
        chain.append(e.attr)
        e = e.value
    if not isinstance(e, ast.Name):
        return False
    return (e.id == "pytest" and chain[-1:] == ["mark"] and len(chain) >= 2) or (e.id == "mark" and len(chain) >= 1)


def _pytestmark_problems(body: list[ast.stmt], where: str, bindings: dict[str, list[ast.expr]]) -> list[str]:
    out: list[str] = []
    for st in body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(st, ast.Assign):
            targets, value = st.targets, st.value
        elif isinstance(st, (ast.AnnAssign, ast.AugAssign)):
            targets, value = [st.target], st.value
        if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in targets) or value is None:
            continue
        marks = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
        if not all(_is_mark(m) for m in marks):
            out.append(f"{where}: pytestmark is not a literal mark or list of marks (fails closed)")
        hits = _denied_refs(value, DENY_MARKS, bindings)
        if hits:
            out.append(f"{where}: pytestmark carries {sorted(hits)}")
    return out


def _level_statements(body: list[ast.stmt]) -> list[ast.stmt]:
    """The statements that run at this level: everything but nested def/class bodies (their decorators stay)."""
    return [st for st in body if not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]


def _bound_names(body: list[ast.stmt]) -> dict[str, int]:
    """How many times each name is bound at this level (def, class, assignment target, import)."""
    out: dict[str, int] = {}

    def add(name: str) -> None:
        out[name] = out.get(name, 0) + 1

    for st in body:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add(st.name)
        elif isinstance(st, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = st.targets if isinstance(st, ast.Assign) else [st.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        add(n.id)
        elif isinstance(st, (ast.Import, ast.ImportFrom)):
            for a in st.names:
                add((a.asname or a.name).split(".")[0])
    return out


def _asserts(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for n in ast.walk(fn):
        if isinstance(n, ast.Assert):
            return True
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
            if name == "raises" or "assert" in name.lower():
                return True
    return False


@dataclass
class _Located:
    tree: ast.Module
    cls: ast.ClassDef | None
    fn: ast.FunctionDef | ast.AsyncFunctionDef
    bindings: dict[str, list[ast.expr]]


def _locate(text: str, rel: str, cls_name: str | None, fn_name: str) -> tuple[_Located | None, list[str]]:
    try:
        tree = ast.parse(text, filename=rel)
    except (SyntaxError, ValueError) as e:
        return None, [f"{rel}: does not parse ({e})"]
    bindings = _module_bindings(tree)
    scope = tree.body
    cls: ast.ClassDef | None = None
    problems: list[str] = []
    if cls_name is not None:
        if _bound_names(tree.body).get(cls_name, 0) != 1:
            return None, [f"{rel}: class {cls_name} must be bound exactly once at module level"]
        found = [st for st in tree.body if isinstance(st, ast.ClassDef) and st.name == cls_name]
        if not found:
            return None, [f"{rel}: no top-level class {cls_name}"]
        cls = found[0]
        scope = cls.body
        if any(isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)) and st.name == "__init__" for st in scope):
            problems.append(f"{rel}::{cls_name}: has __init__ (pytest does not collect it)")
        if "__test__" in _bound_names(scope):
            problems.append(f"{rel}::{cls_name}: sets __test__ (it may hide the class from collection)")
    if _bound_names(scope).get(fn_name, 0) != 1:
        return None, problems + [f"{rel}: {fn_name} must be bound exactly once at its level (a def, not a re-export)"]
    defs = [st for st in scope if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)) and st.name == fn_name]
    if not defs:
        return None, problems + [f"{rel}: {fn_name} is not a def at its level"]
    return _Located(tree, cls, defs[0], bindings), problems


def evidence_problems(root: Path, nodeid: str) -> list[str]:
    """Rule 3a for one nodeid at the working tree; [] when it is a collectable, unskipped, asserting test."""
    m = _NODEID_RE.fullmatch(nodeid)
    if m is None:
        return [f"{nodeid}: not a nodeid tests/**/test_*.py::[Test*::]test_*"]
    rel, cls_name, fn_name = m.group(1), m.group(2), m.group(3)
    if rel in FAST_SUITE_IGNORED:
        return [f"{nodeid}: {rel} is outside the fast suite"]
    path = root / rel
    if not path.is_file():
        return [f"{nodeid}: {rel} does not exist"]
    loc, problems = _locate(path.read_text(encoding="utf-8"), rel, cls_name, fn_name)
    if loc is None:
        return [f"{nodeid}: {p}" for p in problems]
    b = loc.bindings
    problems += _pytestmark_problems(loc.tree.body, f"{rel} (module)", b)
    for st in _level_statements(loc.tree.body):
        if not (
            isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in st.targets)
        ):
            hits = _denied_refs(st, DENY_BODY, {})
            if hits:
                problems.append(f"{rel} (module level) references {sorted(hits)}")
    if loc.cls is not None:
        for d in loc.cls.decorator_list:
            hits = _denied_refs(d, DENY_MARKS, b)
            if hits:
                problems.append(f"{rel}::{loc.cls.name} decorator carries {sorted(hits)}")
        problems += _pytestmark_problems(loc.cls.body, f"{rel}::{loc.cls.name}", b)
        for st in _level_statements(loc.cls.body):
            hits = _denied_refs(st, DENY_BODY, {})
            if hits and not (
                isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in st.targets)
            ):
                problems.append(f"{rel}::{loc.cls.name} (class level) references {sorted(hits)}")
    for d in loc.fn.decorator_list:
        hits = _denied_refs(d, DENY_MARKS, b)
        if hits:
            problems.append(f"decorator carries {sorted(hits)}")
    for st in loc.fn.body:
        hits = _denied_refs(st, DENY_BODY, {})
        if hits:
            problems.append(f"body references {sorted(hits)}")
    if not _asserts(loc.fn):
        problems.append("body never asserts (no assert, pytest.raises or assert* call)")
    return [f"{nodeid}: {p}" for p in problems]


# ── rule 3b: the red gate ────────────────────────────────────────────────────


def _is_strict_xfail(e: ast.expr, bindings: dict[str, list[ast.expr]], depth: int = 0) -> bool:
    if isinstance(e, ast.Name) and e.id in bindings and depth < 3:
        return any(_is_strict_xfail(v, bindings, depth + 1) for v in bindings[e.id])
    if isinstance(e, (ast.List, ast.Tuple)):
        return any(_is_strict_xfail(x, bindings, depth) for x in e.elts)
    if not isinstance(e, ast.Call):
        return False
    f = e.func
    name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
    return name == "xfail" and any(
        k.arg == "strict" and isinstance(k.value, ast.Constant) and k.value.value is True for k in e.keywords
    )


def red_gate_problems(root: Path, item_id: str, rg: dict) -> list[str]:
    commit, nodeid = rg["commit"], rg["nodeid"]
    try:
        git(root, "cat-file", "-e", f"{commit}^{{commit}}")
    except GitUnavailable:
        return [f"item {item_id}: red_gate commit {commit[:12]} does not exist in this clone"]
    try:
        git(root, "merge-base", "--is-ancestor", commit, "HEAD")
    except GitUnavailable:
        return [f"item {item_id}: red_gate commit {commit[:12]} is not an ancestor of HEAD (a squash merge drops it)"]
    m = _NODEID_RE.fullmatch(nodeid)
    assert m is not None
    rel, cls_name, fn_name = m.group(1), m.group(2), m.group(3)
    text = show(root, commit, rel)
    if not text:
        return [f"item {item_id}: {rel} does not exist at red_gate commit {commit[:12]}"]
    loc, problems = _locate(text, rel, cls_name, fn_name)
    if loc is None:
        return [f"item {item_id}: at {commit[:12]}, {p}" for p in problems]
    marks = list(loc.fn.decorator_list)
    if loc.cls is not None:
        marks += loc.cls.decorator_list
    if not any(_is_strict_xfail(d, loc.bindings) for d in marks):
        return [f"item {item_id}: at {commit[:12]}, {nodeid} did not carry xfail(strict=True) — not a red gate"]
    return []


# ── rule 4: functions ────────────────────────────────────────────────────────


@dataclass
class FunctionStatus:
    key: tuple[str, str]
    span: int | None  # None: missing; -1: ambiguous
    pin: int | None
    limit: int
    descendant_of: str | None = None

    @property
    def ok(self) -> bool:
        return self.span is not None and 0 < self.span <= self.limit and self.pin is None

    def describe(self) -> str:
        span = "NOT FOUND" if self.span is None else "AMBIGUOUS" if self.span == -1 else f"{self.span} lines"
        pin = f"pinned {self.pin}" if self.pin is not None else "unpinned"
        via = f" (descendant of {self.descendant_of})" if self.descendant_of else ""
        return f"{self.key[0]}::{self.key[1]} = {span} ({pin}; max {self.limit}){via}"


def _measure(root: Path, rel: str, qualname: str) -> int | None:
    path = root / rel
    if not path.is_file():
        return None
    try:
        nodes = FL.defs_in(path.read_text(encoding="utf-8"), rel).get(qualname, [])
    except (SyntaxError, ValueError):
        return None
    if len(nodes) > 1:
        return -1
    return FL.span(nodes[0]) if nodes else None


def descendants(baseline: FL.Baseline, roots: set[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """{key: root} for every function split or moved out of a root, transitively (baseline exceptions)."""
    children: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for x in baseline.exceptions:
        for kind in ("split_from", "moved_from"):
            if kind in x:
                src = (x[kind]["file"], x[kind]["qualname"])
                children.setdefault(src, set()).add((x["file"], x["qualname"]))
    out: dict[tuple[str, str], str] = {}
    for r in roots:
        stack = [r]
        while stack:
            for c in sorted(children.get(stack.pop(), ())):
                if c not in roots and c not in out:
                    out[c] = f"{r[0]}::{r[1]}"
                    stack.append(c)
    return out


def module_statement_count(root: Path, rel: str) -> int:
    """NIT-1: module-level and class-body statements that are not def/class/import/docstring."""
    path = root / rel
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return -1
    n = 0
    stack: list[list[ast.stmt]] = [tree.body]
    while stack:
        for st in stack.pop():
            if isinstance(st, ast.ClassDef):
                stack.append(st.body)
            elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom)):
                continue
            elif isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) and isinstance(st.value.value, str):
                continue
            else:
                n += 1
    return n


# ── rule 5: arms ─────────────────────────────────────────────────────────────


def golden_arms(text: str) -> set[str] | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    arms = data.get("arms") if isinstance(data, dict) else None
    return set(arms) if isinstance(arms, dict) else None


# ── rule 7: history ──────────────────────────────────────────────────────────


def _view_v1(g: dict) -> dict:
    """The immutable and append-only fields of one gate, as format 1 defines them. Lenient: unknown keys are
    ignored and no value is re-validated, so a later key addition or a tightened validator never fails an old
    version. A missing or mistyped field it needs raises KeyError/TypeError/AttributeError."""
    return {
        "decided": g["decided"],
        "source": g["source"],
        "functions": {(f["file"], f["qualname"]): f["max_lines"] for f in g["functions"]},
        "debt_snapshot": [tuple(p) for p in g["debt_snapshot"]],
        "accepted_new_debt": [tuple(d.get(k) for k in sorted(_DEBT_KEYS)) for d in g["accepted_new_debt"]],
        "retired": [tuple(r.get(k) for k in sorted(_RETIRED_KEYS)) for r in g["retired"]],
        "golden": g["arms"]["golden"],
        "golden_test": g["arms"]["golden_test"],
        "arms": {a["arm"]: a["scope"] for a in g["arms"]["required"]},
        "items": {
            it["id"]: {k: json.dumps(it.get(k), sort_keys=True) for k in ("ref", "scope", "site", "source")}
            for it in g["items"]
        },
    }


#: How to read the immutable view of a gate written in each format. A version is compared with HEAD through ITS
#: OWN format's view (applied to both sides), so HEAD may add keys without failing the history.
HISTORY_VIEWS: dict[int, Callable[[dict], dict]] = {1: _view_v1}


def history_views(text: str) -> dict[str, dict]:
    """{gate key: view} of one historical version, read by its own ``gate_format_version``. Raises GateFileError."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise GateFileError(f"not JSON: {e}") from e
    fmt = data.get("gate_format_version") if isinstance(data, dict) else None
    view = HISTORY_VIEWS.get(fmt) if isinstance(fmt, int) and not isinstance(fmt, bool) else None
    if view is None:
        raise GateFileError(f"gate_format_version {fmt!r} has no history reader")
    if not isinstance(data.get("gates"), dict):
        raise GateFileError("gates is not an object")
    try:
        return {k: {"fmt": fmt, **view(g)} for k, g in data["gates"].items()}
    except (KeyError, TypeError, AttributeError) as e:
        raise GateFileError(f"a field format {fmt} defines is missing or mistyped ({e!r})") from e


def compare_views(key: str, o: dict, h: dict, at: str) -> list[str]:
    out: list[str] = []
    w = f"gate {key}"
    for k in ("decided", "source", "golden", "golden_test"):
        if h[k] != o[k]:
            out.append(f"{w}: `{k}` changed ({at}); it is immutable")
    if h["debt_snapshot"] != o["debt_snapshot"]:
        out.append(f"{w}: debt_snapshot changed ({at}); it is immutable (record accepted_new_debt)")
    for k in ("retired", "accepted_new_debt"):
        if problem := append_only_problem(base=o[k], head=h[k], what=f"{w}: `{k}` is"):
            out.append(f"{problem} ({at})")
    for fk, n in o["functions"].items():
        if fk not in h["functions"]:
            out.append(f"{w}: function {fk[0]}::{fk[1]} was removed ({at}); retire it instead")
        elif h["functions"][fk] > n:
            out.append(f"{w}: {fk[0]}::{fk[1]} max_lines rose {n} -> {h['functions'][fk]} ({at}); it may only fall")
    for arm, scope in o["arms"].items():
        if arm not in h["arms"]:
            out.append(f"{w}: arm {arm!r} was removed ({at}); retire it instead")
        elif h["arms"][arm] != scope:
            out.append(f"{w}: arm {arm!r} scope changed ({at}); it is immutable")
    for iid, fields in o["items"].items():
        if iid not in h["items"]:
            out.append(f"{w}: item {iid} was removed ({at}); retire it instead")
            continue
        for k, v in fields.items():
            if h["items"][iid][k] != v:
                out.append(f"{w}: item {iid} `{k}` changed ({at}); it is immutable")
    return out


def history_problems(root: Path, head: list[Gate]) -> list[str]:
    shas = git(
        root,
        "log",
        "--first-parent",
        "--diff-merges=first-parent",
        "--no-patch",
        "--format=%H",
        "HEAD",
        "--",
        GATES_REL,
    ).split()
    now = {g.key: g.raw for g in head}
    out: list[str] = []
    for sha in shas:
        text = show(root, sha, GATES_REL)
        if not text:
            out.append(f"{GATES_REL} is absent at first-parent commit {sha[:12]} (deleted on main's history)")
            continue
        try:
            old = history_views(text)
        except GateFileError as e:
            out.append(f"{GATES_REL} at first-parent commit {sha[:12]} is malformed ({e}) — a failure, never a skip")
            continue
        at = f"since {sha[:12]}"
        for key, o in old.items():
            if key not in now:
                out.append(f"gate {key} was removed ({at}); a gate is never removed")
                continue
            out += compare_views(key, o, HISTORY_VIEWS[o["fmt"]](now[key]), at)
    return out


# ── rule 8: the bump diff ────────────────────────────────────────────────────


def diff_base(root: Path) -> str:
    """What this change is judged against, for rule 8 and the close-out: ``HEAD^1`` on a ``push`` ONLY (pushes reach
    only main, through a merged PR; never ``push_base``, which calls ``gh api``). The repo also allows REBASE merges:
    a rebase-merged ``[bump, x]`` lands as two first-parent commits, and the push run of ``x`` reads CLOSED, but the
    PR's own run (merge-base, ACTIVE) is the required gate, so that is covered. Only a direct multi-commit push that
    skips a PR is not (a residual, M56). A ``v*`` TAG push (release.yml, #1372) is also a ``push``: on the
    bump commit it reads ACTIVE as on the PR; on a later commit it reads CLOSED (print-only), which is harmless
    because the bump PR enforced the gate (docs/plans/archive/release_pipeline.md, SF4). Every other event, local runs,
    pull requests, ``schedule`` and ``workflow_dispatch`` included, uses the merge-base with origin/main (then
    main): a dispatch can run on any branch head, and judging it against its own parent let a branch whose first
    commit bumped the version read as already released (review SF1). On main the merge-base is HEAD itself."""
    event = os.environ.get("GITHUB_EVENT_NAME") or ""
    if event == "push":
        return git(root, "rev-parse", "HEAD^1").strip()
    for ref in ("origin/main", "main"):
        try:
            mb = git(root, "merge-base", ref, "HEAD").strip()
        except GitUnavailable:
            continue
        if mb:
            return mb
    raise GitUnavailable("no origin/main or main to find the merge-base with")


def released_at(root: Path, base: str) -> Version | None:
    """What ``base`` had released: the LOWER of its newest released ``## [X.Y.Z]`` CHANGELOG header and its
    ``pyproject.toml`` version (both must say so); None when either cannot be read."""
    found = re.findall(r"^## \[(\d[^\]]*)\]", show(root, base, "CHANGELOG.md"), re.M)
    try:
        project = tomllib.loads(show(root, base, "pyproject.toml")).get("project", {})
        if not found:
            return None
        return min(parse_version(found[0], "CHANGELOG at the base"), parse_version(project.get("version"), "pyproject"))
    except (GateFileError, tomllib.TOMLDecodeError, AttributeError):  # AttributeError: `project` is not a table
        return None


def closed_gate_keys(root: Path, gates: list[Gate]) -> set[str]:
    """The gates released before this change (the close-out). Raises GitUnavailable."""
    released = released_at(root, diff_base(root))
    return {g.key for g in gates if released is not None and released >= g.version}


def bump_problems(root: Path, base: str, gates: list[Gate], head_version: Version) -> list[str]:
    sources, errors = version_sources(lambda rel: show(root, base, rel))
    base_version, verrors = max_version(sources)
    if base_version is None:
        return [f"the version at the base {base[:12]} is unreadable ({'; '.join(errors + verrors)})"]
    out: list[str] = []
    for g in gates:
        if base_version < g.version <= head_version:
            now = (root / GATES_REL).read_text(encoding="utf-8") if (root / GATES_REL).is_file() else ""
            if show(root, base, GATES_REL) != now:
                out.append(
                    f"this diff raises the version {_fmt_version(base_version)} -> {_fmt_version(head_version)} "
                    f"across gate {g.key} AND changes {GATES_REL}: the gate file may not change in the bump diff "
                    "(no last-minute satisfy, retire or accept)"
                )
    return out


# ── rule 9: wiring ───────────────────────────────────────────────────────────


def wiring_problems(root: Path) -> list[str]:
    try:
        import yaml
    except ImportError:
        return ["pyyaml is not installed: the release gate's CI wiring cannot be verified (the lint job installs it)"]
    try:
        doc = yaml.safe_load((root / WORKFLOW_REL).read_text(encoding="utf-8"))
        jobs = doc["jobs"]
        lint, twin = jobs[LINT_JOB], jobs[TWIN_JOB]
        lsteps, tsteps = lint["steps"], twin["steps"]
    except (OSError, yaml.YAMLError, KeyError, TypeError) as e:
        return [f"{WORKFLOW_REL}: cannot read the `{LINT_JOB}`/`{TWIN_JOB}` steps ({e})"]
    out: list[str] = []
    if "defaults" in doc:
        out.append(f"{WORKFLOW_REL}: workflow-level `defaults` is banned (it can rewrite every step's shell)")

    def run(s: dict) -> str:
        r = s.get("run") if isinstance(s, dict) else None
        return r if isinstance(r, str) else ""

    for name, job, steps in ((LINT_JOB, lint, lsteps), (TWIN_JOB, twin, tsteps)):
        where = f"{WORKFLOW_REL} `{name}`"
        if not isinstance(job, dict) or not isinstance(steps, list) or not all(isinstance(s, dict) for s in steps):
            out.append(f"{where}: the job and its steps must be mappings")
            continue
        for k in ("if", "continue-on-error", "defaults"):
            if k in job:
                out.append(f"{where}: a job-level `{k}` is banned on a job that carries the release gate")
        checkout = [s for s in steps if str(s.get("uses", "")).startswith("actions/checkout@")]
        if not checkout or any((s.get("with") or {}).get("fetch-depth") != 0 for s in checkout):
            out.append(f"{where}: every checkout needs `with: fetch-depth: 0` (the history rule needs main's history)")
        want = LINT_RUN if name == LINT_JOB else TWIN_RUN
        gate = [i for i, s in enumerate(steps) if "lint_release_gates.py" in run(s)]
        if len(gate) != 1 or run(steps[gate[0]]).strip() != want:
            out.append(f"{where}: needs exactly one step whose run is `{want}` (no `|| true`)")
            continue
        g = steps[gate[0]]
        if set(g) != {"name", "run"}:
            out.append(f"{where}: the release-gate step may carry only `name` and `run` (found {sorted(set(g))})")
        if name == TWIN_JOB:
            suite = [i for i, s in enumerate(steps) if "--cov" in run(s) and "pytest" in run(s)]
            if len(suite) != 1 or run(steps[suite[0]]).count(SUITE_FLAG) != 1:
                out.append(f"{where}: the fast suite (the one `--cov` pytest step) must pass `{SUITE_FLAG}` once")
            elif gate[0] < suite[0]:
                out.append(f"{where}: the release-gate step must come after the fast suite that writes {JUNIT_FILE}")
            others = [
                s.get("name") for i, s in enumerate(steps) if i not in (gate[0], *suite) and JUNIT_FILE in json.dumps(s)
            ]
            if others:
                out.append(f"{where}: only the fast suite and the release-gate step may name {JUNIT_FILE} ({others})")
    return out


# ── rule 6: the runtime twin ─────────────────────────────────────────────────


def _junit_cases(xml_path: Path) -> dict[tuple[str, str], str]:
    """{(classname, name): outcome} from a JUnit report; raises Unanswerable when it cannot be read."""
    try:
        tree = ElementTree.parse(xml_path)
    except (OSError, ElementTree.ParseError) as e:
        raise Unanswerable(f"JUnit report {xml_path} unreadable ({e}) — the fast suite must write it") from e
    out: dict[tuple[str, str], str] = {}
    for tc in tree.iter("testcase"):
        outcome = "passed"
        for tag in ("failure", "error", "skipped"):
            if tc.find(tag) is not None:
                outcome = tag
                break
        out[(tc.get("classname", ""), tc.get("name", ""))] = outcome
    return out


def junit_problems(cases: dict[tuple[str, str], str], wanted: dict[str, str]) -> list[str]:
    """``wanted``: {nodeid (an optional ``[param]`` selects one case): why}. Every case must be present and passed."""
    from _pytest.junitxml import mangle_test_address

    out: list[str] = []
    for nodeid, why in sorted(wanted.items()):
        parts = mangle_test_address(nodeid)
        cls, name = ".".join(parts[:-1]), parts[-1]
        hits = {
            k: v
            for k, v in cases.items()
            if k[0] == cls and (k[1] == name or ("[" not in name and k[1].startswith(name + "[")))
        }
        if not hits:
            out.append(
                f"{nodeid} ({why}): not in the fast suite's JUnit report (skipped at collection, deselected, renamed?)"
            )
        for (_, case), outcome in sorted(hits.items()):
            if outcome != "passed":
                out.append(f"{nodeid} ({why}): case {case} {outcome} in the fast suite (a conftest skip or xfail?)")
    anchor = ".".join(mangle_test_address(ANCHOR_TEST_FILE + "::x")[:-1])
    if not any(k[0] == anchor and v == "passed" for k, v in cases.items()):
        out.append(f"the JUnit report carries no passed case of {ANCHOR_TEST_FILE}: it is not the fast suite's report")
    return out


def wanted_runtime(gates: list[Gate], golden_carried: dict[str, set[str] | None]) -> dict[str, str]:
    out: dict[str, str] = {}
    for g in gates:
        for it in g.items:
            if it["evidence"] and it["red_gate"] is not None:
                for n in it["evidence"]:
                    out[n] = f"gate {g.key} item {it['id']}"
        carried = golden_carried.get(g.key) or set()
        for a in g.arms["required"]:
            if a["evidence"] and a["arm"] in carried:
                out[f"{g.arms['golden_test']}[{a['arm']}]"] = f"gate {g.key} arm {a['arm']}"
                for n in a["evidence"]:
                    out[n] = f"gate {g.key} arm {a['arm']}"
    return out


# ── main ─────────────────────────────────────────────────────────────────────


def _git_state(root: Path) -> str | None:
    """None when git can answer the history questions; else the reason it cannot."""
    try:
        if git(root, "rev-parse", "--is-shallow-repository").strip() == "true":
            return "the clone is shallow (fetch-depth: 0 is required)"
        git(root, "rev-parse", "--verify", "HEAD")
    except GitUnavailable as e:
        return str(e)
    return None


def _git_rule(name: str, fn: Callable[..., list], *args: object) -> list:
    """Run a git-backed rule. Git failing mid-rule is exit 2 in CI (raised as Unanswerable), an INFO skip locally."""
    try:
        return fn(*args)
    except GitUnavailable as e:
        if _in_ci():
            raise Unanswerable(f"{name}: git could not answer ({e})") from e
        print(f"INFO: release gates: {name} skipped ({e})")
        return []


def _read_golden(root: Path, rel: str) -> set[str] | None:
    p = root / rel
    return golden_arms(p.read_text(encoding="utf-8")) if p.is_file() else None


def check(root: Path) -> tuple[list[str], int]:
    """Every rule but the runtime twin. Returns (violations, exit code when not 0/1)."""
    started = time.monotonic()
    failures: list[str] = []
    try:
        gates = parse_gates((root / GATES_REL).read_text(encoding="utf-8"))
    except OSError as e:
        return [f"{GATES_REL}: unreadable ({e})"], 0
    except GateFileError as e:
        return [f"{GATES_REL}: {e}"], 0

    sources, errors = version_sources(
        lambda rel: (root / rel).read_text(encoding="utf-8") if (root / rel).is_file() else ""
    )
    version, verrors = max_version(sources)
    failures += errors + verrors
    shown = " / ".join(f"{k} {v}" for k, v in sources.items())

    git_reason = _git_state(root)
    if git_reason is not None:
        if _in_ci():
            print(f"ERROR: release gates: git cannot answer in CI ({git_reason})", file=sys.stderr)
            return failures, 2
        print(
            f"INFO: release gates: git-backed rules (red gates, close-out, history, bump diff) skipped ({git_reason})"
        )

    try:
        baseline = FL.parse_baseline((root / FL.BASELINE_REL).read_text(encoding="utf-8"))
    except (OSError, FL.BaselineError) as e:
        return failures + [f"{FL.BASELINE_REL}: {e} — the function conditions cannot be measured"], 0

    closed_keys: set[str] = set()  # without git, nothing is closed: every rule stays enforced
    if git_reason is None:
        found = _git_rule("the close-out check", lambda: [closed_gate_keys(root, gates)])
        closed_keys = found[0] if found else set()

    for g in gates:
        # S1 close-out: a gate whose version was RELEASED BEFORE this change (a released CHANGELOG header at the
        # base, not at HEAD: the bump diff itself adds the header and stays fully gated) has done its job. Its item,
        # arm, function and new-debt rules then only print, so a later regrowth or an evidence refactor cannot
        # brick CI for good; its shape, history, bump and wiring rules stay enforced.
        closed = g.key in closed_keys
        active = version is not None and version >= g.version
        # A version that cannot be read is treated as ACTIVE: it fails closed.
        active = active or version is None
        state = (
            "CLOSED (released before this change; item/arm/function/debt rules print only)"
            if closed
            else ("ACTIVE" if active else "inactive")
        )
        vtxt = _fmt_version(version) if version is not None else "UNREADABLE"
        print(f"release gate {g.key}: {state} (version {vtxt}: max of {shown or 'nothing readable'})")
        retired = g.retired
        gate_fail: list[str] = []
        outstanding: list[str] = []

        # rule 3: items
        satisfied = 0
        for it in g.items:
            iid = it["id"]
            if ("item", iid) in retired:
                continue
            ev, rg = it["evidence"], it["red_gate"]
            if ev and rg is None:  # (a red_gate with no evidence is a shape error: its nodeid must be evidence)
                gate_fail.append(f"item {iid}: evidence recorded without a red_gate (rule 3b)")
            if ev and rg is not None:
                probs = [p for n in ev for p in evidence_problems(root, n)]
                if git_reason is None:
                    probs += _git_rule(f"item {iid}'s red-gate check", red_gate_problems, root, iid, rg)
                gate_fail += [f"item {iid}: {p}" for p in probs]
                if not probs:
                    satisfied += 1
                    continue
            outstanding.append(iid)
        n_items = sum(1 for it in g.items if ("item", it["id"]) not in retired)

        # rule 4: functions
        roots = {(f["file"], f["qualname"]) for f in g.functions}
        # A retired function leaves the gate WITH its descendants (owner decision 2026-10-16, #1323): whatever
        # replaces it (an item) carries its own bound, so its split-outs are not held to the 200 ceiling and may be
        # accepted as new debt like any other function.
        live_roots = {k for k in roots if ("function", f"{k[0]}::{k[1]}") not in retired}
        statuses = [
            FunctionStatus(k, _measure(root, k[0], k[1]), baseline.entries.get(k), f["max_lines"])
            for f in g.functions
            if ("function", f"{f['file']}::{f['qualname']}") not in retired
            for k in [(f["file"], f["qualname"])]
        ]
        desc = descendants(baseline, live_roots)
        statuses += [
            FunctionStatus(k, _measure(root, k[0], k[1]), baseline.entries.get(k), MAX_LINES_CEILING, via)
            for k, via in sorted(desc.items())
        ]
        for s in statuses:
            if s.span is None and s.descendant_of is not None:
                continue  # a descendant that no longer exists carries no debt
            print(f"  function: {s.describe()}" + ("" if s.ok else "  OUTSTANDING"))
        bad_functions = [s for s in statuses if not s.ok and not (s.span is None and s.descendant_of is not None)]
        for rel in sorted({k[0] for k in roots}):
            print(f"  module-level statements outside defs: {rel} = {module_statement_count(root, rel)}")

        snapshot = {tuple(p) for p in g.raw["debt_snapshot"]}
        accepted = {(d["file"], d["qualname"]) for d in g.raw["accepted_new_debt"]}
        # Every root, retired ones included: a gated function is never accepted as new debt (#1323 keeps this).
        for k in sorted(accepted & (roots | set(desc))):
            gate_fail.append(
                f"accepted_new_debt names {k[0]}::{k[1]}, a gated function or one split/moved out of one — that rule "
                "has no exception"
            )
        new_debt = sorted(k for k in baseline.entries if k not in snapshot and k not in accepted)

        # rule 5: arms
        carried = _read_golden(root, g.arms["golden"])
        arms_ok = 0
        arms_out: list[str] = []
        for a in g.arms["required"]:
            if ("arm", a["arm"]) in retired:
                continue
            has = carried is not None and a["arm"] in carried
            if a["evidence"] and not has:
                gate_fail.append(f"arm {a['arm']}: evidence recorded but {g.arms['golden']} carries no such arm")
            if a["evidence"] and has:
                probs = [p for n in a["evidence"] for p in evidence_problems(root, n)]
                gate_fail += [f"arm {a['arm']}: {p}" for p in probs]
                if not probs:
                    arms_ok += 1
                    continue
            arms_out.append(a["arm"])
        n_arms = sum(1 for a in g.arms["required"] if ("arm", a["arm"]) not in retired)

        print(
            f"  arms: {arms_ok}/{n_arms} satisfied (golden carries {sorted(carried) if carried is not None else 'NOTHING'})"
            + (f"; outstanding: {', '.join(arms_out)}" if arms_out else "")
        )
        print(
            f"  items: {satisfied}/{n_items} satisfied"
            + (f"; outstanding: {', '.join(sorted(outstanding))}" if outstanding else "")
        )
        print(
            f"  new debt outside debt_snapshot: {len(new_debt)} unaccepted"
            + (f" ({', '.join(f'{a}::{b}' for a, b in new_debt)})" if new_debt else "")
            + f"; {len(accepted)} accepted"
        )
        for r in g.raw["retired"]:
            print(f"  retired {r['kind']} {r['key']} ({r['date']}, {r['ref']}): {r['reason']}")
        over = ", ".join(f"{s.key[1]} {s.span}" for s in bad_functions)
        print(
            f"release gate {g.key}: {state} (version {vtxt}); {len(outstanding)}/{n_items} items outstanding, "
            f"{len(bad_functions)} functions not done ({over or 'none'}), {arms_ok}/{n_arms} arms; "
            f"{len(new_debt)} unaccepted new debt; {len(retired)} retired"
        )
        if active:
            gate_fail += [f"item {i} is outstanding" for i in sorted(outstanding)]
            gate_fail += [f"arm {a} is outstanding" for a in arms_out]
            gate_fail += [f"function {s.describe()} is not done" for s in bad_functions]
            gate_fail += [
                f"new debt {a}::{b} is outside debt_snapshot with no accepted_new_debt record" for a, b in new_debt
            ]
        if closed:
            for f in gate_fail:
                print(f"  closed gate {g.key}, not enforced: {f}")
        else:
            failures += [f"gate {g.key}: {f}" for f in gate_fail]

    if git_reason is None:
        failures += _git_rule("the history rule", history_problems, root, gates)
        bases = _git_rule("the bump-diff rule", lambda: [diff_base(root)])
        if bases and version is not None:
            failures += _git_rule("the bump-diff rule", bump_problems, root, bases[0], gates, version)

    failures += wiring_problems(root)
    print(f"release gates: {time.monotonic() - started:.1f}s")
    return failures, 0


def check_junit(root: Path, xml_rel: str) -> tuple[list[str], int]:
    try:
        gates = parse_gates((root / GATES_REL).read_text(encoding="utf-8"))
    except (OSError, GateFileError) as e:
        return [f"{GATES_REL}: {e}"], 0
    try:
        cases = _junit_cases(root / xml_rel)
    except Unanswerable as e:
        print(f"ERROR: release gates: {e}", file=sys.stderr)
        return [], 2
    carried = {g.key: _read_golden(root, g.arms["golden"]) for g in gates}
    try:
        closed = closed_gate_keys(root, gates)
    except GitUnavailable as e:  # strict: a gate that cannot be shown closed stays enforced
        print(f"INFO: release gates (runtime twin): close-out unknown, every gate enforced ({e})")
        closed = set()
    wanted = wanted_runtime([g for g in gates if g.key not in closed], carried)
    print(
        f"release gates (runtime twin): {len(wanted)} evidence nodeids checked against {xml_rel} ({len(cases)} cases)"
    )
    problems = junit_problems(cases, wanted)
    late = wanted_runtime([g for g in gates if g.key in closed], carried)
    for p in [p for p in junit_problems(cases, late) if "not the fast suite's report" not in p]:
        print(f"  closed gate, not enforced: {p}")
    return problems + wiring_problems(root), 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = REPO_ROOT
    if argv[:1] == ["--junit"] and len(argv) == 2:
        failures, rc = check_junit(root, argv[1])
    elif not argv:
        try:
            failures, rc = check(root)
        except Unanswerable as e:
            print(f"ERROR: release gates: {e}", file=sys.stderr)
            return 2
    else:
        print("usage: lint_release_gates.py [--junit <fast-suite junit xml>]", file=sys.stderr)
        return 2
    sys.stdout.flush()
    if rc:
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return rc
    if failures:
        print("release gates FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("release gates: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())

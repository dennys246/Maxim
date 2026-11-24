#!/usr/bin/env python3
"""Motion-safety lint (#1496): two structural bans on the Reachy motion path.

1. **Clamp idiom.** ``max(-L, min(L, x))`` and ``min(L, max(-L, x))`` turn NaN into a finite limit,
   and ``np.clip`` passes NaN through unrefused, so a NaN from a YAML, a JSON pose file or an LLM argument became a full-range move
   (a NaN ``head_yaw`` swung the body 160 deg in 1 s). On the motion path every command clamp is
   ``maxim.hardware.controller.clamp_finite``, which refuses non-finite input. A clamp that is NOT a
   motion command (a measurement, a count) carries the pragma ``# motion-lint: measurement``.
2. **Raw SDK motion.** Any attribute reference on an SDK handle (``mini.``, ``_mini.``,
   ``reachy_mini.``) to a motion entry point, including an alias (``fn = self.mini.set_target``) and
   ``getattr(<handle>, "<name>")``, outside the sanctioned dispatch points. Replaces the inline CI
   grep (2026-08-07 orient workspace-safety fold), widened to ``scripts/`` and to more entry points.
3. **Auto-yaw left on** (#1499 PR 3, owner 2026-11-01). Every ``ReachyMini(`` call in ``src/`` and ``scripts/``,
   Python or shell, passes ``automatic_body_yaw=False``: the setting is daemon-wide and the SDK defaults it on, so
   one client built without it turns auto-yaw back on for every client. Python files are read as an AST (a call
   split across lines counts; a ``**kwargs`` splat cannot show the value, so it is flagged); shell files, which
   embed Python, by the call's bracketed text.
4. **Media left released** (#1712). A ``ReachyMini(`` call that can connect with ``media_backend="no_media"`` (that
   literal, a non-literal value, or a ``**kwargs`` splat) releases the daemon's media, daemon-wide and past the
   process, and only the SDK's ``__exit__`` re-acquires it. Its file must undo that: a ``with ReachyMini(...)``,
   a ``<handle>.__exit__(`` call on a name ending in ``mini``, or a call to ``release_mini(``
   (``scripts/orient_backbone/live_1_smoke.py``, which calls ``__exit__``). Python files are read as an AST, so a
   comment, a string or another object's ``__exit__`` does not count. The controller owns its own re-acquire (``_reacquire_media``, #1704) and is listed in
   ``MEDIA_UNDO_OWNERS``. A missing ``media_backend`` is the SDK default, which never releases.

Stated residuals (a line-based lint):
- a clamp or a handle split across lines;
- a one-sided ``max(-L, x)`` / ``min(L, x)`` bound (banning it would flag legitimate code), a bare
  ``clip(...)`` imported from numpy, and any clamp helper other than max/min/clip;
- any SDK handle bound to a name not ending in ``mini`` (``sdk = self.mini``, ``with ReachyMini() as
  robot``), or reached through a call, a subscript or parentheses (``self.mini()``, ``minis[0]``,
  ``ReachyMini().goto_target``, ``getattr(self, "mini")``), and ``getattr(handle, name)`` with a dynamic name;
- a ``ReachyMini`` reached under another name (``cls = ReachyMini; cls(...)``, ``from reachy_mini import
  ReachyMini as RM``) or built through ``functools.partial``;
- rule 4 is per file, not per call or per path: a file that undoes one handle passes for all of its calls, and a
  ``release_mini(`` call that is unreachable from the exit is not seen; a shell file is read as text, so there a
  mention in a comment or string counts as an undo, and only a literal ``no_media`` in the call marks it; a
  ``media_backend`` passed positionally (the SDK's tenth argument) is not seen;
- the canned or boolean SDK entry points ``wake_up``, ``goto_sleep``, ``enable/disable_motors``,
  ``enable/disable_wobbling``, ``set_automatic_body_yaw``, gravity compensation and ``cancel_move``
  (none takes a motion value; #1499's wider grep).

A single-line string or log message containing a banned shape is flagged too (reword it), and so is
``max(xs) - min(xs)`` on one line or any ``.clip(`` call; the pragma below covers a non-command clamp.

Motors 2+3 were destroyed by an unclamped pose (docs/agents/embodiment.md section 1).
Approach note: docs/plans/motion_finite_1496.md section 5. Exit 0 clean, 1 on any finding.
"""

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

# Paths (relative to the repo root) where the clamp idiom is banned.
CLAMP_SCOPE = (
    "src/maxim/embodied_runtime/movement.py",
    "src/maxim/motion/",
    "src/maxim/hardware/reachy/",
    "src/maxim/hardware/controller.py",
    "src/maxim/tools/reachy.py",
)
# Any nesting of max() and min() (whatever the bounds) and np.clip: each turns NaN into a bound.
CLAMP_PATTERNS = (
    re.compile(r"\bmax\(.*\bmin\("),
    re.compile(r"\bmin\(.*\bmax\("),
    re.compile(r"\b(?:np|numpy)\.clip\(|\.clip\("),
)
MEASUREMENT_PRAGMA = "# motion-lint: measurement"

RAW_SCOPE = ("src/maxim/", "scripts/")
_ENTRY_POINTS = (
    r"goto_target|set_target\w*|_goto_joint_positions|_set_joint_positions"
    r"|look_at_world|look_at_image|play_move|async_play_move"
)
# Any attribute chain ending in an SDK handle: self.mini, self._maxim.mini, a.b._mini, reachy_mini.
_HANDLE = r"(?<!\w)(?:\w+\.)*\w*mini\b"
RAW_PATTERNS = (
    re.compile(rf"{_HANDLE}\s*\.\s*(?:{_ENTRY_POINTS})\b"),
    re.compile(rf"getattr\(\s*{_HANDLE}\s*,\s*[\"'](?:{_ENTRY_POINTS})[\"']"),
)
# Sanctioned raw dispatch: whole files, or exact stripped lines within a file. Each with its reason.
RAW_ALLOW_FILES = {
    "src/maxim/hardware/reachy/controller.py": "the single clamped + locked dispatch point",
    "src/maxim/motion/movement.py": "move_head / move_antenna: clamped, last-mile-checked primitives",
    "scripts/orient_backbone/live_common.py": "live hardware harness; validates before its raw _goto (#1496)",
    "scripts/orient_backbone/live_1_smoke.py": "live hardware smoke; literal finite targets, maxim-import-free by design",
    "scripts/lint_motion_safety.py": "this lint's own pattern literals",
}
# Selfy's look-at dispatches through ReachyMiniController.look_at_pixel since #1499 (5.2).
RAW_ALLOW_LINES: dict[str, set[str]] = {}

AUTO_YAW_SCOPE = ("src/", "scripts/")
AUTO_YAW_SUFFIXES = (".py", ".sh")
_AUTO_YAW_OFF = re.compile(r"\bautomatic_body_yaw\s*=\s*False\b")
_REACHY_MINI_CALL = re.compile(r"\bReachyMini\(")
# Rule 4: files whose ReachyMini( calls may release the media and that re-acquire through their own seam.
MEDIA_UNDO_OWNERS = {
    "src/maxim/hardware/reachy/controller.py": "re-acquires what its connect released (_reacquire_media, #1704)",
}
_MEDIA_UNDO = re.compile(r"\.__exit__\(|\brelease_mini\(|\bwith\s+ReachyMini\(")  # shell files only


def _is_reachy_mini_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and (
        getattr(node.func, "id", None) == "ReachyMini" or getattr(node.func, "attr", None) == "ReachyMini"
    )


def _terminal_name(node: ast.AST) -> str:
    return getattr(node, "attr", None) or getattr(node, "id", None) or ""


def _undoes_media(tree: ast.AST) -> bool:
    """A real undo in the AST: ``with ReachyMini(...)``, ``<...mini>.__exit__(...)`` or ``release_mini(...)``."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            if any(_is_reachy_mini_call(item.context_expr) for item in node.items):
                return True
        elif isinstance(node, ast.Call):
            if _terminal_name(node.func) == "release_mini":
                return True
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "__exit__"
                and _terminal_name(node.func.value).endswith("mini")
            ):
                return True
    return False


@dataclass(frozen=True)
class Finding:
    """One lint finding. Not persisted (CC3 does not apply)."""

    kind: str  # "clamp", "raw_sdk", "auto_yaw" or "media_undo"
    path: str
    line: int
    text: str


def _in_scope(rel: str, scope: tuple[str, ...]) -> bool:
    return any(rel == s or (s.endswith("/") and rel.startswith(s)) for s in scope)


def _code_lines(source: str) -> list[str]:
    """Source lines with comments, triple-quoted strings (docstrings) and multi-line strings blanked.

    Ordinary single-line strings are kept: the ``getattr(mini, "goto_target")`` form needs them.
    """
    lines = source.splitlines()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return lines
    spans: list[tuple[tuple[int, int], tuple[int, int]]] = []
    fstring_starts: list[tokenize.TokenInfo] = []
    for tok in tokens:
        name = tokenize.tok_name.get(tok.type, "")
        if name == "FSTRING_START":  # Python 3.12+ tokenizes f-strings in pieces
            fstring_starts.append(tok)
        elif name == "FSTRING_END" and fstring_starts:
            start_tok = fstring_starts.pop()
            if start_tok.string.lstrip("rRbBuUfF")[:3] in ('"""', "'''") or start_tok.start[0] != tok.end[0]:
                spans.append((start_tok.start, tok.end))
        elif tok.type == tokenize.COMMENT:
            spans.append((tok.start, tok.end))
        elif tok.type == tokenize.STRING:
            triple = tok.string.lstrip("rRbBuUfF")[:3] in ('"""', "'''")
            if triple or tok.start[0] != tok.end[0]:
                spans.append((tok.start, tok.end))
    for (srow, scol), (erow, ecol) in reversed(spans):
        for row in range(srow, erow + 1):
            text = lines[row - 1]
            start = scol if row == srow else 0
            end = ecol if row == erow else len(text)
            lines[row - 1] = text[:start] + " " * (end - start) + text[end:]
    return lines


def _candidates(root: Path) -> list[Path]:
    """Python files under the scoped paths only (never a whole-repo walk into worktrees or venvs)."""
    paths: set[Path] = set()
    for scope in (*CLAMP_SCOPE, *RAW_SCOPE):
        target = root / scope
        if target.is_file():
            paths.add(target)
        elif target.is_dir():
            paths.update(target.rglob("*.py"))
    return sorted(paths)


def _auto_yaw_off(call: ast.Call) -> bool:
    return any(
        kw.arg == "automatic_body_yaw" and isinstance(kw.value, ast.Constant) and kw.value.value is False
        for kw in call.keywords
    )


def _auto_yaw_findings(rel: str, source: str) -> list[Finding]:
    """Each ``ReachyMini(`` call in *source* that does not pass ``automatic_body_yaw=False``."""
    lines = source.splitlines()
    if rel.endswith(".py"):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None
        if tree is not None:
            return [
                Finding("auto_yaw", rel, node.lineno, lines[node.lineno - 1].strip())
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and (getattr(node.func, "id", None) == "ReachyMini" or getattr(node.func, "attr", None) == "ReachyMini")
                and not _auto_yaw_off(node)
            ]
    findings = []  # a shell file (or Python that does not parse): the call's text, up to its closing bracket
    for match in _REACHY_MINI_CALL.finditer(source):
        depth, end = 0, len(source)
        for i in range(match.end() - 1, len(source)):
            depth += {"(": 1, ")": -1}.get(source[i], 0)
            if depth == 0:
                end = i + 1
                break
        if not _AUTO_YAW_OFF.search(source[match.start() : end]):
            number = source.count("\n", 0, match.start()) + 1
            findings.append(Finding("auto_yaw", rel, number, lines[number - 1].strip()))
    return findings


def _may_release_media(call_text: str | None, call: ast.Call | None) -> bool:
    """Whether a ``ReachyMini(`` call can connect with ``media_backend="no_media"`` (rule 4)."""
    if call is None:  # shell (or unparsable Python): the call's text
        return bool(call_text) and "no_media" in call_text
    if any(kw.arg is None for kw in call.keywords) or any(isinstance(a, ast.Starred) for a in call.args):
        return True  # a splat cannot show the backend
    for kw in call.keywords:
        if kw.arg == "media_backend":
            return not (isinstance(kw.value, ast.Constant) and kw.value.value != "no_media")
    return False


def _media_undo_findings(rel: str, source: str) -> list[Finding]:
    """Each ``ReachyMini(`` call that may release the daemon's media, in a file that never undoes it (rule 4)."""
    if rel in MEDIA_UNDO_OWNERS:
        return []
    lines = source.splitlines()
    tree = None
    if rel.endswith(".py"):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None
    if tree is not None:
        if _undoes_media(tree):
            return []
        hits = [node.lineno for node in ast.walk(tree) if _is_reachy_mini_call(node) and _may_release_media(None, node)]
    else:
        if _MEDIA_UNDO.search(source):
            return []
        hits = []
        for match in _REACHY_MINI_CALL.finditer(source):
            depth, end = 0, len(source)
            for i in range(match.end() - 1, len(source)):
                depth += {"(": 1, ")": -1}.get(source[i], 0)
                if depth == 0:
                    end = i + 1
                    break
            if _may_release_media(source[match.start() : end], None):
                hits.append(source.count("\n", 0, match.start()) + 1)
    return [Finding("media_undo", rel, n, lines[n - 1].strip()) for n in hits]


def _auto_yaw_candidates(root: Path) -> list[Path]:
    paths: set[Path] = set()
    for scope in AUTO_YAW_SCOPE:
        target = root / scope
        if target.is_dir():
            paths.update(p for p in target.rglob("*") if p.suffix in AUTO_YAW_SUFFIXES and p.is_file())
    return sorted(paths)


def scan(root: Path) -> list[Finding]:
    """Scan ``root`` (a repo root) and return every finding."""
    findings: list[Finding] = []
    for path in _candidates(root):
        rel = path.relative_to(root).as_posix()
        check_clamp = _in_scope(rel, CLAMP_SCOPE)
        check_raw = _in_scope(rel, RAW_SCOPE) and rel not in RAW_ALLOW_FILES
        if not (check_clamp or check_raw):
            continue
        allowed_lines = RAW_ALLOW_LINES.get(rel, set())
        source = path.read_text(encoding="utf-8")
        for number, (line, code) in enumerate(zip(source.splitlines(), _code_lines(source)), start=1):
            if not code.strip():
                continue
            if check_clamp and MEASUREMENT_PRAGMA not in line:
                findings.extend(Finding("clamp", rel, number, line.strip()) for p in CLAMP_PATTERNS if p.search(code))
            if check_raw and code.strip() not in allowed_lines:
                if any(p.search(code) for p in RAW_PATTERNS):
                    findings.append(Finding("raw_sdk", rel, number, line.strip()))
    for path in _auto_yaw_candidates(root):
        rel = path.relative_to(root).as_posix()
        if rel != "scripts/lint_motion_safety.py":  # its own pattern literals
            source = path.read_text(encoding="utf-8")
            findings.extend(_auto_yaw_findings(rel, source))
            findings.extend(_media_undo_findings(rel, source))
    return findings


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    findings = scan(root)
    for f in findings:
        if f.kind == "clamp":
            hint = "use maxim.hardware.controller.clamp_finite (refuses NaN), or mark a measurement clamp"
        elif f.kind == "media_undo":
            hint = (
                "a no_media connect releases the daemon's media past this process: use `with ReachyMini(...)`, "
                "or call release_mini(mini) / mini.__exit__(None, None, None) on every exit (#1712)"
            )
        elif f.kind == "auto_yaw":
            hint = "pass automatic_body_yaw=False: the setting is daemon-wide and the SDK defaults it on (#1499)"
        else:
            hint = "route through ReachyMiniController.goto_target or Selfy's validated entry points"
        print(f"{f.path}:{f.line}: [{f.kind}] {f.text}\n    -> {hint}")
    if findings:
        print(f"lint_motion_safety: {len(findings)} finding(s) (#1496, #1499, #1712; docs/plans/motion_finite_1496.md)")
        return 1
    print("lint_motion_safety: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())

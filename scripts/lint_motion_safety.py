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

Stated residuals (a line-based lint):
- a clamp or a handle split across lines;
- a one-sided ``max(-L, x)`` / ``min(L, x)`` bound (banning it would flag legitimate code), a bare
  ``clip(...)`` imported from numpy, and any clamp helper other than max/min/clip;
- any SDK handle bound to a name not ending in ``mini`` (``sdk = self.mini``, ``with ReachyMini() as
  robot``), or reached through a call, a subscript or parentheses (``self.mini()``, ``minis[0]``,
  ``ReachyMini().goto_target``, ``getattr(self, "mini")``), and ``getattr(handle, name)`` with a dynamic name;
- the canned or boolean SDK entry points ``wake_up``, ``goto_sleep``, ``enable/disable_motors``,
  ``enable/disable_wobbling``, ``set_automatic_body_yaw``, gravity compensation and ``cancel_move``
  (none takes a motion value; #1499's wider grep).

A single-line string or log message containing a banned shape is flagged too (reword it), and so is
``max(xs) - min(xs)`` on one line or any ``.clip(`` call; the pragma below covers a non-command clamp.

Motors 2+3 were destroyed by an unclamped pose (docs/agents/embodiment.md section 1).
Approach note: docs/plans/motion_finite_1496.md section 5. Exit 0 clean, 1 on any finding.
"""

from __future__ import annotations

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
RAW_ALLOW_LINES = {
    "src/maxim/embodied_runtime/movement.py": {
        # _enqueue_sdk_look_at binds the SDK fn, then dispatches it through a
        # last-mile-checked wrapper on the motor thread (#1499 reroutes it).
        "sdk_look_at = self.mini.look_at_image",
    },
}


@dataclass(frozen=True)
class Finding:
    """One lint finding. Not persisted (CC3 does not apply)."""

    kind: str  # "clamp" or "raw_sdk"
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
    return findings


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    findings = scan(root)
    for f in findings:
        if f.kind == "clamp":
            hint = "use maxim.hardware.controller.clamp_finite (refuses NaN), or mark a measurement clamp"
        else:
            hint = "route through ReachyMiniController.goto_target or Selfy's validated entry points"
        print(f"{f.path}:{f.line}: [{f.kind}] {f.text}\n    -> {hint}")
    if findings:
        print(f"lint_motion_safety: {len(findings)} finding(s) (#1496, docs/plans/motion_finite_1496.md)")
        return 1
    print("lint_motion_safety: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())

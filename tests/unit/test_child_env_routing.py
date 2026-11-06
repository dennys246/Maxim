"""Every child env a test builds keeps the suite's cloudflared isolation (``tests/_child_env.py``).

``tests/conftest.py`` isolates the system cloudflared dir twice: it exports ``MAXIM_CLOUDFLARED_SYSTEM_DIR`` for
children, and points the reader's in-process default at the same dir, so an in-process strip of every ``MAXIM_*``
(in any spelling) never reaches ``/etc`` (guard: ``test_cloudflared_system_dir_isolation.py``). A CHILD is a fresh
process with the real default, so a child env that drops the variable sends role detection back to the host's
``/etc/cloudflared/config.yml``: ``leader`` on a leader dev box, invisible on CI (2026-11-05 review W1, delta S1).

This scan checks each SITE, not each file:
- a child env that names its own ``HOME`` (``"HOME": ...``, ``env["HOME"] = ...``, ``HOME=...`` in a call) must, within
  a few lines, take ``isolation_env()`` / ``without_maxim_overrides()`` or start from ``os.environ``
  (``**os.environ``, ``dict(os.environ)``, ``os.environ.copy()``);
- a hand-written filter dropping ``MAXIM_*`` (``not k.startswith("MAXIM_")``, tuple prefixes included) is refused
  outside the helper: use ``without_maxim_overrides()``.
It is a heuristic over source text, so its allow-list must give a reason and must not go stale.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
_WINDOW = 12  # lines on either side of a HOME assignment that may carry the isolation

# A dict-literal key, a subscript assignment on anything but os.environ itself (in-process: covered by conftest's
# default), or a keyword argument.
_HOME = re.compile(r"""["']HOME["']\s*:|(?<!environ)\[\s*["']HOME["']\s*\]\s*=(?!=)|(?<![\w.])HOME\s*=(?!=)""")
_CARRIES = re.compile(
    r"""isolation_env\(|without_maxim_overrides\(|\*\*os\.environ\b|dict\(os\.environ\b|os\.environ\.copy\(\)"""
)
_FILTER = re.compile(r"""not\s+[\w.\[\]"']+\.startswith\(\s*\(?[^)]*["']MAXIM_""")

# path (relative to tests/) -> why it may match. Stale entries fail too.
ALLOWED: dict[str, str] = {
    "_child_env.py": "the helper itself",
    "unit/test_child_env_routing.py": "this guard's own patterns and docstring",
    "unit/test_ci_workflow_shape.py": "runs a workflow's shell step under stub binaries, not a maxim process",
}


def _offenders() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(TESTS.rglob("*.py")):
        rel = path.relative_to(TESTS).as_posix()
        lines = path.read_text(encoding="utf-8").splitlines()
        hits = []
        for i, line in enumerate(lines):
            if _FILTER.search(line):
                hits.append(f"line {i + 1}: hand-written MAXIM_* filter: {line.strip()}")
            if _HOME.search(line):
                window = "\n".join(lines[max(0, i - _WINDOW) : i + _WINDOW + 1])
                if not _CARRIES.search(window):
                    hits.append(f"line {i + 1}: a child env names HOME without isolation_env(): {line.strip()}")
        if hits:
            found[rel] = hits
    return found


def test_every_child_env_keeps_the_isolation():
    offenders = {rel: hits for rel, hits in _offenders().items() if rel not in ALLOWED}
    assert not offenders, (
        "give these child envs the suite's isolation (tests/_child_env.py: isolation_env() beside HOME, or "
        f"without_maxim_overrides()), or allow-list one with the reason it never runs maxim: {offenders}"
    )


def test_every_allow_list_entry_still_matches():
    stale = sorted(set(ALLOWED) - set(_offenders()))
    assert not stale, f"these ALLOWED entries no longer match a pattern; remove them: {stale}"


def test_the_scan_catches_the_spellings_the_delta_review_found(tmp_path, monkeypatch):
    """Each form the delta review showed passing the first version of this guard (S1) is caught now."""
    probes = {
        "subscript": 'env = {}\nenv["HOME"] = str(tmp)\nsubprocess.run(cmd, env=env)\n',
        "dict_kw": "env = dict(PATH=p, HOME=tmp)\n",
        "tuple_prefix": 'env = {k: v for k, v in base.items() if not k.startswith(("MAXIM_", "HF_"))}\n',
        "literal": 'env = {"HOME": str(tmp), "PATH": "/usr/bin"}\n',
    }
    for name, body in probes.items():
        (tmp_path / f"probe_{name}.py").write_text(body)
    monkeypatch.setattr(sys.modules[__name__], "TESTS", tmp_path)
    assert set(_offenders()) == {f"probe_{n}.py" for n in probes}

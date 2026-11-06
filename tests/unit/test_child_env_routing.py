"""Child envs a test builds keep the suite's cloudflared isolation (``tests/_child_env.py``): a heuristic source scan.

``tests/conftest.py`` isolates the system cloudflared dir twice: it exports ``MAXIM_CLOUDFLARED_SYSTEM_DIR`` for
children, and points the reader's in-process default at the same dir, so an in-process strip of every ``MAXIM_*``
(in any spelling) never reaches ``/etc`` IN THE TEST PROCESS (guard: ``test_cloudflared_system_dir_isolation.py``).
A CHILD is a fresh process with the real default, so a child env that lacks the variable sends role detection back to
the host's ``/etc/cloudflared/config.yml``: ``leader`` on a leader dev box, invisible on CI (2026-11-05 review W1,
delta rounds 1-2). That includes a child that merely INHERITS an ``os.environ`` a test stripped in place.

This scan checks each SITE, not each file:
- a child env that names its own ``HOME`` (``"HOME": ...``, ``env["HOME"] = ...``, ``setdefault("HOME", ...)``,
  ``HOME=...`` in a call) must, within a few lines, take ``isolation_env()`` / ``without_maxim_overrides()`` or start
  from ``os.environ`` (``**os.environ``, ``dict(os.environ)``, ``os.environ.copy()``);
- a hand-written filter dropping ``MAXIM*`` (``not k.startswith("MAXIM_")``, tuple prefixes included) is refused
  outside the helper: use ``without_maxim_overrides()``;
- a hand-written deletion of ``MAXIM*`` from any mapping (a positive ``startswith`` with ``del``/``.pop(``/``delenv(``
  within two lines) is refused: use ``drop_maxim_overrides()``, which keeps the variable for any child started after.
Known limits (no live instance): a child env with no ``HOME`` at all and an env built far from its strip are not seen;
a ``HOME`` constant or a plain dict that never reaches a subprocess is flagged and needs an allow-list entry. So the
allow-list must give a reason and must not go stale.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
_WINDOW = 12  # lines on either side of a HOME assignment that may carry the isolation

# A dict-literal key, a subscript assignment on anything but os.environ itself (in-process: covered by conftest's
# default), or a keyword argument.
_HOME = re.compile(
    r"""["']HOME["']\s*:|(?<!os\.environ)\[\s*["']HOME["']\s*\]\s*=(?!=)|setdefault\(\s*["']HOME["']"""
    r"""|(?<![\w.])HOME\s*=(?!=)"""
)
_CARRIES = re.compile(
    r"""isolation_env\(|without_maxim_overrides\(|\*\*os\.environ\b|dict\(os\.environ\b|os\.environ\.copy\(\)"""
)
_FILTER = re.compile(r"""not\s+[\w.\[\]"']+\.startswith\(\s*\(?[^)]*["']MAXIM""")
_POSITIVE = re.compile(r"""(?<!not )\b\w+\.startswith\(\s*\(?[^)]*["']MAXIM""")
_DELETES = re.compile(r"""\bdel\s|\.pop\(|delenv\(""")

# path (relative to tests/) -> why it may match. Stale entries fail too.
ALLOWED: dict[str, str] = {
    "_child_env.py": "the helper itself",
    "unit/test_child_env_routing.py": "this guard's own patterns, docstring and probes",
    "unit/test_ci_workflow_shape.py": "runs a workflow's shell step under stub binaries, not a maxim process",
    "unit/test_e4_text_widening_drift.py": (
        "_clear_maxim_env is monkeypatch-scoped to the pin_environment tests, which spawn nothing (the file's one "
        "subprocess runs unstripped); drop_maxim_overrides would leave the isolation var for pin_environment to report"
    ),
    "unit/test_rig_suite.py": (
        "runs bash scripts/rig_suite.sh against a stub suite_at_commit.sh and a fake ssh, never a maxim process (#1604)"
    ),
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
            elif _POSITIVE.search(line) and _DELETES.search("\n".join(lines[i : i + 3])):
                hits.append(f"line {i + 1}: hand-written MAXIM_* deletion (use drop_maxim_overrides): {line.strip()}")
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
    """Each form the delta reviews (rounds 1 and 2) showed passing an earlier version of this guard is caught now."""
    probes = {
        "subscript": 'env = {}\nenv["HOME"] = str(tmp)\nsubprocess.run(cmd, env=env)\n',
        "dict_kw": "env = dict(PATH=p, HOME=tmp)\n",
        "tuple_prefix": 'env = {k: v for k, v in base.items() if not k.startswith(("MAXIM_", "HF_"))}\n',
        "literal": 'env = {"HOME": str(tmp), "PATH": "/usr/bin"}\n',
        "strip_then_child": (
            'for k in [k for k in os.environ if k.startswith("MAXIM_")]:\n'
            "    monkeypatch.delenv(k)\n"
            'subprocess.run([sys.executable, "-m", "maxim"])\n'
        ),
        "copy_then_del": (
            'env = os.environ.copy()\nfor k in [k for k in env if k.startswith("MAXIM_")]:\n    del env[k]\n'
        ),
        "copy_then_pop": 'env = os.environ.copy()\nfor k in list(env):\n    if k.startswith("MAXIM_"):\n        env.pop(k)\n',
        "environ_suffix": 'child_environ = {}\nchild_environ["HOME"] = str(tmp)\n',
        "setdefault": 'env = {}\nenv.setdefault("HOME", str(tmp))\n',
    }
    for name, body in probes.items():
        (tmp_path / f"probe_{name}.py").write_text(body)
    monkeypatch.setattr(sys.modules[__name__], "TESTS", tmp_path)
    assert set(_offenders()) == {f"probe_{n}.py" for n in probes}

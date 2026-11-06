"""Every test site that strips ``MAXIM_*`` or builds a child env from scratch routes through ``tests/_child_env.py``.

The suite's isolation variable ``MAXIM_CLOUDFLARED_SYSTEM_DIR`` lives in the ``MAXIM_*`` namespace, and the suite's
habit is to drop every ``MAXIM_*`` as an operator override. A site that does so by hand, or builds a child env beside
its own ``HOME`` without ``isolation_env()``, silently sends role detection back to the host's
``/etc/cloudflared/config.yml``: ``leader`` on a leader dev box, invisible on CI (2026-11-05 review W1). A forgotten site
is a silent no-op, so this scan makes it a failure.
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]

# A hand-written strip: `not k.startswith("MAXIM_")` (a filter that drops them) or a deletion loop over os.environ.
_STRIP = (
    re.compile(r"""not\s+\w+\.startswith\(\s*["']MAXIM_["']\s*\)"""),
    re.compile(r"""for\s+\w+\s+in\s+\[\s*\w+\s+for\s+\w+\s+in\s+os\.environ\b.*startswith\(\s*["']MAXIM_["']\s*\)"""),
)
# A child env that names its own HOME: built from scratch unless the file inherits os.environ (allow-listed below).
_SCRATCH_HOME = re.compile(r"""["']HOME["']\s*:|\bHOME\s*=\s*str\(""")
_ROUTED = re.compile(r"""from tests\._child_env import""")

# path (relative to tests/) -> why it may match without routing through the helper. Stale entries fail too.
ALLOWED: dict[str, str] = {
    "_child_env.py": "the helper itself",
    "unit/test_child_env_routing.py": "this guard's own patterns",
    "unit/test_e4_text_widening_drift.py": (
        "tests that pin_environment reports unexpected MAXIM_* toggles, so it must clear every one; it never reaches "
        "role detection"
    ),
    "unit/test_ci_workflow_shape.py": "runs a workflow's shell step under stub binaries, not a maxim process",
    "unit/test_console_ws_real_uvicorn.py": "its child env starts from **os.environ, so it inherits the isolation",
    "unit/test_agent_loop_selection_golden.py": "its child env starts from dict(os.environ): it inherits the isolation",
    "unit/test_orchestrator_characterization_golden.py": (
        "its child env starts from dict(os.environ): it inherits the isolation"
    ),
}


def _offenders() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(TESTS.rglob("*.py")):
        rel = path.relative_to(TESTS).as_posix()
        text = path.read_text(encoding="utf-8")
        hits = [f"strip: {ln.strip()}" for ln in text.splitlines() if any(p.search(ln) for p in _STRIP)]
        if _SCRATCH_HOME.search(text) and not _ROUTED.search(text):
            hits.append("a child env names HOME but the file never imports tests._child_env")
        if hits:
            found[rel] = hits
    return found


def test_every_maxim_strip_and_scratch_child_env_routes_through_the_helper():
    offenders = {rel: hits for rel, hits in _offenders().items() if rel not in ALLOWED}
    assert not offenders, (
        "route these through tests/_child_env.py (drop_maxim_overrides / without_maxim_overrides / isolation_env "
        f"beside HOME), or allow-list one with the reason it cannot reach role detection: {offenders}"
    )


def test_every_allow_list_entry_still_matches():
    stale = sorted(set(ALLOWED) - set(_offenders()))
    assert not stale, f"these ALLOWED entries no longer match a pattern; remove them: {stale}"

"""Each nightly lane's roster matches collection, checked at PR time (roadmap 1.3.2 item 7, #940; #1117).

`scripts/check_lane_roster.py` holds each nightly lane to `scripts/lane_rosters/<lane>.json`; without this test a PR
that adds, removes or renames a lane's test would merge green and turn the NEXT nightly red on main. This runs in the
fast suite and compares file-scoped `pytest --collect-only -m <marker>` with the roster (adversarial design pass). A
marker applied some other way (a conftest hook, `getattr(pytest.mark, ...)`) is invisible to the file scan and is
caught only by the nightly, as "ran, not in the roster".
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts import check_lane_roster as C

# An excused skip is a reviewed act: raise the pin in the same PR and say why the test cannot run in the lane
# (the model-cache pin carries the 1.3.1 `ALLOWED_SKIPS <= 4` ratchet from tests/unit/test_model_cache_names.py).
ALLOWED_SKIP_PINS = {
    "slow": 1,
    "model-cache": 4,
    "extras-console": 0,
    "extras-tts": 0,
}  # an extras cell installs its extra and assets


@pytest.mark.parametrize("lane", sorted(C.LANES))
def test_the_roster_is_exactly_what_collection_selects(lane):
    nodeids, problems = C.collect(lane)
    expected, allowed = C.load_roster(C.roster_path(lane))
    assert not problems, problems
    missing, extra = sorted(set(expected) - set(nodeids)), sorted(set(nodeids) - set(expected))
    assert not missing and not extra, (
        f"{lane}-lane roster drift: not collected {missing}, not in the roster {extra}. "
        f"Run `python3 scripts/check_lane_roster.py --lane {lane} --generate` and review the diff."
    )
    assert set(allowed) <= set(expected) and all(r.strip() for r in allowed.values())


@pytest.mark.parametrize("lane", sorted(C.LANES))
def test_the_allowed_skips_cannot_grow_silently(lane):
    _expected, allowed = C.load_roster(C.roster_path(lane))
    assert len(allowed) <= ALLOWED_SKIP_PINS[lane], (
        f"the {lane} lane's allowed_skips grew: an excused skip is a reviewed decision -- raise ALLOWED_SKIP_PINS in "
        "the same PR and say why the test cannot run there"
    )


def test_the_model_cache_lane_excuses_only_dataset_gated_tests():
    """Carried from check_model_cache_lane.py (1.3.1): the lane installs every extra and warms every model, so the
    only excusable skip is a dataset it does not download."""
    _expected, allowed = C.load_roster(C.roster_path("model-cache"))
    assert all("dataset" in reason for reason in allowed.values()), allowed


def test_every_lane_has_a_roster_and_no_roster_is_orphaned():
    assert {p.stem for p in C.ROSTERS.glob("*.json")} == set(C.LANES)


def test_collection_is_scoped_to_files_that_apply_the_marker():
    """A module-level skip in a NON-lane file (the console modules on a box without the extra) must not fail the fast
    suite through this test (wire review); only files that apply the lane's marker are collected."""
    slow = C.marked_files("slow")
    assert "tests/integration/test_cradle_tool_routing.py" in slow
    assert "tests/unit/test_clip_encoder.py" in C.marked_files("requires_model_cache")
    for marker in C.LANES.values():
        files = C.marked_files(marker)
        if marker == "extras_console":
            # The console extra's own real-library test: it imports the extra inside a fixture, so it collects on a box
            # without it (and a module-level skip there would be a collect() problem in
            # test_the_roster_is_exactly_what_collection_selects). Exactly that one file.
            assert [f for f in files if "test_console_" in f] == ["tests/unit/test_console_ws_real_uvicorn.py"]
        else:
            assert not any("test_console_" in f for f in files)
        assert all(f.rsplit("/", 1)[-1].startswith("test_") for f in files)  # the configured python_files pattern


def test_a_module_skipped_at_collection_is_refused_from_real_collect_output(tmp_path, monkeypatch):
    """Executor review (#940): under `-q` pytest leaves the skip count out of its summary, so the first version never
    refused. Real `--collect-only` output from a repo holding a marked module that skips at import."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_ok.py").write_text("import pytest\n\n@pytest.mark.slow\ndef test_ok():\n    pass\n")
    (tests / "test_gone.py").write_text(
        "import pytest\npytest.importorskip('no_such_pkg_xyz')\n\n@pytest.mark.slow\ndef test_gone():\n    pass\n"
    )
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    slow: slow\n")
    nodeids, problems = C.collect("slow", tmp_path)
    assert nodeids == ["tests/test_ok.py::test_ok"]
    assert any("test_gone.py" in p and "skipped or errored at collection" in p for p in problems), problems


# The model-cache and slow lanes run `pytest tests/ -m <marker>`: pytest COLLECTS every module before `-m` deselects,
# so a module that skips at import fails both lanes' roster checks ("module-level collection skipped"), marked or not.
# The lanes install what they collect (.github/actions/model-cache-setup); this is the PR-time twin of that rule.
_LANE_SETUP = C.REPO / ".github" / "actions" / "model-cache-setup" / "action.yml"
_PIP_IMPORT_NAMES = {"pillow": "PIL"}  # a pip argument whose import name differs


def _lane_installed_imports() -> set[str]:
    """Top-level import names the nightly lanes' install step provides: its extras (read from pyproject.toml through
    conftest's ``extra_import_names``) plus the bare packages on the same ``pip install -e ".[...]"`` line."""
    import re

    from tests.conftest import extra_import_names

    match = re.search(r'pip install -e "\.\[([^\]]+)\]"([^\\\n]*)', _LANE_SETUP.read_text(encoding="utf-8"))
    assert match, f'no `pip install -e ".[extras]"` line in {_LANE_SETUP}'
    names = {n for extra in match.group(1).split(",") for n in extra_import_names(extra.strip())}
    for pkg in match.group(2).split():
        if not pkg.startswith("-"):
            dist = re.sub(r"[-.]", "_", pkg.lower())
            names.add(_PIP_IMPORT_NAMES.get(dist, dist))
    return names


def _module_level_importorskips(root: Path) -> list[tuple[str, str]]:
    """``(file, import name)`` for every ``pytest.importorskip("<name>")`` that runs at module import (a top-level
    expression or assignment) in a ``test_*.py`` under ``root/tests``."""
    found = []
    for path in sorted((root / "tests").rglob("test_*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body:
            call = node.value if isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)) else None
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "importorskip"
                and call.args
                and isinstance(call.args[0], ast.Constant)
            ):
                found.append((str(path.relative_to(root)), str(call.args[0].value)))
    return found


def test_no_module_skips_at_collection_for_a_package_the_nightly_lanes_do_not_install():
    """2026-10-31 → 11-21: tests/unit/test_reachy_sdk_contract_1499.py's module-level `importorskip("reachy_mini")`
    turned the model-cache and slow nightlies red for three weeks; the lanes do not install the robot SDK, and the
    module holds no lane test. A module gated on such a package skips its TESTS instead (a module `pytestmark` skipif,
    imports inside the tests), so it still collects in the lanes. A repo-local module (a `scripts/` file the test puts
    on sys.path) always imports."""
    installed = _lane_installed_imports()
    assert {"fastapi", "sentence_transformers", "cryptography", "spacy"} <= installed, installed  # the parse works

    def repo_local(name: str) -> bool:
        return any(any((C.REPO / top).rglob(f"{name}.py")) for top in ("scripts", "tests"))

    offenders = [
        f"{f}: importorskip({n!r})"
        for f, n in _module_level_importorskips(C.REPO)
        if n not in installed and not repo_local(n)
    ]
    assert not offenders, (
        "these modules skip at collection for a package the nightly lanes do not install, which fails the "
        "model-cache and slow roster checks; skip the tests instead (module pytestmark skipif) or install it in "
        f".github/actions/model-cache-setup: {offenders}"
    )


def test_the_module_level_importorskip_scan_sees_top_level_calls_only(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_x.py").write_text(
        "import pytest\nsdk = pytest.importorskip('robot_sdk')\npytest.importorskip('other')\n\n"
        "def test_y():\n    pytest.importorskip('inside')\n"
    )
    assert _module_level_importorskips(tmp_path) == [("tests/test_x.py", "robot_sdk"), ("tests/test_x.py", "other")]

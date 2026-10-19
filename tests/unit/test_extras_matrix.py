"""The R2 extras matrix's helper and its job (#1335; docs/plans/readiness_and_extras_standards.md §"R2's lane").

The lane is a gate, so each way an input could lie (the approach note's table) has a test here, and the job's shape
is pinned: a cell whose steps all skip concludes `success`, so the run/skip decision must sit in the YAML, where a
nightly cannot skip (design pass, finding 3).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

from scripts import check_lane_roster
from scripts import extras_matrix as M

ROOT = Path(__file__).resolve().parents[2]
JOB = yaml.safe_load((ROOT / ".github/workflows/test.yml").read_text())["jobs"]["extras-matrix"]
GATE = "github.event_name != 'pull_request' || steps.scope.outputs.run == 'true'"


# ── pins ──────────────────────────────────────────────────────────────────────


def test_floor_pins_are_each_requirements_lower_bound_without_its_extras():
    reqs = ["piper-tts>=1.3.0", "uvicorn[standard]>=0.29,<1.0", 'av>=14.0.0; python_version >= "3.10"']
    assert M.floor_pins(reqs) == [
        "piper-tts==1.3.0",
        "uvicorn==0.29",  # pip refuses [extras] in a constraints file
        'av==14.0.0; python_version >= "3.10"',
    ]


@pytest.mark.parametrize("req", ["piper-tts", "fastapi<1.0", "x==1.0", "y>=1,>=2"])
def test_a_requirement_without_a_single_floor_is_refused(req):
    with pytest.raises(M.InputError):
        M.floor_pins([req])


def test_latest_pins_are_what_the_unconstrained_resolve_picked():
    report = {"install": [{"metadata": {"name": "Piper_TTS", "version": "1.8.0"}}, {"metadata": {"name": "numpy"}}]}
    with pytest.raises(M.InputError):  # an entry without a version: unreadable, not "nothing to pin"
        M.latest_pins(["piper-tts>=1.3.0"], report)
    report["install"][1]["metadata"]["version"] = "2.5.4"
    assert M.latest_pins(["piper-tts>=1.3.0"], report) == ["piper-tts==1.8.0"]
    with pytest.raises(M.InputError):
        M.latest_pins(["onnxruntime>=1.0"], report)  # resolved nothing for it
    assert M.latest_pins(['winonly>=1; sys_platform == "nonesuch"'], report) == []


def test_every_matrix_extra_declares_a_floor_for_every_requirement():
    for extra in M.EXTRAS:
        assert M.floor_pins(M.requirements(extra))


# ── verify ────────────────────────────────────────────────────────────────────


def test_verify_requires_each_pin_exactly():
    pins = ["fastapi==0.110", "piper-tts==1.3.0", 'gone==1; sys_platform == "nonesuch"']
    assert M.check_pins(pins, {"fastapi": "0.110.0", "piper-tts": "1.3.0"}) == []  # Version equality
    problems = M.check_pins(pins, {"fastapi": "0.115.2"})
    assert problems == ["fastapi: pinned 0.110, installed 0.115.2", "piper-tts: pinned 1.3.0, not installed"]


def _requires_of(graph):
    return lambda name: graph.get(M._norm(name))


def test_only_the_extras_closure_may_be_installed():
    """#1347's vacuity class: a cell that drags `websockets` in some other way must not let `/ws` pass."""
    graph = {
        "pymaxim": ["numpy>=1", 'piper-tts>=1.3; extra == "tts"', 'fastapi>=0.110; extra == "console"'],
        "numpy": [],
        "piper-tts": ["onnxruntime"],
        "onnxruntime": [],
        "fastapi": ["starlette"],
        "pytest": ["pluggy", 'colorama; sys_platform == "win32"'],
        "pluggy": [],
    }
    closure = {"pymaxim", "numpy", "piper-tts", "onnxruntime", "pytest", "pluggy"}
    assert M.closure(["pymaxim[tts]", "pytest"], _requires_of(graph)) == closure
    assert M.check_only_the_extra("tts", closure | {"pip"}, _requires_of(graph)) == []
    assert M.check_only_the_extra("tts", closure | {"websockets"}, _requires_of(graph)) == [
        "installed but outside pymaxim[tts] + pytest: websockets"
    ]


# ── scope ─────────────────────────────────────────────────────────────────────


def test_scope_runs_when_the_diff_cannot_be_read():
    assert M.scope("tts", None)[0] is True


@pytest.mark.parametrize(
    "path", ["src/maxim/models/audio/tts.py", "src/maxim/console/announcer.py", "pyproject.toml", "tests/conftest.py"]
)
def test_scope_runs_when_the_diff_touches_the_extra_or_the_lane(path):
    assert M.scope("tts", ["docs/x.md", path])[0] is True


def test_scope_skips_an_unrelated_diff_and_says_why():
    run, why = M.scope("tts", ["docs/plans/x.md", "src/maxim/memory/hippocampus.py"])
    assert run is False and "tts" in why


def test_an_unknown_extra_is_an_input_error_not_a_skip():
    with pytest.raises(M.InputError):
        M.scope("nonesuch", [])
    assert M.main(["scope", "--extra", "nonesuch"]) == 2


# ── the table, the lanes and the job agree ────────────────────────────────────


def test_the_matrix_the_table_the_lanes_and_the_markers_agree():
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    markers = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]["markers"]
    assert JOB["strategy"]["matrix"]["extra"] == sorted(M.EXTRAS)
    assert JOB["strategy"]["matrix"]["resolution"] == ["floor", "latest"]
    for extra in M.EXTRAS:
        assert extra in extras
        assert check_lane_roster.LANES[f"extras-{extra}"] == f"extras_{extra}"
        assert any(m.startswith(f"extras_{extra}:") for m in markers)
        assert M.marked_test_files(extra), extra


def test_the_job_is_under_the_release_gate_and_no_cell_cancels_another():
    assert "(nightly)" in JOB["name"]  # scripts/check_nightlies.py::NIGHTLY_MARKER
    assert JOB["strategy"]["fail-fast"] is False
    assert "if" not in JOB  # a job-level `if:` cannot read the matrix; and the nightly must never skip a cell


def test_only_a_pull_request_consults_scope_and_every_later_step_is_gated_in_the_yaml():
    steps = JOB["steps"]
    names = [s.get("name") for s in steps]
    scope = steps[names.index("Scope (pull requests only)")]
    assert scope["id"] == "scope" and scope["if"] == "github.event_name == 'pull_request'"
    later = steps[names.index("Scope (pull requests only)") + 1 :]
    for step in later:
        cond = re.sub(r"\s+", " ", str(step.get("if", "")))
        assert GATE in cond, (step.get("name"), cond)
    roster = next(s for s in later if "check_lane_roster.py" in str(s.get("run", "")))
    assert roster["if"].startswith("always() && steps.tests.outcome != 'skipped' && (") and GATE in roster["if"]
    assert (
        roster["run"].strip()
        == "python3 scripts/check_lane_roster.py --lane extras-${{ matrix.extra }} extras-results.xml"
    )


def test_the_cell_installs_only_the_extra_verifies_it_and_runs_offline():
    runs = {s.get("name"): re.sub(r"\s+", " ", str(s.get("run", ""))) for s in JOB["steps"]}
    install = runs["Install only the extra (and pytest)"]
    assert install == 'python -m pip install -e ".[${{ matrix.extra }}]" pytest -c extras-pins.txt'
    assert (
        "verify --extra ${{ matrix.extra }} --pins extras-pins.txt"
        in runs["Verify the pins and that only the extra is installed"]
    )
    test = runs["Run the extra's real-library tests"]
    assert "bash scripts/ci_netns.sh" in test and '"$PY" -m pytest' in test
    assert "--require-extras=${{ matrix.extra }}" in test and "--junitxml=extras-results.xml" in test
    assert '-m "extras_${{ matrix.extra }}"' in test
    names = list(runs)
    assert names.index("Provision the extra's assets (before the network namespace)") < names.index(
        "Run the extra's real-library tests"
    )


# M69 (docs/plans/outstanding.md), mechanized: every extra is in the matrix, or named here. The order of R2's rollout
# (readiness_and_extras_standards.md §R2 "Order"): tts and console in 1.3.3; reachy and audio next; the rest after.
# This list may only SHRINK: an extra leaves it in the PR that adds its cells, and a new extra is born in the matrix.
_NOT_YET_IN_MATRIX = frozenset(
    {
        "audio", "comms", "database", "llm-anthropic", "llm-llama", "llm-openai", "llm-server", "llm-torch", "pi",
        "reachy", "search", "semantic", "sign", "temporal", "training", "vision", "yolo",
    }
)  # fmt: skip
# Not libraries a user installs for a capability: an aggregate of the others, and the dev/test toolchain. CLOSED: a
# new extra may not be parked here to dodge the matrix and the ceiling (the test pins it exactly).
_NOT_A_CAPABILITY = frozenset({"all", "test"})
_NOT_YET_CEILING = 17  # the count with tts and console in (1.3.3); lower it with the list, never raise it


def test_every_extra_is_in_the_matrix_or_named_as_not_yet():
    extras = set(tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"])
    assert extras == set(M.EXTRAS) | _NOT_YET_IN_MATRIX | _NOT_A_CAPABILITY, (
        f"unaccounted: {sorted(extras - set(M.EXTRAS) - _NOT_YET_IN_MATRIX - _NOT_A_CAPABILITY)}; "
        f"stale: {sorted((set(M.EXTRAS) | _NOT_YET_IN_MATRIX | _NOT_A_CAPABILITY) - extras)}"
    )
    assert not set(M.EXTRAS) & _NOT_YET_IN_MATRIX
    assert len(_NOT_YET_IN_MATRIX) <= _NOT_YET_CEILING
    assert _NOT_A_CAPABILITY == {"all", "test"}

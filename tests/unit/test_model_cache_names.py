"""The nightly warm list is derived from source (scripts/model_cache_names.py) and must cover
every model a marked test or the production loaders name. Verified to fail on the pre-fix
workflow's hand-kept tuple, which lacked all-MiniLM-L6-v2 (every nightly run since 2026-08-21)."""

from __future__ import annotations

from pathlib import Path

from scripts import check_model_cache_lane as C
from scripts import model_cache_names as M

PRODUCTION_DEFAULTS = {"all-mpnet-base-v2", "paraphrase-mpnet-base-v2", "clip-ViT-B-32", "all-MiniLM-L6-v2"}


def test_derived_list_covers_production_defaults_and_marked_tests() -> None:
    names = set(M.model_names())
    assert PRODUCTION_DEFAULTS <= names, names
    assert "paraphrase-MiniLM-L6-v2" in names  # the test_model_comparison sweep arm


def test_marked_test_files_are_found() -> None:
    files = {p.name for p in M.marked_test_files()}
    assert "test_baselines.py" in files and "test_clip_encoder.py" in files


def test_lane_check_reads_junit_ids(tmp_path: Path) -> None:
    xml = tmp_path / "r.xml"
    xml.write_text(
        '<testsuites><testsuite tests="3" skipped="2">'
        '<testcase classname="tests.substrate.baselines.test_baselines.TestEmbeddingBaseline" name="test_returns_result_without_deps"><skipped message="installed"/></testcase>'
        '<testcase classname="tests.unit.test_clip_encoder" name="test_x"><skipped message="no reason"/></testcase>'
        '<testcase classname="tests.unit.test_clip_encoder" name="test_y"/>'
        "</testsuite></testsuites>"
    )
    assert C.check(xml) == 1  # test_x skipped and not allow-listed
    xml.write_text(
        '<testsuites><testsuite tests="2" skipped="1">'
        '<testcase classname="tests.substrate.baselines.test_baselines.TestEmbeddingBaseline" name="test_returns_result_without_deps"><skipped message="installed"/></testcase>'
        '<testcase classname="tests.unit.test_clip_encoder" name="test_y"/>'
        "</testsuite></testsuites>"
    )
    assert C.check(xml) == 0


def test_lane_check_refuses_zero_executed(tmp_path: Path) -> None:
    xml = tmp_path / "r.xml"
    xml.write_text(
        '<testsuite tests="1" skipped="1"><testcase classname="tests.unit.t" name="a"><skipped/></testcase></testsuite>'
    )
    assert C.check(xml) == 1


def test_allow_list_names_real_tests() -> None:
    for tid in C.ALLOWED_SKIPS:
        path, _, rest = tid.partition("::")
        assert (M.REPO_ROOT / path).exists(), path
        name = rest.split("::")[-1]
        assert f"def {name}(" in (M.REPO_ROOT / path).read_text(), tid

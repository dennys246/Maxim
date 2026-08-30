"""scripts/lint_harness_provenance.py — both families, on a synthetic scripts tree.

Verified to fail 4/4 in-process writers on the pre-fix origin/main tree (live_common.py,
ear_map.py, loudness_bench_poll.py, 9_hunger_relief_orient.py) and 6/6 spawners on the
spawner-gate rule.
"""

from __future__ import annotations

from pathlib import Path

from scripts import lint_harness_provenance as L

GUARDED_WRITER = "class JsonlLog:\n    def __init__(self, path):\n        preflight_gated_record_or_exit(ROOT, path)\n"


def _tree(tmp_path: Path, writer: str = GUARDED_WRITER) -> Path:
    (tmp_path / "scripts/orient_backbone").mkdir(parents=True)
    (tmp_path / "scripts/orient_backbone/live_common.py").write_text(writer)
    return tmp_path


def test_in_process_writer_without_preflight_is_a_violation(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "scripts/orient_backbone/ear.py").write_text('with open(p, "w") as f:\n    f.write("x")\n')
    (root / "scripts/orient_backbone/dump.py").write_text("json.dump(obj, fh)\n")
    (root / "scripts/orient_backbone/txt.py").write_text("Path(p).write_text(s)\n")
    fails = L.lint(root)
    assert sorted(f.split(":")[0] for f in fails) == [
        "scripts/orient_backbone/dump.py",
        "scripts/orient_backbone/ear.py",
        "scripts/orient_backbone/txt.py",
    ]


def test_guard_references_and_exempt_marker_comply(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "scripts/orient_backbone/a.py").write_text("log = JsonlLog(args.log)\njson.dump(x, fh)\n")
    (root / "scripts/orient_backbone/b.py").write_text("preflight_gated_record_or_exit(R, out)\njson.dump(x, fh)\n")
    (root / "scripts/orient_backbone/c.py").write_text("in_process_code_provenance(R, mf, out_path=o)\nopen(o, 'w')\n")
    (root / "scripts/orient_backbone/d.py").write_text("# provenance-exempt: demo, not evidence\njson.dump(x, fh)\n")
    (root / "scripts/orient_backbone/e.py").write_text("open(p) # read only\n")
    assert L.lint(root) == []


def test_guarded_writer_must_itself_run_the_preflight(tmp_path: Path) -> None:
    root = _tree(tmp_path, writer="class JsonlLog:\n    def __init__(self, path):\n        self._f = open(path, 'a')\n")
    fails = L.lint(root)
    assert len(fails) == 1 and "JsonlLog no longer references preflight_gated_record" in fails[0]


# -- Family 3 -- gated-path writers anywhere under scripts/ (added 2026-08-30) --
#
# `scripts/fail_loud_stage2.py` wrote a new artifact into docs/experiments/data/
# while escaping both existing families: it spawns no `maxim` (not Family 1) and
# sits at the top level of scripts/ rather than under orient_*/ (not Family 2).
# It hand-rolled a dirty-tree check and only STAMPED the flag -- detection, not
# enforcement, the exact Exp 53/53b shape.

GATED_OUT = 'OUT = "docs/experiments/data/thing/baseline.json"\n'


def test_top_level_gated_writer_without_a_preflight_fails(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "scripts/make_artifact.py").write_text(GATED_OUT + "json.dump(payload, fh)\n")
    fails = [f for f in L.lint(root) if "make_artifact.py" in f]
    assert len(fails) == 1
    assert "docs/experiments/data/" in fails[0]


def test_top_level_gated_writer_with_the_preflight_passes(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "scripts/make_artifact.py").write_text(
        GATED_OUT
        + "preflight_gated_record_or_exit(REPO_ROOT, args.out, allow_dirty=args.allow_dirty)\n"
        + "json.dump(payload, fh)\n"
    )
    assert [f for f in L.lint(root) if "make_artifact.py" in f] == []


def test_family_3_accepts_the_spawner_guard_form(tmp_path: Path) -> None:
    """scripts/exp44/campaign.py is guarded via executed_code_provenance(...,
    out_path=), not the in-process preflight. Family 3 keys on WHERE records
    land, so it must accept every form the other families accept -- this was a
    real false positive on the first cut of the family."""
    root = _tree(tmp_path)
    (root / "scripts/spawner.py").write_text(
        GATED_OUT
        + "prov = executed_code_provenance(_REPO, binary, out_path=out, allow_dirty=args.allow_dirty)\n"
        + "json.dump(payload, fh)\n"
    )
    assert [f for f in L.lint(root) if "spawner.py" in f] == []


def test_family_3_ignores_a_script_that_only_reads_gated_data(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "scripts/reader.py").write_text(GATED_OUT + "open(P).read()\n")
    assert [f for f in L.lint(root) if "reader.py" in f] == []


def test_real_tree_is_clean() -> None:
    assert L.lint() == []

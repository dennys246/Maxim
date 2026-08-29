"""scripts/lint_harness_provenance.py — both families, on a synthetic scripts tree.

Verified to fail 4/4 in-process writers on the pre-fix origin/main tree (live_common.py,
ear_map.py, loudness_bench_poll.py, 9_hunger_relief_orient.py).
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


def test_spawner_family_unchanged(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "scripts/bench.py").write_text(
        'import subprocess\nsubprocess.run([sys.executable, "-m", "maxim", "--sim"])\n'
    )
    fails = L.lint(root)
    assert len(fails) == 1 and "assert_repo_interpreter" in fails[0]
    (root / "scripts/bench.py").write_text(
        'import subprocess\n_provenance.assert_repo_interpreter(R, b)\nsubprocess.run([sys.executable, "-m", "maxim", "--sim"])\n'
    )
    assert L.lint(root) == []


def test_real_tree_is_clean() -> None:
    assert L.lint() == []

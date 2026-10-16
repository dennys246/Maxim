"""scripts/lint_release_gates.py — the 1.4.0 decomposition release gate (owner decision 2026-10-15).

Every rule test drives ``main()`` on a fixture git repo with a tiny ``src/maxim`` package, its own function-length
baseline, a golden, evidence tests, a workflow and a gate file. Each attack the design names (approach + adversarial
notes, 2026-10-15) is applied and must FAIL; the clean twin of each must pass. The runtime twin is driven both on
hand-written JUnit and on a REAL pytest run whose conftest skips the evidence test. The last tests run the lint on
THIS checkout (root from this file's path, never ``maxim.__file__``): it passes at 1.3.x and, with the version forced
to 1.4.0, fails listing every outstanding item.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import lint_release_gates as L

GATES = "scripts/release_gates.json"
BASELINE = "src/maxim/utils/function_length_baseline.json"
SIM = "src/maxim/sim.py"
GOLDEN = "tests/fixtures/golden.json"
FIX = "tests/unit/test_fix.py"
FIX_ID = "tests/unit/test_fix.py::test_fixed"
ALPHA_ID = "tests/unit/test_alpha.py::test_alpha_pins_it"
GOLDEN_TEST = "tests/unit/test_golden.py::test_golden"

WORKFLOW = """\
on: {push: {}}
jobs:
  unit-tests:
    runs-on: x
    steps:
      - uses: actions/checkout@v4
        with: {fetch-depth: 0}
      - name: Run required fast suite
        run: python -m pytest tests/ --junitxml=fast-suite.xml --cov=src/maxim
      - name: Release gates, runtime twin
        run: python scripts/lint_release_gates.py --junit fast-suite.xml
  lint:
    runs-on: x
    steps:
      - uses: actions/checkout@v4
        with: {fetch-depth: 0}
      - name: Release gates
        run: python3 scripts/lint_release_gates.py
"""

RED = 'import pytest\n\n\n@pytest.mark.xfail(strict=True, reason="#1: flips with the fix")\ndef test_fixed():\n    assert 1 + 1 == 2\n'
GREEN = "import pytest\n\n\ndef test_fixed():\n    assert 1 + 1 == 2\n"
ALPHA = "def test_alpha_pins_it():\n    assert 'alpha'\n"


def fn(name: str, n: int, tag: str = "x") -> str:
    """A def whose span is exactly ``n`` lines."""
    return f"def {name}():\n" + "".join(f"    {tag}{i} = {i}\n" for i in range(n - 1))


def baseline(entries: dict, exceptions: list | None = None) -> str:
    return json.dumps(
        {
            "baseline_format_version": 2,
            "threshold": 200,
            "entries": [{"file": f, "qualname": q, "lines": n} for (f, q), n in entries.items()],
            "exceptions": exceptions or [],
        }
    )


def gate_dict(**kw) -> dict:
    g = {
        "decided": "2026-10-15",
        "source": "fixture",
        "functions": [{"file": SIM, "qualname": "big", "max_lines": 200, "why": "w"}],
        "debt_snapshot": [[SIM, "big"]],
        "accepted_new_debt": [],
        "arms": {
            "golden": GOLDEN,
            "golden_test": GOLDEN_TEST,
            "required": [{"arm": "alpha", "scope": "s", "evidence": []}],
        },
        "items": [{"id": "1", "ref": "#1", "scope": "s", "evidence": [], "red_gate": None}],
        "retired": [],
    }
    g.update(kw)
    return g


def gates_text(g: dict | None = None, *, key: str = "1.4.0", extra: dict | None = None) -> str:
    gates = {key: g if g is not None else gate_dict()}
    gates.update(extra or {})
    return json.dumps({"gate_format_version": 1, "gates": gates}, indent=1)


def _git(root: Path, *args: str) -> str:
    env = dict(
        os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t"
    )
    return subprocess.run(["git", *args], cwd=root, env=env, capture_output=True, text=True, check=True).stdout


class Repo:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, rel: str, text: str) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def read(self, rel: str) -> str:
        return (self.root / rel).read_text()

    def commit(self, msg: str = "change") -> str:
        _git(self.root, "add", "-A")
        _git(self.root, "commit", "-q", "--allow-empty", "-m", msg)
        return _git(self.root, "rev-parse", "HEAD").strip()

    def version(self, v: str, *, sources: tuple[str, ...] = ("pyproject", "init", "changelog")) -> None:
        if "pyproject" in sources:
            self.write("pyproject.toml", f'[project]\nname = "x"\nversion = "{v}"\n')
        if "init" in sources:
            self.write("src/maxim/__init__.py", f'__version__ = "{v}"\n')
        if "changelog" in sources:
            self.write("CHANGELOG.md", f"# Changelog\n\n## [Unreleased]\n\n## [{v}] - 2026-10-15\n")

    def gate(self, g: dict | None = None, **kw) -> None:
        self.write(GATES, gates_text(g, **kw))

    def edit_gate(self, fn_) -> None:
        doc = json.loads(self.read(GATES))
        fn_(doc["gates"]["1.4.0"])
        self.write(GATES, json.dumps(doc, indent=1))


def run(capsys, *argv: str) -> tuple[int, str]:
    rc = L.main(list(argv))
    out = capsys.readouterr()
    return rc, out.out + out.err


@pytest.fixture
def repo(tmp_path: Path, monkeypatch) -> Repo:
    """main: version 1.3.1, sim.py::big 300 lines pinned, a strict red gate for item 1, the golden without `alpha`,
    a wired workflow and the default gate, all committed."""
    r = Repo(tmp_path)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    r.version("1.3.1")
    r.write(SIM, fn("big", 300) + "\n\n" + fn("small", 10))
    r.write(BASELINE, baseline({(SIM, "big"): 300}))
    r.write(GOLDEN, json.dumps({"arms": {"old": {}}}))
    r.write(FIX, RED)
    r.write(".github/workflows/test.yml", WORKFLOW)
    r.gate()
    r.red = r.commit("base")  # type: ignore[attr-defined]
    monkeypatch.setattr(L, "REPO_ROOT", tmp_path)
    for var in ("GITHUB_EVENT_NAME", "GITHUB_ACTIONS"):
        monkeypatch.delenv(var, raising=False)
    return r


def land_alpha(r: Repo) -> str:
    r.write(GOLDEN, json.dumps({"arms": {"old": {}, "alpha": {}}}))
    r.write("tests/unit/test_alpha.py", ALPHA)
    return r.commit("arms land")


def satisfy_item(r: Repo, red: str) -> None:
    r.write(FIX, GREEN)
    r.edit_gate(lambda g: g["items"][0].update(evidence=[FIX_ID], red_gate={"commit": red, "nodeid": FIX_ID}))


def satisfy_arm(r: Repo) -> None:
    r.edit_gate(lambda g: g["arms"]["required"][0].update(evidence=[ALPHA_ID]))


def finish_function(r: Repo) -> None:
    r.write(SIM, fn("big", 150) + "\n\n" + fn("small", 10))
    r.write(BASELINE, baseline({}))


@pytest.fixture
def done(repo: Repo) -> Repo:
    """Everything the gate asks is done and committed, still at 1.3.1."""
    land_alpha(repo)
    satisfy_item(repo, repo.red)  # type: ignore[attr-defined]
    satisfy_arm(repo)
    finish_function(repo)
    repo.commit("done")
    return repo


# ── rule 1: the version ──────────────────────────────────────────────────────


def test_inactive_gate_passes_and_prints_the_outstanding_counts(repo, capsys):
    rc, out = run(capsys)
    assert rc == 0, out
    assert "release gate 1.4.0: inactive (version 1.3.1" in out
    assert "1/1 items outstanding" in out and "0/1 arms" in out and "1 functions not done (big 300)" in out
    assert "items: 0/1 satisfied; outstanding: 1" in out


def test_a_bump_to_1_4_0_fails_listing_every_outstanding_entry(repo, capsys):
    repo.version("1.4.0")
    rc, out = run(capsys)
    assert rc == 1
    assert "ACTIVE" in out and "item 1 is outstanding" in out and "arm alpha is outstanding" in out
    assert "function src/maxim/sim.py::big = 300 lines (pinned 300; max 200) is not done" in out


def test_a_finished_gate_passes_at_1_4_0(done, capsys):
    done.version("1.4.0")
    rc, out = run(capsys)
    assert rc == 0, out
    assert "release gate 1.4.0: ACTIVE (version 1.4.0); 0/1 items outstanding, 0 functions not done" in out


@pytest.mark.parametrize("source", ["pyproject", "init", "changelog"])
def test_bumping_any_one_version_source_activates_the_gate(repo, capsys, source):
    """Version-source evasion: the gate takes the MAX of the three sources."""
    repo.version("1.4.0", sources=(source,))
    rc, out = run(capsys)
    assert rc == 1 and "ACTIVE" in out and "item 1 is outstanding" in out


@pytest.mark.parametrize("v", ["1.4.0rc1", "1.4.0.dev0", "1.4.0a1", "1.4.0.post1", "2.0.0", "1.10.0"])
def test_prereleases_and_later_versions_are_gated_by_tuple_not_string(repo, capsys, v):
    repo.version(v)
    rc, out = run(capsys)
    assert rc == 1 and "ACTIVE" in out


def test_a_version_below_the_gate_stays_inactive(repo, capsys):
    repo.version("1.3.99")
    rc, out = run(capsys)
    assert rc == 0 and "inactive (version 1.3.99" in out


@pytest.mark.parametrize("bad", ["1!1.4.0", "1.4", "v1.4.0"])
def test_an_unparsable_version_fails_closed(repo, capsys, bad):
    repo.version(bad, sources=("pyproject",))
    rc, out = run(capsys)
    assert rc == 1 and "fails closed" in out and "ACTIVE" in out


def test_a_dynamic_version_fails_closed(repo, capsys):
    repo.write("pyproject.toml", '[project]\nname = "x"\ndynamic = ["version"]\n')
    rc, out = run(capsys)
    assert rc == 1 and "dynamic version" in out


# ── rule 2: shape ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda g: g.update(extra=1), "unknown keys ['extra']"),
        (lambda g: g["items"].append(dict(g["items"][0])), "listed twice"),
        (lambda g: g["functions"][0].update(max_lines=300), "max_lines must be an int in 1..200"),
        (lambda g: g["items"][0].update(ref="1"), "ref must be"),
        (lambda g: g["items"][0].update(evidence=[FIX_ID + "[a]"]), "no [param]"),
        (
            lambda g: g["retired"].append(
                {"kind": "item", "key": "nope", "reason": "r", "ref": "#1", "date": "2026-10-15"}
            ),
            "names an entry",
        ),
        (lambda g: g["items"][0].update(red_gate={"commit": "abc", "nodeid": FIX_ID}), "40-hex"),
        (lambda g: g.update(decided="Oct 15"), "YYYY-MM-DD"),
    ],
)
def test_a_malformed_gate_file_fails(repo, capsys, mutate, needle):
    repo.edit_gate(mutate)
    rc, out = run(capsys)
    assert rc == 1 and needle in out, out


def test_duplicate_json_keys_fail(repo, capsys):
    text = repo.read(GATES).replace('"decided": "2026-10-15"', '"decided": "2026-10-15", "decided": "2026-10-16"')
    repo.write(GATES, text)
    rc, out = run(capsys)
    assert rc == 1 and "duplicate JSON keys ['decided']" in out


def test_a_missing_gate_file_fails(repo, capsys):
    (repo.root / GATES).unlink()
    rc, out = run(capsys)
    assert rc == 1 and "unreadable" in out


# ── rule 3: items, evidence and red gates ────────────────────────────────────


def test_a_satisfied_item_counts(repo, capsys):
    satisfy_item(repo, repo.red)
    rc, out = run(capsys)
    assert rc == 0, out
    assert "items: 1/1 satisfied" in out


def test_evidence_without_a_red_gate_fails(repo, capsys):
    repo.write(FIX, GREEN)
    repo.edit_gate(lambda g: g["items"][0].update(evidence=[FIX_ID]))
    rc, out = run(capsys)
    assert rc == 1 and "evidence recorded without a red_gate" in out


def test_a_trivial_test_with_no_red_gate_history_fails(repo, capsys):
    """The test exists and passes, but it never sat as xfail(strict=True) at the cited commit."""
    repo.write("tests/unit/test_trivial.py", "def test_ok():\n    assert True\n")
    sha = repo.commit("a trivial test")
    nid = "tests/unit/test_trivial.py::test_ok"
    repo.edit_gate(lambda g: g["items"][0].update(evidence=[nid], red_gate={"commit": sha, "nodeid": nid}))
    rc, out = run(capsys)
    assert rc == 1 and "did not carry xfail(strict=True)" in out


def test_a_non_strict_xfail_is_not_a_red_gate(repo, capsys):
    repo.write(FIX, RED.replace("strict=True, ", ""))
    sha = repo.commit("lax")
    satisfy_item(repo, sha)
    rc, out = run(capsys)
    assert rc == 1 and "did not carry xfail(strict=True)" in out


def test_a_class_level_strict_xfail_is_a_red_gate(repo, capsys):
    repo.write(
        "tests/unit/test_cls.py",
        "import pytest\n\n\n@pytest.mark.xfail(strict=True)\nclass TestIt:\n    def test_x(self):\n        assert 1\n",
    )
    sha = repo.commit("red class")
    repo.write("tests/unit/test_cls.py", "class TestIt:\n    def test_x(self):\n        assert 1\n")
    nid = "tests/unit/test_cls.py::TestIt::test_x"
    repo.edit_gate(lambda g: g["items"][0].update(evidence=[nid], red_gate={"commit": sha, "nodeid": nid}))
    rc, out = run(capsys)
    assert rc == 0, out


def test_a_red_gate_commit_off_the_history_fails(repo, capsys):
    """A squash merge drops the red-gate commit: it is no longer an ancestor of HEAD."""
    _git(repo.root, "checkout", "-q", "-b", "side")
    repo.write("tests/unit/other.txt", "x")
    side = repo.commit("side")
    _git(repo.root, "checkout", "-q", "main")
    satisfy_item(repo, side)
    rc, out = run(capsys)
    assert rc == 1 and "not an ancestor of HEAD" in out


@pytest.mark.parametrize(
    ("text", "needle"),
    [
        (
            "import pytest\n\npytestmark = pytest.mark.skip\n\n\n" + GREEN.split("\n\n\n", 1)[1],
            "pytestmark carries ['skip']",
        ),
        (
            "import pytest\n\npytestmark = [pytest.mark.slow]\n\n\n" + GREEN.split("\n\n\n", 1)[1],
            "pytestmark carries ['slow']",
        ),
        (
            "import pytest\n\nMARKS = [pytest.mark.slow]\npytestmark = MARKS\n\n\n" + GREEN.split("\n\n\n", 1)[1],
            "not a literal mark",
        ),
        (
            "import pytest\n\nyaml = pytest.importorskip('yaml')\n\n\n" + GREEN.split("\n\n\n", 1)[1],
            "module level) references ['importorskip']",
        ),
        (
            GREEN.replace("def test_fixed", "@pytest.mark.skip(reason='x')\ndef test_fixed"),
            "decorator carries ['skip']",
        ),
        (
            GREEN.replace("def test_fixed", "@pytest.mark.skipif(True, reason='x')\ndef test_fixed"),
            "decorator carries ['skipif']",
        ),
        (GREEN.replace("def test_fixed", "@pytest.mark.requires_model_cache\ndef test_fixed"), "requires_model_cache"),
        (
            GREEN.replace(
                "def test_fixed():",
                "@pytest.mark.parametrize('a', [1, pytest.param(2, marks=pytest.mark.xfail)])\ndef test_fixed(a):",
            ),
            "decorator carries ['xfail']",
        ),
        (
            GREEN.replace("import pytest\n", "import pytest\n\nLATER = pytest.mark.xfail(strict=True)\n").replace(
                "def test_fixed", "@LATER\ndef test_fixed"
            ),
            "decorator carries ['xfail']",
        ),
        (GREEN.replace("    assert", "    pytest.skip('later')\n    assert"), "body references ['skip']"),
        (GREEN.replace("    assert", "    raise __import__('unittest').SkipTest('x')\n    assert"), "SkipTest"),
        (GREEN + "\n\ndef test_fixed():\n    assert 2\n", "bound exactly once"),
        ("from tests.unit.helpers import test_fixed\n", "not a def at its level"),
        (GREEN.replace("    assert 1 + 1 == 2", "    x = 1"), "never asserts"),
    ],
    ids=[
        "module-pytestmark-skip",
        "module-pytestmark-slow",
        "non-literal-pytestmark",
        "module-importorskip",
        "skip-decorator",
        "skipif-decorator",
        "model-cache-marker",
        "parametrized-xfail",
        "aliased-xfail-decorator",
        "body-skip",
        "body-SkipTest",
        "duplicate-def",
        "re-export",
        "no-assert",
    ],
)
def test_an_evidence_test_that_cannot_count_fails(repo, capsys, text, needle):
    satisfy_item(repo, repo.red)
    repo.write(FIX, text)
    rc, out = run(capsys)
    assert rc == 1 and needle in out, out


@pytest.mark.parametrize(
    ("body", "needle"),
    [
        (
            "class TestIt:\n    def __init__(self):\n        pass\n\n    def test_x(self):\n        assert 1\n",
            "has __init__",
        ),
        ("class TestIt:\n    __test__ = False\n\n    def test_x(self):\n        assert 1\n", "sets __test__"),
        (
            "import pytest\n\n\n@pytest.mark.skip\nclass TestIt:\n    def test_x(self):\n        assert 1\n",
            "decorator carries ['skip']",
        ),
        (
            "import pytest\n\n\nclass TestIt:\n    pytestmark = pytest.mark.xfail\n\n    def test_x(self):\n        assert 1\n",
            "pytestmark carries ['xfail']",
        ),
    ],
    ids=["init", "dunder-test", "class-skip", "class-pytestmark-xfail"],
)
def test_a_class_that_hides_its_test_fails(repo, capsys, body, needle):
    red = "import pytest\n\n\n@pytest.mark.xfail(strict=True)\nclass TestIt:\n    def test_x(self):\n        assert 1\n"
    repo.write("tests/unit/test_cls.py", red)
    sha = repo.commit("red")
    repo.write("tests/unit/test_cls.py", body)
    nid = "tests/unit/test_cls.py::TestIt::test_x"
    repo.edit_gate(lambda g: g["items"][0].update(evidence=[nid], red_gate={"commit": sha, "nodeid": nid}))
    rc, out = run(capsys)
    assert rc == 1 and needle in out, out


def test_evidence_outside_the_fast_suite_fails(repo, capsys):
    nid = "tests/integration/test_memory_hub.py::test_x"
    repo.write("tests/integration/test_memory_hub.py", "def test_x():\n    assert 1\n")
    repo.edit_gate(lambda g: g["items"][0].update(evidence=[nid], red_gate={"commit": repo.red, "nodeid": nid}))
    rc, out = run(capsys)
    assert rc == 1 and "outside the fast suite" in out


# ── rule 4: functions, descendants, debt snapshot ────────────────────────────


def test_a_renamed_gated_function_fails(done, capsys):
    done.write(SIM, fn("big_renamed", 150) + "\n\n" + fn("small", 10))
    done.version("1.4.0")
    rc, out = run(capsys)
    assert rc == 1 and "src/maxim/sim.py::big = NOT FOUND" in out


def test_a_short_wrapper_over_a_pinned_split_out_fails(done, capsys):
    """The body moved into `_body` (300, split_from big, recorded); `big` is a short wrapper. SF-6."""
    done.write(SIM, fn("big", 5) + "\n\n" + fn("_body", 300) + "\n\n" + fn("small", 10))
    exc = {
        "file": SIM,
        "qualname": "_body",
        "from": None,
        "to": 300,
        "split_from": {"file": SIM, "qualname": "big"},
        "date": "2026-10-15",
        "ref": "#1",
        "reason": "r",
    }
    done.write(BASELINE, baseline({(SIM, "_body"): 300}, [exc]))
    rc, out = run(capsys)
    assert rc == 0, out  # inactive: printed, not enforced
    assert "src/maxim/sim.py::_body = 300 lines (pinned 300; max 200) (descendant of src/maxim/sim.py::big)" in out
    done.version("1.4.0")
    rc, out = run(capsys)
    assert rc == 1 and "_body = 300 lines" in out and "is not done" in out


def test_a_split_out_cannot_be_blessed_by_accepted_new_debt(done, capsys):
    exc = {
        "file": SIM,
        "qualname": "_body",
        "from": None,
        "to": 300,
        "split_from": {"file": SIM, "qualname": "big"},
        "date": "2026-10-15",
        "ref": "#1",
        "reason": "r",
    }
    done.write(BASELINE, baseline({}, [exc]))
    rec = {"file": SIM, "qualname": "_body", "reason": "r", "ref": "#1", "date": "2026-10-15"}
    done.edit_gate(lambda g: g["accepted_new_debt"].append(rec))
    rc, out = run(capsys)
    assert rc == 1 and "that rule has no exception" in out


def test_rename_then_wrap_needs_an_accepted_new_debt_record(done, capsys):
    """SF-5: a free move leaves no record, so `moved` (300, pinned) is new debt outside the snapshot."""
    done.write(SIM, fn("big", 5) + "\n\n" + fn("moved", 300) + "\n\n" + fn("small", 10))
    done.write(BASELINE, baseline({(SIM, "moved"): 300}))
    done.version("1.4.0")
    rc, out = run(capsys)
    assert rc == 1 and "new debt src/maxim/sim.py::moved is outside debt_snapshot" in out
    done.version("1.3.1")  # the record lands before the bump: the bump diff may not touch the gate file
    rec = {"file": SIM, "qualname": "moved", "reason": "r", "ref": "#1", "date": "2026-10-15"}
    done.edit_gate(lambda g: g["accepted_new_debt"].append(rec))
    done.commit("accept")
    done.version("1.4.0")
    rc, out = run(capsys)
    assert rc == 0, out


def test_a_short_function_still_pinned_is_not_done(done, capsys):
    done.write(BASELINE, baseline({(SIM, "big"): 150}))
    done.version("1.4.0")
    rc, out = run(capsys)
    assert rc == 1 and "big = 150 lines (pinned 150; max 200) is not done" in out


def test_an_ambiguous_gated_function_is_not_done(done, capsys):
    done.write(
        SIM,
        "if X:\n"
        + "    "
        + fn("big", 5).replace("\n", "\n    ")
        + "\nelse:\n    "
        + fn("big", 5).replace("\n", "\n    "),
    )
    done.version("1.4.0")
    rc, out = run(capsys)
    assert rc == 1 and "big = AMBIGUOUS" in out


def test_a_retired_function_is_not_enforced(done, capsys):
    done.write(SIM, fn("renamed", 150))
    rec = {"kind": "function", "key": f"{SIM}::big", "reason": "r", "ref": "#1", "date": "2026-10-15"}
    done.edit_gate(lambda g: g["retired"].append(rec))
    done.commit("retire")
    done.version("1.4.0")
    rc, out = run(capsys)
    assert rc == 0, out
    assert f"retired function {SIM}::big" in out


def test_span_is_def_line_to_last_line_with_decorators_excluded():
    """NIT-2: the gate measures with lint_function_length.span; pin its semantics here."""
    tree = ast.parse("@dec\n@dec2\ndef f():\n    a = 1\n\n    b = 2\n")
    assert L.FL.span(tree.body[0]) == 4  # type: ignore[arg-type]


# ── rule 5: arms ─────────────────────────────────────────────────────────────


def test_an_arm_needs_its_golden_key_and_direct_evidence(repo, capsys):
    repo.write("tests/unit/test_alpha.py", ALPHA)
    satisfy_arm(repo)
    rc, out = run(capsys)
    assert rc == 1 and "carries no such arm" in out
    land_alpha(repo)
    rc, out = run(capsys)
    assert rc == 0, out
    assert "arms: 1/1 satisfied" in out


def test_a_slice_that_lands_before_the_arms_does_not_brick_ci(repo, capsys):
    """Review B1 (owner decision 2026-10-15): the arms are gated as COMPLETE only. When they land relative to the
    slices is review's (M56): a history rule over the pin made one early slice fail CI at every version, for good."""
    finish_function(repo)
    repo.commit("a slice lands first")
    land_alpha(repo)
    satisfy_arm(repo)
    repo.commit("the arms land after it")
    rc, out = run(capsys)
    assert rc == 0, out
    assert "arms: 1/1 satisfied" in out


# ── rule 7: history (shrink-only) ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda g: g.update(items=[]), "item 1 was removed"),
        (lambda g: g["functions"].clear(), "function src/maxim/sim.py::big was removed"),
        (lambda g: g["arms"]["required"][0].update(arm="beta"), "arm 'alpha' was removed"),
        (lambda g: g["items"][0].update(scope="weaker"), "item 1 `scope` changed"),
        (lambda g: g["debt_snapshot"].append([SIM, "small"]), "debt_snapshot changed"),
        (lambda g: g["arms"].update(golden="tests/fixtures/other.json"), "`golden` changed"),
        (lambda g: g["arms"]["required"][0].update(scope="weaker"), "arm 'alpha' scope changed"),
        (lambda g: g.update(decided="2026-10-16"), "`decided` changed"),
        (lambda g: g.update(source="rewritten"), "`source` changed"),
    ],
)
def test_the_gate_file_only_shrinks_its_obligations(repo, capsys, mutate, needle):
    repo.edit_gate(mutate)
    rc, out = run(capsys)
    assert rc == 1 and needle in out, out


def test_max_lines_may_fall_but_never_rise(repo, capsys):
    repo.edit_gate(lambda g: g["functions"][0].update(max_lines=150))
    repo.commit("tighten")
    rc, out = run(capsys)
    assert rc == 0, out
    repo.edit_gate(lambda g: g["functions"][0].update(max_lines=200))
    rc, out = run(capsys)
    assert rc == 1 and "max_lines rose 150 -> 200" in out


def test_a_gate_is_never_removed(repo, capsys):
    repo.write(GATES, gates_text(key="1.5.0"))
    rc, out = run(capsys)
    assert rc == 1 and "gate 1.4.0 was removed" in out


def test_history_reads_each_version_by_its_own_format(repo, capsys, monkeypatch):
    """Review S2: a schema-v2 HEAD (a new key) against v1 history passes, because each historical version is
    compared through ITS OWN format's immutable view; an old version that a tightened validator would now refuse
    (an extra key) still reads."""
    monkeypatch.setattr(L, "FORMAT_VERSION", 2)
    monkeypatch.setattr(L, "_GATE_KEYS", L._GATE_KEYS | {"new_key"})
    monkeypatch.setitem(L.HISTORY_VIEWS, 2, L._view_v1)
    doc = json.loads(repo.read(GATES))
    doc["gates"]["1.4.0"]["note"] = "a key v1's validator would refuse today"
    repo.write(GATES, json.dumps(doc))
    repo.commit("an old version with a since-refused key")
    doc = json.loads(gates_text(gate_dict(new_key=1)))
    doc["gate_format_version"] = 2
    repo.write(GATES, json.dumps(doc))
    rc, out = run(capsys)
    assert rc == 0, out
    doc["gates"]["1.4.0"]["items"] = []
    repo.write(GATES, json.dumps(doc))
    rc, out = run(capsys)
    assert rc == 1 and "item 1 was removed" in out  # the immutable fields are still compared


def test_an_unknown_historical_format_fails_closed(repo, capsys):
    """SF2: a version written in a format this lint has no reader for is a failure, never a skip, even when it
    would hide a dropped item."""
    doc = json.loads(repo.read(GATES))
    doc["gate_format_version"] = 99
    doc["gates"]["1.4.0"]["items"] = []
    repo.write(GATES, json.dumps(doc))
    repo.commit("format 99, items dropped")
    repo.gate()
    rc, out = run(capsys)
    assert rc == 1 and "is malformed" in out and "no history reader" in out and "a failure, never a skip" in out


def test_a_historical_version_missing_a_field_of_its_own_format_fails(repo, capsys):
    doc = json.loads(repo.read(GATES))
    del doc["gates"]["1.4.0"]["debt_snapshot"]
    repo.write(GATES, json.dumps(doc))
    repo.commit("broken v1")
    repo.gate()
    rc, out = run(capsys)
    assert rc == 1 and "is malformed" in out and "debt_snapshot" in out


def test_moving_an_item_to_a_later_gate_is_a_removal(repo, capsys):
    doc = json.loads(repo.read(GATES))
    item = doc["gates"]["1.4.0"].pop("items")[0]
    doc["gates"]["1.4.0"]["items"] = []
    later = gate_dict(items=[item])
    doc["gates"]["1.5.0"] = later
    repo.write(GATES, json.dumps(doc))
    rc, out = run(capsys)
    assert rc == 1 and "gate 1.4.0: item 1 was removed" in out


def test_retiring_is_append_only(repo, capsys):
    rec = {"kind": "item", "key": "1", "reason": "r", "ref": "#1", "date": "2026-10-15"}
    repo.edit_gate(lambda g: g["retired"].append(rec))
    rc, out = run(capsys)
    assert rc == 0 and "retired item 1" in out, out
    repo.commit("retire")
    repo.edit_gate(lambda g: g["retired"][0].update(reason="rewritten"))
    rc, out = run(capsys)
    assert rc == 1 and "`retired` is append-only" in out


def test_a_malformed_historical_version_fails(repo, capsys):
    repo.write(GATES, "{not json")
    repo.commit("broken")
    repo.gate()
    rc, out = run(capsys)
    assert rc == 1 and "is malformed" in out


def test_a_deleted_then_restored_gate_file_fails(repo, capsys):
    (repo.root / GATES).unlink()
    repo.commit("deleted")
    repo.gate()
    rc, out = run(capsys)
    assert rc == 1 and "is absent at first-parent commit" in out


# ── rule 8: the bump diff ────────────────────────────────────────────────────


def _branch(r: Repo) -> None:
    _git(r.root, "checkout", "-q", "-b", "release")


def test_the_bump_diff_may_not_touch_the_gate_file(done, capsys):
    _branch(done)
    done.version("1.4.0")
    rec = {"file": SIM, "qualname": "small", "reason": "r", "ref": "#1", "date": "2026-10-15"}
    done.edit_gate(lambda g: g["accepted_new_debt"].append(rec))
    done.commit("bump + last-minute record")
    rc, out = run(capsys)
    assert rc == 1 and "AND changes scripts/release_gates.json" in out


def test_a_bump_that_leaves_the_gate_file_alone_passes(done, capsys):
    _branch(done)
    done.version("1.4.0")
    done.commit("bump")
    rc, out = run(capsys)
    assert rc == 0, out


def test_on_push_the_base_is_the_first_parent(done, capsys, monkeypatch):
    done.version("1.4.0")
    done.edit_gate(lambda g: g.update(source="rewritten at the bump"))
    done.commit("bump on main")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    rc, out = run(capsys)
    assert rc == 1 and "AND changes scripts/release_gates.json" in out


def _bump_then_trivial(r: Repo) -> None:
    """Review SF1's shape: a branch whose first commit bumps to 1.4.0 and whose second is trivial."""
    _git(r.root, "checkout", "-q", "-b", "release")
    r.version("1.4.0")
    r.commit("bump")
    r.write("notes.txt", "trivial")
    r.commit("trivial")


@pytest.mark.parametrize("event", ["workflow_dispatch", "schedule", ""])
def test_a_non_push_event_judges_against_the_merge_base(repo, capsys, monkeypatch, event):
    """SF1: a dispatch can run on any branch head. Judged against HEAD^1 it read CLOSED (the parent already carried
    the bump) and skipped rule 8; against the merge-base it is ACTIVE and fails."""
    _bump_then_trivial(repo)
    monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    rc, out = run(capsys)
    assert rc == 1 and "release gate 1.4.0: ACTIVE" in out and "CLOSED" not in out
    assert "item 1 is outstanding" in out


def test_a_push_judges_against_its_first_parent(repo, capsys, monkeypatch):
    """A push reaches only main, through a merged PR: there, HEAD^1 carrying the release means it was released."""
    _bump_then_trivial(repo)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    rc, out = run(capsys)
    assert rc == 0 and "release gate 1.4.0: CLOSED" in out


def test_pushing_the_release_commit_itself_is_active(repo, capsys, monkeypatch):
    """SF2: the release commit's own push compares with the pre-release parent, so it stays fully gated."""
    _release_on_main(repo)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    rc, out = run(capsys)
    assert rc == 1 and "release gate 1.4.0: ACTIVE" in out and "item 1 is outstanding" in out


@pytest.mark.parametrize(
    "pyproject",
    [
        "[project\nname = broken\n",
        '[project]\nname = "x"\ndynamic = ["version"]\n',
        '[project]\nname = "x"\n',
        'project = "x"\n',
    ],
    ids=["unparsable-toml", "dynamic", "no-version", "project-not-a-table"],
)
def test_an_unreadable_base_pyproject_never_closes_the_gate(repo, capsys, monkeypatch, pyproject):
    """P6: with a `## [1.4.0]` header at the base but a pyproject the gate cannot read there, nothing is closed."""
    repo.version("1.4.0", sources=("changelog", "init"))
    repo.write("pyproject.toml", pyproject)
    repo.commit("a base whose pyproject cannot be read")
    repo.version("1.4.0", sources=("pyproject",))
    repo.commit("pyproject repaired")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    rc, out = run(capsys)
    assert rc == 1 and "release gate 1.4.0: ACTIVE" in out and "CLOSED" not in out


def test_nothing_is_closed_when_git_cannot_answer_locally(done, capsys, monkeypatch):
    """P7: a local run whose close-out check cannot read git closes nothing: every rule stays enforced."""
    _release_on_main(done)
    done.write(BASELINE, baseline({(SIM, "big"): 150}))
    done.commit("a pin that would only print were the gate closed")

    def broken(*a, **k):
        raise L.GitUnavailable("simulated: git is down")

    monkeypatch.setattr(L, "diff_base", broken)
    rc, out = run(capsys)
    assert rc == 1 and "CLOSED" not in out and "the close-out check skipped" in out
    assert "big = 150 lines (pinned 150; max 200) is not done" in out


def test_a_changelog_header_alone_does_not_close_the_gate(repo, capsys, monkeypatch):
    """Hardening: the base must carry the release in pyproject too, not only a CHANGELOG header."""
    repo.version("1.4.0", sources=("changelog",))
    repo.commit("a header written ahead")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    repo.write("notes.txt", "trivial")
    repo.commit("trivial")
    rc, out = run(capsys)
    assert rc == 1 and "release gate 1.4.0: ACTIVE" in out


# ── close-out (review S1) ────────────────────────────────────────────────────


def _release_on_main(r: Repo) -> None:
    r.version("1.4.0")
    r.commit("release 1.4.0")


def test_after_the_release_a_regrowth_only_prints(done, capsys):
    """The released header is at the BASE (HEAD^1 on a push): the gate is closed, its rules print."""
    _release_on_main(done)
    done.write(SIM, fn("big", 250) + "\n\n" + fn("small", 10))
    done.write(BASELINE, baseline({(SIM, "big"): 250}))
    done.write(FIX, RED.replace("strict=True, ", ""))  # the evidence test regresses too
    done.commit("regrowth after the release")
    rc, out = run(capsys)
    assert rc == 0, out
    assert "release gate 1.4.0: CLOSED" in out
    assert "closed gate 1.4.0, not enforced: function src/maxim/sim.py::big = 250 lines" in out
    assert "closed gate 1.4.0, not enforced: item 1: tests/unit/test_fix.py::test_fixed: decorator carries" in out


def test_a_closed_gate_still_enforces_shape_and_history(done, capsys):
    _release_on_main(done)
    done.edit_gate(lambda g: g.update(items=[]))
    rc, out = run(capsys)
    assert rc == 1 and "CLOSED" in out and "item 1 was removed" in out
    done.gate()
    done.write(GATES, done.read(GATES).replace('"decided": "2026-10-15"', '"decided": "Oct"'))
    rc, out = run(capsys)
    assert rc == 1 and "YYYY-MM-DD" in out


def test_the_release_diff_itself_is_not_closed(repo, capsys):
    """The other side of the boundary: the header the bump adds is at HEAD, not at the base, so the bump diff is
    fully gated."""
    _git(repo.root, "checkout", "-q", "-b", "release")
    _release_on_main(repo)
    rc, out = run(capsys)
    assert rc == 1 and "release gate 1.4.0: ACTIVE" in out and "item 1 is outstanding" in out
    assert "CLOSED" not in out


# ── git and environment ──────────────────────────────────────────────────────


def test_a_shallow_clone_in_ci_is_exit_2(repo, tmp_path, capsys, monkeypatch):
    repo.commit("second")
    clone = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth=1", f"file://{repo.root}", str(clone)], check=True)
    monkeypatch.setattr(L, "REPO_ROOT", clone)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    rc, out = run(capsys)
    assert rc == 2 and "shallow" in out


def test_no_git_skips_the_git_rules_locally_but_is_exit_2_in_ci(tmp_path, capsys, monkeypatch):
    r = Repo(tmp_path)
    r.version("1.3.1")
    r.write(SIM, fn("big", 300))
    r.write(BASELINE, baseline({(SIM, "big"): 300}))
    r.write(GOLDEN, json.dumps({"arms": {}}))
    r.write(".github/workflows/test.yml", WORKFLOW)
    r.gate()
    monkeypatch.setattr(L, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    rc, out = run(capsys)
    assert rc == 0 and "git-backed rules" in out and "skipped" in out
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    rc, out = run(capsys)
    assert rc == 2


def test_git_failing_mid_rule_is_exit_2_in_ci_and_a_skip_locally(repo, capsys, monkeypatch):
    satisfy_item(repo, repo.red)

    def broken(*a, **k):
        raise L.GitUnavailable("simulated: git show failed")

    monkeypatch.setattr(L, "show", broken)
    rc, out = run(capsys)
    assert rc == 0 and "skipped (simulated" in out
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    rc, out = run(capsys)
    assert rc == 2 and "git could not answer" in out


# ── rule 9: wiring ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("old", "new", "needle"),
    [
        (
            "      - name: Release gates\n        run: python3 scripts/lint_release_gates.py\n",
            "",
            "`lint`: needs exactly one step",
        ),
        (
            "run: python3 scripts/lint_release_gates.py",
            "run: python3 scripts/lint_release_gates.py || true",
            "`lint`: needs exactly one step",
        ),
        (
            "      - name: Release gates\n        run:",
            "      - name: Release gates\n        continue-on-error: true\n        run:",
            "may carry only `name` and `run`",
        ),
        (
            "      - name: Release gates, runtime twin\n        run:",
            "      - name: Release gates, runtime twin\n        if: always()\n        run:",
            "may carry only `name` and `run`",
        ),
        (" --junitxml=fast-suite.xml --cov", " --cov", "must pass `--junitxml=fast-suite.xml` once"),
        (
            "      - name: Release gates, runtime twin\n        run: python scripts/lint_release_gates.py --junit fast-suite.xml\n",
            "",
            "`unit-tests`: needs exactly one step",
        ),
        ("  lint:\n    runs-on: x\n", "  lint:\n    runs-on: x\n    if: false\n", "job-level `if`"),
        (
            "  lint:\n    runs-on: x\n    steps:\n      - uses: actions/checkout@v4\n        with: {fetch-depth: 0}\n",
            "  lint:\n    runs-on: x\n    steps:\n      - uses: actions/checkout@v4\n",
            "fetch-depth: 0",
        ),
        (
            "on: {push: {}}\n",
            "on: {push: {}}\ndefaults: {run: {shell: 'bash {0} || true'}}\n",
            "workflow-level `defaults`",
        ),
    ],
)
def test_the_ci_wiring_is_checked_from_both_steps(repo, capsys, tmp_path, old, new, needle):
    assert old in WORKFLOW
    repo.write(".github/workflows/test.yml", WORKFLOW.replace(old, new))
    rc, out = run(capsys)
    assert rc == 1 and needle in out, out
    xml = _junit(tmp_path, [("tests.unit.test_lint_release_gates", "test_x", "")])
    rc, out = run(capsys, "--junit", str(xml))
    assert rc == 1 and needle in out, out


def test_no_other_step_may_name_the_report(repo, capsys):
    wf = WORKFLOW.replace(
        "      - name: Release gates, runtime twin\n",
        "      - name: Rewrite\n        run: echo '<testsuites/>' > fast-suite.xml\n"
        "      - name: Release gates, runtime twin\n",
    )
    repo.write(".github/workflows/test.yml", wf)
    rc, out = run(capsys)
    assert rc == 1 and "only the fast suite and the release-gate step may name fast-suite.xml" in out


def test_the_twin_must_follow_the_suite(repo, capsys):
    wf = WORKFLOW.replace(
        "      - name: Release gates, runtime twin\n        run: python scripts/lint_release_gates.py --junit fast-suite.xml\n",
        "",
    ).replace(
        "      - name: Run required fast suite\n",
        "      - name: Release gates, runtime twin\n        run: python scripts/lint_release_gates.py --junit fast-suite.xml\n"
        "      - name: Run required fast suite\n",
    )
    repo.write(".github/workflows/test.yml", wf)
    rc, out = run(capsys)
    assert rc == 1 and "must come after the fast suite" in out


# ── rule 6: the runtime twin ─────────────────────────────────────────────────


def _junit(tmp: Path, cases: list[tuple[str, str, str]]) -> Path:
    """cases: (classname, name, outcome tag or '')."""
    body = "".join(
        f'<testcase classname="{c}" name="{n}">' + (f"<{t}/>" if t else "") + "</testcase>" for c, n, t in cases
    )
    p = tmp / "junit.xml"
    p.write_text(f'<?xml version="1.0"?><testsuites><testsuite name="pytest">{body}</testsuite></testsuites>')
    return p


ANCHOR = ("tests.unit.test_lint_release_gates", "test_anchor", "")


def test_the_twin_passes_when_every_evidence_case_passed(repo, capsys, tmp_path):
    satisfy_item(repo, repo.red)
    xml = _junit(tmp_path, [ANCHOR, ("tests.unit.test_fix", "test_fixed", "")])
    rc, out = run(capsys, "--junit", str(xml))
    assert rc == 0, out
    assert "1 evidence nodeids checked" in out


@pytest.mark.parametrize(
    ("cases", "needle"),
    [
        ([ANCHOR], "not in the fast suite's JUnit report"),
        ([ANCHOR, ("tests.unit.test_fix", "test_fixed", "skipped")], "case test_fixed skipped"),
        ([ANCHOR, ("tests.unit.test_fix", "test_fixed", "failure")], "case test_fixed failure"),
        (
            [ANCHOR, ("tests.unit.test_fix", "test_fixed[a]", ""), ("tests.unit.test_fix", "test_fixed[b]", "skipped")],
            "case test_fixed[b] skipped",
        ),
        ([("tests.unit.test_fix", "test_fixed", "")], "not the fast suite's report"),
    ],
)
def test_the_twin_refuses_a_missing_skipped_or_failed_case(repo, capsys, tmp_path, cases, needle):
    satisfy_item(repo, repo.red)
    rc, out = run(capsys, "--junit", str(_junit(tmp_path, cases)))
    assert rc == 1 and needle in out, out


def test_the_twin_checks_a_satisfied_arms_golden_case(repo, capsys, tmp_path):
    land_alpha(repo)
    satisfy_arm(repo)
    alpha = ("tests.unit.test_alpha", "test_alpha_pins_it", "")
    rc, out = run(capsys, "--junit", str(_junit(tmp_path, [ANCHOR, alpha])))
    assert rc == 1 and "tests/unit/test_golden.py::test_golden[alpha]" in out
    golden = ("tests.unit.test_golden", "test_golden[alpha]", "")
    rc, out = run(capsys, "--junit", str(_junit(tmp_path, [ANCHOR, alpha, golden])))
    assert rc == 0, out


def test_the_twin_only_prints_for_a_closed_gate(repo, capsys, tmp_path, monkeypatch):
    satisfy_item(repo, repo.red)
    repo.commit("satisfied")
    _release_on_main(repo)
    repo.write("notes.txt", "after the release")
    repo.commit("later")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    xml = _junit(tmp_path, [ANCHOR, ("tests.unit.test_fix", "test_fixed", "skipped")])
    rc, out = run(capsys, "--junit", str(xml))
    assert rc == 0, out
    assert "closed gate, not enforced: tests/unit/test_fix.py::test_fixed" in out
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")  # merge-base = HEAD here: still released, still closed
    rc, out = run(capsys, "--junit", str(xml))
    assert rc == 0, out


def test_the_twin_enforces_every_gate_when_git_cannot_answer(repo, capsys, tmp_path, monkeypatch):
    """P4: a gate that cannot be shown closed stays enforced: git down in --junit mode is not a close-out."""
    satisfy_item(repo, repo.red)
    repo.commit("satisfied")
    _release_on_main(repo)
    repo.write("notes.txt", "after the release")
    repo.commit("later")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")

    def broken(*a, **k):
        raise L.GitUnavailable("simulated: git is down")

    monkeypatch.setattr(L, "diff_base", broken)
    xml = _junit(tmp_path, [ANCHOR, ("tests.unit.test_fix", "test_fixed", "skipped")])
    rc, out = run(capsys, "--junit", str(xml))
    assert rc == 1 and "close-out unknown, every gate enforced" in out and "case test_fixed skipped" in out


def test_a_missing_junit_report_is_exit_2(repo, capsys, tmp_path):
    rc, out = run(capsys, "--junit", str(tmp_path / "absent.xml"))
    assert rc == 2 and "unreadable" in out


def test_a_conftest_skip_static_rules_cannot_see_is_caught_at_run_time(repo, capsys):
    """The evidence test is clean to the static rules; a conftest hook skips it. Run REAL pytest on the fixture repo
    and feed its JUnit to the twin."""
    satisfy_item(repo, repo.red)
    repo.write(
        "tests/conftest.py",
        "import pytest\n\n\ndef pytest_collection_modifyitems(items):\n"
        "    for it in items:\n        if it.name == 'test_fixed':\n            it.add_marker(pytest.mark.skip(reason='hidden'))\n",
    )
    repo.write(L.ANCHOR_TEST_FILE, "def test_anchor():\n    assert True\n")
    rc, out = run(capsys)
    assert rc == 0, out  # the static rules pass: they cannot see the hook
    xml = repo.root / "fast-suite.xml"
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml", str(xml), "tests/"],
        cwd=repo.root,
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"},
    )
    rc, out = run(capsys, "--junit", str(xml))
    assert rc == 1 and "case test_fixed skipped" in out, out


# ── this checkout ────────────────────────────────────────────────────────────

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def this_checkout(monkeypatch):
    monkeypatch.setattr(L, "REPO_ROOT", REPO)
    for var in ("GITHUB_EVENT_NAME", "GITHUB_ACTIONS"):
        monkeypatch.delenv(var, raising=False)


def test_this_checkout_passes_below_the_gate(this_checkout, capsys):
    rc, out = run(capsys)
    if "inactive" not in out:
        pytest.fail(f"this checkout is at or above a gate; the release-gate test must be revisited:\n{out}")
    assert rc == 0, out


def test_this_checkout_at_1_4_0_fails_listing_every_outstanding_item(this_checkout, capsys, monkeypatch):
    monkeypatch.setattr(
        L, "version_sources", lambda read: ({"pyproject": "1.4.0", "__init__": "1.4.0", "CHANGELOG": "1.4.0"}, [])
    )
    gates = L.parse_gates((REPO / L.GATES_REL).read_text())
    rc, out = run(capsys)
    assert rc == 1
    (g,) = [g for g in gates if g.key == "1.4.0"]
    done = {r["key"] for r in g.raw["retired"]} | {i["id"] for i in g.items if i["evidence"] and i["red_gate"]}
    for it in g.items:
        if it["id"] not in done:
            assert f"item {it['id']} is outstanding" in out, it["id"]


def test_the_real_gate_file_names_what_the_owner_decided():
    """Owner decisions 2026-10-15: three gated functions, five arms, the four #1288 lists with CORE_TOOLS exempt, and
    every #863 site with its file::symbol, pattern and source."""
    (g,) = L.parse_gates((REPO / L.GATES_REL).read_text())
    assert {(f["file"].rsplit("/", 1)[1], f["qualname"]) for f in g.functions} == {
        ("orchestrator.py", "start_simulation_mode"),
        ("agent_loop.py", "run_agentic_loop"),
        ("tool_dispatch.py", "execute_and_learn"),
    }
    assert all(f["max_lines"] == 200 for f in g.functions)
    assert [a["arm"] for a in g.arms["required"]] == [
        "persistent_agent",
        "imagination",
        "new_goal",
        "stall",
        "generative",
    ]
    assert "landed_before" not in g.arms  # review B1: the arms are gated as complete only
    ids = {i["id"]: i for i in g.items}
    assert {i for i in ids if i.startswith("1288.")} == {
        "1288.tool_dispatch-conversation",
        "1288.exec_agent-speak_tools",
        "1288.orch-stall-nudge",
        "1288.console-talk-tools",
    }
    assert all("CORE_TOOLS is exempt" in ids[i]["scope"] for i in ids if i.startswith("1288."))
    assert {
        "1263.plan-writers",
        "1263.memctx",
        "1263.teardown-finally",
        "1263.dn-restart-resubscribe",
        "1263.dn-stop",
        "1263.concept-extractor",
    } <= set(ids)
    n863 = [i for i in ids.values() if i["id"].startswith("863.")]
    assert all({"site", "source"} <= set(i) for i in n863)
    assert sorted(i["id"] for i in n863 if i["id"].startswith("863.orch.")) == [
        "863.orch.sem_trace_imagination_active",
        "863.orch.sem_trace_imagination_failed",
        "863.orch.sem_trace_tools",
    ]  # review SF3: DROP class only, each with its deciding reason
    assert all("WIDEN #863" in i["source"] and "Deciding reason" in i["source"] for i in n863 if ".orch." in i["id"])
    assert sorted(i["id"] for i in n863 if i["id"].startswith("863.loop.")) == [
        "863.loop.auto_sense",
        "863.loop.imagine",
        "863.loop.orient_to_audio.percept",
        "863.loop.orient_to_audio.reflex",
    ]  # review S3: #863's own close condition; no KEEP blocks, no explicit-form handlers
    assert len(g.items) == 30
    snapshot = {tuple(p) for p in g.raw["debt_snapshot"]}
    assert ("src/maxim/simulation/orchestrator.py", "start_simulation_mode") in snapshot

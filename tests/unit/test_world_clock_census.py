"""World clock A-pre PR 0: the census script (``scripts/world_clock_census.py``).

Planted sources prove each resolution form and each section. One test runs the census over this checkout and pins
sites the plan names, so a census that silently stops seeing a family fails here.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("world_clock_census", REPO / "scripts" / "world_clock_census.py")
assert _SPEC is not None and _SPEC.loader is not None
census_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["world_clock_census"] = census_mod
_SPEC.loader.exec_module(census_mod)

PATH = "src/maxim/agents/prompt_builder.py"  # a prompt-scope file, so planted sites get a scope


def _sites(src: str, path: str = PATH) -> list[tuple[str, str, bool]]:
    out = census_mod.scan_text(textwrap.dedent(src), path)
    return [(s.name, s.kind, s.call) for s in out.sites]


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ("import time\ntime.time()\n", "time.time"),
        ("import time as _t\n_t.time()\n", "time.time"),
        ("from time import time as now\nnow()\n", "time.time"),
        ("from time import monotonic\nmonotonic()\n", "time.monotonic"),
        ("from datetime import datetime\ndatetime.now()\n", "datetime.datetime.now"),
        ("import datetime as dt\ndt.datetime.now()\n", "datetime.datetime.now"),
        ("from datetime import datetime as DT\nDT.fromtimestamp(0)\n", "datetime.datetime.fromtimestamp"),
        ("from datetime import date\ndate.today()\n", "datetime.date.today"),
        ("import time\ntime.strftime('%H')\n", "time.strftime"),
        ("import time\ntime.localtime(0)\n", "time.localtime"),
    ],
)
def test_each_alias_form_resolves(src: str, expected: str) -> None:
    names = [name for name, _, _ in _sites(src)]
    assert names == [expected], names


def test_a_reference_counts_not_only_a_call() -> None:
    src = """
    import time
    from dataclasses import dataclass, field

    @dataclass
    class Stamp:
        timestamp: float = field(default_factory=time.time)
    """
    assert _sites(src) == [("time.time", "wall", False)]


def test_ids_and_randomness_are_their_own_kinds() -> None:
    src = """
    import random
    import uuid
    import numpy as np
    uuid.uuid4()
    random.random()
    np.random.rand(3)
    rng = random.Random(7)
    gen = np.random.default_rng(7)
    legacy = np.random.RandomState(7)
    """
    got = _sites(src)
    assert ("uuid.uuid4", "id", True) in got
    assert ("random.random", "random", True) in got
    assert ("numpy.random.rand", "random", True) in got
    # An owned generator is not a process-global draw, and the module names themselves are not references.
    assert all(name not in ("random.Random", "random", "numpy.random") for name, _, _ in got), got
    assert len(got) == 3, got


def test_a_chained_name_is_counted_once() -> None:
    assert _sites("import datetime\ndatetime.datetime.now()\n") == [("datetime.datetime.now", "calendar", True)]
    # Four names deep under a banned prefix: only the outermost reference is the site (the dedup mechanism).
    assert _sites("import numpy as np\nnp.random.mtrand.rand(2)\n") == [("numpy.random.mtrand.rand", "random", True)]


def test_a_conflicting_or_star_binding_is_reported_not_guessed() -> None:
    src = """
    import datetime
    def f():
        from datetime import datetime
        return datetime.now()
    """
    out = census_mod.scan_text(textwrap.dedent(src), PATH)
    assert len(out.conflicts) == 1 and "'datetime'" in out.conflicts[0], out.conflicts
    star = census_mod.scan_text("from time import *\n", PATH)
    assert len(star.conflicts) == 1 and "import *" in star.conflicts[0]
    # Two imports of one module to one target, or a clash between non-clock names, is no conflict.
    assert census_mod.scan_text("import time\ndef g():\n    import time\n", PATH).conflicts == []
    assert census_mod.scan_text("import json as j\nimport jinja2 as j\n", PATH).conflicts == []


def test_unrelated_names_are_not_sites() -> None:
    src = """
    import time
    import os
    time.sleep(1)
    os.path.getmtime('x')
    def time_left(): return 1
    """
    assert _sites(src) == []


def test_the_enclosing_function_is_qualified() -> None:
    src = """
    import time
    class Pool:
        def render(self):
            def inner():
                return time.time()
            return inner()
    x = time.time()
    """
    out = census_mod.scan_text(textwrap.dedent(src), PATH)
    assert [s.function for s in out.sites] == ["Pool.render.inner", "<module>"]


def test_stamp_subtractions_name_the_other_clock() -> None:
    src = """
    import time
    def age(memory):
        return time.time() - memory.timestamp
    def gap(a, b):
        return b.timestamp - a.timestamp
    """
    out = census_mod.scan_text(textwrap.dedent(src), "src/maxim/memory/x.py")
    assert [(r.function, r.other_clock) for r in out.subtractions] == [("age", "wall"), ("gap", "caller-supplied")]


def test_every_stamp_reader_form_is_seen() -> None:
    src = """
    import time
    def a(meta): return time.time() - meta.get("timestamp", time.time())
    def b(mem): return time.time() - getattr(mem, "timestamp", 0)
    def c(event): return time.time() - event["timestamp"]
    def d(entry): return time.time() - entry.accessed_at
    def e(rec): return rec.created_at - rec.consolidated_at
    def f(x): return time.time() - x.duration
    """
    out = census_mod.scan_text(textwrap.dedent(src), "src/maxim/memory/x.py")
    got = [(r.function, r.attr, r.form) for r in out.subtractions]
    assert got == [
        ("a", "timestamp", ".get"),
        ("b", "timestamp", "getattr"),
        ("c", "timestamp", "subscript"),
        ("d", "accessed_at", "attribute"),
        ("e", "created_at", "attribute"),
    ], got


def test_stamp_writers_record_their_input_clock() -> None:
    src = """
    import time
    def make(now):
        a = Percept(timestamp=time.time())
        b = PainSignal(kind=1, timestamp=now)
        c = Percept(text="no stamp")
    """
    out = census_mod.scan_text(textwrap.dedent(src), "src/maxim/agents/x.py")
    assert [(w.target, w.input_clock) for w in out.writers] == [
        ("Percept", "wall"),
        ("PainSignal", "caller-supplied"),
    ]


def test_loader_constructors_and_assignments_are_stamp_writers() -> None:
    src = """
    import time
    class EpisodicMemory:
        @classmethod
        def from_dict(cls, d):
            return cls(timestamp=d["timestamp"], accessed_at=d["accessed_at"])
    class Other:
        @classmethod
        def from_dict(cls, d):
            return cls(timestamp=d["timestamp"])
    def touch(memory):
        memory.accessed_at = time.time()
        memory.name = "x"
    """
    out = census_mod.scan_text(textwrap.dedent(src), "src/maxim/memory/x.py")
    got = [(w.target, w.attrs, w.input_clock) for w in out.writers]
    assert got == [
        ("EpisodicMemory (cls)", "accessed_at,timestamp", "caller-supplied"),
        ("assignment .accessed_at", "accessed_at", "wall"),
    ], got


def test_temporal_callers_record_input_and_output_use() -> None:
    src = """
    import time
    def f(self):
        sig = TemporalSignature.now()
        self._scn.register(time.time())
        return f"phase {self.scn.current_phase()}"
    def g(self, ts):
        return TemporalSignature.from_timestamp(ts).to_dict()
    """
    out = census_mod.scan_text(textwrap.dedent(src), "src/maxim/agents/statistician_agent.py")
    got = [(t.callee, t.input_clock, t.output_use) for t in out.temporal]
    assert got == [
        ("TemporalSignature.now", "wall (inside TemporalSignature.now)", "assigned"),
        ("self._scn.register", "wall", "statement"),
        ("self.scn.current_phase", "caller-supplied", "rendered (f-string)"),
        ("TemporalSignature.from_timestamp", "caller-supplied", "returned"),
    ]


@pytest.mark.parametrize(
    ("path", "function", "kind", "expected"),
    [
        # Ids and randomness are OD-A7's wherever they sit, even inside a fenced file (D-WC1).
        ("src/maxim/runtime/agent_loop.py", "run", "id", "OD-A7, after the 1.4.0 cut (D-WC1)"),
        ("src/maxim/runtime/agent_loop.py", "run", "wall", "A-post: fenced file (OD-A8)"),
        ("src/maxim/agents/prompt_builder.py", "PromptBuilder.build_prompt", "calendar", "PR 2: the port (OD-A3)"),
        ("src/maxim/memory/hippocampus.py", "Hippocampus.store_observation", "wall", "PR 4"),
        ("src/maxim/proprioception/pain.py", "PainDetector._check_for_pain", "wall", "allow: motor-io"),
        ("src/maxim/integration/memory_hub.py", "MemoryHub._on_memory_captured_scn", "wall", "with the stamp half"),
        ("src/maxim/memory/hippocampus_consolidation.py", "ConsolidationMixin.sleep", "wall", "A-post: OD-A4 stamp"),
        ("src/maxim/memory/hippocampus.py", "Hippocampus.unnamed_reader", "wall", "UNPLACED"),
        ("src/maxim/peer/client.py", "Client.ping", "wall", "outside M72's scopes"),
    ],
)
def test_placement_rules(path: str, function: str, kind: str, expected: str) -> None:
    assert census_mod.placement_of(path, function, kind).startswith(expected)


def test_scopes_follow_the_guard_note() -> None:
    assert census_mod.scope_of("src/maxim/agents/context_pool.py") == "machinery+prompt"
    assert census_mod.scope_of("src/maxim/memory/atl.py") == "machinery"
    assert census_mod.scope_of("src/maxim/tools/introspection.py") == "prompt"
    assert census_mod.scope_of("src/maxim/peer/client.py") == "-"
    # A directory prefix needs its slash: decisions_extra.py is not under decisions/.
    assert census_mod.scope_of("src/maxim/decisions_extra.py") == "-"


def test_a_file_that_does_not_parse_raises() -> None:
    with pytest.raises(SyntaxError):
        census_mod.scan_text("def broken(:\n", PATH)


def _git_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    if files:
        subprocess.run(["git", "add", *files], cwd=tmp_path, check=True)
    return tmp_path


def test_an_empty_tracked_tree_is_an_error(tmp_path: Path) -> None:
    root = _git_tree(tmp_path, {})
    (root / "src" / "maxim").mkdir(parents=True)
    assert census_mod.main(["--root", str(root)]) == 2


def test_a_tracked_file_that_does_not_parse_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _git_tree(tmp_path, {"src/maxim/ok.py": "x = 1\n", "src/maxim/deep/bad.py": "def broken(:\n"})
    assert census_mod.main(["--root", str(root)]) == 2
    assert "src/maxim/deep/bad.py" in capsys.readouterr().err


def test_json_output_on_a_clean_tree(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _git_tree(tmp_path, {"src/maxim/deep/a.py": "import time\ndef f():\n    return time.time()\n"})
    assert census_mod.main(["--root", str(root), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [(s["path"], s["function"], s["name"]) for s in data["sites"]] == [("src/maxim/deep/a.py", "f", "time.time")]
    assert data["conflicts"] == []


def test_an_untracked_tree_is_an_error_not_an_empty_census(tmp_path: Path) -> None:
    (tmp_path / "src" / "maxim").mkdir(parents=True)
    (tmp_path / "src" / "maxim" / "a.py").write_text("import time\ntime.time()\n")
    with pytest.raises(census_mod.CensusError):
        census_mod.tracked_files(tmp_path)
    assert census_mod.main(["--root", str(tmp_path)]) == 2


def test_only_tracked_files_are_scanned(tmp_path: Path) -> None:
    src = tmp_path / "src" / "maxim"
    src.mkdir(parents=True)
    (src / "tracked.py").write_text("import time\ntime.time()\n")
    (src / "scratch.py").write_text("import time\ntime.time()\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "src/maxim/tracked.py"], cwd=tmp_path, check=True)
    assert census_mod.tracked_files(tmp_path) == ["src/maxim/tracked.py"]
    assert [s.path for s in census_mod.census(tmp_path).sites] == ["src/maxim/tracked.py"]


def test_the_census_of_this_checkout_sees_the_named_families() -> None:
    result = census_mod.census(REPO)
    placed = {(s.path, s.function, s.placement) for s in result.sites}

    def has(path: str, function: str, placement_prefix: str) -> bool:
        return any(p == path and f == function and pl.startswith(placement_prefix) for p, f, pl in placed)

    assert has("src/maxim/agents/prompt_builder.py", "PromptBuilder.build_prompt", "PR 2")
    assert has("src/maxim/agents/prompt_builder.py", "PromptBuilder._build_engage_prompt", "PR 2")
    assert has("src/maxim/memory/hippocampus.py", "Hippocampus.store_observation", "PR 4")
    assert has("src/maxim/runtime/tool_dispatch.py", "execute_and_learn", "A-post with B2")
    assert has("src/maxim/proprioception/pain_bus.py", "PainBus.publish", "PR 5")
    assert any(s.path == "src/maxim/models/language/cost_tracker.py" and s.kind == "wall" for s in result.sites)
    # Every section is populated on today's tree, and the loader of every persisted Hippocampus stamp is a writer.
    assert result.writers and result.subtractions and result.temporal
    assert any(w.function == "EpisodicMemory.from_dict" for w in result.writers)


def test_every_placement_rule_still_matches_the_tree() -> None:
    """A rule that places nothing names a function or file that moved, so its sites fall through unseen. A rule
    that is empty on purpose says why; one that says so but matches something has a stale reason."""
    result = census_mod.census(REPO)
    problems = []
    for rule, hits in zip(census_mod.PLACEMENT_RULES, census_mod.rule_hits(result), strict=True):
        if rule.empty_today is None and hits == 0:
            problems.append(f"matches nothing: {rule.prefix!r} {rule.func!r} -> {rule.placement}")
        if rule.empty_today is not None and hits:
            problems.append(f"marked empty but matches {hits}: {rule.prefix!r} {rule.func!r}")
    assert not problems, "\n".join(problems)

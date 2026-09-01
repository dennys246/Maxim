"""Exp 42 harness + analyzer smoke tests — CI-safe (no subprocess / no LLM).

Covers docs/experiments/42_substrate_primary_preference.md §4 (frozen metrics)
and §5 (disposition matrix), plus the harness's exploitation-phase extraction
(discovery point → safe_pref / id_pref_a), the C2 telemetry net reduction, and
mock-mode wiring. Verdict-branch tests build records by hand so the FROZEN
both-arms H1 + K-validity math is exercised directly (the mock harness emits a
single canned GRADUATE pattern; these craft the matrix corners).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ANALYZER_PATH = _REPO_ROOT / "scripts" / "analyze_exp42_preference.py"
_HARNESS_PATH = _REPO_ROOT / "scripts" / "benchmark_exp42_preference.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def analyzer():
    return _load(_ANALYZER_PATH, "exp42_analyzer")


@pytest.fixture(scope="module")
def harness():
    return _load(_HARNESS_PATH, "exp42_harness")


# ── record builders ───────────────────────────────────────────────────────


def _rec(
    arm: str,
    seed: int,
    *,
    safe_pref: float | None,
    id_pref_a: float = 0.5,
    n_exploit: int = 14,
    harm_net: float | None = -0.25,
    safe_net: float | None = 0.14,
) -> dict[str, Any]:
    return {
        "experiment": "exp42",
        "arm": arm,
        "seed": seed,
        "n_actions": 40,
        "n_contact": n_exploit + 2,
        "n_exploit": n_exploit,
        "safe_pref": safe_pref,
        "id_pref_a": id_pref_a,
        "harm_net": harm_net,
        "safe_net": safe_net,
    }


def _dataset(*, a_pref, b_pref, a_id=0.1, b_id=0.9, n=5, jitter=0.0, n_exploit=14):
    """Build a 2-arm dataset. a_pref/b_pref are per-arm safe_pref means."""
    recs: list[dict[str, Any]] = []
    for i in range(n):
        d = jitter * (i - n // 2)  # symmetric jitter → non-zero SD, mean preserved
        recs.append(_rec("cradle_pref_a", 42 + i, safe_pref=a_pref + d, id_pref_a=a_id, n_exploit=n_exploit))
        recs.append(_rec("cradle_pref_b", 42 + i, safe_pref=b_pref + d, id_pref_a=b_id, n_exploit=n_exploit))
    return recs


# ── analyzer verdict matrix ───────────────────────────────────────────────


def test_graduate_both_arms(analyzer):
    """Safe preferred in BOTH arms → H1 pass → GRADUATE, exit 0."""
    recs = _dataset(a_pref=0.9, b_pref=0.88, jitter=0.02)
    r = analyzer.analyze(recs)
    assert r.h1_pass is True
    assert r.substrate_signal is True
    assert r.exit_code == analyzer.EXIT_GRADUATE
    assert "GRADUATE" in r.verdict


def test_partial_one_arm(analyzer):
    """Safe preferred in only one arm (the other below threshold) → PARTIAL, exit 4."""
    recs = _dataset(a_pref=0.9, b_pref=0.4, jitter=0.02)
    r = analyzer.analyze(recs)
    assert r.h1_pass is False
    assert r.exit_code == analyzer.EXIT_PARTIAL
    assert "PARTIAL" in r.verdict
    # the winning arm is named
    assert "cradle_pref_a" in r.verdict


def test_reframe_neither_arm(analyzer):
    """Neither arm prefers safe → REFRAME, exit 5."""
    recs = _dataset(a_pref=0.45, b_pref=0.5, jitter=0.02)
    r = analyzer.analyze(recs)
    assert r.h1_pass is False
    assert r.exit_code == analyzer.EXIT_REFRAME
    assert "REFRAME" in r.verdict


def test_void_floored(analyzer):
    """All seeds below the K validity gate → no valid seeds → VOID."""
    recs = _dataset(a_pref=0.9, b_pref=0.9, n_exploit=3)  # < K=10
    r = analyzer.analyze(recs)
    assert r.void is True
    assert "VOID" in r.verdict
    assert r.exit_code == analyzer.EXIT_PARTIAL


def test_void_missing_arm(analyzer):
    recs = [_rec("cradle_pref_a", 42, safe_pref=0.9)]
    r = analyzer.analyze(recs)
    assert r.void is True


def test_threshold_boundary_inclusive(analyzer):
    """safe_pref exactly at 0.66 PASSES (>= threshold)."""
    recs = _dataset(a_pref=0.66, b_pref=0.66)
    r = analyzer.analyze(recs)
    assert r.h1_pass is True
    assert r.exit_code == analyzer.EXIT_GRADUATE


def test_floored_seeds_excluded_but_arm_valid(analyzer):
    """A mix of floored + valid seeds: floored excluded, arm still scored on valid."""
    recs = _dataset(a_pref=0.9, b_pref=0.9, n=5)
    # add 2 floored seeds per arm (should be excluded, not drag the mean)
    recs.append(_rec("cradle_pref_a", 99, safe_pref=0.0, n_exploit=2))
    recs.append(_rec("cradle_pref_b", 99, safe_pref=0.0, n_exploit=2))
    r = analyzer.analyze(recs)
    assert r.arms["cradle_pref_a"].n_seeds_valid == 5
    assert r.arms["cradle_pref_a"].n_floored == 1
    assert r.exit_code == analyzer.EXIT_GRADUATE


def test_c1_identity_flip(analyzer):
    """id_pref_a flips: low in arm A (alpha harmful), high in arm B (alpha safe)."""
    recs = _dataset(a_pref=0.9, b_pref=0.9, a_id=0.1, b_id=0.9)
    r = analyzer.analyze(recs)
    assert r.c1_flip is not None
    assert r.c1_flip >= analyzer.C1_FLIP_THRESHOLD
    assert r.c1_pass is True


def test_c2_relative_learning(analyzer):
    """C2 passes when harm net < safe net in both arms (relative)."""
    recs = _dataset(a_pref=0.9, b_pref=0.9)
    r = analyzer.analyze(recs)
    assert r.c2_pass is True


def test_c2_skipped_without_telemetry(analyzer):
    recs = _dataset(a_pref=0.9, b_pref=0.9)
    for rec in recs:
        rec["harm_net"] = None
        rec["safe_net"] = None
    r = analyzer.analyze(recs)
    assert r.c2_pass is None


# ── harness metric extraction ─────────────────────────────────────────────


def test_contact_predicate(harness):
    assert harness._is_contact("warmth_beta_safe_warm_self")
    assert harness._is_contact("warmth_alpha_harm_touch")
    assert not harness._is_contact("warmth_beta_safe_observe")  # observe is not contact
    assert not harness._is_contact("sense_scene")


def test_safety_and_identity_labels(harness):
    assert harness._safety("warmth_beta_safe_warm_self") == "safe"
    assert harness._safety("warmth_alpha_harm_touch") == "harm"
    assert harness._identity("warmth_alpha_harm_touch") == "alpha"
    assert harness._identity("warmth_beta_safe_warm_self") == "beta"


def test_exploitation_phase_discovery(harness):
    """safe_pref is computed only over contacts AFTER both sources discovered."""
    tools = [
        "warmth_alpha_harm_warm_self",  # discovery: harm seen
        "warmth_beta_safe_warm_self",  # discovery: safe seen → discovery_pos here
        # exploitation phase (4 contacts): 3 safe, 1 harm → safe_pref 0.75
        "warmth_beta_safe_warm_self",
        "warmth_beta_safe_touch",
        "warmth_alpha_harm_warm_self",
        "warmth_beta_safe_warm_self",
    ]
    m = harness.compute_run_metrics(tools)
    assert m["discovery_pos"] == 1
    assert m["n_exploit"] == 4
    assert m["safe_pref"] == 0.75
    # alpha is the harmful identity in arm A: 1 of 4 exploit contacts → 0.25
    assert m["id_pref_a"] == 0.25


def test_no_exploitation_phase_when_one_source(harness):
    """Only one source ever engaged → no discovery → safe_pref None (floored)."""
    tools = ["warmth_beta_safe_warm_self"] * 6
    m = harness.compute_run_metrics(tools)
    assert m["discovery_pos"] is None
    assert m["n_exploit"] == 0
    assert m["safe_pref"] is None


def test_empty_action_sequence(harness):
    m = harness.compute_run_metrics([])
    assert m["n_contact"] == 0
    assert m["safe_pref"] is None


def test_extract_learning_nets(harness):
    cl = {
        "tool:warmth_alpha_harm_warm_self": {"pos": 0.0, "neg": 0.64, "net": -0.32},
        "tool:warmth_beta_safe_warm_self": {"pos": 0.5, "neg": 0.0, "net": 0.16},
        "tool:sense_scene": {"pos": 0.9, "neg": 0.0, "net": 0.9},  # non-contact, ignored
    }
    nets = harness.extract_learning_nets(cl)
    assert nets["harm_net"] == -0.32
    assert nets["safe_net"] == 0.16


# ── end-to-end: mock harness → analyzer = GRADUATE ────────────────────────


def test_mock_pipeline_graduates(harness, analyzer, tmp_path):
    out = tmp_path / "exp42_mock.jsonl"
    rc = harness.main(["--mock", "--trials", "5", "--out", str(out)])
    assert rc == 0
    recs = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(recs) == 10  # 2 arms × 5 seeds
    arms = {r["arm"] for r in recs}
    assert arms == {"cradle_pref_a", "cradle_pref_b"}
    result = analyzer.analyze(recs)
    assert result.exit_code == analyzer.EXIT_GRADUATE
    # both arms prefer safe; identity preference flips between arms
    assert result.c1_pass is True


def test_mock_resume_skips_existing(harness, tmp_path):
    out = tmp_path / "exp42_resume.jsonl"
    harness.main(["--mock", "--trials", "2", "--out", str(out)])
    n1 = len(out.read_text().splitlines())
    harness.main(["--mock", "--trials", "2", "--out", str(out), "--resume"])
    n2 = len(out.read_text().splitlines())
    assert n1 == n2 == 4  # no duplicates on resume


# ── Exp 44 fork: aut_mode threading + provenance (CI-safe, no subprocess) ──


class TestExp44HarnessFork:
    """The Exp 44 additions must leave Exp 42 byte-compatible (default
    aut_mode=substrate-primary) and record arm provenance per run."""

    def test_record_default_mode_is_substrate_primary(self):
        harness = _load(_HARNESS_PATH, "bench_exp42_fork_default")
        rec = harness._record("cradle_pref_a", 42, ["warmth_beta_safe_warm_self"], {}, mock=True, git_hash="x")
        assert rec["aut_mode"] == "substrate-primary"
        assert rec["env_body_state_prompt"] == ""
        assert rec["env_coach_body_layers_disabled"] == ""

    def test_record_llm_primary_provenance(self, monkeypatch):
        harness = _load(_HARNESS_PATH, "bench_exp42_fork_llm")
        monkeypatch.setenv("MAXIM_ENABLE_BODY_STATE_PROMPT", "1")
        monkeypatch.setenv("MAXIM_DISABLE_COACH_BODY_LAYERS", "1")
        rec = harness._record(
            "cradle_pref_a", 42, ["warmth_beta_safe_warm_self"], {}, mock=True, git_hash="x", aut_mode="llm-primary"
        )
        assert rec["aut_mode"] == "llm-primary"
        assert rec["env_body_state_prompt"] == "1"
        assert rec["env_coach_body_layers_disabled"] == "1"

    def test_cli_rejects_unknown_aut_mode(self):
        harness = _load(_HARNESS_PATH, "bench_exp42_fork_cli")
        with pytest.raises(SystemExit):
            harness.main(["--out", "/tmp/x.jsonl", "--aut-mode", "bogus"])


class TestExp44ArmDerivation:
    """ablation_arm must be derived with the SAME truthy semantics the
    runtime parsers use (1/true/t/yes/y/on) — grouping on raw env echoes
    is fragile ("1" vs "true" vs " YES ")."""

    def test_arm_a_default(self, monkeypatch):
        harness = _load(_HARNESS_PATH, "bench_exp44_arm_a")
        monkeypatch.delenv("MAXIM_ENABLE_BODY_STATE_PROMPT", raising=False)
        monkeypatch.delenv("MAXIM_DISABLE_COACH_BODY_LAYERS", raising=False)
        assert harness._ablation_arm() == "A"

    def test_arm_b(self, monkeypatch):
        harness = _load(_HARNESS_PATH, "bench_exp44_arm_b")
        monkeypatch.setenv("MAXIM_ENABLE_BODY_STATE_PROMPT", "y")  # t/y must count
        monkeypatch.setenv("MAXIM_DISABLE_COACH_BODY_LAYERS", " TRUE ")
        assert harness._ablation_arm() == "B"

    def test_arm_c(self, monkeypatch):
        harness = _load(_HARNESS_PATH, "bench_exp44_arm_c")
        monkeypatch.setenv("MAXIM_ENABLE_BODY_STATE_PROMPT", "t")
        monkeypatch.delenv("MAXIM_DISABLE_COACH_BODY_LAYERS", raising=False)
        assert harness._ablation_arm() == "C"

    def test_inconsistent_flagged(self, monkeypatch):
        harness = _load(_HARNESS_PATH, "bench_exp44_arm_x")
        monkeypatch.delenv("MAXIM_ENABLE_BODY_STATE_PROMPT", raising=False)
        monkeypatch.setenv("MAXIM_DISABLE_COACH_BODY_LAYERS", "1")
        assert harness._ablation_arm() == "inconsistent"


class TestOperatorGuards:
    """Guards for the three operator-surface traps of 2026-09-01.

    None was catchable by a test before, because each lived in the seam
    between a tool and the person driving it. All three fired during one
    post-D53 re-validation, and each was recoverable only because data
    happened to survive.
    """

    _MARKER_LINE = "<!-- Analyzer appends '## Results' sections below this line -->\n"

    def _doc(self, tmp_path, below: str):
        p = tmp_path / "exp.md"
        p.write_text("# Exp 42\n\nintro\n\n" + self._MARKER_LINE + "\n" + below)
        return p

    def test_append_preserves_a_recorded_verdict(self, analyzer, tmp_path):
        """THE 2026-09-01 near-miss: --force replaced main's frozen
        2026-06-23 Results section — both tables plus the
        B7-is-not-load-bearing analysis — with one generated block."""
        frozen = "## Results\n\nFrozen run: git `0d6ca70f`. GRADUATE.\n"
        doc = self._doc(tmp_path, frozen)
        analyzer._emit_to_doc(doc, "## Results — re-validation\n\nGRADUATE.\n", append=True)
        out = doc.read_text()
        assert "0d6ca70f" in out, "the frozen record must survive an append"
        assert out.index("## Results\n") < out.index("## Results — re-validation")
        assert out.count("\n## Results") == 2  # marker line also contains the phrase

    def test_default_refuses_and_recommends_append_before_force(self, analyzer, tmp_path):
        """The old refusal existed to protect exactly this content, but its
        message offered only --force. Steering to the destructive option is
        how the record got destroyed."""
        doc = self._doc(tmp_path, "## Results\n\nGRADUATE.\n")
        with pytest.raises(SystemExit) as e:
            analyzer._emit_to_doc(doc, "new\n")
        msg = str(e.value)
        assert "--append" in msg and "--force" in msg
        assert msg.index("--append") < msg.index("--force"), "recommend the safe mode first"
        assert "GRADUATE" in doc.read_text(), "a refusal must not write"

    def test_force_still_replaces(self, analyzer, tmp_path):
        """Kept deliberately: correcting a section this analyzer just wrote."""
        doc = self._doc(tmp_path, "## Results\n\nold\n")
        analyzer._emit_to_doc(doc, "## Results\n\nnew\n", force=True)
        assert "old" not in doc.read_text()

    def test_empty_below_marker_needs_no_flag(self, analyzer, tmp_path):
        doc = self._doc(tmp_path, "")
        analyzer._emit_to_doc(doc, "## Results\n\nfirst\n")
        assert "first" in doc.read_text()

    def test_harness_wipes_a_stale_sandbox(self, harness, tmp_path):
        """Ported from benchmark_cradle_mother.py: a reused data_home RESUMES
        the prior attempt's NAc, so the agent starts pre-trained. exp42 lacked
        this, which is why the re-validation needed a manual `rm -rf` the
        operator had to remember."""
        home = tmp_path / "cradle_pref_a_seed42"
        (home / "sim_reports" / "old_session").mkdir(parents=True)
        stale = home / "sim_reports" / "old_session" / "actions.jsonl"
        stale.write_text('{"tool": "poison"}\n')

        got = harness._prepare_data_home(tmp_path, "cradle_pref_a", 42)

        assert got == home
        assert got.is_dir()
        assert not stale.exists(), "a stale sandbox must be wiped, not reused"
        assert list(got.iterdir()) == [], "the fresh data_home must be empty"

    def test_prepare_data_home_is_idempotent_when_absent(self, harness, tmp_path):
        got = harness._prepare_data_home(tmp_path, "cradle_pref_b", 7)
        assert got.is_dir() and got.name == "cradle_pref_b_seed7"

    def test_workdir_is_configurable(self, harness):
        """Both configurations sharing one workdir left two sessions per
        data_home, recoverable only by reasoning about session order."""
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
            harness.main(["--help"])
        assert "--workdir" in buf.getvalue()

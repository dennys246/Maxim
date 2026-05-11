"""Unit tests for the Roy iteration-log + protocol generator.

Spec: ``src/maxim/analysis/roy_log.py``. The generator consumes a
persisted ``result.json`` (produced by R3's
:func:`maxim.simulation.roy_runner.run_roy_iteration`) and writes:

1. A reproduction-protocol runbook at
   ``docs/experiments/protocols/roy_<id>.md``.
2. An iteration-log entry appended to the appropriate plan doc.

Both artifacts are drafts pending operator review; the generator
just kills the boilerplate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from maxim.analysis.roy_log import (
    LoadedResult,
    default_protocol_path,
    detect_plan_path,
    generate_iteration_log_entry,
    generate_protocol,
    load_iteration_result,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _sample_payload(
    *,
    name: str = "roy-0-smoke",
    description: str = "methodology smoke test",
    aborted_at: str | None = None,
) -> dict[str, Any]:
    """A representative ``result.json`` payload.

    Shape matches ``RoyIterationResult._result_to_dict`` exactly so the
    generator is exercised against the real persisted contract.
    """
    return {
        "_format_version": "1.0",
        "name": name,
        "spec_path": "/repo/scenarios/roy/roy_0_smoke_iteration.yaml",
        "description": description,
        "priming": {
            "name": f"{name}-priming",
            "spec_path": "/tmp/inline_priming.yaml",
            "stages": [
                {
                    "name": "act1_warmup",
                    "source_kind": "fixture",
                    "source_value": "/repo/scenarios/cradle/warmup.yaml",
                    "turns_requested": 4,
                    "session_id": "20260510_120000_call001",
                    "session_dir": "/tmp/sim_reports/20260510_120000_call001",
                    "turns_run": 4,
                    "finish_reason": "complete",
                    "duration_s": 0.05,
                    "error": None,
                },
            ],
            "final_session_id": "20260510_120000_call001",
            "final_session_dir": "/tmp/sim_reports/20260510_120000_call001",
            "total_duration_s": 0.07,
            "aborted_at": None,
        },
        "arms": {
            "a": {
                "name": "a",
                "substrate": "from_priming",
                "system_prompt": "neutral",
                "resumed_from": "20260510_120000_call001",
                "session_id": "20260510_120001_call002",
                "session_dir": "/tmp/sim_reports/20260510_120001_call002",
                "turns_run": 5,
                "finish_reason": "complete",
                "duration_s": 0.02,
                "error": None,
            },
            "b": {
                "name": "b",
                "substrate": "blank",
                "system_prompt": "adversarial",
                "resumed_from": "",
                "session_id": "20260510_120002_call003",
                "session_dir": "/tmp/sim_reports/20260510_120002_call003",
                "turns_run": 5,
                "finish_reason": "complete",
                "duration_s": 0.02,
                "error": None,
            },
            "c": {
                "name": "c",
                "substrate": "blank",
                "system_prompt": "neutral",
                "resumed_from": "",
                "session_id": "20260510_120003_call004",
                "session_dir": "/tmp/sim_reports/20260510_120003_call004",
                "turns_run": 5,
                "finish_reason": "complete",
                "duration_s": 0.02,
                "error": None,
            },
        },
        "pairwise_diffs": {
            "a_vs_b": {
                "available": True,
                "nac": {
                    "available": True,
                    "reward_bias_l2": 1.2345,
                    "reward_bias_top_deltas": [],
                    "goal_reward_bias_l2": 0.0,
                    "goal_reward_bias_top_deltas": [],
                    "percept_valence": {"available": False, "l2": 0.0, "top_deltas": []},
                    "causal_link_count_a": 9,
                    "causal_link_count_b": 5,
                    "causal_link_count_delta": 4,
                },
                "ec": {"available": False, "reason": "missing EC snapshot"},
                "hippocampus": {
                    "available": True,
                    "episode_count_a": 14,
                    "episode_count_b": 5,
                    "episode_count_delta": 9,
                    "memory_count_a": 0,
                    "memory_count_b": 0,
                    "memory_count_delta": 0,
                    "valence": {
                        "mean_a": 0.1,
                        "mean_b": -0.1,
                        "ks_statistic": 0.31,
                        "ks_pvalue": 0.07,
                    },
                    "salience": {"mean_a": 0.5, "mean_b": 0.5, "ks_statistic": 0.0, "ks_pvalue": 1.0},
                },
                "atl": {"available": False, "reason": "missing ATL snapshot"},
            },
            "a_vs_c": {
                "available": True,
                "nac": {
                    "available": True,
                    "reward_bias_l2": 1.5,
                    "reward_bias_top_deltas": [],
                    "goal_reward_bias_l2": 0.0,
                    "goal_reward_bias_top_deltas": [],
                    "percept_valence": {"available": False, "l2": 0.0, "top_deltas": []},
                    "causal_link_count_a": 9,
                    "causal_link_count_b": 5,
                    "causal_link_count_delta": 4,
                },
                "ec": {"available": False, "reason": "missing EC snapshot"},
                "hippocampus": {
                    "available": True,
                    "episode_count_a": 14,
                    "episode_count_b": 5,
                    "episode_count_delta": 9,
                    "memory_count_a": 0,
                    "memory_count_b": 0,
                    "memory_count_delta": 0,
                    "valence": {"mean_a": 0.1, "mean_b": 0.0, "ks_statistic": 0.2, "ks_pvalue": 0.3},
                    "salience": {"mean_a": 0.5, "mean_b": 0.5, "ks_statistic": 0.0, "ks_pvalue": 1.0},
                },
                "atl": {"available": False, "reason": "missing ATL snapshot"},
            },
            "b_vs_c": {
                "available": True,
                "nac": {
                    "available": True,
                    "reward_bias_l2": 0.7,
                    "reward_bias_top_deltas": [],
                    "goal_reward_bias_l2": 0.0,
                    "goal_reward_bias_top_deltas": [],
                    "percept_valence": {"available": False, "l2": 0.0, "top_deltas": []},
                    "causal_link_count_a": 5,
                    "causal_link_count_b": 5,
                    "causal_link_count_delta": 0,
                },
                "ec": {"available": False, "reason": "missing EC snapshot"},
                "hippocampus": {
                    "available": True,
                    "episode_count_a": 5,
                    "episode_count_b": 5,
                    "episode_count_delta": 0,
                    "memory_count_a": 0,
                    "memory_count_b": 0,
                    "memory_count_delta": 0,
                    "valence": {"mean_a": -0.1, "mean_b": 0.0, "ks_statistic": 0.1, "ks_pvalue": 0.9},
                    "salience": {"mean_a": 0.5, "mean_b": 0.5, "ks_statistic": 0.0, "ks_pvalue": 1.0},
                },
                "atl": {"available": False, "reason": "missing ATL snapshot"},
            },
        },
        "total_duration_s": 0.13,
        "artifact_dir": "/home/test/.maxim/roy/roy-0-smoke",
        "aborted_at": aborted_at,
    }


def _write_result(
    artifact_root: Path,
    iteration_id: str,
    payload: dict[str, Any] | None = None,
) -> Path:
    """Persist a ``result.json`` under ``<artifact_root>/<iteration_id>/``."""
    payload = payload or _sample_payload(name=iteration_id)
    iteration_dir = artifact_root / iteration_id
    iteration_dir.mkdir(parents=True, exist_ok=True)
    (iteration_dir / "result.json").write_text(json.dumps(payload))
    return iteration_dir


def _make_plan_doc(tmp_path: Path) -> Path:
    """Write a minimal plan doc with an ``## Iteration log`` section
    so the appender has somewhere to insert.
    """
    p = tmp_path / "plan.md"
    p.write_text(
        "# Test Plan\n"
        "\n"
        "## Overview\n"
        "\n"
        "Some background paragraph.\n"
        "\n"
        "## Iteration log\n"
        "\n"
        "Empty until first run.\n"
        "\n"
        "## Open questions\n"
        "\n"
        "- Something here.\n"
    )
    return p


# ---------------------------------------------------------------------------
# 1. Result loading
# ---------------------------------------------------------------------------


class TestLoadIterationResult:
    def test_loads_persisted_payload(self, tmp_path):
        _write_result(tmp_path, "roy-0-smoke")
        loaded = load_iteration_result("roy-0-smoke", artifact_root=tmp_path)
        assert isinstance(loaded, LoadedResult)
        assert loaded.iteration_id == "roy-0-smoke"
        assert loaded.payload["name"] == "roy-0-smoke"
        assert loaded.result_path == tmp_path / "roy-0-smoke" / "result.json"

    def test_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No Roy iteration result"):
            load_iteration_result("nope", artifact_root=tmp_path)

    def test_malformed_raises(self, tmp_path):
        d = tmp_path / "bad"
        d.mkdir()
        (d / "result.json").write_text(json.dumps(["not", "a", "mapping"]))
        with pytest.raises(ValueError, match="expected mapping"):
            load_iteration_result("bad", artifact_root=tmp_path)


# ---------------------------------------------------------------------------
# 2. Plan auto-detection
# ---------------------------------------------------------------------------


class TestDetectPlanPath:
    def test_cradle_routes_to_grounded(self, tmp_path):
        # Build a fake repo root with the two plan docs.
        (tmp_path / "docs" / "plans").mkdir(parents=True)
        (tmp_path / "docs" / "plans" / "grounded_language_acquisition.md").write_text("# x\n")
        (tmp_path / "docs" / "plans" / "persona_convergence_crucible.md").write_text("# x\n")
        p = detect_plan_path("roy-cradle-warmup", repo_root=tmp_path)
        assert p is not None
        assert p.name == "grounded_language_acquisition.md"

    def test_persona_routes_to_crucible(self, tmp_path):
        (tmp_path / "docs" / "plans").mkdir(parents=True)
        p1 = detect_plan_path("roy-adversarial-1", repo_root=tmp_path)
        p2 = detect_plan_path("roy-cautious-scout", repo_root=tmp_path)
        p3 = detect_plan_path("roy-1-adversarial", repo_root=tmp_path)
        for p in (p1, p2, p3):
            assert p is not None
            assert p.name == "persona_convergence_crucible.md"

    def test_unmatched_returns_none(self, tmp_path):
        (tmp_path / "docs" / "plans").mkdir(parents=True)
        assert detect_plan_path("not-a-roy-name", repo_root=tmp_path) is None

    def test_numeric_iteration_prefix_matches(self, tmp_path):
        """`roy-0-smoke` resolves to the crucible doc (smoke test for
        the numeric prefix rule)."""
        (tmp_path / "docs" / "plans").mkdir(parents=True)
        p = detect_plan_path("roy-0-smoke", repo_root=tmp_path)
        assert p is not None
        assert p.name == "persona_convergence_crucible.md"


# ---------------------------------------------------------------------------
# 3. Protocol generator
# ---------------------------------------------------------------------------


class TestGenerateProtocol:
    def test_renders_expected_sections(self, tmp_path):
        _write_result(tmp_path, "roy-0-smoke")
        loaded = load_iteration_result("roy-0-smoke", artifact_root=tmp_path)
        out = tmp_path / "roy_0_smoke.md"
        rendered = generate_protocol(loaded, out)

        assert out.is_file()
        assert rendered == out.read_text()
        # Required sections.
        for header in (
            "# Roy-0: Smoke",
            "## Background",
            "## Prerequisites",
            "## Running the iteration",
            "## Expected output",
            "## What to do if it fails",
            "## What changed vs prior iterations",
            "## Related docs",
        ):
            assert header in rendered, f"missing protocol section: {header!r}"
        # The recorded duration + spec_path appear.
        assert "roy_0_smoke_iteration.yaml" in rendered
        assert "maxim roy run" in rendered
        # The arms table is populated.
        assert "from_priming" in rendered
        assert "adversarial" in rendered
        # Divergence threshold lines reflect the recorded numbers.
        assert "reward_bias L2" in rendered
        assert "1.5000" in rendered  # a_vs_c reward_bias_l2
        # Generated-by footer marker for re-generation detection.
        assert "generated by `maxim roy log roy-0-smoke`" in rendered

    def test_dry_run_does_not_write(self, tmp_path):
        _write_result(tmp_path, "roy-0-smoke")
        loaded = load_iteration_result("roy-0-smoke", artifact_root=tmp_path)
        target = tmp_path / "protocol.md"
        rendered = generate_protocol(loaded, target, dry_run=True)
        assert rendered
        assert not target.exists()

    def test_keep_edits_refuses_overwrite(self, tmp_path):
        _write_result(tmp_path, "roy-0-smoke")
        loaded = load_iteration_result("roy-0-smoke", artifact_root=tmp_path)
        target = tmp_path / "protocol.md"
        target.write_text("hand-edited\n")
        with pytest.raises(FileExistsError):
            generate_protocol(loaded, target, keep_edits=True)
        # File contents preserved.
        assert target.read_text() == "hand-edited\n"

    def test_aborted_iteration_still_renders(self, tmp_path):
        payload = _sample_payload(name="roy-0-smoke", aborted_at="priming")
        payload["arms"] = {}
        payload["pairwise_diffs"] = {}
        _write_result(tmp_path, "roy-0-smoke", payload=payload)
        loaded = load_iteration_result("roy-0-smoke", artifact_root=tmp_path)
        out = tmp_path / "p.md"
        rendered = generate_protocol(loaded, out)
        # Aborted iterations still produce a renderable protocol — the
        # "no diffs" note appears instead of the threshold table.
        assert "no diffs available" in rendered


# ---------------------------------------------------------------------------
# 4. Iteration log appender — idempotence + plan-doc insertion
# ---------------------------------------------------------------------------


class TestIterationLogAppend:
    def test_appends_under_iteration_log_heading(self, tmp_path):
        _write_result(tmp_path, "roy-0-smoke")
        loaded = load_iteration_result("roy-0-smoke", artifact_root=tmp_path)
        plan = _make_plan_doc(tmp_path)

        block = generate_iteration_log_entry(loaded, plan)
        assert "### Roy-0: Smoke" in block
        assert "<!-- roy-iteration:roy-0-smoke -->" in block
        assert "<!-- /roy-iteration:roy-0-smoke -->" in block

        content = plan.read_text()
        # Block landed inside the Iteration log section, BEFORE Open questions.
        iter_idx = content.index("## Iteration log")
        block_idx = content.index("<!-- roy-iteration:roy-0-smoke -->")
        open_q_idx = content.index("## Open questions")
        assert iter_idx < block_idx < open_q_idx

    def test_idempotent_replace(self, tmp_path):
        """Running twice with the same iteration_id replaces the block
        in-place rather than appending a duplicate."""
        _write_result(tmp_path, "roy-0-smoke")
        loaded = load_iteration_result("roy-0-smoke", artifact_root=tmp_path)
        plan = _make_plan_doc(tmp_path)

        generate_iteration_log_entry(loaded, plan)
        after_first = plan.read_text()
        first_marker_count = after_first.count("<!-- roy-iteration:roy-0-smoke -->")
        first_close_count = after_first.count("<!-- /roy-iteration:roy-0-smoke -->")
        assert first_marker_count == 1
        assert first_close_count == 1

        # Mutate the payload subtly so the new block differs from the prior.
        payload = json.loads((tmp_path / "roy-0-smoke" / "result.json").read_text())
        payload["pairwise_diffs"]["a_vs_b"]["nac"]["reward_bias_l2"] = 99.9999
        (tmp_path / "roy-0-smoke" / "result.json").write_text(json.dumps(payload))
        loaded2 = load_iteration_result("roy-0-smoke", artifact_root=tmp_path)
        generate_iteration_log_entry(loaded2, plan)

        after_second = plan.read_text()
        # Still exactly one marker pair.
        assert after_second.count("<!-- roy-iteration:roy-0-smoke -->") == 1
        assert after_second.count("<!-- /roy-iteration:roy-0-smoke -->") == 1
        # New content reflected.
        assert "99.9999" in after_second

    def test_multiple_iterations_compose(self, tmp_path):
        """Two different iterations append two separate blocks."""
        _write_result(tmp_path, "roy-0-smoke")
        _write_result(tmp_path, "roy-1-adversarial", payload=_sample_payload(name="roy-1-adversarial"))
        plan = _make_plan_doc(tmp_path)

        loaded_0 = load_iteration_result("roy-0-smoke", artifact_root=tmp_path)
        loaded_1 = load_iteration_result("roy-1-adversarial", artifact_root=tmp_path)
        generate_iteration_log_entry(loaded_0, plan)
        generate_iteration_log_entry(loaded_1, plan)

        content = plan.read_text()
        assert "<!-- roy-iteration:roy-0-smoke -->" in content
        assert "<!-- roy-iteration:roy-1-adversarial -->" in content
        # Both blocks live inside the iteration log section.
        iter_idx = content.index("## Iteration log")
        open_q_idx = content.index("## Open questions")
        for marker in (
            "<!-- roy-iteration:roy-0-smoke -->",
            "<!-- roy-iteration:roy-1-adversarial -->",
        ):
            m_idx = content.index(marker)
            assert iter_idx < m_idx < open_q_idx, f"{marker} outside Iteration log section"

    def test_dry_run_does_not_touch_disk(self, tmp_path):
        _write_result(tmp_path, "roy-0-smoke")
        loaded = load_iteration_result("roy-0-smoke", artifact_root=tmp_path)
        plan = _make_plan_doc(tmp_path)
        before = plan.read_text()

        block = generate_iteration_log_entry(loaded, plan, dry_run=True)
        # File unchanged.
        assert plan.read_text() == before
        # Renderer still returned the wrapped block so the CLI can print it.
        assert "<!-- roy-iteration:roy-0-smoke -->" in block
        assert "### Roy-0: Smoke" in block

    def test_missing_section_raises(self, tmp_path):
        _write_result(tmp_path, "roy-0-smoke")
        loaded = load_iteration_result("roy-0-smoke", artifact_root=tmp_path)
        plan = tmp_path / "no_iter.md"
        plan.write_text("# Plan with no log section\n\nNothing here.\n")
        with pytest.raises(ValueError, match="missing a '## Iteration log' section"):
            generate_iteration_log_entry(loaded, plan)

    def test_missing_plan_raises(self, tmp_path):
        _write_result(tmp_path, "roy-0-smoke")
        loaded = load_iteration_result("roy-0-smoke", artifact_root=tmp_path)
        with pytest.raises(FileNotFoundError):
            generate_iteration_log_entry(loaded, tmp_path / "nope.md")


# ---------------------------------------------------------------------------
# 5. CLI surface — picks up auto-detected paths + honors --dry-run
# ---------------------------------------------------------------------------


class TestRoyLogCli:
    def test_cli_writes_protocol_and_log_entry(self, tmp_path, monkeypatch, capsys):
        # Lay out a fake repo with the persona plan doc so detect_plan_path
        # has something to resolve to.
        plans = tmp_path / "docs" / "plans"
        plans.mkdir(parents=True)
        crucible = plans / "persona_convergence_crucible.md"
        crucible.write_text(
            "# Persona Convergence Crucible\n\n"
            "## Methodology\n\nSome content.\n\n"
            "## Iteration log\n\nEmpty until Roy-1 runs.\n\n"
            "## Open questions\n\n- foo\n"
        )
        protocols = tmp_path / "docs" / "experiments" / "protocols"
        protocols.mkdir(parents=True)

        artifact_root = tmp_path / "artifacts"
        artifact_root.mkdir()
        _write_result(artifact_root, "roy-0-smoke")

        # Force the protocol output path so we don't have to fake the repo
        # root detection (which walks up from the installed module path).
        protocol_path = protocols / "roy_roy-0-smoke.md"

        from maxim.roy.cli import run_roy_subcommand

        rc = run_roy_subcommand(
            [
                "log",
                "roy-0-smoke",
                "--artifact-root",
                str(artifact_root),
                "--plan",
                str(crucible),
                "--protocol",
                str(protocol_path),
            ]
        )
        assert rc == 0
        assert protocol_path.is_file()
        assert "<!-- roy-iteration:roy-0-smoke -->" in crucible.read_text()
        captured = capsys.readouterr()
        assert "wrote protocol:" in captured.out
        assert "appended iteration-log entry" in captured.out

    def test_cli_dry_run_does_not_write(self, tmp_path, capsys):
        plans = tmp_path / "docs" / "plans"
        plans.mkdir(parents=True)
        crucible = plans / "persona_convergence_crucible.md"
        crucible.write_text("# x\n\n## Iteration log\n\nEmpty.\n\n## Tail\n")
        protocol_path = tmp_path / "p.md"

        artifact_root = tmp_path / "artifacts"
        artifact_root.mkdir()
        _write_result(artifact_root, "roy-0-smoke")

        before_crucible = crucible.read_text()
        before_protocol_exists = protocol_path.exists()

        from maxim.roy.cli import run_roy_subcommand

        rc = run_roy_subcommand(
            [
                "log",
                "roy-0-smoke",
                "--artifact-root",
                str(artifact_root),
                "--plan",
                str(crucible),
                "--protocol",
                str(protocol_path),
                "--dry-run",
            ]
        )
        assert rc == 0
        # Both targets untouched.
        assert crucible.read_text() == before_crucible
        assert protocol_path.exists() == before_protocol_exists
        captured = capsys.readouterr()
        # Dry-run prints both artifacts inline.
        assert "--- protocol" in captured.out
        assert "--- iteration log entry" in captured.out
        assert "### Roy-0: Smoke" in captured.out

    def test_cli_missing_iteration_returns_error(self, tmp_path, capsys):
        from maxim.roy.cli import run_roy_subcommand

        rc = run_roy_subcommand(
            [
                "log",
                "ghost",
                "--artifact-root",
                str(tmp_path),
            ]
        )
        assert rc == 2
        captured = capsys.readouterr()
        assert "No Roy iteration result" in captured.err

    def test_cli_unauto_detectable_plan_requires_flag(self, tmp_path, capsys):
        artifact_root = tmp_path / "artifacts"
        artifact_root.mkdir()
        # Iteration name that doesn't match any prefix rule.
        _write_result(artifact_root, "wat-experiment", payload=_sample_payload(name="wat-experiment"))

        from maxim.roy.cli import run_roy_subcommand

        rc = run_roy_subcommand(
            [
                "log",
                "wat-experiment",
                "--artifact-root",
                str(artifact_root),
                "--no-protocol",
            ]
        )
        assert rc == 2
        captured = capsys.readouterr()
        assert "cannot auto-detect plan doc" in captured.err


# ---------------------------------------------------------------------------
# 6. Default-path resolver
# ---------------------------------------------------------------------------


class TestDefaultProtocolPath:
    def test_resolves_under_repo_docs(self, tmp_path):
        """`roy-0-smoke` → `roy_0_smoke.md` per the
        `roy_<N>_<persona>.md` convention in
        persona_convergence_crucible.md."""
        (tmp_path / "docs" / "experiments" / "protocols").mkdir(parents=True)
        (tmp_path / "docs" / "plans").mkdir(parents=True)
        out = default_protocol_path("roy-0-smoke", repo_root=tmp_path)
        assert out is not None
        assert out == tmp_path / "docs" / "experiments" / "protocols" / "roy_0_smoke.md"

    def test_persona_slug_uses_underscore_separators(self, tmp_path):
        """`roy-1-adversarial` → `roy_1_adversarial.md`,
        `roy-cradle-warmup` → `roy_cradle_warmup.md`."""
        (tmp_path / "docs" / "experiments" / "protocols").mkdir(parents=True)
        (tmp_path / "docs" / "plans").mkdir(parents=True)
        assert default_protocol_path("roy-1-adversarial", repo_root=tmp_path).name == "roy_1_adversarial.md"
        assert default_protocol_path("roy-cradle-warmup", repo_root=tmp_path).name == "roy_cradle_warmup.md"

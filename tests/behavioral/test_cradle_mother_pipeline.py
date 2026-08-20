"""CI-safe pipeline tests for the cradle-mother fade-curve harness + analyzer.

No subprocess / no LLM: exercises the mother-telemetry parser, the per-act fade
extraction, the mock pipeline, and the analyzer verdict logic (GRADUATE vs
not-graduated) — the parts that decide whether the experiment reads correctly.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HARNESS = _REPO_ROOT / "scripts" / "benchmark_cradle_mother.py"
_ANALYZER = _REPO_ROOT / "scripts" / "analyze_cradle_mother.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


harness = _load(_HARNESS, "cm_harness")


# ── mother-telemetry parsing ────────────────────────────────────────────────


def test_parse_mother_record():
    rec = harness._parse_mother_record(
        "act=act1_fully_guided fed=True guided=True az_prior=0.0 az_stimulus=-0.7 az_guided=-0.0"
    )
    assert rec["act"] == "act1_fully_guided"
    assert rec["fed"] is True and rec["guided"] is True
    assert rec["az_stimulus"] == -0.7


def test_parse_mother_record_none_value_and_non_record():
    rec = harness._parse_mother_record("act=act3_autonomous fed=False guided=False az_prior=None az_stimulus=0.4")
    assert rec["fed"] is False and rec["az_prior"] is None
    assert harness._parse_mother_record("some other log line") is None


def test_extract_fade_computes_directedness(tmp_path):
    log = tmp_path / "log.jsonl"
    lines = [
        # act1: two turns toward the sound (progress > 0), one away (progress < 0)
        {"message": "act=act1_early fed=True credited=True progress=0.3 az_prior=-0.4"},
        {"message": "act=act1_early fed=True credited=True progress=0.2 az_prior=0.5"},
        {"message": "act=act1_early fed=False credited=False progress=-0.3 az_prior=-0.8"},
        # a turn-1 record with progress=None (no prev_stimulus) is not counted
        {"message": "act=act1_early fed=True credited=False progress=None az_prior=0.0"},
        {"message": "unrelated line, no act"},
    ]
    log.write_text("\n".join(json.dumps(x) for x in lines))
    fade = harness._extract_fade(log)
    # 3 measured turns (progress numeric), 2 moved toward → directedness 2/3
    assert fade["act1_early"]["measured"] == 3
    assert abs(fade["act1_early"]["directedness"] - 2 / 3) < 1e-9
    assert abs(fade["act1_early"]["credited_rate"] - 2 / 4) < 1e-9  # 2 of 4 records credited


def test_extract_fade_missing_log_is_empty(tmp_path):
    assert harness._extract_fade(tmp_path / "nope.jsonl") == {}


def test_arm_env_mapping():
    # Both arms share the operant-only + explore conditions; no_feed adds the
    # disable-care toggle (the control).
    assert harness._ARM_ENV["taught"]["MAXIM_OPERANT_ONLY_CREDIT"] == "1"
    assert harness._ARM_ENV["taught"]["MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT"] == "1.5"
    assert harness._ARM_ENV["no_feed"]["MAXIM_CRADLE_MOTHER_DISABLE_CARE"] == "1"
    assert "MAXIM_CRADLE_MOTHER_DISABLE_CARE" not in harness._ARM_ENV["taught"]


def test_mock_fade_shapes_the_arms():
    # taught directedness rises to learned; no_feed stays at chance.
    t = harness._mock_fade("taught", 42)
    c = harness._mock_fade("no_feed", 42)
    assert t["act4_autonomous"]["directedness"] > 0.8
    assert abs(c["act4_autonomous"]["directedness"] - 0.5) < 0.1


# ── analyzer verdict via the mock pipeline ──────────────────────────────────


def _run_analyzer(results_path: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(_ANALYZER), "--in", str(results_path), "--trials", "3"],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout


def test_mock_pipeline_taught_graduates(tmp_path):
    out = tmp_path / "cm.jsonl"
    rc = subprocess.run(
        [
            sys.executable,
            str(_HARNESS),
            "--mock",
            "--trials",
            "3",
            "--out",
            str(out),
            "--workdir",
            str(tmp_path / "work"),
        ],
        capture_output=True,
        text=True,
    ).returncode
    assert rc == 0 and out.exists()
    code, report = _run_analyzer(out)
    assert code == 0, report
    assert "GRADUATE" in report
    assert "LEARNED" in report and "PASS" in report


def test_analyzer_flags_no_learning(tmp_path):
    # taught with a FLAT directedness curve (never learned) must not graduate.
    out = tmp_path / "flat.jsonl"
    rows = []
    flat = {
        "act1_early": {"directedness": 0.5},
        "act2_warming": {"directedness": 0.5},
        "act3_consolidating": {"directedness": 0.5},
        "act4_autonomous": {"directedness": 0.5},
    }
    for arm in ("taught", "no_feed"):
        for seed in (42, 43, 44):
            rows.append({"experiment": "cradle_mother", "arm": arm, "seed": seed, "fade": flat})
    out.write_text("\n".join(json.dumps(r) for r in rows))
    code, report = _run_analyzer(out)
    assert code == 1, report  # NOT GRADUATED
    assert "NOT GRADUATED" in report

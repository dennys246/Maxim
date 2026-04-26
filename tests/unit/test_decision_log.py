"""Tests for the post-mortem lane decision log (P9 of peer_leader_flexibility).

Coverage:

* `build_record` shape — caps + env + tier_decisions populated, secrets
  redacted, URLs reduced to host-only.
* `record_to_json_line` round-trip — append + read_last_record returns
  the original record.
* `append_record` rotation — at 1001 entries the file holds the most
  recent 1000.
* `format_record_human` — multi-line output mentions key fields.
* `maxim doctor --last-decision` CLI — happy path + empty log path.
"""

from __future__ import annotations

import json

import pytest

from maxim.models.language.maxim_peer_backend import _MaximPeerBackend
from maxim.runtime import decision_log
from maxim.runtime.capabilities import RuntimeCapabilities
from maxim.runtime.worker_pool import LaneConfig


@pytest.fixture
def fake_data_home(tmp_path, monkeypatch):
    home = tmp_path / "maxim_home"
    (home / "util").mkdir(parents=True)
    monkeypatch.setenv("MAXIM_DATA_HOME", str(home))
    from maxim.utils import paths

    paths._reset_caches()
    return home


@pytest.fixture(autouse=True)
def _scrub_routing_env(monkeypatch):
    """Drop any inherited routing env vars so the snapshot is deterministic."""
    for k in (
        "MAXIM_LLM_PROFILE",
        "MAXIM_LLM_N_CTX",
        "MAXIM_AUTO_DOWNLOAD_MODELS",
        "MAXIM_AUTO_SPAWN_LLM_SERVER",
        "MAXIM_SKIP_REMOTE_PROBE",
        "MAXIM_LANE_LARGE_REMOTE_URL",
        "MAXIM_LANE_LARGE_REMOTE_MODEL",
        "MAXIM_LANE_LARGE_REMOTE_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)


# ─── build_record / serialization ───────────────────────────────────────────


class TestBuildRecord:
    def test_record_carries_caps_and_lanes(self):
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="cuda", vram_gb=16.0, ram_gb=64.0)
        lane = LaneConfig(
            name="large",
            max_workers=1,
            model_profile="qwen2.5-14b-instruct",
            n_ctx=4096,
        )
        record = decision_log.build_record(
            caps=caps,
            final_lane_configs={"large": lane},
            decision_sources={"large": "tier_table"},
            peer_config_loaded=False,
        )
        assert record.caps["vram_gb"] == 16.0
        assert record.tier_decisions["large"]["profile"] == "qwen2.5-14b-instruct"
        assert record.tier_decisions["large"]["n_ctx"] == 4096
        assert record.tier_decisions["large"]["source"] == "tier_table"

    def test_env_snapshot_redacts_api_keys(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_API_KEY", "secret-token-do-not-leak")
        record = decision_log.build_record(caps=None, final_lane_configs={})
        assert record.env.get("MAXIM_LANE_LARGE_REMOTE_API_KEY") == "<redacted>"
        line = decision_log.record_to_json_line(record)
        assert "secret-token-do-not-leak" not in line

    def test_env_snapshot_reduces_url_to_host(self, monkeypatch):
        monkeypatch.setenv(
            "MAXIM_LANE_LARGE_REMOTE_URL",
            "https://maxim.example.com/v1/secret/path?token=abc",
        )
        record = decision_log.build_record(caps=None, final_lane_configs={})
        assert record.env["MAXIM_LANE_LARGE_REMOTE_URL"] == "maxim.example.com"
        line = decision_log.record_to_json_line(record)
        assert "secret/path" not in line
        assert "token=abc" not in line

    def test_lane_remote_host_is_redacted(self):
        lane = LaneConfig(
            name="large",
            max_workers=1,
            remote_url="https://maxim.example.com/v1/x?key=foo",
            remote_model="claude",
        )
        record = decision_log.build_record(caps=None, final_lane_configs={"large": lane})
        info = record.tier_decisions["large"]
        assert info["remote_host"] == "maxim.example.com"
        assert "/v1" not in info["remote_host"]

    def test_record_to_json_line_is_single_line_and_parses(self):
        caps = RuntimeCapabilities(has_gpu=False, gpu_type=None, vram_gb=0, ram_gb=8)
        record = decision_log.build_record(caps=caps, final_lane_configs={})
        line = decision_log.record_to_json_line(record)
        assert line.endswith("\n")
        assert line.count("\n") == 1
        parsed = json.loads(line)
        assert parsed["caps"]["ram_gb"] == 8


# ─── append_record / read_last_record ───────────────────────────────────────


class TestAppendAndRead:
    def test_append_then_read_round_trip(self, fake_data_home):
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="mps", vram_gb=24, ram_gb=24)
        lane = LaneConfig(name="large", max_workers=1, model_profile="qwen2.5-14b-instruct")
        record = decision_log.build_record(caps=caps, final_lane_configs={"large": lane})
        decision_log.append_record(record)
        loaded = decision_log.read_last_record()
        assert loaded is not None
        assert loaded.tier_decisions["large"]["profile"] == "qwen2.5-14b-instruct"
        assert loaded.caps["gpu_type"] == "mps"

    def test_read_returns_none_when_log_missing(self, fake_data_home):
        assert decision_log.read_last_record() is None

    def test_read_skips_corrupt_trailing_lines(self, fake_data_home):
        # Write one good record then a garbage line.
        record = decision_log.build_record(caps=None, final_lane_configs={})
        decision_log.append_record(record)
        path = fake_data_home / "util" / "lane_decisions.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write("not valid json\n")
        loaded = decision_log.read_last_record()
        # Falls back to the last *valid* line.
        assert loaded is not None

    def test_read_returns_most_recent(self, fake_data_home):
        for profile in ("smollm-1.7b-instruct", "mistral-7b-instruct-v0.2", "qwen2.5-14b-instruct"):
            lane = LaneConfig(name="large", max_workers=1, model_profile=profile)
            decision_log.append_record(decision_log.build_record(caps=None, final_lane_configs={"large": lane}))
        last = decision_log.read_last_record()
        assert last.tier_decisions["large"]["profile"] == "qwen2.5-14b-instruct"


# ─── rotation ───────────────────────────────────────────────────────────────


class TestRotation:
    def test_log_rotates_at_max_records(self, fake_data_home, monkeypatch):
        # Shrink the cap so the test stays fast.
        monkeypatch.setattr(decision_log, "MAX_RECORDS", 5)
        for i in range(8):
            lane = LaneConfig(name="large", max_workers=1, model_profile=f"profile-{i}")
            decision_log.append_record(decision_log.build_record(caps=None, final_lane_configs={"large": lane}))
        path = fake_data_home / "util" / "lane_decisions.jsonl"
        with path.open() as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == 5
        # Most recent record should be profile-7 (zero-indexed iter of 8)
        last = decision_log.read_last_record()
        assert last.tier_decisions["large"]["profile"] == "profile-7"


# ─── format_record_human ────────────────────────────────────────────────────


class TestHumanFormat:
    def test_format_mentions_caps_and_tiers(self):
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="cuda", vram_gb=16.0, ram_gb=64.0)
        lane = LaneConfig(name="large", max_workers=1, model_profile="qwen2.5-14b-instruct", n_ctx=4096)
        record = decision_log.build_record(caps=caps, final_lane_configs={"large": lane})
        rendered = decision_log.format_record_human(record)
        assert "qwen2.5-14b-instruct" in rendered
        assert "n_ctx=4096" in rendered
        assert "16" in rendered  # vram_gb appears in caps line


# ─── maxim doctor --last-decision ───────────────────────────────────────────


class TestDoctorLastDecisionFlag:
    def test_no_log_returns_helpful_message(self, fake_data_home, capsys):
        from maxim.doctor.cli import run_doctor_subcommand

        rc = run_doctor_subcommand(["--last-decision"])
        assert rc == 1
        out = capsys.readouterr().out
        assert "No lane decisions logged" in out

    def test_existing_log_prints_record(self, fake_data_home, capsys):
        caps = RuntimeCapabilities(has_gpu=True, gpu_type="cuda", vram_gb=24, ram_gb=64)
        lane = LaneConfig(name="large", max_workers=1, model_profile="qwen2.5-14b-instruct", n_ctx=8192)
        decision_log.append_record(decision_log.build_record(caps=caps, final_lane_configs={"large": lane}))

        from maxim.doctor.cli import run_doctor_subcommand

        rc = run_doctor_subcommand(["--last-decision"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "qwen2.5-14b-instruct" in out
        assert "n_ctx=8192" in out


# ─── build_primary_router actually appends a record ─────────────────────────


class TestBuildPrimaryRouterAppends:
    def test_append_called_during_router_build(self, fake_data_home, monkeypatch):
        """End-to-end smoke: build_primary_router should drop a record
        on disk for the current invocation."""
        from unittest.mock import patch

        from maxim.models.language.config import LLMConfig
        from maxim.runtime.capabilities import RuntimeCapabilities
        from maxim.runtime.lane_backends import build_primary_router
        from maxim.runtime.llm_server import ProbeResult

        caps = RuntimeCapabilities(has_gpu=False, gpu_type=None, vram_gb=0, ram_gb=16)
        with (
            patch("maxim.models.language.config.load_llm_config", return_value=LLMConfig()),
            patch("maxim.models.language.router.LLMRouter"),
            patch("maxim.runtime.lane_backends._read_persisted_model", return_value=None),
            patch("maxim.runtime.lane_backends._maybe_pin_pre_upgrade_profile"),
            patch(
                "maxim.runtime.lane_backends._ensure_lane_profiles_available",
                side_effect=lambda lc, _caps, _log: lc,
            ),
            patch.object(
                _MaximPeerBackend,
                "health_check",
                return_value=ProbeResult("http://stub", "ok", "HTTP 200", 1.0),
            ),
        ):
            build_primary_router(capabilities=caps)

        record = decision_log.read_last_record()
        assert record is not None
        # Capabilities flowed through
        assert record.caps["ram_gb"] == 16

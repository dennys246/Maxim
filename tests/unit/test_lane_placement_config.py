"""Phase 3b of lane_capability_placement_split.md — the config.json placement
declarative surface + the reusable coherence validator.

Covers the config-layer schema only (parse / validate / round-trip). The
runtime producer (config.json placement → LaneConfig.placement), CLI
re-expression, and multi-element compile are Phase 3c.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from maxim.runtime.config_loader import (
    ConfigurationError,
    LaneTierPlacement,
    load_config,
)
from maxim.runtime.config_writer import write_config
from maxim.runtime.worker_pool import Origin, ProviderPlacement, validate_placement_coherence


def _write_load(tmp_path: Path, lanes: dict) -> object:
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"_format_version": "1.0", "lanes": lanes}))
    return load_config(p)


# ─── validate_placement_coherence (worker_pool) ─────────────────────────────


class TestValidatePlacementCoherence:
    def test_local_requires_model(self):
        with pytest.raises(ValueError, match="local.*requires a 'model'"):
            validate_placement_coherence(ProviderPlacement(origin=Origin.LOCAL))

    def test_peer_requires_url(self):
        with pytest.raises(ValueError, match="peer.*requires a 'url'"):
            validate_placement_coherence(ProviderPlacement(origin=Origin.PEER, model="m"))

    def test_cloud_requires_url_or_model(self):
        with pytest.raises(ValueError, match="cloud.*requires a 'url' or a 'model'"):
            validate_placement_coherence(ProviderPlacement(origin=Origin.CLOUD))

    def test_valid_pass(self):
        # None of these raise.
        validate_placement_coherence(ProviderPlacement(origin=Origin.LOCAL, model="mistral-7b"))
        validate_placement_coherence(ProviderPlacement(origin=Origin.PEER, url="http://10.0.0.9:8100/v1"))
        validate_placement_coherence(ProviderPlacement(origin=Origin.CLOUD, model="claude-sonnet"))  # model only
        validate_placement_coherence(ProviderPlacement(origin=Origin.CLOUD, url="https://x/v1"))  # url only

    def test_where_label_in_message(self):
        with pytest.raises(ValueError, match="lanes.large.placement"):
            validate_placement_coherence(
                ProviderPlacement(origin=Origin.LOCAL), where="config.json: lanes.large.placement[0]"
            )


# ─── LaneTierPlacement dataclass ────────────────────────────────────────────


class TestLaneTierPlacementShape:
    def test_collision_guard(self):
        with pytest.raises(ConfigurationError, match="collide with declared fields"):
            LaneTierPlacement(origin="local", extra={"model": "shadow"})

    def test_defaults(self):
        p = LaneTierPlacement(origin="cloud")
        assert p.model is None and p.url is None and p.api_key_ref is None and p.timeout_s is None and p.extra == {}


# ─── config.json parsing (_parse_lane_tier_placement via load_config) ───────


class TestPlacementParsing:
    def test_absent_placement_is_empty(self, tmp_path):
        cfg = _write_load(tmp_path, {"large": {"remote_url": "http://10.0.0.1:8100/v1"}})
        assert cfg.lanes.large.placement == ()

    def test_valid_multi_entry(self, tmp_path):
        cfg = _write_load(
            tmp_path,
            {
                "large": {
                    "placement": [
                        {"origin": "local", "model": "mistral-7b"},
                        {"origin": "cloud", "model": "claude-sonnet", "api_key_ref": "~/.config/maxim/api_key"},
                    ]
                }
            },
        )
        pl = cfg.lanes.large.placement
        assert len(pl) == 2
        assert pl[0] == LaneTierPlacement(origin="local", model="mistral-7b")
        assert pl[1].origin == "cloud" and pl[1].api_key_ref == "~/.config/maxim/api_key"

    def test_bad_origin_rejected(self, tmp_path):
        with pytest.raises(ConfigurationError, match=r"origin must be one of"):
            _write_load(tmp_path, {"large": {"placement": [{"origin": "gpu"}]}})

    def test_missing_origin_rejected(self, tmp_path):
        with pytest.raises(ConfigurationError, match=r"origin must be one of"):
            _write_load(tmp_path, {"large": {"placement": [{"model": "x"}]}})

    def test_placement_not_a_list_rejected(self, tmp_path):
        with pytest.raises(ConfigurationError, match=r"placement must be a list"):
            _write_load(tmp_path, {"large": {"placement": {"origin": "local"}}})

    def test_entry_not_a_mapping_rejected(self, tmp_path):
        with pytest.raises(ConfigurationError, match=r"placement\[0\] must be a mapping"):
            _write_load(tmp_path, {"large": {"placement": ["local"]}})

    def test_non_string_field_rejected(self, tmp_path):
        with pytest.raises(ConfigurationError, match=r"\.model must be a string or null"):
            _write_load(tmp_path, {"large": {"placement": [{"origin": "local", "model": 123}]}})

    def test_non_positive_timeout_rejected(self, tmp_path):
        with pytest.raises(ConfigurationError, match=r"timeout_s must be positive"):
            _write_load(tmp_path, {"large": {"placement": [{"origin": "peer", "url": "http://x/v1", "timeout_s": 0}]}})

    def test_inline_plaintext_api_key_rejected(self, tmp_path):
        # api_key_ref must be a path / keyring URI, never an inline key (mirrors
        # the remote_api_key_ref validation).
        with pytest.raises(ConfigurationError):
            _write_load(
                tmp_path, {"large": {"placement": [{"origin": "cloud", "model": "x", "api_key_ref": "sk-abc123"}]}}
            )

    def test_unknown_keys_gathered_into_extra(self, tmp_path):
        cfg = _write_load(
            tmp_path,
            {"large": {"placement": [{"origin": "local", "model": "m", "routing_weight": 0.5}]}},
        )
        assert cfg.lanes.large.placement[0].extra == {"routing_weight": 0.5}


# ─── round-trip through config_writer ───────────────────────────────────────


class TestPlacementRoundTrip:
    def test_write_read_preserves_placement(self, tmp_path):
        cfg = _write_load(
            tmp_path,
            {
                "large": {
                    "placement": [
                        {"origin": "local", "model": "mistral-7b"},
                        {"origin": "cloud", "model": "claude-sonnet"},
                    ]
                },
                "small": {"placement": [{"origin": "peer", "url": "http://10.0.0.9:8100/v1"}]},
            },
        )
        p = tmp_path / "config.json"
        write_config(cfg, p)
        cfg2 = load_config(p)
        assert cfg2.lanes.large.placement == cfg.lanes.large.placement
        assert cfg2.lanes.small.placement == cfg.lanes.small.placement

    def test_round_trip_preserves_entry_extra(self, tmp_path):
        cfg = _write_load(
            tmp_path,
            {"large": {"placement": [{"origin": "local", "model": "m", "routing_weight": 0.5}]}},
        )
        p = tmp_path / "config.json"
        write_config(cfg, p)
        # the written JSON inlines the entry extra (no double-nesting)
        written = json.loads(p.read_text())["lanes"]["large"]["placement"][0]
        assert written["routing_weight"] == 0.5
        assert "extra" not in written
        assert load_config(p).lanes.large.placement == cfg.lanes.large.placement

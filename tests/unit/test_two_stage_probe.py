"""Tests for the two-stage probe (Plan 2 R2c).

Stage 1 = liveness (GET /v1/models). Stage 2 = readiness (tiny POST
/v1/chat/completions). Stage 2 only runs when enable_stage2=True and
stage 1 returned ok. Non-HTTP exceptions in stage 2 fall back to
stage-1 ok + WARN log.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from maxim.models.language.types import INFERENCE_BROKEN_BACKOFF_S
from maxim.runtime.llm_server import (
    _probe_stage2_readiness,
    probe_llm_server,
)
from maxim.runtime.probe_cache import load_cache, ttl_for_outcome
from maxim.utils import http as _http


@pytest.fixture
def _fresh_home(tmp_path, monkeypatch):
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
    from maxim.utils.paths import _reset_caches

    _reset_caches()
    yield tmp_path
    _reset_caches()


# ── TTL table ─────────────────────────────────────────────────────────────


def test_ttl_inference_broken_matches_backoff_constant():
    assert ttl_for_outcome("inference_broken") == INFERENCE_BROKEN_BACKOFF_S
    assert ttl_for_outcome("inference_broken") == 15.0


def test_ttl_ok_default_60():
    assert ttl_for_outcome("ok") == 60.0


def test_ttl_auth_rejected_60():
    assert ttl_for_outcome("auth_rejected") == 60.0


def test_ttl_unknown_outcome_default_60():
    assert ttl_for_outcome("martian-weather") == 60.0


# ── Stage-2 probe outcomes ────────────────────────────────────────────────


class _FakeResp:
    def __init__(self, status: int):
        self.status = status


def test_stage2_ok_returns_ok():
    with patch.object(_http, "fetch_url", return_value=_FakeResp(200)):
        r = _probe_stage2_readiness("http://h:8080", None, "mistral-7b")
    assert r.outcome == "ok"


def test_stage2_http_500_returns_inference_broken():
    err = _http.HTTPServerError("peer", status=500, fix_hint="upstream crashed")
    with patch.object(_http, "fetch_url", side_effect=err):
        r = _probe_stage2_readiness("http://h:8080", None, "mistral-7b")
    assert r.outcome == "inference_broken"
    assert "stage2" in r.detail


def test_stage2_timeout_returns_inference_broken():
    err = _http.HTTPTimeout("peer", fix_hint="timed out")
    with patch.object(_http, "fetch_url", side_effect=err):
        r = _probe_stage2_readiness("http://h:8080", None, "mistral-7b")
    assert r.outcome == "inference_broken"


def test_stage2_non_http_exception_falls_back_to_ok():
    # Library bug or JSON parse error — must not make system look dead.
    with patch.object(_http, "fetch_url", side_effect=ValueError("parse bomb")):
        r = _probe_stage2_readiness("http://h:8080", None, "mistral-7b")
    assert r.outcome == "ok"
    assert "fallback" in r.detail


# ── End-to-end probe_llm_server with enable_stage2 ────────────────────────


def test_probe_llm_server_stage2_disabled_preserves_stage1_ok():
    with patch.object(_http, "fetch_url", return_value=_FakeResp(200)):
        r = probe_llm_server("http://h:8080", enable_stage2=False)
    assert r.outcome == "ok"


def test_probe_llm_server_stage2_inference_broken():
    # Stage 1 = 200, stage 2 = 500
    call_count = [0]

    def _responses(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _FakeResp(200)
        raise _http.HTTPServerError("peer", status=500, fix_hint="broken")

    with patch.object(_http, "fetch_url", side_effect=_responses):
        r = probe_llm_server("http://h:8080", enable_stage2=True, model_name="mistral-7b")
    assert r.outcome == "inference_broken"


def test_probe_llm_server_stage2_ok_when_both_succeed():
    with patch.object(_http, "fetch_url", return_value=_FakeResp(200)):
        r = probe_llm_server("http://h:8080", enable_stage2=True, model_name="mistral-7b")
    assert r.outcome == "ok"


# ── Corrupted cache handling ──────────────────────────────────────────────


def test_load_cache_corrupt_returns_empty_and_warns(_fresh_home, caplog):
    import logging

    cache_path = _fresh_home / "util" / "last_probe_status.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{not valid json")

    with caplog.at_level(logging.WARNING, logger="maxim.runtime.probe_cache"):
        result = load_cache()
    assert result == {}
    assert any("probe_cache load failed" in rec.message for rec in caplog.records)


def test_load_cache_missing_file_returns_empty_no_warn(_fresh_home, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="maxim.runtime.probe_cache"):
        result = load_cache()
    assert result == {}
    assert not any("failed" in rec.message for rec in caplog.records)

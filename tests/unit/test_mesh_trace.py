"""Tests for Stage A observability (src/maxim/models/language/mesh_trace.py)."""

from __future__ import annotations

import json
import logging

import pytest

from maxim.models.language.mesh_trace import (
    REQUEST_ID_HEADER,
    TRACE_LOGGER_NAME,
    TraceRecord,
    any_trace_enabled,
    emit_trace,
    generate_request_id,
    lane_trace_enabled,
    peer_log_enabled,
    print_startup_warning_if_enabled,
    start_call,
)


class TestRequestIdGeneration:
    def test_returns_hex_string(self):
        rid = generate_request_id()
        assert isinstance(rid, str)
        assert len(rid) == 32  # UUID4 hex
        int(rid, 16)  # valid hex

    def test_each_is_unique(self):
        ids = {generate_request_id() for _ in range(100)}
        assert len(ids) == 100

    def test_header_name_constant(self):
        assert REQUEST_ID_HEADER == "X-Maxim-Request-Id"


class TestEnvFlags:
    def test_lane_trace_default_off(self, monkeypatch):
        monkeypatch.delenv("MAXIM_LANE_TRACE", raising=False)
        assert lane_trace_enabled() is False

    def test_lane_trace_on_when_1(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LANE_TRACE", "1")
        assert lane_trace_enabled() is True

    @pytest.mark.parametrize("value", ["true", "yes", "on", "Y", "TRUE"])
    def test_lane_trace_accepts_truthy(self, monkeypatch, value):
        monkeypatch.setenv("MAXIM_LANE_TRACE", value)
        assert lane_trace_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_lane_trace_off_for_falsy(self, monkeypatch, value):
        monkeypatch.setenv("MAXIM_LANE_TRACE", value)
        assert lane_trace_enabled() is False

    def test_peer_log_independent_of_lane_trace(self, monkeypatch):
        monkeypatch.setenv("MAXIM_PEER_LOG_REQUESTS", "1")
        monkeypatch.delenv("MAXIM_LANE_TRACE", raising=False)
        assert peer_log_enabled() is True
        assert lane_trace_enabled() is False
        assert any_trace_enabled() is True

    def test_any_trace_enabled_or_logic(self, monkeypatch):
        monkeypatch.delenv("MAXIM_LANE_TRACE", raising=False)
        monkeypatch.delenv("MAXIM_PEER_LOG_REQUESTS", raising=False)
        assert any_trace_enabled() is False
        monkeypatch.setenv("MAXIM_LANE_TRACE", "1")
        assert any_trace_enabled() is True


class TestTraceRecord:
    def _make(self, **overrides) -> TraceRecord:
        defaults = dict(
            request_id="abc123def456",
            provider="lane-infer",
            base_url="http://127.0.0.1:8100/v1",
            model="mistral-7b",
            status="ok",
            http_status=200,
            latency_ms=42.3,
            input_tokens=50,
            output_tokens=10,
        )
        defaults.update(overrides)
        return TraceRecord(**defaults)

    def test_to_dict_includes_all_fields(self):
        rec = self._make()
        d = rec.to_dict()
        assert d["request_id"] == "abc123def456"
        assert d["provider"] == "lane-infer"
        assert d["latency_ms"] == 42.3
        assert d["input_tokens"] == 50

    def test_to_dict_rounds_latency(self):
        rec = self._make(latency_ms=123.456789)
        assert rec.to_dict()["latency_ms"] == 123.5

    def test_format_compact_includes_truncated_id(self):
        rec = self._make()
        out = rec.format_compact()
        assert out.startswith("peer_infer")
        assert "req=abc123de" in out  # first 8 chars
        assert "provider=lane-infer" in out
        assert "status=ok" in out
        assert "latency=42ms" in out
        assert "tokens=50+10" in out

    def test_format_compact_includes_http_status(self):
        rec = self._make(http_status=401)
        assert "http=401" in rec.format_compact()

    def test_format_compact_includes_error(self):
        rec = self._make(status="error", http_status=500, error="timeout")
        assert "err=timeout" in rec.format_compact()

    def test_format_compact_truncates_long_error(self):
        long_err = "A" * 200
        rec = self._make(status="error", error=long_err)
        assert len(rec.format_compact()) < 300  # bounded output


class TestEmitTrace:
    def test_no_log_when_flags_off(self, monkeypatch, caplog):
        monkeypatch.delenv("MAXIM_LANE_TRACE", raising=False)
        monkeypatch.delenv("MAXIM_PEER_LOG_REQUESTS", raising=False)
        rec = TraceRecord(
            request_id="a",
            provider="p",
            base_url="u",
            model="m",
            status="ok",
            http_status=200,
            latency_ms=10.0,
        )
        with caplog.at_level(logging.DEBUG, logger=TRACE_LOGGER_NAME):
            emit_trace(rec)
        assert caplog.records == []

    def test_lane_trace_emits_compact(self, monkeypatch, caplog):
        monkeypatch.setenv("MAXIM_LANE_TRACE", "1")
        monkeypatch.delenv("MAXIM_PEER_LOG_REQUESTS", raising=False)
        rec = TraceRecord(
            request_id="abc12345678",
            provider="p",
            base_url="u",
            model="m",
            status="ok",
            http_status=200,
            latency_ms=10.0,
        )
        with caplog.at_level(logging.INFO, logger=TRACE_LOGGER_NAME):
            emit_trace(rec)
        assert len(caplog.records) == 1
        assert "peer_infer" in caplog.records[0].message
        assert "req=abc12345" in caplog.records[0].message

    def test_peer_log_emits_json(self, monkeypatch, caplog):
        monkeypatch.delenv("MAXIM_LANE_TRACE", raising=False)
        monkeypatch.setenv("MAXIM_PEER_LOG_REQUESTS", "1")
        rec = TraceRecord(
            request_id="abc",
            provider="p",
            base_url="u",
            model="m",
            status="ok",
            http_status=200,
            latency_ms=10.0,
        )
        with caplog.at_level(logging.INFO, logger=TRACE_LOGGER_NAME):
            emit_trace(rec)
        assert len(caplog.records) == 1
        parsed = json.loads(caplog.records[0].message)
        assert parsed["request_id"] == "abc"
        assert parsed["provider"] == "p"

    def test_both_flags_emits_both_lines(self, monkeypatch, caplog):
        monkeypatch.setenv("MAXIM_LANE_TRACE", "1")
        monkeypatch.setenv("MAXIM_PEER_LOG_REQUESTS", "1")
        rec = TraceRecord(
            request_id="a",
            provider="p",
            base_url="u",
            model="m",
            status="ok",
            http_status=200,
            latency_ms=1.0,
        )
        with caplog.at_level(logging.INFO, logger=TRACE_LOGGER_NAME):
            emit_trace(rec)
        assert len(caplog.records) == 2


class TestStartupWarning:
    def test_silent_when_no_flags(self, monkeypatch, capsys):
        monkeypatch.delenv("MAXIM_LANE_TRACE", raising=False)
        monkeypatch.delenv("MAXIM_PEER_LOG_REQUESTS", raising=False)
        print_startup_warning_if_enabled()
        assert capsys.readouterr().err == ""

    def test_loud_when_flag_set(self, monkeypatch, capsys):
        monkeypatch.setenv("MAXIM_LANE_TRACE", "1")
        print_startup_warning_if_enabled()
        err = capsys.readouterr().err
        assert "DEBUG FLAGS ACTIVE" in err
        assert "MAXIM_LANE_TRACE" in err
        assert "!" in err  # attention-grabbing banner

    def test_mentions_both_flags(self, monkeypatch, capsys):
        monkeypatch.setenv("MAXIM_LANE_TRACE", "1")
        monkeypatch.setenv("MAXIM_PEER_LOG_REQUESTS", "1")
        print_startup_warning_if_enabled()
        err = capsys.readouterr().err
        assert "MAXIM_LANE_TRACE" in err
        assert "MAXIM_PEER_LOG_REQUESTS" in err


class TestStartCall:
    def test_returns_id_and_timestamp(self):
        rid, ts = start_call()
        assert isinstance(rid, str)
        assert len(rid) == 32
        assert isinstance(ts, float)
        assert ts > 0

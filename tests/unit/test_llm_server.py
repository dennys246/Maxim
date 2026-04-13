"""Tests for runtime/llm_server.py — extracted from lane_backends.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from maxim.runtime.llm_server import (
    stop_active_spawner,
    register_router,
    _find_active_routers,
    read_persisted_model,
    write_persisted_model,
    llm_server_responding_at,
    profile_has_local_file,
)


class TestStopActiveSpawner:
    def test_stops_spawner(self):
        import maxim.runtime.llm_server as mod

        mock_spawner = MagicMock()
        mod._active_spawner = mock_spawner
        mod._active_model = "test-model"
        mod._llm_start_time = 1000.0

        stop_active_spawner()

        mock_spawner.stop.assert_called_once()
        assert mod._active_spawner is None
        assert mod._active_model is None
        assert mod._llm_start_time is None

    def test_noop_when_no_spawner(self):
        import maxim.runtime.llm_server as mod

        mod._active_spawner = None
        # Should not raise
        stop_active_spawner()

    def test_stop_failure_nonfatal(self):
        import maxim.runtime.llm_server as mod

        mock_spawner = MagicMock()
        mock_spawner.stop.side_effect = RuntimeError("oops")
        mod._active_spawner = mock_spawner
        # Should not raise
        stop_active_spawner()
        assert mod._active_spawner is None


class TestRouterRegistry:
    def test_register_and_find(self):
        import maxim.runtime.llm_server as mod

        # Reset
        mod._active_routers.clear()

        router = MagicMock()
        register_router(router)

        found = _find_active_routers()
        assert len(found) == 1
        assert found[0] is router

        # Cleanup
        mod._active_routers.clear()

    def test_prunes_dead_refs(self):
        import maxim.runtime.llm_server as mod

        mod._active_routers.clear()

        router1 = MagicMock()
        register_router(router1)

        # Simulate router2 being garbage collected
        router2 = MagicMock()
        register_router(router2)
        del router2

        import gc

        gc.collect()

        found = _find_active_routers()
        # router1 should still be there, router2 may be gone
        assert router1 in found

        mod._active_routers.clear()


class TestModelPersistence:
    def test_write_and_read(self, tmp_path):
        state_file = tmp_path / "util" / "active_llm_model.txt"
        with patch("maxim.runtime.llm_server._model_state_file", return_value=state_file):
            write_persisted_model("mistral-7b")
            result = read_persisted_model()
            assert result == "mistral-7b"

    def test_read_missing_returns_none(self, tmp_path):
        state_file = tmp_path / "nonexistent" / "model.txt"
        with patch("maxim.runtime.llm_server._model_state_file", return_value=state_file):
            assert read_persisted_model() is None

    def test_write_none_clears(self, tmp_path):
        state_file = tmp_path / "util" / "active_llm_model.txt"
        with patch("maxim.runtime.llm_server._model_state_file", return_value=state_file):
            write_persisted_model("test")
            write_persisted_model(None)
            result = read_persisted_model()
            assert result is None


class TestLlmServerResponding:
    def test_empty_url_returns_false(self):
        assert llm_server_responding_at("") is False

    def test_200_returns_true(self):
        # Plan 1 R1 migrated llm_server_responding_at off urllib onto
        # maxim.utils.http.fetch_url. Mock at the new layer.
        from maxim.utils import http as _http

        fake_resp = _http.Response(
            status=200,
            headers={},
            content=b"{}",
            elapsed_ms=1.0,
            endpoint=_http._EXTERNAL_ENDPOINT,
            request_id="r",
        )
        with patch("maxim.utils.http.fetch_url", return_value=fake_resp):
            assert llm_server_responding_at("http://localhost:8100/v1") is True

    def test_connection_error_returns_false(self):
        # Non-existent URL — should not raise, just return False
        assert llm_server_responding_at("http://localhost:99999/v1") is False


class TestProfileHasLocalFile:
    def test_empty_name(self):
        assert profile_has_local_file("") is False

    def test_nonexistent_profile(self):
        # Unknown profile — should return False, not raise
        assert profile_has_local_file("nonexistent-model-xyz") is False

    def test_rejects_partial_suffix(self, tmp_path):
        """A profile resolved to a .partial tmp file must return False
        even if the file exists on disk. download_file writes to
        {dest}.partial during the transfer and only os.replace()s to
        the final path after size verification. A stale .partial from a
        crashed download would otherwise be loaded by the spawner and
        fail in cryptic ways."""
        # Create a fake .partial file that DOES exist so the vanilla
        # Path.is_file() check would pass.
        partial = tmp_path / "Qwen2.5-14B-Instruct.Q4_K_M.gguf.partial"
        partial.write_bytes(b"truncated download")
        assert partial.is_file()

        # Mock load_llm_config to resolve the profile to the partial path
        class _FakeCfg:
            model_path = str(partial)

        with patch("maxim.models.language.config.load_llm_config", return_value=_FakeCfg()):
            assert profile_has_local_file("qwen2.5-14b-instruct") is False


class TestImportPaths:
    def test_import_from_lane_backends(self):
        from maxim.runtime.lane_backends import (
            stop_active_spawner,
            register_router,
            _read_persisted_model,
            _write_persisted_model,
            _llm_server_responding_at,
            _profile_has_local_file,
        )

        assert callable(stop_active_spawner)
        assert callable(register_router)
        assert callable(_read_persisted_model)
        assert callable(_write_persisted_model)
        assert callable(_llm_server_responding_at)
        assert callable(_profile_has_local_file)

    def test_globals_accessible(self):
        from maxim.runtime.lane_backends import _swap_lock

        # These are module-level state — just verify they're importable
        assert _swap_lock is not None

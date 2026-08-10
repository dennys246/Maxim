"""Tests for runtime/llm_server.py — extracted from lane_backends.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from maxim.models.language.maxim_peer_backend import _MaximPeerBackend
from maxim.runtime.llm_server import (
    stop_active_spawner,
    register_router,
    _find_active_routers,
    read_persisted_model,
    write_persisted_model,
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


class TestHealthCheckReachability:
    """Tests for _MaximPeerBackend.for_url(...).health_check().is_reachable
    (replacement for the removed llm_server_responding_at)."""

    def _is_reachable(self, url):
        return _MaximPeerBackend.for_url(url).health_check(enable_stage2=False).is_reachable

    def test_200_returns_true(self):
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
            assert self._is_reachable("http://localhost:8100/v1") is True

    def test_connection_error_returns_false(self):
        # Non-existent URL — should not raise, just return False
        assert self._is_reachable("http://localhost:99999/v1") is False


class TestCheckExistingLlmServer:
    """Singleton guard for the harness-on-leader cascade fix.

    ``check_existing_llm_server`` decides whether to reuse an already-running
    llama-cpp-server or spawn a fresh one, and fails loud on a wrong-model
    server. The probe is delegated to ``_MaximPeerBackend.health_check`` so we
    patch that (and ``_read_served_model_id``) rather than touch the network.
    """

    _MODEL_PATH = "/home/u/.maxim/models/Qwen2.5-14B-Instruct.Q4_K_M.gguf"
    _PROFILE = "qwen2.5-14b-instruct"
    _URL = "http://127.0.0.1:8100/v1"

    def _probe(self, outcome: str):
        from maxim.runtime.llm_server import ProbeResult

        return ProbeResult(url=self._URL, outcome=outcome, detail=f"probe={outcome}", latency_ms=1.0)

    def test_spawn_when_no_existing_server(self):
        """Port connection-refused → spawn proceeds."""
        from maxim.runtime.llm_server import check_existing_llm_server

        with patch.object(_MaximPeerBackend, "health_check", return_value=self._probe("connection_refused")):
            decision = check_existing_llm_server(
                url=self._URL,
                api_key="k",
                expected_model_path=self._MODEL_PATH,
                expected_profile=self._PROFILE,
            )
        assert decision == "spawn"

    def test_reuse_when_existing_server_returns_200(self):
        """200 with the right served model → reuse (skip spawn)."""
        from maxim.runtime.llm_server import check_existing_llm_server

        with (
            patch.object(_MaximPeerBackend, "health_check", return_value=self._probe("ok")),
            patch("maxim.runtime.llm_server._read_served_model_id", return_value=self._MODEL_PATH),
        ):
            decision = check_existing_llm_server(
                url=self._URL,
                api_key="k",
                expected_model_path=self._MODEL_PATH,
                expected_profile=self._PROFILE,
            )
        assert decision == "reuse"

    def test_reuse_when_served_model_matches_by_basename(self):
        """The harness symlinks ~/.maxim/models into each sandbox, so the
        served path and the configured path differ in their directory but
        share a basename. Basename match must still reuse."""
        from maxim.runtime.llm_server import check_existing_llm_server

        served = "/tmp/exp37_work/trial001_A/models/Qwen2.5-14B-Instruct.Q4_K_M.gguf"
        with (
            patch.object(_MaximPeerBackend, "health_check", return_value=self._probe("ok")),
            patch("maxim.runtime.llm_server._read_served_model_id", return_value=served),
        ):
            decision = check_existing_llm_server(
                url=self._URL,
                api_key="k",
                expected_model_path=self._MODEL_PATH,
                expected_profile=self._PROFILE,
            )
        assert decision == "reuse"

    def test_reuse_when_existing_server_returns_401(self):
        """Auth-gated but alive → treat as up and reuse (CLAUDE.md 'auth in
        health probes' lesson). The served model can't be read behind auth,
        so ``_read_served_model_id`` must NOT be consulted."""
        from maxim.runtime.llm_server import check_existing_llm_server

        with (
            patch.object(_MaximPeerBackend, "health_check", return_value=self._probe("auth_rejected")),
            patch("maxim.runtime.llm_server._read_served_model_id") as mock_read,
        ):
            decision = check_existing_llm_server(
                url=self._URL,
                api_key="k",
                expected_model_path=self._MODEL_PATH,
                expected_profile=self._PROFILE,
            )
        assert decision == "reuse"
        mock_read.assert_not_called()

    def test_fail_loud_when_existing_server_returns_wrong_model(self):
        """200 but a DIFFERENT served model → RuntimeError with an actionable
        message pointing at the CLAUDE.md lesson."""
        import pytest

        from maxim.runtime.llm_server import check_existing_llm_server

        wrong = "/home/u/.maxim/models/mistral-7b-instruct.Q4_K_M.gguf"
        with (
            patch.object(_MaximPeerBackend, "health_check", return_value=self._probe("ok")),
            patch("maxim.runtime.llm_server._read_served_model_id", return_value=wrong),
        ):
            with pytest.raises(RuntimeError, match=r"Refusing to silently use the wrong model"):
                check_existing_llm_server(
                    url=self._URL,
                    api_key="k",
                    expected_model_path=self._MODEL_PATH,
                    expected_profile=self._PROFILE,
                )

    def test_wrong_model_error_includes_concrete_fix_commands(self):
        """The error message must surface concrete operator-runnable fix
        commands (config_unification.md C2 + 2026-06-09 footgun lesson):
          (a) ``maxim config set llm.profile <profile>``  — adopt served
          (b) ``maxim --llm <configured>``                 — re-spawn to match config
          (c) ``pkill -f llama_cpp.server``                — stop everything
        Without these, the operator has to read CLAUDE.md to figure out
        what to do; the singleton-check error is the natural surface for
        the resolution paths.
        """
        import pytest

        from maxim.runtime.llm_server import check_existing_llm_server

        wrong = "/home/u/.maxim/models/Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf"
        with (
            patch.object(_MaximPeerBackend, "health_check", return_value=self._probe("ok")),
            patch("maxim.runtime.llm_server._read_served_model_id", return_value=wrong),
        ):
            with pytest.raises(RuntimeError) as excinfo:
                check_existing_llm_server(
                    url=self._URL,
                    api_key="k",
                    expected_model_path=self._MODEL_PATH,
                    expected_profile=self._PROFILE,
                )
        msg = str(excinfo.value)
        # (a) adopt-served path
        assert "maxim config set llm.profile" in msg, f"resolution (a) missing: {msg!r}"
        # (b) re-spawn-to-match-config path — must interpolate the
        # configured profile so operator can copy/paste
        assert f"maxim --llm {self._PROFILE}" in msg, f"resolution (b) missing or stale: {msg!r}"
        # (c) kill-everything path
        assert "pkill -f llama_cpp.server" in msg, f"resolution (c) missing: {msg!r}"
        # Pointer to the new CLAUDE.md lesson by name
        assert "config.json" in msg.lower() or "active_llm_model" in msg.lower(), (
            f"message should mention the two state files by name: {msg!r}"
        )

    def test_reuse_when_served_model_unverifiable(self):
        """200 but /v1/models yields no usable id → reuse anyway (avoid the
        kill+respawn cascade) rather than fail loud."""
        from maxim.runtime.llm_server import check_existing_llm_server

        with (
            patch.object(_MaximPeerBackend, "health_check", return_value=self._probe("ok")),
            patch("maxim.runtime.llm_server._read_served_model_id", return_value=None),
        ):
            decision = check_existing_llm_server(
                url=self._URL,
                api_key="k",
                expected_model_path=self._MODEL_PATH,
                expected_profile=self._PROFILE,
            )
        assert decision == "reuse"


class TestProfileHasLocalFile:
    def test_empty_name(self):
        assert profile_has_local_file("") is False

    def test_nonexistent_profile(self):
        """Unknown profile -> False, not raise — HOST-INDEPENDENTLY.

        Pre-fix this was the host-dependent CI-green class: load_llm_config
        silently falls back to the DEFAULT profile for an unknown name, so on
        any machine with the default GGUF downloaded this returned True (and
        CI, with no models, never saw it). The registry membership check makes
        the answer identical everywhere.
        """
        assert profile_has_local_file("nonexistent-model-xyz") is False

    def test_unknown_profile_false_even_when_default_model_exists(self, tmp_path):
        """The exact pre-fix false positive, simulated on any host: an unknown
        profile must NOT inherit the default profile's on-disk model."""
        default_model = tmp_path / "default-model.Q4_K_M.gguf"
        default_model.write_bytes(b"gguf")

        class _FakeCfg:
            model_path = str(default_model)

        # Even if config resolution WOULD hand back an existing default-model
        # path, the registry check must short-circuit to False first.
        with patch("maxim.models.language.config.load_llm_config", return_value=_FakeCfg()):
            assert profile_has_local_file("definitely-not-a-profile") is False

    def test_alias_resolves_to_known_profile(self, tmp_path):
        """Aliases must still pass the registry check (normalize before
        membership) — a real profile name reaching the file check."""
        from maxim.models.language.config import list_llm_profiles, normalize_llm_profile

        known = list_llm_profiles()[0]
        model = tmp_path / "known.Q4_K_M.gguf"
        model.write_bytes(b"gguf")

        class _FakeCfg:
            model_path = str(model)

        assert normalize_llm_profile(known) in list_llm_profiles()
        with patch("maxim.models.language.config.load_llm_config", return_value=_FakeCfg()):
            assert profile_has_local_file(known) is True

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
            _profile_has_local_file,
        )

        assert callable(stop_active_spawner)
        assert callable(register_router)
        assert callable(_read_persisted_model)
        assert callable(_write_persisted_model)
        assert callable(_profile_has_local_file)

    def test_globals_accessible(self):
        from maxim.runtime.lane_backends import _swap_lock

        # These are module-level state — just verify they're importable
        assert _swap_lock is not None

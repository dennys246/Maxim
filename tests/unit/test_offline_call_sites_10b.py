"""#1315 item 10b: the call sites refuse first, so an offline failure is typed and early (notes §6 layer 2).

The child-process census gate (``tests/integration/test_offline_latch.py::test_no_effect_subprocess``) proves each
census site end to end. These pin each refusal in-process, so coverage sees it and a deletion fails a named test,
and they pin the 10a review's NIT checklist that 10b carries. The conftest autouse ``_reset_offline_latch`` fixture
disengages the latch after each test.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from maxim import _offline


def _ledger_events() -> list[str]:
    return [r.event for r in _offline.ledger()]


# ── the latch's own helpers ───────────────────────────────────────────────


def test_local_files_only_kwargs_follow_the_latch() -> None:
    assert _offline.local_files_only_kwargs() == {}
    _offline.engage("test")
    assert _offline.local_files_only_kwargs() == {"local_files_only": True}


def test_mark_degraded_records_and_warns_once_per_site(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="maxim.offline"):
        _offline.mark_degraded("site.a", "no model")
        _offline.mark_degraded("site.a", "no model again")
        _offline.mark_degraded("site.b", "no weight")
    assert [d.site for d in _offline.degraded()] == ["site.a", "site.a", "site.b"]
    assert sum("degraded" in r.getMessage() for r in caplog.records) == 2


def test_a_refusal_records_its_thread() -> None:
    _offline.engage("test")
    seen: list[int] = []

    def other() -> None:
        try:
            _offline.refuse("t", "x")
        except _offline.OfflineRefusal:
            seen.append(threading.get_ident())

    worker = threading.Thread(target=other)
    worker.start()
    worker.join()
    assert _offline.ledger()[-1].thread == seen[0] != threading.get_ident()


# ── the census call sites ─────────────────────────────────────────────────


def test_docker_is_refused_before_the_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.simulation import container_runner

    monkeypatch.setattr(container_runner.subprocess, "run", MagicMock(side_effect=AssertionError("spawned")))
    _offline.engage("test")
    with pytest.raises(_offline.OfflineRefusal):
        container_runner.LocalDockerRunner().ensure_image("alpine:3.19")
    assert _ledger_events() == ["simulation.container_runner"]


def test_ddgs_is_refused_before_its_native_client() -> None:
    from maxim.tools import internet_search

    _offline.engage("test")
    with pytest.raises(_offline.OfflineRefusal):
        internet_search._search_with_ddg_package("maxim", 1)
    assert _ledger_events() == ["tools.internet_search[ddgs]"]


def test_a_missing_yolo_weight_is_refused_before_the_downloader(tmp_path: Path) -> None:
    from maxim.models.vision import ultralytics_engine

    _offline.engage("test")
    with pytest.raises(_offline.OfflineRefusal):
        ultralytics_engine._ensure_weight_file(str(tmp_path / "yolov8n-seg.pt"), "yolov8n-seg.pt")
    assert _ledger_events() == ["ultralytics.attempt_download_asset"]


def test_yolo_takes_only_an_existing_file_offline(tmp_path: Path) -> None:
    from maxim.models.vision.ultralytics_engine import _yolo_weight_path

    weight = tmp_path / "w.pt"
    weight.write_bytes(b"x")
    assert _yolo_weight_path("yolov8n.pt") == "yolov8n.pt"  # online: unchanged
    _offline.engage("test")
    assert _yolo_weight_path(str(weight)) == str(weight)
    with pytest.raises(_offline.OfflineRefusal):
        _yolo_weight_path("yolov8n.pt")  # a bare name: YOLO would download it


def test_twilio_refuses_before_recording_a_send() -> None:
    from maxim.comms.channels.twilio_channel import TwilioChannel

    channel = object.__new__(TwilioChannel)
    channel.conv_manager = MagicMock()
    channel.client = MagicMock()
    channel.voice_enabled = True
    channel.from_number = "+15550000000"
    _offline.engage("test")
    with pytest.raises(_offline.OfflineRefusal):
        channel.send("+15550001111", "hello")
    with pytest.raises(_offline.OfflineRefusal):
        channel.call("+15550001111", "hello")
    channel.conv_manager.record_outbound.assert_not_called()
    assert _ledger_events() == ["comms.twilio.send", "comms.twilio.call"]


def test_a_model_download_is_refused_before_consent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No prompt, no auto flag and no storage preflight is read: the refusal comes first."""
    from maxim.models import download
    from maxim.runtime import llm_server

    monkeypatch.setattr(llm_server, "profile_has_local_file", lambda _name: False)
    monkeypatch.setattr(download, "_auto_download_enabled", MagicMock(side_effect=AssertionError("consent read")))
    _offline.engage("test")
    with pytest.raises(_offline.OfflineRefusal):
        download.ensure_available("smollm-135m-instruct", auto=None, interactive=True)
    with pytest.raises(_offline.OfflineRefusal):
        download.ensure_tts_voice(auto=None, interactive=True, models_dir=tmp_path)
    assert _ledger_events() == ["models.download.ensure_available", "models.download.ensure_tts_voice"]


def test_the_rtm_download_is_refused_typed(tmp_path: Path) -> None:
    from maxim.models.vision import rtm_engine

    _offline.engage("test")
    with pytest.raises(_offline.OfflineRefusal):  # not the FileNotFoundError the swallowed failure used to become
        rtm_engine._resolve_model_path("rtmdet-m.onnx", str(tmp_path))


def test_whisper_without_a_cached_model_is_a_typed_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("faster_whisper")
    import faster_whisper

    import importlib

    transcribe_audio = importlib.import_module("maxim.inference.transcribe_audio")  # the package re-exports a function
    calls: list[dict] = []

    def missing(*_a: object, **kw: object) -> None:
        calls.append(dict(kw))
        raise OSError("not cached")

    monkeypatch.setattr(faster_whisper, "WhisperModel", missing)
    _offline.engage("test")
    with pytest.raises(_offline.OfflineRefusal):
        transcribe_audio.WhisperTranscriber(model_size_or_path="tiny")
    assert calls[0]["local_files_only"] is True


def test_an_encoder_that_cannot_load_offline_says_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("sentence_transformers")
    import sentence_transformers

    from maxim.similarity import encoder

    calls: list[dict] = []

    def missing(*_a: object, **kw: object) -> None:
        calls.append(dict(kw))
        raise OSError("not cached")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", missing)
    monkeypatch.setattr(encoder, "_encoder_model", None)
    _offline.engage("test")
    assert encoder._get_encoder("maxim-offline-test/missing") is None
    assert calls[0]["local_files_only"] is True
    assert [d.site for d in _offline.degraded()] == ["similarity.encoder"]


def test_the_transformers_backend_loads_local_only_without_remote_code_offline() -> None:
    from maxim.models.language.transformers_backend import _PyTorchTransformersBackend

    backend = object.__new__(_PyTorchTransformersBackend)
    backend._local_files_only = False
    backend._trust_remote_code = True
    assert (backend._offline_local_files_only(), backend._offline_trust_remote_code()) == (False, True)
    _offline.engage("test")
    assert (backend._offline_local_files_only(), backend._offline_trust_remote_code()) == (True, False)


# ── the 10a review's checklist ────────────────────────────────────────────


def test_a_robot_run_stopped_by_an_offline_refusal_exits_as_a_refusal() -> None:
    from maxim import cli

    with pytest.raises(_offline.OfflineRefusal):  # re-raised to main(), which exits 2, not logged as a crash (1)
        cli._run_stopped_on(_offline.OfflineRefusal("lane refused"), MagicMock(), verbosity=0)


def test_an_npc_line_keeps_the_offline_refusal() -> None:
    from maxim.runtime.agent_pool import AgentPool

    router = MagicMock()
    router.generate.side_effect = _offline.OfflineRefusal("lane refused")
    pool = object.__new__(AgentPool)
    pool._llm_router = router
    with pytest.raises(_offline.OfflineRefusal):  # not re-typed as ModelInferenceError
        AgentPool._generate_llm_response(pool, "npc", "a guard", "hello")


def test_doctor_keeps_a_self_caught_checks_own_rows() -> None:
    from maxim.doctor import checks

    def check_pair() -> list[checks.CheckResult]:
        try:
            _offline.refuse("socket.connect", "127.0.0.1:8100")
        except _offline.OfflineRefusal:
            pass
        return [
            checks.CheckResult(name="server binary", status="ok", message="installed"),
            checks.CheckResult(name="server", status="fail", message="nothing responding"),
        ]

    _offline.engage("test")
    rows = checks._refused_offline_row(check_pair)()
    assert [(r.name, r.status) for r in rows] == [("server binary", "ok"), ("server", "info")]
    assert rows[1].message.startswith("refused (offline): socket.connect") and "nothing responding" in rows[1].message


def test_doctor_ignores_another_threads_refusal() -> None:
    from maxim.doctor import checks

    def check_quiet() -> checks.CheckResult:
        worker = threading.Thread(target=lambda: pytest.raises(_offline.OfflineRefusal, _offline.refuse, "x", "y"))
        worker.start()
        worker.join()
        return checks.CheckResult(name="quiet", status="fail", message="its own verdict")

    _offline.engage("test")
    row = checks._refused_offline_row(check_quiet)()
    assert (row.status, row.message) == ("fail", "its own verdict")


def test_the_benchmark_does_not_score_a_refused_model() -> None:
    import ast
    import inspect

    from maxim.simulation import benchmark

    tree = ast.parse(inspect.getsource(benchmark))
    tries = [
        [h.type.id for h in node.handlers if isinstance(h.type, ast.Name)]
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
    ]
    scenario_try = [names for names in tries if "SECURITY_REFUSALS" in names]
    assert len(scenario_try) == 1, "the scenario loop's try re-raises SECURITY_REFUSALS"
    names = scenario_try[0]
    assert names.index("SECURITY_REFUSALS") < names.index("Exception")  # before the scenario-failed handler


def test_server_mode_says_refusing_to_serve_on_an_offline_refusal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from maxim import cli

    monkeypatch.setattr(cli, "_start_server_llm", MagicMock(side_effect=_offline.OfflineRefusal("lane refused")))
    _offline.engage("test")
    cli._run_server_mode(MagicMock())
    assert "Refusing to serve" in capsys.readouterr().err


@pytest.mark.parametrize("module", ["maxim.models.download"])
def test_the_download_cli_engages_first(module: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    from maxim.exceptions import ConfigurationError
    from maxim.runtime import config_loader

    monkeypatch.setattr(
        config_loader, "engage_offline_if_configured", MagicMock(side_effect=ConfigurationError("bad config"))
    )
    assert importlib.import_module(module).main() == 2


def test_selfy_engages_before_building_maxim() -> None:
    source = (Path(__file__).resolve().parents[2] / "src" / "maxim" / "embodied_runtime" / "selfy.py").read_text()
    main_block = source[source.index('if __name__ == "__main__":') :]
    assert main_block.index("engage_offline_if_configured()") < main_block.index("Maxim()")


def test_no_http_client_outside_utils_http_widened_grep() -> None:
    """The CI grep step (#1315 item 10b) is present with its pattern."""
    workflow = (Path(__file__).resolve().parents[2] / ".github" / "workflows" / "test.yml").read_text()
    assert r"httpx\.(Async)?Client\(|socket\.socket\(" in workflow

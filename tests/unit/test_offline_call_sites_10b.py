"""#1315 item 10b: the call sites refuse first, so an offline failure is typed and early (notes §6 layer 2).

The child-process census gate (``tests/integration/test_offline_latch.py::test_no_effect_subprocess``) proves each
census site end to end. These pin each refusal in-process, so coverage sees it and a deletion fails a named test,
and they pin the 10a review's NIT checklist that 10b carries. The conftest autouse ``_reset_offline_latch`` fixture
disengages the latch after each test.
"""

from __future__ import annotations

import threading
import types
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

    class LocalEntryNotFoundError(FileNotFoundError):  # huggingface_hub's cache miss, matched by name
        pass

    def missing(*_a: object, **kw: object) -> None:
        calls.append(dict(kw))
        raise LocalEntryNotFoundError("not cached")

    def broken(*_a: object, **_kw: object) -> None:
        raise RuntimeError("CUDA failed")

    monkeypatch.setattr(faster_whisper, "WhisperModel", missing)
    _offline.engage("test")
    with pytest.raises(_offline.OfflineRefusal):
        transcribe_audio.WhisperTranscriber(model_size_or_path="tiny")
    assert calls[0]["local_files_only"] is True
    monkeypatch.setattr(faster_whisper, "WhisperModel", broken)
    with pytest.raises(RuntimeError, match="CUDA"):  # a real load failure is not "not cached": unchanged
        transcribe_audio.WhisperTranscriber(model_size_or_path="tiny")
    assert _ledger_events() == ["faster_whisper.WhisperModel"]


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
    # A list check cannot say which row the latch answered: its rows stand, and one refused row is added.
    assert [(r.name, r.status) for r in rows] == [("server binary", "ok"), ("server", "fail"), ("pair", "info")]
    assert rows[2].message.startswith("refused (offline): socket.connect")


def test_doctor_marks_a_single_row_check_and_wraps_the_per_robot_probe() -> None:
    from maxim.doctor import checks

    def check_server() -> checks.CheckResult:
        try:
            _offline.refuse("socket.connect", "127.0.0.1:8100")
        except _offline.OfflineRefusal:
            pass
        return checks.CheckResult(name="server", status="fail", message="nothing responding")

    _offline.engage("test")
    row = checks._refused_offline_row(check_server)()
    assert (row.name, row.status) == ("server", "info") and "nothing responding" in row.message
    assert hasattr(checks._check_one_robot, "__wrapped__")


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


def test_a_refused_model_is_recorded_not_scored_or_ranked() -> None:
    """Owner decision V26: a refused model's runs are refused (not failed), it is reported and never ranked."""
    from maxim.simulation import benchmark as bm

    runner = object.__new__(bm.BenchmarkRunner)
    refused = runner._aggregate_runs("claude", [bm.RunResult("s", "claude", 0, error="refused: offline", refused=True)])
    assert refused.refused == "refused: offline" and not refused.passed
    errored = runner._aggregate_runs("m", [bm.RunResult("s", "m", 0, error="boom")])
    assert errored.refused is None
    report = bm.BenchmarkReport(
        timestamp="t", suite="s", models=["claude"], runs_per_model=1, results={"claude": refused}
    )
    assert "claude: REFUSED" in report.summary_table()


def test_the_benchmark_loop_records_a_refusal_and_continues() -> None:
    import ast
    import inspect

    from maxim.simulation import benchmark

    tree = ast.parse(inspect.getsource(benchmark))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ExceptHandler)
            and isinstance(node.type, ast.Name)
            and node.type.id == "SECURITY_REFUSALS"
        ):
            assert not any(isinstance(n, ast.Raise) for n in ast.walk(node)), "a refusal must not abort the benchmark"
            break
    else:
        raise AssertionError("the scenario loop has no SECURITY_REFUSALS handler")


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


# ── the 10b review's folds ────────────────────────────────────────────────


def test_refuse_if_offline() -> None:
    _offline.refuse_if_offline("x")  # online: nothing
    _offline.engage("test")
    with pytest.raises(_offline.OfflineRefusal):
        _offline.refuse_if_offline("x", "y")


def test_only_a_hub_local_miss_reads_as_not_cached() -> None:
    class LocalEntryNotFoundError(FileNotFoundError):
        pass

    miss = RuntimeError("wrapped")
    miss.__cause__ = LocalEntryNotFoundError("gone")
    assert _offline.is_hub_local_miss(miss) and not _offline.is_hub_local_miss(RuntimeError("CUDA"))
    assert _offline.load_failure_reason("m", miss) == "m is not cached locally"
    assert _offline.load_failure_reason("m", ValueError("x")) == "m failed to load (ValueError)"


def test_local_files_only_is_passed_only_where_the_loader_takes_it() -> None:
    def old_loader(name: str, device: str = "cpu") -> None: ...

    def new_loader(name: str, local_files_only: bool = False) -> None: ...

    def kw_loader(name: str, **kwargs: object) -> None: ...

    _offline.engage("test")
    assert _offline.local_files_only_kwargs(old_loader) == {}  # sentence-transformers < 2.3: HF_HUB_OFFLINE carries it
    assert _offline.local_files_only_kwargs(new_loader) == {"local_files_only": True}
    assert _offline.local_files_only_kwargs(kw_loader) == {"local_files_only": True}


def test_a_missing_pose_weight_only_turns_pose_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("ultralytics")
    import ultralytics

    from maxim.models.vision import ultralytics_engine as ue

    seg = tmp_path / "seg.pt"
    seg.write_bytes(b"x")
    monkeypatch.setattr(ultralytics, "YOLO", MagicMock())
    monkeypatch.setattr(ue, "_ensure_weight_file", lambda *_a, **_k: str(seg))
    monkeypatch.setattr(ue, "_first_existing", lambda _paths: None)
    _offline.engage("test")
    engine = ue.UltralyticsEngine(load_pose=False)  # builds: pose is optional
    assert engine._pose_load_error is not None
    assert [d.site for d in _offline.degraded()] == ["ultralytics.pose"]
    with pytest.raises(_offline.OfflineRefusal):  # asking for pose still refuses
        ue.UltralyticsEngine(load_pose=True)


def test_the_llama_backend_remembers_its_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.models import download
    from maxim.models.language.llama_backend import _LlamaCppBackend

    backend = object.__new__(_LlamaCppBackend)
    backend.cfg = types.SimpleNamespace(profile="phi-2", model_base=None)
    backend._download_refused = False
    backend._security_refusal = None
    monkeypatch.setattr(
        download, "ensure_available", lambda *a, **k: _offline.refuse("models.download.ensure_available")
    )
    _offline.engage("test")
    for _ in range(3):
        with pytest.raises(_offline.OfflineRefusal):
            backend._fetch_missing_model("/missing.gguf")
    assert _ledger_events() == ["models.download.ensure_available"]  # one booking, not one per call


def test_a_pinned_missing_profile_refuses_and_an_auto_one_re_walks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Owner decision V25."""
    from maxim.models import download
    from maxim.runtime import lane_backends, lane_models
    from maxim.runtime.worker_pool import LaneConfig

    monkeypatch.setattr(lane_backends, "_profile_has_local_file", lambda p: p == "smollm-1.7b-instruct")
    monkeypatch.setattr(download, "llm_models", lambda: {"phi-2": {}, "smollm-1.7b-instruct": {}})
    monkeypatch.setattr(
        lane_models,
        "detect_tiers",
        lambda caps, profile_available: {
            "medium": LaneConfig(name="medium", max_workers=1, model_profile="smollm-1.7b-instruct")
        },
    )
    lanes = {"medium": LaneConfig(name="medium", max_workers=1, model_profile="phi-2")}
    caps = types.SimpleNamespace(has_gpu=False)
    _offline.engage("test")
    out = lane_backends._ensure_lane_profiles_available(dict(lanes), caps, None)  # auto-detected: re-walks
    assert out["medium"].model_profile == "smollm-1.7b-instruct"
    assert [d.site for d in _offline.degraded()] == ["lanes.model_profile"]
    assert "smollm-1.7b-instruct" in _offline.degraded()[0].reason
    monkeypatch.setenv("MAXIM_LLM_PROFILE", "phi-2")
    with pytest.raises(_offline.OfflineRefusal, match="phi-2"):  # pinned (--llm): refused by name, never swapped
        lane_backends._ensure_lane_profiles_available(dict(lanes), caps, None)


def test_a_profile_set_in_config_is_pinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from maxim.runtime import config_loader, lane_backends

    monkeypatch.delenv("MAXIM_LLM_PROFILE", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert not lane_backends._profile_is_pinned("phi-2")
    (tmp_path / "maxim").mkdir()
    (tmp_path / "maxim" / "config.json").write_text(json.dumps({"_format_version": "1.5", "llm": {"profile": "phi-2"}}))
    config_loader.invalidate_config_cache()
    assert lane_backends._profile_is_pinned("phi-2") and not lane_backends._profile_is_pinned("mistral-7b")


def test_the_sandbox_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.simulation import container_runner, sandbox

    monkeypatch.setattr(container_runner.subprocess, "run", MagicMock(side_effect=AssertionError("spawned")))
    _offline.engage("test")
    assert container_runner.check_docker_available(refresh=True) is False  # no probe spawn
    box = sandbox.create_sandbox(backend="auto", populate=False)
    assert isinstance(getattr(box, "_sandbox", box), sandbox.TmpdirSandbox)
    assert [d.site for d in _offline.degraded()] == ["sandbox"]
    with pytest.raises(_offline.OfflineRefusal):
        sandbox.create_sandbox(backend="docker", populate=False)


def test_the_sim_never_runs_without_a_sandbox_on_a_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.simulation import orchestrator, sandbox

    monkeypatch.setattr(sandbox, "create_sandbox", MagicMock(side_effect=_offline.OfflineRefusal("docker refused")))
    with pytest.raises(_offline.OfflineRefusal):
        orchestrator._setup_sim_sandbox(backend="docker", populate=False)


def test_an_offline_report_says_what_it_could_not_do() -> None:
    """Owner decision V27: report.json and decision_log carry the refusals and degraded fallbacks."""
    from maxim.runtime import decision_log
    from maxim.simulation import report

    assert report._offline_stamp() == {} and decision_log._offline_fields() == {}  # online: unchanged
    _offline.engage("test")
    with pytest.raises(_offline.OfflineRefusal):
        _offline.refuse("socket.connect", "x")
    _offline.mark_degraded("similarity.encoder", "m is not cached locally")
    _offline.mark_degraded("similarity.encoder", "m is not cached locally")
    stamp = report._offline_stamp()["offline"]
    assert stamp["refusal_count"] == 1 and stamp["refusals"] == [{"event": "socket.connect", "detail": "x"}]
    assert stamp["degraded"] == [{"site": "similarity.encoder", "reason": "m is not cached locally"}]
    fields = decision_log._offline_fields()
    assert fields == {"offline": True, "offline_degraded": ["similarity.encoder: m is not cached locally"]}
    record = decision_log.LaneDecisionRecord(
        timestamp=0.0, pid=1, maxim_version="x", caps={}, env={}, peer_config_loaded=False, tier_decisions={}, **fields
    )
    assert "degraded: similarity.encoder" in decision_log.format_record_human(record)

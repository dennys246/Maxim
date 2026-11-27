"""#1315 item 10b: the call sites refuse first, so an offline failure is typed and early (notes §6 layer 2).

The child-process census gate (``tests/integration/test_offline_latch.py::test_no_effect_subprocess``) proves each
census site end to end. These pin each refusal in-process, so coverage sees it and a deletion fails a named test,
and they pin the 10a review's NIT checklist that 10b carries. The conftest autouse ``_reset_offline_latch`` fixture
disengages the latch after each test.
"""

from __future__ import annotations

import os
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


def test_doctor_marks_a_single_row_check_and_names_each_robot() -> None:
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
    robot = types.SimpleNamespace(robot_id="reachy-1", config={})
    before = len(_offline.ledger())
    one = checks._check_one_robot(MagicMock(os="macos"), robot)  # says so by name, before any probe
    assert (one.name, one.status, one.retry_id) == ("robot:reachy-1", "info", "robot")
    assert len(_offline.ledger()) == before


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
    assert refused.refused == "offline" and not refused.passed  # the reason, its "refused: " prefix dropped
    errored = runner._aggregate_runs("m", [bm.RunResult("s", "m", 0, error="boom")])
    assert errored.refused is None
    report = bm.BenchmarkReport(
        timestamp="t", suite="s", models=["claude"], runs_per_model=1, results={"claude": refused}
    )
    assert "claude: REFUSED" in report.summary_table()


def test_the_benchmark_loop_records_a_refusal_and_continues() -> None:
    import ast
    import inspect
    import textwrap

    from maxim.simulation import benchmark

    # The scenario loop is BenchmarkRunner.run: a refused model is recorded and the others still run. write_paper's
    # handler re-raises on purpose (a refused paper exits 2, owner decision 2026-11-23), so the search is scoped.
    tree = ast.parse(textwrap.dedent(inspect.getsource(benchmark.BenchmarkRunner.run)))
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
        lambda caps, profile_available, profile=None: {
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
    # the build's own profile (api.run(model=), --aut-model, research's language_model; #1538 hands it to the build,
    # not the env) is pinned: refused by name before any consent prompt, not LocalModelUnavailable's "not downloaded"
    with pytest.raises(_offline.OfflineRefusal, match="phi-2"):
        lane_backends._ensure_lane_profiles_available(dict(lanes), caps, None, profile="phi-2")
    monkeypatch.setenv("MAXIM_LLM_PROFILE", "phi-2")
    with pytest.raises(_offline.OfflineRefusal, match="phi-2"):  # pinned (--llm): refused by name, never swapped
        lane_backends._ensure_lane_profiles_available(dict(lanes), caps, None)


def test_a_profile_set_in_config_is_pinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from maxim.runtime import config_loader, lane_backends

    monkeypatch.delenv("MAXIM_LLM_PROFILE", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert not lane_backends._profile_is_pinned("phi-2", "medium")
    (tmp_path / "maxim").mkdir()
    (tmp_path / "maxim" / "config.json").write_text(json.dumps({"_format_version": "1.5", "llm": {"profile": "phi-2"}}))
    config_loader.invalidate_config_cache()
    assert lane_backends._profile_is_pinned("phi-2", "medium")
    assert not lane_backends._profile_is_pinned("mistral-7b", "medium")


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


def test_an_llm_json_tier_profile_is_pinned(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Owner decision V28: a per-tier profile the operator wrote in llm.json counts as pinned. Live since #1601: a real
    llm.json file (not a stand-in config) is what the check reads."""
    import json

    from maxim.runtime import lane_backends

    monkeypatch.delenv("MAXIM_LLM_PROFILE", raising=False)
    llm_json = tmp_path / "llm.json"
    llm_json.write_text(json.dumps({"tiers": {"medium": {"model_profile": "phi-2"}}}), encoding="utf-8")
    monkeypatch.setenv("MAXIM_LLM_CONFIG", str(llm_json))
    assert lane_backends._profile_is_pinned("phi-2", "medium")
    assert not lane_backends._profile_is_pinned("phi-2", "large")  # another tier's override pins nothing here


def _clear_env(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Unset ``names`` for one test so the restore is recorded: a bare ``delenv(raising=False)`` of an absent variable
    records nothing, and a value the code under test then sets outlives the test (delta round 7, S3)."""
    for name in names:
        monkeypatch.setenv(name, "unset-by-test")
        monkeypatch.delenv(name)


def test_the_cli_cpu_fallback_default_reads_as_a_pin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Owner decision V31 (fail closed): on a GPU-less box the served model comes from load_llm_config, whose only
    input for Maxim's default is MAXIM_LLM_PROFILE, so the default reads as pinned and offline a missing one refuses by
    name. Flip it when #1737 builds the GPU-less router from the lane system."""
    import logging

    from maxim import cli_utils
    from maxim.runtime import lane_backends

    _clear_env(monkeypatch, "MAXIM_LLM_PROFILE", "MAXIM_LLM_N_GPU_LAYERS")
    # no llm.json declares llama.cpp (the checkout's own data/util/llm.json is not this test's input; #1806 PR 0)
    monkeypatch.setattr("maxim.models.language.config.configured_profile_uses_llama_cpp", lambda: False)
    cli_utils.configure_cpu_fallback_model(logging.getLogger("test"), home_dir=str(tmp_path))
    assert os.environ["MAXIM_LLM_PROFILE"] == "smollm-1.7b-instruct"
    assert os.environ["MAXIM_LLM_N_GPU_LAYERS"] == "0"
    assert lane_backends._profile_is_pinned("smollm-1.7b-instruct", "medium")


def test_the_robot_cpu_fallback_default_reads_as_a_pin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """V31, the robot runtime's twin (#1602 is superseded: there is no marker to miss)."""
    import logging

    from maxim.embodied_runtime import agentic_runtime
    from maxim.runtime import lane_backends

    _clear_env(monkeypatch, "MAXIM_LLM_PROFILE", "MAXIM_LLM_N_GPU_LAYERS", "CUDA_VISIBLE_DEVICES")
    # no llm.json declares llama.cpp (the checkout's own data/util/llm.json is not this test's input; #1806 PR 0)
    monkeypatch.setattr("maxim.models.language.config.configured_profile_uses_llama_cpp", lambda: False)
    monkeypatch.setattr(agentic_runtime, "is_gpu_available", lambda: False)
    runtime = types.SimpleNamespace(home_dir=str(tmp_path), log=logging.getLogger("test"))
    agentic_runtime.AgenticRuntimeMixin._apply_llm_backend_fallback(runtime)  # type: ignore[arg-type]
    assert os.environ["MAXIM_LLM_PROFILE"] == "smollm-1.7b-instruct"
    assert os.environ["MAXIM_LLM_N_GPU_LAYERS"] == "0"
    assert lane_backends._profile_is_pinned("smollm-1.7b-instruct", "medium")


def test_an_operator_profile_survives_either_cpu_fallback_unannounced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from maxim import cli_utils
    from maxim.embodied_runtime import agentic_runtime

    _clear_env(monkeypatch, "MAXIM_LLM_N_GPU_LAYERS", "CUDA_VISIBLE_DEVICES")
    # no llm.json declares llama.cpp (the checkout's own data/util/llm.json is not this test's input; #1806 PR 0)
    monkeypatch.setattr("maxim.models.language.config.configured_profile_uses_llama_cpp", lambda: False)
    monkeypatch.setenv("MAXIM_LLM_PROFILE", "phi-2")
    monkeypatch.setattr(agentic_runtime, "is_gpu_available", lambda: False)
    with caplog.at_level(logging.INFO):
        cli_utils.configure_cpu_fallback_model(logging.getLogger("test"), home_dir=str(tmp_path))
        runtime = types.SimpleNamespace(home_dir=str(tmp_path), log=logging.getLogger("test"))
        agentic_runtime.AgenticRuntimeMixin._apply_llm_backend_fallback(runtime)  # type: ignore[arg-type]
    assert os.environ["MAXIM_LLM_PROFILE"] == "phi-2"
    assert os.environ["MAXIM_LLM_N_GPU_LAYERS"] == "0"
    assert (
        "smollm" not in capsys.readouterr().err and "smollm" not in caplog.text
    )  # neither claims a model it did not set


def test_a_re_walk_that_finds_no_model_on_disk_is_recorded_as_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """V27: the offline block is a claim. A re-walk that hands back the missing profile, or another that is not on
    disk, gave the lane no model, and the record says so instead of naming a substitute (delta rounds 7 and 8)."""
    from maxim.models import download
    from maxim.runtime import lane_backends, lane_models
    from maxim.runtime.worker_pool import LaneConfig

    monkeypatch.setattr(lane_backends, "_profile_has_local_file", lambda p: False)
    monkeypatch.setattr(download, "llm_models", lambda: {"phi-2": {}, "smollm-1.7b-instruct": {}})
    _offline.engage("test")
    caps = types.SimpleNamespace(has_gpu=False)
    lanes = {"medium": LaneConfig(name="medium", max_workers=1, model_profile="phi-2")}
    for handed_back in ("phi-2", "smollm-1.7b-instruct"):  # the missing profile itself; another that is missing too
        monkeypatch.setattr(
            lane_models,
            "detect_tiers",
            lambda caps, profile_available, profile=None, _p=handed_back: {
                "medium": LaneConfig(name="medium", max_workers=1, model_profile=_p)
            },
        )
        lane_backends._ensure_lane_profiles_available(dict(lanes), caps, None)
    reasons = {d.reason for d in _offline.degraded()}
    assert reasons == {"phi-2 is not on disk; lane medium has no model on disk"}, reasons


def test_warmup_never_leaves_wait_ready_blocking_on_a_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delta round 1: a remembered refusal raised inside the warmup thread still sets the ready event."""
    from maxim.models.language.router import LLMRouter

    router = object.__new__(LLMRouter)
    router.cfg = types.SimpleNamespace(model_path="m.gguf")
    router._ready_event = threading.Event()
    router._warmup_failed = False
    router._providers = {}
    router._backend_lock = threading.Lock()  # #1657's unload bookkeeping
    router._unloaded = False
    router._warmups_running = 0
    router._deferred_release = []
    router._release_backends = lambda _deferred: None
    backend = MagicMock()
    backend.warmup.side_effect = _offline.OfflineRefusal("remembered")
    router._get_backend = lambda: backend
    router.enabled = lambda: True
    from maxim.models.language import router as router_mod

    said: list[str] = []
    monkeypatch.setattr(router_mod, "warn", lambda msg, *a: said.append(msg % a))
    _offline.engage("test")
    assert router.warmup() is True
    assert router._ready_event.wait(5.0) and router._warmup_failed
    assert said == ["LLM warmup refused: remembered"]  # named a refusal, not "LLM warmup failed"


def test_benchmark_exit_code_and_api_scores_leave_a_refused_model_out() -> None:
    from maxim.simulation import benchmark as bm

    refused = bm.ModelResult(model="claude", refused="offline")
    passed = bm.ModelResult(model="phi-2", passed=True, score=0.7)
    report = bm.BenchmarkReport(timestamp="t", suite="s", models=["claude", "phi-2"], runs_per_model=1)
    report.results = {"phi-2": passed}
    assert report.exit_code() == 0
    report.results = {"phi-2": passed, "claude": refused}
    assert report.exit_code() == 2
    report.results = {"phi-2": bm.ModelResult(model="phi-2")}
    assert report.exit_code() == 1


def test_an_online_decision_record_keeps_its_old_shape() -> None:
    import json

    from maxim.runtime import decision_log

    record = decision_log.LaneDecisionRecord(
        timestamp=0.0, pid=1, maxim_version="x", caps={}, env={}, peer_config_loaded=False, tier_decisions={}
    )
    assert "offline" not in json.loads(decision_log.record_to_json_line(record))
    record.offline = True
    assert json.loads(decision_log.record_to_json_line(record))["offline"] is True


def test_api_benchmark_does_not_score_a_refused_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from maxim import api
    from maxim.simulation import benchmark as bm

    report = bm.BenchmarkReport(timestamp="t", suite="s", models=["claude", "phi-2"], runs_per_model=1)
    report.results = {
        "claude": bm.ModelResult(model="claude", refused="offline"),
        "phi-2": bm.ModelResult(model="phi-2", passed=True, score=0.7),
    }
    runner = MagicMock()
    runner.return_value.run.return_value = report
    monkeypatch.setattr(bm, "BenchmarkRunner", runner)
    suite = tmp_path / "suite.yaml"
    suite.write_text("scenarios: []\n")
    result = api.benchmark(["claude", "phi-2"], suite=str(suite))
    assert set(result.scores) == {"phi-2"}


# The merge with main (2026-11-20 wire review): main's consented-router call sites in cli.py kept broad handlers that
# turned an offline refusal into a template fallback, a warning, or a misread boot failure.


def _refusing_router(
    _profile: str, **_kwargs: object
) -> None:  # cli._cli_local_router also takes fallback_user= (#1680)
    _offline.refuse("router[build]", "lanes that are not in-process: local")


def test_the_curate_router_factory_keeps_the_offline_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim import cli

    monkeypatch.setattr(cli, "_cli_local_router", _refusing_router)
    _offline.engage("test")
    built: list[object] = []
    with pytest.raises(_offline.OfflineRefusal):  # not logged at DEBUG and handed back as the template fallback
        cli._curate_router_factory("claude-sonnet-4-6", built)()
    assert built == []


def test_auto_curation_refused_offline_stops_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim import cli
    from maxim.simulation import foundry

    def curate(**kw: object) -> None:
        kw["llm_router_factory"]()  # type: ignore[operator]

    monkeypatch.setattr(cli, "_cli_local_router", _refusing_router)
    monkeypatch.setattr(foundry, "auto_curate", curate)
    args = types.SimpleNamespace(auto_curate=True, no_curate=False, language_model="claude-sonnet-4-6")
    _offline.engage("test")
    with pytest.raises(_offline.OfflineRefusal):  # not "WARNING: Auto-curation failed" and on into the sim
        cli._auto_curate_before_sim(args, "entity")


def test_a_foundry_refused_offline_never_reports_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim import cli
    from maxim.simulation import foundry

    monkeypatch.setattr(cli, "_cli_local_router", _refusing_router)
    monkeypatch.setattr(foundry, "FoundryRunner", MagicMock(side_effect=AssertionError("ran on templates")))
    _offline.engage("test")
    with pytest.raises(_offline.OfflineRefusal):  # cli.main exits 2, not a template run that returns 0
        cli._handle_foundry(types.SimpleNamespace(language_model="claude-sonnet-4-6"), "castle")


def test_an_early_leader_boot_refused_offline_is_not_a_port_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim import cli
    from maxim.runtime import leader_mode

    def claim(_host: str) -> None:
        _offline.refuse("socket", "bind 127.0.0.1:8100")

    monkeypatch.setattr(
        leader_mode, "detect_role", lambda: types.SimpleNamespace(role="leader", bind_host="127.0.0.1", reason="t")
    )
    monkeypatch.setattr(leader_mode, "claim_leader_services", claim)
    _offline.engage("test")
    with pytest.raises(_offline.OfflineRefusal, match="8100"):
        cli._early_leader_boot()


def test_a_refused_paper_router_is_named_recorded_as_degraded_and_raised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A refused paper router is said, marked degraded offline, and raised: the run exits 2 (V26, 2026-11-23)."""
    from maxim.models import download
    from maxim.models.language import router as router_mod
    from maxim.simulation import benchmark as bm

    monkeypatch.setattr(download, "consented_router", lambda *_a, **_k: _refusing_router("default"))
    monkeypatch.setattr(router_mod, "load_llm_config", lambda: None)
    report = bm.BenchmarkReport(timestamp="t", suite="s", models=["phi-2"], runs_per_model=1)
    report.results = {"phi-2": bm.ModelResult(model="phi-2", passed=True, score=0.7)}
    _offline.engage("test")
    runner = object.__new__(bm.BenchmarkRunner)
    with caplog.at_level("WARNING", logger="maxim.simulation.benchmark"), pytest.raises(_offline.OfflineRefusal):
        runner.write_paper(report, tmp_path)
    assert [d.site for d in _offline.degraded()] == ["benchmark.paper"]
    assert "lanes that are not in-process" in _offline.degraded()[0].reason
    assert "router was refused" in caplog.text
    assert not list(tmp_path.iterdir())  # the router is built first: no experiment log is left behind


def test_a_leader_refused_paper_router_is_raised_but_not_recorded_as_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Delta round 4 NIT: an online LeaderRefusal is raised (exit 2) but books no offline 'degraded' entry."""
    from maxim.exceptions import LeaderRefusal
    from maxim.models import download
    from maxim.models.language import router as router_mod
    from maxim.simulation import benchmark as bm

    def refuse(*_a: object, **_k: object) -> None:
        raise LeaderRefusal("no mesh key")

    monkeypatch.setattr(download, "consented_router", refuse)
    monkeypatch.setattr(router_mod, "load_llm_config", lambda: None)
    report = bm.BenchmarkReport(timestamp="t", suite="s", models=["phi-2"], runs_per_model=1)
    report.results = {"phi-2": bm.ModelResult(model="phi-2", passed=True, score=0.7)}
    with pytest.raises(LeaderRefusal):
        object.__new__(bm.BenchmarkRunner).write_paper(report, tmp_path)
    assert not _offline.degraded()


def test_a_paper_section_refused_mid_write_propagates_through_the_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The same refusal at a section call also raises, through ``cli._write_benchmark_paper``: one exit code (2)
    wherever the paper writer is refused (delta round 10 SHOULD-FIX 1, owner decision 2026-11-23)."""
    from maxim import cli
    from maxim.exceptions import LeaderRefusal
    from maxim.models import download
    from maxim.models.language import router as router_mod
    from maxim.simulation import benchmark as bm

    class _RefusingRouter:  # the real WriterAgent runs: its one LLM call is _complete_text (round 11 NIT)
        calls = 0

        def _complete_text(self, *_a: object, **_k: object) -> tuple[str, None]:
            _RefusingRouter.calls += 1
            raise LeaderRefusal("the leader refused a section call")

    monkeypatch.setattr(download, "consented_router", lambda *_a, **_k: (_RefusingRouter(), None))
    monkeypatch.setattr(router_mod, "load_llm_config", lambda: None)
    report = bm.BenchmarkReport(timestamp="t", suite="s", models=["phi-2"], runs_per_model=1)
    report.results = {"phi-2": bm.ModelResult(model="phi-2", passed=True, score=0.7)}
    runner = object.__new__(bm.BenchmarkRunner)
    with pytest.raises(LeaderRefusal, match="section call"):
        cli._write_benchmark_paper(runner, report, tmp_path)
    assert _RefusingRouter.calls == 1  # the first section's call: refused, not retried or swallowed


# An ExecAgent whose router is refused (#1315 item 10b): the one-router-per-run rule itself is #1228's
# (``MaximAgent(shared_router=)``; tests/unit/test_agent_router_slots_1228.py).


def test_a_refused_run_router_ends_the_proposal_worker_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.agents import exec_agent as exec_mod
    from maxim.agents.maxim_agent import MaximAgent

    exec_agent = MaximAgent(shared_router=MagicMock()).exec_agent

    def refused(_ctx: object) -> None:
        raise _offline.OfflineRefusal("router[dispatch] refused: lane-large")

    monkeypatch.setattr(exec_agent, "_propose_goal", refused)
    said: list[str] = []
    monkeypatch.setattr(exec_mod, "warn", lambda msg, *a: said.append(msg % a))
    exec_agent._work_available.set()
    worker = threading.Thread(target=exec_agent._worker_loop, daemon=True)
    worker.start()
    worker.join(5.0)
    assert not worker.is_alive()  # ended, not spinning or dead on a traceback
    assert said == ["ExecAgent stops proposing: router[dispatch] refused: lane-large"]


def test_propose_goal_keeps_a_refusal_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.agents.maxim_agent import MaximAgent

    exec_agent = MaximAgent(shared_router=MagicMock()).exec_agent
    exec_agent._run_pre_deliberation = lambda _ctx: ""  # type: ignore[method-assign]

    def refused(*_a: object) -> None:
        raise _offline.OfflineRefusal("router[dispatch] refused")

    monkeypatch.setattr(exec_agent, "_invoke_llm_for_goal", refused)
    with pytest.raises(_offline.OfflineRefusal):  # not "Language model error during plan generation"
        exec_agent._propose_goal(exec_agent._memory.build_context())
